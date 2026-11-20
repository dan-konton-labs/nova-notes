//! Sources: `/api/sources/...` (Python `api/routers/sources.py`).
//!
//! `POST /sources` takes form data (multipart, with an optional `file`, or
//! URL-encoded), as the Python router does; a JSON body is accepted there
//! too. `POST /sources/json` is the JSON-only legacy form.

use axum::body::Body;
use axum::extract::{FromRequest, Multipart, Path, Request, State};
use axum::http::{header, HeaderValue, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::get;
use axum::{Form, Json, Router};
use serde_json::{json, Value};
use std::collections::HashMap;

use super::notebooks::{validation_error, ApiJson, ApiQuery};
use crate::error::{AppError, AppResult};
use crate::services::insights::{self, InsightCreation, InsightView, NewSourceInsight};
use crate::services::sources::{
    self, NewSource, SourceListItem, SourceListQuery, SourceStatusView, SourceUpdate, SourceUpload, SourceView,
};
use crate::state::AppState;

/// A rejected request: any response, boxed so `Result`s stay small. `?`
/// converts both an [`AppError`] and a ready-made [`Response`].
pub(crate) struct Rejection(Box<Response>);

impl From<Response> for Rejection {
    fn from(response: Response) -> Self {
        Self(Box::new(response))
    }
}

impl From<AppError> for Rejection {
    fn from(error: AppError) -> Self {
        error.into_response().into()
    }
}

impl IntoResponse for Rejection {
    fn into_response(self) -> Response {
        *self.0
    }
}

pub fn router() -> Router<AppState> {
    Router::new()
        .route("/sources", get(list_sources).post(create_source))
        .route("/sources/json", axum::routing::post(create_source_json))
        .route("/sources/{source_id}", get(get_source).put(update_source).delete(delete_source))
        .route("/sources/{source_id}/download", get(download_source_file).head(check_source_file))
        .route("/sources/{source_id}/status", get(source_status))
        .route("/sources/{source_id}/retry", axum::routing::post(retry_source))
        .route("/sources/{source_id}/insights", get(list_insights).post(create_insight))
}

// ---------------------------------------------------------------------------
// GET /sources
// ---------------------------------------------------------------------------

#[derive(Debug, serde::Deserialize)]
struct ListQuery {
    notebook_id: Option<String>,
    limit: Option<String>,
    offset: Option<String>,
    sort_by: Option<String>,
    sort_order: Option<String>,
}

/// An integer query parameter within bounds, or FastAPI's 422.
fn int_param(name: &str, raw: Option<&str>, default: i64, min: i64, max: Option<i64>) -> Result<i64, Rejection> {
    let Some(raw) = raw else { return Ok(default) };
    let value: i64 = raw.trim().parse().map_err(|_| {
        validation_error("query", "int_parsing", format!("{name}: Input should be a valid integer, unable to parse string as an integer"))
    })?;
    if value < min {
        return Err(validation_error("query", "greater_than_equal", format!("{name}: Input should be greater than or equal to {min}")).into());
    }
    if let Some(max) = max.filter(|max| value > *max) {
        return Err(validation_error("query", "less_than_equal", format!("{name}: Input should be less than or equal to {max}")).into());
    }
    Ok(value)
}

async fn list_sources(State(state): State<AppState>, ApiQuery(query): ApiQuery<ListQuery>) -> Result<Json<Vec<SourceListItem>>, Rejection> {
    let defaults = SourceListQuery::default();
    let query = SourceListQuery {
        notebook_id: query.notebook_id,
        limit: int_param("limit", query.limit.as_deref(), defaults.limit, 1, Some(100))?,
        offset: int_param("offset", query.offset.as_deref(), defaults.offset, 0, None)?,
        sort_by: query.sort_by.unwrap_or(defaults.sort_by),
        sort_order: query.sort_order.unwrap_or(defaults.sort_order),
    };
    Ok(Json(sources::list_sources(&state, &query).await?))
}

// ---------------------------------------------------------------------------
// POST /sources
// ---------------------------------------------------------------------------

/// Python's `str_to_bool` for form booleans.
fn form_bool(value: Option<&String>) -> bool {
    value.is_some_and(|v| matches!(v.to_lowercase().as_str(), "true" | "1" | "yes" | "on"))
}

/// A JSON-encoded list of ids in a form field.
fn form_list(fields: &HashMap<String, String>, name: &str) -> Result<Option<Vec<String>>, Rejection> {
    let Some(raw) = fields.get(name).filter(|raw| !raw.is_empty()) else {
        return Ok(None);
    };
    let parsed: Value = serde_json::from_str(raw).map_err(|_| {
        tracing::error!("Invalid JSON in {name} field");
        AppError::unprocessable(format!("Invalid JSON in {name} field")).into_response()
    })?;
    serde_json::from_value(parsed)
        .map(Some)
        .map_err(|_| AppError::unprocessable(format!("Invalid source data: {name}: Input should be a valid list")).into())
}

/// `parse_source_form_data`: the form fields as a [`NewSource`].
fn new_source_from_form(fields: &HashMap<String, String>) -> Result<NewSource, Rejection> {
    let Some(source_type) = fields.get("type") else {
        return Err(validation_error("body", "missing", "type: Field required").into());
    };
    let text = |name: &str| fields.get(name).cloned();
    Ok(NewSource {
        source_type: source_type.clone(),
        notebook_id: text("notebook_id"),
        notebooks: form_list(fields, "notebooks")?,
        url: text("url"),
        content: text("content"),
        title: text("title"),
        file_path: None,
        transformations: Some(form_list(fields, "transformations")?.unwrap_or_default()),
        embed: form_bool(fields.get("embed")),
        delete_source: form_bool(fields.get("delete_source")),
        async_processing: form_bool(fields.get("async_processing")),
    })
}

pub(crate) fn multipart_problem(e: axum::extract::multipart::MultipartError) -> Rejection {
    if e.status() == StatusCode::PAYLOAD_TOO_LARGE {
        return (StatusCode::PAYLOAD_TOO_LARGE, Json(json!({ "detail": "Request body exceeds the maximum allowed upload size" })))
            .into_response()
            .into();
    }
    validation_error("body", "value_error", e.body_text()).into()
}

async fn read_multipart(mut multipart: Multipart) -> Result<(HashMap<String, String>, Option<SourceUpload>), Rejection> {
    let mut fields = HashMap::new();
    let mut upload = None;
    while let Some(field) = multipart.next_field().await.map_err(multipart_problem)? {
        let name = field.name().unwrap_or_default().to_string();
        if name == "file" {
            let filename = field.file_name().unwrap_or_default().to_string();
            let data = field.bytes().await.map_err(multipart_problem)?;
            // A file input left empty arrives as a nameless, empty part.
            if !(filename.is_empty() && data.is_empty()) {
                upload = Some(SourceUpload { filename, data: data.to_vec() });
            }
        } else {
            let value = field.text().await.map_err(multipart_problem)?;
            fields.insert(name, value);
        }
    }
    Ok((fields, upload))
}

async fn create_source(State(state): State<AppState>, request: Request) -> Result<Json<SourceView>, Rejection> {
    let content_type = request
        .headers()
        .get(header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .unwrap_or_default()
        .to_ascii_lowercase();
    let (new, upload) = if content_type.starts_with("multipart/form-data") {
        let multipart = Multipart::from_request(request, &state).await.map_err(IntoResponse::into_response)?;
        let (fields, upload) = read_multipart(multipart).await?;
        (new_source_from_form(&fields)?, upload)
    } else if content_type.starts_with("application/json") {
        let ApiJson(new) = ApiJson::<NewSource>::from_request(request, &state).await?;
        (new, None)
    } else {
        let fields = match Form::<HashMap<String, String>>::from_request(request, &state).await {
            Ok(Form(fields)) => fields,
            Err(rejection) if rejection.status() == StatusCode::PAYLOAD_TOO_LARGE => return Err(rejection.into_response().into()),
            // No form at all: FastAPI reports the required field as missing.
            Err(_) => HashMap::new(),
        };
        (new_source_from_form(&fields)?, None)
    };
    Ok(Json(sources::create_source(&state, new, upload).await?))
}

async fn create_source_json(State(state): State<AppState>, ApiJson(new): ApiJson<NewSource>) -> AppResult<Json<SourceView>> {
    Ok(Json(sources::create_source(&state, new, None).await?))
}

// ---------------------------------------------------------------------------
// One source
// ---------------------------------------------------------------------------

async fn get_source(State(state): State<AppState>, Path(source_id): Path<String>) -> AppResult<Json<SourceView>> {
    Ok(Json(sources::get_source(&state, &source_id).await?))
}

async fn update_source(
    State(state): State<AppState>,
    Path(source_id): Path<String>,
    ApiJson(body): ApiJson<SourceUpdate>,
) -> AppResult<Json<SourceView>> {
    Ok(Json(sources::update_source(&state, &source_id, &body).await?))
}

async fn delete_source(State(state): State<AppState>, Path(source_id): Path<String>) -> AppResult<Json<Value>> {
    sources::delete_source(&state, &source_id).await?;
    Ok(Json(json!({ "message": "Source deleted successfully" })))
}

async fn source_status(State(state): State<AppState>, Path(source_id): Path<String>) -> AppResult<Json<SourceStatusView>> {
    Ok(Json(sources::source_status(&state, &source_id).await?))
}

async fn retry_source(State(state): State<AppState>, Path(source_id): Path<String>) -> AppResult<Json<SourceView>> {
    Ok(Json(sources::retry_source(&state, &source_id).await?))
}

/// `Content-Disposition: attachment` the way Starlette's `FileResponse`
/// writes it: a quoted name when it needs no escaping, RFC 5987 otherwise.
fn attachment_disposition(filename: &str) -> String {
    let mut encoded = String::with_capacity(filename.len());
    for byte in filename.bytes() {
        if byte.is_ascii_alphanumeric() || b"_.-~/".contains(&byte) {
            encoded.push(byte as char);
        } else {
            encoded.push_str(&format!("%{byte:02X}"));
        }
    }
    if encoded == filename {
        format!("attachment; filename=\"{filename}\"")
    } else {
        format!("attachment; filename*=utf-8''{encoded}")
    }
}

async fn check_source_file(State(state): State<AppState>, Path(source_id): Path<String>) -> AppResult<StatusCode> {
    sources::source_file(&state, &source_id).await?;
    Ok(StatusCode::OK)
}

async fn download_source_file(State(state): State<AppState>, Path(source_id): Path<String>) -> AppResult<Response> {
    let file = sources::source_file(&state, &source_id).await?;
    let handle = tokio::fs::File::open(&file.path).await.map_err(|_| AppError::not_found("File not found on server"))?;
    let length = handle.metadata().await?.len();
    let disposition = HeaderValue::from_str(&attachment_disposition(&file.filename))
        .map_err(|e| AppError::internal(format!("Invalid header value: {e}")))?;
    let mut response = Body::from_stream(tokio_util::io::ReaderStream::new(handle)).into_response();
    let headers = response.headers_mut();
    headers.insert(header::CONTENT_TYPE, HeaderValue::from_static("application/octet-stream"));
    headers.insert(header::CONTENT_LENGTH, HeaderValue::from(length));
    headers.insert(header::CONTENT_DISPOSITION, disposition);
    Ok(response)
}

// ---------------------------------------------------------------------------
// A source's insights
// ---------------------------------------------------------------------------

async fn list_insights(State(state): State<AppState>, Path(source_id): Path<String>) -> AppResult<Json<Vec<InsightView>>> {
    Ok(Json(insights::list_source_insights(&state, &source_id).await?))
}

async fn create_insight(
    State(state): State<AppState>,
    Path(source_id): Path<String>,
    ApiJson(body): ApiJson<NewSourceInsight>,
) -> AppResult<(StatusCode, Json<InsightCreation>)> {
    Ok((StatusCode::ACCEPTED, Json(insights::create_source_insight(&state, &source_id, &body).await?)))
}

#[cfg(test)]
mod tests {
    use axum::body::Body;
    use axum::http::{header, Method, Request, StatusCode};
    use http_body_util::BodyExt;
    use serde_json::{json, Value};

    use crate::ai::AnyModel;
    use crate::domain::Source;
    use crate::testing::{multipart_body, MockEmbeddingModel, MockLanguageModel, TestApp};

    async fn notebook(app: &TestApp) -> String {
        let (status, body) = app.post("/api/notebooks", json!({ "name": "NB", "description": "" })).await;
        assert_eq!(status, StatusCode::OK, "{body}");
        body["id"].as_str().unwrap().to_string()
    }

    async fn transformation(app: &TestApp, title: &str) -> String {
        let (status, body) = app
            .post(
                "/api/transformations",
                json!({ "name": title.to_lowercase(), "title": title, "description": "d", "prompt": "Summarise." }),
            )
            .await;
        assert_eq!(status, StatusCode::OK, "{body}");
        body["id"].as_str().unwrap().to_string()
    }

    fn mock_models(app: &TestApp, replies: &[&str]) -> std::sync::Arc<MockLanguageModel> {
        let language = MockLanguageModel::new(replies.iter().copied());
        app.state.models.set_override(AnyModel::Language(language.clone()));
        app.state.models.set_override(AnyModel::Embedding(MockEmbeddingModel::new()));
        language
    }

    async fn send_multipart(app: &TestApp, path: &str, parts: &[(&str, Option<&str>, &[u8])]) -> (StatusCode, Value) {
        let (content_type, body) = multipart_body(parts);
        let request = Request::builder()
            .method(Method::POST)
            .uri(path)
            .header(header::CONTENT_TYPE, content_type)
            .body(Body::from(body))
            .unwrap();
        let response = app.send(request).await;
        let status = response.status();
        let bytes = response.into_body().collect().await.unwrap().to_bytes();
        (status, serde_json::from_slice(&bytes).unwrap_or(Value::Null))
    }

    async fn chunk_count(app: &TestApp, source_id: &str) -> i64 {
        Source::get(&app.state, source_id).await.unwrap().embedded_chunks(&app.state).await.unwrap()
    }

    #[tokio::test]
    async fn text_source_is_processed_end_to_end() {
        let app = TestApp::new().await;
        let language = mock_models(&app, &["<think>hmm</think>The summary."]);
        let notebook_id = notebook(&app).await;
        let transformation_id = transformation(&app, "Summary").await;

        let notebooks = json!([notebook_id]).to_string();
        let transformations = json!([transformation_id]).to_string();
        let (status, body) = send_multipart(
            &app,
            "/api/sources",
            &[
                ("type", None, b"text"),
                ("title", None, b"My text"),
                ("content", None, b"Rust is a systems programming language. It is fast and safe."),
                ("notebooks", None, notebooks.as_bytes()),
                ("transformations", None, transformations.as_bytes()),
                ("embed", None, b"true"),
            ],
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{body}");
        let source_id = body["id"].as_str().unwrap().to_string();
        assert_eq!(body["title"], "My text");
        assert!(body["full_text"].as_str().unwrap().contains("systems programming"));
        assert_eq!(language.call_count(), 1);

        // The insight and the chunks arrive through their own jobs.
        app.wait_for_idle().await;
        let (_, insights) = app.get(&format!("/api/sources/{source_id}/insights")).await;
        assert_eq!(insights[0]["insight_type"], "Summary");
        assert_eq!(insights[0]["content"], "The summary.");
        assert_eq!(insights[0]["source_id"], source_id.as_str());
        assert!(chunk_count(&app, &source_id).await > 0);

        let (status, view) = app.get(&format!("/api/sources/{source_id}")).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(view["embedded"], true);
        assert_eq!(view["notebooks"], json!([notebook_id]));
        assert_eq!(view["file_available"], Value::Null);

        let (status, list) = app.get(&format!("/api/sources?notebook_id={notebook_id}")).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(list.as_array().unwrap().len(), 1);
        assert_eq!(list[0]["insights_count"], 1);
        assert_eq!(list[0]["embedded"], true);
    }

    #[tokio::test]
    async fn async_sources_are_queued_and_titled_from_the_content() {
        let app = TestApp::new().await;
        mock_models(&app, &[]);
        let notebook_id = notebook(&app).await;
        let (status, body) = app
            .post(
                "/api/sources/json",
                json!({
                    "type": "text",
                    "notebook_id": notebook_id,
                    "content": "Some plain text content.",
                    "async_processing": true,
                }),
            )
            .await;
        assert_eq!(status, StatusCode::OK, "{body}");
        assert_eq!(body["status"], "new");
        assert_eq!(body["processing_info"], json!({ "async": true, "queued": true }));
        let command_id = body["command_id"].as_str().unwrap().to_string();
        assert_eq!(app.wait_for_job(&command_id).await, "completed");

        let source_id = body["id"].as_str().unwrap();
        let (status, status_body) = app.get(&format!("/api/sources/{source_id}/status")).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(status_body["status"], "completed");
        assert_eq!(status_body["message"], "Source processing completed successfully");
        let (_, view) = app.get(&format!("/api/sources/{source_id}")).await;
        assert_eq!(view["full_text"], "Some plain text content.");
        assert_eq!(view["embedded"], false);
    }

    #[tokio::test]
    async fn link_sources_are_fetched() {
        let server = wiremock::MockServer::start().await;
        wiremock::Mock::given(wiremock::matchers::method("GET"))
            .respond_with(
                wiremock::ResponseTemplate::new(200).set_body_raw(
                    "<html><head><title>Local Page</title></head><body><article><p>Hello from the local test page, \
                     with enough words to count as content.</p></article></body></html>",
                    "text/html",
                ),
            )
            .mount(&server)
            .await;
        let app = TestApp::new().await;
        mock_models(&app, &[]);
        // Loopback is allowed (self-hosted), as in the Python build.
        let (status, body) = app.post("/api/sources/json", json!({ "type": "link", "url": server.uri(), "embed": true })).await;
        assert_eq!(status, StatusCode::OK, "{body}");
        assert_eq!(body["title"], "Local Page");
        assert!(body["full_text"].as_str().unwrap().contains("Hello from the local test page"), "{body}");
        assert_eq!(body["asset"]["url"], server.uri().as_str());

        // Link-local (cloud metadata) addresses are refused before any fetch.
        let (status, body) = app.post("/api/sources/json", json!({ "type": "link", "url": "http://169.254.169.254/latest" })).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
    }

    #[tokio::test]
    async fn uploads_are_stored_processed_and_downloadable() {
        let app = TestApp::new().await;
        mock_models(&app, &[]);
        let notebook_id = notebook(&app).await;
        let (status, body) = send_multipart(
            &app,
            "/api/sources",
            &[
                ("type", None, b"upload"),
                ("notebook_id", None, notebook_id.as_bytes()),
                ("file", Some("../notes.txt"), b"Uploaded plain text body."),
            ],
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{body}");
        let source_id = body["id"].as_str().unwrap().to_string();
        assert_eq!(body["full_text"], "Uploaded plain text body.");
        let path = body["asset"]["file_path"].as_str().unwrap().to_string();
        assert!(path.ends_with("notes.txt"), "{path}");
        assert!(crate::services::sources::is_within_uploads(&app.state.config, std::path::Path::new(&path)));

        let (_, view) = app.get(&format!("/api/sources/{source_id}")).await;
        assert_eq!(view["file_available"], true);

        let response = app
            .send(Request::builder().method(Method::HEAD).uri(format!("/api/sources/{source_id}/download")).body(Body::empty()).unwrap())
            .await;
        assert_eq!(response.status(), StatusCode::OK);
        let response = app
            .send(Request::builder().uri(format!("/api/sources/{source_id}/download")).body(Body::empty()).unwrap())
            .await;
        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(response.headers()[header::CONTENT_DISPOSITION], "attachment; filename=\"notes.txt\"");
        let bytes = response.into_body().collect().await.unwrap().to_bytes();
        assert_eq!(&bytes[..], b"Uploaded plain text body.");

        // A second upload of the same name gets its own file.
        let (status, second) = send_multipart(
            &app,
            "/api/sources",
            &[("type", None, b"upload"), ("delete_source", None, b"true"), ("file", Some("notes.txt"), b"Second body.")],
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{second}");
        let second_path = second["asset"]["file_path"].as_str().unwrap();
        assert!(second_path.ends_with("notes (1).txt"), "{second_path}");
        assert!(!std::path::Path::new(second_path).exists(), "delete_source removes the file after extraction");
        let (status, body) = app.get(&format!("/api/sources/{}/download", second["id"].as_str().unwrap())).await;
        assert_eq!((status, body["detail"].as_str()), (StatusCode::NOT_FOUND, Some("File not found on server")));

        // Deleting the source removes its file.
        let (status, body) = app.delete(&format!("/api/sources/{source_id}")).await;
        assert_eq!((status, body["message"].as_str()), (StatusCode::OK, Some("Source deleted successfully")));
        assert!(!std::path::Path::new(&path).exists());
    }

    #[tokio::test]
    async fn rejected_uploads_leave_no_file_behind() {
        let app = TestApp::new().await;
        let uploads = app.state.config.uploads_dir();
        let files = || std::fs::read_dir(&uploads).map(|d| d.count()).unwrap_or(0);

        // Unknown notebook: 404 before anything is written.
        let (status, body) = send_multipart(
            &app,
            "/api/sources",
            &[("type", None, b"upload"), ("notebook_id", None, b"notebook:missing"), ("file", Some("a.txt"), b"x")],
        )
        .await;
        assert_eq!((status, body["detail"].as_str()), (StatusCode::NOT_FOUND, Some("Notebook notebook:missing not found")));
        // Unknown transformation: the stored file is removed again.
        let (status, body) = send_multipart(
            &app,
            "/api/sources",
            &[("type", None, b"upload"), ("transformations", None, b"[\"transformation:nope\"]"), ("file", Some("a.txt"), b"x")],
        )
        .await;
        assert_eq!(status, StatusCode::NOT_FOUND, "{body}");
        assert_eq!(files(), 0);
    }

    #[tokio::test]
    async fn input_guards_answer_like_python() {
        let app = TestApp::new().await;
        for (body, status, detail) in [
            (json!({ "type": "link" }), StatusCode::BAD_REQUEST, "URL is required for link type"),
            (json!({ "type": "text" }), StatusCode::BAD_REQUEST, "Content is required for text type"),
            (json!({ "type": "upload" }), StatusCode::BAD_REQUEST, "File upload or file_path is required for upload type"),
            (json!({ "type": "video", "content": "x" }), StatusCode::BAD_REQUEST, "Invalid source type. Must be link, upload, or text"),
            (
                json!({ "type": "upload", "file_path": "/etc/passwd" }),
                StatusCode::BAD_REQUEST,
                "Invalid file path: must be within the uploads directory",
            ),
            (
                json!({ "type": "upload", "file_path": "../../etc/passwd" }),
                StatusCode::BAD_REQUEST,
                "Invalid file path: must be within the uploads directory",
            ),
        ] {
            let (found, answer) = app.post("/api/sources/json", body.clone()).await;
            assert_eq!((found, answer["detail"].as_str()), (status, Some(detail)), "{body}");
        }
        let (status, body) = app.post("/api/sources/json", json!({ "type": "link", "url": "http://169.254.169.254/latest" })).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
        assert!(app.state.db.query("SELECT id FROM source", crate::db::Params::new()).await.unwrap().is_empty());

        // Array limits and the notebook_id / notebooks conflict are 422s.
        let many: Vec<String> = (0..51).map(|i| format!("notebook:{i}")).collect();
        let (status, body) = app.post("/api/sources/json", json!({ "type": "text", "content": "x", "notebooks": many })).await;
        assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY);
        assert!(body["detail"].as_str().unwrap().contains("at most 50 items"), "{body}");
        let (status, _) = app
            .post("/api/sources/json", json!({ "type": "text", "content": "x", "notebook_id": "notebook:a", "notebooks": [] }))
            .await;
        assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY);

        // Form fields: invalid JSON lists and a missing type.
        let (status, body) =
            send_multipart(&app, "/api/sources", &[("type", None, b"text"), ("content", None, b"x"), ("notebooks", None, b"[oops")]).await;
        assert_eq!((status, body["detail"].as_str()), (StatusCode::UNPROCESSABLE_ENTITY, Some("Invalid JSON in notebooks field")));
        let (status, body) = send_multipart(&app, "/api/sources", &[("content", None, b"x")]).await;
        assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY);
        assert!(body["detail"].is_array(), "{body}");
    }

    #[tokio::test]
    async fn listing_validates_its_parameters() {
        let app = TestApp::new().await;
        for (query, status) in [
            ("limit=0", StatusCode::UNPROCESSABLE_ENTITY),
            ("limit=101", StatusCode::UNPROCESSABLE_ENTITY),
            ("offset=-1", StatusCode::UNPROCESSABLE_ENTITY),
            ("limit=abc", StatusCode::UNPROCESSABLE_ENTITY),
            ("sort_by=bogus", StatusCode::BAD_REQUEST),
            ("sort_order=sideways", StatusCode::BAD_REQUEST),
            ("sort_by=title&sort_order=asc", StatusCode::OK),
            ("sort_by=type", StatusCode::OK),
        ] {
            let (found, body) = app.get(&format!("/api/sources?{query}")).await;
            assert_eq!(found, status, "{query}: {body}");
        }
        let (status, body) = app.get("/api/sources?notebook_id=notebook:missing").await;
        assert_eq!((status, body["detail"].as_str()), (StatusCode::NOT_FOUND, Some("Notebook not found")));
    }

    #[tokio::test]
    async fn missing_sources_are_404() {
        let app = TestApp::new().await;
        for path in [
            "/api/sources/source:missing",
            "/api/sources/source:missing/status",
            "/api/sources/source:missing/insights",
            "/api/sources/source:missing/download",
            "/api/sources/notebook:wrongtable",
        ] {
            let (status, body) = app.get(path).await;
            assert_eq!((status, body["detail"].as_str()), (StatusCode::NOT_FOUND, Some("Source not found")), "{path}");
        }
        let (status, _) = app.put("/api/sources/source:missing", json!({ "title": "x" })).await;
        assert_eq!(status, StatusCode::NOT_FOUND);
        let (status, _) = app.delete("/api/sources/source:missing").await;
        assert_eq!(status, StatusCode::NOT_FOUND);
        let (status, _) = app.post("/api/sources/source:missing/retry", json!({})).await;
        assert_eq!(status, StatusCode::NOT_FOUND);
        let (status, _) = app.post("/api/sources/source:missing/insights", json!({ "transformation_id": "transformation:x" })).await;
        assert_eq!(status, StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn update_retry_and_insight_requests() {
        let app = TestApp::new().await;
        mock_models(&app, &["An insight."]);
        let notebook_id = notebook(&app).await;
        let (status, body) =
            app.post("/api/sources/json", json!({ "type": "text", "content": "Body text.", "notebook_id": notebook_id })).await;
        assert_eq!(status, StatusCode::OK, "{body}");
        let source_id = body["id"].as_str().unwrap().to_string();
        assert_eq!(body["title"], "Processing...");

        let (status, body) = app.put(&format!("/api/sources/{source_id}"), json!({ "title": "Renamed", "topics": ["a"] })).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!((body["title"].as_str(), body["topics"].clone()), (Some("Renamed"), json!(["a"])));

        // A synchronously processed source has a completed command, so retry is allowed.
        let (status, body) = app.post(&format!("/api/sources/{source_id}/retry"), json!({})).await;
        assert_eq!(status, StatusCode::OK, "{body}");
        assert_eq!(body["status"], "queued");
        assert_eq!(app.wait_for_job(body["command_id"].as_str().unwrap()).await, "completed");

        let transformation_id = transformation(&app, "Key points").await;
        let (status, body) =
            app.post(&format!("/api/sources/{source_id}/insights"), json!({ "transformation_id": transformation_id })).await;
        assert_eq!(status, StatusCode::ACCEPTED, "{body}");
        assert_eq!(body["status"], "pending");
        assert_eq!(app.wait_for_job(body["command_id"].as_str().unwrap()).await, "completed");
        // The transformation queues the insight's own `create_insight` job.
        app.wait_for_idle().await;
        let (_, insights) = app.get(&format!("/api/sources/{source_id}/insights")).await;
        assert_eq!(insights[0]["content"], "An insight.", "{insights}");

        let (status, body) =
            app.post(&format!("/api/sources/{source_id}/insights"), json!({ "transformation_id": "transformation:missing" })).await;
        assert_eq!((status, body["detail"].as_str()), (StatusCode::NOT_FOUND, Some("Transformation not found")));

        // A source without text cannot get insights, and an orphan cannot be retried.
        let empty = Source::create(&app.state, &Source { title: Some("Empty".into()), ..Source::default() }).await.unwrap();
        let (status, body) =
            app.post(&format!("/api/sources/{}/insights", empty.id), json!({ "transformation_id": transformation_id })).await;
        assert_eq!((status, body["detail"].as_str()), (StatusCode::BAD_REQUEST, Some("Source has no text content")));
        let (status, body) = app.post(&format!("/api/sources/{}/retry", empty.id), json!({})).await;
        assert_eq!((status, body["detail"].as_str()), (StatusCode::BAD_REQUEST, Some("Source is not associated with any notebooks")));
    }

    #[tokio::test]
    async fn processing_a_deleted_source_fails_permanently() {
        let app = TestApp::new().await;
        let command_id = app
            .state
            .jobs
            .submit(
                "process_source",
                json!({ "source_id": "source:gone", "content_state": { "content": "x" }, "notebook_ids": [], "transformations": [], "embed": false }),
            )
            .await
            .unwrap();
        assert_eq!(app.wait_for_job(&command_id).await, "failed");
        let record = app.state.jobs.get(&command_id).await.unwrap().unwrap();
        assert!(record.error_message.unwrap_or_default().contains("no longer exists"));
    }
}
