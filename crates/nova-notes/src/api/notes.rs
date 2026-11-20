//! Notes: `/api/notes/...`.
//!
//! `POST /notes/upload` (multipart `file` + `notebook_id`) creates a note
//! from a dropped file: [`crate::services::notes::upload_note_file`].

use axum::body::Body;
use axum::extract::{Path, State};
use axum::http::{header, HeaderValue};
use axum::response::{IntoResponse, Response};
use axum::routing::get;
use axum::{Json, Router};
use serde::Deserialize;
use serde_json::{json, Value};

use super::notebooks::{lenient_bool, ApiJson, ApiQuery};
use super::sources::{multipart_problem, Rejection};
use crate::error::{AppError, AppResult};
use crate::services::notes::{self, NewNote, NoteUpdate, NoteView};
use crate::state::AppState;

pub fn router() -> Router<AppState> {
    Router::new()
        .route("/notes", get(list_notes).post(create_note))
        .route("/notes/upload", axum::routing::post(upload_note_file))
        .route("/notes/{note_id}/file", get(note_file))
        .route("/notes/{note_id}", get(get_note).put(update_note).delete(delete_note))
}

#[derive(Debug, Deserialize)]
struct ListQuery {
    notebook_id: Option<String>,
}

async fn list_notes(State(state): State<AppState>, ApiQuery(query): ApiQuery<ListQuery>) -> AppResult<Json<Vec<NoteView>>> {
    Ok(Json(notes::list_notes(&state, query.notebook_id.as_deref()).await?))
}

async fn create_note(State(state): State<AppState>, ApiJson(body): ApiJson<NewNote>) -> AppResult<Json<NoteView>> {
    Ok(Json(notes::create_note(&state, &body).await?))
}

/// `POST /notes/upload`: both fields are required (FastAPI's 422 otherwise).
async fn upload_note_file(State(state): State<AppState>, mut multipart: axum::extract::Multipart) -> Result<Json<NoteView>, Rejection> {
    let (mut file, mut notebook_id) = (None, None);
    while let Some(field) = multipart.next_field().await.map_err(multipart_problem)? {
        match field.name().unwrap_or_default() {
            "file" => {
                let filename = field.file_name().unwrap_or_default().to_string();
                file = Some((filename, field.bytes().await.map_err(multipart_problem)?.to_vec()));
            }
            "notebook_id" => notebook_id = Some(field.text().await.map_err(multipart_problem)?),
            _ => {}
        }
    }
    let Some((filename, data)) = file else {
        return Err(super::validation_error("body", "missing", "file: Field required").into());
    };
    let Some(notebook_id) = notebook_id else {
        return Err(super::validation_error("body", "missing", "notebook_id: Field required").into());
    };
    let upload = notes::NoteUpload { filename, data, notebook_id };
    Ok(Json(notes::upload_note_file(&state, &upload).await?))
}

async fn get_note(State(state): State<AppState>, Path(note_id): Path<String>) -> AppResult<Json<NoteView>> {
    Ok(Json(notes::get_note(&state, &note_id).await?))
}

async fn update_note(
    State(state): State<AppState>,
    Path(note_id): Path<String>,
    ApiJson(body): ApiJson<NoteUpdate>,
) -> AppResult<Json<NoteView>> {
    Ok(Json(notes::update_note(&state, &note_id, &body).await?))
}

async fn delete_note(State(state): State<AppState>, Path(note_id): Path<String>) -> AppResult<Json<Value>> {
    notes::delete_note(&state, &note_id).await?;
    Ok(Json(json!({ "message": "Note deleted successfully" })))
}

#[derive(Debug, Deserialize)]
struct FileQuery {
    #[serde(default, deserialize_with = "lenient_bool")]
    download: Option<bool>,
}

/// `Content-Disposition` the way Starlette's `FileResponse` writes it: a
/// quoted file name when it needs no escaping, RFC 5987 encoding otherwise.
fn content_disposition(kind: &str, filename: &str) -> String {
    let mut encoded = String::with_capacity(filename.len());
    for byte in filename.bytes() {
        if byte.is_ascii_alphanumeric() || b"_.-~/".contains(&byte) {
            encoded.push(byte as char);
        } else {
            encoded.push_str(&format!("%{byte:02X}"));
        }
    }
    if encoded == filename {
        format!("{kind}; filename=\"{filename}\"")
    } else {
        format!("{kind}; filename*=utf-8''{encoded}")
    }
}

fn header_value(value: &str) -> AppResult<HeaderValue> {
    HeaderValue::from_str(value).map_err(|e| AppError::internal(format!("Invalid header value: {e}")))
}

/// Serve the original file of a note created from a dropped or generated file.
async fn note_file(
    State(state): State<AppState>,
    Path(note_id): Path<String>,
    ApiQuery(query): ApiQuery<FileQuery>,
) -> AppResult<Response> {
    let file = notes::note_file(&state, &note_id).await?;
    let handle = tokio::fs::File::open(&file.path)
        .await
        .map_err(|_| AppError::not_found("The file is no longer on disk"))?;
    let length = handle.metadata().await?.len();

    let content_type = if file.mime.starts_with("text/") { format!("{}; charset=utf-8", file.mime) } else { file.mime.to_string() };
    let kind = if query.download.unwrap_or(false) { "attachment" } else { "inline" };
    let mut response = Body::from_stream(tokio_util::io::ReaderStream::new(handle)).into_response();
    let headers = response.headers_mut();
    headers.insert(header::CONTENT_TYPE, header_value(&content_type)?);
    headers.insert(header::CONTENT_LENGTH, HeaderValue::from(length));
    headers.insert(header::CONTENT_DISPOSITION, header_value(&content_disposition(kind, &file.filename))?);
    headers.insert(header::X_CONTENT_TYPE_OPTIONS, HeaderValue::from_static("nosniff"));
    if let Some(policy) = file.content_security_policy {
        headers.insert(header::CONTENT_SECURITY_POLICY, HeaderValue::from_static(policy));
    }
    Ok(response)
}

#[cfg(test)]
mod tests {
    use axum::body::Body;
    use axum::http::{Method, Request, StatusCode};
    use http_body_util::BodyExt;
    use serde_json::{json, Value};

    use super::content_disposition;
    use crate::domain::note::note_file_path;
    use crate::domain::{Note, Notebook};
    use crate::jobs::RetryPolicy;
    use crate::services::notes::{store_note_file, EXPLAINER_CSP_HEADER, INFOGRAPHIC_CSP_HEADER, SVG_MIME};
    use crate::testing::TestApp;

    /// The embedding service is another slice; stand in for its job so saves
    /// get a command id.
    async fn app_with_embed_job() -> TestApp {
        let app = TestApp::new().await;
        app.state.jobs.register("embed_note", RetryPolicy::once(), |_ctx, args| async move {
            Ok(json!({ "embedded": args["note_id"] }))
        });
        app
    }

    async fn notebook(app: &TestApp) -> String {
        Notebook::create(&app.state, "NB", "", None).await.unwrap().id
    }

    #[tokio::test]
    async fn create_a_note() {
        let app = app_with_embed_job().await;
        let nb = notebook(&app).await;

        let (status, body) =
            app.post("/api/notes", json!({ "title": "Idea", "content": "Some content", "notebook_id": nb })).await;
        assert_eq!(status, StatusCode::OK, "{body}");
        let id = body["id"].as_str().unwrap();
        assert!(id.starts_with("note:"));
        let mut keys: Vec<&String> = body.as_object().unwrap().keys().collect();
        keys.sort();
        assert_eq!(
            keys,
            ["attachment", "command_id", "content", "created", "explainer", "id", "infographic", "note_type", "title", "updated"]
        );
        assert_eq!((&body["title"], &body["content"]), (&json!("Idea"), &json!("Some content")));
        assert_eq!(body["note_type"], "human", "the default type");
        assert_eq!((&body["attachment"], &body["explainer"], &body["infographic"]), (&Value::Null, &json!(false), &json!(false)));
        assert!(chrono::DateTime::parse_from_rfc3339(body["created"].as_str().unwrap()).is_ok());

        // The save queued the embedding, and the note was filed in the notebook.
        let command_id = body["command_id"].as_str().unwrap();
        assert!(command_id.starts_with("command:"));
        assert_eq!(app.wait_for_job(command_id).await, "completed");
        let note = Note::get(&app.state, id).await.unwrap();
        assert_eq!(note.notebook_ids(&app.state).await.unwrap(), vec![nb.clone()]);

        // Without a notebook, an explicit null type, and an "ai" note with a title.
        let (status, body) = app.post("/api/notes", json!({ "content": "loose", "note_type": null })).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!((&body["note_type"], &body["title"]), (&Value::Null, &Value::Null));
        let (status, body) = app.post("/api/notes", json!({ "title": "T", "content": "x", "note_type": "ai" })).await;
        assert_eq!((status, &body["note_type"]), (StatusCode::OK, &json!("ai")));
    }

    #[tokio::test]
    async fn invalid_notes_are_rejected_before_saving() {
        let app = app_with_embed_job().await;
        for (body, status, detail) in [
            (json!({ "content": "x", "note_type": "robot" }), StatusCode::BAD_REQUEST, json!("note_type must be 'human' or 'ai'")),
            (json!({ "content": "   " }), StatusCode::BAD_REQUEST, json!("Note content cannot be empty")),
            (json!({ "content": "x", "notebook_id": "notebook:gone" }), StatusCode::NOT_FOUND, json!("Notebook not found")),
            (json!({ "content": "x", "notebook_id": "source:abc" }), StatusCode::NOT_FOUND, json!("Notebook not found")),
        ] {
            let (found, response) = app.post("/api/notes", body.clone()).await;
            assert_eq!(found, status, "{body}");
            assert_eq!(response["detail"], detail, "{body}");
        }
        let (status, body) = app.post("/api/notes", json!({ "title": "no content" })).await;
        assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY);
        assert_eq!(body["detail"][0]["loc"], json!(["body"]));

        // An untitled AI note needs a model to write its title, which this build cannot call yet.
        let (status, _) = app.post("/api/notes", json!({ "content": "an answer", "note_type": "ai" })).await;
        assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY);
        assert!(Note::list(&app.state).await.unwrap().is_empty());
    }

    #[tokio::test]
    async fn list_get_update_and_delete() {
        let app = app_with_embed_job().await;
        let nb = notebook(&app).await;
        let (_, filed) = app.post("/api/notes", json!({ "title": "Filed", "content": "in the notebook", "notebook_id": nb })).await;
        let (_, loose) = app.post("/api/notes", json!({ "title": "Loose", "content": "nowhere" })).await;
        let id = filed["id"].as_str().unwrap();
        let path = format!("/api/notes/{id}");

        let (status, all) = app.get("/api/notes").await;
        assert_eq!(status, StatusCode::OK);
        let titles: Vec<&str> = all.as_array().unwrap().iter().map(|n| n["title"].as_str().unwrap()).collect();
        assert_eq!(titles, ["Loose", "Filed"], "most recently updated first");
        assert_eq!(all[1]["command_id"], Value::Null, "only a save reports a command id");

        let (status, in_notebook) = app.get(&format!("/api/notes?notebook_id={nb}")).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(in_notebook.as_array().unwrap().len(), 1);
        assert_eq!(in_notebook[0]["id"], filed["id"]);
        let (status, body) = app.get("/api/notes?notebook_id=notebook:gone").await;
        assert_eq!((status, &body["detail"]), (StatusCode::NOT_FOUND, &json!("Notebook not found")));

        let (status, body) = app.get(&path).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!((&body["content"], &body["command_id"]), (&json!("in the notebook"), &Value::Null));

        // Only what is sent changes; the save re-queues the embedding.
        let (status, body) = app.put(&path, json!({ "content": "Updated content" })).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!((&body["title"], &body["content"]), (&json!("Filed"), &json!("Updated content")));
        assert!(body["command_id"].as_str().unwrap().starts_with("command:"));
        let (_, body) = app.put(&path, json!({ "title": "Renamed", "note_type": "ai" })).await;
        assert_eq!((&body["title"], &body["note_type"], &body["content"]), (&json!("Renamed"), &json!("ai"), &json!("Updated content")));
        let (status, body) = app.put(&path, json!({ "note_type": "robot" })).await;
        assert_eq!((status, &body["detail"]), (StatusCode::BAD_REQUEST, &json!("note_type must be 'human' or 'ai'")));
        let (status, body) = app.put(&path, json!({ "content": "" })).await;
        assert_eq!((status, &body["detail"]), (StatusCode::BAD_REQUEST, &json!("Note content cannot be empty")));
        assert_eq!(Note::get(&app.state, id).await.unwrap().note_type.as_deref(), Some("ai"));

        let (status, body) = app.delete(&path).await;
        assert_eq!((status, body), (StatusCode::OK, json!({ "message": "Note deleted successfully" })));
        assert!(Note::get(&app.state, id).await.unwrap_err().is_not_found());
        assert!(Note::get(&app.state, loose["id"].as_str().unwrap()).await.is_ok());
        let (_, in_notebook) = app.get(&format!("/api/notes?notebook_id={nb}")).await;
        assert_eq!(in_notebook, json!([]));
    }

    #[tokio::test]
    async fn missing_notes_are_404() {
        let app = TestApp::new().await;
        let nb = notebook(&app).await;
        for gone in ["note:gone", nb.as_str()] {
            let path = format!("/api/notes/{gone}");
            for (status, body) in [
                app.get(&path).await,
                app.put(&path, json!({ "content": "x" })).await,
                app.delete(&path).await,
                app.get(&format!("{path}/file")).await,
            ] {
                assert_eq!(status, StatusCode::NOT_FOUND, "{gone}");
                assert_eq!(body["detail"], "Note not found", "{gone}");
            }
        }
        assert!(Notebook::get(&app.state, &nb).await.is_ok(), "a notebook id never deletes the notebook");
    }

    /// A note with a stored original, as an upload or a generator leaves it.
    async fn file_note(app: &TestApp, name: &str, data: &[u8], mime: &str, explainer: bool) -> Note {
        let ext = name.rfind('.').map(|i| &name[i..]).unwrap_or("");
        let stored = store_note_file(&app.state.config, data, ext).await.unwrap();
        let mut note = Note {
            title: Some(name.into()),
            content: Some("text".into()),
            note_type: Some("human".into()),
            attachment_name: Some(name.into()),
            attachment_file: Some(stored),
            attachment_mime: Some(mime.into()),
            attachment_size: Some(data.len() as i64),
            explainer: explainer.then_some(true),
            ..Note::default()
        };
        note.save(&app.state).await.unwrap();
        note
    }

    async fn fetch(app: &TestApp, path: &str) -> (StatusCode, axum::http::HeaderMap, Vec<u8>) {
        let response = app.send(Request::builder().method(Method::GET).uri(path).body(Body::empty()).unwrap()).await;
        let (status, headers) = (response.status(), response.headers().clone());
        (status, headers, response.into_body().collect().await.unwrap().to_bytes().to_vec())
    }

    #[tokio::test]
    async fn original_files_are_served_safely() {
        let app = TestApp::new().await;

        let pdf = file_note(&app, "a.pdf", b"%PDF-1.4 body", "application/pdf", false).await;
        let (_, note) = app.get(&format!("/api/notes/{}", pdf.id)).await;
        assert_eq!(note["attachment"], json!({ "name": "a.pdf", "type": "application/pdf", "size": 13 }));
        let (status, headers, bytes) = fetch(&app, &format!("/api/notes/{}/file", pdf.id)).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(headers["content-type"], "application/pdf");
        assert_eq!(headers["x-content-type-options"], "nosniff");
        assert_eq!(headers["content-disposition"], "inline; filename=\"a.pdf\"");
        assert_eq!(headers["content-length"], "13");
        assert!(headers.get("content-security-policy").is_none());
        assert_eq!(bytes, b"%PDF-1.4 body");

        let html = file_note(&app, "page.html", b"<p>x</p>", "text/html", false).await;
        let (_, headers, _) = fetch(&app, &format!("/api/notes/{}/file", html.id)).await;
        assert_eq!(headers["content-security-policy"], "sandbox");
        assert_eq!(headers["content-type"], "text/html; charset=utf-8");

        let explainer = file_note(&app, "explainer.html", b"<script></script>", "text/html", true).await;
        let (_, note) = app.get(&format!("/api/notes/{}", explainer.id)).await;
        assert_eq!((&note["explainer"], &note["infographic"]), (&json!(true), &json!(false)));
        let (_, headers, _) = fetch(&app, &format!("/api/notes/{}/file", explainer.id)).await;
        assert_eq!(headers["content-security-policy"], EXPLAINER_CSP_HEADER);

        let svg = file_note(&app, "poster.svg", b"<svg/>", SVG_MIME, false).await;
        let (_, note) = app.get(&format!("/api/notes/{}", svg.id)).await;
        assert_eq!(note["infographic"], true);
        let (_, headers, _) = fetch(&app, &format!("/api/notes/{}/file", svg.id)).await;
        assert_eq!(headers["content-type"], SVG_MIME);
        assert_eq!(headers["content-security-policy"], INFOGRAPHIC_CSP_HEADER);

        // The type comes from the stored extension, not the (tampered) record.
        let tampered = file_note(&app, "a.txt", b"hi", "text/html", false).await;
        let (_, headers, _) = fetch(&app, &format!("/api/notes/{}/file?download=true", tampered.id)).await;
        assert_eq!(headers["content-type"], "text/plain; charset=utf-8");
        assert!(headers.get("content-security-policy").is_none());
        assert_eq!(headers["content-disposition"], "attachment; filename=\"a.txt\"");

        // Deleting the note removes the original.
        let path = note_file_path(&app.state.config, pdf.attachment_file.as_deref().unwrap()).unwrap();
        assert!(path.is_file());
        assert_eq!(app.delete(&format!("/api/notes/{}", pdf.id)).await.0, StatusCode::OK);
        assert!(!path.exists());
    }

    #[tokio::test]
    async fn missing_or_unsafe_files_are_refused() {
        let app = TestApp::new().await;
        let plain = Note::create(&app.state, Some("t"), Some("c"), Some("human")).await.unwrap();
        let (status, body) = app.get(&format!("/api/notes/{}/file", plain.id)).await;
        assert_eq!((status, &body["detail"]), (StatusCode::NOT_FOUND, &json!("This note has no file")));
        let (_, body) = app.get(&format!("/api/notes/{}", plain.id)).await;
        assert_eq!(body["attachment"], Value::Null);

        let gone = file_note(&app, "a.txt", b"hi", "text/plain", false).await;
        std::fs::remove_file(note_file_path(&app.state.config, gone.attachment_file.as_deref().unwrap()).unwrap()).unwrap();
        let (status, body) = app.get(&format!("/api/notes/{}/file", gone.id)).await;
        assert_eq!((status, &body["detail"]), (StatusCode::NOT_FOUND, &json!("The file is no longer on disk")));

        for bad in ["../../etc/passwd", "/etc/passwd", "a.txt", &"x".repeat(32)] {
            let mut note = Note::create(&app.state, Some("t"), Some("c"), Some("human")).await.unwrap();
            note.attachment_name = Some("x.txt".into());
            note.attachment_file = Some(bad.to_string());
            note.save(&app.state).await.unwrap();
            let (status, body) = app.get(&format!("/api/notes/{}/file", note.id)).await;
            assert_eq!(status, StatusCode::BAD_REQUEST, "{bad}");
            assert_eq!(body["detail"], "Invalid stored file name");
        }
    }

    #[test]
    fn file_names_are_escaped_in_the_disposition() {
        assert_eq!(content_disposition("inline", "report-v1.pdf"), "inline; filename=\"report-v1.pdf\"");
        assert_eq!(content_disposition("attachment", "my notes.md"), "attachment; filename*=utf-8''my%20notes.md");
        assert_eq!(content_disposition("inline", "a\"b.txt"), "inline; filename*=utf-8''a%22b.txt");
        assert_eq!(content_disposition("inline", "café.txt"), "inline; filename*=utf-8''caf%C3%A9.txt");
    }
    async fn upload(app: &TestApp, parts: &[(&str, Option<&str>, &[u8])]) -> (StatusCode, Value) {
        let (content_type, body) = crate::testing::multipart_body(parts);
        let request = Request::builder()
            .method(Method::POST)
            .uri("/api/notes/upload")
            .header("content-type", content_type)
            .body(Body::from(body))
            .unwrap();
        let response = app.send(request).await;
        let status = response.status();
        let bytes = response.into_body().collect().await.unwrap().to_bytes();
        (status, serde_json::from_slice(&bytes).unwrap_or(Value::Null))
    }

    #[tokio::test]
    async fn dropped_files_become_notes() {
        let app = app_with_embed_job().await;
        let notebook_id = notebook(&app).await;
        let nb = notebook_id.as_bytes();

        let (status, body) = upload(&app, &[("notebook_id", None, nb), ("file", Some("../dir/notes.md"), b"# Hello\n\nWorld")]).await;
        assert_eq!(status, StatusCode::OK, "{body}");
        assert_eq!(body["title"], "notes.md");
        assert!(body["content"].as_str().unwrap().contains("World"), "{body}");
        assert_eq!(body["attachment"], json!({ "name": "notes.md", "type": "text/markdown", "size": 14 }));
        let (status, notes) = app.get(&format!("/api/notes?notebook_id={notebook_id}")).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(notes.as_array().unwrap().len(), 1);

        let (status, body) = upload(&app, &[("notebook_id", None, nb), ("file", Some("run.exe"), b"MZ")]).await;
        assert_eq!(status, StatusCode::UNSUPPORTED_MEDIA_TYPE, "{body}");
        let (status, body) = upload(&app, &[("file", Some("a.txt"), b"x")]).await;
        assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY, "{body}");
        let (status, body) = upload(&app, &[("notebook_id", None, b"notebook:missing"), ("file", Some("a.txt"), b"x")]).await;
        assert_eq!((status, body["detail"].as_str()), (StatusCode::NOT_FOUND, Some("Notebook not found")));
    }
}
