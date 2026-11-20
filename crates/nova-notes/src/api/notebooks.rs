//! Notebooks: `/api/notebooks/...` and `/api/recently-viewed`.
//!
//! Also home to the request extractors the organisational routers share
//! ([`ApiJson`], [`ApiQuery`]): they answer a malformed body or query string
//! the way FastAPI does, with a 422 and a `{"detail": [...]}` body.

use axum::extract::rejection::{JsonRejection, QueryRejection};
use axum::extract::{FromRequest, FromRequestParts, Path, Query, Request, State};
use axum::http::request::Parts;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use serde::de::{DeserializeOwned, Error as _};
use serde::{Deserialize, Deserializer};
use serde_json::{json, Value};

use crate::error::AppResult;
use crate::services::discovery::{self, DiscoverNotebookRequest, DiscoverNotebookResponse};
use crate::services::notebooks::{
    self, NewNotebook, NotebookDeletePreview, NotebookDeleted, NotebookQuery, NotebookSummary, NotebookUpdate,
    RecentlyViewed,
};
use crate::state::AppState;

pub fn router() -> Router<AppState> {
    Router::new()
        .route("/notebooks", get(list_notebooks).post(create_notebook))
        .route("/recently-viewed", get(recently_viewed))
        // Before `/notebooks/{notebook_id}` so it is not taken for an id.
        .route("/notebooks/discover", post(discover_notebook))
        .route("/notebooks/{notebook_id}/delete-preview", get(delete_preview))
        .route("/notebooks/{notebook_id}", get(get_notebook).put(update_notebook).delete(delete_notebook))
        .route("/notebooks/{notebook_id}/sources/{source_id}", post(add_source).delete(remove_source))
}

// ---------------------------------------------------------------------------
// Request extraction
// ---------------------------------------------------------------------------

/// The 422 FastAPI answers a request that fails validation with: `detail` is
/// a list of `{type, loc, msg}` problems.
pub(crate) fn validation_error(location: &str, kind: &str, message: impl Into<String>) -> Response {
    let detail = json!([{ "type": kind, "loc": [location], "msg": message.into() }]);
    (StatusCode::UNPROCESSABLE_ENTITY, Json(json!({ "detail": detail }))).into_response()
}

/// A JSON request body. Like `axum::Json`, but a missing, malformed or
/// ill-typed body is a FastAPI-style 422.
pub(crate) struct ApiJson<T>(pub T);

impl<S, T> FromRequest<S> for ApiJson<T>
where
    S: Send + Sync,
    T: DeserializeOwned,
{
    type Rejection = Response;

    async fn from_request(request: Request, state: &S) -> Result<Self, Self::Rejection> {
        match Json::<T>::from_request(request, state).await {
            Ok(Json(body)) => Ok(Self(body)),
            // An oversized body keeps its own status.
            Err(rejection) if rejection.status() == StatusCode::PAYLOAD_TOO_LARGE => Err(rejection.into_response()),
            Err(rejection) => {
                let kind = if matches!(rejection, JsonRejection::JsonSyntaxError(_)) { "json_invalid" } else { "value_error" };
                Err(validation_error("body", kind, rejection.body_text()))
            }
        }
    }
}

/// Query-string parameters. Like `axum::extract::Query`, but a value that
/// does not parse is a FastAPI-style 422.
pub(crate) struct ApiQuery<T>(pub T);

impl<S, T> FromRequestParts<S> for ApiQuery<T>
where
    S: Send + Sync,
    T: DeserializeOwned,
{
    type Rejection = Response;

    async fn from_request_parts(parts: &mut Parts, state: &S) -> Result<Self, Self::Rejection> {
        match Query::<T>::from_request_parts(parts, state).await {
            Ok(Query(query)) => Ok(Self(query)),
            Err(rejection @ QueryRejection::FailedToDeserializeQueryString(_)) => {
                Err(validation_error("query", "value_error", rejection.body_text()))
            }
            Err(rejection) => Err(rejection.into_response()),
        }
    }
}

/// A boolean query parameter, accepting the spellings FastAPI does.
pub(crate) fn lenient_bool<'de, D>(deserializer: D) -> Result<Option<bool>, D::Error>
where
    D: Deserializer<'de>,
{
    let Some(raw) = Option::<String>::deserialize(deserializer)? else {
        return Ok(None);
    };
    match raw.trim().to_ascii_lowercase().as_str() {
        "true" | "1" | "yes" | "on" | "t" | "y" => Ok(Some(true)),
        "false" | "0" | "no" | "off" | "f" | "n" => Ok(Some(false)),
        _ => Err(D::Error::custom("Input should be a valid boolean, unable to interpret input")),
    }
}

// ---------------------------------------------------------------------------
// Handlers
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
struct ListQuery {
    #[serde(default, deserialize_with = "lenient_bool")]
    archived: Option<bool>,
    order_by: Option<String>,
    folder_id: Option<String>,
}

async fn list_notebooks(State(state): State<AppState>, ApiQuery(query): ApiQuery<ListQuery>) -> AppResult<Json<Vec<NotebookSummary>>> {
    let query = NotebookQuery { archived: query.archived, order_by: query.order_by, folder_id: query.folder_id };
    Ok(Json(notebooks::list_notebooks(&state, &query).await?))
}

async fn create_notebook(State(state): State<AppState>, ApiJson(body): ApiJson<NewNotebook>) -> AppResult<Json<NotebookSummary>> {
    Ok(Json(notebooks::create_notebook(&state, &body).await?))
}

/// `POST /notebooks/discover`: a 422 for a request the model rejects, then the service.
async fn discover_notebook(State(state): State<AppState>, ApiJson(body): ApiJson<DiscoverNotebookRequest>) -> Response {
    let request = match body.validated() {
        Ok(request) => request,
        Err(e) => return validation_error("body", e.kind, e.message),
    };
    let started: AppResult<DiscoverNotebookResponse> = discovery::discover_notebook(&state, &request).await;
    started.map(Json).into_response()
}

fn default_limit() -> usize {
    12
}

fn limit_1_to_50<'de, D>(deserializer: D) -> Result<usize, D::Error>
where
    D: Deserializer<'de>,
{
    let raw = String::deserialize(deserializer)?;
    match raw.trim().parse::<i64>() {
        Err(_) => Err(D::Error::custom("limit: Input should be a valid integer, unable to parse string as an integer")),
        Ok(limit) if limit < 1 => Err(D::Error::custom("limit: Input should be greater than or equal to 1")),
        Ok(limit) if limit > 50 => Err(D::Error::custom("limit: Input should be less than or equal to 50")),
        Ok(limit) => Ok(limit as usize),
    }
}

#[derive(Debug, Deserialize)]
struct RecentQuery {
    #[serde(default = "default_limit", deserialize_with = "limit_1_to_50")]
    limit: usize,
}

async fn recently_viewed(State(state): State<AppState>, ApiQuery(query): ApiQuery<RecentQuery>) -> AppResult<Json<Vec<RecentlyViewed>>> {
    Ok(Json(notebooks::recently_viewed(&state, query.limit).await?))
}

async fn delete_preview(State(state): State<AppState>, Path(notebook_id): Path<String>) -> AppResult<Json<NotebookDeletePreview>> {
    Ok(Json(notebooks::notebook_delete_preview(&state, &notebook_id).await?))
}

async fn get_notebook(State(state): State<AppState>, Path(notebook_id): Path<String>) -> AppResult<Json<NotebookSummary>> {
    Ok(Json(notebooks::view_notebook(&state, &notebook_id).await?))
}

async fn update_notebook(
    State(state): State<AppState>,
    Path(notebook_id): Path<String>,
    ApiJson(body): ApiJson<NotebookUpdate>,
) -> AppResult<Json<NotebookSummary>> {
    Ok(Json(notebooks::update_notebook(&state, &notebook_id, &body).await?))
}

#[derive(Debug, Deserialize)]
struct DeleteQuery {
    #[serde(default, deserialize_with = "lenient_bool")]
    delete_exclusive_sources: Option<bool>,
}

async fn delete_notebook(
    State(state): State<AppState>,
    Path(notebook_id): Path<String>,
    ApiQuery(query): ApiQuery<DeleteQuery>,
) -> AppResult<Json<NotebookDeleted>> {
    let delete_exclusive_sources = query.delete_exclusive_sources.unwrap_or(false);
    Ok(Json(notebooks::delete_notebook(&state, &notebook_id, delete_exclusive_sources).await?))
}

async fn add_source(State(state): State<AppState>, Path((notebook_id, source_id)): Path<(String, String)>) -> AppResult<Json<Value>> {
    notebooks::add_source_to_notebook(&state, &notebook_id, &source_id).await?;
    Ok(Json(json!({ "message": "Source linked to notebook successfully" })))
}

async fn remove_source(State(state): State<AppState>, Path((notebook_id, source_id)): Path<(String, String)>) -> AppResult<Json<Value>> {
    notebooks::remove_source_from_notebook(&state, &notebook_id, &source_id).await?;
    Ok(Json(json!({ "message": "Source removed from notebook successfully" })))
}

#[cfg(test)]
mod tests {
    use axum::http::StatusCode;
    use serde_json::{json, Value};

    use crate::db::Params;
    use crate::domain::{ChatSession, Note, Notebook, Source};
    use crate::testing::TestApp;

    async fn notebook(app: &TestApp, name: &str) -> String {
        let (status, body) = app.post("/api/notebooks", json!({ "name": name })).await;
        assert_eq!(status, StatusCode::OK, "{body}");
        body["id"].as_str().unwrap().to_string()
    }

    async fn source(app: &TestApp, title: &str) -> Source {
        Source::create(&app.state, &Source { title: Some(title.into()), full_text: Some("text".into()), ..Source::default() })
            .await
            .unwrap()
    }

    fn names(body: &Value) -> Vec<&str> {
        body.as_array().unwrap().iter().map(|nb| nb["name"].as_str().unwrap()).collect()
    }

    #[tokio::test]
    async fn create_get_and_update_a_notebook() {
        let app = TestApp::new().await;

        let (status, created) = app.post("/api/notebooks", json!({ "name": "Research", "description": "About things" })).await;
        assert_eq!(status, StatusCode::OK);
        let id = created["id"].as_str().unwrap();
        assert!(id.starts_with("notebook:"));
        assert!(chrono::DateTime::parse_from_rfc3339(created["created"].as_str().unwrap()).is_ok());
        assert!(chrono::DateTime::parse_from_rfc3339(created["updated"].as_str().unwrap()).is_ok());
        let mut expected = created.clone();
        for (key, value) in [
            ("name", json!("Research")),
            ("description", json!("About things")),
            ("archived", json!(false)),
            ("source_count", json!(0)),
            ("note_count", json!(0)),
            ("folder_id", Value::Null),
        ] {
            assert_eq!(created[key], value, "{key}");
            expected.as_object_mut().unwrap().remove(key);
        }
        // Exactly the NotebookResponse fields.
        let mut rest: Vec<&String> = expected.as_object().unwrap().keys().collect();
        rest.sort();
        assert_eq!(rest, ["created", "id", "updated"]);

        // The description is optional; the name is not.
        let (status, body) = app.post("/api/notebooks", json!({ "name": "Bare" })).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(body["description"], "");
        for name in ["", "   "] {
            let (status, body) = app.post("/api/notebooks", json!({ "name": name })).await;
            assert_eq!(status, StatusCode::BAD_REQUEST);
            assert_eq!(body["detail"], "Notebook name cannot be empty");
        }
        let (status, body) = app.post("/api/notebooks", json!({ "description": "no name" })).await;
        assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY);
        assert_eq!(body["detail"][0]["loc"], json!(["body"]));
        assert!(body["detail"][0]["msg"].as_str().unwrap().contains("name"), "{body}");

        // Counts follow the notebook's sources and notes.
        source(&app, "Paper").await.add_to_notebook(&app.state, id).await.unwrap();
        let note = Note::create(&app.state, Some("N"), Some("note content"), Some("human")).await.unwrap();
        note.add_to_notebook(&app.state, id).await.unwrap();
        let (status, fetched) = app.get(&format!("/api/notebooks/{id}")).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!((fetched["source_count"].as_i64(), fetched["note_count"].as_i64()), (Some(1), Some(1)));
        assert_eq!(fetched["name"], "Research");

        // Update only what is sent.
        let (status, updated) = app.put(&format!("/api/notebooks/{id}"), json!({ "name": "Renamed", "archived": true })).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(updated["name"], "Renamed");
        assert_eq!(updated["archived"], true);
        assert_eq!(updated["description"], "About things", "an absent description is left alone");
        assert_eq!(updated["source_count"], 1);

        // "" and null both clear the description.
        for (payload, expected) in [
            (json!({ "description": "new text" }), "new text"),
            (json!({ "description": "" }), ""),
            (json!({ "description": "again" }), "again"),
            (json!({ "description": null }), ""),
        ] {
            let (status, body) = app.put(&format!("/api/notebooks/{id}"), payload).await;
            assert_eq!(status, StatusCode::OK);
            assert_eq!(body["description"], expected);
            assert_eq!(Notebook::get(&app.state, id).await.unwrap().description, expected);
        }
        let (status, body) = app.put(&format!("/api/notebooks/{id}"), json!({ "name": "  " })).await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert_eq!(body["detail"], "Notebook name cannot be empty");
        assert_eq!(Notebook::get(&app.state, id).await.unwrap().name, "Renamed");
    }

    #[tokio::test]
    async fn list_filters_and_orders() {
        let app = TestApp::new().await;
        let (_, folder) = app.post("/api/folders", json!({ "name": "Work" })).await;
        let folder_id = folder["id"].as_str().unwrap();

        let beta = notebook(&app, "beta").await;
        let alpha = notebook(&app, "Alpha").await;
        let gamma = notebook(&app, "gamma").await;
        app.put(&format!("/api/notebooks/{alpha}"), json!({ "archived": true })).await;
        app.put(&format!("/api/notebooks/{gamma}"), json!({ "folder_id": folder_id })).await;
        source(&app, "S").await.add_to_notebook(&app.state, &beta).await.unwrap();

        // Default order: most recently updated first.
        let (status, body) = app.get("/api/notebooks").await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(names(&body), ["gamma", "Alpha", "beta"]);
        assert_eq!(body[2]["source_count"], 1);
        assert_eq!(body[0]["folder_id"], folder_id);
        assert_eq!(body[1]["folder_id"], Value::Null);

        for (query, expected) in [
            ("?order_by=created", vec!["beta", "Alpha", "gamma"]),
            ("?order_by=created%20desc", vec!["gamma", "Alpha", "beta"]),
            ("?order_by=%20%20Created%20%20%20DESC%20", vec!["gamma", "Alpha", "beta"]),
            ("?archived=true", vec!["Alpha"]),
            ("?archived=false", vec!["gamma", "beta"]),
            ("?archived=0", vec!["gamma", "beta"]),
            ("?folder_id=none", vec!["Alpha", "beta"]),
            ("?folder_id=folder:zzz", vec![]),
        ] {
            let (status, body) = app.get(&format!("/api/notebooks{query}")).await;
            assert_eq!(status, StatusCode::OK, "{query}");
            assert_eq!(names(&body), expected, "{query}");
        }
        let (_, body) = app.get(&format!("/api/notebooks?folder_id={folder_id}&order_by=name%20asc")).await;
        assert_eq!(names(&body), ["gamma"]);

        for order_by in [
            "updated;%20DROP%20TABLE%20notebook",
            "name)%20FETCH%20(SELECT%20*%20FROM%20credential",
            "updated%20LIMIT%201",
            "name%20desc%20extra",
            "archived%20desc",
            "id",
            "name%20asc,%20updated%20desc",
            "updated%20sideways",
            "",
        ] {
            let (status, body) = app.get(&format!("/api/notebooks?order_by={order_by}")).await;
            assert_eq!(status, StatusCode::BAD_REQUEST, "{order_by}");
            let detail = body["detail"].as_str().unwrap();
            assert!(detail.starts_with("Invalid order_by: '"), "{detail}");
            assert!(detail.contains("Allowed fields: created, name, updated"), "{detail}");
        }
        let (status, body) = app.get("/api/notebooks?archived=maybe").await;
        assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY);
        assert_eq!(body["detail"][0]["loc"], json!(["query"]));
        assert_eq!(Notebook::list(&app.state, None).await.unwrap().len(), 3, "nothing was dropped");
    }

    #[tokio::test]
    async fn notebooks_move_between_folders() {
        let app = TestApp::new().await;
        let (_, folder) = app.post("/api/folders", json!({ "name": "Work" })).await;
        let folder_id = folder["id"].as_str().unwrap();
        let id = notebook(&app, "NB").await;
        let path = format!("/api/notebooks/{id}");

        let (status, body) = app.put(&path, json!({ "folder_id": folder_id })).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(body["folder_id"], folder_id);
        let (_, body) = app.get(&path).await;
        assert_eq!(body["folder_id"], folder_id);

        // An absent folder_id leaves the folder alone.
        let (_, body) = app.put(&path, json!({ "name": "Renamed" })).await;
        assert_eq!(body["folder_id"], folder_id);

        // Unknown folders and ids of other tables are 404 and change nothing.
        for target in ["folder:gone", id.as_str(), "not-an-id"] {
            let (status, body) = app.put(&path, json!({ "folder_id": target, "name": "Should not stick" })).await;
            assert_eq!(status, StatusCode::NOT_FOUND, "{target}");
            assert_eq!(body["detail"], "Folder not found");
        }
        let stored = Notebook::get(&app.state, &id).await.unwrap();
        assert_eq!((stored.name.as_str(), stored.folder.as_deref()), ("Renamed", Some(folder_id)));

        // null and "" both take the notebook out of its folder.
        for payload in [json!({ "folder_id": null }), json!({ "folder_id": "" })] {
            app.put(&path, json!({ "folder_id": folder_id })).await;
            let (status, body) = app.put(&path, payload).await;
            assert_eq!(status, StatusCode::OK);
            assert_eq!(body["folder_id"], Value::Null);
            assert_eq!(Notebook::get(&app.state, &id).await.unwrap().folder, None);
        }
    }

    #[tokio::test]
    async fn missing_notebooks_are_404() {
        let app = TestApp::new().await;
        let existing = source(&app, "S").await;
        let id = notebook(&app, "Real").await;

        // A missing id, and an id of another table, for every notebook route.
        for gone in ["notebook:gone", existing.id.as_str()] {
            let path = format!("/api/notebooks/{gone}");
            let source_path = format!("{path}/sources/{}", existing.id);
            let results = [
                (app.get(&path).await, "Notebook not found"),
                (app.put(&path, json!({ "name": "x" })).await, "Notebook not found"),
                (app.delete(&path).await, "Notebook not found"),
                (app.get(&format!("{path}/delete-preview")).await, "Notebook not found"),
                (app.post(&source_path, Value::Null).await, "Notebook or source not found"),
                (app.delete(&source_path).await, "Notebook not found"),
            ];
            for ((status, body), detail) in results {
                assert_eq!(status, StatusCode::NOT_FOUND, "{gone}: {body}");
                assert_eq!(body["detail"], detail, "{gone}");
            }
        }
        // The source was not mistaken for a notebook and deleted.
        assert!(Source::get(&app.state, &existing.id).await.is_ok());

        let (status, body) = app.post(&format!("/api/notebooks/{id}/sources/source:gone"), Value::Null).await;
        assert_eq!(status, StatusCode::NOT_FOUND);
        assert_eq!(body["detail"], "Notebook or source not found");
        let (status, _) = app.post(&format!("/api/notebooks/{id}/sources/{id}"), Value::Null).await;
        assert_eq!(status, StatusCode::NOT_FOUND, "a notebook is not a source");
    }

    #[tokio::test]
    async fn sources_link_and_unlink() {
        let app = TestApp::new().await;
        let id = notebook(&app, "NB").await;
        let paper = source(&app, "Paper").await;
        let path = format!("/api/notebooks/{id}/sources/{}", paper.id);

        // Linking twice still leaves one link.
        for _ in 0..2 {
            let (status, body) = app.post(&path, Value::Null).await;
            assert_eq!(status, StatusCode::OK);
            assert_eq!(body, json!({ "message": "Source linked to notebook successfully" }));
        }
        assert_eq!(paper.notebook_ids(&app.state).await.unwrap(), vec![id.clone()]);
        let (_, body) = app.get(&format!("/api/notebooks/{id}")).await;
        assert_eq!(body["source_count"], 1);

        let (status, body) = app.delete(&path).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(body, json!({ "message": "Source removed from notebook successfully" }));
        assert!(paper.notebook_ids(&app.state).await.unwrap().is_empty());
        assert!(Source::get(&app.state, &paper.id).await.is_ok(), "the source itself is kept");
        // Unlinking something that is not linked is not an error.
        let (status, _) = app.delete(&path).await;
        assert_eq!(status, StatusCode::OK);
    }

    #[tokio::test]
    async fn delete_previews_and_cascades() {
        let app = TestApp::new().await;
        let doomed = notebook(&app, "Doomed").await;
        let survivor = notebook(&app, "Survivor").await;

        let exclusive = source(&app, "Exclusive").await;
        exclusive.add_to_notebook(&app.state, &doomed).await.unwrap();
        let shared = source(&app, "Shared").await;
        shared.add_to_notebook(&app.state, &doomed).await.unwrap();
        shared.add_to_notebook(&app.state, &survivor).await.unwrap();
        let note = Note::create(&app.state, Some("N"), Some("note content"), None).await.unwrap();
        note.add_to_notebook(&app.state, &doomed).await.unwrap();
        let session = ChatSession::create(&app.state, Some("Chat"), None).await.unwrap();
        session.relate_to(&app.state, &doomed).await.unwrap();

        let (status, preview) = app.get(&format!("/api/notebooks/{doomed}/delete-preview")).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(
            preview,
            json!({
                "notebook_id": doomed,
                "notebook_name": "Doomed",
                "note_count": 1,
                "exclusive_source_count": 1,
                "shared_source_count": 1,
            })
        );

        let (status, deleted) = app.delete(&format!("/api/notebooks/{doomed}?delete_exclusive_sources=true")).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(
            deleted,
            json!({
                "message": "Notebook deleted successfully",
                "deleted_notes": 1,
                "deleted_sources": 1,
                "unlinked_sources": 1,
                "deleted_chat_sessions": 1,
            })
        );
        assert_eq!(app.get(&format!("/api/notebooks/{doomed}")).await.0, StatusCode::NOT_FOUND);
        assert!(Source::get(&app.state, &exclusive.id).await.unwrap_err().is_not_found());
        assert!(Source::get(&app.state, &shared.id).await.is_ok());
        assert!(Note::get(&app.state, &note.id).await.unwrap_err().is_not_found());

        // By default the notebook's sources are only unlinked.
        let (status, deleted) = app.delete(&format!("/api/notebooks/{survivor}")).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!((deleted["deleted_sources"].as_i64(), deleted["unlinked_sources"].as_i64()), (Some(0), Some(1)));
        assert!(Source::get(&app.state, &shared.id).await.is_ok());
        let (_, all) = app.get("/api/notebooks").await;
        assert_eq!(all, json!([]));
    }

    #[tokio::test]
    async fn recently_viewed_mixes_notebooks_and_sources() {
        let app = TestApp::new().await;
        let (status, body) = app.get("/api/recently-viewed").await;
        assert_eq!((status, body), (StatusCode::OK, json!([])));

        let first = notebook(&app, "First").await;
        let unseen = notebook(&app, "Never opened").await;
        let paper = source(&app, "Paper").await;
        let untitled = Source::create(&app.state, &Source::default()).await.unwrap();
        let view_source = |id: String| {
            let state = app.state.clone();
            async move {
                state.db.query("UPDATE $id SET last_viewed_at = time::now()", Params::new().record("id", &id)).await.unwrap();
            }
        };

        // Fetching a notebook is what stamps it as viewed.
        app.get(&format!("/api/notebooks/{first}")).await;
        assert!(Notebook::get(&app.state, &first).await.unwrap().last_viewed_at.is_some());
        view_source(paper.id.clone()).await;
        view_source(untitled.id.clone()).await;

        let (status, body) = app.get("/api/recently-viewed").await;
        assert_eq!(status, StatusCode::OK);
        let items = body.as_array().unwrap();
        assert_eq!(items.len(), 3, "{body}");
        assert_eq!(items[0], json!({
            "type": "source",
            "id": untitled.id,
            "title": "Untitled source",
            "last_viewed_at": items[0]["last_viewed_at"],
        }));
        assert!(chrono::DateTime::parse_from_rfc3339(items[0]["last_viewed_at"].as_str().unwrap()).is_ok());
        assert_eq!((&items[1]["type"], &items[1]["title"]), (&json!("source"), &json!("Paper")));
        assert_eq!((&items[2]["type"], &items[2]["id"], &items[2]["title"]), (&json!("notebook"), &json!(first), &json!("First")));
        assert!(!items.iter().any(|item| item["id"] == json!(unseen)));

        // Viewing the notebook again moves it to the front; the limit applies to the mix.
        app.get(&format!("/api/notebooks/{first}")).await;
        let (_, body) = app.get("/api/recently-viewed?limit=2").await;
        let ids: Vec<&str> = body.as_array().unwrap().iter().map(|item| item["id"].as_str().unwrap()).collect();
        assert_eq!(ids, [first.as_str(), untitled.id.as_str()]);

        for limit in ["0", "51", "many"] {
            let (status, body) = app.get(&format!("/api/recently-viewed?limit={limit}")).await;
            assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY, "{limit}");
            assert_eq!(body["detail"][0]["loc"], json!(["query"]));
        }
        assert_eq!(app.get("/api/recently-viewed?limit=50").await.0, StatusCode::OK);
    }
}
