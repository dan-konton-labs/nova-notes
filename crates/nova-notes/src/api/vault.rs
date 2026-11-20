//! `/api/vault/*`: the notes library as an Obsidian vault.
//!
//! Fork-only routes (no Python counterpart): `status` says whether the
//! CouchDB behind the vault is configured and reachable, `files` lists the
//! vault tree (optionally with content, for any consumer that wants to write
//! the files itself), and `push` writes the tree into the LiveSync database.

use axum::extract::State;
use axum::routing::{get, post};
use axum::{Json, Router};
use serde::Deserialize;
use serde_json::json;

use super::ApiQuery;
use crate::error::AppResult;
use crate::services::vault;
use crate::state::AppState;

pub fn router() -> Router<AppState> {
    Router::new()
        .route("/vault/status", get(vault_status))
        .route("/vault/files", get(vault_files))
        .route("/vault/push", post(vault_push))
}

async fn vault_status(State(state): State<AppState>) -> AppResult<Json<vault::VaultStatus>> {
    Ok(Json(vault::status(&state).await?))
}

#[derive(Debug, Default, Deserialize)]
struct FilesQuery {
    /// `true` includes each file's full markdown.
    #[serde(default, deserialize_with = "super::lenient_bool")]
    content: Option<bool>,
}

async fn vault_files(State(state): State<AppState>, ApiQuery(query): ApiQuery<FilesQuery>) -> AppResult<Json<serde_json::Value>> {
    let tree = vault::vault_tree(&state).await?;
    let with_content = query.content.unwrap_or(false);
    let files: Vec<serde_json::Value> = tree
        .iter()
        .map(|f| {
            let mut row = json!({ "path": f.path, "note_id": f.note_id, "size": f.content.len(), "mtime": f.mtime });
            if with_content {
                row["content"] = json!(f.content);
            }
            row
        })
        .collect();
    Ok(Json(json!({ "folder": state.config.vault.folder, "files": files })))
}

async fn vault_push(State(state): State<AppState>) -> AppResult<Json<vault::PushReport>> {
    Ok(Json(vault::push(&state).await?))
}

#[cfg(test)]
mod tests {
    use axum::http::StatusCode;
    use serde_json::json;

    use crate::testing::TestApp;

    #[tokio::test]
    async fn vault_routes_report_files_and_configuration() {
        let app = TestApp::new().await;
        let (_, nb) = app.post("/api/notebooks", json!({ "name": "Garden", "description": "" })).await;
        let (status, note) = app
            .post("/api/notes", json!({ "title": "Tomatoes", "content": "Plant in May", "notebook_id": nb["id"] }))
            .await;
        assert_eq!(status, StatusCode::OK, "{note}");

        let (status, body) = app.get("/api/vault/status").await;
        assert_eq!(status, StatusCode::OK, "{body}");
        assert_eq!(body["configured"], false);
        assert_eq!(body["folder"], "NovaNotes");
        assert_eq!(body["local_files"], 1);

        let (status, body) = app.get("/api/vault/files").await;
        assert_eq!(status, StatusCode::OK, "{body}");
        assert_eq!(body["files"][0]["path"], "NovaNotes/Garden/Tomatoes.md");
        assert!(body["files"][0]["content"].is_null());
        let (_, body) = app.get("/api/vault/files?content=true").await;
        let content = body["files"][0]["content"].as_str().unwrap();
        assert!(content.contains("Plant in May"), "{content}");

        // An unconfigured push is a clear 422, not a hang or a stub.
        let (status, body) = app.post("/api/vault/push", json!({})).await;
        assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY, "{body}");
        assert!(body["detail"].as_str().unwrap().contains("VAULT_COUCHDB_URL"));
    }
}
