//! `/api/embeddings/rebuild...` (Python `api/routers/embedding_rebuild.py`,
//! mounted under `/api/embeddings`).

use axum::extract::{Path, State};
use axum::routing::{get, post};
use axum::{Json, Router};

use super::notebooks::ApiJson;
use crate::error::AppResult;
use crate::services::embedding::{self, RebuildRequest, RebuildStarted, RebuildStatus};
use crate::state::AppState;

pub fn router() -> Router<AppState> {
    Router::new()
        .route("/embeddings/rebuild", post(start_rebuild))
        .route("/embeddings/rebuild/{command_id}/status", get(rebuild_status))
}

async fn start_rebuild(State(state): State<AppState>, ApiJson(body): ApiJson<RebuildRequest>) -> AppResult<Json<RebuildStarted>> {
    Ok(Json(embedding::start_rebuild(&state, &body).await?))
}

async fn rebuild_status(State(state): State<AppState>, Path(command_id): Path<String>) -> AppResult<Json<RebuildStatus>> {
    Ok(Json(embedding::rebuild_status(&state, &command_id).await?))
}

#[cfg(test)]
mod tests {
    use axum::http::StatusCode;
    use serde_json::json;

    use crate::ai::AnyModel;
    use crate::domain::{Note, Source};
    use crate::testing::{MockEmbeddingModel, TestApp};

    #[tokio::test]
    async fn rebuild_queues_one_job_per_item() {
        let app = TestApp::new().await;
        app.state.models.set_override(AnyModel::Embedding(MockEmbeddingModel::new()));
        Note::create(&app.state, Some("N"), Some("Note body"), Some("human")).await.unwrap();
        let source = Source::create(&app.state, &Source { full_text: Some("Source body".into()), ..Source::default() }).await.unwrap();

        let (status, body) = app.post("/api/embeddings/rebuild", json!({ "mode": "all", "include_insights": false })).await;
        assert_eq!(status, StatusCode::OK, "{body}");
        assert_eq!(body["total_items"], 2);
        assert_eq!(body["message"], "Rebuild operation started. Estimated 2 items to process.");
        let command_id = body["command_id"].as_str().unwrap().to_string();
        assert_eq!(app.wait_for_job(&command_id).await, "completed");

        let (status, body) = app.get(&format!("/api/embeddings/rebuild/{command_id}/status")).await;
        assert_eq!(status, StatusCode::OK, "{body}");
        assert_eq!(body["status"], "completed");
        assert_eq!(body["progress"], json!({ "processed": 2, "total": 2, "percentage": 100.0 }));
        assert_eq!(body["stats"], json!({ "sources": 1, "notes": 1, "insights": 0, "failed": 0 }));
        assert!(body["error_message"].is_null());

        // Once the source has chunks, "existing" finds it.
        app.wait_for_idle().await;
        assert!(source.embedded_chunks(&app.state).await.unwrap() > 0);
        let (status, body) = app
            .post("/api/embeddings/rebuild", json!({ "mode": "existing", "include_notes": false, "include_insights": false }))
            .await;
        assert_eq!(status, StatusCode::OK, "{body}");
        assert_eq!(body["total_items"], 1);
    }

    #[tokio::test]
    async fn validation_and_unknown_commands() {
        let app = TestApp::new().await;
        let (status, _) = app.post("/api/embeddings/rebuild", json!({ "mode": "everything" })).await;
        assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY);
        let (status, _) = app.post("/api/embeddings/rebuild", json!({})).await;
        assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY);
        let (status, body) = app.get("/api/embeddings/rebuild/command:missing/status").await;
        assert_eq!((status, body["detail"].as_str()), (StatusCode::NOT_FOUND, Some("Rebuild command not found")));

        // Without an embedding model the job reports the problem in its result.
        let (status, body) = app.post("/api/embeddings/rebuild", json!({ "mode": "all" })).await;
        assert_eq!(status, StatusCode::OK);
        let command_id = body["command_id"].as_str().unwrap();
        app.wait_for_job(command_id).await;
        let (_, body) = app.get(&format!("/api/embeddings/rebuild/{command_id}/status")).await;
        assert_eq!(body["stats"]["sources"], 0);
    }
}
