//! `POST /api/embed` (Python `api/routers/embedding.py`).

use axum::extract::State;
use axum::routing::post;
use axum::{Json, Router};

use super::notebooks::ApiJson;
use crate::error::AppResult;
use crate::services::embedding::{self, EmbedRequest, EmbedResult};
use crate::state::AppState;

pub fn router() -> Router<AppState> {
    Router::new().route("/embed", post(embed_content))
}

async fn embed_content(State(state): State<AppState>, ApiJson(body): ApiJson<EmbedRequest>) -> AppResult<Json<EmbedResult>> {
    Ok(Json(embedding::embed_item(&state, &body).await?))
}

#[cfg(test)]
mod tests {
    use axum::http::StatusCode;
    use serde_json::{json, Value};

    use crate::ai::AnyModel;
    use crate::db::Params;
    use crate::domain::{Note, Source, SourceInsight};
    use crate::testing::{MockEmbeddingModel, TestApp};

    async fn has_embedding(app: &TestApp, id: &str) -> bool {
        let rows = app
            .state
            .db
            .query("SELECT VALUE array::len(embedding OR []) FROM $id", Params::new().record("id", id))
            .await
            .unwrap();
        rows.first().and_then(Value::as_i64).unwrap_or(0) > 0
    }

    #[tokio::test]
    async fn needs_a_model_and_a_known_type() {
        let app = TestApp::new().await;
        let (status, body) = app.post("/api/embed", json!({ "item_id": "note:x", "item_type": "note" })).await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert_eq!(body["detail"], crate::services::embedding::NO_EMBEDDING_MODEL);

        app.state.models.set_override(AnyModel::Embedding(MockEmbeddingModel::new()));
        let (status, body) = app.post("/api/embed", json!({ "item_id": "note:x", "item_type": "podcast" })).await;
        assert_eq!((status, body["detail"].as_str()), (StatusCode::BAD_REQUEST, Some("Item type must be either 'source' or 'note'")));
        let (status, body) = app.post("/api/embed", json!({ "item_id": "note:missing", "item_type": "Note" })).await;
        assert_eq!((status, body["detail"].as_str()), (StatusCode::NOT_FOUND, Some("Note not found")));
        let (status, body) = app.post("/api/embed", json!({ "item_id": "source:missing", "item_type": "source" })).await;
        assert_eq!((status, body["detail"].as_str()), (StatusCode::NOT_FOUND, Some("source not found")));
        let (status, _) = app.post("/api/embed", json!({ "item_type": "source" })).await;
        assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY);
    }

    #[tokio::test]
    async fn notes_and_sources_are_embedded() {
        let app = TestApp::new().await;
        app.state.models.set_override(AnyModel::Embedding(MockEmbeddingModel::new()));

        let note = Note::create(&app.state, Some("N"), Some("A note about embeddings."), Some("human")).await.unwrap();
        let (status, body) = app.post("/api/embed", json!({ "item_id": note.id, "item_type": "note" })).await;
        assert_eq!(status, StatusCode::OK, "{body}");
        assert_eq!(body["message"], "Note embedding job submitted");
        assert_eq!(app.wait_for_job(body["command_id"].as_str().unwrap()).await, "completed");
        assert!(has_embedding(&app, &note.id).await);

        let text = "First paragraph about Rust.\n\n".repeat(200);
        let source = Source::create(&app.state, &Source { full_text: Some(text), ..Source::default() }).await.unwrap();
        let (status, body) = app.post("/api/embed", json!({ "item_id": source.id, "item_type": "source", "async_processing": true })).await;
        assert_eq!(status, StatusCode::OK, "{body}");
        assert_eq!(body["message"], "Embedding queued for background processing");
        let command_id = body["command_id"].as_str().unwrap();
        assert_eq!(app.wait_for_job(command_id).await, "completed");
        let record = app.state.jobs.get(command_id).await.unwrap().unwrap();
        let result = record.result.unwrap();
        assert_eq!(result["success"], true);
        let chunks = source.embedded_chunks(&app.state).await.unwrap();
        assert!(chunks > 1, "{chunks}");
        assert_eq!(result["chunks_created"].as_i64(), Some(chunks));

        // Re-embedding replaces the chunks instead of adding to them.
        let (_, body) = app.post("/api/embed", json!({ "item_id": source.id, "item_type": "source" })).await;
        app.wait_for_job(body["command_id"].as_str().unwrap()).await;
        assert_eq!(source.embedded_chunks(&app.state).await.unwrap(), chunks);
    }

    #[tokio::test]
    async fn embed_jobs_report_missing_or_empty_records() {
        let app = TestApp::new().await;
        app.state.models.set_override(AnyModel::Embedding(MockEmbeddingModel::new()));
        let empty = Source::create(&app.state, &Source::default()).await.unwrap();
        for (job, args) in [
            ("embed_note", json!({ "note_id": "note:missing" })),
            ("embed_insight", json!({ "insight_id": "source_insight:missing" })),
            ("embed_source", json!({ "source_id": empty.id })),
        ] {
            let command_id = app.state.jobs.submit(job, args).await.unwrap();
            assert_eq!(app.wait_for_job(&command_id).await, "completed", "{job}");
            let result = app.state.jobs.get(&command_id).await.unwrap().unwrap().result.unwrap();
            assert_eq!(result["success"], false, "{job}");
            assert!(result["error_message"].is_string(), "{job}");
        }

        // An insight is embedded by its own job.
        let source = Source::create(&app.state, &Source { full_text: Some("t".into()), ..Source::default() }).await.unwrap();
        let command_id = source.add_insight(&app.state, "Summary", "Insight text").await.unwrap();
        app.wait_for_job(&command_id).await;
        let insight_id = source.insights(&app.state).await.unwrap()[0].id.clone();
        app.wait_for_idle().await;
        assert!(has_embedding(&app, &insight_id).await);
        SourceInsight::get(&app.state, &insight_id).await.unwrap();
    }
}
