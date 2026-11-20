//! Insights: `/api/insights/...` (Python `api/routers/insights.py`). A
//! source's insight list and creation live under `/api/sources/{id}/insights`.

use axum::extract::{Path, State};
use axum::routing::{get, post};
use axum::{Json, Router};
use serde::Deserialize;
use serde_json::{json, Value};

use super::notebooks::ApiJson;
use crate::error::AppResult;
use crate::services::insights::{self, InsightView};
use crate::services::notes::NoteView;
use crate::state::AppState;

pub fn router() -> Router<AppState> {
    Router::new()
        .route("/insights/{insight_id}", get(get_insight).delete(delete_insight))
        .route("/insights/{insight_id}/save-as-note", post(save_as_note))
}

/// `SaveAsNoteRequest`.
#[derive(Debug, Default, Deserialize)]
struct SaveAsNote {
    #[serde(default)]
    notebook_id: Option<String>,
}

async fn get_insight(State(state): State<AppState>, Path(insight_id): Path<String>) -> AppResult<Json<InsightView>> {
    Ok(Json(insights::get_insight(&state, &insight_id).await?))
}

async fn delete_insight(State(state): State<AppState>, Path(insight_id): Path<String>) -> AppResult<Json<Value>> {
    insights::delete_insight(&state, &insight_id).await?;
    Ok(Json(json!({ "message": "Insight deleted successfully" })))
}

async fn save_as_note(
    State(state): State<AppState>,
    Path(insight_id): Path<String>,
    ApiJson(body): ApiJson<SaveAsNote>,
) -> AppResult<Json<NoteView>> {
    Ok(Json(insights::save_insight_as_note(&state, &insight_id, body.notebook_id.as_deref()).await?))
}

#[cfg(test)]
mod tests {
    use axum::http::StatusCode;
    use serde_json::json;

    use crate::domain::{Notebook, Source};
    use crate::testing::TestApp;

    async fn source_with_insight(app: &TestApp) -> (Source, String) {
        let source = Source::create(&app.state, &Source { title: Some("Paper".into()), full_text: Some("Text".into()), ..Source::default() })
            .await
            .unwrap();
        // `add_insight` queues `create_insight`; wait for the row it writes.
        let command_id = source.add_insight(&app.state, "Summary", "The gist.").await.unwrap();
        app.wait_for_job(&command_id).await;
        let insight_id = source.insights(&app.state).await.unwrap()[0].id.clone();
        (source, insight_id)
    }

    #[tokio::test]
    async fn get_save_and_delete_an_insight() {
        let app = TestApp::new().await;
        let (source, insight_id) = source_with_insight(&app).await;

        let (status, body) = app.get(&format!("/api/insights/{insight_id}")).await;
        assert_eq!(status, StatusCode::OK, "{body}");
        assert_eq!(body["source_id"], source.id.as_str());
        assert_eq!(body["insight_type"], "Summary");
        assert_eq!(body["content"], "The gist.");
        // Python's isoformat(): `+00:00`, not `Z`.
        assert!(body["created"].as_str().unwrap().ends_with("+00:00"), "{body}");

        let notebook = Notebook::create(&app.state, "NB", "", None).await.unwrap();
        let (status, note) =
            app.post(&format!("/api/insights/{insight_id}/save-as-note"), json!({ "notebook_id": notebook.id })).await;
        assert_eq!(status, StatusCode::OK, "{note}");
        assert_eq!(note["content"], "The gist.");
        assert_eq!(note["title"], "Summary from source Paper");
        let (_, notes) = app.get(&format!("/api/notes?notebook_id={}", notebook.id)).await;
        assert_eq!(notes.as_array().unwrap().len(), 1);

        let (status, body) =
            app.post(&format!("/api/insights/{insight_id}/save-as-note"), json!({ "notebook_id": "notebook:missing" })).await;
        assert_eq!((status, body["detail"].as_str()), (StatusCode::NOT_FOUND, Some("Notebook not found")));
        let (status, _) = app.post(&format!("/api/insights/{insight_id}/save-as-note"), json!({})).await;
        assert_eq!(status, StatusCode::OK);

        let (status, body) = app.delete(&format!("/api/insights/{insight_id}")).await;
        assert_eq!((status, body["message"].as_str()), (StatusCode::OK, Some("Insight deleted successfully")));
        let (status, body) = app.get(&format!("/api/insights/{insight_id}")).await;
        assert_eq!((status, body["detail"].as_str()), (StatusCode::NOT_FOUND, Some("Insight not found")));
    }

    #[tokio::test]
    async fn missing_insights_are_404() {
        let app = TestApp::new().await;
        for id in ["source_insight:missing", "note:wrongtable"] {
            let (status, body) = app.get(&format!("/api/insights/{id}")).await;
            assert_eq!((status, body["detail"].as_str()), (StatusCode::NOT_FOUND, Some("Insight not found")), "{id}");
            let (status, _) = app.delete(&format!("/api/insights/{id}")).await;
            assert_eq!(status, StatusCode::NOT_FOUND);
            let (status, _) = app.post(&format!("/api/insights/{id}/save-as-note"), json!({})).await;
            assert_eq!(status, StatusCode::NOT_FOUND);
        }
    }
}
