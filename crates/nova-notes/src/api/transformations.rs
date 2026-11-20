//! Transformations: `/api/transformations/...` (Python
//! `api/routers/transformations.py`).

use axum::extract::{Path, State};
use axum::routing::{get, post};
use axum::{Json, Router};
use serde_json::{json, Value};

use super::notebooks::ApiJson;
use crate::error::AppResult;
use crate::services::transformations::{
    self, DefaultPrompt, ExecuteTransformation, NewTransformation, TransformationOutput, TransformationUpdate,
    TransformationView,
};
use crate::state::AppState;

pub fn router() -> Router<AppState> {
    Router::new()
        .route("/transformations", get(list_transformations).post(create_transformation))
        .route("/transformations/execute", post(execute_transformation))
        .route("/transformations/default-prompt", get(get_default_prompt).put(update_default_prompt))
        .route(
            "/transformations/{transformation_id}",
            get(get_transformation).put(update_transformation).delete(delete_transformation),
        )
}

async fn list_transformations(State(state): State<AppState>) -> AppResult<Json<Vec<TransformationView>>> {
    Ok(Json(transformations::list_transformations(&state).await?))
}

async fn create_transformation(
    State(state): State<AppState>,
    ApiJson(body): ApiJson<NewTransformation>,
) -> AppResult<Json<TransformationView>> {
    Ok(Json(transformations::create_transformation(&state, &body).await?))
}

async fn execute_transformation(
    State(state): State<AppState>,
    ApiJson(body): ApiJson<ExecuteTransformation>,
) -> AppResult<Json<TransformationOutput>> {
    Ok(Json(transformations::execute_transformation(&state, &body).await?))
}

async fn get_default_prompt(State(state): State<AppState>) -> AppResult<Json<DefaultPrompt>> {
    Ok(Json(transformations::get_default_prompt(&state).await?))
}

async fn update_default_prompt(State(state): State<AppState>, ApiJson(body): ApiJson<DefaultPrompt>) -> AppResult<Json<DefaultPrompt>> {
    Ok(Json(transformations::update_default_prompt(&state, &body).await?))
}

async fn get_transformation(State(state): State<AppState>, Path(id): Path<String>) -> AppResult<Json<TransformationView>> {
    Ok(Json(transformations::get_transformation(&state, &id).await?))
}

async fn update_transformation(
    State(state): State<AppState>,
    Path(id): Path<String>,
    ApiJson(body): ApiJson<TransformationUpdate>,
) -> AppResult<Json<TransformationView>> {
    Ok(Json(transformations::update_transformation(&state, &id, &body).await?))
}

async fn delete_transformation(State(state): State<AppState>, Path(id): Path<String>) -> AppResult<Json<Value>> {
    transformations::delete_transformation(&state, &id).await?;
    Ok(Json(json!({ "message": "Transformation deleted successfully" })))
}

#[cfg(test)]
mod tests {
    use axum::http::StatusCode;
    use nn_llm::AiError;
    use serde_json::{json, Value};

    use crate::ai::AnyModel;
    use crate::testing::{MockLanguageModel, TestApp};

    async fn create(app: &TestApp, name: &str) -> Value {
        let (status, body) = app
            .post(
                "/api/transformations",
                json!({ "name": name, "title": name.to_uppercase(), "description": "d", "prompt": "Make it short.", "apply_default": true }),
            )
            .await;
        assert_eq!(status, StatusCode::OK, "{body}");
        body
    }

    #[tokio::test]
    async fn crud() {
        let app = TestApp::new().await;
        let (_, before) = app.get("/api/transformations").await;
        let baseline = before.as_array().unwrap().len();
        let created = create(&app, "zzz-brief").await;
        let id = created["id"].as_str().unwrap().to_string();
        assert_eq!(created["apply_default"], true);
        assert_eq!(created["model_id"], Value::Null);

        let (status, list) = app.get("/api/transformations").await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(list.as_array().unwrap().len(), baseline + 1);

        let (status, updated) = app.put(&format!("/api/transformations/{id}"), json!({ "title": "Brief", "description": null })).await;
        assert_eq!(status, StatusCode::OK, "{updated}");
        assert_eq!((updated["title"].as_str(), updated["description"].as_str()), (Some("Brief"), Some("")));
        assert_eq!(updated["prompt"], "Make it short.");
        let (_, fetched) = app.get(&format!("/api/transformations/{id}")).await;
        assert_eq!(fetched["title"], "Brief");

        let (status, body) = app.delete(&format!("/api/transformations/{id}")).await;
        assert_eq!((status, body["message"].as_str()), (StatusCode::OK, Some("Transformation deleted successfully")));
        let (status, body) = app.get(&format!("/api/transformations/{id}")).await;
        assert_eq!((status, body["detail"].as_str()), (StatusCode::NOT_FOUND, Some("Transformation not found")));
    }

    #[tokio::test]
    async fn not_found_and_validation() {
        let app = TestApp::new().await;
        for id in ["transformation:missing", "source:wrongtable"] {
            let (status, _) = app.get(&format!("/api/transformations/{id}")).await;
            assert_eq!(status, StatusCode::NOT_FOUND);
            let (status, _) = app.put(&format!("/api/transformations/{id}"), json!({ "title": "x" })).await;
            assert_eq!(status, StatusCode::NOT_FOUND);
            let (status, _) = app.delete(&format!("/api/transformations/{id}")).await;
            assert_eq!(status, StatusCode::NOT_FOUND);
        }
        let (status, body) = app.post("/api/transformations", json!({ "name": "x" })).await;
        assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY, "{body}");
        let (status, body) = app
            .post("/api/transformations", json!({ "name": "x", "title": "X", "description": "", "prompt": "p", "model_id": "model:missing" }))
            .await;
        assert_eq!((status, body["detail"].as_str()), (StatusCode::NOT_FOUND, Some("Model not found")));
        let (status, body) = app
            .post("/api/transformations/execute", json!({ "transformation_id": "transformation:missing", "input_text": "x" }))
            .await;
        assert_eq!((status, body["detail"].as_str()), (StatusCode::NOT_FOUND, Some("Transformation not found")));
    }

    #[tokio::test]
    async fn execute_runs_the_prompt_and_cleans_the_output() {
        let app = TestApp::new().await;
        let model = MockLanguageModel::new(["<think>reasoning</think>\nShort version."]);
        app.state.models.set_override(AnyModel::Language(model.clone()));
        let (status, _) = app
            .put("/api/transformations/default-prompt", json!({ "transformation_instructions": "Always answer in English." }))
            .await;
        assert_eq!(status, StatusCode::OK);
        let (_, prompt) = app.get("/api/transformations/default-prompt").await;
        assert_eq!(prompt["transformation_instructions"], "Always answer in English.");

        let id = create(&app, "short").await["id"].as_str().unwrap().to_string();
        let (status, body) =
            app.post("/api/transformations/execute", json!({ "transformation_id": id, "input_text": "A long text." })).await;
        assert_eq!(status, StatusCode::OK, "{body}");
        assert_eq!(body["output"], "Short version.");
        assert_eq!(body["transformation_id"], id.as_str());
        assert_eq!(body["model_id"], Value::Null);
        let prompt = model.prompt(0);
        assert!(prompt.contains("Always answer in English.") && prompt.contains("Make it short.") && prompt.contains("A long text."));

        // Empty input, a model that only thinks, and provider errors.
        let (status, body) = app.post("/api/transformations/execute", json!({ "transformation_id": id, "input_text": "  " })).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
        app.state.models.set_override(AnyModel::Language(MockLanguageModel::new(["<think>only thoughts</think>"])));
        let (status, body) = app.post("/api/transformations/execute", json!({ "transformation_id": id, "input_text": "x" })).await;
        assert_eq!(status, StatusCode::BAD_GATEWAY, "{body}");
        app.state
            .models
            .set_override(AnyModel::Language(MockLanguageModel::failing(AiError::ContextLength("too long".into()))));
        let (status, _) = app.post("/api/transformations/execute", json!({ "transformation_id": id, "input_text": "x" })).await;
        assert_eq!(status, StatusCode::BAD_GATEWAY);
    }

    #[tokio::test]
    async fn context_length_errors_are_not_retried_by_the_job() {
        let app = TestApp::new().await;
        let model = MockLanguageModel::failing(AiError::ContextLength("too long".into()));
        app.state.models.set_override(AnyModel::Language(model.clone()));
        let id = create(&app, "t").await["id"].as_str().unwrap().to_string();
        let source = crate::domain::Source::create(
            &app.state,
            &crate::domain::Source { full_text: Some("Some text".into()), ..crate::domain::Source::default() },
        )
        .await
        .unwrap();
        let command_id = app
            .state
            .jobs
            .submit("run_transformation", json!({ "source_id": source.id, "transformation_id": id }))
            .await
            .unwrap();
        assert_eq!(app.wait_for_job(&command_id).await, "failed");
        assert_eq!(model.call_count(), 1, "a context-length error is permanent");
    }
}
