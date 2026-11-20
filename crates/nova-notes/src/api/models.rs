//! `/api/models/...`: configured models, default-model slots, provider
//! availability, discovery and sync (Python `api/routers/models.py`).

use axum::extract::{Path, State};
use axum::routing::{delete, get, post};
use axum::{Json, Router};
use serde::Deserialize;
use serde_json::{json, Value};

use crate::api::{ApiJson, ApiQuery};
use crate::domain::DefaultModels;
use crate::error::AppResult;
use crate::services::models::{
    self, AllProvidersSyncResult, AutoAssignResult, DefaultModelsUpdate, DiscoveredModelInfo, ModelResponse,
    ModelTestResult, NewModel, ProviderAvailability, ProviderModelCount, ProviderSyncResult,
};
use crate::state::AppState;

pub fn router() -> Router<AppState> {
    Router::new()
        .route("/models", get(list_models).post(create_model))
        .route("/models/defaults", get(get_defaults).put(update_defaults))
        .route("/models/providers", get(provider_availability))
        .route("/models/discover/{provider}", get(discover_models))
        .route("/models/sync", post(sync_all))
        .route("/models/sync/{provider}", post(sync_provider))
        .route("/models/count/{provider}", get(model_count))
        .route("/models/by-provider/{provider}", get(models_by_provider))
        .route("/models/auto-assign", post(auto_assign))
        .route("/models/{model_id}", delete(delete_model))
        .route("/models/{model_id}/test", post(test_model))
}

#[derive(Debug, Deserialize)]
struct ListQuery {
    #[serde(rename = "type")]
    model_type: Option<String>,
}

async fn list_models(State(state): State<AppState>, ApiQuery(query): ApiQuery<ListQuery>) -> AppResult<Json<Vec<ModelResponse>>> {
    Ok(Json(models::list_models(&state, query.model_type.as_deref()).await?))
}

async fn create_model(State(state): State<AppState>, ApiJson(body): ApiJson<NewModel>) -> AppResult<Json<ModelResponse>> {
    Ok(Json(models::create_model(&state, &body).await?))
}

async fn delete_model(State(state): State<AppState>, Path(id): Path<String>) -> AppResult<Json<Value>> {
    models::delete_model(&state, &id).await?;
    Ok(Json(json!({ "message": "Model deleted successfully" })))
}

async fn test_model(State(state): State<AppState>, Path(id): Path<String>) -> AppResult<Json<ModelTestResult>> {
    Ok(Json(models::test_model(&state, &id).await?))
}

async fn get_defaults(State(state): State<AppState>) -> AppResult<Json<DefaultModels>> {
    Ok(Json(models::get_default_models(&state).await?))
}

async fn update_defaults(
    State(state): State<AppState>,
    ApiJson(body): ApiJson<DefaultModelsUpdate>,
) -> AppResult<Json<DefaultModels>> {
    Ok(Json(models::update_default_models(&state, &body).await?))
}

async fn provider_availability(State(state): State<AppState>) -> Json<ProviderAvailability> {
    Json(models::provider_availability(&state).await)
}

async fn discover_models(State(state): State<AppState>, Path(provider): Path<String>) -> AppResult<Json<Vec<DiscoveredModelInfo>>> {
    Ok(Json(models::discover_provider_models(&state, &provider).await?))
}

async fn sync_provider(State(state): State<AppState>, Path(provider): Path<String>) -> AppResult<Json<ProviderSyncResult>> {
    Ok(Json(models::sync_provider_models(&state, &provider).await?))
}

async fn sync_all(State(state): State<AppState>) -> Json<AllProvidersSyncResult> {
    Json(models::sync_all_providers(&state).await)
}

async fn model_count(State(state): State<AppState>, Path(provider): Path<String>) -> AppResult<Json<ProviderModelCount>> {
    Ok(Json(models::provider_model_count(&state, &provider).await?))
}

async fn models_by_provider(State(state): State<AppState>, Path(provider): Path<String>) -> AppResult<Json<Vec<ModelResponse>>> {
    Ok(Json(models::models_by_provider(&state, &provider).await?))
}

async fn auto_assign(State(state): State<AppState>) -> AppResult<Json<AutoAssignResult>> {
    Ok(Json(models::auto_assign_defaults(&state).await?))
}

#[cfg(test)]
mod tests {
    use axum::http::StatusCode;
    use serde_json::{json, Value};
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    use crate::domain::{DefaultModels, Model};
    use crate::services::models::sync_all_providers_with;
    use crate::testing::TestApp;

    async fn create(app: &TestApp, name: &str, provider: &str, model_type: &str) -> String {
        let (status, body) = app.post("/api/models", json!({ "name": name, "provider": provider, "type": model_type })).await;
        assert_eq!(status, StatusCode::OK, "{body}");
        body["id"].as_str().unwrap().to_string()
    }

    async fn local_credential(app: &TestApp, provider: &str, base_url: String) -> String {
        let (status, body) =
            app.post("/api/credentials", json!({ "name": "Local", "provider": provider, "base_url": base_url })).await;
        assert_eq!(status, StatusCode::CREATED, "{body}");
        body["id"].as_str().unwrap().to_string()
    }

    #[tokio::test]
    async fn create_list_and_delete_models() {
        let app = TestApp::new().await;
        let (status, created) = app
            .post("/api/models", json!({ "name": "gpt-4o", "provider": "openai", "type": "language", "credential": null }))
            .await;
        assert_eq!(status, StatusCode::OK, "{created}");
        let id = created["id"].as_str().unwrap().to_string();
        assert!(id.starts_with("model:"));
        assert_eq!(created["type"], "language");
        assert_eq!(created["credential"], Value::Null);
        let mut keys: Vec<&String> = created.as_object().unwrap().keys().collect();
        keys.sort();
        assert_eq!(keys, ["created", "credential", "id", "name", "provider", "type", "updated"]);

        // Duplicates are case-insensitive on provider, name and type.
        for (name, provider) in [("gpt-4o", "openai"), ("GPT-4O", "OpenAI")] {
            let (status, body) = app.post("/api/models", json!({ "name": name, "provider": provider, "type": "language" })).await;
            assert_eq!(status, StatusCode::BAD_REQUEST);
            assert_eq!(
                body["detail"],
                format!("Model '{name}' already exists for provider '{provider}' with type 'language'")
            );
        }
        // Same name under another provider or type is fine.
        create(&app, "gpt-4o", "azure", "language").await;
        create(&app, "gpt-4o", "openai", "embedding").await;

        let (status, body) = app.post("/api/models", json!({ "name": "x", "provider": "openai", "type": "vision" })).await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert_eq!(
            body["detail"],
            "Invalid model type. Must be one of: ['language', 'embedding', 'text_to_speech', 'speech_to_text']"
        );
        let (status, _) = app.post("/api/models", json!({ "name": "x", "provider": "openai" })).await;
        assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY);

        let (_, all) = app.get("/api/models").await;
        assert_eq!(all.as_array().unwrap().len(), 3);
        let (_, embeddings) = app.get("/api/models?type=embedding").await;
        assert_eq!(embeddings.as_array().unwrap().len(), 1);
        let (_, openai) = app.get("/api/models/by-provider/openai").await;
        let types: Vec<&str> = openai.as_array().unwrap().iter().map(|m| m["type"].as_str().unwrap()).collect();
        assert_eq!(types, ["embedding", "language"]);

        let (status, count) = app.get("/api/models/count/OPENAI").await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(count["provider"], "OPENAI");
        assert_eq!(count["total"], 2);
        assert_eq!(count["counts"], json!({ "embedding": 1, "language": 1, "speech_to_text": 0, "text_to_speech": 0 }));

        let (status, body) = app.delete(&format!("/api/models/{id}")).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(body, json!({ "message": "Model deleted successfully" }));
        let (status, body) = app.delete(&format!("/api/models/{id}")).await;
        assert_eq!(status, StatusCode::NOT_FOUND);
        assert_eq!(body["detail"], "Model not found");
    }

    #[tokio::test]
    async fn defaults_are_presence_keyed_and_required_slots_cannot_be_cleared() {
        let app = TestApp::new().await;
        let chat = create(&app, "gpt-4o", "openai", "language").await;
        let other = create(&app, "claude", "anthropic", "language").await;
        let embed = create(&app, "text-embedding-3-small", "openai", "embedding").await;

        let (status, body) = app.get("/api/models/defaults").await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(body.as_object().unwrap().len(), 7);
        assert!(body.as_object().unwrap().values().all(Value::is_null), "{body}");

        let (status, body) = app
            .put(
                "/api/models/defaults",
                json!({ "default_chat_model": chat, "default_embedding_model": embed, "default_tools_model": other }),
            )
            .await;
        assert_eq!(status, StatusCode::OK, "{body}");
        assert_eq!(body["default_chat_model"], chat);
        assert_eq!(body["default_tools_model"], other);

        // Absent keeps, explicit null clears an optional slot.
        let (_, body) = app.put("/api/models/defaults", json!({ "default_transformation_model": other })).await;
        assert_eq!(body["default_tools_model"], other);
        assert_eq!(body["default_transformation_model"], other);
        let (_, body) = app.put("/api/models/defaults", json!({ "default_tools_model": null })).await;
        assert_eq!(body["default_tools_model"], Value::Null);
        assert_eq!(body["default_chat_model"], chat);

        for field in ["default_chat_model", "default_embedding_model"] {
            let (status, body) = app.put("/api/models/defaults", json!({ field: null, "default_tools_model": other })).await;
            assert_eq!(status, StatusCode::BAD_REQUEST, "{field}");
            assert!(body["detail"].as_str().unwrap().contains(field), "{body}");
        }
        // Nothing was saved by the rejected requests.
        let stored = DefaultModels::load(&app.state).await.unwrap();
        assert_eq!(stored.default_tools_model, None);
        assert_eq!(stored.default_chat_model.as_deref(), Some(chat.as_str()));

        let (status, body) = app.put("/api/models/defaults", json!({ "default_chat_model": other })).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(body["default_chat_model"], other);
        let (_, body) = app.get("/api/models/defaults").await;
        assert_eq!(body["default_chat_model"], other);
    }

    #[tokio::test]
    async fn auto_assign_fills_only_empty_required_slots() {
        let app = TestApp::new().await;
        let (status, body) = app.post("/api/models/auto-assign", json!({})).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(body, json!({ "assigned": {}, "skipped": [], "missing": ["default_chat_model", "default_embedding_model"] }));

        create(&app, "llama3", "ollama", "language").await;
        let gpt = create(&app, "gpt-4o", "openai", "language").await;
        let embed = create(&app, "text-embedding-3-small", "openai", "embedding").await;
        create(&app, "tts-1", "openai", "text_to_speech").await;

        let (_, body) = app.post("/api/models/auto-assign", json!({})).await;
        assert_eq!(body["assigned"], json!({ "default_chat_model": gpt, "default_embedding_model": embed }));
        assert_eq!(body["skipped"], json!([]));
        let defaults = DefaultModels::load(&app.state).await.unwrap();
        assert_eq!(defaults.default_text_to_speech_model, None, "optional slots stay empty");
        assert_eq!(defaults.default_tools_model, None);

        let (_, body) = app.post("/api/models/auto-assign", json!({})).await;
        assert_eq!(body, json!({ "assigned": {}, "skipped": ["default_chat_model", "default_embedding_model"], "missing": [] }));
    }

    #[tokio::test]
    async fn provider_availability_counts_database_credentials() {
        let app = TestApp::new().await;
        local_credential(&app, "openai_compatible", "http://localhost:1234/v1".into()).await;
        let (status, body) = app.get("/api/models/providers").await;
        assert_eq!(status, StatusCode::OK);
        let available: Vec<&str> = body["available"].as_array().unwrap().iter().map(|p| p.as_str().unwrap()).collect();
        assert!(available.contains(&"openai_compatible"));
        assert_eq!(
            body["supported_types"]["openai_compatible"],
            json!(["language", "embedding", "speech_to_text", "text_to_speech"])
        );
        let total = available.len() + body["unavailable"].as_array().unwrap().len();
        assert_eq!(total, 24);
    }

    #[tokio::test]
    async fn discover_and_sync_a_provider_against_its_credential() {
        let app = TestApp::new().await;
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/v1/models"))
            .respond_with(
                ResponseTemplate::new(200)
                    .set_body_json(json!({ "data": [{ "id": "qwen3" }, { "id": "text-embedding-nomic" }, { "id": "whisper-1" }] })),
            )
            .mount(&server)
            .await;
        local_credential(&app, "openai_compatible", format!("{}/v1", server.uri())).await;
        create(&app, "QWEN3", "openai_compatible", "language").await;

        let (status, body) = app.get("/api/models/discover/openai_compatible").await;
        assert_eq!(status, StatusCode::OK, "{body}");
        let found: Vec<(&str, &str)> = body
            .as_array()
            .unwrap()
            .iter()
            .map(|m| (m["name"].as_str().unwrap(), m["model_type"].as_str().unwrap()))
            .collect();
        assert_eq!(found, [("qwen3", "language"), ("text-embedding-nomic", "embedding"), ("whisper-1", "speech_to_text")]);

        let (status, body) = app.post("/api/models/sync/openai_compatible", json!({})).await;
        assert_eq!(status, StatusCode::OK, "{body}");
        assert_eq!(body, json!({ "provider": "openai_compatible", "discovered": 3, "new": 2, "existing": 1 }));
        let (_, body) = app.post("/api/models/sync/openai_compatible", json!({})).await;
        assert_eq!(body["new"], 0);
        assert_eq!(body["existing"], 3);
        assert_eq!(Model::get_by_provider(&app.state, "openai_compatible").await.unwrap().len(), 3);

        // Providers that need a specific credential, and unknown ones, give nothing.
        for provider in ["azure", "vertex", "not-a-provider"] {
            let (status, body) = app.get(&format!("/api/models/discover/{provider}")).await;
            assert_eq!(status, StatusCode::OK);
            assert_eq!(body, json!([]), "{provider}");
            let (_, body) = app.post(&format!("/api/models/sync/{provider}"), json!({})).await;
            assert_eq!(body, json!({ "provider": provider, "discovered": 0, "new": 0, "existing": 0 }));
        }
    }

    #[tokio::test]
    async fn sync_all_reports_every_provider() {
        let app = TestApp::new().await;
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/compat/v1/models"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({ "data": [{ "id": "qwen3" }] })))
            .mount(&server)
            .await;
        // Point the providers with a local default at the mock (it answers
        // 404), so the run never depends on what this machine has running.
        local_credential(&app, "openai_compatible", format!("{}/compat/v1", server.uri())).await;
        local_credential(&app, "ollama", format!("{}/ollama", server.uri())).await;
        local_credential(&app, "omlx", format!("{}/omlx/v1", server.uri())).await;

        let no_env = |_: &str| None;
        let result = sync_all_providers_with(&app.state, &no_env).await;
        assert_eq!(result.results.len(), nn_ai::PROVIDERS.len());
        assert_eq!(result.total_discovered, 1);
        assert_eq!(result.total_new, 1);
        assert_eq!(result.results["openai_compatible"].new, 1);
        assert_eq!(result.results["ollama"].discovered, 0);
        let json = serde_json::to_value(&result).unwrap();
        assert_eq!(json["results"]["openai_compatible"]["provider"], "openai_compatible");
    }

    #[tokio::test]
    async fn model_tests_run_a_real_call() {
        let app = TestApp::new().await;
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/embeddings"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "object": "list",
                "model": "nomic",
                "data": [{ "object": "embedding", "index": 0, "embedding": [0.1, 0.2, 0.3] }],
            })))
            .mount(&server)
            .await;
        let credential = local_credential(&app, "openai_compatible", format!("{}/v1", server.uri())).await;
        let (status, body) = app
            .post(
                "/api/models",
                json!({ "name": "nomic", "provider": "openai_compatible", "type": "embedding", "credential": credential }),
            )
            .await;
        assert_eq!(status, StatusCode::OK, "{body}");
        let id = body["id"].as_str().unwrap();
        assert_eq!(body["credential"], credential);

        let (status, body) = app.post(&format!("/api/models/{id}/test"), json!({})).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(body, json!({ "success": true, "message": "Embedding dimensions: 3", "details": null }));

        // An unreachable server is a failed result, not an error.
        let gone = local_credential(&app, "openai_compatible", "http://127.0.0.1:1/v1".into()).await;
        let (_, body) = app
            .post("/api/models", json!({ "name": "other", "provider": "openai_compatible", "type": "embedding", "credential": gone }))
            .await;
        let (status, body) = app.post(&format!("/api/models/{}/test", body["id"].as_str().unwrap()), json!({})).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(body["success"], false, "{body}");

        let (status, body) = app.post("/api/models/model:missing/test", json!({})).await;
        assert_eq!(status, StatusCode::NOT_FOUND);
        assert_eq!(body["detail"], "Model not found");
    }
}
