//! `/api/credentials/...`: AI provider credentials (Python
//! `api/routers/credentials.py`). Responses never carry a key.

use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use serde::Deserialize;

use crate::api::{validation_error, ApiJson, ApiQuery};
use crate::error::{AppError, AppResult};
use crate::services::credentials::{
    self, CredentialDeleted, CredentialResponse, CredentialStatus, CredentialUpdate, DiscoverModelsResponse,
    MigrationResult, NewCredential, RegisterModelsRequest, RegisterModelsResult,
};
use crate::state::AppState;

pub fn router() -> Router<AppState> {
    Router::new()
        .route("/credentials/status", get(get_status))
        .route("/credentials/env-status", get(get_env_status))
        .route("/credentials", get(list_credentials).post(create_credential))
        .route("/credentials/by-provider/{provider}", get(list_by_provider))
        .route("/credentials/migrate-from-provider-config", post(migrate_from_provider_config))
        .route("/credentials/migrate-encryption", post(migrate_encryption))
        .route("/credentials/migrate-from-env", post(migrate_from_env))
        .route(
            "/credentials/{credential_id}",
            get(get_credential).put(update_credential).delete(delete_credential),
        )
        .route("/credentials/{credential_id}/test", post(test_credential))
        .route("/credentials/{credential_id}/discover", post(discover_models))
        .route("/credentials/{credential_id}/register-models", post(register_models))
}

async fn get_status(State(state): State<AppState>) -> Json<CredentialStatus> {
    Json(credentials::provider_status(&state).await)
}

async fn get_env_status() -> Json<std::collections::BTreeMap<String, bool>> {
    Json(credentials::env_status())
}

#[derive(Debug, Deserialize)]
struct ListQuery {
    provider: Option<String>,
}

async fn list_credentials(
    State(state): State<AppState>,
    ApiQuery(query): ApiQuery<ListQuery>,
) -> AppResult<Json<Vec<CredentialResponse>>> {
    Ok(Json(credentials::list_credentials(&state, query.provider.as_deref()).await?))
}

async fn list_by_provider(State(state): State<AppState>, Path(provider): Path<String>) -> AppResult<Json<Vec<CredentialResponse>>> {
    Ok(Json(credentials::list_credentials_by_provider(&state, &provider).await?))
}

/// 201 on success. A provider outside the registry or a missing required
/// field is a 422, as pydantic rejected `CreateCredentialRequest`.
async fn create_credential(State(state): State<AppState>, ApiJson(body): ApiJson<NewCredential>) -> AppResult<Response> {
    if let Err(e) = body.validate() {
        let message = match e {
            AppError::InvalidInput(message) => message,
            other => other.to_string(),
        };
        let kind = if credentials::supported_provider_message() == message { "literal_error" } else { "value_error" };
        let message = if kind == "value_error" { format!("Value error, {message}") } else { message };
        return Ok(validation_error("body", kind, message));
    }
    let created = credentials::create_credential(&state, &body).await?;
    Ok((StatusCode::CREATED, Json(created)).into_response())
}

async fn get_credential(State(state): State<AppState>, Path(id): Path<String>) -> AppResult<Json<CredentialResponse>> {
    Ok(Json(credentials::get_credential(&state, &id).await?))
}

async fn update_credential(
    State(state): State<AppState>,
    Path(id): Path<String>,
    ApiJson(body): ApiJson<CredentialUpdate>,
) -> AppResult<Json<CredentialResponse>> {
    Ok(Json(credentials::update_credential(&state, &id, &body).await?))
}

#[derive(Debug, Deserialize)]
struct DeleteQuery {
    migrate_to: Option<String>,
}

async fn delete_credential(
    State(state): State<AppState>,
    Path(id): Path<String>,
    ApiQuery(query): ApiQuery<DeleteQuery>,
) -> AppResult<Json<CredentialDeleted>> {
    Ok(Json(credentials::delete_credential(&state, &id, query.migrate_to.as_deref()).await?))
}

async fn test_credential(State(state): State<AppState>, Path(id): Path<String>) -> Json<nn_ai::ConnectionTestResult> {
    Json(credentials::test_credential(&state, &id).await)
}

async fn discover_models(State(state): State<AppState>, Path(id): Path<String>) -> AppResult<Json<DiscoverModelsResponse>> {
    Ok(Json(credentials::discover_credential_models(&state, &id).await?))
}

async fn register_models(
    State(state): State<AppState>,
    Path(id): Path<String>,
    ApiJson(body): ApiJson<RegisterModelsRequest>,
) -> AppResult<Json<RegisterModelsResult>> {
    Ok(Json(credentials::register_credential_models(&state, &id, &body.models).await?))
}

/// `ValueError`s from the migrations were 400s with the text cut to 200
/// characters (the missing-encryption-key case).
fn value_error(e: AppError) -> AppError {
    match e {
        AppError::InvalidInput(message) => AppError::bad_request(message.chars().take(200).collect::<String>()),
        other => other,
    }
}

async fn migrate_from_provider_config(State(state): State<AppState>) -> AppResult<Json<MigrationResult>> {
    Ok(Json(credentials::migrate_from_provider_config(&state).await.map_err(value_error)?))
}

async fn migrate_encryption(State(state): State<AppState>) -> AppResult<Json<MigrationResult>> {
    Ok(Json(credentials::migrate_encryption(&state).await.map_err(value_error)?))
}

async fn migrate_from_env(State(state): State<AppState>) -> AppResult<Json<MigrationResult>> {
    Ok(Json(credentials::migrate_from_env(&state).await.map_err(value_error)?))
}

#[cfg(test)]
mod tests {
    use axum::http::StatusCode;
    use serde_json::{json, Value};
    use wiremock::matchers::{header, method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    use crate::db::Params;
    use crate::domain::{Credential, Model};
    use crate::testing::TestApp;

    async fn create(app: &TestApp, body: Value) -> Value {
        let (status, created) = app.post("/api/credentials", body).await;
        assert_eq!(status, StatusCode::CREATED, "{created}");
        created
    }

    async fn model(app: &TestApp, name: &str, provider: &str, credential: Option<&str>) -> String {
        let mut model = Model {
            name: name.into(),
            provider: provider.into(),
            model_type: "language".into(),
            credential: credential.map(String::from),
            ..Model::default()
        };
        model.save(&app.state).await.unwrap();
        model.id
    }

    #[tokio::test]
    async fn crud_masks_the_key_and_encrypts_it_at_rest() {
        let app = TestApp::new().await;
        let created = create(
            &app,
            json!({ "name": "Main", "provider": "openai", "api_key": "sk-very-secret", "modalities": ["language"], "num_ctx": "8192" }),
        )
        .await;
        let id = created["id"].as_str().unwrap().to_string();
        assert!(id.starts_with("credential:"));
        assert_eq!(created["has_api_key"], true);
        assert_eq!(created["model_count"], 0);
        assert_eq!(created["num_ctx"], 8192);
        assert_eq!(created["decryption_error"], Value::Null);
        assert!(!created.to_string().contains("sk-very-secret"));
        assert!(created.get("api_key").is_none());

        // Stored encrypted, readable through the domain record.
        let rows = app.state.db.query("SELECT api_key FROM credential", Params::new()).await.unwrap();
        let stored = rows[0]["api_key"].as_str().unwrap();
        assert!(!stored.is_empty() && !stored.contains("sk-very-secret"));
        let credential = Credential::get(&app.state, &id).await.unwrap();
        assert_eq!(credential.api_key.unwrap().expose(), "sk-very-secret");

        let (status, fetched) = app.get(&format!("/api/credentials/{id}")).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(fetched["name"], "Main");
        assert!(!fetched.to_string().contains("sk-very-secret"));

        model(&app, "gpt-4o", "openai", Some(&id)).await;
        let (status, list) = app.get("/api/credentials").await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(list.as_array().unwrap().len(), 1);
        assert_eq!(list[0]["model_count"], 1);
        let (_, list) = app.get("/api/credentials?provider=anthropic").await;
        assert!(list.as_array().unwrap().is_empty());
        let (_, list) = app.get("/api/credentials/by-provider/OpenAI").await;
        assert_eq!(list.as_array().unwrap().len(), 1);

        // Partial update: absent keeps, null and "" clear, the key changes.
        let (status, updated) = app
            .put(&format!("/api/credentials/{id}"), json!({ "base_url": "https://gateway.test/v1", "project": "p" }))
            .await;
        assert_eq!(status, StatusCode::OK, "{updated}");
        assert_eq!(updated["base_url"], "https://gateway.test/v1");
        let (_, updated) = app.put(&format!("/api/credentials/{id}"), json!({ "name": "Renamed" })).await;
        assert_eq!(updated["base_url"], "https://gateway.test/v1", "absent field keeps its value");
        assert_eq!(updated["name"], "Renamed");
        let (_, updated) = app.put(&format!("/api/credentials/{id}"), json!({ "base_url": null, "project": "" })).await;
        assert_eq!(updated["base_url"], Value::Null);
        assert_eq!(updated["project"], Value::Null);
        let (_, updated) = app.put(&format!("/api/credentials/{id}"), json!({ "num_ctx": 0, "api_key": "sk-new" })).await;
        assert_eq!(updated["num_ctx"], Value::Null);
        assert_eq!(updated["model_count"], 1);
        assert_eq!(Credential::get(&app.state, &id).await.unwrap().api_key.unwrap().expose(), "sk-new");

        // Delete cascades to linked models.
        let (status, deleted) = app.delete(&format!("/api/credentials/{id}")).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(deleted, json!({ "message": "Credential deleted successfully", "deleted_models": 1 }));
        assert!(Model::get_all(&app.state, None).await.unwrap().is_empty());
        let (status, body) = app.get(&format!("/api/credentials/{id}")).await;
        assert_eq!(status, StatusCode::NOT_FOUND);
        assert_eq!(body["detail"], "Credential not found");
    }

    #[tokio::test]
    async fn create_validates_the_provider_and_required_fields() {
        let app = TestApp::new().await;
        let (status, body) = app.post("/api/credentials", json!({ "name": "Test", "provider": "not_a_real_provider" })).await;
        assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY);
        assert!(body["detail"][0]["msg"].as_str().unwrap().contains("'openai'"), "{body}");
        // Provider names are exact.
        let (status, _) = app.post("/api/credentials", json!({ "name": "Test", "provider": "OpenAI" })).await;
        assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY);
        let (status, body) = app.post("/api/credentials", json!({ "name": "Test", "provider": "anthropic_compatible", "api_key": "k" })).await;
        assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY);
        assert!(body["detail"][0]["msg"].as_str().unwrap().contains("require a base URL"), "{body}");
        let (status, _) = app.post("/api/credentials", json!({ "provider": "openai" })).await;
        assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY, "name is required");

        // A metadata URL is rejected with the validator's message.
        let (status, body) = app
            .post("/api/credentials", json!({ "name": "x", "provider": "openai_compatible", "base_url": "http://169.254.169.254/v1" }))
            .await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert!(body["detail"].as_str().unwrap().contains("Link-local"), "{body}");
        assert!(Credential::get_all(&app.state, None).await.unwrap().is_empty());

        // Updating into an invalid state is a 400.
        let created = create(
            &app,
            json!({ "name": "AC", "provider": "anthropic_compatible", "api_key": "k", "base_url": "https://proxy.test" }),
        )
        .await;
        let id = created["id"].as_str().unwrap();
        let (status, body) = app.put(&format!("/api/credentials/{id}"), json!({ "base_url": null })).await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert_eq!(body["detail"], "Anthropic-compatible credentials require a base URL");
        let (status, body) = app.put("/api/credentials/credential:missing", json!({ "name": "x" })).await;
        assert_eq!(status, StatusCode::NOT_FOUND);
        assert_eq!(body["detail"], "Credential not found");
        let (status, _) = app.put(&format!("/api/credentials/{id}"), json!({ "endpoint": "http://169.254.169.254" })).await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn delete_migrates_models_or_reports_missing_records() {
        let app = TestApp::new().await;
        let old = create(&app, json!({ "name": "Old", "provider": "openai", "api_key": "a" })).await;
        let new = create(&app, json!({ "name": "New", "provider": "openai", "api_key": "b" })).await;
        let (old, new) = (old["id"].as_str().unwrap(), new["id"].as_str().unwrap());
        let model_id = model(&app, "gpt-4o", "openai", Some(old)).await;

        let (status, body) = app.delete(&format!("/api/credentials/{old}?migrate_to={new}")).await;
        assert_eq!(status, StatusCode::OK, "{body}");
        assert_eq!(body["deleted_models"], 0);
        assert_eq!(Model::get(&app.state, &model_id).await.unwrap().credential.as_deref(), Some(new));

        let (status, body) = app.delete(&format!("/api/credentials/{old}")).await;
        assert_eq!(status, StatusCode::NOT_FOUND);
        assert_eq!(body["detail"], "Credential not found");
        let (status, body) = app.delete(&format!("/api/credentials/{new}?migrate_to=credential:nope")).await;
        assert_eq!(status, StatusCode::NOT_FOUND, "{body}");
        let (status, _) = app.delete("/api/credentials/model:wrong-table").await;
        assert_eq!(status, StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn status_and_env_status() {
        let app = TestApp::new().await;
        create(&app, json!({ "name": "Main", "provider": "groq", "api_key": "g" })).await;
        let (status, body) = app.get("/api/credentials/status").await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(body["encryption_configured"], true);
        assert_eq!(body["configured"]["groq"], true);
        assert_eq!(body["source"]["groq"], "database");
        assert_eq!(body["configured"].as_object().unwrap().len(), nn_ai::PROVIDERS.len());

        let (status, body) = app.get("/api/credentials/env-status").await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(body.as_object().unwrap().len(), nn_ai::PROVIDERS.len());
        assert!(body.as_object().unwrap().values().all(Value::is_boolean));
    }

    #[tokio::test]
    async fn connection_tests_report_success_bad_keys_and_unreachable_servers() {
        let app = TestApp::new().await;
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/good/v1/models"))
            .and(header("authorization", "Bearer good-key"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({ "data": [{ "id": "qwen3" }, { "id": "nomic-embed" }] })))
            .mount(&server)
            .await;
        Mock::given(method("GET")).and(path("/bad/v1/models")).respond_with(ResponseTemplate::new(401)).mount(&server).await;

        let good = create(
            &app,
            json!({ "name": "Local", "provider": "openai_compatible", "api_key": "good-key", "base_url": format!("{}/good/v1", server.uri()) }),
        )
        .await;
        let (status, body) = app.post(&format!("/api/credentials/{}/test", good["id"].as_str().unwrap()), json!({})).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(body["provider"], "openai_compatible");
        assert_eq!(body["success"], true, "{body}");
        assert!(body["message"].as_str().unwrap().contains("2 models available"), "{body}");

        let bad = create(
            &app,
            json!({ "name": "Bad", "provider": "openai_compatible", "api_key": "wrong", "base_url": format!("{}/bad/v1", server.uri()) }),
        )
        .await;
        let (_, body) = app.post(&format!("/api/credentials/{}/test", bad["id"].as_str().unwrap()), json!({})).await;
        assert_eq!(body["success"], false, "{body}");

        let gone = create(
            &app,
            json!({ "name": "Gone", "provider": "openai_compatible", "base_url": "http://127.0.0.1:1/v1" }),
        )
        .await;
        let (_, body) = app.post(&format!("/api/credentials/{}/test", gone["id"].as_str().unwrap()), json!({})).await;
        assert_eq!(body["success"], false, "{body}");
        assert!(body["message"].as_str().unwrap().contains("connect"), "{body}");

        // A missing credential is a result, not an error.
        let (status, body) = app.post("/api/credentials/credential:missing/test", json!({})).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(body, json!({ "provider": "unknown", "success": false, "message": "Credential not found" }));
    }

    #[tokio::test]
    async fn discover_and_register_models_for_a_credential() {
        let app = TestApp::new().await;
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/v1/models"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({ "data": [{ "id": "qwen3" }, { "id": "text-embedding-nomic" }] })))
            .mount(&server)
            .await;
        let created = create(
            &app,
            json!({ "name": "Local", "provider": "openai_compatible", "base_url": format!("{}/v1", server.uri()) }),
        )
        .await;
        let id = created["id"].as_str().unwrap();

        let (status, body) = app.post(&format!("/api/credentials/{id}/discover"), json!({})).await;
        assert_eq!(status, StatusCode::OK, "{body}");
        assert_eq!(body["credential_id"], id);
        assert_eq!(body["provider"], "openai_compatible");
        let names: Vec<&str> = body["discovered"].as_array().unwrap().iter().map(|m| m["name"].as_str().unwrap()).collect();
        assert_eq!(names, ["qwen3", "text-embedding-nomic"]);
        assert_eq!(body["discovered"][0]["model_type"], Value::Null);

        let register = json!({ "models": [
            { "name": "qwen3", "provider": "openai_compatible", "model_type": "language" },
            { "name": "text-embedding-nomic", "provider": "openai_compatible", "model_type": "embedding" },
        ] });
        let (status, body) = app.post(&format!("/api/credentials/{id}/register-models"), register.clone()).await;
        assert_eq!(status, StatusCode::OK, "{body}");
        assert_eq!(body, json!({ "created": 2, "existing": 0 }));
        let (_, body) = app.post(&format!("/api/credentials/{id}/register-models"), register).await;
        assert_eq!(body, json!({ "created": 0, "existing": 2 }));
        let models = Model::get_by_credential(&app.state, id).await.unwrap();
        assert_eq!(models.len(), 2);
        let (_, fetched) = app.get(&format!("/api/credentials/{id}")).await;
        assert_eq!(fetched["model_count"], 2);

        let (status, _) = app.post("/api/credentials/credential:missing/discover", json!({})).await;
        assert_eq!(status, StatusCode::NOT_FOUND);
        let (status, _) = app.post("/api/credentials/credential:missing/register-models", json!({ "models": [] })).await;
        assert_eq!(status, StatusCode::NOT_FOUND);
        let (status, _) = app.post(&format!("/api/credentials/{id}/register-models"), json!({})).await;
        assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY);
    }

    #[tokio::test]
    async fn migration_endpoints_answer_with_their_reports() {
        let app = TestApp::new().await;
        let (status, body) = app.post("/api/credentials/migrate-from-env", json!({})).await;
        assert_eq!(status, StatusCode::OK, "{body}");
        assert!(body["message"].as_str().unwrap().starts_with("Migration complete. Migrated"));
        assert!(body["not_configured"].is_array());

        let (status, body) = app.post("/api/credentials/migrate-from-provider-config", json!({})).await;
        assert_eq!(status, StatusCode::OK, "{body}");
        assert_eq!(body["message"], "Migration complete. Migrated 0 credentials.");
        assert!(body.get("not_configured").is_none());

        create(&app, json!({ "name": "Main", "provider": "openai", "api_key": "sk" })).await;
        let (status, body) = app.post("/api/credentials/migrate-encryption", json!({})).await;
        assert_eq!(status, StatusCode::OK, "{body}");
        assert!(body["message"].as_str().unwrap().starts_with("Encryption migration complete."), "{body}");
    }
}
