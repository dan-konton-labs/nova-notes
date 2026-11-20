//! `GET /api/providers`: the provider registry with environment detection
//! (Python `api/routers/providers.py`).

use axum::routing::get;
use axum::{Json, Router};

use crate::services::credentials::{self, ProviderInfoResponse};
use crate::state::AppState;

pub fn router() -> Router<AppState> {
    Router::new().route("/providers", get(list_providers))
}

async fn list_providers() -> Json<Vec<ProviderInfoResponse>> {
    Json(credentials::list_providers())
}

#[cfg(test)]
mod tests {
    use axum::http::StatusCode;

    use crate::testing::TestApp;

    #[tokio::test]
    async fn lists_every_registry_provider_with_metadata() {
        let app = TestApp::new().await;
        let (status, body) = app.get("/api/providers").await;
        assert_eq!(status, StatusCode::OK);
        let providers = body.as_array().unwrap();
        assert_eq!(providers.len(), nn_ai::PROVIDERS.len());
        assert_eq!(providers.len(), 24);
        let openai = providers.iter().find(|p| p["name"] == "openai").unwrap();
        assert_eq!(openai["display_name"], "OpenAI");
        assert!(openai["modalities"].as_array().unwrap().iter().any(|m| m == "language"));
        assert!(openai["env_configured"].is_boolean());
        let mut keys: Vec<&String> = openai.as_object().unwrap().keys().collect();
        keys.sort();
        assert_eq!(keys, ["display_name", "docs_url", "env_configured", "modalities", "name"]);
    }
}
