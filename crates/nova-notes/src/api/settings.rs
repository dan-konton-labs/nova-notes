//! Content-processing settings: `/api/settings`.

use axum::extract::State;
use axum::routing::get;
use axum::{Json, Router};

use super::notebooks::ApiJson;
use crate::domain::ContentSettings;
use crate::error::AppResult;
use crate::services::settings::{self, SettingsUpdate};
use crate::state::AppState;

pub fn router() -> Router<AppState> {
    Router::new().route("/settings", get(get_settings).put(update_settings))
}

async fn get_settings(State(state): State<AppState>) -> AppResult<Json<ContentSettings>> {
    Ok(Json(settings::get_settings(&state).await?))
}

async fn update_settings(State(state): State<AppState>, ApiJson(body): ApiJson<SettingsUpdate>) -> AppResult<Json<ContentSettings>> {
    Ok(Json(settings::update_settings(&state, &body).await?))
}

#[cfg(test)]
mod tests {
    use axum::http::StatusCode;
    use serde_json::json;

    use crate::domain::ContentSettings;
    use crate::testing::TestApp;

    fn defaults() -> serde_json::Value {
        json!({
            "default_content_processing_engine_doc": "auto",
            "default_content_processing_engine_url": "auto",
            "default_embedding_option": "ask",
            "auto_delete_files": "yes",
            "docling_ocr": true,
            "docling_formulas": false,
            "docling_vision": false,
            "youtube_preferred_languages": ["en", "pt", "es", "de", "nl", "en-GB", "fr", "hi", "ja", "zh-CN", "zh-TW"],
        })
    }

    #[tokio::test]
    async fn settings_start_at_the_defaults() {
        let app = TestApp::new().await;
        let (status, body) = app.get("/api/settings").await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(body, defaults());
    }

    #[tokio::test]
    async fn updates_change_only_what_is_sent_and_persist() {
        let app = TestApp::new().await;
        let (status, body) = app
            .put(
                "/api/settings",
                json!({
                    "default_content_processing_engine_url": "crawl4ai",
                    "docling_ocr": false,
                    "youtube_preferred_languages": ["de"],
                    "auto_delete_files": null,
                }),
            )
            .await;
        assert_eq!(status, StatusCode::OK);
        let mut expected = defaults();
        expected["default_content_processing_engine_url"] = json!("crawl4ai");
        expected["docling_ocr"] = json!(false);
        expected["youtube_preferred_languages"] = json!(["de"]);
        assert_eq!(body, expected);
        assert_eq!(app.get("/api/settings").await.1, expected);
        assert_eq!(ContentSettings::load(&app.state).await.unwrap().docling_ocr, Some(false));

        let (status, body) = app.put("/api/settings", json!({ "auto_delete_files": "no", "default_embedding_option": "always" })).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!((&body["auto_delete_files"], &body["default_embedding_option"]), (&json!("no"), &json!("always")));
        assert_eq!(body["default_content_processing_engine_url"], "crawl4ai", "earlier changes stay");
        assert_eq!(app.put("/api/settings", json!({})).await.1, body);
    }

    #[tokio::test]
    async fn invalid_settings_change_nothing() {
        let app = TestApp::new().await;
        for (field, value) in [
            ("default_content_processing_engine_doc", "crawl4ai"),
            ("default_content_processing_engine_url", "docling"),
            ("default_embedding_option", "sometimes"),
            ("auto_delete_files", "maybe"),
        ] {
            let (status, body) =
                app.put("/api/settings", json!({ field: value, "docling_vision": true })).await;
            assert_eq!(status, StatusCode::BAD_REQUEST, "{field}");
            let detail = body["detail"].as_str().unwrap();
            assert!(detail.starts_with(&format!("Invalid {field}: '{value}'. Allowed values: ")), "{detail}");
        }
        let (status, body) = app.put("/api/settings", json!({ "docling_ocr": "yes please" })).await;
        assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY);
        assert_eq!(body["detail"][0]["loc"], json!(["body"]));
        assert_eq!(app.get("/api/settings").await.1, defaults());
    }
}
