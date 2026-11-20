//! `GET /api/capabilities`: which opt-in extraction runtimes this build can use.

use axum::routing::get;
use axum::{Json, Router};

use crate::services::settings::{self, Capabilities};
use crate::state::AppState;

pub fn router() -> Router<AppState> {
    Router::new().route("/capabilities", get(get_capabilities))
}

async fn get_capabilities() -> Json<Capabilities> {
    Json(settings::capabilities())
}

#[cfg(test)]
mod tests {
    use axum::http::StatusCode;
    use serde_json::json;

    use crate::config::env_nonempty;
    use crate::services::settings::Capabilities;
    use crate::testing::TestApp;

    #[tokio::test]
    async fn reports_the_three_flags() {
        let app = TestApp::new().await;
        let (status, body) = app.get("/api/capabilities").await;
        assert_eq!(status, StatusCode::OK);
        // No Docling and no local Crawl4AI here; a remote server only when configured.
        let remote = env_nonempty("CRAWL4AI_API_URL").is_some();
        assert_eq!(
            body,
            json!({ "docling_available": false, "crawl4ai_available": remote, "crawl4ai_remote_configured": remote })
        );
        assert_eq!(body, serde_json::to_value(Capabilities::with_crawl4ai_api_url(env_nonempty("CRAWL4AI_API_URL").as_deref())).unwrap());
    }
}
