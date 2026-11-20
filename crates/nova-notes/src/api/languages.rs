//! `GET /api/languages`: the language choices for podcasts and transcripts.

use axum::routing::get;
use axum::{Json, Router};

use crate::services::settings::{self, Language};
use crate::state::AppState;

pub fn router() -> Router<AppState> {
    Router::new().route("/languages", get(list_languages))
}

async fn list_languages() -> Json<Vec<Language>> {
    Json(settings::languages())
}

#[cfg(test)]
mod tests {
    use axum::http::StatusCode;
    use serde_json::json;

    use crate::testing::TestApp;

    #[tokio::test]
    async fn languages_are_bcp47_codes_sorted_by_name() {
        let app = TestApp::new().await;
        let (status, body) = app.get("/api/languages").await;
        assert_eq!(status, StatusCode::OK);
        let languages = body.as_array().unwrap();
        assert_eq!(languages.len(), 199);
        assert_eq!(languages[0], json!({ "code": "ab-GE", "name": "Abkhazian (Georgia)" }));
        assert!(languages.contains(&json!({ "code": "pt-BR", "name": "Portuguese (Brazil)" })));
        assert!(languages.contains(&json!({ "code": "en-GB", "name": "English (United Kingdom)" })));
        let names: Vec<&str> = languages.iter().map(|l| l["name"].as_str().unwrap()).collect();
        assert!(names.windows(2).all(|pair| pair[0] <= pair[1]));
    }
}
