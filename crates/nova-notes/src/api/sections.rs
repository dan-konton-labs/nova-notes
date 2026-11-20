//! Sections, the top level of section > folder > notebook: `/api/sections/...`.

use axum::extract::{Path, State};
use axum::routing::{get, put};
use axum::{Json, Router};

use super::notebooks::ApiJson;
use crate::error::AppResult;
use crate::services::notebooks::{self, SectionDeleted, SectionInput, SectionSummary};
use crate::state::AppState;

pub fn router() -> Router<AppState> {
    Router::new()
        .route("/sections", get(list_sections).post(create_section))
        .route("/sections/{section_id}", put(rename_section).delete(delete_section))
}

async fn list_sections(State(state): State<AppState>) -> AppResult<Json<Vec<SectionSummary>>> {
    Ok(Json(notebooks::list_sections(&state).await?))
}

async fn create_section(State(state): State<AppState>, ApiJson(body): ApiJson<SectionInput>) -> AppResult<Json<SectionSummary>> {
    Ok(Json(notebooks::create_section(&state, &body).await?))
}

async fn rename_section(
    State(state): State<AppState>,
    Path(section_id): Path<String>,
    ApiJson(body): ApiJson<SectionInput>,
) -> AppResult<Json<SectionSummary>> {
    Ok(Json(notebooks::rename_section(&state, &section_id, &body).await?))
}

async fn delete_section(State(state): State<AppState>, Path(section_id): Path<String>) -> AppResult<Json<SectionDeleted>> {
    Ok(Json(notebooks::delete_section(&state, &section_id).await?))
}

#[cfg(test)]
mod tests {
    use axum::http::StatusCode;
    use serde_json::{json, Value};

    use crate::domain::{Folder, Section};
    use crate::testing::TestApp;

    async fn section(app: &TestApp, name: &str) -> String {
        let (status, body) = app.post("/api/sections", json!({ "name": name })).await;
        assert_eq!(status, StatusCode::OK, "{body}");
        body["id"].as_str().unwrap().to_string()
    }

    #[tokio::test]
    async fn sections_are_listed_by_name_with_folder_counts() {
        let app = TestApp::new().await;
        let (status, body) = app.get("/api/sections").await;
        assert_eq!((status, body), (StatusCode::OK, json!([])));

        let work = section(&app, "work").await;
        section(&app, "Personal").await;
        for name in ["A", "B"] {
            app.post("/api/folders", json!({ "name": name, "section_id": work })).await;
        }
        app.post("/api/folders", json!({ "name": "Loose" })).await;

        let (status, body) = app.get("/api/sections").await;
        assert_eq!(status, StatusCode::OK);
        let names: Vec<&str> = body.as_array().unwrap().iter().map(|s| s["name"].as_str().unwrap()).collect();
        assert_eq!(names, ["Personal", "work"], "case-insensitive name order");
        assert_eq!(body[0]["folder_count"], 0);
        assert_eq!(body[1]["folder_count"], 2);
        let mut keys: Vec<&String> = body[1].as_object().unwrap().keys().collect();
        keys.sort();
        assert_eq!(keys, ["created", "folder_count", "id", "name", "updated"]);
        assert!(chrono::DateTime::parse_from_rfc3339(body[1]["created"].as_str().unwrap()).is_ok());
    }

    #[tokio::test]
    async fn names_are_trimmed_bounded_and_unique() {
        let app = TestApp::new().await;
        let (status, body) = app.post("/api/sections", json!({ "name": "  Personal " })).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!((&body["name"], &body["folder_count"]), (&json!("Personal"), &json!(0)));
        assert_eq!(Section::get(&app.state, body["id"].as_str().unwrap()).await.unwrap().name, "Personal");

        for (name, detail) in [
            ("", "Section name cannot be empty"),
            ("   ", "Section name cannot be empty"),
            (&*"x".repeat(81), "Section name cannot be longer than 80 characters"),
        ] {
            let (status, body) = app.post("/api/sections", json!({ "name": name })).await;
            assert_eq!(status, StatusCode::BAD_REQUEST, "{name:?}");
            assert_eq!(body["detail"], detail);
        }
        assert_eq!(app.post("/api/sections", json!({ "name": "x".repeat(80) })).await.0, StatusCode::OK);

        let (status, body) = app.post("/api/sections", json!({ "name": " personal" })).await;
        assert_eq!(status, StatusCode::CONFLICT);
        assert_eq!(body["detail"], "A section named 'personal' already exists");
        assert_eq!(Section::list(&app.state).await.unwrap().len(), 2);

        let (status, body) = app.post("/api/sections", json!({})).await;
        assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY);
        assert_eq!(body["detail"][0]["loc"], json!(["body"]));
    }

    #[tokio::test]
    async fn rename_a_section() {
        let app = TestApp::new().await;
        let personal = section(&app, "Personal").await;
        section(&app, "Work").await;
        app.post("/api/folders", json!({ "name": "F", "section_id": personal })).await;
        let path = format!("/api/sections/{personal}");

        let (status, body) = app.put(&path, json!({ "name": " Home " })).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!((&body["name"], &body["folder_count"], &body["id"]), (&json!("Home"), &json!(1), &json!(personal)));

        // Its own name in another case is fine; another section's name is not.
        assert_eq!(app.put(&path, json!({ "name": "HOME" })).await.0, StatusCode::OK);
        let (status, body) = app.put(&path, json!({ "name": "work" })).await;
        assert_eq!(status, StatusCode::CONFLICT);
        assert_eq!(body["detail"], "A section named 'work' already exists");
        let (status, body) = app.put(&path, json!({ "name": " " })).await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert_eq!(body["detail"], "Section name cannot be empty");
        assert_eq!(Section::get(&app.state, &personal).await.unwrap().name, "HOME");
    }

    #[tokio::test]
    async fn missing_sections_are_404() {
        let app = TestApp::new().await;
        let (_, folder) = app.post("/api/folders", json!({ "name": "Not a section" })).await;
        let folder_id = folder["id"].as_str().unwrap();
        for gone in ["section:x", folder_id] {
            let path = format!("/api/sections/{gone}");
            for (status, body) in [app.put(&path, json!({ "name": "A" })).await, app.delete(&path).await] {
                assert_eq!(status, StatusCode::NOT_FOUND, "{gone}");
                assert_eq!(body["detail"], "Section not found");
            }
        }
        assert!(Folder::get(&app.state, folder_id).await.is_ok(), "a folder id never deletes the folder");
    }

    #[tokio::test]
    async fn deleting_a_section_keeps_its_folders() {
        let app = TestApp::new().await;
        let personal = section(&app, "Personal").await;
        let other = section(&app, "Other").await;
        for (name, section_id) in [("a", &personal), ("b", &personal), ("c", &other)] {
            app.post("/api/folders", json!({ "name": name, "section_id": section_id })).await;
        }

        let (status, body) = app.delete(&format!("/api/sections/{personal}")).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(body, json!({ "message": "Section deleted successfully", "folders_ungrouped": 2 }));
        assert!(Section::get(&app.state, &personal).await.unwrap_err().is_not_found());

        let (_, folders) = app.get("/api/folders").await;
        let placement: Vec<(&str, &Value)> =
            folders.as_array().unwrap().iter().map(|f| (f["name"].as_str().unwrap(), &f["section_id"])).collect();
        assert_eq!(placement, [("a", &Value::Null), ("b", &Value::Null), ("c", &json!(other))]);
    }
}
