//! Notebook folders, one level of grouping inside a section: `/api/folders/...`.

use axum::extract::{Path, State};
use axum::routing::{get, put};
use axum::{Json, Router};

use super::notebooks::ApiJson;
use crate::error::AppResult;
use crate::services::notebooks::{self, FolderDeleted, FolderInput, FolderSummary};
use crate::state::AppState;

pub fn router() -> Router<AppState> {
    Router::new()
        .route("/folders", get(list_folders).post(create_folder))
        .route("/folders/{folder_id}", put(update_folder).delete(delete_folder))
}

async fn list_folders(State(state): State<AppState>) -> AppResult<Json<Vec<FolderSummary>>> {
    Ok(Json(notebooks::list_folders(&state).await?))
}

async fn create_folder(State(state): State<AppState>, ApiJson(body): ApiJson<FolderInput>) -> AppResult<Json<FolderSummary>> {
    Ok(Json(notebooks::create_folder(&state, &body).await?))
}

async fn update_folder(
    State(state): State<AppState>,
    Path(folder_id): Path<String>,
    ApiJson(body): ApiJson<FolderInput>,
) -> AppResult<Json<FolderSummary>> {
    Ok(Json(notebooks::update_folder(&state, &folder_id, &body).await?))
}

async fn delete_folder(State(state): State<AppState>, Path(folder_id): Path<String>) -> AppResult<Json<FolderDeleted>> {
    Ok(Json(notebooks::delete_folder(&state, &folder_id).await?))
}

#[cfg(test)]
mod tests {
    use axum::http::StatusCode;
    use serde_json::{json, Value};

    use crate::domain::{Folder, Notebook};
    use crate::testing::TestApp;

    async fn folder(app: &TestApp, body: Value) -> Value {
        let (status, folder) = app.post("/api/folders", body).await;
        assert_eq!(status, StatusCode::OK, "{folder}");
        folder
    }

    async fn section(app: &TestApp, name: &str) -> String {
        let (_, body) = app.post("/api/sections", json!({ "name": name })).await;
        body["id"].as_str().unwrap().to_string()
    }

    #[tokio::test]
    async fn folders_are_listed_by_name_with_counts_and_section() {
        let app = TestApp::new().await;
        let (status, body) = app.get("/api/folders").await;
        assert_eq!((status, body), (StatusCode::OK, json!([])));

        let crime = section(&app, "Personal").await;
        let beta = folder(&app, json!({ "name": "beta", "section_id": crime })).await;
        folder(&app, json!({ "name": "Zed" })).await;
        folder(&app, json!({ "name": "Alpha" })).await;
        let beta_id = beta["id"].as_str().unwrap();
        for name in ["one", "two"] {
            Notebook::create(&app.state, name, "", Some(beta_id)).await.unwrap();
        }

        let (status, body) = app.get("/api/folders").await;
        assert_eq!(status, StatusCode::OK);
        let names: Vec<&str> = body.as_array().unwrap().iter().map(|f| f["name"].as_str().unwrap()).collect();
        assert_eq!(names, ["Alpha", "beta", "Zed"]);
        assert_eq!(
            body[1],
            json!({
                "id": beta_id,
                "name": "beta",
                "created": beta["created"],
                "updated": body[1]["updated"],
                "notebook_count": 2,
                "section_id": crime,
            })
        );
        assert_eq!((&body[0]["notebook_count"], &body[0]["section_id"]), (&json!(0), &Value::Null));
    }

    #[tokio::test]
    async fn names_are_trimmed_bounded_and_unique_per_section() {
        let app = TestApp::new().await;
        let created = folder(&app, json!({ "name": "  Research  " })).await;
        assert_eq!((&created["name"], &created["notebook_count"], &created["section_id"]), (&json!("Research"), &json!(0), &Value::Null));
        assert!(chrono::DateTime::parse_from_rfc3339(created["created"].as_str().unwrap()).is_ok());

        for (name, detail) in [
            ("", "Folder name cannot be empty"),
            ("   ", "Folder name cannot be empty"),
            (&*"x".repeat(81), "Folder name cannot be longer than 80 characters"),
        ] {
            let (status, body) = app.post("/api/folders", json!({ "name": name })).await;
            assert_eq!(status, StatusCode::BAD_REQUEST, "{name:?}");
            assert_eq!(body["detail"], detail);
        }
        folder(&app, json!({ "name": "x".repeat(80) })).await;

        let (status, body) = app.post("/api/folders", json!({ "name": " rESEARCH " })).await;
        assert_eq!(status, StatusCode::CONFLICT);
        assert_eq!(body["detail"], "A folder named 'rESEARCH' already exists");

        // The same name is fine in another section, not twice in one.
        let personal = section(&app, "Personal").await;
        let work = section(&app, "Work").await;
        folder(&app, json!({ "name": "Misc", "section_id": personal })).await;
        folder(&app, json!({ "name": "misc", "section_id": work })).await;
        folder(&app, json!({ "name": "MISC" })).await;
        let (status, body) = app.post("/api/folders", json!({ "name": "misc", "section_id": personal })).await;
        assert_eq!(status, StatusCode::CONFLICT);
        assert_eq!(body["detail"], "A folder named 'misc' already exists");

        // A blank section id means no section; an unknown one is a 404 and creates nothing.
        let blank = folder(&app, json!({ "name": "Blank section", "section_id": "  " })).await;
        assert_eq!(blank["section_id"], Value::Null);
        let before = Folder::list(&app.state).await.unwrap().len();
        let notebook = Notebook::create(&app.state, "NB", "", None).await.unwrap();
        for section_id in ["section:nope", notebook.id.as_str(), "nonsense"] {
            let (status, body) = app.post("/api/folders", json!({ "name": "X", "section_id": section_id })).await;
            assert_eq!(status, StatusCode::NOT_FOUND, "{section_id}");
            assert_eq!(body["detail"], "Section not found");
        }
        assert_eq!(Folder::list(&app.state).await.unwrap().len(), before);

        let (status, body) = app.post("/api/folders", json!({ "section_id": personal })).await;
        assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY);
        assert_eq!(body["detail"][0]["loc"], json!(["body"]));
    }

    #[tokio::test]
    async fn rename_and_move_a_folder() {
        let app = TestApp::new().await;
        let personal = section(&app, "Personal").await;
        let work = folder(&app, json!({ "name": "Work", "section_id": personal })).await;
        folder(&app, json!({ "name": "Play", "section_id": personal })).await;
        let id = work["id"].as_str().unwrap();
        let path = format!("/api/folders/{id}");
        for name in ["a", "b", "c"] {
            Notebook::create(&app.state, name, "", Some(id)).await.unwrap();
        }

        let (status, body) = app.put(&path, json!({ "name": " Jobs " })).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!((&body["name"], &body["notebook_count"]), (&json!("Jobs"), &json!(3)));
        assert_eq!(body["section_id"], personal, "an absent section_id keeps the section");

        assert_eq!(app.put(&path, json!({ "name": "JOBS" })).await.0, StatusCode::OK, "its own name in another case");
        let (status, body) = app.put(&path, json!({ "name": "play" })).await;
        assert_eq!(status, StatusCode::CONFLICT);
        assert_eq!(body["detail"], "A folder named 'play' already exists");
        let (status, body) = app.put(&path, json!({ "name": " " })).await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert_eq!(body["detail"], "Folder name cannot be empty");

        // Out of its section, the name no longer clashes with "Play".
        let (status, body) = app.put(&path, json!({ "name": "Play", "section_id": null })).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!((&body["name"], &body["section_id"]), (&json!("Play"), &Value::Null));
        assert_eq!(Folder::get(&app.state, id).await.unwrap().section, None);

        // Moving back into the section where "Play" exists is a conflict.
        let (status, _) = app.put(&path, json!({ "name": "Play", "section_id": personal })).await;
        assert_eq!(status, StatusCode::CONFLICT);
        let (status, body) = app.put(&path, json!({ "name": "Jobs", "section_id": personal })).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(body["section_id"], personal);
        let (status, body) = app.put(&path, json!({ "name": "Jobs", "section_id": "" })).await;
        assert_eq!((status, &body["section_id"]), (StatusCode::OK, &Value::Null));

        let (status, body) = app.put(&path, json!({ "name": "Jobs", "section_id": "section:gone" })).await;
        assert_eq!(status, StatusCode::NOT_FOUND);
        assert_eq!(body["detail"], "Section not found");
    }

    #[tokio::test]
    async fn missing_folders_are_404() {
        let app = TestApp::new().await;
        let notebook = Notebook::create(&app.state, "N", "", None).await.unwrap();
        for gone in ["folder:x", notebook.id.as_str()] {
            let path = format!("/api/folders/{gone}");
            for (status, body) in [app.put(&path, json!({ "name": "A" })).await, app.delete(&path).await] {
                assert_eq!(status, StatusCode::NOT_FOUND, "{gone}");
                assert_eq!(body["detail"], "Folder not found");
            }
        }
        assert!(Notebook::get(&app.state, &notebook.id).await.is_ok(), "a notebook id never deletes the notebook");
    }

    #[tokio::test]
    async fn deleting_a_folder_ungroups_its_notebooks() {
        let app = TestApp::new().await;
        let doomed = folder(&app, json!({ "name": "Doomed" })).await;
        let kept = folder(&app, json!({ "name": "Kept" })).await;
        let (doomed, kept) = (doomed["id"].as_str().unwrap(), kept["id"].as_str().unwrap());
        let a = Notebook::create(&app.state, "a", "", Some(doomed)).await.unwrap();
        Notebook::create(&app.state, "b", "", Some(doomed)).await.unwrap();
        let c = Notebook::create(&app.state, "c", "", Some(kept)).await.unwrap();

        let (status, body) = app.delete(&format!("/api/folders/{doomed}")).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(body, json!({ "message": "Folder deleted successfully", "notebooks_ungrouped": 2 }));
        assert!(Folder::get(&app.state, doomed).await.unwrap_err().is_not_found());
        assert_eq!(Notebook::get(&app.state, &a.id).await.unwrap().folder, None);
        assert_eq!(Notebook::get(&app.state, &c.id).await.unwrap().folder.as_deref(), Some(kept));
        let (_, notebooks) = app.get("/api/notebooks?folder_id=none").await;
        assert_eq!(notebooks.as_array().unwrap().len(), 2);
    }
}
