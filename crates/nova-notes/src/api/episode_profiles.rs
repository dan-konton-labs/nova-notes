//! Episode profiles: `/api/episode-profiles/...`.
//!
//! The Rust counterpart of `api/routers/episode_profiles.py`. `GET` of one
//! profile takes its name; update, delete and duplicate take its record id,
//! as in the Python router.

use axum::extract::{Path, State};
use axum::routing::{get, post};
use axum::{Json, Router};
use serde_json::{json, Value};

use crate::api::ApiJson;
use crate::error::AppResult;
use crate::services::podcasts::{self, EpisodeProfileInput, EpisodeProfileView};
use crate::state::AppState;

pub fn router() -> Router<AppState> {
    Router::new()
        .route("/episode-profiles", get(list).post(create))
        .route("/episode-profiles/{profile}", get(get_by_name).put(update).delete(delete))
        .route("/episode-profiles/{profile}/duplicate", post(duplicate))
}

async fn list(State(state): State<AppState>) -> AppResult<Json<Vec<EpisodeProfileView>>> {
    Ok(Json(podcasts::list_episode_profiles(&state).await?))
}

async fn get_by_name(State(state): State<AppState>, Path(name): Path<String>) -> AppResult<Json<EpisodeProfileView>> {
    Ok(Json(podcasts::get_episode_profile(&state, &name).await?))
}

async fn create(
    State(state): State<AppState>,
    ApiJson(body): ApiJson<EpisodeProfileInput>,
) -> AppResult<Json<EpisodeProfileView>> {
    Ok(Json(podcasts::create_episode_profile(&state, &body).await?))
}

async fn update(
    State(state): State<AppState>,
    Path(profile_id): Path<String>,
    ApiJson(body): ApiJson<EpisodeProfileInput>,
) -> AppResult<Json<EpisodeProfileView>> {
    Ok(Json(podcasts::update_episode_profile(&state, &profile_id, &body).await?))
}

async fn delete(State(state): State<AppState>, Path(profile_id): Path<String>) -> AppResult<Json<Value>> {
    podcasts::delete_episode_profile(&state, &profile_id).await?;
    Ok(Json(json!({ "message": "Episode profile deleted successfully" })))
}

async fn duplicate(State(state): State<AppState>, Path(profile_id): Path<String>) -> AppResult<Json<EpisodeProfileView>> {
    Ok(Json(podcasts::duplicate_episode_profile(&state, &profile_id).await?))
}

#[cfg(test)]
pub(crate) mod tests {
    use axum::http::StatusCode;
    use serde_json::{json, Value};

    use crate::api::speaker_profiles::tests::{create_speaker_profile, model};
    use crate::testing::TestApp;

    /// Create an episode profile through the API; returns its body.
    pub(crate) async fn create_episode_profile(app: &TestApp, name: &str, speaker: &str, llm: &str) -> Value {
        let (status, body) = app
            .post(
                "/api/episode-profiles",
                json!({
                    "name": name,
                    "speaker_config": speaker,
                    "outline_llm": llm,
                    "transcript_llm": llm,
                    "default_briefing": "Discuss the material.",
                    "num_segments": 3,
                }),
            )
            .await;
        assert_eq!(status, StatusCode::OK, "{body}");
        body
    }

    #[tokio::test]
    async fn episode_profile_crud() {
        let app = TestApp::new().await;
        let llm = model(&app, "llm1", "language").await;
        let speaker = create_speaker_profile(&app, "zz_speakers", None).await;
        let speaker_id = speaker["id"].as_str().unwrap();

        // By record id.
        let created = create_episode_profile(&app, "zz_show", speaker_id, &llm).await;
        let id = created["id"].as_str().unwrap().to_string();
        assert!(id.starts_with("episode_profile:"));
        assert_eq!(created["speaker_config"], speaker_id);
        assert_eq!(created["speaker_config_name"], "zz_speakers");
        assert_eq!(created["outline_llm"], llm.as_str());
        assert_eq!(created["description"], "");
        assert_eq!(created["num_segments"], 3);
        assert_eq!(created["max_tokens"], Value::Null);

        // By name (backward compatibility): stored as the record id.
        let (status, by_name) = app
            .post(
                "/api/episode-profiles",
                json!({ "name": "zz_named", "speaker_config": "zz_speakers", "default_briefing": "b" }),
            )
            .await;
        assert_eq!(status, StatusCode::OK, "{by_name}");
        assert_eq!(by_name["speaker_config"], speaker_id);
        assert_eq!(by_name["num_segments"], 5);

        let (status, body) = app.get("/api/episode-profiles/zz_show").await;
        assert_eq!((status, body["id"].as_str()), (StatusCode::OK, Some(id.as_str())));
        assert_eq!(body["speaker_config_name"], "zz_speakers");
        let (_, list) = app.get("/api/episode-profiles").await;
        let listed = list.as_array().unwrap().iter().find(|p| p["id"] == id.as_str()).unwrap();
        assert_eq!(listed["speaker_config_name"], "zz_speakers");

        // Validation.
        let (status, body) = app
            .post(
                "/api/episode-profiles",
                json!({ "name": "bad", "speaker_config": "nobody", "default_briefing": "b" }),
            )
            .await;
        assert_eq!((status, body), (StatusCode::BAD_REQUEST, json!({ "detail": "Speaker profile 'nobody' not found" })));
        for segments in [2, 21] {
            let (status, body) = app
                .post(
                    "/api/episode-profiles",
                    json!({ "name": "bad", "speaker_config": speaker_id, "default_briefing": "b", "num_segments": segments }),
                )
                .await;
            assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
            assert_eq!(body["detail"], "Number of segments must be between 3 and 20");
        }
        let (status, _) = app.post("/api/episode-profiles", json!({ "name": "bad", "speaker_config": speaker_id })).await;
        assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY);

        // Update: fields left out keep their values.
        let (status, body) = app
            .put(
                &format!("/api/episode-profiles/{id}"),
                json!({ "name": "zz_show", "speaker_config": speaker_id, "default_briefing": "New briefing", "max_tokens": 4000 }),
            )
            .await;
        assert_eq!(status, StatusCode::OK, "{body}");
        assert_eq!(body["default_briefing"], "New briefing");
        assert_eq!(body["max_tokens"], 4000);
        assert_eq!(body["num_segments"], 3);
        assert_eq!(body["transcript_llm"], llm.as_str());

        let (status, copy) = app.post(&format!("/api/episode-profiles/{id}/duplicate"), json!({})).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(copy["name"], "zz_show - Copy");
        assert_eq!(copy["speaker_config_name"], "zz_speakers");
        assert_eq!(copy["max_tokens"], 4000);

        // Deleting the speaker profile orphans the episode profile.
        app.delete(&format!("/api/speaker-profiles/{speaker_id}")).await;
        let (_, body) = app.get("/api/episode-profiles/zz_show").await;
        assert_eq!(body["speaker_config_name"], Value::Null);

        let (status, body) = app.delete(&format!("/api/episode-profiles/{id}")).await;
        assert_eq!((status, body), (StatusCode::OK, json!({ "message": "Episode profile deleted successfully" })));
        let (status, body) = app.get("/api/episode-profiles/zz_show").await;
        assert_eq!((status, body), (StatusCode::NOT_FOUND, json!({ "detail": "Episode profile 'zz_show' not found" })));
        let (status, _) = app.delete(&format!("/api/episode-profiles/{id}")).await;
        assert_eq!(status, StatusCode::NOT_FOUND);
        let (status, _) = app.post(&format!("/api/episode-profiles/{id}/duplicate"), json!({})).await;
        assert_eq!(status, StatusCode::NOT_FOUND);
    }
}
