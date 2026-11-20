//! Speaker profiles: `/api/speaker-profiles/...`.
//!
//! The Rust counterpart of `api/routers/speaker_profiles.py`. `GET` of one
//! profile takes its name; update, delete and duplicate take its record id,
//! as in the Python router.

use axum::extract::{Path, State};
use axum::routing::{get, post};
use axum::{Json, Router};
use serde_json::{json, Value};

use crate::api::ApiJson;
use crate::error::AppResult;
use crate::services::podcasts::{self, SpeakerProfileInput, SpeakerProfileView};
use crate::state::AppState;

pub fn router() -> Router<AppState> {
    Router::new()
        .route("/speaker-profiles", get(list).post(create))
        .route("/speaker-profiles/{profile}", get(get_by_name).put(update).delete(delete))
        .route("/speaker-profiles/{profile}/duplicate", post(duplicate))
}

async fn list(State(state): State<AppState>) -> AppResult<Json<Vec<SpeakerProfileView>>> {
    Ok(Json(podcasts::list_speaker_profiles(&state).await?))
}

async fn get_by_name(State(state): State<AppState>, Path(name): Path<String>) -> AppResult<Json<SpeakerProfileView>> {
    Ok(Json(podcasts::get_speaker_profile(&state, &name).await?))
}

async fn create(
    State(state): State<AppState>,
    ApiJson(body): ApiJson<SpeakerProfileInput>,
) -> AppResult<Json<SpeakerProfileView>> {
    Ok(Json(podcasts::create_speaker_profile(&state, &body).await?))
}

async fn update(
    State(state): State<AppState>,
    Path(profile_id): Path<String>,
    ApiJson(body): ApiJson<SpeakerProfileInput>,
) -> AppResult<Json<SpeakerProfileView>> {
    Ok(Json(podcasts::update_speaker_profile(&state, &profile_id, &body).await?))
}

async fn delete(State(state): State<AppState>, Path(profile_id): Path<String>) -> AppResult<Json<Value>> {
    podcasts::delete_speaker_profile(&state, &profile_id).await?;
    Ok(Json(json!({ "message": "Speaker profile deleted successfully" })))
}

async fn duplicate(State(state): State<AppState>, Path(profile_id): Path<String>) -> AppResult<Json<SpeakerProfileView>> {
    Ok(Json(podcasts::duplicate_speaker_profile(&state, &profile_id).await?))
}

#[cfg(test)]
pub(crate) mod tests {
    use axum::http::StatusCode;
    use serde_json::{json, Value};

    use crate::testing::TestApp;

    /// A `model` record of the given type, for profile references.
    pub(crate) async fn model(app: &TestApp, key: &str, model_type: &str) -> String {
        let doc = crate::db::Doc::new().set("name", key).set("provider", "openai").set("type", model_type);
        let row = app.state.db.create_with_id(&format!("model:{key}"), doc).await.unwrap();
        row["id"].as_str().unwrap().to_string()
    }

    pub(crate) fn speakers() -> Value {
        json!([
            { "name": "Ana", "voice_id": "alloy", "backstory": "A host.", "personality": "Warm." },
            { "name": "Ben", "voice_id": "echo", "backstory": "A guest.", "personality": "Curious." }
        ])
    }

    /// Create a speaker profile through the API; returns its body.
    pub(crate) async fn create_speaker_profile(app: &TestApp, name: &str, voice_model: Option<&str>) -> Value {
        let (status, body) = app
            .post("/api/speaker-profiles", json!({ "name": name, "voice_model": voice_model, "speakers": speakers() }))
            .await;
        assert_eq!(status, StatusCode::OK, "{body}");
        body
    }

    #[tokio::test]
    async fn speaker_profile_crud() {
        let app = TestApp::new().await;
        let tts = model(&app, "tts1", "text_to_speech").await;
        let created = create_speaker_profile(&app, "zz_hosts", Some(&tts)).await;
        let id = created["id"].as_str().unwrap().to_string();
        assert!(id.starts_with("speaker_profile:"));
        assert_eq!(created["description"], "");
        assert_eq!(created["voice_model"], tts.as_str());
        assert_eq!(created["speakers"], speakers());

        let (status, body) = app.get("/api/speaker-profiles/zz_hosts").await;
        assert_eq!((status, body["id"].as_str()), (StatusCode::OK, Some(id.as_str())));
        let (status, list) = app.get("/api/speaker-profiles").await;
        assert_eq!(status, StatusCode::OK);
        assert!(list.as_array().unwrap().iter().any(|p| p["id"] == id.as_str()));

        // Update: a speaker override with its own voice model is kept as a link.
        let mut new_speakers = speakers();
        new_speakers[1]["voice_model"] = json!(tts);
        let (status, body) = app
            .put(
                &format!("/api/speaker-profiles/{id}"),
                json!({ "name": "zz_hosts", "description": "Two hosts", "speakers": new_speakers }),
            )
            .await;
        assert_eq!(status, StatusCode::OK, "{body}");
        assert_eq!(body["description"], "Two hosts");
        assert_eq!(body["voice_model"], tts.as_str(), "left out: kept");
        assert_eq!(body["speakers"][1]["voice_model"], tts.as_str());

        // Validation.
        for (speakers, message) in [
            (json!([]), "Must have between 1 and 4 speakers"),
            (json!([{ "name": "Ana", "voice_id": "a", "backstory": "b" }]), "Speaker missing required field: personality"),
        ] {
            let (status, body) =
                app.post("/api/speaker-profiles", json!({ "name": "bad", "speakers": speakers })).await;
            assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
            assert_eq!(body["detail"], message);
        }
        let (status, _) = app.post("/api/speaker-profiles", json!({ "name": "no speakers" })).await;
        assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY);

        let (status, copy) = app.post(&format!("/api/speaker-profiles/{id}/duplicate"), json!({})).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(copy["name"], "zz_hosts - Copy");
        assert_ne!(copy["id"], id.as_str());
        assert_eq!(copy["speakers"][1]["voice_model"], tts.as_str());

        let (status, body) = app.delete(&format!("/api/speaker-profiles/{id}")).await;
        assert_eq!((status, body), (StatusCode::OK, json!({ "message": "Speaker profile deleted successfully" })));
        let (status, body) = app.get("/api/speaker-profiles/zz_hosts").await;
        assert_eq!((status, body), (StatusCode::NOT_FOUND, json!({ "detail": "Speaker profile 'zz_hosts' not found" })));
        for path in [format!("/api/speaker-profiles/{id}"), format!("/api/speaker-profiles/{id}/duplicate")] {
            let (status, _) = app.post(&path, json!({})).await;
            assert!(status == StatusCode::NOT_FOUND || status == StatusCode::METHOD_NOT_ALLOWED, "{path}");
        }
        let (status, _) = app.delete(&format!("/api/speaker-profiles/{id}")).await;
        assert_eq!(status, StatusCode::NOT_FOUND);
        let (status, _) = app
            .put(&format!("/api/speaker-profiles/{id}"), json!({ "name": "x", "speakers": speakers() }))
            .await;
        assert_eq!(status, StatusCode::NOT_FOUND);
    }
}
