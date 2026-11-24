//! Podcast episodes: `/api/podcasts/...` (generation, script reading, job
//! status, episodes, audio, publishing, the Feed URL and token rotation).
//!
//! The Rust counterpart of `api/routers/podcasts.py`. The logic is in
//! `services::podcasts` and `services::podcast_feed`; these handlers extract
//! and shape. The token-gated feed routes are in `api::podcast_feed`.

use axum::body::Bytes;
use axum::extract::{Path, State};
use axum::http::HeaderMap;
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use serde::Deserialize;
use serde_json::{json, Value};

use crate::api::podcast_feed::{forwarded_origin, serve_file_or_error};
use crate::api::{validation_error, ApiJson, ApiQuery};
use crate::error::AppResult;
use crate::services::podcast_feed::{self, FeedUrl};
use crate::services::podcasts::{
    self, EpisodeDeleted, EpisodeView, GenerateRequest, PodcastJobStatus, PublishRequest, PublishedState,
    RetrySubmitted, ScriptRequest, SubmittedJob,
};
use crate::state::AppState;

pub fn router() -> Router<AppState> {
    Router::new()
        .route("/podcasts/generate", post(generate))
        .route("/podcasts/read-script", post(read_script))
        .route("/podcasts/jobs/{job_id}", get(job_status))
        .route("/podcasts/episodes", get(list_episodes))
        .route("/podcasts/episodes/{episode_id}", get(get_episode).delete(delete_episode))
        .route("/podcasts/episodes/{episode_id}/audio", get(episode_audio))
        .route("/podcasts/episodes/{episode_id}/publish", post(publish))
        .route("/podcasts/episodes/{episode_id}/unpublish", post(unpublish))
        .route("/podcasts/episodes/{episode_id}/retry", post(retry))
        .route("/podcasts/feed-url", get(feed_url))
        .route("/podcasts/feed-token/rotate", post(rotate_feed_token))
}

async fn generate(State(state): State<AppState>, ApiJson(body): ApiJson<GenerateRequest>) -> AppResult<Json<SubmittedJob>> {
    Ok(Json(podcasts::submit_generation(&state, &body).await?))
}

async fn read_script(State(state): State<AppState>, ApiJson(body): ApiJson<ScriptRequest>) -> AppResult<Json<SubmittedJob>> {
    Ok(Json(podcasts::submit_script(&state, &body).await?))
}

async fn job_status(State(state): State<AppState>, Path(job_id): Path<String>) -> AppResult<Json<PodcastJobStatus>> {
    Ok(Json(podcasts::job_status(&state, &job_id).await?))
}

#[derive(Debug, Deserialize)]
struct EpisodesQuery {
    notebook_id: Option<String>,
}

async fn list_episodes(
    State(state): State<AppState>,
    ApiQuery(query): ApiQuery<EpisodesQuery>,
) -> AppResult<Json<Vec<EpisodeView>>> {
    Ok(Json(podcasts::list_episodes(&state, query.notebook_id.as_deref()).await?))
}

async fn get_episode(State(state): State<AppState>, Path(episode_id): Path<String>) -> AppResult<Json<EpisodeView>> {
    Ok(Json(podcasts::get_episode(&state, &episode_id).await?))
}

async fn delete_episode(State(state): State<AppState>, Path(episode_id): Path<String>) -> AppResult<Json<EpisodeDeleted>> {
    Ok(Json(podcasts::delete_episode(&state, &episode_id).await?))
}

/// The episode's audio for the app's player, with byte ranges; HEAD too.
async fn episode_audio(
    State(state): State<AppState>,
    Path(episode_id): Path<String>,
    headers: HeaderMap,
) -> AppResult<Response> {
    let file = podcasts::episode_audio(&state, &episode_id).await?;
    let name = file.path.file_name().map(|n| n.to_string_lossy().into_owned());
    serve_file_or_error(&file, &headers, name.as_deref()).await
}

/// The body is optional (FastAPI `Optional[PublishEpisodeRequest] = None`):
/// none at all, `{}` and `{"summary": ...}` are all accepted.
async fn publish(State(state): State<AppState>, Path(episode_id): Path<String>, body: Bytes) -> Response {
    let request = if body.iter().all(u8::is_ascii_whitespace) {
        PublishRequest::default()
    } else {
        match serde_json::from_slice::<Option<PublishRequest>>(&body) {
            Ok(request) => request.unwrap_or_default(),
            Err(e) => return validation_error("body", "value_error", e.to_string()),
        }
    };
    match podcasts::publish_episode(&state, &episode_id, request.summary.as_deref()).await {
        Ok(published) => Json::<PublishedState>(published).into_response(),
        Err(e) => e.into_response(),
    }
}

async fn unpublish(State(state): State<AppState>, Path(episode_id): Path<String>) -> AppResult<Json<Value>> {
    podcasts::unpublish_episode(&state, &episode_id).await?;
    Ok(Json(json!({ "published_at": null })))
}

async fn retry(State(state): State<AppState>, Path(episode_id): Path<String>) -> AppResult<Json<RetrySubmitted>> {
    Ok(Json(podcasts::retry_episode(&state, &episode_id).await?))
}

async fn feed_url(State(state): State<AppState>, headers: HeaderMap) -> AppResult<Json<FeedUrl>> {
    Ok(Json(podcast_feed::feed_url(&state, &forwarded_origin(&state.config, &headers)).await?))
}

async fn rotate_feed_token(State(state): State<AppState>, headers: HeaderMap) -> AppResult<Json<FeedUrl>> {
    Ok(Json(podcast_feed::rotate_feed_url(&state, &forwarded_origin(&state.config, &headers)).await?))
}

#[cfg(test)]
mod tests {
    use axum::body::Body;
    use axum::http::{Method, Request, StatusCode};
    use nn_llm::AiError;
    use serde_json::{json, Value};

    use crate::ai::AnyModel;
    use crate::api::episode_profiles::tests::create_episode_profile;
    use crate::api::podcast_feed::tests::{episode_with_audio, forwarded, read, token};
    use crate::api::speaker_profiles::tests::{create_speaker_profile, model};
    use crate::testing::{silent_mp3_frames, MockLanguageModel, MockTextToSpeechModel, TestApp};

    const OUTLINE: &str = r#"{"segments": [
        {"name": "Opening", "description": "Why offline-first apps matter.", "size": "short"},
        {"name": "Middle", "description": "How sync works.", "size": "short"},
        {"name": "Close", "description": "What to remember.", "size": "short"}
    ]}"#;
    const TRANSCRIPT: &str =
        r#"{"transcript": [{"speaker": "Ana", "dialogue": "Welcome to the show."}, {"speaker": "Ben", "dialogue": "Glad to be here."}]}"#;

    /// Profiles named `zz_show` / `zz_speakers` with mock models behind them.
    async fn setup(app: &TestApp) -> std::sync::Arc<MockTextToSpeechModel> {
        let llm = model(app, "llm1", "language").await;
        let tts_id = model(app, "tts1", "text_to_speech").await;
        let speaker = create_speaker_profile(app, "zz_speakers", Some(&tts_id)).await;
        create_episode_profile(app, "zz_show", speaker["id"].as_str().unwrap(), &llm).await;
        app.state.models.set_override(AnyModel::Language(MockLanguageModel::new([OUTLINE, TRANSCRIPT])));
        let tts = MockTextToSpeechModel::new();
        app.state.models.set_override(AnyModel::TextToSpeech(tts.clone()));
        tts
    }

    async fn generate(app: &TestApp, name: &str) -> String {
        let (status, body) = app
            .post(
                "/api/podcasts/generate",
                json!({
                    "episode_profile": "zz_show",
                    "speaker_profile": "zz_speakers",
                    "episode_name": name,
                    "content": "Offline-first apps keep working without a network.",
                }),
            )
            .await;
        assert_eq!(status, StatusCode::OK, "{body}");
        assert_eq!(body["status"], "submitted");
        assert_eq!(body["episode_profile"], "zz_show");
        assert_eq!(body["message"], format!("Podcast generation started for episode '{name}'"));
        body["job_id"].as_str().unwrap().to_string()
    }

    async fn only_episode(app: &TestApp) -> Value {
        let (status, list) = app.get("/api/podcasts/episodes").await;
        assert_eq!(status, StatusCode::OK);
        let list = list.as_array().unwrap();
        assert_eq!(list.len(), 1, "{list:?}");
        list[0].clone()
    }

    #[tokio::test]
    async fn generated_episode_end_to_end() {
        let app = crate::api::podcast_feed::tests::proxied_app().await;
        let tts = setup(&app).await;
        let job_id = generate(&app, "Offline first").await;
        assert_eq!(app.wait_for_job(&job_id).await, "completed");

        let (status, job) = app.get(&format!("/api/podcasts/jobs/{job_id}")).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(job["status"], "completed");
        assert_eq!(job["result"]["success"], true);
        let episode_id = job["result"]["episode_id"].as_str().unwrap().to_string();
        assert!(episode_id.starts_with("episode:"));

        let episode = only_episode(&app).await;
        assert_eq!(episode["id"], episode_id.as_str());
        assert_eq!(episode["name"], "Offline first");
        assert_eq!(episode["job_status"], "completed");
        assert_eq!(episode["progress"], 100);
        assert_eq!(episode["episode_profile"]["name"], "zz_show");
        assert_eq!(episode["episode_profile"]["outline_model_name"], "llm1");
        assert_eq!(episode["speaker_profile"]["voice_model_provider"], "openai");
        assert_eq!(episode["transcript"]["transcript"][0]["speaker"], "Ana");
        assert_eq!(episode["outline"]["segments"].as_array().unwrap().len(), 3);
        assert_eq!(episode["audio_url"], format!("/api/podcasts/episodes/{episode_id}/audio"));
        assert!(episode["briefing"].as_str().unwrap().contains("Discuss the material."));
        assert_eq!(episode["published_at"], Value::Null);
        assert!(!tts.calls.lock().unwrap().is_empty());

        // The stored progress is 100 and the audio is under the podcasts folder.
        let row = app.state.db.select(&episode_id).await.unwrap().unwrap();
        assert_eq!(row["progress"], 100);
        let audio_file = row["audio_file"].as_str().unwrap().to_string();
        assert!(audio_file.starts_with("episodes/"), "{audio_file}");
        let audio_path = app.state.config.podcasts_dir().join(&audio_file);
        let audio = std::fs::read(&audio_path).unwrap();
        assert!(!audio.is_empty());

        let (status, one) = app.get(&format!("/api/podcasts/episodes/{episode_id}")).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(one, episode);

        // Download: whole, with a range, and HEAD.
        let url = format!("/api/podcasts/episodes/{episode_id}/audio");
        let (status, headers, body) = read(&app, forwarded(Method::GET, &url)).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(headers["content-type"], "audio/mpeg");
        assert!(headers["content-disposition"].to_str().unwrap().starts_with("attachment; filename=\""));
        assert_eq!(body, audio);
        let ranged = Request::builder().uri(&url).header("range", "bytes=10-19").body(Body::empty()).unwrap();
        let (status, headers, body) = read(&app, ranged).await;
        assert_eq!(status, StatusCode::PARTIAL_CONTENT);
        assert_eq!(headers["content-range"], format!("bytes 10-19/{}", audio.len()).as_str());
        assert_eq!(body, &audio[10..20]);
        let (status, headers, body) = read(&app, forwarded(Method::HEAD, &url)).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(headers["content-length"], audio.len().to_string().as_str());
        assert!(body.is_empty());

        // Publish (no body at all), republish, unpublish.
        let request = Request::builder().method(Method::POST).uri(format!("/api/podcasts/episodes/{episode_id}/publish")).body(Body::empty()).unwrap();
        let (status, _, body) = read(&app, request).await;
        assert_eq!(status, StatusCode::OK, "{}", String::from_utf8_lossy(&body));
        let published: Value = serde_json::from_slice(&body).unwrap();
        assert!(published["published_at"].is_string());
        assert!(published["duration_seconds"].is_number());
        assert!(published["feed_summary"].as_str().unwrap().contains("offline-first"), "{published}");
        let (status, again) =
            app.post(&format!("/api/podcasts/episodes/{episode_id}/publish"), json!({ "summary": "  Edited notes  " })).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(again["published_at"], published["published_at"]);
        assert_eq!(again["feed_summary"], "Edited notes");
        let (_, one) = app.get(&format!("/api/podcasts/episodes/{episode_id}")).await;
        assert_eq!(one["published_at"], published["published_at"]);

        let token = token(&app).await;
        let feed = format!("/api/podcasts/feed/{token}/feed.xml");
        let (_, _, xml) = read(&app, forwarded(Method::GET, &feed)).await;
        let xml = String::from_utf8(xml).unwrap();
        assert!(xml.contains("<title>Offline first</title>"), "{xml}");
        assert!(xml.contains("Edited notes"));

        let (status, body) = app.post(&format!("/api/podcasts/episodes/{episode_id}/unpublish"), json!({})).await;
        assert_eq!((status, body), (StatusCode::OK, json!({ "published_at": null })));
        let (status, body) = app.post(&format!("/api/podcasts/episodes/{episode_id}/unpublish"), json!({})).await;
        assert_eq!((status, body), (StatusCode::OK, json!({ "published_at": null })));
        let (_, one) = app.get(&format!("/api/podcasts/episodes/{episode_id}")).await;
        assert_eq!((one["published_at"].clone(), one["feed_summary"].clone()), (Value::Null, Value::Null));
        let (_, _, xml) = read(&app, forwarded(Method::GET, &feed)).await;
        assert!(!String::from_utf8(xml).unwrap().contains("<item>"));

        // Delete removes the record, the audio and the episode's folder.
        let (status, body) = app.delete(&format!("/api/podcasts/episodes/{episode_id}")).await;
        assert_eq!(
            (status, body),
            (StatusCode::OK, json!({ "message": "Episode deleted successfully", "episode_id": episode_id }))
        );
        assert!(!audio_path.exists());
        assert!(!audio_path.parent().unwrap().parent().unwrap().exists(), "episode folder removed");
        let (status, body) = app.get(&format!("/api/podcasts/episodes/{episode_id}")).await;
        assert_eq!((status, body), (StatusCode::NOT_FOUND, json!({ "detail": "Episode not found" })));
        let (status, _) = app.delete(&format!("/api/podcasts/episodes/{episode_id}")).await;
        assert_eq!(status, StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn generation_requests_are_checked_before_queueing() {
        let app = TestApp::new().await;
        setup(&app).await;
        let body = |episode_profile: &str, speaker: &str, content: Option<&str>, notebook: Option<&str>| {
            json!({
                "episode_profile": episode_profile,
                "speaker_profile": speaker,
                "episode_name": "x",
                "content": content,
                "notebook_id": notebook,
            })
        };
        let cases = [
            (body("nope", "zz_speakers", Some("c"), None), StatusCode::NOT_FOUND, "Episode profile 'nope' not found"),
            (body("zz_show", "nobody", Some("c"), None), StatusCode::NOT_FOUND, "Speaker profile 'nobody' not found"),
            (
                body("zz_show", "zz_speakers", None, None),
                StatusCode::BAD_REQUEST,
                "Content is required - provide either content or notebook_id",
            ),
            (body("zz_show", "zz_speakers", None, Some("notebook:missing")), StatusCode::NOT_FOUND, "Notebook not found"),
        ];
        for (request, status, detail) in cases {
            let (got, body) = app.post("/api/podcasts/generate", request).await;
            assert_eq!((got, body["detail"].as_str()), (status, Some(detail)));
        }
        let (status, _) = app.post("/api/podcasts/generate", json!({ "episode_profile": "zz_show" })).await;
        assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY);
        let (_, list) = app.get("/api/podcasts/episodes").await;
        assert_eq!(list, json!([]));

        let (status, job) = app.get("/api/podcasts/jobs/command:nothing").await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(job["status"], "unknown");
    }

    #[tokio::test]
    async fn script_episode_and_its_checks() {
        let app = TestApp::new().await;
        let tts = setup(&app).await;
        let (status, body) = app
            .post(
                "/api/podcasts/read-script",
                json!({ "speaker_profile": "zz_speakers", "episode_name": "Read aloud", "script": "Ana: Hello there.\n\nBen: Hi, Ana." }),
            )
            .await;
        assert_eq!(status, StatusCode::OK, "{body}");
        assert_eq!(body["episode_profile"], "script");
        assert_eq!(body["message"], "Reading script for episode 'Read aloud'");
        assert_eq!(app.wait_for_job(body["job_id"].as_str().unwrap()).await, "completed");

        let episode = only_episode(&app).await;
        assert_eq!(episode["episode_profile"]["mode"], "script");
        assert_eq!(episode["progress"], 100);
        assert_eq!(episode["outline"], Value::Null);
        let turns = episode["transcript"]["transcript"].as_array().unwrap();
        assert_eq!(turns.len(), 2);
        assert_eq!((turns[1]["speaker"].as_str(), turns[1]["dialogue"].as_str()), (Some("Ben"), Some("Hi, Ana.")));
        assert!(episode["audio_url"].is_string());
        let voices: Vec<String> = tts.calls.lock().unwrap().iter().map(|(_, voice)| voice.clone()).collect();
        assert_eq!(voices, ["alloy", "echo"]);

        let script = |speaker: &str, script: &str| {
            json!({ "speaker_profile": speaker, "episode_name": "x", "script": script })
        };
        let (status, _) = app.post("/api/podcasts/read-script", script("zz_speakers", "   ")).await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        let (status, body) = app.post("/api/podcasts/read-script", script("nobody", "Hi.")).await;
        assert_eq!((status, body["detail"].as_str()), (StatusCode::NOT_FOUND, Some("Speaker profile 'nobody' not found")));
        create_speaker_profile(&app, "zz_voiceless", None).await;
        let (status, body) = app.post("/api/podcasts/read-script", script("zz_voiceless", "Hi.")).await;
        assert_eq!(
            (status, body["detail"].as_str()),
            (StatusCode::BAD_REQUEST, Some("Speaker profile 'zz_voiceless' has no voice model"))
        );
        let (_, list) = app.get("/api/podcasts/episodes").await;
        assert_eq!(list.as_array().unwrap().len(), 1, "nothing else was queued");
    }

    #[tokio::test]
    async fn a_failed_episode_can_be_retried() {
        let app = TestApp::new().await;
        setup(&app).await;
        app.state.models.set_override(AnyModel::Language(MockLanguageModel::failing(AiError::Authentication(
            "Invalid API key".into(),
        ))));
        let job_id = generate(&app, "Doomed").await;
        assert_eq!(app.wait_for_job(&job_id).await, "failed");
        let failed = only_episode(&app).await;
        assert_eq!(failed["job_status"], "failed");
        assert!(failed["error_message"].as_str().unwrap().contains("Invalid API key"), "{failed}");
        assert_eq!(failed["audio_url"], Value::Null);
        let failed_id = failed["id"].as_str().unwrap().to_string();

        // Publishing a failed episode is refused.
        let (status, body) = app.post(&format!("/api/podcasts/episodes/{failed_id}/publish"), json!({})).await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert!(body["detail"].as_str().unwrap().contains("not completed"), "{body}");

        app.state.models.set_override(AnyModel::Language(MockLanguageModel::new([OUTLINE, TRANSCRIPT])));
        let (status, body) = app.post(&format!("/api/podcasts/episodes/{failed_id}/retry"), json!({})).await;
        assert_eq!(status, StatusCode::OK, "{body}");
        assert_eq!(body["message"], "Retry submitted successfully");
        let retry_job = body["job_id"].as_str().unwrap().to_string();
        assert_ne!(retry_job, job_id);
        assert_eq!(app.wait_for_job(&retry_job).await, "completed");

        let (status, _) = app.get(&format!("/api/podcasts/episodes/{failed_id}")).await;
        assert_eq!(status, StatusCode::NOT_FOUND, "the failed episode is replaced");
        let episode = only_episode(&app).await;
        assert_eq!((episode["name"].as_str(), episode["job_status"].as_str()), (Some("Doomed"), Some("completed")));

        // Only failed episodes can be retried.
        let (status, body) = app.post(&format!("/api/podcasts/episodes/{}/retry", episode["id"].as_str().unwrap()), json!({})).await;
        assert_eq!(
            (status, body["detail"].as_str()),
            (StatusCode::BAD_REQUEST, Some("Episode is not in a failed state (current: completed)"))
        );
        let (status, _) = app.post("/api/podcasts/episodes/episode:missing/retry", json!({})).await;
        assert_eq!(status, StatusCode::NOT_FOUND);
    }

    async fn command_count(app: &TestApp) -> usize {
        app.state.db.query("SELECT id FROM command", crate::db::Params::new()).await.unwrap().len()
    }

    /// Issue #34: podcast content is stored in the job arguments and on the
    /// episode, so it is capped well below the request body limit.
    #[tokio::test]
    async fn generation_refuses_overlong_content() {
        let app = TestApp::new().await;
        setup(&app).await;
        let content = "a".repeat(crate::services::podcasts::MAX_PODCAST_CONTENT_CHARS + 1);
        let (status, body) = app
            .post(
                "/api/podcasts/generate",
                json!({ "episode_profile": "zz_show", "speaker_profile": "zz_speakers", "episode_name": "x", "content": content }),
            )
            .await;
        assert_eq!(
            (status, body["detail"].as_str()),
            (StatusCode::BAD_REQUEST, Some("The podcast content is too long (2,000,000 characters at most)."))
        );
        assert_eq!(command_count(&app).await, 0, "nothing was stored");
    }

    /// Issue #34: a full job queue answers 429 instead of growing without
    /// bound, and a retry refused that way keeps the failed episode.
    #[tokio::test]
    async fn a_full_job_queue_is_a_429_and_a_refused_retry_keeps_the_episode() {
        let app = TestApp::new().await;
        setup(&app).await;
        app.state.models.set_override(AnyModel::Language(MockLanguageModel::failing(AiError::Authentication(
            "Invalid API key".into(),
        ))));
        let job_id = generate(&app, "Doomed").await;
        assert_eq!(app.wait_for_job(&job_id).await, "failed");
        let failed_id = only_episode(&app).await["id"].as_str().unwrap().to_string();
        let commands = command_count(&app).await;

        // One slow job fills a queue of one.
        app.state.jobs.register("zz_hold", crate::jobs::RetryPolicy::once(), |_ctx, _args| async move {
            tokio::time::sleep(std::time::Duration::from_secs(60)).await;
            Ok(Value::Null)
        });
        app.state.jobs.set_max_pending(1);
        let hold = app.state.jobs.submit("zz_hold", Value::Null).await.unwrap();

        let (status, body) = app
            .post(
                "/api/podcasts/generate",
                json!({ "episode_profile": "zz_show", "speaker_profile": "zz_speakers", "episode_name": "y", "content": "c" }),
            )
            .await;
        assert_eq!(status, StatusCode::TOO_MANY_REQUESTS, "{body}");
        assert!(body["detail"].as_str().unwrap().contains("Too many background jobs are queued"), "{body}");

        let (status, _) = app.post(&format!("/api/podcasts/episodes/{failed_id}/retry"), json!({})).await;
        assert_eq!(status, StatusCode::TOO_MANY_REQUESTS);
        let (status, _) = app.get(&format!("/api/podcasts/episodes/{failed_id}")).await;
        assert_eq!(status, StatusCode::OK, "the failed episode is still there");
        assert_eq!(command_count(&app).await, commands + 1, "refused submissions store no command row");

        // Canceling the slow job frees its place.
        assert!(app.state.jobs.cancel(&hold).await.unwrap());
        for _ in 0..200 {
            if app.state.jobs.pending() == 0 {
                break;
            }
            tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        }
        app.state.models.set_override(AnyModel::Language(MockLanguageModel::new([OUTLINE, TRANSCRIPT])));
        let (status, body) = app.post(&format!("/api/podcasts/episodes/{failed_id}/retry"), json!({})).await;
        assert_eq!(status, StatusCode::OK, "{body}");
        assert_eq!(app.wait_for_job(body["job_id"].as_str().unwrap()).await, "completed");
    }

    #[tokio::test]
    async fn audio_route_refusals_and_publish_rules() {
        let app = TestApp::new().await;
        episode_with_audio(&app, "silent", "episodes/silent/audio/silent.mp3", &silent_mp3_frames(10), None).await;
        episode_with_audio(&app, "escape", "../outside.mp3", b"", None).await;
        episode_with_audio(&app, "vanished", "episodes/vanished/audio/vanished.mp3", b"", None).await;
        episode_with_audio(&app, "wave", "episodes/wave/audio/wave.wav", b"RIFF....WAVEfmt ", None).await;
        app.state
            .db
            .create_with_id(
                "episode:noaudio",
                crate::db::Doc::new()
                    .set("name", "No audio")
                    .set("episode_profile", json!({}))
                    .set("speaker_profile", json!({}))
                    .set("briefing", "")
                    .set("content", ""),
            )
            .await
            .unwrap();

        for (path, status, detail) in [
            ("episode:missing", StatusCode::NOT_FOUND, "Episode not found"),
            ("notebook:silent", StatusCode::NOT_FOUND, "Episode not found"),
            ("episode:noaudio", StatusCode::NOT_FOUND, "Episode has no audio file"),
            ("episode:escape", StatusCode::FORBIDDEN, "Access to file denied"),
            ("episode:vanished", StatusCode::NOT_FOUND, "Audio file not found on disk"),
        ] {
            let (got, body) = app.get(&format!("/api/podcasts/episodes/{path}/audio")).await;
            assert_eq!((got, body), (status, json!({ "detail": detail })), "{path}");
        }
        let (status, headers, _) = read(&app, forwarded(Method::GET, "/api/podcasts/episodes/episode:wave/audio")).await;
        assert_eq!((status, headers["content-type"].to_str().unwrap()), (StatusCode::OK, "audio/wav"));

        // An imported episode (audio, no job) is completed and can be published.
        let (_, episode) = app.get("/api/podcasts/episodes/episode:silent").await;
        assert_eq!(episode["job_status"], "completed");
        let (status, body) = app.post("/api/podcasts/episodes/episode:silent/publish", json!({})).await;
        assert_eq!(status, StatusCode::OK, "{body}");
        assert_eq!(body["feed_summary"], "Episode silent", "no outline: the name");
        let (status, body) =
            app.post("/api/podcasts/episodes/episode:silent/publish", json!({ "summary": "x".repeat(4001) })).await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert!(body["detail"].as_str().unwrap().to_lowercase().contains("summary"));
        let (status, _) = app.post("/api/podcasts/episodes/episode:silent/publish", json!({ "summary": 5 })).await;
        assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY);

        // A WAV episode cannot be published, and the message says what to do.
        let (status, body) = app.post("/api/podcasts/episodes/episode:wave/publish", json!({})).await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        let detail = body["detail"].as_str().unwrap();
        assert!(detail.contains("'.wav'") && detail.contains("ffmpeg"), "{detail}");
        for (key, keyword) in [("noaudio", "no audio"), ("vanished", "not found"), ("escape", "outside")] {
            let (status, body) = app.post(&format!("/api/podcasts/episodes/episode:{key}/publish"), json!({})).await;
            assert_eq!(status, StatusCode::BAD_REQUEST, "{key}");
            assert!(body["detail"].as_str().unwrap().to_lowercase().contains(keyword), "{key}: {body}");
        }
        let (status, _) = app.post("/api/podcasts/episodes/episode:missing/publish", json!({})).await;
        assert_eq!(status, StatusCode::NOT_FOUND);
        let (status, _) = app.post("/api/podcasts/episodes/episode:missing/unpublish", json!({})).await;
        assert_eq!(status, StatusCode::NOT_FOUND);

        // Deleting an episode whose path escapes the folder never touches the outside file.
        let outside = app.state.config.podcasts_dir().join("../outside.mp3");
        std::fs::write(&outside, b"keep me").unwrap();
        let (status, _) = app.delete("/api/podcasts/episodes/episode:escape").await;
        assert_eq!(status, StatusCode::OK);
        assert!(outside.exists());
    }

    #[tokio::test]
    async fn feed_url_and_token_rotation() {
        let app = crate::api::podcast_feed::tests::proxied_app().await;
        let (status, _, body) = read(&app, forwarded(Method::GET, "/api/podcasts/feed-url")).await;
        assert_eq!(status, StatusCode::OK);
        let url = serde_json::from_slice::<Value>(&body).unwrap()["feed_url"].as_str().unwrap().to_string();
        let old_token = token(&app).await;
        assert!(url.ends_with(&format!("/api/podcasts/feed/{old_token}/feed.xml")), "{url}");
        let path = url.split_once(".net:8502").map(|(_, p)| p.to_string()).unwrap_or(url.clone());

        let (status, _, body) = read(&app, forwarded(Method::POST, "/api/podcasts/feed-token/rotate")).await;
        assert_eq!(status, StatusCode::OK);
        let new_url = serde_json::from_slice::<Value>(&body).unwrap()["feed_url"].as_str().unwrap().to_string();
        assert_ne!(new_url, url);
        let new_token = token(&app).await;
        assert_ne!(new_token, old_token);
        let (status, _, _) = read(&app, forwarded(Method::GET, &path)).await;
        assert_eq!(status, StatusCode::NOT_FOUND, "the old URL stops working");
        let (status, _, _) = read(&app, forwarded(Method::GET, &format!("/api/podcasts/feed/{new_token}/feed.xml"))).await;
        assert_eq!(status, StatusCode::OK);

        // Without an HTTPS origin: 422, and nothing is rotated.
        if std::env::var("OPEN_NOTEBOOK_PUBLIC_URL").is_err() && std::env::var("API_URL").is_err() {
            let (status, body) = app.get("/api/podcasts/feed-url").await;
            assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY);
            assert!(body["detail"].as_str().unwrap().contains("OPEN_NOTEBOOK_PUBLIC_URL"));
            let (status, _) = app.post("/api/podcasts/feed-token/rotate", json!({})).await;
            assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY);
            assert_eq!(token(&app).await, new_token);
        }
    }
}
