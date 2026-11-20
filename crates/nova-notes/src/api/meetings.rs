//! Meetings: `/api/notebooks/{notebook_id}/meetings/...` (Python
//! `api/routers/meetings.py`). The logic lives in [`crate::services::meetings`].

use axum::extract::{Multipart, Path, State};
use axum::http::StatusCode;
use axum::response::IntoResponse;
use axum::routing::{get, patch, post};
use axum::{Json, Router};
use serde_json::json;

use super::notebooks::ApiJson;
use super::sources::{multipart_problem, Rejection};
use crate::domain::meeting::FieldError;
use crate::services::meetings::{
    self, ApproveResult, ChunkResult, DiscardResult, EditMeeting, FinishMeeting, MeetingView, StartMeeting,
};
use crate::state::AppState;

pub fn router() -> Router<AppState> {
    Router::new()
        .route("/notebooks/{notebook_id}/meetings", get(list_meetings).post(start_meeting))
        .route("/notebooks/{notebook_id}/meetings/{meeting_id}/chunks/{sequence}", post(upload_chunk))
        .route("/notebooks/{notebook_id}/meetings/{meeting_id}/finish", post(finish_meeting))
        .route("/notebooks/{notebook_id}/meetings/{meeting_id}/summarize", post(retry_summary))
        .route("/notebooks/{notebook_id}/meetings/{meeting_id}", patch(edit_meeting).delete(discard))
        .route("/notebooks/{notebook_id}/meetings/{meeting_id}/approve", post(approve))
}

/// FastAPI's 422 for a body field that breaks a constraint.
fn field_problem(location: &str, e: FieldError) -> Rejection {
    let detail = json!([{ "type": e.kind, "loc": [location, e.field], "msg": e.message }]);
    (StatusCode::UNPROCESSABLE_ENTITY, Json(json!({ "detail": detail }))).into_response().into()
}

async fn list_meetings(
    State(state): State<AppState>,
    Path(notebook_id): Path<String>,
) -> Result<Json<Vec<MeetingView>>, Rejection> {
    Ok(Json(meetings::list_meetings(&state, &notebook_id).await?))
}

async fn start_meeting(
    State(state): State<AppState>,
    Path(notebook_id): Path<String>,
    ApiJson(body): ApiJson<StartMeeting>,
) -> Result<Json<MeetingView>, Rejection> {
    body.check().map_err(|e| field_problem("body", e))?;
    Ok(Json(meetings::start_meeting(&state, &notebook_id, &body).await?))
}

/// `POST .../chunks/{sequence}`: multipart with the WAV chunk as `audio`.
async fn upload_chunk(
    State(state): State<AppState>,
    Path((notebook_id, meeting_id, sequence)): Path<(String, String, String)>,
    mut multipart: Multipart,
) -> Result<Json<ChunkResult>, Rejection> {
    let Ok(sequence) = sequence.trim().parse::<i64>() else {
        return Err(field_problem(
            "path",
            FieldError::new(
                "sequence",
                "int_parsing",
                "Input should be a valid integer, unable to parse string as an integer",
            ),
        ));
    };
    let mut audio = None;
    while let Some(field) = multipart.next_field().await.map_err(multipart_problem)? {
        if field.name() == Some("audio") {
            audio = Some(field.bytes().await.map_err(multipart_problem)?);
        }
    }
    let Some(audio) = audio else {
        return Err(field_problem("body", FieldError::new("audio", "missing", "Field required")));
    };
    Ok(Json(meetings::upload_chunk(&state, &notebook_id, &meeting_id, sequence, &audio).await?))
}

async fn finish_meeting(
    State(state): State<AppState>,
    Path((notebook_id, meeting_id)): Path<(String, String)>,
    ApiJson(body): ApiJson<FinishMeeting>,
) -> Result<Json<MeetingView>, Rejection> {
    body.check().map_err(|e| field_problem("body", e))?;
    Ok(Json(meetings::finish_meeting(&state, &notebook_id, &meeting_id, &body).await?))
}

async fn retry_summary(
    State(state): State<AppState>,
    Path((notebook_id, meeting_id)): Path<(String, String)>,
) -> Result<Json<MeetingView>, Rejection> {
    Ok(Json(meetings::retry_summary(&state, &notebook_id, &meeting_id).await?))
}

async fn edit_meeting(
    State(state): State<AppState>,
    Path((notebook_id, meeting_id)): Path<(String, String)>,
    ApiJson(body): ApiJson<EditMeeting>,
) -> Result<Json<MeetingView>, Rejection> {
    body.check().map_err(|e| field_problem("body", e))?;
    Ok(Json(meetings::edit_meeting(&state, &notebook_id, &meeting_id, &body).await?))
}

async fn approve(
    State(state): State<AppState>,
    Path((notebook_id, meeting_id)): Path<(String, String)>,
) -> Result<Json<ApproveResult>, Rejection> {
    Ok(Json(meetings::approve(&state, &notebook_id, &meeting_id).await?))
}

async fn discard(
    State(state): State<AppState>,
    Path((notebook_id, meeting_id)): Path<(String, String)>,
) -> Result<Json<DiscardResult>, Rejection> {
    Ok(Json(meetings::discard_meeting(&state, &notebook_id, &meeting_id).await?))
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use axum::body::Body;
    use axum::http::{header, Method, Request, StatusCode};
    use http_body_util::BodyExt;
    use nn_llm::AiError;
    use serde_json::{json, Value};

    use crate::ai::AnyModel;
    use crate::db::Params;
    use crate::domain::meeting::Meeting;
    use crate::services::meetings::{parse_summary, summarize_meeting, validate_wav, MAX_AUDIO_BYTES};
    use crate::testing::{multipart_body, MockEmbeddingModel, MockLanguageModel, MockSpeechToTextModel, TestApp};

    const QUOTE: &str = "Priya will send the budget to finance by Friday.";

    /// Mono 16-bit PCM at 16 kHz with `frames` silent samples.
    fn wav(frames: u32) -> Vec<u8> {
        let data_len = frames * 2;
        let mut out = Vec::new();
        out.extend_from_slice(b"RIFF");
        out.extend_from_slice(&(36 + data_len).to_le_bytes());
        out.extend_from_slice(b"WAVEfmt ");
        out.extend_from_slice(&16u32.to_le_bytes());
        out.extend_from_slice(&1u16.to_le_bytes()); // PCM
        out.extend_from_slice(&1u16.to_le_bytes()); // mono
        out.extend_from_slice(&16_000u32.to_le_bytes());
        out.extend_from_slice(&32_000u32.to_le_bytes());
        out.extend_from_slice(&2u16.to_le_bytes());
        out.extend_from_slice(&16u16.to_le_bytes());
        out.extend_from_slice(b"data");
        out.extend_from_slice(&data_len.to_le_bytes());
        out.extend(std::iter::repeat_n(0u8, data_len as usize));
        out
    }

    fn summary_reply() -> String {
        json!({
            "summary": "Budget follow-up agreed.",
            "actions": [{ "text": "Send budget", "owner": "Priya", "due": "Friday", "evidence": QUOTE, "done": true }],
        })
        .to_string()
    }

    async fn notebook(app: &TestApp) -> String {
        let (status, body) = app.post("/api/notebooks", json!({ "name": "Meetings" })).await;
        assert_eq!(status, StatusCode::OK, "{body}");
        body["id"].as_str().unwrap().to_string()
    }

    fn with_stt(app: &TestApp, transcript: &str) -> Arc<MockSpeechToTextModel> {
        let stt = MockSpeechToTextModel::new(transcript);
        app.state.models.set_override(AnyModel::SpeechToText(stt.clone()));
        app.state.models.set_override(AnyModel::Embedding(MockEmbeddingModel::new()));
        stt
    }

    fn with_chat(app: &TestApp, replies: &[&str]) -> Arc<MockLanguageModel> {
        let language = MockLanguageModel::new(replies.iter().copied());
        app.state.models.set_override(AnyModel::Language(language.clone()));
        language
    }

    async fn start(app: &TestApp, notebook_id: &str) -> Value {
        let (status, body) =
            app.post(&format!("/api/notebooks/{notebook_id}/meetings"), json!({ "title": "Sync" })).await;
        assert_eq!(status, StatusCode::OK, "{body}");
        body
    }

    async fn upload(
        app: &TestApp,
        notebook_id: &str,
        meeting_id: &str,
        sequence: &str,
        audio: &[u8],
    ) -> (StatusCode, Value) {
        let (content_type, body) = multipart_body(&[("audio", Some("chunk.wav"), audio)]);
        let request = Request::builder()
            .method(Method::POST)
            .uri(format!("/api/notebooks/{notebook_id}/meetings/{meeting_id}/chunks/{sequence}"))
            .header(header::CONTENT_TYPE, content_type)
            .body(Body::from(body))
            .unwrap();
        let response = app.send(request).await;
        let status = response.status();
        let bytes = response.into_body().collect().await.unwrap().to_bytes();
        (status, serde_json::from_slice(&bytes).unwrap_or(Value::Null))
    }

    async fn finish(app: &TestApp, notebook_id: &str, meeting_id: &str, total: i64) -> Value {
        let path = format!("/api/notebooks/{notebook_id}/meetings/{meeting_id}/finish");
        let (status, body) = app.post(&path, json!({ "total_chunks": total })).await;
        assert_eq!(status, StatusCode::OK, "{body}");
        body
    }

    async fn wait_summary(app: &TestApp, meeting_id: &str) -> Meeting {
        let meeting = Meeting::get(&app.state, meeting_id).await.unwrap();
        assert!(!meeting.summary_job.is_empty(), "summary was queued");
        assert_eq!(app.wait_for_job(&meeting.summary_job).await, "completed");
        Meeting::get(&app.state, meeting_id).await.unwrap()
    }

    async fn count(app: &TestApp, sql: &str) -> usize {
        app.state.db.query(sql, Params::new()).await.unwrap().len()
    }

    /// A meeting in review with one transcribed chunk and its summary done.
    async fn reviewed(app: &TestApp, notebook_id: &str) -> Meeting {
        with_stt(app, QUOTE);
        with_chat(app, &[&summary_reply()]);
        let meeting = start(app, notebook_id).await;
        let id = meeting["id"].as_str().unwrap();
        assert_eq!(upload(app, notebook_id, id, "0", &wav(1600)).await.0, StatusCode::OK);
        finish(app, notebook_id, id, 1).await;
        wait_summary(app, id).await
    }

    #[tokio::test]
    async fn start_needs_a_speech_to_text_model() {
        let app = TestApp::new().await;
        let nb = notebook(&app).await;
        let (status, body) = app.post(&format!("/api/notebooks/{nb}/meetings"), json!({ "title": "Sync" })).await;
        assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY);
        assert_eq!(body["detail"], "Configure a default speech-to-text model in Settings");
        assert_eq!(count(&app, "SELECT * FROM meeting").await, 0);
    }

    #[tokio::test]
    async fn start_validates_its_input() {
        let app = TestApp::new().await;
        let nb = notebook(&app).await;
        with_stt(&app, "words");
        let path = format!("/api/notebooks/{nb}/meetings");
        let (status, body) = app.post(&path, json!({ "title": "" })).await;
        assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY);
        assert_eq!(body["detail"][0]["loc"], json!(["body", "title"]));
        let (status, _) = app.post(&path, json!({})).await;
        assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY);
        let (status, body) = app.post(&path, json!({ "title": "Sync", "capture_mode": "camera" })).await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert_eq!(body["detail"], "Unsupported audio source");
        let (status, _) = app.post("/api/notebooks/notebook:missing/meetings", json!({ "title": "Sync" })).await;
        assert_eq!(status, StatusCode::NOT_FOUND);

        let (status, body) = app.post(&path, json!({ "title": "Sync", "capture_mode": "shared" })).await;
        assert_eq!(status, StatusCode::OK, "{body}");
        assert!(body["id"].as_str().unwrap().starts_with("meeting:"));
        assert_eq!(body["status"], "recording");
        assert_eq!(body["capture_mode"], "shared");
        assert_eq!(body["notebook_id"], nb);
        assert_eq!(body["summary_status"], "idle");
        assert_eq!(body["revision"], 0);
        assert_eq!(body["actions"], json!([]));
        assert_eq!(body["missing_chunks"], json!([]));
    }

    #[tokio::test]
    async fn lifecycle_records_gaps_summarises_and_publishes_once() {
        let app = TestApp::new().await;
        let nb = notebook(&app).await;
        let stt = with_stt(&app, &format!("  {QUOTE} "));
        let chat = with_chat(&app, &[&summary_reply()]);
        let meeting = start(&app, &nb).await;
        let id = meeting["id"].as_str().unwrap().to_string();

        // Repeat deliveries are idempotent and do not reach the provider again.
        for sequence in ["0", "0", "2"] {
            let (status, body) = upload(&app, &nb, &id, sequence, &wav(1600)).await;
            assert_eq!(status, StatusCode::OK, "{body}");
            assert_eq!(body["text"], QUOTE);
        }
        assert_eq!(stt.calls.lock().unwrap().len(), 2);
        assert_eq!(count(&app, "SELECT * FROM meeting_chunk").await, 2);

        // While recording, the view shows the chunks received so far.
        let (status, list) = app.get(&format!("/api/notebooks/{nb}/meetings")).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(list[0]["total_chunks"], 3);
        assert_eq!(list[0]["transcript"], format!("[00:00] {QUOTE}\n\n[00:40] {QUOTE}"));

        let finished = finish(&app, &nb, &id, 3).await;
        assert_eq!(finished["status"], "review");
        assert_eq!(finished["missing_chunks"], json!([1]));
        assert!(finished["transcript"].as_str().unwrap().contains("[00:20] [Missing audio section]"));
        // Drafts stay out of notes until approval.
        assert_eq!(count(&app, "SELECT * FROM note").await, 0);

        let done = wait_summary(&app, &id).await;
        assert_eq!(chat.call_count(), 1);
        assert_eq!(done.summary, "Budget follow-up agreed.");
        assert_eq!(done.actions[0].owner, "Priya");
        assert!(!done.actions[0].done, "generated actions start unchecked");
        assert_eq!(done.revision, 1);
        let (_, list) = app.get(&format!("/api/notebooks/{nb}/meetings")).await;
        assert_eq!(list[0]["summary_status"], "completed");
        assert_eq!(list[0]["summary"], "Budget follow-up agreed.");
        assert_eq!(list[0]["summary_error"], "");

        // Finishing again changes nothing.
        let again = finish(&app, &nb, &id, 5).await;
        assert_eq!(again["total_chunks"], 3);

        // Uploads after finishing are rejected.
        let (status, body) = upload(&app, &nb, &id, "3", &wav(1600)).await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert_eq!(body["detail"], "Meeting is finished or exceeds four hours");

        let edit = json!({
            "revision": 1, "title": "Reviewed", "summary": "Plan agreed",
            "actions": [], "transcript": finished["transcript"],
        });
        let (status, edited) = app.patch(&format!("/api/notebooks/{nb}/meetings/{id}"), edit).await;
        assert_eq!(status, StatusCode::OK, "{edited}");
        assert_eq!(edited["revision"], 2);
        assert_eq!(edited["summary_token"], "");
        assert_eq!(edited["summary_status"], "idle");

        let approve_path = format!("/api/notebooks/{nb}/meetings/{id}/approve");
        let (status, approved) = app.post(&approve_path, json!({})).await;
        assert_eq!(status, StatusCode::OK, "{approved}");
        let note_id = approved["note_id"].as_str().unwrap().to_string();
        assert_eq!(note_id, format!("note:meeting_{}", id.split_once(':').unwrap().1));
        let (_, again) = app.post(&approve_path, json!({})).await;
        assert_eq!(again["note_id"], note_id);

        let (status, notes) = app.get(&format!("/api/notes?notebook_id={nb}")).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(notes.as_array().unwrap().len(), 1);
        let (_, note) = app.get(&format!("/api/notes/{note_id}")).await;
        assert_eq!(note["title"], "Reviewed");
        assert_eq!(note["note_type"], "human");
        assert!(note["content"]
            .as_str()
            .unwrap()
            .starts_with("## Summary\nPlan agreed\n\n## Actions\n\n\n## Transcript\n"));

        // Editing an approved meeting rewrites its note.
        let edit = json!({
            "revision": 2, "title": "Reviewed", "summary": "Updated after approval",
            "actions": [], "transcript": finished["transcript"],
        });
        let (status, _) = app.patch(&format!("/api/notebooks/{nb}/meetings/{id}"), edit).await;
        assert_eq!(status, StatusCode::OK);
        let (_, note) = app.get(&format!("/api/notes/{note_id}")).await;
        assert!(note["content"].as_str().unwrap().contains("Updated after approval"));

        // Approved meetings are deleted through their note.
        let (status, body) = app.delete(&format!("/api/notebooks/{nb}/meetings/{id}")).await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert_eq!(body["detail"], "Delete the approved note through Notes instead");
        let (status, body) = app.delete(&format!("/api/notes/{note_id}")).await;
        assert_eq!(status, StatusCode::OK, "{body}");
        assert_eq!(count(&app, "SELECT * FROM meeting").await, 0);
        assert_eq!(count(&app, "SELECT * FROM meeting_chunk").await, 0);
    }

    #[tokio::test]
    async fn invalid_audio_never_reaches_the_provider() {
        let app = TestApp::new().await;
        let nb = notebook(&app).await;
        let stt = with_stt(&app, "words");
        let id = start(&app, &nb).await["id"].as_str().unwrap().to_string();
        let mut truncated = wav(1600);
        truncated.truncate(truncated.len() - 2);
        for audio in [Vec::new(), b"not WAV".to_vec(), wav(16_000 * 26), truncated] {
            let (status, body) = upload(&app, &nb, &id, "0", &audio).await;
            assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
        }
        assert!(stt.calls.lock().unwrap().is_empty());
        assert!(validate_wav(&vec![b'x'; MAX_AUDIO_BYTES + 1]).is_err());
        assert!(validate_wav(&wav(16_000 * 25)).is_ok());

        for sequence in ["-1", "720"] {
            let (status, _) = upload(&app, &nb, &id, sequence, &wav(1600)).await;
            assert_eq!(status, StatusCode::BAD_REQUEST);
        }
        let (status, _) = upload(&app, &nb, &id, "first", &wav(1600)).await;
        assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY);
        let (content_type, body) = multipart_body(&[("other", None, b"x")]);
        let request = Request::builder()
            .method(Method::POST)
            .uri(format!("/api/notebooks/{nb}/meetings/{id}/chunks/0"))
            .header(header::CONTENT_TYPE, content_type)
            .body(Body::from(body))
            .unwrap();
        assert_eq!(app.send(request).await.status(), StatusCode::UNPROCESSABLE_ENTITY);
    }

    #[tokio::test]
    async fn chunk_upload_without_a_model_is_a_configuration_error() {
        let app = TestApp::new().await;
        let nb = notebook(&app).await;
        with_stt(&app, "words");
        let id = start(&app, &nb).await["id"].as_str().unwrap().to_string();
        app.state.models.clear_overrides();
        let (status, body) = upload(&app, &nb, &id, "0", &wav(1600)).await;
        assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY);
        assert_eq!(body["detail"], "Configure a default speech-to-text model in Settings");
        assert_eq!(count(&app, "SELECT * FROM meeting_chunk").await, 0);
    }

    #[tokio::test]
    async fn meetings_are_scoped_to_their_notebook() {
        let app = TestApp::new().await;
        let nb = notebook(&app).await;
        let other = notebook(&app).await;
        with_stt(&app, "words");
        let id = start(&app, &nb).await["id"].as_str().unwrap().to_string();
        let (status, body) =
            app.post(&format!("/api/notebooks/{other}/meetings/{id}/finish"), json!({ "total_chunks": 0 })).await;
        assert_eq!(status, StatusCode::NOT_FOUND);
        assert_eq!(body["detail"], "Meeting not found in this notebook");
        let (status, body) = app.delete(&format!("/api/notebooks/{nb}/meetings/note:abc")).await;
        assert_eq!(status, StatusCode::NOT_FOUND);
        assert_eq!(body["detail"], "Meeting not found");
        let (status, _) = app.post(&format!("/api/notebooks/{nb}/meetings/meeting:missing/approve"), json!({})).await;
        assert_eq!(status, StatusCode::NOT_FOUND);
        let (_, list) = app.get(&format!("/api/notebooks/{other}/meetings")).await;
        assert_eq!(list, json!([]));
        let (status, _) = app.get("/api/notebooks/notebook:missing/meetings").await;
        assert_eq!(status, StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn recording_drafts_cannot_be_edited_or_published() {
        let app = TestApp::new().await;
        let nb = notebook(&app).await;
        with_stt(&app, "words");
        let id = start(&app, &nb).await["id"].as_str().unwrap().to_string();
        let edit = json!({ "title": "T", "summary": "", "actions": [], "transcript": "x" });
        let (status, body) = app.patch(&format!("/api/notebooks/{nb}/meetings/{id}"), edit).await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert_eq!(body["detail"], "Finish recording before editing");
        let (status, body) = app.post(&format!("/api/notebooks/{nb}/meetings/{id}/approve"), json!({})).await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert_eq!(body["detail"], "Finish the meeting and review its transcript first");
        let (status, body) = app.post(&format!("/api/notebooks/{nb}/meetings/{id}/summarize"), json!({})).await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert_eq!(body["detail"], "Finish recording before generating a summary");
    }

    #[tokio::test]
    async fn finish_without_audio_queues_no_summary() {
        let app = TestApp::new().await;
        let nb = notebook(&app).await;
        with_stt(&app, "words");
        let id = start(&app, &nb).await["id"].as_str().unwrap().to_string();
        let path = format!("/api/notebooks/{nb}/meetings/{id}/finish");
        let (status, body) = app.post(&path, json!({ "total_chunks": 721 })).await;
        assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY);
        assert_eq!(body["detail"][0]["loc"], json!(["body", "total_chunks"]));
        let finished = finish(&app, &nb, &id, 2).await;
        assert_eq!(finished["missing_chunks"], json!([0, 1]));
        assert_eq!(finished["summary_job"], "");
        assert_eq!(finished["summary_status"], "idle");
    }

    #[tokio::test]
    async fn stale_editor_cannot_overwrite_a_newer_meeting() {
        let app = TestApp::new().await;
        let nb = notebook(&app).await;
        let meeting = reviewed(&app, &nb).await;
        let edit = json!({ "revision": 0, "title": "Old", "summary": "Old", "actions": [], "transcript": "Old" });
        let (status, body) = app.patch(&format!("/api/notebooks/{nb}/meetings/{}", meeting.id), edit).await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert_eq!(body["detail"], "Meeting changed; reload it before saving your edits");
        assert_eq!(Meeting::get(&app.state, &meeting.id).await.unwrap().summary, "Budget follow-up agreed.");
    }

    #[tokio::test]
    async fn edit_with_actions_renders_the_note_content() {
        let app = TestApp::new().await;
        let nb = notebook(&app).await;
        let meeting = reviewed(&app, &nb).await;
        let (status, approved) =
            app.post(&format!("/api/notebooks/{nb}/meetings/{}/approve", meeting.id), json!({})).await;
        assert_eq!(status, StatusCode::OK);
        let action = json!({ "text": "Send budget", "owner": "Priya", "due": "Friday", "evidence": "Priya will send it", "done": true });
        let edit = json!({
            "revision": meeting.revision, "title": "Reviewed", "summary": "Plan agreed",
            "actions": [action], "transcript": "Words",
        });
        let (status, body) = app.patch(&format!("/api/notebooks/{nb}/meetings/{}", meeting.id), edit).await;
        assert_eq!(status, StatusCode::OK, "{body}");
        assert_eq!(body["actions"][0]["done"], true);
        let (_, note) = app.get(&format!("/api/notes/{}", approved["note_id"].as_str().unwrap())).await;
        let content = note["content"].as_str().unwrap();
        assert!(content.contains("- [x] Send budget — Priya — Friday"), "{content}");
        assert!(content.contains("Evidence: Priya will send it"));

        let bad = json!({ "revision": 0, "title": "T", "summary": "", "actions": [{ "text": "", "evidence": "e" }], "transcript": "x" });
        let (status, _) = app.patch(&format!("/api/notebooks/{nb}/meetings/{}", meeting.id), bad).await;
        assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY);
    }

    #[tokio::test]
    async fn summary_failure_keeps_the_transcript_and_can_be_retried() {
        let app = TestApp::new().await;
        let nb = notebook(&app).await;
        with_stt(&app, QUOTE);
        app.state.models.set_override(AnyModel::Language(MockLanguageModel::failing(AiError::Network("down".into()))));
        let id = start(&app, &nb).await["id"].as_str().unwrap().to_string();
        upload(&app, &nb, &id, "0", &wav(1600)).await;
        let finished = finish(&app, &nb, &id, 1).await;
        let failed = wait_summary(&app, &id).await;
        assert!(failed.summary_error.starts_with("Summary generation failed"));
        assert_eq!(failed.transcript, finished["transcript"].as_str().unwrap());
        let (_, list) = app.get(&format!("/api/notebooks/{nb}/meetings")).await;
        assert_eq!(list[0]["summary_status"], "completed");
        assert!(list[0]["summary_error"].as_str().unwrap().starts_with("Summary generation failed"));

        // An ungrounded action is a failure too. The model is held so the
        // retry's own failure cannot land before the response is built.
        let invented = summary_reply().replace(QUOTE, "An invented quote");
        let chat = with_chat(&app, &[&invented]);
        chat.hold();
        let (status, body) = app.post(&format!("/api/notebooks/{nb}/meetings/{id}/summarize"), json!({})).await;
        assert_eq!(status, StatusCode::OK, "{body}");
        assert_eq!(body["summary_error"], "", "retry clears the old error");
        chat.release();
        let failed = wait_summary(&app, &id).await;
        assert!(!failed.summary_error.is_empty());
        assert!(failed.actions.is_empty());

        with_chat(&app, &[&format!("<think>hmm</think>```json\n{}\n```", summary_reply())]);
        app.post(&format!("/api/notebooks/{nb}/meetings/{id}/summarize"), json!({})).await;
        let done = wait_summary(&app, &id).await;
        assert_eq!(done.summary, "Budget follow-up agreed.");
        assert_eq!(done.summary_error, "");
    }

    #[tokio::test]
    async fn stale_summary_cannot_overwrite_user_edits() {
        let app = TestApp::new().await;
        let nb = notebook(&app).await;
        let meeting = reviewed(&app, &nb).await;
        let chat = with_chat(&app, &[&summary_reply()]);
        // Editing clears the token, so the earlier job's token is stale.
        let edit = json!({ "revision": meeting.revision, "title": "Mine", "summary": "Mine", "actions": [], "transcript": "Mine" });
        app.patch(&format!("/api/notebooks/{nb}/meetings/{}", meeting.id), edit).await;
        assert!(!summarize_meeting(&app.state, &meeting.id, &meeting.summary_token).await.unwrap());
        assert_eq!(chat.call_count(), 0);
        assert_eq!(Meeting::get(&app.state, &meeting.id).await.unwrap().summary, "Mine");
    }

    #[tokio::test]
    async fn discard_removes_the_draft_and_its_chunks() {
        let app = TestApp::new().await;
        let nb = notebook(&app).await;
        with_stt(&app, "words");
        let id = start(&app, &nb).await["id"].as_str().unwrap().to_string();
        upload(&app, &nb, &id, "0", &wav(1600)).await;
        let (status, body) = app.delete(&format!("/api/notebooks/{nb}/meetings/{id}")).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(body, json!({ "deleted": true }));
        assert_eq!(count(&app, "SELECT * FROM meeting").await, 0);
        assert_eq!(count(&app, "SELECT * FROM meeting_chunk").await, 0);
    }

    #[tokio::test]
    async fn deleting_the_notebook_removes_its_drafts() {
        let app = TestApp::new().await;
        let nb = notebook(&app).await;
        with_stt(&app, "words");
        let id = start(&app, &nb).await["id"].as_str().unwrap().to_string();
        upload(&app, &nb, &id, "0", &wav(1600)).await;
        let (status, _) = app.delete(&format!("/api/notebooks/{nb}")).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(count(&app, "SELECT * FROM meeting").await, 0);
        assert_eq!(count(&app, "SELECT * FROM meeting_chunk").await, 0);
    }

    #[test]
    fn summary_requires_grounded_actions() {
        let reply = format!("```json\n{}\n```", summary_reply());
        let parsed = parse_summary(&reply, QUOTE).unwrap();
        assert!(!parsed.actions[0].done);
        let invented = summary_reply().replace(QUOTE, "An invented quote");
        assert!(parse_summary(&invented, QUOTE).is_err());
        assert!(parse_summary("not json", QUOTE).is_err());
        assert!(parse_summary(r#"{"summary": "", "actions": []}"#, QUOTE).is_err());
    }

    #[tokio::test]
    async fn summary_job_is_registered() {
        let app = TestApp::new().await;
        assert!(app.state.jobs.registered().contains(&"summarize_meeting".to_string()));
    }
}
