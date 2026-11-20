//! Source chat: the Rust counterpart of `api/routers/source_chat.py` and
//! `open_notebook/graphs/source_chat.py`.
//!
//! A session is a `chat_session` linked to one source through `refers_to`.
//! Each turn builds the source's budgeted context
//! ([`build_source_context`]), renders `source_chat/system`, asks the model
//! and stores question and answer, the answer carrying the turn's context
//! indicators. Sending a message answers as a stream of
//! [`SourceChatEvent`]s (`user_message`, `ai_message`, `context_indicators`,
//! `complete`, or `error`), the events the Python build sent over SSE.

use futures::stream::BoxStream;
use futures::StreamExt;
use serde::{Deserialize, Serialize};

use crate::db::{with_table, Json};
use crate::domain::{ChatSession, Source, StoredMessage};
use crate::error::{AppError, AppResult};
use crate::jobs::JobQueue;
use crate::services::chat::{default_title, generate_chat_reply, load_session, normalize_record_id, session_time, ChatMessageOut, SuccessResponse};
use crate::services::context::{build_source_context, SOURCE_CHAT_CONTEXT_TOKENS};
use crate::state::AppState;

/// Source chat has no background jobs.
pub fn register_jobs(_jobs: &JobQueue) {}

/// The `chat_message` field holding an answer's context indicators.
const INDICATORS_FIELD: &str = "context_indicators";

// ---------------------------------------------------------------------------
// Requests, responses and events
// ---------------------------------------------------------------------------

/// `CreateSourceChatSessionRequest`. `source_id` is required by the API but
/// the path decides which source the session belongs to.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CreateSourceChatSessionRequest {
    pub source_id: String,
    #[serde(default)]
    pub title: Option<String>,
    #[serde(default)]
    pub model_override: Option<String>,
}

/// `UpdateSourceChatSessionRequest`: `null` / absent fields are left alone.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct UpdateSourceChatSessionRequest {
    #[serde(default)]
    pub title: Option<String>,
    #[serde(default)]
    pub model_override: Option<String>,
}

/// Which sources, insights and notes reached the model in a turn.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ContextIndicator {
    #[serde(default)]
    pub sources: Vec<String>,
    #[serde(default)]
    pub insights: Vec<String>,
    #[serde(default)]
    pub notes: Vec<String>,
}

/// `SourceChatSessionResponse`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SourceChatSessionResponse {
    pub id: String,
    pub title: String,
    /// The source id as the client sent it in the path.
    pub source_id: String,
    pub model_override: Option<String>,
    pub created: String,
    pub updated: String,
    pub message_count: Option<i64>,
}

/// `SourceChatSessionWithMessagesResponse`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SourceChatSessionWithMessages {
    #[serde(flatten)]
    pub session: SourceChatSessionResponse,
    pub messages: Vec<ChatMessageOut>,
    pub context_indicators: Option<ContextIndicator>,
}

/// `SendMessageRequest`.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SendMessageRequest {
    pub message: String,
    #[serde(default)]
    pub model_override: Option<String>,
}

/// One server-sent event of a source chat turn (`data: <json>`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum SourceChatEvent {
    UserMessage { content: String, timestamp: Option<String> },
    AiMessage { content: String, timestamp: Option<String> },
    ContextIndicators { data: ContextIndicator },
    Complete,
    Error { message: String },
}

/// A verified turn, ready to run with [`stream_message`].
#[derive(Debug, Clone)]
pub struct SourceChatTurn {
    pub source_id: String,
    pub session_id: String,
    pub message: String,
    pub model_override: Option<String>,
}

// ---------------------------------------------------------------------------
// Verification
// ---------------------------------------------------------------------------

/// The source by bare or prefixed id; missing → `NotFound(missing)`.
async fn load_source(state: &AppState, source_id: &str, missing: &'static str) -> AppResult<Source> {
    let id = normalize_record_id(Source::TABLE, source_id);
    match Source::get(state, &id).await {
        Ok(source) => Ok(source),
        Err(AppError::NotFound(_) | AppError::InvalidInput(_)) => Err(AppError::not_found(missing)),
        Err(e) => Err(e),
    }
}

/// `get_verified_source_session`: the source, the session, and the link
/// between them. A missing source or session is `NotFound(missing)`; a
/// session of another source is "Session not found for this source".
async fn verified_session(state: &AppState, source_id: &str, session_id: &str, missing: &'static str) -> AppResult<(Source, ChatSession)> {
    let source = load_source(state, source_id, missing).await?;
    let session = load_session(state, session_id, missing).await?;
    if !session.belongs_to(state, &source.id).await? {
        return Err(AppError::not_found("Session not found for this source"));
    }
    Ok((source, session))
}

fn session_response(session: &ChatSession, source_id: &str, count: i64) -> SourceChatSessionResponse {
    SourceChatSessionResponse {
        id: session.id.clone(),
        title: session.title.clone().filter(|t| !t.is_empty()).unwrap_or_else(|| "Untitled Session".into()),
        source_id: source_id.to_string(),
        model_override: session.model_override.clone(),
        created: session_time(session.created),
        updated: session_time(session.updated),
        message_count: Some(count),
    }
}

async fn message_count(state: &AppState, session_id: &str) -> i64 {
    StoredMessage::count(state, session_id).await.unwrap_or_else(|e| {
        tracing::warn!("Could not fetch message count for session {session_id}: {e}");
        0
    })
}

// ---------------------------------------------------------------------------
// Sessions
// ---------------------------------------------------------------------------

/// `POST /api/sources/{source_id}/chat/sessions`.
pub async fn create_session(state: &AppState, source_id: &str, request: &CreateSourceChatSessionRequest) -> AppResult<SourceChatSessionResponse> {
    let source = load_source(state, source_id, "Source not found").await?;
    let title = request.title.clone().filter(|t| !t.is_empty()).unwrap_or_else(|| default_title("Source Chat"));
    let session = ChatSession::create(state, Some(&title), request.model_override.as_deref()).await?;
    session.relate_to(state, &source.id).await?;
    Ok(session_response(&session, source_id, 0))
}

/// `GET /api/sources/{source_id}/chat/sessions`: newest first.
pub async fn list_sessions(state: &AppState, source_id: &str) -> AppResult<Vec<SourceChatSessionResponse>> {
    let source = load_source(state, source_id, "Source not found").await?;
    let mut sessions = ChatSession::for_target(state, &source.id).await?;
    sessions.sort_by_key(|s| std::cmp::Reverse(s.created));
    let mut out = Vec::with_capacity(sessions.len());
    for session in &sessions {
        out.push(session_response(session, source_id, message_count(state, &session.id).await));
    }
    Ok(out)
}

/// `GET /api/sources/{source_id}/chat/sessions/{session_id}`: the session,
/// its messages and the context indicators of the latest answer.
pub async fn get_session(state: &AppState, source_id: &str, session_id: &str) -> AppResult<SourceChatSessionWithMessages> {
    let (_, session) = verified_session(state, source_id, session_id, "Source or session not found").await?;
    let messages: Vec<ChatMessageOut> = session.messages(state).await?.iter().map(ChatMessageOut::from).collect();
    let context_indicators = StoredMessage::latest_value(state, &session.id, INDICATORS_FIELD)
        .await?
        .and_then(|value| serde_json::from_value::<ContextIndicator>(value).ok());
    let count = messages.len() as i64;
    Ok(SourceChatSessionWithMessages { session: session_response(&session, source_id, count), messages, context_indicators })
}

/// `PUT /api/sources/{source_id}/chat/sessions/{session_id}`.
pub async fn update_session(
    state: &AppState,
    source_id: &str,
    session_id: &str,
    request: &UpdateSourceChatSessionRequest,
) -> AppResult<SourceChatSessionResponse> {
    let (_, mut session) = verified_session(state, source_id, session_id, "Source or session not found").await?;
    if let Some(title) = &request.title {
        session.title = Some(title.clone());
    }
    if let Some(model_override) = &request.model_override {
        session.model_override = Some(model_override.clone());
    }
    session.save(state).await?;
    Ok(session_response(&session, source_id, message_count(state, &session.id).await))
}

/// Set or clear (`None`) a session's model override. The JSON API's update
/// treats `null` as "leave alone", as the Python does, so it cannot clear
/// one; the web UI's "Default model" choice uses this instead.
pub async fn set_model_override(
    state: &AppState,
    source_id: &str,
    session_id: &str,
    model_override: Option<&str>,
) -> AppResult<SourceChatSessionResponse> {
    let (_, mut session) = verified_session(state, source_id, session_id, "Source or session not found").await?;
    session.model_override = model_override.filter(|m| !m.is_empty()).map(str::to_string);
    session.save(state).await?;
    Ok(session_response(&session, source_id, message_count(state, &session.id).await))
}

/// `DELETE /api/sources/{source_id}/chat/sessions/{session_id}`.
pub async fn delete_session(state: &AppState, source_id: &str, session_id: &str) -> AppResult<SuccessResponse> {
    let (_, session) = verified_session(state, source_id, session_id, "Source or session not found").await?;
    session.delete(state).await?;
    Ok(SuccessResponse { success: true, message: "Source chat session deleted successfully".into() })
}

// ---------------------------------------------------------------------------
// Messages
// ---------------------------------------------------------------------------

/// Check a message before streaming starts: source, session and link exist
/// and the message is not empty. The request's `model_override` (when
/// non-empty) wins over the session's.
pub async fn prepare_message(state: &AppState, source_id: &str, session_id: &str, request: &SendMessageRequest) -> AppResult<SourceChatTurn> {
    let (source, session) = verified_session(state, source_id, session_id, "Source or session not found").await?;
    if request.message.is_empty() {
        return Err(AppError::bad_request("Message content is required"));
    }
    let model_override =
        request.model_override.clone().filter(|m| !m.is_empty()).or_else(|| session.model_override.clone());
    Ok(SourceChatTurn { source_id: source.id, session_id: session.id, message: request.message.clone(), model_override })
}

/// The answer of one turn, already stored.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SourceChatAnswer {
    pub content: String,
    pub context_indicators: ContextIndicator,
}

/// Run one turn: build the context, ask the model, store question and
/// answer. Nothing is stored when it fails.
pub async fn run_turn(state: &AppState, turn: &SourceChatTurn) -> AppResult<SourceChatAnswer> {
    let context = build_source_context(state, &turn.source_id, Some(SOURCE_CHAT_CONTEXT_TOKENS)).await?;
    let mut indicators = ContextIndicator::default();
    let source = context.sources.first();
    if let Some(source) = source {
        if context.metadata.source_text_status.is_available() {
            indicators.sources.push(source.id.clone());
        }
    }
    indicators.insights = context.insights.iter().map(|i| i.id.clone()).collect();

    let topics = match source {
        Some(s) => Source::get(state, &with_table(Source::TABLE, &s.id)).await.map(|s| s.topics).unwrap_or_default(),
        None => Vec::new(),
    };
    let source_data = source.map(|s| serde_json::json!({ "id": s.id, "title": s.title, "full_text": s.full_text, "topics": topics }));
    let prompt_data = serde_json::json!({
        "source": source_data,
        "insights": context.insights,
        "context": context.formatted(),
        "context_indicators": indicators,
    });
    let system_prompt = state.prompts.render("source_chat/system", prompt_data)?;

    let history = StoredMessage::history(state, &turn.session_id).await?;
    let content = generate_chat_reply(state, system_prompt, &history, &turn.message, turn.model_override.as_deref()).await?;

    StoredMessage::append(state, &turn.session_id, StoredMessage::HUMAN, &turn.message).await?;
    let indicators_json: Json = serde_json::to_value(&indicators)?;
    StoredMessage::append_with(state, &turn.session_id, StoredMessage::AI, &content, &[(INDICATORS_FIELD, indicators_json)]).await?;
    Ok(SourceChatAnswer { content, context_indicators: indicators })
}

/// Run a turn as the event stream the source chat UI consumes. The turn
/// runs in its own task, so it completes (and is stored) even when the
/// client goes away mid-stream.
pub fn stream_message(state: AppState, turn: SourceChatTurn) -> BoxStream<'static, SourceChatEvent> {
    async_stream::stream! {
        yield SourceChatEvent::UserMessage { content: turn.message.clone(), timestamp: None };
        let task_state = state.clone();
        let task_turn = turn.clone();
        let result = match tokio::spawn(async move { run_turn(&task_state, &task_turn).await }).await {
            Ok(result) => result,
            Err(e) => Err(AppError::from(e)),
        };
        match result {
            Ok(answer) => {
                yield SourceChatEvent::AiMessage { content: answer.content, timestamp: None };
                yield SourceChatEvent::ContextIndicators { data: answer.context_indicators };
                yield SourceChatEvent::Complete;
            }
            Err(e) => {
                tracing::error!("Error in source chat streaming: {e}");
                yield SourceChatEvent::Error { message: e.detail() };
            }
        }
    }
    .boxed()
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use axum::http::StatusCode;
    use serde_json::{json, Value};

    use super::*;
    use crate::ai::AnyModel;
    use crate::db::Doc;
    use crate::testing::{MockLanguageModel, TestApp};

    async fn source(app: &TestApp, title: &str, text: &str) -> Source {
        Source::create(&app.state, &Source { title: Some(title.into()), full_text: Some(text.into()), ..Source::default() })
            .await
            .unwrap()
    }

    fn mock(app: &TestApp, replies: &[&str]) -> Arc<MockLanguageModel> {
        let model = MockLanguageModel::new(replies.iter().copied());
        app.state.models.set_override(AnyModel::Language(model.clone()));
        model
    }

    fn events(body: &str) -> Vec<Value> {
        body.split("\n\n")
            .filter(|chunk| !chunk.is_empty())
            .map(|chunk| serde_json::from_str(chunk.strip_prefix("data: ").expect("data line")).unwrap())
            .collect()
    }

    #[tokio::test]
    async fn session_crud_and_404s() {
        let app = TestApp::new().await;
        let s = source(&app, "S", "text").await;
        let other = source(&app, "O", "other").await;
        let bare = s.id.split_once(':').unwrap().1.to_string();

        let (status, created) = app.post(&format!("/api/sources/{bare}/chat/sessions"), json!({ "source_id": bare, "title": "Mine" })).await;
        assert_eq!(status, StatusCode::OK, "{created}");
        assert_eq!(created["source_id"], bare, "the path id is echoed");
        assert_eq!(created["title"], "Mine");
        let sid = created["id"].as_str().unwrap().to_string();
        let (_, untitled) = app.post(&format!("/api/sources/{}/chat/sessions", s.id), json!({ "source_id": s.id })).await;
        assert!(untitled["title"].as_str().unwrap().starts_with("Source Chat "));
        let (_, foreign) = app.post(&format!("/api/sources/{}/chat/sessions", other.id), json!({ "source_id": other.id })).await;
        let foreign = foreign["id"].as_str().unwrap().to_string();

        let (status, list) = app.get(&format!("/api/sources/{}/chat/sessions", s.id)).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(list.as_array().unwrap().len(), 2);

        let (status, one) = app.get(&format!("/api/sources/{bare}/chat/sessions/{}", sid.split_once(':').unwrap().1)).await;
        assert_eq!(status, StatusCode::OK, "{one}");
        assert_eq!(one["id"], sid);
        assert_eq!(one["messages"], json!([]));
        assert!(one["context_indicators"].is_null());

        let (status, updated) =
            app.put(&format!("/api/sources/{}/chat/sessions/{sid}", s.id), json!({ "title": "New", "model_override": "model:m" })).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(updated["title"], "New");
        assert_eq!(updated["model_override"], "model:m");
        let (_, updated) = app.put(&format!("/api/sources/{}/chat/sessions/{sid}", s.id), json!({ "model_override": null })).await;
        assert_eq!(updated["model_override"], "model:m", "null leaves the override alone");

        // 404s, with the Python detail strings.
        for (status, body) in [
            app.post("/api/sources/gone/chat/sessions", json!({ "source_id": "gone" })).await,
            app.get("/api/sources/source:gone/chat/sessions").await,
        ] {
            assert_eq!(status, StatusCode::NOT_FOUND);
            assert_eq!(body["detail"], "Source not found");
        }
        let (status, body) = app.get(&format!("/api/sources/{}/chat/sessions/gone", s.id)).await;
        assert_eq!(status, StatusCode::NOT_FOUND);
        assert_eq!(body["detail"], "Source or session not found");
        let path = format!("/api/sources/{}/chat/sessions/{foreign}", s.id);
        for (status, body) in [
            app.get(&path).await,
            app.put(&path, json!({ "title": "x" })).await,
            app.delete(&path).await,
            app.post(&format!("{path}/messages"), json!({ "message": "hi" })).await,
        ] {
            assert_eq!(status, StatusCode::NOT_FOUND);
            assert_eq!(body["detail"], "Session not found for this source");
        }
        let (status, body) = app.post(&format!("/api/sources/{}/chat/sessions/{sid}/messages", s.id), json!({ "message": "" })).await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert_eq!(body["detail"], "Message content is required");

        let (status, body) = app.delete(&format!("/api/sources/{}/chat/sessions/{sid}", s.id)).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(body["message"], "Source chat session deleted successfully");
        let (status, _) = app.get(&format!("/api/sources/{}/chat/sessions/{sid}", s.id)).await;
        assert_eq!(status, StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn streaming_turns_persist_history_and_indicators() {
        let app = TestApp::new().await;
        let model = mock(&app, &["<think>hmm</think>first answer", "second answer"]);
        let s = source(&app, "Waves", "Waves carry energy across space.").await;
        let insight = app
            .state
            .db
            .create("source_insight", Doc::new().record("source", &s.id).set("insight_type", "summary").set("content", "About waves"))
            .await
            .unwrap()["id"]
            .as_str()
            .unwrap()
            .to_string();
        let (_, session) = app.post(&format!("/api/sources/{}/chat/sessions", s.id), json!({ "source_id": s.id })).await;
        let sid = session["id"].as_str().unwrap().to_string();
        let path = format!("/api/sources/{}/chat/sessions/{sid}/messages", s.id);

        let request = axum::http::Request::builder()
            .method("POST")
            .uri(&path)
            .header("content-type", "application/json")
            .body(axum::body::Body::from(r#"{"message":"What do waves carry?"}"#))
            .unwrap();
        let response = app.send(request).await;
        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(response.headers()["content-type"], "text/event-stream");
        assert_eq!(response.headers()["cache-control"], "no-cache");
        assert_eq!(response.headers()["x-accel-buffering"], "no");
        let bytes = http_body_util::BodyExt::collect(response.into_body()).await.unwrap().to_bytes();
        let first = events(&String::from_utf8_lossy(&bytes));
        assert_eq!(
            first,
            [
                json!({ "type": "user_message", "content": "What do waves carry?", "timestamp": null }),
                json!({ "type": "ai_message", "content": "first answer", "timestamp": null }),
                json!({ "type": "context_indicators", "data": { "sources": [s.id], "insights": [insight], "notes": [] } }),
                json!({ "type": "complete" }),
            ]
        );
        // Clone: a borrowed guard would live to the end of the test and
        // deadlock the next model call.
        let system = model.calls.lock().unwrap()[0][0].content.clone();
        assert!(system.contains("**Source ID:** source:"), "{system}");
        assert!(system.contains("Waves carry energy across space."));
        assert!(system.contains("About waves"));

        // The second turn sends only its own answer and carries the history.
        let (_, body) = app.text(axum::http::Method::POST, &path, Some(json!({ "message": "More?" }))).await;
        let second = events(&body);
        let answers: Vec<&Value> = second.iter().filter(|e| e["type"] == "ai_message").collect();
        assert_eq!(answers.len(), 1);
        assert_eq!(answers[0]["content"], "second answer");
        assert_eq!(model.calls.lock().unwrap()[1].len(), 4);

        let (_, one) = app.get(&format!("/api/sources/{}/chat/sessions/{sid}", s.id)).await;
        assert_eq!(one["message_count"], 4);
        let kinds: Vec<&str> = one["messages"].as_array().unwrap().iter().map(|m| m["type"].as_str().unwrap()).collect();
        assert_eq!(kinds, ["human", "ai", "human", "ai"]);
        assert_eq!(one["context_indicators"], json!({ "sources": [s.id], "insights": [insight], "notes": [] }));
    }

    #[tokio::test]
    async fn failed_turn_streams_an_error_and_is_not_stored() {
        let app = TestApp::new().await;
        mock(&app, &["", "the answer"]);
        let s = source(&app, "S", "text").await;
        let (_, session) = app.post(&format!("/api/sources/{}/chat/sessions", s.id), json!({ "source_id": s.id })).await;
        let sid = session["id"].as_str().unwrap().to_string();
        let path = format!("/api/sources/{}/chat/sessions/{sid}/messages", s.id);

        let (status, body) = app.text(axum::http::Method::POST, &path, Some(json!({ "message": "question" }))).await;
        assert_eq!(status, StatusCode::OK);
        let failed = events(&body);
        assert_eq!(failed[0]["type"], "user_message");
        assert_eq!(failed.last().unwrap()["type"], "error");
        assert!(failed.last().unwrap()["message"].as_str().unwrap().contains("empty response"));
        assert!(!failed.iter().any(|e| e["type"] == "ai_message"));
        assert_eq!(StoredMessage::count(&app.state, &sid).await.unwrap(), 0);

        let (_, body) = app.text(axum::http::Method::POST, &path, Some(json!({ "message": "question" }))).await;
        let retried = events(&body);
        assert_eq!(retried.iter().filter(|e| e["type"] == "ai_message").map(|e| e["content"].clone()).collect::<Vec<_>>(), [json!("the answer")]);
        let history = StoredMessage::history(&app.state, &sid).await.unwrap();
        let pairs: Vec<(&str, &str)> = history.iter().map(|m| (m.role.as_str(), m.content.as_str())).collect();
        assert_eq!(pairs, [("human", "question"), ("ai", "the answer")]);
    }

    #[tokio::test]
    async fn session_model_override_is_used() {
        let app = TestApp::new().await;
        let model = mock(&app, &["ok"]);
        let s = source(&app, "S", "text").await;
        let (_, session) =
            app.post(&format!("/api/sources/{}/chat/sessions", s.id), json!({ "source_id": s.id, "model_override": "model:none" })).await;
        let sid = session["id"].as_str().unwrap().to_string();
        let path = format!("/api/sources/{}/chat/sessions/{sid}/messages", s.id);
        let (_, body) = app.text(axum::http::Method::POST, &path, Some(json!({ "message": "q" }))).await;
        let e = events(&body);
        assert_eq!(e.last().unwrap(), &json!({ "type": "error", "message": "Model with ID model:none not found" }));
        assert_eq!(model.call_count(), 0);

        // An empty request override falls back to the session's; a real one wins.
        let row = app.state.db.create("model", Doc::new().set("name", "m").set("provider", "openai").set("type", "language")).await.unwrap();
        let (_, body) =
            app.text(axum::http::Method::POST, &path, Some(json!({ "message": "q", "model_override": row["id"] }))).await;
        assert_eq!(events(&body).last().unwrap()["type"], "complete");
        assert_eq!(model.call_count(), 1);
    }
}
