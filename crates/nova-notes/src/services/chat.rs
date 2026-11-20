//! Notebook chat: the Rust counterpart of `api/routers/chat.py`,
//! `api/routers/_chat_shared.py` and `open_notebook/graphs/chat.py`.
//!
//! A session is a `chat_session` record linked to its notebook through
//! `refers_to`; its history is the `chat_message` rows (see
//! [`crate::domain::chat`]). One turn renders the `chat/system` prompt with
//! the notebook and the client-selected context, sends it with the history
//! and the new question, and stores question and answer together only when
//! the model produced a non-empty answer, so a failed turn leaves no
//! unanswered question behind (Python's `invoke_chat_turn` rollback).

use chrono::{DateTime, Utc};
use nn_llm::{ChatMessage as LlmMessage, ChatOptions};
use serde::{Deserialize, Serialize};

use crate::db::{Json, Params};
use crate::domain::{ChatSession, DefaultModelType, Notebook, StoredMessage};
use crate::error::{AppError, AppResult};
use crate::jobs::JobQueue;
use crate::services::context::{load_notebook, provision_language_model, py_repr};
use crate::services::notebooks::{double_option, timestamp};
use crate::state::AppState;

/// Notebook chat has no background jobs.
pub fn register_jobs(_jobs: &JobQueue) {}

/// Output budget of a chat answer (Python `max_tokens=8192`).
pub const CHAT_MAX_TOKENS: u32 = 8192;

/// The error a blank model answer produces (after thinking tags are removed).
pub const EMPTY_REPLY_MESSAGE: &str =
    "The model returned an empty response. Try again, or pick a different model if this keeps happening.";

const NOTEBOOK_NOT_FOUND: &str = "Notebook not found";
const SESSION_NOT_FOUND: &str = "Session not found";

// ---------------------------------------------------------------------------
// Shared with source chat
// ---------------------------------------------------------------------------

/// One message as the API reports it (`ChatMessage` in `_chat_shared.py`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ChatMessageOut {
    pub id: String,
    /// `human` or `ai`.
    #[serde(rename = "type")]
    pub message_type: String,
    pub content: String,
    /// Always `null`, as in the Python build.
    pub timestamp: Option<String>,
}

impl From<&StoredMessage> for ChatMessageOut {
    fn from(m: &StoredMessage) -> Self {
        let message_type = if m.role.is_empty() { "unknown".to_string() } else { m.role.clone() };
        Self { id: m.id.clone(), message_type, content: m.content.clone(), timestamp: None }
    }
}

/// `SuccessResponse`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SuccessResponse {
    pub success: bool,
    pub message: String,
}

/// Ensure a record id carries its table prefix (`normalize_record_id`).
pub fn normalize_record_id(table: &str, id: &str) -> String {
    let prefix = format!("{table}:");
    if id.starts_with(&prefix) {
        id.to_string()
    } else {
        format!("{prefix}{id}")
    }
}

/// Load a chat session by bare or prefixed id; anything that is not an
/// existing session is `NotFound` with `missing` as the detail.
pub(crate) async fn load_session(state: &AppState, session_id: &str, missing: &'static str) -> AppResult<ChatSession> {
    let id = normalize_record_id(ChatSession::TABLE, session_id);
    match ChatSession::get(state, &id).await {
        Ok(session) => Ok(session),
        Err(AppError::NotFound(_) | AppError::InvalidInput(_)) => Err(AppError::not_found(missing)),
        Err(e) => Err(e),
    }
}

/// Default title of a new session: `"<prefix> <seconds>"`, as Python's
/// `f"Chat Session {loop.time():.0f}"`.
pub(crate) fn default_title(prefix: &str) -> String {
    format!("{prefix} {}", Utc::now().timestamp())
}

/// The created / updated time as the API reports it.
pub(crate) fn session_time(value: Option<DateTime<Utc>>) -> String {
    timestamp(value)
}

/// Send one chat turn: `system_prompt`, the stored `history` and the new
/// `question`. Picks the model as Python's `provision_langchain_model`
/// (`model_id`, else the default chat model; the large-context model for
/// a very long prompt). Thinking tags are stripped; a blank answer is
/// [`AppError::IncompleteGeneration`].
pub async fn generate_chat_reply(
    state: &AppState,
    system_prompt: String,
    history: &[StoredMessage],
    question: &str,
    model_id: Option<&str>,
) -> AppResult<String> {
    let mut messages = vec![LlmMessage::system(system_prompt)];
    for message in history {
        match message.role.as_str() {
            StoredMessage::HUMAN => messages.push(LlmMessage::user(message.content.clone())),
            StoredMessage::AI => messages.push(LlmMessage::assistant(message.content.clone())),
            _ => {}
        }
    }
    messages.push(LlmMessage::user(question));

    // Python sized the prompt with `str(payload)`; the text of every
    // message is a close enough measure.
    let payload: String = messages.iter().map(|m| m.content.as_str()).collect::<Vec<_>>().join("\n");
    let model = provision_language_model(state, &payload, model_id, DefaultModelType::Chat).await?;
    let reply = model.chat(&messages, &ChatOptions::default().with_max_tokens(CHAT_MAX_TOKENS)).await?;
    let cleaned = nn_content::clean_thinking_content(&reply.content);
    if cleaned.trim().is_empty() {
        return Err(AppError::IncompleteGeneration(EMPTY_REPLY_MESSAGE.into()));
    }
    Ok(cleaned)
}

// ---------------------------------------------------------------------------
// Requests and responses
// ---------------------------------------------------------------------------

/// `CreateSessionRequest`.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CreateSessionRequest {
    pub notebook_id: String,
    #[serde(default)]
    pub title: Option<String>,
    #[serde(default)]
    pub model_override: Option<String>,
}

/// `UpdateSessionRequest`: only fields present in the body change; an
/// explicit `null` clears the field.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct UpdateSessionRequest {
    #[serde(default, deserialize_with = "double_option", skip_serializing_if = "Option::is_none")]
    pub title: Option<Option<String>>,
    #[serde(default, deserialize_with = "double_option", skip_serializing_if = "Option::is_none")]
    pub model_override: Option<Option<String>>,
}

/// `ChatSessionResponse`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ChatSessionResponse {
    pub id: String,
    pub title: String,
    pub notebook_id: Option<String>,
    pub created: String,
    pub updated: String,
    pub message_count: Option<i64>,
    pub model_override: Option<String>,
}

/// `ChatSessionWithMessagesResponse`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ChatSessionWithMessages {
    #[serde(flatten)]
    pub session: ChatSessionResponse,
    pub messages: Vec<ChatMessageOut>,
}

/// `ExecuteChatRequest`. `context` is the client-built context object
/// (normally the `context` of [`crate::services::context::build_context`]).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ExecuteChatRequest {
    pub session_id: String,
    pub message: String,
    pub context: serde_json::Map<String, Json>,
    #[serde(default)]
    pub model_override: Option<String>,
}

/// `ExecuteChatResponse`: the session's whole history after the turn.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExecuteChatResponse {
    pub session_id: String,
    pub messages: Vec<ChatMessageOut>,
}

fn session_response(session: &ChatSession, title: String, notebook_id: Option<String>, count: i64) -> ChatSessionResponse {
    ChatSessionResponse {
        id: session.id.clone(),
        title,
        notebook_id,
        created: session_time(session.created),
        updated: session_time(session.updated),
        message_count: Some(count),
        model_override: session.model_override.clone(),
    }
}

/// The notebook (or other target) a session is attached to, if any.
async fn session_target(state: &AppState, session: &ChatSession) -> AppResult<Option<String>> {
    Ok(session.targets(state).await?.into_iter().next())
}

// ---------------------------------------------------------------------------
// Operations
// ---------------------------------------------------------------------------

/// `GET /api/chat/sessions?notebook_id=`: the notebook's sessions, most
/// recently updated first.
pub async fn list_sessions(state: &AppState, notebook_id: &str) -> AppResult<Vec<ChatSessionResponse>> {
    let notebook = load_notebook(state, notebook_id, NOTEBOOK_NOT_FOUND).await?;
    let mut out = Vec::new();
    for session in notebook.chat_sessions(state).await? {
        let count = StoredMessage::count(state, &session.id).await.unwrap_or_else(|e| {
            tracing::warn!("Could not fetch message count for session {}: {e}", session.id);
            0
        });
        let title = session.title.clone().filter(|t| !t.is_empty()).unwrap_or_else(|| "Untitled Session".into());
        out.push(session_response(&session, title, Some(notebook_id.to_string()), count));
    }
    Ok(out)
}

/// `POST /api/chat/sessions`.
pub async fn create_session(state: &AppState, request: &CreateSessionRequest) -> AppResult<ChatSessionResponse> {
    let notebook = load_notebook(state, &request.notebook_id, NOTEBOOK_NOT_FOUND).await?;
    let title = request.title.clone().filter(|t| !t.is_empty()).unwrap_or_else(|| default_title("Chat Session"));
    let session = ChatSession::create(state, Some(&title), request.model_override.as_deref()).await?;
    session.relate_to(state, &notebook.id).await?;
    Ok(session_response(&session, session.title.clone().unwrap_or_default(), Some(request.notebook_id.clone()), 0))
}

/// `GET /api/chat/sessions/{id}`: a session with its messages.
pub async fn get_session(state: &AppState, session_id: &str) -> AppResult<ChatSessionWithMessages> {
    let session = load_session(state, session_id, SESSION_NOT_FOUND).await?;
    let messages: Vec<ChatMessageOut> = session.messages(state).await?.iter().map(ChatMessageOut::from).collect();
    let notebook_id = session_target(state, &session).await?;
    if notebook_id.is_none() {
        tracing::warn!("No notebook relationship found for session {session_id} - may be an orphaned session");
    }
    let title = session.title.clone().filter(|t| !t.is_empty()).unwrap_or_else(|| "Untitled Session".into());
    let count = messages.len() as i64;
    Ok(ChatSessionWithMessages { session: session_response(&session, title, notebook_id, count), messages })
}

/// `PUT /api/chat/sessions/{id}`.
pub async fn update_session(state: &AppState, session_id: &str, request: &UpdateSessionRequest) -> AppResult<ChatSessionResponse> {
    let mut session = load_session(state, session_id, SESSION_NOT_FOUND).await?;
    if let Some(title) = &request.title {
        session.title = title.clone();
    }
    if let Some(model_override) = &request.model_override {
        session.model_override = model_override.clone();
    }
    session.save(state).await.map_err(not_found(SESSION_NOT_FOUND))?;
    if matches!(request.title, Some(None)) {
        // `save` never clears a title; an explicit null does.
        state.db.query("UPDATE $id SET title = NONE", Params::new().record("id", &session.id)).await?;
        session.title = None;
    }
    let notebook_id = session_target(state, &session).await?;
    let count = StoredMessage::count(state, &session.id).await.unwrap_or(0);
    Ok(session_response(&session, session.title.clone().unwrap_or_default(), notebook_id, count))
}

/// `DELETE /api/chat/sessions/{id}`: the session, its links and messages.
pub async fn delete_session(state: &AppState, session_id: &str) -> AppResult<SuccessResponse> {
    let session = load_session(state, session_id, SESSION_NOT_FOUND).await?;
    session.delete(state).await.map_err(not_found(SESSION_NOT_FOUND))?;
    Ok(SuccessResponse { success: true, message: "Session deleted successfully".into() })
}

/// `POST /api/chat/execute`: one notebook chat turn.
///
/// The request's `model_override` wins over the session's. The prompt gets
/// the session's notebook (name and description) and the request context,
/// rendered as Python's `str(dict)`.
pub async fn execute_chat(state: &AppState, request: &ExecuteChatRequest) -> AppResult<ExecuteChatResponse> {
    let session = load_session(state, &request.session_id, SESSION_NOT_FOUND).await?;
    let notebook = match session_target(state, &session).await? {
        Some(id) if crate::db::table_of(&id) == Some(Notebook::TABLE) => Notebook::get(state, &id).await.ok(),
        _ => None,
    };
    let model_override = request.model_override.clone().or_else(|| session.model_override.clone());

    let context = (!request.context.is_empty()).then(|| py_repr(&Json::Object(request.context.clone())));
    let notebook_data = notebook.as_ref().map(|n| serde_json::json!({ "id": n.id, "name": n.name, "description": n.description }));
    let system_prompt = state.prompts.render("chat/system", serde_json::json!({ "notebook": notebook_data, "context": context }))?;

    let history = session.messages(state).await?;
    let reply = generate_chat_reply(state, system_prompt, &history, &request.message, model_override.as_deref())
        .await
        .inspect_err(|e| {
            tracing::error!("Error executing chat: {e} (session {}, model override {:?})", request.session_id, model_override)
        })?;

    StoredMessage::append(state, &session.id, StoredMessage::HUMAN, &request.message).await?;
    StoredMessage::append(state, &session.id, StoredMessage::AI, &reply).await?;

    let messages = session.messages(state).await?.iter().map(ChatMessageOut::from).collect();
    Ok(ExecuteChatResponse { session_id: request.session_id.clone(), messages })
}

fn not_found(detail: &'static str) -> impl Fn(AppError) -> AppError {
    move |e| if e.is_not_found() { AppError::not_found(detail) } else { e }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use axum::http::StatusCode;
    use nn_llm::{AiError, Role};
    use serde_json::json;

    use crate::ai::AnyModel;
    use crate::testing::{MockLanguageModel, TestApp};

    async fn notebook(app: &TestApp) -> String {
        let (_, nb) = app.post("/api/notebooks", json!({ "name": "Physics", "description": "All about waves" })).await;
        nb["id"].as_str().unwrap().to_string()
    }

    fn mock(app: &TestApp, replies: &[&str]) -> Arc<MockLanguageModel> {
        let model = MockLanguageModel::new(replies.iter().copied());
        app.state.models.set_override(AnyModel::Language(model.clone()));
        model
    }

    #[tokio::test]
    async fn session_crud_per_notebook() {
        let app = TestApp::new().await;
        let nb = notebook(&app).await;
        let other = notebook(&app).await;

        let (status, created) = app.post("/api/chat/sessions", json!({ "notebook_id": nb, "title": "First" })).await;
        assert_eq!(status, StatusCode::OK, "{created}");
        assert_eq!(created["title"], "First");
        assert_eq!(created["notebook_id"], nb);
        assert_eq!(created["message_count"], 0);
        assert!(created["model_override"].is_null());
        let id = created["id"].as_str().unwrap().to_string();
        assert!(id.starts_with("chat_session:"));

        let (_, untitled) = app.post("/api/chat/sessions", json!({ "notebook_id": nb })).await;
        assert!(untitled["title"].as_str().unwrap().starts_with("Chat Session "), "{untitled}");
        app.post("/api/chat/sessions", json!({ "notebook_id": other, "title": "Elsewhere" })).await;

        let (status, list) = app.get(&format!("/api/chat/sessions?notebook_id={nb}")).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(list.as_array().unwrap().len(), 2, "{list}");
        assert!(list.as_array().unwrap().iter().all(|s| s["notebook_id"] == nb));

        // Bare ids are prefixed.
        let bare = id.split_once(':').unwrap().1;
        let (status, one) = app.get(&format!("/api/chat/sessions/{bare}")).await;
        assert_eq!(status, StatusCode::OK, "{one}");
        assert_eq!(one["id"], id);
        assert_eq!(one["notebook_id"], nb);
        assert_eq!(one["messages"], json!([]));
        assert_eq!(one["message_count"], 0);

        let (status, updated) = app.put(&format!("/api/chat/sessions/{id}"), json!({ "title": "Renamed", "model_override": "model:x" })).await;
        assert_eq!(status, StatusCode::OK, "{updated}");
        assert_eq!(updated["title"], "Renamed");
        assert_eq!(updated["model_override"], "model:x");
        // Absent fields are kept; an explicit null clears.
        let (_, updated) = app.put(&format!("/api/chat/sessions/{id}"), json!({ "model_override": null })).await;
        assert_eq!(updated["title"], "Renamed");
        assert!(updated["model_override"].is_null());

        let (status, deleted) = app.delete(&format!("/api/chat/sessions/{id}")).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(deleted, json!({ "success": true, "message": "Session deleted successfully" }));
        let (status, body) = app.get(&format!("/api/chat/sessions/{id}")).await;
        assert_eq!(status, StatusCode::NOT_FOUND);
        assert_eq!(body["detail"], "Session not found");
    }

    #[tokio::test]
    async fn not_found_and_validation() {
        let app = TestApp::new().await;
        let (status, body) = app.delete("/api/chat/sessions/gone").await;
        assert_eq!(status, StatusCode::NOT_FOUND);
        assert_eq!(body["detail"], "Session not found");
        let (status, body) = app.put("/api/chat/sessions/gone", json!({ "title": "x" })).await;
        assert_eq!(status, StatusCode::NOT_FOUND);
        assert_eq!(body["detail"], "Session not found");
        let (status, body) = app.get("/api/chat/sessions?notebook_id=notebook:gone").await;
        assert_eq!(status, StatusCode::NOT_FOUND);
        assert_eq!(body["detail"], "Notebook not found");
        let (status, body) = app.post("/api/chat/sessions", json!({ "notebook_id": "notebook:gone" })).await;
        assert_eq!(status, StatusCode::NOT_FOUND);
        assert_eq!(body["detail"], "Notebook not found");
        let (status, _) = app.get("/api/chat/sessions").await;
        assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY);
        let (status, _) = app.post("/api/chat/execute", json!({ "session_id": "x", "message": "hi" })).await;
        assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY, "context is required");
        let (status, body) = app.post("/api/chat/execute", json!({ "session_id": "gone", "message": "hi", "context": {} })).await;
        assert_eq!(status, StatusCode::NOT_FOUND);
        assert_eq!(body["detail"], "Session not found");
    }

    #[tokio::test]
    async fn execute_persists_history_and_sends_context() {
        let app = TestApp::new().await;
        let model = mock(&app, &["<think>pondering</think>First answer", "Second answer"]);
        let nb = notebook(&app).await;
        let (_, session) = app.post("/api/chat/sessions", json!({ "notebook_id": nb })).await;
        let sid = session["id"].as_str().unwrap();

        let context = json!({ "sources": [{ "id": "source:abc", "title": "Waves", "full_text": "Waves carry energy" }], "notes": [] });
        let (status, body) =
            app.post("/api/chat/execute", json!({ "session_id": sid, "message": "What do waves carry?", "context": context })).await;
        assert_eq!(status, StatusCode::OK, "{body}");
        assert_eq!(body["session_id"], sid);
        let messages = body["messages"].as_array().unwrap();
        assert_eq!(messages.len(), 2);
        assert_eq!(messages[0]["type"], "human");
        assert_eq!(messages[0]["content"], "What do waves carry?");
        assert_eq!(messages[1]["type"], "ai");
        assert_eq!(messages[1]["content"], "First answer", "thinking is stripped");
        assert!(messages[1]["timestamp"].is_null());

        let calls = model.calls.lock().unwrap().clone();
        let system = &calls[0][0];
        assert_eq!(system.role, Role::System);
        assert!(system.content.contains("**Name:** Physics"), "{}", system.content);
        assert!(system.content.contains("**Description:** All about waves"));
        assert!(system.content.contains("'full_text': 'Waves carry energy'"), "{}", system.content);
        assert_eq!(calls[0].last().unwrap().content, "What do waves carry?");
        assert_eq!(model.options.lock().unwrap()[0].max_tokens, Some(8192));

        // The second turn carries the stored history; an empty context is left out.
        let (_, body) = app.post("/api/chat/execute", json!({ "session_id": sid, "message": "And then?", "context": {} })).await;
        assert_eq!(body["messages"].as_array().unwrap().len(), 4);
        let calls = model.calls.lock().unwrap().clone();
        let roles: Vec<Role> = calls[1].iter().map(|m| m.role).collect();
        assert_eq!(roles, [Role::System, Role::User, Role::Assistant, Role::User]);
        assert_eq!(calls[1][2].content, "First answer");
        assert!(!calls[1][0].content.contains("# CONTEXT"));

        let (_, one) = app.get(&format!("/api/chat/sessions/{sid}")).await;
        assert_eq!(one["message_count"], 4);
        let (_, list) = app.get(&format!("/api/chat/sessions?notebook_id={nb}")).await;
        assert_eq!(list[0]["message_count"], 4);
    }

    #[tokio::test]
    async fn failed_or_empty_turns_leave_no_history() {
        let app = TestApp::new().await;
        mock(&app, &["<think>only reasoning</think>", "   ", "the answer"]);
        let nb = notebook(&app).await;
        let (_, session) = app.post("/api/chat/sessions", json!({ "notebook_id": nb })).await;
        let sid = session["id"].as_str().unwrap();

        for _ in 0..2 {
            let (status, body) = app.post("/api/chat/execute", json!({ "session_id": sid, "message": "q", "context": {} })).await;
            assert_eq!(status, StatusCode::BAD_GATEWAY);
            assert!(body["detail"].as_str().unwrap().contains("empty response"), "{body}");
        }
        let (_, one) = app.get(&format!("/api/chat/sessions/{sid}")).await;
        assert_eq!(one["messages"], json!([]));

        let (status, body) = app.post("/api/chat/execute", json!({ "session_id": sid, "message": "q", "context": {} })).await;
        assert_eq!(status, StatusCode::OK);
        let pairs: Vec<(String, String)> = body["messages"]
            .as_array()
            .unwrap()
            .iter()
            .map(|m| (m["type"].as_str().unwrap().into(), m["content"].as_str().unwrap().into()))
            .collect();
        assert_eq!(pairs, [("human".into(), "q".into()), ("ai".into(), "the answer".into())]);

        // A provider failure keeps its status class and leaves no trace.
        app.state.models.set_override(AnyModel::Language(MockLanguageModel::failing(AiError::RateLimit("slow down".into()))));
        let (status, body) = app.post("/api/chat/execute", json!({ "session_id": sid, "message": "again", "context": {} })).await;
        assert_eq!(status, StatusCode::TOO_MANY_REQUESTS);
        assert_eq!(body["detail"], "slow down");
        let (_, one) = app.get(&format!("/api/chat/sessions/{sid}")).await;
        assert_eq!(one["message_count"], 2);
    }

    #[tokio::test]
    async fn model_override_is_respected() {
        let app = TestApp::new().await;
        let model = mock(&app, &["ok"]);
        let nb = notebook(&app).await;
        let (_, session) = app.post("/api/chat/sessions", json!({ "notebook_id": nb, "model_override": "model:missing" })).await;
        assert_eq!(session["model_override"], "model:missing");
        let sid = session["id"].as_str().unwrap();

        // The session's override is used: it names no model, so the turn fails.
        let (status, body) = app.post("/api/chat/execute", json!({ "session_id": sid, "message": "q", "context": {} })).await;
        assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY, "{body}");
        assert_eq!(body["detail"], "Model with ID model:missing not found");
        assert_eq!(model.call_count(), 0);

        // A per-request override wins over the session's.
        let row = app
            .state
            .db
            .create("model", crate::db::Doc::new().set("name", "m").set("provider", "openai").set("type", "language"))
            .await
            .unwrap();
        let model_id = row["id"].as_str().unwrap();
        let (status, body) =
            app.post("/api/chat/execute", json!({ "session_id": sid, "message": "q", "context": {}, "model_override": model_id })).await;
        assert_eq!(status, StatusCode::OK, "{body}");
        assert_eq!(model.call_count(), 1);
    }
}
