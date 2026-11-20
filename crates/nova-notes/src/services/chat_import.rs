//! One-off import of chat history written by the Python build.
//!
//! The Python build keeps chat messages in a LangGraph SQLite checkpoint;
//! this build keeps them in the `chat_message` table. `scripts/
//! export_langgraph_chats.py` (run with the Python image, which can read its
//! own checkpoints) writes one JSON object per chat session to a file:
//!
//! ```json
//! {"session": "chat_session:abc", "messages": [{"role": "human", "content": "..."},
//!  {"role": "ai", "content": "..."}], "context_indicators": {"sources": ["source:x"]}}
//! ```
//!
//! [`import_chat_history`] writes those messages into `chat_message`. It is
//! safe to run more than once: a session that already has messages is left
//! alone. Sessions that no longer exist are reported and skipped. Session
//! `updated` times are not touched, so the session lists keep their order.

use std::io::BufRead;

use serde::{Deserialize, Serialize};
use serde_json::Value as Json;

use crate::db::Doc;
use crate::domain::chat::{ChatSession, StoredMessage};
use crate::error::{AppError, AppResult};
use crate::services::chat::normalize_record_id;
use crate::state::AppState;

/// The field a source chat's latest answer carries its context indicators in.
const INDICATORS_FIELD: &str = "context_indicators";

/// One exported message.
#[derive(Debug, Clone, Deserialize)]
pub struct ExportedMessage {
    pub role: String,
    #[serde(default)]
    pub content: String,
}

/// One exported chat session (a line of the export file).
#[derive(Debug, Clone, Deserialize)]
pub struct ExportedSession {
    pub session: String,
    #[serde(default)]
    pub messages: Vec<ExportedMessage>,
    /// A source chat's indicators (`{"sources": [...], "insights": [...], "notes": [...]}`).
    #[serde(default)]
    pub context_indicators: Option<Json>,
}

/// What an import did (or, on a dry run, would do).
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize)]
pub struct ImportReport {
    /// Sessions whose messages were written.
    pub imported_sessions: usize,
    /// Messages written.
    pub imported_messages: usize,
    /// Sessions that already had messages in this database.
    pub skipped_existing: usize,
    /// Sessions with no human or AI messages in the export.
    pub skipped_empty: usize,
    /// Sessions in the export that do not exist in this database.
    pub missing_sessions: Vec<String>,
    /// Messages of other kinds (system, tool) that were left out.
    pub ignored_messages: usize,
}

/// Import every session in `export` (JSON lines). With `dry_run` nothing is
/// written, but the report says what would be.
pub async fn import_chat_history(state: &AppState, export: impl BufRead, dry_run: bool) -> AppResult<ImportReport> {
    let mut report = ImportReport::default();
    for (index, line) in export.lines().enumerate() {
        let line = line.map_err(|e| AppError::FileOperation(format!("Could not read the export: {e}")))?;
        if line.trim().is_empty() {
            continue;
        }
        let session: ExportedSession = serde_json::from_str(&line)
            .map_err(|e| AppError::invalid(format!("Line {} of the export is not a chat session: {e}", index + 1)))?;
        import_session(state, session, dry_run, &mut report).await?;
    }
    Ok(report)
}

async fn import_session(
    state: &AppState,
    exported: ExportedSession,
    dry_run: bool,
    report: &mut ImportReport,
) -> AppResult<()> {
    let session_id = normalize_record_id(ChatSession::TABLE, exported.session.trim());
    match ChatSession::get(state, &session_id).await {
        Ok(_) => {}
        Err(e) if e.is_not_found() => {
            report.missing_sessions.push(session_id);
            return Ok(());
        }
        Err(e) => return Err(e),
    }
    if StoredMessage::count(state, &session_id).await? > 0 {
        report.skipped_existing += 1;
        return Ok(());
    }

    let total = exported.messages.len();
    let messages: Vec<&ExportedMessage> =
        exported.messages.iter().filter(|m| m.role == StoredMessage::HUMAN || m.role == StoredMessage::AI).collect();
    report.ignored_messages += total - messages.len();
    if messages.is_empty() {
        report.skipped_empty += 1;
        return Ok(());
    }

    // The indicators belong on the latest answer, where the source chat reads them.
    let last_ai = messages.iter().rposition(|m| m.role == StoredMessage::AI);
    let indicators = exported.context_indicators.as_ref().and_then(clean_indicators);
    if !dry_run {
        for (seq, message) in messages.iter().enumerate() {
            let mut doc = Doc::new()
                .record("session", &session_id)
                .set("role", message.role.as_str())
                .set("content", message.content.as_str())
                .set("seq", seq as i64)
                .set("imported_from", "langgraph");
            if Some(seq) == last_ai {
                if let Some(indicators) = &indicators {
                    doc = doc.set(INDICATORS_FIELD, indicators.clone());
                }
            }
            state.db.create("chat_message", doc).await?;
        }
    }
    report.imported_sessions += 1;
    report.imported_messages += messages.len();
    Ok(())
}

/// Python's indicator lists may hold `None` entries; keep the strings only.
fn clean_indicators(value: &Json) -> Option<Json> {
    let object = value.as_object()?;
    let mut clean = serde_json::Map::new();
    for key in ["sources", "insights", "notes"] {
        let ids: Vec<Json> = object
            .get(key)
            .and_then(Json::as_array)
            .map(|items| items.iter().filter(|v| v.as_str().is_some_and(|s| !s.is_empty())).cloned().collect())
            .unwrap_or_default();
        clean.insert(key.to_string(), Json::Array(ids));
    }
    Some(Json::Object(clean))
}

#[cfg(test)]
mod tests {
    use axum::http::StatusCode;
    use serde_json::json;

    use super::*;
    use crate::testing::TestApp;

    async fn notebook_session(app: &TestApp) -> String {
        let (_, nb) = app.post("/api/notebooks", json!({ "name": "N" })).await;
        let (status, session) =
            app.post("/api/chat/sessions", json!({ "notebook_id": nb["id"], "title": "Old chat" })).await;
        assert_eq!(status, StatusCode::OK, "{session}");
        session["id"].as_str().unwrap().to_string()
    }

    fn export(lines: &[Json]) -> std::io::Cursor<Vec<u8>> {
        let text: Vec<String> = lines.iter().map(|l| l.to_string()).collect();
        std::io::Cursor::new(format!("{}\n\n", text.join("\n")).into_bytes())
    }

    #[tokio::test]
    async fn imports_notebook_chat_history_once() {
        let app = TestApp::new().await;
        let session = notebook_session(&app).await;
        let (_, before) = app.get(&format!("/api/chat/sessions/{session}")).await;
        let updated_before = before["updated"].clone();
        let bare = session.trim_start_matches("chat_session:");
        let file = [json!({
            "session": bare,
            "messages": [
                { "role": "human", "content": "What is a heat pump?" },
                { "role": "system", "content": "never stored" },
                { "role": "ai", "content": "A device that moves heat." },
                { "role": "human", "content": "Thanks" },
                { "role": "ai", "content": "You're welcome." }
            ]
        })];

        let dry = import_chat_history(&app.state, export(&file), true).await.unwrap();
        assert_eq!((dry.imported_sessions, dry.imported_messages, dry.ignored_messages), (1, 4, 1));
        assert_eq!(StoredMessage::count(&app.state, &session).await.unwrap(), 0, "a dry run writes nothing");

        let report = import_chat_history(&app.state, export(&file), false).await.unwrap();
        assert_eq!((report.imported_sessions, report.imported_messages), (1, 4));

        // The API shows the history in order, and the session keeps its updated time.
        let (_, after) = app.get(&format!("/api/chat/sessions/{session}")).await;
        let messages = after["messages"].as_array().unwrap();
        let shown: Vec<(&str, &str)> =
            messages.iter().map(|m| (m["type"].as_str().unwrap(), m["content"].as_str().unwrap())).collect();
        assert_eq!(
            shown,
            [("human", "What is a heat pump?"), ("ai", "A device that moves heat."), ("human", "Thanks"), ("ai", "You're welcome.")]
        );
        assert_eq!(after["message_count"], 4);
        assert_eq!(after["updated"], updated_before);

        // A second run leaves the session alone.
        let again = import_chat_history(&app.state, export(&file), false).await.unwrap();
        assert_eq!((again.imported_sessions, again.skipped_existing), (0, 1));
        assert_eq!(StoredMessage::count(&app.state, &session).await.unwrap(), 4);
    }

    #[tokio::test]
    async fn source_chat_gets_its_context_indicators() {
        let app = TestApp::new().await;
        let (_, source) = app
            .post("/api/sources/json", json!({ "type": "text", "content": "Heat pumps move heat.", "title": "S", "async_processing": false }))
            .await;
        let source_id = source["id"].as_str().unwrap().to_string();
        let (status, session) =
            app.post(&format!("/api/sources/{source_id}/chat/sessions"), json!({ "source_id": source_id, "title": "Ask" })).await;
        assert_eq!(status, StatusCode::OK, "{session}");
        let session_id = session["id"].as_str().unwrap().to_string();

        let file = [json!({
            "session": session_id,
            "messages": [{ "role": "human", "content": "Summarise" }, { "role": "ai", "content": "It moves heat." }],
            "context_indicators": { "sources": [source_id, null], "insights": [], "notes": null }
        })];
        let report = import_chat_history(&app.state, export(&file), false).await.unwrap();
        assert_eq!(report.imported_messages, 2);

        let (status, body) = app.get(&format!("/api/sources/{source_id}/chat/sessions/{session_id}")).await;
        assert_eq!(status, StatusCode::OK, "{body}");
        assert_eq!(body["messages"].as_array().unwrap().len(), 2);
        assert_eq!(body["context_indicators"]["sources"], json!([source_id]));
        assert_eq!(body["context_indicators"]["notes"], json!([]));
    }

    #[tokio::test]
    async fn missing_and_empty_sessions_are_reported_and_bad_lines_refused() {
        let app = TestApp::new().await;
        let session = notebook_session(&app).await;
        let file = [
            json!({ "session": "chat_session:gone", "messages": [{ "role": "human", "content": "hi" }] }),
            json!({ "session": session, "messages": [{ "role": "tool", "content": "x" }] }),
        ];
        let report = import_chat_history(&app.state, export(&file), false).await.unwrap();
        assert_eq!(report.missing_sessions, vec!["chat_session:gone".to_string()]);
        assert_eq!((report.skipped_empty, report.ignored_messages, report.imported_sessions), (1, 1, 0));

        let broken = std::io::Cursor::new(b"{\"session\": \"x\"}\nnot json\n".to_vec());
        let error = import_chat_history(&app.state, broken, true).await.unwrap_err();
        assert!(error.to_string().contains("Line 2"), "{error}");
    }
}
