//! Chat sessions and their message history.
//!
//! The Python build keeps messages in a LangGraph SQLite checkpoint keyed by
//! session id. This build stores them in the `chat_message` table, one row
//! per message, so history lives in the same database as everything else.
//! (History written by the Python build is not imported.)

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use super::link_id;
use crate::db::{Doc, Json, Params};
use crate::error::{AppError, AppResult};
use crate::state::AppState;

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ChatSession {
    #[serde(default)]
    pub id: String,
    #[serde(default)]
    pub title: Option<String>,
    /// Model id overriding the default chat model for this session.
    #[serde(default)]
    pub model_override: Option<String>,
    #[serde(default)]
    pub created: Option<DateTime<Utc>>,
    #[serde(default)]
    pub updated: Option<DateTime<Utc>>,
}

impl ChatSession {
    pub const TABLE: &'static str = "chat_session";

    /// Missing → `NotFound`.
    pub async fn get(state: &AppState, id: &str) -> AppResult<Self> {
        state.db.get(id).await
    }

    pub async fn create(state: &AppState, title: Option<&str>, model_override: Option<&str>) -> AppResult<Self> {
        let row = state
            .db
            .create(Self::TABLE, Doc::new().set_some("title", title).set("model_override", model_override))
            .await?;
        Ok(serde_json::from_value(row)?)
    }

    /// Write `title` and `model_override` back (a `None` override clears it).
    pub async fn save(&mut self, state: &AppState) -> AppResult<()> {
        let doc = Doc::new().set_some("title", self.title.as_ref()).set("model_override", &self.model_override);
        let row = state.db.update(&self.id, doc).await?;
        *self = serde_json::from_value(row)?;
        Ok(())
    }

    /// Attach the session to a notebook or a source.
    pub async fn relate_to(&self, state: &AppState, target_id: &str) -> AppResult<()> {
        if target_id.is_empty() {
            return Err(AppError::invalid("Notebook ID must be provided"));
        }
        state.db.relate(&self.id, "refers_to", target_id, Doc::new()).await?;
        Ok(())
    }

    /// Sessions attached to a notebook or source, most recently updated first.
    pub async fn for_target(state: &AppState, target_id: &str) -> AppResult<Vec<Self>> {
        let rows = state
            .db
            .query(
                "SELECT in AS chat_session FROM refers_to WHERE out = $id FETCH chat_session",
                Params::new().record("id", target_id),
            )
            .await?;
        let mut sessions: Vec<Self> = super::notebook::fetched(rows, "chat_session");
        sessions.sort_by_key(|s| std::cmp::Reverse(s.updated));
        Ok(sessions)
    }

    /// Whether this session is attached to the given notebook or source.
    pub async fn belongs_to(&self, state: &AppState, target_id: &str) -> AppResult<bool> {
        let rows = state
            .db
            .query(
                "SELECT id FROM refers_to WHERE in = $sid AND out = $target LIMIT 1",
                Params::new().record("sid", &self.id).record("target", target_id),
            )
            .await?;
        Ok(!rows.is_empty())
    }

    /// Ids of the notebooks / sources this session is attached to.
    pub async fn targets(&self, state: &AppState) -> AppResult<Vec<String>> {
        let rows = state
            .db
            .query("SELECT VALUE out FROM refers_to WHERE in = $sid", Params::new().record("sid", &self.id))
            .await?;
        Ok(rows.iter().filter_map(Json::as_str).map(String::from).collect())
    }

    pub async fn messages(&self, state: &AppState) -> AppResult<Vec<StoredMessage>> {
        StoredMessage::history(state, &self.id).await
    }

    /// Delete the session, its links and its messages.
    pub async fn delete(&self, state: &AppState) -> AppResult<()> {
        state
            .db
            .query_all(
                "DELETE chat_message WHERE session = $id; DELETE refers_to WHERE in = $id;",
                Params::new().record("id", &self.id),
            )
            .await?;
        state.db.delete(&self.id).await
    }
}

/// One message of a chat session. `role` is `"human"` or `"ai"`, the names the
/// API has always used for message `type`.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct StoredMessage {
    #[serde(default)]
    pub id: String,
    #[serde(default, deserialize_with = "link_id")]
    pub session: Option<String>,
    #[serde(default)]
    pub role: String,
    #[serde(default)]
    pub content: String,
    /// Position within the session, starting at 0.
    #[serde(default)]
    pub seq: i64,
    #[serde(default)]
    pub created: Option<DateTime<Utc>>,
}

impl StoredMessage {
    pub const HUMAN: &'static str = "human";
    pub const AI: &'static str = "ai";

    /// Every message of a session, oldest first.
    pub async fn history(state: &AppState, session_id: &str) -> AppResult<Vec<Self>> {
        state
            .db
            .query_as(
                "SELECT * FROM chat_message WHERE session = $sid ORDER BY seq ASC, created ASC",
                Params::new().record("sid", session_id),
            )
            .await
    }

    /// Number of messages in a session.
    pub async fn count(state: &AppState, session_id: &str) -> AppResult<i64> {
        let rows = state
            .db
            .query(
                "SELECT count() AS count FROM chat_message WHERE session = $sid GROUP ALL",
                Params::new().record("sid", session_id),
            )
            .await?;
        Ok(rows.first().and_then(|r| r.get("count")).and_then(Json::as_i64).unwrap_or(0))
    }

    /// Append a message and bump the session's `updated` time.
    pub async fn append(state: &AppState, session_id: &str, role: &str, content: &str) -> AppResult<Self> {
        let seq = Self::count(state, session_id).await?;
        let row = state
            .db
            .create(
                "chat_message",
                Doc::new().record("session", session_id).set("role", role).set("content", content).set("seq", seq),
            )
            .await?;
        state
            .db
            .query("UPDATE $id SET updated = time::now()", Params::new().record("id", session_id))
            .await?;
        Ok(serde_json::from_value(row)?)
    }

    /// Like [`StoredMessage::append`], with extra fields stored on the message
    /// row (the table is schemaless), e.g. a source chat's context indicators.
    pub async fn append_with(
        state: &AppState,
        session_id: &str,
        role: &str,
        content: &str,
        extra: &[(&str, Json)],
    ) -> AppResult<Self> {
        let seq = Self::count(state, session_id).await?;
        let mut doc = Doc::new().record("session", session_id).set("role", role).set("content", content).set("seq", seq);
        for (key, value) in extra {
            doc = doc.set(key, value);
        }
        let row = state.db.create("chat_message", doc).await?;
        state
            .db
            .query("UPDATE $id SET updated = time::now()", Params::new().record("id", session_id))
            .await?;
        Ok(serde_json::from_value(row)?)
    }

    /// The value of an extra `field` on the session's most recent message
    /// that has it, if any.
    pub async fn latest_value(state: &AppState, session_id: &str, field: &str) -> AppResult<Option<Json>> {
        let field = crate::db::ensure_safe_identifier(field, "field")?;
        let rows = state
            .db
            .query(
                &format!(
                    "SELECT {field}, seq FROM chat_message WHERE session = $sid AND {field} != NONE ORDER BY seq DESC LIMIT 1"
                ),
                Params::new().record("sid", session_id),
            )
            .await?;
        Ok(rows.into_iter().next().and_then(|mut row| row.get_mut(field).map(Json::take)).filter(|v| !v.is_null()))
    }

    /// Remove every message of a session.
    pub async fn clear(state: &AppState, session_id: &str) -> AppResult<()> {
        state
            .db
            .query("DELETE chat_message WHERE session = $sid", Params::new().record("sid", session_id))
            .await
            .map(|_| ())
    }
}
