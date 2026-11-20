//! Meeting drafts (`meeting`) and their transcribed chunks (`meeting_chunk`).
//!
//! The Rust counterpart of `open_notebook/domain/meeting.py`. Drafts are kept
//! apart from notes, search and embeddings until they are approved. Both
//! tables are SCHEMALESS (migration 105); `notebook_id`, `note_id` and
//! `meeting_chunk.meeting_id` are plain `"table:id"` strings, not record links.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use crate::db::{Doc, Params};
use crate::domain::null_default;
use crate::error::{AppError, AppResult};
use crate::state::AppState;

/// Four hours at twenty seconds per chunk.
pub const MAX_CHUNKS: i64 = 720;
/// Longest transcript a meeting may hold, in characters.
pub const MAX_TRANSCRIPT: usize = 400_000;
/// Longest summary, in characters.
pub const MAX_SUMMARY: usize = 30_000;
/// Most actions a meeting may hold.
pub const MAX_ACTIONS: usize = 100;
/// Longest title, in characters.
pub const MAX_TITLE: usize = 200;
/// Seconds of audio in one chunk; used for the `[mm:ss]` stamps.
pub const CHUNK_SECONDS: i64 = 20;

/// A request field that breaks a Pydantic constraint of the Python models.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FieldError {
    pub field: &'static str,
    /// Pydantic's error type (`string_too_short`, `less_than_equal`, ...).
    pub kind: &'static str,
    pub message: String,
}

impl FieldError {
    pub fn new(field: &'static str, kind: &'static str, message: impl Into<String>) -> Self {
        Self { field, kind, message: message.into() }
    }
}

impl From<FieldError> for AppError {
    /// For callers outside the HTTP layer: a 422 naming the field.
    fn from(e: FieldError) -> Self {
        AppError::unprocessable(format!("{}: {}", e.field, e.message))
    }
}

/// One action item: a commitment backed by a quotation from the transcript.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct MeetingAction {
    pub text: String,
    #[serde(default)]
    pub owner: String,
    #[serde(default)]
    pub due: String,
    pub evidence: String,
    #[serde(default)]
    pub done: bool,
}

impl MeetingAction {
    /// Python's field constraints: `text` 1..=2000, `owner`/`due` <= 200,
    /// `evidence` 1..=3000 characters.
    pub fn check(&self) -> Result<(), FieldError> {
        check_len("text", &self.text, 1, 2000)?;
        check_len("owner", &self.owner, 0, 200)?;
        check_len("due", &self.due, 0, 200)?;
        check_len("evidence", &self.evidence, 1, 3000)
    }
}

/// What the summary model returns.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct MeetingSummary {
    pub summary: String,
    #[serde(default)]
    pub actions: Vec<MeetingAction>,
}

impl MeetingSummary {
    pub fn check(&self) -> Result<(), FieldError> {
        check_len("summary", &self.summary, 1, MAX_SUMMARY)?;
        check_actions(&self.actions)
    }
}

/// Pydantic's string length constraint, counted in characters.
pub fn check_len(field: &'static str, value: &str, min: usize, max: usize) -> Result<(), FieldError> {
    let len = value.chars().count();
    if len < min {
        let unit = if min == 1 { "character" } else { "characters" };
        return Err(FieldError::new(field, "string_too_short", format!("String should have at least {min} {unit}")));
    }
    if len > max {
        return Err(FieldError::new(field, "string_too_long", format!("String should have at most {max} characters")));
    }
    Ok(())
}

/// At most [`MAX_ACTIONS`] actions, each valid.
pub fn check_actions(actions: &[MeetingAction]) -> Result<(), FieldError> {
    if actions.len() > MAX_ACTIONS {
        return Err(FieldError::new(
            "actions",
            "too_long",
            format!("List should have at most {MAX_ACTIONS} items after validation"),
        ));
    }
    actions.iter().try_for_each(MeetingAction::check)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Meeting {
    #[serde(default)]
    pub id: String,
    #[serde(default, deserialize_with = "null_default")]
    pub notebook_id: String,
    #[serde(default, deserialize_with = "null_default")]
    pub title: String,
    #[serde(default = "default_capture_mode", deserialize_with = "capture_mode_or_default")]
    pub capture_mode: String,
    #[serde(default = "default_status", deserialize_with = "status_or_default")]
    pub status: String,
    #[serde(default, deserialize_with = "null_default")]
    pub transcript: String,
    #[serde(default, deserialize_with = "null_default")]
    pub summary: String,
    #[serde(default, deserialize_with = "null_default")]
    pub actions: Vec<MeetingAction>,
    #[serde(default, deserialize_with = "null_default")]
    pub total_chunks: i64,
    #[serde(default, deserialize_with = "null_default")]
    pub missing_chunks: Vec<i64>,
    #[serde(default, deserialize_with = "null_default")]
    pub revision: i64,
    #[serde(default, deserialize_with = "null_default")]
    pub interrupted: bool,
    #[serde(default, deserialize_with = "null_default")]
    pub summary_token: String,
    #[serde(default, deserialize_with = "null_default")]
    pub summary_job: String,
    #[serde(default, deserialize_with = "null_default")]
    pub summary_error: String,
    #[serde(default, deserialize_with = "null_default")]
    pub note_id: String,
    #[serde(default)]
    pub created: Option<DateTime<Utc>>,
    #[serde(default)]
    pub updated: Option<DateTime<Utc>>,
}

fn default_capture_mode() -> String {
    "microphone".into()
}

fn default_status() -> String {
    "recording".into()
}

fn capture_mode_or_default<'de, D: serde::Deserializer<'de>>(d: D) -> Result<String, D::Error> {
    Ok(Option::<String>::deserialize(d)?.unwrap_or_else(default_capture_mode))
}

fn status_or_default<'de, D: serde::Deserializer<'de>>(d: D) -> Result<String, D::Error> {
    Ok(Option::<String>::deserialize(d)?.unwrap_or_else(default_status))
}

/// One transcribed chunk.
#[derive(Debug, Clone, Deserialize)]
pub struct MeetingChunk {
    pub sequence: i64,
    #[serde(default, deserialize_with = "null_default")]
    pub text: String,
}

/// `[mm:ss]` for the start of chunk `sequence` (Python: `seq*20//60`, `seq*20%60`).
pub fn chunk_stamp(sequence: i64) -> String {
    let seconds = sequence * CHUNK_SECONDS;
    format!("[{:02}:{:02}]", seconds / 60, seconds % 60)
}

impl Meeting {
    pub const TABLE: &'static str = "meeting";

    /// A new draft in the `recording` state.
    pub fn new(notebook_id: &str, title: &str, capture_mode: &str) -> Self {
        Self {
            id: String::new(),
            notebook_id: notebook_id.to_string(),
            title: title.to_string(),
            capture_mode: capture_mode.to_string(),
            status: default_status(),
            transcript: String::new(),
            summary: String::new(),
            actions: Vec::new(),
            total_chunks: 0,
            missing_chunks: Vec::new(),
            revision: 0,
            interrupted: false,
            summary_token: String::new(),
            summary_job: String::new(),
            summary_error: String::new(),
            note_id: String::new(),
            created: None,
            updated: None,
        }
    }

    /// Missing → `NotFound` (`"meeting with id ... not found"`).
    pub async fn get(state: &AppState, id: &str) -> AppResult<Self> {
        if id.is_empty() {
            return Err(AppError::invalid("ID cannot be empty"));
        }
        state.db.get(id).await
    }

    /// Insert this draft and return it as stored (with id and timestamps).
    pub async fn create(&self, state: &AppState) -> AppResult<Self> {
        let doc = Doc::from(self).remove("created").remove("updated");
        let row = state.db.create(Self::TABLE, doc).await?;
        Ok(serde_json::from_value(row)?)
    }

    /// The notebook's meetings, newest first.
    pub async fn list_for_notebook(state: &AppState, notebook_id: &str) -> AppResult<Vec<Self>> {
        state
            .db
            .query_as(
                "SELECT * FROM meeting WHERE notebook_id = $notebook_id ORDER BY created DESC",
                Params::new().set("notebook_id", notebook_id),
            )
            .await
    }

    /// Transcribed chunks in sequence order.
    pub async fn chunks(state: &AppState, meeting_id: &str) -> AppResult<Vec<MeetingChunk>> {
        state
            .db
            .query_as(
                "SELECT sequence, text FROM meeting_chunk WHERE meeting_id = $meeting_id ORDER BY sequence",
                Params::new().set("meeting_id", meeting_id),
            )
            .await
    }

    /// The stored text of one chunk, if it was already transcribed.
    pub async fn chunk_text(state: &AppState, meeting_id: &str, sequence: i64) -> AppResult<Option<String>> {
        let rows: Vec<MeetingChunk> = state
            .db
            .query_as(
                "SELECT sequence, text FROM meeting_chunk WHERE meeting_id = $meeting_id AND sequence = $sequence",
                Params::new().set("meeting_id", meeting_id).set("sequence", sequence),
            )
            .await?;
        Ok(rows.into_iter().next().map(|c| c.text))
    }

    /// The note body an approved meeting publishes.
    pub fn content(&self) -> String {
        let actions: Vec<String> = self
            .actions
            .iter()
            .map(|a| {
                let mut line = format!("- [{}] {}", if a.done { 'x' } else { ' ' }, a.text);
                if !a.owner.is_empty() {
                    line.push_str(&format!(" — {}", a.owner));
                }
                if !a.due.is_empty() {
                    line.push_str(&format!(" — {}", a.due));
                }
                line.push_str(&format!("\n  Evidence: {}", a.evidence));
                line
            })
            .collect();
        format!(
            "## Summary\n{}\n\n## Actions\n{}\n\n## Transcript\n{}",
            self.summary,
            actions.join("\n"),
            self.transcript
        )
    }

    /// The record key without the table (`meeting:abc` → `abc`).
    pub fn key(&self) -> &str {
        self.id.split_once(':').map_or(self.id.as_str(), |(_, key)| key)
    }
}
