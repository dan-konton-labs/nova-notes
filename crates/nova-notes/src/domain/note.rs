//! Notes, and the files that can be attached to them.

use std::path::PathBuf;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::json;

use super::Notebook;
use crate::config::Config;
use crate::db::{retry_conflicts, Doc, Json, Params};
use crate::error::{AppError, AppResult};
use crate::state::AppState;

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Note {
    #[serde(default)]
    pub id: String,
    #[serde(default)]
    pub title: Option<String>,
    /// `"human"` or `"ai"`.
    #[serde(default)]
    pub note_type: Option<String>,
    #[serde(default)]
    pub content: Option<String>,
    /// A blank note opened in the editor that has yet to receive a save.
    #[serde(default, deserialize_with = "super::null_default")]
    pub untouched_draft: bool,
    /// The user's file name of an attached original (fork: dropped files).
    #[serde(default)]
    pub attachment_name: Option<String>,
    /// Stored file name under the note files folder.
    #[serde(default)]
    pub attachment_file: Option<String>,
    #[serde(default)]
    pub attachment_mime: Option<String>,
    #[serde(default)]
    pub attachment_size: Option<i64>,
    /// True when the note is a generated explainer (fork).
    #[serde(default)]
    pub explainer: Option<bool>,
    /// The meeting this note is the transcript of (fork).
    #[serde(default)]
    pub meeting_id: Option<String>,
    #[serde(default)]
    pub created: Option<DateTime<Utc>>,
    #[serde(default)]
    pub updated: Option<DateTime<Utc>>,
}

impl Note {
    pub const TABLE: &'static str = "note";

    /// Missing → `NotFound`. The embedding vector is not loaded.
    pub async fn get(state: &AppState, id: &str) -> AppResult<Self> {
        let rows: Vec<Self> = state.db.query_as("SELECT * OMIT embedding, doc_state FROM $id", Params::new().record("id", id)).await?;
        rows.into_iter().next().ok_or_else(|| AppError::not_found(format!("note with id {id} not found")))
    }

    /// Visible notes, most recently updated first.
    pub async fn list(state: &AppState) -> AppResult<Vec<Self>> {
        state.db.query_as("SELECT * OMIT embedding, doc_state FROM note WHERE untouched_draft != true ORDER BY updated DESC", Params::new()).await
    }

    fn validate(&self) -> AppResult<()> {
        match self.content.as_deref() {
            Some(content) if content.trim().is_empty() => Err(AppError::invalid("Note content cannot be empty")),
            _ => Ok(()),
        }
    }

    fn doc(&self) -> Doc {
        let has_text = [self.title.as_deref(), self.content.as_deref()]
            .into_iter()
            .flatten()
            .any(|text| !text.trim().is_empty());
        Doc::new()
            .set_some("title", self.title.as_ref())
            .set_some("note_type", self.note_type.as_ref())
            .set_some("content", self.content.as_ref())
            .set_some("untouched_draft", (self.untouched_draft && has_text).then_some(false))
            .set_some("attachment_name", self.attachment_name.as_ref())
            .set_some("attachment_file", self.attachment_file.as_ref())
            .set_some("attachment_mime", self.attachment_mime.as_ref())
            .set_some("attachment_size", self.attachment_size)
            .set_some("explainer", self.explainer)
            .set_some("meeting_id", self.meeting_id.as_ref())
    }

    /// Create a note and queue its embedding.
    pub async fn create(state: &AppState, title: Option<&str>, content: Option<&str>, note_type: Option<&str>) -> AppResult<Self> {
        let mut note = Note {
            title: title.map(String::from),
            content: content.map(String::from),
            note_type: note_type.map(String::from),
            ..Note::default()
        };
        note.save(state).await?;
        Ok(note)
    }

    /// Create (when `id` is empty) or update the note, then queue the
    /// `embed_note` job if it has content. Returns that job's command id.
    pub async fn save(&mut self, state: &AppState) -> AppResult<Option<String>> {
        self.validate()?;
        let row = if self.id.is_empty() {
            state.db.create(Self::TABLE, self.doc()).await?
        } else {
            // An `embed_note` job may be writing this note's embedding.
            let _write = state.db.lock_record(&self.id).await;
            state.db.update(&self.id, self.doc()).await?
        };
        let mut saved: Note = serde_json::from_value(strip_embedding(row))?;
        std::mem::swap(self, &mut saved);

        Ok(self.queue_embedding(state).await)
    }

    async fn queue_embedding(&self, state: &AppState) -> Option<String> {
        if self.content.as_deref().is_some_and(|c| !c.trim().is_empty()) {
            match state.jobs.submit("embed_note", json!({ "note_id": self.id })).await {
                Ok(command_id) => return Some(command_id),
                Err(e) => tracing::error!("Failed to submit embed_note command for {}: {e}", self.id),
            }
        }
        None
    }

    /// File the note in a notebook. Missing notebook → `NotFound`.
    pub async fn add_to_notebook(&self, state: &AppState, notebook_id: &str) -> AppResult<()> {
        if notebook_id.is_empty() {
            return Err(AppError::invalid("Notebook ID must be provided"));
        }
        Notebook::get(state, notebook_id).await?;
        state.db.relate(&self.id, "artifact", notebook_id, Doc::new()).await?;
        Ok(())
    }

    /// Ids of the notebooks holding this note.
    pub async fn notebook_ids(&self, state: &AppState) -> AppResult<Vec<String>> {
        let rows = state
            .db
            .query("SELECT VALUE out FROM artifact WHERE in = $id", Params::new().record("id", &self.id))
            .await?;
        Ok(rows.iter().filter_map(Json::as_str).map(String::from).collect())
    }

    /// The note as model context: `{id, title, content}`, with the content cut
    /// to 100 characters for the short form.
    pub fn context(&self, long: bool) -> Json {
        let content = match (&self.content, long) {
            (Some(content), false) => Some(content.chars().take(100).collect::<String>()),
            (content, _) => content.clone(),
        };
        json!({ "id": self.id, "title": self.title, "content": content })
    }

    /// Delete the note, the meeting it transcribes (if any) and its attachment.
    ///
    /// Every step is a delete, so a step that hits a transaction conflict (an
    /// `embed_note` job writing the note's embedding at that moment) is simply
    /// run again rather than failing the request.
    pub async fn delete(&self, state: &AppState) -> AppResult<()> {
        let _write = state.db.lock_record(&self.id).await;
        retry_conflicts(|| state.db.query("DELETE artifact WHERE in = $id", Params::new().record("id", &self.id))).await?;
        retry_conflicts(|| state.db.delete(&self.id)).await?;
        if let Some(meeting_id) = self.meeting_id.as_deref().filter(|m| !m.is_empty()) {
            retry_conflicts(|| {
                state.db.query_all(
                    "BEGIN TRANSACTION; DELETE meeting_chunk WHERE meeting_id = $meeting_id; DELETE $id; COMMIT TRANSACTION;",
                    Params::new().record("id", meeting_id).set("meeting_id", meeting_id),
                )
            })
            .await?;
        }
        if let Some(stored) = self.attachment_file.as_deref() {
            remove_note_file(&state.config, stored);
        }
        Ok(())
    }
}

fn strip_embedding(mut row: Json) -> Json {
    if let Some(map) = row.as_object_mut() {
        map.remove("embedding");
    }
    row
}

/// A stored attachment name is `<32 hex chars>.<2-8 lowercase letters>`. The
/// user's own file name is never used as a path.
pub fn is_stored_note_file_name(stored: &str) -> bool {
    let Some((stem, ext)) = stored.split_once('.') else {
        return false;
    };
    stem.len() == 32
        && stem.chars().all(|c| c.is_ascii_digit() || ('a'..='f').contains(&c))
        && (2..=8).contains(&ext.len())
        && ext.chars().all(|c| c.is_ascii_lowercase())
}

/// Resolve a stored attachment name to its path inside the note files folder.
pub fn note_file_path(config: &Config, stored: &str) -> AppResult<PathBuf> {
    if !is_stored_note_file_name(stored) {
        return Err(AppError::invalid("Invalid stored file name"));
    }
    Ok(config.note_files_dir().join(stored))
}

/// Best-effort removal; a missing file or bad name is only logged.
pub fn remove_note_file(config: &Config, stored: &str) {
    match note_file_path(config, stored) {
        Ok(path) => {
            if let Err(e) = std::fs::remove_file(&path) {
                if e.kind() != std::io::ErrorKind::NotFound {
                    tracing::warn!("Failed to remove note file {stored}: {e}");
                }
            }
        }
        Err(e) => tracing::warn!("Failed to remove note file {stored}: {e}"),
    }
}

/// The rich editor's copy of a note (rust migration 2): the BlockSuite
/// document, the content it was saved with, its mode and its links.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct NoteDoc {
    /// Base64 Yjs update.
    #[serde(default)]
    pub doc_state: Option<String>,
    /// SHA-256 (hex) of `content` when `doc_state` was saved.
    #[serde(default)]
    pub doc_hash: Option<String>,
    /// `"page"` or `"edgeless"`.
    #[serde(default)]
    pub doc_mode: Option<String>,
}

/// A note named in a link list: its id and title.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct NoteRef {
    pub id: String,
    #[serde(default)]
    pub title: Option<String>,
}

impl Note {
    /// Create and file a blank human note in one write.
    pub async fn create_draft(state: &AppState, notebook_id: &str) -> AppResult<String> {
        Notebook::get(state, notebook_id).await?;
        let id = format!("note:{}", uuid::Uuid::new_v4().simple());
        state
            .db
            .query_all(
                "BEGIN TRANSACTION; CREATE $id SET note_type = 'human', untouched_draft = true; \
                 RELATE $id->artifact->$notebook_id; COMMIT TRANSACTION;",
                Params::new().record("id", &id).record("notebook_id", notebook_id),
            )
            .await?;
        Ok(id)
    }

    /// The editor document of a note (all fields `None` for a note never
    /// opened in the rich editor). Missing note → `NotFound`.
    pub async fn doc_state(state: &AppState, id: &str) -> AppResult<NoteDoc> {
        let rows: Vec<NoteDoc> =
            state.db.query_as("SELECT doc_state, doc_hash, doc_mode FROM $id", Params::new().record("id", id)).await?;
        rows.into_iter().next().ok_or_else(|| AppError::not_found(format!("note with id {id} not found")))
    }

    /// Store a complete editor snapshot atomically. Only changed text needs a new embedding.
    pub async fn save_doc_state(
        state: &AppState,
        id: &str,
        title: Option<&str>,
        content: Option<&str>,
        doc: &NoteDoc,
        links: &[String],
    ) -> AppResult<(Self, Option<String>)> {
        let write = state.db.lock_record(id).await;
        let changed = Self::get(state, id).await?.content.as_deref() != content;
        let rows = state
            .db
            .query_as::<Self>(
                "UPDATE $id SET title = $title, content = $markdown, doc_state = $doc_state, doc_hash = $doc_hash, \
                 doc_mode = $doc_mode, links = $links, untouched_draft = NONE, \
                 embedding = IF $markdown = NONE THEN NONE ELSE embedding END \
                 RETURN id, title, content, note_type, created, updated, attachment_name, attachment_file, \
                 attachment_mime, attachment_size, explainer, meeting_id",
                Params::new()
                    .record("id", id)
                    .set("title", title)
                    .set("markdown", content)
                    .set("doc_state", &doc.doc_state)
                    .set("doc_hash", &doc.doc_hash)
                    .set("doc_mode", &doc.doc_mode)
                    .records("links", links),
            )
            .await?;
        drop(write);
        let note = rows.into_iter().next().ok_or_else(|| AppError::not_found("Note not found"))?;
        let command = if changed { note.queue_embedding(state).await } else { None };
        Ok((note, command))
    }

    /// The ids among `ids` that are existing notes.
    pub async fn existing_ids(state: &AppState, ids: &[String]) -> AppResult<Vec<String>> {
        if ids.is_empty() {
            return Ok(Vec::new());
        }
        let rows = state.db.query("SELECT VALUE id FROM note WHERE id IN $ids", Params::new().records("ids", ids)).await?;
        Ok(rows.iter().filter_map(Json::as_str).map(String::from).collect())
    }

    /// Notes linking to `id`, most recently updated first.
    pub async fn backlinks(state: &AppState, id: &str) -> AppResult<Vec<NoteRef>> {
        state
            .db
            .query_as(
                "SELECT id, title, updated FROM note WHERE links CONTAINS $id AND id != $id ORDER BY updated DESC",
                Params::new().record("id", id),
            )
            .await
    }
}
