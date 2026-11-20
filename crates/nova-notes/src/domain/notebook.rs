//! Notebooks and the folders / sections that organise them.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use super::{link_id, null_default, require_name, ChatSession, Note, Source, SourceInsight};
use crate::db::{Doc, Json, Params};
use crate::error::{AppError, AppResult};
use crate::state::AppState;

// ---------------------------------------------------------------------------
// Section / Folder (fork: sidebar organisation)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Section {
    #[serde(default)]
    pub id: String,
    #[serde(default, deserialize_with = "null_default")]
    pub name: String,
    #[serde(default)]
    pub created: Option<DateTime<Utc>>,
    #[serde(default)]
    pub updated: Option<DateTime<Utc>>,
}

impl Section {
    pub const TABLE: &'static str = "section";

    pub async fn get(state: &AppState, id: &str) -> AppResult<Self> {
        state.db.get(id).await
    }

    pub async fn list(state: &AppState) -> AppResult<Vec<Self>> {
        state.db.query_as("SELECT * FROM section ORDER BY name ASC", Params::new()).await
    }

    pub async fn create(state: &AppState, name: &str) -> AppResult<Self> {
        require_name("Section", name)?;
        let row = state.db.create(Self::TABLE, Doc::new().set("name", name)).await?;
        Ok(serde_json::from_value(row)?)
    }

    pub async fn rename(state: &AppState, id: &str, name: &str) -> AppResult<Self> {
        require_name("Section", name)?;
        let row = state.db.update(id, Doc::new().set("name", name)).await?;
        Ok(serde_json::from_value(row)?)
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Folder {
    #[serde(default)]
    pub id: String,
    #[serde(default, deserialize_with = "null_default")]
    pub name: String,
    #[serde(default, deserialize_with = "link_id")]
    pub section: Option<String>,
    #[serde(default)]
    pub created: Option<DateTime<Utc>>,
    #[serde(default)]
    pub updated: Option<DateTime<Utc>>,
}

impl Folder {
    pub const TABLE: &'static str = "folder";

    pub async fn get(state: &AppState, id: &str) -> AppResult<Self> {
        state.db.get(id).await
    }

    pub async fn list(state: &AppState) -> AppResult<Vec<Self>> {
        state.db.query_as("SELECT * FROM folder ORDER BY name ASC", Params::new()).await
    }

    pub async fn create(state: &AppState, name: &str, section: Option<&str>) -> AppResult<Self> {
        require_name("Folder", name)?;
        let mut doc = Doc::new().set("name", name);
        if let Some(section) = section {
            doc = doc.record("section", section);
        }
        let row = state.db.create(Self::TABLE, doc).await?;
        Ok(serde_json::from_value(row)?)
    }

    pub async fn rename(state: &AppState, id: &str, name: &str) -> AppResult<Self> {
        require_name("Folder", name)?;
        let row = state.db.update(id, Doc::new().set("name", name)).await?;
        Ok(serde_json::from_value(row)?)
    }

    /// Move the folder into a section, or out of any section with `None`.
    pub async fn set_section(state: &AppState, id: &str, section: Option<&str>) -> AppResult<Self> {
        let row = state.db.update(id, Doc::new().record_or_none("section", section)).await?;
        Ok(serde_json::from_value(row)?)
    }
}

// ---------------------------------------------------------------------------
// Notebook
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Notebook {
    #[serde(default)]
    pub id: String,
    #[serde(default, deserialize_with = "null_default")]
    pub name: String,
    #[serde(default, deserialize_with = "null_default")]
    pub description: String,
    #[serde(default, deserialize_with = "null_default")]
    pub archived: bool,
    #[serde(default)]
    pub last_viewed_at: Option<DateTime<Utc>>,
    #[serde(default, deserialize_with = "link_id")]
    pub folder: Option<String>,
    #[serde(default)]
    pub created: Option<DateTime<Utc>>,
    #[serde(default)]
    pub updated: Option<DateTime<Utc>>,
}

/// What deleting a notebook would remove.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct DeletePreview {
    pub note_count: i64,
    pub exclusive_source_count: i64,
    pub shared_source_count: i64,
}

/// What deleting a notebook did remove.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct DeleteSummary {
    pub deleted_notes: i64,
    pub deleted_sources: i64,
    pub unlinked_sources: i64,
    pub deleted_chat_sessions: i64,
}

/// For each source of the notebook: how many *other* notebooks also hold it.
const SOURCE_SHARING_QUERY: &str = "
    SELECT id, count(->reference[WHERE out != $notebook_id].out) AS assigned_others
    FROM (SELECT VALUE <-reference.in AS sources FROM $notebook_id)[0]";

impl Notebook {
    pub const TABLE: &'static str = "notebook";

    /// Missing → `NotFound`.
    pub async fn get(state: &AppState, id: &str) -> AppResult<Self> {
        state.db.get(id).await
    }

    /// Every notebook. `order_by` is validated against an allowlist.
    pub async fn list(state: &AppState, order_by: Option<&str>) -> AppResult<Vec<Self>> {
        let rows = state.db.select_all(Self::TABLE, order_by).await?;
        Ok(rows.into_iter().filter_map(|row| serde_json::from_value(row).ok()).collect())
    }

    pub async fn create(state: &AppState, name: &str, description: &str, folder: Option<&str>) -> AppResult<Self> {
        require_name("Notebook", name)?;
        let mut doc = Doc::new().set("name", name).set("description", description).set("archived", false);
        if let Some(folder) = folder {
            doc = doc.record("folder", folder);
        }
        let row = state.db.create(Self::TABLE, doc).await?;
        Ok(serde_json::from_value(row)?)
    }

    /// Write `name`, `description`, `archived` and `last_viewed_at` back.
    /// The folder is changed with [`Notebook::set_folder`].
    pub async fn save(&mut self, state: &AppState) -> AppResult<()> {
        require_name("Notebook", &self.name)?;
        let mut doc = Doc::new()
            .set("name", &self.name)
            .set("description", &self.description)
            .set("archived", self.archived);
        if let Some(viewed) = self.last_viewed_at {
            doc = doc.datetime("last_viewed_at", viewed);
        }
        let row = state.db.update(&self.id, doc).await?;
        *self = serde_json::from_value(row)?;
        Ok(())
    }

    /// Put the notebook in a folder, or take it out of its folder with `None`.
    pub async fn set_folder(state: &AppState, id: &str, folder: Option<&str>) -> AppResult<Self> {
        let row = state.db.update(id, Doc::new().record_or_none("folder", folder)).await?;
        Ok(serde_json::from_value(row)?)
    }

    /// Record that the notebook was just opened.
    pub async fn touch_viewed(state: &AppState, id: &str) -> AppResult<()> {
        state
            .db
            .query("UPDATE $id SET last_viewed_at = time::now()", Params::new().record("id", id))
            .await
            .map(|_| ())
    }

    /// The notebook's sources, most recently updated first.
    pub async fn sources(&self, state: &AppState, include_full_text: bool) -> AppResult<Vec<Source>> {
        let projection = if include_full_text { "" } else { " OMIT source.full_text" };
        let rows = state
            .db
            .query(
                &format!(
                    "SELECT *{projection} FROM (SELECT in AS source FROM reference WHERE out = $id FETCH source) ORDER BY source.updated DESC"
                ),
                Params::new().record("id", &self.id),
            )
            .await?;
        Ok(fetched(rows, "source"))
    }

    /// The notebook's visible notes, most recently updated first.
    pub async fn notes(&self, state: &AppState, include_content: bool) -> AppResult<Vec<Note>> {
        let mut notes = self.all_notes(state, include_content).await?;
        notes.retain(|note| !note.untouched_draft);
        Ok(notes)
    }

    async fn all_notes(&self, state: &AppState, include_content: bool) -> AppResult<Vec<Note>> {
        let projection = if include_content { " OMIT note.embedding, note.doc_state" } else { " OMIT note.content, note.embedding, note.doc_state" };
        let rows = state
            .db
            .query(
                &format!(
                    "SELECT *{projection} FROM (SELECT in AS note FROM artifact WHERE out = $id FETCH note) ORDER BY note.updated DESC"
                ),
                Params::new().record("id", &self.id),
            )
            .await?;
        Ok(fetched(rows, "note"))
    }

    /// Chat sessions attached to this notebook, most recently updated first.
    pub async fn chat_sessions(&self, state: &AppState) -> AppResult<Vec<ChatSession>> {
        ChatSession::for_target(state, &self.id).await
    }

    /// The whole notebook as one markdown document: every source (full text
    /// plus insights) and every note. Used as podcast / explainer input.
    pub async fn context(&self, state: &AppState) -> AppResult<String> {
        let sources = self.sources(state, true).await?;
        let notes = self.notes(state, true).await?;
        let source_ids: Vec<String> = sources.iter().map(|s| s.id.clone()).collect();
        let mut insights = SourceInsight::for_sources(state, &source_ids).await?;

        let mut blocks = Vec::new();
        for source in &sources {
            let title = source.title.as_deref().filter(|t| !t.is_empty()).unwrap_or("Untitled source");
            let mut parts = Vec::new();
            if let Some(text) = source.full_text.as_deref().filter(|t| !t.is_empty()) {
                parts.push(text.to_string());
            }
            let lines: Vec<String> = insights
                .remove(&source.id)
                .unwrap_or_default()
                .iter()
                .filter(|i| !i.content.is_empty())
                .map(|i| {
                    let kind = if i.insight_type.is_empty() { "Insight" } else { i.insight_type.as_str() };
                    format!("- {kind}: {}", i.content)
                })
                .collect();
            if !lines.is_empty() {
                parts.push(format!("Insights:\n{}", lines.join("\n")));
            }
            let content = parts.join("\n\n");
            let content = content.trim();
            if !content.is_empty() {
                blocks.push(format!("## Source: {title}\n\n{content}"));
            }
        }
        for note in &notes {
            let title = note.title.as_deref().filter(|t| !t.is_empty()).unwrap_or("Untitled note");
            let content = note.content.as_deref().unwrap_or("").trim();
            if !content.is_empty() {
                blocks.push(format!("## Note: {title}\n\n{content}"));
            }
        }
        Ok(blocks.join("\n\n"))
    }

    async fn source_sharing(&self, state: &AppState) -> AppResult<Vec<(String, i64)>> {
        let rows = state.db.query(SOURCE_SHARING_QUERY, Params::new().record("notebook_id", &self.id)).await?;
        Ok(rows
            .iter()
            .filter_map(|row| {
                let id = row.get("id")?.as_str()?.to_string();
                Some((id, row.get("assigned_others").and_then(Json::as_i64).unwrap_or(0)))
            })
            .collect())
    }

    async fn count(state: &AppState, edge: &str, notebook_id: &str) -> AppResult<i64> {
        let rows = state
            .db
            .query(
                &format!("SELECT count() AS count FROM {edge} WHERE out = $notebook_id GROUP ALL"),
                Params::new().record("notebook_id", notebook_id),
            )
            .await?;
        Ok(rows.first().and_then(|r| r.get("count")).and_then(Json::as_i64).unwrap_or(0))
    }

    pub async fn delete_preview(&self, state: &AppState) -> AppResult<DeletePreview> {
        let sharing = self.source_sharing(state).await?;
        let exclusive = sharing.iter().filter(|(_, others)| *others == 0).count() as i64;
        Ok(DeletePreview {
            note_count: Self::count(state, "artifact", &self.id).await?,
            exclusive_source_count: exclusive,
            shared_source_count: sharing.len() as i64 - exclusive,
        })
    }

    /// Delete the notebook with its notes, meetings and chat sessions. Sources
    /// are unlinked; with `delete_exclusive_sources`, those held by no other
    /// notebook are deleted too.
    pub async fn delete(&self, state: &AppState, delete_exclusive_sources: bool) -> AppResult<DeleteSummary> {
        if self.id.is_empty() {
            return Err(AppError::invalid("Cannot delete notebook without an ID"));
        }
        let mut summary = DeleteSummary::default();
        let by_notebook = || Params::new().record("notebook_id", &self.id);

        for note in self.all_notes(state, false).await? {
            note.delete(state).await?;
            summary.deleted_notes += 1;
        }

        // Meetings store the notebook id as a plain string.
        state
            .db
            .query_all(
                "DELETE meeting_chunk WHERE meeting_id IN (SELECT VALUE type::string(id) FROM meeting WHERE notebook_id = $notebook_id); \
                 DELETE meeting WHERE notebook_id = $notebook_id;",
                Params::new().set("notebook_id", &self.id),
            )
            .await?;
        state.db.query("DELETE artifact WHERE out = $notebook_id", by_notebook()).await?;

        if delete_exclusive_sources {
            for (source_id, others) in self.source_sharing(state).await? {
                if others > 0 {
                    summary.unlinked_sources += 1;
                    continue;
                }
                match Source::get(state, &source_id).await {
                    Ok(source) => match source.delete(state).await {
                        Ok(()) => summary.deleted_sources += 1,
                        Err(e) => tracing::warn!("Failed to delete exclusive source {source_id}: {e}"),
                    },
                    Err(e) => tracing::warn!("Failed to delete exclusive source {source_id}: {e}"),
                }
            }
        } else {
            summary.unlinked_sources = Self::count(state, "reference", &self.id).await?;
        }
        state.db.query("DELETE reference WHERE out = $notebook_id", by_notebook()).await?;

        for session in self.chat_sessions(state).await? {
            session.delete(state).await?;
            summary.deleted_chat_sessions += 1;
        }

        state.db.delete(&self.id).await?;
        tracing::info!("Deleted notebook {}: {summary:?}", self.id);
        Ok(summary)
    }
}

/// Pull the fetched record out of each `{ <key>: {...} }` row, skipping
/// dangling links (which come back as a bare id or null).
pub(crate) fn fetched<T: serde::de::DeserializeOwned>(rows: Vec<Json>, key: &str) -> Vec<T> {
    rows.into_iter()
        .filter_map(|mut row| row.get_mut(key).map(Json::take))
        .filter(Json::is_object)
        .filter_map(|record| serde_json::from_value(record).ok())
        .collect()
}
