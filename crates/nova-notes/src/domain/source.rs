//! Sources, their embedded chunks and their insights.

use std::collections::HashMap;
use std::path::Path;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::json;

use super::{link_id, null_default, Note, Notebook};
use crate::db::{Doc, Json, Params};
use crate::error::{AppError, AppResult};
use crate::state::AppState;

/// Where a source came from: an uploaded file or a URL.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Asset {
    #[serde(default)]
    pub file_path: Option<String>,
    #[serde(default)]
    pub url: Option<String>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Source {
    #[serde(default)]
    pub id: String,
    #[serde(default)]
    pub asset: Option<Asset>,
    #[serde(default)]
    pub title: Option<String>,
    #[serde(default, deserialize_with = "null_default")]
    pub topics: Vec<String>,
    #[serde(default)]
    pub full_text: Option<String>,
    #[serde(default)]
    pub last_viewed_at: Option<DateTime<Utc>>,
    /// `command:<id>` of the job that processes (or processed) this source.
    #[serde(default, deserialize_with = "link_id")]
    pub command: Option<String>,
    #[serde(default)]
    pub created: Option<DateTime<Utc>>,
    #[serde(default)]
    pub updated: Option<DateTime<Utc>>,
}

impl Source {
    pub const TABLE: &'static str = "source";

    /// Missing → `NotFound`.
    pub async fn get(state: &AppState, id: &str) -> AppResult<Self> {
        state.db.get(id).await
    }

    /// Create the record. Does not process, embed or link it to a notebook.
    pub async fn create(state: &AppState, new: &Source) -> AppResult<Self> {
        let row = state.db.create(Self::TABLE, new.doc()).await?;
        Ok(serde_json::from_value(row)?)
    }

    fn doc(&self) -> Doc {
        // Unset optional fields are left alone, as the Python model's save did.
        let mut doc = Doc::new()
            .set_some("asset", self.asset.as_ref())
            .set_some("title", self.title.as_ref())
            .set("topics", &self.topics)
            .set_some("full_text", self.full_text.as_ref());
        if let Some(viewed) = self.last_viewed_at {
            doc = doc.datetime("last_viewed_at", viewed);
        }
        if let Some(command) = &self.command {
            doc = doc.record("command", command);
        }
        doc
    }

    /// Write every set field back. Does **not** re-embed; call
    /// [`Source::vectorize`] for that.
    pub async fn save(&mut self, state: &AppState) -> AppResult<()> {
        let row = state.db.update(&self.id, self.doc()).await?;
        *self = serde_json::from_value(row)?;
        Ok(())
    }

    /// Status of the processing job: `None` when the source has no job,
    /// `"unknown"` when the job record is gone.
    pub async fn status(&self, state: &AppState) -> Option<String> {
        let command = self.command.as_deref()?;
        match state.jobs.get(command).await {
            Ok(Some(record)) => Some(record.status_str().to_string()),
            Ok(None) => Some("unknown".to_string()),
            Err(e) => {
                tracing::warn!("Failed to get command status for {command}: {e}");
                Some("unknown".to_string())
            }
        }
    }

    /// Status, timing and result of the processing job, when there is one.
    pub async fn processing_progress(&self, state: &AppState) -> Option<Json> {
        let command = self.command.as_deref()?;
        let record = state.jobs.get(command).await.ok().flatten()?;
        let metadata = record.result.as_ref().and_then(|r| r.get("execution_metadata"));
        Some(json!({
            "status": record.status_str(),
            "started_at": metadata.and_then(|m| m.get("started_at")),
            "completed_at": metadata.and_then(|m| m.get("completed_at")),
            "error": record.error_message,
            "result": record.result,
        }))
    }

    /// The source as model context: `{id, title, insights}` plus `full_text`
    /// for the long form. Pass `insights` when they were already batch-loaded.
    pub async fn context(&self, state: &AppState, long: bool, insights: Option<Vec<SourceInsight>>) -> AppResult<Json> {
        let insights = match insights {
            Some(insights) => insights,
            None => self.insights(state).await?,
        };
        let mut context = json!({ "id": self.id, "title": self.title, "insights": insights });
        if long {
            context["full_text"] = json!(self.full_text);
        }
        Ok(context)
    }

    /// Number of embedded chunks.
    pub async fn embedded_chunks(&self, state: &AppState) -> AppResult<i64> {
        let rows = state
            .db
            .query(
                "SELECT count() AS chunks FROM source_embedding WHERE source = $id GROUP ALL",
                Params::new().record("id", &self.id),
            )
            .await?;
        Ok(rows.first().and_then(|r| r.get("chunks")).and_then(Json::as_i64).unwrap_or(0))
    }

    pub async fn insights(&self, state: &AppState) -> AppResult<Vec<SourceInsight>> {
        state
            .db
            .query_as("SELECT * OMIT embedding FROM source_insight WHERE source = $id", Params::new().record("id", &self.id))
            .await
    }

    /// Ids of the notebooks holding this source.
    pub async fn notebook_ids(&self, state: &AppState) -> AppResult<Vec<String>> {
        let rows = state
            .db
            .query("SELECT VALUE out FROM reference WHERE in = $id", Params::new().record("id", &self.id))
            .await?;
        Ok(rows.iter().filter_map(Json::as_str).map(String::from).collect())
    }

    /// Link the source to a notebook. Missing notebook → `NotFound`.
    pub async fn add_to_notebook(&self, state: &AppState, notebook_id: &str) -> AppResult<()> {
        if notebook_id.is_empty() {
            return Err(AppError::invalid("Notebook ID must be provided"));
        }
        Notebook::get(state, notebook_id).await?;
        state.db.relate(&self.id, "reference", notebook_id, Doc::new()).await?;
        Ok(())
    }

    /// Queue the `embed_source` job and return its command id.
    pub async fn vectorize(&self, state: &AppState) -> AppResult<String> {
        if self.full_text.as_deref().is_none_or(|t| t.trim().is_empty()) {
            return Err(AppError::invalid(format!("Source {} has no text to vectorize", self.id)));
        }
        state.jobs.submit("embed_source", json!({ "source_id": self.id })).await
    }

    /// Queue the `create_insight` job (which stores and embeds the insight)
    /// and return its command id.
    pub async fn add_insight(&self, state: &AppState, insight_type: &str, content: &str) -> AppResult<String> {
        if insight_type.is_empty() || content.is_empty() {
            return Err(AppError::invalid("Insight type and content must be provided"));
        }
        state
            .jobs
            .submit(
                "create_insight",
                json!({ "source_id": self.id, "insight_type": insight_type, "content": content }),
            )
            .await
    }

    /// Delete the source, its uploaded file, its chunks and its insights.
    pub async fn delete(&self, state: &AppState) -> AppResult<()> {
        if let Some(path) = self.asset.as_ref().and_then(|a| a.file_path.as_deref()) {
            let path = Path::new(path);
            if path.exists() {
                match std::fs::remove_file(path) {
                    Ok(()) => tracing::info!("Deleted file for source {}: {}", self.id, path.display()),
                    Err(e) => tracing::warn!(
                        "Failed to delete file {} for source {}: {e}. Continuing with database deletion.",
                        path.display(),
                        self.id
                    ),
                }
            }
        }
        let cleanup = state
            .db
            .query_all(
                "DELETE source_embedding WHERE source = $source_id; DELETE source_insight WHERE source = $source_id;",
                Params::new().record("source_id", &self.id),
            )
            .await;
        if let Err(e) = cleanup {
            tracing::warn!("Failed to delete embeddings/insights for source {}: {e}. Continuing with source deletion.", self.id);
        }
        state.db.query("DELETE reference WHERE in = $id", Params::new().record("id", &self.id)).await?;
        state.db.delete(&self.id).await
    }
}

/// One embedded chunk of a source (the vector itself is not loaded).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SourceEmbedding {
    #[serde(default)]
    pub id: String,
    #[serde(default, deserialize_with = "link_id")]
    pub source: Option<String>,
    #[serde(default, deserialize_with = "null_default")]
    pub order: i64,
    #[serde(default, deserialize_with = "null_default")]
    pub content: String,
}

impl SourceEmbedding {
    pub const TABLE: &'static str = "source_embedding";

    pub async fn get(state: &AppState, id: &str) -> AppResult<Self> {
        let rows: Vec<Self> = state
            .db
            .query_as("SELECT * OMIT embedding FROM $id", Params::new().record("id", id))
            .await?;
        rows.into_iter().next().ok_or_else(|| AppError::not_found(format!("source_embedding with id {id} not found")))
    }

    pub async fn source(&self, state: &AppState) -> AppResult<Source> {
        match &self.source {
            Some(source) => Source::get(state, source).await,
            None => Err(AppError::not_found(format!("Source for embedding {} not found", self.id))),
        }
    }
}

/// The output of a transformation run against a source.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SourceInsight {
    #[serde(default)]
    pub id: String,
    #[serde(default, deserialize_with = "link_id")]
    pub source: Option<String>,
    #[serde(default, deserialize_with = "null_default")]
    pub insight_type: String,
    #[serde(default, deserialize_with = "null_default")]
    pub content: String,
    #[serde(default)]
    pub created: Option<DateTime<Utc>>,
    #[serde(default)]
    pub updated: Option<DateTime<Utc>>,
}

impl SourceInsight {
    pub const TABLE: &'static str = "source_insight";

    /// Missing → `NotFound`.
    pub async fn get(state: &AppState, id: &str) -> AppResult<Self> {
        let rows: Vec<Self> = state
            .db
            .query_as("SELECT * OMIT embedding FROM $id", Params::new().record("id", id))
            .await?;
        rows.into_iter().next().ok_or_else(|| AppError::not_found(format!("source_insight with id {id} not found")))
    }

    /// Insights for many sources in one query, keyed by source id. Every
    /// requested id is present in the map, possibly with an empty list.
    pub async fn for_sources(state: &AppState, source_ids: &[String]) -> AppResult<HashMap<String, Vec<Self>>> {
        let mut grouped: HashMap<String, Vec<Self>> =
            source_ids.iter().filter(|id| !id.is_empty()).map(|id| (id.clone(), Vec::new())).collect();
        if grouped.is_empty() {
            return Ok(grouped);
        }
        let rows: Vec<Self> = state
            .db
            .query_as(
                "SELECT * OMIT embedding FROM source_insight WHERE source IN $source_ids",
                Params::new().records("source_ids", grouped.keys()),
            )
            .await
            .map_err(|e| AppError::database(format!("Failed to fetch insights for sources: {e}")))?;
        for insight in rows {
            if let Some(source) = insight.source.clone() {
                grouped.entry(source).or_default().push(insight);
            }
        }
        Ok(grouped)
    }

    pub async fn source(&self, state: &AppState) -> AppResult<Source> {
        match &self.source {
            Some(source) => Source::get(state, source).await,
            None => Err(AppError::not_found(format!("Source for insight {} not found", self.id))),
        }
    }

    /// Copy the insight into a new note, optionally filed in a notebook.
    pub async fn save_as_note(&self, state: &AppState, notebook_id: Option<&str>) -> AppResult<Note> {
        let source = self.source(state).await?;
        let title = format!("{} from source {}", self.insight_type, source.title.as_deref().unwrap_or("None"));
        let note = Note::create(state, Some(&title), Some(&self.content), None).await?;
        if let Some(notebook_id) = notebook_id {
            note.add_to_notebook(state, notebook_id).await?;
        }
        Ok(note)
    }

    pub async fn delete(&self, state: &AppState) -> AppResult<()> {
        state.db.delete(&self.id).await
    }
}
