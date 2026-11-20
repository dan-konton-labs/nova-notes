//! Transformations: reusable prompts run against a source to produce insights.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use super::{link_id, null_default, settings};
use crate::db::{Doc, Params};
use crate::error::AppResult;
use crate::state::AppState;

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Transformation {
    #[serde(default)]
    pub id: String,
    #[serde(default, deserialize_with = "null_default")]
    pub name: String,
    #[serde(default, deserialize_with = "null_default")]
    pub title: String,
    #[serde(default, deserialize_with = "null_default")]
    pub description: String,
    #[serde(default, deserialize_with = "null_default")]
    pub prompt: String,
    /// Run automatically on every new source.
    #[serde(default, deserialize_with = "null_default")]
    pub apply_default: bool,
    /// Model to run with instead of the default transformation model.
    #[serde(default, deserialize_with = "link_id")]
    pub model_id: Option<String>,
    #[serde(default)]
    pub created: Option<DateTime<Utc>>,
    #[serde(default)]
    pub updated: Option<DateTime<Utc>>,
}

impl Transformation {
    pub const TABLE: &'static str = "transformation";

    /// Missing → `NotFound`.
    pub async fn get(state: &AppState, id: &str) -> AppResult<Self> {
        state.db.get(id).await
    }

    pub async fn list(state: &AppState, order_by: Option<&str>) -> AppResult<Vec<Self>> {
        let rows = state.db.select_all(Self::TABLE, order_by).await?;
        Ok(rows.into_iter().filter_map(|row| serde_json::from_value(row).ok()).collect())
    }

    /// Transformations flagged to run on every new source.
    pub async fn defaults(state: &AppState) -> AppResult<Vec<Self>> {
        state.db.query_as("SELECT * FROM transformation WHERE apply_default = true", Params::new()).await
    }

    fn doc(&self) -> Doc {
        Doc::new()
            .set("name", &self.name)
            .set("title", &self.title)
            .set("description", &self.description)
            .set("prompt", &self.prompt)
            .set("apply_default", self.apply_default)
            .record_or_none("model_id", self.model_id.as_deref())
    }

    /// Create (when `id` is empty) or update the transformation.
    pub async fn save(&mut self, state: &AppState) -> AppResult<()> {
        let row = if self.id.is_empty() {
            state.db.create(Self::TABLE, self.doc()).await?
        } else {
            state.db.update(&self.id, self.doc()).await?
        };
        *self = serde_json::from_value(row)?;
        Ok(())
    }

    pub async fn delete(&self, state: &AppState) -> AppResult<()> {
        state.db.delete(&self.id).await
    }
}

/// The `open_notebook:default_prompts` singleton.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DefaultPrompts {
    /// Instructions prepended to every transformation run.
    #[serde(default)]
    pub transformation_instructions: Option<String>,
}

impl DefaultPrompts {
    pub const RECORD_ID: &'static str = "open_notebook:default_prompts";

    pub async fn load(state: &AppState) -> AppResult<Self> {
        settings::load(state, Self::RECORD_ID).await
    }

    pub async fn save(&self, state: &AppState) -> AppResult<Self> {
        settings::save(state, Self::RECORD_ID, self).await
    }
}
