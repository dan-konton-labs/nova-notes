//! Settings singletons: single records in the `open_notebook` table
//! (`open_notebook:content_settings`, `open_notebook:default_models`, ...).

use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};

use crate::db::{Doc, Json};
use crate::error::AppResult;
use crate::state::AppState;

/// Load a singleton. A missing record, and any missing field, takes the
/// type's defaults.
pub async fn load<T: DeserializeOwned>(state: &AppState, record_id: &str) -> AppResult<T> {
    let row = state.db.select(record_id).await?.unwrap_or_else(|| Json::Object(Default::default()));
    Ok(serde_json::from_value(row)?)
}

/// Write every field of a singleton (creating the record if needed) and
/// return it as stored.
pub async fn save<T: Serialize + DeserializeOwned>(state: &AppState, record_id: &str, value: &T) -> AppResult<T> {
    let row = state.db.upsert(record_id, Doc::from(value)).await?;
    Ok(serde_json::from_value(row)?)
}

fn default_engine() -> Option<String> {
    Some("auto".to_string())
}
fn default_embedding_option() -> Option<String> {
    Some("ask".to_string())
}
fn default_auto_delete() -> Option<String> {
    Some("yes".to_string())
}
fn default_true() -> Option<bool> {
    Some(true)
}
fn default_false() -> Option<bool> {
    Some(false)
}
fn default_youtube_languages() -> Option<Vec<String>> {
    Some(
        ["en", "pt", "es", "de", "nl", "en-GB", "fr", "hi", "ja", "zh-CN", "zh-TW"]
            .iter()
            .map(|s| s.to_string())
            .collect(),
    )
}

/// The `open_notebook:content_settings` singleton: how sources are processed.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ContentSettings {
    /// `auto`, `docling` or `simple`.
    #[serde(default = "default_engine")]
    pub default_content_processing_engine_doc: Option<String>,
    /// `auto`, `firecrawl`, `jina`, `crawl4ai` or `simple`.
    #[serde(default = "default_engine")]
    pub default_content_processing_engine_url: Option<String>,
    /// `ask`, `always` or `never`.
    #[serde(default = "default_embedding_option")]
    pub default_embedding_option: Option<String>,
    /// `yes` or `no`: delete uploaded files once processed.
    #[serde(default = "default_auto_delete")]
    pub auto_delete_files: Option<String>,
    #[serde(default = "default_true")]
    pub docling_ocr: Option<bool>,
    #[serde(default = "default_false")]
    pub docling_formulas: Option<bool>,
    #[serde(default = "default_false")]
    pub docling_vision: Option<bool>,
    #[serde(default = "default_youtube_languages")]
    pub youtube_preferred_languages: Option<Vec<String>>,
}

impl Default for ContentSettings {
    fn default() -> Self {
        Self {
            default_content_processing_engine_doc: default_engine(),
            default_content_processing_engine_url: default_engine(),
            default_embedding_option: default_embedding_option(),
            auto_delete_files: default_auto_delete(),
            docling_ocr: default_true(),
            docling_formulas: default_false(),
            docling_vision: default_false(),
            youtube_preferred_languages: default_youtube_languages(),
        }
    }
}

impl ContentSettings {
    pub const RECORD_ID: &'static str = "open_notebook:content_settings";

    pub async fn load(state: &AppState) -> AppResult<Self> {
        load(state, Self::RECORD_ID).await
    }

    pub async fn save(&self, state: &AppState) -> AppResult<Self> {
        save(state, Self::RECORD_ID, self).await
    }
}
