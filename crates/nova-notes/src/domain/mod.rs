//! Records and the queries that belong to them.
//!
//! The Rust counterpart of `open_notebook/domain/`. Each record is a plain
//! struct that deserializes from a database row (ids are `"table:id"` strings,
//! datetimes are `DateTime<Utc>`), with its queries as associated functions
//! taking `&AppState`.

pub mod chat;
pub mod credential;
pub mod meeting;
pub mod model;
pub mod note;
pub mod notebook;
pub mod search;
pub mod settings;
pub mod source;
pub mod transformation;

pub use chat::{ChatSession, StoredMessage};
pub use credential::{ConfigSource, Credential, ResolvedProviderConfig};
pub use model::{DefaultModelType, DefaultModels, Model, ModelDisplayInfo};
pub use note::Note;
pub use notebook::{Folder, Notebook, Section};
pub use settings::ContentSettings;
pub use source::{Asset, Source, SourceEmbedding, SourceInsight};
pub use transformation::{DefaultPrompts, Transformation};

use serde::{Deserialize, Deserializer};

use crate::db::Json;
use crate::error::{AppError, AppResult};

/// Deserialize `null` (or a missing field, with `#[serde(default)]`) as the
/// type's default. Database rows hold NONE where Rust wants `""`, `false` or `[]`.
pub fn null_default<'de, D, T>(deserializer: D) -> Result<T, D::Error>
where
    D: Deserializer<'de>,
    T: Default + Deserialize<'de>,
{
    Ok(Option::<T>::deserialize(deserializer)?.unwrap_or_default())
}

/// Deserialize a record link that may arrive as an id string, as a fetched
/// record object (`FETCH`), or as null: keep only the id.
pub fn link_id<'de, D>(deserializer: D) -> Result<Option<String>, D::Error>
where
    D: Deserializer<'de>,
{
    Ok(match Json::deserialize(deserializer)? {
        Json::String(id) if !id.is_empty() => Some(id),
        Json::Object(map) => map.get("id").and_then(Json::as_str).map(String::from),
        _ => None,
    })
}

/// Reject an empty or whitespace-only name with the message the Python
/// validators used (`"<Thing> name cannot be empty"`).
pub(crate) fn require_name(kind: &str, name: &str) -> AppResult<()> {
    if name.trim().is_empty() {
        Err(AppError::invalid(format!("{kind} name cannot be empty")))
    } else {
        Ok(())
    }
}

#[cfg(test)]
mod tests;
