//! Configured models and the default-model slots.
//!
//! The record half of `open_notebook/ai/models.py`: the `model` table
//! ([`Model`]), the `open_notebook:default_models` singleton
//! ([`DefaultModels`]) and the lookup that decides which model a slot or a
//! model type resolves to. Turning a model into a live client is
//! `crate::ai`'s job.

use std::collections::HashMap;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use super::credential::{credential_link, resolve_provider_config, ConfigSource, Credential, ResolvedProviderConfig};
use super::{link_id, null_default, settings};
use crate::db::{table_of, with_table, Doc, Params};
use crate::error::{AppError, AppResult};
use crate::state::AppState;

/// The model types a `model` record can have, in the order the Python API
/// listed them in its error message.
pub const MODEL_TYPES: [&str; 4] = ["language", "embedding", "text_to_speech", "speech_to_text"];

pub fn is_valid_model_type(model_type: &str) -> bool {
    MODEL_TYPES.contains(&model_type)
}

/// One configured model: a provider's model name, its type and optionally
/// the credential it runs with.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Model {
    #[serde(default)]
    pub id: String,
    #[serde(default, deserialize_with = "null_default")]
    pub name: String,
    #[serde(default, deserialize_with = "null_default")]
    pub provider: String,
    /// `language`, `embedding`, `text_to_speech` or `speech_to_text`.
    #[serde(rename = "type", default, deserialize_with = "null_default")]
    pub model_type: String,
    /// `credential:<id>`. Without one the provider's default credential (or
    /// the environment) is used.
    #[serde(default, deserialize_with = "link_id")]
    pub credential: Option<String>,
    #[serde(default)]
    pub created: Option<DateTime<Utc>>,
    #[serde(default)]
    pub updated: Option<DateTime<Utc>>,
}

/// What an episode listing shows for a model reference.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelDisplayInfo {
    pub provider: String,
    pub name: String,
}

impl Model {
    pub const TABLE: &'static str = "model";

    /// Missing → `NotFound`.
    pub async fn get(state: &AppState, id: &str) -> AppResult<Self> {
        let id = with_table(Self::TABLE, id);
        if table_of(&id) != Some(Self::TABLE) {
            return Err(AppError::not_found(format!("model with id {id} not found")));
        }
        state.db.get(&id).await
    }

    /// A model about to be used (Python's `ModelManager.get_model` checks):
    /// a missing model or one with an unknown type is a `Configuration`
    /// error with the Python message. Database failures stay database errors.
    pub async fn get_for_use(state: &AppState, id: &str) -> AppResult<Self> {
        let model = match Self::get(state, id).await {
            Ok(model) => model,
            Err(AppError::NotFound(_) | AppError::InvalidInput(_)) => {
                return Err(AppError::config(format!("Model with ID {id} not found")))
            }
            Err(e) => return Err(e),
        };
        if !is_valid_model_type(&model.model_type) {
            return Err(AppError::config(format!("Invalid model type: {}", model.model_type)));
        }
        Ok(model)
    }

    pub async fn get_all(state: &AppState, order_by: Option<&str>) -> AppResult<Vec<Self>> {
        let rows = state.db.select_all(Self::TABLE, order_by).await?;
        Ok(rows.into_iter().filter_map(|row| serde_json::from_value(row).ok()).collect())
    }

    /// Models of one type (`language`, `embedding`, ...).
    pub async fn get_by_type(state: &AppState, model_type: &str) -> AppResult<Vec<Self>> {
        state
            .db
            .query_as("SELECT * FROM model WHERE type = $model_type", Params::new().set("model_type", model_type))
            .await
    }

    /// Models linked to a credential.
    pub async fn get_by_credential(state: &AppState, credential_id: &str) -> AppResult<Vec<Self>> {
        state
            .db
            .query_as(
                "SELECT * FROM model WHERE credential = $cred_id",
                Params::new().record("cred_id", credential_link(credential_id)),
            )
            .await
    }

    /// Models of a provider (exact match), ordered by type then name, as
    /// `GET /models/by-provider/{provider}` lists them.
    pub async fn get_by_provider(state: &AppState, provider: &str) -> AppResult<Vec<Self>> {
        state
            .db
            .query_as(
                "SELECT * FROM model WHERE provider = $provider ORDER BY type, name",
                Params::new().set("provider", provider),
            )
            .await
    }

    /// An existing model with the same provider, name and type, compared
    /// case-insensitively (the duplicate check before creating one).
    pub async fn find_duplicate(
        state: &AppState,
        provider: &str,
        name: &str,
        model_type: &str,
    ) -> AppResult<Option<Self>> {
        let found: Vec<Self> = state
            .db
            .query_as(
                "SELECT * FROM model WHERE string::lowercase(provider) = $provider AND string::lowercase(name) = $name \
                 AND string::lowercase(type) = $model_type LIMIT 1",
                Params::new()
                    .set("provider", provider.to_lowercase())
                    .set("name", name.to_lowercase())
                    .set("model_type", model_type.to_lowercase()),
            )
            .await?;
        Ok(found.into_iter().next())
    }

    /// How many models a provider has of each type (case-insensitive
    /// provider). Every type is present, zero when there are none.
    pub async fn count_by_type(state: &AppState, provider: &str) -> AppResult<HashMap<String, i64>> {
        let rows = state
            .db
            .query(
                "SELECT type, count() AS count FROM model WHERE string::lowercase(provider) = string::lowercase($provider) GROUP BY type",
                Params::new().set("provider", provider),
            )
            .await?;
        let mut counts: HashMap<String, i64> = MODEL_TYPES.iter().map(|t| (t.to_string(), 0)).collect();
        for row in rows {
            if let (Some(model_type), Some(count)) = (row["type"].as_str(), row["count"].as_i64()) {
                if let Some(slot) = counts.get_mut(model_type) {
                    *slot = count;
                }
            }
        }
        Ok(counts)
    }

    /// `{provider, name}` for many model ids in one query, keyed by id.
    /// Unknown ids are absent; a failed query gives an empty map, so display
    /// degrades instead of breaking the caller (as in Python).
    pub async fn display_info_for_ids<I, S>(state: &AppState, ids: I) -> HashMap<String, ModelDisplayInfo>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        let mut ids: Vec<String> = ids
            .into_iter()
            .map(|id| id.as_ref().to_string())
            .filter(|id| !id.is_empty())
            .collect();
        ids.sort();
        ids.dedup();
        let mut found = HashMap::new();
        if ids.is_empty() {
            return found;
        }
        let rows = match state
            .db
            .query("SELECT id, name, provider FROM model WHERE id IN $model_ids", Params::new().records("model_ids", &ids))
            .await
        {
            Ok(rows) => rows,
            Err(e) => {
                tracing::error!("Error batch-fetching model display info: {e}");
                return found;
            }
        };
        for row in rows {
            if let Some(id) = row["id"].as_str() {
                let text = |key: &str| row[key].as_str().unwrap_or_default().to_string();
                found.insert(id.to_string(), ModelDisplayInfo { provider: text("provider"), name: text("name") });
            }
        }
        found
    }

    fn doc(&self) -> Doc {
        Doc::new()
            .set("name", &self.name)
            .set("provider", &self.provider)
            .set("type", &self.model_type)
            .record_or_none("credential", self.credential.as_deref().filter(|c| !c.is_empty()).map(credential_link))
    }

    /// Create (when `id` is empty) or update the model.
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
        if self.id.is_empty() {
            return Err(AppError::invalid("Cannot delete object without an ID"));
        }
        state.db.delete(&self.id).await
    }

    /// The linked credential, or `None` when there is none or it cannot be
    /// loaded (missing, undecryptable); the caller then falls back to the
    /// provider's default configuration.
    pub async fn credential(&self, state: &AppState) -> Option<Credential> {
        let id = self.credential.as_deref().filter(|c| !c.is_empty())?;
        match Credential::get(state, id).await {
            Ok(credential) => Some(credential),
            Err(e) => {
                tracing::warn!("Could not load credential {id} for model {}: {e}", self.id);
                None
            }
        }
    }

    /// The configuration this model's client is built with: its linked
    /// credential when that loads, otherwise the provider's default
    /// credential, otherwise the environment. `None` when nothing is
    /// configured. A linked credential of another provider is a
    /// `Configuration` error: its key must not go to this model's provider.
    pub async fn provider_config(
        &self,
        state: &AppState,
    ) -> AppResult<Option<(ResolvedProviderConfig, ConfigSource)>> {
        if let Some(credential) = self.credential(state).await {
            if nn_ai::normalize_provider_name(&credential.provider) != nn_ai::normalize_provider_name(&self.provider) {
                return Err(AppError::config(format!(
                    "Model '{}' uses the {} provider but is linked to a credential for {}. Link it to a {} credential in Manage → Models.",
                    self.name, self.provider, credential.provider, self.provider
                )));
            }
            return Ok(Some((credential.provider_config(), ConfigSource::Database)));
        }
        resolve_provider_config(state, &self.provider).await
    }
}

// ---------------------------------------------------------------------------
// Default models
// ---------------------------------------------------------------------------

/// What a caller asks the defaults for (Python's `get_default_model`
/// `model_type` strings).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DefaultModelType {
    Chat,
    Transformation,
    Tools,
    LargeContext,
    Embedding,
    TextToSpeech,
    SpeechToText,
}

impl DefaultModelType {
    pub const ALL: [Self; 7] = [
        Self::Chat,
        Self::Transformation,
        Self::Tools,
        Self::LargeContext,
        Self::Embedding,
        Self::TextToSpeech,
        Self::SpeechToText,
    ];

    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Chat => "chat",
            Self::Transformation => "transformation",
            Self::Tools => "tools",
            Self::LargeContext => "large_context",
            Self::Embedding => "embedding",
            Self::TextToSpeech => "text_to_speech",
            Self::SpeechToText => "speech_to_text",
        }
    }

    pub fn parse(value: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|kind| kind.as_str() == value)
    }

    /// The `DefaultModels` field this type reads first.
    pub fn slot(&self) -> &'static str {
        match self {
            Self::Chat => "default_chat_model",
            Self::Transformation => "default_transformation_model",
            Self::Tools => "default_tools_model",
            Self::LargeContext => "large_context_model",
            Self::Embedding => "default_embedding_model",
            Self::TextToSpeech => "default_text_to_speech_model",
            Self::SpeechToText => "default_speech_to_text_model",
        }
    }

    /// The `model` record type the slot must hold.
    pub fn model_type(&self) -> &'static str {
        match self {
            Self::Chat | Self::Transformation | Self::Tools | Self::LargeContext => "language",
            Self::Embedding => "embedding",
            Self::TextToSpeech => "text_to_speech",
            Self::SpeechToText => "speech_to_text",
        }
    }

    /// The text slots fall back to the chat model when unset (#1098); the
    /// audio and embedding slots do not.
    pub fn falls_back_to_chat(&self) -> bool {
        matches!(self, Self::Transformation | Self::Tools | Self::LargeContext)
    }
}

/// The `open_notebook:default_models` singleton. Values are model ids
/// stored as strings.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct DefaultModels {
    #[serde(default, deserialize_with = "link_id")]
    pub default_chat_model: Option<String>,
    #[serde(default, deserialize_with = "link_id")]
    pub default_transformation_model: Option<String>,
    #[serde(default, deserialize_with = "link_id")]
    pub large_context_model: Option<String>,
    #[serde(default, deserialize_with = "link_id")]
    pub default_text_to_speech_model: Option<String>,
    #[serde(default, deserialize_with = "link_id")]
    pub default_speech_to_text_model: Option<String>,
    #[serde(default, deserialize_with = "link_id")]
    pub default_embedding_model: Option<String>,
    #[serde(default, deserialize_with = "link_id")]
    pub default_tools_model: Option<String>,
}

impl DefaultModels {
    pub const RECORD_ID: &'static str = "open_notebook:default_models";

    /// Every slot, in the order of the Python model.
    pub const SLOTS: [&'static str; 7] = [
        "default_chat_model",
        "default_transformation_model",
        "large_context_model",
        "default_text_to_speech_model",
        "default_speech_to_text_model",
        "default_embedding_model",
        "default_tools_model",
    ];

    /// Slots that can be reassigned but never cleared.
    pub const REQUIRED: [&'static str; 2] = ["default_chat_model", "default_embedding_model"];

    /// Always read fresh from the database; a missing record is all-empty.
    pub async fn load(state: &AppState) -> AppResult<Self> {
        settings::load(state, Self::RECORD_ID).await
    }

    /// Write every slot (an empty slot is removed from the record).
    pub async fn save(&self, state: &AppState) -> AppResult<Self> {
        settings::save(state, Self::RECORD_ID, self).await
    }

    fn slot_mut(&mut self, slot: &str) -> Option<&mut Option<String>> {
        Some(match slot {
            "default_chat_model" => &mut self.default_chat_model,
            "default_transformation_model" => &mut self.default_transformation_model,
            "large_context_model" => &mut self.large_context_model,
            "default_text_to_speech_model" => &mut self.default_text_to_speech_model,
            "default_speech_to_text_model" => &mut self.default_speech_to_text_model,
            "default_embedding_model" => &mut self.default_embedding_model,
            "default_tools_model" => &mut self.default_tools_model,
            _ => return None,
        })
    }

    /// A slot's value by field name; `""` counts as unset.
    pub fn slot(&self, slot: &str) -> Option<&str> {
        let value = match slot {
            "default_chat_model" => &self.default_chat_model,
            "default_transformation_model" => &self.default_transformation_model,
            "large_context_model" => &self.large_context_model,
            "default_text_to_speech_model" => &self.default_text_to_speech_model,
            "default_speech_to_text_model" => &self.default_speech_to_text_model,
            "default_embedding_model" => &self.default_embedding_model,
            "default_tools_model" => &self.default_tools_model,
            _ => return None,
        };
        value.as_deref().filter(|v| !v.is_empty())
    }

    /// Assign or clear a slot by field name (`PUT /models/defaults`
    /// semantics for one field that was present in the request). Clearing a
    /// required slot is `InvalidInput` with the Python message; an unknown
    /// field is `InvalidInput` too.
    pub fn set_slot(&mut self, slot: &str, value: Option<String>) -> AppResult<()> {
        let value = value.filter(|v| !v.is_empty());
        if value.is_none() && Self::REQUIRED.contains(&slot) {
            return Err(AppError::invalid(format!("{slot} is required and cannot be cleared, only reassigned")));
        }
        let field = self.slot_mut(slot).ok_or_else(|| AppError::invalid(format!("Unknown default model slot: {slot}")))?;
        *field = value;
        Ok(())
    }

    /// The model id a request for `kind` resolves to: its own slot, or for
    /// the text slots the chat model when theirs is empty. `None` when
    /// nothing is configured.
    pub fn model_id_for(&self, kind: DefaultModelType) -> Option<&str> {
        self.slot(kind.slot()).or_else(|| {
            kind.falls_back_to_chat().then(|| self.slot(DefaultModelType::Chat.slot())).flatten()
        })
    }

    /// The default model record for `kind`, or `None` when no slot is set or
    /// the configured model is gone or unusable (logged, as in Python's
    /// `get_default_model`, so the user is pointed at Manage → Models).
    pub async fn model_for(state: &AppState, kind: DefaultModelType) -> AppResult<Option<Model>> {
        let defaults = Self::load(state).await?;
        let Some(id) = defaults.model_id_for(kind) else {
            tracing::warn!(
                "No default model configured for type '{}'. Please go to Manage → Models and set a default model.",
                kind.as_str()
            );
            return Ok(None);
        };
        match Model::get_for_use(state, id).await {
            Ok(model) => Ok(Some(model)),
            Err(e @ (AppError::Configuration(_) | AppError::InvalidInput(_))) => {
                tracing::error!(
                    "Failed to load default model for type '{}': {e}. The configured model_id '{id}' may have been \
                     deleted or misconfigured. Please go to Manage → Models and reconfigure the default model.",
                    kind.as_str()
                );
                Ok(None)
            }
            Err(e) => Err(e),
        }
    }
}

/// Fill the empty required slots (chat, embedding) from the available
/// models, leaving optional slots alone (#1098). `choose` picks one model
/// from the candidates of the slot's type (the provider-priority logic lives
/// with the endpoint). Returns `(assigned, skipped, missing)`.
pub fn auto_assign_required(
    defaults: &mut DefaultModels,
    models: &[Model],
    choose: impl Fn(&[&Model]) -> Option<String>,
) -> (Vec<(String, String)>, Vec<String>, Vec<String>) {
    let mut assigned = Vec::new();
    let mut skipped = Vec::new();
    let mut missing = Vec::new();
    for (slot, model_type) in [("default_chat_model", "language"), ("default_embedding_model", "embedding")] {
        if defaults.slot(slot).is_some() {
            skipped.push(slot.to_string());
            continue;
        }
        let candidates: Vec<&Model> = models.iter().filter(|m| m.model_type == model_type).collect();
        if candidates.is_empty() {
            missing.push(slot.to_string());
            continue;
        }
        if let Some(id) = choose(&candidates) {
            if let Some(field) = defaults.slot_mut(slot) {
                *field = Some(id.clone());
            }
            assigned.push((slot.to_string(), id));
        }
    }
    (assigned, skipped, missing)
}

/// Every model ordered by provider then name (the auto-assign input).
pub async fn all_models_by_provider(state: &AppState) -> AppResult<Vec<Model>> {
    state.db.query_as("SELECT * FROM model ORDER BY provider, name", Params::new()).await
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;
    use crate::crypto::SecretString;

    async fn state() -> (AppState, tempfile::TempDir) {
        AppState::for_tests().await.unwrap()
    }

    async fn model(state: &AppState, provider: &str, name: &str, model_type: &str) -> Model {
        let mut model =
            Model { name: name.into(), provider: provider.into(), model_type: model_type.into(), ..Model::default() };
        model.save(state).await.unwrap();
        model
    }

    #[tokio::test]
    async fn model_crud_and_queries() {
        let (state, _dir) = state().await;
        let gpt = model(&state, "openai", "gpt-4o", "language").await;
        assert!(gpt.id.starts_with("model:"));
        assert!(gpt.created.is_some());
        let embed = model(&state, "openai", "text-embedding-3", "embedding").await;
        model(&state, "anthropic", "claude", "language").await;
        model(&state, "OpenAI", "tts-1", "text_to_speech").await;

        let raw = state.db.select(&gpt.id).await.unwrap().unwrap();
        assert_eq!(raw["type"], "language");
        assert!(raw.get("credential").is_none());

        let language = Model::get_by_type(&state, "language").await.unwrap();
        assert_eq!(language.len(), 2);
        assert!(language.iter().all(|m| m.model_type == "language"));

        let by_provider = Model::get_by_provider(&state, "openai").await.unwrap();
        assert_eq!(by_provider.iter().map(|m| m.name.as_str()).collect::<Vec<_>>(), ["text-embedding-3", "gpt-4o"]);

        let counts = Model::count_by_type(&state, "OPENAI").await.unwrap();
        assert_eq!(counts["language"], 1);
        assert_eq!(counts["embedding"], 1);
        assert_eq!(counts["text_to_speech"], 1);
        assert_eq!(counts["speech_to_text"], 0);

        // Duplicate check is case-insensitive on all three.
        assert_eq!(Model::find_duplicate(&state, "OpenAI", "GPT-4O", "language").await.unwrap().unwrap().id, gpt.id);
        assert!(Model::find_duplicate(&state, "anthropic", "gpt-4o", "language").await.unwrap().is_none());
        assert!(Model::find_duplicate(&state, "openai", "gpt-4o", "embedding").await.unwrap().is_none());

        let info = Model::display_info_for_ids(&state, [gpt.id.as_str(), gpt.id.as_str(), "model:gone", ""]).await;
        assert_eq!(info.len(), 1);
        assert_eq!(info[&gpt.id], ModelDisplayInfo { provider: "openai".into(), name: "gpt-4o".into() });
        assert!(Model::display_info_for_ids(&state, Vec::<String>::new()).await.is_empty());

        let ordered = all_models_by_provider(&state).await.unwrap();
        assert_eq!(ordered.first().unwrap().provider, "OpenAI");
        assert_eq!(Model::get_all(&state, Some("name asc")).await.unwrap().len(), 4);

        let mut renamed = Model::get(&state, &embed.id).await.unwrap();
        renamed.name = "text-embedding-3-large".into();
        renamed.save(&state).await.unwrap();
        assert_eq!(Model::get(&state, &embed.id).await.unwrap().name, "text-embedding-3-large");

        embed.delete(&state).await.unwrap();
        assert!(Model::get(&state, &embed.id).await.unwrap_err().is_not_found());
        assert!(Model::default().delete(&state).await.is_err());
    }

    #[tokio::test]
    async fn models_link_to_credentials() {
        let (state, _dir) = state().await;
        let mut cred = Credential {
            name: "Work".into(),
            provider: "openai".into(),
            api_key: Some(SecretString::new("sk-work")),
            ..Credential::default()
        };
        cred.save(&state).await.unwrap();

        let mut linked = Model {
            name: "gpt-4o".into(),
            provider: "openai".into(),
            model_type: "language".into(),
            credential: Some(cred.id.clone()),
            ..Model::default()
        };
        linked.save(&state).await.unwrap();
        assert_eq!(linked.credential.as_deref(), Some(cred.id.as_str()));
        model(&state, "openai", "unlinked", "language").await;

        assert_eq!(Model::get_by_credential(&state, &cred.id).await.unwrap().len(), 1);
        let bare = cred.id.trim_start_matches("credential:");
        assert_eq!(Model::get_by_credential(&state, bare).await.unwrap()[0].id, linked.id);
        assert_eq!(cred.linked_models(&state).await.unwrap()[0].id, linked.id);

        // The linked credential's configuration is used.
        let loaded = linked.credential(&state).await.unwrap();
        assert_eq!(loaded.api_key.unwrap().expose(), "sk-work");
        let (config, source) = linked.provider_config(&state).await.unwrap().unwrap();
        assert_eq!(source, ConfigSource::Database);
        assert_eq!(config.api_key.as_deref(), Some("sk-work"));

        // Unlinking clears the field.
        linked.credential = None;
        linked.save(&state).await.unwrap();
        assert!(state.db.select(&linked.id).await.unwrap().unwrap().get("credential").is_none());
        assert!(cred.linked_models(&state).await.unwrap().is_empty());

        // A dangling link falls back to the provider's default credential.
        linked.credential = Some("credential:deleted".into());
        linked.save(&state).await.unwrap();
        assert!(linked.credential(&state).await.is_none());
        let (config, _) = linked.provider_config(&state).await.unwrap().unwrap();
        assert_eq!(config.api_key.as_deref(), Some("sk-work"));
    }

    #[tokio::test]
    async fn models_for_use_are_checked() {
        let (state, _dir) = state().await;
        let error = Model::get_for_use(&state, "model:missing").await.unwrap_err();
        assert!(matches!(&error, AppError::Configuration(m) if m == "Model with ID model:missing not found"), "{error:?}");
        let created = state
            .db
            .create("model", Doc::new().set("name", "x").set("provider", "openai").set("type", "vision"))
            .await
            .unwrap();
        let id = created["id"].as_str().unwrap();
        let error = Model::get_for_use(&state, id).await.unwrap_err();
        assert_eq!(error.to_string(), "Invalid model type: vision");
        assert!(is_valid_model_type("speech_to_text"));
        assert!(!is_valid_model_type("chat"));
    }

    #[test]
    fn default_slots_resolve_with_text_fallback_to_chat() {
        let defaults = DefaultModels {
            default_chat_model: Some("model:chat".into()),
            default_embedding_model: Some("model:embed".into()),
            default_tools_model: Some(String::new()),
            ..DefaultModels::default()
        };
        for kind in [DefaultModelType::Transformation, DefaultModelType::Tools, DefaultModelType::LargeContext] {
            assert_eq!(defaults.model_id_for(kind), Some("model:chat"), "{kind:?}");
        }
        assert_eq!(defaults.model_id_for(DefaultModelType::Chat), Some("model:chat"));
        assert_eq!(defaults.model_id_for(DefaultModelType::Embedding), Some("model:embed"));
        assert_eq!(defaults.model_id_for(DefaultModelType::TextToSpeech), None);
        assert_eq!(defaults.model_id_for(DefaultModelType::SpeechToText), None);

        let specific = DefaultModels { default_transformation_model: Some("model:t".into()), ..defaults.clone() };
        assert_eq!(specific.model_id_for(DefaultModelType::Transformation), Some("model:t"));
        assert_eq!(DefaultModels::default().model_id_for(DefaultModelType::Tools), None);

        for kind in DefaultModelType::ALL {
            assert_eq!(DefaultModelType::parse(kind.as_str()), Some(kind));
            assert!(DefaultModels::SLOTS.contains(&kind.slot()));
        }
        assert_eq!(DefaultModelType::parse("vision"), None);
    }

    #[test]
    fn set_slot_keeps_required_slots_filled() {
        let mut defaults = DefaultModels {
            default_chat_model: Some("model:chat".into()),
            default_transformation_model: Some("model:transform".into()),
            ..DefaultModels::default()
        };
        defaults.set_slot("default_transformation_model", None).unwrap();
        assert_eq!(defaults.default_transformation_model, None);
        for slot in DefaultModels::REQUIRED {
            let error = defaults.set_slot(slot, None).unwrap_err();
            assert_eq!(error.status(), axum::http::StatusCode::BAD_REQUEST);
            assert!(error.to_string().contains(slot));
        }
        assert_eq!(defaults.default_chat_model.as_deref(), Some("model:chat"));
        defaults.set_slot("default_chat_model", Some("model:new-chat".into())).unwrap();
        assert_eq!(defaults.default_chat_model.as_deref(), Some("model:new-chat"));
        assert!(defaults.set_slot("default_vision_model", Some("x".into())).is_err());
    }

    #[test]
    fn auto_assign_fills_only_required_slots() {
        let m = |id: &str, t: &str| Model { id: id.into(), model_type: t.into(), ..Model::default() };
        let models =
            [m("model:lang", "language"), m("model:embed", "embedding"), m("model:tts", "text_to_speech"), m("model:stt", "speech_to_text")];
        let first = |c: &[&Model]| c.first().map(|m| m.id.clone());

        let mut defaults = DefaultModels::default();
        let (assigned, skipped, missing) = auto_assign_required(&mut defaults, &models, first);
        assert_eq!(
            assigned,
            [("default_chat_model".to_string(), "model:lang".to_string()), ("default_embedding_model".into(), "model:embed".into())]
        );
        assert!(skipped.is_empty() && missing.is_empty());
        assert_eq!(defaults.default_text_to_speech_model, None);
        assert_eq!(defaults.default_transformation_model, None);

        let mut filled = DefaultModels { default_chat_model: Some("model:existing".into()), ..DefaultModels::default() };
        let (assigned, skipped, missing) = auto_assign_required(&mut filled, &models[2..], first);
        assert!(assigned.is_empty());
        assert_eq!(skipped, ["default_chat_model"]);
        assert_eq!(missing, ["default_embedding_model"]);
    }

    #[tokio::test]
    async fn default_models_singleton_round_trips() {
        let (state, _dir) = state().await;
        // Migration 1 seeds the record with an empty chat model.
        let seeded = DefaultModels::load(&state).await.unwrap();
        assert_eq!(seeded, DefaultModels::default());

        let chat = model(&state, "openai", "gpt-4o", "language").await;
        let mut defaults = seeded;
        defaults.set_slot("default_chat_model", Some(chat.id.clone())).unwrap();
        defaults.set_slot("default_text_to_speech_model", Some("model:tts".into())).unwrap();
        defaults.save(&state).await.unwrap();

        let raw = state.db.select(DefaultModels::RECORD_ID).await.unwrap().unwrap();
        assert_eq!(raw["default_chat_model"], json!(chat.id));
        let mut loaded = DefaultModels::load(&state).await.unwrap();
        assert_eq!(loaded.default_chat_model.as_deref(), Some(chat.id.as_str()));

        // Clearing removes the field.
        loaded.set_slot("default_text_to_speech_model", None).unwrap();
        loaded.save(&state).await.unwrap();
        let raw = state.db.select(DefaultModels::RECORD_ID).await.unwrap().unwrap();
        assert!(raw.get("default_text_to_speech_model").is_none());

        // Resolution to records: chat directly, tools via the fallback, a
        // deleted model or an empty slot gives None.
        let found = DefaultModels::model_for(&state, DefaultModelType::Tools).await.unwrap().unwrap();
        assert_eq!(found.id, chat.id);
        assert!(DefaultModels::model_for(&state, DefaultModelType::Embedding).await.unwrap().is_none());
        loaded.set_slot("default_embedding_model", Some("model:deleted".into())).unwrap();
        loaded.save(&state).await.unwrap();
        assert!(DefaultModels::model_for(&state, DefaultModelType::Embedding).await.unwrap().is_none());

        // Values stored by Python as record links read back as ids too.
        state.db.upsert(DefaultModels::RECORD_ID, Doc::new().record("default_tools_model", &chat.id)).await.unwrap();
        assert_eq!(DefaultModels::load(&state).await.unwrap().default_tools_model.as_deref(), Some(chat.id.as_str()));
    }
}
