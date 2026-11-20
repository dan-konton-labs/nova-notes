//! Credentials: one record per AI provider account, API key encrypted at rest.
//!
//! The Rust counterpart of `open_notebook/domain/credential.py`, plus the
//! database-first / environment-fallback lookup of
//! `open_notebook/ai/key_provider.py` and `create_credential_from_env` from
//! `api/credentials_service.py`.
//!
//! A [`Credential`] holds its key decrypted in a [`SecretString`]; it is
//! written back encrypted by [`Credential::save`]. The type is deliberately
//! not `Serialize`, so a credential cannot be returned from an endpoint with
//! its key by accident, and its `Debug` output never shows the key.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::Map;

use super::model::Model;
use super::null_default;
use crate::crypto::{Cipher, SecretString, UNDECRYPTABLE_PLACEHOLDER};
use crate::db::{table_of, with_table, Doc, Json, Params};
use crate::error::{AppError, AppResult};
use crate::state::AppState;

/// Provider options that are top-level fields on [`Credential`] but are
/// stored inside the flexible `config` object (migration 15).
pub const CONFIG_EXTRAS: [&str; 1] = ["num_ctx"];

/// The message a credential carries when its stored key could not be decrypted.
pub const DECRYPTION_ERROR_MESSAGE: &str = "Failed to decrypt API key. The encryption key may have changed.";

/// Name given to credentials created from environment variables.
pub const ENV_CREDENTIAL_NAME: &str = "Default (Migrated from env)";

/// One provider account.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Credential {
    /// `credential:<id>`, empty until saved.
    pub id: String,
    pub name: String,
    pub provider: String,
    /// `language`, `embedding`, `speech_to_text`, `text_to_speech`.
    pub modalities: Vec<String>,
    /// The decrypted key. For a record whose key could not be decrypted (see
    /// [`get_all`](Self::get_all)) this is the `UNDECRYPTABLE` placeholder.
    pub api_key: Option<SecretString>,
    /// Set (never stored) when the key could not be decrypted.
    pub decryption_error: Option<String>,
    pub base_url: Option<String>,
    /// Azure OpenAI.
    pub endpoint: Option<String>,
    pub api_version: Option<String>,
    pub endpoint_llm: Option<String>,
    pub endpoint_embedding: Option<String>,
    pub endpoint_stt: Option<String>,
    pub endpoint_tts: Option<String>,
    /// Google Vertex AI.
    pub project: Option<String>,
    pub location: Option<String>,
    pub credentials_path: Option<String>,
    /// The stored `config` object as read. Keys this version does not know are
    /// kept and written back untouched; [`CONFIG_EXTRAS`] keys are replaced
    /// by their top-level fields on save.
    pub config: Option<Map<String, Json>>,
    /// Ollama context window override (stored as `config.num_ctx`).
    pub num_ctx: Option<i64>,
    pub created: Option<DateTime<Utc>>,
    pub updated: Option<DateTime<Utc>>,
}

/// A `credential` row as stored, key still encrypted.
#[derive(Deserialize)]
struct Row {
    #[serde(default)]
    id: String,
    #[serde(default, deserialize_with = "null_default")]
    name: String,
    #[serde(default, deserialize_with = "null_default")]
    provider: String,
    #[serde(default, deserialize_with = "null_default")]
    modalities: Vec<String>,
    #[serde(default)]
    api_key: Option<String>,
    #[serde(default)]
    base_url: Option<String>,
    #[serde(default)]
    endpoint: Option<String>,
    #[serde(default)]
    api_version: Option<String>,
    #[serde(default)]
    endpoint_llm: Option<String>,
    #[serde(default)]
    endpoint_embedding: Option<String>,
    #[serde(default)]
    endpoint_stt: Option<String>,
    #[serde(default)]
    endpoint_tts: Option<String>,
    #[serde(default)]
    project: Option<String>,
    #[serde(default)]
    location: Option<String>,
    #[serde(default)]
    credentials_path: Option<String>,
    #[serde(default)]
    config: Option<Map<String, Json>>,
    #[serde(default)]
    created: Option<DateTime<Utc>>,
    #[serde(default)]
    updated: Option<DateTime<Utc>>,
}

/// `num_ctx` from the config bag, validated as an integer the way the
/// Python model's field validation did: numbers with no fractional part and
/// numeric strings are accepted, anything else is rejected.
fn config_int(config: Option<&Map<String, Json>>, key: &str) -> AppResult<Option<i64>> {
    let Some(value) = config.and_then(|c| c.get(key)) else { return Ok(None) };
    let parsed = match value {
        Json::Null => return Ok(None),
        Json::Number(n) => n.as_i64().or_else(|| n.as_f64().filter(|f| f.fract() == 0.0).map(|f| f as i64)),
        Json::String(s) => s.trim().parse::<i64>().ok(),
        _ => None,
    };
    parsed.map(Some).ok_or_else(|| AppError::invalid(format!("{key} must be an integer")))
}

/// Python truthiness for an optional string: `None` and `""` are unset.
fn set(value: &Option<String>) -> Option<&str> {
    value.as_deref().filter(|v| !v.is_empty())
}

impl Credential {
    pub const TABLE: &'static str = "credential";

    fn from_row(row: Row, api_key: Option<SecretString>) -> AppResult<Self> {
        let num_ctx = config_int(row.config.as_ref(), "num_ctx")?;
        Ok(Self {
            id: row.id,
            name: row.name,
            provider: row.provider,
            modalities: row.modalities,
            api_key,
            decryption_error: None,
            base_url: row.base_url,
            endpoint: row.endpoint,
            api_version: row.api_version,
            endpoint_llm: row.endpoint_llm,
            endpoint_embedding: row.endpoint_embedding,
            endpoint_stt: row.endpoint_stt,
            endpoint_tts: row.endpoint_tts,
            project: row.project,
            location: row.location,
            credentials_path: row.credentials_path,
            config: row.config,
            num_ctx,
            created: row.created,
            updated: row.updated,
        })
    }

    /// Parse a database row, decrypting its key. A key that does not decrypt
    /// is an error (see [`crate::crypto::is_decryption_error`]).
    pub fn from_db_row(state: &AppState, row: Json) -> AppResult<Self> {
        let row: Row = serde_json::from_value(row)?;
        let api_key = match row.api_key.as_deref() {
            None => None,
            // An empty stored value is not ciphertext; Python kept it as is.
            Some("") => Some(SecretString::new("")),
            Some(stored) => Some(SecretString::new(Cipher::from_config(&state.config)?.decrypt(stored)?)),
        };
        Self::from_row(row, api_key)
    }

    /// The stand-in `get_all` returns for a row that cannot be read: name,
    /// provider and modalities only, an error message, and the placeholder
    /// key when the row had one (so `has_api_key` stays truthful).
    fn undecryptable(row: &Json) -> Self {
        let text = |key: &str| row.get(key).and_then(Json::as_str).map(String::from);
        let datetime = |key: &str| row.get(key).and_then(|v| serde_json::from_value(v.clone()).ok());
        let has_key = row.get("api_key").and_then(Json::as_str).is_some_and(|k| !k.is_empty());
        Self {
            id: text("id").unwrap_or_default(),
            name: text("name").unwrap_or_else(|| "Unknown".into()),
            provider: text("provider").unwrap_or_else(|| "unknown".into()),
            modalities: row
                .get("modalities")
                .and_then(|m| serde_json::from_value(m.clone()).ok())
                .unwrap_or_default(),
            api_key: has_key.then(|| SecretString::new(UNDECRYPTABLE_PLACEHOLDER)),
            decryption_error: Some(DECRYPTION_ERROR_MESSAGE.into()),
            created: datetime("created"),
            updated: datetime("updated"),
            ..Self::default()
        }
    }

    /// Load one credential, decrypting its key. Missing → `NotFound`; a key
    /// that does not decrypt → a decryption error; no encryption key
    /// configured (and the record has a key) → `Configuration`.
    pub async fn get(state: &AppState, id: &str) -> AppResult<Self> {
        let id = with_table(Self::TABLE, id);
        if table_of(&id) != Some(Self::TABLE) {
            return Err(AppError::not_found(format!("credential with id {id} not found")));
        }
        match state.db.select(&id).await? {
            Some(row) => Self::from_db_row(state, row),
            None => Err(AppError::not_found(format!("credential with id {id} not found"))),
        }
    }

    /// Every credential. A row that cannot be decrypted or parsed is not
    /// dropped: it comes back as a stand-in with
    /// [`decryption_error`](Self::decryption_error) set, so the UI can show it
    /// and offer to delete it.
    pub async fn get_all(state: &AppState, order_by: Option<&str>) -> AppResult<Vec<Self>> {
        let rows = state.db.select_all(Self::TABLE, order_by).await?;
        Ok(rows
            .into_iter()
            .map(|row| match Self::from_db_row(state, row.clone()) {
                Ok(credential) => credential,
                Err(e) => {
                    tracing::warn!(
                        "Failed to decrypt credential {}: {e}",
                        row.get("id").and_then(Json::as_str).unwrap_or("unknown")
                    );
                    Self::undecryptable(&row)
                }
            })
            .collect())
    }

    /// Credentials for a provider (case-insensitive), oldest first. Rows that
    /// cannot be decrypted are skipped.
    pub async fn get_by_provider(state: &AppState, provider: &str) -> AppResult<Vec<Self>> {
        let rows = state
            .db
            .query(
                "SELECT * FROM credential WHERE string::lowercase(provider) = string::lowercase($provider) ORDER BY created ASC",
                Params::new().set("provider", provider),
            )
            .await?;
        Ok(rows
            .into_iter()
            .filter_map(|row| match Self::from_db_row(state, row) {
                Ok(credential) => Some(credential),
                Err(e) => {
                    tracing::warn!("Skipping invalid credential: {e}");
                    None
                }
            })
            .collect())
    }

    /// The credential used for a provider when a model names none: the
    /// oldest readable one. Accepts `openai-compatible` as well as
    /// `openai_compatible`.
    pub async fn default_for_provider(state: &AppState, provider: &str) -> AppResult<Option<Self>> {
        let provider = nn_ai::normalize_provider_name(provider);
        Ok(Self::get_by_provider(state, &provider).await?.into_iter().next())
    }

    /// Models that use this credential.
    pub async fn linked_models(&self, state: &AppState) -> AppResult<Vec<Model>> {
        if self.id.is_empty() {
            return Ok(Vec::new());
        }
        Model::get_by_credential(state, &self.id).await
    }

    pub fn has_api_key(&self) -> bool {
        self.api_key.is_some()
    }

    /// Provider-specific required fields (the shared rule of the Python
    /// create and update paths): anthropic_compatible needs both a base URL
    /// and an API key.
    pub fn check_required_fields(&self) -> AppResult<()> {
        check_required_fields(&self.provider, self.base_url.as_deref(), self.api_key.as_ref().map(SecretString::expose))
    }

    /// The `config` object to store: the bag as read, with every
    /// [`CONFIG_EXTRAS`] key replaced by its field (or removed when unset).
    /// `None` when that leaves it empty.
    pub fn stored_config(&self) -> Option<Map<String, Json>> {
        let mut config = self.config.clone().unwrap_or_default();
        match self.num_ctx {
            Some(n) => config.insert("num_ctx".into(), Json::from(n)),
            None => config.remove("num_ctx"),
        };
        (!config.is_empty()).then_some(config)
    }

    fn doc(&self, cipher: Option<&Cipher>) -> AppResult<Doc> {
        let api_key = match (&self.api_key, cipher) {
            (Some(key), Some(cipher)) => Some(cipher.encrypt(key.expose())),
            (Some(_), None) => return Err(AppError::internal("credential key present without a cipher")),
            (None, _) => None,
        };
        let optional = |doc: Doc, key: &str, value: &Option<String>| match value {
            Some(v) => doc.set(key, v),
            None => doc.none(key),
        };
        let mut doc = Doc::new()
            .set("name", &self.name)
            .set("provider", &self.provider)
            .set("modalities", &self.modalities);
        doc = optional(doc, "api_key", &api_key);
        for (key, value) in [
            ("base_url", &self.base_url),
            ("endpoint", &self.endpoint),
            ("api_version", &self.api_version),
            ("endpoint_llm", &self.endpoint_llm),
            ("endpoint_embedding", &self.endpoint_embedding),
            ("endpoint_stt", &self.endpoint_stt),
            ("endpoint_tts", &self.endpoint_tts),
            ("project", &self.project),
            ("location", &self.location),
            ("credentials_path", &self.credentials_path),
        ] {
            doc = optional(doc, key, value);
        }
        // `UPDATE ... MERGE` merges nested objects key by key, so a key left out
        // of `config` would survive. An unset extra is written as NONE (null
        // becomes NONE), which removes it.
        Ok(match self.stored_config() {
            Some(mut config) => {
                for key in CONFIG_EXTRAS {
                    config.entry(key).or_insert(Json::Null);
                }
                doc.set("config", config)
            }
            None => doc.none("config"),
        })
    }

    /// Create (when `id` is empty) or update the credential, encrypting the
    /// key. Clearing a field (`None`) removes it from the record.
    ///
    /// Refuses to save a stand-in whose key could not be decrypted (that
    /// would overwrite a recoverable key with the placeholder), and a
    /// credential missing a provider-required field.
    pub async fn save(&mut self, state: &AppState) -> AppResult<()> {
        if self.decryption_error.is_some() {
            return Err(AppError::invalid("Cannot save a credential whose API key could not be decrypted"));
        }
        self.check_required_fields()?;
        let cipher = match self.api_key {
            Some(_) => Some(Cipher::from_config(&state.config)?),
            None => None,
        };
        let doc = self.doc(cipher.as_deref())?;
        let row = if self.id.is_empty() {
            state.db.create(Self::TABLE, doc).await?
        } else {
            state.db.update(&self.id, doc).await?
        };
        // Keep the decrypted key in memory; take everything else from the
        // stored record.
        let row: Row = serde_json::from_value(row)?;
        let api_key = self.api_key.take();
        *self = Self::from_row(row, api_key)?;
        Ok(())
    }

    /// Delete the record. Linked models are left alone; deciding what
    /// happens to them is the caller's job.
    pub async fn delete(&self, state: &AppState) -> AppResult<()> {
        if self.id.is_empty() {
            return Err(AppError::invalid("Cannot delete object without an ID"));
        }
        state.db.delete(&self.id).await
    }

    /// The configuration a provider client is built from (Python's
    /// `to_esperanto_config`): only the fields that are set, Azure's base URL
    /// doubling as its endpoint, and Vertex's project / location under their
    /// `vertex_` names.
    pub fn provider_config(&self) -> ResolvedProviderConfig {
        let provider = self.provider.to_ascii_lowercase();
        let owned = |v: &Option<String>| set(v).map(String::from);
        let is_vertex = provider == "vertex";
        let endpoint = owned(&self.endpoint)
            .or_else(|| (provider == "azure").then(|| owned(&self.base_url)).flatten());
        let extra = self
            .config
            .iter()
            .flatten()
            .filter(|(k, _)| !CONFIG_EXTRAS.contains(&k.as_str()))
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect();
        ResolvedProviderConfig {
            api_key: self.api_key.as_ref().filter(|k| !k.is_empty()).map(|k| k.expose().to_string()),
            base_url: owned(&self.base_url),
            endpoint,
            api_version: owned(&self.api_version),
            endpoint_llm: owned(&self.endpoint_llm),
            endpoint_embedding: owned(&self.endpoint_embedding),
            endpoint_stt: owned(&self.endpoint_stt),
            endpoint_tts: owned(&self.endpoint_tts),
            project: owned(&self.project).filter(|_| !is_vertex),
            location: owned(&self.location).filter(|_| !is_vertex),
            vertex_project: owned(&self.project).filter(|_| is_vertex),
            vertex_location: owned(&self.location).filter(|_| is_vertex),
            credentials_path: owned(&self.credentials_path),
            num_ctx: self.num_ctx,
            extra,
        }
    }
}

/// The required-field rule for providers that need both a base URL and an
/// API key (anthropic_compatible). `InvalidInput` with the Python message.
///
/// The Python create path raised this from request validation (422); the
/// update path returned it as a 400. The API layer picks the status.
pub fn check_required_fields(provider: &str, base_url: Option<&str>, api_key: Option<&str>) -> AppResult<()> {
    if provider.eq_ignore_ascii_case("anthropic_compatible") {
        if base_url.is_none_or(|u| u.trim().is_empty()) {
            return Err(AppError::invalid("Anthropic-compatible credentials require a base URL"));
        }
        if api_key.is_none_or(|k| k.trim().is_empty()) {
            return Err(AppError::invalid("Anthropic-compatible credentials require an API key"));
        }
    }
    Ok(())
}

/// Whether a provider name is one the credential API accepts: exactly a
/// registry name (`openai`, `openai_compatible`; not `OpenAI` or ` openai`).
pub fn is_supported_provider(provider: &str) -> bool {
    nn_ai::PROVIDERS.iter().any(|spec| spec.name == provider)
}

/// Modalities offered by default for a new credential of this provider.
pub fn default_modalities(provider: &str) -> Vec<String> {
    match nn_ai::provider(provider) {
        Some(spec) => spec.modality_names().into_iter().map(String::from).collect(),
        None => vec!["language".into()],
    }
}

// ---------------------------------------------------------------------------
// Resolved provider configuration
// ---------------------------------------------------------------------------

/// The settings a provider client is built from, resolved from a credential
/// or the environment: the key/value pairs of Python's
/// `Credential.to_esperanto_config()`. Serializing it gives exactly that dict
/// (unset fields are omitted), so it contains the API key in clear: never
/// return or log it. `Debug` redacts the key.
#[derive(Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ResolvedProviderConfig {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub api_key: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub base_url: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub endpoint: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub api_version: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub endpoint_llm: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub endpoint_embedding: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub endpoint_stt: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub endpoint_tts: Option<String>,
    /// Non-Vertex providers.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub project: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub location: Option<String>,
    /// Vertex (#1151: esperanto's Vertex providers take these names).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub vertex_project: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub vertex_location: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub credentials_path: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub num_ctx: Option<i64>,
    /// Other keys of the credential's `config` object (written by a newer
    /// version, or provider options such as `keep_alive`). Empty for every
    /// credential this version writes, so the serialized form matches Python.
    #[serde(default, skip_serializing_if = "Map::is_empty")]
    pub extra: Map<String, Json>,
}

impl std::fmt::Debug for ResolvedProviderConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ResolvedProviderConfig")
            .field("api_key", &self.api_key.as_ref().map(|_| "[redacted]"))
            .field("base_url", &self.base_url)
            .field("endpoint", &self.endpoint)
            .field("api_version", &self.api_version)
            .field("endpoint_llm", &self.endpoint_llm)
            .field("endpoint_embedding", &self.endpoint_embedding)
            .field("endpoint_stt", &self.endpoint_stt)
            .field("endpoint_tts", &self.endpoint_tts)
            .field("project", &self.project.as_ref().or(self.vertex_project.as_ref()))
            .field("location", &self.location.as_ref().or(self.vertex_location.as_ref()))
            .field("credentials_path", &self.credentials_path)
            .field("num_ctx", &self.num_ctx)
            .field("extra_keys", &self.extra.keys().collect::<Vec<_>>())
            .finish()
    }
}

impl ResolvedProviderConfig {
    /// The URL-valued settings, for re-validation right before a request
    /// (Python's `_URL_CONFIG_KEYS`).
    pub fn urls(&self) -> Vec<(&'static str, &str)> {
        [
            ("base_url", &self.base_url),
            ("endpoint", &self.endpoint),
            ("endpoint_llm", &self.endpoint_llm),
            ("endpoint_embedding", &self.endpoint_embedding),
            ("endpoint_stt", &self.endpoint_stt),
            ("endpoint_tts", &self.endpoint_tts),
        ]
        .into_iter()
        .filter_map(|(key, value)| value.as_deref().map(|v| (key, v)))
        .collect()
    }
}

impl From<ResolvedProviderConfig> for nn_ai::ProviderConfig {
    fn from(config: ResolvedProviderConfig) -> Self {
        let mut extra = config.extra;
        if let Some(num_ctx) = config.num_ctx {
            extra.insert("num_ctx".into(), Json::from(num_ctx));
        }
        nn_ai::ProviderConfig {
            api_key: config.api_key,
            base_url: config.base_url,
            endpoint: config.endpoint,
            api_version: config.api_version,
            endpoint_llm: config.endpoint_llm,
            endpoint_embedding: config.endpoint_embedding,
            endpoint_stt: config.endpoint_stt,
            endpoint_tts: config.endpoint_tts,
            project: config.vertex_project.or(config.project),
            location: config.vertex_location.or(config.location),
            credentials_path: config.credentials_path,
            extra,
            timeout: None,
        }
    }
}

// ---------------------------------------------------------------------------
// Database-first lookup with the (deprecated) environment fallback
// ---------------------------------------------------------------------------

/// An unsaved credential built from the provider's environment variables
/// (Python's `create_credential_from_env`), or `None` when the provider is
/// unknown or not configured there. `lookup` reads a variable
/// (`|name| std::env::var(name).ok()` in production). Variable names come
/// from the `nn_ai` registry.
pub fn credential_from_env(provider: &str, lookup: impl Fn(&str) -> Option<String>) -> Option<Credential> {
    let spec = nn_ai::provider(provider)?;
    if !spec.is_env_configured(&lookup) {
        return None;
    }
    let var = |name: &str| lookup(name).filter(|v| !v.is_empty());
    let mut credential = Credential {
        name: ENV_CREDENTIAL_NAME.into(),
        provider: spec.name.into(),
        modalities: default_modalities(spec.name),
        ..Credential::default()
    };
    match spec.name {
        "ollama" => credential.base_url = var("OLLAMA_API_BASE"),
        "omlx" => {
            credential.api_key = var("OMLX_API_KEY").map(SecretString::new);
            credential.base_url = var("OMLX_API_BASE");
        }
        "vertex" => {
            credential.project = var("VERTEX_PROJECT");
            credential.location = var("VERTEX_LOCATION");
            credential.credentials_path = var("GOOGLE_APPLICATION_CREDENTIALS");
        }
        "azure" => {
            credential.api_key = var("AZURE_OPENAI_API_KEY").map(SecretString::new);
            credential.endpoint = var("AZURE_OPENAI_ENDPOINT");
            credential.api_version = var("AZURE_OPENAI_API_VERSION");
            credential.endpoint_llm = var("AZURE_OPENAI_ENDPOINT_LLM");
            credential.endpoint_embedding = var("AZURE_OPENAI_ENDPOINT_EMBEDDING");
            credential.endpoint_stt = var("AZURE_OPENAI_ENDPOINT_STT");
            credential.endpoint_tts = var("AZURE_OPENAI_ENDPOINT_TTS");
        }
        "openai_compatible" | "anthropic_compatible" => {
            let prefix = spec.name.to_ascii_uppercase();
            credential.api_key = var(&format!("{prefix}_API_KEY")).map(SecretString::new);
            credential.base_url = var(&format!("{prefix}_BASE_URL"));
        }
        "google" => {
            credential.api_key = var("GOOGLE_API_KEY").or_else(|| var("GEMINI_API_KEY")).map(SecretString::new);
        }
        _ => {
            // Simple key providers, plus the optional *_BASE_URL endpoint
            // override for providers that declare one (ADR-012).
            credential.api_key = spec.required_env.first().and_then(|name| var(name)).map(SecretString::new);
            credential.base_url = spec
                .base_url_env()
                .and_then(&lookup)
                .map(|v| v.trim().to_string())
                .filter(|v| !v.is_empty());
        }
    }
    Some(credential)
}

/// The provider configuration from environment variables alone.
pub fn provider_config_from_env(
    provider: &str,
    lookup: impl Fn(&str) -> Option<String>,
) -> Option<ResolvedProviderConfig> {
    credential_from_env(provider, lookup).map(|c| c.provider_config())
}

/// Where a provider's configuration came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum ConfigSource {
    Database,
    Environment,
}

/// Resolve a provider's configuration: its default credential in the
/// database first, then the deprecated environment variables (read through
/// `lookup`). `None` when neither has it. A database failure falls through
/// to the environment, as in Python.
pub async fn resolve_provider_config_with(
    state: &AppState,
    provider: &str,
    lookup: impl Fn(&str) -> Option<String>,
) -> AppResult<Option<(ResolvedProviderConfig, ConfigSource)>> {
    match Credential::default_for_provider(state, provider).await {
        Ok(Some(credential)) => return Ok(Some((credential.provider_config(), ConfigSource::Database))),
        Ok(None) => {}
        Err(e) => tracing::debug!("Could not load credential from database for {provider}: {e}"),
    }
    Ok(provider_config_from_env(provider, lookup).map(|c| (c, ConfigSource::Environment)))
}

/// [`resolve_provider_config_with`] over the process environment.
pub async fn resolve_provider_config(
    state: &AppState,
    provider: &str,
) -> AppResult<Option<(ResolvedProviderConfig, ConfigSource)>> {
    resolve_provider_config_with(state, provider, |name| std::env::var(name).ok()).await
}

/// A record link field for queries that compare against credentials.
pub(crate) fn credential_link(id: &str) -> String {
    with_table(Credential::TABLE, id)
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use serde_json::json;

    use super::*;
    use crate::crypto::reference;

    async fn state() -> (AppState, tempfile::TempDir) {
        AppState::for_tests().await.unwrap()
    }

    fn env(vars: &[(&str, &str)]) -> impl Fn(&str) -> Option<String> {
        let vars: HashMap<String, String> = vars.iter().map(|(k, v)| (k.to_string(), v.to_string())).collect();
        move |name: &str| vars.get(name).cloned()
    }

    async fn raw(state: &AppState, id: &str) -> Json {
        state.db.select(id).await.unwrap().unwrap()
    }

    #[tokio::test]
    async fn create_read_update_delete_with_the_key_encrypted_at_rest() {
        let (state, _dir) = state().await;
        let mut cred = Credential {
            name: "Production".into(),
            provider: "openai".into(),
            modalities: vec!["language".into(), "embedding".into()],
            api_key: Some(SecretString::new("sk-live-secret-123")),
            base_url: Some("https://api.example.com/v1".into()),
            ..Credential::default()
        };
        cred.save(&state).await.unwrap();
        assert!(cred.id.starts_with("credential:"));
        assert!(cred.created.is_some());
        assert_eq!(cred.api_key.as_ref().unwrap().expose(), "sk-live-secret-123");

        // At rest: marked ciphertext, never the key.
        let row = raw(&state, &cred.id).await;
        let stored = row["api_key"].as_str().unwrap();
        assert!(stored.starts_with(crate::crypto::PBKDF2_MARKER));
        assert!(!row.to_string().contains("sk-live-secret-123"));
        assert!(row.get("config").is_none());

        // Read back decrypted.
        let mut again = Credential::get(&state, &cred.id).await.unwrap();
        assert_eq!(again.api_key.as_ref().unwrap().expose(), "sk-live-secret-123");
        assert_eq!(again.modalities, ["language", "embedding"]);
        assert_eq!(again.base_url.as_deref(), Some("https://api.example.com/v1"));
        // The bare id works too.
        let bare = cred.id.trim_start_matches("credential:");
        assert_eq!(Credential::get(&state, bare).await.unwrap().id, cred.id);

        // Debug never shows the key.
        assert!(!format!("{again:?}").contains("sk-live-secret-123"));
        assert!(!format!("{:?}", again.provider_config()).contains("sk-live-secret-123"));

        // Saving again does not re-wrap the key, and clearing a field removes it.
        again.name = "Renamed".into();
        again.base_url = None;
        again.save(&state).await.unwrap();
        let row = raw(&state, &cred.id).await;
        assert!(row.get("base_url").is_none());
        assert_eq!(row["name"], "Renamed");
        let reread = Credential::get(&state, &cred.id).await.unwrap();
        assert_eq!(reread.api_key.as_ref().unwrap().expose(), "sk-live-secret-123");
        assert_eq!(reread.created, cred.created);

        // A new key replaces the old one; removing the key clears it.
        again.api_key = Some(SecretString::new("sk-rotated"));
        again.save(&state).await.unwrap();
        assert_eq!(Credential::get(&state, &cred.id).await.unwrap().api_key.unwrap().expose(), "sk-rotated");
        again.api_key = None;
        again.save(&state).await.unwrap();
        assert!(raw(&state, &cred.id).await.get("api_key").is_none());
        assert!(!Credential::get(&state, &cred.id).await.unwrap().has_api_key());

        again.delete(&state).await.unwrap();
        assert!(Credential::get(&state, &cred.id).await.unwrap_err().is_not_found());
        assert!(Credential::get(&state, "model:abc").await.unwrap_err().is_not_found());
        assert!(Credential::default().delete(&state).await.is_err());
    }

    #[tokio::test]
    async fn rows_written_by_python_decrypt() {
        let (state, _dir) = state().await;
        let marked = state
            .db
            .create(
                "credential",
                Doc::new().set("name", "From Python").set("provider", "OpenAI").set("api_key", reference::MARKED),
            )
            .await
            .unwrap();
        let legacy = state
            .db
            .create(
                "credential",
                Doc::new().set("name", "Legacy").set("provider", "openai").set("api_key", reference::LEGACY),
            )
            .await
            .unwrap();
        let plain = state
            .db
            .create("credential", Doc::new().set("name", "Plain").set("provider", "openai").set("api_key", "sk-plain"))
            .await
            .unwrap();
        let get = |row: &Json| row["id"].as_str().unwrap().to_string();
        let key = |c: Credential| c.api_key.unwrap().expose().to_string();
        assert_eq!(key(Credential::get(&state, &get(&marked)).await.unwrap()), reference::MARKED_PLAINTEXT);
        assert_eq!(key(Credential::get(&state, &get(&legacy)).await.unwrap()), reference::LEGACY_PLAINTEXT);
        assert_eq!(key(Credential::get(&state, &get(&plain)).await.unwrap()), "sk-plain");

        // Case-insensitive lookup, oldest first.
        let by_provider = Credential::get_by_provider(&state, "openai").await.unwrap();
        assert_eq!(by_provider.iter().map(|c| c.name.as_str()).collect::<Vec<_>>(), ["From Python", "Legacy", "Plain"]);
        assert_eq!(Credential::default_for_provider(&state, "OPENAI").await.unwrap().unwrap().name, "From Python");
    }

    #[tokio::test]
    async fn undecryptable_rows_are_reported_not_leaked() {
        let (state, _dir) = state().await;
        state
            .db
            .create("credential", Doc::new().set("name", "Good").set("provider", "anthropic").set("api_key", "sk-ok"))
            .await
            .unwrap();
        let bad = state
            .db
            .create(
                "credential",
                Doc::new()
                    .set("name", "Foreign")
                    .set("provider", "anthropic")
                    .set("modalities", ["language"])
                    .set("base_url", "https://x.example")
                    .set("api_key", reference::MARKED_OTHER_KEY),
            )
            .await
            .unwrap();
        let bad_id = bad["id"].as_str().unwrap();

        let error = Credential::get(&state, bad_id).await.unwrap_err();
        assert!(crate::crypto::is_decryption_error(&error), "{error:?}");
        assert!(!error.to_string().contains("gAAAA"));

        // get_by_provider skips it; get_all keeps a stand-in.
        let by_provider = Credential::get_by_provider(&state, "anthropic").await.unwrap();
        assert_eq!(by_provider.len(), 1);
        let all = Credential::get_all(&state, Some("name asc")).await.unwrap();
        assert_eq!(all.len(), 2);
        let mut stand_in = all.into_iter().find(|c| c.id == bad_id).unwrap();
        assert_eq!(stand_in.decryption_error.as_deref(), Some(DECRYPTION_ERROR_MESSAGE));
        assert_eq!(stand_in.name, "Foreign");
        assert_eq!(stand_in.modalities, ["language"]);
        assert!(stand_in.has_api_key());
        assert_eq!(stand_in.api_key.as_ref().unwrap().expose(), UNDECRYPTABLE_PLACEHOLDER);
        assert!(stand_in.created.is_some());

        // Saving the stand-in would destroy the recoverable ciphertext.
        assert!(stand_in.save(&state).await.is_err());
        assert_eq!(raw(&state, bad_id).await["api_key"], reference::MARKED_OTHER_KEY);
        // It can still be deleted.
        stand_in.delete(&state).await.unwrap();
        assert!(state.db.select(bad_id).await.unwrap().is_none());
    }

    #[tokio::test]
    async fn missing_encryption_key() {
        let (mut state, _dir) = state().await;
        let mut config = (*state.config).clone();
        config.encryption_key = None;
        state.config = std::sync::Arc::new(config);

        // No key needed for a credential without one.
        let mut keyless = Credential { name: "Local".into(), provider: "ollama".into(), ..Credential::default() };
        keyless.save(&state).await.unwrap();
        assert_eq!(Credential::get(&state, &keyless.id).await.unwrap().name, "Local");

        let mut with_key = Credential {
            name: "Cloud".into(),
            provider: "openai".into(),
            api_key: Some(SecretString::new("sk-x")),
            ..Credential::default()
        };
        let error = with_key.save(&state).await.unwrap_err();
        assert!(matches!(error, AppError::Configuration(_)), "{error:?}");
        assert!(error.to_string().starts_with("OPEN_NOTEBOOK_ENCRYPTION_KEY is not set."));
        assert!(with_key.id.is_empty());
    }

    #[tokio::test]
    async fn config_bag_round_trips_and_keeps_unknown_keys() {
        let (state, _dir) = state().await;
        let mut cred = Credential {
            name: "Local Ollama".into(),
            provider: "ollama".into(),
            num_ctx: Some(16384),
            ..Credential::default()
        };
        assert_eq!(cred.stored_config(), Some(json!({"num_ctx": 16384}).as_object().unwrap().clone()));
        cred.save(&state).await.unwrap();
        let row = raw(&state, &cred.id).await;
        assert_eq!(row["config"], json!({"num_ctx": 16384}));
        assert!(row.get("num_ctx").is_none());
        assert_eq!(Credential::get(&state, &cred.id).await.unwrap().num_ctx, Some(16384));

        // A key written by a newer version survives; clearing num_ctx keeps it.
        state
            .db
            .update(&cred.id, Doc::new().set("config", json!({"num_ctx": "8192", "future_option": "keep-me"})))
            .await
            .unwrap();
        let mut loaded = Credential::get(&state, &cred.id).await.unwrap();
        assert_eq!(loaded.num_ctx, Some(8192), "numeric strings are coerced like pydantic");
        loaded.num_ctx = None;
        loaded.save(&state).await.unwrap();
        assert_eq!(raw(&state, &cred.id).await["config"], json!({"future_option": "keep-me"}));

        // Nothing left: the field is cleared, not stored as {}.
        let mut plain = Credential { config: None, ..loaded.clone() };
        plain.save(&state).await.unwrap();
        assert!(raw(&state, &cred.id).await.get("config").is_none());

        // A value that is not an integer is rejected.
        state.db.update(&cred.id, Doc::new().set("config", json!({"num_ctx": "not-an-int"}))).await.unwrap();
        assert!(Credential::get(&state, &cred.id).await.is_err());
        assert_eq!(config_int(json!({"n": 4096.0}).as_object(), "n").unwrap(), Some(4096));
        assert!(config_int(json!({"n": 1.5}).as_object(), "n").is_err());
    }

    #[test]
    fn provider_config_matches_to_esperanto_config() {
        let ollama = Credential {
            provider: "ollama".into(),
            base_url: Some("http://localhost:11434".into()),
            num_ctx: Some(32768),
            ..Credential::default()
        };
        assert_eq!(
            serde_json::to_value(ollama.provider_config()).unwrap(),
            json!({"base_url": "http://localhost:11434", "num_ctx": 32768})
        );
        let unset = Credential { num_ctx: None, ..ollama };
        assert!(serde_json::to_value(unset.provider_config()).unwrap().get("num_ctx").is_none());

        let vertex = Credential {
            provider: "vertex".into(),
            project: Some("my-gcp-project".into()),
            location: Some("us-central1".into()),
            credentials_path: Some("/secrets/sa.json".into()),
            ..Credential::default()
        };
        assert_eq!(
            serde_json::to_value(vertex.provider_config()).unwrap(),
            json!({"vertex_project": "my-gcp-project", "vertex_location": "us-central1", "credentials_path": "/secrets/sa.json"})
        );
        let other = Credential { provider: "openai".into(), ..vertex };
        assert_eq!(
            serde_json::to_value(other.provider_config()).unwrap(),
            json!({"project": "my-gcp-project", "location": "us-central1", "credentials_path": "/secrets/sa.json"})
        );

        // Azure: the form's base URL doubles as the endpoint unless one is set.
        let azure = Credential {
            provider: "Azure".into(),
            api_key: Some(SecretString::new("az-key")),
            base_url: Some("https://x.openai.azure.com".into()),
            api_version: Some("2024-02-01".into()),
            ..Credential::default()
        };
        assert_eq!(
            serde_json::to_value(azure.provider_config()).unwrap(),
            json!({"api_key": "az-key", "base_url": "https://x.openai.azure.com", "endpoint": "https://x.openai.azure.com", "api_version": "2024-02-01"})
        );
        let explicit = Credential { endpoint: Some("https://y.openai.azure.com".into()), ..azure };
        assert_eq!(explicit.provider_config().endpoint.as_deref(), Some("https://y.openai.azure.com"));
        // Empty strings are unset, as in Python.
        let blank = Credential { provider: "openai".into(), base_url: Some(String::new()), ..Credential::default() };
        assert_eq!(serde_json::to_value(blank.provider_config()).unwrap(), json!({}));
    }

    #[test]
    fn converts_into_the_nn_ai_config() {
        let vertex = Credential {
            provider: "vertex".into(),
            project: Some("p".into()),
            location: Some("us-central1".into()),
            config: json!({"num_ctx": 1, "task_type": "RETRIEVAL_DOCUMENT"}).as_object().cloned(),
            num_ctx: Some(2048),
            ..Credential::default()
        };
        let resolved = vertex.provider_config();
        assert_eq!(resolved.extra, *json!({"task_type": "RETRIEVAL_DOCUMENT"}).as_object().unwrap());
        let config: nn_ai::ProviderConfig = resolved.into();
        assert_eq!(config.project.as_deref(), Some("p"));
        assert_eq!(config.location.as_deref(), Some("us-central1"));
        assert_eq!(config.extra["num_ctx"], 2048);
        assert_eq!(config.extra["task_type"], "RETRIEVAL_DOCUMENT");

        let keyed = Credential {
            provider: "openai".into(),
            api_key: Some(SecretString::new("sk-k")),
            base_url: Some("https://api.example.com/v1".into()),
            ..Credential::default()
        };
        let config: nn_ai::ProviderConfig = keyed.provider_config().into();
        assert_eq!(config.api_key.as_deref(), Some("sk-k"));
        assert_eq!(config.base_url.as_deref(), Some("https://api.example.com/v1"));
        assert_eq!(config.project, None);
    }

    #[test]
    fn provider_validation() {
        for provider in ["openai", "anthropic", "ollama", "omlx", "azure", "vertex", "openai_compatible", "anthropic_compatible"] {
            assert!(is_supported_provider(provider), "{provider}");
        }
        for provider in ["openai ", " openai", "OpenAI", "opnai", "not_a_real_provider", "", "sqlite"] {
            assert!(!is_supported_provider(provider), "{provider:?}");
        }
        for (key, url) in [(None, Some("https://api.example.com")), (Some("sk-test"), None), (Some(" "), Some(" "))] {
            let error = check_required_fields("anthropic_compatible", url, key).unwrap_err();
            assert_eq!(error.status(), axum::http::StatusCode::BAD_REQUEST);
        }
        assert!(check_required_fields("anthropic_compatible", Some("https://a.example"), Some("sk")).is_ok());
        assert!(check_required_fields("openai", None, None).is_ok());
        assert_eq!(default_modalities("voyage"), ["embedding"]);
        assert_eq!(default_modalities("nope"), ["language"]);
    }

    #[tokio::test]
    async fn anthropic_compatible_requires_both_fields_on_save() {
        let (state, _dir) = state().await;
        let mut cred = Credential {
            name: "Proxy".into(),
            provider: "anthropic_compatible".into(),
            base_url: Some("https://proxy.example".into()),
            ..Credential::default()
        };
        assert_eq!(
            cred.save(&state).await.unwrap_err().to_string(),
            "Anthropic-compatible credentials require an API key"
        );
        assert!(Credential::get_all(&state, None).await.unwrap().is_empty());
    }

    #[test]
    fn env_fallback_builds_credentials_from_registry_names() {
        let sf = credential_from_env(
            "siliconflow",
            env(&[("SILICONFLOW_API_KEY", "sf-key"), ("SILICONFLOW_BASE_URL", "https://api.siliconflow.cn/v1")]),
        )
        .unwrap();
        assert_eq!(sf.name, ENV_CREDENTIAL_NAME);
        assert_eq!(sf.api_key.unwrap().expose(), "sf-key");
        assert_eq!(sf.base_url.as_deref(), Some("https://api.siliconflow.cn/v1"));

        for value in [None, Some(""), Some("   ")] {
            let mut vars = vec![("ZAI_API_KEY", "zai-key")];
            if let Some(v) = value {
                vars.push(("ZAI_BASE_URL", v));
            }
            assert_eq!(credential_from_env("zai", env(&vars)).unwrap().base_url, None);
        }
        let zai = credential_from_env(
            "zai",
            env(&[("ZAI_API_KEY", "k"), ("ZAI_BASE_URL", " https://open.bigmodel.cn/api/paas/v4 ")]),
        )
        .unwrap();
        assert_eq!(zai.base_url.as_deref(), Some("https://open.bigmodel.cn/api/paas/v4"));

        // A provider without a declared override ignores a stray base URL.
        let openai =
            credential_from_env("openai", env(&[("OPENAI_API_KEY", "sk"), ("OPENAI_BASE_URL", "https://x")])).unwrap();
        assert_eq!(openai.base_url, None);

        let google = credential_from_env("google", env(&[("GEMINI_API_KEY", "gem")])).unwrap();
        assert_eq!(google.api_key.unwrap().expose(), "gem");

        let vertex = provider_config_from_env(
            "vertex",
            env(&[("VERTEX_PROJECT", "p"), ("VERTEX_LOCATION", "eu"), ("GOOGLE_APPLICATION_CREDENTIALS", "/sa.json")]),
        )
        .unwrap();
        assert_eq!(
            serde_json::to_value(vertex).unwrap(),
            json!({"vertex_project": "p", "vertex_location": "eu", "credentials_path": "/sa.json"})
        );

        let azure = credential_from_env(
            "azure",
            env(&[
                ("AZURE_OPENAI_API_KEY", "az"),
                ("AZURE_OPENAI_ENDPOINT", "https://e"),
                ("AZURE_OPENAI_API_VERSION", "v"),
                ("AZURE_OPENAI_ENDPOINT_TTS", "https://tts"),
            ]),
        )
        .unwrap();
        assert_eq!(azure.endpoint.as_deref(), Some("https://e"));
        assert_eq!(azure.endpoint_tts.as_deref(), Some("https://tts"));

        let compat = credential_from_env(
            "openai-compatible",
            env(&[("OPENAI_COMPATIBLE_BASE_URL", "http://localhost:1234/v1")]),
        )
        .unwrap();
        assert_eq!(compat.provider, "openai_compatible");
        assert_eq!(compat.base_url.as_deref(), Some("http://localhost:1234/v1"));
        assert!(compat.api_key.is_none());

        let ollama = credential_from_env("ollama", env(&[("OLLAMA_API_BASE", "http://ollama:11434")])).unwrap();
        assert_eq!(ollama.base_url.as_deref(), Some("http://ollama:11434"));
        assert_eq!(ollama.modalities, ["language", "embedding"]);

        // Not configured, partly configured (azure needs all three) or unknown.
        assert!(credential_from_env("openai", env(&[("OPENAI_API_KEY", "  ")])).is_none());
        assert!(credential_from_env("azure", env(&[("AZURE_OPENAI_API_KEY", "az")])).is_none());
        assert!(credential_from_env("nope", env(&[])).is_none());
    }

    #[tokio::test]
    async fn database_first_then_environment() {
        let (state, _dir) = state().await;
        let lookup = env(&[("GROQ_API_KEY", "env-key")]);
        let (config, source) = resolve_provider_config_with(&state, "groq", &lookup).await.unwrap().unwrap();
        assert_eq!(source, ConfigSource::Environment);
        assert_eq!(config.api_key.as_deref(), Some("env-key"));
        assert!(resolve_provider_config_with(&state, "mistral", &lookup).await.unwrap().is_none());

        let mut cred = Credential {
            name: "DB".into(),
            provider: "groq".into(),
            api_key: Some(SecretString::new("db-key")),
            ..Credential::default()
        };
        cred.save(&state).await.unwrap();
        let (config, source) = resolve_provider_config_with(&state, "Groq", &lookup).await.unwrap().unwrap();
        assert_eq!(source, ConfigSource::Database);
        assert_eq!(config.api_key.as_deref(), Some("db-key"));
    }

    #[test]
    fn credential_links_take_the_table() {
        assert_eq!(credential_link("abc"), "credential:abc");
        assert_eq!(credential_link("credential:abc"), "credential:abc");
    }
}
