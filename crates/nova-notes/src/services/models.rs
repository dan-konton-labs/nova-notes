//! Configured models, the default-model slots, provider availability, and
//! provider-wide model discovery and sync.
//!
//! The Rust counterpart of `api/routers/models.py` plus the database side of
//! `open_notebook/ai/model_discovery.py` (syncing discovered models into the
//! `model` table, counts per provider) and `test_individual_model` from
//! `open_notebook/ai/connection_tester.py`. Listing a provider's models and
//! calling it are `nn_ai`'s job.
//!
//! Provider-wide discovery uses the provider's default credential, falling
//! back to its environment variables, as Python's `provision_provider_keys`
//! did. Functions that read the environment have a `_with` form taking the
//! lookup.

use std::collections::BTreeMap;

use axum::http::StatusCode;
use serde::{Deserialize, Deserializer, Serialize};

use super::credentials::{
    credential_provider, discover_with_config, existing_model_keys, process_env, provider_mismatch, truncate, EnvLookup,
};
use crate::ai::net::prepare_client_config;
use crate::domain::credential::resolve_provider_config_with;
use crate::domain::model::{all_models_by_provider, auto_assign_required, is_valid_model_type, MODEL_TYPES};
use crate::domain::{Credential, DefaultModels, Model};
use crate::error::{AppError, AppResult};
use crate::jobs::JobQueue;
use crate::services::notebooks::timestamp;
use crate::state::AppState;

/// Models have no background jobs (the Python build has no commands for
/// them).
pub fn register_jobs(_jobs: &JobQueue) {}

/// A 500 carrying the error text, the way the Python models router reported
/// unexpected failures (`f"Error fetching models: {str(e)}"`). Other
/// statuses pass through.
fn with_context(context: &'static str) -> impl Fn(AppError) -> AppError {
    move |e| {
        if e.status() == StatusCode::INTERNAL_SERVER_ERROR && !matches!(e, AppError::Http { .. }) {
            tracing::error!("{context}: {e}");
            AppError::http(StatusCode::INTERNAL_SERVER_ERROR, format!("{context}: {e}"))
        } else {
            e
        }
    }
}

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// A model as the API returns it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModelResponse {
    pub id: String,
    pub name: String,
    pub provider: String,
    #[serde(rename = "type")]
    pub model_type: String,
    pub credential: Option<String>,
    pub created: String,
    pub updated: String,
}

impl From<&Model> for ModelResponse {
    fn from(model: &Model) -> Self {
        Self {
            id: model.id.clone(),
            name: model.name.clone(),
            provider: model.provider.clone(),
            model_type: model.model_type.clone(),
            credential: model.credential.clone(),
            created: timestamp(model.created),
            updated: timestamp(model.updated),
        }
    }
}

/// `POST /models` (`ModelCreate`).
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct NewModel {
    pub name: String,
    pub provider: String,
    #[serde(rename = "type")]
    pub model_type: String,
    /// `credential:<id>` to run the model with.
    #[serde(default)]
    pub credential: Option<String>,
}

/// `POST /models/{id}/test`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModelTestResult {
    pub success: bool,
    pub message: String,
    pub details: Option<String>,
}

fn present<'de, D: Deserializer<'de>>(deserializer: D) -> Result<Option<Option<String>>, D::Error> {
    Option::<String>::deserialize(deserializer).map(Some)
}

/// `PUT /models/defaults`: keyed on presence. An absent slot (`None`) is
/// left alone, an explicit null (`Some(None)`) clears it, a value assigns it.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct DefaultModelsUpdate {
    #[serde(default, deserialize_with = "present", skip_serializing_if = "Option::is_none")]
    pub default_chat_model: Option<Option<String>>,
    #[serde(default, deserialize_with = "present", skip_serializing_if = "Option::is_none")]
    pub default_transformation_model: Option<Option<String>>,
    #[serde(default, deserialize_with = "present", skip_serializing_if = "Option::is_none")]
    pub large_context_model: Option<Option<String>>,
    #[serde(default, deserialize_with = "present", skip_serializing_if = "Option::is_none")]
    pub default_text_to_speech_model: Option<Option<String>>,
    #[serde(default, deserialize_with = "present", skip_serializing_if = "Option::is_none")]
    pub default_speech_to_text_model: Option<Option<String>>,
    #[serde(default, deserialize_with = "present", skip_serializing_if = "Option::is_none")]
    pub default_embedding_model: Option<Option<String>>,
    #[serde(default, deserialize_with = "present", skip_serializing_if = "Option::is_none")]
    pub default_tools_model: Option<Option<String>>,
}

impl DefaultModelsUpdate {
    /// `(slot, value)` for every slot present in the update, in the order of
    /// [`DefaultModels::SLOTS`].
    fn present_slots(&self) -> Vec<(&'static str, Option<String>)> {
        [
            ("default_chat_model", &self.default_chat_model),
            ("default_transformation_model", &self.default_transformation_model),
            ("large_context_model", &self.large_context_model),
            ("default_text_to_speech_model", &self.default_text_to_speech_model),
            ("default_speech_to_text_model", &self.default_speech_to_text_model),
            ("default_embedding_model", &self.default_embedding_model),
            ("default_tools_model", &self.default_tools_model),
        ]
        .into_iter()
        .filter_map(|(slot, value)| value.as_ref().map(|v| (slot, v.clone())))
        .collect()
    }
}

/// `GET /models/providers`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProviderAvailability {
    pub available: Vec<String>,
    pub unavailable: Vec<String>,
    /// Model types each available provider supports.
    pub supported_types: BTreeMap<String, Vec<String>>,
}

/// A model a provider reports, with its guessed type.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DiscoveredModelInfo {
    pub name: String,
    pub provider: String,
    pub model_type: String,
    pub description: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProviderSyncResult {
    pub provider: String,
    pub discovered: usize,
    pub new: usize,
    pub existing: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AllProvidersSyncResult {
    pub results: BTreeMap<String, ProviderSyncResult>,
    pub total_discovered: usize,
    pub total_new: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProviderModelCount {
    pub provider: String,
    pub counts: BTreeMap<String, i64>,
    pub total: i64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AutoAssignResult {
    /// Slot name → assigned model id.
    pub assigned: BTreeMap<String, String>,
    /// Slots that already had a model.
    pub skipped: Vec<String>,
    /// Slots with no model of their type to choose from.
    pub missing: Vec<String>,
}

// ---------------------------------------------------------------------------
// CRUD
// ---------------------------------------------------------------------------

/// Every model, or those of one type.
pub async fn list_models(state: &AppState, model_type: Option<&str>) -> AppResult<Vec<ModelResponse>> {
    let models = match model_type.filter(|t| !t.is_empty()) {
        Some(model_type) => Model::get_by_type(state, model_type).await,
        None => Model::get_all(state, None).await,
    }
    .map_err(with_context("Error fetching models"))?;
    Ok(models.iter().map(ModelResponse::from).collect())
}

/// The message for a model type outside [`MODEL_TYPES`] (Python printed
/// its list).
fn invalid_type_message() -> String {
    let types: Vec<String> = MODEL_TYPES.iter().map(|t| format!("'{t}'")).collect();
    format!("Invalid model type. Must be one of: [{}]", types.join(", "))
}

/// Create a model. The type must be valid, and provider + name + type must
/// not already exist (case-insensitive).
pub async fn create_model(state: &AppState, input: &NewModel) -> AppResult<ModelResponse> {
    if !is_valid_model_type(&input.model_type) {
        return Err(AppError::bad_request(invalid_type_message()));
    }
    // A model sends its credential's key to the model's provider: the two
    // must match.
    if let Some(credential) = input.credential.as_deref().filter(|c| !c.is_empty()) {
        if let Some(provider) = credential_provider(state, credential).await.map_err(with_context("Error creating model"))? {
            if provider != nn_ai::normalize_provider_name(&input.provider) {
                return Err(provider_mismatch(&provider, &input.provider));
            }
        }
    }
    let result = async {
        if Model::find_duplicate(state, &input.provider, &input.name, &input.model_type).await?.is_some() {
            return Err(AppError::bad_request(format!(
                "Model '{}' already exists for provider '{}' with type '{}'",
                input.name, input.provider, input.model_type
            )));
        }
        let mut model = Model {
            name: input.name.clone(),
            provider: input.provider.clone(),
            model_type: input.model_type.clone(),
            credential: input.credential.clone().filter(|c| !c.is_empty()),
            ..Model::default()
        };
        model.save(state).await?;
        Ok(ModelResponse::from(&model))
    };
    result.await.map_err(with_context("Error creating model"))
}

async fn find_model(state: &AppState, id: &str) -> AppResult<Model> {
    match Model::get(state, id).await {
        Ok(model) => Ok(model),
        Err(e @ AppError::Database { .. }) => Err(e),
        Err(_) => Err(AppError::not_found("Model not found")),
    }
}

/// Delete a model. Missing → 404 "Model not found".
pub async fn delete_model(state: &AppState, id: &str) -> AppResult<()> {
    let result = async {
        let model = find_model(state, id).await?;
        model.delete(state).await
    };
    result.await.map_err(with_context("Error deleting model"))
}

/// The client configuration a model runs with (its credential, else the
/// provider default credential, else the environment), URLs re-validated.
async fn model_client_config(state: &AppState, model: &Model) -> AppResult<nn_ai::ProviderConfig> {
    prepare_client_config(&state.config.providers, &model.provider, model.provider_config(state).await?)
        .await
        .map_err(|e| AppError::config(format!("The endpoint configured for {} is not allowed: {e}", model.provider)))
}

/// Test one model end to end with a real call. Missing → 404; every other
/// outcome is a result with a user-facing message.
pub async fn test_model(state: &AppState, id: &str) -> AppResult<ModelTestResult> {
    let model = Model::get(state, id).await.map_err(|_| AppError::not_found("Model not found"))?;
    let (success, message) = if !is_valid_model_type(&model.model_type) {
        nn_ai::tester::normalize_error_message(&format!("Invalid model type: {}", model.model_type))
    } else {
        match model_client_config(state, &model).await {
            Ok(config) => {
                let result =
                    state.models.clients().test_model(&model.provider, &model.name, &model.model_type, &config).await;
                (result.success, result.message)
            }
            Err(e) => nn_ai::tester::normalize_error_message(&e.to_string()),
        }
    };
    if !success {
        tracing::debug!("Test individual model error for {id}: {message}");
    }
    Ok(ModelTestResult { success, message: truncate(&message, 200), details: None })
}

// ---------------------------------------------------------------------------
// Defaults
// ---------------------------------------------------------------------------

/// The default-model slots.
pub async fn get_default_models(state: &AppState) -> AppResult<DefaultModels> {
    DefaultModels::load(state).await.map_err(with_context("Error fetching default models"))
}

/// Apply a presence-keyed update to the slots. Clearing a required slot
/// (chat, embedding) is a 400 and nothing is saved.
pub async fn update_default_models(state: &AppState, update: &DefaultModelsUpdate) -> AppResult<DefaultModels> {
    let result = async {
        let mut defaults = DefaultModels::load(state).await?;
        for (slot, value) in update.present_slots() {
            defaults.set_slot(slot, value)?;
        }
        defaults.save(state).await
    };
    result.await.map_err(with_context("Error updating default models"))
}

/// Providers tried first when auto-assigning, best first.
pub const PROVIDER_PRIORITY: [&str; 13] = [
    "openai",
    "anthropic",
    "google",
    "mistral",
    "groq",
    "deepseek",
    "xai",
    "openrouter",
    "ollama",
    "azure",
    "openai_compatible",
    "dashscope",
    "minimax",
];

/// Preferred model-name patterns within a provider, best first.
pub fn model_preferences(provider: &str) -> &'static [&'static str] {
    match provider {
        "openai" => &["gpt-4o", "gpt-4", "gpt-3.5-turbo"],
        "anthropic" => &["claude-3-5-sonnet", "claude-3-opus", "claude-3-sonnet"],
        "google" => &["gemini-3.5-flash", "gemini-2.5-flash", "gemini-2.5-pro"],
        "mistral" => &["mistral-large", "mixtral"],
        "groq" => &["llama-3.3", "llama-3.1", "mixtral"],
        "dashscope" => &["qwen-max", "qwen-plus", "qwen-turbo"],
        "minimax" => &["MiniMax-M3", "MiniMax-M2.5", "MiniMax-M2.5-highspeed"],
        _ => &[],
    }
}

/// Python's `_get_preferred_model`: the first provider in priority order
/// that has candidates, its preferred model (by name pattern) or else its
/// first; failing that the first candidate.
pub fn preferred_model(candidates: &[&Model]) -> Option<String> {
    for provider in PROVIDER_PRIORITY {
        let models: Vec<&&Model> = candidates.iter().filter(|m| m.provider == provider).collect();
        let Some(first) = models.first() else { continue };
        for pattern in model_preferences(provider) {
            let pattern = pattern.to_lowercase();
            if let Some(model) = models.iter().find(|m| m.name.to_lowercase().contains(&pattern)) {
                return Some(model.id.clone());
            }
        }
        return Some(first.id.clone());
    }
    candidates.first().map(|m| m.id.clone())
}

/// Fill the empty required slots (chat, embedding) from the configured
/// models; optional slots are never touched.
pub async fn auto_assign_defaults(state: &AppState) -> AppResult<AutoAssignResult> {
    let result = async {
        let mut defaults = DefaultModels::load(state).await?;
        let models = all_models_by_provider(state).await?;
        let (assigned, skipped, missing) = auto_assign_required(&mut defaults, &models, preferred_model);
        if !assigned.is_empty() {
            defaults.save(state).await?;
        }
        Ok(AutoAssignResult { assigned: assigned.into_iter().collect(), skipped, missing })
    };
    result.await.map_err(with_context("Error auto-assigning defaults"))
}

// ---------------------------------------------------------------------------
// Provider availability
// ---------------------------------------------------------------------------

/// The providers `GET /models/providers` reports on, with the single
/// environment variable that makes each available, in Python's order.
const SIMPLE_PROVIDERS: [(&str, &str); 20] = [
    ("openai", "OPENAI_API_KEY"),
    ("anthropic", "ANTHROPIC_API_KEY"),
    ("google", "GOOGLE_API_KEY"),
    ("groq", "GROQ_API_KEY"),
    ("mistral", "MISTRAL_API_KEY"),
    ("deepseek", "DEEPSEEK_API_KEY"),
    ("xai", "XAI_API_KEY"),
    ("openrouter", "OPENROUTER_API_KEY"),
    ("voyage", "VOYAGE_API_KEY"),
    ("elevenlabs", "ELEVENLABS_API_KEY"),
    ("deepgram", "DEEPGRAM_API_KEY"),
    ("ollama", "OLLAMA_API_BASE"),
    ("omlx", "OMLX_API_BASE"),
    ("dashscope", "DASHSCOPE_API_KEY"),
    ("minimax", "MINIMAX_API_KEY"),
    ("novita", "NOVITA_API_KEY"),
    ("siliconflow", "SILICONFLOW_API_KEY"),
    ("zai", "ZAI_API_KEY"),
    ("ppq", "PPQ_API_KEY"),
    ("cohere", "COHERE_API_KEY"),
];

/// Environment-variable suffix of each model type for the mode-specific
/// Azure and OpenAI-compatible variables.
fn mode_suffix(modality: nn_ai::Modality) -> &'static str {
    match modality {
        nn_ai::Modality::Language => "LLM",
        nn_ai::Modality::Embedding => "EMBEDDING",
        nn_ai::Modality::SpeechToText => "STT",
        nn_ai::Modality::TextToSpeech => "TTS",
    }
}

fn azure_support(env: EnvLookup<'_>, mode: &str) -> bool {
    let set = |name: &str| env(name).is_some();
    let generic = set("AZURE_OPENAI_API_KEY") && set("AZURE_OPENAI_ENDPOINT") && set("AZURE_OPENAI_API_VERSION");
    let specific = set(&format!("AZURE_OPENAI_API_KEY_{mode}"))
        && set(&format!("AZURE_OPENAI_ENDPOINT_{mode}"))
        && set(&format!("AZURE_OPENAI_API_VERSION_{mode}"));
    generic || specific
}

fn openai_compatible_support(env: EnvLookup<'_>, mode: &str) -> bool {
    let set = |name: &str| env(name).is_some();
    set("OPENAI_COMPATIBLE_BASE_URL")
        || set(&format!("OPENAI_COMPATIBLE_BASE_URL_{mode}"))
        || set("OPENAI_COMPATIBLE_API_KEY")
        || set(&format!("OPENAI_COMPATIBLE_API_KEY_{mode}"))
}

/// Whether `check` holds for any model type's mode.
fn any_mode(env: EnvLookup<'_>, check: fn(EnvLookup<'_>, &str) -> bool) -> bool {
    nn_ai::Modality::ALL.into_iter().any(|m| check(env, mode_suffix(m)))
}

async fn has_credential(state: &AppState, provider: &str) -> bool {
    Credential::get_by_provider(state, provider).await.map(|c| !c.is_empty()).unwrap_or(false)
}

/// Which providers can be used (a credential in the database, or the
/// environment variables), and which model types each supports.
pub async fn provider_availability_with(state: &AppState, env: EnvLookup<'_>) -> ProviderAvailability {
    let set = |name: &str| env(name).is_some();
    let non_blank = |name: &str| env(name).is_some_and(|v| !v.trim().is_empty());

    let mut status: Vec<(&str, bool)> = Vec::new();
    for (provider, variable) in SIMPLE_PROVIDERS {
        status.push((provider, has_credential(state, provider).await || set(variable)));
    }
    if let Some(google) = status.iter_mut().find(|(p, _)| *p == "google") {
        if !google.1 {
            google.1 = set("GEMINI_API_KEY");
        }
    }
    let azure_db = has_credential(state, "azure").await;
    let compatible_db = has_credential(state, "openai_compatible").await;
    status.push(("vertex", has_credential(state, "vertex").await || set("VERTEX_PROJECT")));
    status.push(("azure", azure_db || any_mode(env, azure_support)));
    status.push(("openai_compatible", compatible_db || any_mode(env, openai_compatible_support)));
    status.push((
        "anthropic_compatible",
        has_credential(state, "anthropic_compatible").await
            || (non_blank("ANTHROPIC_COMPATIBLE_BASE_URL") && non_blank("ANTHROPIC_COMPATIBLE_API_KEY")),
    ));

    let mut supported_types = BTreeMap::new();
    for (provider, available) in &status {
        if !available {
            continue;
        }
        let Some(spec) = nn_ai::provider(provider) else { continue };
        let supported = nn_ai::Modality::ALL.into_iter().filter(|m| spec.supports(*m));
        let types: Vec<String> = match *provider {
            "openai_compatible" => supported
                .filter(|m| compatible_db || openai_compatible_support(env, mode_suffix(*m)))
                .map(|m| m.as_str().to_string())
                .collect(),
            "azure" => supported
                .filter(|m| azure_db || azure_support(env, mode_suffix(*m)))
                .map(|m| m.as_str().to_string())
                .collect(),
            "anthropic_compatible" => vec!["language".to_string()],
            _ => supported.map(|m| m.as_str().to_string()).collect(),
        };
        supported_types.insert(provider.to_string(), types);
    }

    ProviderAvailability {
        available: status.iter().filter(|(_, a)| *a).map(|(p, _)| p.to_string()).collect(),
        unavailable: status.iter().filter(|(_, a)| !*a).map(|(p, _)| p.to_string()).collect(),
        supported_types,
    }
}

/// [`provider_availability_with`] over the process environment.
pub async fn provider_availability(state: &AppState) -> ProviderAvailability {
    provider_availability_with(state, &process_env).await
}

// ---------------------------------------------------------------------------
// Discovery and sync
// ---------------------------------------------------------------------------

/// Providers whose models can be listed with the provider-wide
/// configuration. Azure and Vertex need a specific credential (use the
/// credential discovery endpoint).
fn discoverable(provider: &str) -> bool {
    nn_ai::PROVIDERS.iter().any(|spec| spec.name == provider) && !matches!(provider, "azure" | "vertex")
}

/// List a provider's models with its default credential (or environment)
/// without registering them. Unknown providers and failed listings give an
/// empty list.
pub async fn discover_provider_models_with(
    state: &AppState,
    provider: &str,
    env: EnvLookup<'_>,
) -> AppResult<Vec<DiscoveredModelInfo>> {
    if !discoverable(provider) {
        if nn_ai::provider(provider).is_some() {
            tracing::info!(
                "Provider '{provider}' requires credential-based discovery. Use the /credentials/{{id}}/discover endpoint instead."
            );
        } else {
            tracing::warn!("No discovery function for provider: {provider}");
        }
        return Ok(Vec::new());
    }
    let resolved = resolve_provider_config_with(state, provider, |name| env(name)).await?;
    let discovered = match prepare_client_config(&state.config.providers, provider, resolved).await {
        Ok(config) => discover_with_config(state, provider, &config).await,
        Err(e) => {
            tracing::warn!("Failed to discover {provider} models: {e}");
            Vec::new()
        }
    };
    Ok(discovered
        .into_iter()
        .map(|m| DiscoveredModelInfo {
            name: m.name,
            provider: m.provider,
            model_type: m.model_type.as_str().to_string(),
            description: m.description,
        })
        .collect())
}

/// [`discover_provider_models_with`] over the process environment.
pub async fn discover_provider_models(state: &AppState, provider: &str) -> AppResult<Vec<DiscoveredModelInfo>> {
    discover_provider_models_with(state, provider, &process_env)
        .await
        .map_err(|e| {
            tracing::error!("Error discovering models for {provider}: {e}");
            AppError::http(StatusCode::INTERNAL_SERVER_ERROR, "Error discovering models. Check server logs for details.")
        })
}

/// Discover a provider's models and register the new ones (unlinked, as in
/// Python's `sync_provider_models`). Models already present with the same
/// name and type (case-insensitive) are counted as existing.
pub async fn sync_provider_models_with(
    state: &AppState,
    provider: &str,
    env: EnvLookup<'_>,
) -> AppResult<ProviderSyncResult> {
    let discovered = discover_provider_models_with(state, provider, env).await?;
    let mut result = ProviderSyncResult { provider: provider.to_string(), discovered: discovered.len(), new: 0, existing: 0 };
    if discovered.is_empty() {
        return Ok(result);
    }
    let mut existing_keys = existing_model_keys(state, provider).await.unwrap_or_else(|e| {
        tracing::warn!("Failed to fetch existing models for {provider}: {e}");
        Default::default()
    });
    for found in discovered {
        let key = (found.name.to_lowercase(), found.model_type.to_lowercase());
        if existing_keys.contains(&key) {
            result.existing += 1;
            continue;
        }
        let mut model =
            Model { name: found.name.clone(), provider: found.provider.clone(), model_type: found.model_type.clone(), ..Model::default() };
        match model.save(state).await {
            Ok(()) => {
                tracing::info!("Registered new model: {}/{} ({})", found.provider, found.name, found.model_type);
                existing_keys.insert(key);
                result.new += 1;
            }
            Err(e) => tracing::warn!("Failed to register model {}: {e}", found.name),
        }
    }
    tracing::info!(
        "Synced {provider}: {} discovered, {} new, {} existing",
        result.discovered,
        result.new,
        result.existing
    );
    Ok(result)
}

/// [`sync_provider_models_with`] over the process environment.
pub async fn sync_provider_models(state: &AppState, provider: &str) -> AppResult<ProviderSyncResult> {
    sync_provider_models_with(state, provider, &process_env).await.map_err(|e| {
        tracing::error!("Error syncing models for {provider}: {e}");
        AppError::http(StatusCode::INTERNAL_SERVER_ERROR, "Error syncing models. Check server logs for details.")
    })
}

/// Sync every provider concurrently. A provider that fails counts as
/// nothing discovered.
pub async fn sync_all_providers_with(state: &AppState, env: EnvLookup<'_>) -> AllProvidersSyncResult {
    let runs = nn_ai::PROVIDERS.iter().map(|spec| async move {
        let result = sync_provider_models_with(state, spec.name, env).await.unwrap_or_else(|e| {
            tracing::error!("Error syncing {}: {e}", spec.name);
            ProviderSyncResult { provider: spec.name.to_string(), discovered: 0, new: 0, existing: 0 }
        });
        (spec.name.to_string(), result)
    });
    let results: BTreeMap<String, ProviderSyncResult> = futures::future::join_all(runs).await.into_iter().collect();
    AllProvidersSyncResult {
        total_discovered: results.values().map(|r| r.discovered).sum(),
        total_new: results.values().map(|r| r.new).sum(),
        results,
    }
}

/// [`sync_all_providers_with`] over the process environment.
pub async fn sync_all_providers(state: &AppState) -> AllProvidersSyncResult {
    sync_all_providers_with(state, &process_env).await
}

/// How many models a provider has of each type (case-insensitive).
pub async fn provider_model_count(state: &AppState, provider: &str) -> AppResult<ProviderModelCount> {
    let counts = Model::count_by_type(state, provider).await.map_err(with_context("Error getting model count"))?;
    let counts: BTreeMap<String, i64> = counts.into_iter().collect();
    Ok(ProviderModelCount { provider: provider.to_string(), total: counts.values().sum(), counts })
}

/// A provider's models (exact provider name), by type then name.
pub async fn models_by_provider(state: &AppState, provider: &str) -> AppResult<Vec<ModelResponse>> {
    let models = Model::get_by_provider(state, provider).await.map_err(with_context("Error fetching models"))?;
    Ok(models.iter().map(ModelResponse::from).collect())
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::*;

    fn env(vars: &[(&str, &str)]) -> impl Fn(&str) -> Option<String> + Send + Sync {
        let vars: HashMap<String, String> = vars.iter().map(|(k, v)| (k.to_string(), v.to_string())).collect();
        move |name: &str| vars.get(name).cloned()
    }

    fn model(id: &str, provider: &str, name: &str) -> Model {
        Model { id: id.into(), provider: provider.into(), name: name.into(), model_type: "language".into(), ..Model::default() }
    }

    #[test]
    fn preferred_model_follows_provider_priority_and_name_preferences() {
        let ollama = model("model:o", "ollama", "llama3");
        let mini = model("model:m", "openai", "o3-mini");
        let gpt4o = model("model:g", "openai", "GPT-4o-2024");
        let custom = model("model:c", "custom", "x");
        assert_eq!(preferred_model(&[&ollama, &mini, &gpt4o]).as_deref(), Some("model:g"));
        assert_eq!(preferred_model(&[&ollama, &mini]).as_deref(), Some("model:m"));
        assert_eq!(preferred_model(&[&custom, &ollama]).as_deref(), Some("model:o"));
        assert_eq!(preferred_model(&[&custom]).as_deref(), Some("model:c"));
        assert_eq!(preferred_model(&[]), None);
    }

    async fn availability(vars: &[(&str, &str)]) -> ProviderAvailability {
        let (state, _dir) = AppState::for_tests().await.unwrap();
        provider_availability_with(&state, &env(vars)).await
    }

    #[tokio::test]
    async fn openai_compatible_modes_follow_the_environment() {
        let all = ["language", "embedding", "speech_to_text", "text_to_speech"];
        let a = availability(&[("OPENAI_COMPATIBLE_BASE_URL", "http://localhost:1234/v1")]).await;
        assert!(a.available.contains(&"openai_compatible".to_string()));
        assert_eq!(a.supported_types["openai_compatible"], all);

        let a = availability(&[
            ("OPENAI_COMPATIBLE_BASE_URL_LLM", "http://localhost:1234/v1"),
            ("OPENAI_COMPATIBLE_BASE_URL_EMBEDDING", "http://localhost:8080/v1"),
        ])
        .await;
        assert_eq!(a.supported_types["openai_compatible"], ["language", "embedding"]);

        for (variable, expected) in [
            ("OPENAI_COMPATIBLE_BASE_URL_LLM", "language"),
            ("OPENAI_COMPATIBLE_BASE_URL_EMBEDDING", "embedding"),
            ("OPENAI_COMPATIBLE_BASE_URL_STT", "speech_to_text"),
            ("OPENAI_COMPATIBLE_BASE_URL_TTS", "text_to_speech"),
        ] {
            let a = availability(&[(variable, "http://localhost:9000/v1")]).await;
            assert_eq!(a.supported_types["openai_compatible"], [expected], "{variable}");
        }

        let a = availability(&[
            ("OPENAI_COMPATIBLE_BASE_URL", "http://localhost:1234/v1"),
            ("OPENAI_COMPATIBLE_BASE_URL_LLM", "http://localhost:5678/v1"),
        ])
        .await;
        assert_eq!(a.supported_types["openai_compatible"], all);

        let a = availability(&[]).await;
        assert!(a.unavailable.contains(&"openai_compatible".to_string()));
        assert!(!a.supported_types.contains_key("openai_compatible"));
        assert!(a.available.is_empty(), "{:?}", a.available);
        assert_eq!(a.unavailable.len(), 24);
    }

    #[tokio::test]
    async fn other_providers_follow_their_variables() {
        let a = availability(&[
            ("ANTHROPIC_COMPATIBLE_BASE_URL", "   "),
            ("ANTHROPIC_COMPATIBLE_API_KEY", "   "),
            ("GEMINI_API_KEY", "g"),
            ("VERTEX_PROJECT", "p"),
            ("AZURE_OPENAI_API_KEY_TTS", "k"),
            ("AZURE_OPENAI_ENDPOINT_TTS", "https://x"),
            ("AZURE_OPENAI_API_VERSION_TTS", "v"),
            ("GROQ_API_KEY", "gk"),
        ])
        .await;
        assert!(a.unavailable.contains(&"anthropic_compatible".to_string()), "blank values do not count");
        assert!(!a.supported_types.contains_key("anthropic_compatible"));
        assert_eq!(a.supported_types["google"], ["language", "embedding", "speech_to_text", "text_to_speech"]);
        assert_eq!(a.supported_types["vertex"], ["language", "embedding", "text_to_speech"]);
        assert_eq!(a.supported_types["azure"], ["text_to_speech"]);
        assert_eq!(a.supported_types["groq"], ["language", "speech_to_text"]);

        let a = availability(&[("ANTHROPIC_COMPATIBLE_BASE_URL", "https://x"), ("ANTHROPIC_COMPATIBLE_API_KEY", "k")]).await;
        assert_eq!(a.supported_types["anthropic_compatible"], ["language"]);
    }
}
