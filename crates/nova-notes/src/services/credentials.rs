//! AI provider credentials: CRUD with encrypted keys, connection tests,
//! credential-scoped model discovery and registration, configuration status,
//! the provider listing, and the one-off migrations (environment variables,
//! the old `provider_configs` singleton, the encryption-scheme upgrade).
//!
//! The Rust counterpart of `api/credentials_service.py` and the logic of
//! `api/routers/credentials.py` / `api/routers/providers.py`. Responses never
//! carry a key: [`CredentialResponse`] has `has_api_key` only.
//!
//! Functions that read provider environment variables come in two forms:
//! `foo(state)` reads the process environment, `foo_with(state, env)` takes
//! the lookup (tests pass a fixed map).

use std::collections::{BTreeMap, HashSet};

use axum::http::StatusCode;
use serde::{Deserialize, Deserializer, Serialize};
use serde_json::Map;

use crate::ai::net;
use crate::crypto::{self, Cipher, Reencrypt, SecretString};
use crate::db::{table_of, with_table, Doc, Json, Params};
use crate::domain::credential::{
    check_required_fields, credential_from_env, default_modalities, is_supported_provider,
};
use crate::domain::credential::ConfigSource;
use crate::domain::{Credential, Model};
use crate::error::{AppError, AppResult};
use crate::jobs::JobQueue;
use crate::services::notebooks::timestamp;
use crate::state::AppState;

/// Credentials have no background jobs (the Python build has no commands
/// for them).
pub fn register_jobs(_jobs: &JobQueue) {}

// ---------------------------------------------------------------------------
// Environment lookup
// ---------------------------------------------------------------------------

/// Reads one environment variable. Production passes [`process_env`].
pub type EnvLookup<'a> = &'a (dyn Fn(&str) -> Option<String> + Send + Sync);

/// The process environment.
pub fn process_env(name: &str) -> Option<String> {
    std::env::var(name).ok()
}

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

/// The first `max` characters of `text` (Python's `str(e)[:200]`).
pub(crate) fn truncate(text: &str, max: usize) -> String {
    text.chars().take(max).collect()
}

/// Server-side failures become a 500 with the Python router's fixed detail;
/// everything else (404, 400, ...) passes through. The original error is
/// logged, never returned.
pub(crate) fn mask(detail: &'static str) -> impl Fn(AppError) -> AppError {
    move |e| {
        if e.status() == StatusCode::INTERNAL_SERVER_ERROR && !matches!(e, AppError::Http { .. }) {
            tracing::error!("{detail}: {e}");
            AppError::http(StatusCode::INTERNAL_SERVER_ERROR, detail)
        } else {
            e
        }
    }
}

/// `validate_url` (under the configured provider URL policy) as the
/// credential endpoints call it: an empty value passes; a rejected URL is a
/// 400 with the validator's message.
pub(crate) async fn check_url(state: &AppState, url: &str) -> AppResult<()> {
    let config = nn_ai::ProviderConfig { base_url: Some(url.to_string()), ..nn_ai::ProviderConfig::default() };
    net::validate_config_urls(&config, &state.config.providers.url_policy)
        .await
        .map_err(|e| AppError::bad_request(truncate(&e, 200)))
}

/// What a credential may point at, checked when it is saved: a first-party
/// provider's base URL must be on its own domain, and a Vertex AI credential
/// takes no base URL, a plain region name and a service-account file inside
/// the configured directories. A violation is a 400. The same rules are
/// applied again at use time ([`net::prepare_client_config`]).
fn check_targets(state: &AppState, credential: &Credential) -> AppResult<()> {
    let net_config = &state.config.providers;
    let provider = credential.provider.to_lowercase();
    if let Some(problem) = credential.base_url.as_deref().and_then(|url| net::base_url_problem(net_config, &provider, url)) {
        return Err(AppError::bad_request(problem));
    }
    if provider == "vertex" {
        if let Some(location) = &credential.location {
            net::check_vertex_location(location).map_err(AppError::bad_request)?;
        }
        if let Some(path) = credential.credentials_path.as_deref().filter(|p| !p.trim().is_empty()) {
            net::confine_vertex_credentials(net_config, path).map_err(AppError::bad_request)?;
        }
    }
    Ok(())
}

/// The client configuration a stored credential runs with
/// ([`net::prepare_client_config`]); the error is a user-facing reason.
pub(crate) async fn credential_client_config(state: &AppState, credential: &Credential) -> Result<nn_ai::ProviderConfig, String> {
    net::prepare_client_config(
        &state.config.providers,
        &credential.provider,
        Some((credential.provider_config(), ConfigSource::Database)),
    )
    .await
}

/// The provider of a credential record, read without decrypting its key.
/// `None` when the record does not exist.
pub(crate) async fn credential_provider(state: &AppState, id: &str) -> AppResult<Option<String>> {
    let Ok(id) = credential_id(id) else { return Ok(None) };
    Ok(state
        .db
        .select(&id)
        .await?
        .and_then(|row| row.get("provider").and_then(Json::as_str).map(nn_ai::normalize_provider_name)))
}

/// The 400 for linking a model of one provider to another provider's
/// credential (its key would be sent to the wrong service).
pub(crate) fn provider_mismatch(credential_provider: &str, model_provider: &str) -> AppError {
    AppError::bad_request(format!(
        "Credential provider '{credential_provider}' does not match model provider '{model_provider}'"
    ))
}

/// List a provider's models with a prepared `config`. Failures (bad key,
/// unreachable server, an unknown provider) are logged and give an empty
/// list, as in the Python discovery functions.
pub(crate) async fn discover_with_config(
    state: &AppState,
    provider: &str,
    config: &nn_ai::ProviderConfig,
) -> Vec<nn_ai::DiscoveredModel> {
    match state.models.clients().discover_models(provider, config).await {
        Ok(models) => models,
        Err(e) => {
            tracing::warn!("Failed to discover {provider} models: {e}");
            Vec::new()
        }
    }
}

/// `credential:<id>` for a path id, or 404 when it names another table.
fn credential_id(id: &str) -> AppResult<String> {
    let id = with_table(Credential::TABLE, id);
    match table_of(&id) {
        Some(table) if table == Credential::TABLE => Ok(id),
        _ => Err(AppError::not_found("Credential not found")),
    }
}

/// Load a credential for an endpoint that answers a missing one with the
/// generic "Credential not found" and anything else with `detail`.
async fn load(state: &AppState, id: &str, detail: &'static str) -> AppResult<Credential> {
    match Credential::get(state, id).await {
        Ok(credential) => Ok(credential),
        Err(e) if e.is_not_found() => Err(AppError::not_found("Credential not found")),
        Err(e) => {
            tracing::error!("{detail}: {e}");
            Err(AppError::http(StatusCode::INTERNAL_SERVER_ERROR, detail))
        }
    }
}

/// Load a credential for an endpoint that lets the record's own 404 message
/// through and answers anything else with `detail`.
async fn load_keeping_not_found(state: &AppState, id: &str, detail: &'static str) -> AppResult<Credential> {
    match Credential::get(state, id).await {
        Ok(credential) => Ok(credential),
        Err(e) if e.is_not_found() => Err(e),
        Err(e) => {
            tracing::error!("{detail}: {e}");
            Err(AppError::http(StatusCode::INTERNAL_SERVER_ERROR, detail))
        }
    }
}

/// Link the provider's models that have no credential to `credential_id`
/// (the env and provider-config migrations).
async fn link_unassigned_models(state: &AppState, provider: &str, credential_id: &str) -> AppResult<usize> {
    let models: Vec<Model> = state
        .db
        .query_as(
            "SELECT * FROM model WHERE string::lowercase(provider) = $provider AND credential IS NONE",
            Params::new().set("provider", provider.to_lowercase()),
        )
        .await?;
    let count = models.len();
    for mut model in models {
        model.credential = Some(credential_id.to_string());
        model.save(state).await?;
    }
    Ok(count)
}

// ---------------------------------------------------------------------------
// Deserialization helpers for request bodies
// ---------------------------------------------------------------------------

/// An integer the way pydantic's lax mode accepts one: a JSON integer, a
/// float with no fractional part, or a numeric string.
fn parse_int<E: serde::de::Error>(value: Json) -> Result<Option<i64>, E> {
    match value {
        Json::Null => Ok(None),
        Json::Number(n) => n
            .as_i64()
            .or_else(|| n.as_f64().filter(|f| f.fract() == 0.0).map(|f| f as i64))
            .map(Some)
            .ok_or_else(|| E::custom("num_ctx: Input should be a valid integer")),
        Json::String(s) => s
            .trim()
            .parse::<i64>()
            .map(Some)
            .map_err(|_| E::custom("num_ctx: Input should be a valid integer, unable to parse string as an integer")),
        _ => Err(E::custom("num_ctx: Input should be a valid integer")),
    }
}

fn lenient_int<'de, D: Deserializer<'de>>(deserializer: D) -> Result<Option<i64>, D::Error> {
    parse_int(Json::deserialize(deserializer)?)
}

/// A field that was present in the body (`Some(value)`, `value` possibly
/// `None` for an explicit null). With `#[serde(default)]`, absent is `None`.
fn present<'de, D, T>(deserializer: D) -> Result<Option<Option<T>>, D::Error>
where
    D: Deserializer<'de>,
    T: Deserialize<'de>,
{
    Option::<T>::deserialize(deserializer).map(Some)
}

fn present_int<'de, D: Deserializer<'de>>(deserializer: D) -> Result<Option<Option<i64>>, D::Error> {
    parse_int(Json::deserialize(deserializer)?).map(Some)
}

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// A credential as the API returns it. Never contains the key.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CredentialResponse {
    pub id: String,
    pub name: String,
    pub provider: String,
    pub modalities: Vec<String>,
    pub base_url: Option<String>,
    pub endpoint: Option<String>,
    pub api_version: Option<String>,
    pub endpoint_llm: Option<String>,
    pub endpoint_embedding: Option<String>,
    pub endpoint_stt: Option<String>,
    pub endpoint_tts: Option<String>,
    pub project: Option<String>,
    pub location: Option<String>,
    pub credentials_path: Option<String>,
    pub num_ctx: Option<i64>,
    pub has_api_key: bool,
    pub created: String,
    pub updated: String,
    pub model_count: usize,
    pub decryption_error: Option<String>,
}

impl CredentialResponse {
    /// Python's `credential_to_response`.
    pub fn from_credential(credential: &Credential, model_count: usize) -> Self {
        Self {
            id: credential.id.clone(),
            name: credential.name.clone(),
            provider: credential.provider.clone(),
            modalities: credential.modalities.clone(),
            base_url: credential.base_url.clone(),
            endpoint: credential.endpoint.clone(),
            api_version: credential.api_version.clone(),
            endpoint_llm: credential.endpoint_llm.clone(),
            endpoint_embedding: credential.endpoint_embedding.clone(),
            endpoint_stt: credential.endpoint_stt.clone(),
            endpoint_tts: credential.endpoint_tts.clone(),
            project: credential.project.clone(),
            location: credential.location.clone(),
            credentials_path: credential.credentials_path.clone(),
            num_ctx: credential.num_ctx,
            has_api_key: credential.has_api_key(),
            created: timestamp(credential.created),
            updated: timestamp(credential.updated),
            model_count,
            decryption_error: credential.decryption_error.clone(),
        }
    }
}

/// `POST /credentials` (`CreateCredentialRequest`).
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct NewCredential {
    pub name: String,
    /// One of the registry provider names, exactly (`openai`, not `OpenAI`).
    pub provider: String,
    #[serde(default)]
    pub modalities: Vec<String>,
    #[serde(default)]
    pub api_key: Option<String>,
    #[serde(default)]
    pub base_url: Option<String>,
    #[serde(default)]
    pub endpoint: Option<String>,
    #[serde(default)]
    pub api_version: Option<String>,
    #[serde(default)]
    pub endpoint_llm: Option<String>,
    #[serde(default)]
    pub endpoint_embedding: Option<String>,
    #[serde(default)]
    pub endpoint_stt: Option<String>,
    #[serde(default)]
    pub endpoint_tts: Option<String>,
    #[serde(default)]
    pub project: Option<String>,
    #[serde(default)]
    pub location: Option<String>,
    #[serde(default)]
    pub credentials_path: Option<String>,
    #[serde(default, deserialize_with = "lenient_int")]
    pub num_ctx: Option<i64>,
}

impl NewCredential {
    /// The request validation of `CreateCredentialRequest`: the provider must
    /// be a registry name and provider-required fields must be present.
    /// `InvalidInput` with the pydantic message; the API turns it into a 422.
    pub fn validate(&self) -> AppResult<()> {
        if !is_supported_provider(&self.provider) {
            return Err(AppError::invalid(supported_provider_message()));
        }
        check_required_fields(&self.provider, self.base_url.as_deref(), self.api_key.as_deref())
    }

    fn urls(&self) -> [&Option<String>; 6] {
        [&self.base_url, &self.endpoint, &self.endpoint_llm, &self.endpoint_embedding, &self.endpoint_stt, &self.endpoint_tts]
    }
}

/// The pydantic literal error for an unsupported provider.
pub fn supported_provider_message() -> String {
    let names: Vec<String> = nn_ai::PROVIDERS.iter().map(|spec| format!("'{}'", spec.name)).collect();
    match names.split_last() {
        Some((last, rest)) if !rest.is_empty() => format!("Input should be {} or {last}", rest.join(", ")),
        Some((last, _)) => format!("Input should be {last}"),
        None => "Input should be a supported provider".to_string(),
    }
}

/// `PUT /credentials/{id}` (`UpdateCredentialRequest`). `name`,
/// `modalities` and `api_key` change only when non-null. The other fields
/// are keyed on presence: absent (`None`) keeps the value, an explicit null
/// or `""` (`Some(None)` / `Some(Some(""))`) clears it.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct CredentialUpdate {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub modalities: Option<Vec<String>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub api_key: Option<String>,
    #[serde(default, deserialize_with = "present", skip_serializing_if = "Option::is_none")]
    pub base_url: Option<Option<String>>,
    #[serde(default, deserialize_with = "present", skip_serializing_if = "Option::is_none")]
    pub endpoint: Option<Option<String>>,
    #[serde(default, deserialize_with = "present", skip_serializing_if = "Option::is_none")]
    pub api_version: Option<Option<String>>,
    #[serde(default, deserialize_with = "present", skip_serializing_if = "Option::is_none")]
    pub endpoint_llm: Option<Option<String>>,
    #[serde(default, deserialize_with = "present", skip_serializing_if = "Option::is_none")]
    pub endpoint_embedding: Option<Option<String>>,
    #[serde(default, deserialize_with = "present", skip_serializing_if = "Option::is_none")]
    pub endpoint_stt: Option<Option<String>>,
    #[serde(default, deserialize_with = "present", skip_serializing_if = "Option::is_none")]
    pub endpoint_tts: Option<Option<String>>,
    #[serde(default, deserialize_with = "present", skip_serializing_if = "Option::is_none")]
    pub project: Option<Option<String>>,
    #[serde(default, deserialize_with = "present", skip_serializing_if = "Option::is_none")]
    pub location: Option<Option<String>>,
    #[serde(default, deserialize_with = "present", skip_serializing_if = "Option::is_none")]
    pub credentials_path: Option<Option<String>>,
    /// `0` or null clears the override.
    #[serde(default, deserialize_with = "present_int", skip_serializing_if = "Option::is_none")]
    pub num_ctx: Option<Option<i64>>,
}

impl CredentialUpdate {
    fn urls(&self) -> [&Option<Option<String>>; 6] {
        [&self.base_url, &self.endpoint, &self.endpoint_llm, &self.endpoint_embedding, &self.endpoint_stt, &self.endpoint_tts]
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CredentialDeleted {
    pub message: String,
    pub deleted_models: usize,
}

/// One model of a credential's discovery listing.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DiscoveredModelResponse {
    pub name: String,
    pub provider: String,
    /// Always null here, as in the Python build: the user picks the type
    /// when registering.
    pub model_type: Option<String>,
    pub description: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DiscoverModelsResponse {
    pub credential_id: String,
    pub provider: String,
    pub discovered: Vec<DiscoveredModelResponse>,
}

/// A model to register, with the type the user chose.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RegisterModel {
    pub name: String,
    pub provider: String,
    pub model_type: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RegisterModelsRequest {
    pub models: Vec<RegisterModel>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct RegisterModelsResult {
    pub created: usize,
    pub existing: usize,
}

/// `GET /credentials/status`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CredentialStatus {
    pub configured: BTreeMap<String, bool>,
    /// `database`, `environment` or `none`.
    pub source: BTreeMap<String, String>,
    pub encryption_configured: bool,
}

/// The result of a migration endpoint.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MigrationResult {
    pub message: String,
    pub migrated: Vec<String>,
    pub skipped: Vec<String>,
    /// Only the environment migration reports this.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub not_configured: Option<Vec<String>>,
    pub errors: Vec<String>,
}

/// One entry of `GET /providers`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProviderInfoResponse {
    pub name: String,
    pub display_name: String,
    pub modalities: Vec<String>,
    pub docs_url: Option<String>,
    pub env_configured: bool,
}

// ---------------------------------------------------------------------------
// Listing and CRUD
// ---------------------------------------------------------------------------

async fn responses(state: &AppState, credentials: Vec<Credential>) -> AppResult<Vec<CredentialResponse>> {
    let mut result = Vec::with_capacity(credentials.len());
    for credential in credentials {
        let count = credential.linked_models(state).await?.len();
        result.push(CredentialResponse::from_credential(&credential, count));
    }
    Ok(result)
}

/// Every credential (by provider, oldest first), or those of one provider.
/// Undecryptable records are included with `decryption_error` set.
pub async fn list_credentials(state: &AppState, provider: Option<&str>) -> AppResult<Vec<CredentialResponse>> {
    let result = async {
        let credentials = match provider.filter(|p| !p.is_empty()) {
            Some(provider) => Credential::get_by_provider(state, provider).await?,
            None => Credential::get_all(state, Some("provider, created")).await?,
        };
        responses(state, credentials).await
    };
    result.await.map_err(mask("Failed to list credentials"))
}

/// `GET /credentials/by-provider/{provider}`.
pub async fn list_credentials_by_provider(state: &AppState, provider: &str) -> AppResult<Vec<CredentialResponse>> {
    let result = async {
        let credentials = Credential::get_by_provider(state, &provider.to_lowercase()).await?;
        responses(state, credentials).await
    };
    result.await.map_err(mask("Failed to list credentials for provider"))
}

/// One credential. Missing or unreadable → 404 "Credential not found".
pub async fn get_credential(state: &AppState, id: &str) -> AppResult<CredentialResponse> {
    let credential = match Credential::get(state, id).await {
        Ok(credential) => credential,
        Err(e) => {
            if !e.is_not_found() {
                tracing::error!("Error fetching credential {id}: {e}");
            }
            return Err(AppError::not_found("Credential not found"));
        }
    };
    let count = match credential.linked_models(state).await {
        Ok(models) => models.len(),
        Err(e) => {
            tracing::error!("Error fetching credential {id}: {e}");
            return Err(AppError::not_found("Credential not found"));
        }
    };
    Ok(CredentialResponse::from_credential(&credential, count))
}

/// Create a credential, its key encrypted at rest.
pub async fn create_credential(state: &AppState, input: &NewCredential) -> AppResult<CredentialResponse> {
    input.validate()?;
    crypto::require_encryption_key(&state.config)?;
    for url in input.urls().into_iter().flatten() {
        check_url(state, url).await?;
    }
    let mut credential = Credential {
        name: input.name.clone(),
        provider: input.provider.to_lowercase(),
        modalities: input.modalities.clone(),
        api_key: input.api_key.as_deref().filter(|k| !k.is_empty()).map(SecretString::new),
        base_url: input.base_url.clone(),
        endpoint: input.endpoint.clone(),
        api_version: input.api_version.clone(),
        endpoint_llm: input.endpoint_llm.clone(),
        endpoint_embedding: input.endpoint_embedding.clone(),
        endpoint_stt: input.endpoint_stt.clone(),
        endpoint_tts: input.endpoint_tts.clone(),
        project: input.project.clone(),
        location: input.location.clone(),
        credentials_path: input.credentials_path.clone(),
        num_ctx: input.num_ctx,
        ..Credential::default()
    };
    check_targets(state, &credential)?;
    credential.save(state).await.map_err(mask("Failed to create credential"))?;
    Ok(CredentialResponse::from_credential(&credential, 0))
}

/// Where a credential sends its key: base URL, endpoints and the Vertex key
/// file, normalised (trimmed, no trailing slash, blank as unset).
fn targets(credential: &Credential) -> Vec<Option<String>> {
    [
        &credential.base_url,
        &credential.endpoint,
        &credential.endpoint_llm,
        &credential.endpoint_embedding,
        &credential.endpoint_stt,
        &credential.endpoint_tts,
        &credential.credentials_path,
    ]
    .into_iter()
    .map(|value| value.as_deref().map(|v| v.trim().trim_end_matches('/').to_string()).filter(|v| !v.is_empty()))
    .collect()
}

/// Whether going from `before` to `after` ([`targets`]) points the key at a
/// new address.
fn moves_key(provider: &str, before: &[Option<String>], after: &[Option<String>]) -> bool {
    let own_domain = |url: &str| match nn_ai::base_url_rule(provider) {
        rule @ nn_ai::BaseUrlRule::Domains(_) => url::Url::parse(url)
            .ok()
            .and_then(|u| u.host_str().map(|host| rule.allows(u.scheme(), host)))
            .unwrap_or(false),
        _ => false,
    };
    before.iter().zip(after).enumerate().any(|(index, (old, new))| match new {
        Some(new) if Some(new) != old.as_ref() => !(index == 0 && own_domain(new)),
        _ => false,
    })
}

/// The 400 detail when a credential's address changes without its key.
pub const REKEY_DETAIL: &str =
    "Re-enter the API key to change the base URL, endpoints or credentials path: a stored key is never sent to a new address";

/// Partially update a credential (see [`CredentialUpdate`]). Changing
/// where it points while it holds a key requires `api_key` in the same
/// request ([`REKEY_DETAIL`]).
pub async fn update_credential(state: &AppState, id: &str, update: &CredentialUpdate) -> AppResult<CredentialResponse> {
    crypto::require_encryption_key(&state.config)?;
    for url in update.urls().into_iter().flatten().flatten() {
        check_url(state, url).await?;
    }
    let mut credential = load(state, id, "Failed to update credential").await?;
    let targets_before = targets(&credential);
    let had_key = credential.api_key.as_ref().is_some_and(|k| !k.expose().trim().is_empty());

    let cleared = |value: &Option<String>| value.clone().filter(|v| !v.is_empty());
    if let Some(name) = &update.name {
        credential.name = name.clone();
    }
    if let Some(modalities) = &update.modalities {
        credential.modalities = modalities.clone();
    }
    if let Some(api_key) = &update.api_key {
        credential.api_key = Some(SecretString::new(api_key.as_str()));
    }
    for (field, value) in [
        (&mut credential.base_url, &update.base_url),
        (&mut credential.endpoint, &update.endpoint),
        (&mut credential.api_version, &update.api_version),
        (&mut credential.endpoint_llm, &update.endpoint_llm),
        (&mut credential.endpoint_embedding, &update.endpoint_embedding),
        (&mut credential.endpoint_stt, &update.endpoint_stt),
        (&mut credential.endpoint_tts, &update.endpoint_tts),
        (&mut credential.project, &update.project),
        (&mut credential.location, &update.location),
        (&mut credential.credentials_path, &update.credentials_path),
    ] {
        if let Some(value) = value {
            *field = cleared(value);
        }
    }
    if let Some(num_ctx) = update.num_ctx {
        credential.num_ctx = num_ctx.filter(|n| *n != 0);
    }

    // A stored key is only ever sent where it was entered for: pointing a
    // credential somewhere new (base URL, any endpoint, the Vertex key file)
    // needs the key in the same request. Clearing a target, or moving a
    // first-party provider within its own domain, is not "somewhere new".
    // The record cannot change its provider (`provider` is not an update
    // field).
    if had_key && update.api_key.is_none() && moves_key(&credential.provider, &targets_before, &targets(&credential)) {
        return Err(AppError::bad_request(REKEY_DETAIL));
    }
    credential
        .check_required_fields()
        .map_err(|e| AppError::bad_request(truncate(&e.to_string(), 200)))?;
    check_targets(state, &credential)?;
    credential.save(state).await.map_err(mask("Failed to update credential"))?;
    let count = credential.linked_models(state).await.map_err(mask("Failed to update credential"))?.len();
    Ok(CredentialResponse::from_credential(&credential, count))
}

/// Delete a credential. Its models move to `migrate_to` when given,
/// otherwise they are deleted with it. Works on records whose key cannot be
/// decrypted (the key is never needed).
pub async fn delete_credential(state: &AppState, id: &str, migrate_to: Option<&str>) -> AppResult<CredentialDeleted> {
    let id = credential_id(id)?;
    let result = async {
        if !state.db.exists(&id).await? {
            return Err(AppError::not_found("Credential not found"));
        }
        let linked = Model::get_by_credential(state, &id).await?;
        let mut deleted_models = 0;
        match migrate_to.filter(|m| !m.is_empty()) {
            Some(target) if !linked.is_empty() => {
                let target = load(state, target, "Failed to delete credential").await?;
                let target_provider = nn_ai::normalize_provider_name(&target.provider);
                if let Some(model) = linked.iter().find(|m| nn_ai::normalize_provider_name(&m.provider) != target_provider) {
                    return Err(provider_mismatch(&target_provider, &model.provider));
                }
                for mut model in linked {
                    model.credential = Some(target.id.clone());
                    model.save(state).await?;
                }
            }
            _ => {
                for model in linked {
                    model.delete(state).await?;
                    deleted_models += 1;
                }
            }
        }
        state.db.delete(&id).await?;
        Ok(CredentialDeleted { message: "Credential deleted successfully".into(), deleted_models })
    };
    result.await.map_err(mask("Failed to delete credential"))
}

// ---------------------------------------------------------------------------
// Test, discover, register
// ---------------------------------------------------------------------------

fn failed_test(provider: &str, error: &str) -> nn_ai::ConnectionTestResult {
    let (success, message) = nn_ai::tester::classify_provider_test_error(error);
    nn_ai::ConnectionTestResult { provider: provider.to_string(), success, message }
}

/// Test a credential's connection. Never fails: every outcome, a missing
/// credential included, is a result with a user-facing message.
pub async fn test_credential(state: &AppState, id: &str) -> nn_ai::ConnectionTestResult {
    let credential = match Credential::get(state, id).await {
        Ok(credential) => credential,
        Err(e) => {
            tracing::debug!("Test connection error for credential {id}: {e}");
            // The Python build ran this text through the classifier, whose
            // "not found" marker reported a missing credential as a valid key.
            if e.is_not_found() {
                return nn_ai::ConnectionTestResult {
                    provider: "unknown".into(),
                    success: false,
                    message: "Credential not found".into(),
                };
            }
            return failed_test("unknown", &e.to_string());
        }
    };
    let provider = credential.provider.to_lowercase();
    let config = match credential_client_config(state, &credential).await {
        Ok(config) => config,
        Err(e) => return failed_test(&provider, &e),
    };
    state.models.clients().test_provider_connection(&provider, &config).await
}

/// The models a credential can use, names only (the user picks each type
/// when registering). A failed listing gives an empty list.
pub async fn discover_credential_models(state: &AppState, id: &str) -> AppResult<DiscoverModelsResponse> {
    let credential = load_keeping_not_found(state, id, "Failed to discover models").await?;
    let provider = credential.provider.to_lowercase();
    let discovered = match credential_client_config(state, &credential).await {
        Ok(config) => discover_with_config(state, &provider, &config).await,
        Err(e) => {
            tracing::warn!("Failed to discover {provider} models: {e}");
            Vec::new()
        }
    };
    let discovered = discovered
        .into_iter()
        .map(|m| DiscoveredModelResponse { name: m.name, provider: m.provider, model_type: None, description: m.description })
        .collect();
    Ok(DiscoverModelsResponse { credential_id: credential.id, provider, discovered })
}

/// Lower-cased `(name, type)` of the provider's existing models.
pub(crate) async fn existing_model_keys(state: &AppState, provider: &str) -> AppResult<HashSet<(String, String)>> {
    let rows = state
        .db
        .query(
            "SELECT string::lowercase(name) AS name, string::lowercase(type) AS type FROM model \
             WHERE string::lowercase(provider) = $provider",
            Params::new().set("provider", provider.to_lowercase()),
        )
        .await?;
    Ok(rows
        .iter()
        .map(|row| {
            let text = |key: &str| row.get(key).and_then(Json::as_str).unwrap_or_default().to_string();
            (text("name"), text("type"))
        })
        .collect())
}

/// Create the given models linked to the credential, skipping those that
/// already exist for its provider (same name and type, case-insensitive).
pub async fn register_credential_models(
    state: &AppState,
    id: &str,
    models: &[RegisterModel],
) -> AppResult<RegisterModelsResult> {
    let credential = load_keeping_not_found(state, id, "Failed to register models").await?;
    let credential_provider = nn_ai::normalize_provider_name(&credential.provider);
    if let Some(data) = models
        .iter()
        .find(|m| !m.provider.is_empty() && nn_ai::normalize_provider_name(&m.provider) != credential_provider)
    {
        return Err(provider_mismatch(&credential_provider, &data.provider));
    }
    let result = async {
        let mut existing_keys = existing_model_keys(state, &credential.provider).await?;
        let mut result = RegisterModelsResult { created: 0, existing: 0 };
        for data in models {
            let key = (data.name.to_lowercase(), data.model_type.to_lowercase());
            if existing_keys.contains(&key) {
                result.existing += 1;
                continue;
            }
            let provider = if data.provider.is_empty() { credential.provider.clone() } else { data.provider.clone() };
            let mut model = Model {
                name: data.name.clone(),
                provider,
                model_type: data.model_type.clone(),
                credential: Some(credential.id.clone()),
                ..Model::default()
            };
            model.save(state).await?;
            existing_keys.insert(key);
            result.created += 1;
        }
        Ok(result)
    };
    result.await.map_err(mask("Failed to register models"))
}

// ---------------------------------------------------------------------------
// Status and provider listing
// ---------------------------------------------------------------------------

fn env_configured(provider: &nn_ai::ProviderSpec, env: EnvLookup<'_>) -> bool {
    provider.is_env_configured(env)
}

/// Whether each provider is configured, and from where, plus whether an
/// encryption key is set.
pub async fn provider_status_with(state: &AppState, env: EnvLookup<'_>) -> CredentialStatus {
    let mut configured = BTreeMap::new();
    let mut source = BTreeMap::new();
    for spec in nn_ai::PROVIDERS {
        let from_env = env_configured(spec, env);
        let from_db = Credential::get_by_provider(state, spec.name).await.map(|c| !c.is_empty()).unwrap_or(false);
        configured.insert(spec.name.to_string(), from_db || from_env);
        let origin = if from_db {
            "database"
        } else if from_env {
            "environment"
        } else {
            "none"
        };
        source.insert(spec.name.to_string(), origin.to_string());
    }
    CredentialStatus { configured, source, encryption_configured: crypto::is_configured(&state.config) }
}

/// [`provider_status_with`] over the process environment.
pub async fn provider_status(state: &AppState) -> CredentialStatus {
    provider_status_with(state, &process_env).await
}

/// Which providers are configured through environment variables.
pub fn env_status_with(env: EnvLookup<'_>) -> BTreeMap<String, bool> {
    nn_ai::PROVIDERS.iter().map(|spec| (spec.name.to_string(), env_configured(spec, env))).collect()
}

/// [`env_status_with`] over the process environment.
pub fn env_status() -> BTreeMap<String, bool> {
    env_status_with(&process_env)
}

/// Every supported provider in display order, with its registry metadata.
pub fn list_providers_with(env: EnvLookup<'_>) -> Vec<ProviderInfoResponse> {
    nn_ai::PROVIDERS
        .iter()
        .map(|spec| ProviderInfoResponse {
            name: spec.name.to_string(),
            display_name: spec.display_name.to_string(),
            modalities: spec.modality_names().into_iter().map(String::from).collect(),
            docs_url: spec.docs_url.map(String::from),
            env_configured: env_configured(spec, env),
        })
        .collect()
}

/// [`list_providers_with`] over the process environment.
pub fn list_providers() -> Vec<ProviderInfoResponse> {
    list_providers_with(&process_env)
}

// ---------------------------------------------------------------------------
// Migrations
// ---------------------------------------------------------------------------

/// Create a credential for every provider configured through environment
/// variables that has none in the database yet, and link the provider's
/// unassigned models to it.
pub async fn migrate_from_env_with(state: &AppState, env: EnvLookup<'_>) -> AppResult<MigrationResult> {
    crypto::require_encryption_key(&state.config)?;
    let mut migrated = Vec::new();
    let mut skipped = Vec::new();
    let mut not_configured = Vec::new();
    let mut errors = Vec::new();

    for spec in nn_ai::PROVIDERS {
        let provider = spec.name;
        let outcome: AppResult<()> = async {
            if !env_configured(spec, env) {
                not_configured.push(provider.to_string());
                return Ok(());
            }
            let existing = Credential::get_by_provider(state, provider).await?;
            if !existing.is_empty() {
                tracing::info!("[{provider}] Already has {} credential(s) in DB, skipping", existing.len());
                skipped.push(provider.to_string());
                return Ok(());
            }
            let Some(mut credential) = credential_from_env(provider, env) else {
                not_configured.push(provider.to_string());
                return Ok(());
            };
            // Same URL checks as credentials created through the API.
            if let Some(base_url) = credential.base_url.as_deref() {
                nn_content::validate_url(base_url).await.map_err(|e| AppError::invalid(e.to_string()))?;
            }
            credential.save(state).await?;
            let linked = link_unassigned_models(state, provider, &credential.id).await?;
            tracing::info!("[{provider}] Credential saved (id={}), {linked} unassigned model(s) linked", credential.id);
            migrated.push(provider.to_string());
            Ok(())
        }
        .await;
        if let Err(e) = outcome {
            tracing::error!("[{provider}] Migration FAILED: {e}");
            errors.push(format!("{provider}: {e}"));
        }
    }

    Ok(MigrationResult {
        message: format!("Migration complete. Migrated {} providers.", migrated.len()),
        migrated,
        skipped,
        not_configured: Some(not_configured),
        errors,
    })
}

/// [`migrate_from_env_with`] over the process environment.
pub async fn migrate_from_env(state: &AppState) -> AppResult<MigrationResult> {
    migrate_from_env_with(state, &process_env).await.map_err(mask("Migration from environment variables failed"))
}

/// The record the pre-credential builds kept every provider key in.
pub const PROVIDER_CONFIG_RECORD: &str = "open_notebook:provider_configs";

/// One readable entry of the old `provider_configs` singleton.
struct OldProviderCredential {
    name: String,
    api_key: Option<SecretString>,
    fields: Map<String, Json>,
}

impl OldProviderCredential {
    fn text(&self, key: &str) -> Option<String> {
        self.fields.get(key).and_then(Json::as_str).map(String::from)
    }
}

/// The singleton's entries per provider, keys decrypted. Entries that cannot
/// be read are skipped, as Python's `ProviderConfig.get_instance` did.
async fn old_provider_credentials(state: &AppState, cipher: &Cipher) -> AppResult<Vec<(String, Vec<OldProviderCredential>)>> {
    let Some(record) = state.db.select(PROVIDER_CONFIG_RECORD).await? else { return Ok(Vec::new()) };
    let Some(groups) = record.get("credentials").and_then(Json::as_object) else { return Ok(Vec::new()) };
    let mut result = Vec::new();
    for (provider, entries) in groups {
        let Some(entries) = entries.as_array() else { continue };
        let mut readable = Vec::new();
        for entry in entries {
            let Some(fields) = entry.as_object() else { continue };
            let api_key = match fields.get("api_key") {
                Some(Json::String(stored)) if !stored.is_empty() => match cipher.decrypt(stored) {
                    Ok(key) => Some(SecretString::new(key)),
                    Err(_) => continue,
                },
                _ => None,
            };
            let name = fields.get("name").and_then(Json::as_str).unwrap_or("Default").to_string();
            readable.push(OldProviderCredential { name, api_key, fields: fields.clone() });
        }
        result.push((provider.clone(), readable));
    }
    Ok(result)
}

/// Turn the old `provider_configs` singleton into credential records,
/// skipping names that already exist for the provider.
pub async fn migrate_from_provider_config(state: &AppState) -> AppResult<MigrationResult> {
    let result = async {
        crypto::require_encryption_key(&state.config)?;
        let cipher = Cipher::from_config(&state.config)?;
        let mut migrated = Vec::new();
        let mut skipped = Vec::new();
        let mut errors = Vec::new();

        for (provider, entries) in old_provider_credentials(state, &cipher).await? {
            for old in entries {
                let label = format!("{provider}/{}", old.name);
                let outcome: AppResult<bool> = async {
                    let existing = Credential::get_by_provider(state, &provider).await?;
                    if existing.iter().any(|c| c.name == old.name) {
                        return Ok(false);
                    }
                    let mut credential = Credential {
                        name: old.name.clone(),
                        provider: provider.clone(),
                        modalities: default_modalities(&provider),
                        api_key: old.api_key.clone(),
                        base_url: old.text("base_url"),
                        endpoint: old.text("endpoint"),
                        api_version: old.text("api_version"),
                        endpoint_llm: old.text("endpoint_llm"),
                        endpoint_embedding: old.text("endpoint_embedding"),
                        endpoint_stt: old.text("endpoint_stt"),
                        endpoint_tts: old.text("endpoint_tts"),
                        project: old.text("project"),
                        location: old.text("location"),
                        credentials_path: old.text("credentials_path"),
                        ..Credential::default()
                    };
                    credential.save(state).await?;
                    link_unassigned_models(state, &provider, &credential.id).await?;
                    Ok(true)
                }
                .await;
                match outcome {
                    Ok(true) => migrated.push(label),
                    Ok(false) => {
                        tracing::info!("[{label}] Already exists in DB, skipping");
                        skipped.push(label);
                    }
                    Err(e) => {
                        tracing::error!("[{label}] Migration FAILED: {e}");
                        errors.push(format!("{label}: {e}"));
                    }
                }
            }
        }

        Ok(MigrationResult {
            message: format!("Migration complete. Migrated {} credentials.", migrated.len()),
            migrated,
            skipped,
            not_configured: None,
            errors,
        })
    };
    result.await.map_err(mask("Migration from provider config failed"))
}

/// Record one settled (not migrated) re-encryption outcome.
fn settle(outcome: &Reencrypt, label: &str, skipped: &mut Vec<String>, errors: &mut Vec<String>) {
    match outcome {
        Reencrypt::Skipped(reason) => skipped.push(format!("{label}: {reason}")),
        Reencrypt::Error(reason) => {
            tracing::warn!("[{label}] Left untouched: {reason}");
            errors.push(format!("{label}: {reason}"));
        }
        Reencrypt::Migrated(_) => {}
    }
}

/// Re-encrypt every stored key into the versioned PBKDF2 format: the
/// `credential` rows and the old `provider_configs` entries. Idempotent;
/// values that do not decrypt are left untouched and reported.
pub async fn migrate_encryption(state: &AppState) -> AppResult<MigrationResult> {
    let result = async {
        crypto::require_encryption_key(&state.config)?;
        let cipher = Cipher::from_config(&state.config)?;
        let mut migrated = Vec::new();
        let mut skipped = Vec::new();
        let mut errors = Vec::new();

        let rows = state.db.query("SELECT id, api_key FROM credential", Params::new()).await?;
        for row in rows {
            let id = row.get("id").and_then(Json::as_str).unwrap_or_default().to_string();
            match crypto::reencrypt_stored_value(&cipher, row.get("api_key").unwrap_or(&Json::Null)) {
                Reencrypt::Migrated(value) => match state.db.update(&id, Doc::new().set("api_key", value)).await {
                    Ok(_) => migrated.push(id),
                    Err(e) => {
                        tracing::error!("[{id}] Write FAILED: {e}");
                        errors.push(format!("{id}: write-failed"));
                    }
                },
                other => settle(&other, &id, &mut skipped, &mut errors),
            }
        }

        if let Some(record) = state.db.select(PROVIDER_CONFIG_RECORD).await? {
            if let Some(mut groups) = record.get("credentials").and_then(Json::as_object).cloned() {
                let mut pending = Vec::new();
                for (provider, entries) in groups.iter_mut() {
                    let Some(entries) = entries.as_array_mut() else {
                        errors.push(format!("provider_configs/{provider}: unexpected-group-type"));
                        continue;
                    };
                    for (index, entry) in entries.iter_mut().enumerate() {
                        let Some(entry) = entry.as_object_mut() else {
                            errors.push(format!("provider_configs/{provider}/{index}: unexpected-entry-type"));
                            continue;
                        };
                        let name = entry.get("name").and_then(Json::as_str).map(String::from).unwrap_or_else(|| index.to_string());
                        let label = format!("provider_configs/{provider}/{name}");
                        match crypto::reencrypt_stored_value(&cipher, entry.get("api_key").unwrap_or(&Json::Null)) {
                            Reencrypt::Migrated(value) => {
                                entry.insert("api_key".into(), Json::String(value));
                                pending.push(label);
                            }
                            other => settle(&other, &label, &mut skipped, &mut errors),
                        }
                    }
                }
                if !pending.is_empty() {
                    match state.db.upsert(PROVIDER_CONFIG_RECORD, Doc::new().set("credentials", Json::Object(groups))).await {
                        Ok(_) => migrated.extend(pending),
                        Err(e) => {
                            tracing::error!("Singleton rewrite FAILED: {e}");
                            errors.extend(pending.into_iter().map(|label| format!("{label}: write-failed")));
                        }
                    }
                }
            }
        }

        Ok(MigrationResult {
            message: format!(
                "Encryption migration complete. Migrated {} record(s) with {} error(s).",
                migrated.len(),
                errors.len()
            ),
            migrated,
            skipped,
            not_configured: None,
            errors,
        })
    };
    result.await.map_err(mask("Encryption migration failed"))
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::*;

    fn env(vars: &[(&str, &str)]) -> impl Fn(&str) -> Option<String> + Send + Sync {
        let vars: HashMap<String, String> = vars.iter().map(|(k, v)| (k.to_string(), v.to_string())).collect();
        move |name: &str| vars.get(name).cloned()
    }

    #[test]
    fn provider_message_lists_every_registry_name() {
        let message = supported_provider_message();
        assert!(message.starts_with("Input should be 'openai', 'anthropic'"), "{message}");
        assert!(message.ends_with(" or 'anthropic_compatible'"), "{message}");
    }

    #[test]
    fn update_bodies_distinguish_absent_null_and_values() {
        let update: CredentialUpdate =
            serde_json::from_value(serde_json::json!({ "base_url": null, "endpoint": "", "num_ctx": "4096" })).unwrap();
        assert_eq!(update.base_url, Some(None));
        assert_eq!(update.endpoint, Some(Some(String::new())));
        assert_eq!(update.project, None);
        assert_eq!(update.num_ctx, Some(Some(4096)));
        let update: CredentialUpdate = serde_json::from_value(serde_json::json!({ "num_ctx": null })).unwrap();
        assert_eq!(update.num_ctx, Some(None));
        assert!(serde_json::from_value::<CredentialUpdate>(serde_json::json!({ "num_ctx": "big" })).is_err());
        assert!(serde_json::from_value::<NewCredential>(serde_json::json!({ "name": "n", "provider": "openai", "num_ctx": 1.5 })).is_err());
    }

    #[tokio::test]
    async fn env_detection_drives_status_and_provider_listing() {
        let (state, _dir) = AppState::for_tests().await.unwrap();
        let lookup = env(&[("NOVITA_API_KEY", "nv-key"), ("GEMINI_API_KEY", "g-key"), ("GROQ_API_KEY", "  ")]);

        let providers = list_providers_with(&lookup);
        assert_eq!(providers.len(), nn_ai::PROVIDERS.len());
        let configured: Vec<&str> = providers.iter().filter(|p| p.env_configured).map(|p| p.name.as_str()).collect();
        assert_eq!(configured, ["google", "novita"]);

        let envs = env_status_with(&lookup);
        assert!(envs["novita"] && envs["google"]);
        assert!(!envs["groq"], "blank values do not count");

        let mut credential = Credential { name: "Main".into(), provider: "openai".into(), api_key: Some("sk".into()), ..Credential::default() };
        credential.save(&state).await.unwrap();
        let status = provider_status_with(&state, &lookup).await;
        assert!(status.encryption_configured);
        assert_eq!(status.source["openai"], "database");
        assert_eq!(status.source["novita"], "environment");
        assert_eq!(status.source["groq"], "none");
        assert!(status.configured["openai"] && status.configured["novita"] && !status.configured["groq"]);
    }

    #[tokio::test]
    async fn env_migration_creates_credentials_and_links_unassigned_models() {
        let (state, _dir) = AppState::for_tests().await.unwrap();
        let mut orphan = Model { name: "deepseek-chat".into(), provider: "deepseek".into(), model_type: "language".into(), ..Model::default() };
        orphan.save(&state).await.unwrap();
        let mut existing = Credential { name: "Mine".into(), provider: "groq".into(), api_key: Some("gk".into()), ..Credential::default() };
        existing.save(&state).await.unwrap();

        let lookup = env(&[
            ("DEEPSEEK_API_KEY", "ds-secret"),
            ("GROQ_API_KEY", "groq-secret"),
            ("SILICONFLOW_API_KEY", "sf-key"),
            ("SILICONFLOW_BASE_URL", "http://169.254.169.254/v1"),
            ("ZAI_API_KEY", "z-key"),
            ("ZAI_BASE_URL", " https://open.bigmodel.cn/api/paas/v4 "),
        ]);
        let result = migrate_from_env_with(&state, &lookup).await.unwrap();
        assert_eq!(result.migrated, ["deepseek", "zai"]);
        assert_eq!(result.skipped, ["groq"]);
        assert_eq!(result.message, "Migration complete. Migrated 2 providers.");
        // A rejected base URL is an error for that provider only.
        assert_eq!(result.errors.len(), 1, "{:?}", result.errors);
        assert!(result.errors[0].starts_with("siliconflow: "), "{:?}", result.errors);
        assert!(result.not_configured.as_ref().unwrap().contains(&"openai".to_string()));

        let created = Credential::default_for_provider(&state, "deepseek").await.unwrap().unwrap();
        assert_eq!(created.name, crate::domain::credential::ENV_CREDENTIAL_NAME);
        assert_eq!(created.api_key.as_ref().unwrap().expose(), "ds-secret");
        assert_eq!(Model::get(&state, &orphan.id).await.unwrap().credential.as_deref(), Some(created.id.as_str()));
        let zai = Credential::default_for_provider(&state, "zai").await.unwrap().unwrap();
        assert_eq!(zai.base_url.as_deref(), Some("https://open.bigmodel.cn/api/paas/v4"));

        // Running again changes nothing.
        let again = migrate_from_env_with(&state, &lookup).await.unwrap();
        assert!(again.migrated.is_empty());
        assert_eq!(again.skipped, ["groq", "deepseek", "zai"]);
    }

    // -----------------------------------------------------------------------
    // Where a stored key may go (issue #32), through the HTTP surface
    // -----------------------------------------------------------------------

    use axum::http::StatusCode;
    use serde_json::{json, Value};
    use wiremock::matchers::{header, method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    use crate::testing::TestApp;

    async fn create(app: &TestApp, body: Value) -> String {
        let (status, created) = app.post("/api/credentials", body).await;
        assert_eq!(status, StatusCode::CREATED, "{created}");
        created["id"].as_str().unwrap().to_string()
    }

    async fn models_server(key: &str) -> MockServer {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/v1/models"))
            .and(header("authorization", format!("Bearer {key}").as_str()))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({ "data": [{ "id": "m" }] })))
            .mount(&server)
            .await;
        server
    }

    #[tokio::test]
    async fn moving_a_credential_to_a_new_url_needs_its_key() {
        let app = TestApp::new().await;
        let home = models_server("sk-real").await;
        let attacker = MockServer::start().await;
        Mock::given(method("GET")).respond_with(ResponseTemplate::new(200).set_body_json(json!({ "data": [] }))).mount(&attacker).await;
        let id = create(
            &app,
            json!({ "name": "Local", "provider": "openai_compatible", "api_key": "sk-real", "base_url": format!("{}/v1", home.uri()) }),
        )
        .await;

        // The attack: change only the URL, then test. Refused, nothing stored.
        let (status, body) = app.put(&format!("/api/credentials/{id}"), json!({ "base_url": format!("{}/v1", attacker.uri()) })).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
        assert_eq!(body["detail"], REKEY_DETAIL);
        for field in ["endpoint", "endpoint_llm", "endpoint_embedding", "endpoint_stt", "endpoint_tts", "credentials_path"] {
            let (status, _) = app.put(&format!("/api/credentials/{id}"), json!({ field: format!("{}/x", attacker.uri()) })).await;
            assert_eq!(status, StatusCode::BAD_REQUEST, "{field}");
        }
        let (_, body) = app.post(&format!("/api/credentials/{id}/test"), json!({})).await;
        assert_eq!(body["success"], true, "{body}");
        assert!(attacker.received_requests().await.unwrap().is_empty(), "the stored key reached the new host");
        let stored = Credential::get(&app.state, &id).await.unwrap();
        assert_eq!(stored.base_url.as_deref(), Some(format!("{}/v1", home.uri()).as_str()));

        // Re-posting the same URL (the web form always sends it), renaming,
        // or trying to switch provider is fine; the provider stays.
        let (status, body) = app
            .put(&format!("/api/credentials/{id}"), json!({ "name": "Renamed", "base_url": format!("{}/v1/", home.uri()), "provider": "openai" }))
            .await;
        assert_eq!(status, StatusCode::OK, "{body}");
        assert_eq!(body["provider"], "openai_compatible");

        // With the key in the same request the move goes through, and the
        // new host gets the new key only.
        let (status, body) = app
            .put(&format!("/api/credentials/{id}"), json!({ "base_url": format!("{}/v1", attacker.uri()), "api_key": "sk-for-attacker" }))
            .await;
        assert_eq!(status, StatusCode::OK, "{body}");
        app.post(&format!("/api/credentials/{id}/test"), json!({})).await;
        let sent = attacker.received_requests().await.unwrap();
        assert_eq!(sent.len(), 1);
        assert_eq!(sent[0].headers.get("authorization").unwrap(), "Bearer sk-for-attacker");

        // A credential without a key can move freely (nothing to leak).
        let keyless = create(&app, json!({ "name": "Ollama", "provider": "ollama", "base_url": "http://127.0.0.1:11434" })).await;
        let (status, body) = app.put(&format!("/api/credentials/{keyless}"), json!({ "base_url": "http://127.0.0.1:11435" })).await;
        assert_eq!(status, StatusCode::OK, "{body}");
    }

    #[tokio::test]
    async fn the_web_form_goes_through_the_same_rule() {
        let app = TestApp::new().await;
        let id = create(&app, json!({ "name": "Local", "provider": "openai_compatible", "api_key": "sk-real", "base_url": "http://127.0.0.1:1/v1" })).await;
        let request = axum::http::Request::builder()
            .method("POST")
            .uri(format!("/settings/models/credentials/{id}"))
            .header("content-type", "application/x-www-form-urlencoded")
            .body(axum::body::Body::from("provider=openai_compatible&name=Local&api_key=&base_url=https%3A%2F%2Fattacker.example%2Fv1"))
            .unwrap();
        let response = app.send(request).await;
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        assert_eq!(Credential::get(&app.state, &id).await.unwrap().base_url.as_deref(), Some("http://127.0.0.1:1/v1"));
    }

    #[tokio::test]
    async fn first_party_credentials_only_point_at_their_provider() {
        let app = TestApp::new().await;
        let (status, body) = app
            .post("/api/credentials", json!({ "name": "x", "provider": "openai", "api_key": "sk", "base_url": "https://attacker.example/v1" }))
            .await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
        assert!(body["detail"].as_str().unwrap().contains("openai.com"), "{body}");
        let id = create(&app, json!({ "name": "x", "provider": "openai", "api_key": "sk-real", "base_url": "https://api.openai.com/v1" })).await;
        // Even with the key re-entered, a foreign host is refused for openai.
        let (status, _) = app
            .put(&format!("/api/credentials/{id}"), json!({ "base_url": "https://attacker.example/v1", "api_key": "sk-real" }))
            .await;
        assert_eq!(status, StatusCode::BAD_REQUEST);

        // A row written before these checks is refused at use, not followed.
        let mut old = Credential::get(&app.state, &id).await.unwrap();
        old.base_url = Some("https://attacker.example/v1".into());
        old.save(&app.state).await.unwrap();
        assert!(credential_client_config(&app.state, &old).await.is_err(), "the key would go to the stored URL");
        let (_, body) = app.post(&format!("/api/credentials/{id}/test"), json!({})).await;
        assert_eq!(body["success"], false, "{body}");

        // Vertex takes no base URL, a plain location and a confined key file.
        let vertex = |extra: Value| {
            let mut body = json!({ "name": "v", "provider": "vertex", "project": "p", "location": "us-central1" });
            body.as_object_mut().unwrap().extend(extra.as_object().unwrap().clone());
            body
        };
        for extra in [
            json!({ "base_url": "https://attacker.example" }),
            json!({ "location": "attacker.example/x" }),
            json!({ "credentials_path": "/etc/hosts" }),
        ] {
            let (status, body) = app.post("/api/credentials", vertex(extra.clone())).await;
            assert_eq!(status, StatusCode::BAD_REQUEST, "{extra}: {body}");
        }
        let dir = app.state.config.providers.vertex_credentials_dirs[0].clone();
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("sa.json"), "{}").unwrap();
        create(&app, vertex(json!({ "credentials_path": dir.join("sa.json").display().to_string() }))).await;
    }

    #[tokio::test]
    async fn models_only_use_credentials_of_their_own_provider() {
        let app = TestApp::new().await;
        let anthropic = create(&app, json!({ "name": "A", "provider": "anthropic", "api_key": "sk-ant" })).await;
        let openai = create(&app, json!({ "name": "O", "provider": "openai", "api_key": "sk-oai" })).await;

        let (status, body) = app
            .post("/api/models", json!({ "name": "gpt-4o", "provider": "openai", "type": "language", "credential": anthropic }))
            .await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
        assert!(body["detail"].as_str().unwrap().contains("does not match"), "{body}");
        let (status, body) = app
            .post("/api/models", json!({ "name": "gpt-4o", "provider": "openai", "type": "language", "credential": openai }))
            .await;
        assert_eq!(status, StatusCode::OK, "{body}");
        let model_id = body["id"].as_str().unwrap().to_string();

        let register = json!({ "models": [{ "name": "gpt-4o", "provider": "openai", "model_type": "language" }] });
        let (status, _) = app.post(&format!("/api/credentials/{anthropic}/register-models"), register).await;
        assert_eq!(status, StatusCode::BAD_REQUEST);

        let (status, _) = app.delete(&format!("/api/credentials/{openai}?migrate_to={anthropic}")).await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert!(Model::get(&app.state, &model_id).await.is_ok(), "nothing was deleted or moved");

        // A link made before the check is refused at use, not followed.
        let mut model = Model::get(&app.state, &model_id).await.unwrap();
        model.credential = Some(anthropic.clone());
        model.save(&app.state).await.unwrap();
        let err = model.provider_config(&app.state).await.unwrap_err();
        assert!(err.to_string().contains("linked to a credential for anthropic"), "{err}");
        let (status, body) = app.post(&format!("/api/models/{model_id}/test"), json!({})).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(body["success"], false, "{body}");
    }
}
