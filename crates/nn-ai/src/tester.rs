//! Connection testing: is this credential usable, does this model answer?
//!
//! A port of `open_notebook/ai/connection_tester.py` and of
//! `test_credential()` in `api/credentials_service.py`, with the user-facing
//! messages kept.
//!
//! Two questions with deliberately different answers to the same failure:
//!
//! * [`test_provider_connection`] asks "do these credentials reach a working
//!   provider?". Only a rejected key, missing permissions or an unreachable
//!   endpoint fail. A rate limit or a retired test model still proves the
//!   connection works, so both count as success.
//! * [`test_model`] validates one specific model, so a missing model is a
//!   failure.

use std::time::Duration;

use nn_llm::{AiError, ChatMessage, ChatOptions, TranscriptionRequest};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::audio::TEST_SPEECH_MP3;
use crate::config::{non_blank, ProviderConfig};
use crate::error::{http_status, truncate};
use crate::factory::Clients;
use crate::http::encode_component;
use crate::providers::anthropic::{normalize_anthropic_compatible_base_url, ANTHROPIC_VERSION};
use crate::providers::ollama::OLLAMA_DEFAULT_BASE_URL;
use crate::providers::openai::{DEFAULT_AZURE_API_VERSION, OMLX_DEFAULT_BASE_URL};
use crate::registry::{self, Modality};

const PING_TIMEOUT: Duration = Duration::from_secs(10);

const INVALID_KEY: &str = "Invalid API key";
const NO_PERMISSION: &str = "API key lacks required permissions";
const TIMED_OUT: &str = "Connection timed out - check network/endpoint";
const UNREACHABLE: &str = "Connection error - check network/endpoint";
const RATE_LIMITED: &str = "Rate limited - but connection works";

/// Outcome of a connection or model test, ready to return from the API.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ConnectionTestResult {
    /// Normalised provider id the test ran against.
    pub provider: String,
    pub success: bool,
    /// Short user-facing explanation.
    pub message: String,
}

impl ConnectionTestResult {
    fn new(provider: &str, (success, message): (bool, String)) -> Self {
        Self { provider: provider.to_string(), success, message }
    }
}

// ---------------------------------------------------------------------------
// Error text classification (string rules ported from the original)
// ---------------------------------------------------------------------------

/// True when `code` appears as a standalone number, not as digits inside a
/// longer one (the date in `claude-3-haiku-20240307` contains `403`).
fn contains_http_status(text: &str, code: u16) -> bool {
    let needle = code.to_string();
    let bytes = text.as_bytes();
    text.match_indices(&needle).any(|(start, _)| {
        let end = start + needle.len();
        let digit_before = start > 0 && bytes[start - 1].is_ascii_digit();
        let digit_after = end < bytes.len() && bytes[end].is_ascii_digit();
        !digit_before && !digit_after
    })
}

/// Removes provider request ids (`req_011Cga403VtXy`), whose random digits
/// can read as an HTTP status.
fn strip_request_ids(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    while let Some(start) = rest.find("req_") {
        let after = &rest[start + 4..];
        let id_len = after.bytes().take_while(u8::is_ascii_alphanumeric).count();
        out.push_str(&rest[..start]);
        if id_len < 8 {
            out.push_str(&rest[start..start + 4 + id_len]);
        }
        rest = &after[id_len..];
    }
    out.push_str(rest);
    out
}

/// Says why an error disproves a working connection, or `None` when the
/// provider answered after authenticating (missing model, quota, bad request).
pub fn connection_failure_reason(error: &str) -> Option<&'static str> {
    let error = strip_request_ids(error);
    let lower = error.to_lowercase();
    if contains_http_status(&error, 401) || lower.contains("unauthorized") {
        Some(INVALID_KEY)
    } else if contains_http_status(&error, 403) || lower.contains("forbidden") {
        Some(NO_PERMISSION)
    } else if lower.contains("timeout") || lower.contains("timed out") {
        Some(TIMED_OUT)
    } else if ["connection", "network", "getaddrinfo", "name resolution", "failed to establish"]
        .iter()
        .any(|k| lower.contains(k))
    {
        Some(UNREACHABLE)
    } else {
        None
    }
}

/// True for throttling and quota responses. Being rate-limited proves the
/// request authenticated.
pub fn is_rate_limit(error: &str) -> bool {
    let lower = error.to_lowercase();
    (lower.contains("rate") && lower.contains("limit"))
        || contains_http_status(error, 429)
        || lower.contains("quota")
        || lower.contains("resource has been exhausted")
        || lower.contains("resource exhausted")
}

/// Classifies an error text for the single-model test: `(success, message)`.
pub fn normalize_error_message(error: &str) -> (bool, String) {
    if let Some(reason) = connection_failure_reason(error) {
        return (false, reason.to_string());
    }
    if is_rate_limit(error) {
        return (true, RATE_LIMITED.to_string());
    }
    let lower = error.to_lowercase();
    if (lower.contains("not found") || lower.contains("not_found")) && lower.contains("model") {
        return (false, "Model not found on this provider".to_string());
    }
    (false, error.to_string())
}

/// Substrings showing the provider answered but the test model is missing,
/// retired or unsupported: proof that credentials and endpoint work.
const MODEL_UNAVAILABLE_MARKERS: &[&str] =
    &["not found", "not_found", "not supported", "does not exist", "deprecated", "unavailable", "no longer available"];

/// Classifies an error text for the provider test: `(success, message)`.
pub fn classify_provider_test_error(error: &str) -> (bool, String) {
    if let Some(reason) = connection_failure_reason(error) {
        return (false, reason.to_string());
    }
    if is_rate_limit(error) {
        return (true, RATE_LIMITED.to_string());
    }
    let lower = error.to_lowercase();
    if MODEL_UNAVAILABLE_MARKERS.iter().any(|marker| lower.contains(marker)) {
        return (true, "API key valid (test model unavailable)".to_string());
    }
    (false, format!("Error: {}", truncate(error, 100)))
}

/// What the typed part of an [`AiError`] already settles for both tests.
fn settled_by_variant(err: &AiError) -> Option<(bool, String)> {
    Some(match err {
        // Includes the generic Vertex credentials-file message.
        AiError::Configuration(message) => (false, message.clone()),
        AiError::Authentication(_) if http_status(err) == Some(403) => (false, NO_PERMISSION.to_string()),
        AiError::Authentication(_) => (false, INVALID_KEY.to_string()),
        AiError::Timeout(_) => (false, TIMED_OUT.to_string()),
        AiError::Network(_) => (false, UNREACHABLE.to_string()),
        AiError::RateLimit(_) => (true, RATE_LIMITED.to_string()),
        _ => return None,
    })
}

fn provider_test_outcome(err: &AiError) -> (bool, String) {
    if let Some(outcome) = settled_by_variant(err) {
        return outcome;
    }
    match err {
        AiError::ModelNotFound(_) => (true, "API key valid (test model unavailable)".to_string()),
        other => classify_provider_test_error(&other.to_string()),
    }
}

fn model_test_outcome(err: &AiError) -> (bool, String) {
    if let Some(outcome) = settled_by_variant(err) {
        return outcome;
    }
    match err {
        AiError::ModelNotFound(_) => (false, "Model not found on this provider".to_string()),
        other => normalize_error_message(&other.to_string()),
    }
}

// ---------------------------------------------------------------------------
// Server pings for URL-based providers
// ---------------------------------------------------------------------------

/// Wording and response shape of one kind of model-listing ping.
struct Ping {
    cannot_connect: &'static str,
    timed_out: &'static str,
    /// JSON key holding the model array and the key naming each model.
    list_key: &'static str,
    name_key: &'static str,
    /// `"Connected. 3 {connected_suffix}: a, b, c"`.
    connected_suffix: &'static str,
    empty: &'static str,
    status_prefix: &'static str,
    /// Treat 404/405 as "reachable, but cannot list models".
    listing_optional: bool,
}

const SERVER_PING: Ping = Ping {
    cannot_connect: "Cannot connect to server. Check the URL is correct.",
    timed_out: "Connection timed out. Check if server is accessible.",
    list_key: "data",
    name_key: "id",
    connected_suffix: "models available",
    empty: "Connected successfully (no models listed)",
    status_prefix: "Server",
    listing_optional: false,
};

const OLLAMA_PING: Ping = Ping {
    cannot_connect: "Cannot connect to Ollama. Check if Ollama server is running.",
    timed_out: "Connection timed out. Check if Ollama server is accessible.",
    list_key: "models",
    name_key: "name",
    ..SERVER_PING
};

const ANTHROPIC_COMPATIBLE_PING: Ping = Ping { listing_optional: true, ..SERVER_PING };

const AZURE_PING: Ping = Ping {
    cannot_connect: "Cannot connect to Azure endpoint. Check the URL.",
    timed_out: "Connection timed out. Check the endpoint URL.",
    connected_suffix: "models",
    empty: "Connected successfully (no models found)",
    status_prefix: "Azure",
    ..SERVER_PING
};

async fn ping(request: reqwest::RequestBuilder, kind: &Ping) -> (bool, String) {
    let response = match request.timeout(PING_TIMEOUT).send().await {
        Ok(response) => response,
        Err(err) if err.is_timeout() => return (false, kind.timed_out.to_string()),
        Err(err) if err.is_connect() => return (false, kind.cannot_connect.to_string()),
        Err(err) => return (false, format!("Connection error: {}", truncate(&err.without_url().to_string(), 100))),
    };
    match response.status().as_u16() {
        200 => {
            let json: Value = match response.json().await {
                Ok(json) => json,
                Err(_) => return (false, "Connection error: the server did not return a model list".to_string()),
            };
            let models = json.get(kind.list_key).and_then(Value::as_array).map(Vec::as_slice).unwrap_or_default();
            if models.is_empty() {
                return (true, kind.empty.to_string());
            }
            let mut names = models
                .iter()
                .take(3)
                .map(|m| m.get(kind.name_key).and_then(Value::as_str).unwrap_or("unknown"))
                .collect::<Vec<_>>()
                .join(", ");
            if models.len() > 3 {
                names.push_str(&format!(" (+{} more)", models.len() - 3));
            }
            (true, format!("Connected. {} {}: {names}", models.len(), kind.connected_suffix))
        }
        401 => (false, INVALID_KEY.to_string()),
        403 => (false, NO_PERMISSION.to_string()),
        status @ (404 | 405) if kind.listing_optional => {
            (true, format!("Connected, but model listing is unsupported (status {status}); add models manually."))
        }
        status => (false, format!("{} returned status {status}", kind.status_prefix)),
    }
}

/// Joins a base URL with `/models` without doubling an existing suffix.
fn models_url(base: &str) -> String {
    let trimmed = base.trim().trim_end_matches('/');
    if trimmed.ends_with("/models") {
        trimmed.to_string()
    } else {
        format!("{trimmed}/models")
    }
}

// ---------------------------------------------------------------------------
// The tests
// ---------------------------------------------------------------------------

/// Voices known to work for a provider's speech models. Providers missing
/// here have their voices looked up through the model's voice listing.
pub fn default_test_voice(provider: &str) -> Option<&'static str> {
    Some(match provider {
        "openai" | "azure" | "openai_compatible" => "alloy",
        "google" | "vertex" => "Kore",
        "deepgram" => "aura-2-thalia-en",
        "xai" => "eve",
        _ => return None,
    })
}

impl Clients {
    /// See [`test_provider_connection`].
    pub async fn test_provider_connection(&self, provider: &str, config: &ProviderConfig) -> ConnectionTestResult {
        let name = registry::normalize_provider_name(provider);
        let outcome = self.provider_outcome(&name, config).await;
        if !outcome.0 {
            tracing::debug!(provider = %name, message = %outcome.1, "provider connection test failed");
        }
        ConnectionTestResult::new(&name, outcome)
    }

    async fn provider_outcome(&self, provider: &str, config: &ProviderConfig) -> (bool, String) {
        let client = self.http();
        let bearer = |request: reqwest::RequestBuilder| match config.key() {
            Some(key) => request.bearer_auth(key),
            None => request,
        };

        match provider {
            "ollama" => {
                let base = config.url().unwrap_or(OLLAMA_DEFAULT_BASE_URL);
                let base = base.strip_suffix("/v1").unwrap_or(base);
                return ping(bearer(client.get(format!("{base}/api/tags"))), &OLLAMA_PING).await;
            }
            "openai_compatible" => {
                let Some(base) = config.url() else {
                    return (false, "No base URL configured".to_string());
                };
                return ping(bearer(client.get(models_url(base))), &SERVER_PING).await;
            }
            "omlx" => {
                let base = config.url().unwrap_or(OMLX_DEFAULT_BASE_URL);
                return ping(bearer(client.get(models_url(base))), &SERVER_PING).await;
            }
            "anthropic_compatible" => {
                let Some(base) = config.url() else {
                    return (false, "No base URL configured".to_string());
                };
                let url = format!("{}/models", normalize_anthropic_compatible_base_url(base));
                let mut request = client.get(url).header("anthropic-version", ANTHROPIC_VERSION);
                if let Some(key) = config.key() {
                    request = request.header("x-api-key", key);
                }
                return ping(request, &ANTHROPIC_COMPATIBLE_PING).await;
            }
            "azure" => {
                // Deployment names vary per user, so list models instead of
                // invoking one.
                let Some(endpoint) = non_blank(&config.endpoint).or_else(|| config.url()) else {
                    return (false, "No Azure endpoint configured".to_string());
                };
                let Some(key) = config.key() else {
                    return (false, "No Azure API key configured".to_string());
                };
                let version = non_blank(&config.api_version).unwrap_or(DEFAULT_AZURE_API_VERSION);
                let url = format!(
                    "{}/openai/models?api-version={}",
                    endpoint.trim_end_matches('/'),
                    encode_component(version)
                );
                return ping(client.get(url).header("api-key", key), &AZURE_PING).await;
            }
            _ => {}
        }

        // Standard providers: a minimal call with the registry's test model.
        let Some(spec) = registry::provider(provider) else {
            return (false, format!("Unknown provider: {provider}"));
        };
        let Some(test_model) = spec.test_model else {
            return (false, format!("No test model configured for {provider}"));
        };
        let result = match spec.test_model_type {
            Modality::Language => match self.language_model(provider, test_model, config) {
                Ok(model) => model.chat(&[ChatMessage::user("Hi")], &ChatOptions::default()).await.map(|_| ()),
                Err(err) => Err(err),
            },
            Modality::Embedding => match self.embedding_model(provider, test_model, config) {
                Ok(model) => model.embed(&["test".to_string()]).await.map(|_| ()),
                Err(err) => Err(err),
            },
            Modality::TextToSpeech => {
                // As in the original, speech providers are only checked for a
                // usable configuration: synthesising audio costs money.
                return match self.text_to_speech_model(provider, test_model, config) {
                    Ok(_) => (true, "Connection successful (key format valid)".to_string()),
                    Err(err) => provider_test_outcome(&err),
                };
            }
            Modality::SpeechToText => return (false, "Unsupported test type: speech_to_text".to_string()),
        };
        match result {
            Ok(()) => (true, "Connection successful".to_string()),
            Err(err) => provider_test_outcome(&err),
        }
    }

    /// See [`test_model`].
    pub async fn test_model(
        &self,
        provider: &str,
        model: &str,
        model_type: &str,
        config: &ProviderConfig,
    ) -> ConnectionTestResult {
        let name = registry::normalize_provider_name(provider);
        let outcome = match Modality::parse(model_type) {
            None => (false, format!("Unsupported model type: {model_type}")),
            Some(modality) => match self.exercise_model(&name, model, modality, config).await {
                Ok(message) => (true, message),
                Err(err) => {
                    let outcome = model_test_outcome(&err);
                    if !outcome.0 {
                        tracing::debug!(provider = %name, model, error = %err, "model test failed");
                    }
                    outcome
                }
            },
        };
        ConnectionTestResult::new(&name, outcome)
    }

    /// Makes one real call to the model and describes the result.
    async fn exercise_model(
        &self,
        provider: &str,
        model: &str,
        modality: Modality,
        config: &ProviderConfig,
    ) -> Result<String, AiError> {
        match modality {
            Modality::Language => {
                let llm = self.language_model(provider, model, config)?;
                let reply = llm.chat(&[ChatMessage::user("Hi!")], &ChatOptions::default()).await?;
                let text: String = reply.content.chars().take(100).collect();
                Ok(format!("Response: {}", if text.is_empty() { "(empty response)" } else { &text }))
            }
            Modality::Embedding => {
                let embedder = self.embedding_model(provider, model, config)?;
                let vectors = embedder.embed(&["This is a test.".to_string()]).await?;
                Ok(match vectors.first() {
                    Some(vector) => format!("Embedding dimensions: {}", vector.len()),
                    None => "Embedding successful".to_string(),
                })
            }
            Modality::TextToSpeech => {
                let tts = self.text_to_speech_model(provider, model, config)?;
                let voice = match default_test_voice(provider) {
                    Some(voice) => voice.to_string(),
                    // Voice ids are account- or model-specific here; take the
                    // first one the provider lists.
                    None => tts
                        .voices()
                        .await
                        .ok()
                        .and_then(|voices| voices.into_iter().next())
                        .map(|voice| voice.id)
                        .unwrap_or_else(|| "alloy".to_string()),
                };
                let audio = tts.synthesize("Hello from Open Notebook", &voice).await?;
                Ok(format!("Audio generated: {} bytes", audio.bytes.len()))
            }
            Modality::SpeechToText => {
                let stt = self.speech_to_text_model(provider, model, config)?;
                let text = stt
                    .transcribe(TranscriptionRequest {
                        audio: bytes::Bytes::from_static(TEST_SPEECH_MP3),
                        filename: "test_speech.mp3".to_string(),
                        content_type: Some("audio/mpeg".to_string()),
                        language: Some("en".to_string()),
                        prompt: None,
                    })
                    .await?;
                let text = text.trim();
                Ok(if text.is_empty() {
                    "Connection successful (test clip produced no transcription)".to_string()
                } else {
                    format!("Transcription: {}", text.chars().take(100).collect::<String>())
                })
            }
        }
    }
}

/// Tests whether `config` reaches a working `provider`.
///
/// URL-based providers (Ollama, oMLX, OpenAI-compatible,
/// Anthropic-compatible, Azure) are pinged by listing their models. All
/// others make one minimal call with the registry's test model. Never
/// returns an error: every outcome is a [`ConnectionTestResult`].
pub async fn test_provider_connection(provider: &str, config: &ProviderConfig) -> ConnectionTestResult {
    Clients::new().test_provider_connection(provider, config).await
}

/// [`test_provider_connection`] on the caller's connection pool.
pub async fn test_provider_connection_with_client(
    provider: &str,
    config: &ProviderConfig,
    client: &reqwest::Client,
) -> ConnectionTestResult {
    Clients::with_client(client.clone()).test_provider_connection(provider, config).await
}

/// Tests one model end to end with a real call: a short chat, one embedding,
/// a sentence of speech, or a transcription of a bundled clip.
/// `model_type` is `language`, `embedding`, `text_to_speech` or
/// `speech_to_text`.
pub async fn test_model(provider: &str, model: &str, model_type: &str, config: &ProviderConfig) -> ConnectionTestResult {
    Clients::new().test_model(provider, model, model_type, config).await
}

/// [`test_model`] on the caller's connection pool.
pub async fn test_model_with_client(
    provider: &str,
    model: &str,
    model_type: &str,
    config: &ProviderConfig,
    client: &reqwest::Client,
) -> ConnectionTestResult {
    Clients::with_client(client.clone()).test_model(provider, model, model_type, config).await
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::providers::testing::{header, request, request_json};
    use serde_json::json;
    use wiremock::matchers::{method, path, query_param};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    // Realistic provider error strings, from tests/test_connection_test_classification.py.
    const GOOGLE_RETIRED_MODEL_404: &str =
        "404 models/gemini-2.0-flash is not found for API version v1beta, or is not supported for generateContent.";
    const ANTHROPIC_RETIRED_HAIKU_404: &str = "Error code: 404 - {'type': 'error', 'error': {'type': 'not_found_error', 'message': 'model: claude-3-haiku-20240307'}, 'request_id': 'req_test'}";
    const ANTHROPIC_404_WITH_403_IN_REQUEST_ID: &str = "Error code: 404 - {'type': 'error', 'error': {'type': 'not_found_error', 'message': 'model: claude-haiku-4-5-20251001'}, 'request_id': 'req_011Cga403VtXy92ABc'}";
    const ANTHROPIC_PERM_403_WITH_REQUEST_ID: &str = "Error code: 403 - {'type': 'error', 'error': {'type': 'permission_error', 'message': 'Your API key does not have permission to use the specified resource.'}, 'request_id': 'req_011Cga403VtXy92ABc'}";
    const GOOGLE_DEPRECATED: &str = "400 Model gemini-1.5-pro has been deprecated.";
    const GOOGLE_BAD_KEY_401: &str = "401 API key not valid. Please pass a valid API key.";
    const GOOGLE_PERM_403: &str = "403 Permission denied on resource project.";
    const GOOGLE_QUOTA_429: &str = "429 Resource has been exhausted (e.g. check quota).";
    const DNS_FAILURE: &str = "Connection error: [Errno -2] Name or service not known (getaddrinfo failed)";
    const TIMEOUT: &str = "Request timed out after 10s";

    #[test]
    fn real_failures_return_a_reason() {
        for (msg, fragment) in [
            (GOOGLE_BAD_KEY_401, "Invalid API key"),
            (GOOGLE_PERM_403, "lacks required permissions"),
            (DNS_FAILURE, "Connection error"),
            (TIMEOUT, "timed out"),
        ] {
            assert!(connection_failure_reason(msg).unwrap().contains(fragment), "{msg}");
            let (success, message) = classify_provider_test_error(msg);
            assert!(!success && message.contains(fragment), "{msg}");
        }
    }

    #[test]
    fn provider_reached_returns_none() {
        for msg in [GOOGLE_RETIRED_MODEL_404, GOOGLE_DEPRECATED, GOOGLE_QUOTA_429, ANTHROPIC_RETIRED_HAIKU_404] {
            assert_eq!(connection_failure_reason(msg), None, "{msg}");
        }
    }

    #[test]
    fn digits_inside_dates_and_request_ids_are_not_statuses() {
        // "20240307" contains 403 but is not an HTTP 403.
        assert_eq!(connection_failure_reason(ANTHROPIC_RETIRED_HAIKU_404), None);
        assert_eq!(connection_failure_reason(ANTHROPIC_404_WITH_403_IN_REQUEST_ID), None);
        // Stripping request ids must not hide a genuine permission failure.
        assert_eq!(connection_failure_reason(ANTHROPIC_PERM_403_WITH_REQUEST_ID), Some(NO_PERMISSION));
        assert!(contains_http_status("got 403.", 403));
        assert!(contains_http_status("403", 403));
        assert!(!contains_http_status("14031", 403));
        assert!(!contains_http_status("4030", 403));
        assert_eq!(strip_request_ids("a req_011Cga403VtXy92ABc b req_test c"), "a  b req_test c");
    }

    #[test]
    fn rate_limits_are_detected_in_their_common_phrasings() {
        for msg in [
            "429 Too Many Requests",
            "Resource has been exhausted (e.g. check quota).",
            "You have hit your rate limit",
            "quota exceeded for this project",
        ] {
            assert!(is_rate_limit(msg), "{msg}");
        }
        assert!(!is_rate_limit(GOOGLE_RETIRED_MODEL_404));
    }

    #[test]
    fn provider_test_treats_model_and_quota_problems_as_success() {
        let (success, message) = classify_provider_test_error(GOOGLE_RETIRED_MODEL_404);
        assert!(success && message.contains("test model unavailable"));
        let (success, message) = classify_provider_test_error(ANTHROPIC_RETIRED_HAIKU_404);
        assert!(success && message.contains("test model unavailable") && !message.contains("permissions"));
        assert!(classify_provider_test_error(GOOGLE_DEPRECATED).0);
        let (success, message) = classify_provider_test_error(GOOGLE_QUOTA_429);
        assert!(success && message.contains("connection works"));
    }

    #[test]
    fn unrecognised_errors_stay_failures_and_are_truncated() {
        let (success, message) = classify_provider_test_error("TypeError: create_language() missing required config 'base_url'");
        assert!(!success && message.starts_with("Error:"));
        let (_, long) = classify_provider_test_error(&"x".repeat(150));
        assert_eq!(long, format!("Error: {}...", "x".repeat(100)));
    }

    #[test]
    fn individual_model_semantics_differ() {
        let (success, message) = normalize_error_message(GOOGLE_RETIRED_MODEL_404);
        assert!(!success && message.contains("Model not found"));
        assert!(classify_provider_test_error(GOOGLE_RETIRED_MODEL_404).0);
        let (success, message) = normalize_error_message(GOOGLE_BAD_KEY_401);
        assert!(!success && message.contains("Invalid API key"));
        assert_eq!(normalize_error_message("weird failure"), (false, "weird failure".to_string()));
        assert_eq!(normalize_error_message(GOOGLE_QUOTA_429), (true, RATE_LIMITED.to_string()));
    }

    #[test]
    fn typed_errors_are_settled_without_string_guessing() {
        let auth = crate::error::http_error("openai", 401, "{}", None);
        assert_eq!(provider_test_outcome(&auth), (false, "Invalid API key".to_string()));
        let forbidden = crate::error::http_error("openai", 403, "{}", None);
        assert_eq!(provider_test_outcome(&forbidden), (false, "API key lacks required permissions".to_string()));
        let missing = crate::error::http_error("anthropic", 404, r#"{"error":{"type":"not_found_error","message":"model: claude-3-haiku-20240307"}}"#, None);
        assert_eq!(provider_test_outcome(&missing), (true, "API key valid (test model unavailable)".to_string()));
        assert_eq!(model_test_outcome(&missing), (false, "Model not found on this provider".to_string()));
        let limited = crate::error::http_error("openai", 429, "{}", None);
        assert_eq!(model_test_outcome(&limited), (true, "Rate limited - but connection works".to_string()));
        assert_eq!(provider_test_outcome(&AiError::Network("x".into())), (false, UNREACHABLE.to_string()));
        assert_eq!(model_test_outcome(&AiError::Timeout("x".into())), (false, TIMED_OUT.to_string()));
        let config = AiError::Configuration("Invalid or inaccessible credentials file".into());
        assert_eq!(provider_test_outcome(&config), (false, "Invalid or inaccessible credentials file".to_string()));
        let server = crate::error::http_error("openai", 500, r#"{"error":{"message":"boom"}}"#, None);
        assert_eq!(provider_test_outcome(&server), (false, "Error: openai: boom (HTTP 500)".to_string()));
        assert_eq!(model_test_outcome(&server), (false, "openai: boom (HTTP 500)".to_string()));
    }

    fn models(n: usize) -> Value {
        json!({"data": (0..n).map(|i| json!({"id": format!("model-{i}")})).collect::<Vec<_>>()})
    }

    #[tokio::test]
    async fn openai_compatible_test_lists_models() {
        let server = MockServer::start().await;
        Mock::given(method("GET")).and(path("/v1/models")).respond_with(ResponseTemplate::new(200).set_body_json(models(5))).mount(&server).await;
        let config = ProviderConfig::new().with_base_url(format!("{}/v1", server.uri())).with_api_key("local-key");
        let result = test_provider_connection("openai-compatible", &config).await;
        assert_eq!(
            result,
            ConnectionTestResult {
                provider: "openai_compatible".into(),
                success: true,
                message: "Connected. 5 models available: model-0, model-1, model-2 (+2 more)".into(),
            }
        );
        assert_eq!(header(&request(&server, 0).await, "authorization"), Some("Bearer local-key"));

        // oMLX goes through the same ping.
        let result = test_provider_connection("omlx", &ProviderConfig::new().with_base_url(format!("{}/v1/models", server.uri()))).await;
        assert!(result.success, "{result:?}");
    }

    #[tokio::test]
    async fn server_ping_reports_statuses_and_missing_urls() {
        let server = MockServer::start().await;
        Mock::given(method("GET")).and(path("/empty/models")).respond_with(ResponseTemplate::new(200).set_body_json(json!({"data": []}))).mount(&server).await;
        Mock::given(method("GET")).and(path("/denied/models")).respond_with(ResponseTemplate::new(401)).mount(&server).await;
        Mock::given(method("GET")).and(path("/forbidden/models")).respond_with(ResponseTemplate::new(403)).mount(&server).await;
        let at = |prefix: &str| ProviderConfig::new().with_base_url(format!("{}/{prefix}", server.uri()));
        let message = |result: ConnectionTestResult| (result.success, result.message);

        assert_eq!(message(test_provider_connection("openai_compatible", &at("empty")).await), (true, "Connected successfully (no models listed)".into()));
        assert_eq!(message(test_provider_connection("openai_compatible", &at("denied")).await), (false, "Invalid API key".into()));
        assert_eq!(message(test_provider_connection("openai_compatible", &at("forbidden")).await), (false, "API key lacks required permissions".into()));
        assert_eq!(message(test_provider_connection("openai_compatible", &at("missing")).await), (false, "Server returned status 404".into()));
        assert_eq!(message(test_provider_connection("openai_compatible", &ProviderConfig::new()).await), (false, "No base URL configured".into()));
        assert_eq!(message(test_provider_connection("anthropic_compatible", &ProviderConfig::new()).await), (false, "No base URL configured".into()));
    }

    #[tokio::test]
    async fn unreachable_servers_get_provider_specific_wording() {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let dead = format!("http://{}", listener.local_addr().unwrap());
        drop(listener);
        let config = ProviderConfig::new().with_base_url(&dead).with_api_key("k");
        let message = |provider: &'static str| {
            let config = config.clone();
            async move { test_provider_connection(provider, &config).await.message }
        };
        assert_eq!(message("ollama").await, "Cannot connect to Ollama. Check if Ollama server is running.");
        assert_eq!(message("openai_compatible").await, "Cannot connect to server. Check the URL is correct.");
        assert_eq!(message("anthropic_compatible").await, "Cannot connect to server. Check the URL is correct.");
        assert_eq!(message("azure").await, "Cannot connect to Azure endpoint. Check the URL.");
        // Standard providers report the generic network wording.
        assert_eq!(message("openai").await, "Connection error - check network/endpoint");
    }

    #[tokio::test]
    async fn ollama_and_azure_pings_use_their_own_endpoints() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/api/tags"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({"models": [{"name": "llama3.2:latest"}, {"name": "bge-m3:latest"}]})))
            .mount(&server)
            .await;
        Mock::given(method("GET"))
            .and(path("/openai/models"))
            .and(query_param("api-version", "2024-10-21"))
            .respond_with(ResponseTemplate::new(200).set_body_json(models(2)))
            .mount(&server)
            .await;

        let ollama = test_provider_connection("ollama", &ProviderConfig::new().with_base_url(server.uri())).await;
        assert_eq!(ollama.message, "Connected. 2 models available: llama3.2:latest, bge-m3:latest");

        let azure_config = ProviderConfig { endpoint: Some(format!("{}/", server.uri())), ..ProviderConfig::new().with_api_key("azure-key") };
        let azure = test_provider_connection("azure", &azure_config).await;
        assert_eq!(azure.message, "Connected. 2 models: model-0, model-1");
        assert_eq!(header(&request(&server, 1).await, "api-key"), Some("azure-key"));

        let message = |config: ProviderConfig| async move { test_provider_connection("azure", &config).await.message };
        assert_eq!(message(ProviderConfig::new().with_api_key("k")).await, "No Azure endpoint configured");
        assert_eq!(message(ProviderConfig::new().with_base_url(server.uri())).await, "No Azure API key configured");
    }

    #[tokio::test]
    async fn anthropic_compatible_ping_tolerates_missing_listing() {
        let server = MockServer::start().await;
        Mock::given(method("GET")).and(path("/v1/models")).respond_with(ResponseTemplate::new(404)).mount(&server).await;
        let config = ProviderConfig::new().with_base_url(server.uri()).with_api_key("k");
        let result = test_provider_connection("anthropic_compatible", &config).await;
        assert!(result.success);
        assert_eq!(result.message, "Connected, but model listing is unsupported (status 404); add models manually.");
        let sent = request(&server, 0).await;
        assert_eq!(header(&sent, "x-api-key"), Some("k"));
        assert_eq!(header(&sent, "anthropic-version"), Some("2023-06-01"));
    }

    #[tokio::test]
    async fn standard_provider_test_calls_the_registry_test_model() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({"choices": [{"message": {"content": "Hello"}}]})))
            .mount(&server)
            .await;
        let config = ProviderConfig::new().with_api_key("sk").with_base_url(format!("{}/v1", server.uri()));
        let result = test_provider_connection("openai", &config).await;
        assert_eq!((result.success, result.message.as_str()), (true, "Connection successful"));
        let body = request_json(&server, 0).await;
        assert_eq!(body["model"], json!("gpt-3.5-turbo"));
        assert_eq!(body["messages"], json!([{"role": "user", "content": "Hi"}]));
    }

    #[tokio::test]
    async fn standard_provider_test_classifies_failures() {
        let server = MockServer::start().await;
        Mock::given(method("POST")).and(path("/bad-key/chat/completions")).respond_with(ResponseTemplate::new(401).set_body_json(json!({"error": {"message": "Incorrect API key"}}))).mount(&server).await;
        Mock::given(method("POST")).and(path("/retired/chat/completions")).respond_with(ResponseTemplate::new(404).set_body_json(json!({"error": {"message": "The model `gpt-3.5-turbo` does not exist", "code": "model_not_found"}}))).mount(&server).await;
        Mock::given(method("POST")).and(path("/busy/chat/completions")).respond_with(ResponseTemplate::new(429).set_body_json(json!({"error": {"message": "Rate limit reached"}}))).mount(&server).await;
        Mock::given(method("POST")).and(path("/v1/embeddings")).respond_with(ResponseTemplate::new(200).set_body_json(json!({"data": [{"index": 0, "embedding": [0.1]}]}))).mount(&server).await;
        let at = |prefix: &str| ProviderConfig::new().with_api_key("sk").with_base_url(format!("{}/{prefix}", server.uri()));
        let outcome = |provider: &'static str, config: ProviderConfig| async move {
            let result = test_provider_connection(provider, &config).await;
            (result.success, result.message)
        };

        assert_eq!(outcome("openai", at("bad-key")).await, (false, "Invalid API key".into()));
        assert_eq!(outcome("openai", at("retired")).await, (true, "API key valid (test model unavailable)".into()));
        assert_eq!(outcome("openai", at("busy")).await, (true, "Rate limited - but connection works".into()));
        // Embedding-only and speech-only providers follow their test type.
        assert_eq!(outcome("voyage", at("v1")).await, (true, "Connection successful".into()));
        assert_eq!(outcome("elevenlabs", at("v1")).await, (true, "Connection successful (key format valid)".into()));
        assert_eq!(outcome("deepgram", ProviderConfig::new()).await, (false, "No API key configured for deepgram".into()));
        assert_eq!(outcome("skynet", at("v1")).await, (false, "Unknown provider: skynet".into()));
        let vertex = ProviderConfig {
            project: Some("p".into()),
            location: Some("us-central1".into()),
            credentials_path: Some("/nonexistent/sa.json".into()),
            ..Default::default()
        };
        assert_eq!(outcome("vertex", vertex).await, (false, "Invalid or inaccessible credentials file".into()));
    }

    #[tokio::test]
    async fn model_test_exercises_each_modality() {
        let server = MockServer::start().await;
        let long_reply = "word ".repeat(60);
        Mock::given(method("POST")).and(path("/v1/chat/completions")).respond_with(ResponseTemplate::new(200).set_body_json(json!({"choices": [{"message": {"content": long_reply}}]}))).mount(&server).await;
        Mock::given(method("POST")).and(path("/v1/embeddings")).respond_with(ResponseTemplate::new(200).set_body_json(json!({"data": [{"index": 0, "embedding": [0.1, 0.2, 0.3]}]}))).mount(&server).await;
        Mock::given(method("POST")).and(path("/v1/audio/speech")).respond_with(ResponseTemplate::new(200).insert_header("content-type", "audio/mpeg").set_body_bytes(vec![0u8; 1234])).mount(&server).await;
        Mock::given(method("POST")).and(path("/v1/audio/transcriptions")).respond_with(ResponseTemplate::new(200).set_body_json(json!({"text": "Hello there"}))).mount(&server).await;
        let config = ProviderConfig::new().with_api_key("sk").with_base_url(format!("{}/v1", server.uri()));

        let chat = test_model("openai", "gpt-4o", "language", &config).await;
        assert!(chat.success);
        assert_eq!(chat.message, format!("Response: {}", &long_reply[..100]));
        assert_eq!(request_json(&server, 0).await["messages"], json!([{"role": "user", "content": "Hi!"}]));

        let embedding = test_model("openai", "text-embedding-3-small", "embedding", &config).await;
        assert_eq!(embedding.message, "Embedding dimensions: 3");
        assert_eq!(request_json(&server, 1).await["input"], json!(["This is a test."]));

        let speech = test_model("openai", "tts-1", "text_to_speech", &config).await;
        assert_eq!(speech.message, "Audio generated: 1234 bytes");
        assert_eq!(request_json(&server, 2).await, json!({"model": "tts-1", "input": "Hello from Open Notebook", "voice": "alloy", "response_format": "mp3"}));

        let transcription = test_model("openai", "whisper-1", "speech_to_text", &config).await;
        assert_eq!(transcription.message, "Transcription: Hello there");
        assert!(String::from_utf8_lossy(&request(&server, 3).await.body).contains("filename=\"test_speech.mp3\""));

        let odd = test_model("openai", "gpt-4o", "vision", &config).await;
        assert_eq!((odd.success, odd.message.as_str()), (false, "Unsupported model type: vision"));
        let mismatch = test_model("anthropic", "claude", "embedding", &config).await;
        assert_eq!((mismatch.success, mismatch.message.as_str()), (false, "Anthropic does not offer embedding models"));
    }

    #[tokio::test]
    async fn model_test_reports_missing_models_and_empty_results() {
        let server = MockServer::start().await;
        Mock::given(method("POST")).and(path("/gone/chat/completions")).respond_with(ResponseTemplate::new(404).set_body_json(json!({"error": {"message": "The model `x` does not exist", "code": "model_not_found"}}))).mount(&server).await;
        Mock::given(method("POST")).and(path("/quiet/chat/completions")).respond_with(ResponseTemplate::new(200).set_body_json(json!({"choices": [{"message": {"content": ""}}]}))).mount(&server).await;
        Mock::given(method("POST")).and(path("/quiet/audio/transcriptions")).respond_with(ResponseTemplate::new(200).set_body_json(json!({"text": "  "}))).mount(&server).await;
        let at = |prefix: &str| ProviderConfig::new().with_api_key("sk").with_base_url(format!("{}/{prefix}", server.uri()));

        let gone = test_model("openai", "x", "language", &at("gone")).await;
        assert_eq!((gone.success, gone.message.as_str()), (false, "Model not found on this provider"));
        let quiet = test_model("openai", "x", "language", &at("quiet")).await;
        assert_eq!((quiet.success, quiet.message.as_str()), (true, "Response: (empty response)"));
        let silent = test_model("groq", "whisper-large-v3", "speech_to_text", &at("quiet")).await;
        assert_eq!(silent.message, "Connection successful (test clip produced no transcription)");
    }

    #[tokio::test]
    async fn speech_test_looks_up_a_voice_when_none_is_known() {
        let server = MockServer::start().await;
        Mock::given(method("GET")).and(path("/v1/voices")).respond_with(ResponseTemplate::new(200).set_body_json(json!({"voices": [{"voice_id": "voice-abc", "name": "Rachel"}]}))).mount(&server).await;
        Mock::given(method("POST")).and(path("/v1/text-to-speech/voice-abc")).respond_with(ResponseTemplate::new(200).insert_header("content-type", "audio/mpeg").set_body_bytes(vec![1u8; 10])).mount(&server).await;
        let config = ProviderConfig::new().with_api_key("xi").with_base_url(format!("{}/v1", server.uri()));
        let result = test_model("elevenlabs", "eleven_multilingual_v2", "text_to_speech", &config).await;
        assert_eq!((result.success, result.message.as_str()), (true, "Audio generated: 10 bytes"));
    }
}
