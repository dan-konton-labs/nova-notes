//! Provider configuration: everything a stored credential can carry.
//!
//! The server loads a credential from the database, decrypts it and fills a
//! [`ProviderConfig`]. This crate never reads credentials from the
//! environment.

use std::fmt;
use std::time::Duration;

use serde_json::{Map, Value};

/// Chat calls are limited to 180 s unless [`ProviderConfig::timeout`] says
/// otherwise (the original sets `ESPERANTO_LLM_TIMEOUT=180`).
pub const DEFAULT_CHAT_TIMEOUT: Duration = Duration::from_secs(180);
/// Default limit for one embedding batch.
pub const DEFAULT_EMBEDDING_TIMEOUT: Duration = Duration::from_secs(120);
/// Default limit for speech synthesis and transcription.
pub const DEFAULT_SPEECH_TIMEOUT: Duration = Duration::from_secs(300);

/// Configuration for one provider account. Every field is optional; each
/// provider documents which ones it needs and reports a
/// [`Configuration`](nn_llm::AiError::Configuration) error when one is missing.
///
/// Keys read from [`extra`](Self::extra) (the credential's `config` object):
///
/// | key | used by | meaning |
/// |---|---|---|
/// | `num_ctx` | ollama | context window, default 8192 |
/// | `keep_alive` | ollama | passed through to `/api/chat` |
/// | `max_tokens` | anthropic | output limit when the caller sets none |
/// | `embedding_batch_size` | all embedders | overrides the provider batch limit |
/// | `dimensions` | openai family, voyage, cohere, google | output vector size |
/// | `input_type` | cohere, voyage | defaults to `search_document` / `document` |
/// | `task_type` | google, vertex | embedding task type, unset by default |
/// | `language` | xai text-to-speech | defaults to `auto` |
/// | `output_format` | elevenlabs | defaults to `mp3_44100_128` |
#[derive(Clone, Default, PartialEq)]
pub struct ProviderConfig {
    pub api_key: Option<String>,
    pub base_url: Option<String>,

    // Azure OpenAI
    pub endpoint: Option<String>,
    pub api_version: Option<String>,
    pub endpoint_llm: Option<String>,
    pub endpoint_embedding: Option<String>,
    pub endpoint_stt: Option<String>,
    pub endpoint_tts: Option<String>,

    // Google Vertex AI
    pub project: Option<String>,
    pub location: Option<String>,
    /// Path to a service-account JSON file readable by this process.
    pub credentials_path: Option<String>,

    /// The credential's free-form `config` object.
    pub extra: Map<String, Value>,
    /// Overrides the per-modality default request timeout.
    pub timeout: Option<Duration>,
}

impl ProviderConfig {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_api_key(mut self, api_key: impl Into<String>) -> Self {
        self.api_key = Some(api_key.into());
        self
    }

    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = Some(base_url.into());
        self
    }

    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = Some(timeout);
        self
    }

    pub fn with_extra(mut self, key: impl Into<String>, value: impl Into<Value>) -> Self {
        self.extra.insert(key.into(), value.into());
        self
    }

    /// The API key, if one is set and not blank.
    pub(crate) fn key(&self) -> Option<&str> {
        non_blank(&self.api_key)
    }

    /// The base URL without trailing slashes, if one is set and not blank.
    pub(crate) fn url(&self) -> Option<&str> {
        non_blank(&self.base_url).map(|u| u.trim_end_matches('/'))
    }

    pub(crate) fn extra_str(&self, key: &str) -> Option<&str> {
        self.extra.get(key).and_then(Value::as_str).map(str::trim).filter(|s| !s.is_empty())
    }

    /// Reads an integer option, accepting a JSON number or a numeric string
    /// (form fields often arrive as strings).
    pub(crate) fn extra_u64(&self, key: &str) -> Option<u64> {
        match self.extra.get(key)? {
            Value::Number(n) => n.as_u64(),
            Value::String(s) => s.trim().parse().ok(),
            _ => None,
        }
    }
}

pub(crate) fn non_blank(value: &Option<String>) -> Option<&str> {
    value.as_deref().map(str::trim).filter(|s| !s.is_empty())
}

fn looks_secret(key: &str) -> bool {
    let key = key.to_ascii_lowercase();
    ["key", "token", "secret", "password", "authorization"].iter().any(|w| key.contains(w))
}

/// `api_key` and secret-looking `extra` entries are never printed.
impl fmt::Debug for ProviderConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let extra: Map<String, Value> = self
            .extra
            .iter()
            .map(|(k, v)| {
                let shown = if looks_secret(k) { Value::from("[redacted]") } else { v.clone() };
                (k.clone(), shown)
            })
            .collect();
        f.debug_struct("ProviderConfig")
            .field("api_key", &self.api_key.as_ref().map(|_| "[redacted]"))
            .field("base_url", &self.base_url)
            .field("endpoint", &self.endpoint)
            .field("api_version", &self.api_version)
            .field("endpoint_llm", &self.endpoint_llm)
            .field("endpoint_embedding", &self.endpoint_embedding)
            .field("endpoint_stt", &self.endpoint_stt)
            .field("endpoint_tts", &self.endpoint_tts)
            .field("project", &self.project)
            .field("location", &self.location)
            .field("credentials_path", &self.credentials_path)
            .field("extra", &extra)
            .field("timeout", &self.timeout)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn debug_redacts_the_api_key_and_secret_extras() {
        let config = ProviderConfig::new()
            .with_api_key("sk-super-secret")
            .with_base_url("https://example.test/v1")
            .with_extra("access_token", "tok-123")
            .with_extra("num_ctx", 4096);
        let shown = format!("{config:?}");
        assert!(!shown.contains("sk-super-secret"));
        assert!(!shown.contains("tok-123"));
        assert!(shown.contains("[redacted]"));
        assert!(shown.contains("https://example.test/v1"));
        assert!(shown.contains("4096"));
    }

    #[test]
    fn blank_values_count_as_missing() {
        let config = ProviderConfig { api_key: Some("  ".into()), base_url: Some(" ".into()), ..Default::default() };
        assert_eq!(config.key(), None);
        assert_eq!(config.url(), None);
    }

    #[test]
    fn base_url_loses_trailing_slashes() {
        let config = ProviderConfig::new().with_base_url(" http://localhost:1234/v1// ");
        assert_eq!(config.url(), Some("http://localhost:1234/v1"));
    }

    #[test]
    fn integer_extras_accept_numbers_and_strings() {
        let config = ProviderConfig::new().with_extra("num_ctx", 16384).with_extra("embedding_batch_size", "8");
        assert_eq!(config.extra_u64("num_ctx"), Some(16384));
        assert_eq!(config.extra_u64("embedding_batch_size"), Some(8));
        assert_eq!(config.extra_u64("missing"), None);
    }
}
