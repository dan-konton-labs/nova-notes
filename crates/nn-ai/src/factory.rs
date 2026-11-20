//! Turning `(provider, model, config)` into a model object.
//!
//! This is the Rust counterpart of esperanto's `AIFactory`. The free
//! functions use a process-wide connection pool; [`Clients`] (or the
//! `*_with_client` variants) lets the server supply its own `reqwest::Client`,
//! for example one configured with a private CA bundle.
//!
//! Construction never touches the network. It fails with
//! [`AiError::Unsupported`] for an unknown provider or a modality the
//! provider does not offer, and with [`AiError::Configuration`] when the
//! credential lacks something the provider needs.

use std::sync::Arc;

use nn_llm::{
    AiError, AiResult, SharedEmbeddingModel, SharedLanguageModel, SharedSpeechToTextModel, SharedTextToSpeechModel,
};

use crate::config::{ProviderConfig, DEFAULT_CHAT_TIMEOUT, DEFAULT_EMBEDDING_TIMEOUT, DEFAULT_SPEECH_TIMEOUT};
use crate::http::shared_client;
use crate::providers::anthropic::AnthropicChat;
use crate::providers::cohere::{CohereChat, CohereEmbedder};
use crate::providers::deepgram::{DeepgramSpeech, DeepgramTranscriber};
use crate::providers::elevenlabs::{ElevenLabsSpeech, ElevenLabsTranscriber};
use crate::providers::google::{GeminiChat, GeminiEmbedder, GeminiSpeech, GeminiTranscriber, GoogleEndpoint};
use crate::providers::minimax::MiniMaxSpeech;
use crate::providers::ollama::{OllamaChat, OllamaEmbedder};
use crate::providers::openai::{OpenAiChat, OpenAiEmbedder, OpenAiSpeech, OpenAiTranscriber};
use crate::providers::voyage::VoyageEmbedder;
use crate::providers::Batched;
use crate::registry::{self, Modality, ProviderSpec};

/// Builds models on top of one HTTP connection pool. Cheap to clone.
#[derive(Clone, Debug)]
pub struct Clients {
    http: reqwest::Client,
}

impl Default for Clients {
    fn default() -> Self {
        Self::new()
    }
}

/// Checks the provider exists, offers `modality`, and that a model was named.
fn resolve(provider: &str, model: &str, modality: Modality) -> AiResult<&'static ProviderSpec> {
    let spec = registry::provider(provider)
        .ok_or_else(|| AiError::Unsupported(format!("Unknown provider: {}", provider.trim())))?;
    if !spec.supports(modality) {
        return Err(unsupported(spec, modality));
    }
    if model.trim().is_empty() {
        return Err(AiError::Configuration(format!("No model name given for {}", spec.display_name)));
    }
    Ok(spec)
}

fn unsupported(spec: &ProviderSpec, modality: Modality) -> AiError {
    AiError::Unsupported(format!("{} does not offer {} models", spec.display_name, modality.human()))
}

impl Clients {
    /// Uses the process-wide connection pool.
    pub fn new() -> Self {
        Self { http: shared_client() }
    }

    /// Uses the caller's client (and with it the caller's TLS, proxy and
    /// pool settings). Per-request timeouts are still applied by this crate.
    pub fn with_client(client: reqwest::Client) -> Self {
        Self { http: client }
    }

    pub fn http(&self) -> &reqwest::Client {
        &self.http
    }

    pub fn language_model(&self, provider: &str, model: &str, config: &ProviderConfig) -> AiResult<SharedLanguageModel> {
        let spec = resolve(provider, model, Modality::Language)?;
        let (http, name, model) = (&self.http, spec.name, model.trim());
        Ok(match name {
            "anthropic" | "anthropic_compatible" => Arc::new(AnthropicChat::new(http, name, model, config)?),
            "google" | "vertex" => {
                Arc::new(GeminiChat::new(GoogleEndpoint::resolve(http, name, model, config, DEFAULT_CHAT_TIMEOUT)?))
            }
            "cohere" => Arc::new(CohereChat::new(http, model, config)?),
            "ollama" => Arc::new(OllamaChat::new(http, model, config)),
            "openai" | "groq" | "mistral" | "deepseek" | "xai" | "openrouter" | "dashscope" | "minimax" | "novita"
            | "siliconflow" | "zai" | "ppq" | "omlx" | "openai_compatible" | "azure" => {
                Arc::new(OpenAiChat::new(http, name, model, config)?)
            }
            _ => return Err(unsupported(spec, Modality::Language)),
        })
    }

    pub fn embedding_model(&self, provider: &str, model: &str, config: &ProviderConfig) -> AiResult<SharedEmbeddingModel> {
        let spec = resolve(provider, model, Modality::Embedding)?;
        let (http, name, model) = (&self.http, spec.name, model.trim());
        Ok(match name {
            "google" | "vertex" => {
                let endpoint = GoogleEndpoint::resolve(http, name, model, config, DEFAULT_EMBEDDING_TIMEOUT)?;
                Arc::new(Batched::new(GeminiEmbedder::new(endpoint, config), config))
            }
            "cohere" => Arc::new(Batched::new(CohereEmbedder::new(http, model, config)?, config)),
            "voyage" => Arc::new(Batched::new(VoyageEmbedder::new(http, model, config)?, config)),
            "ollama" => Arc::new(Batched::new(OllamaEmbedder::new(http, model, config), config)),
            "openai" | "mistral" | "openrouter" | "ppq" | "omlx" | "openai_compatible" | "azure" => {
                Arc::new(Batched::new(OpenAiEmbedder::new(http, name, model, config)?, config))
            }
            _ => return Err(unsupported(spec, Modality::Embedding)),
        })
    }

    pub fn text_to_speech_model(
        &self,
        provider: &str,
        model: &str,
        config: &ProviderConfig,
    ) -> AiResult<SharedTextToSpeechModel> {
        let spec = resolve(provider, model, Modality::TextToSpeech)?;
        let (http, name, model) = (&self.http, spec.name, model.trim());
        Ok(match name {
            "google" | "vertex" => {
                Arc::new(GeminiSpeech::new(GoogleEndpoint::resolve(http, name, model, config, DEFAULT_SPEECH_TIMEOUT)?))
            }
            "elevenlabs" => Arc::new(ElevenLabsSpeech::new(http, model, config)?),
            "deepgram" => Arc::new(DeepgramSpeech::new(http, model, config)?),
            "minimax" => Arc::new(MiniMaxSpeech::new(http, model, config)?),
            "openai" | "mistral" | "xai" | "openrouter" | "ppq" | "openai_compatible" | "azure" => {
                Arc::new(OpenAiSpeech::new(http, name, model, config)?)
            }
            _ => return Err(unsupported(spec, Modality::TextToSpeech)),
        })
    }

    pub fn speech_to_text_model(
        &self,
        provider: &str,
        model: &str,
        config: &ProviderConfig,
    ) -> AiResult<SharedSpeechToTextModel> {
        let spec = resolve(provider, model, Modality::SpeechToText)?;
        let (http, name, model) = (&self.http, spec.name, model.trim());
        Ok(match name {
            "google" => {
                Arc::new(GeminiTranscriber::new(GoogleEndpoint::resolve(http, name, model, config, DEFAULT_SPEECH_TIMEOUT)?))
            }
            "elevenlabs" => Arc::new(ElevenLabsTranscriber::new(http, model, config)?),
            "deepgram" => Arc::new(DeepgramTranscriber::new(http, model, config)?),
            "openai" | "groq" | "mistral" | "openrouter" | "ppq" | "openai_compatible" | "azure" => {
                Arc::new(OpenAiTranscriber::new(http, name, model, config)?)
            }
            _ => return Err(unsupported(spec, Modality::SpeechToText)),
        })
    }
}

/// A chat model for `provider` (a registry id such as `openai`).
pub fn language_model(provider: &str, model: &str, config: &ProviderConfig) -> AiResult<SharedLanguageModel> {
    Clients::new().language_model(provider, model, config)
}

/// An embedding model; batching to the provider's limit is built in.
pub fn embedding_model(provider: &str, model: &str, config: &ProviderConfig) -> AiResult<SharedEmbeddingModel> {
    Clients::new().embedding_model(provider, model, config)
}

pub fn text_to_speech_model(provider: &str, model: &str, config: &ProviderConfig) -> AiResult<SharedTextToSpeechModel> {
    Clients::new().text_to_speech_model(provider, model, config)
}

pub fn speech_to_text_model(provider: &str, model: &str, config: &ProviderConfig) -> AiResult<SharedSpeechToTextModel> {
    Clients::new().speech_to_text_model(provider, model, config)
}

/// [`language_model`] on the caller's connection pool.
pub fn language_model_with_client(
    provider: &str,
    model: &str,
    config: &ProviderConfig,
    client: &reqwest::Client,
) -> AiResult<SharedLanguageModel> {
    Clients::with_client(client.clone()).language_model(provider, model, config)
}

/// [`embedding_model`] on the caller's connection pool.
pub fn embedding_model_with_client(
    provider: &str,
    model: &str,
    config: &ProviderConfig,
    client: &reqwest::Client,
) -> AiResult<SharedEmbeddingModel> {
    Clients::with_client(client.clone()).embedding_model(provider, model, config)
}

/// [`text_to_speech_model`] on the caller's connection pool.
pub fn text_to_speech_model_with_client(
    provider: &str,
    model: &str,
    config: &ProviderConfig,
    client: &reqwest::Client,
) -> AiResult<SharedTextToSpeechModel> {
    Clients::with_client(client.clone()).text_to_speech_model(provider, model, config)
}

/// [`speech_to_text_model`] on the caller's connection pool.
pub fn speech_to_text_model_with_client(
    provider: &str,
    model: &str,
    config: &ProviderConfig,
    client: &reqwest::Client,
) -> AiResult<SharedSpeechToTextModel> {
    Clients::with_client(client.clone()).speech_to_text_model(provider, model, config)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::registry::PROVIDERS;

    /// A credential with every field filled in, pointing nowhere.
    fn dummy_config() -> ProviderConfig {
        ProviderConfig {
            api_key: Some("dummy-key".into()),
            base_url: Some("http://127.0.0.1:9/v1".into()),
            endpoint: Some("http://127.0.0.1:9".into()),
            api_version: Some("2024-10-21".into()),
            project: Some("project".into()),
            location: Some("us-central1".into()),
            credentials_path: Some("/nonexistent/service-account.json".into()),
            ..Default::default()
        }
    }

    /// Whether construction succeeds; an error other than `Unsupported` is
    /// a test failure.
    fn constructs(clients: &Clients, spec: &ProviderSpec, modality: Modality) -> bool {
        let config = dummy_config();
        let result = match modality {
            Modality::Language => clients.language_model(spec.name, "model", &config).map(|m| m.provider().to_string()),
            Modality::Embedding => clients.embedding_model(spec.name, "model", &config).map(|m| m.provider().to_string()),
            Modality::TextToSpeech => {
                clients.text_to_speech_model(spec.name, "model", &config).map(|m| m.provider().to_string())
            }
            Modality::SpeechToText => {
                clients.speech_to_text_model(spec.name, "model", &config).map(|m| m.provider().to_string())
            }
        };
        match result {
            Ok(provider) => {
                assert_eq!(provider, spec.name, "the model must report its registry id");
                true
            }
            Err(AiError::Unsupported(_)) => false,
            Err(other) => panic!("{} {modality}: unexpected error {other:?}", spec.name),
        }
    }

    #[test]
    fn every_provider_constructs_exactly_the_modalities_it_advertises() {
        let clients = Clients::new();
        for spec in PROVIDERS {
            for modality in Modality::ALL {
                assert_eq!(
                    constructs(&clients, spec, modality),
                    spec.supports(modality),
                    "{} / {modality}",
                    spec.name
                );
            }
        }
    }

    #[test]
    fn unknown_providers_and_modalities_are_unsupported_with_clear_messages() {
        let config = dummy_config();
        let err = language_model("skynet", "m", &config).err().unwrap();
        assert!(matches!(&err, AiError::Unsupported(m) if m == "Unknown provider: skynet"));
        let err = embedding_model("anthropic", "m", &config).err().unwrap();
        assert!(matches!(&err, AiError::Unsupported(m) if m == "Anthropic does not offer embedding models"));
        let err = speech_to_text_model("vertex", "m", &config).err().unwrap();
        assert!(matches!(&err, AiError::Unsupported(m) if m == "Google Vertex AI does not offer speech-to-text models"));
        let err = text_to_speech_model("groq", "m", &config).err().unwrap();
        assert!(matches!(&err, AiError::Unsupported(m) if m == "Groq does not offer text-to-speech models"));
    }

    #[test]
    fn missing_credentials_are_configuration_errors() {
        let empty = ProviderConfig::default();
        for (provider, expected) in [
            ("openai", "No API key configured for openai"),
            ("anthropic", "No API key configured for anthropic"),
            ("google", "No API key configured for google"),
            ("cohere", "No API key configured for cohere"),
            ("openai_compatible", "No base URL configured for openai_compatible"),
            ("anthropic_compatible", "Anthropic-compatible models require a base URL and API key"),
            ("azure", "No Azure endpoint configured"),
            ("vertex", "No Vertex AI project configured"),
        ] {
            let err = language_model(provider, "m", &empty).err().unwrap();
            assert!(matches!(&err, AiError::Configuration(m) if m == expected), "{provider}: {err:?}");
        }
        assert!(matches!(embedding_model("voyage", "m", &empty).err(), Some(AiError::Configuration(_))));
        assert!(matches!(text_to_speech_model("elevenlabs", "m", &empty).err(), Some(AiError::Configuration(_))));
        assert!(matches!(speech_to_text_model("deepgram", "m", &empty).err(), Some(AiError::Configuration(_))));
        // Local providers work without any configuration.
        assert!(language_model("ollama", "llama3.2", &empty).is_ok());
        assert!(embedding_model("omlx", "bge", &empty).is_ok());
        let err = language_model("openai", "  ", &dummy_config()).err().unwrap();
        assert!(matches!(&err, AiError::Configuration(m) if m == "No model name given for OpenAI"));
    }

    #[test]
    fn provider_names_are_normalised_and_models_trimmed() {
        let model = language_model("OpenAI-Compatible", " local-model ", &dummy_config()).unwrap();
        assert_eq!(model.provider(), "openai_compatible");
        assert_eq!(model.model_name(), "local-model");
        let client = reqwest::Client::new();
        let model = language_model_with_client("anthropic", "claude", &dummy_config(), &client).unwrap();
        assert_eq!(model.provider(), "anthropic");
    }
}
