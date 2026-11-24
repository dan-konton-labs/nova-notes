//! Multi-provider AI clients for Nova Notes.
//!
//! This crate implements the provider-neutral traits of `nn-llm`
//! (`LanguageModel`, `EmbeddingModel`, `TextToSpeechModel`,
//! `SpeechToTextModel`) for every provider in the [`registry`]. It is the
//! Rust counterpart of the `esperanto` library plus `open_notebook/ai/*.py`
//! in the Python original.
//!
//! The crate has no database access and reads no environment variables:
//! everything it needs arrives in a [`ProviderConfig`], which the server
//! fills from a decrypted credential.
//!
//! ```no_run
//! use nn_ai::{language_model, ProviderConfig};
//! use nn_llm::{ChatMessage, ChatOptions};
//!
//! # async fn demo() -> nn_llm::AiResult<()> {
//! let config = ProviderConfig::new().with_api_key("sk-...");
//! let model = language_model("openai", "gpt-4o-mini", &config)?;
//! let reply = model.chat(&[ChatMessage::user("Hello")], &ChatOptions::default()).await?;
//! println!("{}", reply.content);
//! # Ok(())
//! # }
//! ```
//!
//! # Layout
//!
//! * [`registry`]: the static provider table (names, modalities, env vars,
//!   test models, discovery URLs).
//! * [`config`]: [`ProviderConfig`] and the default timeouts.
//! * [`factory`]: `(provider, model, config)` to a model object, via free
//!   functions or [`Clients`] when the caller owns the HTTP client.
//! * [`discovery`]: listing the models a credential can use, with a guessed
//!   model type.
//! * [`tester`]: connection and single-model tests with user-facing messages.
//! * [`error`]: HTTP and transport failures mapped onto `nn_llm::AiError`,
//!   and the messages shown to users.
//! * [`sse`]: Server-Sent Events and NDJSON reassembly shared by every
//!   streaming implementation.
//! * [`audio`]: WAV framing for raw PCM and the clips used by tests.
//!
//! # Not handled here
//!
//! URL validation against SSRF and DNS pinning belong to the server, which
//! must validate user-supplied URLs before putting them in a
//! [`ProviderConfig`]. Custom TLS settings (private CA bundles) are supplied
//! by passing a configured `reqwest::Client` to [`Clients::with_client`].

pub mod audio;
pub mod config;
pub mod discovery;
pub mod error;
pub mod factory;
mod http;
mod providers;
pub mod registry;
pub mod sse;
pub mod tester;

pub use config::{ProviderConfig, DEFAULT_CHAT_TIMEOUT, DEFAULT_EMBEDDING_TIMEOUT, DEFAULT_SPEECH_TIMEOUT};
pub use discovery::{
    audio_seed, classify_model_type, discover_models, discover_models_with_client, DiscoveredModel,
    ANTHROPIC_FALLBACK_MODELS,
};
pub use error::{classify_error_text, http_status, user_message};
pub use factory::{
    embedding_model, embedding_model_with_client, language_model, language_model_with_client, speech_to_text_model,
    speech_to_text_model_with_client, text_to_speech_model, text_to_speech_model_with_client, Clients,
};
pub use providers::anthropic::normalize_anthropic_compatible_base_url;
pub use registry::{base_url_rule, normalize_provider_name, provider, BaseUrlRule, providers, EnvConfig, Modality, ProviderInfo, ProviderSpec, PROVIDERS};
pub use tester::{
    default_test_voice, test_model, test_model_with_client, test_provider_connection,
    test_provider_connection_with_client, ConnectionTestResult,
};
