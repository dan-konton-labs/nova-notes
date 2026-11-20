//! Provider-neutral AI traits.
//!
//! This crate is the contract between the provider implementations (`nn-ai`)
//! and everything that consumes a model (content extraction, podcasts, the
//! server). It contains no HTTP code and no provider knowledge, so consumers
//! can be built and tested against mock models.

use std::fmt;
use std::sync::Arc;

use async_trait::async_trait;
use bytes::Bytes;
use futures::stream::BoxStream;
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// Everything that can go wrong when calling a model.
///
/// The variants are the classification the server needs to pick an HTTP status
/// and to decide whether a background job may retry.
#[derive(Debug, Clone, thiserror::Error)]
pub enum AiError {
    /// Bad, missing or revoked API key (HTTP 401/403 from the provider).
    #[error("{0}")]
    Authentication(String),
    /// Provider rate limit or quota exhausted (HTTP 429).
    #[error("{0}")]
    RateLimit(String),
    /// The request exceeds the model's context window.
    #[error("{0}")]
    ContextLength(String),
    /// The model (or deployment) does not exist for this provider.
    #[error("{0}")]
    ModelNotFound(String),
    /// The provider rejected the request as malformed (HTTP 400/422).
    #[error("{0}")]
    InvalidRequest(String),
    /// The provider is missing configuration we need (no key, no base URL...).
    #[error("{0}")]
    Configuration(String),
    /// Could not reach the provider: DNS, connect, TLS, reset.
    #[error("{0}")]
    Network(String),
    /// The provider did not answer in time.
    #[error("{0}")]
    Timeout(String),
    /// The provider answered with an error we do not classify further.
    #[error("{message}")]
    Provider { status: Option<u16>, message: String },
    /// The model returned empty or truncated output.
    #[error("{0}")]
    Incomplete(String),
    /// This provider / model type combination is not supported.
    #[error("{0}")]
    Unsupported(String),
    /// Anything else.
    #[error("{0}")]
    Other(String),
}

impl AiError {
    /// True when sending the same request again could plausibly succeed.
    pub fn is_retryable(&self) -> bool {
        match self {
            AiError::RateLimit(_) | AiError::Network(_) | AiError::Timeout(_) => true,
            AiError::Provider { status, .. } => matches!(status, Some(s) if *s >= 500),
            _ => false,
        }
    }
}

pub type AiResult<T> = Result<T, AiError>;

// ---------------------------------------------------------------------------
// Chat
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    System,
    User,
    Assistant,
}

impl Role {
    pub fn as_str(&self) -> &'static str {
        match self {
            Role::System => "system",
            Role::User => "user",
            Role::Assistant => "assistant",
        }
    }
}

impl fmt::Display for Role {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: Role,
    pub content: String,
}

impl ChatMessage {
    pub fn new(role: Role, content: impl Into<String>) -> Self {
        Self { role, content: content.into() }
    }
    pub fn system(content: impl Into<String>) -> Self {
        Self::new(Role::System, content)
    }
    pub fn user(content: impl Into<String>) -> Self {
        Self::new(Role::User, content)
    }
    pub fn assistant(content: impl Into<String>) -> Self {
        Self::new(Role::Assistant, content)
    }
}

/// Per-call generation settings. `None` means "use the provider default".
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ChatOptions {
    pub temperature: Option<f32>,
    pub max_tokens: Option<u32>,
    pub top_p: Option<f32>,
    /// Ask the provider for a JSON object response where it supports that
    /// (OpenAI `response_format`, Gemini `responseMimeType`, Ollama `format`).
    /// Providers without a JSON mode ignore it; callers must still parse
    /// defensively.
    pub json: bool,
}

impl ChatOptions {
    pub fn json() -> Self {
        Self { json: true, ..Self::default() }
    }
    pub fn with_max_tokens(mut self, max_tokens: u32) -> Self {
        self.max_tokens = Some(max_tokens);
        self
    }
    pub fn with_temperature(mut self, temperature: f32) -> Self {
        self.temperature = Some(temperature);
        self
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Usage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ChatResponse {
    /// The assistant text, exactly as the provider returned it (thinking tags
    /// are *not* stripped here).
    pub content: String,
    /// Model name the provider reports having used.
    pub model: String,
    /// Provider finish reason, normalised to lowercase when present
    /// (`stop`, `length`, `content_filter`, ...).
    pub finish_reason: Option<String>,
    pub usage: Option<Usage>,
}

/// A stream of text deltas. Concatenating every `Ok` item gives the full reply.
pub type TextStream = BoxStream<'static, AiResult<String>>;

#[async_trait]
pub trait LanguageModel: Send + Sync {
    /// Provider id as used in the provider registry (`openai`, `anthropic`...).
    fn provider(&self) -> &str;
    fn model_name(&self) -> &str;

    async fn chat(&self, messages: &[ChatMessage], options: &ChatOptions) -> AiResult<ChatResponse>;

    /// Streaming variant of [`chat`](Self::chat). Errors that happen before the
    /// first byte are returned from the call; later errors arrive in the stream.
    async fn chat_stream(&self, messages: &[ChatMessage], options: &ChatOptions) -> AiResult<TextStream>;

    /// Convenience: single user prompt, default options, text out.
    async fn complete(&self, prompt: &str) -> AiResult<String> {
        let reply = self.chat(&[ChatMessage::user(prompt)], &ChatOptions::default()).await?;
        Ok(reply.content)
    }
}

// ---------------------------------------------------------------------------
// Embeddings
// ---------------------------------------------------------------------------

#[async_trait]
pub trait EmbeddingModel: Send + Sync {
    fn provider(&self) -> &str;
    fn model_name(&self) -> &str;

    /// One vector per input, in input order. Implementations batch internally
    /// when the provider caps the batch size.
    async fn embed(&self, texts: &[String]) -> AiResult<Vec<Vec<f32>>>;
}

// ---------------------------------------------------------------------------
// Speech
// ---------------------------------------------------------------------------

/// Encoded audio as returned by a text-to-speech provider.
#[derive(Debug, Clone, PartialEq)]
pub struct AudioData {
    pub bytes: Bytes,
    /// MIME type, e.g. `audio/mpeg` or `audio/wav`.
    pub content_type: String,
}

impl AudioData {
    /// File extension (no dot) matching `content_type`; `mp3` when unknown.
    pub fn extension(&self) -> &'static str {
        let ct = self.content_type.to_ascii_lowercase();
        if ct.contains("wav") {
            "wav"
        } else if ct.contains("ogg") || ct.contains("opus") {
            "ogg"
        } else if ct.contains("flac") {
            "flac"
        } else if ct.contains("aac") {
            "aac"
        } else if ct.contains("mp4") || ct.contains("m4a") {
            "m4a"
        } else if ct.contains("pcm") || ct.contains("l16") {
            "pcm"
        } else {
            "mp3"
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Voice {
    pub id: String,
    pub name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub language: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gender: Option<String>,
}

#[async_trait]
pub trait TextToSpeechModel: Send + Sync {
    fn provider(&self) -> &str;
    fn model_name(&self) -> &str;

    /// Speak `text` with the provider voice id `voice`.
    async fn synthesize(&self, text: &str, voice: &str) -> AiResult<AudioData>;

    /// Voices the provider offers, when it can list them. Default: none.
    async fn voices(&self) -> AiResult<Vec<Voice>> {
        Ok(Vec::new())
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct TranscriptionRequest {
    pub audio: Bytes,
    /// File name including extension; providers sniff the format from it.
    pub filename: String,
    pub content_type: Option<String>,
    /// ISO 639-1 hint, when the caller knows the language.
    pub language: Option<String>,
    /// Optional context prompt (previous segment text, vocabulary).
    pub prompt: Option<String>,
}

#[async_trait]
pub trait SpeechToTextModel: Send + Sync {
    fn provider(&self) -> &str;
    fn model_name(&self) -> &str;

    /// Plain transcript text.
    async fn transcribe(&self, request: TranscriptionRequest) -> AiResult<String>;
}

pub type SharedLanguageModel = Arc<dyn LanguageModel>;
pub type SharedEmbeddingModel = Arc<dyn EmbeddingModel>;
pub type SharedTextToSpeechModel = Arc<dyn TextToSpeechModel>;
pub type SharedSpeechToTextModel = Arc<dyn SpeechToTextModel>;
