//! The OpenAI wire format and everything that speaks it.
//!
//! One implementation serves OpenAI itself, the hosted OpenAI-compatible
//! providers (Groq, Mistral, DeepSeek, xAI, OpenRouter, DashScope, MiniMax,
//! Novita, SiliconFlow, Z.ai, PPQ), local servers (oMLX, any
//! `openai_compatible` endpoint) and Azure OpenAI, which differs only in how
//! URLs are built and how the key is sent.
//!
//! Per-provider quirks handled here:
//!
//! * Azure: `{endpoint}/openai/deployments/{deployment}/{path}?api-version=..`,
//!   `api-key` header, per-modality endpoint overrides.
//! * OpenAI reasoning models (o-series, gpt-5 family) take
//!   `max_completion_tokens` and reject `temperature`/`top_p`.
//! * Servers that reject a parameter (`response_format`, `max_tokens`,
//!   sampling settings) get one adapted retry instead of a hard failure.
//! * Speech: Mistral and xAI use their own request bodies on the same base
//!   URL; OpenRouter transcription takes base64 JSON instead of multipart.

use async_trait::async_trait;
use base64::Engine;
use nn_llm::{
    AiError, AiResult, AudioData, ChatMessage, ChatOptions, ChatResponse, LanguageModel, SpeechToTextModel,
    TextStream, TextToSpeechModel, TranscriptionRequest, Voice,
};
use reqwest::multipart::{Form, Part};
use reqwest::RequestBuilder;
use serde_json::{json, Map, Value};

use super::{content_text, finish_reason, float, usage, vector, EmbedBatch};
use crate::audio::{audio_format, audio_mime};
use crate::config::{non_blank, ProviderConfig, DEFAULT_CHAT_TIMEOUT, DEFAULT_EMBEDDING_TIMEOUT, DEFAULT_SPEECH_TIMEOUT};
use crate::error::{http_error, redact};
use crate::http::{encode_component, join_url, Http};
use crate::registry::Modality;
use crate::sse::{text_stream, Frame, Framing, Step};

/// The API version used when an Azure credential does not carry one. The
/// credential form has no field for it, so without a default every
/// UI-created Azure credential would be unusable.
pub(crate) const DEFAULT_AZURE_API_VERSION: &str = "2024-10-21";
pub(crate) const OMLX_DEFAULT_BASE_URL: &str = "http://localhost:11435/v1";

/// The provider's API root, when it has a fixed one.
pub(crate) fn default_base_url(provider: &str) -> Option<&'static str> {
    Some(match provider {
        "openai" => "https://api.openai.com/v1",
        "groq" => "https://api.groq.com/openai/v1",
        "mistral" => "https://api.mistral.ai/v1",
        "deepseek" => "https://api.deepseek.com/v1",
        "xai" => "https://api.x.ai/v1",
        "openrouter" => "https://openrouter.ai/api/v1",
        "dashscope" => "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "minimax" => "https://api.minimax.io/v1",
        "novita" => "https://api.novita.ai/openai",
        "siliconflow" => "https://api.siliconflow.com/v1",
        "zai" => "https://api.z.ai/api/paas/v4",
        "ppq" => "https://api.ppq.ai/v1",
        "omlx" => OMLX_DEFAULT_BASE_URL,
        _ => return None,
    })
}

/// Local servers usually run without authentication.
fn key_is_optional(provider: &str) -> bool {
    matches!(provider, "omlx" | "openai_compatible")
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum Route {
    /// `{base}/{path}` with a bearer token.
    Standard { base: String },
    /// Azure deployment URLs with an `api-key` header.
    Azure { endpoint: String, api_version: String },
}

/// Where one model of an OpenAI-style API lives and how to authenticate.
#[derive(Clone)]
pub(crate) struct Endpoint {
    pub http: Http,
    pub provider: String,
    pub model: String,
    route: Route,
    api_key: Option<String>,
}

impl Endpoint {
    pub fn resolve(
        client: &reqwest::Client,
        provider: &str,
        model: &str,
        config: &ProviderConfig,
        modality: Modality,
    ) -> AiResult<Self> {
        let default_timeout = match modality {
            Modality::Language => DEFAULT_CHAT_TIMEOUT,
            Modality::Embedding => DEFAULT_EMBEDDING_TIMEOUT,
            Modality::SpeechToText | Modality::TextToSpeech => DEFAULT_SPEECH_TIMEOUT,
        };
        let http = Http::new(client, provider, config, default_timeout);
        let api_key = config.key().map(str::to_string);

        let route = if provider == "azure" {
            let specific = match modality {
                Modality::Language => &config.endpoint_llm,
                Modality::Embedding => &config.endpoint_embedding,
                Modality::SpeechToText => &config.endpoint_stt,
                Modality::TextToSpeech => &config.endpoint_tts,
            };
            // The credential form stores the resource endpoint as "base URL".
            let endpoint = non_blank(specific)
                .or_else(|| non_blank(&config.endpoint))
                .or_else(|| non_blank(&config.base_url))
                .ok_or_else(|| AiError::Configuration("No Azure endpoint configured".to_string()))?;
            if api_key.is_none() {
                return Err(AiError::Configuration("No Azure API key configured".to_string()));
            }
            let endpoint = endpoint.trim_end_matches('/');
            let endpoint = endpoint.strip_suffix("/openai").unwrap_or(endpoint);
            Route::Azure {
                endpoint: endpoint.to_string(),
                api_version: non_blank(&config.api_version).unwrap_or(DEFAULT_AZURE_API_VERSION).to_string(),
            }
        } else {
            let base = config
                .url()
                .or_else(|| default_base_url(provider))
                .ok_or_else(|| AiError::Configuration(format!("No base URL configured for {provider}")))?;
            if api_key.is_none() && !key_is_optional(provider) {
                return Err(AiError::Configuration(format!("No API key configured for {provider}")));
            }
            Route::Standard { base: base.to_string() }
        };

        Ok(Self { http, provider: provider.to_string(), model: model.to_string(), route, api_key })
    }

    pub fn url(&self, path: &str) -> String {
        match &self.route {
            Route::Standard { base } => join_url(base, path),
            Route::Azure { endpoint, api_version } => format!(
                "{endpoint}/openai/deployments/{}/{}?api-version={}",
                encode_component(&self.model),
                path.trim_start_matches('/'),
                encode_component(api_version)
            ),
        }
    }

    fn authorize(&self, request: RequestBuilder) -> RequestBuilder {
        match (&self.route, &self.api_key) {
            (_, None) => request,
            (Route::Azure { .. }, Some(key)) => request.header("api-key", key),
            (Route::Standard { .. }, Some(key)) => request.bearer_auth(key),
        }
    }

    pub fn post(&self, path: &str) -> RequestBuilder {
        self.authorize(self.http.client.post(self.url(path)))
    }

    pub fn get(&self, path: &str) -> RequestBuilder {
        self.authorize(self.http.client.get(self.url(path)))
    }

    fn is_azure(&self) -> bool {
        matches!(self.route, Route::Azure { .. })
    }
}

/// An `{"error": ...}` object delivered with a success status or inside a
/// stream (OpenRouter and several gateways do this).
fn embedded_error(provider: &str, json: &Value, secret: Option<&str>) -> Option<AiError> {
    let error = json.get("error").filter(|e| !e.is_null())?;
    let status = error
        .get("code")
        .and_then(|c| c.as_u64().or_else(|| c.as_str()?.parse().ok()))
        .filter(|c| (400..600).contains(c));
    let body = json.to_string();
    Some(match status {
        Some(status) => http_error(provider, status as u16, &body, secret),
        None => {
            let detail = crate::error::parse_error_body(&body).message;
            let detail = if detail.is_empty() { "the provider reported an error".to_string() } else { detail };
            AiError::Provider { status: None, message: format!("{provider}: {}", redact(&detail, secret)) }
        }
    })
}

// ---------------------------------------------------------------------------
// Chat
// ---------------------------------------------------------------------------

/// OpenAI models that take `max_completion_tokens` and fixed sampling.
pub(crate) fn is_reasoning_model(model: &str) -> bool {
    let name = model.to_ascii_lowercase();
    ["o1", "o3", "o4", "gpt-5"].iter().any(|prefix| name.starts_with(prefix))
}

/// Request adjustments, either known up front (reasoning models) or learned
/// from a provider rejecting a parameter.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct Tweaks {
    /// Send `max_completion_tokens` instead of `max_tokens`.
    pub completion_tokens: bool,
    /// Leave out `temperature` and `top_p`.
    pub no_sampling: bool,
    /// Leave out `response_format` even when JSON was requested.
    pub no_response_format: bool,
}

impl Tweaks {
    fn initial(provider: &str, model: &str) -> Self {
        let reasoning = matches!(provider, "openai" | "azure") && is_reasoning_model(model);
        Self { completion_tokens: reasoning, no_sampling: reasoning, no_response_format: false }
    }

    /// Adjusts for the parameter `err` complains about. Returns whether
    /// anything changed, i.e. whether a retry can help.
    fn adapt(&mut self, err: &AiError, options: &ChatOptions) -> bool {
        if !matches!(err, AiError::InvalidRequest(_) | AiError::ContextLength(_) | AiError::Provider { .. }) {
            return false;
        }
        let text = err.to_string().to_lowercase();
        let rejected = ["unsupported", "not supported", "does not support", "unknown", "unrecognized", "invalid", "not allowed", "extra inputs", "must be"]
            .iter()
            .any(|w| text.contains(w));
        if !rejected {
            return false;
        }
        if !self.completion_tokens && text.contains("max_completion_tokens") {
            self.completion_tokens = true;
            return true;
        }
        if !self.no_sampling
            && (options.temperature.is_some() || options.top_p.is_some())
            && (text.contains("temperature") || text.contains("top_p"))
        {
            self.no_sampling = true;
            return true;
        }
        if !self.no_response_format
            && options.json
            && (text.contains("response_format") || text.contains("json_object"))
        {
            self.no_response_format = true;
            return true;
        }
        false
    }
}

pub(crate) fn chat_body(model: &str, messages: &[ChatMessage], options: &ChatOptions, stream: bool, tweaks: Tweaks) -> Value {
    let mut body = Map::new();
    body.insert("model".into(), json!(model));
    body.insert(
        "messages".into(),
        messages.iter().map(|m| json!({"role": m.role.as_str(), "content": m.content})).collect(),
    );
    if !tweaks.no_sampling {
        if let Some(temperature) = options.temperature {
            body.insert("temperature".into(), float(temperature));
        }
        if let Some(top_p) = options.top_p {
            body.insert("top_p".into(), float(top_p));
        }
    }
    if let Some(max_tokens) = options.max_tokens {
        let key = if tweaks.completion_tokens { "max_completion_tokens" } else { "max_tokens" };
        body.insert(key.into(), json!(max_tokens));
    }
    if options.json && !tweaks.no_response_format {
        body.insert("response_format".into(), json!({"type": "json_object"}));
    }
    if stream {
        body.insert("stream".into(), json!(true));
    }
    Value::Object(body)
}

pub(crate) struct OpenAiChat {
    endpoint: Endpoint,
}

impl OpenAiChat {
    pub fn new(client: &reqwest::Client, provider: &str, model: &str, config: &ProviderConfig) -> AiResult<Self> {
        Ok(Self { endpoint: Endpoint::resolve(client, provider, model, config, Modality::Language)? })
    }

    /// Sends the chat request, retrying with an adapted body when the
    /// provider rejects a parameter.
    async fn request(&self, messages: &[ChatMessage], options: &ChatOptions, stream: bool) -> AiResult<reqwest::Response> {
        let ep = &self.endpoint;
        let mut tweaks = Tweaks::initial(&ep.provider, &ep.model);
        let mut attempts_left = 3;
        loop {
            let body = chat_body(&ep.model, messages, options, stream, tweaks);
            let request = ep.post("chat/completions").json(&body);
            let result = if stream {
                ep.http.send_streaming(request.header("accept", "text/event-stream")).await
            } else {
                ep.http.send(request).await
            };
            attempts_left -= 1;
            match result {
                Ok(response) => return Ok(response),
                Err(err) if attempts_left > 0 && tweaks.adapt(&err, options) => {
                    tracing::debug!(provider = %ep.provider, model = %ep.model, ?tweaks, "retrying chat request with adapted parameters");
                }
                Err(err) => return Err(err),
            }
        }
    }
}

#[async_trait]
impl LanguageModel for OpenAiChat {
    fn provider(&self) -> &str {
        &self.endpoint.provider
    }

    fn model_name(&self) -> &str {
        &self.endpoint.model
    }

    async fn chat(&self, messages: &[ChatMessage], options: &ChatOptions) -> AiResult<ChatResponse> {
        let ep = &self.endpoint;
        let response = self.request(messages, options, false).await?;
        let body = response.bytes().await.map_err(|e| crate::error::transport_error(&ep.provider, e))?;
        let json: Value =
            serde_json::from_slice(&body).map_err(|_| ep.http.malformed("the response was not valid JSON"))?;
        let Some(choice) = json.pointer("/choices/0") else {
            return Err(embedded_error(&ep.provider, &json, ep.api_key.as_deref())
                .unwrap_or_else(|| ep.http.malformed("no choices in the chat completion")));
        };
        Ok(ChatResponse {
            content: choice.pointer("/message/content").map(content_text).unwrap_or_default(),
            model: json.get("model").and_then(Value::as_str).unwrap_or(&ep.model).to_string(),
            finish_reason: finish_reason(choice.get("finish_reason")),
            usage: usage(json.pointer("/usage/prompt_tokens"), json.pointer("/usage/completion_tokens")),
        })
    }

    async fn chat_stream(&self, messages: &[ChatMessage], options: &ChatOptions) -> AiResult<TextStream> {
        let ep = &self.endpoint;
        let response = self.request(messages, options, true).await?;
        let provider = ep.provider.clone();
        let secret = ep.api_key.clone();
        Ok(text_stream(response, Framing::Sse, ep.http.timeout, &ep.provider, move |frame: Frame| {
            let Ok(json) = serde_json::from_str::<Value>(&frame.data) else {
                return Step::Skip;
            };
            if let Some(err) = embedded_error(&provider, &json, secret.as_deref()) {
                return Step::Fail(err);
            }
            match json.pointer("/choices/0/delta/content") {
                Some(content) => Step::Text(content_text(content)),
                None => Step::Skip,
            }
        }))
    }
}

// ---------------------------------------------------------------------------
// Embeddings
// ---------------------------------------------------------------------------

pub(crate) struct OpenAiEmbedder {
    endpoint: Endpoint,
    dimensions: Option<u64>,
}

impl OpenAiEmbedder {
    pub fn new(client: &reqwest::Client, provider: &str, model: &str, config: &ProviderConfig) -> AiResult<Self> {
        Ok(Self {
            endpoint: Endpoint::resolve(client, provider, model, config, Modality::Embedding)?,
            dimensions: config.extra_u64("dimensions"),
        })
    }
}

/// Reads `data[*].embedding`, ordered by each item's `index`.
fn parse_embeddings(json: &Value) -> Option<Vec<Vec<f32>>> {
    let data = json.get("data")?.as_array()?;
    let mut items: Vec<(u64, Vec<f32>)> = Vec::with_capacity(data.len());
    for (position, item) in data.iter().enumerate() {
        let index = item.get("index").and_then(Value::as_u64).unwrap_or(position as u64);
        items.push((index, vector(item.get("embedding")?)?));
    }
    items.sort_by_key(|(index, _)| *index);
    Some(items.into_iter().map(|(_, v)| v).collect())
}

#[async_trait]
impl EmbedBatch for OpenAiEmbedder {
    fn provider(&self) -> &str {
        &self.endpoint.provider
    }

    fn model_name(&self) -> &str {
        &self.endpoint.model
    }

    fn batch_limit(&self) -> usize {
        // OpenAI accepts 2048 inputs but also caps the tokens per request;
        // local servers and Mistral have much smaller limits.
        match self.endpoint.provider.as_str() {
            "mistral" | "omlx" | "openai_compatible" => 32,
            _ => 100,
        }
    }

    async fn embed_batch(&self, texts: &[String]) -> AiResult<Vec<Vec<f32>>> {
        let ep = &self.endpoint;
        let mut body = json!({"model": ep.model, "input": texts});
        if let Some(dimensions) = self.dimensions {
            body["dimensions"] = json!(dimensions);
        }
        let json = ep.http.json(ep.post("embeddings").json(&body)).await?;
        match parse_embeddings(&json) {
            Some(vectors) => Ok(vectors),
            None => Err(embedded_error(&ep.provider, &json, ep.api_key.as_deref())
                .unwrap_or_else(|| ep.http.malformed("no embeddings in the response"))),
        }
    }
}

// ---------------------------------------------------------------------------
// Text to speech
// ---------------------------------------------------------------------------

const OPENAI_VOICES: &[&str] =
    &["alloy", "ash", "ballad", "coral", "echo", "fable", "nova", "onyx", "sage", "shimmer", "verse"];
const XAI_VOICES: &[&str] = &["eve", "ara", "rex", "sal", "leo"];
/// Voices of OpenRouter's default speech model (Microsoft neural voice names).
const MAI_VOICES: &[&str] = &["en-US-AvaNeural", "en-US-AndrewNeural", "en-US-EmmaNeural", "en-US-BrianNeural"];

fn static_voices(ids: &[&str]) -> Vec<Voice> {
    ids.iter().map(|id| Voice { id: id.to_string(), name: id.to_string(), language: None, gender: None }).collect()
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SpeechDialect {
    /// `POST /audio/speech {model, input, voice}` returning audio bytes.
    OpenAi,
    /// Mistral Voxtral: `voice_id`, audio returned as base64 JSON.
    Mistral,
    /// xAI: `POST /tts {text, voice_id, language}` returning audio bytes.
    Xai,
}

pub(crate) struct OpenAiSpeech {
    endpoint: Endpoint,
    dialect: SpeechDialect,
    language: String,
}

impl OpenAiSpeech {
    pub fn new(client: &reqwest::Client, provider: &str, model: &str, config: &ProviderConfig) -> AiResult<Self> {
        let dialect = match provider {
            "mistral" => SpeechDialect::Mistral,
            "xai" => SpeechDialect::Xai,
            _ => SpeechDialect::OpenAi,
        };
        Ok(Self {
            endpoint: Endpoint::resolve(client, provider, model, config, Modality::TextToSpeech)?,
            dialect,
            language: config.extra_str("language").unwrap_or("auto").to_string(),
        })
    }

    fn request(&self, text: &str, voice: &str) -> RequestBuilder {
        let ep = &self.endpoint;
        let voice = voice.trim();
        match self.dialect {
            SpeechDialect::OpenAi => {
                let voice = if voice.is_empty() { "alloy" } else { voice };
                ep.post("audio/speech").json(&json!({
                    "model": ep.model,
                    "input": text,
                    "voice": voice,
                    "response_format": "mp3",
                }))
            }
            SpeechDialect::Mistral => {
                let mut body = json!({"model": ep.model, "input": text, "response_format": "mp3"});
                if !voice.is_empty() {
                    body["voice_id"] = json!(voice);
                }
                ep.post("audio/speech").json(&body)
            }
            SpeechDialect::Xai => {
                let voice = if voice.is_empty() { "eve" } else { voice };
                ep.post("tts").json(&json!({"text": text, "voice_id": voice, "language": self.language}))
            }
        }
    }
}

/// Decodes speech delivered as base64 inside JSON (`audio_data`, `audio`,
/// `data`), which is how Mistral returns it.
fn audio_from_json(body: &[u8]) -> Option<Vec<u8>> {
    let json: Value = serde_json::from_slice(body).ok()?;
    let encoded = ["audio_data", "audio", "data"].iter().find_map(|k| json.get(*k).and_then(Value::as_str))?;
    base64::engine::general_purpose::STANDARD.decode(encoded.trim()).ok()
}

#[async_trait]
impl TextToSpeechModel for OpenAiSpeech {
    fn provider(&self) -> &str {
        &self.endpoint.provider
    }

    fn model_name(&self) -> &str {
        &self.endpoint.model
    }

    async fn synthesize(&self, text: &str, voice: &str) -> AiResult<AudioData> {
        let http = &self.endpoint.http;
        let (bytes, content_type) = http.bytes(self.request(text, voice)).await?;
        if content_type.as_deref().is_some_and(|ct| ct.contains("json")) {
            let audio = audio_from_json(&bytes).ok_or_else(|| http.malformed("no audio in the response"))?;
            return Ok(AudioData { bytes: audio.into(), content_type: "audio/mpeg".to_string() });
        }
        if bytes.is_empty() {
            return Err(http.malformed("the audio response was empty"));
        }
        let content_type = content_type.filter(|ct| ct.starts_with("audio/")).unwrap_or_else(|| "audio/mpeg".to_string());
        Ok(AudioData { bytes, content_type })
    }

    async fn voices(&self) -> AiResult<Vec<Voice>> {
        let ep = &self.endpoint;
        match (self.dialect, ep.provider.as_str()) {
            (SpeechDialect::Xai, _) => Ok(static_voices(XAI_VOICES)),
            (SpeechDialect::Mistral, _) => {
                let json = ep.http.json(ep.get("audio/voices")).await?;
                Ok(parse_voice_list(&json))
            }
            (SpeechDialect::OpenAi, "openai" | "azure") => Ok(static_voices(OPENAI_VOICES)),
            (SpeechDialect::OpenAi, "openrouter") => {
                let model = ep.model.to_ascii_lowercase();
                if model.contains("mai-voice") {
                    Ok(static_voices(MAI_VOICES))
                } else if model.starts_with("openai/") {
                    Ok(static_voices(OPENAI_VOICES))
                } else {
                    Ok(Vec::new())
                }
            }
            _ => Ok(Vec::new()),
        }
    }
}

/// Reads a voice listing in any of the usual envelopes (`items`, `data`,
/// `voices` or a bare array) with `id`/`voice_id` and `name` fields.
pub(crate) fn parse_voice_list(json: &Value) -> Vec<Voice> {
    let items = ["items", "data", "voices"]
        .iter()
        .find_map(|k| json.get(*k).and_then(Value::as_array))
        .or_else(|| json.as_array());
    let text = |item: &Value, keys: &[&str]| {
        keys.iter().find_map(|k| item.get(*k).and_then(Value::as_str)).map(str::to_string)
    };
    items
        .into_iter()
        .flatten()
        .filter_map(|item| {
            let id = text(item, &["id", "voice_id", "slug"])?;
            Some(Voice {
                name: text(item, &["name", "voice_name"]).unwrap_or_else(|| id.clone()),
                language: text(item, &["language", "locale"])
                    .or_else(|| item.pointer("/languages/0").and_then(Value::as_str).map(str::to_string)),
                gender: text(item, &["gender"]),
                id,
            })
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Speech to text
// ---------------------------------------------------------------------------

pub(crate) struct OpenAiTranscriber {
    endpoint: Endpoint,
}

impl OpenAiTranscriber {
    pub fn new(client: &reqwest::Client, provider: &str, model: &str, config: &ProviderConfig) -> AiResult<Self> {
        Ok(Self { endpoint: Endpoint::resolve(client, provider, model, config, Modality::SpeechToText)? })
    }

    fn multipart(&self, request: &TranscriptionRequest) -> Form {
        let part = |mime: &str| {
            let length = request.audio.len() as u64;
            Part::stream_with_length(request.audio.clone(), length).file_name(request.filename.clone()).mime_str(mime)
        };
        let guessed = audio_mime(&request.filename);
        let file = request
            .content_type
            .as_deref()
            .and_then(|mime| part(mime).ok())
            .or_else(|| part(guessed).ok())
            .unwrap_or_else(|| Part::stream(request.audio.clone()).file_name(request.filename.clone()));

        let mut form = Form::new().part("file", file).text("response_format", "json");
        // Azure names the model in the URL; everyone else wants it in the form.
        if !self.endpoint.is_azure() {
            form = form.text("model", self.endpoint.model.clone());
        }
        if let Some(language) = non_blank(&request.language) {
            form = form.text("language", language.to_string());
        }
        if let Some(prompt) = non_blank(&request.prompt) {
            form = form.text("prompt", prompt.to_string());
        }
        form
    }

    /// OpenRouter takes the audio as base64 inside a JSON body.
    fn openrouter_body(&self, request: &TranscriptionRequest) -> Value {
        let mut body = json!({
            "model": self.endpoint.model,
            "input_audio": {
                "data": base64::engine::general_purpose::STANDARD.encode(&request.audio),
                "format": audio_format(&request.filename),
            },
        });
        if let Some(language) = non_blank(&request.language) {
            body["language"] = json!(language);
        }
        if let Some(prompt) = non_blank(&request.prompt) {
            body["prompt"] = json!(prompt);
        }
        body
    }
}

#[async_trait]
impl SpeechToTextModel for OpenAiTranscriber {
    fn provider(&self) -> &str {
        &self.endpoint.provider
    }

    fn model_name(&self) -> &str {
        &self.endpoint.model
    }

    async fn transcribe(&self, request: TranscriptionRequest) -> AiResult<String> {
        let ep = &self.endpoint;
        let builder = ep.post("audio/transcriptions");
        let builder = if ep.provider == "openrouter" {
            builder.json(&self.openrouter_body(&request))
        } else {
            builder.multipart(self.multipart(&request))
        };
        let (bytes, _) = ep.http.bytes(builder).await?;
        match serde_json::from_slice::<Value>(&bytes) {
            Ok(Value::String(text)) => Ok(text.trim().to_string()),
            Ok(json) => match json.get("text").and_then(Value::as_str) {
                Some(text) => Ok(text.trim().to_string()),
                None => Err(embedded_error(&ep.provider, &json, ep.api_key.as_deref())
                    .unwrap_or_else(|| ep.http.malformed("no text in the transcription"))),
            },
            // Some servers answer with the bare transcript.
            Err(_) => Ok(String::from_utf8_lossy(&bytes).trim().to_string()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::http::shared_client;
    use crate::providers::testing::{collect_text, collect_until_error, header, request, request_json};
    use crate::providers::Batched;
    use bytes::Bytes;
    use nn_llm::EmbeddingModel;
    use wiremock::matchers::{body_string_contains, header as header_is, method, path, query_param};
    use wiremock::{Mock, MockServer, Request, ResponseTemplate};

    fn config(server: &MockServer) -> ProviderConfig {
        ProviderConfig::new().with_api_key("sk-test").with_base_url(format!("{}/v1", server.uri()))
    }

    fn chat(provider: &str, model: &str, config: &ProviderConfig) -> OpenAiChat {
        OpenAiChat::new(&shared_client(), provider, model, config).unwrap()
    }

    fn completion(content: &str) -> Value {
        json!({
            "id": "chatcmpl-1",
            "model": "gpt-4o-mini-2024-07-18",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 12, "completion_tokens": 3, "total_tokens": 15},
        })
    }

    fn messages() -> Vec<ChatMessage> {
        vec![ChatMessage::system("Be brief."), ChatMessage::user("Hi"), ChatMessage::assistant("Hello"), ChatMessage::user("Bye")]
    }

    #[test]
    fn default_base_urls_cover_the_openai_family() {
        for provider in [
            "openai", "groq", "mistral", "deepseek", "xai", "openrouter", "dashscope", "minimax", "novita",
            "siliconflow", "zai", "ppq", "omlx",
        ] {
            let base = default_base_url(provider).unwrap_or_else(|| panic!("{provider} has no base URL"));
            assert!(base.starts_with("http"), "{provider}");
            assert!(!base.ends_with('/'), "{provider}");
        }
        assert_eq!(default_base_url("openai_compatible"), None);
        assert_eq!(default_base_url("azure"), None);
        // Registry discovery URLs must sit under the same roots.
        for spec in crate::registry::PROVIDERS {
            if let (Some(url), Some(base)) = (spec.openai_compat_discovery_url, default_base_url(spec.name)) {
                let root = base.trim_end_matches("/v1");
                assert!(url.starts_with(root), "{}: {url} is not under {root}", spec.name);
            }
        }
    }

    #[tokio::test]
    async fn chat_sends_the_openai_shape_and_parses_the_reply() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .and(header_is("authorization", "Bearer sk-test"))
            .respond_with(ResponseTemplate::new(200).set_body_json(completion("See you.")))
            .mount(&server)
            .await;

        let model = chat("openai", "gpt-4o-mini", &config(&server));
        let options = ChatOptions { temperature: Some(0.5), max_tokens: Some(64), top_p: Some(0.9), json: false };
        let reply = model.chat(&messages(), &options).await.unwrap();

        assert_eq!(reply.content, "See you.");
        assert_eq!(reply.model, "gpt-4o-mini-2024-07-18");
        assert_eq!(reply.finish_reason.as_deref(), Some("stop"));
        assert_eq!(reply.usage, Some(nn_llm::Usage { prompt_tokens: 12, completion_tokens: 3 }));
        assert_eq!(model.provider(), "openai");
        assert_eq!(model.model_name(), "gpt-4o-mini");

        assert_eq!(
            request_json(&server, 0).await,
            json!({
                "model": "gpt-4o-mini",
                "messages": [
                    {"role": "system", "content": "Be brief."},
                    {"role": "user", "content": "Hi"},
                    {"role": "assistant", "content": "Hello"},
                    {"role": "user", "content": "Bye"},
                ],
                "temperature": 0.5,
                "top_p": 0.9,
                "max_tokens": 64,
            })
        );
    }

    #[test]
    fn default_options_send_only_model_and_messages() {
        let body = chat_body("m", &[ChatMessage::user("x")], &ChatOptions::default(), false, Tweaks::default());
        assert_eq!(body, json!({"model": "m", "messages": [{"role": "user", "content": "x"}]}));
        let body = chat_body("m", &[ChatMessage::user("x")], &ChatOptions::json(), true, Tweaks::default());
        assert_eq!(body["response_format"], json!({"type": "json_object"}));
        assert_eq!(body["stream"], json!(true));
    }

    #[test]
    fn reasoning_models_are_detected_by_name() {
        for name in ["o1", "o3-mini", "o4-mini", "gpt-5", "GPT-5-mini", "gpt-5.2-pro"] {
            assert!(is_reasoning_model(name), "{name}");
        }
        for name in ["gpt-4o", "gpt-4.1-mini", "gpt-3.5-turbo", "chatgpt-4o-latest", "omni-moderation"] {
            assert!(!is_reasoning_model(name), "{name}");
        }
        assert_eq!(Tweaks::initial("openai", "o3"), Tweaks { completion_tokens: true, no_sampling: true, no_response_format: false });
        assert!(Tweaks::initial("azure", "gpt-5-mini").completion_tokens);
        // Other providers name their own models; no OpenAI rules apply.
        assert_eq!(Tweaks::initial("groq", "o3"), Tweaks::default());
    }

    #[tokio::test]
    async fn reasoning_models_get_max_completion_tokens_and_no_sampling() {
        let server = MockServer::start().await;
        Mock::given(method("POST")).respond_with(ResponseTemplate::new(200).set_body_json(completion("ok"))).mount(&server).await;
        let model = chat("openai", "gpt-5-mini", &config(&server));
        let options = ChatOptions { temperature: Some(0.2), max_tokens: Some(500), top_p: Some(0.5), json: true };
        model.chat(&[ChatMessage::user("x")], &options).await.unwrap();
        assert_eq!(
            request_json(&server, 0).await,
            json!({
                "model": "gpt-5-mini",
                "messages": [{"role": "user", "content": "x"}],
                "max_completion_tokens": 500,
                "response_format": {"type": "json_object"},
            })
        );
    }

    #[tokio::test]
    async fn a_rejected_parameter_is_dropped_and_the_request_retried() {
        let server = MockServer::start().await;
        // Local servers often accept only json_schema.
        Mock::given(method("POST"))
            .and(body_string_contains("response_format"))
            .respond_with(ResponseTemplate::new(400).set_body_json(
                json!({"error": "'response_format.type' must be 'json_schema' or 'text'"}),
            ))
            .mount(&server)
            .await;
        Mock::given(method("POST")).respond_with(ResponseTemplate::new(200).set_body_json(completion("{}"))).mount(&server).await;

        let model = chat("openai_compatible", "local", &config(&server));
        let reply = model.chat(&[ChatMessage::user("x")], &ChatOptions::json()).await.unwrap();
        assert_eq!(reply.content, "{}");
        assert_eq!(server.received_requests().await.unwrap().len(), 2);
        assert!(request_json(&server, 1).await.get("response_format").is_none());
    }

    #[tokio::test]
    async fn unsupported_max_tokens_switches_to_max_completion_tokens() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(body_string_contains("\"max_tokens\""))
            .respond_with(ResponseTemplate::new(400).set_body_json(json!({"error": {
                "message": "Unsupported parameter: 'max_tokens' is not supported with this model. Use 'max_completion_tokens' instead.",
                "type": "invalid_request_error", "param": "max_tokens", "code": "unsupported_parameter"
            }})))
            .mount(&server)
            .await;
        Mock::given(method("POST")).respond_with(ResponseTemplate::new(200).set_body_json(completion("ok"))).mount(&server).await;

        let model = chat("openai_compatible", "some-new-model", &config(&server));
        model.chat(&[ChatMessage::user("x")], &ChatOptions::default().with_max_tokens(10)).await.unwrap();
        let second = request_json(&server, 1).await;
        assert_eq!(second["max_completion_tokens"], json!(10));
        assert!(second.get("max_tokens").is_none());
    }

    #[tokio::test]
    async fn errors_that_adapting_cannot_fix_are_returned_once() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .respond_with(ResponseTemplate::new(400).set_body_json(json!({"error": {"message": "messages must not be empty"}})))
            .mount(&server)
            .await;
        let model = chat("openai", "gpt-4o", &config(&server));
        let err = model.chat(&[], &ChatOptions::json()).await.unwrap_err();
        assert!(matches!(err, AiError::InvalidRequest(_)));
        assert_eq!(server.received_requests().await.unwrap().len(), 1);
    }

    #[tokio::test]
    async fn chat_stream_yields_text_deltas_only() {
        let server = MockServer::start().await;
        let sse = concat!(
            ": OPENROUTER PROCESSING\n\n",
            "data: {\"choices\":[{\"delta\":{\"role\":\"assistant\",\"content\":\"\"}}]}\n\n",
            "data: {\"choices\":[{\"delta\":{\"content\":\"Hel\"}}]}\n\n",
            "data: {\"choices\":[{\"delta\":{\"content\":\"lo\"}}]}\n\n",
            "data: {\"choices\":[{\"delta\":{\"reasoning_content\":\"thinking\"}}]}\n\n",
            "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n",
            "data: {\"choices\":[],\"usage\":{\"prompt_tokens\":1,\"completion_tokens\":2}}\n\n",
            "data: [DONE]\n\n",
        );
        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .respond_with(ResponseTemplate::new(200).insert_header("content-type", "text/event-stream").set_body_string(sse))
            .mount(&server)
            .await;

        let model = chat("groq", "llama-3.3-70b", &config(&server));
        let stream = model.chat_stream(&[ChatMessage::user("Hi")], &ChatOptions::default()).await.unwrap();
        assert_eq!(collect_text(stream).await, vec!["Hel", "lo"]);
        assert_eq!(request_json(&server, 0).await["stream"], json!(true));
    }

    #[tokio::test]
    async fn errors_before_and_during_a_stream_are_reported() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(body_string_contains("missing-model"))
            .respond_with(ResponseTemplate::new(404).set_body_json(
                json!({"error": {"message": "The model `missing-model` does not exist", "code": "model_not_found"}}),
            ))
            .mount(&server)
            .await;
        let sse = "data: {\"choices\":[{\"delta\":{\"content\":\"par\"}}]}\n\ndata: {\"error\":{\"code\":429,\"message\":\"Rate limited upstream\"}}\n\n";
        Mock::given(method("POST")).respond_with(ResponseTemplate::new(200).set_body_string(sse)).mount(&server).await;

        let missing = chat("openrouter", "missing-model", &config(&server));
        let err = missing.chat_stream(&[ChatMessage::user("x")], &ChatOptions::default()).await.err().unwrap();
        assert!(matches!(err, AiError::ModelNotFound(_)), "{err:?}");

        let model = chat("openrouter", "vendor/model", &config(&server));
        let stream = model.chat_stream(&[ChatMessage::user("x")], &ChatOptions::default()).await.unwrap();
        let (text, err) = collect_until_error(stream).await;
        assert_eq!(text, vec!["par"]);
        assert!(matches!(err, Some(AiError::RateLimit(_))), "{err:?}");
    }

    #[tokio::test]
    async fn an_error_object_with_status_200_is_an_error() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({"error": {"code": 401, "message": "No auth credentials found"}})))
            .mount(&server)
            .await;
        let model = chat("openrouter", "vendor/model", &config(&server));
        let err = model.chat(&[ChatMessage::user("x")], &ChatOptions::default()).await.unwrap_err();
        assert!(matches!(err, AiError::Authentication(_)), "{err:?}");
    }

    #[tokio::test]
    async fn http_failures_map_to_typed_errors_without_the_key() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .respond_with(ResponseTemplate::new(401).set_body_json(json!({"error": {"message": "Incorrect API key provided: sk-test", "code": "invalid_api_key"}})))
            .mount(&server)
            .await;
        let model = chat("openai", "gpt-4o", &config(&server));
        let err = model.chat(&[ChatMessage::user("x")], &ChatOptions::default()).await.unwrap_err();
        assert!(matches!(err, AiError::Authentication(_)));
        assert!(!err.to_string().contains("sk-test"));
    }

    #[test]
    fn missing_settings_are_configuration_errors() {
        let client = shared_client();
        let no_key = OpenAiChat::new(&client, "openai", "gpt-4o", &ProviderConfig::default()).err().unwrap();
        assert!(matches!(&no_key, AiError::Configuration(m) if m == "No API key configured for openai"));
        let no_url = OpenAiChat::new(&client, "openai_compatible", "m", &ProviderConfig::default()).err().unwrap();
        assert!(matches!(&no_url, AiError::Configuration(m) if m == "No base URL configured for openai_compatible"));
        // Local servers need no key.
        assert!(OpenAiChat::new(&client, "omlx", "m", &ProviderConfig::default()).is_ok());
        let local = ProviderConfig::new().with_base_url("http://localhost:1234/v1/");
        let model = OpenAiChat::new(&client, "openai_compatible", "m", &local).unwrap();
        assert_eq!(model.endpoint.url("chat/completions"), "http://localhost:1234/v1/chat/completions");

        let azure = OpenAiChat::new(&client, "azure", "dep", &ProviderConfig::new().with_api_key("k")).err().unwrap();
        assert!(matches!(&azure, AiError::Configuration(m) if m == "No Azure endpoint configured"));
        let azure = ProviderConfig { endpoint: Some("https://r.openai.azure.com".into()), ..Default::default() };
        let azure = OpenAiChat::new(&client, "azure", "dep", &azure).err().unwrap();
        assert!(matches!(&azure, AiError::Configuration(m) if m == "No Azure API key configured"));
    }

    #[test]
    fn azure_urls_use_deployments_api_version_and_modality_endpoints() {
        let client = shared_client();
        let config = ProviderConfig {
            api_key: Some("azure-key".into()),
            endpoint: Some("https://main.openai.azure.com/".into()),
            api_version: Some("2025-01-01-preview".into()),
            endpoint_embedding: Some("https://embed.openai.azure.com".into()),
            endpoint_tts: Some("https://tts.openai.azure.com/openai/".into()),
            ..Default::default()
        };
        let url = |modality, path: &str| Endpoint::resolve(&client, "azure", "my dep", &config, modality).unwrap().url(path);
        assert_eq!(
            url(Modality::Language, "chat/completions"),
            "https://main.openai.azure.com/openai/deployments/my%20dep/chat/completions?api-version=2025-01-01-preview"
        );
        assert_eq!(
            url(Modality::Embedding, "embeddings"),
            "https://embed.openai.azure.com/openai/deployments/my%20dep/embeddings?api-version=2025-01-01-preview"
        );
        assert_eq!(
            url(Modality::TextToSpeech, "audio/speech"),
            "https://tts.openai.azure.com/openai/deployments/my%20dep/audio/speech?api-version=2025-01-01-preview"
        );
        // No STT override: falls back to the main endpoint.
        assert!(url(Modality::SpeechToText, "audio/transcriptions").starts_with("https://main.openai.azure.com/openai/deployments/"));

        // The form's "base URL" is the endpoint, and the API version has a default.
        let form = ProviderConfig::new().with_api_key("k").with_base_url("https://form.openai.azure.com");
        let endpoint = Endpoint::resolve(&client, "azure", "gpt-4o", &form, Modality::Language).unwrap();
        assert_eq!(
            endpoint.url("chat/completions"),
            "https://form.openai.azure.com/openai/deployments/gpt-4o/chat/completions?api-version=2024-10-21"
        );
    }

    #[tokio::test]
    async fn azure_requests_carry_the_api_key_header() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/openai/deployments/gpt-4o-prod/chat/completions"))
            .and(query_param("api-version", "2024-10-21"))
            .and(header_is("api-key", "azure-key"))
            .respond_with(ResponseTemplate::new(200).set_body_json(completion("hi")))
            .mount(&server)
            .await;
        let config = ProviderConfig { api_key: Some("azure-key".into()), endpoint: Some(server.uri()), ..Default::default() };
        let model = chat("azure", "gpt-4o-prod", &config);
        assert_eq!(model.chat(&[ChatMessage::user("x")], &ChatOptions::default()).await.unwrap().content, "hi");
        let sent = request(&server, 0).await;
        assert!(header(&sent, "authorization").is_none());
    }

    #[tokio::test]
    async fn embeddings_are_batched_and_ordered_by_index() {
        let server = MockServer::start().await;
        // Echo each input's number back as its vector, in reverse order, to
        // prove ordering comes from `index`.
        Mock::given(method("POST"))
            .and(path("/v1/embeddings"))
            .respond_with(|req: &Request| {
                let body: Value = serde_json::from_slice(&req.body).unwrap();
                let mut data: Vec<Value> = body["input"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .enumerate()
                    .map(|(i, text)| json!({"index": i, "embedding": [text.as_str().unwrap().parse::<f32>().unwrap(), 0.5]}))
                    .collect();
                data.reverse();
                ResponseTemplate::new(200).set_body_json(json!({"data": data, "model": body["model"]}))
            })
            .mount(&server)
            .await;

        let config = config(&server).with_extra("dimensions", 2);
        let embedder = OpenAiEmbedder::new(&shared_client(), "mistral", "mistral-embed", &config).unwrap();
        assert_eq!(embedder.batch_limit(), 32);
        let model = Batched::new(embedder, &config);
        let texts: Vec<String> = (0..70).map(|i| i.to_string()).collect();
        let vectors = model.embed(&texts).await.unwrap();

        assert_eq!(vectors.len(), 70);
        for (i, v) in vectors.iter().enumerate() {
            assert_eq!(v, &vec![i as f32, 0.5]);
        }
        let requests = server.received_requests().await.unwrap();
        let sizes: Vec<usize> = requests
            .iter()
            .map(|r| serde_json::from_slice::<Value>(&r.body).unwrap()["input"].as_array().unwrap().len())
            .collect();
        assert_eq!(sizes, vec![32, 32, 6]);
        let first = request_json(&server, 0).await;
        assert_eq!(first["model"], json!("mistral-embed"));
        assert_eq!(first["dimensions"], json!(2));
    }

    #[tokio::test]
    async fn null_embeddings_are_rejected() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({"data": [{"index": 0, "embedding": [null, null]}]})))
            .mount(&server)
            .await;
        let config = config(&server);
        let model = Batched::new(OpenAiEmbedder::new(&shared_client(), "openai_compatible", "m", &config).unwrap(), &config);
        let err = model.embed(&["a".to_string()]).await.unwrap_err();
        assert!(matches!(err, AiError::Provider { status: None, .. }));
    }

    #[tokio::test]
    async fn speech_posts_json_and_returns_the_audio() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/audio/speech"))
            .and(header_is("authorization", "Bearer sk-test"))
            .respond_with(ResponseTemplate::new(200).insert_header("content-type", "audio/mpeg").set_body_bytes(b"ID3fake".to_vec()))
            .mount(&server)
            .await;
        let tts = OpenAiSpeech::new(&shared_client(), "openai", "gpt-4o-mini-tts", &config(&server)).unwrap();
        let audio = tts.synthesize("Hello there", "nova").await.unwrap();
        assert_eq!(audio.bytes, Bytes::from_static(b"ID3fake"));
        assert_eq!(audio.content_type, "audio/mpeg");
        assert_eq!(audio.extension(), "mp3");
        assert_eq!(
            request_json(&server, 0).await,
            json!({"model": "gpt-4o-mini-tts", "input": "Hello there", "voice": "nova", "response_format": "mp3"})
        );
        // An empty voice falls back to a valid one.
        tts.synthesize("x", "").await.unwrap();
        assert_eq!(request_json(&server, 1).await["voice"], json!("alloy"));
        assert!(tts.voices().await.unwrap().iter().any(|v| v.id == "alloy"));
    }

    #[tokio::test]
    async fn xai_speech_uses_its_own_endpoint_and_body() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/tts"))
            .respond_with(ResponseTemplate::new(200).insert_header("content-type", "audio/mpeg").set_body_bytes(vec![1u8, 2, 3]))
            .mount(&server)
            .await;
        let tts = OpenAiSpeech::new(&shared_client(), "xai", "grok-tts", &config(&server)).unwrap();
        let audio = tts.synthesize("Hi", "ara").await.unwrap();
        assert_eq!(audio.bytes.len(), 3);
        assert_eq!(request_json(&server, 0).await, json!({"text": "Hi", "voice_id": "ara", "language": "auto"}));
        assert_eq!(tts.voices().await.unwrap().len(), 5);
    }

    #[tokio::test]
    async fn mistral_speech_decodes_base64_json_and_lists_voices() {
        let server = MockServer::start().await;
        let encoded = base64::engine::general_purpose::STANDARD.encode(b"mp3-bytes");
        Mock::given(method("POST"))
            .and(path("/v1/audio/speech"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({"audio_data": encoded})))
            .mount(&server)
            .await;
        Mock::given(method("GET"))
            .and(path("/v1/audio/voices"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "items": [{"id": "v-1", "name": "Marie", "languages": ["fr"], "gender": "female"}, {"name": "no id"}],
                "total": 2,
            })))
            .mount(&server)
            .await;
        let tts = OpenAiSpeech::new(&shared_client(), "mistral", "voxtral-mini-tts-2603", &config(&server)).unwrap();
        let audio = tts.synthesize("Bonjour", "v-1").await.unwrap();
        assert_eq!(audio.bytes, Bytes::from_static(b"mp3-bytes"));
        assert_eq!(audio.content_type, "audio/mpeg");
        assert_eq!(
            request_json(&server, 0).await,
            json!({"model": "voxtral-mini-tts-2603", "input": "Bonjour", "voice_id": "v-1", "response_format": "mp3"})
        );
        let voices = tts.voices().await.unwrap();
        assert_eq!(
            voices,
            vec![Voice { id: "v-1".into(), name: "Marie".into(), language: Some("fr".into()), gender: Some("female".into()) }]
        );
    }

    fn transcription() -> TranscriptionRequest {
        TranscriptionRequest {
            audio: Bytes::from_static(b"RIFFfakeaudio"),
            filename: "clip.wav".to_string(),
            content_type: None,
            language: Some("en".to_string()),
            prompt: Some("Nova Notes".to_string()),
        }
    }

    #[tokio::test]
    async fn transcription_uploads_multipart_form_data() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/audio/transcriptions"))
            .and(header_is("authorization", "Bearer sk-test"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({"text": " Hello there. "})))
            .mount(&server)
            .await;
        let stt = OpenAiTranscriber::new(&shared_client(), "groq", "whisper-large-v3", &config(&server)).unwrap();
        assert_eq!(stt.transcribe(transcription()).await.unwrap(), "Hello there.");

        let sent = request(&server, 0).await;
        assert!(header(&sent, "content-type").unwrap().starts_with("multipart/form-data; boundary="));
        let body = String::from_utf8_lossy(&sent.body);
        for expected in [
            "name=\"model\"\r\n\r\nwhisper-large-v3",
            "name=\"language\"\r\n\r\nen",
            "name=\"prompt\"\r\n\r\nNova Notes",
            "name=\"response_format\"\r\n\r\njson",
            "name=\"file\"; filename=\"clip.wav\"",
            "content-type: audio/wav",
            "RIFFfakeaudio",
        ] {
            assert!(body.to_lowercase().contains(&expected.to_lowercase()), "missing {expected:?} in {body}");
        }
    }

    #[tokio::test]
    async fn azure_transcription_names_the_deployment_in_the_url_only() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/openai/deployments/whisper/audio/transcriptions"))
            .respond_with(ResponseTemplate::new(200).set_body_string("plain transcript\n"))
            .mount(&server)
            .await;
        let config = ProviderConfig { api_key: Some("k".into()), endpoint_stt: Some(server.uri()), ..Default::default() };
        let stt = OpenAiTranscriber::new(&shared_client(), "azure", "whisper", &config).unwrap();
        assert_eq!(stt.transcribe(transcription()).await.unwrap(), "plain transcript");
        let body = String::from_utf8_lossy(&request(&server, 0).await.body).to_string();
        assert!(!body.contains("name=\"model\""));
    }

    #[tokio::test]
    async fn openrouter_transcription_sends_base64_json() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/audio/transcriptions"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({"text": "hi"})))
            .mount(&server)
            .await;
        let stt = OpenAiTranscriber::new(&shared_client(), "openrouter", "openai/whisper-1", &config(&server)).unwrap();
        assert_eq!(stt.transcribe(transcription()).await.unwrap(), "hi");
        assert_eq!(
            request_json(&server, 0).await,
            json!({
                "model": "openai/whisper-1",
                "input_audio": {"data": base64::engine::general_purpose::STANDARD.encode(b"RIFFfakeaudio"), "format": "wav"},
                "language": "en",
                "prompt": "Nova Notes",
            })
        );
    }
}
