//! Google Gemini, through the Gemini API (`google`) and Vertex AI (`vertex`).
//!
//! Both speak the same `generateContent` wire format; they differ in the URL
//! prefix and in authentication (API key header versus a service-account
//! OAuth token). Embeddings are the exception: the Gemini API has
//! `batchEmbedContents`, Vertex uses the generic `predict` method.
//!
//! Gemini text-to-speech returns raw PCM, which is wrapped in a WAV header so
//! the audio is directly playable. Speech-to-text sends the audio inline with
//! a transcription instruction.

use async_trait::async_trait;
use base64::Engine;
use nn_llm::{
    AiError, AiResult, AudioData, ChatMessage, ChatOptions, ChatResponse, LanguageModel, Role, SpeechToTextModel,
    TextStream, TextToSpeechModel, TranscriptionRequest, Voice,
};
use reqwest::RequestBuilder;
use serde_json::{json, Map, Value};

use super::vertex_auth::ServiceAccount;
use super::{finish_reason, float, require_key, usage, vector, EmbedBatch};
use crate::audio::{audio_mime, is_raw_pcm, mime_param, pcm16_to_wav};
use crate::config::{non_blank, ProviderConfig};
use crate::error::http_error;
use crate::http::{encode_component, Http};
use crate::sse::{text_stream, Frame, Framing, Step};

pub(crate) const GEMINI_API_ROOT: &str = "https://generativelanguage.googleapis.com/v1beta";

const GEMINI_VOICES: &[&str] = &[
    "Zephyr", "Puck", "Charon", "Kore", "Fenrir", "Leda", "Orus", "Aoede", "Callirrhoe", "Autonoe", "Enceladus",
    "Iapetus", "Umbriel", "Algieba", "Despina", "Erinome", "Algenib", "Rasalgethi", "Laomedeia", "Achernar",
    "Alnilam", "Schedar", "Gacrux", "Pulcherrima", "Achird", "Zubenelgenubi", "Vindemiatrix", "Sadachbia",
    "Sadaltager", "Sulafat",
];
const DEFAULT_VOICE: &str = "Kore";
/// Sample rate of Gemini speech output when the MIME type does not say.
const DEFAULT_PCM_RATE: u32 = 24_000;

#[derive(Clone)]
enum Auth {
    ApiKey(String),
    ServiceAccount(ServiceAccount),
}

/// Where a Gemini model lives and how to authenticate against it.
#[derive(Clone)]
pub(crate) struct GoogleEndpoint {
    http: Http,
    provider: String,
    model: String,
    /// URL of the `models` collection the model belongs to.
    models_url: String,
    auth: Auth,
}

/// Adds the API version when a custom Gemini base URL lacks one.
pub(crate) fn gemini_root(base_url: Option<&str>) -> String {
    match base_url {
        None => GEMINI_API_ROOT.to_string(),
        Some(url) => {
            let last = url.rsplit('/').next().unwrap_or_default();
            let versioned = last.starts_with('v') && last[1..].starts_with(|c: char| c.is_ascii_digit());
            if versioned {
                url.to_string()
            } else {
                format!("{url}/v1beta")
            }
        }
    }
}

impl GoogleEndpoint {
    pub fn resolve(
        client: &reqwest::Client,
        provider: &str,
        model: &str,
        config: &ProviderConfig,
        default_timeout: std::time::Duration,
    ) -> AiResult<Self> {
        let http = Http::new(client, provider, config, default_timeout);
        let model = model.trim().trim_start_matches("models/").to_string();
        let (models_url, auth) = if provider == "vertex" {
            let project = non_blank(&config.project)
                .ok_or_else(|| AiError::Configuration("No Vertex AI project configured".to_string()))?;
            let location = non_blank(&config.location)
                .ok_or_else(|| AiError::Configuration("No Vertex AI location configured".to_string()))?;
            let credentials = non_blank(&config.credentials_path).ok_or_else(|| {
                AiError::Configuration("No service account credentials file configured for Vertex AI".to_string())
            })?;
            let host = match config.url() {
                Some(url) => url.to_string(),
                None if location == "global" => "https://aiplatform.googleapis.com".to_string(),
                None => format!("https://{location}-aiplatform.googleapis.com"),
            };
            let models_url = format!(
                "{host}/v1/projects/{}/locations/{}/publishers/google/models",
                encode_component(project),
                encode_component(location)
            );
            (models_url, Auth::ServiceAccount(ServiceAccount::new(credentials, http.clone())))
        } else {
            let key = require_key(provider, config)?;
            (format!("{}/models", gemini_root(config.url())), Auth::ApiKey(key))
        };
        Ok(Self { http, provider: provider.to_string(), model, models_url, auth })
    }

    fn is_vertex(&self) -> bool {
        matches!(self.auth, Auth::ServiceAccount(_))
    }

    pub fn url(&self, method: &str) -> String {
        format!("{}/{}:{method}", self.models_url, encode_component(&self.model))
    }

    /// A POST to `{model}:{method}` with credentials attached. The API key
    /// travels in a header, never in the URL.
    async fn post(&self, method: &str, body: &Value) -> AiResult<RequestBuilder> {
        let request = self.http.client.post(self.url(method)).json(body);
        Ok(match &self.auth {
            Auth::ApiKey(key) => request.header("x-goog-api-key", key),
            Auth::ServiceAccount(account) => request.bearer_auth(account.token().await?),
        })
    }

    fn secret(&self) -> Option<&str> {
        match &self.auth {
            Auth::ApiKey(key) => Some(key),
            Auth::ServiceAccount(_) => None,
        }
    }

    async fn generate(&self, body: &Value) -> AiResult<Value> {
        self.generate_capped(body, crate::http::MAX_JSON_RESPONSE_BYTES).await
    }

    /// `generateContent` with a body limit (speech replies carry base64 audio).
    async fn generate_capped(&self, body: &Value, cap: usize) -> AiResult<Value> {
        let request = self.post("generateContent", body).await?;
        self.http.json_capped(request, cap).await
    }
}

// ---------------------------------------------------------------------------
// Chat
// ---------------------------------------------------------------------------

pub(crate) fn generate_body(messages: &[ChatMessage], options: &ChatOptions) -> Value {
    let mut system: Vec<&str> = Vec::new();
    let mut contents: Vec<Value> = Vec::new();
    for message in messages.iter().filter(|m| !m.content.trim().is_empty()) {
        let role = match message.role {
            Role::System => {
                system.push(&message.content);
                continue;
            }
            Role::User => "user",
            Role::Assistant => "model",
        };
        contents.push(json!({"role": role, "parts": [{"text": message.content}]}));
    }
    let mut system = system.join("\n\n");
    if contents.is_empty() && !system.is_empty() {
        contents.push(json!({"role": "user", "parts": [{"text": std::mem::take(&mut system)}]}));
    }

    let mut generation = Map::new();
    if let Some(temperature) = options.temperature {
        generation.insert("temperature".into(), float(temperature));
    }
    if let Some(top_p) = options.top_p {
        generation.insert("topP".into(), float(top_p));
    }
    if let Some(max_tokens) = options.max_tokens {
        generation.insert("maxOutputTokens".into(), json!(max_tokens));
    }
    if options.json {
        generation.insert("responseMimeType".into(), json!("application/json"));
    }

    let mut body = Map::new();
    body.insert("contents".into(), Value::Array(contents));
    if !system.is_empty() {
        body.insert("systemInstruction".into(), json!({"parts": [{"text": system}]}));
    }
    if !generation.is_empty() {
        body.insert("generationConfig".into(), Value::Object(generation));
    }
    Value::Object(body)
}

/// Concatenated answer text of the first candidate, without thought parts.
fn candidate_text(json: &Value) -> String {
    json.pointer("/candidates/0/content/parts")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter(|part| part.get("thought").and_then(Value::as_bool) != Some(true))
        .filter_map(|part| part.get("text").and_then(Value::as_str))
        .collect()
}

pub(crate) struct GeminiChat {
    endpoint: GoogleEndpoint,
}

impl GeminiChat {
    pub fn new(endpoint: GoogleEndpoint) -> Self {
        Self { endpoint }
    }
}

#[async_trait]
impl LanguageModel for GeminiChat {
    fn provider(&self) -> &str {
        &self.endpoint.provider
    }

    fn model_name(&self) -> &str {
        &self.endpoint.model
    }

    async fn chat(&self, messages: &[ChatMessage], options: &ChatOptions) -> AiResult<ChatResponse> {
        let ep = &self.endpoint;
        let json = ep.generate(&generate_body(messages, options)).await?;
        // A blocked prompt has no candidates, only a block reason.
        let reason = json.pointer("/candidates/0/finishReason").or_else(|| json.pointer("/promptFeedback/blockReason"));
        Ok(ChatResponse {
            content: candidate_text(&json),
            model: json.get("modelVersion").and_then(Value::as_str).unwrap_or(&ep.model).to_string(),
            finish_reason: finish_reason(reason),
            usage: usage(
                json.pointer("/usageMetadata/promptTokenCount"),
                json.pointer("/usageMetadata/candidatesTokenCount"),
            ),
        })
    }

    async fn chat_stream(&self, messages: &[ChatMessage], options: &ChatOptions) -> AiResult<TextStream> {
        let ep = &self.endpoint;
        let request = ep.post("streamGenerateContent", &generate_body(messages, options)).await?;
        let response = ep.http.send_streaming(request.query(&[("alt", "sse")])).await?;
        let provider = ep.provider.clone();
        let secret = ep.secret().map(str::to_string);
        Ok(text_stream(response, Framing::Sse, ep.http.timeout, &ep.provider, move |frame: Frame| {
            let Ok(json) = serde_json::from_str::<Value>(&frame.data) else {
                return Step::Skip;
            };
            if let Some(error) = json.get("error") {
                let status = error.get("code").and_then(Value::as_u64).filter(|c| (400..600).contains(c)).unwrap_or(500);
                return Step::Fail(http_error(&provider, status as u16, &frame.data, secret.as_deref()));
            }
            Step::Text(candidate_text(&json))
        }))
    }
}

// ---------------------------------------------------------------------------
// Embeddings
// ---------------------------------------------------------------------------

pub(crate) struct GeminiEmbedder {
    endpoint: GoogleEndpoint,
    task_type: Option<String>,
    dimensions: Option<u64>,
}

impl GeminiEmbedder {
    pub fn new(endpoint: GoogleEndpoint, config: &ProviderConfig) -> Self {
        Self {
            endpoint,
            task_type: config.extra_str("task_type").map(str::to_string),
            dimensions: config.extra_u64("dimensions"),
        }
    }

    fn gemini_body(&self, texts: &[String]) -> Value {
        let model = format!("models/{}", self.endpoint.model);
        let requests: Vec<Value> = texts
            .iter()
            .map(|text| {
                let mut request = json!({"model": model, "content": {"parts": [{"text": text}]}});
                if let Some(task_type) = &self.task_type {
                    request["taskType"] = json!(task_type);
                }
                if let Some(dimensions) = self.dimensions {
                    request["outputDimensionality"] = json!(dimensions);
                }
                request
            })
            .collect();
        json!({"requests": requests})
    }

    fn vertex_body(&self, texts: &[String]) -> Value {
        let instances: Vec<Value> = texts
            .iter()
            .map(|text| {
                let mut instance = json!({"content": text});
                if let Some(task_type) = &self.task_type {
                    instance["task_type"] = json!(task_type);
                }
                instance
            })
            .collect();
        let mut parameters = json!({"autoTruncate": true});
        if let Some(dimensions) = self.dimensions {
            parameters["outputDimensionality"] = json!(dimensions);
        }
        json!({"instances": instances, "parameters": parameters})
    }
}

#[async_trait]
impl EmbedBatch for GeminiEmbedder {
    fn provider(&self) -> &str {
        &self.endpoint.provider
    }

    fn model_name(&self) -> &str {
        &self.endpoint.model
    }

    fn batch_limit(&self) -> usize {
        if !self.endpoint.is_vertex() {
            100
        } else if self.endpoint.model.contains("gemini-embedding") {
            // Vertex accepts a single input per request for these models.
            1
        } else {
            // Vertex allows 250 inputs but only 20k tokens per request.
            16
        }
    }

    async fn embed_batch(&self, texts: &[String]) -> AiResult<Vec<Vec<f32>>> {
        let ep = &self.endpoint;
        let (method, body, list, item) = if ep.is_vertex() {
            ("predict", self.vertex_body(texts), "predictions", "/embeddings/values")
        } else {
            ("batchEmbedContents", self.gemini_body(texts), "embeddings", "/values")
        };
        let json = ep.http.json(ep.post(method, &body).await?).await?;
        json.get(list)
            .and_then(Value::as_array)
            .and_then(|items| items.iter().map(|i| i.pointer(item).and_then(vector)).collect::<Option<Vec<_>>>())
            .ok_or_else(|| ep.http.malformed("no embeddings in the response"))
    }
}

// ---------------------------------------------------------------------------
// Text to speech
// ---------------------------------------------------------------------------

pub(crate) struct GeminiSpeech {
    endpoint: GoogleEndpoint,
}

impl GeminiSpeech {
    pub fn new(endpoint: GoogleEndpoint) -> Self {
        Self { endpoint }
    }
}

pub(crate) fn speech_body(text: &str, voice: &str) -> Value {
    let voice = if voice.trim().is_empty() { DEFAULT_VOICE } else { voice.trim() };
    json!({
        "contents": [{"role": "user", "parts": [{"text": text}]}],
        "generationConfig": {
            "responseModalities": ["AUDIO"],
            "speechConfig": {"voiceConfig": {"prebuiltVoiceConfig": {"voiceName": voice}}},
        },
    })
}

/// Extracts the audio part of a speech response, wrapping raw PCM as WAV.
fn speech_audio(json: &Value) -> Option<AudioData> {
    let parts = json.pointer("/candidates/0/content/parts")?.as_array()?;
    let inline = parts.iter().find_map(|part| part.get("inlineData").or_else(|| part.get("inline_data")))?;
    let mime = inline.get("mimeType").or_else(|| inline.get("mime_type")).and_then(Value::as_str).unwrap_or_default();
    let data = base64::engine::general_purpose::STANDARD.decode(inline.get("data")?.as_str()?).ok()?;
    if mime.is_empty() || is_raw_pcm(mime) {
        let rate = mime_param(mime, "rate").unwrap_or(DEFAULT_PCM_RATE);
        let channels = mime_param(mime, "channels").unwrap_or(1) as u16;
        Some(AudioData { bytes: pcm16_to_wav(&data, rate, channels), content_type: "audio/wav".to_string() })
    } else {
        let content_type = mime.split(';').next().unwrap_or(mime).trim().to_string();
        Some(AudioData { bytes: data.into(), content_type })
    }
}

#[async_trait]
impl TextToSpeechModel for GeminiSpeech {
    fn provider(&self) -> &str {
        &self.endpoint.provider
    }

    fn model_name(&self) -> &str {
        &self.endpoint.model
    }

    async fn synthesize(&self, text: &str, voice: &str) -> AiResult<AudioData> {
        let json = self.endpoint.generate_capped(&speech_body(text, voice), crate::http::MAX_AUDIO_RESPONSE_BYTES).await?;
        speech_audio(&json).ok_or_else(|| self.endpoint.http.malformed("no audio in the response"))
    }

    async fn voices(&self) -> AiResult<Vec<Voice>> {
        Ok(GEMINI_VOICES
            .iter()
            .map(|name| Voice { id: name.to_string(), name: name.to_string(), language: None, gender: None })
            .collect())
    }
}

// ---------------------------------------------------------------------------
// Speech to text
// ---------------------------------------------------------------------------

pub(crate) struct GeminiTranscriber {
    endpoint: GoogleEndpoint,
}

impl GeminiTranscriber {
    pub fn new(endpoint: GoogleEndpoint) -> Self {
        Self { endpoint }
    }
}

pub(crate) fn transcription_body(request: &TranscriptionRequest) -> Value {
    let mut instruction =
        "Transcribe the speech in this audio verbatim. Reply with the transcript only, without commentary or timestamps."
            .to_string();
    if let Some(language) = non_blank(&request.language) {
        instruction.push_str(&format!(" The spoken language is {language}."));
    }
    if let Some(prompt) = non_blank(&request.prompt) {
        instruction.push_str(&format!(" Context that may help with names and terms: {prompt}"));
    }
    let mime = non_blank(&request.content_type).unwrap_or_else(|| audio_mime(&request.filename));
    json!({
        "contents": [{
            "role": "user",
            "parts": [
                {"text": instruction},
                {"inlineData": {
                    "mimeType": mime,
                    "data": base64::engine::general_purpose::STANDARD.encode(&request.audio),
                }},
            ],
        }],
        "generationConfig": {"temperature": 0},
    })
}

#[async_trait]
impl SpeechToTextModel for GeminiTranscriber {
    fn provider(&self) -> &str {
        &self.endpoint.provider
    }

    fn model_name(&self) -> &str {
        &self.endpoint.model
    }

    async fn transcribe(&self, request: TranscriptionRequest) -> AiResult<String> {
        let json = self.endpoint.generate(&transcription_body(&request)).await?;
        Ok(candidate_text(&json).trim().to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{DEFAULT_CHAT_TIMEOUT, DEFAULT_EMBEDDING_TIMEOUT};
    use crate::http::shared_client;
    use crate::providers::testing::{collect_text, collect_until_error, header, request, request_json};
    use crate::providers::vertex_auth::tests::{mount_token_endpoint, service_account_file};
    use crate::providers::Batched;
    use bytes::Bytes;
    use nn_llm::EmbeddingModel;
    use wiremock::matchers::{method, path, query_param};
    use wiremock::{Mock, MockServer, Request, ResponseTemplate};

    fn gemini(model: &str, server: &MockServer) -> GoogleEndpoint {
        let config = ProviderConfig::new().with_api_key("AIza-test").with_base_url(server.uri());
        GoogleEndpoint::resolve(&shared_client(), "google", model, &config, DEFAULT_CHAT_TIMEOUT).unwrap()
    }

    fn vertex_config(server: &MockServer, credentials: &tempfile::NamedTempFile) -> ProviderConfig {
        ProviderConfig {
            project: Some("demo-project".into()),
            location: Some("europe-west4".into()),
            credentials_path: Some(credentials.path().to_str().unwrap().to_string()),
            base_url: Some(server.uri()),
            ..Default::default()
        }
    }

    fn reply(text: &str) -> Value {
        json!({
            "candidates": [{
                "content": {"role": "model", "parts": [{"text": "pondering", "thought": true}, {"text": text}]},
                "finishReason": "STOP",
            }],
            "usageMetadata": {"promptTokenCount": 7, "candidatesTokenCount": 2, "totalTokenCount": 9},
            "modelVersion": "gemini-2.5-flash-001",
        })
    }

    #[test]
    fn endpoint_urls_follow_each_api() {
        let client = shared_client();
        let google = GoogleEndpoint::resolve(&client, "google", "models/gemini-2.5-flash", &ProviderConfig::new().with_api_key("k"), DEFAULT_CHAT_TIMEOUT).unwrap();
        assert_eq!(google.url("generateContent"), "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent");

        assert_eq!(gemini_root(Some("https://proxy.test/google")), "https://proxy.test/google/v1beta");
        assert_eq!(gemini_root(Some("https://proxy.test/v1")), "https://proxy.test/v1");
        assert_eq!(gemini_root(Some("https://proxy.test/v1beta")), "https://proxy.test/v1beta");

        let vertex = |location: &str| {
            let config = ProviderConfig {
                project: Some("p1".into()),
                location: Some(location.into()),
                credentials_path: Some("/tmp/sa.json".into()),
                ..Default::default()
            };
            GoogleEndpoint::resolve(&client, "vertex", "gemini-2.5-flash", &config, DEFAULT_CHAT_TIMEOUT).unwrap().url("generateContent")
        };
        assert_eq!(
            vertex("us-central1"),
            "https://us-central1-aiplatform.googleapis.com/v1/projects/p1/locations/us-central1/publishers/google/models/gemini-2.5-flash:generateContent"
        );
        assert_eq!(
            vertex("global"),
            "https://aiplatform.googleapis.com/v1/projects/p1/locations/global/publishers/google/models/gemini-2.5-flash:generateContent"
        );
    }

    #[test]
    fn missing_settings_are_configuration_errors() {
        let client = shared_client();
        let resolve = |provider: &str, config: ProviderConfig| {
            GoogleEndpoint::resolve(&client, provider, "m", &config, DEFAULT_CHAT_TIMEOUT).err().map(|e| e.to_string())
        };
        assert_eq!(resolve("google", ProviderConfig::new()).as_deref(), Some("No API key configured for google"));
        assert_eq!(resolve("vertex", ProviderConfig::new()).as_deref(), Some("No Vertex AI project configured"));
        let project = ProviderConfig { project: Some("p".into()), ..Default::default() };
        assert_eq!(resolve("vertex", project.clone()).as_deref(), Some("No Vertex AI location configured"));
        let located = ProviderConfig { location: Some("us-central1".into()), ..project };
        assert_eq!(
            resolve("vertex", located).as_deref(),
            Some("No service account credentials file configured for Vertex AI")
        );
    }

    #[tokio::test]
    async fn chat_uses_system_instruction_and_generation_config() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1beta/models/gemini-2.5-flash:generateContent"))
            .respond_with(ResponseTemplate::new(200).set_body_json(reply("{\"ok\":true}")))
            .mount(&server)
            .await;
        let model = GeminiChat::new(gemini("gemini-2.5-flash", &server));
        let messages = [ChatMessage::system("Reply in JSON."), ChatMessage::user("Hi"), ChatMessage::assistant("Hello"), ChatMessage::user("Go")];
        let options = ChatOptions { temperature: Some(0.2), top_p: Some(0.8), max_tokens: Some(100), json: true };
        let out = model.chat(&messages, &options).await.unwrap();

        assert_eq!(out.content, "{\"ok\":true}");
        assert_eq!(out.model, "gemini-2.5-flash-001");
        assert_eq!(out.finish_reason.as_deref(), Some("stop"));
        assert_eq!(out.usage, Some(nn_llm::Usage { prompt_tokens: 7, completion_tokens: 2 }));

        let sent = request(&server, 0).await;
        assert_eq!(header(&sent, "x-goog-api-key"), Some("AIza-test"));
        assert_eq!(sent.url.query(), None, "the key must not be in the URL");
        assert_eq!(
            request_json(&server, 0).await,
            json!({
                "contents": [
                    {"role": "user", "parts": [{"text": "Hi"}]},
                    {"role": "model", "parts": [{"text": "Hello"}]},
                    {"role": "user", "parts": [{"text": "Go"}]},
                ],
                "systemInstruction": {"parts": [{"text": "Reply in JSON."}]},
                "generationConfig": {
                    "temperature": 0.2,
                    "topP": 0.8,
                    "maxOutputTokens": 100,
                    "responseMimeType": "application/json",
                },
            })
        );
    }

    #[test]
    fn default_options_send_no_generation_config() {
        let body = generate_body(&[ChatMessage::user("x")], &ChatOptions::default());
        assert_eq!(body, json!({"contents": [{"role": "user", "parts": [{"text": "x"}]}]}));
        let body = generate_body(&[ChatMessage::system("only")], &ChatOptions::default());
        assert_eq!(body, json!({"contents": [{"role": "user", "parts": [{"text": "only"}]}]}));
    }

    #[tokio::test]
    async fn blocked_prompts_report_a_content_filter_finish() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({"promptFeedback": {"blockReason": "SAFETY"}})))
            .mount(&server)
            .await;
        let out = GeminiChat::new(gemini("gemini-2.5-flash", &server)).chat(&[ChatMessage::user("x")], &ChatOptions::default()).await.unwrap();
        assert_eq!(out.content, "");
        assert_eq!(out.finish_reason.as_deref(), Some("content_filter"));
    }

    #[tokio::test]
    async fn stream_uses_alt_sse_and_yields_text() {
        let server = MockServer::start().await;
        let sse = concat!(
            "data: {\"candidates\":[{\"content\":{\"role\":\"model\",\"parts\":[{\"text\":\"plan\",\"thought\":true}]}}]}\r\n\r\n",
            "data: {\"candidates\":[{\"content\":{\"role\":\"model\",\"parts\":[{\"text\":\"Hel\"}]}}]}\r\n\r\n",
            "data: {\"candidates\":[{\"content\":{\"role\":\"model\",\"parts\":[{\"text\":\"lo\"},{\"text\":\"!\"}]},\"finishReason\":\"STOP\"}],\"usageMetadata\":{\"promptTokenCount\":1}}\r\n\r\n",
        );
        Mock::given(method("POST"))
            .and(path("/v1beta/models/gemini-2.5-flash:streamGenerateContent"))
            .and(query_param("alt", "sse"))
            .respond_with(ResponseTemplate::new(200).insert_header("content-type", "text/event-stream").set_body_string(sse))
            .mount(&server)
            .await;
        let model = GeminiChat::new(gemini("gemini-2.5-flash", &server));
        let stream = model.chat_stream(&[ChatMessage::user("Hi")], &ChatOptions::default()).await.unwrap();
        assert_eq!(collect_text(stream).await, vec!["Hel", "lo!"]);
    }

    #[tokio::test]
    async fn errors_are_classified_before_and_during_a_stream() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1beta/models/gemini-0:generateContent"))
            .respond_with(ResponseTemplate::new(404).set_body_json(json!({"error": {
                "code": 404, "status": "NOT_FOUND",
                "message": "models/gemini-0 is not found for API version v1beta, or is not supported for generateContent.",
            }})))
            .mount(&server)
            .await;
        Mock::given(method("POST"))
            .and(path("/v1beta/models/gemini-busy:generateContent"))
            .respond_with(ResponseTemplate::new(429).set_body_json(json!({"error": {
                "code": 429, "status": "RESOURCE_EXHAUSTED", "message": "Resource has been exhausted (e.g. check quota).",
            }})))
            .mount(&server)
            .await;
        let sse = "data: {\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"a\"}]}}]}\n\ndata: {\"error\":{\"code\":503,\"message\":\"The model is overloaded.\",\"status\":\"UNAVAILABLE\"}}\n\n";
        Mock::given(method("POST")).and(query_param("alt", "sse")).respond_with(ResponseTemplate::new(200).set_body_string(sse)).mount(&server).await;

        let user = [ChatMessage::user("x")];
        let err = GeminiChat::new(gemini("gemini-0", &server)).chat(&user, &ChatOptions::default()).await.unwrap_err();
        assert!(matches!(err, AiError::ModelNotFound(_)), "{err:?}");
        let err = GeminiChat::new(gemini("gemini-busy", &server)).chat(&user, &ChatOptions::default()).await.unwrap_err();
        assert!(matches!(err, AiError::RateLimit(_)), "{err:?}");
        let stream = GeminiChat::new(gemini("gemini-2.5-flash", &server)).chat_stream(&user, &ChatOptions::default()).await.unwrap();
        let (text, err) = collect_until_error(stream).await;
        assert_eq!(text, vec!["a"]);
        assert!(matches!(err, Some(AiError::Provider { status: Some(503), .. })), "{err:?}");
    }

    #[tokio::test]
    async fn vertex_chat_authenticates_with_a_service_account_token() {
        let server = MockServer::start().await;
        mount_token_endpoint(&server, "ya29.vertex").await;
        Mock::given(method("POST"))
            .and(path("/v1/projects/demo-project/locations/europe-west4/publishers/google/models/gemini-2.5-flash:generateContent"))
            .respond_with(ResponseTemplate::new(200).set_body_json(reply("Hallo")))
            .mount(&server)
            .await;
        let credentials = service_account_file(&format!("{}/token", server.uri()));
        let config = vertex_config(&server, &credentials);
        let endpoint = GoogleEndpoint::resolve(&shared_client(), "vertex", "gemini-2.5-flash", &config, DEFAULT_CHAT_TIMEOUT).unwrap();
        let model = GeminiChat::new(endpoint);
        assert_eq!(model.chat(&[ChatMessage::user("Hi")], &ChatOptions::default()).await.unwrap().content, "Hallo");
        assert_eq!(model.provider(), "vertex");

        let requests = server.received_requests().await.unwrap();
        let call = requests.iter().find(|r| r.url.path().ends_with(":generateContent")).unwrap();
        assert_eq!(header(call, "authorization"), Some("Bearer ya29.vertex"));
        assert!(header(call, "x-goog-api-key").is_none());
    }

    #[tokio::test]
    async fn gemini_embeddings_use_batch_embed_contents() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1beta/models/gemini-embedding-001:batchEmbedContents"))
            .respond_with(|req: &Request| {
                let body: Value = serde_json::from_slice(&req.body).unwrap();
                let embeddings: Vec<Value> = body["requests"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|r| json!({"values": [r["content"]["parts"][0]["text"].as_str().unwrap().parse::<f32>().unwrap()]}))
                    .collect();
                ResponseTemplate::new(200).set_body_json(json!({"embeddings": embeddings}))
            })
            .mount(&server)
            .await;
        let config = ProviderConfig::new()
            .with_api_key("AIza-test")
            .with_base_url(server.uri())
            .with_extra("task_type", "RETRIEVAL_DOCUMENT")
            .with_extra("dimensions", 768);
        let endpoint = GoogleEndpoint::resolve(&shared_client(), "google", "gemini-embedding-001", &config, DEFAULT_EMBEDDING_TIMEOUT).unwrap();
        let model = Batched::new(GeminiEmbedder::new(endpoint, &config), &config);
        let texts: Vec<String> = (0..130).map(|i| i.to_string()).collect();
        let vectors = model.embed(&texts).await.unwrap();
        assert_eq!(vectors, (0..130).map(|i| vec![i as f32]).collect::<Vec<_>>());

        assert_eq!(server.received_requests().await.unwrap().len(), 2);
        let first = request_json(&server, 0).await;
        assert_eq!(first["requests"].as_array().unwrap().len(), 100);
        assert_eq!(
            first["requests"][0],
            json!({
                "model": "models/gemini-embedding-001",
                "content": {"parts": [{"text": "0"}]},
                "taskType": "RETRIEVAL_DOCUMENT",
                "outputDimensionality": 768,
            })
        );
    }

    #[tokio::test]
    async fn vertex_embeddings_use_predict() {
        let server = MockServer::start().await;
        mount_token_endpoint(&server, "ya29.vertex").await;
        Mock::given(method("POST"))
            .and(path("/v1/projects/demo-project/locations/europe-west4/publishers/google/models/text-embedding-005:predict"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "predictions": [
                    {"embeddings": {"values": [0.1, 0.2], "statistics": {"token_count": 2}}},
                    {"embeddings": {"values": [0.3, 0.4]}},
                ],
            })))
            .mount(&server)
            .await;
        let credentials = service_account_file(&format!("{}/token", server.uri()));
        let config = vertex_config(&server, &credentials);
        let endpoint = GoogleEndpoint::resolve(&shared_client(), "vertex", "text-embedding-005", &config, DEFAULT_EMBEDDING_TIMEOUT).unwrap();
        let embedder = GeminiEmbedder::new(endpoint, &config);
        assert_eq!(embedder.batch_limit(), 16);
        let vectors = embedder.embed_batch(&["a".to_string(), "b".to_string()]).await.unwrap();
        assert_eq!(vectors, vec![vec![0.1, 0.2], vec![0.3, 0.4]]);

        let requests = server.received_requests().await.unwrap();
        let call = requests.iter().find(|r| r.url.path().ends_with(":predict")).unwrap();
        assert_eq!(
            serde_json::from_slice::<Value>(&call.body).unwrap(),
            json!({"instances": [{"content": "a"}, {"content": "b"}], "parameters": {"autoTruncate": true}})
        );
    }

    #[tokio::test]
    async fn speech_wraps_raw_pcm_in_a_wav_header() {
        let server = MockServer::start().await;
        let pcm = vec![7u8; 4800];
        Mock::given(method("POST"))
            .and(path("/v1beta/models/gemini-2.5-flash-preview-tts:generateContent"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "candidates": [{"content": {"parts": [{"inlineData": {
                    "mimeType": "audio/L16;codec=pcm;rate=24000",
                    "data": base64::engine::general_purpose::STANDARD.encode(&pcm),
                }}]}}],
            })))
            .mount(&server)
            .await;
        let tts = GeminiSpeech::new(gemini("gemini-2.5-flash-preview-tts", &server));
        let audio = tts.synthesize("Hello", "Puck").await.unwrap();

        assert_eq!(audio.content_type, "audio/wav");
        assert_eq!(audio.extension(), "wav");
        assert_eq!(&audio.bytes[0..4], b"RIFF");
        assert_eq!(u32::from_le_bytes(audio.bytes[24..28].try_into().unwrap()), 24_000);
        assert_eq!(&audio.bytes[44..], &pcm[..]);
        assert_eq!(
            request_json(&server, 0).await,
            json!({
                "contents": [{"role": "user", "parts": [{"text": "Hello"}]}],
                "generationConfig": {
                    "responseModalities": ["AUDIO"],
                    "speechConfig": {"voiceConfig": {"prebuiltVoiceConfig": {"voiceName": "Puck"}}},
                },
            })
        );
        assert_eq!(speech_body("x", " ")["generationConfig"]["speechConfig"]["voiceConfig"]["prebuiltVoiceConfig"]["voiceName"], json!("Kore"));
        assert!(tts.voices().await.unwrap().iter().any(|v| v.id == "Kore"));
    }

    #[test]
    fn encoded_speech_audio_passes_through() {
        let json = json!({"candidates": [{"content": {"parts": [
            {"text": "ignored"},
            {"inline_data": {"mime_type": "audio/mpeg", "data": base64::engine::general_purpose::STANDARD.encode(b"mp3")}},
        ]}}]});
        let audio = speech_audio(&json).unwrap();
        assert_eq!(audio.content_type, "audio/mpeg");
        assert_eq!(audio.bytes, Bytes::from_static(b"mp3"));
        assert!(speech_audio(&json!({"candidates": [{"content": {"parts": [{"text": "no audio"}]}}]})).is_none());
    }

    #[tokio::test]
    async fn transcription_sends_the_audio_inline() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1beta/models/gemini-2.5-flash:generateContent"))
            .respond_with(ResponseTemplate::new(200).set_body_json(reply(" Hello there.\n")))
            .mount(&server)
            .await;
        let stt = GeminiTranscriber::new(gemini("gemini-2.5-flash", &server));
        let request = TranscriptionRequest {
            audio: Bytes::from_static(b"ID3audio"),
            filename: "memo.mp3".into(),
            content_type: None,
            language: Some("en".into()),
            prompt: Some("Nova".into()),
        };
        assert_eq!(stt.transcribe(request).await.unwrap(), "Hello there.");

        let body = request_json(&server, 0).await;
        let parts = &body["contents"][0]["parts"];
        let instruction = parts[0]["text"].as_str().unwrap();
        assert!(instruction.contains("Transcribe"));
        assert!(instruction.contains("language is en"));
        assert!(instruction.contains("Nova"));
        assert_eq!(
            parts[1],
            json!({"inlineData": {"mimeType": "audio/mpeg", "data": base64::engine::general_purpose::STANDARD.encode(b"ID3audio")}})
        );
        assert_eq!(body["generationConfig"], json!({"temperature": 0}));
    }
}
