//! Anthropic Messages API, for Anthropic itself and for any
//! `anthropic_compatible` endpoint.
//!
//! The Messages API differs from the OpenAI shape in three ways that matter
//! here: system prompts are a top-level `system` field rather than messages,
//! `max_tokens` is mandatory, and streaming uses named SSE events.

use async_trait::async_trait;
use nn_llm::{AiError, AiResult, ChatMessage, ChatOptions, ChatResponse, LanguageModel, Role, TextStream};
use reqwest::RequestBuilder;
use serde_json::{json, Map, Value};

use super::{content_text, finish_reason, float, require_key, usage};
use crate::config::{ProviderConfig, DEFAULT_CHAT_TIMEOUT};
use crate::error::http_error;
use crate::http::{join_url, Http};
use crate::sse::{text_stream, Frame, Framing, Step};

pub(crate) const ANTHROPIC_VERSION: &str = "2023-06-01";
pub(crate) const ANTHROPIC_API_ROOT: &str = "https://api.anthropic.com/v1";

/// Returns an Anthropic-style API base URL ending in `/v1`.
///
/// Accepts a bare host, a `/v1` URL or a `/v1/models` URL and collapses them
/// all to the `/v1` form, so callers can append `/messages` or `/models`
/// exactly once.
pub fn normalize_anthropic_compatible_base_url(base_url: &str) -> String {
    let mut url = base_url.trim().trim_end_matches('/');
    if let Some(stripped) = url.strip_suffix("/models") {
        url = stripped.trim_end_matches('/');
    }
    if url.ends_with("/v1") {
        url.to_string()
    } else {
        format!("{url}/v1")
    }
}

/// Output limit used when the caller sets none (the API requires one).
fn default_max_tokens(model: &str) -> u32 {
    let name = model.to_ascii_lowercase();
    let small_output =
        ["claude-3-opus", "claude-3-sonnet", "claude-3-haiku", "claude-2", "claude-instant"].iter().any(|m| name.contains(m));
    if small_output {
        4096
    } else {
        8192
    }
}

pub(crate) struct AnthropicChat {
    http: Http,
    provider: String,
    model: String,
    base: String,
    api_key: String,
    default_max_tokens: u32,
}

/// Builds the Messages API body. System messages are hoisted into `system`;
/// empty messages are dropped because the API rejects empty text blocks.
pub(crate) fn messages_body(
    model: &str,
    messages: &[ChatMessage],
    options: &ChatOptions,
    default_max_tokens: u32,
    stream: bool,
    sampling: bool,
) -> Value {
    let mut system: Vec<&str> = Vec::new();
    let mut turns: Vec<Value> = Vec::new();
    for message in messages.iter().filter(|m| !m.content.trim().is_empty()) {
        match message.role {
            Role::System => system.push(&message.content),
            role => turns.push(json!({"role": role.as_str(), "content": message.content})),
        }
    }
    let mut system = system.join("\n\n");
    if turns.is_empty() && !system.is_empty() {
        // The API needs at least one user turn; a lone system prompt is the
        // whole instruction.
        turns.push(json!({"role": "user", "content": std::mem::take(&mut system)}));
    }

    let mut body = Map::new();
    body.insert("model".into(), json!(model));
    body.insert("max_tokens".into(), json!(options.max_tokens.unwrap_or(default_max_tokens)));
    if !system.is_empty() {
        body.insert("system".into(), json!(system));
    }
    body.insert("messages".into(), Value::Array(turns));
    if sampling {
        // Current models reject requests that set both; temperature wins.
        if let Some(temperature) = options.temperature {
            body.insert("temperature".into(), float(temperature));
        } else if let Some(top_p) = options.top_p {
            body.insert("top_p".into(), float(top_p));
        }
    }
    if stream {
        body.insert("stream".into(), json!(true));
    }
    Value::Object(body)
}

/// HTTP status equivalent of an error `type` delivered inside a stream.
fn status_for_error_type(kind: &str) -> u16 {
    match kind {
        "authentication_error" => 401,
        "permission_error" => 403,
        "not_found_error" => 404,
        "request_too_large" => 413,
        "rate_limit_error" => 429,
        "invalid_request_error" => 400,
        "overloaded_error" => 529,
        _ => 500,
    }
}

impl AnthropicChat {
    pub fn new(client: &reqwest::Client, provider: &str, model: &str, config: &ProviderConfig) -> AiResult<Self> {
        let (base, api_key) = if provider == "anthropic_compatible" {
            match (config.url(), config.key()) {
                (Some(url), Some(key)) => (normalize_anthropic_compatible_base_url(url), key.to_string()),
                _ => {
                    return Err(AiError::Configuration(
                        "Anthropic-compatible models require a base URL and API key".to_string(),
                    ))
                }
            }
        } else {
            let base = config.url().map(normalize_anthropic_compatible_base_url);
            (base.unwrap_or_else(|| ANTHROPIC_API_ROOT.to_string()), require_key(provider, config)?)
        };
        Ok(Self {
            http: Http::new(client, provider, config, DEFAULT_CHAT_TIMEOUT),
            provider: provider.to_string(),
            model: model.to_string(),
            base,
            api_key,
            default_max_tokens: config
                .extra_u64("max_tokens")
                .map(|n| n as u32)
                .unwrap_or_else(|| default_max_tokens(model)),
        })
    }

    fn post(&self, body: &Value) -> RequestBuilder {
        self.http
            .client
            .post(join_url(&self.base, "messages"))
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", ANTHROPIC_VERSION)
            .json(body)
    }

    /// Sends the request; if the model rejects sampling parameters (newer
    /// models deprecate them), retries once without.
    async fn request(&self, messages: &[ChatMessage], options: &ChatOptions, stream: bool) -> AiResult<reqwest::Response> {
        let mut sampling = true;
        loop {
            let body = messages_body(&self.model, messages, options, self.default_max_tokens, stream, sampling);
            let request = self.post(&body);
            let result =
                if stream { self.http.send_streaming(request).await } else { self.http.send(request).await };
            match result {
                Err(AiError::InvalidRequest(message))
                    if sampling
                        && (options.temperature.is_some() || options.top_p.is_some())
                        && (message.contains("temperature") || message.contains("top_p")) =>
                {
                    tracing::debug!(model = %self.model, "retrying without sampling parameters");
                    sampling = false;
                }
                other => return other,
            }
        }
    }
}

#[async_trait]
impl LanguageModel for AnthropicChat {
    fn provider(&self) -> &str {
        &self.provider
    }

    fn model_name(&self) -> &str {
        &self.model
    }

    async fn chat(&self, messages: &[ChatMessage], options: &ChatOptions) -> AiResult<ChatResponse> {
        let response = self.request(messages, options, false).await?;
        let body = crate::http::read_body(&self.provider, response, crate::http::MAX_JSON_RESPONSE_BYTES).await?;
        let json: Value =
            serde_json::from_slice(&body).map_err(|_| self.http.malformed("the response was not valid JSON"))?;
        let content = json.get("content").ok_or_else(|| self.http.malformed("no content in the message"))?;
        Ok(ChatResponse {
            content: content_text(content),
            model: json.get("model").and_then(Value::as_str).unwrap_or(&self.model).to_string(),
            finish_reason: finish_reason(json.get("stop_reason")),
            usage: usage(json.pointer("/usage/input_tokens"), json.pointer("/usage/output_tokens")),
        })
    }

    async fn chat_stream(&self, messages: &[ChatMessage], options: &ChatOptions) -> AiResult<TextStream> {
        let response = self.request(messages, options, true).await?;
        let provider = self.provider.clone();
        let secret = self.api_key.clone();
        Ok(text_stream(response, Framing::Sse, self.http.timeout, &self.provider, move |frame: Frame| {
            let Ok(json) = serde_json::from_str::<Value>(&frame.data) else {
                return Step::Skip;
            };
            let kind = json.get("type").and_then(Value::as_str).or(frame.event.as_deref()).unwrap_or_default();
            match kind {
                "content_block_delta" => match json.pointer("/delta/text").and_then(Value::as_str) {
                    Some(text) if json.pointer("/delta/type").and_then(Value::as_str) != Some("thinking_delta") => {
                        Step::Text(text.to_string())
                    }
                    _ => Step::Skip,
                },
                "message_stop" => Step::Done,
                "error" => {
                    let error_type = json.pointer("/error/type").and_then(Value::as_str).unwrap_or_default();
                    Step::Fail(http_error(&provider, status_for_error_type(error_type), &frame.data, Some(&secret)))
                }
                _ => Step::Skip,
            }
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::http::shared_client;
    use crate::providers::testing::{collect_text, collect_until_error, header, request, request_json};
    use wiremock::matchers::{body_string_contains, method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    fn model(provider: &str, name: &str, server: &MockServer) -> AnthropicChat {
        let config = ProviderConfig::new().with_api_key("sk-ant-test").with_base_url(server.uri());
        AnthropicChat::new(&shared_client(), provider, name, &config).unwrap()
    }

    fn reply(text: &str) -> Value {
        json!({
            "id": "msg_1", "type": "message", "role": "assistant", "model": "claude-sonnet-4-5-20250929",
            "content": [{"type": "thinking", "thinking": "..."}, {"type": "text", "text": text}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 20, "output_tokens": 5},
        })
    }

    #[test]
    fn base_urls_normalise_to_v1() {
        for (input, expected) in [
            ("https://api.example.com", "https://api.example.com/v1"),
            ("https://api.example.com/", "https://api.example.com/v1"),
            ("https://api.example.com/v1", "https://api.example.com/v1"),
            ("https://api.example.com/v1/", "https://api.example.com/v1"),
            ("https://api.example.com/v1/models", "https://api.example.com/v1"),
            ("https://api.example.com/v1/models/", "https://api.example.com/v1"),
            (" https://host/anthropic/models ", "https://host/anthropic/v1"),
        ] {
            assert_eq!(normalize_anthropic_compatible_base_url(input), expected, "{input}");
        }
    }

    #[tokio::test]
    async fn chat_hoists_system_prompts_and_parses_text_blocks() {
        let server = MockServer::start().await;
        Mock::given(method("POST")).and(path("/v1/messages")).respond_with(ResponseTemplate::new(200).set_body_json(reply("Hello!"))).mount(&server).await;

        let model = model("anthropic", "claude-sonnet-4-5", &server);
        let messages = [
            ChatMessage::system("You are terse."),
            ChatMessage::user("Hi"),
            ChatMessage::assistant("Hello"),
            ChatMessage::system("Answer in French."),
            ChatMessage::assistant("  "),
            ChatMessage::user("Bye"),
        ];
        let options = ChatOptions { temperature: Some(0.3), top_p: Some(0.9), max_tokens: Some(256), json: true };
        let out = model.chat(&messages, &options).await.unwrap();

        assert_eq!(out.content, "Hello!");
        assert_eq!(out.model, "claude-sonnet-4-5-20250929");
        assert_eq!(out.finish_reason.as_deref(), Some("stop"));
        assert_eq!(out.usage, Some(nn_llm::Usage { prompt_tokens: 20, completion_tokens: 5 }));

        let sent = request(&server, 0).await;
        assert_eq!(header(&sent, "x-api-key"), Some("sk-ant-test"));
        assert_eq!(header(&sent, "anthropic-version"), Some("2023-06-01"));
        assert!(header(&sent, "authorization").is_none());
        assert_eq!(
            request_json(&server, 0).await,
            json!({
                "model": "claude-sonnet-4-5",
                "max_tokens": 256,
                "system": "You are terse.\n\nAnswer in French.",
                "messages": [
                    {"role": "user", "content": "Hi"},
                    {"role": "assistant", "content": "Hello"},
                    {"role": "user", "content": "Bye"},
                ],
                "temperature": 0.3,
            })
        );
    }

    #[test]
    fn max_tokens_always_has_a_value() {
        let body = |model: &str| messages_body(model, &[ChatMessage::user("x")], &ChatOptions::default(), default_max_tokens(model), false, true);
        assert_eq!(body("claude-sonnet-4-5")["max_tokens"], json!(8192));
        assert_eq!(body("claude-3-5-haiku-latest")["max_tokens"], json!(8192));
        assert_eq!(body("claude-3-haiku-20240307")["max_tokens"], json!(4096));
        let only_top_p = ChatOptions { top_p: Some(0.5), ..Default::default() };
        let body = messages_body("m", &[ChatMessage::user("x")], &only_top_p, 100, true, true);
        assert_eq!(body, json!({"model": "m", "max_tokens": 100, "messages": [{"role": "user", "content": "x"}], "top_p": 0.5, "stream": true}));
    }

    #[test]
    fn a_lone_system_prompt_becomes_the_user_turn() {
        let body = messages_body("m", &[ChatMessage::system("Summarise X")], &ChatOptions::default(), 10, false, true);
        assert_eq!(body, json!({"model": "m", "max_tokens": 10, "messages": [{"role": "user", "content": "Summarise X"}]}));
    }

    #[tokio::test]
    async fn rejected_sampling_parameters_are_dropped_on_retry() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(body_string_contains("temperature"))
            .respond_with(ResponseTemplate::new(400).set_body_json(json!({
                "type": "error",
                "error": {"type": "invalid_request_error", "message": "`temperature` is deprecated for this model."},
            })))
            .mount(&server)
            .await;
        Mock::given(method("POST")).respond_with(ResponseTemplate::new(200).set_body_json(reply("ok"))).mount(&server).await;

        let model = model("anthropic", "claude-opus-4-8", &server);
        let out = model.chat(&[ChatMessage::user("x")], &ChatOptions::default().with_temperature(0.2)).await.unwrap();
        assert_eq!(out.content, "ok");
        assert!(request_json(&server, 1).await.get("temperature").is_none());
    }

    #[tokio::test]
    async fn stream_yields_text_deltas_and_ignores_other_events() {
        let server = MockServer::start().await;
        let sse = concat!(
            "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_1\",\"usage\":{\"input_tokens\":3}}}\n\n",
            "event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"thinking\",\"thinking\":\"\"}}\n\n",
            "event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"thinking_delta\",\"thinking\":\"hmm\"}}\n\n",
            "event: ping\ndata: {\"type\": \"ping\"}\n\n",
            "event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":1,\"delta\":{\"type\":\"text_delta\",\"text\":\"Bon\"}}\n\n",
            "event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":1,\"delta\":{\"type\":\"text_delta\",\"text\":\"jour\"}}\n\n",
            "event: content_block_stop\ndata: {\"type\":\"content_block_stop\",\"index\":1}\n\n",
            "event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"output_tokens\":2}}\n\n",
            "event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n",
        );
        Mock::given(method("POST"))
            .and(path("/v1/messages"))
            .respond_with(ResponseTemplate::new(200).insert_header("content-type", "text/event-stream").set_body_string(sse))
            .mount(&server)
            .await;
        let model = model("anthropic", "claude-haiku-4-5", &server);
        let stream = model.chat_stream(&[ChatMessage::user("Salut")], &ChatOptions::default()).await.unwrap();
        assert_eq!(collect_text(stream).await, vec!["Bon", "jour"]);
        assert_eq!(request_json(&server, 0).await["stream"], json!(true));
    }

    #[tokio::test]
    async fn stream_error_events_end_the_stream_with_a_typed_error() {
        let server = MockServer::start().await;
        let sse = concat!(
            "event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"par\"}}\n\n",
            "event: error\ndata: {\"type\":\"error\",\"error\":{\"type\":\"overloaded_error\",\"message\":\"Overloaded\"}}\n\n",
        );
        Mock::given(method("POST")).respond_with(ResponseTemplate::new(200).set_body_string(sse)).mount(&server).await;
        let model = model("anthropic", "claude-haiku-4-5", &server);
        let stream = model.chat_stream(&[ChatMessage::user("x")], &ChatOptions::default()).await.unwrap();
        let (text, err) = collect_until_error(stream).await;
        assert_eq!(text, vec!["par"]);
        let err = err.unwrap();
        assert!(matches!(err, AiError::Provider { status: Some(529), .. }), "{err:?}");
        assert!(err.is_retryable());
    }

    #[tokio::test]
    async fn missing_models_and_bad_keys_are_classified() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(body_string_contains("retired"))
            .respond_with(ResponseTemplate::new(404).set_body_json(json!({
                "type": "error", "error": {"type": "not_found_error", "message": "model: retired"}, "request_id": "req_011Cga403VtXy92ABc",
            })))
            .mount(&server)
            .await;
        Mock::given(method("POST"))
            .respond_with(ResponseTemplate::new(401).set_body_json(json!({
                "type": "error", "error": {"type": "authentication_error", "message": "invalid x-api-key"},
            })))
            .mount(&server)
            .await;
        let err = model("anthropic", "retired", &server).chat(&[ChatMessage::user("x")], &ChatOptions::default()).await.unwrap_err();
        assert!(matches!(err, AiError::ModelNotFound(_)), "{err:?}");
        let err = model("anthropic", "claude", &server).chat(&[ChatMessage::user("x")], &ChatOptions::default()).await.unwrap_err();
        assert!(matches!(err, AiError::Authentication(_)), "{err:?}");
    }

    #[test]
    fn compatible_endpoints_need_both_url_and_key() {
        let client = shared_client();
        for config in [
            ProviderConfig::new().with_api_key("k"),
            ProviderConfig::new().with_base_url("https://x.test"),
            ProviderConfig::new(),
        ] {
            let err = AnthropicChat::new(&client, "anthropic_compatible", "m", &config).err().unwrap();
            assert!(matches!(&err, AiError::Configuration(m) if m == "Anthropic-compatible models require a base URL and API key"));
        }
        let ok = ProviderConfig::new().with_api_key("k").with_base_url("https://x.test/anthropic/v1/models");
        let model = AnthropicChat::new(&client, "anthropic_compatible", "m", &ok).unwrap();
        assert_eq!(model.base, "https://x.test/anthropic/v1");
        assert_eq!(model.provider(), "anthropic_compatible");

        let err = AnthropicChat::new(&client, "anthropic", "m", &ProviderConfig::new()).err().unwrap();
        assert!(matches!(err, AiError::Configuration(_)));
        let official = AnthropicChat::new(&client, "anthropic", "m", &ProviderConfig::new().with_api_key("k")).unwrap();
        assert_eq!(official.base, "https://api.anthropic.com/v1");
    }
}
