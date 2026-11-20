//! Ollama's native API: `/api/chat` (NDJSON streaming) and `/api/embed`.

use async_trait::async_trait;
use nn_llm::{AiError, AiResult, ChatMessage, ChatOptions, ChatResponse, LanguageModel, TextStream};
use reqwest::RequestBuilder;
use serde_json::{json, Map, Value};

use super::{finish_reason, float, usage, vectors, EmbedBatch};
use crate::config::{ProviderConfig, DEFAULT_CHAT_TIMEOUT, DEFAULT_EMBEDDING_TIMEOUT};
use crate::error::http_error;
use crate::http::{join_url, Http};
use crate::sse::{text_stream, Frame, Framing, Step};

pub(crate) const OLLAMA_DEFAULT_BASE_URL: &str = "http://localhost:11434";
/// Ollama's own default context window is far smaller; the original raises
/// it to 8192 unless the credential says otherwise.
const DEFAULT_NUM_CTX: u64 = 8192;

struct OllamaEndpoint {
    http: Http,
    model: String,
    base: String,
    /// Only needed behind an authenticating proxy or for Ollama's cloud.
    api_key: Option<String>,
}

impl OllamaEndpoint {
    fn new(client: &reqwest::Client, model: &str, config: &ProviderConfig, timeout: std::time::Duration) -> Self {
        let base = config.url().unwrap_or(OLLAMA_DEFAULT_BASE_URL);
        // People paste the OpenAI-compatible URL; the native API has no /v1.
        let base = base.strip_suffix("/v1").or_else(|| base.strip_suffix("/api")).unwrap_or(base);
        Self {
            http: Http::new(client, "ollama", config, timeout),
            model: model.to_string(),
            base: base.to_string(),
            api_key: config.key().map(str::to_string),
        }
    }

    fn post(&self, path: &str, body: &Value) -> RequestBuilder {
        let request = self.http.client.post(join_url(&self.base, path)).json(body);
        match &self.api_key {
            Some(key) => request.bearer_auth(key),
            None => request,
        }
    }
}

pub(crate) struct OllamaChat {
    endpoint: OllamaEndpoint,
    num_ctx: u64,
    keep_alive: Option<Value>,
}

impl OllamaChat {
    pub fn new(client: &reqwest::Client, model: &str, config: &ProviderConfig) -> Self {
        Self {
            endpoint: OllamaEndpoint::new(client, model, config, DEFAULT_CHAT_TIMEOUT),
            num_ctx: config.extra_u64("num_ctx").filter(|n| *n > 0).unwrap_or(DEFAULT_NUM_CTX),
            keep_alive: config.extra.get("keep_alive").filter(|v| !v.is_null()).cloned(),
        }
    }

    fn body(&self, messages: &[ChatMessage], options: &ChatOptions, stream: bool) -> Value {
        let mut model_options = Map::new();
        model_options.insert("num_ctx".into(), json!(self.num_ctx));
        if let Some(temperature) = options.temperature {
            model_options.insert("temperature".into(), float(temperature));
        }
        if let Some(top_p) = options.top_p {
            model_options.insert("top_p".into(), float(top_p));
        }
        if let Some(max_tokens) = options.max_tokens {
            model_options.insert("num_predict".into(), json!(max_tokens));
        }

        let mut body = Map::new();
        body.insert("model".into(), json!(self.endpoint.model));
        body.insert(
            "messages".into(),
            messages.iter().map(|m| json!({"role": m.role.as_str(), "content": m.content})).collect(),
        );
        body.insert("stream".into(), json!(stream));
        body.insert("options".into(), Value::Object(model_options));
        if options.json {
            body.insert("format".into(), json!("json"));
        }
        if let Some(keep_alive) = &self.keep_alive {
            body.insert("keep_alive".into(), keep_alive.clone());
        }
        Value::Object(body)
    }
}

#[async_trait]
impl LanguageModel for OllamaChat {
    fn provider(&self) -> &str {
        "ollama"
    }

    fn model_name(&self) -> &str {
        &self.endpoint.model
    }

    async fn chat(&self, messages: &[ChatMessage], options: &ChatOptions) -> AiResult<ChatResponse> {
        let ep = &self.endpoint;
        let json = ep.http.json(ep.post("api/chat", &self.body(messages, options, false))).await?;
        if let Some(error) = json.get("error").and_then(Value::as_str) {
            return Err(AiError::Provider { status: None, message: format!("ollama: {error}") });
        }
        let content = json
            .pointer("/message/content")
            .and_then(Value::as_str)
            .ok_or_else(|| ep.http.malformed("no message content in the reply"))?;
        Ok(ChatResponse {
            content: content.to_string(),
            model: json.get("model").and_then(Value::as_str).unwrap_or(&ep.model).to_string(),
            finish_reason: finish_reason(json.get("done_reason")),
            usage: usage(json.get("prompt_eval_count"), json.get("eval_count")),
        })
    }

    async fn chat_stream(&self, messages: &[ChatMessage], options: &ChatOptions) -> AiResult<TextStream> {
        let ep = &self.endpoint;
        let response = ep.http.send_streaming(ep.post("api/chat", &self.body(messages, options, true))).await?;
        Ok(text_stream(response, Framing::NdJson, ep.http.timeout, "ollama", |frame: Frame| {
            let Ok(json) = serde_json::from_str::<Value>(&frame.data) else {
                return Step::Skip;
            };
            if json.get("error").is_some() {
                return Step::Fail(http_error("ollama", 500, &frame.data, None));
            }
            let text = json.pointer("/message/content").and_then(Value::as_str).unwrap_or_default();
            if !text.is_empty() {
                // A final line may carry both the last text and done=true; the
                // body ends right after it either way.
                Step::Text(text.to_string())
            } else if json.get("done").and_then(Value::as_bool) == Some(true) {
                Step::Done
            } else {
                Step::Skip
            }
        }))
    }
}

pub(crate) struct OllamaEmbedder {
    endpoint: OllamaEndpoint,
}

impl OllamaEmbedder {
    pub fn new(client: &reqwest::Client, model: &str, config: &ProviderConfig) -> Self {
        Self { endpoint: OllamaEndpoint::new(client, model, config, DEFAULT_EMBEDDING_TIMEOUT) }
    }
}

#[async_trait]
impl EmbedBatch for OllamaEmbedder {
    fn provider(&self) -> &str {
        "ollama"
    }

    fn model_name(&self) -> &str {
        &self.endpoint.model
    }

    fn batch_limit(&self) -> usize {
        // No documented cap; modest batches keep a CPU-bound server
        // inside the request timeout.
        32
    }

    async fn embed_batch(&self, texts: &[String]) -> AiResult<Vec<Vec<f32>>> {
        let ep = &self.endpoint;
        let json = ep.http.json(ep.post("api/embed", &json!({"model": ep.model, "input": texts}))).await?;
        vectors(json.get("embeddings")).ok_or_else(|| ep.http.malformed("no embeddings in the response"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::http::shared_client;
    use crate::providers::testing::{collect_text, collect_until_error, header, request, request_json};
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    fn config(server: &MockServer) -> ProviderConfig {
        ProviderConfig::new().with_base_url(server.uri())
    }

    #[test]
    fn base_url_defaults_and_tolerates_pasted_suffixes() {
        let client = shared_client();
        assert_eq!(OllamaChat::new(&client, "m", &ProviderConfig::new()).endpoint.base, "http://localhost:11434");
        let v1 = ProviderConfig::new().with_base_url("http://box:11434/v1/");
        assert_eq!(OllamaChat::new(&client, "m", &v1).endpoint.base, "http://box:11434");
    }

    #[tokio::test]
    async fn chat_sends_options_and_parses_the_reply() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/api/chat"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "model": "llama3.2:latest",
                "message": {"role": "assistant", "content": "Hi!"},
                "done": true, "done_reason": "stop", "prompt_eval_count": 11, "eval_count": 3,
            })))
            .mount(&server)
            .await;
        let config = config(&server).with_extra("num_ctx", 32768).with_extra("keep_alive", "10m");
        let model = OllamaChat::new(&shared_client(), "llama3.2", &config);
        let options = ChatOptions { temperature: Some(0.1), top_p: Some(0.5), max_tokens: Some(200), json: true };
        let out = model.chat(&[ChatMessage::system("Short."), ChatMessage::user("Hi")], &options).await.unwrap();

        assert_eq!(out.content, "Hi!");
        assert_eq!(out.model, "llama3.2:latest");
        assert_eq!(out.finish_reason.as_deref(), Some("stop"));
        assert_eq!(out.usage, Some(nn_llm::Usage { prompt_tokens: 11, completion_tokens: 3 }));
        assert!(header(&request(&server, 0).await, "authorization").is_none());
        assert_eq!(
            request_json(&server, 0).await,
            json!({
                "model": "llama3.2",
                "messages": [{"role": "system", "content": "Short."}, {"role": "user", "content": "Hi"}],
                "stream": false,
                "options": {"num_ctx": 32768, "temperature": 0.1, "top_p": 0.5, "num_predict": 200},
                "format": "json",
                "keep_alive": "10m",
            })
        );
    }

    #[tokio::test]
    async fn default_context_window_is_8192() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({"message": {"content": "x"}, "done": true})))
            .mount(&server)
            .await;
        let model = OllamaChat::new(&shared_client(), "m", &config(&server));
        model.chat(&[ChatMessage::user("x")], &ChatOptions::default()).await.unwrap();
        assert_eq!(request_json(&server, 0).await["options"], json!({"num_ctx": 8192}));
    }

    #[tokio::test]
    async fn stream_reads_ndjson_lines() {
        let server = MockServer::start().await;
        let ndjson = concat!(
            "{\"model\":\"m\",\"message\":{\"role\":\"assistant\",\"content\":\"Hel\"},\"done\":false}\n",
            "{\"model\":\"m\",\"message\":{\"role\":\"assistant\",\"content\":\"\",\"thinking\":\"hmm\"},\"done\":false}\n",
            "{\"model\":\"m\",\"message\":{\"role\":\"assistant\",\"content\":\"lo\"},\"done\":false}\n",
            "{\"model\":\"m\",\"message\":{\"role\":\"assistant\",\"content\":\"\"},\"done\":true,\"done_reason\":\"stop\",\"eval_count\":2}\n",
        );
        Mock::given(method("POST"))
            .and(path("/api/chat"))
            .respond_with(ResponseTemplate::new(200).insert_header("content-type", "application/x-ndjson").set_body_string(ndjson))
            .mount(&server)
            .await;
        let model = OllamaChat::new(&shared_client(), "m", &config(&server));
        let stream = model.chat_stream(&[ChatMessage::user("Hi")], &ChatOptions::default()).await.unwrap();
        assert_eq!(collect_text(stream).await, vec!["Hel", "lo"]);
        assert_eq!(request_json(&server, 0).await["stream"], json!(true));
    }

    #[tokio::test]
    async fn errors_are_classified_before_and_during_a_stream() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/api/chat"))
            .and(wiremock::matchers::body_string_contains("\"missing\""))
            .respond_with(ResponseTemplate::new(404).set_body_json(json!({"error": "model 'missing' not found"})))
            .mount(&server)
            .await;
        Mock::given(method("POST"))
            .respond_with(ResponseTemplate::new(200).set_body_string("{\"message\":{\"content\":\"a\"},\"done\":false}\n{\"error\":\"llama runner process has terminated\"}\n"))
            .mount(&server)
            .await;
        let user = [ChatMessage::user("x")];
        let err = OllamaChat::new(&shared_client(), "missing", &config(&server)).chat(&user, &ChatOptions::default()).await.unwrap_err();
        assert!(matches!(&err, AiError::ModelNotFound(m) if m.contains("model 'missing' not found")), "{err:?}");

        let stream = OllamaChat::new(&shared_client(), "m", &config(&server)).chat_stream(&user, &ChatOptions::default()).await.unwrap();
        let (text, err) = collect_until_error(stream).await;
        assert_eq!(text, vec!["a"]);
        assert!(err.unwrap().to_string().contains("llama runner process has terminated"));
    }

    #[tokio::test]
    async fn embeddings_use_api_embed_and_send_the_key_when_present() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/api/embed"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({"model": "nomic-embed-text", "embeddings": [[0.5], [1.5]]})))
            .mount(&server)
            .await;
        let config = config(&server).with_api_key("proxy-key");
        let embedder = OllamaEmbedder::new(&shared_client(), "nomic-embed-text", &config);
        let out = embedder.embed_batch(&["a".to_string(), "b".to_string()]).await.unwrap();
        assert_eq!(out, vec![vec![0.5], vec![1.5]]);
        assert_eq!(request_json(&server, 0).await, json!({"model": "nomic-embed-text", "input": ["a", "b"]}));
        assert_eq!(header(&request(&server, 0).await, "authorization"), Some("Bearer proxy-key"));
    }
}
