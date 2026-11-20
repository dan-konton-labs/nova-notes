//! Cohere's native v2 API: `/v2/chat` and `/v2/embed`.

use async_trait::async_trait;
use nn_llm::{AiResult, ChatMessage, ChatOptions, ChatResponse, LanguageModel, TextStream};
use reqwest::RequestBuilder;
use serde_json::{json, Map, Value};

use super::{content_text, finish_reason, float, require_key, usage, vectors, EmbedBatch};
use crate::config::{ProviderConfig, DEFAULT_CHAT_TIMEOUT, DEFAULT_EMBEDDING_TIMEOUT};
use crate::error::http_error;
use crate::http::{join_url, Http};
use crate::sse::{text_stream, Frame, Framing, Step};

pub(crate) const COHERE_API_ROOT: &str = "https://api.cohere.com";

/// Strips a version suffix so both `https://host` and `https://host/v2` work
/// as base URLs.
fn api_root(config: &ProviderConfig) -> String {
    let url = config.url().unwrap_or(COHERE_API_ROOT);
    url.strip_suffix("/v2").or_else(|| url.strip_suffix("/v1")).unwrap_or(url).to_string()
}

struct CohereEndpoint {
    http: Http,
    model: String,
    root: String,
    api_key: String,
}

impl CohereEndpoint {
    fn new(client: &reqwest::Client, model: &str, config: &ProviderConfig, timeout: std::time::Duration) -> AiResult<Self> {
        Ok(Self {
            http: Http::new(client, "cohere", config, timeout),
            model: model.to_string(),
            root: api_root(config),
            api_key: require_key("cohere", config)?,
        })
    }

    fn post(&self, path: &str, body: &Value) -> RequestBuilder {
        self.http.client.post(join_url(&self.root, path)).bearer_auth(&self.api_key).json(body)
    }
}

pub(crate) fn chat_body(model: &str, messages: &[ChatMessage], options: &ChatOptions, stream: bool) -> Value {
    let mut body = Map::new();
    body.insert("model".into(), json!(model));
    body.insert(
        "messages".into(),
        messages.iter().map(|m| json!({"role": m.role.as_str(), "content": m.content})).collect(),
    );
    if let Some(temperature) = options.temperature {
        body.insert("temperature".into(), float(temperature));
    }
    if let Some(top_p) = options.top_p {
        body.insert("p".into(), float(top_p));
    }
    if let Some(max_tokens) = options.max_tokens {
        body.insert("max_tokens".into(), json!(max_tokens));
    }
    if options.json {
        body.insert("response_format".into(), json!({"type": "json_object"}));
    }
    if stream {
        body.insert("stream".into(), json!(true));
    }
    Value::Object(body)
}

pub(crate) struct CohereChat {
    endpoint: CohereEndpoint,
}

impl CohereChat {
    pub fn new(client: &reqwest::Client, model: &str, config: &ProviderConfig) -> AiResult<Self> {
        Ok(Self { endpoint: CohereEndpoint::new(client, model, config, DEFAULT_CHAT_TIMEOUT)? })
    }
}

#[async_trait]
impl LanguageModel for CohereChat {
    fn provider(&self) -> &str {
        "cohere"
    }

    fn model_name(&self) -> &str {
        &self.endpoint.model
    }

    async fn chat(&self, messages: &[ChatMessage], options: &ChatOptions) -> AiResult<ChatResponse> {
        let ep = &self.endpoint;
        let json = ep.http.json(ep.post("v2/chat", &chat_body(&ep.model, messages, options, false))).await?;
        let content =
            json.pointer("/message/content").ok_or_else(|| ep.http.malformed("no message content in the reply"))?;
        // Billed units leave out cached and system tokens; prefer raw counts.
        let tokens = |kind: &str| {
            json.pointer(&format!("/usage/tokens/{kind}")).or_else(|| json.pointer(&format!("/usage/billed_units/{kind}")))
        };
        Ok(ChatResponse {
            content: content_text(content),
            model: ep.model.clone(),
            finish_reason: finish_reason(json.get("finish_reason")),
            usage: usage(tokens("input_tokens"), tokens("output_tokens")),
        })
    }

    async fn chat_stream(&self, messages: &[ChatMessage], options: &ChatOptions) -> AiResult<TextStream> {
        let ep = &self.endpoint;
        let request = ep.post("v2/chat", &chat_body(&ep.model, messages, options, true));
        let response = ep.http.send_streaming(request).await?;
        let secret = ep.api_key.clone();
        Ok(text_stream(response, Framing::Sse, ep.http.timeout, "cohere", move |frame: Frame| {
            let Ok(json) = serde_json::from_str::<Value>(&frame.data) else {
                return Step::Skip;
            };
            let kind = json.get("type").and_then(Value::as_str).or(frame.event.as_deref()).unwrap_or_default();
            match kind {
                "content-delta" => match json.pointer("/delta/message/content/text").and_then(Value::as_str) {
                    Some(text) => Step::Text(text.to_string()),
                    None => Step::Skip,
                },
                "message-end" => match json.pointer("/delta/error").and_then(Value::as_str) {
                    Some(_) => Step::Fail(http_error("cohere", 500, &json["delta"].to_string(), Some(&secret))),
                    None => Step::Done,
                },
                _ => Step::Skip,
            }
        }))
    }
}

pub(crate) struct CohereEmbedder {
    endpoint: CohereEndpoint,
    input_type: String,
    dimensions: Option<u64>,
}

impl CohereEmbedder {
    pub fn new(client: &reqwest::Client, model: &str, config: &ProviderConfig) -> AiResult<Self> {
        Ok(Self {
            endpoint: CohereEndpoint::new(client, model, config, DEFAULT_EMBEDDING_TIMEOUT)?,
            input_type: config.extra_str("input_type").unwrap_or("search_document").to_string(),
            dimensions: config.extra_u64("dimensions"),
        })
    }
}

#[async_trait]
impl EmbedBatch for CohereEmbedder {
    fn provider(&self) -> &str {
        "cohere"
    }

    fn model_name(&self) -> &str {
        &self.endpoint.model
    }

    fn batch_limit(&self) -> usize {
        96
    }

    async fn embed_batch(&self, texts: &[String]) -> AiResult<Vec<Vec<f32>>> {
        let ep = &self.endpoint;
        let mut body = json!({
            "model": ep.model,
            "texts": texts,
            "input_type": self.input_type,
            "embedding_types": ["float"],
        });
        if let Some(dimensions) = self.dimensions {
            body["output_dimension"] = json!(dimensions);
        }
        let json = ep.http.json(ep.post("v2/embed", &body)).await?;
        vectors(json.pointer("/embeddings/float")).ok_or_else(|| ep.http.malformed("no embeddings in the response"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::http::shared_client;
    use crate::providers::testing::{collect_text, header, request, request_json};
    use nn_llm::AiError;
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    fn config(server: &MockServer) -> ProviderConfig {
        ProviderConfig::new().with_api_key("co-test").with_base_url(server.uri())
    }

    #[test]
    fn base_urls_with_or_without_a_version_resolve_to_the_root() {
        assert_eq!(api_root(&ProviderConfig::new()), "https://api.cohere.com");
        assert_eq!(api_root(&ProviderConfig::new().with_base_url("https://proxy.test/v2/")), "https://proxy.test");
        assert_eq!(api_root(&ProviderConfig::new().with_base_url("https://proxy.test/cohere")), "https://proxy.test/cohere");
    }

    #[tokio::test]
    async fn chat_uses_the_v2_shape() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v2/chat"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "id": "c1",
                "finish_reason": "COMPLETE",
                "message": {"role": "assistant", "content": [{"type": "text", "text": "Hi there"}]},
                "usage": {"billed_units": {"input_tokens": 3, "output_tokens": 2}, "tokens": {"input_tokens": 70, "output_tokens": 4}},
            })))
            .mount(&server)
            .await;
        let model = CohereChat::new(&shared_client(), "command-a-03-2025", &config(&server)).unwrap();
        let options = ChatOptions { temperature: Some(0.3), top_p: Some(0.75), max_tokens: Some(50), json: true };
        let out = model.chat(&[ChatMessage::system("Be kind."), ChatMessage::user("Hi")], &options).await.unwrap();

        assert_eq!(out.content, "Hi there");
        assert_eq!(out.finish_reason.as_deref(), Some("stop"));
        assert_eq!(out.usage, Some(nn_llm::Usage { prompt_tokens: 70, completion_tokens: 4 }));
        assert_eq!(header(&request(&server, 0).await, "authorization"), Some("Bearer co-test"));
        assert_eq!(
            request_json(&server, 0).await,
            json!({
                "model": "command-a-03-2025",
                "messages": [{"role": "system", "content": "Be kind."}, {"role": "user", "content": "Hi"}],
                "temperature": 0.3,
                "p": 0.75,
                "max_tokens": 50,
                "response_format": {"type": "json_object"},
            })
        );
    }

    #[tokio::test]
    async fn stream_reads_content_delta_events() {
        let server = MockServer::start().await;
        let sse = concat!(
            "event: message-start\ndata: {\"type\":\"message-start\",\"id\":\"c1\"}\n\n",
            "event: content-start\ndata: {\"type\":\"content-start\",\"index\":0,\"delta\":{\"message\":{\"content\":{\"type\":\"text\",\"text\":\"\"}}}}\n\n",
            "event: content-delta\ndata: {\"type\":\"content-delta\",\"index\":0,\"delta\":{\"message\":{\"content\":{\"text\":\"Bon\"}}}}\n\n",
            "event: content-delta\ndata: {\"type\":\"content-delta\",\"index\":0,\"delta\":{\"message\":{\"content\":{\"text\":\"jour\"}}}}\n\n",
            "event: content-end\ndata: {\"type\":\"content-end\",\"index\":0}\n\n",
            "event: message-end\ndata: {\"type\":\"message-end\",\"delta\":{\"finish_reason\":\"COMPLETE\"}}\n\n",
            "data: {\"type\":\"content-delta\",\"delta\":{\"message\":{\"content\":{\"text\":\"late\"}}}}\n\n",
        );
        Mock::given(method("POST")).and(path("/v2/chat")).respond_with(ResponseTemplate::new(200).set_body_string(sse)).mount(&server).await;
        let model = CohereChat::new(&shared_client(), "command-a-03-2025", &config(&server)).unwrap();
        let stream = model.chat_stream(&[ChatMessage::user("Salut")], &ChatOptions::default()).await.unwrap();
        assert_eq!(collect_text(stream).await, vec!["Bon", "jour"]);
        assert_eq!(request_json(&server, 0).await["stream"], json!(true));
    }

    #[tokio::test]
    async fn embeddings_request_float_vectors() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v2/embed"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "id": "e1", "embeddings": {"float": [[0.5, 1.0], [1.5, 2.0]]}, "texts": ["a", "b"],
            })))
            .mount(&server)
            .await;
        let embedder = CohereEmbedder::new(&shared_client(), "embed-v4.0", &config(&server)).unwrap();
        assert_eq!(embedder.batch_limit(), 96);
        let out = embedder.embed_batch(&["a".to_string(), "b".to_string()]).await.unwrap();
        assert_eq!(out, vec![vec![0.5, 1.0], vec![1.5, 2.0]]);
        assert_eq!(
            request_json(&server, 0).await,
            json!({"model": "embed-v4.0", "texts": ["a", "b"], "input_type": "search_document", "embedding_types": ["float"]})
        );
    }

    #[tokio::test]
    async fn errors_use_cohere_message_field() {
        let server = MockServer::start().await;
        Mock::given(method("POST")).respond_with(ResponseTemplate::new(401).set_body_json(json!({"message": "invalid api token"}))).mount(&server).await;
        let model = CohereChat::new(&shared_client(), "command-a-03-2025", &config(&server)).unwrap();
        let err = model.chat(&[ChatMessage::user("x")], &ChatOptions::default()).await.unwrap_err();
        assert!(matches!(&err, AiError::Authentication(m) if m == "cohere: invalid api token (HTTP 401)"), "{err:?}");
        assert!(matches!(CohereChat::new(&shared_client(), "m", &ProviderConfig::new()).err(), Some(AiError::Configuration(_))));
    }
}
