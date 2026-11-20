//! Provider implementations, one module per wire format.
//!
//! Each module exposes constructors that validate the configuration up front
//! and return a type implementing the matching `nn-llm` trait. The pieces
//! every wire format needs (embedding batching, JSON field access, required
//! settings) live here.

use async_trait::async_trait;
use nn_llm::{AiError, AiResult, EmbeddingModel, Usage};
use serde_json::Value;

use crate::config::ProviderConfig;

pub(crate) mod anthropic;
pub(crate) mod cohere;
pub(crate) mod deepgram;
pub(crate) mod elevenlabs;
pub(crate) mod google;
pub(crate) mod minimax;
pub(crate) mod ollama;
pub(crate) mod openai;
pub(crate) mod vertex_auth;
pub(crate) mod voyage;

/// The API key, or a configuration error naming the provider.
pub(crate) fn require_key(provider: &str, config: &ProviderConfig) -> AiResult<String> {
    config
        .key()
        .map(str::to_string)
        .ok_or_else(|| AiError::Configuration(format!("No API key configured for {provider}")))
}

/// One request's worth of embedding work. [`Batched`] turns it into an
/// [`EmbeddingModel`] that respects the provider's batch limit.
#[async_trait]
pub(crate) trait EmbedBatch: Send + Sync {
    fn provider(&self) -> &str;
    fn model_name(&self) -> &str;
    /// Largest number of inputs the provider accepts in one request.
    fn batch_limit(&self) -> usize;
    /// One vector per input, in input order.
    async fn embed_batch(&self, texts: &[String]) -> AiResult<Vec<Vec<f32>>>;
}

/// Splits inputs into provider-sized batches, sent one after another so the
/// output order always matches the input order.
pub(crate) struct Batched<T> {
    inner: T,
    limit: usize,
}

impl<T: EmbedBatch> Batched<T> {
    /// `embedding_batch_size` in the credential's config overrides the
    /// provider limit (useful for small local servers).
    pub fn new(inner: T, config: &ProviderConfig) -> Self {
        let limit = config
            .extra_u64("embedding_batch_size")
            .map(|n| n as usize)
            .unwrap_or_else(|| inner.batch_limit())
            .max(1);
        Self { inner, limit }
    }
}

#[async_trait]
impl<T: EmbedBatch> EmbeddingModel for Batched<T> {
    fn provider(&self) -> &str {
        self.inner.provider()
    }

    fn model_name(&self) -> &str {
        self.inner.model_name()
    }

    async fn embed(&self, texts: &[String]) -> AiResult<Vec<Vec<f32>>> {
        let mut vectors = Vec::with_capacity(texts.len());
        for batch in texts.chunks(self.limit) {
            let embedded = self.inner.embed_batch(batch).await?;
            if embedded.len() != batch.len() {
                return Err(AiError::Provider {
                    status: None,
                    message: format!(
                        "{}: unexpected response: {} embeddings returned for {} inputs",
                        self.inner.provider(),
                        embedded.len(),
                        batch.len()
                    ),
                });
            }
            vectors.extend(embedded);
        }
        Ok(vectors)
    }
}

/// A JSON number that prints the way the `f32` does (`0.7`, not the widened
/// `0.699999988079071`).
pub(crate) fn float(value: f32) -> Value {
    value.to_string().parse::<f64>().map(Value::from).unwrap_or(Value::Null)
}

/// Parses a JSON array of numbers into an embedding vector.
pub(crate) fn vector(value: &Value) -> Option<Vec<f32>> {
    value.as_array()?.iter().map(|n| n.as_f64().map(|f| f as f32)).collect()
}

/// Parses a JSON array of embedding vectors.
pub(crate) fn vectors(value: Option<&Value>) -> Option<Vec<Vec<f32>>> {
    value?.as_array()?.iter().map(vector).collect()
}

/// Text of a message `content` field that is either a string or a list of
/// typed parts (`[{"type": "text", "text": "..."}]`). Non-text parts such as
/// thinking blocks are skipped.
pub(crate) fn content_text(content: &Value) -> String {
    match content {
        Value::String(text) => text.clone(),
        Value::Array(parts) => parts
            .iter()
            .filter(|part| part.get("type").and_then(Value::as_str).is_none_or(|t| t == "text"))
            .filter_map(|part| part.get("text").and_then(Value::as_str))
            .collect(),
        _ => String::new(),
    }
}

pub(crate) fn usage(prompt: Option<&Value>, completion: Option<&Value>) -> Option<Usage> {
    let count = |v: Option<&Value>| v.and_then(Value::as_u64).map(|n| n as u32);
    match (count(prompt), count(completion)) {
        (None, None) => None,
        (prompt, completion) => {
            Some(Usage { prompt_tokens: prompt.unwrap_or(0), completion_tokens: completion.unwrap_or(0) })
        }
    }
}

/// Lowercases a provider finish reason and maps the common spellings onto
/// the OpenAI vocabulary (`stop`, `length`, `content_filter`).
pub(crate) fn finish_reason(raw: Option<&Value>) -> Option<String> {
    let raw = raw?.as_str()?.trim().to_ascii_lowercase();
    if raw.is_empty() {
        return None;
    }
    let mapped = match raw.as_str() {
        "end_turn" | "stop_sequence" | "complete" | "stop" => "stop",
        "max_tokens" | "length" => "length",
        "safety" | "recitation" | "blocklist" | "prohibited_content" | "spii" | "refusal" | "content_filter"
        | "error_toxic" => "content_filter",
        other => other,
    };
    Some(mapped.to_string())
}

#[cfg(test)]
pub(crate) mod testing {
    //! Helpers shared by the provider tests.

    use futures::StreamExt;
    use nn_llm::TextStream;
    use serde_json::Value;
    use wiremock::{MockServer, Request};

    /// The JSON body of the `index`-th request the mock server received.
    pub async fn request_json(server: &MockServer, index: usize) -> Value {
        let requests = server.received_requests().await.expect("request recording is on");
        serde_json::from_slice(&requests[index].body).expect("request body is JSON")
    }

    pub async fn request(server: &MockServer, index: usize) -> Request {
        server.received_requests().await.expect("request recording is on")[index].clone()
    }

    pub fn header<'a>(request: &'a Request, name: &str) -> Option<&'a str> {
        request.headers.get(name).and_then(|v| v.to_str().ok())
    }

    /// Collects a text stream, panicking on the first error.
    pub async fn collect_text(mut stream: TextStream) -> Vec<String> {
        let mut out = Vec::new();
        while let Some(item) = stream.next().await {
            out.push(item.expect("stream item"));
        }
        out
    }

    /// Collects a text stream into its deltas and the error that ended it.
    pub async fn collect_until_error(mut stream: TextStream) -> (Vec<String>, Option<nn_llm::AiError>) {
        let mut out = Vec::new();
        while let Some(item) = stream.next().await {
            match item {
                Ok(text) => out.push(text),
                Err(err) => return (out, Some(err)),
            }
        }
        (out, None)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    struct FakeEmbedder {
        limit: usize,
        calls: Mutex<Vec<usize>>,
        drop_one: bool,
    }

    #[async_trait]
    impl EmbedBatch for FakeEmbedder {
        fn provider(&self) -> &str {
            "fake"
        }
        fn model_name(&self) -> &str {
            "fake-embed"
        }
        fn batch_limit(&self) -> usize {
            self.limit
        }
        async fn embed_batch(&self, texts: &[String]) -> AiResult<Vec<Vec<f32>>> {
            self.calls.lock().unwrap().push(texts.len());
            let mut out: Vec<Vec<f32>> = texts.iter().map(|t| vec![t.parse::<f32>().unwrap()]).collect();
            if self.drop_one {
                out.pop();
            }
            Ok(out)
        }
    }

    fn inputs(n: usize) -> Vec<String> {
        (0..n).map(|i| i.to_string()).collect()
    }

    #[tokio::test]
    async fn batches_respect_the_limit_and_preserve_order() {
        let fake = FakeEmbedder { limit: 3, calls: Mutex::new(Vec::new()), drop_one: false };
        let model = Batched::new(fake, &ProviderConfig::default());
        let out = model.embed(&inputs(8)).await.unwrap();
        assert_eq!(out, (0..8).map(|i| vec![i as f32]).collect::<Vec<_>>());
        assert_eq!(*model.inner.calls.lock().unwrap(), vec![3, 3, 2]);
    }

    #[tokio::test]
    async fn empty_input_makes_no_request() {
        let fake = FakeEmbedder { limit: 3, calls: Mutex::new(Vec::new()), drop_one: false };
        let model = Batched::new(fake, &ProviderConfig::default());
        assert!(model.embed(&[]).await.unwrap().is_empty());
        assert!(model.inner.calls.lock().unwrap().is_empty());
    }

    #[tokio::test]
    async fn configured_batch_size_overrides_the_provider_limit() {
        let fake = FakeEmbedder { limit: 100, calls: Mutex::new(Vec::new()), drop_one: false };
        let config = ProviderConfig::default().with_extra("embedding_batch_size", 2);
        let model = Batched::new(fake, &config);
        model.embed(&inputs(5)).await.unwrap();
        assert_eq!(*model.inner.calls.lock().unwrap(), vec![2, 2, 1]);
    }

    #[tokio::test]
    async fn a_short_response_is_an_error_not_a_misaligned_result() {
        let fake = FakeEmbedder { limit: 4, calls: Mutex::new(Vec::new()), drop_one: true };
        let model = Batched::new(fake, &ProviderConfig::default());
        let err = model.embed(&inputs(4)).await.unwrap_err();
        assert!(err.to_string().contains("3 embeddings returned for 4 inputs"));
    }

    #[test]
    fn content_text_reads_strings_and_text_parts() {
        assert_eq!(content_text(&serde_json::json!("plain")), "plain");
        let parts = serde_json::json!([
            {"type": "thinking", "thinking": "hmm", "text": "hidden"},
            {"type": "text", "text": "Hello "},
            {"text": "world"},
        ]);
        assert_eq!(content_text(&parts), "Hello world");
        assert_eq!(content_text(&Value::Null), "");
    }

    #[test]
    fn finish_reasons_are_normalised() {
        let norm = |s: &str| finish_reason(Some(&Value::from(s)));
        assert_eq!(norm("STOP").as_deref(), Some("stop"));
        assert_eq!(norm("end_turn").as_deref(), Some("stop"));
        assert_eq!(norm("MAX_TOKENS").as_deref(), Some("length"));
        assert_eq!(norm("SAFETY").as_deref(), Some("content_filter"));
        assert_eq!(norm("tool_calls").as_deref(), Some("tool_calls"));
        assert_eq!(finish_reason(Some(&Value::Null)), None);
        assert_eq!(finish_reason(None), None);
    }

    #[test]
    fn floats_serialise_without_widening_noise() {
        assert_eq!(float(0.7).to_string(), "0.7");
        assert_eq!(float(1.0).to_string(), "1.0");
        assert_eq!(float(0.25).to_string(), "0.25");
    }

    #[test]
    fn vectors_reject_non_numeric_values() {
        assert_eq!(vector(&serde_json::json!([0.5, 1, -2])), Some(vec![0.5, 1.0, -2.0]));
        assert_eq!(vector(&serde_json::json!([0.5, null])), None);
        assert_eq!(vectors(Some(&serde_json::json!([[1.0], [2.0]]))), Some(vec![vec![1.0], vec![2.0]]));
    }
}
