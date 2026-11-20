//! Voyage AI embeddings (`POST /v1/embeddings`).

use async_trait::async_trait;
use nn_llm::AiResult;
use serde_json::{json, Value};

use super::{require_key, vector, EmbedBatch};
use crate::config::{ProviderConfig, DEFAULT_EMBEDDING_TIMEOUT};
use crate::http::{join_url, Http};

pub(crate) const VOYAGE_API_ROOT: &str = "https://api.voyageai.com/v1";

pub(crate) struct VoyageEmbedder {
    http: Http,
    model: String,
    base: String,
    api_key: String,
    input_type: String,
    dimensions: Option<u64>,
}

impl VoyageEmbedder {
    pub fn new(client: &reqwest::Client, model: &str, config: &ProviderConfig) -> AiResult<Self> {
        Ok(Self {
            http: Http::new(client, "voyage", config, DEFAULT_EMBEDDING_TIMEOUT),
            model: model.to_string(),
            base: config.url().unwrap_or(VOYAGE_API_ROOT).to_string(),
            api_key: require_key("voyage", config)?,
            input_type: config.extra_str("input_type").unwrap_or("document").to_string(),
            dimensions: config.extra_u64("dimensions"),
        })
    }
}

#[async_trait]
impl EmbedBatch for VoyageEmbedder {
    fn provider(&self) -> &str {
        "voyage"
    }

    fn model_name(&self) -> &str {
        &self.model
    }

    fn batch_limit(&self) -> usize {
        128
    }

    async fn embed_batch(&self, texts: &[String]) -> AiResult<Vec<Vec<f32>>> {
        let mut body = json!({"model": self.model, "input": texts, "input_type": self.input_type});
        if let Some(dimensions) = self.dimensions {
            body["output_dimension"] = json!(dimensions);
        }
        let request =
            self.http.client.post(join_url(&self.base, "embeddings")).bearer_auth(&self.api_key).json(&body);
        let json = self.http.json(request).await?;
        let mut items: Vec<(u64, Vec<f32>)> = json
            .get("data")
            .and_then(Value::as_array)
            .and_then(|data| {
                data.iter()
                    .enumerate()
                    .map(|(position, item)| {
                        let index = item.get("index").and_then(Value::as_u64).unwrap_or(position as u64);
                        Some((index, vector(item.get("embedding")?)?))
                    })
                    .collect()
            })
            .ok_or_else(|| self.http.malformed("no embeddings in the response"))?;
        items.sort_by_key(|(index, _)| *index);
        Ok(items.into_iter().map(|(_, v)| v).collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::http::shared_client;
    use crate::providers::testing::{header, request, request_json};
    use nn_llm::AiError;
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    #[tokio::test]
    async fn embeds_with_document_input_type_and_orders_by_index() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/embeddings"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "object": "list",
                "data": [{"index": 1, "embedding": [2.0]}, {"index": 0, "embedding": [1.0]}],
                "model": "voyage-3-lite",
                "usage": {"total_tokens": 4},
            })))
            .mount(&server)
            .await;
        let config = ProviderConfig::new().with_api_key("pa-test").with_base_url(format!("{}/v1", server.uri()));
        let embedder = VoyageEmbedder::new(&shared_client(), "voyage-3-lite", &config).unwrap();
        let out = embedder.embed_batch(&["a".to_string(), "b".to_string()]).await.unwrap();
        assert_eq!(out, vec![vec![1.0], vec![2.0]]);
        assert_eq!(embedder.batch_limit(), 128);
        assert_eq!(header(&request(&server, 0).await, "authorization"), Some("Bearer pa-test"));
        assert_eq!(
            request_json(&server, 0).await,
            json!({"model": "voyage-3-lite", "input": ["a", "b"], "input_type": "document"})
        );
    }

    #[test]
    fn a_key_is_required() {
        let err = VoyageEmbedder::new(&shared_client(), "voyage-3", &ProviderConfig::new()).err().unwrap();
        assert!(matches!(err, AiError::Configuration(_)));
    }
}
