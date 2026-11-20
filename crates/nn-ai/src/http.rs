//! Thin HTTP layer shared by every provider: timeouts, status checking and
//! error mapping in one place.

use std::sync::OnceLock;
use std::time::Duration;

use bytes::Bytes;
use nn_llm::{AiError, AiResult};
use reqwest::{RequestBuilder, Response};
use serde_json::Value;

use crate::config::ProviderConfig;
use crate::error::{http_error, transport_error};

/// The process-wide client used when the caller does not bring its own.
pub(crate) fn shared_client() -> reqwest::Client {
    static CLIENT: OnceLock<reqwest::Client> = OnceLock::new();
    CLIENT
        .get_or_init(|| {
            reqwest::Client::builder()
                .user_agent("nova-notes")
                .connect_timeout(Duration::from_secs(30))
                .build()
                .unwrap_or_default()
        })
        .clone()
}

/// A client bound to one provider: knows its name (for error messages), its
/// request timeout and the secret to scrub from anything it reports.
#[derive(Clone)]
pub(crate) struct Http {
    pub client: reqwest::Client,
    pub provider: String,
    pub timeout: Duration,
    secret: Option<String>,
}

impl Http {
    pub fn new(client: &reqwest::Client, provider: &str, config: &ProviderConfig, default_timeout: Duration) -> Self {
        Self {
            client: client.clone(),
            provider: provider.to_string(),
            timeout: config.timeout.unwrap_or(default_timeout),
            secret: config.key().map(str::to_string),
        }
    }

    /// Sends a request whose whole exchange must finish within the timeout.
    pub async fn send(&self, request: RequestBuilder) -> AiResult<Response> {
        let response =
            request.timeout(self.timeout).send().await.map_err(|e| transport_error(&self.provider, e))?;
        self.check(response).await
    }

    /// Sends a request whose body will be consumed as a stream: the timeout
    /// covers the wait for the response headers only. The caller applies it
    /// again as an idle timeout between chunks.
    pub async fn send_streaming(&self, request: RequestBuilder) -> AiResult<Response> {
        let response = match tokio::time::timeout(self.timeout, request.send()).await {
            Ok(result) => result.map_err(|e| transport_error(&self.provider, e))?,
            Err(_) => {
                return Err(AiError::Timeout(format!(
                    "{}: request timed out (no response within {} seconds)",
                    self.provider,
                    self.timeout.as_secs()
                )))
            }
        };
        self.check(response).await
    }

    async fn check(&self, response: Response) -> AiResult<Response> {
        let status = response.status();
        if status.is_success() {
            return Ok(response);
        }
        // A body that cannot be read still leaves the status to report.
        let body = response.text().await.unwrap_or_default();
        Err(http_error(&self.provider, status.as_u16(), &body, self.secret.as_deref()))
    }

    pub async fn json(&self, request: RequestBuilder) -> AiResult<Value> {
        let response = self.send(request).await?;
        let body = response.bytes().await.map_err(|e| transport_error(&self.provider, e))?;
        serde_json::from_slice(&body).map_err(|_| self.malformed("the response was not valid JSON"))
    }

    /// Returns the body and its `Content-Type` (without parameters).
    pub async fn bytes(&self, request: RequestBuilder) -> AiResult<(Bytes, Option<String>)> {
        let response = self.send(request).await?;
        let content_type = response
            .headers()
            .get(reqwest::header::CONTENT_TYPE)
            .and_then(|v| v.to_str().ok())
            .map(|v| v.split(';').next().unwrap_or(v).trim().to_ascii_lowercase())
            .filter(|v| !v.is_empty());
        let body = response.bytes().await.map_err(|e| transport_error(&self.provider, e))?;
        Ok((body, content_type))
    }

    /// An error for a 2xx response whose content is not what the API promises.
    pub fn malformed(&self, what: &str) -> AiError {
        AiError::Provider { status: None, message: format!("{}: unexpected response: {what}", self.provider) }
    }
}

/// Joins a base URL and a path with exactly one slash between them.
pub(crate) fn join_url(base: &str, path: &str) -> String {
    format!("{}/{}", base.trim_end_matches('/'), path.trim_start_matches('/'))
}

/// Percent-encodes a value for use as one URL path segment or query value.
pub(crate) fn encode_component(value: &str) -> String {
    let mut out = String::with_capacity(value.len());
    for byte in value.bytes() {
        if byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.' | b'~') {
            out.push(byte as char);
        } else {
            out.push_str(&format!("%{byte:02X}"));
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    fn http(config: &ProviderConfig) -> Http {
        Http::new(&shared_client(), "testprov", config, Duration::from_secs(5))
    }

    #[test]
    fn urls_join_with_a_single_slash() {
        assert_eq!(join_url("http://h/v1/", "/models"), "http://h/v1/models");
        assert_eq!(join_url("http://h/v1", "models"), "http://h/v1/models");
    }

    #[test]
    fn components_are_percent_encoded() {
        assert_eq!(encode_component("gpt-4o_mini.v1~x"), "gpt-4o_mini.v1~x");
        assert_eq!(encode_component("my dep/ø?"), "my%20dep%2F%C3%B8%3F");
    }

    #[tokio::test]
    async fn error_statuses_become_typed_errors_with_the_provider_message() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/x"))
            .respond_with(
                ResponseTemplate::new(429).set_body_json(serde_json::json!({"error": {"message": "slow down"}})),
            )
            .mount(&server)
            .await;
        let http = http(&ProviderConfig::default());
        let err = http.json(http.client.get(format!("{}/x", server.uri()))).await.unwrap_err();
        assert!(matches!(err, AiError::RateLimit(_)));
        assert_eq!(err.to_string(), "testprov: slow down (HTTP 429)");
    }

    #[tokio::test]
    async fn connection_refused_is_a_network_error_without_the_url() {
        // Bind then drop to get a port nothing listens on.
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        drop(listener);
        let http = http(&ProviderConfig::default().with_api_key("sk-secret-key"));
        let url = format!("http://127.0.0.1:{port}/v1/chat?marker=findme");
        let err = http.json(http.client.get(url).bearer_auth("sk-secret-key")).await.unwrap_err();
        assert!(matches!(err, AiError::Network(_)), "{err:?}");
        let text = err.to_string();
        assert!(!text.contains("findme"));
        assert!(!text.contains("sk-secret-key"));
        assert!(err.is_retryable());
    }

    #[tokio::test]
    async fn slow_responses_are_timeouts() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .respond_with(ResponseTemplate::new(200).set_delay(Duration::from_secs(5)))
            .mount(&server)
            .await;
        let config = ProviderConfig::default().with_timeout(Duration::from_millis(100));
        let http = http(&config);
        let err = http.json(http.client.get(server.uri())).await.unwrap_err();
        assert!(matches!(err, AiError::Timeout(_)), "{err:?}");
        let err = http.send_streaming(http.client.get(server.uri())).await.unwrap_err();
        assert!(matches!(err, AiError::Timeout(_)), "{err:?}");
    }

    #[tokio::test]
    async fn non_json_success_is_reported_as_malformed() {
        let server = MockServer::start().await;
        Mock::given(method("GET")).respond_with(ResponseTemplate::new(200).set_body_string("<html>")).mount(&server).await;
        let http = http(&ProviderConfig::default());
        let err = http.json(http.client.get(server.uri())).await.unwrap_err();
        assert!(matches!(err, AiError::Provider { status: None, .. }));
    }
}
