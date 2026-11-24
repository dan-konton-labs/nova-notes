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

/// Largest JSON (or other non-audio) response body read from a provider.
pub const MAX_JSON_RESPONSE_BYTES: usize = 32 * 1024 * 1024;
/// Largest audio response read from a provider: raw audio, or JSON carrying
/// base64 / hex audio (Gemini and MiniMax speech).
pub const MAX_AUDIO_RESPONSE_BYTES: usize = 512 * 1024 * 1024;
/// How much of an error response body is read to find the provider's message.
pub const MAX_ERROR_BODY_BYTES: usize = 64 * 1024;

/// The process-wide client used when the caller does not bring its own.
/// Redirects are never followed: a provider endpoint that redirects is an
/// error, so keys in custom headers cannot follow a redirect elsewhere.
pub(crate) fn shared_client() -> reqwest::Client {
    static CLIENT: OnceLock<reqwest::Client> = OnceLock::new();
    CLIENT
        .get_or_init(|| {
            reqwest::Client::builder()
                .user_agent("nova-notes")
                .connect_timeout(Duration::from_secs(30))
                .redirect(reqwest::redirect::Policy::none())
                .build()
                .unwrap_or_default()
        })
        .clone()
}

/// Reads a response body, failing once it exceeds `cap` bytes.
pub(crate) async fn read_body(provider: &str, mut response: Response, cap: usize) -> AiResult<Bytes> {
    let too_large = || AiError::Provider {
        status: None,
        message: format!("{provider}: unexpected response: the response is larger than {} MiB", cap / (1024 * 1024)),
    };
    if response.content_length().is_some_and(|len| len > cap as u64) {
        return Err(too_large());
    }
    let mut body = Vec::new();
    while let Some(chunk) = response.chunk().await.map_err(|e| transport_error(provider, e))? {
        if body.len() + chunk.len() > cap {
            return Err(too_large());
        }
        body.extend_from_slice(&chunk);
    }
    Ok(Bytes::from(body))
}

/// Reads at most `cap` bytes of a body and drops the rest (error bodies,
/// where only the start matters). A read failure ends the body early.
pub(crate) async fn read_prefix(mut response: Response, cap: usize) -> Vec<u8> {
    let mut body = Vec::new();
    while let Ok(Some(chunk)) = response.chunk().await {
        let room = cap - body.len();
        body.extend_from_slice(&chunk[..chunk.len().min(room)]);
        if body.len() >= cap {
            break;
        }
    }
    body
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
        let body = read_prefix(response, MAX_ERROR_BODY_BYTES).await;
        let body = String::from_utf8_lossy(&body);
        Err(http_error(&self.provider, status.as_u16(), &body, self.secret.as_deref()))
    }

    pub async fn json(&self, request: RequestBuilder) -> AiResult<Value> {
        self.json_capped(request, MAX_JSON_RESPONSE_BYTES).await
    }

    /// [`json`](Self::json) with a body limit other than
    /// [`MAX_JSON_RESPONSE_BYTES`] (JSON that carries audio).
    pub async fn json_capped(&self, request: RequestBuilder, cap: usize) -> AiResult<Value> {
        let response = self.send(request).await?;
        let body = read_body(&self.provider, response, cap).await?;
        serde_json::from_slice(&body).map_err(|_| self.malformed("the response was not valid JSON"))
    }

    /// Returns the body (up to [`MAX_AUDIO_RESPONSE_BYTES`]) and its
    /// `Content-Type` (without parameters).
    pub async fn bytes(&self, request: RequestBuilder) -> AiResult<(Bytes, Option<String>)> {
        let response = self.send(request).await?;
        let content_type = response
            .headers()
            .get(reqwest::header::CONTENT_TYPE)
            .and_then(|v| v.to_str().ok())
            .map(|v| v.split(';').next().unwrap_or(v).trim().to_ascii_lowercase())
            .filter(|v| !v.is_empty());
        let body = read_body(&self.provider, response, MAX_AUDIO_RESPONSE_BYTES).await?;
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

    #[tokio::test]
    async fn bodies_over_the_cap_are_refused() {
        let server = MockServer::start().await;
        Mock::given(method("GET")).respond_with(ResponseTemplate::new(200).set_body_string("x".repeat(100))).mount(&server).await;
        let response = shared_client().get(server.uri()).send().await.unwrap();
        let err = read_body("testprov", response, 10).await.unwrap_err();
        assert!(err.to_string().contains("larger than"), "{err}");
        let response = shared_client().get(server.uri()).send().await.unwrap();
        assert_eq!(read_body("testprov", response, 100).await.unwrap().len(), 100);
        let response = shared_client().get(server.uri()).send().await.unwrap();
        assert_eq!(read_prefix(response, 7).await, b"xxxxxxx");
    }

    #[tokio::test]
    async fn redirects_are_not_followed_so_key_headers_stay_put() {
        let target = MockServer::start().await;
        Mock::given(method("GET")).respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({}))).mount(&target).await;
        let redirector = MockServer::start().await;
        Mock::given(method("GET"))
            .respond_with(ResponseTemplate::new(302).insert_header("location", format!("{}/stolen", target.uri())))
            .mount(&redirector)
            .await;
        let http = http(&ProviderConfig::default().with_api_key("sk-ant-secret"));
        let request = http.client.get(format!("{}/v1/models", redirector.uri())).header("x-api-key", "sk-ant-secret");
        let err = http.json(request).await.unwrap_err();
        assert!(err.to_string().contains("HTTP 302"), "{err}");
        assert!(target.received_requests().await.unwrap().is_empty(), "the redirect was followed");
    }

    #[tokio::test]
    async fn non_json_error_pages_are_not_reflected() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .respond_with(ResponseTemplate::new(500).set_body_string("consul: internal secret value"))
            .mount(&server)
            .await;
        let http = http(&ProviderConfig::default());
        let err = http.json(http.client.get(server.uri())).await.unwrap_err();
        assert_eq!(err.to_string(), "testprov: internal server error (HTTP 500)");
    }
}
