//! Remote web engines: Jina Reader, Firecrawl and a remote Crawl4AI server.
//!
//! Each takes a page URL and returns Markdown produced by the service. They
//! are called with `ExtractContext::client`; the page URL has already passed
//! the SSRF guard. Error text never contains API keys.

use reqwest::header::{ACCEPT, AUTHORIZATION};
use reqwest::StatusCode;
use serde_json::{json, Map, Value};
use url::Url;

use super::fetch::network_error;
use super::ExtractContext;
use crate::error::ContentError;

/// A page as returned by a web engine.
#[derive(Debug, Clone, PartialEq, Default)]
pub(crate) struct EnginePage {
    pub title: Option<String>,
    pub content: String,
    pub metadata: Map<String, Value>,
}

fn clean(value: Option<&Value>) -> Option<String> {
    value.and_then(Value::as_str).map(str::trim).filter(|s| !s.is_empty()).map(str::to_string)
}

fn service_status(service: &str, status: StatusCode, body: &str) -> ContentError {
    let detail: String = body.chars().take(300).collect();
    match status.as_u16() {
        401 | 403 => ContentError::Configuration(format!(
            "{service} rejected the credentials (HTTP {}); check the API key or token",
            status.as_u16()
        )),
        code => ContentError::ExternalService(format!("{service} returned HTTP {code}: {}", detail.trim())),
    }
}

/// Jina Reader: `GET {reader}/{page url}`, JSON response.
pub(crate) async fn jina(url: &Url, ctx: &ExtractContext) -> Result<EnginePage, ContentError> {
    let endpoint = format!("{}/{}", ctx.jina_reader_url.trim_end_matches('/'), url);
    let mut request = ctx
        .client
        .get(&endpoint)
        .header(ACCEPT, "application/json")
        .header("X-Return-Format", "markdown")
        .timeout(ctx.limits.engine_timeout);
    if let Some(key) = ctx.jina_api_key.as_deref().filter(|k| !k.trim().is_empty()) {
        request = request.header(AUTHORIZATION, format!("Bearer {}", key.trim()));
    }
    let response = request.send().await.map_err(network_error)?;
    let status = response.status();
    let body = response.text().await.map_err(network_error)?;
    if !status.is_success() {
        return Err(service_status("Jina Reader", status, &body));
    }

    let mut page = EnginePage::default();
    match serde_json::from_str::<Value>(&body) {
        Ok(json) => {
            let data = json.get("data").unwrap_or(&json);
            page.title = clean(data.get("title"));
            page.content = data.get("content").and_then(Value::as_str).unwrap_or("").trim().to_string();
            for key in ["description", "publishedTime"] {
                if let Some(value) = clean(data.get(key)) {
                    page.metadata.insert(key.to_string(), json!(value));
                }
            }
        }
        // Without JSON negotiation the reader answers in its text envelope.
        Err(_) => {
            let (title, content) = parse_jina_text(&body);
            page.title = title;
            page.content = content;
        }
    }
    Ok(page)
}

/// Split Jina's plain-text envelope (`Title: ...`, `URL Source: ...`,
/// `Markdown Content:`) into title and content.
fn parse_jina_text(body: &str) -> (Option<String>, String) {
    let title = body
        .lines()
        .next()
        .and_then(|line| line.strip_prefix("Title:"))
        .map(|t| t.trim().to_string())
        .filter(|t| !t.is_empty());
    let content = match body.split_once("Markdown Content:") {
        Some((_, content)) => content,
        None if title.is_some() => body.split_once('\n').map_or("", |(_, rest)| rest),
        None => body,
    };
    (title, content.trim().to_string())
}

/// Firecrawl scrape API (`/v2/scrape`, falling back to `/v1/scrape` on
/// self-hosted servers that predate v2).
pub(crate) async fn firecrawl(url: &Url, ctx: &ExtractContext) -> Result<EnginePage, ContentError> {
    let base = ctx.firecrawl_api_url.trim_end_matches('/');
    let mut payload = json!({
        "url": url.as_str(),
        "formats": ["markdown"],
        "onlyMainContent": true,
        "waitFor": ctx.firecrawl_wait_for_ms,
    });
    let proxy = ctx.firecrawl_proxy.trim().to_ascii_lowercase();
    if matches!(proxy.as_str(), "basic" | "stealth" | "auto") {
        payload["proxy"] = json!(proxy);
    }

    let mut last_status = None;
    for version in ["v2", "v1"] {
        let mut request =
            ctx.client.post(format!("{base}/{version}/scrape")).json(&payload).timeout(ctx.limits.engine_timeout);
        if let Some(key) = ctx.firecrawl_api_key.as_deref().filter(|k| !k.trim().is_empty()) {
            request = request.header(AUTHORIZATION, format!("Bearer {}", key.trim()));
        }
        let response = request.send().await.map_err(network_error)?;
        let status = response.status();
        let body = response.text().await.map_err(network_error)?;
        let parsed = serde_json::from_str::<Value>(&body).ok();

        // A 404 without Firecrawl's JSON envelope means the route itself is missing.
        if status == StatusCode::NOT_FOUND && parsed.as_ref().and_then(|j| j.get("success")).is_none() {
            last_status = Some((status, body));
            continue;
        }
        let failure = parsed.as_ref().and_then(|j| clean(j.get("error")));
        if !status.is_success() {
            return Err(service_status("Firecrawl", status, failure.as_deref().unwrap_or(&body)));
        }
        let json = parsed.ok_or_else(|| ContentError::ExternalService("Firecrawl returned a non-JSON response".into()))?;
        if json.get("success").and_then(Value::as_bool) == Some(false) {
            return Err(ContentError::ExternalService(format!(
                "Firecrawl could not scrape the page: {}",
                failure.unwrap_or_else(|| "no reason given".to_string())
            )));
        }

        let data = json.get("data").unwrap_or(&json);
        let meta = data.get("metadata");
        if meta.and_then(|m| m.get("statusCode")).and_then(Value::as_u64).is_some_and(|code| code == 404 || code == 410) {
            return Err(ContentError::NotFound("the page returned HTTP 404 to Firecrawl".to_string()));
        }
        let mut page = EnginePage {
            title: clean(meta.and_then(|m| m.get("title"))).or_else(|| clean(meta.and_then(|m| m.get("ogTitle")))),
            content: data.get("markdown").and_then(Value::as_str).unwrap_or("").trim().to_string(),
            metadata: Map::new(),
        };
        for key in ["description", "language", "sourceURL"] {
            if let Some(value) = clean(meta.and_then(|m| m.get(key))) {
                page.metadata.insert(key.to_string(), json!(value));
            }
        }
        return Ok(page);
    }
    let (status, body) = last_status.unwrap_or((StatusCode::NOT_FOUND, String::new()));
    Err(service_status("Firecrawl", status, &body))
}

/// Remote Crawl4AI server: `POST {base}/crawl`.
pub(crate) async fn crawl4ai(url: &Url, ctx: &ExtractContext) -> Result<EnginePage, ContentError> {
    let base = ctx
        .crawl4ai_api_url
        .as_deref()
        .map(|b| b.trim().trim_end_matches('/'))
        .filter(|b| !b.is_empty())
        .ok_or_else(|| ContentError::Configuration("no Crawl4AI server URL is configured".to_string()))?;
    let payload = json!({
        "urls": [url.as_str()],
        "browser_config": { "type": "BrowserConfig", "params": { "headless": true } },
        "crawler_config": { "type": "CrawlerRunConfig", "params": { "cache_mode": "bypass", "stream": false } },
    });
    let mut request = ctx.client.post(format!("{base}/crawl")).json(&payload).timeout(ctx.limits.engine_timeout);
    if let Some(token) = ctx.crawl4ai_api_token.as_deref().filter(|t| !t.trim().is_empty()) {
        request = request.header(AUTHORIZATION, format!("Bearer {}", token.trim()));
    }
    let response = request.send().await.map_err(network_error)?;
    let status = response.status();
    let body = response.text().await.map_err(network_error)?;
    if !status.is_success() {
        return Err(service_status("Crawl4AI", status, &body));
    }
    let json: Value = serde_json::from_str(&body)
        .map_err(|_| ContentError::ExternalService("Crawl4AI returned a non-JSON response".to_string()))?;

    let result = json
        .get("results")
        .and_then(Value::as_array)
        .and_then(|results| results.first())
        .or_else(|| json.get("result"))
        .ok_or_else(|| ContentError::ExternalService("Crawl4AI returned no result for the page".to_string()))?;
    if result.get("success").and_then(Value::as_bool) == Some(false) {
        if result.get("status_code").and_then(Value::as_u64).is_some_and(|code| code == 404 || code == 410) {
            return Err(ContentError::NotFound("the page returned HTTP 404 to Crawl4AI".to_string()));
        }
        return Err(ContentError::ExternalService(format!(
            "Crawl4AI could not crawl the page: {}",
            clean(result.get("error_message")).unwrap_or_else(|| "no reason given".to_string())
        )));
    }

    // `markdown` is a string on old servers and an object on current ones.
    let content = match result.get("markdown") {
        Some(Value::String(markdown)) => markdown.clone(),
        Some(Value::Object(variants)) => ["fit_markdown", "raw_markdown", "markdown_with_citations"]
            .iter()
            .find_map(|key| clean(variants.get(*key)))
            .unwrap_or_default(),
        _ => String::new(),
    };
    let meta = result.get("metadata");
    let mut page = EnginePage {
        title: clean(meta.and_then(|m| m.get("title"))).or_else(|| clean(meta.and_then(|m| m.get("og:title")))),
        content: content.trim().to_string(),
        metadata: Map::new(),
    };
    if let Some(description) = clean(meta.and_then(|m| m.get("description"))) {
        page.metadata.insert("description".to_string(), json!(description));
    }
    Ok(page)
}

#[cfg(test)]
mod tests {
    use wiremock::matchers::{body_partial_json, header, method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    use super::*;

    fn page_url() -> Url {
        Url::parse("https://example.com/articles/one?x=1").unwrap()
    }

    #[tokio::test]
    async fn jina_json_response_with_key() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/https://example.com/articles/one"))
            .and(header("authorization", "Bearer jina-key"))
            .and(header("accept", "application/json"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "code": 200,
                "data": {"title": " Article One ", "content": "# Article One\n\nBody.\n", "description": "About one", "url": "https://example.com/articles/one"}
            })))
            .mount(&server)
            .await;
        let ctx = ExtractContext {
            jina_reader_url: format!("{}/", server.uri()),
            jina_api_key: Some("jina-key".into()),
            ..ExtractContext::default()
        };
        let page = jina(&page_url(), &ctx).await.unwrap();
        assert_eq!(page.title.as_deref(), Some("Article One"));
        assert_eq!(page.content, "# Article One\n\nBody.");
        assert_eq!(page.metadata["description"], "About one");
    }

    #[tokio::test]
    async fn jina_text_envelope_without_key_and_errors() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/https://example.com/articles/one"))
            .respond_with(ResponseTemplate::new(200).set_body_string(
                "Title: Plain Title\n\nURL Source: https://example.com/articles/one\n\nMarkdown Content:\nFirst paragraph.\n\nSecond.",
            ))
            .mount(&server)
            .await;
        let ctx = ExtractContext { jina_reader_url: server.uri(), ..ExtractContext::default() };
        let page = jina(&page_url(), &ctx).await.unwrap();
        assert_eq!(page.title.as_deref(), Some("Plain Title"));
        assert_eq!(page.content, "First paragraph.\n\nSecond.");
        let request = &server.received_requests().await.unwrap()[0];
        assert!(request.headers.get("authorization").is_none());

        let failing = MockServer::start().await;
        Mock::given(method("GET")).respond_with(ResponseTemplate::new(429).set_body_string("slow down")).mount(&failing).await;
        let ctx = ExtractContext { jina_reader_url: failing.uri(), ..ExtractContext::default() };
        let error = jina(&page_url(), &ctx).await.unwrap_err();
        assert!(matches!(&error, ContentError::ExternalService(m) if m.contains("429")), "{error:?}");

        let denied = MockServer::start().await;
        Mock::given(method("GET")).respond_with(ResponseTemplate::new(401)).mount(&denied).await;
        let ctx = ExtractContext {
            jina_reader_url: denied.uri(),
            jina_api_key: Some("super-secret".into()),
            ..ExtractContext::default()
        };
        let error = jina(&page_url(), &ctx).await.unwrap_err();
        assert!(matches!(error, ContentError::Configuration(_)), "{error:?}");
        assert!(!error.to_string().contains("super-secret"));
    }

    #[test]
    fn jina_text_parsing_edge_cases() {
        assert_eq!(parse_jina_text("just markdown"), (None, "just markdown".to_string()));
        assert_eq!(parse_jina_text("Title: T\nbody line"), (Some("T".to_string()), "body line".to_string()));
    }

    #[tokio::test]
    async fn firecrawl_scrape() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v2/scrape"))
            .and(header("authorization", "Bearer fc-key"))
            .and(body_partial_json(json!({
                "url": "https://example.com/articles/one?x=1",
                "formats": ["markdown"],
                "proxy": "stealth",
                "waitFor": 1500
            })))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "success": true,
                "data": {"markdown": "Scraped **content**.", "metadata": {"title": "FC Title", "language": "en", "statusCode": 200}}
            })))
            .mount(&server)
            .await;
        let ctx = ExtractContext {
            firecrawl_api_url: server.uri(),
            firecrawl_api_key: Some("fc-key".into()),
            firecrawl_proxy: "Stealth".into(),
            firecrawl_wait_for_ms: 1500,
            ..ExtractContext::default()
        };
        let page = firecrawl(&page_url(), &ctx).await.unwrap();
        assert_eq!(page.title.as_deref(), Some("FC Title"));
        assert_eq!(page.content, "Scraped **content**.");
        assert_eq!(page.metadata["language"], "en");
    }

    #[tokio::test]
    async fn firecrawl_falls_back_to_v1_and_reports_failures() {
        let old = MockServer::start().await;
        Mock::given(method("POST")).and(path("/v2/scrape")).respond_with(ResponseTemplate::new(404)).mount(&old).await;
        Mock::given(method("POST"))
            .and(path("/v1/scrape"))
            .respond_with(
                ResponseTemplate::new(200)
                    .set_body_json(json!({"success": true, "data": {"markdown": "v1 body", "metadata": {}}})),
            )
            .mount(&old)
            .await;
        let ctx = ExtractContext { firecrawl_api_url: old.uri(), ..ExtractContext::default() };
        assert_eq!(firecrawl(&page_url(), &ctx).await.unwrap().content, "v1 body");

        let failing = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v2/scrape"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({"success": false, "error": "blocked by site"})))
            .mount(&failing)
            .await;
        let ctx = ExtractContext { firecrawl_api_url: failing.uri(), ..ExtractContext::default() };
        let error = firecrawl(&page_url(), &ctx).await.unwrap_err();
        assert!(matches!(&error, ContentError::ExternalService(m) if m.contains("blocked by site")), "{error:?}");

        let gone = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v2/scrape"))
            .respond_with(ResponseTemplate::new(200).set_body_json(
                json!({"success": true, "data": {"markdown": "Not found page", "metadata": {"statusCode": 404}}}),
            ))
            .mount(&gone)
            .await;
        let ctx = ExtractContext { firecrawl_api_url: gone.uri(), ..ExtractContext::default() };
        assert!(matches!(firecrawl(&page_url(), &ctx).await, Err(ContentError::NotFound(_))));

        let payment = MockServer::start().await;
        Mock::given(method("POST"))
            .respond_with(ResponseTemplate::new(402).set_body_json(json!({"success": false, "error": "Insufficient credits"})))
            .mount(&payment)
            .await;
        let ctx = ExtractContext { firecrawl_api_url: payment.uri(), ..ExtractContext::default() };
        let error = firecrawl(&page_url(), &ctx).await.unwrap_err();
        assert!(matches!(&error, ContentError::ExternalService(m) if m.contains("Insufficient credits")), "{error:?}");
    }

    #[tokio::test]
    async fn crawl4ai_markdown_shapes() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/crawl"))
            .and(header("authorization", "Bearer c4-token"))
            .and(body_partial_json(json!({"urls": ["https://example.com/articles/one?x=1"]})))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "success": true,
                "results": [{
                    "success": true,
                    "markdown": {"raw_markdown": "raw text", "fit_markdown": ""},
                    "metadata": {"title": "C4 Title", "description": "desc"}
                }]
            })))
            .mount(&server)
            .await;
        let ctx = ExtractContext {
            crawl4ai_api_url: Some(format!("{}/", server.uri())),
            crawl4ai_api_token: Some("c4-token".into()),
            ..ExtractContext::default()
        };
        let page = crawl4ai(&page_url(), &ctx).await.unwrap();
        assert_eq!(page.title.as_deref(), Some("C4 Title"));
        assert_eq!(page.content, "raw text");
        assert_eq!(page.metadata["description"], "desc");

        let legacy = MockServer::start().await;
        Mock::given(method("POST"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({"results": [{"success": true, "markdown": "string form"}]})))
            .mount(&legacy)
            .await;
        let ctx = ExtractContext { crawl4ai_api_url: Some(legacy.uri()), ..ExtractContext::default() };
        assert_eq!(crawl4ai(&page_url(), &ctx).await.unwrap().content, "string form");
    }

    #[tokio::test]
    async fn crawl4ai_failures() {
        let no_url = crawl4ai(&page_url(), &ExtractContext::default()).await.unwrap_err();
        assert!(matches!(no_url, ContentError::Configuration(_)));

        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .respond_with(ResponseTemplate::new(200).set_body_json(
                json!({"results": [{"success": false, "error_message": "net::ERR_NAME_NOT_RESOLVED", "status_code": 0}]}),
            ))
            .mount(&server)
            .await;
        let ctx = ExtractContext { crawl4ai_api_url: Some(server.uri()), ..ExtractContext::default() };
        let error = crawl4ai(&page_url(), &ctx).await.unwrap_err();
        assert!(matches!(&error, ContentError::ExternalService(m) if m.contains("ERR_NAME_NOT_RESOLVED")), "{error:?}");

        let denied = MockServer::start().await;
        Mock::given(method("POST")).respond_with(ResponseTemplate::new(403)).mount(&denied).await;
        let ctx = ExtractContext { crawl4ai_api_url: Some(denied.uri()), ..ExtractContext::default() };
        assert!(matches!(crawl4ai(&page_url(), &ctx).await, Err(ContentError::Configuration(_))));
    }
}
