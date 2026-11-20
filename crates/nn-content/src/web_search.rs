//! Keyless web search.
//!
//! Port of `open_notebook/utils/web_search.py`. The original called the
//! `ddgs` Python package, which tries several search engines in turn. Here
//! each engine is a [`Searcher`] that reads that engine's result page
//! ([`DuckDuckGoSearcher`], [`BingSearcher`], [`BraveSearcher`]) or API
//! ([`WikipediaSearcher`]), and [`MultiSearcher`] tries them in turn so one
//! engine's bot check or rate limit does not stop a search. A self-hosted
//! SearXNG instance ([`SearxngSearcher`]) can be put in front of them as the
//! dependable primary engine.
//!
//! Everything returned is untrusted data from the open web: titles, snippets
//! and URLs must never be placed in system instructions. URLs are meant to be
//! handed to the normal source pipeline, which applies the SSRF guard.

use std::collections::{HashMap, HashSet};
use std::time::Duration;

use async_trait::async_trait;
use scraper::{Html, Selector};
use serde::{Deserialize, Serialize};
use url::Url;

/// Hosts that are social, video, login-walled or otherwise hostile to text
/// extraction. Matched against the host and every parent domain.
pub const BLOCKED_DOMAINS: &[&str] = &[
    "facebook.com",
    "instagram.com",
    "twitter.com",
    "x.com",
    "tiktok.com",
    "youtube.com",
    "youtu.be",
    "vimeo.com",
    "linkedin.com",
    "pinterest.com",
    "quora.com",
    "reddit.com",
    "threads.net",
    "t.me",
    "wsj.com",
    "ft.com",
    "bloomberg.com",
    "nytimes.com",
];

const BLOCKED_EXTENSIONS: &[&str] = &[".pdf", ".zip", ".mp3", ".mp4", ".jpg", ".jpeg", ".png", ".gif"];

/// Default DuckDuckGo HTML endpoint.
pub const DUCKDUCKGO_HTML_URL: &str = "https://html.duckduckgo.com/html/";

/// A search hit after normalisation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SearchResult {
    pub url: String,
    pub title: String,
    pub snippet: String,
    /// Host without a leading `www.`.
    pub domain: String,
    /// Index of the query (in the list given to [`search_many`]) that found it.
    #[serde(default)]
    pub query_index: usize,
    /// Position within that query's results, starting at 0.
    #[serde(default)]
    pub rank: usize,
}

/// A hit as returned by a search backend.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct SearchHit {
    pub title: String,
    pub href: String,
    pub body: String,
}

#[derive(Debug, Clone, thiserror::Error)]
pub enum WebSearchError {
    /// The request could not be completed (connection, timeout).
    #[error("search request failed: {0}")]
    Request(String),
    /// The search service refused or rate-limited the request.
    #[error("search service returned HTTP {0}")]
    Status(u16),
    /// The response was not a result page (bot check, layout change).
    #[error("search response could not be read: {0}")]
    Parse(String),
}

/// A search backend: one query in, raw hits out.
#[async_trait]
pub trait Searcher: Send + Sync {
    async fn search(&self, query: &str, max_results: usize, timeout: Duration) -> Result<Vec<SearchHit>, WebSearchError>;
}

/// Host of `url` in lower case, without a leading `www.`. Empty when the URL
/// has no host.
pub fn domain_of(url: &str) -> String {
    let host = Url::parse(url.trim()).ok().and_then(|u| u.host_str().map(str::to_ascii_lowercase)).unwrap_or_default();
    host.strip_prefix("www.").map(str::to_string).unwrap_or(host)
}

/// Canonical form used for de-duplication: lower-case scheme and host, no
/// `www.`, no default port, no trailing slash, no fragment.
pub fn normalize_url(url: &str) -> String {
    let trimmed = url.trim();
    let Ok(parsed) = Url::parse(trimmed) else {
        return trimmed.to_string();
    };
    let host = parsed.host_str().unwrap_or("").to_ascii_lowercase();
    let host = host.strip_prefix("www.").unwrap_or(&host);
    let port = match parsed.port() {
        Some(port) if port != 80 && port != 443 => format!(":{port}"),
        _ => String::new(),
    };
    let path = parsed.path().trim_end_matches('/');
    let query = parsed.query().map(|q| format!("?{q}")).unwrap_or_default();
    format!("{}://{host}{port}{path}{query}", parsed.scheme())
}

/// `http(s)` only, not a blocked host and not an obviously non-HTML file.
pub fn is_allowed_url(url: &str) -> bool {
    let Ok(parsed) = Url::parse(url.trim()) else {
        return false;
    };
    let host = parsed.host_str().unwrap_or("").to_ascii_lowercase();
    if !matches!(parsed.scheme(), "http" | "https") || host.is_empty() {
        return false;
    }
    let labels: Vec<&str> = host.split('.').collect();
    for start in 0..labels.len().saturating_sub(1) {
        if BLOCKED_DOMAINS.contains(&labels[start..].join(".").as_str()) {
            return false;
        }
    }
    let path = parsed.path().to_lowercase();
    !BLOCKED_EXTENSIONS.iter().any(|ext| path.ends_with(ext))
}

/// Drop disallowed and duplicate URLs and keep at most `per_domain` results
/// per domain, preserving order.
pub fn dedupe_and_limit(results: &[SearchResult], per_domain: usize) -> Vec<SearchResult> {
    let mut seen: HashSet<String> = HashSet::new();
    let mut per: HashMap<String, usize> = HashMap::new();
    let mut out = Vec::new();
    for result in results {
        if !is_allowed_url(&result.url) {
            continue;
        }
        let key = normalize_url(&result.url);
        if seen.contains(&key) {
            continue;
        }
        let domain = if result.domain.is_empty() { domain_of(&result.url) } else { result.domain.clone() };
        let count = per.entry(domain).or_insert(0);
        if *count >= per_domain {
            continue;
        }
        *count += 1;
        seen.insert(key);
        out.push(result.clone());
    }
    out
}

/// Settings for [`search_many`]. The defaults are the original's.
#[derive(Debug, Clone, PartialEq)]
pub struct SearchOptions {
    pub results_per_query: usize,
    pub per_domain: usize,
    /// Per-query timeout handed to the searcher.
    pub timeout: Duration,
    /// Pause between queries, to stay under rate limits.
    pub delay: Duration,
}

impl Default for SearchOptions {
    fn default() -> Self {
        Self { results_per_query: 8, per_domain: 2, timeout: Duration::from_secs(10), delay: Duration::from_secs(1) }
    }
}

/// Run `queries` one after another and merge the results.
///
/// A failing query (rate limit, timeout, no results) is logged and skipped.
/// The merged list is filtered with [`dedupe_and_limit`].
pub async fn search_many(queries: &[String], options: &SearchOptions, searcher: &dyn Searcher) -> Vec<SearchResult> {
    let mut collected = Vec::new();
    for (query_index, query) in queries.iter().enumerate() {
        if query_index > 0 && !options.delay.is_zero() {
            tokio::time::sleep(options.delay).await;
        }
        let hits = match searcher.search(query, options.results_per_query, options.timeout).await {
            Ok(hits) => hits,
            Err(e) => {
                tracing::warn!("web search failed for query {query:?}: {e}");
                continue;
            }
        };
        for (rank, hit) in hits.into_iter().enumerate() {
            let url = hit.href.trim().to_string();
            if url.is_empty() {
                continue;
            }
            collected.push(SearchResult {
                domain: domain_of(&url),
                url,
                title: hit.title.trim().to_string(),
                snippet: hit.body.trim().to_string(),
                query_index,
                rank,
            });
        }
    }
    dedupe_and_limit(&collected, options.per_domain)
}

/// DuckDuckGo search over its HTML (no JavaScript, no API key) endpoint.
#[derive(Debug, Clone)]
pub struct DuckDuckGoSearcher {
    client: reqwest::Client,
    endpoint: String,
    region: Option<String>,
}

impl DuckDuckGoSearcher {
    pub fn new(client: reqwest::Client) -> Self {
        Self { client, endpoint: DUCKDUCKGO_HTML_URL.to_string(), region: None }
    }

    /// Use another endpoint (a mirror, or a mock server in tests).
    pub fn with_endpoint(mut self, endpoint: impl Into<String>) -> Self {
        self.endpoint = endpoint.into();
        self
    }

    /// DuckDuckGo region code such as `us-en` or `de-de`.
    pub fn with_region(mut self, region: impl Into<String>) -> Self {
        self.region = Some(region.into());
        self
    }
}

#[async_trait]
impl Searcher for DuckDuckGoSearcher {
    async fn search(&self, query: &str, max_results: usize, timeout: Duration) -> Result<Vec<SearchHit>, WebSearchError> {
        let mut form = vec![("q", query.to_string()), ("b", String::new())];
        if let Some(region) = &self.region {
            form.push(("kl", region.clone()));
        }
        let response = self
            .client
            .post(&self.endpoint)
            .header(reqwest::header::USER_AGENT, crate::extract::DEFAULT_USER_AGENT)
            .header(reqwest::header::ACCEPT, "text/html")
            .form(&form)
            .timeout(timeout)
            .send()
            .await
            .map_err(|e| WebSearchError::Request(e.without_url().to_string()))?;
        let status = response.status();
        if !status.is_success() {
            return Err(WebSearchError::Status(status.as_u16()));
        }
        let html = response.text().await.map_err(|e| WebSearchError::Request(e.without_url().to_string()))?;
        let mut hits = parse_duckduckgo_html(&html)?;
        hits.truncate(max_results);
        Ok(hits)
    }
}

/// Parse a DuckDuckGo HTML result page into hits. Sponsored entries are
/// skipped and redirect links are unwrapped to their target.
pub fn parse_duckduckgo_html(html: &str) -> Result<Vec<SearchHit>, WebSearchError> {
    let document = Html::parse_document(html);
    let result = Selector::parse("div.result").expect("valid selector");
    let link = Selector::parse("a.result__a").expect("valid selector");
    let snippet = Selector::parse(".result__snippet").expect("valid selector");
    let no_results = Selector::parse(".no-results").expect("valid selector");
    let challenge = Selector::parse("form#challenge-form, .anomaly-modal__title").expect("valid selector");

    let mut hits = Vec::new();
    for entry in document.select(&result) {
        if entry.value().classes().any(|c| c == "result--ad") {
            continue;
        }
        let Some(anchor) = entry.select(&link).next() else {
            continue;
        };
        let Some(href) = anchor.value().attr("href").and_then(unwrap_duckduckgo_link) else {
            continue;
        };
        let title = collapse_whitespace(&anchor.text().collect::<String>());
        let body =
            entry.select(&snippet).next().map(|s| collapse_whitespace(&s.text().collect::<String>())).unwrap_or_default();
        hits.push(SearchHit { title, href, body });
    }

    if hits.is_empty() {
        if document.select(&challenge).next().is_some() {
            return Err(WebSearchError::Parse("the search service asked for a bot check".to_string()));
        }
        if document.select(&no_results).next().is_none() && document.select(&result).next().is_none() {
            return Err(WebSearchError::Parse("no result list in the response".to_string()));
        }
    }
    Ok(hits)
}

/// Resolve DuckDuckGo's `/l/?uddg=<target>` redirect links. Ad click-through
/// links (`/y.js`) give `None`.
fn unwrap_duckduckgo_link(href: &str) -> Option<String> {
    let absolute = if let Some(rest) = href.strip_prefix("//") {
        format!("https://{rest}")
    } else if href.starts_with('/') {
        format!("https://duckduckgo.com{href}")
    } else {
        href.to_string()
    };
    let parsed = Url::parse(&absolute).ok()?;
    let is_ddg = parsed.host_str().is_some_and(|h| h == "duckduckgo.com" || h.ends_with(".duckduckgo.com"));
    if is_ddg {
        if parsed.path().starts_with("/y.js") {
            return None;
        }
        if parsed.path().starts_with("/l/") {
            return parsed.query_pairs().find(|(key, _)| key == "uddg").map(|(_, value)| value.into_owned());
        }
    }
    Some(absolute)
}


// ---------------------------------------------------------------------------
// More engines
// ---------------------------------------------------------------------------

/// Default Bing results page.
pub const BING_SEARCH_URL: &str = "https://www.bing.com/search";
/// Default Brave Search results page.
pub const BRAVE_SEARCH_URL: &str = "https://search.brave.com/search";
/// Default Wikipedia search API (English).
pub const WIKIPEDIA_API_URL: &str = "https://en.wikipedia.org/w/api.php";

/// GET `endpoint` with `query` and read the body as HTML, mapping failures to
/// [`WebSearchError`].
async fn fetch_page(
    client: &reqwest::Client,
    endpoint: &str,
    query: &[(&str, String)],
    timeout: Duration,
) -> Result<String, WebSearchError> {
    let response = client
        .get(endpoint)
        .header(reqwest::header::USER_AGENT, crate::extract::DEFAULT_USER_AGENT)
        .header(reqwest::header::ACCEPT, "text/html,application/xhtml+xml")
        .header(reqwest::header::ACCEPT_LANGUAGE, "en-US,en;q=0.9")
        .query(query)
        .timeout(timeout)
        .send()
        .await
        .map_err(|e| WebSearchError::Request(e.without_url().to_string()))?;
    let status = response.status();
    if !status.is_success() {
        return Err(WebSearchError::Status(status.as_u16()));
    }
    response.text().await.map_err(|e| WebSearchError::Request(e.without_url().to_string()))
}

/// Bing web search over its normal results page.
#[derive(Debug, Clone)]
pub struct BingSearcher {
    client: reqwest::Client,
    endpoint: String,
}

impl BingSearcher {
    pub fn new(client: reqwest::Client) -> Self {
        Self { client, endpoint: BING_SEARCH_URL.to_string() }
    }

    pub fn with_endpoint(mut self, endpoint: impl Into<String>) -> Self {
        self.endpoint = endpoint.into();
        self
    }
}

#[async_trait]
impl Searcher for BingSearcher {
    async fn search(&self, query: &str, max_results: usize, timeout: Duration) -> Result<Vec<SearchHit>, WebSearchError> {
        let params = [("q", query.to_string()), ("count", max_results.clamp(1, 50).to_string())];
        let html = fetch_page(&self.client, &self.endpoint, &params, timeout).await?;
        let mut hits = parse_bing_html(&html)?;
        hits.truncate(max_results);
        Ok(hits)
    }
}

/// Parse a Bing results page. Bing's `/ck/a?...&u=a1<base64url>` click
/// links are unwrapped to their target.
pub fn parse_bing_html(html: &str) -> Result<Vec<SearchHit>, WebSearchError> {
    let document = Html::parse_document(html);
    let result = Selector::parse("li.b_algo").expect("valid selector");
    let link = Selector::parse("h2 a").expect("valid selector");
    let snippet = Selector::parse(".b_caption p, p.b_lineclamp2, p.b_lineclamp3, p.b_lineclamp4, .b_algoSlug").expect("valid selector");
    let results_list = Selector::parse("#b_results").expect("valid selector");
    let challenge = Selector::parse("#b_captcha, form[action*='captcha'], .captcha").expect("valid selector");

    let mut hits = Vec::new();
    for entry in document.select(&result) {
        let Some(anchor) = entry.select(&link).next() else {
            continue;
        };
        let Some(href) = anchor.value().attr("href").and_then(unwrap_bing_link) else {
            continue;
        };
        let title = collapse_whitespace(&anchor.text().collect::<String>());
        let body =
            entry.select(&snippet).next().map(|s| collapse_whitespace(&s.text().collect::<String>())).unwrap_or_default();
        hits.push(SearchHit { title, href, body });
    }
    if hits.is_empty() {
        if document.select(&challenge).next().is_some() {
            return Err(WebSearchError::Parse("the search service asked for a bot check".to_string()));
        }
        if document.select(&results_list).next().is_none() {
            return Err(WebSearchError::Parse("no result list in the response".to_string()));
        }
    }
    Ok(hits)
}

/// Resolve Bing's click-tracking links. `u` holds `a1` plus the target URL in
/// unpadded URL-safe base64. Other links are returned as they are.
fn unwrap_bing_link(href: &str) -> Option<String> {
    use base64::Engine as _;
    let parsed = Url::parse(href).ok()?;
    let is_bing = parsed.host_str().is_some_and(|h| h == "bing.com" || h.ends_with(".bing.com"));
    if !is_bing {
        return Some(href.to_string());
    }
    if !parsed.path().starts_with("/ck/") {
        return None;
    }
    let encoded = parsed.query_pairs().find(|(key, _)| key == "u").map(|(_, value)| value.into_owned())?;
    let encoded = encoded.strip_prefix("a1").unwrap_or(&encoded);
    let bytes = base64::engine::general_purpose::URL_SAFE_NO_PAD.decode(encoded.trim_end_matches('=')).ok()?;
    let target = String::from_utf8(bytes).ok()?;
    target.starts_with("http").then_some(target)
}

/// Brave Search over its normal results page.
#[derive(Debug, Clone)]
pub struct BraveSearcher {
    client: reqwest::Client,
    endpoint: String,
}

impl BraveSearcher {
    pub fn new(client: reqwest::Client) -> Self {
        Self { client, endpoint: BRAVE_SEARCH_URL.to_string() }
    }

    pub fn with_endpoint(mut self, endpoint: impl Into<String>) -> Self {
        self.endpoint = endpoint.into();
        self
    }
}

#[async_trait]
impl Searcher for BraveSearcher {
    async fn search(&self, query: &str, max_results: usize, timeout: Duration) -> Result<Vec<SearchHit>, WebSearchError> {
        let params = [("q", query.to_string()), ("source", "web".to_string())];
        let html = fetch_page(&self.client, &self.endpoint, &params, timeout).await?;
        let mut hits = parse_brave_html(&html)?;
        hits.truncate(max_results);
        Ok(hits)
    }
}

/// Parse a Brave Search results page (web results only).
pub fn parse_brave_html(html: &str) -> Result<Vec<SearchHit>, WebSearchError> {
    let document = Html::parse_document(html);
    let result = Selector::parse("div.snippet[data-type='web']").expect("valid selector");
    let link = Selector::parse("a[href]").expect("valid selector");
    let title = Selector::parse(".title, .search-snippet-title").expect("valid selector");
    let snippet = Selector::parse(".generic-snippet .content, .snippet-description").expect("valid selector");
    let page_marker = Selector::parse("#results, #mixed-main, main").expect("valid selector");
    let challenge = Selector::parse("form#challenge-form, .captcha, #pow-captcha").expect("valid selector");

    let mut hits = Vec::new();
    for entry in document.select(&result) {
        let Some(href) = entry
            .select(&link)
            .filter_map(|a| a.value().attr("href"))
            .find(|h| h.starts_with("http") && !domain_of(h).ends_with("brave.com"))
        else {
            continue;
        };
        let title = entry.select(&title).next().map(|t| collapse_whitespace(&t.text().collect::<String>())).unwrap_or_default();
        let body =
            entry.select(&snippet).next().map(|s| collapse_whitespace(&s.text().collect::<String>())).unwrap_or_default();
        hits.push(SearchHit { title, href: href.to_string(), body });
    }
    if hits.is_empty() {
        if document.select(&challenge).next().is_some() {
            return Err(WebSearchError::Parse("the search service asked for a bot check".to_string()));
        }
        if document.select(&page_marker).next().is_none() {
            return Err(WebSearchError::Parse("no result list in the response".to_string()));
        }
    }
    Ok(hits)
}

/// Wikipedia's own search API: always available and keyless, but only ever
/// returns Wikipedia articles, so it is the last resort.
#[derive(Debug, Clone)]
pub struct WikipediaSearcher {
    client: reqwest::Client,
    endpoint: String,
}

impl WikipediaSearcher {
    pub fn new(client: reqwest::Client) -> Self {
        Self { client, endpoint: WIKIPEDIA_API_URL.to_string() }
    }

    pub fn with_endpoint(mut self, endpoint: impl Into<String>) -> Self {
        self.endpoint = endpoint.into();
        self
    }
}

#[async_trait]
impl Searcher for WikipediaSearcher {
    async fn search(&self, query: &str, max_results: usize, timeout: Duration) -> Result<Vec<SearchHit>, WebSearchError> {
        let params = [
            ("action", "query".to_string()),
            ("list", "search".to_string()),
            ("srsearch", query.to_string()),
            ("srlimit", max_results.clamp(1, 50).to_string()),
            ("format", "json".to_string()),
        ];
        let body = fetch_page(&self.client, &self.endpoint, &params, timeout).await?;
        parse_wikipedia_json(&body, &self.endpoint)
    }
}

/// Parse a Wikipedia `list=search` response into article links on the same
/// wiki as `endpoint`.
pub fn parse_wikipedia_json(body: &str, endpoint: &str) -> Result<Vec<SearchHit>, WebSearchError> {
    let json: serde_json::Value =
        serde_json::from_str(body).map_err(|e| WebSearchError::Parse(format!("not a search response: {e}")))?;
    let results = json
        .pointer("/query/search")
        .and_then(|v| v.as_array())
        .ok_or_else(|| WebSearchError::Parse("no result list in the response".to_string()))?;
    let base = Url::parse(endpoint)
        .ok()
        .and_then(|u| u.host_str().map(|h| format!("{}://{h}", u.scheme())))
        .unwrap_or_else(|| "https://en.wikipedia.org".to_string());
    let tags = regex::Regex::new(r"<[^>]+>").expect("valid regex");
    Ok(results
        .iter()
        .filter_map(|r| {
            let title = r.get("title")?.as_str()?.to_string();
            let mut href = Url::parse(&format!("{base}/wiki/")).ok()?;
            href.path_segments_mut().ok()?.pop().push(&title.replace(' ', "_"));
            let snippet = r.get("snippet").and_then(|s| s.as_str()).unwrap_or_default();
            let body = collapse_whitespace(&decode_entities(&tags.replace_all(snippet, "")));
            Some(SearchHit { title, href: href.to_string(), body })
        })
        .collect())
}

/// Words of a query worth matching: longer than three letters and not a
/// filler word. Lower case, without punctuation.
fn query_terms(query: &str) -> Vec<String> {
    const FILLER: &[&str] = &[
        "what", "when", "where", "which", "while", "with", "within", "without", "does", "about", "from", "into", "that",
        "this", "these", "those", "there", "their", "they", "have", "will", "would", "should", "could", "your", "best",
        "overview", "official", "source", "latest", "news", "research", "analysis", "criticism", "controversy",
    ];
    let mut terms: Vec<String> = query
        .split(|c: char| !c.is_alphanumeric())
        .map(str::to_lowercase)
        .filter(|w| w.chars().count() > 3 && !FILLER.contains(&w.as_str()))
        .collect();
    terms.dedup();
    terms
}

/// Whether at least half the hits mention two of the query's words (or its
/// only word) in their title or snippet. Word stems are matched, so "pumps"
/// matches "pump".
fn mostly_relevant(query: &str, hits: &[SearchHit]) -> bool {
    let terms = query_terms(query);
    if terms.len() < 2 {
        return true;
    }
    let stems: Vec<String> = terms.iter().map(|t| t.trim_end_matches('s').chars().take(6).collect()).collect();
    let relevant = hits
        .iter()
        .filter(|hit| {
            let text = format!("{} {}", hit.title, hit.body).to_lowercase();
            stems.iter().filter(|stem| text.contains(stem.as_str())).count() >= 2
        })
        .count();
    relevant * 2 >= hits.len()
}

fn decode_entities(text: &str) -> String {
    text.replace("&quot;", "\"").replace("&#039;", "'").replace("&lt;", "<").replace("&gt;", ">").replace("&amp;", "&")
}

/// A self-hosted SearXNG instance, through its JSON API
/// (`GET {base}/search?q=...&format=json`). The instance must allow the
/// `json` format (`search.formats` in its `settings.yml`).
#[derive(Debug, Clone)]
pub struct SearxngSearcher {
    client: reqwest::Client,
    base_url: String,
}

impl SearxngSearcher {
    /// `base_url` is the instance root, for example `https://searxng.example.ts.net`.
    pub fn new(client: reqwest::Client, base_url: impl Into<String>) -> Self {
        Self { client, base_url: base_url.into().trim_end_matches('/').to_string() }
    }
}

#[async_trait]
impl Searcher for SearxngSearcher {
    async fn search(&self, query: &str, max_results: usize, timeout: Duration) -> Result<Vec<SearchHit>, WebSearchError> {
        let params = [("q", query.to_string()), ("format", "json".to_string()), ("categories", "general".to_string())];
        let body = fetch_page(&self.client, &format!("{}/search", self.base_url), &params, timeout).await?;
        let mut hits = parse_searxng_json(&body)?;
        hits.truncate(max_results);
        Ok(hits)
    }
}

/// Parse a SearXNG JSON response. A 403 from SearXNG (handled as a status
/// error before this) usually means the `json` format is not enabled.
pub fn parse_searxng_json(body: &str) -> Result<Vec<SearchHit>, WebSearchError> {
    let json: serde_json::Value =
        serde_json::from_str(body).map_err(|e| WebSearchError::Parse(format!("not a SearXNG JSON response: {e}")))?;
    let results = json
        .get("results")
        .and_then(|v| v.as_array())
        .ok_or_else(|| WebSearchError::Parse("no result list in the response".to_string()))?;
    Ok(results
        .iter()
        .filter_map(|r| {
            let href = r.get("url")?.as_str()?.trim().to_string();
            href.starts_with("http").then(|| SearchHit {
                title: collapse_whitespace(r.get("title").and_then(|t| t.as_str()).unwrap_or_default()),
                href,
                body: collapse_whitespace(r.get("content").and_then(|t| t.as_str()).unwrap_or_default()),
            })
        })
        .collect())
}

/// Tries several engines for each query, like `ddgs`'s automatic backend.
///
/// Engines are tried in order, starting one further along for each query so
/// the load is spread; the last engine is only a fallback. An engine is
/// skipped for the rest of this searcher's life (one discovery run) when it
/// answers with a bot check, an error page, or results that mostly don't
/// mention the query (Bing does this to automated requests: it answers for
/// the first word only). A rate limit (HTTP 429) only skips that engine for
/// the current query.
pub struct MultiSearcher {
    /// Tried first, in order, for every query (a self-hosted SearXNG).
    primary: Vec<(&'static str, Box<dyn Searcher>)>,
    engines: Vec<(&'static str, Box<dyn Searcher>)>,
    benched: std::sync::Mutex<HashSet<&'static str>>,
    next: std::sync::atomic::AtomicUsize,
}

impl MultiSearcher {
    pub fn new(engines: Vec<(&'static str, Box<dyn Searcher>)>) -> Self {
        Self {
            primary: Vec::new(),
            engines,
            benched: std::sync::Mutex::new(HashSet::new()),
            next: std::sync::atomic::AtomicUsize::new(0),
        }
    }

    /// Put `engine` in front of the others: it is tried first for every query
    /// (unless it gets benched), and the rest are only used when it fails.
    pub fn with_primary(mut self, name: &'static str, engine: Box<dyn Searcher>) -> Self {
        self.primary.push((name, engine));
        self
    }

    /// DuckDuckGo, Bing and Brave, then Wikipedia as the last resort.
    pub fn with_default_engines(client: reqwest::Client) -> Self {
        Self::new(vec![
            ("duckduckgo", Box::new(DuckDuckGoSearcher::new(client.clone())) as Box<dyn Searcher>),
            ("bing", Box::new(BingSearcher::new(client.clone()))),
            ("brave", Box::new(BraveSearcher::new(client.clone()))),
            ("wikipedia", Box::new(WikipediaSearcher::new(client))),
        ])
    }

    fn is_benched(&self, name: &str) -> bool {
        self.benched.lock().map(|b| b.contains(name)).unwrap_or(false)
    }

    fn bench(&self, name: &'static str) {
        if let Ok(mut benched) = self.benched.lock() {
            benched.insert(name);
        }
    }
}

#[async_trait]
impl Searcher for MultiSearcher {
    async fn search(&self, query: &str, max_results: usize, timeout: Duration) -> Result<Vec<SearchHit>, WebSearchError> {
        if self.engines.is_empty() && self.primary.is_empty() {
            return Err(WebSearchError::Request("no search engines configured".to_string()));
        }
        // Primary engines first, in order. Then the rest: the last one
        // (Wikipedia by default) is a fallback, the others take turns.
        let rotating = self.engines.len().saturating_sub(1).max(1);
        let start = self.next.fetch_add(1, std::sync::atomic::Ordering::Relaxed) % rotating;
        let order = (0..rotating).map(|i| (start + i) % rotating).chain(rotating..self.engines.len());
        let candidates = self.primary.iter().chain(order.filter_map(|index| self.engines.get(index)));
        let mut last_error = None;
        for (name, engine) in candidates {
            if self.is_benched(name) {
                continue;
            }
            match engine.search(query, max_results, timeout).await {
                Ok(hits) if hits.is_empty() => tracing::debug!("{name} found nothing for {query:?}; trying the next engine"),
                Ok(hits) if !mostly_relevant(query, &hits) => {
                    tracing::info!("{name} results don't match the query; not using {name} for this run");
                    self.bench(name);
                    last_error = Some(WebSearchError::Parse(format!("{name} returned results unrelated to the query")));
                }
                Ok(hits) => return Ok(hits),
                Err(e) => {
                    tracing::info!("{name} search failed ({e}); trying the next engine");
                    if !matches!(e, WebSearchError::Status(429) | WebSearchError::Request(_)) {
                        self.bench(name);
                    }
                    last_error = Some(e);
                }
            }
        }
        match last_error {
            Some(e) => Err(e),
            None => Ok(Vec::new()),
        }
    }
}

fn collapse_whitespace(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

#[cfg(test)]
mod tests {
    use super::*;
    use wiremock::matchers::{body_string_contains, method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    fn sr(url: &str) -> SearchResult {
        SearchResult {
            url: url.to_string(),
            title: "t".to_string(),
            snippet: "snip".to_string(),
            domain: domain_of(url),
            query_index: 0,
            rank: 0,
        }
    }

    #[test]
    fn url_filters_and_normalize() {
        assert!(!is_allowed_url("ftp://example.com/a"));
        assert!(!is_allowed_url("https://www.youtube.com/watch?v=1"));
        assert!(!is_allowed_url("https://m.facebook.com/x"));
        assert!(!is_allowed_url("https://example.com/paper.PDF"));
        assert!(!is_allowed_url("not a url"));
        assert!(is_allowed_url("https://example.org/a"));
        // A blocked name must match a whole label run, not a suffix of one.
        assert!(is_allowed_url("https://notreddit.com/a"));
        assert!(is_allowed_url("https://netflix.com/a"));
        assert_eq!(normalize_url("https://www.Example.com/a/#frag"), "https://example.com/a");
        assert_eq!(normalize_url("HTTP://Example.com:8080/a/b/?q=1#x"), "http://example.com:8080/a/b?q=1");
        assert_eq!(normalize_url("https://example.com:443/"), "https://example.com");
        assert_eq!(domain_of("https://WWW.Example.com/x"), "example.com");
        assert_eq!(domain_of("nonsense"), "");
    }

    #[test]
    fn dedupe_and_domain_limit() {
        let results = vec![
            sr("https://a.com/1"),
            sr("https://www.a.com/1/"),
            sr("https://a.com/2"),
            sr("https://a.com/3"),
            sr("https://b.com/1"),
            sr("https://youtube.com/watch?v=1"),
        ];
        let urls: Vec<String> = dedupe_and_limit(&results, 2).into_iter().map(|r| r.url).collect();
        assert_eq!(urls, vec!["https://a.com/1", "https://a.com/2", "https://b.com/1"]);
    }

    struct FakeSearcher;

    #[async_trait]
    impl Searcher for FakeSearcher {
        async fn search(&self, query: &str, _max: usize, _timeout: Duration) -> Result<Vec<SearchHit>, WebSearchError> {
            if query == "bad" {
                return Err(WebSearchError::Status(429));
            }
            Ok(vec![
                SearchHit { href: format!("https://{query}.com/x"), title: format!(" {query} "), body: "b".into() },
                SearchHit { href: "  ".into(), title: "no url".into(), body: String::new() },
            ])
        }
    }

    #[tokio::test]
    async fn search_many_tolerates_failures() {
        let queries = vec!["ok1".to_string(), "bad".to_string(), "ok2".to_string()];
        let options = SearchOptions { delay: Duration::ZERO, ..SearchOptions::default() };
        let out = search_many(&queries, &options, &FakeSearcher).await;
        assert_eq!(out.iter().map(|r| r.domain.as_str()).collect::<Vec<_>>(), vec!["ok1.com", "ok2.com"]);
        assert_eq!(out.iter().map(|r| r.query_index).collect::<Vec<_>>(), vec![0, 2]);
        assert_eq!(out[0].title, "ok1");
        assert_eq!(out[0].rank, 0);
    }

    const RESULT_PAGE: &str = r#"<!DOCTYPE html><html><body><div id="links" class="results">
      <div class="result results_links results_links_deep result--ad">
        <h2 class="result__title"><a class="result__a" href="//duckduckgo.com/y.js?ad_domain=shop.example">Buy things</a></h2>
        <a class="result__snippet" href="x">Sponsored text</a>
      </div>
      <div class="result results_links results_links_deep web-result">
        <div class="links_main links_deep result__body">
          <h2 class="result__title">
            <a rel="nofollow" class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fwww.rust-lang.org%2Flearn%3Fa%3D1%26b%3D2&amp;rut=abc">Learn <b>Rust</b></a>
          </h2>
          <a class="result__snippet" href="//duckduckgo.com/l/?uddg=x">Get started with
             <b>Rust</b> today.</a>
        </div>
      </div>
      <div class="result results_links results_links_deep web-result">
        <h2 class="result__title"><a class="result__a" href="https://doc.rust-lang.org/book/">The Book</a></h2>
      </div>
      <div class="result results_links web-result">
        <h2 class="result__title"><a class="result__a" href="https://example.org/third">Third</a></h2>
        <a class="result__snippet">Third snippet</a>
      </div>
    </div></body></html>"#;

    #[test]
    fn parses_duckduckgo_result_page() {
        let hits = parse_duckduckgo_html(RESULT_PAGE).unwrap();
        assert_eq!(
            hits,
            vec![
                SearchHit {
                    title: "Learn Rust".into(),
                    href: "https://www.rust-lang.org/learn?a=1&b=2".into(),
                    body: "Get started with Rust today.".into(),
                },
                SearchHit { title: "The Book".into(), href: "https://doc.rust-lang.org/book/".into(), body: String::new() },
                SearchHit { title: "Third".into(), href: "https://example.org/third".into(), body: "Third snippet".into() },
            ]
        );
    }

    #[test]
    fn empty_and_unreadable_pages() {
        let none = r#"<html><body><div class="no-results">No results.</div></body></html>"#;
        assert_eq!(parse_duckduckgo_html(none).unwrap(), Vec::new());
        assert!(matches!(parse_duckduckgo_html("<html><body>hello</body></html>"), Err(WebSearchError::Parse(_))));
        let challenge = r#"<html><body><form id="challenge-form"></form></body></html>"#;
        assert!(matches!(parse_duckduckgo_html(challenge), Err(WebSearchError::Parse(_))));
    }

    #[tokio::test]
    async fn duckduckgo_searcher_posts_query_and_limits_results() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/html/"))
            .and(body_string_contains("q=rust+language"))
            .respond_with(ResponseTemplate::new(200).set_body_raw(RESULT_PAGE, "text/html; charset=utf-8"))
            .mount(&server)
            .await;
        Mock::given(method("POST"))
            .and(path("/limited/"))
            .respond_with(ResponseTemplate::new(202))
            .mount(&server)
            .await;
        Mock::given(method("POST"))
            .and(path("/blocked/"))
            .respond_with(ResponseTemplate::new(429))
            .mount(&server)
            .await;

        let searcher =
            DuckDuckGoSearcher::new(reqwest::Client::new()).with_endpoint(format!("{}/html/", server.uri()));
        let hits = searcher.search("rust language", 2, Duration::from_secs(5)).await.unwrap();
        assert_eq!(hits.len(), 2);
        assert_eq!(hits[0].href, "https://www.rust-lang.org/learn?a=1&b=2");

        let results = search_many(&["rust language".to_string()], &SearchOptions::default(), &searcher).await;
        assert_eq!(results.len(), 3);
        assert_eq!(results[1].domain, "doc.rust-lang.org");
        assert_eq!(results[2].rank, 2);

        let blocked =
            DuckDuckGoSearcher::new(reqwest::Client::new()).with_endpoint(format!("{}/blocked/", server.uri()));
        assert!(matches!(
            blocked.search("x", 5, Duration::from_secs(5)).await,
            Err(WebSearchError::Status(429))
        ));
        // DuckDuckGo answers 202 with a challenge page when it rate-limits.
        let limited =
            DuckDuckGoSearcher::new(reqwest::Client::new()).with_endpoint(format!("{}/limited/", server.uri()));
        assert!(limited.search("x", 5, Duration::from_secs(5)).await.is_err());
    }

    // --- More engines -------------------------------------------------------

    #[test]
    fn parses_bing_results_and_unwraps_click_links() {
        use base64::Engine as _;
        let target = "https://example.org/heat-pumps";
        let encoded = base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(target);
        let page = format!(
            r#"<html><body><ol id="b_results">
              <li class="b_algo"><h2><a href="https://www.bing.com/ck/a?!&amp;&amp;p=abc&amp;u=a1{encoded}&amp;ntb=1"><strong>Heat</strong> pumps</a></h2>
                <div class="b_caption"><p class="b_lineclamp3">How they   work in the cold.</p></div></li>
              <li class="b_algo"><h2><a href="https://direct.example.com/page">Direct link</a></h2></li>
              <li class="b_algo"><h2><a href="https://www.bing.com/search?q=related">Related search</a></h2></li>
            </ol></body></html>"#
        );
        let hits = parse_bing_html(&page).unwrap();
        assert_eq!(hits.len(), 2, "{hits:?}");
        assert_eq!(hits[0], SearchHit { title: "Heat pumps".into(), href: target.into(), body: "How they work in the cold.".into() });
        assert_eq!(hits[1].href, "https://direct.example.com/page");

        let captcha = r#"<html><body><div id="b_captcha">Verify</div></body></html>"#;
        assert!(matches!(parse_bing_html(captcha), Err(WebSearchError::Parse(m)) if m.contains("bot check")));
        let empty = r#"<html><body><ol id="b_results"></ol></body></html>"#;
        assert!(parse_bing_html(empty).unwrap().is_empty());
    }

    #[test]
    fn parses_brave_web_results_only() {
        let page = r#"<html><body><main><div id="results">
          <div class="snippet" data-type="web"><a href="https://example.org/a"><div class="title">First   result</div></a>
            <div class="generic-snippet"><div class="content">Snippet one</div></div></div>
          <div class="snippet" data-type="news"><a href="https://news.example.com/x"><div class="title">News</div></a></div>
          <div class="snippet" data-type="web"><a href="https://search.brave.com/internal">Internal</a></div>
        </div></main></body></html>"#;
        let hits = parse_brave_html(page).unwrap();
        assert_eq!(hits, vec![SearchHit { title: "First result".into(), href: "https://example.org/a".into(), body: "Snippet one".into() }]);
        assert!(parse_brave_html("<html><body><form id=\"challenge-form\"></form></body></html>").is_err());
    }

    #[test]
    fn parses_wikipedia_search_json() {
        let body = r#"{"query":{"search":[{"title":"Air source heat pump","snippet":"Air-to-air <span class=\"searchmatch\">heat</span> pumps &amp; more"}]}}"#;
        let hits = parse_wikipedia_json(body, "https://de.wikipedia.org/w/api.php").unwrap();
        assert_eq!(
            hits,
            vec![SearchHit {
                title: "Air source heat pump".into(),
                href: "https://de.wikipedia.org/wiki/Air_source_heat_pump".into(),
                body: "Air-to-air heat pumps & more".into(),
            }]
        );
        assert!(parse_wikipedia_json("not json", WIKIPEDIA_API_URL).is_err());
    }

    /// An engine that always gives the same answer and counts its calls.
    struct Scripted {
        answer: Result<Vec<SearchHit>, WebSearchError>,
        calls: std::sync::Arc<std::sync::atomic::AtomicUsize>,
    }

    #[async_trait]
    impl Searcher for Scripted {
        async fn search(&self, _q: &str, _max: usize, _timeout: Duration) -> Result<Vec<SearchHit>, WebSearchError> {
            self.calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            self.answer.clone()
        }
    }

    fn scripted(answer: Result<Vec<SearchHit>, WebSearchError>) -> (Box<dyn Searcher>, std::sync::Arc<std::sync::atomic::AtomicUsize>) {
        let calls = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        (Box::new(Scripted { answer, calls: calls.clone() }), calls)
    }

    fn hit(url: &str) -> SearchHit {
        SearchHit { title: "Heat pumps in cold climates".into(), href: url.into(), body: String::new() }
    }

    #[tokio::test]
    async fn multi_searcher_falls_back_and_benches_blocked_engines() {
        let (blocked, blocked_calls) = scripted(Err(WebSearchError::Parse("the search service asked for a bot check".into())));
        let (working, working_calls) = scripted(Ok(vec![hit("https://b.example/1")]));
        let (fallback, fallback_calls) = scripted(Ok(vec![hit("https://wiki.example/1")]));
        let multi = MultiSearcher::new(vec![("blocked", blocked), ("working", working), ("fallback", fallback)]);
        let t = Duration::from_secs(1);

        for _ in 0..4 {
            let hits = multi.search("heat pumps cold climates", 5, t).await.unwrap();
            assert_eq!(hits[0].href, "https://b.example/1");
        }
        assert_eq!(blocked_calls.load(std::sync::atomic::Ordering::SeqCst), 1, "a blocked engine is tried once, then skipped");
        assert_eq!(working_calls.load(std::sync::atomic::Ordering::SeqCst), 4);
        assert_eq!(fallback_calls.load(std::sync::atomic::Ordering::SeqCst), 0, "the last engine is only a fallback");
    }

    #[tokio::test]
    async fn multi_searcher_uses_the_fallback_and_reports_the_last_error() {
        let (empty, _) = scripted(Ok(Vec::new()));
        let (limited, _) = scripted(Err(WebSearchError::Status(429)));
        let (fallback, _) = scripted(Ok(vec![hit("https://wiki.example/1")]));
        let multi = MultiSearcher::new(vec![("empty", empty), ("limited", limited), ("fallback", fallback)]);
        assert_eq!(multi.search("heat pumps cold climates", 5, Duration::from_secs(1)).await.unwrap()[0].href, "https://wiki.example/1");

        let (down, _) = scripted(Err(WebSearchError::Request("connection refused".into())));
        let (also_down, _) = scripted(Err(WebSearchError::Status(503)));
        let multi = MultiSearcher::new(vec![("down", down), ("also_down", also_down)]);
        assert!(multi.search("heat pumps cold climates", 5, Duration::from_secs(1)).await.is_err());
        assert!(MultiSearcher::new(Vec::new()).search("q", 5, Duration::from_secs(1)).await.is_err());
    }

    #[tokio::test]
    async fn multi_searcher_skips_engines_with_unrelated_results() {
        let film = SearchHit { title: "Heat (1995 film) - Wikipedia".into(), href: "https://w.example/heat".into(), body: "A crime drama".into() };
        let (degraded, degraded_calls) = scripted(Ok(vec![film.clone(), film]));
        let (good, _) = scripted(Ok(vec![hit("https://good.example/1")]));
        let (fallback, _) = scripted(Ok(vec![hit("https://wiki.example/1")]));
        let multi = MultiSearcher::new(vec![("degraded", degraded), ("good", good), ("fallback", fallback)]);
        let t = Duration::from_secs(1);
        for _ in 0..3 {
            assert_eq!(multi.search("How do heat pumps work in cold climates?", 5, t).await.unwrap()[0].href, "https://good.example/1");
        }
        assert_eq!(degraded_calls.load(std::sync::atomic::Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn rate_limits_skip_only_the_current_query() {
        let (limited, limited_calls) = scripted(Err(WebSearchError::Status(429)));
        let (good, _) = scripted(Ok(vec![hit("https://good.example/1")]));
        let (fallback, _) = scripted(Ok(vec![hit("https://wiki.example/1")]));
        let multi = MultiSearcher::new(vec![("limited", limited), ("good", good), ("fallback", fallback)]);
        for _ in 0..4 {
            multi.search("heat pumps cold climates", 5, Duration::from_secs(1)).await.unwrap();
        }
        assert_eq!(limited_calls.load(std::sync::atomic::Ordering::SeqCst), 2, "tried again on later queries (it starts half of them)");
    }

    #[test]
    fn relevance_uses_meaningful_query_words() {
        assert_eq!(query_terms("How do heat pumps work in cold climates? overview"), ["heat", "pumps", "work", "cold", "climates"]);
        let on_topic = [hit("https://a.example")];
        assert!(mostly_relevant("heat pumps cold climates", &on_topic));
        let off_topic = [SearchHit { title: "DO | English meaning".into(), href: "https://d.example".into(), body: "verb".into() }];
        assert!(!mostly_relevant("How do heat pumps work in cold climates?", &off_topic));
        assert!(mostly_relevant("rust", &off_topic), "a one-word query cannot be judged");
    }

    #[test]
    fn parses_searxng_json() {
        let body = r#"{"query":"heat pumps","results":[
            {"url":"https://example.org/a","title":"Heat  pumps","content":"How they work"},
            {"url":"ftp://bad.example/x","title":"Not http"},
            {"title":"No url"}]}"#;
        assert_eq!(
            parse_searxng_json(body).unwrap(),
            vec![SearchHit { title: "Heat pumps".into(), href: "https://example.org/a".into(), body: "How they work".into() }]
        );
        assert!(parse_searxng_json("<html>").is_err());
        assert!(parse_searxng_json(r#"{"query":"x"}"#).is_err());
    }

    #[tokio::test]
    async fn searxng_is_asked_for_json_and_goes_first() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/search"))
            .and(wiremock::matchers::query_param("q", "heat pumps cold climates"))
            .and(wiremock::matchers::query_param("format", "json"))
            .respond_with(ResponseTemplate::new(200).set_body_string(
                r#"{"results":[{"url":"https://searx.example/1","title":"Heat pumps in cold climates","content":""}]}"#,
            ))
            .mount(&server)
            .await;
        let (rotating, rotating_calls) = scripted(Ok(vec![hit("https://other.example/1")]));
        let (fallback, _) = scripted(Ok(vec![hit("https://wiki.example/1")]));
        let multi = MultiSearcher::new(vec![("rotating", rotating), ("fallback", fallback)])
            .with_primary("searxng", Box::new(SearxngSearcher::new(reqwest::Client::new(), format!("{}/", server.uri()))));
        for _ in 0..3 {
            let hits = multi.search("heat pumps cold climates", 5, Duration::from_secs(5)).await.unwrap();
            assert_eq!(hits[0].href, "https://searx.example/1");
        }
        assert_eq!(rotating_calls.load(std::sync::atomic::Ordering::SeqCst), 0, "the others are only a fallback");

        // SearXNG refusing JSON (403) benches it and the others take over.
        let refusing = MockServer::start().await;
        Mock::given(method("GET")).respond_with(ResponseTemplate::new(403)).mount(&refusing).await;
        let (rotating, _) = scripted(Ok(vec![hit("https://other.example/1")]));
        let (fallback, _) = scripted(Ok(vec![hit("https://wiki.example/1")]));
        let multi = MultiSearcher::new(vec![("rotating", rotating), ("fallback", fallback)])
            .with_primary("searxng", Box::new(SearxngSearcher::new(reqwest::Client::new(), refusing.uri())));
        let hits = multi.search("heat pumps cold climates", 5, Duration::from_secs(5)).await.unwrap();
        assert_eq!(hits[0].href, "https://other.example/1");
    }

    #[tokio::test]
    async fn bing_and_wikipedia_searchers_send_the_query() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/bing"))
            .and(wiremock::matchers::query_param("q", "heat pumps"))
            .respond_with(ResponseTemplate::new(200).set_body_string(
                r#"<html><body><ol id="b_results"><li class="b_algo"><h2><a href="https://example.org/x">X</a></h2></li></ol></body></html>"#,
            ))
            .mount(&server)
            .await;
        Mock::given(method("GET"))
            .and(path("/w/api.php"))
            .and(wiremock::matchers::query_param("srsearch", "heat pumps"))
            .respond_with(ResponseTemplate::new(200).set_body_string(r#"{"query":{"search":[{"title":"Heat pump","snippet":""}]}}"#))
            .mount(&server)
            .await;
        let client = reqwest::Client::new();
        let t = Duration::from_secs(5);
        let bing = BingSearcher::new(client.clone()).with_endpoint(format!("{}/bing", server.uri()));
        assert_eq!(bing.search("heat pumps", 5, t).await.unwrap()[0].href, "https://example.org/x");
        let wiki = WikipediaSearcher::new(client).with_endpoint(format!("{}/w/api.php", server.uri()));
        let hits = wiki.search("heat pumps", 5, t).await.unwrap();
        assert!(hits[0].href.ends_with("/wiki/Heat_pump"), "{hits:?}");
    }
}
