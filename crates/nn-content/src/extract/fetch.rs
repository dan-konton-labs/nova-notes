//! The guarded HTTP fetcher used for user-supplied URLs.
//!
//! Every hop is validated by the SSRF guard before a connection is made, and
//! redirects are followed by hand so each target is validated too. Without a
//! proxy the connection is pinned to the addresses that were vetted, so a
//! second DNS answer cannot redirect it (DNS rebinding). Bodies are read up
//! to `Limits::max_download_bytes`.

use bytes::{Bytes, BytesMut};
use reqwest::header::{HeaderMap, ACCEPT, ACCEPT_LANGUAGE, CONTENT_DISPOSITION, CONTENT_TYPE, LOCATION};
use reqwest::{Method, StatusCode};
use url::Url;

use super::ExtractContext;
use crate::error::ContentError;
use crate::url_validation::{prepare_pinned_http_target_with, validate_url_with};

/// A downloaded response body.
#[derive(Debug, Clone)]
pub(crate) struct Fetched {
    /// The URL that produced the body, after redirects.
    pub final_url: Url,
    /// Media type without parameters, lower case (`text/html`).
    pub content_type: Option<String>,
    /// The `charset` parameter of the content type.
    pub charset: Option<String>,
    /// File name from `Content-Disposition`.
    pub file_name: Option<String>,
    pub body: Bytes,
}

/// GET `url` and read the body, enforcing the guard, the redirect limit and
/// the size limit. HTTP error statuses become typed errors.
pub(crate) async fn get(url: &Url, ctx: &ExtractContext, accept: &str) -> Result<Fetched, ContentError> {
    let (final_url, mut response) = send(Method::GET, url, ctx, accept).await?;
    let status = response.status();
    if !status.is_success() {
        return Err(status_error(status, &final_url));
    }

    let limit = ctx.limits.max_download_bytes;
    if response.content_length().is_some_and(|length| length > limit) {
        return Err(ContentError::TooLarge { limit });
    }
    let (content_type, charset) = content_type_of(response.headers());
    let file_name = disposition_file_name(response.headers());

    let mut body = BytesMut::new();
    while let Some(chunk) = response.chunk().await.map_err(network_error)? {
        if (body.len() + chunk.len()) as u64 > limit {
            return Err(ContentError::TooLarge { limit });
        }
        body.extend_from_slice(&chunk);
    }
    Ok(Fetched { final_url, content_type, charset, file_name, body: body.freeze() })
}

/// Best-effort HEAD request for the media type of `url`. Guard violations are
/// errors; anything else that goes wrong (servers that reject HEAD, timeouts)
/// is `Ok(None)`.
pub(crate) async fn head_content_type(url: &Url, ctx: &ExtractContext) -> Result<Option<String>, ContentError> {
    match send(Method::HEAD, url, ctx, "*/*").await {
        Ok((_, response)) if response.status().is_success() => Ok(content_type_of(response.headers()).0),
        Ok(_) => Ok(None),
        Err(error @ ContentError::UrlNotAllowed(_)) => Err(error),
        Err(_) => Ok(None),
    }
}

async fn send(
    method: Method,
    url: &Url,
    ctx: &ExtractContext,
    accept: &str,
) -> Result<(Url, reqwest::Response), ContentError> {
    let mut current = url.clone();
    for _ in 0..=ctx.limits.max_redirects {
        let client = client_for(&current, ctx).await?;
        let response = client
            .request(method.clone(), current.clone())
            .header(ACCEPT, accept)
            .header(ACCEPT_LANGUAGE, "en-US,en;q=0.8,*;q=0.5")
            .send()
            .await
            .map_err(network_error)?;

        if !matches!(response.status().as_u16(), 301 | 302 | 303 | 307 | 308) {
            return Ok((current, response));
        }
        let location = response
            .headers()
            .get(LOCATION)
            .and_then(|value| value.to_str().ok())
            .ok_or_else(|| ContentError::ExternalService("redirect without a Location header".to_string()))?;
        current = current
            .join(location)
            .map_err(|e| ContentError::ExternalService(format!("redirect to an unparseable location: {e}")))?;
    }
    Err(ContentError::Network(format!("more than {} redirects", ctx.limits.max_redirects)))
}

/// Validate `url` and build a client that can only reach what was validated.
async fn client_for(url: &Url, ctx: &ExtractContext) -> Result<reqwest::Client, ContentError> {
    let mut builder = reqwest::Client::builder()
        .redirect(reqwest::redirect::Policy::none())
        .connect_timeout(ctx.limits.connect_timeout)
        .timeout(ctx.limits.request_timeout)
        .user_agent(ctx.user_agent.as_str());

    match &ctx.proxy {
        Some(proxy) => {
            // The proxy resolves the name, so the address cannot be pinned;
            // the check still rejects what resolves badly from here.
            validate_url_with(url.as_str(), &ctx.url_policy, ctx.resolver.as_ref()).await?;
            let proxy = reqwest::Proxy::all(proxy.as_str())
                .map_err(|_| ContentError::Configuration("the configured proxy URL is not valid".to_string()))?;
            builder = builder.proxy(proxy);
        }
        None => {
            let target = prepare_pinned_http_target_with(url.as_str(), &ctx.url_policy, ctx.resolver.as_ref())
                .await
                .map_err(|error| {
                    if error.is_unresolved_host() {
                        ContentError::Network(error.to_string())
                    } else {
                        ContentError::UrlNotAllowed(error)
                    }
                })?;
            builder = target.apply(builder.no_proxy());
        }
    }
    builder.build().map_err(|e| ContentError::Configuration(format!("cannot build the HTTP client: {e}")))
}

pub(crate) fn network_error(error: reqwest::Error) -> ContentError {
    let kind = if error.is_timeout() {
        "request timed out"
    } else if error.is_connect() {
        "connection failed"
    } else {
        "request failed"
    };
    // The URL is dropped: it can carry credentials or tokens in its query.
    ContentError::Network(format!("{kind}: {}", error.without_url()))
}

pub(crate) fn status_error(status: StatusCode, url: &Url) -> ContentError {
    let host = url.host_str().unwrap_or("the server");
    match status.as_u16() {
        404 | 410 => ContentError::NotFound(format!("HTTP {} from {host}", status.as_u16())),
        code => ContentError::ExternalService(format!("HTTP {code} from {host}")),
    }
}

fn content_type_of(headers: &HeaderMap) -> (Option<String>, Option<String>) {
    let Some(raw) = headers.get(CONTENT_TYPE).and_then(|value| value.to_str().ok()) else {
        return (None, None);
    };
    let mut parts = raw.split(';');
    let media_type = parts.next().map(|m| m.trim().to_ascii_lowercase()).filter(|m| !m.is_empty());
    let charset = parts.find_map(|parameter| {
        let (key, value) = parameter.split_once('=')?;
        key.trim().eq_ignore_ascii_case("charset").then(|| value.trim().trim_matches('"').to_string())
    });
    (media_type, charset)
}

fn disposition_file_name(headers: &HeaderMap) -> Option<String> {
    let raw = headers.get(CONTENT_DISPOSITION)?.to_str().ok()?;
    raw.split(';').find_map(|parameter| {
        let (key, value) = parameter.split_once('=')?;
        let key = key.trim().to_ascii_lowercase();
        let value = value.trim().trim_matches('"');
        let name = match key.as_str() {
            "filename" => value,
            // RFC 5987: charset'language'percent-encoded-name
            "filename*" => value.rsplit('\'').next()?,
            _ => return None,
        };
        let name = name.rsplit(['/', '\\']).next()?.trim();
        (!name.is_empty()).then(|| name.to_string())
    })
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    use super::*;
    use crate::url_validation::UrlPolicy;

    fn url(server: &MockServer, path: &str) -> Url {
        Url::parse(&format!("{}{path}", server.uri())).unwrap()
    }

    #[tokio::test]
    async fn reads_body_and_headers() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/doc"))
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header("content-type", "Text/HTML; Charset=\"ISO-8859-1\"")
                    .insert_header("content-disposition", "attachment; filename=\"../evil/report.pdf\"")
                    .set_body_bytes(b"hello".to_vec()),
            )
            .mount(&server)
            .await;
        let fetched = get(&url(&server, "/doc"), &ExtractContext::default(), "*/*").await.unwrap();
        assert_eq!(&fetched.body[..], b"hello");
        assert_eq!(fetched.content_type.as_deref(), Some("text/html"));
        assert_eq!(fetched.charset.as_deref(), Some("ISO-8859-1"));
        assert_eq!(fetched.file_name.as_deref(), Some("report.pdf"));
        assert_eq!(fetched.final_url.path(), "/doc");
    }

    #[tokio::test]
    async fn follows_and_revalidates_redirects() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/start"))
            .respond_with(ResponseTemplate::new(302).insert_header("location", "/middle"))
            .mount(&server)
            .await;
        Mock::given(method("GET"))
            .and(path("/middle"))
            .respond_with(ResponseTemplate::new(301).insert_header("location", format!("{}/end", server.uri())))
            .mount(&server)
            .await;
        Mock::given(method("GET"))
            .and(path("/end"))
            .respond_with(ResponseTemplate::new(200).set_body_string("done"))
            .mount(&server)
            .await;
        Mock::given(method("GET"))
            .and(path("/to-metadata"))
            .respond_with(ResponseTemplate::new(302).insert_header("location", "http://169.254.169.254/latest/meta-data/"))
            .mount(&server)
            .await;
        Mock::given(method("GET"))
            .and(path("/to-file"))
            .respond_with(ResponseTemplate::new(302).insert_header("location", "file:///etc/passwd"))
            .mount(&server)
            .await;
        Mock::given(method("GET"))
            .and(path("/loop"))
            .respond_with(ResponseTemplate::new(302).insert_header("location", "/loop"))
            .mount(&server)
            .await;

        let ctx = ExtractContext::default();
        let fetched = get(&url(&server, "/start"), &ctx, "*/*").await.unwrap();
        assert_eq!(&fetched.body[..], b"done");
        assert_eq!(fetched.final_url.path(), "/end");

        let blocked = get(&url(&server, "/to-metadata"), &ctx, "*/*").await.unwrap_err();
        assert!(matches!(&blocked, ContentError::UrlNotAllowed(e) if e.0.contains("Link-local")), "{blocked:?}");
        let blocked = get(&url(&server, "/to-file"), &ctx, "*/*").await.unwrap_err();
        assert!(matches!(&blocked, ContentError::UrlNotAllowed(e) if e.0.contains("scheme")), "{blocked:?}");

        let looping = get(&url(&server, "/loop"), &ctx, "*/*").await.unwrap_err();
        assert!(matches!(&looping, ContentError::Network(m) if m.contains("redirects")), "{looping:?}");
    }

    #[tokio::test]
    async fn strict_policy_blocks_the_loopback_server_itself() {
        let server = MockServer::start().await;
        Mock::given(method("GET")).respond_with(ResponseTemplate::new(200)).mount(&server).await;
        let ctx = ExtractContext { url_policy: UrlPolicy::public_only(), ..ExtractContext::default() };
        let error = get(&url(&server, "/x"), &ctx, "*/*").await.unwrap_err();
        assert!(matches!(error, ContentError::UrlNotAllowed(_)), "{error:?}");
        assert!(server.received_requests().await.unwrap().is_empty());
    }

    #[tokio::test]
    async fn size_limit_applies_with_and_without_content_length() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/big"))
            .respond_with(ResponseTemplate::new(200).set_body_bytes(vec![b'x'; 5000]))
            .mount(&server)
            .await;
        let mut ctx = ExtractContext::default();
        ctx.limits.max_download_bytes = 1000;
        let error = get(&url(&server, "/big"), &ctx, "*/*").await.unwrap_err();
        assert!(matches!(error, ContentError::TooLarge { limit: 1000 }), "{error:?}");
        ctx.limits.max_download_bytes = 5000;
        assert_eq!(get(&url(&server, "/big"), &ctx, "*/*").await.unwrap().body.len(), 5000);
    }

    #[tokio::test]
    async fn http_errors_are_typed() {
        let server = MockServer::start().await;
        for (route, status) in [("/gone", 410), ("/missing", 404), ("/denied", 403), ("/broken", 500)] {
            Mock::given(method("GET")).and(path(route)).respond_with(ResponseTemplate::new(status)).mount(&server).await;
        }
        let ctx = ExtractContext::default();
        assert!(matches!(get(&url(&server, "/missing"), &ctx, "*/*").await, Err(ContentError::NotFound(_))));
        assert!(matches!(get(&url(&server, "/gone"), &ctx, "*/*").await, Err(ContentError::NotFound(_))));
        assert!(matches!(get(&url(&server, "/denied"), &ctx, "*/*").await, Err(ContentError::ExternalService(_))));
        assert!(matches!(get(&url(&server, "/broken"), &ctx, "*/*").await, Err(ContentError::ExternalService(_))));
    }

    #[tokio::test]
    async fn connection_and_dns_failures_are_network_errors() {
        // A port nothing listens on.
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        drop(listener);
        let mut ctx = ExtractContext::default();
        ctx.limits.connect_timeout = Duration::from_secs(2);
        let refused = get(&Url::parse(&format!("http://127.0.0.1:{port}/")).unwrap(), &ctx, "*/*").await.unwrap_err();
        assert!(matches!(refused, ContentError::Network(_)), "{refused:?}");

        ctx.resolver = std::sync::Arc::new(crate::url_validation::StaticResolver::new());
        let unresolved = get(&Url::parse("http://no-such-host.invalid/").unwrap(), &ctx, "*/*").await.unwrap_err();
        assert!(matches!(unresolved, ContentError::Network(_)), "{unresolved:?}");
    }

    #[tokio::test]
    async fn requests_are_pinned_to_the_vetted_address() {
        // The name resolves (through the injected resolver) to the mock
        // server; the system resolver could not resolve it at all.
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/pinned"))
            .respond_with(ResponseTemplate::new(200).set_body_string("via pin"))
            .mount(&server)
            .await;
        let address = server.address();
        let resolver = crate::url_validation::StaticResolver::new().with("pinned-host.invalid", &[address.ip()]);
        let ctx = ExtractContext { resolver: std::sync::Arc::new(resolver), ..ExtractContext::default() };
        let target = Url::parse(&format!("http://pinned-host.invalid:{}/pinned", address.port())).unwrap();
        let fetched = get(&target, &ctx, "*/*").await.unwrap();
        assert_eq!(&fetched.body[..], b"via pin");
        let requests = server.received_requests().await.unwrap();
        assert_eq!(requests[0].headers.get("host").unwrap().to_str().unwrap(), format!("pinned-host.invalid:{}", address.port()));
    }

    #[tokio::test]
    async fn head_probe_is_best_effort() {
        let server = MockServer::start().await;
        Mock::given(method("HEAD"))
            .and(path("/paper"))
            .respond_with(ResponseTemplate::new(200).insert_header("content-type", "application/pdf"))
            .mount(&server)
            .await;
        Mock::given(method("HEAD")).and(path("/nohead")).respond_with(ResponseTemplate::new(405)).mount(&server).await;
        let ctx = ExtractContext::default();
        assert_eq!(head_content_type(&url(&server, "/paper"), &ctx).await.unwrap().as_deref(), Some("application/pdf"));
        assert_eq!(head_content_type(&url(&server, "/nohead"), &ctx).await.unwrap(), None);
        let blocked = head_content_type(&Url::parse("http://169.254.169.254/").unwrap(), &ctx).await;
        assert!(matches!(blocked, Err(ContentError::UrlNotAllowed(_))));
    }

    #[test]
    fn disposition_names() {
        let mut headers = HeaderMap::new();
        headers.insert(CONTENT_DISPOSITION, "inline; filename*=UTF-8''na%C3%AFve.pdf".parse().unwrap());
        assert_eq!(disposition_file_name(&headers).as_deref(), Some("na%C3%AFve.pdf"));
        headers.insert(CONTENT_DISPOSITION, "attachment".parse().unwrap());
        assert_eq!(disposition_file_name(&headers), None);
    }
}
