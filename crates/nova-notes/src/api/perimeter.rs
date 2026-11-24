//! The request perimeter: which hosts and sites may talk to this server, and
//! the security headers every response carries.
//!
//! - [`check_host`] refuses a `Host` that is not this server's (DNS rebinding:
//!   a page on `attacker.example` re-pointed at 127.0.0.1 sends
//!   `Host: attacker.example`).
//! - [`check_origin`] refuses state-changing requests a browser sends on behalf
//!   of another site (CSRF against both the JSON API and the web forms).
//! - [`security_headers`] adds `nosniff`, `Referrer-Policy`, `X-Frame-Options`
//!   and, on HTML pages, the app's `Content-Security-Policy`.

use std::sync::LazyLock;

use axum::extract::{Request, State};
use axum::http::{header, HeaderMap, HeaderValue, Method, StatusCode};
use axum::middleware::Next;
use axum::response::{IntoResponse, Response};
use axum::Json;
use base64::Engine;
use serde_json::json;
use sha2::{Digest, Sha256};

use crate::config::{strip_port, Config};
use crate::state::AppState;

/// The `Content-Security-Policy` of the app's own HTML pages.
///
/// - Scripts: this server's files plus the inline boot scripts by hash.
///   `wasm-unsafe-eval` is for the editor's code highlighter (WebAssembly);
///   `blob:` for the meeting recorder's AudioWorklet and the editor's worker.
/// - Images only from this server, `data:` and `blob:`: rendered markdown
///   cannot make the browser fetch a remote URL.
/// - `frame-src 'self'`: a framed explainer or uploaded HTML note cannot
///   navigate its frame to another site (the parent's policy governs that).
/// - `frame-ancestors 'self'`: other sites cannot frame the app.
pub static PAGE_CSP: LazyLock<String> = LazyLock::new(|| {
    let hashes: Vec<String> = crate::web::INLINE_SCRIPTS.iter().map(|script| script_hash(script)).collect();
    format!(
        "default-src 'self'; script-src 'self' 'wasm-unsafe-eval' blob: {}; style-src 'self' 'unsafe-inline'; \
         img-src 'self' data: blob:; font-src 'self' data:; media-src 'self' data: blob:; \
         connect-src 'self' data: blob:; worker-src 'self' blob:; frame-src 'self'; frame-ancestors 'self'; \
         object-src 'none'; base-uri 'none'; form-action 'self'",
        hashes.join(" ")
    )
});

/// The CSP source expression allowing one inline script.
pub fn script_hash(script: &str) -> String {
    format!("'sha256-{}'", base64::engine::general_purpose::STANDARD.encode(Sha256::digest(script.as_bytes())))
}

/// Add the security headers a response does not already set. Responses with
/// their own CSP (note files, blobs, explainers) keep it; only HTML gets the
/// page policy, so attachments, audio and JSON are untouched apart from the
/// generic headers.
pub async fn security_headers(request: Request, next: Next) -> Response {
    let mut response = next.run(request).await;
    let is_html = response
        .headers()
        .get(header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .is_some_and(|v| v.trim_start().to_ascii_lowercase().starts_with("text/html"));
    let headers = response.headers_mut();
    if is_html && !headers.contains_key(header::CONTENT_SECURITY_POLICY) {
        if let Ok(value) = HeaderValue::from_str(&PAGE_CSP) {
            headers.insert(header::CONTENT_SECURITY_POLICY, value);
        }
    }
    for (name, value) in [
        (header::X_CONTENT_TYPE_OPTIONS, "nosniff"),
        // Keep `same-origin`, not `no-referrer`: under `no-referrer` browsers send
        // `Origin: null` on form posts, and over plain HTTP (no Sec-Fetch-Site)
        // `check_origin` would then refuse the app's own forms.
        (header::REFERRER_POLICY, "same-origin"),
        (header::X_FRAME_OPTIONS, "SAMEORIGIN"),
    ] {
        if !headers.contains_key(&name) {
            headers.insert(name, HeaderValue::from_static(value));
        }
    }
    response
}

fn refuse(status: StatusCode, detail: String) -> Response {
    (status, Json(json!({ "detail": detail }))).into_response()
}

/// The host a request was addressed to: the `Host` header, or the URI's
/// authority (HTTP/2). `X-Forwarded-Host` is never consulted: a rebound page
/// can set it to anything.
fn request_host(request: &Request) -> Option<&str> {
    request
        .headers()
        .get(header::HOST)
        .and_then(|v| v.to_str().ok())
        .or_else(|| request.uri().authority().map(|a| a.as_str()))
}

/// Refuse requests addressed to a host this server does not answer to.
/// Requests with no host at all (HTTP/1.0 tools) pass: a browser always
/// sends one, so DNS rebinding is still caught. `/health` is exempt so
/// container and orchestrator probes by IP keep working.
pub async fn check_host(State(state): State<AppState>, request: Request, next: Next) -> Response {
    if request.uri().path() == "/health" {
        return next.run(request).await;
    }
    if let Some(raw) = request_host(&request) {
        let allowed = strip_port(raw).is_some_and(|host| state.config.host_is_allowed(&host));
        if !allowed {
            let shown: String = raw.chars().filter(|c| c.is_ascii_graphic()).take(255).collect();
            let host = strip_port(raw).unwrap_or_else(|| shown.clone());
            tracing::warn!(
                "Refused a request for host {shown:?}. If that is this server's name (a Tailscale or LAN host name), \
                 add {host:?} to OPEN_NOTEBOOK_ALLOWED_HOSTS (comma-separated; a leading dot such as \
                 .tailnet-name.ts.net allows every name under it) or set OPEN_NOTEBOOK_PUBLIC_URL."
            );
            return refuse(
                StatusCode::BAD_REQUEST,
                format!(
                    "This server does not answer to the host name {host:?}. If it should, add it to \
                     OPEN_NOTEBOOK_ALLOWED_HOSTS (comma-separated) or set OPEN_NOTEBOOK_PUBLIC_URL, then restart."
                ),
            );
        }
    }
    next.run(request).await
}

/// Whether a state-changing request comes from a page of another site.
///
/// Requests with neither `Sec-Fetch-Site` nor `Origin` come from a non-browser
/// client (curl, scripts, the vault sync tool) and pass. Origins listed in
/// `CORS_ORIGINS`, and the public URL's origin, are trusted. Otherwise
/// `Sec-Fetch-Site` must be `same-origin` or `none` (typed by the user), and
/// without it the `Origin` must match the `Host` the request was sent to.
/// `same-site` is refused too: another port on the same host, or a sibling
/// subdomain, is a different application.
pub(crate) fn is_cross_site(config: &Config, headers: &HeaderMap) -> bool {
    let origin = headers.get(header::ORIGIN).and_then(|v| v.to_str().ok()).map(str::trim);
    if let Some(origin) = origin {
        let listed = config.cors_origins.iter().any(|o| o != "*" && o.trim_end_matches('/').eq_ignore_ascii_case(origin));
        let public = config.public_origins.iter().any(|o| o.eq_ignore_ascii_case(origin));
        // With `CORS_ORIGINS=*` (password required for that), a cross-site
        // caller still has to present the password itself, never ride a cookie.
        let wildcard_with_bearer = config.cors_wildcard() && headers.contains_key(header::AUTHORIZATION);
        if listed || public || wildcard_with_bearer {
            return false;
        }
    }
    if let Some(site) = headers.get("sec-fetch-site").and_then(|v| v.to_str().ok()) {
        return !matches!(site.trim().to_ascii_lowercase().as_str(), "same-origin" | "none");
    }
    let Some(origin) = origin else {
        return false;
    };
    // Compare host and port with the Host header; the scheme may differ
    // behind a TLS-terminating proxy, so the Host is read with the Origin's
    // scheme (and so its default port).
    let Ok(origin) = url::Url::parse(origin) else {
        return true; // `null` (sandboxed frames, privacy redirects) or garbage.
    };
    let Some(host) = headers.get(header::HOST).and_then(|v| v.to_str().ok()) else {
        return true;
    };
    let Ok(target) = url::Url::parse(&format!("{}://{}/", origin.scheme(), host.trim())) else {
        return true;
    };
    let authority = |url: &url::Url| url.host_str().map(|h| (h.to_ascii_lowercase(), url.port_or_known_default()));
    authority(&origin).is_none() || authority(&origin) != authority(&target)
}

/// How a client reached this server, as far as the request can be believed.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct RequestOrigin {
    /// The client used HTTPS (decides the cookie's `Secure` flag and name).
    pub https: bool,
    /// The `host[:port]` the client used, when it can be trusted for
    /// building absolute links.
    pub host: Option<String>,
}

/// Where the request came from, for absolute links (the podcast feed) and the
/// cookie's `Secure` flag.
///
/// `X-Forwarded-Proto` / `X-Forwarded-Host` are believed only with
/// `OPEN_NOTEBOOK_TRUST_PROXY_HEADERS` (a proxy in front overwrites them);
/// otherwise anyone could put a host of their choosing into the links. Without
/// that setting the request counts as HTTPS when its `Host` is the host of an
/// `https://` `OPEN_NOTEBOOK_PUBLIC_URL` / `API_URL`, and the host is the
/// `Host` header when the allow-list is not `*`. A forwarded host must pass
/// the allow-list too, since some proxies pass a client's value through.
pub fn request_origin(config: &Config, headers: &HeaderMap) -> RequestOrigin {
    let first = |name: &str| {
        headers
            .get(name)
            .and_then(|v| v.to_str().ok())
            .and_then(|v| v.split(',').next())
            .map(str::trim)
            .filter(|v| !v.is_empty())
            .map(String::from)
    };
    let wildcard = config.allowed_hosts.iter().any(|entry| entry == "*");
    let allowed = |raw: &String| strip_port(raw).is_some_and(|host| config.host_is_allowed(&host));
    let host_header = first(header::HOST.as_str()).filter(|h| !wildcard && allowed(h));
    let public_https = |host: &str| {
        let Ok(target) = url::Url::parse(&format!("https://{host}/")) else {
            return false;
        };
        config.public_origins.iter().filter_map(|o| url::Url::parse(o).ok()).any(|origin| {
            origin.scheme() == "https"
                && origin.host_str().map(str::to_ascii_lowercase) == target.host_str().map(str::to_ascii_lowercase)
                && origin.port_or_known_default() == target.port_or_known_default()
        })
    };
    if config.trust_proxy_headers {
        let host = first("x-forwarded-host").filter(allowed).or(host_header);
        let https = first("x-forwarded-proto").is_some_and(|p| p.eq_ignore_ascii_case("https"))
            || host.as_deref().is_some_and(public_https);
        return RequestOrigin { https, host };
    }
    let https = host_header.as_deref().is_some_and(public_https);
    RequestOrigin { https, host: host_header }
}

/// Refuse state-changing requests (anything but GET, HEAD and OPTIONS) that
/// a browser sends on behalf of another site. Covers the JSON API and every
/// web form, so a page elsewhere cannot submit to this one.
pub async fn check_origin(State(state): State<AppState>, request: Request, next: Next) -> Response {
    let safe = matches!(*request.method(), Method::GET | Method::HEAD | Method::OPTIONS);
    if !safe && is_cross_site(&state.config, request.headers()) {
        tracing::warn!(
            "Refused a cross-site {} {} (Origin {:?}, Sec-Fetch-Site {:?})",
            request.method(),
            super::redact_feed_token(request.uri().path()),
            request.headers().get(header::ORIGIN),
            request.headers().get("sec-fetch-site"),
        );
        return refuse(StatusCode::FORBIDDEN, "Cross-site request refused".into());
    }
    next.run(request).await
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    fn headers(pairs: &[(&'static str, &str)]) -> HeaderMap {
        let mut map = HeaderMap::new();
        for (name, value) in pairs {
            map.insert(*name, HeaderValue::from_str(value).unwrap());
        }
        map
    }

    #[test]
    fn cross_site_detection() {
        let mut config = Config::for_tests(Path::new("/tmp/nn"));
        let cross = |config: &Config, pairs: &[(&'static str, &str)]| is_cross_site(config, &headers(pairs));

        // Non-browser clients send neither header.
        assert!(!cross(&config, &[("host", "127.0.0.1:5055")]));
        // Fetch metadata decides when present.
        assert!(!cross(&config, &[("sec-fetch-site", "same-origin"), ("origin", "http://127.0.0.1:5055")]));
        assert!(!cross(&config, &[("sec-fetch-site", "none")]));
        assert!(cross(&config, &[("sec-fetch-site", "cross-site"), ("origin", "https://evil.example")]));
        assert!(cross(&config, &[("sec-fetch-site", "same-site"), ("origin", "http://localhost:3000")]));
        // Without it, Origin must match Host, port included.
        assert!(!cross(&config, &[("host", "127.0.0.1:5055"), ("origin", "http://127.0.0.1:5055")]));
        assert!(!cross(&config, &[("host", "notes.lan"), ("origin", "http://notes.lan")]));
        assert!(!cross(&config, &[("host", "notes.lan:443"), ("origin", "https://notes.lan")]));
        assert!(!cross(&config, &[("host", "[::1]:5055"), ("origin", "http://[::1]:5055")]));
        assert!(cross(&config, &[("host", "localhost:5055"), ("origin", "http://localhost:3000")]));
        assert!(cross(&config, &[("host", "127.0.0.1:5055"), ("origin", "https://evil.example")]));
        assert!(cross(&config, &[("host", "127.0.0.1:5055"), ("origin", "null")]));
        assert!(cross(&config, &[("origin", "http://127.0.0.1:5055")]));

        // Listed origins and the public URL's origin are trusted.
        config.cors_origins = vec!["https://app.example".into()];
        config.public_origins = vec!["https://notes.example.ts.net".into()];
        assert!(!cross(&config, &[("sec-fetch-site", "cross-site"), ("origin", "https://app.example")]));
        assert!(!cross(&config, &[("host", "127.0.0.1:5055"), ("origin", "https://notes.example.ts.net")]));
        assert!(cross(&config, &[("sec-fetch-site", "cross-site"), ("origin", "https://other.example")]));

        // A wildcard trusts no one on cookies alone.
        config.cors_origins = vec!["*".into()];
        config.password = Some("hunter2".into());
        assert!(cross(&config, &[("sec-fetch-site", "cross-site"), ("origin", "https://evil.example")]));
        assert!(!cross(
            &config,
            &[("sec-fetch-site", "cross-site"), ("origin", "https://app.example"), ("authorization", "Bearer hunter2")]
        ));
    }

    #[test]
    fn forwarded_headers_count_only_behind_a_trusted_proxy() {
        let mut config = Config::for_tests(Path::new("/tmp/nn"));
        config.allowed_hosts = vec!["notes.example.ts.net".into()];
        let origin = |config: &Config, pairs: &[(&'static str, &str)]| request_origin(config, &headers(pairs));
        let forged = [("host", "127.0.0.1:5055"), ("x-forwarded-proto", "https"), ("x-forwarded-host", "evil.example")];

        // Untrusted: the forwarded headers change nothing.
        assert_eq!(origin(&config, &forged), RequestOrigin { https: false, host: Some("127.0.0.1:5055".into()) });
        // An https public URL marks requests to its host as HTTPS.
        config.public_origins = vec!["https://notes.example.ts.net".into()];
        assert_eq!(
            origin(&config, &[("host", "notes.example.ts.net")]),
            RequestOrigin { https: true, host: Some("notes.example.ts.net".into()) }
        );
        assert!(!origin(&config, &[("host", "notes.example.ts.net:5055")]).https);
        // A `*` allow-list (or no Host) gives no host to build links from.
        config.allowed_hosts = vec!["*".into()];
        assert_eq!(origin(&config, &forged).host, None);
        assert_eq!(origin(&config, &[]), RequestOrigin::default());

        // Trusted: believed, but the forwarded host must still be one of ours.
        config.allowed_hosts = vec!["notes.example.ts.net".into()];
        config.trust_proxy_headers = true;
        assert_eq!(origin(&config, &forged), RequestOrigin { https: true, host: Some("127.0.0.1:5055".into()) });
        assert_eq!(
            origin(&config, &[("x-forwarded-proto", "https"), ("x-forwarded-host", "notes.example.ts.net:8502, proxy")]),
            RequestOrigin { https: true, host: Some("notes.example.ts.net:8502".into()) }
        );
    }

    #[test]
    fn script_hashes_match_the_csp_format() {
        // Known value: printf 'alert(1)' | openssl dgst -sha256 -binary | base64
        assert_eq!(script_hash("alert(1)"), "'sha256-bhHHL3z2vDgxUt0W3dWQOrprscmda2Y5pLsLg4GF+pI='");
        for script in crate::web::INLINE_SCRIPTS {
            assert!(PAGE_CSP.contains(&script_hash(script)));
        }
    }
}
