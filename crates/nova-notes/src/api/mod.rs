//! The JSON API.
//!
//! Paths, request bodies, response bodies and status codes follow the Python
//! build's FastAPI app (`api/main.py` and `api/routers/*`), so existing
//! clients work against either. Each submodule is one of its routers and
//! exposes `router()`, with paths relative to `/api`.

pub mod auth;
pub mod capabilities;
pub mod chat;
pub mod commands;
pub mod config;
pub mod credentials;
pub mod embedding;
pub mod embedding_rebuild;
pub mod episode_profiles;
pub mod explainers;
pub mod folders;
pub mod infographics;
pub mod insights;
pub mod languages;
pub mod meetings;
pub mod models;
pub mod notebooks;
pub mod notes;
pub mod perimeter;
pub mod podcast_feed;
pub mod podcast_feed_settings;
pub mod podcasts;
pub mod providers;
pub mod search;
pub mod sections;
pub mod settings;
pub mod source_chat;
pub mod sources;
pub mod speaker_profiles;
pub mod transformations;
pub mod vault;

use std::time::Instant;

use axum::extract::{DefaultBodyLimit, Request, State};
use axum::http::{header, HeaderValue, Method, StatusCode};
use axum::middleware::{self, Next};
use axum::response::{IntoResponse, Redirect, Response};
use axum::routing::get;
use axum::{Json, Router};
use serde_json::json;
use tower_http::cors::{AllowHeaders, AllowMethods, AllowOrigin, CorsLayer};

use crate::config::Config;
use crate::state::AppState;

/// Request extractors that answer bad input the way FastAPI does (422 with a
/// `{"detail": [...]}` list). Use these instead of `axum::Json` / `Query`.
#[allow(unused_imports)]
pub(crate) use notebooks::{lenient_bool, validation_error, ApiJson, ApiQuery};

/// Paths that never need the instance password.
const PUBLIC_PATHS: [&str; 5] = ["/health", "/login", "/logout", "/api/auth/status", "/api/config"];
/// Prefixes that never need the instance password. The Podcast Feed
/// authenticates itself with the Feed Token in the path; keep the trailing
/// slash so `/api/podcasts/feed-url` does not match.
const PUBLIC_PREFIXES: [&str; 2] = ["/api/podcasts/feed/", "/static/"];

/// Assemble the whole application: JSON API under `/api`, web UI everywhere else.
pub fn build_router(state: AppState) -> Router {
    let api = Router::new()
        .merge(auth::router())
        .merge(config::router())
        .merge(notebooks::router())
        .merge(folders::router())
        .merge(sections::router())
        .merge(search::router())
        .merge(models::router())
        .merge(transformations::router())
        .merge(notes::router())
        .merge(meetings::router())
        .merge(explainers::router())
        .merge(infographics::router())
        .merge(embedding::router())
        .merge(embedding_rebuild::router())
        .merge(settings::router())
        .merge(sources::router())
        .merge(insights::router())
        .merge(commands::router())
        .merge(podcasts::router())
        .merge(podcast_feed::router())
        .merge(podcast_feed_settings::router())
        .merge(episode_profiles::router())
        .merge(speaker_profiles::router())
        .merge(chat::router())
        .merge(source_chat::router())
        .merge(credentials::router())
        .merge(providers::router())
        .merge(capabilities::router())
        .merge(languages::router())
        .merge(vault::router())
        .fallback(api_not_found);

    Router::new()
        .route("/health", get(health))
        .nest("/api", api)
        .merge(crate::web::router())
        .layer(DefaultBodyLimit::max(state.config.max_upload_bytes))
        .layer(middleware::from_fn_with_state(state.clone(), require_password))
        .layer(middleware::from_fn_with_state(state.clone(), perimeter::check_origin))
        .layer(middleware::from_fn_with_state(state.clone(), limit_body_size))
        .layer(middleware::from_fn(log_requests))
        .layer(cors_layer(&state.config))
        .layer(middleware::from_fn_with_state(state.clone(), perimeter::check_host))
        .layer(middleware::from_fn(perimeter::security_headers))
        .with_state(state)
}

async fn health() -> Json<serde_json::Value> {
    Json(json!({ "status": "healthy" }))
}

async fn api_not_found() -> Response {
    (StatusCode::NOT_FOUND, Json(json!({ "detail": "Not Found" }))).into_response()
}

/// Cross-origin access: none unless CORS_ORIGINS lists origins. `*` is
/// honoured only with a password set (and never with credentials).
fn cors_layer(config: &Config) -> CorsLayer {
    if config.cors_wildcard() {
        // Wildcard origins never get credentials: together they would reflect
        // any Origin, which defeats the allowlist.
        return CorsLayer::new()
            .allow_origin(AllowOrigin::any())
            .allow_methods(AllowMethods::any())
            .allow_headers(AllowHeaders::any());
    }
    let origins: Vec<HeaderValue> =
        config.cors_origins.iter().filter(|o| *o != "*").filter_map(|o| HeaderValue::from_str(o).ok()).collect();
    if origins.is_empty() {
        // The default: no cross-origin access, so no CORS headers at all.
        return CorsLayer::new();
    }
    CorsLayer::new()
        .allow_origin(AllowOrigin::list(origins))
        .allow_methods(AllowMethods::mirror_request())
        .allow_headers(AllowHeaders::mirror_request())
        .allow_credentials(true)
}

/// Replace the Feed Token in a path so it never reaches the logs.
pub fn redact_feed_token(path: &str) -> String {
    const PREFIX: &str = "/api/podcasts/feed/";
    match path.strip_prefix(PREFIX) {
        Some(rest) => {
            let tail = rest.find('/').map(|i| &rest[i..]).unwrap_or("");
            format!("{PREFIX}[redacted]{tail}")
        }
        None => path.to_string(),
    }
}

async fn log_requests(request: Request, next: Next) -> Response {
    let method = request.method().clone();
    let path = redact_feed_token(request.uri().path());
    let started = Instant::now();
    let response = next.run(request).await;
    if path != "/health" && !path.starts_with("/static/") {
        tracing::info!("{method} {path} -> {} ({} ms)", response.status().as_u16(), started.elapsed().as_millis());
    }
    response
}

fn payload_too_large() -> Response {
    (
        StatusCode::PAYLOAD_TOO_LARGE,
        Json(json!({ "detail": "Request body exceeds the maximum allowed upload size" })),
    )
        .into_response()
}

/// Reject a declared oversized body before any work is done on it. Bodies
/// without a truthful Content-Length are capped by `DefaultBodyLimit` as they
/// are read.
async fn limit_body_size(State(state): State<AppState>, request: Request, next: Next) -> Response {
    let declared = request
        .headers()
        .get(header::CONTENT_LENGTH)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.parse::<usize>().ok());
    if let Some(length) = declared {
        if length > state.config.max_upload_bytes {
            tracing::warn!(
                "Rejected {} {}: declared body of {length} bytes exceeds the {}-byte limit",
                request.method(),
                redact_feed_token(request.uri().path()),
                state.config.max_upload_bytes
            );
            return payload_too_large();
        }
    }
    let response = next.run(request).await;
    if response.status() == StatusCode::PAYLOAD_TOO_LARGE {
        return payload_too_large();
    }
    response
}

fn unauthorized(detail: &str) -> Response {
    (
        StatusCode::UNAUTHORIZED,
        [(header::WWW_AUTHENTICATE, HeaderValue::from_static("Bearer"))],
        Json(json!({ "detail": detail })),
    )
        .into_response()
}

/// Password check for everything except the public paths. API clients send
/// `Authorization: Bearer <password>`; the web UI carries a session cookie set
/// by its login page. With no password configured, authentication is off.
async fn require_password(State(state): State<AppState>, request: Request, next: Next) -> Response {
    let Some(password) = state.config.password.as_deref() else {
        return next.run(request).await;
    };
    let path = request.uri().path();
    // Only a real CORS preflight skips the check, so a route that ever
    // answers OPTIONS is not exposed by accident.
    let preflight = request.method() == Method::OPTIONS && request.headers().contains_key(header::ACCESS_CONTROL_REQUEST_METHOD);
    if preflight
        || PUBLIC_PATHS.contains(&path)
        || PUBLIC_PREFIXES.iter().any(|prefix| path.starts_with(prefix))
    {
        return next.run(request).await;
    }

    if auth::has_valid_session(&state, request.headers(), password).await {
        return next.run(request).await;
    }

    let is_api = path.starts_with("/api/");
    let Some(header_value) = request.headers().get(header::AUTHORIZATION) else {
        if is_api {
            return unauthorized("Missing authorization header");
        }
        let next_path = request.uri().path_and_query().map(|p| p.as_str()).unwrap_or("/");
        return Redirect::to(&format!("/login?next={}", auth::encode_query_value(next_path))).into_response();
    };
    let Some(credentials) = header_value
        .to_str()
        .ok()
        .and_then(|v| v.split_once(' '))
        .filter(|(scheme, _)| scheme.eq_ignore_ascii_case("bearer"))
        .map(|(_, credentials)| credentials)
    else {
        return unauthorized("Invalid authorization header format");
    };
    let client = auth::client_ip(&state.config, request.headers(), request.extensions());
    if let Err(wait) = state.login_throttle.check(client) {
        return too_many_attempts(wait);
    }
    if auth::constant_time_eq(credentials.as_bytes(), password.as_bytes()) {
        state.login_throttle.record_success(client);
        next.run(request).await
    } else {
        tracing::warn!("Failed API password check from {client}");
        state.login_throttle.record_failure(client);
        unauthorized("Invalid password")
    }
}

/// 429 for a client that has to wait after failed password checks.
pub(crate) fn too_many_attempts(wait: std::time::Duration) -> Response {
    let seconds = wait.as_secs() + u64::from(wait.subsec_nanos() > 0);
    (
        StatusCode::TOO_MANY_REQUESTS,
        [(header::RETRY_AFTER, HeaderValue::from(seconds.max(1)))],
        Json(json!({ "detail": auth::throttled_detail(wait) })),
    )
        .into_response()
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::Body;
    use http_body_util::BodyExt;
    use tower::ServiceExt;

    async fn app(password: Option<&str>) -> (Router, tempfile::TempDir) {
        let (mut state, dir) = AppState::for_tests().await.unwrap();
        let mut config = (*state.config).clone();
        config.password = password.map(String::from);
        config.max_upload_bytes = 1024;
        state.config = std::sync::Arc::new(config);
        (build_router(state), dir)
    }

    async fn send(app: &Router, request: Request) -> (StatusCode, serde_json::Value) {
        let response = app.clone().oneshot(request).await.unwrap();
        let status = response.status();
        let bytes = response.into_body().collect().await.unwrap().to_bytes();
        (status, serde_json::from_slice(&bytes).unwrap_or(serde_json::Value::Null))
    }

    fn get(path: &str) -> Request {
        Request::builder().uri(path).body(Body::empty()).unwrap()
    }

    #[test]
    fn feed_tokens_are_redacted() {
        assert_eq!(redact_feed_token("/api/podcasts/feed/s3cret/rss.xml"), "/api/podcasts/feed/[redacted]/rss.xml");
        assert_eq!(redact_feed_token("/api/podcasts/feed/s3cret"), "/api/podcasts/feed/[redacted]");
        assert_eq!(redact_feed_token("/api/podcasts/feed-url"), "/api/podcasts/feed-url");
        assert_eq!(redact_feed_token("/api/notebooks"), "/api/notebooks");
    }

    #[tokio::test]
    async fn health_and_unknown_api_paths() {
        let (app, _dir) = app(None).await;
        let (status, body) = send(&app, get("/health")).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(body["status"], "healthy");
        let (status, body) = send(&app, get("/api/definitely-not-a-route")).await;
        assert_eq!(status, StatusCode::NOT_FOUND);
        assert_eq!(body["detail"], "Not Found");
    }

    #[tokio::test]
    async fn no_password_means_no_auth() {
        let (app, _dir) = app(None).await;
        let (status, body) = send(&app, get("/api/auth/status")).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(body["auth_enabled"], false);
        let (status, _) = send(&app, get("/api/commands/jobs")).await;
        assert_eq!(status, StatusCode::OK);
    }

    #[tokio::test]
    async fn password_protects_the_api() {
        let (app, _dir) = app(Some("hunter2")).await;

        let (status, body) = send(&app, get("/api/auth/status")).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(body["auth_enabled"], true);
        let (status, _) = send(&app, get("/health")).await;
        assert_eq!(status, StatusCode::OK);

        let (status, body) = send(&app, get("/api/commands/jobs")).await;
        assert_eq!(status, StatusCode::UNAUTHORIZED);
        assert_eq!(body["detail"], "Missing authorization header");

        let with = |value: &str| Request::builder().uri("/api/commands/jobs").header("authorization", value).body(Body::empty()).unwrap();
        let (status, body) = send(&app, with("Basic abc")).await;
        assert_eq!(status, StatusCode::UNAUTHORIZED);
        assert_eq!(body["detail"], "Invalid authorization header format");
        let (status, body) = send(&app, with("Bearer wrong")).await;
        assert_eq!(status, StatusCode::UNAUTHORIZED);
        assert_eq!(body["detail"], "Invalid password");
        let (status, _) = send(&app, with("Bearer hunter2")).await;
        assert_eq!(status, StatusCode::OK);

        // The old password-derived cookie (SHA-256 of the password) no longer
        // opens anything; sessions are server-side now (see web tests).
        let old = {
            use sha2::{Digest, Sha256};
            hex::encode(Sha256::digest(b"nova-notes-session-v1:hunter2"))
        };
        let request = Request::builder()
            .uri("/api/commands/jobs")
            .header("cookie", format!("{}={old}", auth::SESSION_COOKIE))
            .body(Body::empty())
            .unwrap();
        let (status, _) = send(&app, request).await;
        assert_eq!(status, StatusCode::UNAUTHORIZED);

        // A CORS preflight is never challenged; a bare OPTIONS is.
        let request = Request::builder()
            .method(Method::OPTIONS)
            .uri("/api/commands/jobs")
            .header("origin", "https://app.example")
            .header("access-control-request-method", "GET")
            .body(Body::empty())
            .unwrap();
        let (status, _) = send(&app, request).await;
        assert_ne!(status, StatusCode::UNAUTHORIZED);
        // (The CORS layer answers every OPTIONS itself today, so a bare one
        // never reaches a handler; `options_only_skip_the_password_as_a_preflight`
        // checks the password layer on its own.)

        // Pages redirect to the login form instead of answering 401.
        let response = app.clone().oneshot(get("/some/page?x=1")).await.unwrap();
        assert_eq!(response.status(), StatusCode::SEE_OTHER);
        assert_eq!(response.headers()["location"], "/login?next=%2Fsome%2Fpage%3Fx%3D1");
    }

    #[tokio::test]
    async fn failed_bearer_passwords_are_throttled_per_client() {
        let (app, _dir) = app(Some("hunter2")).await;
        let from = |ip: &str, password: &str| {
            let mut request = Request::builder()
                .uri("/api/commands/jobs")
                .header("authorization", format!("Bearer {password}"))
                .body(Body::empty())
                .unwrap();
            let addr: std::net::SocketAddr = format!("{ip}:40000").parse().unwrap();
            request.extensions_mut().insert(axum::extract::ConnectInfo(addr));
            request
        };
        for _ in 0..5 {
            let (status, body) = send(&app, from("192.0.2.10", "wrong")).await;
            assert_eq!(status, StatusCode::UNAUTHORIZED);
            assert_eq!(body["detail"], "Invalid password");
        }
        // Now even the right password waits, without being tried.
        let response = app.clone().oneshot(from("192.0.2.10", "hunter2")).await.unwrap();
        assert_eq!(response.status(), StatusCode::TOO_MANY_REQUESTS);
        assert!(response.headers().contains_key(header::RETRY_AFTER));
        let bytes = response.into_body().collect().await.unwrap().to_bytes();
        let body: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert!(body["detail"].as_str().unwrap().starts_with("Too many failed sign-in attempts"), "{body}");
        // Another client is unaffected.
        let (status, _) = send(&app, from("192.0.2.11", "hunter2")).await;
        assert_eq!(status, StatusCode::OK);
    }

    #[tokio::test]
    async fn options_only_skip_the_password_as_a_preflight() {
        let (mut state, _dir) = AppState::for_tests().await.unwrap();
        let mut config = (*state.config).clone();
        config.password = Some("hunter2".into());
        state.config = std::sync::Arc::new(config);
        // A route that answers OPTIONS, behind the password layer alone.
        let router = Router::new()
            .route("/api/thing", axum::routing::options(|| async { "secret" }))
            .layer(middleware::from_fn_with_state(state.clone(), require_password))
            .with_state(state);
        let bare = Request::builder().method(Method::OPTIONS).uri("/api/thing").body(Body::empty()).unwrap();
        let (status, body) = send(&router, bare).await;
        assert_eq!(status, StatusCode::UNAUTHORIZED);
        assert_eq!(body["detail"], "Missing authorization header");
        let preflight = Request::builder()
            .method(Method::OPTIONS)
            .uri("/api/thing")
            .header("access-control-request-method", "GET")
            .body(Body::empty())
            .unwrap();
        let response = router.clone().oneshot(preflight).await.unwrap();
        assert_eq!(response.status(), StatusCode::OK);
    }

    async fn app_with(edit: impl FnOnce(&mut Config)) -> (Router, tempfile::TempDir) {
        let (mut state, dir) = AppState::for_tests().await.unwrap();
        let mut config = (*state.config).clone();
        edit(&mut config);
        state.config = std::sync::Arc::new(config);
        (build_router(state), dir)
    }

    fn with_headers(method: Method, path: &str, pairs: &[(&str, &str)]) -> Request {
        let mut builder = Request::builder().method(method).uri(path);
        for (name, value) in pairs {
            builder = builder.header(*name, *value);
        }
        let body = if path.starts_with("/api/") { Body::from(r#"{"name":"N","description":""}"#) } else { Body::from("") };
        builder.body(body).unwrap()
    }

    #[tokio::test]
    async fn default_cors_lets_no_other_site_read_the_api() {
        let (app, _dir) = app(None).await;
        let response = app
            .clone()
            .oneshot(with_headers(Method::GET, "/api/notebooks", &[("origin", "https://evil.example")]))
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        assert!(response.headers().get("access-control-allow-origin").is_none(), "{:?}", response.headers());

        let preflight = with_headers(
            Method::OPTIONS,
            "/api/notebooks",
            &[("origin", "https://evil.example"), ("access-control-request-method", "POST"), ("access-control-request-headers", "content-type")],
        );
        let response = app.clone().oneshot(preflight).await.unwrap();
        assert!(response.headers().get("access-control-allow-origin").is_none(), "{:?}", response.headers());

        // `*` without a password is ignored rather than handing the API to every site.
        let (app, _dir) = app_with(|c| c.cors_origins = vec!["*".into()]).await;
        let response = app
            .clone()
            .oneshot(with_headers(Method::GET, "/api/notebooks", &[("origin", "https://evil.example")]))
            .await
            .unwrap();
        assert!(response.headers().get("access-control-allow-origin").is_none(), "{:?}", response.headers());
    }

    #[tokio::test]
    async fn listed_cors_origins_still_work() {
        let (app, _dir) = app_with(|c| c.cors_origins = vec!["https://app.example".into()]).await;
        let preflight = with_headers(
            Method::OPTIONS,
            "/api/notebooks",
            &[("origin", "https://app.example"), ("access-control-request-method", "POST"), ("access-control-request-headers", "content-type")],
        );
        let response = app.clone().oneshot(preflight).await.unwrap();
        assert_eq!(response.headers()["access-control-allow-origin"], "https://app.example");
        assert_eq!(response.headers()["access-control-allow-credentials"], "true");
        let post = with_headers(
            Method::POST,
            "/api/notebooks",
            &[("origin", "https://app.example"), ("sec-fetch-site", "cross-site"), ("content-type", "application/json")],
        );
        let response = app.clone().oneshot(post).await.unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(response.headers()["access-control-allow-origin"], "https://app.example");
    }

    #[tokio::test]
    async fn unknown_host_names_are_refused() {
        let (app, _dir) = app(None).await;
        // DNS rebinding: the attacker's name resolves to 127.0.0.1.
        let (status, body) = send(&app, with_headers(Method::GET, "/api/notebooks", &[("host", "attacker.example:5055")])).await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert!(body["detail"].as_str().unwrap().contains("OPEN_NOTEBOOK_ALLOWED_HOSTS"), "{body}");
        let (status, _) = send(&app, with_headers(Method::GET, "/notebooks", &[("host", "attacker.example")])).await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        let (status, _) = send(&app, with_headers(Method::GET, "/api/notebooks", &[("host", "bad host")])).await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        for host in ["127.0.0.1:5055", "localhost:5055", "[::1]:5055", "LOCALHOST"] {
            let (status, _) = send(&app, with_headers(Method::GET, "/api/notebooks", &[("host", host)])).await;
            assert_eq!(status, StatusCode::OK, "{host}");
        }
        // Health probes by container IP keep working.
        let (status, _) = send(&app, with_headers(Method::GET, "/health", &[("host", "172.17.0.2:5055")])).await;
        assert_eq!(status, StatusCode::OK);

        // A tailnet name is one setting away.
        let (app, _dir) = app_with(|c| c.allowed_hosts = vec!["notes.tailnet-name.ts.net".into()]).await;
        let (status, _) = send(&app, with_headers(Method::GET, "/api/notebooks", &[("host", "notes.tailnet-name.ts.net")])).await;
        assert_eq!(status, StatusCode::OK);
        let (status, _) = send(&app, with_headers(Method::GET, "/api/notebooks", &[("host", "attacker.example")])).await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn cross_site_writes_are_refused_on_the_api_and_the_web_forms() {
        let (app, _dir) = app(None).await;
        let cross = [("origin", "https://evil.example"), ("sec-fetch-site", "cross-site"), ("content-type", "application/json")];
        let (status, body) = send(&app, with_headers(Method::POST, "/api/notebooks", &cross)).await;
        assert_eq!(status, StatusCode::FORBIDDEN);
        assert_eq!(body["detail"], "Cross-site request refused");
        // Another port on the same host is a different application.
        let sibling = [("host", "127.0.0.1:5055"), ("origin", "http://127.0.0.1:3000"), ("content-type", "application/json")];
        let (status, _) = send(&app, with_headers(Method::DELETE, "/api/notebooks/notebook:x", &sibling)).await;
        assert_eq!(status, StatusCode::FORBIDDEN);
        // Web forms: an auto-submitted form from another site.
        let form = [("origin", "https://evil.example"), ("sec-fetch-site", "cross-site"), ("content-type", "application/x-www-form-urlencoded")];
        for path in ["/notes/quick", "/login", "/podcasts/feed/rotate", "/logout"] {
            let (status, _) = send(&app, with_headers(Method::POST, path, &form)).await;
            assert_eq!(status, StatusCode::FORBIDDEN, "{path}");
        }
        // Reads are not affected, and same-origin writes and non-browser clients pass.
        let (status, _) = send(&app, with_headers(Method::GET, "/api/notebooks", &cross)).await;
        assert_eq!(status, StatusCode::OK);
        let same = [("origin", "http://127.0.0.1:5055"), ("sec-fetch-site", "same-origin"), ("content-type", "application/json")];
        let (status, _) = send(&app, with_headers(Method::POST, "/api/notebooks", &same)).await;
        assert_eq!(status, StatusCode::OK);
        let (status, _) = send(&app, with_headers(Method::POST, "/api/notebooks", &[("content-type", "application/json")])).await;
        assert_eq!(status, StatusCode::OK);
        let quick = [("host", "127.0.0.1:5055"), ("origin", "http://127.0.0.1:5055")];
        let response = app.clone().oneshot(with_headers(Method::POST, "/notes/quick", &quick)).await.unwrap();
        assert_ne!(response.status(), StatusCode::FORBIDDEN);
    }

    #[tokio::test]
    async fn responses_carry_security_headers() {
        let (app, _dir) = app(None).await;
        let page = app.clone().oneshot(get("/notebooks")).await.unwrap();
        assert_eq!(page.status(), StatusCode::OK);
        let headers = page.headers();
        let csp = headers["content-security-policy"].to_str().unwrap();
        for directive in ["default-src 'self'", "img-src 'self' data: blob:", "frame-src 'self'", "frame-ancestors 'self'", "object-src 'none'", "base-uri 'none'", "form-action 'self'"] {
            assert!(csp.contains(directive), "{directive} in {csp}");
        }
        assert!(!csp.contains("'unsafe-eval'") && !csp.contains("script-src 'self' 'unsafe-inline'"), "{csp}");
        assert_eq!(headers["x-content-type-options"], "nosniff");
        assert_eq!(headers["referrer-policy"], "same-origin");
        assert_eq!(headers["x-frame-options"], "SAMEORIGIN");

        let json = app.clone().oneshot(get("/api/notebooks")).await.unwrap();
        assert_eq!(json.headers()["x-content-type-options"], "nosniff");
        assert_eq!(json.headers()["referrer-policy"], "same-origin");
        assert!(json.headers().get("content-security-policy").is_none());
        // Refusals carry them too.
        let refused = app.clone().oneshot(with_headers(Method::GET, "/api/notebooks", &[("host", "attacker.example")])).await.unwrap();
        assert_eq!(refused.headers()["x-frame-options"], "SAMEORIGIN");
    }

    #[tokio::test]
    async fn oversized_bodies_are_rejected_up_front() {
        let (app, _dir) = app(None).await;
        let request = Request::builder()
            .method(Method::POST)
            .uri("/api/notebooks")
            .header("content-length", "5000")
            .body(Body::from(vec![b'x'; 5000]))
            .unwrap();
        let (status, body) = send(&app, request).await;
        assert_eq!(status, StatusCode::PAYLOAD_TOO_LARGE);
        assert_eq!(body["detail"], "Request body exceeds the maximum allowed upload size");
    }
}
