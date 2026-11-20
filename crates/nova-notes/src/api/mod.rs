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
        .layer(middleware::from_fn_with_state(state.clone(), limit_body_size))
        .layer(middleware::from_fn(log_requests))
        .layer(cors_layer(&state.config))
        .with_state(state)
}

async fn health() -> Json<serde_json::Value> {
    Json(json!({ "status": "healthy" }))
}

async fn api_not_found() -> Response {
    (StatusCode::NOT_FOUND, Json(json!({ "detail": "Not Found" }))).into_response()
}

fn cors_layer(config: &Config) -> CorsLayer {
    if config.cors_origins.iter().any(|o| o == "*") {
        // Wildcard origins never get credentials: together they would reflect
        // any Origin, which defeats the allowlist.
        return CorsLayer::new()
            .allow_origin(AllowOrigin::any())
            .allow_methods(AllowMethods::any())
            .allow_headers(AllowHeaders::any());
    }
    let origins: Vec<HeaderValue> = config.cors_origins.iter().filter_map(|o| HeaderValue::from_str(o).ok()).collect();
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
    if request.method() == Method::OPTIONS
        || PUBLIC_PATHS.contains(&path)
        || PUBLIC_PREFIXES.iter().any(|prefix| path.starts_with(prefix))
    {
        return next.run(request).await;
    }

    if auth::has_valid_session_cookie(request.headers(), password) {
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
    let supplied = header_value
        .to_str()
        .ok()
        .and_then(|v| v.split_once(' '))
        .filter(|(scheme, _)| scheme.eq_ignore_ascii_case("bearer"))
        .map(|(_, credentials)| credentials);
    match supplied {
        None => unauthorized("Invalid authorization header format"),
        Some(credentials) if auth::constant_time_eq(credentials.as_bytes(), password.as_bytes()) => next.run(request).await,
        Some(_) => unauthorized("Invalid password"),
    }
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

        // The web UI's session cookie works too.
        let cookie = format!("{}={}", auth::SESSION_COOKIE, auth::session_token("hunter2"));
        let request = Request::builder().uri("/api/commands/jobs").header("cookie", cookie).body(Body::empty()).unwrap();
        let (status, _) = send(&app, request).await;
        assert_eq!(status, StatusCode::OK);

        // Preflight is never challenged.
        let request = Request::builder().method(Method::OPTIONS).uri("/api/commands/jobs").body(Body::empty()).unwrap();
        let (status, _) = send(&app, request).await;
        assert_ne!(status, StatusCode::UNAUTHORIZED);

        // Pages redirect to the login form instead of answering 401.
        let response = app.clone().oneshot(get("/some/page?x=1")).await.unwrap();
        assert_eq!(response.status(), StatusCode::SEE_OTHER);
        assert_eq!(response.headers()["location"], "/login?next=%2Fsome%2Fpage%3Fx%3D1");
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
