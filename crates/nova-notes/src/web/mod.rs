//! The server-rendered web UI.
//!
//! Pages are rendered with maud and enhanced with htmx: forms post to the
//! server, which answers with an HTML fragment (or a redirect) instead of
//! JSON. Every page calls the same [`crate::services`] functions the JSON API
//! uses, so the two can never disagree about behaviour.
//!
//! This module holds the shared pieces: the page layout, static assets, the
//! login page, markdown rendering and error pages. Feature pages live in the
//! submodules.

use axum::extract::{Path, Query, State};
use axum::http::{header, HeaderMap, HeaderValue, StatusCode};
use axum::response::{IntoResponse, Redirect, Response};
use axum::routing::get;
use axum::{Form, Router};
use include_dir::{include_dir, Dir};
use maud::{html, Markup, PreEscaped, DOCTYPE};
use serde::Deserialize;

use crate::api::auth;
use crate::error::AppError;
use crate::state::AppState;

mod helpers;
pub use helpers::*;

pub mod discover;
pub mod notebooks;
pub mod podcasts;
pub mod search;
pub mod settings;
pub mod sources;
pub mod studio;
pub mod workspace;

static STATIC: Dir<'static> = include_dir!("$CARGO_MANIFEST_DIR/static");

/// Feature areas. Each has `src/web/<area>.rs`, `static/css/<area>.css` and
/// `static/js/<area>.js`; every page loads all of them.
const AREAS: [&str; 8] = ["notebooks", "studio", "sources", "search", "podcasts", "settings", "discover", "workspace"];

pub fn router() -> Router<AppState> {
    Router::new()
        .route("/", get(|| async { Redirect::to("/notes") }))
        .route("/static/{*path}", get(static_file))
        .route("/login", get(login_page).post(login_submit))
        .route("/logout", get(logout).post(logout))
        .merge(notebooks::router())
        .merge(studio::router())
        .merge(sources::router())
        .merge(search::router())
        .merge(podcasts::router())
        .merge(settings::router())
        .merge(discover::router())
        .merge(workspace::router())
        .fallback(not_found_page)
}

// ---------------------------------------------------------------------------
// Layout
// ---------------------------------------------------------------------------

/// One entry of the sidebar.
pub(crate) struct NavItem {
    pub(crate) label: &'static str,
    pub(crate) href: &'static str,
    pub(crate) colour: &'static str,
}

/// The sidebar, grouped as in the Next.js frontend: collect, process, create, manage.
/// The Notes view lists the same pages under "Studio".
pub(crate) const NAVIGATION: [(&str, &[NavItem]); 3] = [
    (
        "Process",
        &[
            NavItem { label: "Notebooks", href: "/notebooks", colour: "var(--teal)" },
            NavItem { label: "Ask and Search", href: "/search", colour: "var(--gold)" },
        ],
    ),
    ("Create", &[NavItem { label: "Podcasts", href: "/podcasts", colour: "var(--mauve)" }]),
    (
        "Manage",
        &[
            NavItem { label: "Sources", href: "/sources", colour: "var(--sage)" },
            NavItem { label: "Models", href: "/settings/models", colour: "var(--plum)" },
            NavItem { label: "Transformations", href: "/transformations", colour: "var(--fern)" },
            NavItem { label: "Settings", href: "/settings", colour: "var(--ink-faint)" },
            NavItem { label: "Advanced", href: "/advanced", colour: "var(--ink-faint)" },
        ],
    ),
];

/// The nav item to highlight: the longest href that prefixes the path, so
/// `/settings/models` highlights Models and not Settings.
fn active_href(path: &str) -> Option<&'static str> {
    NAVIGATION
        .iter()
        .flat_map(|(_, items)| items.iter())
        .map(|item| item.href)
        .filter(|href| path == *href || path.starts_with(&format!("{href}/")))
        .max_by_key(|href| href.len())
}

/// Context every page needs to render its chrome.
#[derive(Debug, Clone, Default)]
pub struct PageContext {
    /// Path of the current page, for highlighting the nav.
    pub path: String,
    /// Whether a password is configured (shows the sign-out button).
    pub auth_enabled: bool,
}

impl PageContext {
    pub fn new(state: &AppState, path: impl Into<String>) -> Self {
        Self { path: path.into(), auth_enabled: state.config.password.is_some() }
    }
}

pub(crate) fn head(title: &str) -> Markup {
    html! {
        meta charset="utf-8";
        meta name="viewport" content="width=device-width, initial-scale=1";
        meta name="color-scheme" content="light dark";
        title { (title) " · Nova Notes" }
        script { (PreEscaped(r#"
            try {
                if (localStorage.getItem("nn-sidebar-collapsed") === "true") {
                    document.documentElement.classList.add("sidebar-collapsed");
                }
            } catch (error) {}
        "#)) }
        link rel="stylesheet" href=(asset_url("app.css"));
        @for area in AREAS {
            link rel="stylesheet" href=(asset_url(&format!("css/{area}.css")));
        }
        script src=(asset_url("vendor/htmx.min.js")) defer {}
        script src=(asset_url("app.js")) defer {}
        @for area in AREAS {
            script src=(asset_url(&format!("js/{area}.js"))) defer {}
        }
    }
}

/// A full page: sidebar, main column, toast area.
pub fn page(ctx: &PageContext, title: &str, body: Markup) -> Markup {
    let active = active_href(&ctx.path);
    html! {
        (DOCTYPE)
        html lang="en" {
            head { (head(title)) }
            body hx-headers=r#"{"X-Requested-With": "htmx"}"# {
                div.shell {
                    nav.sidebar id="site-sidebar" aria-label="Main" {
                        div.sidebar-header {
                            a.brand href="/notebooks" {
                                span.pebbles aria-hidden="true" { i {} i {} i {} }
                                "Nova Notes"
                            }
                            button.btn.btn-ghost.btn-icon.sidebar-toggle type="button" data-sidebar-toggle
                                aria-controls="sidebar-links" aria-expanded="true" aria-label="Collapse sidebar" title="Collapse sidebar" {
                                (PreEscaped(r#"<svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.3" aria-hidden="true"><rect x="1.5" y="2.5" width="13" height="11" rx="2"/><path d="M6 2.5v11"/></svg>"#))
                            }
                        }
                        // Studio pages sit beside the Notes view; this is the way back.
                        a.btn.back-to-notes href="/notes" title="Back to Notes" aria-label="Back to Notes" {
                            span aria-hidden="true" { "‹" } span.back-to-notes-label { "Notes" }
                        }
                        div.sidebar-links id="sidebar-links" {
                            div.nav-group {
                                div.nav-title { "Collect" }
                                form method="post" action="/notes/quick" data-note-create {
                                    button.nav-link.nav-action type="submit" {
                                        span.dot.quick-note-dot aria-hidden="true" { "+" }
                                        "Quick Note"
                                    }
                                }
                            }
                            @for (group, items) in NAVIGATION {
                                div.nav-group {
                                    div.nav-title { (group) }
                                    @for item in items {
                                        a.nav-link href=(item.href) aria-current=[(active == Some(item.href)).then_some("page")] {
                                            span.dot style=(format!("background:{}", item.colour)) {}
                                            (item.label)
                                        }
                                    }
                                }
                            }
                            div.sidebar-foot {
                                button.btn.btn-ghost.btn-sm type="button" data-theme-toggle { "Toggle theme" }
                                @if ctx.auth_enabled {
                                    form method="post" action="/logout" {
                                        button.btn.btn-ghost.btn-sm type="submit" { "Sign out" }
                                    }
                                }
                                span { "v" (crate::VERSION) }
                            }
                        }
                    }
                    main.main id="main" { (body) }
                }
                button.btn.menu-toggle type="button" data-menu-toggle aria-controls="site-sidebar" aria-expanded="false" { "Menu" }
                a.btn.notes-fab href="/notes" { "‹ Notes" }
                div id="toasts" aria-live="polite" {}
            }
        }
    }
}

/// A page heading with an optional subtitle and actions on the right.
pub fn page_head(title: &str, subtitle: Option<&str>, actions: Markup) -> Markup {
    html! {
        div.page-head {
            div {
                h1 { (title) }
                @if let Some(subtitle) = subtitle { p { (subtitle) } }
            }
            div.row { (actions) }
        }
    }
}

/// An empty-state box.
pub fn empty_state(title: &str, hint: &str) -> Markup {
    html! { div.empty { strong { (title) } (hint) } }
}

/// A status badge for a background job status (`new`, `running`, `completed`, ...).
pub fn status_badge(status: Option<&str>) -> Markup {
    let (class, label) = match status.unwrap_or("") {
        "completed" => ("ok", "Ready"),
        "running" => ("info", "Processing"),
        "new" | "queued" | "pending" => ("info", "Queued"),
        "failed" | "error" => ("bad", "Failed"),
        "canceled" | "cancelled" => ("warn", "Canceled"),
        "" => return html! {},
        other => ("", other),
    };
    html! { span class=(format!("badge {class}")) { (label) } }
}

/// A short, human date ("13 Nov 2026, 19:24") from an optional timestamp.
pub fn human_date(value: Option<chrono::DateTime<chrono::Utc>>) -> String {
    value.map(|d| d.format("%-d %b %Y, %H:%M").to_string()).unwrap_or_default()
}

// ---------------------------------------------------------------------------
// Markdown
// ---------------------------------------------------------------------------

/// Render untrusted markdown (model output, user notes, extracted web pages)
/// to sanitised HTML.
pub fn markdown(text: &str) -> Markup {
    use pulldown_cmark::{html::push_html, Options, Parser};
    let mut options = Options::empty();
    options.insert(Options::ENABLE_TABLES);
    options.insert(Options::ENABLE_STRIKETHROUGH);
    options.insert(Options::ENABLE_TASKLISTS);
    let mut raw = String::with_capacity(text.len() * 3 / 2);
    push_html(&mut raw, Parser::new_ext(text, options));
    let clean = ammonia::Builder::default()
        .link_rel(Some("noopener noreferrer nofollow"))
        .add_generic_attributes(["align"])
        .clean(&raw)
        .to_string();
    html! { div.prose { (PreEscaped(clean)) } }
}

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// Whether the request came from htmx (and so wants a fragment, not a page).
pub fn is_htmx(headers: &HeaderMap) -> bool {
    headers.contains_key("hx-request")
}

/// An [`AppError`] rendered as HTML. Handlers in the web UI return
/// `Result<_, WebError>`; `?` converts any `AppError`.
#[derive(Debug)]
pub struct WebError(pub AppError);

impl From<AppError> for WebError {
    fn from(e: AppError) -> Self {
        Self(e)
    }
}

pub type WebResult<T> = Result<T, WebError>;

impl IntoResponse for WebError {
    fn into_response(self) -> Response {
        let status = self.0.status();
        if status.is_server_error() {
            tracing::error!(%status, "{}", self.0);
        }
        // The client script reads `data-error-detail` for its toast, so the
        // same body serves both a full page load and an htmx swap.
        let body = html! {
            (DOCTYPE)
            html lang="en" {
                head { (head("Error")) }
                body {
                    div.center-page {
                        div.card.stack {
                            h2 { (status.canonical_reason().unwrap_or("Error")) }
                            p.muted data-error-detail { (self.0.detail()) }
                            div.row { a.btn href="/notebooks" { "Back to notebooks" } }
                        }
                    }
                }
            }
        };
        (status, body).into_response()
    }
}

async fn not_found_page() -> Response {
    WebError(AppError::not_found("There is no page at this address.")).into_response()
}

// ---------------------------------------------------------------------------
// Static files
// ---------------------------------------------------------------------------

/// URL of an embedded static file, with a suffix that changes whenever the
/// file's contents do, so browsers pick up a new build at once rather than
/// keeping the cached copy for a day.
pub fn asset_url(path: &str) -> String {
    use std::hash::{Hash, Hasher};
    let Some(file) = STATIC.get_file(path) else {
        return format!("/static/{path}?v={}", crate::VERSION);
    };
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    file.contents().hash(&mut hasher);
    format!("/static/{path}?v={:016x}", hasher.finish())
}

async fn static_file(Path(path): Path<String>) -> Response {
    let Some(file) = STATIC.get_file(&path) else {
        return (StatusCode::NOT_FOUND, "Not found").into_response();
    };
    let mime = mime_guess::from_path(&path).first_or_octet_stream();
    (
        [
            (header::CONTENT_TYPE, HeaderValue::from_str(mime.as_ref()).unwrap_or(HeaderValue::from_static("application/octet-stream"))),
            (header::CACHE_CONTROL, HeaderValue::from_static("public, max-age=86400")),
            (header::X_CONTENT_TYPE_OPTIONS, HeaderValue::from_static("nosniff")),
        ],
        file.contents(),
    )
        .into_response()
}

// ---------------------------------------------------------------------------
// Login
// ---------------------------------------------------------------------------

#[derive(Debug, Default, Deserialize)]
struct LoginQuery {
    next: Option<String>,
    error: Option<String>,
}

/// Only same-site relative paths are followed after login.
fn safe_next(next: Option<&str>) -> String {
    safe_path_or(next, "/notebooks")
}

fn login_form(next: &str, error: Option<&str>) -> Markup {
    html! {
        (DOCTYPE)
        html lang="en" {
            head { (head("Sign in")) }
            body {
                div.center-page {
                    form.card.stack method="post" action="/login" {
                        div.row {
                            span.pebbles aria-hidden="true" { i {} i {} i {} }
                            h2 { "Nova Notes" }
                        }
                        @if let Some(error) = error { div.flash.bad role="alert" { (error) } }
                        input type="hidden" name="next" value=(next);
                        div.field {
                            label for="password" { "Password" }
                            input #password type="password" name="password" autocomplete="current-password" required autofocus;
                        }
                        button.btn.btn-primary type="submit" { "Sign in" }
                    }
                }
            }
        }
    }
}

async fn login_page(State(state): State<AppState>, Query(query): Query<LoginQuery>) -> Response {
    let next = safe_next(query.next.as_deref());
    if state.config.password.is_none() {
        return Redirect::to(&next).into_response();
    }
    let error = query.error.as_deref().map(|_| "That password is not right.");
    login_form(&next, error).into_response()
}

#[derive(Debug, Deserialize)]
struct LoginForm {
    password: String,
    next: Option<String>,
}

fn session_cookie(value: &str, max_age: i64, secure: bool) -> HeaderValue {
    let secure = if secure { "; Secure" } else { "" };
    HeaderValue::from_str(&format!(
        "{}={value}; Path=/; HttpOnly; SameSite=Lax; Max-Age={max_age}{secure}",
        auth::SESSION_COOKIE
    ))
    .expect("cookie value is ASCII")
}

/// Mark the cookie Secure when the request reached us over HTTPS (directly or
/// through a proxy such as Tailscale Serve), so it still works on plain-HTTP
/// localhost.
fn is_https(headers: &HeaderMap) -> bool {
    headers
        .get("x-forwarded-proto")
        .and_then(|v| v.to_str().ok())
        .is_some_and(|v| v.eq_ignore_ascii_case("https"))
}

async fn login_submit(State(state): State<AppState>, headers: HeaderMap, Form(form): Form<LoginForm>) -> Response {
    let next = safe_next(form.next.as_deref());
    let Some(password) = state.config.password.as_deref() else {
        return Redirect::to(&next).into_response();
    };
    if !auth::constant_time_eq(form.password.as_bytes(), password.as_bytes()) {
        tracing::warn!("Failed sign-in attempt");
        let target = format!("/login?error=1&next={}", auth::encode_query_value(&next));
        return Redirect::to(&target).into_response();
    }
    let cookie = session_cookie(&auth::session_token(password), 60 * 60 * 24 * 30, is_https(&headers));
    let mut response = Redirect::to(&next).into_response();
    response.headers_mut().insert(header::SET_COOKIE, cookie);
    response
}

async fn logout(headers: HeaderMap) -> Response {
    let mut response = Redirect::to("/login").into_response();
    response.headers_mut().insert(header::SET_COOKIE, session_cookie("", 0, is_https(&headers)));
    response
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::Body;
    use axum::http::{Method, Request};

    use crate::testing::TestApp;

    #[test]
    fn nav_highlights_the_most_specific_item() {
        assert_eq!(active_href("/settings/models"), Some("/settings/models"));
        assert_eq!(active_href("/settings"), Some("/settings"));
        assert_eq!(active_href("/notebooks/notebook:abc"), Some("/notebooks"));
        assert_eq!(active_href("/notes"), None);
        assert_eq!(active_href("/sourcesx"), None);
    }

    #[test]
    fn next_paths_stay_on_this_site() {
        assert_eq!(safe_next(Some("/notebooks/x")), "/notebooks/x");
        assert_eq!(safe_next(Some("//evil.example")), "/notebooks");
        assert_eq!(safe_next(Some("https://evil.example")), "/notebooks");
        assert_eq!(safe_next(Some("/\\evil.example")), "/notebooks");
        assert_eq!(safe_next(None), "/notebooks");
    }

    #[test]
    fn markdown_is_sanitised() {
        let out = markdown("# Title\n\n<script>alert(1)</script>\n\n[x](javascript:alert(1)) **bold**\n\n| a | b |\n|---|---|\n| 1 | 2 |").into_string();
        assert!(out.contains("<h1>Title</h1>"), "{out}");
        assert!(out.contains("<strong>bold</strong>"));
        assert!(out.contains("<table>"));
        assert!(!out.contains("<script"), "{out}");
        assert!(!out.contains("javascript:"), "{out}");
    }

    #[tokio::test]
    async fn static_assets_are_served() {
        let app = TestApp::new().await;
        for (path, mime) in [("/static/app.css", "text/css"), ("/static/app.js", "javascript"), ("/static/vendor/htmx.min.js", "javascript")] {
            let response = app.send(Request::builder().uri(path).body(Body::empty()).unwrap()).await;
            assert_eq!(response.status(), StatusCode::OK, "{path}");
            assert!(response.headers()[header::CONTENT_TYPE].to_str().unwrap().contains(mime), "{path}");
        }
        let (status, _) = app.text(Method::GET, "/static/nope.css", None).await;
        assert_eq!(status, StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn unknown_pages_render_an_html_404() {
        let app = TestApp::new().await;
        let (status, body) = app.text(Method::GET, "/no/such/page", None).await;
        assert_eq!(status, StatusCode::NOT_FOUND);
        assert!(body.contains("data-error-detail"), "{body}");
        // The JSON API keeps its JSON 404.
        let (status, body) = app.get("/api/no/such/route").await;
        assert_eq!(status, StatusCode::NOT_FOUND);
        assert_eq!(body["detail"], "Not Found");
    }

    #[tokio::test]
    async fn login_sets_a_session_cookie_that_opens_the_app() {
        let mut app = TestApp::new().await;
        let mut config = (*app.state.config).clone();
        config.password = Some("hunter2".into());
        app.state.config = std::sync::Arc::new(config);
        app.router = crate::api::build_router(app.state.clone());

        // Without a session, pages redirect to the login form.
        let response = app.send(Request::builder().uri("/notebooks").body(Body::empty()).unwrap()).await;
        assert_eq!(response.status(), StatusCode::SEE_OTHER);
        let (status, body) = app.text(Method::GET, "/login?next=/notebooks", None).await;
        assert_eq!(status, StatusCode::OK);
        assert!(body.contains("name=\"password\""));

        let form = |password: &str| {
            Request::builder()
                .method(Method::POST)
                .uri("/login")
                .header(header::CONTENT_TYPE, "application/x-www-form-urlencoded")
                .body(Body::from(format!("password={password}&next=%2Fsources")))
                .unwrap()
        };
        let wrong = app.send(form("nope")).await;
        assert_eq!(wrong.status(), StatusCode::SEE_OTHER);
        assert!(wrong.headers()[header::LOCATION].to_str().unwrap().starts_with("/login?error=1"));
        assert!(wrong.headers().get(header::SET_COOKIE).is_none());

        let right = app.send(form("hunter2")).await;
        assert_eq!(right.status(), StatusCode::SEE_OTHER);
        assert_eq!(right.headers()[header::LOCATION], "/sources");
        let cookie = right.headers()[header::SET_COOKIE].to_str().unwrap().to_string();
        assert!(cookie.contains("HttpOnly") && cookie.contains("SameSite=Lax") && !cookie.contains("Secure"), "{cookie}");
        let pair = cookie.split(';').next().unwrap().to_string();

        let (status, _) = {
            let response = app
                .send(Request::builder().uri("/api/commands/jobs").header(header::COOKIE, &pair).body(Body::empty()).unwrap())
                .await;
            (response.status(), ())
        };
        assert_eq!(status, StatusCode::OK);

        let out = app.send(Request::builder().method(Method::POST).uri("/logout").body(Body::empty()).unwrap()).await;
        assert!(out.headers()[header::SET_COOKIE].to_str().unwrap().contains("Max-Age=0"));
    }
}
