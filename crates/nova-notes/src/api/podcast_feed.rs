//! Token-gated Podcast Feed routes: `/api/podcasts/feed/{token}/...`.
//!
//! The one place the password middleware does not apply (see
//! `PUBLIC_PREFIXES` in `api/mod.rs`): the Feed Token in the path is the
//! credential. Everything here checks it in the service, and every miss is
//! the same `404 {"detail": "Not found"}`. GET routes answer HEAD too.
//!
//! Also home to [`serve_file`], the byte-range file response the episode
//! audio and artwork routes share.

use std::path::Path;

use axum::body::Body;
use axum::extract::rejection::PathRejection;
use axum::extract::{Path as UrlPath, State};
use axum::http::{header, HeaderMap, HeaderValue, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::get;
use axum::Router;
use tokio::io::{AsyncReadExt, AsyncSeekExt};
use tokio_util::io::ReaderStream;

use crate::config::Config;
use crate::error::{AppError, AppResult};
use crate::services::podcast_feed::{self, feed_not_found, ForwardedOrigin};
use crate::services::podcasts::ServedFile;
use crate::state::AppState;

pub fn router() -> Router<AppState> {
    Router::new()
        .route("/podcasts/feed/{token}/{name}", get(feed_document_or_artwork))
        .route("/podcasts/feed/{token}/episodes/{filename}", get(feed_episode_audio))
}

/// The request's own address, the last resort for the feed's public address.
/// Forwarded headers count only behind a trusted proxy (see
/// [`crate::api::perimeter::request_origin`]).
pub(crate) fn forwarded_origin(config: &Config, headers: &HeaderMap) -> ForwardedOrigin {
    let origin = crate::api::perimeter::request_origin(config, headers);
    ForwardedOrigin { proto: Some(if origin.https { "https" } else { "http" }.into()), host: origin.host }
}

/// `feed.xml` or `artwork.<ext>`; any other name is the uniform 404.
async fn feed_document_or_artwork(
    State(state): State<AppState>,
    path: Result<UrlPath<(String, String)>, PathRejection>,
    headers: HeaderMap,
) -> AppResult<Response> {
    let Ok(UrlPath((token, name))) = path else {
        return Err(feed_not_found());
    };
    if name == "feed.xml" {
        let xml = podcast_feed::feed_xml(&state, &token, &forwarded_origin(&state.config, &headers)).await?;
        let length = xml.len();
        return Ok((
            [
                (header::CONTENT_TYPE, HeaderValue::from_static(nn_podcast::feed::FEED_MEDIA_TYPE)),
                (header::CONTENT_LENGTH, HeaderValue::from(length)),
                // The links carry the Feed Token: no shared cache may keep it.
                (header::CACHE_CONTROL, HeaderValue::from_static("private")),
            ],
            xml,
        )
            .into_response());
    }
    match name.strip_prefix("artwork") {
        Some(extension) if extension.starts_with('.') => {
            let file = podcast_feed::feed_artwork(&state, &token, extension).await?;
            serve_file(&file, &headers, None).await.map_err(|_| feed_not_found())
        }
        _ => Err(feed_not_found()),
    }
}

/// A Published Episode's audio, with byte ranges (podcast apps seek and resume).
async fn feed_episode_audio(
    State(state): State<AppState>,
    path: Result<UrlPath<(String, String)>, PathRejection>,
    headers: HeaderMap,
) -> AppResult<Response> {
    let Ok(UrlPath((token, filename))) = path else {
        return Err(feed_not_found());
    };
    let file = podcast_feed::feed_episode_audio(&state, &token, &filename).await?;
    serve_file(&file, &headers, None).await.map_err(|_| feed_not_found())
}

// ---------------------------------------------------------------------------
// Files with byte ranges
// ---------------------------------------------------------------------------

/// What a `Range` header asks of a file of `length` bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RangeRequest {
    /// No (usable) range: send everything.
    Whole,
    /// `start..=end`.
    Part(u64, u64),
    /// Nothing of the file matches: 416.
    Unsatisfiable,
}

/// One `bytes=` range (`a-b`, `a-`, `-n`). Several ranges, other units and
/// malformed values are ignored, which RFC 9110 allows (the whole file is sent).
fn parse_range(value: Option<&str>, length: u64) -> RangeRequest {
    let Some(spec) = value.and_then(|v| v.trim().strip_prefix("bytes=")) else {
        return RangeRequest::Whole;
    };
    if spec.contains(',') {
        return RangeRequest::Whole;
    }
    let Some((start, end)) = spec.trim().split_once('-') else {
        return RangeRequest::Whole;
    };
    let (start, end) = (start.trim(), end.trim());
    if start.is_empty() {
        let Ok(suffix) = end.parse::<u64>() else {
            return RangeRequest::Whole;
        };
        if suffix == 0 || length == 0 {
            return RangeRequest::Unsatisfiable;
        }
        return RangeRequest::Part(length.saturating_sub(suffix), length - 1);
    }
    let Ok(start) = start.parse::<u64>() else {
        return RangeRequest::Whole;
    };
    let end = if end.is_empty() {
        length.saturating_sub(1)
    } else {
        match end.parse::<u64>() {
            Ok(end) if end >= start => end.min(length.saturating_sub(1)),
            _ => return RangeRequest::Whole,
        }
    };
    if start >= length {
        RangeRequest::Unsatisfiable
    } else {
        RangeRequest::Part(start, end)
    }
}

fn http_date(modified: std::time::SystemTime) -> Option<HeaderValue> {
    let at: chrono::DateTime<chrono::Utc> = modified.into();
    HeaderValue::from_str(&at.format("%a, %d %b %Y %H:%M:%S GMT").to_string()).ok()
}

/// Send a file with `Content-Length`, `Accept-Ranges` and single byte ranges
/// (206, or 416 with `Content-Range: bytes */<len>`), streamed from disk.
/// `attachment` adds `Content-Disposition: attachment; filename="..."`.
pub(crate) async fn serve_file(file: &ServedFile, headers: &HeaderMap, attachment: Option<&str>) -> std::io::Result<Response> {
    serve_path(&file.path, &file.media_type, headers, attachment).await
}

async fn serve_path(path: &Path, media_type: &str, headers: &HeaderMap, attachment: Option<&str>) -> std::io::Result<Response> {
    let mut handle = tokio::fs::File::open(path).await?;
    let metadata = handle.metadata().await?;
    let length = metadata.len();
    let range = parse_range(headers.get(header::RANGE).and_then(|v| v.to_str().ok()), length);

    let mut builder = Response::builder()
        .header(header::CONTENT_TYPE, media_type)
        .header(header::ACCEPT_RANGES, "bytes");
    if let Some(modified) = metadata.modified().ok().and_then(http_date) {
        builder = builder.header(header::LAST_MODIFIED, modified);
    }
    if let Some(name) = attachment {
        let safe: String = name.chars().filter(|c| c.is_ascii_graphic() && *c != '"' && *c != '\\').collect();
        builder = builder.header(header::CONTENT_DISPOSITION, format!("attachment; filename=\"{safe}\""));
    }

    let response = match range {
        RangeRequest::Unsatisfiable => builder
            .status(StatusCode::RANGE_NOT_SATISFIABLE)
            .header(header::CONTENT_RANGE, format!("bytes */{length}"))
            .body(Body::empty()),
        RangeRequest::Whole => builder
            .status(StatusCode::OK)
            .header(header::CONTENT_LENGTH, length)
            .body(Body::from_stream(ReaderStream::new(handle))),
        RangeRequest::Part(start, end) => {
            let count = end - start + 1;
            handle.seek(std::io::SeekFrom::Start(start)).await?;
            builder
                .status(StatusCode::PARTIAL_CONTENT)
                .header(header::CONTENT_RANGE, format!("bytes {start}-{end}/{length}"))
                .header(header::CONTENT_LENGTH, count)
                .body(Body::from_stream(ReaderStream::new(handle.take(count))))
        }
    };
    response.map_err(|e| std::io::Error::other(e.to_string()))
}

/// [`serve_file`] for routes where a vanished file is a plain error.
pub(crate) async fn serve_file_or_error(file: &ServedFile, headers: &HeaderMap, attachment: Option<&str>) -> AppResult<Response> {
    serve_file(file, headers, attachment).await.map_err(|e| {
        if e.kind() == std::io::ErrorKind::NotFound {
            AppError::not_found("File not found on disk")
        } else {
            AppError::from(e)
        }
    })
}

#[cfg(test)]
pub(crate) mod tests {
    use axum::body::Body;
    use axum::http::{Method, Request, StatusCode};
    use chrono::{Duration, Utc};
    use http_body_util::BodyExt;
    use serde_json::{json, Value};

    use super::{parse_range, RangeRequest};
    use crate::db::Doc;
    use crate::testing::{silent_mp3_frames, TestApp};

    pub(crate) const HOST: &str = "lab.example.ts.net:8502";
    pub(crate) const BASE: &str = "https://lab.example.ts.net:8502";

    /// An app behind a trusted proxy answering to [`HOST`], so the
    /// forwarded headers [`forwarded`] sends decide the feed's address.
    pub(crate) async fn proxied_app() -> TestApp {
        TestApp::with_config(|config| {
            config.trust_proxy_headers = true;
            config.allowed_hosts.push("lab.example.ts.net".into());
        })
        .await
    }

    /// A request carrying the forwarded headers the public address comes from.
    pub(crate) fn forwarded(method: Method, path: &str) -> Request<Body> {
        Request::builder()
            .method(method)
            .uri(path)
            .header("x-forwarded-proto", "https")
            .header("x-forwarded-host", HOST)
            .body(Body::empty())
            .unwrap()
    }

    pub(crate) async fn read(app: &TestApp, request: Request<Body>) -> (StatusCode, axum::http::HeaderMap, Vec<u8>) {
        let response = app.send(request).await;
        let status = response.status();
        let headers = response.headers().clone();
        let body = response.into_body().collect().await.unwrap().to_bytes().to_vec();
        (status, headers, body)
    }

    pub(crate) async fn token(app: &TestApp) -> String {
        crate::services::podcast_feed::ensure_feed_token(&app.state).await.unwrap()
    }

    /// Write `bytes` under the podcasts folder and create an episode pointing at it.
    pub(crate) async fn episode_with_audio(
        app: &TestApp,
        key: &str,
        relative: &str,
        bytes: &[u8],
        published: Option<chrono::DateTime<Utc>>,
    ) -> String {
        let root = app.state.config.podcasts_dir();
        if !bytes.is_empty() {
            let path = root.join(relative);
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
            std::fs::write(&path, bytes).unwrap();
        }
        let mut doc = Doc::new()
            .set("name", format!("Episode {key}"))
            .set("episode_profile", json!({ "name": "tech_discussion" }))
            .set("speaker_profile", json!({ "name": "tech_experts" }))
            .set("briefing", "Secret generation briefing")
            .set("content", "content")
            .set("audio_file", relative);
        if let Some(at) = published {
            doc = doc.datetime("published_at", at).set("duration_seconds", 42).set("feed_summary", format!("Notes for {key}"));
        }
        let row = app.state.db.create_with_id(&format!("episode:{key}"), doc).await.unwrap();
        row["id"].as_str().unwrap().to_string()
    }

    /// The full app with an instance password set.
    pub(crate) fn router_with_password(app: &TestApp, password: &str) -> axum::Router {
        let mut state = app.state.clone();
        let mut config = (*state.config).clone();
        config.password = Some(password.into());
        state.config = std::sync::Arc::new(config);
        crate::api::build_router(state)
    }

    fn items(xml: &str) -> Vec<String> {
        xml.split("<item>").skip(1).map(|item| item.split("</item>").next().unwrap().to_string()).collect()
    }

    #[test]
    fn ranges_parse_like_starlette() {
        assert_eq!(parse_range(None, 100), RangeRequest::Whole);
        assert_eq!(parse_range(Some("bytes=10-19"), 100), RangeRequest::Part(10, 19));
        assert_eq!(parse_range(Some("bytes=90-"), 100), RangeRequest::Part(90, 99));
        assert_eq!(parse_range(Some("bytes=90-500"), 100), RangeRequest::Part(90, 99));
        assert_eq!(parse_range(Some("bytes=-10"), 100), RangeRequest::Part(90, 99));
        assert_eq!(parse_range(Some("bytes=-500"), 100), RangeRequest::Part(0, 99));
        assert_eq!(parse_range(Some("bytes=100-"), 100), RangeRequest::Unsatisfiable);
        assert_eq!(parse_range(Some("bytes=-0"), 100), RangeRequest::Unsatisfiable);
        assert_eq!(parse_range(Some("bytes=5-1"), 100), RangeRequest::Whole);
        assert_eq!(parse_range(Some("bytes=0-1,5-6"), 100), RangeRequest::Whole);
        assert_eq!(parse_range(Some("items=0-1"), 100), RangeRequest::Whole);
    }

    #[tokio::test]
    async fn feed_answers_a_valid_token_and_a_uniform_404_otherwise() {
        let app = proxied_app().await;
        let token = token(&app).await;
        let feed = format!("/api/podcasts/feed/{token}/feed.xml");

        let (status, headers, body) = read(&app, forwarded(Method::GET, &feed)).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(headers["content-type"], "application/rss+xml");
        let xml = String::from_utf8(body.clone()).unwrap();
        assert!(xml.starts_with("<?xml version='1.0' encoding='utf-8'?>"), "{xml}");
        assert!(xml.contains("<title>Open Notebook Podcasts</title>"));
        assert!(xml.contains("<itunes:block>yes</itunes:block>"));
        assert!(xml.contains(&format!("href=\"{BASE}{feed}\"")), "self link: {xml}");
        assert!(items(&xml).is_empty());

        // HEAD: same headers, no body.
        let (status, head_headers, head_body) = read(&app, forwarded(Method::HEAD, &feed)).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(head_headers["content-type"], "application/rss+xml");
        assert_eq!(head_headers["content-length"], body.len().to_string().as_str());
        assert!(head_body.is_empty());

        for path in [
            "/api/podcasts/feed/not-the-token/feed.xml".to_string(),
            "/api/podcasts/feed/t%C3%B6k/feed.xml".to_string(),
            "/api/podcasts/feed/%FF%FE/feed.xml".to_string(),
            format!("/api/podcasts/feed/{token}/cover.png"),
            format!("/api/podcasts/feed/{token}/artwork.png"),
        ] {
            for method in [Method::GET, Method::HEAD] {
                let (status, _, body) = read(&app, forwarded(method.clone(), &path)).await;
                assert_eq!(status, StatusCode::NOT_FOUND, "{method} {path}");
                if method == Method::GET {
                    assert_eq!(serde_json::from_slice::<Value>(&body).unwrap(), json!({ "detail": "Not found" }), "{path}");
                }
            }
        }

        // A wrong token is 404 even when no public address can be built; the
        // right one explains what is missing.
        if std::env::var("OPEN_NOTEBOOK_PUBLIC_URL").is_err() && std::env::var("API_URL").is_err() {
            let (status, body) = app.get("/api/podcasts/feed/nope/feed.xml").await;
            assert_eq!((status, body), (StatusCode::NOT_FOUND, json!({ "detail": "Not found" })));
            let (status, body) = app.get(&feed).await;
            assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY);
            assert!(body["detail"].as_str().unwrap().contains("OPEN_NOTEBOOK_PUBLIC_URL"));
        }
    }

    /// Without `OPEN_NOTEBOOK_TRUST_PROXY_HEADERS`, a client cannot put a host
    /// of its choosing (and so the Feed Token) into the feed's links.
    #[tokio::test]
    async fn forwarded_headers_count_only_behind_a_trusted_proxy() {
        if std::env::var("OPEN_NOTEBOOK_PUBLIC_URL").is_ok() || std::env::var("API_URL").is_ok() {
            return;
        }
        let app = TestApp::with_config(|config| config.allowed_hosts.push("lab.example.ts.net".into())).await;
        let token = token(&app).await;
        let forged = |path: &str| {
            Request::builder()
                .uri(path)
                .header("x-forwarded-proto", "https")
                .header("x-forwarded-host", "evil.example")
                .body(Body::empty())
                .unwrap()
        };
        let (status, _, body) = read(&app, forged(&format!("/api/podcasts/feed/{token}/feed.xml"))).await;
        assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY);
        let body = String::from_utf8(body).unwrap();
        assert!(!body.contains("evil.example") && body.contains("OPEN_NOTEBOOK_TRUST_PROXY_HEADERS"), "{body}");
        let (status, _, body) = read(&app, forged("/api/podcasts/feed-url")).await;
        assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY);
        assert!(!String::from_utf8(body).unwrap().contains("evil.example"));

        // Behind a trusted proxy the forwarded host still has to be one of ours.
        let app = proxied_app().await;
        let (status, _, body) = read(&app, forged("/api/podcasts/feed-url")).await;
        assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY);
        assert!(!String::from_utf8(body).unwrap().contains("evil.example"));
        let (status, _, body) = read(&app, forwarded(Method::GET, "/api/podcasts/feed-url")).await;
        assert_eq!(status, StatusCode::OK);
        assert!(String::from_utf8(body).unwrap().contains(BASE));
    }

    /// The feed carries the Feed Token, so no shared cache may keep it; audio
    /// and artwork are never content-sniffed.
    #[tokio::test]
    async fn feed_responses_are_private_and_never_sniffed() {
        let app = proxied_app().await;
        let token = token(&app).await;
        let (status, headers, _) = read(&app, forwarded(Method::GET, &format!("/api/podcasts/feed/{token}/feed.xml"))).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(headers["cache-control"], "private");
        episode_with_audio(&app, "ep1", "episodes/ep1/audio/ep1.mp3", &silent_mp3_frames(5), Some(Utc::now())).await;
        let (status, headers, _) = read(&app, forwarded(Method::GET, &format!("/api/podcasts/feed/{token}/episodes/ep1.mp3"))).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(headers["x-content-type-options"], "nosniff");
    }

    #[tokio::test]
    async fn no_token_configured_looks_like_a_wrong_token() {
        let app = proxied_app().await;
        let (status, body) = app.get("/api/podcasts/feed/anything/feed.xml").await;
        assert_eq!((status, body), (StatusCode::NOT_FOUND, json!({ "detail": "Not found" })));
    }

    #[tokio::test]
    async fn published_episodes_become_items_newest_first() {
        let app = proxied_app().await;
        let token = token(&app).await;
        let audio = silent_mp3_frames(20);
        let now = Utc::now();
        episode_with_audio(&app, "old", "episodes/old/audio/old.mp3", &audio, Some(now - Duration::days(2))).await;
        episode_with_audio(&app, "new", "episodes/new/audio/new.mp3", &audio, Some(now - Duration::hours(1))).await;
        episode_with_audio(&app, "draft", "episodes/draft/audio/draft.mp3", &audio, None).await;
        episode_with_audio(&app, "m4a", "episodes/m4a/audio/m4a.m4a", b"fake m4a bytes", Some(now - Duration::days(1))).await;
        // Published, but its file is gone: left out.
        episode_with_audio(&app, "gone", "episodes/gone/audio/gone.mp3", b"", Some(now)).await;

        let (status, _, body) = read(&app, forwarded(Method::GET, &format!("/api/podcasts/feed/{token}/feed.xml"))).await;
        assert_eq!(status, StatusCode::OK);
        let xml = String::from_utf8(body).unwrap();
        let entries = items(&xml);
        assert_eq!(entries.len(), 3, "{xml}");
        assert!(entries[0].contains("<title>Episode new</title>"));
        assert!(entries[1].contains("<title>Episode m4a</title>"));
        assert!(entries[2].contains("<title>Episode old</title>"));
        assert!(entries[0].contains("<description>Notes for new</description>"));
        assert!(entries[0].contains("<guid isPermaLink=\"false\">open-notebook-episode-new</guid>"));
        assert!(entries[0].contains(&format!(
            "<enclosure url=\"{BASE}/api/podcasts/feed/{token}/episodes/new.mp3\" length=\"{}\" type=\"audio/mpeg\" />",
            audio.len()
        )));
        assert!(entries[0].contains("<itunes:duration>42</itunes:duration>"));
        assert!(entries[1].contains("type=\"audio/x-m4a\""));
        assert!(!xml.contains("Secret generation briefing"), "the briefing never reaches the feed");

        // Every enclosure URL serves the advertised file.
        for item in &entries {
            let url = item.split("<enclosure url=\"").nth(1).unwrap().split('"').next().unwrap();
            let (status, headers, body) = read(&app, forwarded(Method::GET, url.strip_prefix(BASE).unwrap())).await;
            assert_eq!(status, StatusCode::OK, "{url}");
            let length = item.split("length=\"").nth(1).unwrap().split('"').next().unwrap();
            assert_eq!(headers["content-length"], length);
            assert_eq!(body.len().to_string(), length);
        }

        // Deleting a published episode removes it from the feed.
        let (status, _) = app.delete("/api/podcasts/episodes/episode:new").await;
        assert_eq!(status, StatusCode::OK);
        let (_, _, body) = read(&app, forwarded(Method::GET, &format!("/api/podcasts/feed/{token}/feed.xml"))).await;
        assert_eq!(items(&String::from_utf8(body).unwrap()).len(), 2);
    }

    #[tokio::test]
    async fn episode_audio_supports_ranges_head_and_uniform_404s() {
        let app = proxied_app().await;
        let token = token(&app).await;
        let audio: Vec<u8> = (0..12_000u32).map(|i| (i % 251) as u8).collect();
        episode_with_audio(&app, "ep1", "episodes/ep1/audio/ep1.mp3", &audio, Some(Utc::now())).await;
        let url = format!("/api/podcasts/feed/{token}/episodes/ep1.mp3");
        let ranged = |range: &str| {
            Request::builder().uri(&url).header("range", range).body(Body::empty()).unwrap()
        };

        let (status, headers, body) = read(&app, forwarded(Method::GET, &url)).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(headers["content-type"], "audio/mpeg");
        assert_eq!(headers["content-length"], audio.len().to_string().as_str());
        assert_eq!(headers["accept-ranges"], "bytes");
        assert_eq!(body, audio);

        let (status, headers, body) = read(&app, ranged("bytes=100-199")).await;
        assert_eq!(status, StatusCode::PARTIAL_CONTENT);
        assert_eq!(headers["content-range"], format!("bytes 100-199/{}", audio.len()).as_str());
        assert_eq!(headers["content-length"], "100");
        assert_eq!(body, &audio[100..200]);
        let (status, _, body) = read(&app, ranged("bytes=0-1")).await;
        assert_eq!((status, body.as_slice()), (StatusCode::PARTIAL_CONTENT, &audio[0..2]));
        let (status, headers, body) = read(&app, ranged("bytes=10000-")).await;
        assert_eq!(status, StatusCode::PARTIAL_CONTENT);
        assert_eq!(headers["content-range"], format!("bytes 10000-{}/{}", audio.len() - 1, audio.len()).as_str());
        assert_eq!(body, &audio[10000..]);
        let (status, _, body) = read(&app, ranged("bytes=-50")).await;
        assert_eq!((status, body.as_slice()), (StatusCode::PARTIAL_CONTENT, &audio[audio.len() - 50..]));
        let (status, headers, _) = read(&app, ranged("bytes=20000-30000")).await;
        assert_eq!(status, StatusCode::RANGE_NOT_SATISFIABLE);
        assert_eq!(headers["content-range"], format!("bytes */{}", audio.len()).as_str());

        let (status, headers, body) = read(&app, forwarded(Method::HEAD, &url)).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(headers["content-length"], audio.len().to_string().as_str());
        assert_eq!(headers["accept-ranges"], "bytes");
        assert!(body.is_empty());

        // Every miss is the same 404.
        episode_with_audio(&app, "draft", "episodes/draft/audio/draft.mp3", &audio, None).await;
        episode_with_audio(&app, "vanished", "episodes/vanished/audio/vanished.mp3", b"", Some(Utc::now())).await;
        episode_with_audio(&app, "escape", "../../etc/passwd.mp3", b"", Some(Utc::now())).await;
        for path in [
            "/api/podcasts/feed/not-the-token/episodes/ep1.mp3".to_string(),
            format!("/api/podcasts/feed/{token}/episodes/ep9.mp3"),
            format!("/api/podcasts/feed/{token}/episodes/draft.mp3"),
            format!("/api/podcasts/feed/{token}/episodes/ep1.m4a"),
            format!("/api/podcasts/feed/{token}/episodes/ep1"),
            format!("/api/podcasts/feed/{token}/episodes/ep1.wav"),
            format!("/api/podcasts/feed/{token}/episodes/vanished.mp3"),
            format!("/api/podcasts/feed/{token}/episodes/escape.mp3"),
        ] {
            let (status, _, body) = read(&app, forwarded(Method::GET, &path)).await;
            assert_eq!(status, StatusCode::NOT_FOUND, "{path}");
            assert_eq!(serde_json::from_slice::<Value>(&body).unwrap(), json!({ "detail": "Not found" }), "{path}");
            let (status, _, _) = read(&app, forwarded(Method::HEAD, &path)).await;
            assert_eq!(status, StatusCode::NOT_FOUND, "HEAD {path}");
        }
    }

    #[tokio::test]
    async fn the_feed_needs_no_password_but_everything_else_does() {
        let app = proxied_app().await;
        let token = token(&app).await;
        episode_with_audio(&app, "ep1", "episodes/ep1/audio/ep1.mp3", &silent_mp3_frames(5), Some(Utc::now())).await;
        let router = router_with_password(&app, "hunter2");
        let send = |path: String, method: Method| {
            let router = router.clone();
            async move {
                use tower::ServiceExt;
                let request = Request::builder()
                    .method(method)
                    .uri(path)
                    .header("x-forwarded-proto", "https")
                    .header("x-forwarded-host", HOST)
                    .body(Body::empty())
                    .unwrap();
                router.oneshot(request).await.unwrap().status()
            }
        };
        assert_eq!(send(format!("/api/podcasts/feed/{token}/feed.xml"), Method::GET).await, StatusCode::OK);
        assert_eq!(send(format!("/api/podcasts/feed/{token}/feed.xml"), Method::HEAD).await, StatusCode::OK);
        assert_eq!(send(format!("/api/podcasts/feed/{token}/episodes/ep1.mp3"), Method::GET).await, StatusCode::OK);
        assert_eq!(send("/api/podcasts/feed/nope/feed.xml".into(), Method::GET).await, StatusCode::NOT_FOUND);
        assert_eq!(send("/api/podcasts/feed/nope/episodes/ep1.mp3".into(), Method::GET).await, StatusCode::NOT_FOUND);
        for path in ["/api/podcasts/episodes", "/api/podcasts/feed-url", "/api/podcasts/feed-settings", "/api/podcasts/episodes/episode:ep1/audio"] {
            assert_eq!(send(path.into(), Method::GET).await, StatusCode::UNAUTHORIZED, "{path}");
        }
    }
}
