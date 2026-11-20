//! Show Metadata and artwork of the Podcast Feed: `/api/podcasts/feed-settings`.
//!
//! The Rust counterpart of `api/routers/podcast_feed_settings.py`. These
//! routes need the app password; the token-gated artwork a podcast app
//! fetches is in `api::podcast_feed`.

use axum::extract::{Multipart, State};
use axum::http::HeaderMap;
use axum::response::{IntoResponse, Response};
use axum::routing::get;
use axum::{Json, Router};
use nn_podcast::artwork::MAX_BYTES;
use nn_podcast::feed::{FeedSettingsUpdate, FeedSettingsView};

use crate::api::podcast_feed::serve_file_or_error;
use crate::api::{validation_error, ApiJson};
use crate::error::{AppError, AppResult};
use crate::services::podcast_feed;
use crate::state::AppState;

pub fn router() -> Router<AppState> {
    Router::new()
        .route("/podcasts/feed-settings", get(get_settings).put(update_settings))
        .route("/podcasts/feed-settings/artwork", get(artwork_preview).post(upload_artwork))
}

async fn get_settings(State(state): State<AppState>) -> AppResult<Json<FeedSettingsView>> {
    Ok(Json(podcast_feed::feed_settings(&state).await?))
}

async fn update_settings(
    State(state): State<AppState>,
    ApiJson(update): ApiJson<FeedSettingsUpdate>,
) -> AppResult<Json<FeedSettingsView>> {
    Ok(Json(podcast_feed::update_feed_settings(&state, &update).await?))
}

/// Read the `file` part, keeping at most `MAX_BYTES + 1` bytes: one byte
/// over the limit is enough to know the file is too large.
async fn read_file_part(multipart: &mut Multipart) -> Result<Option<Vec<u8>>, Box<Response>> {
    let bad_body = |e: axum::extract::multipart::MultipartError| {
        let status = e.status();
        Box::new(if status == axum::http::StatusCode::PAYLOAD_TOO_LARGE {
            (status, e.body_text()).into_response()
        } else {
            validation_error("body", "value_error", e.body_text())
        })
    };
    while let Some(mut field) = multipart.next_field().await.map_err(bad_body)? {
        if field.name() != Some("file") {
            continue;
        }
        let mut data = Vec::new();
        while let Some(chunk) = field.chunk().await.map_err(bad_body)? {
            let room = (MAX_BYTES + 1).saturating_sub(data.len());
            data.extend_from_slice(&chunk[..chunk.len().min(room)]);
            if data.len() > MAX_BYTES {
                break;
            }
        }
        return Ok(Some(data));
    }
    Ok(None)
}

/// Replace the show artwork (square PNG or JPEG, 1400-3000 px, 10 MB at most).
async fn upload_artwork(State(state): State<AppState>, mut multipart: Multipart) -> Response {
    let data = match read_file_part(&mut multipart).await {
        Ok(Some(data)) => data,
        Ok(None) => return validation_error("body", "missing", "Field required: file"),
        Err(response) => return *response,
    };
    match podcast_feed::upload_artwork(&state, data).await {
        Ok(view) => Json::<FeedSettingsView>(view).into_response(),
        Err(e) => e.into_response(),
    }
}

/// The current artwork, for the settings page preview.
async fn artwork_preview(State(state): State<AppState>, headers: HeaderMap) -> AppResult<Response> {
    let file = podcast_feed::artwork_preview(&state).await?;
    serve_file_or_error(&file, &headers, None).await.map_err(|e| {
        if e.is_not_found() {
            AppError::not_found("No artwork has been uploaded")
        } else {
            e
        }
    })
}

#[cfg(test)]
mod tests {
    use axum::body::Body;
    use axum::http::{header, Method, Request, StatusCode};
    use serde_json::{json, Value};

    use crate::api::podcast_feed::tests::{forwarded, read, token, BASE};
    use crate::testing::{multipart_body, TestApp};

    /// A structurally valid JPEG header of the given size (what the artwork
    /// check reads), with some scan data.
    fn jpeg(size: u16, seed: u8) -> Vec<u8> {
        let mut out = vec![0xFF, 0xD8, 0xFF, 0xE0, 0x00, 0x10];
        out.extend(b"JFIF\0\x01\x01\0\0\x01\0\x01\0\0");
        out.extend([0xFF, 0xDB, 0x00, 0x43, 0x00]);
        out.extend([seed.max(1); 64]);
        out.extend([0xFF, 0xC0, 0x00, 0x11, 0x08]);
        out.extend(size.to_be_bytes());
        out.extend(size.to_be_bytes());
        out.extend([0x03, 0x01, 0x22, 0x00, 0x02, 0x11, 0x01, 0x03, 0x11, 0x01]);
        out.extend([0xFF, 0xDA, 0x00, 0x0C, 0x03, 0x01, 0x00, 0x02, 0x11, 0x03, 0x11, 0x00, 0x3F, 0x00]);
        out.extend([seed; 40]);
        out.extend([0xFF, 0xD9]);
        out
    }

    async fn upload(app: &TestApp, field: &str, bytes: &[u8]) -> (StatusCode, Value) {
        let (content_type, body) = multipart_body(&[(field, Some("cover.png"), bytes)]);
        let request = Request::builder()
            .method(Method::POST)
            .uri("/api/podcasts/feed-settings/artwork")
            .header(header::CONTENT_TYPE, content_type)
            .body(Body::from(body))
            .unwrap();
        let (status, _, body) = read(app, request).await;
        (status, serde_json::from_slice(&body).unwrap_or(Value::Null))
    }

    #[tokio::test]
    async fn settings_default_edit_trim_and_refuse() {
        let app = TestApp::new().await;
        let (status, body) = app.get("/api/podcasts/feed-settings").await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(body["title"], "Open Notebook Podcasts");
        assert_eq!(body["has_artwork"], false);
        assert_eq!(body["artwork_version"], Value::Null);
        let default_author = body["author"].clone();

        let (status, body) = app
            .put("/api/podcasts/feed-settings", json!({ "title": "  Lab Radio  ", "author": "Dan", "language": "pt-BR" }))
            .await;
        assert_eq!(status, StatusCode::OK, "{body}");
        assert_eq!((body["title"].as_str(), body["author"].as_str(), body["language"].as_str()), (Some("Lab Radio"), Some("Dan"), Some("pt-BR")));

        // A field left out keeps its value; a blank one goes back to the default.
        let (_, body) = app.put("/api/podcasts/feed-settings", json!({ "author": "" })).await;
        assert_eq!(body["title"], "Lab Radio");
        assert_eq!(body["author"], default_author);

        // A refused edit changes nothing, and names the field.
        let (status, body) =
            app.put("/api/podcasts/feed-settings", json!({ "title": "New", "language": "not a language" })).await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert!(body["detail"].as_str().unwrap().contains("language"), "{body}");
        let (status, body) = app.put("/api/podcasts/feed-settings", json!({ "title": "x".repeat(256) })).await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert!(body["detail"].as_str().unwrap().contains("title"), "{body}");
        let (status, _) = app.put("/api/podcasts/feed-settings", json!({ "title": "x".repeat(255) })).await;
        assert_eq!(status, StatusCode::OK);
        let (_, body) = app.put("/api/podcasts/feed-settings", json!({ "title": "Lab Radio" })).await;
        assert_eq!(body["language"], "pt-BR");

        // The feed shows the edits, escaped.
        let (_, body) = app.put("/api/podcasts/feed-settings", json!({ "description": "<b>Tom & Jerry</b>" })).await;
        assert_eq!(body["description"], "<b>Tom & Jerry</b>");
        let token = token(&app).await;
        let (status, _, xml) = read(&app, forwarded(Method::GET, &format!("/api/podcasts/feed/{token}/feed.xml"))).await;
        assert_eq!(status, StatusCode::OK);
        let xml = String::from_utf8(xml).unwrap();
        assert!(xml.contains("<title>Lab Radio</title>"), "{xml}");
        assert!(!xml.contains("<b>Tom"), "{xml}");

        // Bad JSON is FastAPI's 422.
        let (status, _) = app.put("/api/podcasts/feed-settings", json!({ "title": 5 })).await;
        assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY);
    }

    #[tokio::test]
    async fn artwork_upload_preview_and_feed() {
        let app = TestApp::new().await;
        let (status, body) = app.get("/api/podcasts/feed-settings/artwork").await;
        assert_eq!((status, body), (StatusCode::NOT_FOUND, json!({ "detail": "No artwork has been uploaded" })));

        // Refusals store nothing.
        for (bytes, keyword) in [
            (Vec::new(), "empty"),
            (b"plain text, not an image".to_vec(), "image"),
            (jpeg(1399, 1), "1400"),
            (jpeg(3001, 1), "3000"),
        ] {
            let (status, body) = upload(&app, "file", &bytes).await;
            assert_eq!(status, StatusCode::BAD_REQUEST, "{keyword}: {body}");
            assert!(body["detail"].as_str().unwrap().to_lowercase().contains(keyword), "{keyword}: {body}");
        }
        let (_, body) = app.get("/api/podcasts/feed-settings").await;
        assert_eq!(body["has_artwork"], false);

        // No `file` part.
        let (status, _) = upload(&app, "image", &jpeg(1400, 1)).await;
        assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY);

        let image = jpeg(1400, 7);
        let (status, body) = upload(&app, "file", &image).await;
        assert_eq!(status, StatusCode::OK, "{body}");
        assert_eq!(body["has_artwork"], true);
        let version = body["artwork_version"].as_str().unwrap().to_string();

        let (status, headers, preview) = read(&app, forwarded(Method::GET, "/api/podcasts/feed-settings/artwork")).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(headers["content-type"], "image/jpeg");
        assert_eq!(preview, image);

        // A podcast app fetches it through the token-gated URL in the feed.
        let token = token(&app).await;
        let (_, _, xml) = read(&app, forwarded(Method::GET, &format!("/api/podcasts/feed/{token}/feed.xml"))).await;
        let xml = String::from_utf8(xml).unwrap();
        let artwork_url = format!("{BASE}/api/podcasts/feed/{token}/artwork.jpg");
        assert!(xml.contains(&artwork_url), "{xml}");
        assert!(xml.contains(&version), "{xml}");
        let (status, headers, served) =
            read(&app, forwarded(Method::GET, &format!("/api/podcasts/feed/{token}/artwork.jpg"))).await;
        assert_eq!((status, served.as_slice()), (StatusCode::OK, image.as_slice()));
        assert_eq!(headers["content-type"], "image/jpeg");
        let (status, _, body) = read(&app, forwarded(Method::HEAD, &format!("/api/podcasts/feed/{token}/artwork.jpg"))).await;
        assert_eq!(status, StatusCode::OK);
        assert!(body.is_empty());
        for path in ["/api/podcasts/feed/nope/artwork.jpg".to_string(), format!("/api/podcasts/feed/{token}/artwork.png")] {
            let (status, _, body) = read(&app, forwarded(Method::GET, &path)).await;
            assert_eq!(status, StatusCode::NOT_FOUND, "{path}");
            assert_eq!(serde_json::from_slice::<Value>(&body).unwrap(), json!({ "detail": "Not found" }));
        }

        // A refused upload keeps the existing artwork; a new one changes the version.
        let (status, _) = upload(&app, "file", &jpeg(1000, 1)).await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        let (_, body) = app.get("/api/podcasts/feed-settings").await;
        assert_eq!(body["artwork_version"], version.as_str());
        let (_, body) = upload(&app, "file", &jpeg(1500, 9)).await;
        assert_ne!(body["artwork_version"], version.as_str());
    }
}
