//! The Podcast Feed (Nova Notes fork): its settings, the Feed Token, the RSS
//! document and the token-gated files.
//!
//! The Rust counterpart of `open_notebook/domain/podcast_feed.py` and the
//! logic of `api/routers/podcast_feed.py`, `podcast_feed_settings.py` and the
//! feed URL routes. The XML, token, artwork and publishing rules are
//! `nn_podcast::feed` / `artwork`; this module stores the settings singleton
//! (`open_notebook:podcast_feed_settings`) and resolves the public address.
//!
//! Every token-gated miss (wrong or missing token, unknown or unpublished
//! episode, a URL that is not exactly the one the feed gives out, a missing
//! file) is the same `404 {"detail": "Not found"}`, so nothing can be probed.

use nn_podcast::artwork::{self, current_artwork};
use nn_podcast::feed::{
    self, parse_episode_audio_filename, published_audio_file, FeedSettingsUpdate, FeedSettingsView,
    PodcastFeedSettings, FEED_SETTINGS_RECORD_ID,
};
use serde::{Deserialize, Serialize};

use crate::config::env_nonempty;
use crate::domain::settings;
use crate::error::{AppError, AppResult};
use crate::jobs::JobQueue;
use crate::services::podcasts::{find_episode_by_key, podcast_error, published_episodes, ServedFile};
use crate::state::AppState;

/// The feed has no background jobs.
pub fn register_jobs(_jobs: &JobQueue) {}

/// Startup task: mint the Feed Token once, after migrations (the Python
/// lifespan does the same). Idempotent; the token is never logged.
pub async fn on_startup(state: &AppState) -> AppResult<()> {
    ensure_feed_token(state).await.map(|_| ())
}

/// The one answer for every token-gated miss.
pub fn feed_not_found() -> AppError {
    AppError::not_found("Not found")
}

pub async fn load_settings(state: &AppState) -> AppResult<PodcastFeedSettings> {
    settings::load(state, FEED_SETTINGS_RECORD_ID).await
}

async fn save_settings(state: &AppState, value: &PodcastFeedSettings) -> AppResult<PodcastFeedSettings> {
    settings::save(state, FEED_SETTINGS_RECORD_ID, value).await
}

/// The Feed Token, minted and saved on first use.
pub async fn ensure_feed_token(state: &AppState) -> AppResult<String> {
    let mut current = load_settings(state).await?;
    if current.ensure_feed_token() {
        current = save_settings(state, &current).await?;
    }
    current.feed_token.ok_or_else(|| AppError::internal("The Feed Token could not be saved"))
}

/// Replace the Feed Token. Every URL carrying the old one stops working.
pub async fn rotate_feed_token(state: &AppState) -> AppResult<String> {
    let mut current = load_settings(state).await?;
    current.rotate_feed_token();
    let saved = save_settings(state, &current).await?;
    saved.feed_token.ok_or_else(|| AppError::internal("The Feed Token could not be saved"))
}

/// The request's `X-Forwarded-Proto` / `X-Forwarded-Host`, the last resort
/// for the feed's public address.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ForwardedOrigin {
    pub proto: Option<String>,
    pub host: Option<String>,
}

/// The HTTPS origin podcast apps should use: `OPEN_NOTEBOOK_PUBLIC_URL`, then
/// `API_URL`, then the forwarded headers. None of them HTTPS → 422.
pub fn public_base_url(origin: &ForwardedOrigin) -> AppResult<String> {
    feed::resolve_public_base_url(
        env_nonempty("OPEN_NOTEBOOK_PUBLIC_URL").as_deref(),
        env_nonempty("API_URL").as_deref(),
        origin.proto.as_deref(),
        origin.host.as_deref(),
    )
    .map_err(podcast_error)
}

/// `FeedUrlResponse`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FeedUrl {
    pub feed_url: String,
}

/// The Podcast Feed URL to paste into a podcast app (it contains the Feed Token).
pub async fn feed_url(state: &AppState, origin: &ForwardedOrigin) -> AppResult<FeedUrl> {
    let base_url = public_base_url(origin)?;
    let token = ensure_feed_token(state).await?;
    Ok(FeedUrl { feed_url: feed::feed_url(&base_url, &token) })
}

/// Rotate the Feed Token and return the new URL. Without a usable public
/// address nothing is rotated (422).
pub async fn rotate_feed_url(state: &AppState, origin: &ForwardedOrigin) -> AppResult<FeedUrl> {
    let base_url = public_base_url(origin)?;
    let token = rotate_feed_token(state).await?;
    Ok(FeedUrl { feed_url: feed::feed_url(&base_url, &token) })
}

// ---------------------------------------------------------------------------
// Show Metadata and artwork (app password)
// ---------------------------------------------------------------------------

/// The Show Metadata with defaults filled in.
pub async fn feed_settings(state: &AppState) -> AppResult<FeedSettingsView> {
    Ok(load_settings(state).await?.view(&state.config.podcasts_dir()))
}

/// Edit the Show Metadata. A field left out is kept, a blank one goes back
/// to its default; a refused edit (400) changes nothing.
pub async fn update_feed_settings(state: &AppState, update: &FeedSettingsUpdate) -> AppResult<FeedSettingsView> {
    let mut current = load_settings(state).await?;
    if current.apply_update(update).map_err(podcast_error)? {
        current = save_settings(state, &current).await?;
    }
    Ok(current.view(&state.config.podcasts_dir()))
}

/// Replace the show artwork (a square PNG or JPEG of 1400 to 3000 pixels, at
/// most 10 MB). A refused upload (400) stores nothing.
pub async fn upload_artwork(state: &AppState, data: Vec<u8>) -> AppResult<FeedSettingsView> {
    let podcasts = state.config.podcasts_dir();
    let stored = tokio::task::spawn_blocking(move || {
        let found = artwork::validate_artwork(&data)?;
        artwork::save_artwork(&podcasts, &data, &found)?;
        Ok::<_, nn_podcast::PodcastError>(found)
    })
    .await?
    .map_err(podcast_error)?;
    let mut current = load_settings(state).await?;
    current.artwork_ext = Some(stored.extension.to_string());
    current.artwork_version = Some(stored.version);
    let saved = save_settings(state, &current).await?;
    Ok(saved.view(&state.config.podcasts_dir()))
}

/// The current artwork, for the settings page preview.
pub async fn artwork_preview(state: &AppState) -> AppResult<ServedFile> {
    let current = load_settings(state).await?;
    match current_artwork(&state.config.podcasts_dir(), current.artwork_ext.as_deref()) {
        Some((path, media_type)) => Ok(ServedFile { path, media_type: media_type.to_string() }),
        None => Err(AppError::not_found("No artwork has been uploaded")),
    }
}

// ---------------------------------------------------------------------------
// Token-gated (no app password)
// ---------------------------------------------------------------------------

/// The settings, or the uniform 404 when `token` is not the Feed Token.
async fn require_feed_token(state: &AppState, token: &str) -> AppResult<PodcastFeedSettings> {
    let current = load_settings(state).await?;
    if current.feed_token_matches(token) {
        Ok(current)
    } else {
        Err(feed_not_found())
    }
}

/// The RSS document: Published Episodes whose audio is still there, newest
/// published first. The public address is resolved after the token check so
/// a wrong token is always the uniform 404.
pub async fn feed_xml(state: &AppState, token: &str, origin: &ForwardedOrigin) -> AppResult<Vec<u8>> {
    let current = require_feed_token(state, token).await?;
    let base_url = public_base_url(origin)?;
    let episodes = published_episodes(state).await?.iter().map(|e| e.to_engine()).collect();
    let podcasts = state.config.podcasts_dir();
    Ok(tokio::task::spawn_blocking(move || feed::render_feed(&current, episodes, &base_url, &podcasts)).await?)
}

/// The show artwork for a podcast app: `artwork<ext>` must be the stored one.
pub async fn feed_artwork(state: &AppState, token: &str, extension: &str) -> AppResult<ServedFile> {
    let current = require_feed_token(state, token).await?;
    let found = current_artwork(&state.config.podcasts_dir(), current.artwork_ext.as_deref());
    match found {
        Some((path, media_type)) if current.artwork_ext.as_deref() == Some(extension) => {
            Ok(ServedFile { path, media_type: media_type.to_string() })
        }
        _ => Err(feed_not_found()),
    }
}

/// A Published Episode's audio behind a feed enclosure URL (`<key>.mp3` /
/// `<key>.m4a`). A database failure is not hidden as a 404.
pub async fn feed_episode_audio(state: &AppState, token: &str, filename: &str) -> AppResult<ServedFile> {
    require_feed_token(state, token).await?;
    let (key, suffix) = parse_episode_audio_filename(filename).ok_or_else(feed_not_found)?;
    let episode = find_episode_by_key(state, key).await?.ok_or_else(feed_not_found)?;
    let (path, media_type) =
        published_audio_file(&episode.to_engine(), &suffix, &state.config.podcasts_dir()).ok_or_else(feed_not_found)?;
    Ok(ServedFile { path, media_type: media_type.to_string() })
}
