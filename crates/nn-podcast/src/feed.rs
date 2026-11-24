//! The Podcast Feed: RSS 2.0 with iTunes tags, its settings and its token.
//!
//! One private feed for the whole instance lists every Published Episode. A
//! podcast app cannot log in, so the **Feed Token** in the URL path is the only
//! credential: `/api/podcasts/feed/<token>/feed.xml`, and the same prefix for
//! episode audio and artwork. This module builds those URLs and the XML, and
//! holds the rules for the token and the Show Metadata. Routes, the settings
//! record and environment variables belong to the server crate.
//!
//! The XML is written the way Python's `xml.etree.ElementTree` serialised it
//! in the original (same declaration, namespace order, attribute order,
//! escaping and empty-element form), so the two produce identical bytes.

use std::borrow::Cow;
use std::path::Path;
use std::sync::OnceLock;

use base64::Engine;
use chrono::{DateTime, Utc};
use rand::RngCore;
use regex::Regex;
use serde::{Deserialize, Deserializer, Serialize};

use crate::artwork::current_artwork;
use crate::audio_paths::resolve_contained_audio_path;
use crate::error::{PodcastError, Result};
use crate::models::{published_newest_first, PodcastEpisode};
use crate::publishing::{audio_suffix, publishable_audio_type};

pub const ITUNES_NS: &str = "http://www.itunes.com/dtds/podcast-1.0.dtd";
pub const ATOM_NS: &str = "http://www.w3.org/2005/Atom";

pub const DEFAULT_TITLE: &str = "Open Notebook Podcasts";
pub const DEFAULT_AUTHOR: &str = "Open Notebook";
pub const DEFAULT_DESCRIPTION: &str = "Podcast episodes generated in Open Notebook.";
pub const DEFAULT_LANGUAGE: &str = "en";

/// Media type of the feed document.
pub const FEED_MEDIA_TYPE: &str = "application/rss+xml";
/// The one path prefix the password login does not apply to. Everything under
/// it must check the Feed Token.
pub const FEED_PATH_PREFIX: &str = "/api/podcasts/feed/";

pub const TITLE_MAX: usize = 255;
pub const AUTHOR_MAX: usize = 255;
pub const DESCRIPTION_MAX: usize = 4000;

// ---------------------------------------------------------------------------
// Settings and the Feed Token
// ---------------------------------------------------------------------------

/// Instance-wide settings of the Podcast Feed (the singleton record
/// `open_notebook:podcast_feed_settings`). `None` means "use the default".
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct PodcastFeedSettings {
    /// Feed Token embedded in the Podcast Feed URL.
    #[serde(default)]
    pub feed_token: Option<String>,
    #[serde(default)]
    pub show_title: Option<String>,
    #[serde(default)]
    pub show_author: Option<String>,
    #[serde(default)]
    pub show_description: Option<String>,
    #[serde(default)]
    pub show_language: Option<String>,
    /// Extension of the stored artwork (`.png` or `.jpg`).
    #[serde(default)]
    pub artwork_ext: Option<String>,
    /// Content hash of the artwork; changes with the image.
    #[serde(default)]
    pub artwork_version: Option<String>,
}

/// Record ID of the settings singleton in the original's database.
pub const FEED_SETTINGS_RECORD_ID: &str = "open_notebook:podcast_feed_settings";

/// A new Feed Token: 32 random bytes as 43 URL-safe characters (Python's
/// `secrets.token_urlsafe(32)`).
pub fn generate_feed_token() -> String {
    let mut bytes = [0u8; 32];
    rand::rngs::OsRng.fill_bytes(&mut bytes);
    base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(bytes)
}

/// Compare two tokens without stopping at the first differing byte.
fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    a.iter().zip(b).fold(0u8, |diff, (x, y)| diff | (x ^ y)) == 0
}

impl PodcastFeedSettings {
    /// Mint the Feed Token if there is none. Returns `true` when one was
    /// minted, in which case the caller saves the record. Idempotent: once
    /// saved, every later call leaves the token alone.
    pub fn ensure_feed_token(&mut self) -> bool {
        if self.feed_token.as_deref().is_some_and(|token| !token.is_empty()) {
            return false;
        }
        self.feed_token = Some(generate_feed_token());
        true
    }

    /// Replace the Feed Token and return the new one. Every URL carrying the
    /// old one stops working once the caller has saved the record.
    pub fn rotate_feed_token(&mut self) -> &str {
        self.feed_token.insert(generate_feed_token())
    }

    /// True when `token` is the configured Feed Token. No token configured and
    /// a wrong token are the same `false`: the routes answer both with one
    /// uniform 404 so nothing can be probed.
    pub fn feed_token_matches(&self, token: &str) -> bool {
        match self.feed_token.as_deref() {
            Some(expected) if !expected.is_empty() => constant_time_eq(token.as_bytes(), expected.as_bytes()),
            _ => false,
        }
    }

    /// What the settings page shows: defaults filled in, and the artwork
    /// version only while its file exists.
    pub fn view(&self, podcasts_folder: &Path) -> FeedSettingsView {
        let show = ShowMetadata::from_settings(self, None);
        let has_artwork = current_artwork(podcasts_folder, self.artwork_ext.as_deref()).is_some();
        FeedSettingsView {
            title: show.title,
            author: show.author,
            description: show.description,
            language: show.language,
            has_artwork,
            artwork_version: if has_artwork { self.artwork_version.clone() } else { None },
        }
    }

    /// Apply an edit of the Show Metadata. Every provided field is checked
    /// first, so a refused edit changes nothing. Returns `true` when at least
    /// one field was provided (the caller then saves the record).
    pub fn apply_update(&mut self, update: &FeedSettingsUpdate) -> Result<bool> {
        let title = update.title.as_ref().map(|v| checked_text("title", v.as_deref(), TITLE_MAX)).transpose()?;
        let author =
            update.author.as_ref().map(|v| checked_text("author", v.as_deref(), AUTHOR_MAX)).transpose()?;
        let description = update
            .description
            .as_ref()
            .map(|v| checked_text("description", v.as_deref(), DESCRIPTION_MAX))
            .transpose()?;
        let language = update.language.as_ref().map(|v| checked_language(v.as_deref())).transpose()?;

        let mut changed = false;
        let mut set = |field: &mut Option<String>, value: Option<Option<String>>| {
            if let Some(value) = value {
                *field = value;
                changed = true;
            }
        };
        set(&mut self.show_title, title);
        set(&mut self.show_author, author);
        set(&mut self.show_description, description);
        set(&mut self.show_language, language);
        Ok(changed)
    }
}

/// The Show Metadata as the settings page reads it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FeedSettingsView {
    pub title: String,
    pub author: String,
    pub description: String,
    pub language: String,
    pub has_artwork: bool,
    pub artwork_version: Option<String>,
}

fn provided<'de, D: Deserializer<'de>>(deserializer: D) -> std::result::Result<Option<Option<String>>, D::Error> {
    Option::<String>::deserialize(deserializer).map(Some)
}

/// An edit of the Show Metadata. A field left out (`None`) is kept; a blank or
/// null one (`Some(None)`, `Some(Some(""))`) goes back to its default.
#[derive(Debug, Clone, Default, PartialEq, Eq, Deserialize)]
pub struct FeedSettingsUpdate {
    #[serde(default, deserialize_with = "provided")]
    pub title: Option<Option<String>>,
    #[serde(default, deserialize_with = "provided")]
    pub author: Option<Option<String>>,
    #[serde(default, deserialize_with = "provided")]
    pub description: Option<Option<String>>,
    #[serde(default, deserialize_with = "provided")]
    pub language: Option<Option<String>>,
}

/// Trim a Show Metadata text and enforce its length limit (in characters).
/// Blank becomes `None`, which means "use the default".
pub fn checked_text(label: &str, value: Option<&str>, limit: usize) -> Result<Option<String>> {
    let text = value.unwrap_or_default().trim();
    let length = text.chars().count();
    if length > limit {
        return Err(PodcastError::InvalidInput(format!(
            "The show {label} is too long ({length} characters; the limit is {limit})."
        )));
    }
    Ok((!text.is_empty()).then(|| text.to_string()))
}

/// Trim and check a show language: BCP 47-style, such as `en`, `pt-BR` or
/// `zh-Hans-CN`. Blank becomes `None`.
pub fn checked_language(value: Option<&str>) -> Result<Option<String>> {
    static PATTERN: OnceLock<Regex> = OnceLock::new();
    let pattern =
        PATTERN.get_or_init(|| Regex::new(r"^[A-Za-z]{2,3}(-[A-Za-z0-9]{2,8})*$").expect("valid regex"));
    let text = value.unwrap_or_default().trim();
    if !text.is_empty() && !pattern.is_match(text) {
        return Err(PodcastError::InvalidInput(
            "The show language must be a language code such as 'en' or 'pt-BR'.".into(),
        ));
    }
    Ok((!text.is_empty()).then(|| text.to_string()))
}

// ---------------------------------------------------------------------------
// URLs
// ---------------------------------------------------------------------------

/// Return the HTTPS origin podcast apps should use to reach this API.
///
/// Sources, first HTTPS one wins: `public_url` (the original's
/// `OPEN_NOTEBOOK_PUBLIC_URL`), then `api_url` (`API_URL`), then the request's
/// `X-Forwarded-Proto` / `X-Forwarded-Host` headers. Podcast apps need absolute
/// URLs and Apple Podcasts expects HTTPS, so a source that is not HTTPS is
/// skipped. With none left the result is [`PodcastError::Configuration`].
pub fn resolve_public_base_url(
    public_url: Option<&str>,
    api_url: Option<&str>,
    forwarded_proto: Option<&str>,
    forwarded_host: Option<&str>,
) -> Result<String> {
    let first = |header: &str| header.split(',').next().unwrap_or_default().trim().to_string();
    let forwarded = forwarded_host
        .filter(|host| !host.is_empty())
        .map(|host| format!("{}://{}", first(forwarded_proto.unwrap_or_default()), first(host)));

    let https_origin = [public_url, api_url, forwarded.as_deref()]
        .into_iter()
        .flatten()
        .map(str::trim)
        .find(|candidate| candidate.to_lowercase().starts_with("https://"))
        .map(|candidate| candidate.trim_end_matches('/').to_string());
    https_origin.ok_or_else(|| {
            PodcastError::Configuration(
                "The Podcast Feed needs an HTTPS address. Set OPEN_NOTEBOOK_PUBLIC_URL \
                 to the HTTPS origin podcast apps should use (for example \
                 https://your-host.your-tailnet.ts.net:8502), or, behind a reverse \
                 proxy that sets X-Forwarded-Proto and X-Forwarded-Host, set \
                 OPEN_NOTEBOOK_TRUST_PROXY_HEADERS=true."
                .into(),
        )
    })
}

/// The Podcast Feed URL a podcast app subscribes to.
pub fn feed_url(base_url: &str, token: &str) -> String {
    format!("{base_url}{FEED_PATH_PREFIX}{token}/feed.xml")
}

/// The URL of a Published Episode's audio. `suffix` is `.mp3` or `.m4a`.
pub fn episode_audio_url(base_url: &str, token: &str, episode_key: &str, suffix: &str) -> String {
    format!("{base_url}{FEED_PATH_PREFIX}{token}/episodes/{episode_key}{suffix}")
}

/// The URL of the show artwork. The version in the query makes podcast apps
/// refetch a replaced image.
pub fn artwork_url(base_url: &str, token: &str, extension: &str, version: &str) -> String {
    format!("{base_url}{FEED_PATH_PREFIX}{token}/artwork{extension}?v={version}")
}

/// Replace the Feed Token in a log line or URL with `<redacted>`: everything
/// after `/api/podcasts/feed/` up to the next `/`, `?`, `#`, quote or space.
pub fn redact_feed_token(text: &str) -> Cow<'_, str> {
    static PATTERN: OnceLock<Regex> = OnceLock::new();
    let pattern =
        PATTERN.get_or_init(|| Regex::new(r#"(/api/podcasts/feed/)[^/?#\s"']+"#).expect("valid regex"));
    pattern.replace_all(text, "${1}<redacted>")
}

// ---------------------------------------------------------------------------
// Show and items
// ---------------------------------------------------------------------------

/// The show-level description of the Podcast Feed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ShowMetadata {
    pub title: String,
    pub author: String,
    pub description: String,
    pub language: String,
    pub artwork_url: Option<String>,
}

impl Default for ShowMetadata {
    fn default() -> Self {
        Self {
            title: DEFAULT_TITLE.into(),
            author: DEFAULT_AUTHOR.into(),
            description: DEFAULT_DESCRIPTION.into(),
            language: DEFAULT_LANGUAGE.into(),
            artwork_url: None,
        }
    }
}

impl ShowMetadata {
    /// Settings values, with the default wherever one is unset or empty.
    pub fn from_settings(settings: &PodcastFeedSettings, artwork_url: Option<String>) -> Self {
        let or_default = |value: &Option<String>, default: &str| {
            value.as_deref().filter(|v| !v.is_empty()).unwrap_or(default).to_string()
        };
        Self {
            title: or_default(&settings.show_title, DEFAULT_TITLE),
            author: or_default(&settings.show_author, DEFAULT_AUTHOR),
            description: or_default(&settings.show_description, DEFAULT_DESCRIPTION),
            language: or_default(&settings.show_language, DEFAULT_LANGUAGE),
            artwork_url,
        }
    }
}

/// One Published Episode as a feed item.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FeedItem {
    pub title: String,
    pub description: String,
    pub published_at: DateTime<Utc>,
    pub guid: String,
    pub enclosure_url: String,
    pub enclosure_bytes: u64,
    pub enclosure_type: String,
    pub duration_seconds: u64,
}

/// The stable, non-permalink GUID of an episode's feed item.
pub fn episode_guid(episode_key: &str) -> String {
    format!("open-notebook-episode-{episode_key}")
}

/// Feed items for Published Episodes whose audio is still publishable, in the
/// order given.
///
/// The feed must never point at nothing: an episode that is not published,
/// whose file vanished, whose path is no longer valid or whose format is not
/// publishable is left out (and logged), not served.
pub fn feed_items(
    episodes: &[PodcastEpisode],
    base_url: &str,
    token: &str,
    podcasts_folder: &Path,
) -> Vec<FeedItem> {
    let mut items = Vec::new();
    for episode in episodes {
        let Some(published_at) = episode.published_at else {
            continue;
        };
        let audio = resolve_contained_audio_path(podcasts_folder, episode.audio_file.as_deref())
            .filter(|path| path.is_file())
            .and_then(|path| {
                let suffix = audio_suffix(&path);
                let mime = publishable_audio_type(&suffix)?;
                let bytes = path.metadata().ok()?.len();
                Some((suffix, mime, bytes))
            });
        let Some((suffix, mime, bytes)) = audio else {
            tracing::warn!("Leaving published episode {} out of the feed", episode.id.as_deref().unwrap_or("?"));
            continue;
        };
        let key = episode.key();
        items.push(FeedItem {
            title: episode.name.clone(),
            description: episode
                .feed_summary
                .clone()
                .filter(|summary| !summary.is_empty())
                .unwrap_or_else(|| episode.name.clone()),
            published_at,
            guid: episode_guid(key),
            enclosure_url: episode_audio_url(base_url, token, key, &suffix),
            enclosure_bytes: bytes,
            enclosure_type: mime.to_string(),
            duration_seconds: episode.duration_seconds.unwrap_or(0),
        });
    }
    items
}

/// Split the last path segment of an episode audio URL (`ep1.mp3`) into the
/// episode key and the suffix. `None` unless it is exactly `<key>.mp3` or
/// `<key>.m4a` (the suffix may be in any case).
pub fn parse_episode_audio_filename(filename: &str) -> Option<(&str, String)> {
    let (key, extension) = filename.rsplit_once('.')?;
    let suffix = format!(".{}", extension.to_lowercase());
    (!key.is_empty() && publishable_audio_type(&suffix).is_some()).then_some((key, suffix))
}

/// The audio file and MIME type behind a feed enclosure URL, given the episode
/// the URL names and the suffix it asked for.
///
/// `None` for every way this can fail (an unpublished episode, a suffix that
/// differs from the file, a missing file, a path outside the podcasts folder):
/// the route answers all of them with the same 404.
pub fn published_audio_file(
    episode: &PodcastEpisode,
    suffix: &str,
    podcasts_folder: &Path,
) -> Option<(std::path::PathBuf, &'static str)> {
    let mime = publishable_audio_type(suffix)?;
    episode.published_at?;
    let path = resolve_contained_audio_path(podcasts_folder, episode.audio_file.as_deref())?;
    (path.is_file() && audio_suffix(&path) == suffix).then_some((path, mime))
}

// ---------------------------------------------------------------------------
// XML
// ---------------------------------------------------------------------------

/// Characters XML 1.0 cannot carry at all, escaped or not.
fn is_xml_char(c: char) -> bool {
    matches!(c, '\t' | '\n' | '\r' | '\u{20}'..='\u{D7FF}' | '\u{E000}'..='\u{FFFD}' | '\u{10000}'..='\u{10FFFF}')
}

fn escape_text(text: &str, out: &mut String) {
    for c in text.chars().filter(|&c| is_xml_char(c)) {
        match c {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            _ => out.push(c),
        }
    }
}

fn escape_attribute(text: &str, out: &mut String) {
    for c in text.chars().filter(|&c| is_xml_char(c)) {
        match c {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '"' => out.push_str("&quot;"),
            '\r' => out.push_str("&#13;"),
            '\n' => out.push_str("&#10;"),
            '\t' => out.push_str("&#09;"),
            _ => out.push(c),
        }
    }
}

/// Write `<tag attr="..">text</tag>`, or `<tag attr=".." />` when there is no
/// text, as ElementTree does.
fn element(out: &mut String, tag: &str, attributes: &[(&str, &str)], text: &str) {
    out.push('<');
    out.push_str(tag);
    for (name, value) in attributes {
        out.push(' ');
        out.push_str(name);
        out.push_str("=\"");
        escape_attribute(value, out);
        out.push('"');
    }
    let mut escaped = String::new();
    escape_text(text, &mut escaped);
    if escaped.is_empty() {
        out.push_str(" />");
    } else {
        out.push('>');
        out.push_str(&escaped);
        out.push_str("</");
        out.push_str(tag);
        out.push('>');
    }
}

/// `Thu, 05 Nov 2026 18:45:00 GMT`.
fn rfc2822_gmt(at: &DateTime<Utc>) -> String {
    at.format("%a, %d %b %Y %H:%M:%S GMT").to_string()
}

/// Serialise the Podcast Feed: show info plus one item per episode, in the
/// order given. The channel carries `itunes:block` = `yes`, which keeps a
/// private feed out of Apple's public directory.
pub fn build_feed_xml(feed_url: &str, base_url: &str, items: &[FeedItem], show: &ShowMetadata) -> Vec<u8> {
    let mut out = String::from("<?xml version='1.0' encoding='utf-8'?>\n");
    out.push_str(&format!(
        "<rss xmlns:atom=\"{ATOM_NS}\" xmlns:itunes=\"{ITUNES_NS}\" version=\"2.0\"><channel>"
    ));
    element(&mut out, "title", &[], &show.title);
    element(&mut out, "link", &[], base_url);
    element(&mut out, "description", &[], &show.description);
    element(&mut out, "language", &[], &show.language);
    element(
        &mut out,
        "atom:link",
        &[("href", feed_url), ("rel", "self"), ("type", FEED_MEDIA_TYPE)],
        "",
    );
    element(&mut out, "itunes:author", &[], &show.author);
    if let Some(artwork) = show.artwork_url.as_deref().filter(|url| !url.is_empty()) {
        element(&mut out, "itunes:image", &[("href", artwork)], "");
    }
    // Private feed: keep it out of Apple's public directory.
    element(&mut out, "itunes:block", &[], "yes");
    for item in items {
        out.push_str("<item>");
        element(&mut out, "title", &[], &item.title);
        element(&mut out, "description", &[], &item.description);
        element(&mut out, "pubDate", &[], &rfc2822_gmt(&item.published_at));
        element(&mut out, "guid", &[("isPermaLink", "false")], &item.guid);
        element(
            &mut out,
            "enclosure",
            &[
                ("url", &item.enclosure_url),
                ("length", &item.enclosure_bytes.to_string()),
                ("type", &item.enclosure_type),
            ],
            "",
        );
        element(&mut out, "itunes:duration", &[], &item.duration_seconds.to_string());
        out.push_str("</item>");
    }
    out.push_str("</channel></rss>");
    out.into_bytes()
}

/// Everything the feed route does after the token check: the Published
/// Episodes among `episodes` (newest published first) whose audio is still
/// there, the Show Metadata, and the artwork when its file exists.
pub fn render_feed(
    settings: &PodcastFeedSettings,
    episodes: Vec<PodcastEpisode>,
    base_url: &str,
    podcasts_folder: &Path,
) -> Vec<u8> {
    let token = settings.feed_token.as_deref().unwrap_or_default();
    let published = published_newest_first(episodes);
    let items = feed_items(&published, base_url, token, podcasts_folder);
    // Never advertise artwork whose file is gone: the feed must not point at nothing.
    let artwork = current_artwork(podcasts_folder, settings.artwork_ext.as_deref()).map(|_| {
        artwork_url(
            base_url,
            token,
            settings.artwork_ext.as_deref().unwrap_or_default(),
            settings.artwork_version.as_deref().unwrap_or_default(),
        )
    });
    let show = ShowMetadata::from_settings(settings, artwork);
    build_feed_xml(&feed_url(base_url, token), base_url, &items, &show)
}

#[cfg(test)]
mod tests {
    use chrono::TimeZone;
    use serde_json::json;

    use super::*;

    const TOKEN: &str = "tok-abc123";
    const PUBLIC_URL: &str = "https://lab.example.ts.net:8502";

    fn settings() -> PodcastFeedSettings {
        PodcastFeedSettings { feed_token: Some(TOKEN.into()), ..PodcastFeedSettings::default() }
    }

    fn newer() -> DateTime<Utc> {
        Utc.with_ymd_and_hms(2026, 11, 5, 18, 45, 0).unwrap()
    }

    fn older() -> DateTime<Utc> {
        Utc.with_ymd_and_hms(2026, 11, 2, 8, 0, 0).unwrap()
    }

    fn episode_row(key: &str, name: &str, published_at: Option<DateTime<Utc>>, ext: &str) -> PodcastEpisode {
        serde_json::from_value(json!({
            "id": format!("episode:{key}"),
            "name": name,
            "episode_profile": {"name": "default"},
            "speaker_profile": {"name": "default"},
            "briefing": "INTERNAL generation prompt, never shown in the feed",
            "content": "source text",
            "audio_file": format!("episodes/{key}/audio/{key}.{ext}"),
            "published_at": published_at,
            "duration_seconds": 26,
            "feed_summary": format!("About {name}"),
        }))
        .unwrap()
    }

    fn write_audio(root: &Path, episodes: &[PodcastEpisode], bytes: usize) {
        for episode in episodes {
            let path = root.join(episode.audio_file.as_deref().unwrap());
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
            std::fs::write(path, vec![b'A'; bytes]).unwrap();
        }
    }

    fn itunes<'a>(node: roxmltree::Node<'a, 'a>, name: &str) -> Option<roxmltree::Node<'a, 'a>> {
        node.children().find(|n| n.has_tag_name((ITUNES_NS, name)))
    }

    fn child<'a>(node: roxmltree::Node<'a, 'a>, name: &str) -> Option<roxmltree::Node<'a, 'a>> {
        node.children().find(|n| n.tag_name().namespace().is_none() && n.tag_name().name() == name)
    }

    fn text<'a>(node: roxmltree::Node<'a, 'a>, name: &str) -> &'a str {
        child(node, name).and_then(|n| n.text()).unwrap_or_default()
    }

    fn with_channel<T>(xml: &[u8], check: impl FnOnce(roxmltree::Node) -> T) -> T {
        let source = std::str::from_utf8(xml).unwrap();
        let document = roxmltree::Document::parse(source).unwrap();
        let rss = document.root_element();
        assert_eq!(rss.tag_name().name(), "rss");
        assert_eq!(rss.attribute("version"), Some("2.0"));
        check(child(rss, "channel").expect("channel"))
    }

    fn items<'a>(channel: roxmltree::Node<'a, 'a>) -> Vec<roxmltree::Node<'a, 'a>> {
        channel.children().filter(|n| n.has_tag_name("item")).collect()
    }

    // --- the document ----------------------------------------------------------

    #[test]
    fn empty_feed_is_well_formed_rss_with_default_show_info() {
        let root = tempfile::tempdir().unwrap();
        let xml = render_feed(&settings(), Vec::new(), PUBLIC_URL, root.path());
        with_channel(&xml, |channel| {
            assert!(items(channel).is_empty());
            assert_eq!(text(channel, "title"), "Open Notebook Podcasts");
            assert_eq!(text(channel, "link"), PUBLIC_URL);
            assert_eq!(text(channel, "description"), "Podcast episodes generated in Open Notebook.");
            assert_eq!(text(channel, "language"), "en");
            assert_eq!(itunes(channel, "author").unwrap().text(), Some("Open Notebook"));
            assert!(itunes(channel, "image").is_none());
            // Blocked from the public directory.
            assert_eq!(itunes(channel, "block").unwrap().text(), Some("yes"));
            let link = channel.children().find(|n| n.has_tag_name((ATOM_NS, "link"))).unwrap();
            assert_eq!(link.attribute("rel"), Some("self"));
            assert_eq!(link.attribute("type"), Some("application/rss+xml"));
            assert_eq!(
                link.attribute("href"),
                Some("https://lab.example.ts.net:8502/api/podcasts/feed/tok-abc123/feed.xml")
            );
        });
    }

    #[test]
    fn bytes_match_the_original_serialiser() {
        // Reference output of the Python original for the same inputs.
        let empty = build_feed_xml(&feed_url(PUBLIC_URL, TOKEN), PUBLIC_URL, &[], &ShowMetadata::default());
        assert_eq!(
            String::from_utf8(empty).unwrap(),
            "<?xml version='1.0' encoding='utf-8'?>\n<rss xmlns:atom=\"http://www.w3.org/2005/Atom\" xmlns:itunes=\"http://www.itunes.com/dtds/podcast-1.0.dtd\" version=\"2.0\"><channel><title>Open Notebook Podcasts</title><link>https://lab.example.ts.net:8502</link><description>Podcast episodes generated in Open Notebook.</description><language>en</language><atom:link href=\"https://lab.example.ts.net:8502/api/podcasts/feed/tok-abc123/feed.xml\" rel=\"self\" type=\"application/rss+xml\" /><itunes:author>Open Notebook</itunes:author><itunes:block>yes</itunes:block></channel></rss>"
        );

        let feed_items = [
            FeedItem {
                title: "Tom & \"Jerry\" <live>".into(),
                description: "Line one\nLine 'two' > three\ttab".into(),
                published_at: newer(),
                guid: "open-notebook-episode-ep1".into(),
                enclosure_url: format!("{PUBLIC_URL}/api/podcasts/feed/{TOKEN}/episodes/ep1.mp3?a=1&b=\"2\""),
                enclosure_bytes: 417_000,
                enclosure_type: "audio/mpeg".into(),
                duration_seconds: 26,
            },
            FeedItem {
                title: String::new(),
                description: "Café שלום".into(),
                published_at: Utc.with_ymd_and_hms(2026, 1, 2, 3, 4, 5).unwrap(),
                guid: "open-notebook-episode-old".into(),
                enclosure_url: format!("{PUBLIC_URL}/x.m4a"),
                enclosure_bytes: 0,
                enclosure_type: "audio/x-m4a".into(),
                duration_seconds: 0,
            },
        ];
        let show = ShowMetadata {
            title: "Dan's Lab Radio".into(),
            author: "Dan & Co".into(),
            description: "Notes <from> the lab.".into(),
            language: "pt-BR".into(),
            artwork_url: Some(format!("{PUBLIC_URL}/api/podcasts/feed/{TOKEN}/artwork.png?v=abc&x=1")),
        };
        let full = build_feed_xml(&feed_url(PUBLIC_URL, TOKEN), PUBLIC_URL, &feed_items, &show);
        assert_eq!(
            String::from_utf8(full).unwrap(),
            "<?xml version='1.0' encoding='utf-8'?>\n<rss xmlns:atom=\"http://www.w3.org/2005/Atom\" xmlns:itunes=\"http://www.itunes.com/dtds/podcast-1.0.dtd\" version=\"2.0\"><channel><title>Dan's Lab Radio</title><link>https://lab.example.ts.net:8502</link><description>Notes &lt;from&gt; the lab.</description><language>pt-BR</language><atom:link href=\"https://lab.example.ts.net:8502/api/podcasts/feed/tok-abc123/feed.xml\" rel=\"self\" type=\"application/rss+xml\" /><itunes:author>Dan &amp; Co</itunes:author><itunes:image href=\"https://lab.example.ts.net:8502/api/podcasts/feed/tok-abc123/artwork.png?v=abc&amp;x=1\" /><itunes:block>yes</itunes:block><item><title>Tom &amp; \"Jerry\" &lt;live&gt;</title><description>Line one\nLine 'two' &gt; three\ttab</description><pubDate>Thu, 05 Nov 2026 18:45:00 GMT</pubDate><guid isPermaLink=\"false\">open-notebook-episode-ep1</guid><enclosure url=\"https://lab.example.ts.net:8502/api/podcasts/feed/tok-abc123/episodes/ep1.mp3?a=1&amp;b=&quot;2&quot;\" length=\"417000\" type=\"audio/mpeg\" /><itunes:duration>26</itunes:duration></item><item><title /><description>Café שלום</description><pubDate>Fri, 02 Jan 2026 03:04:05 GMT</pubDate><guid isPermaLink=\"false\">open-notebook-episode-old</guid><enclosure url=\"https://lab.example.ts.net:8502/x.m4a\" length=\"0\" type=\"audio/x-m4a\" /><itunes:duration>0</itunes:duration></item></channel></rss>"
        );

        let odd = build_feed_xml(
            "u\n\"q\"\t<&>\r",
            "b",
            &[],
            &ShowMetadata { title: "a\rb".into(), ..ShowMetadata::default() },
        );
        let odd = String::from_utf8(odd).unwrap();
        assert!(odd.contains("<title>a\rb</title><link>b</link>"));
        assert!(odd.contains("<atom:link href=\"u&#10;&quot;q&quot;&#09;&lt;&amp;&gt;&#13;\" rel=\"self\""));
    }

    #[test]
    fn markup_in_text_cannot_break_the_feed() {
        let root = tempfile::tempdir().unwrap();
        let mut edited = settings();
        edited.show_title = Some("A & B <script>\"x\"</script>".into());
        edited.show_author = Some("]]> <!-- & -->".into());
        let xml = render_feed(&edited, Vec::new(), PUBLIC_URL, root.path());
        with_channel(&xml, |channel| {
            assert_eq!(text(channel, "title"), "A & B <script>\"x\"</script>");
            assert_eq!(itunes(channel, "author").unwrap().text(), Some("]]> <!-- & -->"));
        });
    }

    #[test]
    fn characters_xml_cannot_carry_are_dropped() {
        let show = ShowMetadata { title: "Bell\u{7}, null\u{0} and \u{FFFE}end".into(), ..ShowMetadata::default() };
        let xml = build_feed_xml("https://x/feed.xml", "https://x", &[], &show);
        with_channel(&xml, |channel| assert_eq!(text(channel, "title"), "Bell, null and end"));
    }

    // --- items -------------------------------------------------------------------

    #[test]
    fn published_episode_becomes_an_item() {
        let root = tempfile::tempdir().unwrap();
        let episodes = vec![episode_row("ep1", "Offline first", Some(newer()), "mp3")];
        write_audio(root.path(), &episodes, 417_000);
        let xml = render_feed(&settings(), episodes, PUBLIC_URL, root.path());
        assert!(!String::from_utf8_lossy(&xml).contains("INTERNAL generation prompt"));
        with_channel(&xml, |channel| {
            let found = items(channel);
            assert_eq!(found.len(), 1);
            let item = found[0];
            assert_eq!(text(item, "title"), "Offline first");
            assert_eq!(text(item, "description"), "About Offline first");
            assert_eq!(text(item, "pubDate"), "Thu, 05 Nov 2026 18:45:00 GMT");
            assert_eq!(
                DateTime::parse_from_rfc2822(text(item, "pubDate")).unwrap().with_timezone(&Utc),
                newer()
            );
            assert_eq!(itunes(item, "duration").unwrap().text(), Some("26"));
            let guid = child(item, "guid").unwrap();
            assert_eq!(guid.text(), Some("open-notebook-episode-ep1"));
            assert_eq!(guid.attribute("isPermaLink"), Some("false"));
            let enclosure = child(item, "enclosure").unwrap();
            let attributes: Vec<(&str, &str)> = enclosure.attributes().map(|a| (a.name(), a.value())).collect();
            assert_eq!(
                attributes,
                [
                    ("url", "https://lab.example.ts.net:8502/api/podcasts/feed/tok-abc123/episodes/ep1.mp3"),
                    ("length", "417000"),
                    ("type", "audio/mpeg"),
                ]
            );
        });
    }

    #[test]
    fn items_are_newest_published_first_and_drafts_are_left_out() {
        let root = tempfile::tempdir().unwrap();
        let episodes = vec![
            episode_row("old", "Older", Some(older()), "mp3"),
            episode_row("draft", "Draft", None, "mp3"),
            episode_row("new", "Newer", Some(newer()), "mp3"),
        ];
        write_audio(root.path(), &episodes, 10);
        let xml = render_feed(&settings(), episodes, PUBLIC_URL, root.path());
        with_channel(&xml, |channel| {
            let titles: Vec<&str> = items(channel).into_iter().map(|item| text(item, "title")).collect();
            assert_eq!(titles, ["Newer", "Older"]);
        });
    }

    #[test]
    fn m4a_episode_has_the_m4a_mime_type() {
        let root = tempfile::tempdir().unwrap();
        let episodes = vec![episode_row("ep1", "Talk", Some(newer()), "m4a")];
        write_audio(root.path(), &episodes, 10);
        let found = feed_items(&episodes, PUBLIC_URL, TOKEN, root.path());
        assert_eq!(found[0].enclosure_type, "audio/x-m4a");
        assert!(found[0].enclosure_url.ends_with("/episodes/ep1.m4a"));
    }

    #[test]
    fn episodes_that_cannot_be_served_are_left_out() {
        let root = tempfile::tempdir().unwrap();
        let gone = episode_row("gone", "Gone", Some(newer()), "mp3");
        let wav = episode_row("wav", "Wav", Some(newer()), "wav");
        let mut escaping = episode_row("esc", "Escaping", Some(newer()), "mp3");
        escaping.audio_file = Some("../../etc/passwd.mp3".into());
        let mut no_audio = episode_row("none", "None", Some(newer()), "mp3");
        no_audio.audio_file = None;
        write_audio(root.path(), std::slice::from_ref(&wav), 10);

        let found = feed_items(&[gone, wav, escaping, no_audio], PUBLIC_URL, TOKEN, root.path());
        assert!(found.is_empty(), "{found:?}");
    }

    #[test]
    fn description_falls_back_to_the_name_and_duration_to_zero() {
        let root = tempfile::tempdir().unwrap();
        let mut episode = episode_row("ep1", "Offline first", Some(newer()), "mp3");
        episode.feed_summary = None;
        episode.duration_seconds = None;
        write_audio(root.path(), std::slice::from_ref(&episode), 5);
        let found = feed_items(&[episode], PUBLIC_URL, TOKEN, root.path());
        assert_eq!(found[0].description, "Offline first");
        assert_eq!(found[0].duration_seconds, 0);
        assert_eq!(found[0].enclosure_bytes, 5);
    }

    // --- artwork -------------------------------------------------------------------

    #[test]
    fn artwork_is_advertised_only_while_its_file_exists() {
        let root = tempfile::tempdir().unwrap();
        let data = crate::artwork::testing::png(1400, 1400, 1);
        let artwork = crate::artwork::validate_artwork(&data).unwrap();
        crate::artwork::save_artwork(root.path(), &data, &artwork).unwrap();
        let mut with_art = settings();
        with_art.artwork_ext = Some(artwork.extension.to_string());
        with_art.artwork_version = Some(artwork.version.clone());

        let xml = render_feed(&with_art, Vec::new(), PUBLIC_URL, root.path());
        with_channel(&xml, |channel| {
            assert_eq!(
                itunes(channel, "image").unwrap().attribute("href"),
                Some(format!("{PUBLIC_URL}/api/podcasts/feed/{TOKEN}/artwork.png?v={}", artwork.version).as_str())
            );
        });
        let view = with_art.view(root.path());
        assert!(view.has_artwork);
        assert_eq!(view.artwork_version, Some(artwork.version.clone()));

        std::fs::remove_file(crate::artwork::artwork_path(root.path(), ".png")).unwrap();
        let xml = render_feed(&with_art, Vec::new(), PUBLIC_URL, root.path());
        with_channel(&xml, |channel| assert!(itunes(channel, "image").is_none()));
        let view = with_art.view(root.path());
        assert!(!view.has_artwork);
        assert_eq!(view.artwork_version, None);
    }

    // --- audio behind an enclosure URL ---------------------------------------------

    #[test]
    fn enclosure_filenames_are_parsed_strictly() {
        assert_eq!(parse_episode_audio_filename("ep1.mp3"), Some(("ep1", ".mp3".to_string())));
        assert_eq!(parse_episode_audio_filename("a.b.M4A"), Some(("a.b", ".m4a".to_string())));
        for bad in ["ep1", ".mp3", "ep1.wav", "ep1.", ""] {
            assert_eq!(parse_episode_audio_filename(bad), None, "{bad}");
        }
    }

    #[test]
    fn only_the_advertised_file_is_served() {
        let root = tempfile::tempdir().unwrap();
        let published = episode_row("ep1", "Talk", Some(newer()), "mp3");
        let draft = episode_row("ep2", "Draft", None, "mp3");
        let missing = episode_row("ep3", "Missing", Some(newer()), "mp3");
        let mut escaping = episode_row("ep4", "Escaping", Some(newer()), "mp3");
        escaping.audio_file = Some("../../etc/passwd.mp3".into());
        write_audio(root.path(), &[published.clone(), draft.clone()], 10);

        let (path, mime) = published_audio_file(&published, ".mp3", root.path()).unwrap();
        assert!(path.ends_with("episodes/ep1/audio/ep1.mp3"));
        assert_eq!(mime, "audio/mpeg");

        assert_eq!(published_audio_file(&published, ".m4a", root.path()), None, "extension mismatch");
        assert_eq!(published_audio_file(&published, ".wav", root.path()), None);
        assert_eq!(published_audio_file(&draft, ".mp3", root.path()), None, "unpublished");
        assert_eq!(published_audio_file(&missing, ".mp3", root.path()), None, "file missing");
        assert_eq!(published_audio_file(&escaping, ".mp3", root.path()), None, "outside the folder");
    }

    #[test]
    fn every_enclosure_url_resolves_back_to_its_file() {
        let root = tempfile::tempdir().unwrap();
        let episodes =
            vec![episode_row("a1", "One", Some(newer()), "mp3"), episode_row("b2", "Two", Some(older()), "m4a")];
        write_audio(root.path(), &episodes, 10);
        for (episode, item) in episodes.iter().zip(feed_items(&episodes, PUBLIC_URL, TOKEN, root.path())) {
            let filename = item.enclosure_url.rsplit('/').next().unwrap();
            let (key, suffix) = parse_episode_audio_filename(filename).unwrap();
            assert_eq!(key, episode.key());
            let (_, mime) = published_audio_file(episode, &suffix, root.path()).unwrap();
            assert_eq!(mime, item.enclosure_type);
        }
    }

    // --- base URL --------------------------------------------------------------------

    #[test]
    fn public_url_wins_and_a_trailing_slash_is_ignored() {
        let api = Some("https://api.example.ts.net:8502");
        assert_eq!(resolve_public_base_url(Some(PUBLIC_URL), api, None, None).unwrap(), PUBLIC_URL);
        let with_slash = format!("{PUBLIC_URL}/");
        assert_eq!(resolve_public_base_url(Some(&with_slash), None, None, None).unwrap(), PUBLIC_URL);
        assert_eq!(resolve_public_base_url(Some("  HTTPS://Host/  "), None, None, None).unwrap(), "HTTPS://Host");
    }

    #[test]
    fn api_url_is_used_when_no_public_url_is_set() {
        assert_eq!(
            resolve_public_base_url(None, Some("https://api.example.ts.net:8502"), None, None).unwrap(),
            "https://api.example.ts.net:8502"
        );
    }

    #[test]
    fn forwarded_headers_are_the_last_resort() {
        assert_eq!(
            resolve_public_base_url(None, None, Some("https"), Some("fwd.example.ts.net")).unwrap(),
            "https://fwd.example.ts.net"
        );
        assert_eq!(
            resolve_public_base_url(None, None, Some("https, http"), Some("a.example, b.example")).unwrap(),
            "https://a.example"
        );
        assert_eq!(
            resolve_public_base_url(None, Some("https://api.example"), Some("https"), Some("fwd.example")).unwrap(),
            "https://api.example"
        );
    }

    #[test]
    fn sources_that_are_not_https_are_skipped() {
        assert_eq!(
            resolve_public_base_url(Some("http://lab.local:8502"), Some("https://api.example.ts.net:8502"), None, None)
                .unwrap(),
            "https://api.example.ts.net:8502"
        );
    }

    #[test]
    fn no_https_origin_is_a_clear_error() {
        for (proto, host) in [(Some("http"), Some("x")), (None, None), (Some("https"), None), (None, Some("x"))] {
            let error = resolve_public_base_url(None, Some(""), proto, host).unwrap_err();
            assert!(matches!(error, PodcastError::Configuration(_)));
            assert!(error.to_string().contains("OPEN_NOTEBOOK_PUBLIC_URL"));
        }
    }

    #[test]
    fn urls_carry_the_token_as_a_path_segment() {
        assert_eq!(feed_url(PUBLIC_URL, TOKEN), format!("{PUBLIC_URL}/api/podcasts/feed/{TOKEN}/feed.xml"));
        assert_eq!(
            episode_audio_url(PUBLIC_URL, TOKEN, "ep1", ".m4a"),
            format!("{PUBLIC_URL}/api/podcasts/feed/{TOKEN}/episodes/ep1.m4a")
        );
        assert_eq!(
            artwork_url(PUBLIC_URL, TOKEN, ".jpg", "0123456789ab"),
            format!("{PUBLIC_URL}/api/podcasts/feed/{TOKEN}/artwork.jpg?v=0123456789ab")
        );
        assert!(feed_url(PUBLIC_URL, TOKEN).contains(FEED_PATH_PREFIX));
    }

    // --- the token ---------------------------------------------------------------------

    #[test]
    fn first_call_mints_a_strong_token_and_later_calls_keep_it() {
        let mut settings = PodcastFeedSettings::default();
        assert!(settings.ensure_feed_token());
        let token = settings.feed_token.clone().unwrap();
        assert_eq!(token.len(), 43);
        assert!(token.chars().all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_'));

        assert!(!settings.ensure_feed_token());
        assert_eq!(settings.feed_token.as_deref(), Some(token.as_str()));

        // A restart: the record is read back from storage unchanged.
        let mut reloaded: PodcastFeedSettings =
            serde_json::from_value(serde_json::to_value(&settings).unwrap()).unwrap();
        assert!(!reloaded.ensure_feed_token());
        assert_eq!(reloaded.feed_token, Some(token));
    }

    #[test]
    fn different_instances_and_rotations_get_different_tokens() {
        let mut first = PodcastFeedSettings::default();
        let mut second = PodcastFeedSettings::default();
        first.ensure_feed_token();
        second.ensure_feed_token();
        assert_ne!(first.feed_token, second.feed_token);

        let before = first.feed_token.clone().unwrap();
        let rotated = first.rotate_feed_token().to_string();
        assert_ne!(rotated, before);
        assert_eq!(rotated.len(), 43);
        assert!(first.feed_token_matches(&rotated));
        assert!(!first.feed_token_matches(&before), "the old URL stops working");
        assert_ne!(first.rotate_feed_token(), rotated);
    }

    #[test]
    fn wrong_missing_and_odd_tokens_do_not_match() {
        let configured = settings();
        assert!(configured.feed_token_matches(TOKEN));
        for wrong in ["not-the-token", "", "tok-abc12", "tok-abc1234", "tök", "TOK-ABC123"] {
            assert!(!configured.feed_token_matches(wrong), "{wrong}");
        }
        let unconfigured = PodcastFeedSettings::default();
        assert!(!unconfigured.feed_token_matches(TOKEN));
        assert!(!unconfigured.feed_token_matches(""));
        let blank = PodcastFeedSettings { feed_token: Some(String::new()), ..PodcastFeedSettings::default() };
        assert!(!blank.feed_token_matches(""));
    }

    #[test]
    fn tokens_are_redacted_from_log_lines() {
        assert_eq!(
            redact_feed_token("GET /api/podcasts/feed/s3cr3t-Tok_en/feed.xml HTTP/1.1"),
            "GET /api/podcasts/feed/<redacted>/feed.xml HTTP/1.1"
        );
        assert_eq!(
            redact_feed_token("\"/api/podcasts/feed/a/episodes/b/c.mp3\" and /api/podcasts/feed/zzz?x=1"),
            "\"/api/podcasts/feed/<redacted>/episodes/b/c.mp3\" and /api/podcasts/feed/<redacted>?x=1"
        );
        assert_eq!(redact_feed_token("/api/podcasts/feed-url"), "/api/podcasts/feed-url");
        assert_eq!(redact_feed_token("/api/podcasts/episodes"), "/api/podcasts/episodes");
    }

    // --- Show Metadata -------------------------------------------------------------------

    fn update(value: serde_json::Value) -> FeedSettingsUpdate {
        serde_json::from_value(value).unwrap()
    }

    #[test]
    fn unset_fields_show_the_defaults() {
        let root = tempfile::tempdir().unwrap();
        assert_eq!(
            serde_json::to_value(settings().view(root.path())).unwrap(),
            json!({
                "title": "Open Notebook Podcasts",
                "author": "Open Notebook",
                "description": "Podcast episodes generated in Open Notebook.",
                "language": "en",
                "has_artwork": false,
                "artwork_version": null,
            })
        );
    }

    #[test]
    fn edits_are_stored_and_appear_in_the_feed() {
        let root = tempfile::tempdir().unwrap();
        let mut edited = settings();
        let changed = edited
            .apply_update(&update(json!({
                "title": "Dan's Lab Radio", "author": "Dan",
                "description": "Notes from the lab.", "language": "pt-BR",
            })))
            .unwrap();
        assert!(changed);
        assert_eq!(edited.view(root.path()).language, "pt-BR");
        let xml = render_feed(&edited, Vec::new(), PUBLIC_URL, root.path());
        with_channel(&xml, |channel| {
            assert_eq!(text(channel, "title"), "Dan's Lab Radio");
            assert_eq!(itunes(channel, "author").unwrap().text(), Some("Dan"));
            assert_eq!(text(channel, "description"), "Notes from the lab.");
            assert_eq!(text(channel, "language"), "pt-BR");
        });
    }

    #[test]
    fn a_field_left_out_keeps_its_value_and_a_blank_one_resets() {
        let mut edited = settings();
        edited.apply_update(&update(json!({"title": "Lab Radio", "author": "Dan"}))).unwrap();
        edited.apply_update(&update(json!({"author": "Daniel"}))).unwrap();
        assert_eq!(edited.show_title.as_deref(), Some("Lab Radio"));
        assert_eq!(edited.show_author.as_deref(), Some("Daniel"));

        edited.apply_update(&update(json!({"title": "   "}))).unwrap();
        assert_eq!(edited.show_title, None);
        edited.apply_update(&update(json!({"author": null}))).unwrap();
        assert_eq!(edited.show_author, None);

        assert!(!edited.apply_update(&update(json!({}))).unwrap());
    }

    #[test]
    fn values_are_trimmed() {
        let mut edited = settings();
        edited.apply_update(&update(json!({"title": "  Lab Radio \n", "language": " en-GB "}))).unwrap();
        assert_eq!(edited.show_title.as_deref(), Some("Lab Radio"));
        assert_eq!(edited.show_language.as_deref(), Some("en-GB"));
    }

    #[test]
    fn invalid_values_are_refused_with_the_field_named() {
        let cases = [
            (json!({"title": "t".repeat(256)}), "title"),
            (json!({"author": "a".repeat(256)}), "author"),
            (json!({"description": "d".repeat(4001)}), "description"),
            (json!({"language": "english"}), "language"),
            (json!({"language": "en_US"}), "language"),
            (json!({"language": "e"}), "language"),
        ];
        for (body, keyword) in cases {
            let error = settings().apply_update(&update(body)).unwrap_err();
            assert!(matches!(error, PodcastError::InvalidInput(_)));
            assert!(error.to_string().to_lowercase().contains(keyword), "{keyword}: {error}");
        }
        assert_eq!(
            checked_text("title", Some(&"t".repeat(256)), TITLE_MAX).unwrap_err().to_string(),
            "The show title is too long (256 characters; the limit is 255)."
        );
    }

    #[test]
    fn limits_are_inclusive() {
        let mut edited = settings();
        edited
            .apply_update(&update(json!({
                "title": "t".repeat(255), "author": "a".repeat(255),
                "description": "d".repeat(4000), "language": "zh-Hans-CN",
            })))
            .unwrap();
        assert_eq!(edited.show_language.as_deref(), Some("zh-Hans-CN"));
    }

    #[test]
    fn a_refused_edit_changes_nothing() {
        let mut edited = settings();
        edited.apply_update(&update(json!({"title": "Before"}))).unwrap();
        let before = edited.clone();
        assert!(edited.apply_update(&update(json!({"title": "After", "language": "nope!"}))).is_err());
        assert_eq!(edited, before);
    }
}
