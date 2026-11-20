//! YouTube: title and transcript from the video's caption tracks.
//!
//! The watch page gives the title and the InnerTube API key; the InnerTube
//! `player` endpoint (Android client) lists the caption tracks, with the
//! watch page's own player response as the fallback. The chosen track is the
//! hinted language, then `youtube_languages` in order (English first by
//! default), then whatever exists; creator-made tracks beat auto-generated
//! ones of the same language. Videos without captions are
//! [`ContentError::NoTranscriptFound`]: downloading and transcribing the
//! audio would need a YouTube stream decipherer, which is out of scope.
//!
//! Errors that mean YouTube refused the request (bot check, rate limit,
//! empty caption bodies) are `ExternalService` errors whose text contains
//! "YouTube transcript", which [`ContentError::user_message`] turns into the
//! proxy / cookies advice.

use std::sync::OnceLock;

use regex::Regex;
use reqwest::header::{ACCEPT_LANGUAGE, COOKIE, USER_AGENT};
use serde_json::{json, Value};
use url::Url;

use super::fetch::network_error;
use super::xml::{self, XmlNode};
use super::{ExtractContext, Extracted};
use crate::error::ContentError;

/// Transcript languages tried when the settings name none; the list the
/// Python original passed to content-core.
pub const DEFAULT_YOUTUBE_LANGUAGES: &[&str] =
    &["en", "pt", "es", "de", "nl", "en-GB", "fr", "hi", "ja", "zh-CN", "zh-TW"];

const INNERTUBE_CLIENT: &str = "ANDROID";
const INNERTUBE_CLIENT_VERSION: &str = "20.10.38";

/// The 11-character video id of a YouTube URL (`watch?v=`, `youtu.be/`,
/// `/shorts/`, `/embed/`, `/live/`, `/v/`), or `None` when the URL is not a
/// video link.
pub fn youtube_video_id(url: &str) -> Option<String> {
    let parsed = Url::parse(url.trim()).ok()?;
    let host = parsed.host_str()?.to_ascii_lowercase();
    let valid = |id: &str| {
        (id.len() == 11 && id.chars().all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_')).then(|| id.to_string())
    };
    let mut segments = parsed.path_segments()?.filter(|s| !s.is_empty());
    if host == "youtu.be" {
        return valid(segments.next()?);
    }
    let youtube = host == "youtube.com"
        || host.ends_with(".youtube.com")
        || host == "youtube-nocookie.com"
        || host.ends_with(".youtube-nocookie.com");
    if !youtube {
        return None;
    }
    match segments.next()? {
        "watch" => parsed.query_pairs().find(|(key, _)| key == "v").and_then(|(_, id)| valid(&id)),
        "shorts" | "embed" | "live" | "v" | "e" => valid(segments.next()?),
        _ => None,
    }
}

#[derive(Debug, Clone, PartialEq)]
struct CaptionTrack {
    base_url: String,
    language: String,
    name: String,
    generated: bool,
}

pub(crate) async fn extract_youtube(url: &str, ctx: &ExtractContext) -> Result<Extracted, ContentError> {
    let video_id = youtube_video_id(url)
        .ok_or_else(|| ContentError::InvalidInput("the YouTube URL does not name a video".to_string()))?;
    let client = youtube_client(ctx)?;
    let cookies = cookie_header(ctx).await?;
    let base = ctx.youtube_base_url.trim_end_matches('/');
    let blocked = |reason: &str| {
        ContentError::ExternalService(format!("YouTube transcript extraction failed for video {video_id}: {reason}"))
    };

    let page_url = format!("{base}/watch?v={video_id}&hl=en");
    let response = client
        .get(&page_url)
        .header(USER_AGENT, ctx.user_agent.as_str())
        .header(ACCEPT_LANGUAGE, "en-US,en;q=0.9")
        .header(COOKIE, &cookies)
        .timeout(ctx.limits.engine_timeout)
        .send()
        .await
        .map_err(network_error)?;
    match response.status().as_u16() {
        200..=299 => {}
        429 => return Err(blocked("too many requests (HTTP 429)")),
        404 | 410 => return Err(ContentError::NotFound(format!("YouTube video {video_id} does not exist"))),
        code => return Err(blocked(&format!("watch page returned HTTP {code}"))),
    }
    let page = response.text().await.map_err(network_error)?;

    let page_player = initial_player_response(&page);
    let api_key = innertube_api_key(&page);
    if api_key.is_none() && page_player.is_none() {
        if page.contains("class=\"g-recaptcha\"") || page.contains("www.google.com/recaptcha") {
            return Err(blocked("YouTube asked for a captcha (the server IP is blocked)"));
        }
        return Err(blocked("the watch page did not contain video data"));
    }

    let mut player = None;
    if let Some(key) = &api_key {
        match innertube_player(&client, base, key, &video_id, &cookies, ctx).await {
            Ok(response) => player = Some(response),
            Err(error) => tracing::warn!("InnerTube player request failed, using the watch page data: {error}"),
        }
    }
    let player = match (player, page_player.clone()) {
        // The Android response sometimes omits captions the web page has.
        (Some(innertube), Some(web)) if caption_tracks(&innertube).is_empty() && !caption_tracks(&web).is_empty() => web,
        (Some(innertube), _) => innertube,
        (None, Some(web)) => web,
        (None, None) => return Err(blocked("could not load the video's player data")),
    };

    check_playability(&player, &video_id)?;

    let details = player.get("videoDetails").or_else(|| page_player.as_ref().and_then(|p| p.get("videoDetails")));
    let text = |key: &str| details.and_then(|d| d.get(key)).and_then(Value::as_str).map(str::trim).filter(|s| !s.is_empty());
    let title = text("title").map(str::to_string).or_else(|| page_title(&page));
    let channel = text("author").map(str::to_string);
    let duration = text("lengthSeconds").and_then(|s| s.parse::<u64>().ok());

    let tracks = caption_tracks(&player);
    let preferences = language_preferences(ctx);
    let Some(track) = choose_track(&tracks, &preferences) else {
        return Err(ContentError::NoTranscriptFound(format!("video {video_id} has no caption tracks")));
    };

    let track_url = Url::parse(&track.base_url)
        .or_else(|_| Url::parse(&format!("{base}/")).and_then(|b| b.join(&track.base_url)))
        .ok()
        .filter(|u| matches!(u.scheme(), "http" | "https"))
        .ok_or_else(|| blocked("the caption track URL is not valid"))?;
    let response = client
        .get(track_url)
        .header(USER_AGENT, ctx.user_agent.as_str())
        .header(COOKIE, &cookies)
        .timeout(ctx.limits.engine_timeout)
        .send()
        .await
        .map_err(network_error)?;
    match response.status().as_u16() {
        200..=299 => {}
        429 => return Err(blocked("too many requests for the caption track (HTTP 429)")),
        code => return Err(blocked(&format!("caption track returned HTTP {code}"))),
    }
    let body = response.text().await.map_err(network_error)?;
    if body.trim().is_empty() {
        return Err(blocked("the caption track came back empty (YouTube may require a proof-of-origin token)"));
    }
    let segments = parse_caption_xml(&body).map_err(|e| blocked(&format!("unreadable caption track: {e}")))?;
    let content = join_segments(&segments);

    let mut metadata = json!({
        "video_id": video_id,
        "language": track.language,
        "caption_name": track.name,
        "caption_kind": if track.generated { "auto" } else { "manual" },
        "available_languages": tracks.iter().map(|t| t.language.clone()).collect::<Vec<_>>(),
    });
    if let Some(channel) = channel {
        metadata["channel"] = json!(channel);
    }
    if let Some(duration) = duration {
        metadata["duration_seconds"] = json!(duration);
    }
    Ok(Extracted {
        title,
        content,
        identified_type: "youtube".to_string(),
        url: None,
        file_path: None,
        metadata,
    })
}

fn youtube_client(ctx: &ExtractContext) -> Result<reqwest::Client, ContentError> {
    match ctx.youtube_proxy.as_deref().map(str::trim).filter(|p| !p.is_empty()) {
        None => Ok(ctx.client.clone()),
        Some(proxy) => {
            let proxy = reqwest::Proxy::all(proxy)
                .map_err(|_| ContentError::Configuration("the YouTube proxy URL is not valid".to_string()))?;
            reqwest::Client::builder()
                .proxy(proxy)
                .connect_timeout(ctx.limits.connect_timeout)
                .build()
                .map_err(|e| ContentError::Configuration(format!("cannot build the YouTube HTTP client: {e}")))
        }
    }
}

/// The `Cookie` header for YouTube: the consent cookies that skip the EU
/// consent page, plus the YouTube cookies of the configured cookie file.
async fn cookie_header(ctx: &ExtractContext) -> Result<String, ContentError> {
    let mut cookies = vec!["CONSENT=YES+cb".to_string(), "SOCS=CAI".to_string()];
    if let Some(path) = &ctx.youtube_cookies_file {
        let text = tokio::fs::read_to_string(path).await.map_err(|e| {
            ContentError::Configuration(format!("the YouTube cookies file cannot be read: {e}"))
        })?;
        cookies.extend(parse_netscape_cookies(&text));
    }
    Ok(cookies.join("; "))
}

/// `name=value` pairs for YouTube domains from a Netscape `cookies.txt`.
fn parse_netscape_cookies(text: &str) -> Vec<String> {
    text.lines()
        .filter_map(|line| {
            let line = line.strip_prefix("#HttpOnly_").unwrap_or(line);
            if line.starts_with('#') || line.trim().is_empty() {
                return None;
            }
            let fields: Vec<&str> = line.split('\t').collect();
            if fields.len() < 7 {
                return None;
            }
            let domain = fields[0].trim_start_matches('.').to_ascii_lowercase();
            if domain != "youtube.com" && !domain.ends_with(".youtube.com") {
                return None;
            }
            let (name, value) = (fields[5].trim(), fields[6].trim());
            (!name.is_empty()).then(|| format!("{name}={value}"))
        })
        .collect()
}

async fn innertube_player(
    client: &reqwest::Client,
    base: &str,
    key: &str,
    video_id: &str,
    cookies: &str,
    ctx: &ExtractContext,
) -> Result<Value, ContentError> {
    let body = json!({
        "context": { "client": { "clientName": INNERTUBE_CLIENT, "clientVersion": INNERTUBE_CLIENT_VERSION, "hl": "en" } },
        "videoId": video_id,
    });
    let response = client
        .post(format!("{base}/youtubei/v1/player"))
        .query(&[("key", key), ("prettyPrint", "false")])
        .header(COOKIE, cookies)
        .json(&body)
        .timeout(ctx.limits.engine_timeout)
        .send()
        .await
        .map_err(network_error)?;
    if !response.status().is_success() {
        return Err(ContentError::ExternalService(format!("InnerTube player returned HTTP {}", response.status().as_u16())));
    }
    response.json::<Value>().await.map_err(|e| ContentError::ExternalService(format!("InnerTube player response: {e}")))
}

fn check_playability(player: &Value, video_id: &str) -> Result<(), ContentError> {
    let status = player.pointer("/playabilityStatus/status").and_then(Value::as_str).unwrap_or("OK");
    let reason = player
        .pointer("/playabilityStatus/reason")
        .and_then(Value::as_str)
        .unwrap_or("")
        .trim()
        .to_string();
    match status {
        "OK" | "LIVE_STREAM_OFFLINE" => Ok(()),
        "ERROR" => Err(ContentError::NotFound(format!("YouTube video {video_id} is unavailable: {reason}"))),
        "LOGIN_REQUIRED" if reason.to_ascii_lowercase().contains("bot") => Err(ContentError::ExternalService(format!(
            "YouTube transcript extraction failed for video {video_id}: request blocked ({reason})"
        ))),
        _ => Err(ContentError::ExternalService(format!(
            "YouTube transcript extraction failed for video {video_id}: video not playable ({status}: {reason})"
        ))),
    }
}

fn initial_player_response(page: &str) -> Option<Value> {
    let marker = "ytInitialPlayerResponse";
    let mut search_from = 0;
    while let Some(found) = page[search_from..].find(marker) {
        let after = search_from + found + marker.len();
        let rest = page[after..].trim_start();
        if let Some(rest) = rest.strip_prefix('=') {
            let rest = rest.trim_start();
            if rest.starts_with('{') {
                if let Some(Ok(value)) = serde_json::Deserializer::from_str(rest).into_iter::<Value>().next() {
                    return Some(value);
                }
            }
        }
        search_from = after;
    }
    None
}

fn innertube_api_key(page: &str) -> Option<String> {
    static RE: OnceLock<Regex> = OnceLock::new();
    let pattern = RE.get_or_init(|| Regex::new(r#""INNERTUBE_API_KEY"\s*:\s*"([A-Za-z0-9_-]+)""#).expect("valid regex"));
    pattern.captures(page).map(|c| c[1].to_string())
}

fn page_title(page: &str) -> Option<String> {
    static OG: OnceLock<Regex> = OnceLock::new();
    static TITLE: OnceLock<Regex> = OnceLock::new();
    let og = OG.get_or_init(|| Regex::new(r#"<meta\s+property="og:title"\s+content="([^"]*)""#).expect("valid regex"));
    let title = TITLE.get_or_init(|| Regex::new(r"(?is)<title[^>]*>(.*?)</title>").expect("valid regex"));
    let raw = og
        .captures(page)
        .map(|c| c[1].to_string())
        .or_else(|| title.captures(page).map(|c| c[1].trim().trim_end_matches("- YouTube").trim().to_string()))?;
    let text = decode_entities(&raw).trim().to_string();
    (!text.is_empty() && text != "YouTube").then_some(text)
}

fn caption_tracks(player: &Value) -> Vec<CaptionTrack> {
    let Some(tracks) = player.pointer("/captions/playerCaptionsTracklistRenderer/captionTracks").and_then(Value::as_array)
    else {
        return Vec::new();
    };
    tracks
        .iter()
        .filter_map(|track| {
            let base_url = track.get("baseUrl")?.as_str()?.to_string();
            let language = track.get("languageCode")?.as_str()?.to_string();
            let name = track
                .pointer("/name/simpleText")
                .and_then(Value::as_str)
                .map(str::to_string)
                .or_else(|| {
                    track.pointer("/name/runs").and_then(Value::as_array).map(|runs| {
                        runs.iter().filter_map(|r| r.get("text").and_then(Value::as_str)).collect::<String>()
                    })
                })
                .unwrap_or_else(|| language.clone());
            let generated = track.get("kind").and_then(Value::as_str) == Some("asr");
            Some(CaptionTrack { base_url, language, name, generated })
        })
        .collect()
}

fn language_preferences(ctx: &ExtractContext) -> Vec<String> {
    let mut preferences: Vec<String> = Vec::new();
    let candidates = ctx
        .language
        .iter()
        .cloned()
        .chain(ctx.youtube_languages.iter().cloned())
        .chain(std::iter::once("en".to_string()));
    for language in candidates {
        let language = language.trim().to_string();
        if !language.is_empty() && !preferences.iter().any(|p| p.eq_ignore_ascii_case(&language)) {
            preferences.push(language);
        }
    }
    preferences
}

fn base_language(code: &str) -> String {
    code.split(['-', '_']).next().unwrap_or(code).to_ascii_lowercase()
}

/// Preferred language first (exact code, then same base language),
/// creator-made before auto-generated; then any creator-made track; then any.
fn choose_track<'a>(tracks: &'a [CaptionTrack], preferences: &[String]) -> Option<&'a CaptionTrack> {
    for preference in preferences {
        for generated in [false, true] {
            if let Some(track) =
                tracks.iter().find(|t| t.generated == generated && t.language.eq_ignore_ascii_case(preference))
            {
                return Some(track);
            }
        }
        let wanted = base_language(preference);
        for generated in [false, true] {
            if let Some(track) = tracks.iter().find(|t| t.generated == generated && base_language(&t.language) == wanted) {
                return Some(track);
            }
        }
    }
    tracks.iter().find(|t| !t.generated).or_else(|| tracks.first())
}

/// Caption lines from the classic (`<transcript><text>`) or srv3
/// (`<timedtext><body><p>`) format.
fn parse_caption_xml(body: &str) -> Result<Vec<String>, String> {
    let root = xml::parse(body)?;
    let mut nodes: Vec<&XmlNode> = Vec::new();
    match root.name.as_str() {
        "transcript" => root.find_all("text", &mut nodes),
        "timedtext" => root.find_all("p", &mut nodes),
        other => return Err(format!("unexpected caption document <{other}>")),
    }
    Ok(nodes
        .into_iter()
        .map(|node| clean_caption(&node.text()))
        .filter(|line| !line.is_empty())
        .collect())
}

/// Caption text is HTML-escaped inside the XML and may carry `<font>` tags.
fn clean_caption(raw: &str) -> String {
    static TAGS: OnceLock<Regex> = OnceLock::new();
    let tags = TAGS.get_or_init(|| Regex::new(r"</?[a-zA-Z][^>]*>").expect("valid regex"));
    let decoded = decode_entities(raw);
    let stripped = tags.replace_all(&decoded, " ");
    stripped.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn decode_entities(text: &str) -> String {
    static ENTITY: OnceLock<Regex> = OnceLock::new();
    let entity = ENTITY.get_or_init(|| Regex::new(r"&(#[0-9]{1,7}|#[xX][0-9a-fA-F]{1,6}|[a-zA-Z]{2,8});").expect("valid regex"));
    entity
        .replace_all(text, |caps: &regex::Captures| {
            let name = &caps[1];
            let decoded = if let Some(hex) = name.strip_prefix("#x").or_else(|| name.strip_prefix("#X")) {
                u32::from_str_radix(hex, 16).ok().and_then(char::from_u32)
            } else if let Some(decimal) = name.strip_prefix('#') {
                decimal.parse::<u32>().ok().and_then(char::from_u32)
            } else {
                match name {
                    "amp" => Some('&'),
                    "lt" => Some('<'),
                    "gt" => Some('>'),
                    "quot" => Some('"'),
                    "apos" => Some('\''),
                    "nbsp" => Some(' '),
                    _ => None,
                }
            };
            decoded.map_or_else(|| caps[0].to_string(), String::from)
        })
        .into_owned()
}

/// Join caption lines into paragraphs of roughly 600 characters, breaking
/// after a line that ends a sentence.
fn join_segments(lines: &[String]) -> String {
    let mut paragraphs = Vec::new();
    let mut current = String::new();
    for line in lines {
        if !current.is_empty() {
            current.push(' ');
        }
        current.push_str(line);
        if current.len() >= 600 && line.ends_with(['.', '!', '?']) {
            paragraphs.push(std::mem::take(&mut current));
        }
    }
    if !current.is_empty() {
        paragraphs.push(current);
    }
    paragraphs.join("\n\n")
}

#[cfg(test)]
mod tests {
    use wiremock::matchers::{method, path, query_param};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    use super::*;
    use crate::extract::{extract_content, ExtractInput};

    const VIDEO: &str = "dQw4w9WgXcQ";

    #[test]
    fn video_ids() {
        for url in [
            "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            "https://youtube.com/watch?feature=share&v=dQw4w9WgXcQ&t=10s",
            "https://m.youtube.com/watch?v=dQw4w9WgXcQ",
            "https://music.youtube.com/watch?v=dQw4w9WgXcQ&list=x",
            "https://youtu.be/dQw4w9WgXcQ?si=abc",
            "https://www.youtube.com/shorts/dQw4w9WgXcQ",
            "https://www.youtube.com/embed/dQw4w9WgXcQ?start=3",
            "https://www.youtube.com/live/dQw4w9WgXcQ",
            "https://www.youtube-nocookie.com/embed/dQw4w9WgXcQ",
        ] {
            assert_eq!(youtube_video_id(url).as_deref(), Some(VIDEO), "{url}");
        }
        for url in [
            "https://www.youtube.com/channel/UC123",
            "https://www.youtube.com/watch?v=short",
            "https://www.youtube.com/@someone",
            "https://notyoutube.com/watch?v=dQw4w9WgXcQ",
            "https://youtu.be/",
            "not a url",
        ] {
            assert_eq!(youtube_video_id(url), None, "{url}");
        }
    }

    fn track(language: &str, generated: bool) -> CaptionTrack {
        CaptionTrack { base_url: format!("u-{language}-{generated}"), language: language.into(), name: language.into(), generated }
    }

    #[test]
    fn track_choice() {
        let prefs = |list: &[&str]| list.iter().map(|s| s.to_string()).collect::<Vec<_>>();
        let tracks = vec![track("de", false), track("en", true), track("en-GB", false), track("fr", true)];
        assert_eq!(choose_track(&tracks, &prefs(&["fr", "en"])).unwrap().language, "fr");
        // Exact code first (creator-made before auto, as youtube-transcript-api
        // does), then the same base language.
        assert_eq!(choose_track(&tracks, &prefs(&["en"])).unwrap().language, "en");
        assert_eq!(choose_track(&tracks, &prefs(&["en-US"])).unwrap().language, "en-GB");
        let only_auto_en = vec![track("de", false), track("en", true)];
        assert_eq!(choose_track(&only_auto_en, &prefs(&["en"])).unwrap().language, "en");
        // Nothing preferred: a creator-made track, then anything.
        assert_eq!(choose_track(&tracks, &prefs(&["ja"])).unwrap().language, "de");
        assert_eq!(choose_track(&[track("ko", true)], &prefs(&["ja"])).unwrap().language, "ko");
        assert!(choose_track(&[], &prefs(&["en"])).is_none());

        let ctx = ExtractContext { language: Some("pt".into()), youtube_languages: prefs(&["es", "PT"]), ..ExtractContext::default() };
        assert_eq!(language_preferences(&ctx), prefs(&["pt", "es", "en"]));
        assert_eq!(language_preferences(&ExtractContext::default())[0], "en");
    }

    #[test]
    fn caption_formats() {
        let classic = r#"<?xml version="1.0" encoding="utf-8" ?><transcript><text start="0.5" dur="1.2">Hello &amp;amp; welcome</text><text start="2" dur="1">it&amp;#39;s   &lt;font color=&quot;#E5E5E5&quot;&gt;great&lt;/font&gt;</text><text start="3" dur="1">  </text></transcript>"#;
        assert_eq!(parse_caption_xml(classic).unwrap(), vec!["Hello & welcome".to_string(), "it's great".to_string()]);
        let srv3 = r#"<timedtext format="3"><body><p t="0" d="1000"><s>Line</s><s t="300"> one</s></p><p t="1000" d="500">Line two</p></body></timedtext>"#;
        assert_eq!(parse_caption_xml(srv3).unwrap(), vec!["Line one".to_string(), "Line two".to_string()]);
        assert!(parse_caption_xml("<html></html>").is_err());
        assert!(parse_caption_xml("").is_err());
    }

    #[test]
    fn segments_become_paragraphs() {
        let lines: Vec<String> = (0..40).map(|i| format!("Sentence number {i} is here.")).collect();
        let text = join_segments(&lines);
        assert!(text.contains("\n\n"));
        assert!(text.split("\n\n").all(|p| p.ends_with('.')));
        assert_eq!(join_segments(&["a".into(), "b".into()]), "a b");
    }

    #[test]
    fn cookies_file_parsing() {
        let text = "# Netscape HTTP Cookie File\n.youtube.com\tTRUE\t/\tTRUE\t0\tSID\tabc\n#HttpOnly_.youtube.com\tTRUE\t/\tTRUE\t0\tHSID\tdef\n.google.com\tTRUE\t/\tTRUE\t0\tNID\tzzz\nbroken line\n";
        assert_eq!(parse_netscape_cookies(text), vec!["SID=abc".to_string(), "HSID=def".to_string()]);
    }

    fn watch_page(player: &Value, api_key: bool) -> String {
        let key = if api_key { r#"ytcfg.set({"INNERTUBE_API_KEY":"AIzaTestKey_123","INNERTUBE_CONTEXT":{}});"# } else { "" };
        format!(
            r#"<!DOCTYPE html><html><head><title>Never Gonna - YouTube</title><meta property="og:title" content="Never Gonna &amp; Friends"></head><body><script>{key}</script><script>var ytInitialPlayerResponse = {player};var meta = document.createElement('meta');</script></body></html>"#
        )
    }

    fn player(server: &MockServer, tracks: Value) -> Value {
        let _ = server;
        json!({
            "playabilityStatus": {"status": "OK"},
            "videoDetails": {"videoId": VIDEO, "title": "Never Gonna Give You Up", "author": "Rick Astley", "lengthSeconds": "213"},
            "captions": {"playerCaptionsTracklistRenderer": {"captionTracks": tracks}}
        })
    }

    async fn youtube_ctx(server: &MockServer) -> ExtractContext {
        ExtractContext { youtube_base_url: server.uri(), ..ExtractContext::default() }
    }

    async fn run(ctx: &ExtractContext) -> Result<Extracted, ContentError> {
        extract_content(ExtractInput::url(format!("https://www.youtube.com/watch?v={VIDEO}")), ctx).await
    }

    #[tokio::test]
    async fn transcript_via_innertube_in_the_preferred_language() {
        let server = MockServer::start().await;
        let tracks = json!([
            {"baseUrl": format!("{}/api/timedtext?v={VIDEO}&lang=de", server.uri()), "languageCode": "de", "name": {"simpleText": "German"}},
            {"baseUrl": format!("{}/api/timedtext?v={VIDEO}&lang=en&kind=asr", server.uri()), "languageCode": "en", "kind": "asr", "name": {"runs": [{"text": "English (auto-generated)"}]}},
            {"baseUrl": format!("{}/api/timedtext?v={VIDEO}&lang=es", server.uri()), "languageCode": "es", "name": {"simpleText": "Spanish"}}
        ]);
        // The page itself has no captions: the InnerTube answer supplies them.
        let page = watch_page(&player(&server, json!([])), true);
        Mock::given(method("GET"))
            .and(path("/watch"))
            .and(query_param("v", VIDEO))
            .respond_with(ResponseTemplate::new(200).set_body_raw(page, "text/html"))
            .mount(&server)
            .await;
        Mock::given(method("POST"))
            .and(path("/youtubei/v1/player"))
            .and(query_param("key", "AIzaTestKey_123"))
            .respond_with(ResponseTemplate::new(200).set_body_json(player(&server, tracks)))
            .mount(&server)
            .await;
        Mock::given(method("GET"))
            .and(path("/api/timedtext"))
            .and(query_param("lang", "en"))
            .respond_with(ResponseTemplate::new(200).set_body_raw(
                r#"<?xml version="1.0" encoding="utf-8" ?><transcript><text start="0" dur="2">We&amp;#39;re no strangers</text><text start="2" dur="2">to love.</text></transcript>"#,
                "text/xml",
            ))
            .mount(&server)
            .await;
        Mock::given(method("GET"))
            .and(path("/api/timedtext"))
            .and(query_param("lang", "es"))
            .respond_with(ResponseTemplate::new(200).set_body_raw(
                r#"<transcript><text start="0" dur="2">Nunca te voy a dejar.</text></transcript>"#,
                "text/xml",
            ))
            .mount(&server)
            .await;

        let ctx = youtube_ctx(&server).await;
        let out = run(&ctx).await.unwrap();
        assert_eq!(out.identified_type, "youtube");
        assert_eq!(out.title.as_deref(), Some("Never Gonna Give You Up"));
        assert_eq!(out.content, "We're no strangers to love.");
        assert_eq!(out.url.as_deref(), Some(format!("https://www.youtube.com/watch?v={VIDEO}").as_str()));
        assert_eq!(out.metadata["language"], "en");
        assert_eq!(out.metadata["caption_kind"], "auto");
        assert_eq!(out.metadata["channel"], "Rick Astley");
        assert_eq!(out.metadata["duration_seconds"], 213);
        assert_eq!(out.metadata["available_languages"], json!(["de", "en", "es"]));

        // The language hint wins over English.
        let hinted = ExtractContext { language: Some("es".into()), ..youtube_ctx(&server).await };
        let out = run(&hinted).await.unwrap();
        assert_eq!(out.content, "Nunca te voy a dejar.");
        assert_eq!(out.metadata["caption_kind"], "manual");
    }

    #[tokio::test]
    async fn watch_page_player_is_the_fallback() {
        let server = MockServer::start().await;
        let tracks = json!([{"baseUrl": format!("{}/api/timedtext?lang=fr", server.uri()), "languageCode": "fr", "name": {"simpleText": "French"}}]);
        let mut data = player(&server, tracks);
        data["videoDetails"] = json!({});
        Mock::given(method("GET"))
            .and(path("/watch"))
            .respond_with(ResponseTemplate::new(200).set_body_raw(watch_page(&data, false), "text/html"))
            .mount(&server)
            .await;
        Mock::given(method("GET"))
            .and(path("/api/timedtext"))
            .respond_with(ResponseTemplate::new(200).set_body_raw(
                r#"<timedtext format="3"><body><p t="0" d="1000">Bonjour à tous.</p></body></timedtext>"#,
                "text/xml",
            ))
            .mount(&server)
            .await;
        let out = run(&youtube_ctx(&server).await).await.unwrap();
        // No English track: whatever exists is used. Title from og:title.
        assert_eq!(out.content, "Bonjour à tous.");
        assert_eq!(out.title.as_deref(), Some("Never Gonna & Friends"));
        assert_eq!(out.metadata["language"], "fr");
    }

    #[tokio::test]
    async fn missing_captions_and_blocking() {
        // No caption tracks at all.
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/watch"))
            .respond_with(ResponseTemplate::new(200).set_body_raw(watch_page(&player(&server, json!([])), false), "text/html"))
            .mount(&server)
            .await;
        let error = run(&youtube_ctx(&server).await).await.unwrap_err();
        assert!(matches!(error, ContentError::NoTranscriptFound(_)), "{error:?}");
        assert!(error.user_message(Some("https://youtu.be/dQw4w9WgXcQ")).contains("Speech-to-Text"));

        // Bot check.
        let bot = MockServer::start().await;
        let blocked = json!({"playabilityStatus": {"status": "LOGIN_REQUIRED", "reason": "Sign in to confirm you're not a bot"}});
        Mock::given(method("GET"))
            .and(path("/watch"))
            .respond_with(ResponseTemplate::new(200).set_body_raw(watch_page(&blocked, false), "text/html"))
            .mount(&bot)
            .await;
        let error = run(&youtube_ctx(&bot).await).await.unwrap_err();
        assert!(matches!(&error, ContentError::ExternalService(m) if m.contains("YouTube transcript")), "{error:?}");
        assert!(error.user_message(Some("https://www.youtube.com/watch?v=dQw4w9WgXcQ")).contains("proxy"));

        // Captcha page.
        let captcha = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/watch"))
            .respond_with(ResponseTemplate::new(200).set_body_raw("<html><div class=\"g-recaptcha\"></div></html>", "text/html"))
            .mount(&captcha)
            .await;
        let error = run(&youtube_ctx(&captcha).await).await.unwrap_err();
        assert!(matches!(&error, ContentError::ExternalService(m) if m.contains("captcha")), "{error:?}");

        // Rate limited.
        let limited = MockServer::start().await;
        Mock::given(method("GET")).respond_with(ResponseTemplate::new(429)).mount(&limited).await;
        let error = run(&youtube_ctx(&limited).await).await.unwrap_err();
        assert!(matches!(&error, ContentError::ExternalService(m) if m.contains("429")), "{error:?}");

        // Removed video.
        let gone = MockServer::start().await;
        let unavailable = json!({"playabilityStatus": {"status": "ERROR", "reason": "Video unavailable"}});
        Mock::given(method("GET"))
            .respond_with(ResponseTemplate::new(200).set_body_raw(watch_page(&unavailable, false), "text/html"))
            .mount(&gone)
            .await;
        assert!(matches!(run(&youtube_ctx(&gone).await).await, Err(ContentError::NotFound(_))));

        // Caption track that comes back empty.
        let empty = MockServer::start().await;
        let tracks = json!([{"baseUrl": format!("{}/api/timedtext?lang=en", empty.uri()), "languageCode": "en"}]);
        Mock::given(method("GET"))
            .and(path("/watch"))
            .respond_with(ResponseTemplate::new(200).set_body_raw(watch_page(&player(&empty, tracks), false), "text/html"))
            .mount(&empty)
            .await;
        Mock::given(method("GET")).and(path("/api/timedtext")).respond_with(ResponseTemplate::new(200)).mount(&empty).await;
        let error = run(&youtube_ctx(&empty).await).await.unwrap_err();
        assert!(matches!(&error, ContentError::ExternalService(m) if m.contains("empty")), "{error:?}");
    }

    #[tokio::test]
    async fn cookies_file_is_sent_and_a_missing_one_is_a_configuration_error() {
        let server = MockServer::start().await;
        Mock::given(method("GET")).respond_with(ResponseTemplate::new(429)).mount(&server).await;
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("cookies.txt");
        std::fs::write(&file, ".youtube.com\tTRUE\t/\tTRUE\t0\tLOGIN_INFO\tsecret-cookie\n").unwrap();
        let ctx = ExtractContext { youtube_cookies_file: Some(file), ..youtube_ctx(&server).await };
        let _ = run(&ctx).await;
        let request = &server.received_requests().await.unwrap()[0];
        let cookie = request.headers.get("cookie").unwrap().to_str().unwrap();
        assert!(cookie.contains("LOGIN_INFO=secret-cookie") && cookie.contains("CONSENT=YES+cb"), "{cookie}");

        let ctx = ExtractContext { youtube_cookies_file: Some(dir.path().join("missing.txt")), ..youtube_ctx(&server).await };
        assert!(matches!(run(&ctx).await, Err(ContentError::Configuration(_))));
    }
}
