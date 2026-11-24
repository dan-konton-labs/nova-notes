//! Content extraction: one entry point for pasted text, URLs and files.
//!
//! This replaces the `content-core` library the Python original called from
//! `open_notebook/graphs/source.py`. [`extract_content`] takes an
//! [`ExtractInput`] and an [`ExtractContext`] and returns [`Extracted`]:
//! Markdown or plain text, an optional title, and a stable type string.
//!
//! | Input | Handling |
//! |-------|----------|
//! | pasted text | passed through (a full pasted HTML document is converted to Markdown) |
//! | web page URL | fetched through the SSRF guard, main content converted to Markdown; or the Jina, Firecrawl or remote Crawl4AI engines |
//! | document / media URL | downloaded to a temporary file, then handled as a file |
//! | YouTube URL | title and caption-track transcript |
//! | pdf, docx, pptx, odt, odp, epub, html | text and structure as Markdown |
//! | xlsx, xls, xlsb, ods, csv, tsv | Markdown tables |
//! | markdown, text, json, yaml, xml, source code | as text |
//! | audio, video | transcribed through the supplied speech-to-text model (ffmpeg for video and long audio) |
//! | images, legacy `.doc` / `.ppt`, archives, other binaries | [`ContentError::UnsupportedType`] |
//!
//! The caller owns configuration: nothing here reads the environment (except
//! `PATH`, to locate ffmpeg) or a database. The caller is also responsible
//! for confining file paths to its uploads directory before calling in.

mod detect;
mod documents;
mod engines;
mod fetch;
mod files;
pub mod html;
mod markdown;
mod media;
mod web;
mod xml;
mod youtube;

use std::fmt;
use std::path::{Path, PathBuf};
use std::str::FromStr;
use std::sync::Arc;
use std::time::Duration;

use nn_llm::SpeechToTextModel;
use serde::{Deserialize, Serialize};
use serde_json::json;

use crate::error::{is_youtube_url, ContentError};
use crate::url_validation::{validate_url_with, Resolver, SystemResolver, UrlPolicy};

pub use youtube::{youtube_video_id, DEFAULT_YOUTUBE_LANGUAGES};

/// User agent sent by the built-in fetcher. Many sites serve an error page to
/// clients that do not look like a browser.
pub const DEFAULT_USER_AGENT: &str = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 \
(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36";

/// Default Jina Reader endpoint.
pub const DEFAULT_JINA_READER_URL: &str = "https://r.jina.ai";
/// Default Firecrawl API base URL.
pub const DEFAULT_FIRECRAWL_API_URL: &str = "https://api.firecrawl.dev";
/// Default YouTube origin.
pub const DEFAULT_YOUTUBE_BASE_URL: &str = "https://www.youtube.com";

/// What to extract: exactly one of pasted text, a URL, or a local file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ExtractInput {
    /// Text pasted by the user.
    Text(String),
    /// An `http(s)` URL: web page, document, media file or YouTube video.
    Url(String),
    /// A file on local disk.
    File {
        path: PathBuf,
        /// The name the user uploaded the file under, when `path` is a
        /// server-generated name. Only its extension is used.
        file_name: Option<String>,
        /// MIME type declared by the uploader. A hint only: magic bytes win.
        mime_type: Option<String>,
    },
}

impl ExtractInput {
    pub fn text(content: impl Into<String>) -> Self {
        Self::Text(content.into())
    }

    pub fn url(url: impl Into<String>) -> Self {
        Self::Url(url.into())
    }

    pub fn file(path: impl Into<PathBuf>) -> Self {
        Self::File { path: path.into(), file_name: None, mime_type: None }
    }

    /// Set the original file name. No effect on text and URL inputs.
    pub fn with_file_name(mut self, name: impl Into<String>) -> Self {
        if let Self::File { file_name, .. } = &mut self {
            *file_name = Some(name.into());
        }
        self
    }

    /// Set the declared MIME type. No effect on text and URL inputs.
    pub fn with_mime_type(mut self, mime: impl Into<String>) -> Self {
        if let Self::File { mime_type, .. } = &mut self {
            *mime_type = Some(mime.into());
        }
        self
    }
}

/// How web pages are turned into text.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum UrlEngine {
    /// Firecrawl, Jina, Crawl4AI (each only when configured), then the
    /// built-in engine; the first that returns content wins.
    #[default]
    Auto,
    /// Built-in: plain HTTP fetch and HTML-to-Markdown conversion. No JavaScript.
    Simple,
    /// Jina Reader (`r.jina.ai`). Works without a key, at a lower rate limit.
    Jina,
    /// Firecrawl scrape API. Needs an API key (or a self-hosted base URL).
    Firecrawl,
    /// A remote Crawl4AI server. Needs its base URL.
    Crawl4ai,
}

impl UrlEngine {
    pub fn as_str(&self) -> &'static str {
        match self {
            UrlEngine::Auto => "auto",
            UrlEngine::Simple => "simple",
            UrlEngine::Jina => "jina",
            UrlEngine::Firecrawl => "firecrawl",
            UrlEngine::Crawl4ai => "crawl4ai",
        }
    }
}

impl fmt::Display for UrlEngine {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl FromStr for UrlEngine {
    type Err = ContentError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.trim().to_ascii_lowercase().as_str() {
            "auto" => Ok(UrlEngine::Auto),
            "simple" => Ok(UrlEngine::Simple),
            "jina" => Ok(UrlEngine::Jina),
            "firecrawl" => Ok(UrlEngine::Firecrawl),
            "crawl4ai" => Ok(UrlEngine::Crawl4ai),
            other => Err(ContentError::Configuration(format!("unknown URL engine '{other}'"))),
        }
    }
}

/// Size and time limits.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Limits {
    /// Largest response body the built-in fetcher will download.
    pub max_download_bytes: u64,
    /// TCP / TLS connect timeout of the built-in fetcher.
    pub connect_timeout: Duration,
    /// Whole-request timeout of the built-in fetcher, per hop.
    pub request_timeout: Duration,
    /// Redirects followed (each target is re-validated) before giving up.
    pub max_redirects: usize,
    /// Timeout for one call to a remote engine (Jina, Firecrawl, Crawl4AI) or YouTube.
    pub engine_timeout: Duration,
    /// Timeout for one speech-to-text request (`CCORE_STT_TIMEOUT` in the original).
    pub stt_timeout: Duration,
    /// Timeout for one ffmpeg / ffprobe run.
    pub ffmpeg_timeout: Duration,
}

impl Default for Limits {
    fn default() -> Self {
        Self {
            max_download_bytes: 100 * 1024 * 1024,
            connect_timeout: Duration::from_secs(15),
            request_timeout: Duration::from_secs(120),
            max_redirects: 10,
            engine_timeout: Duration::from_secs(120),
            stt_timeout: Duration::from_secs(3600),
            ffmpeg_timeout: Duration::from_secs(1800),
        }
    }
}

/// Where to find ffmpeg and ffprobe.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum FfmpegLocation {
    /// Look both up on `PATH` at the time they are needed.
    #[default]
    Auto,
    /// Use these binaries. Without `ffprobe`, durations are not probed.
    Paths { ffmpeg: PathBuf, ffprobe: Option<PathBuf> },
    /// Behave as if ffmpeg were not installed.
    Disabled,
}

/// Audio and video transcription settings.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AudioOptions {
    /// Audio longer than this is split into segments of this length before
    /// transcription (`CCORE_AUDIO_SEGMENT_MINUTES` in the original, in
    /// seconds here). `0` never splits: the file, or a video's audio track,
    /// is sent whole.
    pub segment_seconds: u64,
    /// Segments transcribed in parallel (`CCORE_AUDIO_CONCURRENCY`), 1 to 10.
    pub concurrency: usize,
    /// Largest file sent to the speech-to-text model in one request. Files
    /// above it are always re-encoded and split. 25 MB is the common provider cap.
    pub max_upload_bytes: u64,
    pub ffmpeg: FfmpegLocation,
}

impl Default for AudioOptions {
    fn default() -> Self {
        Self {
            segment_seconds: 600,
            concurrency: 3,
            max_upload_bytes: 25 * 1024 * 1024,
            ffmpeg: FfmpegLocation::Auto,
        }
    }
}

/// Everything extraction needs from its caller.
///
/// `Default` is a working zero-configuration setup: the `auto` URL engine
/// with no remote engine configured (so the built-in one runs), no
/// speech-to-text model, the self-hosted URL policy.
#[derive(Clone)]
pub struct ExtractContext {
    /// Client for third-party API calls (Jina, Firecrawl, Crawl4AI) and for
    /// YouTube. User-supplied URLs are *not* fetched with it: the built-in
    /// fetcher builds its own clients so it can pin DNS and vet every redirect.
    pub client: reqwest::Client,
    pub url_engine: UrlEngine,
    /// `JINA_API_KEY`.
    pub jina_api_key: Option<String>,
    /// Jina Reader endpoint; the page URL is appended as the path.
    pub jina_reader_url: String,
    /// `FIRECRAWL_API_KEY`.
    pub firecrawl_api_key: Option<String>,
    /// `FIRECRAWL_API_URL`.
    pub firecrawl_api_url: String,
    /// `CCORE_FIRECRAWL_PROXY`: `basic`, `stealth` or `auto`.
    pub firecrawl_proxy: String,
    /// `CCORE_FIRECRAWL_WAIT_FOR`: milliseconds Firecrawl waits for JavaScript.
    pub firecrawl_wait_for_ms: u64,
    /// `CRAWL4AI_API_URL`: base URL of a remote Crawl4AI server.
    pub crawl4ai_api_url: Option<String>,
    /// `CRAWL4AI_API_TOKEN`.
    pub crawl4ai_api_token: Option<String>,
    /// Model used to transcribe audio and video.
    pub speech_to_text: Option<Arc<dyn SpeechToTextModel>>,
    /// ISO 639-1 language hint: passed to the speech-to-text model and tried
    /// first when choosing a YouTube caption track.
    pub language: Option<String>,
    /// Caption languages tried after `language`, in order, before falling
    /// back to whatever track exists.
    pub youtube_languages: Vec<String>,
    /// YouTube origin (overridable for tests).
    pub youtube_base_url: String,
    /// `CCORE_YOUTUBE_PROXY`: proxy URL used for YouTube requests only.
    pub youtube_proxy: Option<String>,
    /// `CCORE_YOUTUBE_COOKIES_FILE`: Netscape-format cookie file sent to YouTube.
    pub youtube_cookies_file: Option<PathBuf>,
    /// Proxy URL for the built-in fetcher. With a proxy, DNS for the target
    /// is resolved by the proxy, so addresses are checked but not pinned.
    pub proxy: Option<String>,
    pub user_agent: String,
    pub url_policy: UrlPolicy,
    /// Resolver used by the SSRF guard and for DNS pinning.
    pub resolver: Arc<dyn Resolver>,
    pub limits: Limits,
    pub audio: AudioOptions,
}

impl Default for ExtractContext {
    fn default() -> Self {
        Self {
            client: reqwest::Client::new(),
            url_engine: UrlEngine::Auto,
            jina_api_key: None,
            jina_reader_url: DEFAULT_JINA_READER_URL.to_string(),
            firecrawl_api_key: None,
            firecrawl_api_url: DEFAULT_FIRECRAWL_API_URL.to_string(),
            firecrawl_proxy: "auto".to_string(),
            firecrawl_wait_for_ms: 3000,
            crawl4ai_api_url: None,
            crawl4ai_api_token: None,
            speech_to_text: None,
            language: None,
            youtube_languages: DEFAULT_YOUTUBE_LANGUAGES.iter().map(|s| s.to_string()).collect(),
            youtube_base_url: DEFAULT_YOUTUBE_BASE_URL.to_string(),
            youtube_proxy: None,
            youtube_cookies_file: None,
            proxy: None,
            user_agent: DEFAULT_USER_AGENT.to_string(),
            url_policy: UrlPolicy::default(),
            resolver: Arc::new(SystemResolver),
            limits: Limits::default(),
            audio: AudioOptions::default(),
        }
    }
}

impl fmt::Debug for ExtractContext {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let set = |present: bool| if present { "<set>" } else { "<unset>" };
        f.debug_struct("ExtractContext")
            .field("url_engine", &self.url_engine)
            .field("jina_api_key", &set(self.jina_api_key.is_some()))
            .field("jina_reader_url", &self.jina_reader_url)
            .field("firecrawl_api_key", &set(self.firecrawl_api_key.is_some()))
            .field("firecrawl_api_url", &self.firecrawl_api_url)
            .field("crawl4ai_api_url", &self.crawl4ai_api_url)
            .field("crawl4ai_api_token", &set(self.crawl4ai_api_token.is_some()))
            .field(
                "speech_to_text",
                &self.speech_to_text.as_ref().map(|m| format!("{}/{}", m.provider(), m.model_name())),
            )
            .field("language", &self.language)
            .field("youtube_languages", &self.youtube_languages)
            .field("youtube_proxy", &set(self.youtube_proxy.is_some()))
            .field("youtube_cookies_file", &self.youtube_cookies_file)
            .field("proxy", &set(self.proxy.is_some()))
            .field("url_policy", &self.url_policy)
            .field("limits", &self.limits)
            .field("audio", &self.audio)
            .finish_non_exhaustive()
    }
}

/// The result of an extraction.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Extracted {
    /// Title declared by the content itself (page title, document
    /// properties, video title). `None` when the content declares none: the
    /// file name is never used as a stand-in.
    pub title: Option<String>,
    /// Markdown or plain text. Never empty or whitespace-only.
    pub content: String,
    /// One of `text`, `url`, `youtube`, `pdf`, `docx`, `pptx`, `odt`, `odp`,
    /// `xlsx`, `xls`, `xlsb`, `ods`, `csv`, `epub`, `html`, `markdown`,
    /// `audio`, `video`. A URL that served a document or media file reports
    /// the document's type.
    pub identified_type: String,
    /// The URL the input named (not the final URL after redirects; that is
    /// `metadata.final_url`).
    pub url: Option<String>,
    /// The file the input named. `None` for URL downloads.
    pub file_path: Option<PathBuf>,
    /// Type-specific details: `mime_type`, `engine`, `final_url`, `language`,
    /// `pages`, `sheets`, `slides`, `duration_seconds`, `segments`, `author`...
    pub metadata: serde_json::Value,
}

/// Verdict of [`check_file_support`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FileSupport {
    pub supported: bool,
    /// The same string [`Extracted::identified_type`] would carry, or the
    /// detected MIME type when the file is not supported.
    pub identified_type: String,
    pub mime_type: String,
    /// Why the file is not supported.
    pub reason: Option<String>,
}

/// Extract text from `input`.
///
/// Errors are typed ([`ContentError`]); content that extracts to nothing is
/// [`ContentError::EmptyContent`] rather than an empty success.
pub async fn extract_content(input: ExtractInput, ctx: &ExtractContext) -> Result<Extracted, ContentError> {
    let extracted = match input {
        ExtractInput::Text(text) => extract_text(text).await?,
        ExtractInput::Url(url) => extract_url(&url, ctx).await?,
        ExtractInput::File { path, file_name, mime_type } => {
            let mut extracted = files::extract_file(&path, file_name.as_deref(), mime_type.as_deref(), ctx).await?;
            extracted.file_path = Some(path);
            extracted
        }
    };
    if extracted.content.trim().is_empty() {
        return Err(ContentError::EmptyContent(format!(
            "{} input produced no text",
            extracted.identified_type
        )));
    }
    Ok(extracted)
}

/// Decide from the file header whether [`extract_content`] can handle `path`,
/// without extracting it. Uses the same detection as extraction, so the two
/// cannot disagree. Audio and video count as supported even though they also
/// need a speech-to-text model at extraction time.
pub async fn check_file_support(
    path: &Path,
    file_name: Option<&str>,
    mime_type: Option<&str>,
) -> Result<FileSupport, ContentError> {
    let detected = files::detect_file(path, file_name, mime_type).await?;
    Ok(match detected.kind.unsupported_reason() {
        Some(reason) => FileSupport {
            supported: false,
            identified_type: detected.mime.clone(),
            mime_type: detected.mime,
            reason: Some(reason),
        },
        None => FileSupport {
            supported: true,
            identified_type: detected.kind.identified_type().to_string(),
            mime_type: detected.mime,
            reason: None,
        },
    })
}

async fn extract_text(text: String) -> Result<Extracted, ContentError> {
    let trimmed = text.trim_start();
    let head: String = trimmed.chars().take(32).collect::<String>().to_ascii_lowercase();
    if head.starts_with("<!doctype html") || head.starts_with("<html") {
        let (page, original) = files::run_blocking(move || {
            let page = html::html_to_markdown(&text, &html::HtmlOptions::default());
            Ok((page, text))
        })
        .await?;
        if !page.markdown.trim().is_empty() {
            return Ok(Extracted {
                title: page.title,
                content: page.markdown,
                identified_type: "text".to_string(),
                url: None,
                file_path: None,
                metadata: json!({ "mime_type": "text/html", "converted_from": "html" }),
            });
        }
        return Ok(plain_text(original));
    }
    Ok(plain_text(text))
}

fn plain_text(text: String) -> Extracted {
    Extracted {
        title: None,
        content: text,
        identified_type: "text".to_string(),
        url: None,
        file_path: None,
        metadata: json!({ "mime_type": "text/plain" }),
    }
}

async fn extract_url(raw: &str, ctx: &ExtractContext) -> Result<Extracted, ContentError> {
    let url = raw.trim();
    if url.is_empty() {
        return Err(ContentError::InvalidInput("the URL is empty".to_string()));
    }
    // Validate before any engine sees the URL, so a blocked address is never
    // forwarded to a third-party fetcher either.
    validate_url_with(url, &ctx.url_policy, ctx.resolver.as_ref()).await?;
    let parsed = url::Url::parse(url).map_err(|e| ContentError::InvalidInput(format!("malformed URL: {e}")))?;

    let mut extracted = if is_youtube_url(url) && youtube_video_id(url).is_some() {
        youtube::extract_youtube(url, ctx).await?
    } else {
        web::extract_web(&parsed, ctx).await?
    };
    extracted.url = Some(url.to_string());
    extracted.file_path = None;
    Ok(extracted)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn pasted_text_passes_through_unchanged() {
        let text = "  Some pasted notes.\n\n# Not parsed\n<b>kept</b>  ";
        let out = extract_content(ExtractInput::text(text), &ExtractContext::default()).await.unwrap();
        assert_eq!(out.content, text);
        assert_eq!(out.title, None);
        assert_eq!(out.identified_type, "text");
        assert_eq!(out.url, None);
        assert_eq!(out.file_path, None);
    }

    #[tokio::test]
    async fn pasted_html_document_is_converted() {
        let html = "<!DOCTYPE html><html><head><title>Pasted Page</title></head><body><h1>Heading</h1><p>Body <a href=\"https://example.com/x\">link</a>.</p></body></html>";
        let out = extract_content(ExtractInput::text(html), &ExtractContext::default()).await.unwrap();
        assert_eq!(out.title.as_deref(), Some("Pasted Page"));
        assert_eq!(out.content, "# Heading\n\nBody [link](https://example.com/x).");
        assert_eq!(out.identified_type, "text");
    }

    #[tokio::test]
    async fn empty_text_is_an_error() {
        for text in ["", "   \n\t "] {
            let error = extract_content(ExtractInput::text(text), &ExtractContext::default()).await.unwrap_err();
            assert!(matches!(error, ContentError::EmptyContent(_)), "{error:?}");
        }
    }

    #[tokio::test]
    async fn invalid_urls_are_rejected_before_any_request() {
        let ctx = ExtractContext::default();
        let error = extract_content(ExtractInput::url("ftp://example.com/file"), &ctx).await.unwrap_err();
        assert!(matches!(error, ContentError::UrlNotAllowed(_)), "{error:?}");
        let error = extract_content(ExtractInput::url("http://169.254.169.254/latest"), &ctx).await.unwrap_err();
        assert!(matches!(error, ContentError::UrlNotAllowed(_)), "{error:?}");
        let error = extract_content(ExtractInput::url("   "), &ctx).await.unwrap_err();
        assert!(matches!(error, ContentError::InvalidInput(_)), "{error:?}");
    }

    #[test]
    fn url_engine_parses_setting_strings() {
        for (text, engine) in [
            ("auto", UrlEngine::Auto),
            ("Simple", UrlEngine::Simple),
            (" jina ", UrlEngine::Jina),
            ("firecrawl", UrlEngine::Firecrawl),
            ("crawl4ai", UrlEngine::Crawl4ai),
        ] {
            assert_eq!(text.parse::<UrlEngine>().unwrap(), engine);
            assert_eq!(engine.to_string().parse::<UrlEngine>().unwrap(), engine);
        }
        assert!("docling".parse::<UrlEngine>().is_err());
        assert_eq!(serde_json::to_string(&UrlEngine::Crawl4ai).unwrap(), "\"crawl4ai\"");
        assert_eq!(serde_json::from_str::<UrlEngine>("\"firecrawl\"").unwrap(), UrlEngine::Firecrawl);
    }

    #[test]
    fn debug_output_does_not_leak_secrets() {
        let ctx = ExtractContext {
            jina_api_key: Some("jina-secret".into()),
            firecrawl_api_key: Some("fc-secret".into()),
            crawl4ai_api_token: Some("c4-secret".into()),
            proxy: Some("http://user:proxy-secret@proxy:3128".into()),
            youtube_proxy: Some("http://user:yt-secret@proxy:3128".into()),
            ..ExtractContext::default()
        };
        let debug = format!("{ctx:?}");
        assert!(!debug.contains("secret"), "{debug}");
        assert!(debug.contains("<set>"));
    }

    #[test]
    fn input_builders() {
        let input = ExtractInput::file("/tmp/abc123").with_file_name("report.pdf").with_mime_type("application/pdf");
        assert_eq!(
            input,
            ExtractInput::File {
                path: PathBuf::from("/tmp/abc123"),
                file_name: Some("report.pdf".into()),
                mime_type: Some("application/pdf".into()),
            }
        );
        assert_eq!(ExtractInput::url("https://e.com").with_file_name("x"), ExtractInput::Url("https://e.com".into()));
    }
}
