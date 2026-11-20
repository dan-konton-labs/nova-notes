//! Extraction errors.
//!
//! The variants mirror the typed errors of the `content-core` library the
//! Python original used, because the server maps each one to a fixed,
//! user-facing sentence ([`ContentError::user_message`]). The `Display` text
//! of a variant is the *internal* detail: it may contain local paths, proxy
//! addresses or provider messages and belongs in the log, not in an API
//! response.

use nn_llm::AiError;

use crate::url_validation::UrlValidationError;

/// Message shown when a YouTube video has no usable transcript.
pub const YOUTUBE_NO_TRANSCRIPT_MESSAGE: &str = "Could not extract content from this YouTube video. \
No transcript or subtitles are available. \
Try configuring a Speech-to-Text model in Settings \
to transcribe the audio instead.";

/// Message shown when extraction succeeded but produced no text.
pub const EMPTY_CONTENT_MESSAGE: &str = "Could not extract any text content from this source. \
The content may be empty, inaccessible, or in an unsupported format.";

#[derive(Debug, thiserror::Error)]
pub enum ContentError {
    /// The URL, path or text handed in is malformed.
    #[error("invalid input: {0}")]
    InvalidInput(String),
    /// The URL (or a redirect target) is rejected by the SSRF guard.
    #[error("{0}")]
    UrlNotAllowed(#[from] UrlValidationError),
    /// The remote resource does not exist (HTTP 404 / 410, removed video).
    #[error("not found: {0}")]
    NotFound(String),
    /// Connection, DNS or timeout failure while fetching.
    #[error("network error: {0}")]
    Network(String),
    /// The response body is larger than `Limits::max_download_bytes`.
    #[error("content exceeds the download limit of {limit} bytes")]
    TooLarge { limit: u64 },
    /// The input type is recognised but cannot be extracted (images, legacy
    /// binary Office files, archives, unknown binary data).
    #[error("unsupported type '{identified_type}': {reason}")]
    UnsupportedType { identified_type: String, reason: String },
    /// The file is missing, unreadable or corrupt.
    #[error("file error: {0}")]
    FileOperation(String),
    /// Something the input needs is not configured: no speech-to-text model,
    /// no ffmpeg, an engine selected without its API key or base URL.
    #[error("configuration error: {0}")]
    Configuration(String),
    /// A remote extraction service (Jina, Firecrawl, Crawl4AI, YouTube, the
    /// origin server) answered with an error.
    #[error("external service error: {0}")]
    ExternalService(String),
    /// The speech-to-text model failed.
    #[error("speech-to-text failed: {0}")]
    Transcription(#[source] AiError),
    /// The YouTube video exists but has no caption track.
    #[error("no transcript found: {0}")]
    NoTranscriptFound(String),
    /// Extraction ran but the source contains no text.
    #[error("no text content: {0}")]
    EmptyContent(String),
}

impl ContentError {
    /// The fixed, safe-to-display sentence for this failure.
    ///
    /// Port of `_extraction_error` in `open_notebook/graphs/source.py`. `url`
    /// is the source URL when the input was one; it only influences the
    /// wording for YouTube links.
    pub fn user_message(&self, url: Option<&str>) -> String {
        let youtube = url.is_some_and(is_youtube_url);
        match self {
            ContentError::NoTranscriptFound(_) => YOUTUBE_NO_TRANSCRIPT_MESSAGE.to_string(),
            ContentError::EmptyContent(_) if youtube => YOUTUBE_NO_TRANSCRIPT_MESSAGE.to_string(),
            ContentError::EmptyContent(_) => EMPTY_CONTENT_MESSAGE.to_string(),
            ContentError::ExternalService(detail)
                if youtube && detail.to_lowercase().contains("youtube transcript") =>
            {
                "YouTube blocked or failed the transcript request. If this keeps \
                 happening, configure a YouTube proxy (a residential proxy) or a \
                 YouTube cookies file for the worker."
                    .to_string()
            }
            ContentError::NotFound(_) => {
                "The page was not found (it may have been removed or moved). Check the URL.".to_string()
            }
            ContentError::Network(_) => "Could not reach this address (connection, timeout or DNS error). \
                 Check the URL and try again."
                .to_string(),
            ContentError::InvalidInput(_) => "This URL or input is not valid.".to_string(),
            // The guard's own wording is written for end users and names no
            // internal detail beyond the host the user typed.
            ContentError::UrlNotAllowed(e) => e.to_string(),
            ContentError::TooLarge { .. } => {
                "This content is larger than the configured download limit.".to_string()
            }
            ContentError::UnsupportedType { .. } => "This file type is not supported.".to_string(),
            ContentError::FileOperation(_) => "The file could not be read. It may be corrupted or in an \
                 unsupported format."
                .to_string(),
            ContentError::Configuration(_) => "Content extraction is not configured correctly. Check the content \
                 processing engine and speech-to-text settings; the worker log has \
                 the details."
                .to_string(),
            ContentError::ExternalService(_) | ContentError::Transcription(_) => {
                "The content extraction service failed.".to_string()
            }
        }
    }
}

/// Whether `url` points at YouTube, judged by host name rather than substring.
pub fn is_youtube_url(url: &str) -> bool {
    let Ok(parsed) = url::Url::parse(url.trim()) else {
        return false;
    };
    let host = parsed.host_str().unwrap_or("").to_ascii_lowercase();
    host == "youtube.com"
        || host == "youtu.be"
        || host.ends_with(".youtube.com")
        || host == "youtube-nocookie.com"
        || host.ends_with(".youtube-nocookie.com")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn youtube_urls_match_by_hostname() {
        for (url, expected) in [
            ("https://www.youtube.com/watch?v=abc", true),
            ("https://youtube.com/shorts/abc", true),
            ("https://m.youtube.com/watch?v=abc", true),
            ("https://youtu.be/abc", true),
            ("https://notyoutube.com/watch?v=abc", false),
            ("https://example.com/youtube.com/abc", false),
            ("", false),
        ] {
            assert_eq!(is_youtube_url(url), expected, "{url}");
        }
    }

    #[test]
    fn typed_errors_become_fixed_user_messages() {
        let yt = Some("https://www.youtube.com/watch?v=abc123");
        let page = Some("https://example.com/article");
        let cases: Vec<(ContentError, Option<&str>, &str)> = vec![
            (ContentError::NoTranscriptFound("none".into()), yt, "Speech-to-Text model"),
            (
                ContentError::ExternalService(
                    "YouTube transcript extraction failed for video abc123: IpBlocked".into(),
                ),
                yt,
                "YouTube proxy",
            ),
            (ContentError::ExternalService("STT provider failed".into()), yt, "service failed"),
            (ContentError::NotFound("HTTP 404".into()), page, "page was not found"),
            (ContentError::Network("DNS failure".into()), page, "Could not reach"),
            (ContentError::InvalidInput("malformed".into()), page, "not valid"),
            (
                ContentError::UnsupportedType { identified_type: "image/png".into(), reason: "x".into() },
                None,
                "not supported",
            ),
            (ContentError::FileOperation("corrupted PDF".into()), None, "could not be read"),
            (ContentError::Configuration("no key".into()), page, "not configured correctly"),
            (ContentError::ExternalService("firecrawl 500".into()), page, "service failed"),
            (ContentError::EmptyContent("empty".into()), yt, "Speech-to-Text model"),
            (ContentError::EmptyContent("empty".into()), page, "Could not extract any text"),
        ];
        for (error, url, expected) in cases {
            let message = error.user_message(url);
            assert!(message.contains(expected), "{error:?} -> {message}");
        }
    }

    #[test]
    fn raw_detail_stays_out_of_the_user_message() {
        let secret = "proxy http://user:hunter2@10.0.0.5:3128 failed for /app/data/x.pdf";
        let error = ContentError::Network(secret.into());
        let message = error.user_message(Some("https://example.com/article"));
        assert!(!message.contains("hunter2"));
        assert!(!message.contains("/app/data"));
        assert!(error.to_string().contains(secret));
    }

    #[test]
    fn blocked_non_youtube_url_gets_generic_service_message() {
        let error = ContentError::ExternalService("YouTube transcript blocked".into());
        let message = error.user_message(Some("https://notyoutube.com/watch?v=abc"));
        assert!(message.contains("service failed"));
    }
}
