//! Content handling for Nova Notes: turning URLs, files and pasted text into
//! clean Markdown or text, plus the text utilities the rest of the app needs.
//!
//! | Module | Port of |
//! |---|---|
//! | [`extract`] | the `content-core` calls in `open_notebook/graphs/source.py` |
//! | [`chunking`] | `open_notebook/utils/chunking.py` |
//! | [`text`] | `open_notebook/utils/text_utils.py` |
//! | [`tokens`] | `open_notebook/utils/token_utils.py` |
//! | [`url_validation`] | `open_notebook/utils/url_validation.py` |
//! | [`web_search`] | `open_notebook/utils/web_search.py` |
//!
//! Nothing here touches a database or reads credentials from the
//! environment: the caller passes engine settings in through
//! [`ExtractContext`].

pub mod chunking;
pub mod error;
pub mod extract;
pub mod text;
pub mod tokens;
pub mod url_validation;
pub mod web_search;

pub use chunking::{chunk_text, chunk_text_with, detect_content_type, ChunkConfig, ContentType};
pub use error::{is_youtube_url, ContentError, EMPTY_CONTENT_MESSAGE, YOUTUBE_NO_TRANSCRIPT_MESSAGE};
pub use extract::{
    check_file_support, extract_content, AudioOptions, ExtractContext, ExtractInput, Extracted, FfmpegLocation,
    FileSupport, Limits, UrlEngine,
};
pub use text::{clean_thinking_content, parse_thinking_content};
pub use tokens::token_count;
pub use url_validation::{validate_url, UrlPolicy, UrlValidationError};
