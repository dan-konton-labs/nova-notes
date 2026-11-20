//! Content-type detection and token-based chunking for embeddings.
//!
//! Port of `open_notebook/utils/chunking.py`, including the behaviour of the
//! three LangChain splitters it delegated to:
//!
//! * plain text: `RecursiveCharacterTextSplitter` with separators
//!   `"\n\n"`, `"\n"`, `". "`, `", "`, `" "`, `""`, sizes measured in
//!   `o200k_base` tokens ([`crate::tokens::token_count`]);
//! * Markdown: `MarkdownHeaderTextSplitter` on `#`, `##`, `###` with headers
//!   kept in the chunk text;
//! * HTML: `HTMLHeaderTextSplitter` on `h1`, `h2`, `h3`.
//!
//! Header-split sections that are still above the chunk size get a secondary
//! plain-text split, and fragments below the minimum size are dropped.
//!
//! The Python module read its sizes from the environment at import time. Here
//! the caller owns the environment: pass the raw values of
//! `OPEN_NOTEBOOK_CHUNK_SIZE`, `OPEN_NOTEBOOK_CHUNK_OVERLAP` and
//! `OPEN_NOTEBOOK_MIN_CHUNK_SIZE` to [`ChunkConfig::from_settings`].

use std::sync::OnceLock;

use regex::Regex;
use scraper::{ElementRef, Html, Node, Selector};

use crate::text::is_printable;
use crate::tokens::token_count;

/// Default maximum chunk size in tokens.
pub const DEFAULT_CHUNK_SIZE: usize = 400;
/// Default minimum chunk size in tokens.
pub const DEFAULT_MIN_CHUNK_SIZE: usize = 5;
/// Heuristic confidence needed to override a plain-text file extension.
pub const HIGH_CONFIDENCE_THRESHOLD: f64 = 0.8;

const SEPARATORS: [&str; 6] = ["\n\n", "\n", ". ", ", ", " ", ""];

/// Chunk sizes, all in tokens.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ChunkConfig {
    /// Maximum chunk size.
    pub chunk_size: usize,
    /// Overlap between neighbouring chunks of a plain-text split.
    pub chunk_overlap: usize,
    /// Chunks below this are dropped (0 disables the filter).
    pub min_chunk_size: usize,
}

impl Default for ChunkConfig {
    fn default() -> Self {
        Self {
            chunk_size: DEFAULT_CHUNK_SIZE,
            chunk_overlap: default_overlap(DEFAULT_CHUNK_SIZE),
            min_chunk_size: DEFAULT_MIN_CHUNK_SIZE,
        }
    }
}

fn default_overlap(chunk_size: usize) -> usize {
    (chunk_size as f64 * 0.15) as usize
}

impl ChunkConfig {
    /// Build a configuration from the raw setting strings, applying the same
    /// validation as the original: chunk size has a floor of 100 and defaults
    /// to 400; overlap defaults to 15% of the chunk size, is clamped to 0 when
    /// negative and reset to 15% when it is not smaller than the chunk size;
    /// the minimum defaults to 5 and is clamped to 0 when negative. Values
    /// that do not parse as integers fall back to the default.
    pub fn from_settings(chunk_size: Option<&str>, chunk_overlap: Option<&str>, min_chunk_size: Option<&str>) -> Self {
        let size = match chunk_size.filter(|s| !s.is_empty()).map(parse_int) {
            Some(Some(value)) if value < 100 => {
                tracing::warn!("chunk size ({value}) is too small, using the minimum of 100");
                100
            }
            Some(Some(value)) => {
                if value > 8192 {
                    tracing::warn!("chunk size ({value}) is very large; some embedding models may reject it");
                }
                value as usize
            }
            Some(None) => {
                tracing::warn!("invalid chunk size setting, using the default of {DEFAULT_CHUNK_SIZE}");
                DEFAULT_CHUNK_SIZE
            }
            None => DEFAULT_CHUNK_SIZE,
        };

        let overlap = match chunk_overlap.filter(|s| !s.is_empty()).map(parse_int) {
            Some(Some(value)) if value < 0 => {
                tracing::warn!("chunk overlap ({value}) cannot be negative, using 0");
                0
            }
            Some(Some(value)) if value as usize >= size => {
                tracing::warn!("chunk overlap ({value}) must be smaller than the chunk size ({size}), using 15%");
                default_overlap(size)
            }
            Some(Some(value)) => value as usize,
            Some(None) => {
                tracing::warn!("invalid chunk overlap setting, using 15% of the chunk size");
                default_overlap(size)
            }
            None => default_overlap(size),
        };

        let min = match min_chunk_size.map(parse_int) {
            Some(Some(value)) if value < 0 => {
                tracing::warn!("minimum chunk size ({value}) cannot be negative, using 0");
                0
            }
            Some(Some(value)) => value as usize,
            Some(None) => {
                tracing::warn!("invalid minimum chunk size setting, using the default of {DEFAULT_MIN_CHUNK_SIZE}");
                DEFAULT_MIN_CHUNK_SIZE
            }
            None => DEFAULT_MIN_CHUNK_SIZE,
        };

        Self { chunk_size: size, chunk_overlap: overlap, min_chunk_size: min }
    }
}

fn parse_int(raw: &str) -> Option<i64> {
    raw.trim().parse::<i64>().ok()
}

/// Content type used to pick a chunking strategy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ContentType {
    Html,
    Markdown,
    Plain,
}

impl ContentType {
    /// `html`, `markdown` or `plain`.
    pub fn as_str(&self) -> &'static str {
        match self {
            ContentType::Html => "html",
            ContentType::Markdown => "markdown",
            ContentType::Plain => "plain",
        }
    }
}

/// Content type implied by the extension of `file_path`, if it is a known one.
pub fn detect_content_type_from_extension(file_path: Option<&str>) -> Option<ContentType> {
    let path = file_path?;
    let name = path.trim_end_matches('/').rsplit('/').next()?;
    let dot = name.rfind('.')?;
    if dot == 0 || dot + 1 == name.len() {
        return None;
    }
    match name[dot..].to_lowercase().as_str() {
        ".html" | ".htm" | ".xhtml" => Some(ContentType::Html),
        ".md" | ".markdown" | ".mdown" | ".mkd" => Some(ContentType::Markdown),
        ".txt" | ".text" | ".py" | ".js" | ".ts" | ".java" | ".c" | ".cpp" | ".go" | ".rs" | ".rb" | ".php"
        | ".sh" | ".bash" | ".zsh" | ".sql" | ".json" | ".yaml" | ".yml" | ".xml" | ".csv" | ".tsv" => {
            Some(ContentType::Plain)
        }
        _ => None,
    }
}

fn re(cell: &'static OnceLock<Regex>, pattern: &str) -> &'static Regex {
    cell.get_or_init(|| Regex::new(pattern).expect("valid regex"))
}

fn html_score(text: &str) -> f64 {
    static DOCTYPE: OnceLock<Regex> = OnceLock::new();
    static HTML_TAG: OnceLock<Regex> = OnceLock::new();
    static HEADING: OnceLock<Regex> = OnceLock::new();
    static CLOSING: OnceLock<Regex> = OnceLock::new();

    let mut score = 0.0_f64;
    let mut indicators = 0;

    if re(&DOCTYPE, r"(?i)<!DOCTYPE\s+html").is_match(text) {
        score += 0.4;
        indicators += 1;
    }
    if re(&HTML_TAG, r"(?i)<html[\s>]").is_match(text) {
        score += 0.3;
        indicators += 1;
    }

    let lowered = text.to_lowercase();
    for tag in ["<head", "<body", "<div", "<span", "<p>", "<table", "<form"] {
        if lowered.contains(tag) {
            score += 0.1;
            indicators += 1;
            if indicators >= 5 {
                break;
            }
        }
    }

    if re(&HEADING, r"(?i)<h[1-6][\s>]").is_match(text) {
        score += 0.15;
    }
    if re(&CLOSING, r"</\w+>").is_match(text) {
        score += 0.1;
    }

    score.min(1.0)
}

fn markdown_score(text: &str) -> f64 {
    static HEADERS: OnceLock<Regex> = OnceLock::new();
    static LINKS: OnceLock<Regex> = OnceLock::new();
    static FENCE: OnceLock<Regex> = OnceLock::new();
    static INLINE_CODE: OnceLock<Regex> = OnceLock::new();
    static BULLETS: OnceLock<Regex> = OnceLock::new();
    static NUMBERED: OnceLock<Regex> = OnceLock::new();
    static EMPHASIS: OnceLock<Regex> = OnceLock::new();
    static QUOTE: OnceLock<Regex> = OnceLock::new();

    let mut score = 0.0_f64;

    let headers = re(&HEADERS, r"(?m)^#{1,6}\s+.+").find_iter(text).count();
    if headers >= 3 {
        score += 0.35;
    } else if headers >= 1 {
        score += 0.2;
    }

    let links = re(&LINKS, r"\[.+?\]\(.+?\)").find_iter(text).count();
    if links >= 2 {
        score += 0.25;
    } else if links >= 1 {
        score += 0.15;
    }

    if re(&FENCE, r"(?m)^```").is_match(text) {
        score += 0.2;
    }
    if re(&INLINE_CODE, r"`[^`]+`").is_match(text) {
        score += 0.1;
    }

    let list_items = re(&BULLETS, r"(?m)^[\*\-\+]\s+").find_iter(text).count()
        + re(&NUMBERED, r"(?m)^\d+\.\s+").find_iter(text).count();
    if list_items >= 3 {
        score += 0.15;
    } else if list_items >= 1 {
        score += 0.08;
    }

    if re(&EMPHASIS, r"\*\*.+?\*\*|__.+?__").is_match(text) {
        score += 0.1;
    }
    if re(&QUOTE, r"(?m)^>\s+").is_match(text) {
        score += 0.1;
    }

    score.min(1.0)
}

/// Guess the content type from the text itself. Returns the type and a
/// confidence between 0 and 1. Only the first 5000 characters are examined.
pub fn detect_content_type_from_heuristics(text: &str) -> (ContentType, f64) {
    if text.chars().take(10).count() < 10 {
        return (ContentType::Plain, 0.5);
    }
    let sample_end = text.char_indices().nth(5000).map_or(text.len(), |(i, _)| i);
    let sample = &text[..sample_end];

    let html = html_score(sample);
    if html >= HIGH_CONFIDENCE_THRESHOLD {
        return (ContentType::Html, html);
    }
    let markdown = markdown_score(sample);
    if markdown >= HIGH_CONFIDENCE_THRESHOLD {
        return (ContentType::Markdown, markdown);
    }

    if html > markdown && html > 0.3 {
        (ContentType::Html, html)
    } else if markdown > 0.3 {
        (ContentType::Markdown, markdown)
    } else {
        (ContentType::Plain, 0.6)
    }
}

/// Detect the content type: a recognised file extension wins, except that a
/// plain-text extension is overridden by heuristics at
/// [`HIGH_CONFIDENCE_THRESHOLD`] or above. Without a recognised extension the
/// heuristics decide.
pub fn detect_content_type(text: &str, file_path: Option<&str>) -> ContentType {
    let extension_type = detect_content_type_from_extension(file_path);
    let (heuristic_type, confidence) = detect_content_type_from_heuristics(text);
    match extension_type {
        None => heuristic_type,
        Some(ContentType::Plain) if confidence >= HIGH_CONFIDENCE_THRESHOLD => heuristic_type,
        Some(extension_type) => extension_type,
    }
}

/// Split `text` into chunks of at most (approximately) `DEFAULT_CHUNK_SIZE`
/// tokens using the default [`ChunkConfig`]. See [`chunk_text_with`].
pub fn chunk_text(text: &str, content_type: Option<ContentType>, file_path: Option<&str>) -> Vec<String> {
    chunk_text_with(text, content_type, file_path, &ChunkConfig::default())
}

/// Split `text` into chunks using the splitter that fits its content type.
///
/// Empty or whitespace-only text gives no chunks; text within the chunk size
/// is returned as a single, unmodified chunk. `content_type` is detected from
/// `file_path` and the text when not given.
pub fn chunk_text_with(
    text: &str,
    content_type: Option<ContentType>,
    file_path: Option<&str>,
    config: &ChunkConfig,
) -> Vec<String> {
    if text.trim().is_empty() {
        return Vec::new();
    }
    if token_count(text) <= config.chunk_size {
        return vec![text.to_string()];
    }

    let content_type = content_type.unwrap_or_else(|| detect_content_type(text, file_path));
    let chunks = match content_type {
        ContentType::Html => secondary_split(split_html_by_headers(text), config),
        ContentType::Markdown => secondary_split(split_markdown_by_headers(text), config),
        ContentType::Plain => split_recursive(text, config),
    };

    let mut chunks: Vec<String> =
        chunks.iter().map(|c| c.trim()).filter(|c| !c.is_empty()).map(str::to_string).collect();

    // Never let the minimum-size filter empty the result.
    if config.min_chunk_size > 0 && chunks.len() > 1 {
        let keep: Vec<bool> = chunks.iter().map(|c| token_count(c) >= config.min_chunk_size).collect();
        if keep.iter().any(|&k| k) {
            let mut flags = keep.into_iter();
            chunks.retain(|_| flags.next().unwrap_or(true));
        }
    }
    chunks
}

fn secondary_split(chunks: Vec<String>, config: &ChunkConfig) -> Vec<String> {
    let mut result = Vec::with_capacity(chunks.len());
    for chunk in chunks {
        if token_count(&chunk) > config.chunk_size {
            result.extend(split_recursive(&chunk, config));
        } else {
            result.push(chunk);
        }
    }
    result
}

// --- RecursiveCharacterTextSplitter ----------------------------------------

/// Recursive separator-based split measured in tokens.
fn split_recursive(text: &str, config: &ChunkConfig) -> Vec<String> {
    split_recursive_with(text, &SEPARATORS, config)
}

fn split_recursive_with(text: &str, separators: &[&str], config: &ChunkConfig) -> Vec<String> {
    let mut separator = separators.last().copied().unwrap_or("");
    let mut remaining: &[&str] = &[];
    for (i, candidate) in separators.iter().enumerate() {
        if candidate.is_empty() {
            separator = candidate;
            break;
        }
        if text.contains(candidate) {
            separator = candidate;
            remaining = &separators[i + 1..];
            break;
        }
    }

    let mut final_chunks = Vec::new();
    let mut good: Vec<(String, usize)> = Vec::new();
    for piece in split_keeping_separator(text, separator) {
        let length = token_count(&piece);
        if length < config.chunk_size {
            good.push((piece, length));
            continue;
        }
        if !good.is_empty() {
            final_chunks.extend(merge_splits(std::mem::take(&mut good), config));
        }
        if remaining.is_empty() {
            final_chunks.push(piece);
        } else {
            final_chunks.extend(split_recursive_with(&piece, remaining, config));
        }
    }
    if !good.is_empty() {
        final_chunks.extend(merge_splits(good, config));
    }
    final_chunks
}

/// Split on `separator`, leaving the separator at the start of the piece that
/// follows it. An empty separator splits into characters.
fn split_keeping_separator(text: &str, separator: &str) -> Vec<String> {
    if separator.is_empty() {
        return text.chars().map(String::from).collect();
    }
    let mut pieces = Vec::new();
    for (i, part) in text.split(separator).enumerate() {
        let piece = if i == 0 { part.to_string() } else { format!("{separator}{part}") };
        if !piece.is_empty() {
            pieces.push(piece);
        }
    }
    pieces
}

/// Greedily pack consecutive pieces into chunks of at most `chunk_size`
/// tokens, carrying up to `chunk_overlap` tokens of trailing pieces into the
/// next chunk.
fn merge_splits(splits: Vec<(String, usize)>, config: &ChunkConfig) -> Vec<String> {
    let mut docs = Vec::new();
    let mut current: std::collections::VecDeque<(String, usize)> = std::collections::VecDeque::new();
    let mut total = 0usize;

    for (piece, length) in splits {
        if total + length > config.chunk_size && !current.is_empty() {
            if let Some(doc) = join_pieces(&current) {
                docs.push(doc);
            }
            while total > config.chunk_overlap || (total + length > config.chunk_size && total > 0) {
                match current.pop_front() {
                    Some((_, dropped)) => total -= dropped,
                    None => break,
                }
            }
        }
        total += length;
        current.push_back((piece, length));
    }
    if let Some(doc) = join_pieces(&current) {
        docs.push(doc);
    }
    docs
}

fn join_pieces(pieces: &std::collections::VecDeque<(String, usize)>) -> Option<String> {
    let joined: String = pieces.iter().map(|(piece, _)| piece.as_str()).collect();
    let trimmed = joined.trim();
    (!trimmed.is_empty()).then(|| trimmed.to_string())
}

// --- MarkdownHeaderTextSplitter --------------------------------------------

/// Header text for levels 1 to 3; `None` when no header of that level is open.
type HeaderPath = [Option<String>; 3];

fn header_depth(path: &HeaderPath) -> usize {
    path.iter().filter(|h| h.is_some()).count()
}

/// Split Markdown into sections at `#`, `##` and `###` headers. Header lines
/// stay in the text, lines are stripped, fenced code blocks are never split,
/// and paragraphs within a section are joined with a Markdown hard break
/// (`"  \n"`).
fn split_markdown_by_headers(text: &str) -> Vec<String> {
    let mut sections: Vec<(String, HeaderPath)> = Vec::new();
    let mut current: Vec<String> = Vec::new();
    let mut current_path: HeaderPath = Default::default();
    let mut open_path: HeaderPath = Default::default();
    let mut fence: Option<&'static str> = None;

    for raw in text.split('\n') {
        let line: String = raw.trim().chars().filter(|&c| is_printable(c)).collect();

        match fence {
            None => {
                if line.starts_with("```") && line.matches("```").count() == 1 {
                    fence = Some("```");
                } else if line.starts_with("~~~") {
                    fence = Some("~~~");
                }
            }
            Some(open) => {
                if line.starts_with(open) {
                    fence = None;
                }
            }
        }
        if fence.is_some() {
            current.push(line);
            continue;
        }

        let header_level = (1..=3usize).rev().find(|&level| {
            let rest = line.strip_prefix(&"#".repeat(level));
            matches!(rest, Some(rest) if rest.is_empty() || rest.starts_with(' '))
        });

        if let Some(level) = header_level {
            for deeper in open_path.iter_mut().skip(level - 1) {
                *deeper = None;
            }
            open_path[level - 1] = Some(line[level..].trim().to_string());

            if !current.is_empty() {
                sections.push((current.join("\n"), current_path.clone()));
                current.clear();
            }
            current.push(line);
        } else if !line.is_empty() {
            current.push(line);
        } else if !current.is_empty() {
            sections.push((current.join("\n"), current_path.clone()));
            current.clear();
        }

        current_path = open_path.clone();
    }
    if !current.is_empty() {
        sections.push((current.join("\n"), current_path));
    }

    let mut aggregated: Vec<(String, HeaderPath)> = Vec::new();
    for (content, path) in sections {
        if let Some((last_content, last_path)) = aggregated.last_mut() {
            let same_section = *last_path == path;
            let last_line_is_header = last_content.rsplit('\n').next().is_some_and(|l| l.starts_with('#'));
            // A header directly followed by a deeper header stays with it.
            let nested_under_header = header_depth(last_path) < header_depth(&path) && last_line_is_header;
            if same_section || nested_under_header {
                last_content.push_str("  \n");
                last_content.push_str(&content);
                *last_path = path;
                continue;
            }
        }
        aggregated.push((content, path));
    }
    aggregated.into_iter().map(|(content, _)| content).collect()
}

// --- HTMLHeaderTextSplitter ------------------------------------------------

/// Split HTML at `h1`, `h2` and `h3`. Each header becomes its own chunk and
/// the text of the elements between two headers is joined with `"  \n"`.
/// Only text directly inside an element counts for that element, visited in
/// document order. Comments are ignored.
fn split_html_by_headers(html: &str) -> Vec<String> {
    let document = Html::parse_document(html);
    let body_selector = Selector::parse("body").expect("valid selector");
    let root = document.select(&body_selector).next().unwrap_or_else(|| document.root_element());

    let mut chunks = Vec::new();
    let mut pending: Vec<String> = Vec::new();
    let flush = |pending: &mut Vec<String>, chunks: &mut Vec<String>| {
        if !pending.is_empty() {
            chunks.push(pending.join("  \n"));
            pending.clear();
        }
    };

    for element in root.descendants().filter_map(ElementRef::wrap) {
        let own_text: Vec<&str> = element
            .children()
            .filter_map(|child| match child.value() {
                Node::Text(text) => Some(text.trim()),
                _ => None,
            })
            .filter(|t| !t.is_empty())
            .collect();
        if own_text.is_empty() {
            continue;
        }
        let node_text = own_text.join(" ");

        if matches!(element.value().name(), "h1" | "h2" | "h3") {
            flush(&mut pending, &mut chunks);
            chunks.push(node_text);
        } else {
            pending.push(node_text);
        }
    }
    flush(&mut pending, &mut chunks);
    chunks
}

#[cfg(test)]
mod tests {
    use super::*;

    const CHUNK_SIZE: usize = DEFAULT_CHUNK_SIZE;

    fn build_text_with_max_tokens(fragment: &str, max_tokens: usize) -> String {
        let mut text = String::new();
        loop {
            let candidate = format!("{text}{fragment}");
            if token_count(&candidate) > max_tokens {
                return text;
            }
            text = candidate;
        }
    }

    fn build_text_exceeding_tokens(fragment: &str, threshold: usize) -> String {
        let mut text = fragment.to_string();
        while token_count(&text) <= threshold {
            text.push_str(fragment);
        }
        text
    }

    fn assert_chunks_within_token_limit(chunks: &[String]) {
        assert!(!chunks.is_empty());
        for chunk in chunks {
            assert!(token_count(chunk) <= CHUNK_SIZE, "chunk has {} tokens", token_count(chunk));
        }
    }

    // --- extension detection ---

    #[test]
    fn html_extensions() {
        assert_eq!(detect_content_type_from_extension(Some("file.html")), Some(ContentType::Html));
        assert_eq!(detect_content_type_from_extension(Some("file.htm")), Some(ContentType::Html));
        assert_eq!(detect_content_type_from_extension(Some("file.xhtml")), Some(ContentType::Html));
        assert_eq!(detect_content_type_from_extension(Some("/path/to/file.HTML")), Some(ContentType::Html));
    }

    #[test]
    fn markdown_extensions() {
        assert_eq!(detect_content_type_from_extension(Some("file.md")), Some(ContentType::Markdown));
        assert_eq!(detect_content_type_from_extension(Some("file.markdown")), Some(ContentType::Markdown));
        assert_eq!(detect_content_type_from_extension(Some("file.mdown")), Some(ContentType::Markdown));
        assert_eq!(detect_content_type_from_extension(Some("/path/to/README.MD")), Some(ContentType::Markdown));
    }

    #[test]
    fn plain_text_extensions() {
        assert_eq!(detect_content_type_from_extension(Some("file.txt")), Some(ContentType::Plain));
        assert_eq!(detect_content_type_from_extension(Some("file.text")), Some(ContentType::Plain));
    }

    #[test]
    fn code_extensions_are_plain() {
        for name in ["file.py", "file.js", "file.json", "file.yaml"] {
            assert_eq!(detect_content_type_from_extension(Some(name)), Some(ContentType::Plain), "{name}");
        }
    }

    #[test]
    fn unknown_extensions() {
        for name in ["file.xyz", "file.docx", "file.pdf"] {
            assert_eq!(detect_content_type_from_extension(Some(name)), None, "{name}");
        }
    }

    #[test]
    fn no_extension_none_and_empty() {
        assert_eq!(detect_content_type_from_extension(Some("Makefile")), None);
        assert_eq!(detect_content_type_from_extension(Some("README")), None);
        assert_eq!(detect_content_type_from_extension(Some(".bashrc")), None);
        assert_eq!(detect_content_type_from_extension(None), None);
        assert_eq!(detect_content_type_from_extension(Some("")), None);
    }

    // --- heuristics ---

    #[test]
    fn html_detection_doctype() {
        let (kind, confidence) =
            detect_content_type_from_heuristics("<!DOCTYPE html><html><body>Content</body></html>");
        assert_eq!(kind, ContentType::Html);
        assert!(confidence >= 0.8);
    }

    #[test]
    fn html_detection_tags() {
        let (kind, confidence) = detect_content_type_from_heuristics(
            "<html><head><title>Test</title></head><body><div><p>Content</p></div></body></html>",
        );
        assert_eq!(kind, ContentType::Html);
        assert!(confidence >= 0.5);
    }

    #[test]
    fn markdown_detection_headers() {
        let text = "# Main Title\n\n## Section 1\n\nSome content here.\n\n## Section 2\n\nMore content.\n\n### Subsection\n\nDetails here.\n";
        let (kind, confidence) = detect_content_type_from_heuristics(text);
        assert_eq!(kind, ContentType::Markdown);
        assert!(confidence >= 0.3);
    }

    #[test]
    fn markdown_detection_links() {
        let text = "# Documentation\n\nCheck out [this link](https://example.com) and [another one](https://test.com).\n\n## References\n\nHere's some more text with [links](url) and `inline code`.";
        let (kind, confidence) = detect_content_type_from_heuristics(text);
        assert_eq!(kind, ContentType::Markdown);
        assert!(confidence >= 0.4);
    }

    #[test]
    fn markdown_detection_code_blocks() {
        let text = "# Code Example\n\n```python\ndef hello():\n    print(\"Hello, World!\")\n```\n\nSome explanation text.\n";
        let (kind, confidence) = detect_content_type_from_heuristics(text);
        assert_eq!(kind, ContentType::Markdown);
        assert!(confidence >= 0.5);
    }

    #[test]
    fn plain_text_detection() {
        let text = "This is just regular plain text.\nIt has multiple lines but no special formatting.\nNo headers, no links, no HTML tags.\nJust regular sentences and paragraphs.";
        let (kind, confidence) = detect_content_type_from_heuristics(text);
        assert_eq!(kind, ContentType::Plain);
        assert_eq!(confidence, 0.6);
    }

    #[test]
    fn short_and_empty_text_default_to_plain() {
        assert_eq!(detect_content_type_from_heuristics("Hi"), (ContentType::Plain, 0.5));
        assert_eq!(detect_content_type_from_heuristics(""), (ContentType::Plain, 0.5));
    }

    // --- combined detection ---

    #[test]
    fn extension_takes_priority() {
        let kind = detect_content_type("# Header\n\nSome [link](url) content", Some("file.txt"));
        assert_eq!(kind, ContentType::Plain);
    }

    #[test]
    fn no_extension_uses_heuristics() {
        let kind = detect_content_type("<!DOCTYPE html><html><body>Test</body></html>", None);
        assert_eq!(kind, ContentType::Html);
    }

    #[test]
    fn extension_html_and_markdown() {
        assert_eq!(detect_content_type("some text", Some("file.html")), ContentType::Html);
        assert_eq!(detect_content_type("some text", Some("file.md")), ContentType::Markdown);
    }

    #[test]
    fn high_confidence_override() {
        let html =
            "<!DOCTYPE html><html><head><title>Test</title></head><body><div><p>Content</p></div></body></html>";
        assert_eq!(detect_content_type(html, Some("file.txt")), ContentType::Html);
    }

    // --- chunking ---

    #[test]
    fn empty_text() {
        assert!(chunk_text("", None, None).is_empty());
        assert!(chunk_text("   ", None, None).is_empty());
    }

    #[test]
    fn short_text_is_not_chunked() {
        let text = "This is a short text.";
        assert_eq!(chunk_text(text, None, None), vec![text.to_string()]);
    }

    #[test]
    fn text_at_chunk_limit() {
        let text = build_text_with_max_tokens("This is a sentence. ", CHUNK_SIZE);
        assert!(token_count(&text) <= CHUNK_SIZE);
        assert_eq!(chunk_text(&text, None, None).len(), 1);
    }

    #[test]
    fn long_text_is_chunked() {
        let text = build_text_exceeding_tokens("This is a sentence. ", CHUNK_SIZE);
        let chunks = chunk_text(&text, None, None);
        assert!(chunks.len() > 1);
        assert_chunks_within_token_limit(&chunks);
    }

    #[test]
    fn cjk_text_is_chunked_by_tokens() {
        let text = build_text_exceeding_tokens("這是一段中文內容，用來驗證分塊邏輯。", CHUNK_SIZE);
        let chunks = chunk_text(&text, Some(ContentType::Plain), None);
        assert!(chunks.len() > 1);
        assert_chunks_within_token_limit(&chunks);
    }

    #[test]
    fn mixed_language_text_is_chunked_by_tokens() {
        let fragment = "This paragraph mixes English and 中文內容 to verify token-based chunking. ";
        let text = build_text_exceeding_tokens(fragment, CHUNK_SIZE);
        let chunks = chunk_text(&text, Some(ContentType::Plain), None);
        assert!(chunks.len() > 1);
        assert_chunks_within_token_limit(&chunks);
    }

    #[test]
    fn explicit_content_type_html() {
        let html = "<html>\n<body>\n<h1>Main Title</h1>\n<p>First paragraph with lots of content.</p>\n<h2>Section</h2>\n<p>Second paragraph.</p>\n</body>\n</html>";
        assert!(!chunk_text(html, Some(ContentType::Html), None).is_empty());
    }

    #[test]
    fn explicit_content_type_markdown() {
        let md = "# Main Title\n\nIntroduction paragraph.\n\n## Section 1\n\nContent for section 1.\n\n## Section 2\n\nContent for section 2.\n";
        assert!(!chunk_text(md, Some(ContentType::Markdown), None).is_empty());
    }

    #[test]
    fn explicit_content_type_plain() {
        let text = build_text_exceeding_tokens("Word ", CHUNK_SIZE);
        let chunks = chunk_text(&text, Some(ContentType::Plain), None);
        assert!(chunks.len() > 1);
        assert_chunks_within_token_limit(&chunks);
    }

    #[test]
    fn file_path_detection() {
        assert_eq!(chunk_text("Some content here", None, Some("document.md")).len(), 1);
    }

    #[test]
    fn secondary_chunking_for_large_sections() {
        let large = build_text_exceeding_tokens("這是一段很長的章節內容，用來測試次級分塊。", CHUNK_SIZE);
        let chunks = chunk_text(&format!("# Title\n\n{large}"), Some(ContentType::Markdown), None);
        assert!(chunks.len() > 1);
        assert_chunks_within_token_limit(&chunks);
    }

    #[test]
    fn drops_degenerate_short_chunks() {
        let large =
            build_text_exceeding_tokens("This is a paragraph with enough content to be useful. ", CHUNK_SIZE);
        let chunks = chunk_text(&format!("# Real Title\n\n{large}\n\n# .\n"), Some(ContentType::Markdown), None);
        assert!(!chunks.is_empty());
        assert!(chunks.iter().all(|c| token_count(c) >= DEFAULT_MIN_CHUNK_SIZE));
        assert!(chunks.iter().all(|c| !matches!(c.trim(), "." | "," | ";" | "#")));
    }

    #[test]
    fn filter_never_empties_result() {
        let text = ". ".repeat(200);
        assert!(!chunk_text(&text, Some(ContentType::Plain), None).is_empty());
    }

    // --- splitter behaviour beyond the original test suite ---

    #[test]
    fn plain_chunks_overlap_and_cover_the_text() {
        let sentences: Vec<String> = (0..400).map(|i| format!("Sentence number {i} ends here.")).collect();
        let text = sentences.join(" ");
        let chunks = chunk_text(&text, Some(ContentType::Plain), None);
        assert!(chunks.len() > 2);
        assert_chunks_within_token_limit(&chunks);
        for i in [0, 199, 399] {
            let needle = format!("Sentence number {i} ends here");
            assert!(chunks.iter().any(|c| c.contains(&needle)), "missing {needle}");
        }
        // Consecutive chunks share trailing/leading text (the overlap).
        let tail: String = chunks[0].chars().rev().take(20).collect::<String>().chars().rev().collect();
        assert!(chunks[1].contains(tail.trim()), "no overlap between first two chunks");
    }

    #[test]
    fn paragraph_boundaries_are_preferred() {
        let paragraph = build_text_with_max_tokens("Alpha beta gamma delta. ", 150);
        let text = [paragraph.trim(), paragraph.trim(), paragraph.trim(), paragraph.trim()].join("\n\n");
        let chunks = chunk_text(&text, Some(ContentType::Plain), None);
        assert!(chunks.len() >= 2);
        // Every chunk is a whole number of paragraphs: no paragraph is cut.
        for chunk in &chunks {
            for part in chunk.split("\n\n") {
                assert_eq!(part, paragraph.trim());
            }
        }
    }

    #[test]
    fn markdown_sections_split_at_headers() {
        let body = build_text_with_max_tokens("Words in a section. ", 200);
        let md = format!(
            "# One\n\n{body}\n\n## Two\n\n{body}\n\n```\n# not a header\n```\n\n### Three\n\n{body}\n\n#### Four stays inline\n\n{body}"
        );
        let sections = split_markdown_by_headers(&md);
        assert_eq!(sections.len(), 3);
        assert!(sections[0].starts_with("# One  \n"));
        assert!(sections[1].starts_with("## Two  \n"));
        assert!(sections[1].contains("```\n# not a header\n```"));
        assert!(sections[2].starts_with("### Three  \n"));
        assert!(sections[2].contains("#### Four stays inline"));

        let chunks = chunk_text(&md, Some(ContentType::Markdown), None);
        assert_chunks_within_token_limit(&chunks);
        assert!(chunks.iter().any(|c| c.starts_with("## Two")));
    }

    #[test]
    fn markdown_header_followed_by_subheader_stays_together() {
        let sections = split_markdown_by_headers("# Top\n## Sub\ntext\n\n# Next\nmore");
        assert_eq!(sections, vec!["# Top  \n## Sub\ntext".to_string(), "# Next\nmore".to_string()]);
    }

    #[test]
    fn html_splits_on_headers() {
        let chunks = split_html_by_headers(
            "<html><body><h1>Main Title</h1><p>First <b>bold</b> paragraph.</p><div>Loose text<p>Nested</p></div><h2>Section</h2><p>Second paragraph.</p><h4>Minor</h4><p>Tail</p></body></html>",
        );
        assert_eq!(
            chunks,
            vec![
                "Main Title".to_string(),
                "First paragraph.  \nbold  \nLoose text  \nNested".to_string(),
                "Section".to_string(),
                "Second paragraph.  \nMinor  \nTail".to_string(),
            ]
        );
    }

    #[test]
    fn large_html_is_chunked_within_limit() {
        let body = build_text_exceeding_tokens("Paragraph text for the page. ", CHUNK_SIZE);
        let html = format!("<html><body><h1>The complete title of this rather long page</h1><p>{body}</p><h2>Another section title</h2><p>{body}</p></body></html>");
        let chunks = chunk_text(&html, None, None);
        assert!(chunks.len() > 2);
        assert_chunks_within_token_limit(&chunks);
        assert!(chunks.iter().any(|c| c == "The complete title of this rather long page"));
    }

    // --- configuration ---

    #[test]
    fn config_defaults() {
        let config = ChunkConfig::default();
        assert_eq!((config.chunk_size, config.chunk_overlap, config.min_chunk_size), (400, 60, 5));
        assert_eq!(ChunkConfig::from_settings(None, None, None), config);
        assert_eq!(ChunkConfig::from_settings(Some(""), Some(""), None), config);
    }

    #[test]
    fn config_validation_matches_original() {
        let c = ChunkConfig::from_settings(Some("1500"), Some("150"), Some("0"));
        assert_eq!((c.chunk_size, c.chunk_overlap, c.min_chunk_size), (1500, 150, 0));

        // Too small: floor of 100, overlap follows the effective size.
        let c = ChunkConfig::from_settings(Some("50"), None, None);
        assert_eq!((c.chunk_size, c.chunk_overlap), (100, 15));

        // Unparseable values fall back to defaults.
        let c = ChunkConfig::from_settings(Some("abc"), Some("x"), Some("y"));
        assert_eq!((c.chunk_size, c.chunk_overlap, c.min_chunk_size), (400, 60, 5));

        // Negative overlap -> 0; overlap >= size -> 15%.
        assert_eq!(ChunkConfig::from_settings(Some("1000"), Some("-5"), None).chunk_overlap, 0);
        assert_eq!(ChunkConfig::from_settings(Some("1000"), Some("1000"), None).chunk_overlap, 150);

        // Negative minimum -> 0; empty minimum is invalid -> default.
        assert_eq!(ChunkConfig::from_settings(None, None, Some("-1")).min_chunk_size, 0);
        assert_eq!(ChunkConfig::from_settings(None, None, Some("")).min_chunk_size, 5);
        assert_eq!(ChunkConfig::from_settings(Some(" 800 "), None, None).chunk_size, 800);
    }

    #[test]
    fn custom_config_controls_chunk_size() {
        let text = build_text_exceeding_tokens("This is a sentence. ", 300);
        let config = ChunkConfig { chunk_size: 100, chunk_overlap: 10, min_chunk_size: 5 };
        let chunks = chunk_text_with(&text, Some(ContentType::Plain), None, &config);
        assert!(chunks.len() >= 3);
        assert!(chunks.iter().all(|c| token_count(c) <= 100));
    }
}
