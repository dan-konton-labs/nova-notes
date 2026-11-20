//! Text clean-up helpers.
//!
//! Port of `open_notebook/utils/text_utils.py`: stripping `<think>` blocks
//! from model output, flattening structured LLM content, and the two
//! character filters.

use std::sync::OnceLock;

use regex::Regex;
use unicode_general_category::{get_general_category, GeneralCategory as Gc};

/// Content longer than this many characters is returned untouched by
/// [`parse_thinking_content`].
pub const THINKING_PARSE_LIMIT: usize = 100_000;

fn think_pattern() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"(?s)<think>(.*?)</think>").expect("valid regex"))
}

fn blank_lines_pattern() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"\n\s*\n\s*\n").expect("valid regex"))
}

/// Remove every non-ASCII character.
pub fn remove_non_ascii(text: &str) -> String {
    text.chars().filter(char::is_ascii).collect()
}

pub(crate) fn is_other_category(c: char) -> bool {
    matches!(
        get_general_category(c),
        Gc::Control | Gc::Format | Gc::Surrogate | Gc::PrivateUse | Gc::Unassigned
    )
}

/// Python's `str.isprintable` for a single character.
pub(crate) fn is_printable(c: char) -> bool {
    if c == ' ' {
        return true;
    }
    !is_other_category(c)
        && !matches!(
            get_general_category(c),
            Gc::SpaceSeparator | Gc::LineSeparator | Gc::ParagraphSeparator
        )
}

/// Python's `\w` for a single character: letters and numbers of any script,
/// plus underscore.
fn is_word_char(c: char) -> bool {
    c == '_'
        || matches!(
            get_general_category(c),
            Gc::UppercaseLetter
                | Gc::LowercaseLetter
                | Gc::TitlecaseLetter
                | Gc::ModifierLetter
                | Gc::OtherLetter
                | Gc::DecimalNumber
                | Gc::LetterNumber
                | Gc::OtherNumber
        )
}

/// Normalise exotic whitespace and drop control characters and symbols,
/// keeping letters, digits, whitespace and `. , ! ? -`.
pub fn remove_non_printable(text: &str) -> String {
    let normalised: String = text
        .chars()
        .map(|c| match c {
            '\u{2000}'..='\u{200B}' | '\u{202F}' | '\u{205F}' | '\u{3000}' => ' ',
            '\u{2028}' | '\u{2029}' | '\r' => '\n',
            other => other,
        })
        .filter(|&c| c == '\n' || c == '\t' || !is_other_category(c))
        .map(|c| if c == '\u{a0}' { ' ' } else { c })
        .collect();

    normalised
        .trim()
        .chars()
        .filter(|&c| is_word_char(c) || c.is_whitespace() || matches!(c, '.' | ',' | '!' | '?' | '-'))
        .collect()
}

/// Split model output into `(thinking, answer)`.
///
/// Handles well-formed `<think>...</think>` blocks (any number of them),
/// output where the opening tag is missing but `</think>` is present, and
/// output truncated inside an unclosed `<think>` (all of it is reasoning, so
/// the answer is empty). Content above [`THINKING_PARSE_LIMIT`] characters is
/// returned unchanged.
pub fn parse_thinking_content(content: &str) -> (String, String) {
    if content.len() > THINKING_PARSE_LIMIT && content.chars().count() > THINKING_PARSE_LIMIT {
        return (String::new(), content.to_string());
    }

    let pattern = think_pattern();
    let thoughts: Vec<&str> = pattern
        .captures_iter(content)
        .map(|c| c.get(1).map_or("", |m| m.as_str()).trim())
        .collect();
    if !thoughts.is_empty() {
        let thinking = thoughts.join("\n\n");
        let without_blocks = pattern.replace_all(content, "");
        let cleaned = blank_lines_pattern().replace_all(&without_blocks, "\n\n").trim().to_string();
        return (thinking, cleaned);
    }

    const CLOSE: &str = "</think>";
    if let Some(end) = content.find(CLOSE) {
        let thinking = content[..end].trim().to_string();
        let cleaned = content[end + CLOSE.len()..].trim().to_string();
        return (thinking, cleaned);
    }

    if let Some(rest) = content.trim_start().strip_prefix("<think>") {
        return (rest.trim().to_string(), String::new());
    }

    (String::new(), content.to_string())
}

/// The answer part of [`parse_thinking_content`].
pub fn clean_thinking_content(content: &str) -> String {
    parse_thinking_content(content).1
}

/// Flatten LLM response content to text.
///
/// Accepts a plain string or the list-of-parts envelope some providers return
/// (`[{"type": "text", "text": "..."}]`); parts without a `text` field are
/// skipped. `null` gives an empty string; any other value is rendered as JSON.
pub fn extract_text_content(content: &serde_json::Value) -> String {
    use serde_json::Value;
    match content {
        Value::Null => String::new(),
        Value::String(s) => s.clone(),
        Value::Array(parts) => parts
            .iter()
            .filter_map(|part| match part {
                Value::String(s) => Some(s.clone()),
                Value::Object(map) => map.get("text").map(|text| match text {
                    Value::String(s) => s.clone(),
                    other => other.to_string(),
                }),
                _ => None,
            })
            .collect(),
        other => other.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn remove_non_ascii_strips_everything_above_127() {
        let result = remove_non_ascii("Hello 世界 café naïve émoji 🎉");
        assert_eq!(result, "Hello  caf nave moji ");
        assert!(result.is_ascii());
    }

    #[test]
    fn remove_non_ascii_keeps_pure_ascii() {
        let text = "Hello World 123 !@#";
        assert_eq!(remove_non_ascii(text), text);
    }

    #[test]
    fn remove_non_printable_normalises_unicode_whitespace() {
        let result = remove_non_printable("Hello\u{2000}World\u{200b}\u{202f}Test");
        assert_eq!(result, "Hello World  Test");
    }

    #[test]
    fn remove_non_printable_preserves_newlines_and_tabs() {
        let result = remove_non_printable("Line1\nLine2\tTabbed");
        assert_eq!(result, "Line1\nLine2\tTabbed");
    }

    #[test]
    fn remove_non_printable_drops_controls_and_symbols() {
        assert_eq!(remove_non_printable("a\u{0007}b\u{200d}c"), "abc");
        assert_eq!(remove_non_printable("  price: $5 (five) – café_1!  "), "price 5 five  café_1!");
        assert_eq!(remove_non_printable("one\r\ntwo\u{2028}three"), "one\n\ntwo\nthree");
        assert_eq!(remove_non_printable("non\u{a0}breaking"), "non breaking");
    }

    #[test]
    fn parse_thinking_basic() {
        let (thinking, cleaned) = parse_thinking_content("<think>This is my thinking</think>Here is my answer");
        assert_eq!(thinking, "This is my thinking");
        assert_eq!(cleaned, "Here is my answer");
    }

    #[test]
    fn parse_thinking_multiple_blocks() {
        let (thinking, cleaned) =
            parse_thinking_content("<think>First thought</think>Answer<think>Second thought</think>More");
        assert_eq!(thinking, "First thought\n\nSecond thought");
        assert!(!cleaned.contains("<think>"));
        assert_eq!(cleaned, "AnswerMore");
    }

    #[test]
    fn parse_thinking_collapses_blank_lines_left_behind() {
        let (_, cleaned) = parse_thinking_content("Intro\n\n<think>x</think>\n\n\nAnswer");
        assert_eq!(cleaned, "Intro\n\nAnswer");
    }

    #[test]
    fn parse_thinking_without_tags() {
        let (thinking, cleaned) = parse_thinking_content("Just regular content");
        assert_eq!(thinking, "");
        assert_eq!(cleaned, "Just regular content");
    }

    #[test]
    fn parse_thinking_missing_open_tag() {
        let (thinking, cleaned) = parse_thinking_content("Some thinking content</think>Here is my answer");
        assert_eq!(thinking, "Some thinking content");
        assert_eq!(cleaned, "Here is my answer");
    }

    #[test]
    fn parse_thinking_truncated_without_close_tag() {
        let (thinking, cleaned) = parse_thinking_content("<think>Let me reason about this at length and then");
        assert_eq!(thinking, "Let me reason about this at length and then");
        assert_eq!(cleaned, "");
        assert_eq!(clean_thinking_content("  <think>partial"), "");
    }

    #[test]
    fn parse_thinking_skips_very_large_content() {
        let large = "x".repeat(200_000);
        let (thinking, cleaned) = parse_thinking_content(&large);
        assert_eq!(thinking, "");
        assert_eq!(cleaned, large);

        let tagged = format!("<think>{}</think>answer", "y".repeat(150_000));
        let (thinking, cleaned) = parse_thinking_content(&tagged);
        assert_eq!(thinking, "");
        assert_eq!(cleaned, tagged);
    }

    #[test]
    fn clean_thinking_returns_only_the_answer() {
        let result = clean_thinking_content("<think>Internal thoughts</think>Public response");
        assert_eq!(result, "Public response");
    }

    #[test]
    fn extract_text_content_handles_strings_and_part_lists() {
        assert_eq!(extract_text_content(&json!(null)), "");
        assert_eq!(extract_text_content(&json!("plain")), "plain");
        assert_eq!(
            extract_text_content(&json!([
                {"type": "text", "text": "Hello ", "extras": {"signature": "abc"}},
                "world",
                {"type": "image", "url": "x"},
                42
            ])),
            "Hello world"
        );
        assert_eq!(extract_text_content(&json!(123)), "123");
    }
}
