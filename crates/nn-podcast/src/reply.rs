//! Turning a model's reply into an outline or a transcript.
//!
//! Models wrap JSON in thinking tags, code fences and prose, and occasionally
//! leave a trailing comma. Everything here is local and deterministic: strip
//! what is not the answer, find the JSON object, validate it. Asking the model
//! again is the caller's job (see [`crate::generate`]).

use std::fmt;
use std::sync::OnceLock;

use regex::Regex;
use serde_json::Value;

use crate::models::{Dialogue, Outline, Segment, SegmentSize};

/// Why a reply could not be used.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReplyError {
    /// Nothing was left after removing thinking content.
    Empty,
    /// No JSON object with the expected shape could be found.
    Unparseable { snippet: String },
    /// JSON was found but its content breaks a rule.
    Invalid { what: &'static str, snippet: String, detail: String },
}

impl fmt::Display for ReplyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ReplyError::Empty => write!(f, "Invalid json output: the model returned an empty reply"),
            ReplyError::Unparseable { snippet } => write!(f, "Invalid json output: {snippet}"),
            ReplyError::Invalid { what, snippet, detail } => {
                write!(f, "Failed to parse {what} from completion {snippet}. Got: {detail}")
            }
        }
    }
}

impl ReplyError {
    /// The part worth telling the model when asking it to correct itself.
    pub(crate) fn detail(&self) -> String {
        match self {
            ReplyError::Empty => "the reply was empty".into(),
            ReplyError::Unparseable { .. } => "the reply was not valid JSON".into(),
            ReplyError::Invalid { detail, .. } => detail.clone(),
        }
    }
}

const SNIPPET_CHARS: usize = 500;

fn snippet(text: &str) -> String {
    let trimmed = text.trim();
    if trimmed.chars().count() <= SNIPPET_CHARS {
        return trimmed.to_string();
    }
    let cut: String = trimmed.chars().take(SNIPPET_CHARS).collect();
    format!("{cut}...")
}

// ---------------------------------------------------------------------------
// Thinking content
// ---------------------------------------------------------------------------

fn think_pattern() -> &'static Regex {
    static PATTERN: OnceLock<Regex> = OnceLock::new();
    PATTERN.get_or_init(|| Regex::new(r"(?s)<think>(.*?)</think>").expect("valid regex"))
}

fn blank_lines_pattern() -> &'static Regex {
    static PATTERN: OnceLock<Regex> = OnceLock::new();
    PATTERN.get_or_init(|| Regex::new(r"\n\s*\n\s*\n").expect("valid regex"))
}

/// Byte offset of the first position in `text` where a complete JSON object or
/// array starts.
fn first_json_start(text: &str) -> Option<usize> {
    text.char_indices()
        .filter(|(_, c)| matches!(c, '{' | '['))
        .map(|(i, _)| i)
        .find(|&i| decode_at(text, i).is_some())
}

/// Decode one JSON value starting at byte `start`, ignoring whatever follows.
fn decode_at(text: &str, start: usize) -> Option<Value> {
    serde_json::Deserializer::from_str(&text[start..]).into_iter::<Value>().next()?.ok()
}

/// Split a reply into `(thinking, answer)`.
///
/// Handles closed `<think>...</think>` blocks and the unclosed `<think>` some
/// models produce: after an unclosed tag, everything up to the first complete
/// JSON value is thinking, and when there is no JSON, everything is.
pub fn parse_thinking_content(content: &str) -> (String, String) {
    let pattern = think_pattern();
    let mut thinking: String = pattern
        .captures_iter(content)
        .map(|captures| captures[1].trim().to_string())
        .collect::<Vec<_>>()
        .join("\n\n");
    let mut cleaned = pattern.replace_all(content, "").into_owned();

    if let Some(index) = cleaned.find("<think>") {
        let before = cleaned[..index].to_string();
        let after = cleaned[index + "<think>".len()..].to_string();
        let (unclosed, answer) = match first_json_start(&after) {
            Some(position) => (
                after[..position].trim().to_string(),
                format!("{before}{}", after[position..].trim()).trim().to_string(),
            ),
            None => (after.trim().to_string(), before.trim().to_string()),
        };
        thinking = if thinking.is_empty() {
            unclosed
        } else {
            format!("{thinking}\n\n{unclosed}").trim().to_string()
        };
        cleaned = answer;
    }

    let cleaned = blank_lines_pattern().replace_all(&cleaned, "\n\n").trim().to_string();
    (thinking, cleaned)
}

/// The reply with thinking content removed.
pub fn clean_thinking_content(content: &str) -> String {
    parse_thinking_content(content).1
}

// ---------------------------------------------------------------------------
// JSON extraction
// ---------------------------------------------------------------------------

/// Remove commas that directly precede a closing bracket, outside strings.
fn remove_trailing_commas(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut in_string = false;
    let mut escaped = false;
    for (index, c) in text.char_indices() {
        if in_string {
            out.push(c);
            if escaped {
                escaped = false;
            } else if c == '\\' {
                escaped = true;
            } else if c == '"' {
                in_string = false;
            }
            continue;
        }
        match c {
            '"' => {
                in_string = true;
                out.push(c);
            }
            ',' => {
                let next = text[index + 1..].chars().find(|c| !c.is_whitespace());
                if !matches!(next, Some('}') | Some(']')) {
                    out.push(c);
                }
            }
            _ => out.push(c),
        }
    }
    out
}

/// Accept `value` when it is the answer: an object carrying `root_key`, or the
/// bare list that belongs under it.
fn as_answer(value: Value, root_key: &str) -> Option<Value> {
    match value {
        Value::Object(ref map) if map.contains_key(root_key) => Some(value),
        Value::Array(ref items) if !items.is_empty() && items.iter().all(Value::is_object) => {
            let mut wrapped = serde_json::Map::new();
            wrapped.insert(root_key.to_string(), value);
            Some(Value::Object(wrapped))
        }
        _ => None,
    }
}

fn find_answer(text: &str, root_key: &str) -> Option<Value> {
    if let Ok(value) = serde_json::from_str::<Value>(text.trim()) {
        if let Some(answer) = as_answer(value, root_key) {
            return Some(answer);
        }
    }
    // Leading prose, a code fence, trailing commentary: try every position
    // where a JSON value could start and take the first that is the answer.
    text.char_indices()
        .filter(|(_, c)| matches!(c, '{' | '['))
        .find_map(|(i, _)| decode_at(text, i).and_then(|value| as_answer(value, root_key)))
}

/// Find the JSON answer in a model reply: an object with `root_key`.
///
/// Thinking tags are removed first. A bare list is accepted and wrapped, and
/// one level of trailing-comma damage is repaired. Truncated JSON is never
/// salvaged: a reply cut off mid-answer is an error, not a shorter answer.
pub fn extract_json(reply: &str, root_key: &str) -> std::result::Result<Value, ReplyError> {
    let cleaned = clean_thinking_content(reply);
    if cleaned.trim().is_empty() {
        return Err(ReplyError::Empty);
    }
    find_answer(&cleaned, root_key)
        .or_else(|| find_answer(&remove_trailing_commas(&cleaned), root_key))
        .ok_or_else(|| ReplyError::Unparseable { snippet: snippet(&cleaned) })
}

// ---------------------------------------------------------------------------
// Outline and transcript
// ---------------------------------------------------------------------------

fn required_string<'a>(entry: &'a Value, key: &str, path: &str) -> std::result::Result<&'a str, String> {
    match entry.get(key) {
        Some(Value::String(text)) => Ok(text),
        Some(_) => Err(format!("{path}.{key}: Input should be a valid string")),
        None => Err(format!("{path}.{key}: Field required")),
    }
}

fn entries<'a>(answer: &'a Value, root_key: &str) -> std::result::Result<&'a Vec<Value>, String> {
    answer
        .get(root_key)
        .and_then(Value::as_array)
        .ok_or_else(|| format!("{root_key}: Input should be a valid list"))
}

/// Parse an outline reply: `{"segments": [{"name", "description", "size"}]}`.
///
/// `size` defaults to `medium` when absent and must otherwise be one of
/// `short`, `medium`, `long` (case and surrounding space are forgiven).
pub fn parse_outline(reply: &str) -> std::result::Result<Outline, ReplyError> {
    let answer = extract_json(reply, "segments")?;
    let invalid = |detail: String| ReplyError::Invalid {
        what: "Outline",
        snippet: snippet(&answer.to_string()),
        detail,
    };

    let mut segments = Vec::new();
    for (index, entry) in entries(&answer, "segments").map_err(invalid)?.iter().enumerate() {
        let path = format!("segments.{index}");
        let name = required_string(entry, "name", &path).map_err(invalid)?;
        let description = required_string(entry, "description", &path).map_err(invalid)?;
        let size = match entry.get("size") {
            None | Some(Value::Null) => SegmentSize::default(),
            Some(Value::String(text)) => SegmentSize::parse(text).ok_or_else(|| {
                invalid(format!("{path}.size: Input should be 'short', 'medium' or 'long' (got '{text}')"))
            })?,
            Some(_) => return Err(invalid(format!("{path}.size: Input should be 'short', 'medium' or 'long'"))),
        };
        segments.push(Segment { name: name.to_string(), description: description.to_string(), size });
    }
    if segments.is_empty() {
        return Err(invalid("segments: the outline has no segments".into()));
    }
    Ok(Outline { segments })
}

/// Parse a transcript reply: `{"transcript": [{"speaker", "dialogue"}]}`.
///
/// Every speaker must be one of `speaker_names` after trimming; anything else
/// fails with the original's `Invalid speaker name '...'. Must be one of: ...`.
/// Turns whose dialogue is empty are dropped (a text-to-speech call with no
/// text fails the whole episode); a transcript left with no turns is invalid.
pub fn parse_transcript(
    reply: &str,
    speaker_names: &[String],
) -> std::result::Result<Vec<Dialogue>, ReplyError> {
    let answer = extract_json(reply, "transcript")?;
    let invalid = |detail: String| ReplyError::Invalid {
        what: "Transcript",
        snippet: snippet(&answer.to_string()),
        detail,
    };

    let mut turns = Vec::new();
    for (index, entry) in entries(&answer, "transcript").map_err(invalid)?.iter().enumerate() {
        let path = format!("transcript.{index}");
        let speaker = required_string(entry, "speaker", &path).map_err(invalid)?.trim();
        if speaker.is_empty() {
            return Err(invalid(format!("{path}.speaker: Speaker name cannot be empty")));
        }
        if !speaker_names.iter().any(|name| name == speaker) {
            return Err(invalid(format!(
                "{path}.speaker: Invalid speaker name '{speaker}'. Must be one of: {}",
                speaker_names.join(", ")
            )));
        }
        let dialogue = required_string(entry, "dialogue", &path).map_err(invalid)?;
        if dialogue.trim().is_empty() {
            tracing::warn!("Dropping an empty dialogue turn for {speaker}");
            continue;
        }
        turns.push(Dialogue::new(speaker, dialogue));
    }
    if turns.is_empty() {
        return Err(invalid("transcript: the transcript has no dialogue".into()));
    }
    Ok(turns)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn names() -> Vec<String> {
        vec!["Marcus Thompson".to_string(), "Elena Vasquez".to_string()]
    }

    const OUTLINE: &str = r#"{"segments": [{"name": "Intro", "description": "Open", "size": "short"}, {"name": "Body", "description": "Detail"}]}"#;
    const TRANSCRIPT: &str = r#"{"transcript": [{"speaker": "Marcus Thompson", "dialogue": "Hello."}, {"speaker": " Elena Vasquez ", "dialogue": "Hi."}]}"#;

    // --- thinking content --------------------------------------------------

    #[test]
    fn closed_think_blocks_are_removed() {
        let (thinking, cleaned) =
            parse_thinking_content("<think>Let me analyze this</think>Here's my answer");
        assert_eq!(thinking, "Let me analyze this");
        assert_eq!(cleaned, "Here's my answer");
    }

    #[test]
    fn several_think_blocks_are_joined() {
        let (thinking, cleaned) =
            parse_thinking_content("<think>one</think>A\n<think>\ntwo\n</think>\nB");
        assert_eq!(thinking, "one\n\ntwo");
        assert_eq!(cleaned, "A\n\nB");
    }

    #[test]
    fn unclosed_think_keeps_the_json_that_follows() {
        let reply = format!("<think>reasoning about [the] outline {OUTLINE}");
        let (thinking, cleaned) = parse_thinking_content(&reply);
        assert_eq!(thinking, "reasoning about [the] outline");
        assert_eq!(cleaned, OUTLINE);
    }

    #[test]
    fn unclosed_think_without_json_leaves_nothing() {
        let (thinking, cleaned) = parse_thinking_content("Sure.<think>all of the output went here");
        assert_eq!(thinking, "all of the output went here");
        assert_eq!(cleaned, "Sure.");
        assert_eq!(parse_outline("<think>everything is thinking").unwrap_err(), ReplyError::Empty);
    }

    #[test]
    fn content_without_tags_is_untouched() {
        assert_eq!(clean_thinking_content("  plain  "), "plain");
    }

    // --- JSON extraction -----------------------------------------------------

    #[test]
    fn plain_json_parses() {
        let outline = parse_outline(OUTLINE).unwrap();
        assert_eq!(outline.segments.len(), 2);
        assert_eq!(outline.segments[0].size, SegmentSize::Short);
        assert_eq!(outline.segments[1].size, SegmentSize::Medium);
    }

    #[test]
    fn code_fences_are_tolerated() {
        let fenced = format!("```json\n{OUTLINE}\n```");
        assert_eq!(parse_outline(&fenced).unwrap().segments.len(), 2);
        let bare_fence = format!("```\n{TRANSCRIPT}\n```\n");
        assert_eq!(parse_transcript(&bare_fence, &names()).unwrap().len(), 2);
    }

    #[test]
    fn leading_prose_and_trailing_commentary_are_tolerated() {
        let reply = format!("Sure! Here is the outline you asked for [1]:\n\n{OUTLINE}\n\nLet me know if you want changes.");
        assert_eq!(parse_outline(&reply).unwrap().segments[1].name, "Body");
    }

    #[test]
    fn thinking_then_fenced_json_parses() {
        let reply = format!("<think>\nplan {{not json}}\n</think>\n```json\n{TRANSCRIPT}\n```");
        let turns = parse_transcript(&reply, &names()).unwrap();
        assert_eq!(turns[1], Dialogue::new("Elena Vasquez", "Hi."));
    }

    #[test]
    fn trailing_commas_are_repaired() {
        let reply = r#"{"segments": [{"name": "A, b", "description": "x,]", "size": "long",},],}"#;
        let outline = parse_outline(reply).unwrap();
        assert_eq!(outline.segments[0].name, "A, b");
        assert_eq!(outline.segments[0].description, "x,]");
        assert_eq!(outline.segments[0].size, SegmentSize::Long);
    }

    #[test]
    fn a_bare_list_is_accepted() {
        let reply = r#"[{"speaker": "Marcus Thompson", "dialogue": "Solo."}]"#;
        assert_eq!(parse_transcript(reply, &names()).unwrap().len(), 1);
    }

    #[test]
    fn an_answer_wrapped_in_another_object_is_found() {
        let reply = format!("{{\"result\": {OUTLINE}}}");
        assert_eq!(parse_outline(&reply).unwrap().segments.len(), 2);
    }

    #[test]
    fn truncated_json_is_not_salvaged() {
        let cut = &TRANSCRIPT[..TRANSCRIPT.len() - 30];
        let error = parse_transcript(cut, &names()).unwrap_err();
        assert!(matches!(error, ReplyError::Unparseable { .. }), "{error:?}");
        assert!(error.to_string().starts_with("Invalid json output: {\"transcript\""));
    }

    #[test]
    fn text_without_json_is_unparseable() {
        let error = parse_outline("I cannot help with that.").unwrap_err();
        assert_eq!(error.to_string(), "Invalid json output: I cannot help with that.");
    }

    #[test]
    fn empty_reply_is_reported_as_empty() {
        assert_eq!(parse_outline("   ").unwrap_err(), ReplyError::Empty);
        assert!(ReplyError::Empty.to_string().starts_with("Invalid json output"));
    }

    #[test]
    fn long_garbage_is_cut_in_the_message() {
        let error = parse_outline(&"x".repeat(5000)).unwrap_err().to_string();
        assert!(error.len() < 600, "{}", error.len());
        assert!(error.ends_with("..."));
    }

    // --- validation ------------------------------------------------------------

    #[test]
    fn size_is_case_insensitive_but_must_be_known() {
        let ok = r#"{"segments": [{"name": "A", "description": "d", "size": " LONG "}]}"#;
        assert_eq!(parse_outline(ok).unwrap().segments[0].size, SegmentSize::Long);

        let bad = r#"{"segments": [{"name": "A", "description": "d", "size": "huge"}]}"#;
        let error = parse_outline(bad).unwrap_err().to_string();
        assert!(error.starts_with("Failed to parse Outline from completion"), "{error}");
        assert!(error.contains("segments.0.size"), "{error}");
    }

    #[test]
    fn outline_needs_segments_with_name_and_description() {
        let empty = parse_outline(r#"{"segments": []}"#).unwrap_err().to_string();
        assert!(empty.contains("no segments"), "{empty}");

        let missing = parse_outline(r#"{"segments": [{"name": "A"}]}"#).unwrap_err().to_string();
        assert!(missing.contains("segments.0.description: Field required"), "{missing}");

        let not_a_list = parse_outline(r#"{"segments": "none"}"#).unwrap_err().to_string();
        assert!(not_a_list.contains("segments: Input should be a valid list"), "{not_a_list}");
    }

    #[test]
    fn placeholder_speaker_name_is_rejected_with_the_original_message() {
        let reply = r#"{"transcript": [{"speaker": "...", "dialogue": "..."}]}"#;
        let error = parse_transcript(reply, &names()).unwrap_err().to_string();
        assert!(
            error.contains("Invalid speaker name '...'. Must be one of: Marcus Thompson, Elena Vasquez"),
            "{error}"
        );
        assert!(error.starts_with("Failed to parse Transcript from completion {\"transcript\""));
    }

    #[test]
    fn speaker_names_are_trimmed_and_case_sensitive() {
        let turns = parse_transcript(TRANSCRIPT, &names()).unwrap();
        assert_eq!(turns[1].speaker, "Elena Vasquez");

        let wrong_case = r#"{"transcript": [{"speaker": "marcus thompson", "dialogue": "x"}]}"#;
        assert!(parse_transcript(wrong_case, &names()).is_err());

        let empty = r#"{"transcript": [{"speaker": "  ", "dialogue": "x"}]}"#;
        let error = parse_transcript(empty, &names()).unwrap_err().to_string();
        assert!(error.contains("Speaker name cannot be empty"), "{error}");
    }

    #[test]
    fn empty_dialogue_turns_are_dropped() {
        let reply = r#"{"transcript": [{"speaker": "Marcus Thompson", "dialogue": "  "}, {"speaker": "Elena Vasquez", "dialogue": "Real words."}]}"#;
        let turns = parse_transcript(reply, &names()).unwrap();
        assert_eq!(turns, [Dialogue::new("Elena Vasquez", "Real words.")]);

        let all_empty = r#"{"transcript": [{"speaker": "Marcus Thompson", "dialogue": ""}]}"#;
        let error = parse_transcript(all_empty, &names()).unwrap_err().to_string();
        assert!(error.contains("no dialogue"), "{error}");
    }

    #[test]
    fn dialogue_must_be_a_string() {
        let reply = r#"{"transcript": [{"speaker": "Marcus Thompson", "dialogue": 5}]}"#;
        let error = parse_transcript(reply, &names()).unwrap_err().to_string();
        assert!(error.contains("transcript.0.dialogue: Input should be a valid string"), "{error}");
    }
}
