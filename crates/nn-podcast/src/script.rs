//! Turn a user-written script (plain text or Markdown) into podcast dialogue.
//!
//! The script is read as written: no model rewrites it. Markdown syntax is
//! stripped so the voice doesn't read symbols aloud, each paragraph becomes one
//! clip, and a line starting with a speaker name from the chosen profile
//! (`Max: ...` or `**Max:** ...`) switches to that speaker's voice. Text before
//! any label is read by the profile's first speaker.
//!
//! This is a port of `open_notebook/podcasts/script.py`. Two of its patterns
//! use look-around and back-references, which the `regex` crate does not have;
//! those are written out by hand below with the same matching rules.

use std::collections::HashMap;
use std::sync::OnceLock;

use regex::Regex;

use crate::error::{PodcastError, Result};
use crate::models::Dialogue;

/// Clips longer than this are split at sentence ends; some TTS providers cap a
/// single request well below a long paragraph (OpenAI 4096, ElevenLabs 5000).
pub const MAX_CLIP_CHARS: usize = 1500;

/// Longest script the original's API accepted (characters).
pub const MAX_SCRIPT_CHARS: usize = 500_000;

struct Patterns {
    frontmatter: Regex,
    label: Regex,
    html_tag: Regex,
    image: Regex,
    link: Regex,
    heading: Regex,
    blockquote: Regex,
    list_marker: Regex,
    whitespace: Regex,
}

fn patterns() -> &'static Patterns {
    static PATTERNS: OnceLock<Patterns> = OnceLock::new();
    PATTERNS.get_or_init(|| {
        let compile = |pattern: &str| Regex::new(pattern).expect("valid regex");
        Patterns {
            frontmatter: compile(r"(?s)\A---\n.*?\n---\n"),
            label: compile(r"^\**\s*([^:*\n]{1,60}?)\s*\**\s*:\s*\**\s*(.*)$"),
            html_tag: compile(r"<[^>]+>"),
            image: compile(r"!\[[^\]]*\]\([^)]*\)"),
            link: compile(r"\[([^\]]+)\]\([^)]*\)"),
            heading: compile(r"^\s{0,3}#{1,6}\s*"),
            blockquote: compile(r"^\s*>+\s?"),
            list_marker: compile(r"^\s*(?:[-*+]|\d+[.)])\s+"),
            whitespace: compile(r"\s+"),
        }
    })
}

/// `^\s*([-*_])(\s*\1){2,}\s*$`: a Markdown horizontal rule.
fn is_horizontal_rule(line: &str) -> bool {
    let mut marks = line.chars().filter(|c| !c.is_whitespace());
    let Some(first) = marks.next() else {
        return false;
    };
    matches!(first, '-' | '*' | '_') && {
        let mut count = 1;
        marks.all(|c| {
            count += 1;
            c == first
        }) && count >= 3
    }
}

/// `(\*\*|__|\*|_|~~|`)(?=\S)(.+?)(?<=\S)\1` replaced by its content: remove
/// emphasis, strikethrough and code delimiters that tightly wrap text.
fn strip_inline_markup(line: &str) -> String {
    const DELIMITERS: [&str; 6] = ["**", "__", "*", "_", "~~", "`"];
    let chars: Vec<char> = line.chars().collect();
    let starts_with = |at: usize, delimiter: &[char]| {
        chars.len() >= at + delimiter.len() && chars[at..at + delimiter.len()] == *delimiter
    };

    let mut out = String::with_capacity(line.len());
    let mut position = 0;
    'scan: while position < chars.len() {
        for delimiter in DELIMITERS {
            let delimiter: Vec<char> = delimiter.chars().collect();
            if !starts_with(position, &delimiter) {
                continue;
            }
            let start = position + delimiter.len();
            if start >= chars.len() || chars[start].is_whitespace() {
                continue;
            }
            // Shortest content (at least one character, not ending in
            // whitespace) that is followed by the same delimiter.
            let close = (start + 1..chars.len())
                .find(|&end| !chars[end - 1].is_whitespace() && starts_with(end, &delimiter));
            if let Some(end) = close {
                out.extend(&chars[start..end]);
                position = end + delimiter.len();
                continue 'scan;
            }
        }
        out.push(chars[position]);
        position += 1;
    }
    out
}

fn clean_line(line: &str) -> String {
    let p = patterns();
    let line = p.html_tag.replace_all(line, " ");
    let line = p.image.replace_all(&line, "");
    let line = p.link.replace_all(&line, "$1");
    let line = p.heading.replace(&line, "");
    let line = p.blockquote.replace(&line, "");
    let line = p.list_marker.replace(&line, "");
    let line = strip_inline_markup(&line).replace('`', "");
    p.whitespace.replace_all(&line, " ").trim().to_string()
}

/// `(?<=[.!?…])\s+` split: break after sentence-ending punctuation, dropping
/// the whitespace run that follows it.
fn split_sentences(text: &[char]) -> Vec<&[char]> {
    let mut sentences = Vec::new();
    let mut start = 0;
    let mut position = 0;
    while position < text.len() {
        let after_sentence_end = position > 0 && matches!(text[position - 1], '.' | '!' | '?' | '…');
        if after_sentence_end && text[position].is_whitespace() {
            let mut end = position;
            while end < text.len() && text[end].is_whitespace() {
                end += 1;
            }
            sentences.push(&text[start..position]);
            start = end;
            position = end;
        } else {
            position += 1;
        }
    }
    sentences.push(&text[start..]);
    sentences
}

fn trimmed(chars: &[char]) -> &[char] {
    let start = chars.iter().position(|c| !c.is_whitespace()).unwrap_or(chars.len());
    let end = chars.iter().rposition(|c| !c.is_whitespace()).map_or(start, |i| i + 1);
    &chars[start..end]
}

fn split_long(text: &str, limit: usize) -> Vec<String> {
    let chars: Vec<char> = text.chars().collect();
    if chars.len() <= limit {
        return vec![text.to_string()];
    }
    let mut chunks: Vec<String> = Vec::new();
    let mut current: Vec<char> = Vec::new();
    for sentence in split_sentences(&chars) {
        let mut sentence = sentence;
        while sentence.len() > limit {
            // One huge sentence: cut at a space.
            let cut = match sentence[..limit].iter().rposition(|&c| c == ' ') {
                Some(space) if space > 0 => space,
                _ => limit,
            };
            if !current.is_empty() {
                chunks.push(current.iter().collect());
                current.clear();
            }
            chunks.push(trimmed(&sentence[..cut]).iter().collect());
            sentence = trimmed(&sentence[cut..]);
        }
        if sentence.is_empty() {
            continue;
        }
        if !current.is_empty() && current.len() + 1 + sentence.len() > limit {
            chunks.push(current.iter().collect());
            current = sentence.to_vec();
        } else {
            if !current.is_empty() {
                current.push(' ');
            }
            current.extend_from_slice(sentence);
        }
    }
    if !current.is_empty() {
        chunks.push(current.iter().collect());
    }
    chunks
}

/// Return the clips for `text`, in reading order.
///
/// Fails with [`PodcastError::InvalidInput`] when there is nothing to read or
/// the profile has no speakers.
pub fn parse_script(text: &str, speaker_names: &[String]) -> Result<Vec<Dialogue>> {
    let Some(first_speaker) = speaker_names.first() else {
        return Err(PodcastError::InvalidInput("The speaker profile has no speakers".into()));
    };
    let by_lower: HashMap<String, &String> =
        speaker_names.iter().map(|name| (name.trim().to_lowercase(), name)).collect();
    let p = patterns();

    let text = text.replace("\r\n", "\n").replace('\r', "\n");
    let text = p.frontmatter.replace(&text, "");

    let mut paragraphs: Vec<(&String, String)> = Vec::new();
    let mut speaker = first_speaker;
    let mut paragraph_speaker = first_speaker;
    let mut paragraph_lines: Vec<String> = Vec::new();

    fn flush<'a>(
        paragraphs: &mut Vec<(&'a String, String)>,
        speaker: &'a String,
        lines: &mut Vec<String>,
    ) {
        let body = lines.join(" ").trim().to_string();
        if !body.is_empty() {
            paragraphs.push((speaker, body));
        }
        lines.clear();
    }

    for raw in text.split('\n') {
        if raw.trim().is_empty() || is_horizontal_rule(raw) {
            flush(&mut paragraphs, paragraph_speaker, &mut paragraph_lines);
            continue;
        }
        let mut line = raw;
        if let Some(label) = p.label.captures(raw.trim()) {
            let name = label.get(1).map_or("", |m| m.as_str()).trim().to_lowercase();
            if let Some(known) = by_lower.get(&name) {
                flush(&mut paragraphs, paragraph_speaker, &mut paragraph_lines);
                speaker = known;
                line = label.get(2).map_or("", |m| m.as_str());
            }
        }
        if !paragraph_lines.is_empty() && speaker != paragraph_speaker {
            flush(&mut paragraphs, paragraph_speaker, &mut paragraph_lines);
        }
        paragraph_speaker = speaker;
        let cleaned = clean_line(line);
        if !cleaned.is_empty() {
            paragraph_lines.push(cleaned);
        }
    }
    flush(&mut paragraphs, paragraph_speaker, &mut paragraph_lines);

    let turns: Vec<Dialogue> = paragraphs
        .into_iter()
        .flat_map(|(who, body)| {
            split_long(&body, MAX_CLIP_CHARS).into_iter().map(move |chunk| Dialogue::new(who.clone(), chunk))
        })
        .collect();
    if turns.is_empty() {
        return Err(PodcastError::InvalidInput("The script has no text to read".into()));
    }
    Ok(turns)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn speakers() -> Vec<String> {
        vec!["Velvet Noir".to_string(), "Max".to_string()]
    }

    fn turns(pairs: &[(&str, &str)]) -> Vec<Dialogue> {
        pairs.iter().map(|(speaker, dialogue)| Dialogue::new(*speaker, *dialogue)).collect()
    }

    #[test]
    fn unlabelled_text_goes_to_first_speaker_one_clip_per_paragraph() {
        let parsed = parse_script("Hello there.\nStill me.\n\nNew paragraph.", &speakers()).unwrap();
        assert_eq!(
            parsed,
            turns(&[("Velvet Noir", "Hello there. Still me."), ("Velvet Noir", "New paragraph.")])
        );
    }

    #[test]
    fn speaker_labels_switch_voice_case_insensitively() {
        let script = "velvet noir: Welcome back.\nMAX: Thanks.\nAnd more from Max.";
        assert_eq!(
            parse_script(script, &speakers()).unwrap(),
            turns(&[("Velvet Noir", "Welcome back."), ("Max", "Thanks. And more from Max.")])
        );
    }

    #[test]
    fn markdown_bold_labels_and_formatting_are_stripped() {
        let script = "---\ntitle: Ep 1\n---\n\
                      # The Case\n\n\
                      **Max:** This is *really* `odd`, see [the file](http://x.y).\n\n\
                      - first point\n> quoted line\n\n***\n<br>";
        assert_eq!(
            parse_script(script, &speakers()).unwrap(),
            turns(&[
                ("Velvet Noir", "The Case"),
                ("Max", "This is really odd, see the file."),
                ("Max", "first point quoted line"),
            ])
        );
    }

    #[test]
    fn unknown_label_is_read_as_text() {
        assert_eq!(
            parse_script("Note: read this aloud.", &speakers()).unwrap(),
            turns(&[("Velvet Noir", "Note: read this aloud.")])
        );
    }

    #[test]
    fn long_paragraph_splits_on_sentence_boundaries() {
        let sentence = "This sentence is exactly fifty characters long ok. ";
        let text = sentence.repeat(100);
        let parsed = parse_script(&text, &speakers()).unwrap();
        assert!(parsed.len() > 1);
        assert!(parsed.iter().all(|t| t.dialogue.chars().count() <= MAX_CLIP_CHARS));
        assert!(parsed.iter().all(|t| t.dialogue.ends_with('.')));
        let joined = parsed.iter().map(|t| t.dialogue.as_str()).collect::<Vec<_>>().join(" ");
        assert_eq!(joined, text.trim());
    }

    #[test]
    fn single_overlong_sentence_is_hard_split() {
        let parsed = parse_script(&"word ".repeat(1000), &speakers()).unwrap();
        assert!(parsed.len() > 1);
        assert!(parsed.iter().all(|t| {
            let length = t.dialogue.chars().count();
            0 < length && length <= MAX_CLIP_CHARS
        }));
    }

    #[test]
    fn unbroken_text_is_cut_at_the_limit() {
        let parsed = parse_script(&"x".repeat(3200), &speakers()).unwrap();
        let lengths: Vec<usize> = parsed.iter().map(|t| t.dialogue.len()).collect();
        assert_eq!(lengths, [1500, 1500, 200]);
    }

    #[test]
    fn empty_script_raises() {
        let error = parse_script("  \n# \n---\n", &speakers()).unwrap_err();
        assert_eq!(error.to_string(), "The script has no text to read");
        assert!(matches!(error, PodcastError::InvalidInput(_)));
    }

    #[test]
    fn no_speakers_raises() {
        let error = parse_script("Hello.", &[]).unwrap_err();
        assert_eq!(error.to_string(), "The speaker profile has no speakers");
    }

    #[test]
    fn windows_line_endings_and_label_only_lines() {
        let script = "Max:\r\nFirst line.\r\n\r\nVelvet Noir: Reply.\r";
        assert_eq!(
            parse_script(script, &speakers()).unwrap(),
            turns(&[("Max", "First line."), ("Velvet Noir", "Reply.")])
        );
    }

    #[test]
    fn horizontal_rules_separate_paragraphs() {
        for rule in ["---", "***", "___", " - - - ", "*****"] {
            assert!(is_horizontal_rule(rule), "{rule}");
        }
        for not_rule in ["--", "-*-", "a---", "", "  "] {
            assert!(!is_horizontal_rule(not_rule), "{not_rule}");
        }
        assert_eq!(
            parse_script("One.\n- - -\nTwo.", &speakers()).unwrap(),
            turns(&[("Velvet Noir", "One."), ("Velvet Noir", "Two.")])
        );
    }

    #[test]
    fn inline_markup_needs_tight_delimiters() {
        assert_eq!(strip_inline_markup("a **bold** and __strong__ word"), "a bold and strong word");
        assert_eq!(strip_inline_markup("~~gone~~ `code` _em_"), "gone code em");
        assert_eq!(strip_inline_markup("2 * 3 * 4"), "2 * 3 * 4");
        assert_eq!(strip_inline_markup("snake_case_name"), "snakecasename");
        assert_eq!(strip_inline_markup("**unclosed"), "**unclosed");
        assert_eq!(strip_inline_markup("***both***"), "*both*");
        assert_eq!(strip_inline_markup("* a*b *"), "* a*b *");
        assert_eq!(strip_inline_markup("_ x _y_"), "_ x y");
        assert_eq!(strip_inline_markup("**a ** b**"), "a ** b");
    }

    #[test]
    fn label_forms_accepted_by_the_original() {
        let script = "Max: one\n**Velvet Noir**: two\n* Max: three\nMax : four\n\
                      A very long label that is not a speaker: text";
        assert_eq!(
            parse_script(script, &speakers()).unwrap(),
            turns(&[
                ("Max", "one"),
                ("Velvet Noir", "two"),
                ("Max", "three"),
                ("Max", "four A very long label that is not a speaker: text"),
            ])
        );
    }

    #[test]
    fn sentences_end_at_period_bang_question_and_ellipsis() {
        assert_eq!(split_long("Aa. Bb! Cc? Dd… Ee", 6), ["Aa.", "Bb!", "Cc?", "Dd… Ee"]);
        assert_eq!(
            parse_script("Hello.  World!\tNext?   End… Done", &speakers()).unwrap(),
            turns(&[("Velvet Noir", "Hello. World! Next? End… Done")])
        );
    }

    #[test]
    fn images_are_dropped_and_links_keep_their_text() {
        assert_eq!(clean_line("See ![diagram](a.png) and [the docs](http://d.oc)."), "See and the docs.");
        assert_eq!(clean_line("1) numbered <b>item</b>"), "numbered item");
        assert_eq!(clean_line("### Title"), "Title");
    }
}
