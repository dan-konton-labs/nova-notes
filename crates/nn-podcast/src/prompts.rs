//! The outline and transcript prompts.
//!
//! The two templates in `templates/` are byte-for-byte copies of the
//! original's `prompts/podcast/*.jinja`, rendered with `minijinja` using the
//! variable names podcast-creator passed. They are compiled into the binary;
//! user text only ever reaches them as plain variables, never as template
//! source.

use std::sync::OnceLock;

use minijinja::Environment;
use serde::Serialize;
use serde_json::{json, Value};

use crate::error::{PodcastError, Result};
use crate::models::{Content, Dialogue, Outline, Segment, Speaker};

const OUTLINE_TEMPLATE: &str = include_str!("../templates/outline.jinja");
const TRANSCRIPT_TEMPLATE: &str = include_str!("../templates/transcript.jinja");

const FORMAT_INSTRUCTIONS_HEAD: &str = "The output should be formatted as a JSON instance that conforms to the JSON schema below.

As an example, for the schema {\"properties\": {\"foo\": {\"title\": \"Foo\", \"description\": \"a list of strings\", \"type\": \"array\", \"items\": {\"type\": \"string\"}}}, \"required\": [\"foo\"]}
the object {\"foo\": [\"bar\", \"baz\"]} is a well-formatted instance of the schema. The object {\"properties\": {\"foo\": [\"bar\", \"baz\"]}} is not well-formatted.

Here is the output schema:
```
";

const OUTLINE_SCHEMA: &str = r##"{"$defs": {"Segment": {"properties": {"name": {"description": "Name of the segment", "title": "Name", "type": "string"}, "description": {"description": "Description of the segment", "title": "Description", "type": "string"}, "size": {"default": "medium", "description": "Size of the segment", "enum": ["short", "medium", "long"], "title": "Size", "type": "string"}}, "required": ["name", "description"], "title": "Segment", "type": "object"}}, "properties": {"segments": {"description": "List of segments", "items": {"$ref": "#/$defs/Segment"}, "title": "Segments", "type": "array"}}, "required": ["segments"]}"##;

const TRANSCRIPT_SCHEMA: &str = r##"{"$defs": {"Dialogue": {"properties": {"speaker": {"description": "Speaker name", "title": "Speaker", "type": "string"}, "dialogue": {"description": "Dialogue", "title": "Dialogue", "type": "string"}}, "required": ["speaker", "dialogue"], "title": "Dialogue", "type": "object"}}, "properties": {"transcript": {"description": "Transcript", "items": {"$ref": "#/$defs/Dialogue"}, "title": "Transcript", "type": "array"}}, "required": ["transcript"]}"##;

/// Which of the two embedded templates to render.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PromptTemplate {
    Outline,
    Transcript,
}

impl PromptTemplate {
    pub fn name(self) -> &'static str {
        match self {
            PromptTemplate::Outline => "outline",
            PromptTemplate::Transcript => "transcript",
        }
    }

    /// The template source, as embedded.
    pub fn source(self) -> &'static str {
        match self {
            PromptTemplate::Outline => OUTLINE_TEMPLATE,
            PromptTemplate::Transcript => TRANSCRIPT_TEMPLATE,
        }
    }

    /// The "Formatting instructions" block for this template: the JSON schema
    /// of the expected answer, worded as LangChain's Pydantic parser did.
    pub fn format_instructions(self) -> String {
        let schema = match self {
            PromptTemplate::Outline => OUTLINE_SCHEMA,
            PromptTemplate::Transcript => TRANSCRIPT_SCHEMA,
        };
        format!("{FORMAT_INSTRUCTIONS_HEAD}{schema}\n```")
    }
}

fn environment() -> &'static Environment<'static> {
    static ENVIRONMENT: OnceLock<Environment<'static>> = OnceLock::new();
    ENVIRONMENT.get_or_init(|| {
        let mut env = Environment::new();
        for template in [PromptTemplate::Outline, PromptTemplate::Transcript] {
            env.add_template(template.name(), template.source())
                .expect("embedded podcast prompt templates are valid");
        }
        env
    })
}

/// Render one of the embedded templates with an arbitrary context. The typed
/// [`OutlinePrompt`] and [`TranscriptPrompt`] build the context the pipeline
/// uses; this entry point exists for callers that need something else.
pub fn render_template<S: Serialize>(template: PromptTemplate, context: S) -> Result<String> {
    environment()
        .get_template(template.name())
        .and_then(|compiled| compiled.render(context))
        .map_err(|e| {
            PodcastError::Configuration(format!(
                "Failed to render the podcast {} prompt: {e}",
                template.name()
            ))
        })
}

fn speakers_context(speakers: &[Speaker]) -> Vec<Value> {
    speakers
        .iter()
        .map(|s| json!({"name": s.name.trim(), "backstory": s.backstory, "personality": s.personality}))
        .collect()
}

/// Inputs of the outline prompt.
#[derive(Debug, Clone)]
pub struct OutlinePrompt<'a> {
    pub briefing: &'a str,
    pub content: &'a Content,
    pub speakers: &'a [Speaker],
    pub num_segments: u32,
    /// Language name as shown to the model (`"Portuguese"`), not a code.
    pub language: Option<&'a str>,
}

impl OutlinePrompt<'_> {
    pub fn render(&self) -> Result<String> {
        render_template(
            PromptTemplate::Outline,
            json!({
                "briefing": self.briefing,
                "num_segments": self.num_segments,
                "context": self.content,
                "speakers": speakers_context(self.speakers),
                "language": self.language,
                "format_instructions": PromptTemplate::Outline.format_instructions(),
            }),
        )
    }
}

/// Inputs of the transcript prompt for one segment.
#[derive(Debug, Clone)]
pub struct TranscriptPrompt<'a> {
    pub briefing: &'a str,
    pub content: &'a Content,
    pub speakers: &'a [Speaker],
    pub outline: &'a Outline,
    pub segment: &'a Segment,
    pub is_final: bool,
    /// Dialogue written for the earlier segments.
    pub transcript: &'a [Dialogue],
    /// Language name as shown to the model (`"Portuguese"`), not a code.
    pub language: Option<&'a str>,
}

impl TranscriptPrompt<'_> {
    pub fn render(&self) -> Result<String> {
        let speaker_names: Vec<&str> = self.speakers.iter().map(|s| s.name.trim()).collect();
        render_template(
            PromptTemplate::Transcript,
            json!({
                "briefing": self.briefing,
                "outline": outline_text(self.outline),
                "context": self.content,
                "segment": segment_text(self.segment),
                "is_final": self.is_final,
                "turns": self.segment.size.turns(),
                "speakers": speakers_context(self.speakers),
                "speaker_names": speaker_names,
                "transcript": transcript_text(self.transcript),
                "language": self.language,
                "format_instructions": PromptTemplate::Transcript.format_instructions(),
            }),
        )
    }
}

/// The outline as the transcript prompt shows it: a numbered list.
fn outline_text(outline: &Outline) -> String {
    outline
        .segments
        .iter()
        .enumerate()
        .map(|(i, segment)| format!("{}. {}", i + 1, segment_text(segment)))
        .collect::<Vec<_>>()
        .join("\n")
}

fn segment_text(segment: &Segment) -> String {
    format!("{} ({}): {}", segment.name, segment.size.as_str(), segment.description)
}

/// Earlier dialogue as the transcript prompt shows it: one `Name: words` line
/// per turn. Empty when nothing has been written yet, which hides the block.
fn transcript_text(transcript: &[Dialogue]) -> String {
    transcript
        .iter()
        .map(|turn| format!("{}: {}", turn.speaker, turn.dialogue))
        .collect::<Vec<_>>()
        .join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::SegmentSize;

    fn speakers() -> Vec<Value> {
        vec![
            json!({"name": "Marcus Thompson", "backstory": "Former consultant", "personality": "Strategic"}),
            json!({"name": "Elena Vasquez", "backstory": "Serial entrepreneur", "personality": "Pragmatic"}),
        ]
    }

    fn names(speakers: &[Value]) -> Value {
        Value::Array(speakers.iter().map(|s| s["name"].clone()).collect())
    }

    fn render_transcript_with(speakers: Vec<Value>, overrides: Value) -> String {
        let mut data = json!({
            "briefing": "A briefing",
            "context": "Some content",
            "speaker_names": names(&speakers),
            "speakers": speakers,
            "outline": "An outline",
            "segment": "A segment",
            "turns": 6,
            "is_final": false,
            "transcript": [],
            "format_instructions": "Return JSON",
        });
        for (key, value) in overrides.as_object().unwrap() {
            data[key] = value.clone();
        }
        render_template(PromptTemplate::Transcript, data).unwrap()
    }

    fn render_transcript(overrides: Value) -> String {
        render_transcript_with(speakers(), overrides)
    }

    fn render_outline(overrides: Value) -> String {
        let mut data = json!({
            "briefing": "A briefing",
            "context": "Some content",
            "speakers": speakers(),
            "num_segments": 6,
            "format_instructions": "Return JSON",
        });
        for (key, value) in overrides.as_object().unwrap() {
            data[key] = value.clone();
        }
        render_template(PromptTemplate::Outline, data).unwrap()
    }

    type Renderer = fn(Value) -> String;
    const RENDERERS: [(&str, Renderer); 2] =
        [("transcript", render_transcript), ("outline", render_outline)];

    /// Parse the JSON example the prompt shows the model: it is what the model
    /// imitates, so it has to be valid JSON in its own right.
    fn example_object(rendered: &str, root_key: &str) -> Value {
        let prefix = format!("{{\"{root_key}\":");
        let line = rendered
            .lines()
            .find(|line| line.starts_with(&prefix))
            .unwrap_or_else(|| panic!("no {root_key} example found in the rendered prompt"));
        serde_json::from_str(line).unwrap()
    }

    fn example_speakers(example: &Value) -> Vec<String> {
        example["transcript"]
            .as_array()
            .unwrap()
            .iter()
            .map(|entry| entry["speaker"].as_str().unwrap().to_string())
            .collect()
    }

    // --- The example is valid and complete -------------------------------

    #[test]
    fn transcript_example_parses_and_names_the_speakers() {
        let example = example_object(&render_transcript(json!({})), "transcript");
        assert_eq!(example_speakers(&example), ["Marcus Thompson", "Elena Vasquez"]);
    }

    #[test]
    fn transcript_example_survives_json_special_characters() {
        let speakers = vec![
            json!({"name": "Dr. \"Alex\" Chen\\", "backstory": "b", "personality": "p"}),
            json!({"name": "Jamie\tRodriguez", "backstory": "b", "personality": "p"}),
        ];
        let example = example_object(&render_transcript_with(speakers, json!({})), "transcript");
        assert_eq!(example_speakers(&example), ["Dr. \"Alex\" Chen\\", "Jamie\tRodriguez"]);
    }

    #[test]
    fn transcript_example_dialogue_is_speakable() {
        let example = example_object(&render_transcript(json!({})), "transcript");
        for entry in example["transcript"].as_array().unwrap() {
            let dialogue = entry["dialogue"].as_str().unwrap();
            assert!(!dialogue.trim().is_empty());
            for marker in ["<", ">", "[", "]", "...", "\u{2026}", "TODO"] {
                assert!(!dialogue.contains(marker), "{marker} in {dialogue}");
            }
        }
    }

    #[test]
    fn outline_example_parses_with_valid_sizes() {
        let example = example_object(&render_outline(json!({})), "segments");
        let segments = example["segments"].as_array().unwrap();
        assert!(!segments.is_empty());
        for segment in segments {
            assert!(SegmentSize::parse(segment["size"].as_str().unwrap()).is_some());
            assert!(!segment["name"].as_str().unwrap().contains('<'));
            assert!(!segment["description"].as_str().unwrap().contains('<'));
        }
    }

    // --- The example cannot pass as a complete answer --------------------

    #[test]
    fn transcript_example_points_at_the_turn_minimum() {
        let rendered = render_transcript(json!({"turns": 6}));
        assert!(rendered.contains("two-entry EXCERPT"));
        assert!(rendered.contains("return at least 6 entries rather than the two shown"));
    }

    #[test]
    fn solo_excerpt_is_not_called_a_two_entry_one() {
        let rendered = render_transcript_with(speakers()[..1].to_vec(), json!({"turns": 3}));
        assert!(rendered.contains("one-entry EXCERPT"));
        assert!(!rendered.contains("two-entry"));
        assert!(rendered.contains("rather than the one shown"));
    }

    #[test]
    fn outline_example_points_at_the_segment_count() {
        let rendered = render_outline(json!({"num_segments": 6}));
        assert!(rendered.contains("two-entry EXCERPT"));
        assert!(rendered.contains("return all 6 of them rather than the two shown"));
    }

    // --- No English sample in a non-English run ---------------------------

    #[test]
    fn no_json_sample_when_a_language_is_set() {
        for (name, renderer) in RENDERERS {
            let rendered = renderer(json!({"language": "Hebrew"}));
            assert!(!rendered.contains("EXCERPT"), "{name}");
            assert!(!rendered.contains("Let's pick up where we left off"), "{name}");
            assert!(!rendered.contains("Setting the scene"), "{name}");
            assert!(!rendered.lines().any(|line| line.starts_with("{\"")), "{name}");
        }
    }

    #[test]
    fn transcript_still_states_the_structure_in_prose() {
        let rendered = render_transcript(json!({"turns": 6, "language": "Hebrew"}));
        assert!(rendered.contains("root key \"transcript\""));
        assert!(rendered.contains("at least 6 entries"));
        assert!(rendered.contains("\"speaker\" and \"dialogue\""));
        assert!(!rendered.contains("{{ speaker_names|join"));
        assert!(rendered.contains("Marcus Thompson, Elena Vasquez"));
    }

    #[test]
    fn outline_still_states_the_structure_in_prose() {
        let rendered = render_outline(json!({"num_segments": 6, "language": "Hebrew"}));
        assert!(rendered.contains("root key \"segments\""));
        assert!(rendered.contains("exactly 6 entries"));
        assert!(rendered.contains("\"size\" must be exactly one of \"short\", \"medium\" or \"long\""));
    }

    #[test]
    fn english_language_keeps_the_sample() {
        for (name, renderer) in RENDERERS {
            assert!(renderer(json!({"language": "English"})).contains("EXCERPT"), "{name}");
        }
    }

    #[test]
    fn language_instruction_keeps_the_json_contract_in_english() {
        for (name, renderer) in RENDERERS {
            assert!(
                renderer(json!({"language": "Hebrew"})).contains("except the JSON contract below"),
                "{name}"
            );
        }
    }

    // --- No copyable skeletons ---------------------------------------------

    const COPYABLE_SKELETONS: [&str; 11] = [
        "[Actual Speaker Name]",
        "[Speaker's dialogue based on their personality and expertise]",
        "[Segment Name]",
        "[Description of the segment content]",
        "{\"transcript\": [...]}",
        "{\"segments\": [...]}",
        "<the complete words this speaker says out loud, written out in full>",
        "<the real title of this segment>",
        "<the real title of the next segment>",
        "<what is discussed in this segment, including the key points and questions to cover>",
        "<what is discussed in that segment, including the key points and questions to cover>",
    ];

    #[test]
    fn prompts_have_no_placeholder_skeleton() {
        for (name, renderer) in RENDERERS {
            let rendered = renderer(json!({}));
            for skeleton in COPYABLE_SKELETONS {
                assert!(!rendered.contains(skeleton), "{name}: {skeleton}");
            }
        }
    }

    #[test]
    fn transcript_example_uses_real_speaker_names() {
        let rendered = render_transcript(json!({}));
        assert!(rendered.contains("{\"transcript\": [{\"speaker\": \"Marcus Thompson\""));
        assert!(rendered.contains("\"speaker\": \"Elena Vasquez\""));
    }

    #[test]
    fn solo_transcript_example_uses_the_only_speaker() {
        let rendered = render_transcript_with(speakers()[..1].to_vec(), json!({}));
        assert!(rendered.contains("{\"transcript\": [{\"speaker\": \"Marcus Thompson\""));
        assert!(!rendered.contains("Elena Vasquez"));
        assert!(rendered.contains("SOLO podcast"));
    }

    #[test]
    fn second_speaker_is_guarded_against_a_short_name_list() {
        let rendered = render_transcript(json!({"speaker_names": ["Marcus Thompson"]}));
        assert!(!rendered.contains("\"speaker\": \"\""));
        assert_eq!(rendered.matches("\"speaker\": \"Marcus Thompson\"").count(), 2);
    }

    #[test]
    fn prompts_ban_placeholder_content() {
        let transcript = render_transcript(json!({}));
        assert!(transcript.contains("Never emit placeholder or elided content"));
        assert!(transcript.contains("Never shorten or truncate the list"));
        let outline = render_outline(json!({}));
        assert!(outline.contains("Never emit placeholder or elided content"));
        assert!(outline.contains("never shorten or truncate the list"));
    }

    #[test]
    fn no_code_fence_contradicts_the_no_fence_instruction() {
        let no_fence_rule = "Do NOT wrap the JSON in ```json code blocks";
        for (name, renderer) in RENDERERS {
            let rendered = renderer(json!({}));
            assert!(rendered.contains(no_fence_rule), "{name}");
            assert!(!rendered.replace(no_fence_rule, "").contains("```"), "{name}");
        }
    }

    // --- Language instruction ----------------------------------------------

    #[test]
    fn transcript_includes_the_language_instruction() {
        let rendered = render_transcript(json!({"language": "Hebrew"}));
        assert!(rendered.contains("IMPORTANT LANGUAGE INSTRUCTION"));
        assert!(rendered.matches("Hebrew").count() >= 3);
    }

    #[test]
    fn outline_includes_the_language_instruction() {
        let rendered = render_outline(json!({"language": "Hebrew"}));
        assert!(rendered.contains("IMPORTANT LANGUAGE INSTRUCTION"));
        assert!(rendered.contains("segment names, descriptions"));
    }

    #[test]
    fn no_language_instruction_without_a_language() {
        for (name, renderer) in RENDERERS {
            assert!(!renderer(json!({})).contains("IMPORTANT LANGUAGE INSTRUCTION"), "{name}");
            assert!(
                !renderer(json!({"language": null})).contains("IMPORTANT LANGUAGE INSTRUCTION"),
                "{name}"
            );
        }
    }

    // --- Context forms -------------------------------------------------------

    #[test]
    fn context_may_be_one_text_or_several_pieces() {
        let text = render_outline(json!({"context": "Only one text"}));
        assert!(text.contains("Only one text"));
        assert!(!text.contains("<content_piece>"));

        let pieces = render_outline(json!({"context": ["First piece", "Second piece"]}));
        assert_eq!(pieces.matches("<content_piece>").count(), 2);
        assert!(pieces.contains("First piece") && pieces.contains("Second piece"));
    }

    // --- Typed prompts -------------------------------------------------------

    fn typed_speakers() -> Vec<Speaker> {
        vec![
            Speaker::new("Ana", "v1", "host", "curious"),
            Speaker::new("Bruno", "v2", "expert", "calm"),
        ]
    }

    #[test]
    fn typed_outline_prompt_renders_with_the_schema() {
        let speakers = typed_speakers();
        let content = Content::from("RAG combines retrieval and generation.");
        let text = OutlinePrompt {
            briefing: "Explain RAG",
            content: &content,
            speakers: &speakers,
            num_segments: 3,
            language: None,
        }
        .render()
        .unwrap();
        assert!(text.contains("Explain RAG"));
        assert!(text.contains("RAG combines retrieval and generation."));
        assert!(text.contains("- **Ana**: host"));
        assert!(text.contains("Personality: calm"));
        assert!(text.contains("<think>"));
        assert!(text.contains("\"required\": [\"segments\"]"));
        assert!(text.contains("Create 3 distinct segments"));
    }

    #[test]
    fn typed_transcript_prompt_renders_for_duo_and_solo() {
        let content = Content::from("RAG combines retrieval and generation.");
        let segment = Segment {
            name: "Intro".into(),
            description: "What RAG is".into(),
            size: SegmentSize::Short,
        };
        let outline = Outline { segments: vec![segment.clone()] };
        for speakers in [typed_speakers(), typed_speakers()[..1].to_vec()] {
            let text = TranscriptPrompt {
                briefing: "Explain RAG",
                content: &content,
                speakers: &speakers,
                outline: &outline,
                segment: &segment,
                is_final: true,
                transcript: &[],
                language: None,
            }
            .render()
            .unwrap();
            assert!(text.contains("1. Intro (short): What RAG is"));
            assert!(text.contains("at least 3"));
            assert!(text.contains("This is the final segment"));
            assert!(!text.contains("Here is the current transcript so far"));
            assert!(text.contains("\"required\": [\"transcript\"]"));
            for speaker in &speakers {
                assert!(text.contains(&speaker.name));
            }
            assert_eq!(text.contains("SOLO podcast"), speakers.len() == 1);
        }
    }

    #[test]
    fn typed_transcript_prompt_carries_the_running_transcript() {
        let content = Content::from("c");
        let speakers = typed_speakers();
        let segment =
            Segment { name: "Mid".into(), description: "More".into(), size: SegmentSize::Long };
        let outline = Outline { segments: vec![segment.clone(), segment.clone()] };
        let earlier = [Dialogue::new("Ana", "Welcome back."), Dialogue::new("Bruno", "Glad to be here.")];
        let text = TranscriptPrompt {
            briefing: "b",
            content: &content,
            speakers: &speakers,
            outline: &outline,
            segment: &segment,
            is_final: false,
            transcript: &earlier,
            language: Some("Portuguese"),
        }
        .render()
        .unwrap();
        assert!(text.contains("Here is the current transcript so far"));
        assert!(text.contains("Ana: Welcome back.\nBruno: Glad to be here."));
        assert!(text.contains("at least 10"));
        assert!(!text.contains("This is the final segment"));
        assert!(text.contains("You MUST generate ALL dialogue and content in Portuguese"));
    }

    #[test]
    fn embedded_templates_match_the_original_files() {
        // The crate ships copies; this fails when they drift from the Python
        // repository checked out next to the workspace (skipped when absent).
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../open-notebook/prompts/podcast");
        for template in [PromptTemplate::Outline, PromptTemplate::Transcript] {
            let Ok(original) = std::fs::read_to_string(root.join(format!("{}.jinja", template.name())))
            else {
                continue;
            };
            assert_eq!(template.source(), original, "{}", template.name());
        }
    }
}
