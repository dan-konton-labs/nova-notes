//! Domain types: profiles, episodes, outlines and dialogue.
//!
//! These are plain `serde` structs with the field names of the database
//! records in the original (`open_notebook/podcasts/models.py`). Nothing here
//! touches a database: loading and saving belong to the server crate. The
//! validation rules of the original live in the `validate*` and `require_*`
//! methods and return its error messages.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Deserializer, Serialize};
use serde_json::{Map, Value};

use crate::error::{PodcastError, Result};

/// Marker stored as a script episode's `episode_profile` snapshot: there is no
/// episode profile, so the UI and the retry endpoint key off this instead.
pub fn script_episode_profile() -> Map<String, Value> {
    let mut marker = Map::new();
    marker.insert("name".into(), Value::String("script".into()));
    marker.insert("mode".into(), Value::String("script".into()));
    marker
}

/// Briefing stored on an episode that was read from a script.
pub const SCRIPT_BRIEFING: &str = "Script read as written.";

const SPEAKER_REQUIRED_FIELDS: [&str; 4] = ["name", "voice_id", "backstory", "personality"];

fn default_num_segments() -> i64 {
    5
}

fn null_as_empty<'de, D: Deserializer<'de>>(deserializer: D) -> std::result::Result<String, D::Error> {
    Ok(Option::<String>::deserialize(deserializer)?.unwrap_or_default())
}

// ---------------------------------------------------------------------------
// Episode profile
// ---------------------------------------------------------------------------

/// Generation settings for a kind of episode (table `episode_profile`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EpisodeProfile {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    /// Unique profile name.
    pub name: String,
    #[serde(default)]
    pub description: Option<String>,
    /// `speaker_profile` record ID. `None` when the referenced speaker profile
    /// no longer exists.
    #[serde(default)]
    pub speaker_config: Option<String>,
    /// Model record ID for outline generation.
    #[serde(default)]
    pub outline_llm: Option<String>,
    /// Model record ID for transcript generation.
    #[serde(default)]
    pub transcript_llm: Option<String>,
    /// Podcast language as a BCP 47 code (`pt-BR`, `en-US`).
    #[serde(default)]
    pub language: Option<String>,
    pub default_briefing: String,
    #[serde(default = "default_num_segments")]
    pub num_segments: i64,
    /// Max output tokens for each outline / transcript call.
    #[serde(default)]
    pub max_tokens: Option<u32>,
}

impl EpisodeProfile {
    pub fn validate(&self) -> Result<()> {
        if !(3..=20).contains(&self.num_segments) {
            return Err(PodcastError::InvalidInput(
                "Number of segments must be between 3 and 20".into(),
            ));
        }
        Ok(())
    }

    /// The outline model reference, or the original's "not configured" error.
    pub fn require_outline_llm(&self) -> Result<&str> {
        non_empty(&self.outline_llm).ok_or_else(|| {
            PodcastError::InvalidInput(format!(
                "Episode profile '{}' has no outline model configured. \
                 Please update the profile to select an outline model.",
                self.name
            ))
        })
    }

    /// The transcript model reference, or the original's "not configured" error.
    pub fn require_transcript_llm(&self) -> Result<&str> {
        non_empty(&self.transcript_llm).ok_or_else(|| {
            PodcastError::InvalidInput(format!(
                "Episode profile '{}' has no transcript model configured. \
                 Please update the profile to select a transcript model.",
                self.name
            ))
        })
    }

    /// The speaker profile reference, or the original's "not configured" error.
    pub fn require_speaker_config(&self) -> Result<&str> {
        non_empty(&self.speaker_config).ok_or_else(|| {
            PodcastError::InvalidInput(format!(
                "Episode profile '{}' has no speaker profile configured. \
                 Please update the profile to select a speaker profile.",
                self.name
            ))
        })
    }

    /// The briefing sent to the models: the profile's default, plus the
    /// per-episode suffix when one was given.
    pub fn briefing(&self, briefing_suffix: Option<&str>) -> String {
        compose_briefing(&self.default_briefing, briefing_suffix)
    }
}

/// `default_briefing`, followed by `Additional instructions: <suffix>` when a
/// non-empty suffix is given.
pub fn compose_briefing(default_briefing: &str, briefing_suffix: Option<&str>) -> String {
    match briefing_suffix {
        Some(suffix) if !suffix.is_empty() => {
            format!("{default_briefing}\n\nAdditional instructions: {suffix}")
        }
        _ => default_briefing.to_string(),
    }
}

fn non_empty(value: &Option<String>) -> Option<&str> {
    value.as_deref().filter(|v| !v.is_empty())
}

// ---------------------------------------------------------------------------
// Speaker profile
// ---------------------------------------------------------------------------

/// One voice in a speaker profile.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Speaker {
    pub name: String,
    /// Provider voice id handed to the text-to-speech model.
    pub voice_id: String,
    #[serde(deserialize_with = "null_as_empty")]
    pub backstory: String,
    #[serde(deserialize_with = "null_as_empty")]
    pub personality: String,
    /// Model record ID overriding the profile's `voice_model` for this speaker.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub voice_model: Option<String>,
    /// Any other keys stored on the record, kept so a round trip loses nothing.
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

impl Speaker {
    pub fn new(
        name: impl Into<String>,
        voice_id: impl Into<String>,
        backstory: impl Into<String>,
        personality: impl Into<String>,
    ) -> Self {
        Self {
            name: name.into(),
            voice_id: voice_id.into(),
            backstory: backstory.into(),
            personality: personality.into(),
            voice_model: None,
            extra: Map::new(),
        }
    }
}

/// Voice and personality configuration for one to four speakers
/// (table `speaker_profile`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SpeakerProfile {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    /// Unique profile name.
    pub name: String,
    #[serde(default)]
    pub description: Option<String>,
    /// Model record ID for text to speech.
    #[serde(default)]
    pub voice_model: Option<String>,
    pub speakers: Vec<Speaker>,
}

impl SpeakerProfile {
    /// Build a profile from a JSON value (a request body or a database row),
    /// applying the original's validation with its messages: the speaker count
    /// first, then each speaker's required fields.
    pub fn from_value(value: Value) -> Result<Self> {
        if let Some(speakers) = value.get("speakers").and_then(Value::as_array) {
            validate_speaker_values(speakers)?;
        }
        let profile: SpeakerProfile = serde_json::from_value(value)
            .map_err(|e| PodcastError::InvalidInput(format!("Invalid speaker profile: {e}")))?;
        profile.validate()?;
        Ok(profile)
    }

    pub fn validate(&self) -> Result<()> {
        if !(1..=4).contains(&self.speakers.len()) {
            return Err(PodcastError::InvalidInput("Must have between 1 and 4 speakers".into()));
        }
        Ok(())
    }

    /// The stricter checks podcast-creator applied when a profile was used:
    /// non-empty and unique speaker names, unique voice ids.
    pub fn validate_for_generation(&self) -> Result<()> {
        self.validate()?;
        let mut names: Vec<&str> = Vec::new();
        let mut voices: Vec<&str> = Vec::new();
        for speaker in &self.speakers {
            let name = speaker.name.trim();
            if name.is_empty() {
                return Err(PodcastError::InvalidInput("Speaker name cannot be empty".into()));
            }
            if names.contains(&name) {
                return Err(PodcastError::InvalidInput("Speaker names must be unique".into()));
            }
            if voices.contains(&speaker.voice_id.as_str()) {
                return Err(PodcastError::InvalidInput("Voice IDs must be unique".into()));
            }
            names.push(name);
            voices.push(&speaker.voice_id);
        }
        Ok(())
    }

    /// The voice model reference, or the original's "not configured" error.
    pub fn require_voice_model(&self) -> Result<&str> {
        non_empty(&self.voice_model).ok_or_else(|| {
            PodcastError::InvalidInput(format!(
                "Speaker profile '{}' has no voice model configured. \
                 Please update the profile to select a voice model.",
                self.name
            ))
        })
    }

    /// Speaker names in profile order, trimmed.
    pub fn speaker_names(&self) -> Vec<String> {
        self.speakers.iter().map(|s| s.name.trim().to_string()).collect()
    }

    pub fn speaker_by_name(&self, name: &str) -> Result<&Speaker> {
        self.speakers
            .iter()
            .find(|s| s.name.trim() == name)
            .ok_or_else(|| PodcastError::InvalidInput(format!("Speaker '{name}' not found in profile")))
    }
}

/// The original's `validate_speakers` on raw JSON: between one and four
/// entries, each carrying the four required keys.
pub fn validate_speaker_values(speakers: &[Value]) -> Result<()> {
    if !(1..=4).contains(&speakers.len()) {
        return Err(PodcastError::InvalidInput("Must have between 1 and 4 speakers".into()));
    }
    for speaker in speakers {
        for field in SPEAKER_REQUIRED_FIELDS {
            if speaker.get(field).is_none() {
                return Err(PodcastError::InvalidInput(format!(
                    "Speaker missing required field: {field}"
                )));
            }
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Episode
// ---------------------------------------------------------------------------

/// A generated (or script-read) episode (table `episode`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PodcastEpisode {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    pub name: String,
    /// Snapshot of the episode profile used. Editing a profile never changes
    /// past episodes.
    pub episode_profile: Map<String, Value>,
    /// Snapshot of the speaker profile used.
    pub speaker_profile: Map<String, Value>,
    /// Full briefing used for generation. Never shown in the Podcast Feed.
    pub briefing: String,
    /// Source content (or the script, for a script episode).
    pub content: String,
    /// Audio path relative to the podcasts folder (see [`crate::audio_paths`]).
    #[serde(default)]
    pub audio_file: Option<String>,
    /// `{"transcript": [{"speaker", "dialogue"}, ...]}`.
    #[serde(default)]
    pub transcript: Option<Value>,
    /// `{"segments": [{"name", "description", "size"}, ...]}`.
    #[serde(default)]
    pub outline: Option<Value>,
    /// Background job record ID.
    #[serde(default)]
    pub command: Option<String>,
    /// Notebook the episode was generated from.
    #[serde(default)]
    pub notebook: Option<String>,
    /// Generation progress, 0 to 100.
    #[serde(default)]
    pub progress: Option<i64>,
    /// When the episode was put in the Podcast Feed; `None` means it is not a
    /// Published Episode. Doubles as the feed item's `pubDate`.
    #[serde(default)]
    pub published_at: Option<DateTime<Utc>>,
    /// Audio length, read when publishing.
    #[serde(default)]
    pub duration_seconds: Option<u64>,
    /// Episode description shown in podcast apps.
    #[serde(default)]
    pub feed_summary: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub created: Option<DateTime<Utc>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub updated: Option<DateTime<Utc>>,
}

impl PodcastEpisode {
    pub fn validate(&self) -> Result<()> {
        if let Some(progress) = self.progress {
            if !(0..=100).contains(&progress) {
                return Err(PodcastError::InvalidInput(
                    "Episode progress must be between 0 and 100".into(),
                ));
            }
        }
        Ok(())
    }

    /// The record key: the part of the id after `episode:`.
    pub fn key(&self) -> &str {
        let id = self.id.as_deref().unwrap_or("");
        id.split_once(':').map_or(id, |(_, key)| key)
    }

    /// True for an episode read from a script (no episode profile, no outline).
    pub fn is_script(&self) -> bool {
        self.episode_profile.get("mode").and_then(Value::as_str) == Some("script")
    }

    pub fn is_published(&self) -> bool {
        self.published_at.is_some()
    }
}

/// Published Episodes only, newest published first.
pub fn published_newest_first(episodes: Vec<PodcastEpisode>) -> Vec<PodcastEpisode> {
    let mut published: Vec<PodcastEpisode> =
        episodes.into_iter().filter(PodcastEpisode::is_published).collect();
    published.sort_by_key(|episode| std::cmp::Reverse(episode.published_at));
    published
}

// ---------------------------------------------------------------------------
// Outline and dialogue
// ---------------------------------------------------------------------------

/// How much dialogue a segment gets.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SegmentSize {
    Short,
    #[default]
    Medium,
    Long,
}

impl SegmentSize {
    /// Minimum number of dialogue turns asked of the transcript model.
    pub fn turns(self) -> u32 {
        match self {
            SegmentSize::Short => 3,
            SegmentSize::Medium => 6,
            SegmentSize::Long => 10,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            SegmentSize::Short => "short",
            SegmentSize::Medium => "medium",
            SegmentSize::Long => "long",
        }
    }

    pub(crate) fn parse(value: &str) -> Option<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "short" => Some(SegmentSize::Short),
            "medium" => Some(SegmentSize::Medium),
            "long" => Some(SegmentSize::Long),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Segment {
    pub name: String,
    pub description: String,
    #[serde(default)]
    pub size: SegmentSize,
}

#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct Outline {
    pub segments: Vec<Segment>,
}

impl Outline {
    /// The value stored in `PodcastEpisode::outline`.
    pub fn to_record(&self) -> Value {
        serde_json::to_value(self).unwrap_or(Value::Null)
    }
}

/// One spoken turn.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Dialogue {
    pub speaker: String,
    pub dialogue: String,
}

impl Dialogue {
    pub fn new(speaker: impl Into<String>, dialogue: impl Into<String>) -> Self {
        Self { speaker: speaker.into(), dialogue: dialogue.into() }
    }
}

/// The value stored in `PodcastEpisode::transcript`:
/// `{"transcript": [{"speaker", "dialogue"}, ...]}`.
pub fn transcript_record(transcript: &[Dialogue]) -> Value {
    let mut record = Map::new();
    record.insert("transcript".into(), serde_json::to_value(transcript).unwrap_or(Value::Null));
    Value::Object(record)
}

/// Source material for a generated episode: one text, or several pieces that
/// the prompts present separately.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum Content {
    Text(String),
    Pieces(Vec<String>),
}

impl From<String> for Content {
    fn from(text: String) -> Self {
        Content::Text(text)
    }
}

impl From<&str> for Content {
    fn from(text: &str) -> Self {
        Content::Text(text.to_string())
    }
}

impl From<Vec<String>> for Content {
    fn from(pieces: Vec<String>) -> Self {
        Content::Pieces(pieces)
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    fn speaker_json(name: &str, voice: &str) -> Value {
        json!({"name": name, "voice_id": voice, "backstory": "b", "personality": "p"})
    }

    fn message(result: Result<impl std::fmt::Debug>) -> String {
        result.unwrap_err().to_string()
    }

    #[test]
    fn episode_profile_accepts_max_tokens_and_defaults_to_none() {
        let with: EpisodeProfile = serde_json::from_value(json!({
            "name": "Long Form", "speaker_config": "default",
            "default_briefing": "Test briefing", "max_tokens": 12000
        }))
        .unwrap();
        assert_eq!(with.max_tokens, Some(12000));
        assert_eq!(with.num_segments, 5);
        with.validate().unwrap();

        let without: EpisodeProfile = serde_json::from_value(json!({
            "name": "Default", "speaker_config": "default", "default_briefing": "Test briefing"
        }))
        .unwrap();
        assert_eq!(without.max_tokens, None);
    }

    #[test]
    fn episode_profile_segment_limits() {
        let mut profile: EpisodeProfile =
            serde_json::from_value(json!({"name": "p", "default_briefing": "b"})).unwrap();
        for ok in [3, 20] {
            profile.num_segments = ok;
            profile.validate().unwrap();
        }
        for bad in [2, 21, -1] {
            profile.num_segments = bad;
            assert_eq!(message(profile.validate()), "Number of segments must be between 3 and 20");
        }
    }

    #[test]
    fn episode_profile_names_the_missing_model() {
        let profile: EpisodeProfile =
            serde_json::from_value(json!({"name": "Deep Dive", "default_briefing": "b"})).unwrap();
        assert_eq!(
            message(profile.require_outline_llm()),
            "Episode profile 'Deep Dive' has no outline model configured. \
             Please update the profile to select an outline model."
        );
        assert_eq!(
            message(profile.require_transcript_llm()),
            "Episode profile 'Deep Dive' has no transcript model configured. \
             Please update the profile to select a transcript model."
        );
        assert_eq!(
            message(profile.require_speaker_config()),
            "Episode profile 'Deep Dive' has no speaker profile configured. \
             Please update the profile to select a speaker profile."
        );
    }

    #[test]
    fn briefing_suffix_is_appended_as_additional_instructions() {
        assert_eq!(compose_briefing("Base", None), "Base");
        assert_eq!(compose_briefing("Base", Some("")), "Base");
        assert_eq!(
            compose_briefing("Base", Some("Focus on X")),
            "Base\n\nAdditional instructions: Focus on X"
        );
    }

    #[test]
    fn speaker_profile_count_limits() {
        let none = json!({"name": "p", "speakers": []});
        assert_eq!(message(SpeakerProfile::from_value(none)), "Must have between 1 and 4 speakers");

        let five: Vec<Value> = (0..5).map(|i| speaker_json(&format!("S{i}"), &format!("v{i}"))).collect();
        assert_eq!(
            message(SpeakerProfile::from_value(json!({"name": "p", "speakers": five}))),
            "Must have between 1 and 4 speakers"
        );

        let four: Vec<Value> = (0..4).map(|i| speaker_json(&format!("S{i}"), &format!("v{i}"))).collect();
        let profile = SpeakerProfile::from_value(json!({"name": "p", "speakers": four})).unwrap();
        assert_eq!(profile.speakers.len(), 4);
    }

    #[test]
    fn speaker_profile_names_the_missing_field() {
        for field in SPEAKER_REQUIRED_FIELDS {
            let mut speaker = speaker_json("Ana", "v1");
            speaker.as_object_mut().unwrap().remove(field);
            assert_eq!(
                message(SpeakerProfile::from_value(json!({"name": "p", "speakers": [speaker]}))),
                format!("Speaker missing required field: {field}")
            );
        }
    }

    #[test]
    fn speaker_profile_round_trips_overrides_and_unknown_keys() {
        let value = json!({
            "id": "speaker_profile:sp1",
            "name": "velvet_noir",
            "description": null,
            "voice_model": "model:tts1",
            "speakers": [
                {"name": "Velvet Noir", "voice_id": "af_bella", "backstory": null,
                 "personality": "p", "voice_model": "model:tts2", "legacy": 1},
                {"name": "Max", "voice_id": "am_adam", "backstory": "b", "personality": "p"}
            ]
        });
        let profile = SpeakerProfile::from_value(value).unwrap();
        assert_eq!(profile.speakers[0].backstory, "");
        assert_eq!(profile.speakers[0].voice_model.as_deref(), Some("model:tts2"));
        assert_eq!(profile.speakers[0].extra.get("legacy"), Some(&json!(1)));
        assert_eq!(profile.speaker_names(), ["Velvet Noir", "Max"]);
        assert_eq!(profile.speaker_by_name("Max").unwrap().voice_id, "am_adam");
        assert_eq!(
            message(profile.speaker_by_name("Nobody")),
            "Speaker 'Nobody' not found in profile"
        );

        let back = serde_json::to_value(&profile).unwrap();
        assert_eq!(back["speakers"][0]["legacy"], json!(1));
        assert_eq!(back["speakers"][1].get("voice_model"), None);
        assert_eq!(back["voice_model"], json!("model:tts1"));
    }

    #[test]
    fn speaker_profile_voice_model_is_required_for_generation() {
        let profile = SpeakerProfile::from_value(
            json!({"name": "duo", "speakers": [speaker_json("Ana", "v1")]}),
        )
        .unwrap();
        assert_eq!(
            message(profile.require_voice_model()),
            "Speaker profile 'duo' has no voice model configured. \
             Please update the profile to select a voice model."
        );
    }

    #[test]
    fn generation_needs_unique_names_and_voices() {
        let profile = |speakers: Vec<Value>| {
            SpeakerProfile::from_value(json!({"name": "p", "speakers": speakers})).unwrap()
        };
        profile(vec![speaker_json("Ana", "v1"), speaker_json("Bruno", "v2")])
            .validate_for_generation()
            .unwrap();
        assert_eq!(
            message(profile(vec![speaker_json("Ana", "v1"), speaker_json("Ana", "v2")])
                .validate_for_generation()),
            "Speaker names must be unique"
        );
        assert_eq!(
            message(profile(vec![speaker_json("Ana", "v1"), speaker_json("Bruno", "v1")])
                .validate_for_generation()),
            "Voice IDs must be unique"
        );
        assert_eq!(
            message(profile(vec![speaker_json("  ", "v1")]).validate_for_generation()),
            "Speaker name cannot be empty"
        );
    }

    fn episode(overrides: Value) -> PodcastEpisode {
        let mut base = json!({
            "id": "episode:abc",
            "name": "Ep",
            "episode_profile": {"name": "d"},
            "speaker_profile": {"name": "d"},
            "briefing": "b",
            "content": "c"
        });
        for (key, value) in overrides.as_object().unwrap() {
            base[key] = value.clone();
        }
        serde_json::from_value(base).unwrap()
    }

    #[test]
    fn episode_progress_range() {
        episode(json!({"progress": 40})).validate().unwrap();
        episode(json!({})).validate().unwrap();
        assert!(episode(json!({"progress": 101})).validate().is_err());
        assert!(episode(json!({"progress": -1})).validate().is_err());
    }

    #[test]
    fn episode_key_and_script_marker() {
        let generated = episode(json!({}));
        assert_eq!(generated.key(), "abc");
        assert!(!generated.is_script());

        let script = episode(json!({"episode_profile": Value::Object(script_episode_profile())}));
        assert!(script.is_script());
        assert_eq!(script.episode_profile["name"], json!("script"));
    }

    #[test]
    fn published_episodes_are_sorted_newest_first() {
        let episodes = vec![
            episode(json!({"id": "episode:old", "published_at": "2026-11-02T08:00:00Z"})),
            episode(json!({"id": "episode:draft"})),
            episode(json!({"id": "episode:new", "published_at": "2026-11-05T18:45:00Z"})),
        ];
        let keys: Vec<String> =
            published_newest_first(episodes).iter().map(|e| e.key().to_string()).collect();
        assert_eq!(keys, ["new", "old"]);
    }

    #[test]
    fn nullable_publication_fields_serialise_as_null() {
        let value = serde_json::to_value(episode(json!({}))).unwrap();
        for field in ["published_at", "duration_seconds", "feed_summary"] {
            assert_eq!(value.get(field), Some(&Value::Null), "{field}");
        }
    }

    #[test]
    fn outline_and_transcript_records_have_the_stored_shape() {
        let outline = Outline {
            segments: vec![Segment {
                name: "Intro".into(),
                description: "Hello".into(),
                size: SegmentSize::Short,
            }],
        };
        assert_eq!(
            outline.to_record(),
            json!({"segments": [{"name": "Intro", "description": "Hello", "size": "short"}]})
        );
        assert_eq!(
            transcript_record(&[Dialogue::new("Ana", "Hi")]),
            json!({"transcript": [{"speaker": "Ana", "dialogue": "Hi"}]})
        );
        assert_eq!(SegmentSize::Short.turns(), 3);
        assert_eq!(SegmentSize::Medium.turns(), 6);
        assert_eq!(SegmentSize::Long.turns(), 10);
    }
}
