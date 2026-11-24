//! Podcasts: episode and speaker profiles, episodes and the two generation
//! jobs (`generate_podcast`, `read_script_podcast`).
//!
//! The Rust counterpart of `api/podcast_service.py`, the profile and episode
//! routers' logic and `commands/podcast_commands.py`. The podcast engine
//! itself (outline, transcript, speech, audio, publishing rules) is
//! `nn_podcast`; this module loads and saves the records around it and turns
//! profile model references into clients through `state.models`.
//!
//! Records (same tables and fields as the Python build):
//! - `episode_profile`: `speaker_config`, `outline_llm`, `transcript_llm` are record links.
//! - `speaker_profile`: `voice_model` and each `speakers[*].voice_model` are record links.
//! - `episode`: profile snapshots, `command` / `notebook` links, `progress`,
//!   `audio_file` relative to the podcasts folder, publication state.

use std::collections::{BTreeMap, HashMap};
use std::path::PathBuf;
use std::time::{Duration, Instant};

use chrono::{DateTime, SecondsFormat, Utc};
use nn_podcast::audio_paths::{podcasts_root, realpath};
use nn_podcast::models::{script_episode_profile, transcript_record, validate_speaker_values, SCRIPT_BRIEFING};
use nn_podcast::publishing::audio_suffix;
use nn_podcast::{
    build_episode_output_dir, create_podcast, resolve_contained_audio_path, to_relative_audio_path, AudioOptions,
    FfmpegMode, GenerationInput, PodcastError, PodcastFailure, PodcastOutput, PodcastRequest, PodcastSource,
    ProgressUpdate, RefusalPolicy, TtsModels,
};
use serde::{Deserialize, Deserializer, Serialize};
use serde_json::{json, Map};
use surrealdb::sql::Value as SqlValue;

use crate::ai::AnyModel;
use crate::db::{from_json, table_of, thing, with_table, Doc, Json, Params};
use crate::domain::{link_id, null_default, Model, Notebook};
use crate::error::{AppError, AppResult};
use crate::jobs::{JobContext, JobQueue, RetryPolicy};
use crate::state::AppState;

/// Command names, as the Python build registers them.
pub const GENERATE_PODCAST: &str = "generate_podcast";
pub const READ_SCRIPT_PODCAST: &str = "read_script_podcast";

const EPISODE_PROFILE_TABLE: &str = "episode_profile";
const SPEAKER_PROFILE_TABLE: &str = "speaker_profile";
const EPISODE_TABLE: &str = "episode";

/// Register the two podcast jobs. Both run once: a second run would make a
/// second episode (the Python commands use `max_attempts: 1`).
pub fn register_jobs(jobs: &JobQueue) {
    jobs.register(GENERATE_PODCAST, RetryPolicy::once(), |ctx, args| async move { run_generate_job(ctx, args).await });
    jobs.register(READ_SCRIPT_PODCAST, RetryPolicy::once(), |ctx, args| async move { run_script_job(ctx, args).await });
}

// ---------------------------------------------------------------------------
// Errors and small helpers
// ---------------------------------------------------------------------------

/// `PodcastError` → `AppError`, by the status the Python API used for the
/// matching exception (ValueError / InvalidInputError → 400, ...).
pub(crate) fn podcast_error(error: PodcastError) -> AppError {
    podcast_error_with_message(&error, error.to_string())
}

/// A failed generation, with its hint (`NOTE: ...`) in the message.
pub(crate) fn failure_error(failure: &PodcastFailure) -> AppError {
    podcast_error_with_message(&failure.error, failure.message())
}

fn podcast_error_with_message(error: &PodcastError, message: String) -> AppError {
    match error {
        PodcastError::InvalidInput(_) => AppError::InvalidInput(message),
        PodcastError::Configuration(_) => AppError::Configuration(message),
        PodcastError::Refusal(_) | PodcastError::InvalidOutput(_) => AppError::ExternalService(message),
        PodcastError::Incomplete(_) => AppError::IncompleteGeneration(message),
        PodcastError::Ai(ai) => match AppError::from(ai.clone()) {
            AppError::Authentication(_) => AppError::Authentication(message),
            AppError::RateLimit(_) => AppError::RateLimit(message),
            AppError::ContextLength(_) => AppError::ContextLength(message),
            AppError::Configuration(_) => AppError::Configuration(message),
            AppError::Network(_) => AppError::Network(message),
            AppError::IncompleteGeneration(_) => AppError::IncompleteGeneration(message),
            _ => AppError::ExternalService(message),
        },
        PodcastError::Audio(_) => AppError::Internal(message),
        PodcastError::Io { .. } => AppError::FileOperation(message),
    }
}

/// Deserialize a field that may be absent (`None`), null (`Some(None)`) or set.
pub(crate) fn provided<'de, D, T>(deserializer: D) -> Result<Option<Option<T>>, D::Error>
where
    D: Deserializer<'de>,
    T: Deserialize<'de>,
{
    Option::<T>::deserialize(deserializer).map(Some)
}

/// Python's `datetime.isoformat()` for an aware UTC value.
pub(crate) fn iso(at: &DateTime<Utc>) -> String {
    at.to_rfc3339_opts(SecondsFormat::Micros, false)
}

fn non_empty(value: Option<&str>) -> Option<&str> {
    value.map(str::trim).filter(|v| !v.is_empty())
}

/// `id` as a record id of `table`; an id of any other table (or a malformed
/// one) is `None`.
fn id_in_table(table: &str, id: &str) -> Option<String> {
    let id = with_table(table, id.trim());
    (table_of(&id) == Some(table) && thing(&id).is_ok()).then_some(id)
}

/// A model reference from a request: empty means "none"; anything else must
/// be a `model:` record id.
fn model_ref(value: Option<&str>) -> AppResult<Option<String>> {
    match non_empty(value) {
        None => Ok(None),
        Some(id) if table_of(id) == Some(Model::TABLE) && thing(id).is_ok() => Ok(Some(id.to_string())),
        Some(id) => Err(AppError::invalid(format!("Invalid model id: {id:?}"))),
    }
}

fn env_u32(name: &str) -> Option<u32> {
    crate::config::env_nonempty(name).and_then(|v| v.trim().parse::<u32>().ok())
}

// ---------------------------------------------------------------------------
// Episode profiles
// ---------------------------------------------------------------------------

fn default_num_segments() -> i64 {
    5
}

/// An `episode_profile` row.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct EpisodeProfileRecord {
    #[serde(default)]
    pub id: String,
    #[serde(default, deserialize_with = "null_default")]
    pub name: String,
    #[serde(default)]
    pub description: Option<String>,
    /// `speaker_profile:<id>`, `None` when orphaned.
    #[serde(default, deserialize_with = "link_id")]
    pub speaker_config: Option<String>,
    #[serde(default, deserialize_with = "link_id")]
    pub outline_llm: Option<String>,
    #[serde(default, deserialize_with = "link_id")]
    pub transcript_llm: Option<String>,
    #[serde(default)]
    pub language: Option<String>,
    #[serde(default, deserialize_with = "null_default")]
    pub default_briefing: String,
    #[serde(default = "default_num_segments")]
    pub num_segments: i64,
    #[serde(default)]
    pub max_tokens: Option<i64>,
}

impl EpisodeProfileRecord {
    fn validate(&self) -> AppResult<()> {
        if !(3..=20).contains(&self.num_segments) {
            return Err(AppError::invalid("Number of segments must be between 3 and 20"));
        }
        Ok(())
    }

    fn doc(&self) -> Doc {
        let text = |doc: Doc, key: &str, value: &Option<String>| match value {
            Some(v) => doc.set(key, v),
            None => doc.none(key),
        };
        let doc = Doc::new().set("name", &self.name);
        let doc = text(doc, "description", &self.description);
        let doc = doc
            .record_or_none("speaker_config", self.speaker_config.as_deref())
            .record_or_none("outline_llm", self.outline_llm.as_deref())
            .record_or_none("transcript_llm", self.transcript_llm.as_deref());
        let doc = text(doc, "language", &self.language);
        let doc = doc.set("default_briefing", &self.default_briefing).set("num_segments", self.num_segments);
        match self.max_tokens {
            Some(max) => doc.set("max_tokens", max),
            None => doc.none("max_tokens"),
        }
    }

    /// The engine's view of the profile.
    pub fn to_engine(&self) -> nn_podcast::EpisodeProfile {
        nn_podcast::EpisodeProfile {
            id: Some(self.id.clone()),
            name: self.name.clone(),
            description: self.description.clone(),
            speaker_config: self.speaker_config.clone(),
            outline_llm: self.outline_llm.clone(),
            transcript_llm: self.transcript_llm.clone(),
            language: self.language.clone(),
            default_briefing: self.default_briefing.clone(),
            num_segments: self.num_segments,
            max_tokens: self.max_tokens.and_then(|m| u32::try_from(m).ok()),
        }
    }
}

/// `EpisodeProfileResponse`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EpisodeProfileView {
    pub id: String,
    pub name: String,
    pub description: String,
    /// `speaker_profile` record id (null when orphaned).
    pub speaker_config: Option<String>,
    /// The speaker profile's name, for display.
    pub speaker_config_name: Option<String>,
    pub outline_llm: Option<String>,
    pub transcript_llm: Option<String>,
    pub language: Option<String>,
    pub default_briefing: String,
    pub num_segments: i64,
    pub max_tokens: Option<i64>,
}

impl EpisodeProfileView {
    fn new(profile: EpisodeProfileRecord, speaker_name: Option<String>) -> Self {
        Self {
            id: profile.id,
            name: profile.name,
            description: profile.description.unwrap_or_default(),
            speaker_config: profile.speaker_config,
            speaker_config_name: speaker_name,
            outline_llm: profile.outline_llm,
            transcript_llm: profile.transcript_llm,
            language: profile.language,
            default_briefing: profile.default_briefing,
            num_segments: profile.num_segments,
            max_tokens: profile.max_tokens,
        }
    }
}

/// `EpisodeProfileCreate`: the body of create and update. On update only the
/// fields present in the body change (`description`, `num_segments` and the
/// optional fields may be left out).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct EpisodeProfileInput {
    pub name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// A `speaker_profile` record id, or (for backward compatibility) a profile name.
    pub speaker_config: String,
    #[serde(default, deserialize_with = "provided", skip_serializing_if = "Option::is_none")]
    pub outline_llm: Option<Option<String>>,
    #[serde(default, deserialize_with = "provided", skip_serializing_if = "Option::is_none")]
    pub transcript_llm: Option<Option<String>>,
    #[serde(default, deserialize_with = "provided", skip_serializing_if = "Option::is_none")]
    pub language: Option<Option<String>>,
    pub default_briefing: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub num_segments: Option<i64>,
    #[serde(default, deserialize_with = "provided", skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<Option<i64>>,
}

async fn episode_profile_rows(state: &AppState, sql: &str, params: Params) -> AppResult<Vec<EpisodeProfileRecord>> {
    let rows = state.db.query(sql, params).await?;
    Ok(rows
        .into_iter()
        .filter_map(|row| match serde_json::from_value(row) {
            Ok(profile) => Some(profile),
            Err(e) => {
                tracing::error!("Skipping an unreadable episode profile: {e}");
                None
            }
        })
        .collect())
}

/// The episode profile with this name, if any.
pub async fn find_episode_profile(state: &AppState, name: &str) -> AppResult<Option<EpisodeProfileRecord>> {
    Ok(episode_profile_rows(state, "SELECT * FROM episode_profile WHERE name = $name", Params::new().set("name", name))
        .await?
        .into_iter()
        .next())
}

/// The episode profile with this record id. Missing → 404 with the Python message.
pub async fn load_episode_profile(state: &AppState, profile_id: &str) -> AppResult<EpisodeProfileRecord> {
    let missing = || AppError::not_found(format!("{EPISODE_PROFILE_TABLE} with id {profile_id} not found"));
    let id = id_in_table(EPISODE_PROFILE_TABLE, profile_id).ok_or_else(missing)?;
    match state.db.select(&id).await? {
        Some(row) => Ok(serde_json::from_value(row)?),
        None => Err(missing()),
    }
}

async fn speaker_name(state: &AppState, speaker_config: Option<&str>) -> AppResult<Option<String>> {
    match speaker_config {
        Some(reference) => Ok(resolve_speaker_profile(state, reference).await?.map(|s| s.name)),
        None => Ok(None),
    }
}

/// Every episode profile, by name.
pub async fn list_episode_profiles(state: &AppState) -> AppResult<Vec<EpisodeProfileView>> {
    let profiles = episode_profile_rows(state, "SELECT * FROM episode_profile ORDER BY name ASC", Params::new()).await?;
    let names: HashMap<String, String> = state
        .db
        .query("SELECT id, name FROM speaker_profile", Params::new())
        .await?
        .into_iter()
        .filter_map(|row| Some((row["id"].as_str()?.to_string(), row["name"].as_str()?.to_string())))
        .collect();
    Ok(profiles
        .into_iter()
        .map(|profile| {
            let name = profile.speaker_config.as_ref().and_then(|id| names.get(id)).cloned();
            EpisodeProfileView::new(profile, name)
        })
        .collect())
}

/// One episode profile by name. Missing → 404 `Episode profile '<name>' not found`.
pub async fn get_episode_profile(state: &AppState, name: &str) -> AppResult<EpisodeProfileView> {
    let profile = find_episode_profile(state, name)
        .await?
        .ok_or_else(|| AppError::not_found(format!("Episode profile '{name}' not found")))?;
    let speaker = speaker_name(state, profile.speaker_config.as_deref()).await?;
    Ok(EpisodeProfileView::new(profile, speaker))
}

/// An incoming `speaker_config` (record id or name) → the speaker profile. Unknown → 400.
async fn require_speaker_config(state: &AppState, value: &str) -> AppResult<SpeakerProfileRecord> {
    resolve_speaker_profile(state, value)
        .await?
        .ok_or_else(|| AppError::invalid(format!("Speaker profile '{value}' not found")))
}

fn saved_episode_profile(row: Json) -> AppResult<EpisodeProfileRecord> {
    Ok(serde_json::from_value(row)?)
}

pub async fn create_episode_profile(state: &AppState, input: &EpisodeProfileInput) -> AppResult<EpisodeProfileView> {
    let speaker = require_speaker_config(state, &input.speaker_config).await?;
    let profile = EpisodeProfileRecord {
        id: String::new(),
        name: input.name.clone(),
        description: Some(input.description.clone().unwrap_or_default()),
        speaker_config: Some(speaker.id.clone()),
        outline_llm: model_ref(input.outline_llm.clone().flatten().as_deref())?,
        transcript_llm: model_ref(input.transcript_llm.clone().flatten().as_deref())?,
        language: input.language.clone().flatten().filter(|l| !l.trim().is_empty()),
        default_briefing: input.default_briefing.clone(),
        num_segments: input.num_segments.unwrap_or_else(default_num_segments),
        max_tokens: input.max_tokens.flatten(),
    };
    profile.validate()?;
    let saved = saved_episode_profile(state.db.create(EPISODE_PROFILE_TABLE, profile.doc()).await?)?;
    Ok(EpisodeProfileView::new(saved, Some(speaker.name)))
}

pub async fn update_episode_profile(
    state: &AppState,
    profile_id: &str,
    input: &EpisodeProfileInput,
) -> AppResult<EpisodeProfileView> {
    let mut profile = load_episode_profile(state, profile_id).await?;
    let speaker = require_speaker_config(state, &input.speaker_config).await?;
    profile.name = input.name.clone();
    profile.speaker_config = Some(speaker.id.clone());
    profile.default_briefing = input.default_briefing.clone();
    if let Some(description) = &input.description {
        profile.description = Some(description.clone());
    }
    if let Some(value) = &input.outline_llm {
        profile.outline_llm = model_ref(value.as_deref())?;
    }
    if let Some(value) = &input.transcript_llm {
        profile.transcript_llm = model_ref(value.as_deref())?;
    }
    if let Some(value) = &input.language {
        profile.language = value.clone().filter(|l| !l.trim().is_empty());
    }
    if let Some(segments) = input.num_segments {
        profile.num_segments = segments;
    }
    if let Some(max_tokens) = input.max_tokens {
        profile.max_tokens = max_tokens;
    }
    profile.validate()?;
    let saved = saved_episode_profile(state.db.update(&profile.id, profile.doc()).await?)?;
    Ok(EpisodeProfileView::new(saved, Some(speaker.name)))
}

pub async fn delete_episode_profile(state: &AppState, profile_id: &str) -> AppResult<()> {
    let profile = load_episode_profile(state, profile_id).await?;
    state.db.delete(&profile.id).await
}

/// A copy named `<name> - Copy`.
pub async fn duplicate_episode_profile(state: &AppState, profile_id: &str) -> AppResult<EpisodeProfileView> {
    let original = load_episode_profile(state, profile_id).await?;
    let copy = EpisodeProfileRecord { id: String::new(), name: format!("{} - Copy", original.name), ..original };
    let saved = saved_episode_profile(state.db.create(EPISODE_PROFILE_TABLE, copy.doc()).await?)?;
    let speaker = speaker_name(state, saved.speaker_config.as_deref()).await?;
    Ok(EpisodeProfileView::new(saved, speaker))
}

// ---------------------------------------------------------------------------
// Speaker profiles
// ---------------------------------------------------------------------------

/// A `speaker_profile` row. `speakers` stay raw JSON objects, as stored.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SpeakerProfileRecord {
    #[serde(default)]
    pub id: String,
    #[serde(default, deserialize_with = "null_default")]
    pub name: String,
    #[serde(default)]
    pub description: Option<String>,
    #[serde(default, deserialize_with = "link_id")]
    pub voice_model: Option<String>,
    #[serde(default, deserialize_with = "null_default")]
    pub speakers: Vec<Map<String, Json>>,
}

impl SpeakerProfileRecord {
    fn validate(&self) -> AppResult<()> {
        let speakers: Vec<Json> = self.speakers.iter().cloned().map(Json::Object).collect();
        validate_speaker_values(&speakers).map_err(podcast_error)
    }

    /// Speakers for the database: each `voice_model` override bound as a record link.
    fn speakers_value(&self) -> AppResult<SqlValue> {
        let mut values = Vec::with_capacity(self.speakers.len());
        for speaker in &self.speakers {
            let mut fields: BTreeMap<String, SqlValue> = BTreeMap::new();
            for (key, value) in speaker {
                if key == "voice_model" {
                    if let Some(id) = model_ref(value.as_str())? {
                        fields.insert(key.clone(), SqlValue::Thing(thing(&id)?));
                    }
                } else {
                    fields.insert(key.clone(), from_json(value.clone()));
                }
            }
            values.push(SqlValue::from(fields));
        }
        Ok(SqlValue::from(values))
    }

    fn doc(&self) -> AppResult<Doc> {
        let doc = Doc::new().set("name", &self.name);
        let doc = match &self.description {
            Some(description) => doc.set("description", description),
            None => doc.none("description"),
        };
        Ok(doc.record_or_none("voice_model", self.voice_model.as_deref()).raw("speakers", self.speakers_value()?))
    }

    /// The engine's view, with the original's checks on the speakers.
    pub fn to_engine(&self) -> AppResult<nn_podcast::SpeakerProfile> {
        nn_podcast::SpeakerProfile::from_value(serde_json::to_value(self)?).map_err(podcast_error)
    }
}

/// `SpeakerProfileResponse`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SpeakerProfileView {
    pub id: String,
    pub name: String,
    pub description: String,
    pub voice_model: Option<String>,
    pub speakers: Vec<Map<String, Json>>,
}

impl From<SpeakerProfileRecord> for SpeakerProfileView {
    fn from(profile: SpeakerProfileRecord) -> Self {
        Self {
            id: profile.id,
            name: profile.name,
            description: profile.description.unwrap_or_default(),
            voice_model: profile.voice_model,
            speakers: profile.speakers,
        }
    }
}

/// `SpeakerProfileCreate`: the body of create and update.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SpeakerProfileInput {
    pub name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(default, deserialize_with = "provided", skip_serializing_if = "Option::is_none")]
    pub voice_model: Option<Option<String>>,
    pub speakers: Vec<Map<String, Json>>,
}

async fn speaker_profile_rows(state: &AppState, sql: &str, params: Params) -> AppResult<Vec<SpeakerProfileRecord>> {
    let rows = state.db.query(sql, params).await?;
    Ok(rows
        .into_iter()
        .filter_map(|row| match serde_json::from_value(row) {
            Ok(profile) => Some(profile),
            Err(e) => {
                tracing::error!("Skipping an unreadable speaker profile: {e}");
                None
            }
        })
        .collect())
}

/// A speaker profile by record id (`speaker_profile:...`) or by unique name.
pub async fn resolve_speaker_profile(state: &AppState, reference: &str) -> AppResult<Option<SpeakerProfileRecord>> {
    if reference.starts_with("speaker_profile:") {
        if thing(reference).is_err() {
            return Ok(None);
        }
        return match state.db.select(reference).await? {
            Some(row) => Ok(Some(serde_json::from_value(row)?)),
            None => Ok(None),
        };
    }
    Ok(speaker_profile_rows(state, "SELECT * FROM speaker_profile WHERE name = $name", Params::new().set("name", reference))
        .await?
        .into_iter()
        .next())
}

/// The speaker profile with this record id. Missing → 404 with the Python message.
pub async fn load_speaker_profile(state: &AppState, profile_id: &str) -> AppResult<SpeakerProfileRecord> {
    let missing = || AppError::not_found(format!("{SPEAKER_PROFILE_TABLE} with id {profile_id} not found"));
    let id = id_in_table(SPEAKER_PROFILE_TABLE, profile_id).ok_or_else(missing)?;
    match state.db.select(&id).await? {
        Some(row) => Ok(serde_json::from_value(row)?),
        None => Err(missing()),
    }
}

pub async fn list_speaker_profiles(state: &AppState) -> AppResult<Vec<SpeakerProfileView>> {
    Ok(speaker_profile_rows(state, "SELECT * FROM speaker_profile ORDER BY name ASC", Params::new())
        .await?
        .into_iter()
        .map(SpeakerProfileView::from)
        .collect())
}

/// One speaker profile by name. Missing → 404 `Speaker profile '<name>' not found`.
pub async fn get_speaker_profile(state: &AppState, name: &str) -> AppResult<SpeakerProfileView> {
    speaker_profile_rows(state, "SELECT * FROM speaker_profile WHERE name = $name", Params::new().set("name", name))
        .await?
        .into_iter()
        .next()
        .map(SpeakerProfileView::from)
        .ok_or_else(|| AppError::not_found(format!("Speaker profile '{name}' not found")))
}

fn saved_speaker_profile(row: Json) -> AppResult<SpeakerProfileView> {
    Ok(serde_json::from_value::<SpeakerProfileRecord>(row)?.into())
}

pub async fn create_speaker_profile(state: &AppState, input: &SpeakerProfileInput) -> AppResult<SpeakerProfileView> {
    let profile = SpeakerProfileRecord {
        id: String::new(),
        name: input.name.clone(),
        description: Some(input.description.clone().unwrap_or_default()),
        voice_model: model_ref(input.voice_model.clone().flatten().as_deref())?,
        speakers: input.speakers.clone(),
    };
    profile.validate()?;
    saved_speaker_profile(state.db.create(SPEAKER_PROFILE_TABLE, profile.doc()?).await?)
}

pub async fn update_speaker_profile(
    state: &AppState,
    profile_id: &str,
    input: &SpeakerProfileInput,
) -> AppResult<SpeakerProfileView> {
    let mut profile = load_speaker_profile(state, profile_id).await?;
    profile.name = input.name.clone();
    profile.speakers = input.speakers.clone();
    if let Some(description) = &input.description {
        profile.description = Some(description.clone());
    }
    if let Some(voice_model) = &input.voice_model {
        profile.voice_model = model_ref(voice_model.as_deref())?;
    }
    profile.validate()?;
    saved_speaker_profile(state.db.update(&profile.id, profile.doc()?).await?)
}

pub async fn delete_speaker_profile(state: &AppState, profile_id: &str) -> AppResult<()> {
    let profile = load_speaker_profile(state, profile_id).await?;
    state.db.delete(&profile.id).await
}

/// A copy named `<name> - Copy`.
pub async fn duplicate_speaker_profile(state: &AppState, profile_id: &str) -> AppResult<SpeakerProfileView> {
    let original = load_speaker_profile(state, profile_id).await?;
    let copy = SpeakerProfileRecord { id: String::new(), name: format!("{} - Copy", original.name), ..original };
    saved_speaker_profile(state.db.create(SPEAKER_PROFILE_TABLE, copy.doc()?).await?)
}

// ---------------------------------------------------------------------------
// Episodes
// ---------------------------------------------------------------------------

/// An `episode` row, read leniently (legacy rows may lack fields).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct EpisodeRecord {
    #[serde(default)]
    pub id: String,
    #[serde(default, deserialize_with = "null_default")]
    pub name: String,
    #[serde(default, deserialize_with = "null_default")]
    pub episode_profile: Map<String, Json>,
    #[serde(default, deserialize_with = "null_default")]
    pub speaker_profile: Map<String, Json>,
    #[serde(default, deserialize_with = "null_default")]
    pub briefing: String,
    #[serde(default, deserialize_with = "null_default")]
    pub content: String,
    #[serde(default)]
    pub audio_file: Option<String>,
    #[serde(default)]
    pub transcript: Option<Json>,
    #[serde(default)]
    pub outline: Option<Json>,
    #[serde(default, deserialize_with = "link_id")]
    pub command: Option<String>,
    #[serde(default, deserialize_with = "link_id")]
    pub notebook: Option<String>,
    #[serde(default)]
    pub progress: Option<i64>,
    #[serde(default)]
    pub published_at: Option<DateTime<Utc>>,
    #[serde(default)]
    pub duration_seconds: Option<u64>,
    #[serde(default)]
    pub feed_summary: Option<String>,
    #[serde(default)]
    pub created: Option<DateTime<Utc>>,
    #[serde(default)]
    pub updated: Option<DateTime<Utc>>,
}

impl EpisodeRecord {
    /// The engine's view of the episode (publishing rules, feed items).
    pub fn to_engine(&self) -> nn_podcast::PodcastEpisode {
        nn_podcast::PodcastEpisode {
            id: Some(self.id.clone()),
            name: self.name.clone(),
            episode_profile: self.episode_profile.clone(),
            speaker_profile: self.speaker_profile.clone(),
            briefing: self.briefing.clone(),
            content: self.content.clone(),
            audio_file: self.audio_file.clone(),
            transcript: self.transcript.clone(),
            outline: self.outline.clone(),
            command: self.command.clone(),
            notebook: self.notebook.clone(),
            progress: self.progress,
            published_at: self.published_at,
            duration_seconds: self.duration_seconds,
            feed_summary: self.feed_summary.clone(),
            created: self.created,
            updated: self.updated,
        }
    }

    pub fn is_script(&self) -> bool {
        self.episode_profile.get("mode").and_then(Json::as_str) == Some("script")
    }
}

fn episode_not_found() -> AppError {
    AppError::not_found("Episode not found")
}

/// An episode by record id. Missing (or an id of another table) → 404 `Episode not found`.
pub async fn load_episode(state: &AppState, episode_id: &str) -> AppResult<EpisodeRecord> {
    let id = id_in_table(EPISODE_TABLE, episode_id).ok_or_else(episode_not_found)?;
    match state.db.select(&id).await? {
        Some(row) => Ok(serde_json::from_value(row)?),
        None => Err(episode_not_found()),
    }
}

/// An episode by its key (the part after `episode:`), `None` when missing.
pub(crate) async fn find_episode_by_key(state: &AppState, key: &str) -> AppResult<Option<EpisodeRecord>> {
    let id = format!("{EPISODE_TABLE}:{key}");
    if thing(&id).is_err() {
        return Ok(None);
    }
    match state.db.select(&id).await? {
        Some(row) => Ok(Some(serde_json::from_value(row)?)),
        None => Ok(None),
    }
}

/// Every episode with `published_at` set (the feed sorts them).
pub(crate) async fn published_episodes(state: &AppState) -> AppResult<Vec<EpisodeRecord>> {
    state.db.query_as("SELECT * FROM episode WHERE published_at != NONE", Params::new()).await
}

/// `PodcastEpisodeResponse`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EpisodeView {
    pub id: String,
    pub name: String,
    /// The profile snapshot, plus `outline_model_provider` / `_name` and
    /// `transcript_model_provider` / `_name` when the references resolve.
    pub episode_profile: Map<String, Json>,
    /// The profile snapshot, plus `voice_model_provider` / `_name` when it resolves.
    pub speaker_profile: Map<String, Json>,
    pub briefing: String,
    pub audio_file: Option<String>,
    /// `/api/podcasts/episodes/<id>/audio` while the file is there.
    pub audio_url: Option<String>,
    pub transcript: Option<Json>,
    pub outline: Option<Json>,
    pub created: Option<String>,
    pub job_status: Option<String>,
    pub error_message: Option<String>,
    pub notebook_id: Option<String>,
    pub progress: Option<i64>,
    pub published_at: Option<String>,
    pub duration_seconds: Option<u64>,
    pub feed_summary: Option<String>,
}

const EPISODE_PROFILE_MODEL_FIELDS: [(&str, &str, &str); 2] = [
    ("outline_llm", "outline_model_provider", "outline_model_name"),
    ("transcript_llm", "transcript_model_provider", "transcript_model_name"),
];
const SPEAKER_PROFILE_MODEL_FIELDS: [(&str, &str, &str); 1] = [("voice_model", "voice_model_provider", "voice_model_name")];

fn snapshot_model_ids(episodes: &[EpisodeRecord]) -> Vec<String> {
    let mut ids = Vec::new();
    for episode in episodes {
        for (field, _, _) in EPISODE_PROFILE_MODEL_FIELDS {
            if let Some(id) = episode.episode_profile.get(field).and_then(Json::as_str) {
                ids.push(id.to_string());
            }
        }
        for (field, _, _) in SPEAKER_PROFILE_MODEL_FIELDS {
            if let Some(id) = episode.speaker_profile.get(field).and_then(Json::as_str) {
                ids.push(id.to_string());
            }
        }
    }
    ids
}

/// A snapshot with the display fields of every model reference that resolves.
/// Unresolvable references and legacy snapshots are left untouched.
fn with_model_fields(
    snapshot: &Map<String, Json>,
    fields: &[(&str, &str, &str)],
    models: &HashMap<String, crate::domain::ModelDisplayInfo>,
) -> Map<String, Json> {
    let mut enriched = snapshot.clone();
    for (reference, provider_field, name_field) in fields {
        let info = enriched.get(*reference).and_then(Json::as_str).and_then(|id| models.get(id)).cloned();
        if let Some(info) = info {
            enriched.insert(provider_field.to_string(), Json::String(info.provider));
            enriched.insert(name_field.to_string(), Json::String(info.name));
        }
    }
    enriched
}

fn episode_view(
    state: &AppState,
    episode: EpisodeRecord,
    job_status: Option<String>,
    error_message: Option<String>,
    models: &HashMap<String, crate::domain::ModelDisplayInfo>,
) -> EpisodeView {
    let has_audio = resolve_contained_audio_path(&state.config.podcasts_dir(), episode.audio_file.as_deref())
        .is_some_and(|path| path.exists());
    let progress = nn_podcast::progress::effective_progress(job_status.as_deref(), episode.progress);
    EpisodeView {
        audio_url: has_audio.then(|| format!("/api/podcasts/episodes/{}/audio", episode.id)),
        episode_profile: with_model_fields(&episode.episode_profile, &EPISODE_PROFILE_MODEL_FIELDS, models),
        speaker_profile: with_model_fields(&episode.speaker_profile, &SPEAKER_PROFILE_MODEL_FIELDS, models),
        id: episode.id,
        name: episode.name,
        briefing: episode.briefing,
        audio_file: episode.audio_file,
        transcript: episode.transcript,
        outline: episode.outline,
        created: episode.created.as_ref().map(iso),
        job_status,
        error_message,
        notebook_id: episode.notebook,
        progress,
        published_at: episode.published_at.as_ref().map(iso),
        duration_seconds: episode.duration_seconds,
        feed_summary: episode.feed_summary,
    }
}

/// Episodes, newest first, optionally only those generated from a notebook.
/// Episodes with neither a job nor audio (abandoned records) are left out.
pub async fn list_episodes(state: &AppState, notebook_id: Option<&str>) -> AppResult<Vec<EpisodeView>> {
    let episodes: Vec<EpisodeRecord> = match non_empty(notebook_id) {
        Some(notebook) => {
            state
                .db
                .query_as(
                    "SELECT * FROM episode WHERE notebook = $notebook ORDER BY created DESC",
                    Params::new().record("notebook", notebook),
                )
                .await?
        }
        None => state.db.query_as("SELECT * FROM episode ORDER BY created DESC", Params::new()).await?,
    };
    let commands: Vec<String> = episodes.iter().filter_map(|e| e.command.clone()).collect();
    let jobs = match state.jobs.get_many(&commands).await {
        Ok(jobs) => jobs,
        Err(e) => {
            tracing::warn!("Error batch-fetching podcast job statuses: {e}");
            HashMap::new()
        }
    };
    let models = Model::display_info_for_ids(state, snapshot_model_ids(&episodes)).await;

    let mut views = Vec::with_capacity(episodes.len());
    for episode in episodes {
        if episode.command.is_none() && non_empty(episode.audio_file.as_deref()).is_none() {
            continue;
        }
        let (status, error) = match &episode.command {
            Some(command) => match jobs.get(command) {
                Some(job) => (Some(job.status_str().to_string()), job.error_message.clone()),
                None => (Some("unknown".to_string()), None),
            },
            // No job but audio: an imported episode.
            None => (Some("completed".to_string()), None),
        };
        views.push(episode_view(state, episode, status, error, &models));
    }
    Ok(views)
}

/// Status and error of an episode's job: `(None, None)` without a job,
/// `"unknown"` when the job record is gone or unreadable.
async fn job_detail(state: &AppState, episode: &EpisodeRecord) -> (Option<String>, Option<String>) {
    let Some(command) = &episode.command else {
        return (None, None);
    };
    match state.jobs.get(command).await {
        Ok(Some(job)) => (Some(job.status_str().to_string()), job.error_message),
        _ => (Some("unknown".to_string()), None),
    }
}

pub async fn get_episode(state: &AppState, episode_id: &str) -> AppResult<EpisodeView> {
    let episode = load_episode(state, episode_id).await?;
    let (status, error) = match job_detail(state, &episode).await {
        (None, _) if non_empty(episode.audio_file.as_deref()).is_some() => (Some("completed".to_string()), None),
        (None, _) => (Some("unknown".to_string()), None),
        detail => detail,
    };
    let models = Model::display_info_for_ids(state, snapshot_model_ids(std::slice::from_ref(&episode))).await;
    Ok(episode_view(state, episode, status, error, &models))
}

/// A file to send, with its media type.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ServedFile {
    pub path: PathBuf,
    pub media_type: String,
}

fn audio_media_type(path: &std::path::Path) -> &'static str {
    match audio_suffix(path).as_str() {
        ".m4a" => "audio/x-m4a",
        ".wav" => "audio/wav",
        _ => "audio/mpeg",
    }
}

/// The audio file of an episode, for the app's player and download.
pub async fn episode_audio(state: &AppState, episode_id: &str) -> AppResult<ServedFile> {
    let episode = load_episode(state, episode_id).await?;
    let Some(audio_file) = non_empty(episode.audio_file.as_deref()) else {
        return Err(AppError::not_found("Episode has no audio file"));
    };
    let Some(path) = resolve_contained_audio_path(&state.config.podcasts_dir(), Some(audio_file)) else {
        tracing::warn!("Blocked audio access outside podcasts directory for episode {episode_id}: {audio_file}");
        return Err(AppError::http(axum::http::StatusCode::FORBIDDEN, "Access to file denied"));
    };
    if !path.is_file() {
        return Err(AppError::not_found("Audio file not found on disk"));
    }
    Ok(ServedFile { media_type: audio_media_type(&path).to_string(), path })
}

/// Unlink an episode's audio, and the episode's working directory
/// (`episodes/<uuid>/`: clips, outline, transcript) when the audio sits in
/// one. Paths outside the podcasts folder are refused and logged.
async fn remove_episode_files(state: &AppState, episode: &EpisodeRecord) {
    let Some(audio_file) = non_empty(episode.audio_file.as_deref()) else {
        return;
    };
    let podcasts = state.config.podcasts_dir();
    let Some(path) = resolve_contained_audio_path(&podcasts, Some(audio_file)) else {
        tracing::warn!("Refusing to delete audio file outside podcasts directory for episode {}: {audio_file}", episode.id);
        return;
    };
    if path.exists() {
        match tokio::fs::remove_file(&path).await {
            Ok(()) => tracing::info!("Deleted audio file: {}", path.display()),
            Err(e) => tracing::warn!("Failed to delete audio file {}: {e}", path.display()),
        }
    }
    // `episodes/<dir>/audio/<file>`: the layout generation writes.
    let parts: Vec<&str> = audio_file.split('/').collect();
    if let ["episodes", dir, "audio", _file] = parts.as_slice() {
        let episodes_root = podcasts_root(&podcasts).join("episodes");
        let folder = realpath(&episodes_root.join(dir));
        if folder.parent() == Some(episodes_root.as_path()) && folder.is_dir() {
            if let Err(e) = tokio::fs::remove_dir_all(&folder).await {
                tracing::warn!("Failed to delete episode folder {}: {e}", folder.display());
            }
        }
    }
}

/// What `DELETE /podcasts/episodes/{id}` answers.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EpisodeDeleted {
    pub message: String,
    pub episode_id: String,
}

/// Delete an episode, its audio and its working files.
pub async fn delete_episode(state: &AppState, episode_id: &str) -> AppResult<EpisodeDeleted> {
    let episode = load_episode(state, episode_id).await?;
    remove_episode_files(state, &episode).await;
    state.db.delete(&episode.id).await?;
    tracing::info!("Deleted podcast episode: {episode_id}");
    Ok(EpisodeDeleted { message: "Episode deleted successfully".into(), episode_id: episode_id.to_string() })
}

// ---------------------------------------------------------------------------
// Publishing
// ---------------------------------------------------------------------------

/// The body of `POST /podcasts/episodes/{id}/publish`.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PublishRequest {
    #[serde(default)]
    pub summary: Option<String>,
}

/// `PublishEpisodeResponse`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PublishedState {
    pub published_at: Option<String>,
    pub duration_seconds: Option<u64>,
    pub feed_summary: Option<String>,
}

/// Explain why a WAV episode cannot go in the feed and what fixes it.
fn explain_publish_refusal(error: PodcastError) -> AppError {
    let message = error.to_string();
    if matches!(error, PodcastError::InvalidInput(_)) && message.to_lowercase().contains("format '.wav'") {
        return AppError::invalid(format!(
            "{message} This episode was saved as WAV: Gemini and Vertex AI voices produce WAV clips, which are \
             combined into a WAV file when ffmpeg is not installed. Install ffmpeg on the server and generate the \
             episode again to get an MP3 that can be published."
        ));
    }
    podcast_error(error)
}

/// Put a completed episode in the Podcast Feed (idempotent: a republish keeps
/// the original publish time). Refusals are 400s with a specific message.
pub async fn publish_episode(state: &AppState, episode_id: &str, summary: Option<&str>) -> AppResult<PublishedState> {
    let episode = load_episode(state, episode_id).await?;
    let (job_status, _) = job_detail(state, &episode).await;
    let podcasts = state.config.podcasts_dir();
    let summary = summary.map(String::from);
    let mut engine = episode.to_engine();
    let engine = tokio::task::spawn_blocking(move || {
        nn_podcast::publishing::publish_episode(&mut engine, summary.as_deref(), job_status.as_deref(), &podcasts, Utc::now())
            .map(|()| engine)
    })
    .await?
    .map_err(explain_publish_refusal)?;

    let published_at = engine.published_at.unwrap_or_else(Utc::now);
    let doc = Doc::new()
        .datetime("published_at", published_at)
        .set_some("duration_seconds", engine.duration_seconds)
        .set_some("feed_summary", engine.feed_summary.clone());
    state.db.update(&episode.id, doc).await?;
    Ok(PublishedState {
        published_at: Some(iso(&published_at)),
        duration_seconds: engine.duration_seconds,
        feed_summary: engine.feed_summary,
    })
}

/// Take an episode out of the Podcast Feed. A no-op if it is not in it.
pub async fn unpublish_episode(state: &AppState, episode_id: &str) -> AppResult<()> {
    let episode = load_episode(state, episode_id).await?;
    let mut engine = episode.to_engine();
    if nn_podcast::publishing::unpublish_episode(&mut engine) {
        let doc = Doc::new().none("published_at").none("duration_seconds").none("feed_summary");
        state.db.update(&episode.id, doc).await?;
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Submitting jobs
// ---------------------------------------------------------------------------

/// `PodcastGenerationRequest`.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct GenerateRequest {
    pub episode_profile: String,
    pub speaker_profile: String,
    pub episode_name: String,
    #[serde(default)]
    pub content: Option<String>,
    #[serde(default)]
    pub notebook_id: Option<String>,
    #[serde(default)]
    pub briefing_suffix: Option<String>,
}

/// `ScriptPodcastRequest`.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ScriptRequest {
    pub speaker_profile: String,
    pub episode_name: String,
    pub script: String,
    #[serde(default)]
    pub notebook_id: Option<String>,
}

/// `PodcastGenerationResponse`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SubmittedJob {
    pub job_id: String,
    pub status: String,
    pub message: String,
    pub episode_profile: String,
    pub episode_name: String,
}

/// The arguments of a `generate_podcast` job (same fields as the Python command).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct GeneratePodcastArgs {
    pub episode_profile: String,
    /// Speaker profile record id (or name). Omitted: the episode profile's `speaker_config`.
    #[serde(default)]
    pub speaker_profile: Option<String>,
    pub episode_name: String,
    pub content: String,
    #[serde(default)]
    pub briefing_suffix: Option<String>,
    #[serde(default)]
    pub notebook_id: Option<String>,
}

/// The arguments of a `read_script_podcast` job.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ReadScriptArgs {
    pub speaker_profile: String,
    pub episode_name: String,
    pub script: String,
    #[serde(default)]
    pub notebook_id: Option<String>,
}

fn notebook_ref(notebook_id: Option<&str>) -> Option<String> {
    non_empty(notebook_id).map(|id| with_table(Notebook::TABLE, id))
}

/// The longest content an episode is generated from by default
/// (`OPEN_NOTEBOOK_MAX_PODCAST_CONTENT_CHARS`). It is stored twice (job
/// arguments and the episode) and sent whole to the outline model, so it is
/// capped well below the request body limit.
pub const MAX_PODCAST_CONTENT_CHARS: usize = 2_000_000;

fn max_podcast_content_chars() -> usize {
    crate::config::env_nonempty("OPEN_NOTEBOOK_MAX_PODCAST_CONTENT_CHARS")
        .and_then(|raw| raw.trim().parse::<usize>().ok())
        .filter(|n| *n > 0)
        .unwrap_or(MAX_PODCAST_CONTENT_CHARS)
}

/// `2000000` → `"2,000,000"`.
fn with_thousands(n: usize) -> String {
    let digits = n.to_string();
    let mut out = String::with_capacity(digits.len() + digits.len() / 3);
    for (i, c) in digits.chars().enumerate() {
        if i > 0 && (digits.len() - i).is_multiple_of(3) {
            out.push(',');
        }
        out.push(c);
    }
    out
}

/// Check a generation request and build its job arguments: the episode and
/// speaker profiles must exist (404 otherwise) and there must be content,
/// given directly or taken from the notebook, of at most
/// [`MAX_PODCAST_CONTENT_CHARS`] characters (400 otherwise).
async fn generation_args(
    state: &AppState,
    episode_profile: &str,
    speaker_profile: &str,
    episode_name: &str,
    content: Option<&str>,
    notebook_id: Option<&str>,
    briefing_suffix: Option<&str>,
) -> AppResult<GeneratePodcastArgs> {
    if find_episode_profile(state, episode_profile).await?.is_none() {
        return Err(AppError::not_found(format!("Episode profile '{episode_profile}' not found")));
    }
    let speaker = resolve_speaker_profile(state, speaker_profile)
        .await?
        .ok_or_else(|| AppError::not_found(format!("Speaker profile '{speaker_profile}' not found")))?;
    let notebook = notebook_ref(notebook_id);
    let mut content = non_empty(content).map(String::from);
    if content.is_none() {
        if let Some(notebook_id) = &notebook {
            let notebook = Notebook::get(state, notebook_id).await.map_err(|e| {
                if e.is_not_found() {
                    AppError::not_found("Notebook not found")
                } else {
                    e
                }
            })?;
            content = Some(notebook.context(state).await?).filter(|c| !c.trim().is_empty());
        }
    }
    let content =
        content.ok_or_else(|| AppError::invalid("Content is required - provide either content or notebook_id"))?;
    let max = max_podcast_content_chars();
    if content.chars().count() > max {
        return Err(AppError::invalid(format!(
            "The podcast content is too long ({} characters at most).",
            with_thousands(max)
        )));
    }
    Ok(GeneratePodcastArgs {
        episode_profile: episode_profile.to_string(),
        speaker_profile: Some(speaker.id),
        episode_name: episode_name.to_string(),
        content,
        briefing_suffix: briefing_suffix.map(String::from),
        notebook_id: notebook,
    })
}

/// Check a script request and build its job arguments. An empty script, an
/// unknown profile (404) or one without a voice model (400) fail the request
/// instead of a background job.
async fn script_args(
    state: &AppState,
    speaker_profile: &str,
    episode_name: &str,
    script: &str,
    notebook_id: Option<&str>,
) -> AppResult<ReadScriptArgs> {
    if script.chars().count() > nn_podcast::script::MAX_SCRIPT_CHARS {
        return Err(AppError::invalid("The script is too long (500,000 characters at most)."));
    }
    let speaker = resolve_speaker_profile(state, speaker_profile)
        .await?
        .ok_or_else(|| AppError::not_found(format!("Speaker profile '{speaker_profile}' not found")))?;
    if non_empty(speaker.voice_model.as_deref()).is_none() {
        return Err(AppError::invalid(format!("Speaker profile '{}' has no voice model", speaker.name)));
    }
    let names: Vec<String> =
        speaker.speakers.iter().filter_map(|s| s.get("name").and_then(Json::as_str)).map(String::from).collect();
    let script = script.to_string();
    let (script, _) = parse_script_blocking(script, names).await?;
    Ok(ReadScriptArgs {
        speaker_profile: speaker.id,
        episode_name: episode_name.to_string(),
        script,
        notebook_id: notebook_ref(notebook_id),
    })
}

/// Parse a script on the blocking pool: a long script is CPU work that would
/// otherwise hold up an async worker thread. Hands the script back.
async fn parse_script_blocking(script: String, speaker_names: Vec<String>) -> AppResult<(String, Vec<nn_podcast::models::Dialogue>)> {
    tokio::task::spawn_blocking(move || {
        let turns = nn_podcast::parse_script(&script, &speaker_names).map_err(podcast_error)?;
        Ok((script, turns))
    })
    .await
    .map_err(|e| AppError::Internal(format!("Parsing the script was interrupted: {e}")))?
}

async fn submit(state: &AppState, name: &str, args: impl Serialize) -> AppResult<String> {
    let job_id = state.jobs.submit(name, serde_json::to_value(args)?).await?;
    Ok(job_id)
}

/// Queue a generated episode. Returns at once with the job id.
pub async fn submit_generation(state: &AppState, request: &GenerateRequest) -> AppResult<SubmittedJob> {
    let args = generation_args(
        state,
        &request.episode_profile,
        &request.speaker_profile,
        &request.episode_name,
        request.content.as_deref(),
        request.notebook_id.as_deref(),
        request.briefing_suffix.as_deref(),
    )
    .await?;
    let job_id = submit(state, GENERATE_PODCAST, &args).await?;
    tracing::info!("Submitted podcast generation job: {job_id} for episode '{}'", request.episode_name);
    Ok(SubmittedJob {
        job_id,
        status: "submitted".into(),
        message: format!("Podcast generation started for episode '{}'", request.episode_name),
        episode_profile: request.episode_profile.clone(),
        episode_name: request.episode_name.clone(),
    })
}

/// Queue an episode read from a script. Returns at once with the job id.
pub async fn submit_script(state: &AppState, request: &ScriptRequest) -> AppResult<SubmittedJob> {
    let args = script_args(state, &request.speaker_profile, &request.episode_name, &request.script, request.notebook_id.as_deref())
        .await?;
    let job_id = submit(state, READ_SCRIPT_PODCAST, &args).await?;
    tracing::info!("Submitted script podcast job {job_id} for '{}'", request.episode_name);
    Ok(SubmittedJob {
        job_id,
        status: "submitted".into(),
        message: format!("Reading script for episode '{}'", request.episode_name),
        episode_profile: "script".into(),
        episode_name: request.episode_name.clone(),
    })
}

/// What `POST /podcasts/episodes/{id}/retry` answers.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RetrySubmitted {
    pub job_id: String,
    pub message: String,
}

/// Retry a failed episode: delete it (and its files) and queue the same job again.
pub async fn retry_episode(state: &AppState, episode_id: &str) -> AppResult<RetrySubmitted> {
    let episode = load_episode(state, episode_id).await?;
    let (status, _) = job_detail(state, &episode).await;
    if !matches!(status.as_deref(), Some("failed") | Some("error")) {
        return Err(AppError::bad_request(format!(
            "Episode is not in a failed state (current: {})",
            status.as_deref().unwrap_or("None")
        )));
    }
    let snapshot_name = |snapshot: &Map<String, Json>| {
        snapshot.get("name").and_then(Json::as_str).filter(|n| !n.is_empty()).map(String::from)
    };
    let speaker = snapshot_name(&episode.speaker_profile);

    let (command, args) = if episode.is_script() {
        let speaker = speaker.ok_or_else(|| {
            AppError::bad_request("Cannot retry: speaker profile name missing from stored data")
        })?;
        let args = script_args(state, &speaker, &episode.name, &episode.content, episode.notebook.as_deref()).await?;
        (READ_SCRIPT_PODCAST, serde_json::to_value(args)?)
    } else {
        let (Some(profile), Some(speaker)) = (snapshot_name(&episode.episode_profile), speaker) else {
            return Err(AppError::bad_request(
                "Cannot retry: episode or speaker profile name missing from stored data",
            ));
        };
        let args = generation_args(
            state,
            &profile,
            &speaker,
            &episode.name,
            Some(&episode.content),
            episode.notebook.as_deref(),
            None,
        )
        .await?;
        (GENERATE_PODCAST, serde_json::to_value(args)?)
    };

    // Take a place in the queue first: a full queue (429) must not cost the episode.
    let slot = state.jobs.reserve()?;
    remove_episode_files(state, &episode).await;
    state.db.delete(&episode.id).await?;
    let job_id = state.jobs.submit_reserved(slot, command, args).await?;
    Ok(RetrySubmitted { job_id, message: "Retry submitted successfully".into() })
}

/// `GET /podcasts/jobs/{job_id}`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PodcastJobStatus {
    pub job_id: String,
    pub status: String,
    pub result: Option<Json>,
    pub error_message: Option<String>,
    pub created: Option<String>,
    pub updated: Option<String>,
    pub progress: Option<Json>,
}

pub async fn job_status(state: &AppState, job_id: &str) -> AppResult<PodcastJobStatus> {
    let job = state.jobs.get(job_id).await.map_err(|e| {
        tracing::error!("Failed to get podcast job status: {e}");
        AppError::internal("Failed to get job status")
    })?;
    Ok(match job {
        Some(job) => PodcastJobStatus {
            job_id: job_id.to_string(),
            status: job.status_str().to_string(),
            result: job.result,
            error_message: job.error_message,
            created: job.created.as_ref().map(iso),
            updated: job.updated.as_ref().map(iso),
            progress: job.progress,
        },
        None => PodcastJobStatus {
            job_id: job_id.to_string(),
            status: "unknown".into(),
            result: None,
            error_message: None,
            created: None,
            updated: None,
            progress: None,
        },
    })
}

// ---------------------------------------------------------------------------
// The jobs
// ---------------------------------------------------------------------------

/// `PodcastGenerationOutput`.
#[derive(Debug, Clone, Serialize)]
struct GenerationOutput {
    success: bool,
    episode_id: Option<String>,
    audio_file_path: Option<String>,
    transcript: Option<Json>,
    outline: Option<Json>,
    processing_time: f64,
    error_message: Option<String>,
}

async fn write_progress(state: &AppState, episode_id: &str, percent: i64) {
    let outcome = state
        .db
        .query("UPDATE $id SET progress = $progress", Params::new().record("id", episode_id).set("progress", percent))
        .await;
    if let Err(e) = outcome {
        tracing::warn!("Failed to write podcast progress {percent}%: {e}");
    }
}

fn audio_options() -> AudioOptions {
    let mut options = AudioOptions::default();
    if let Some(batch) = env_u32("TTS_BATCH_SIZE").filter(|n| *n > 0) {
        options.batch_size = batch as usize;
    }
    if cfg!(test) {
        // Tests join the mock clips the same way on every machine.
        options.ffmpeg = FfmpegMode::Disabled;
        options.batch_pause = Duration::ZERO;
    }
    options
}

/// `PODCAST_REFUSAL_RETRIES` and `PODCAST_REFUSAL_FALLBACK_MODEL` (a model
/// name on the transcript model's provider and credential).
async fn refusal_policy(state: &AppState, transcript_model_id: &str) -> RefusalPolicy {
    let mut policy = RefusalPolicy::default();
    if let Some(raw) = crate::config::env_nonempty("PODCAST_REFUSAL_RETRIES") {
        if let Ok(retries) = raw.trim().parse::<i64>() {
            policy.retries = retries.max(0) as u32;
        }
    }
    let Some(fallback) = crate::config::env_nonempty("PODCAST_REFUSAL_FALLBACK_MODEL") else {
        return policy;
    };
    let base = match Model::get_for_use(state, transcript_model_id).await {
        Ok(model) => model,
        Err(e) => {
            tracing::warn!("Cannot set up the refusal fallback model: {e}");
            return policy;
        }
    };
    let model = Model { id: String::new(), name: fallback.trim().to_string(), model_type: "language".into(), ..base };
    match state.models.build(state, &model).await {
        Ok(AnyModel::Language(llm)) => policy.fallback = Some(llm),
        Ok(_) => {}
        Err(e) => tracing::warn!("Cannot set up the refusal fallback model {}/{}: {e}", model.provider, model.name),
    }
    policy
}

/// The profile's speech model plus per-speaker overrides. With `strict`, a
/// speaker override that does not resolve fails; otherwise it is logged and
/// the profile's model is used for that speaker (the Python generate path).
async fn tts_models(state: &AppState, profile: &nn_podcast::SpeakerProfile, strict: bool) -> AppResult<TtsModels> {
    let voice_model = profile.require_voice_model().map_err(podcast_error)?;
    let mut models = TtsModels::new(state.models.text_to_speech_by_id(state, voice_model).await?);
    for speaker in &profile.speakers {
        let Some(model_id) = non_empty(speaker.voice_model.as_deref()) else {
            continue;
        };
        match state.models.text_to_speech_by_id(state, model_id).await {
            Ok(model) => models = models.with_speaker(speaker.name.trim(), model),
            Err(e) if strict => return Err(e),
            Err(e) => tracing::warn!("Failed to resolve per-speaker TTS for '{}': {e}", speaker.name),
        }
    }
    Ok(models)
}

/// Create the episode record for a starting job and set progress 1.
async fn create_episode_record(state: &AppState, ctx: &JobContext, doc: Doc, notebook: Option<&str>) -> AppResult<String> {
    let doc = doc.record("command", &ctx.command_id);
    let doc = match notebook {
        Some(notebook) => doc.record("notebook", notebook),
        None => doc,
    };
    let row = state.db.create(EPISODE_TABLE, doc).await?;
    let id = row["id"].as_str().ok_or_else(|| AppError::database("Episode record has no id"))?.to_string();
    write_progress(state, &id, 1).await;
    Ok(id)
}

/// Run the engine, writing every progress percentage to the episode.
async fn run_engine(state: &AppState, episode_id: &str, request: PodcastRequest) -> Result<PodcastOutput, PodcastFailure> {
    let (sender, mut receiver) = tokio::sync::mpsc::unbounded_channel::<u8>();
    let writer_state = state.clone();
    let writer_id = episode_id.to_string();
    let writer = tokio::spawn(async move {
        while let Some(mut percent) = receiver.recv().await {
            // Only the latest value matters.
            while let Ok(newer) = receiver.try_recv() {
                percent = newer;
            }
            write_progress(&writer_state, &writer_id, i64::from(percent)).await;
        }
    });
    let report = move |update: ProgressUpdate| {
        let _ = sender.send(update.percent);
    };
    let result = create_podcast(request, &report).await;
    drop(report);
    if let Err(e) = writer.await {
        tracing::warn!("Failed to flush podcast progress: {e}");
    }
    result
}

/// Store what a failed run produced, then hand back the job's error.
async fn record_failure(state: &AppState, episode_id: &str, failure: &PodcastFailure) -> AppError {
    let mut doc = Doc::new();
    if let Some(outline) = &failure.outline {
        doc = doc.set("outline", outline.to_record());
    }
    if let Some(transcript) = &failure.transcript {
        doc = doc.set("transcript", transcript_record(transcript));
    }
    if doc.contains("outline") || doc.contains("transcript") {
        if let Err(e) = state.db.update(episode_id, doc).await {
            tracing::warn!("Could not save the partial results of episode {episode_id}: {e}");
        }
    }
    tracing::error!("Podcast generation failed: {}", failure.message());
    failure_error(failure)
}

/// Save the finished episode (relative audio path, transcript, outline) and set progress 100.
async fn record_success(state: &AppState, episode_id: &str, output: &PodcastOutput) -> AppResult<String> {
    let audio_file = to_relative_audio_path(&state.config.podcasts_dir(), &output.audio_file).map_err(podcast_error)?;
    let mut doc = Doc::new().set("audio_file", &audio_file).set("transcript", output.transcript_record());
    if let Some(outline) = output.outline_record() {
        doc = doc.set("outline", outline);
    }
    state.db.update(episode_id, doc).await?;
    write_progress(state, episode_id, 100).await;
    Ok(audio_file)
}

fn parse_args<T: serde::de::DeserializeOwned>(args: Json, command: &str) -> AppResult<T> {
    serde_json::from_value(args).map_err(|e| AppError::invalid(format!("Invalid arguments for {command}: {e}")))
}

async fn run_generate_job(ctx: JobContext, args: Json) -> AppResult<Json> {
    let started = Instant::now();
    let state = ctx.state.clone();
    let args: GeneratePodcastArgs = parse_args(args, GENERATE_PODCAST)?;
    tracing::info!("Starting podcast generation for episode: {}", args.episode_name);

    // 1. Profiles.
    let episode_profile = find_episode_profile(&state, &args.episode_profile)
        .await?
        .ok_or_else(|| AppError::invalid(format!("Episode profile '{}' not found", args.episode_profile)))?;
    let engine_episode_profile = episode_profile.to_engine();
    let explicit_speaker = non_empty(args.speaker_profile.as_deref());
    let speaker_ref = match explicit_speaker {
        Some(reference) => reference.to_string(),
        None => engine_episode_profile.require_speaker_config().map_err(podcast_error)?.to_string(),
    };
    let speaker_profile = match resolve_speaker_profile(&state, &speaker_ref).await? {
        Some(profile) => profile,
        None if explicit_speaker.is_some() => {
            return Err(AppError::invalid(format!("Speaker profile '{speaker_ref}' not found")))
        }
        None => {
            return Err(AppError::invalid(format!(
                "Episode profile '{}' references a speaker profile that no longer exists. \
                 Please update the profile to select a speaker profile.",
                episode_profile.name
            )))
        }
    };
    let engine_speaker_profile = speaker_profile.to_engine()?;

    // 2. Models.
    let outline_id = engine_episode_profile.require_outline_llm().map_err(podcast_error)?.to_string();
    let transcript_id = engine_episode_profile.require_transcript_llm().map_err(podcast_error)?.to_string();
    engine_speaker_profile.require_voice_model().map_err(podcast_error)?;
    let outline_model = state.models.language_by_id(&state, &outline_id).await?;
    let transcript_model = state.models.language_by_id(&state, &transcript_id).await?;
    let tts = tts_models(&state, &engine_speaker_profile, false).await?;
    let mut input = GenerationInput::from_profile(
        &engine_episode_profile,
        args.content.clone(),
        args.briefing_suffix.as_deref(),
        outline_model,
        transcript_model,
    )
    .map_err(podcast_error)?;
    input.options.refusal = refusal_policy(&state, &transcript_id).await;

    // 3. The episode record, linked to this job.
    let episode_snapshot = state.db.select(&episode_profile.id).await?.unwrap_or_else(|| json!({}));
    let speaker_snapshot = state.db.select(&speaker_profile.id).await?.unwrap_or_else(|| json!({}));
    let doc = Doc::new()
        .set("name", &args.episode_name)
        .set("episode_profile", episode_snapshot)
        .set("speaker_profile", speaker_snapshot)
        .set("briefing", &input.briefing)
        .set("content", &args.content);
    let episode_id = create_episode_record(&state, &ctx, doc, args.notebook_id.as_deref()).await?;

    // 4. Generate.
    let (episode_dir_name, output_dir) = build_episode_output_dir(&state.config.podcasts_dir());
    let request = PodcastRequest {
        output_dir,
        episode_dir_name,
        speaker_profile: engine_speaker_profile,
        tts,
        source: PodcastSource::Generate(Box::new(input)),
        audio: audio_options(),
    };
    let output = match run_engine(&state, &episode_id, request).await {
        Ok(output) => output,
        Err(failure) => return Err(record_failure(&state, &episode_id, &failure).await),
    };
    let audio_file = record_success(&state, &episode_id, &output).await?;
    let processing_time = started.elapsed().as_secs_f64();
    tracing::info!("Successfully generated podcast episode: {episode_id} in {processing_time:.2}s");
    Ok(serde_json::to_value(GenerationOutput {
        success: true,
        episode_id: Some(episode_id),
        audio_file_path: Some(audio_file),
        transcript: Some(output.transcript_record()),
        outline: output.outline_record(),
        processing_time,
        error_message: None,
    })?)
}

async fn run_script_job(ctx: JobContext, args: Json) -> AppResult<Json> {
    let started = Instant::now();
    let state = ctx.state.clone();
    let args: ReadScriptArgs = parse_args(args, READ_SCRIPT_PODCAST)?;

    let speaker_profile = resolve_speaker_profile(&state, &args.speaker_profile)
        .await?
        .ok_or_else(|| AppError::invalid(format!("Speaker profile '{}' not found", args.speaker_profile)))?;
    let engine_speaker_profile = speaker_profile.to_engine()?;
    let tts = tts_models(&state, &engine_speaker_profile, true).await?;
    let (_, turns) = parse_script_blocking(args.script.clone(), engine_speaker_profile.speaker_names()).await?;

    let speaker_snapshot = state.db.select(&speaker_profile.id).await?.unwrap_or_else(|| json!({}));
    let doc = Doc::new()
        .set("name", &args.episode_name)
        .set("episode_profile", Json::Object(script_episode_profile()))
        .set("speaker_profile", speaker_snapshot)
        .set("briefing", SCRIPT_BRIEFING)
        .set("content", &args.script)
        .set("transcript", transcript_record(&turns));
    let episode_id = create_episode_record(&state, &ctx, doc, args.notebook_id.as_deref()).await?;

    let (episode_dir_name, output_dir) = build_episode_output_dir(&state.config.podcasts_dir());
    let request = PodcastRequest {
        output_dir,
        episode_dir_name,
        speaker_profile: engine_speaker_profile,
        tts,
        source: PodcastSource::Script(args.script.clone()),
        audio: audio_options(),
    };
    let output = match run_engine(&state, &episode_id, request).await {
        Ok(output) => output,
        Err(failure) => return Err(record_failure(&state, &episode_id, &failure).await),
    };
    let audio_file = record_success(&state, &episode_id, &output).await?;
    let processing_time = started.elapsed().as_secs_f64();
    tracing::info!("Read script into episode {episode_id} ({} clips) in {processing_time:.2}s", output.transcript.len());
    Ok(serde_json::to_value(GenerationOutput {
        success: true,
        episode_id: Some(episode_id),
        audio_file_path: Some(audio_file),
        transcript: Some(output.transcript_record()),
        outline: None,
        processing_time,
        error_message: None,
    })?)
}

#[cfg(test)]
mod tests {
    use axum::http::StatusCode;

    use crate::testing::TestApp;

    #[test]
    fn thousands_separators() {
        assert_eq!(super::with_thousands(2_000_000), "2,000,000");
        assert_eq!(super::with_thousands(999), "999");
        assert_eq!(super::with_thousands(12_345), "12,345");
    }

    /// Issue #34: the API applies the script length limit the web form does,
    /// before any parsing or lookup.
    #[tokio::test]
    async fn read_script_refuses_an_overlong_script() {
        let app = TestApp::new().await;
        let script = "*a ".repeat(nn_podcast::script::MAX_SCRIPT_CHARS / 3 + 1);
        let (status, body) = app
            .post(
                "/api/podcasts/read-script",
                serde_json::json!({ "speaker_profile": "nobody", "episode_name": "x", "script": script }),
            )
            .await;
        assert_eq!(
            (status, body["detail"].as_str()),
            (StatusCode::BAD_REQUEST, Some("The script is too long (500,000 characters at most)."))
        );
    }
}
