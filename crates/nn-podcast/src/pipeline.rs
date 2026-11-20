//! The whole job: from source content (or a script) to an episode on disk.
//!
//! [`create_podcast`] is what the original's `generate_podcast` and
//! `read_script_podcast` commands ran once they had loaded their records:
//!
//! ```text
//! generated:  outline -> transcript (one call per segment) -> clips -> episode file
//! script:     parse the script                             -> clips -> episode file
//! ```
//!
//! Output directory, as in the original:
//!
//! ```text
//! <output_dir>/outline.json             {"segments": [...]}       (generated only)
//! <output_dir>/transcript.json          [{"speaker", "dialogue"}, ...]
//! <output_dir>/clips/0000.mp3 ...       one file per dialogue turn
//! <output_dir>/audio/<episode_dir_name>.mp3
//! ```
//!
//! The caller owns everything around it: creating the episode record (and
//! writing progress 1), resolving profiles to models, storing the result with
//! [`crate::audio_paths::to_relative_audio_path`], and writing progress 100
//! after the final save.

use std::fmt;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::{Duration, Instant};

use nn_llm::SharedLanguageModel;
use serde::Serialize;
use serde_json::Value;

use crate::audio::{self, AudioOptions, CombineMethod, TtsModels};
use crate::error::{PodcastError, Result};
use crate::generate::{generate_outline, generate_transcript, GenerationOptions, OutlineRequest, TranscriptRequest};
use crate::language::resolve_optional_language;
use crate::models::{transcript_record, Content, Dialogue, EpisodeProfile, Outline, SpeakerProfile};
use crate::progress::{ProgressEvent, ProgressTracker, ProgressUpdate};
use crate::refusal_guard::model_key;
use crate::script::parse_script;

/// Everything needed to generate an episode's outline and transcript.
#[derive(Clone)]
pub struct GenerationInput {
    pub content: Content,
    /// The full briefing (profile default plus any per-episode suffix).
    pub briefing: String,
    pub num_segments: u32,
    /// Language as a BCP 47 or ISO 639 code (`EpisodeProfile::language`).
    pub language: Option<String>,
    pub outline_model: SharedLanguageModel,
    pub transcript_model: SharedLanguageModel,
    pub options: GenerationOptions,
}

impl fmt::Debug for GenerationInput {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("GenerationInput")
            .field("content", &self.content)
            .field("briefing", &self.briefing)
            .field("num_segments", &self.num_segments)
            .field("language", &self.language)
            .field("outline_model", &model_key(self.outline_model.as_ref()))
            .field("transcript_model", &model_key(self.transcript_model.as_ref()))
            .field("options", &self.options)
            .finish()
    }
}

impl GenerationInput {
    /// Fill the input from an episode profile, the way the original's command
    /// did: briefing from the profile's default plus `briefing_suffix`, and
    /// the profile's segment count, language and `max_tokens`.
    pub fn from_profile(
        profile: &EpisodeProfile,
        content: impl Into<Content>,
        briefing_suffix: Option<&str>,
        outline_model: SharedLanguageModel,
        transcript_model: SharedLanguageModel,
    ) -> Result<Self> {
        profile.validate()?;
        Ok(Self {
            content: content.into(),
            briefing: profile.briefing(briefing_suffix),
            num_segments: profile.num_segments as u32,
            language: profile.language.clone(),
            outline_model,
            transcript_model,
            options: GenerationOptions { max_tokens: profile.max_tokens, ..GenerationOptions::default() },
        })
    }
}

/// Where the dialogue comes from.
#[derive(Debug, Clone)]
pub enum PodcastSource {
    /// Two language models write an outline and then the dialogue.
    Generate(Box<GenerationInput>),
    /// A user-written script is read as written; no language model runs.
    Script(String),
}

/// One episode to produce.
#[derive(Debug, Clone)]
pub struct PodcastRequest {
    /// Directory for this episode's files. Use
    /// [`crate::audio_paths::build_episode_output_dir`].
    pub output_dir: PathBuf,
    /// File stem of the episode's audio file: the directory's UUID, never the
    /// episode name the user typed.
    pub episode_dir_name: String,
    pub speaker_profile: SpeakerProfile,
    pub tts: TtsModels,
    pub source: PodcastSource,
    pub audio: AudioOptions,
}

/// A finished episode.
#[derive(Debug, Clone)]
pub struct PodcastOutput {
    /// `None` for a script episode.
    pub outline: Option<Outline>,
    pub transcript: Vec<Dialogue>,
    /// Absolute or working-directory-relative path of the episode audio, as
    /// built from `output_dir`. Store it with `to_relative_audio_path`.
    pub audio_file: PathBuf,
    /// Length of the audio in seconds; `None` if it could not be measured.
    pub duration_seconds: Option<f64>,
    /// The per-turn clips, in transcript order.
    pub clips: Vec<PathBuf>,
    pub combine_method: CombineMethod,
    pub output_dir: PathBuf,
    pub processing_time: Duration,
}

impl PodcastOutput {
    /// The value for `PodcastEpisode::transcript`: `{"transcript": [...]}`.
    pub fn transcript_record(&self) -> Value {
        transcript_record(&self.transcript)
    }

    /// The value for `PodcastEpisode::outline`, `None` for a script episode.
    pub fn outline_record(&self) -> Option<Value> {
        self.outline.as_ref().map(Outline::to_record)
    }
}

/// A failed job, with whatever had been produced before it failed so the
/// caller can still store the outline and transcript on the episode.
#[derive(Debug)]
pub struct PodcastFailure {
    pub error: PodcastError,
    pub outline: Option<Outline>,
    pub transcript: Option<Vec<Dialogue>>,
}

impl PodcastFailure {
    /// The message for the failed job record: the error plus, when one
    /// applies, a `NOTE:` with what to do about it.
    pub fn message(&self) -> String {
        self.error.message_with_hint()
    }
}

impl fmt::Display for PodcastFailure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message())
    }
}

impl std::error::Error for PodcastFailure {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.error)
    }
}

impl From<PodcastError> for PodcastFailure {
    fn from(error: PodcastError) -> Self {
        Self { error, outline: None, transcript: None }
    }
}

async fn write_json<T: Serialize>(path: &Path, value: &T, pretty: bool) -> Result<()> {
    let encoded = if pretty { serde_json::to_vec_pretty(value) } else { serde_json::to_vec(value) }
        .map_err(|e| PodcastError::io(format!("Failed to encode {}", path.display()), e.into()))?;
    tokio::fs::write(path, encoded)
        .await
        .map_err(|e| PodcastError::io(format!("Failed to write {}", path.display()), e))
}

/// Produce one episode: dialogue (generated or from a script), a clip per
/// turn, and the combined audio file.
///
/// `progress` receives a [`ProgressUpdate`] each time the percentage grows:
/// 5 when the outline is done, 5 to 35 across the transcript segments, 35 when
/// audio starts and up to 95 as clips are written. It may be called from
/// several clips at once and must return quickly; hand slow work (a database
/// write) to a channel.
pub async fn create_podcast(
    request: PodcastRequest,
    progress: &(dyn Fn(ProgressUpdate) + Send + Sync),
) -> std::result::Result<PodcastOutput, PodcastFailure> {
    let started = Instant::now();
    let PodcastRequest { output_dir, episode_dir_name, speaker_profile, tts, source, audio: audio_options } = request;

    speaker_profile.validate_for_generation()?;
    tokio::fs::create_dir_all(&output_dir)
        .await
        .map_err(|e| PodcastError::io(format!("Failed to create {}", output_dir.display()), e))?;

    let tracker = Mutex::new(ProgressTracker::new());
    let report = |event: ProgressEvent| {
        let grown = tracker.lock().unwrap_or_else(|poisoned| poisoned.into_inner()).apply(event);
        if let Some(percent) = grown {
            progress(ProgressUpdate { percent, phase: event.phase(), event });
        }
    };

    let mut outline: Option<Outline> = None;
    let transcript: Vec<Dialogue> = match source {
        PodcastSource::Generate(input) => {
            let language = resolve_optional_language(input.language.as_deref())?;
            let generated = generate_outline(
                input.outline_model.as_ref(),
                &OutlineRequest {
                    briefing: &input.briefing,
                    content: &input.content,
                    speakers: &speaker_profile.speakers,
                    num_segments: input.num_segments,
                    language: language.as_deref(),
                },
                &input.options,
            )
            .await?;
            write_json(&output_dir.join("outline.json"), &generated, false).await?;
            report(ProgressEvent::OutlineGenerated { segments: generated.segments.len() });
            let generated = outline.insert(generated);

            let result = generate_transcript(
                input.transcript_model.as_ref(),
                &TranscriptRequest {
                    briefing: &input.briefing,
                    content: &input.content,
                    speakers: &speaker_profile.speakers,
                    outline: generated,
                    language: language.as_deref(),
                },
                &input.options,
                &|index, total, _segment| report(ProgressEvent::TranscriptSegmentStarted { index, total }),
            )
            .await;
            match result {
                Ok(transcript) => transcript,
                Err(error) => return Err(PodcastFailure { error, outline, transcript: None }),
            }
        }
        PodcastSource::Script(script) => parse_script(&script, &speaker_profile.speaker_names())?,
    };

    let audio_stage = async {
        write_json(&output_dir.join("transcript.json"), &transcript, true).await?;
        tracing::info!("Routing {} dialogue segments for sequential batch processing", transcript.len());
        report(ProgressEvent::AudioStarted { clips: transcript.len() });

        let clips = audio::synthesize_clips(
            &transcript,
            &speaker_profile,
            &tts,
            &output_dir.join("clips"),
            &audio_options,
            &|_done, _total| report(ProgressEvent::ClipGenerated),
        )
        .await?;
        let combined =
            audio::combine_clips(&clips, &output_dir.join("audio"), &episode_dir_name, &audio_options).await?;
        Ok::<_, PodcastError>((clips, combined))
    };
    let (clips, combined) = match audio_stage.await {
        Ok(done) => done,
        Err(error) => return Err(PodcastFailure { error, outline, transcript: Some(transcript) }),
    };

    let measured = combined.path.clone();
    let duration_seconds = match tokio::task::spawn_blocking(move || audio::duration_seconds(&measured)).await {
        Ok(Ok(seconds)) => Some(seconds),
        Ok(Err(error)) => {
            tracing::warn!("{error}");
            None
        }
        Err(_) => None,
    };

    let processing_time = started.elapsed();
    tracing::info!(
        "Generated podcast audio {} ({} clips) in {:.2}s",
        combined.path.display(),
        clips.len(),
        processing_time.as_secs_f64()
    );
    Ok(PodcastOutput {
        outline,
        transcript,
        audio_file: combined.path,
        duration_seconds,
        clips,
        combine_method: combined.method,
        output_dir,
        processing_time,
    })
}
