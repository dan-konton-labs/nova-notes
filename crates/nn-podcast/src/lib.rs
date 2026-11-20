//! The Nova Notes podcast engine.
//!
//! Database-free and HTTP-free: the server crate owns records, routes,
//! settings and background jobs, and calls into this crate with plain values
//! and model trait objects from `nn-llm`. Nothing here reads the environment,
//! except that `ffmpeg` is looked up on `PATH` when combining audio.
//!
//! # Making an episode
//!
//! [`create_podcast`] runs the whole job and reports progress through a
//! callback. Its stages are also usable on their own:
//!
//! | Stage | Module |
//! |---|---|
//! | Prompt templates (the original's `outline.jinja` / `transcript.jinja`) | [`prompts`] |
//! | Outline and transcript from language models | [`generate`] |
//! | Reply clean-up and validation | [`reply`] |
//! | Provider refusals: retry, then fall back | [`refusal_guard`] |
//! | A user-written script as dialogue, no language model | [`script`] |
//! | Speech clips, combining, duration | [`audio`] |
//! | Progress percentages | [`progress`] |
//!
//! # The Podcast Feed (Nova Notes fork)
//!
//! | Concern | Module |
//! |---|---|
//! | Stored audio paths and containment | [`audio_paths`] |
//! | RSS XML, Feed Token, Show Metadata, URLs | [`feed`] |
//! | Artwork validation and storage | [`artwork`] |
//! | Publish / unpublish rules | [`publishing`] |
//!
//! Domain types with the database records' field names are in [`models`].

pub mod artwork;
pub mod audio;
pub mod audio_paths;
pub mod error;
pub mod feed;
pub mod generate;
pub mod language;
pub mod models;
pub mod pipeline;
pub mod progress;
pub mod prompts;
pub mod publishing;
pub mod refusal_guard;
pub mod reply;
pub mod retry;
pub mod script;

pub use audio::{duration_seconds, AudioOptions, CombineMethod, FfmpegMode, TtsModels};
pub use audio_paths::{build_episode_output_dir, resolve_contained_audio_path, to_relative_audio_path};
pub use error::{explain_generation_failure, PodcastError, Result};
pub use generate::{generate_outline, generate_transcript, GenerationOptions, OutlineRequest, TranscriptRequest};
pub use models::{
    Content, Dialogue, EpisodeProfile, Outline, PodcastEpisode, Segment, SegmentSize, Speaker, SpeakerProfile,
};
pub use pipeline::{create_podcast, GenerationInput, PodcastFailure, PodcastOutput, PodcastRequest, PodcastSource};
pub use progress::{ProgressEvent, ProgressPhase, ProgressTracker, ProgressUpdate};
pub use refusal_guard::RefusalPolicy;
pub use retry::RetryConfig;
pub use script::parse_script;
