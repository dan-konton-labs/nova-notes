//! Speech synthesis and audio assembly.
//!
//! * [`synthesize_clips`] speaks every dialogue turn with its speaker's voice,
//!   a few turns at a time, into numbered files in a `clips/` directory.
//! * [`combine_clips`] joins the clips into one episode file. With `ffmpeg` on
//!   `PATH` it decodes everything and encodes MP3, like the original. Without
//!   it, MP3 clips of one format are joined frame by frame and WAV clips of
//!   one format are joined under a new header; anything else is an error that
//!   says so.
//! * [`duration_seconds`] measures MP3, M4A/MP4 and WAV files without external
//!   tools (the original used `mutagen`).

mod ffmpeg;
mod mp3;
mod mp4;
mod wav;

use std::collections::HashMap;
use std::fmt;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use futures::future::try_join_all;
use nn_llm::{AiError, AudioData, SharedTextToSpeechModel, TextToSpeechModel};

use crate::error::{PodcastError, Result};
use crate::models::{Dialogue, SpeakerProfile};
use crate::retry::{retry, RetryConfig};

pub use ffmpeg::find_executable;

/// Silence inserted between consecutive clips unless the caller says otherwise.
pub const DEFAULT_AUDIO_GAP_MS: u32 = 400;
/// Text-to-speech requests sent in parallel (the original's `TTS_BATCH_SIZE`).
pub const DEFAULT_TTS_BATCH_SIZE: usize = 5;

/// Whether to use an `ffmpeg` binary for combining clips.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum FfmpegMode {
    /// Use `ffmpeg` when it is on `PATH`, the pure-Rust joiner otherwise.
    #[default]
    Auto,
    /// Never run `ffmpeg`.
    Disabled,
    /// Use this binary.
    Path(PathBuf),
}

/// Settings of the audio half of the pipeline.
#[derive(Debug, Clone)]
pub struct AudioOptions {
    /// Clips synthesized concurrently; batches run one after another. Lower it
    /// for rate-limited or single-threaded speech servers.
    pub batch_size: usize,
    /// Pause between batches.
    pub batch_pause: Duration,
    /// Silence between consecutive clips, in milliseconds. `0` disables it.
    pub gap_ms: u32,
    /// Applied to transient speech errors (rate limit, network, timeout, 5xx).
    pub retry: RetryConfig,
    pub ffmpeg: FfmpegMode,
}

impl Default for AudioOptions {
    fn default() -> Self {
        Self {
            batch_size: DEFAULT_TTS_BATCH_SIZE,
            batch_pause: Duration::from_secs(1),
            gap_ms: DEFAULT_AUDIO_GAP_MS,
            retry: RetryConfig::default(),
            ffmpeg: FfmpegMode::Auto,
        }
    }
}

/// The speech models of a speaker profile: the profile's own, plus the models
/// of speakers that override it (`Speaker::voice_model`), keyed by speaker name.
#[derive(Clone)]
pub struct TtsModels {
    pub default: SharedTextToSpeechModel,
    pub per_speaker: HashMap<String, SharedTextToSpeechModel>,
}

impl TtsModels {
    pub fn new(default: SharedTextToSpeechModel) -> Self {
        Self { default, per_speaker: HashMap::new() }
    }

    pub fn with_speaker(mut self, speaker_name: impl Into<String>, model: SharedTextToSpeechModel) -> Self {
        self.per_speaker.insert(speaker_name.into(), model);
        self
    }

    pub fn for_speaker(&self, speaker_name: &str) -> &dyn TextToSpeechModel {
        self.per_speaker.get(speaker_name).unwrap_or(&self.default).as_ref()
    }
}

impl fmt::Debug for TtsModels {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let key = |model: &SharedTextToSpeechModel| format!("{}/{}", model.provider(), model.model_name());
        let overrides: HashMap<&String, String> =
            self.per_speaker.iter().map(|(name, model)| (name, key(model))).collect();
        f.debug_struct("TtsModels")
            .field("default", &key(&self.default))
            .field("per_speaker", &overrides)
            .finish()
    }
}

// ---------------------------------------------------------------------------
// Synthesis
// ---------------------------------------------------------------------------

/// File extension for a clip, from what the bytes are rather than what the
/// provider says they are; the content type decides only when the bytes are
/// not recognised.
fn clip_extension(audio: &AudioData) -> &'static str {
    let bytes = audio.bytes.as_ref();
    if bytes.starts_with(b"RIFF") && bytes.get(8..12) == Some(b"WAVE") {
        "wav"
    } else if bytes.starts_with(b"ID3") || mp3::FrameHeader::parse(bytes).is_some() {
        "mp3"
    } else if bytes.starts_with(b"OggS") {
        "ogg"
    } else if bytes.starts_with(b"fLaC") {
        "flac"
    } else if bytes.get(4..8) == Some(b"ftyp") {
        "m4a"
    } else {
        audio.extension()
    }
}

async fn synthesize_clip(
    index: usize,
    turn: &Dialogue,
    voice: &str,
    model: &dyn TextToSpeechModel,
    clips_dir: &Path,
    retry_config: &RetryConfig,
) -> Result<PathBuf> {
    tracing::info!("Generating audio clip {index:04} for {}", turn.speaker);
    let what = format!("{}/{} clip {index:04}", model.provider(), model.model_name());
    let audio =
        retry(retry_config, &what, AiError::is_retryable, || model.synthesize(&turn.dialogue, voice)).await?;
    if audio.bytes.is_empty() {
        return Err(PodcastError::Audio(format!(
            "The voice model returned no audio for clip {index:04} ({})",
            turn.speaker
        )));
    }
    let extension = clip_extension(&audio);
    if extension == "pcm" {
        return Err(PodcastError::Audio(format!(
            "The voice model returned raw PCM audio ({}) for clip {index:04}; \
             clips must be in a container format such as MP3 or WAV",
            audio.content_type
        )));
    }
    let path = clips_dir.join(format!("{index:04}.{extension}"));
    tokio::fs::write(&path, &audio.bytes)
        .await
        .map_err(|e| PodcastError::io(format!("Failed to write audio clip {}", path.display()), e))?;
    tracing::info!("Generated audio clip: {}", path.display());
    Ok(path)
}

/// Speak every turn of `transcript` into `clips_dir/NNNN.<ext>`.
///
/// Turns are processed in batches of [`AudioOptions::batch_size`]: the clips
/// of a batch are requested concurrently, batches run in sequence. The
/// returned paths are in transcript order whatever order the clips finished
/// in. `on_clip(done, total)` is called as each clip is written.
pub async fn synthesize_clips(
    transcript: &[Dialogue],
    speaker_profile: &SpeakerProfile,
    tts: &TtsModels,
    clips_dir: &Path,
    options: &AudioOptions,
    on_clip: &(dyn Fn(usize, usize) + Send + Sync),
) -> Result<Vec<PathBuf>> {
    let total = transcript.len();
    let batch_size = options.batch_size.max(1);
    // Resolve every voice before spending anything on speech.
    let voices: Vec<&str> = transcript
        .iter()
        .map(|turn| speaker_profile.speaker_by_name(&turn.speaker).map(|speaker| speaker.voice_id.as_str()))
        .collect::<Result<_>>()?;

    tokio::fs::create_dir_all(clips_dir)
        .await
        .map_err(|e| PodcastError::io(format!("Failed to create {}", clips_dir.display()), e))?;
    tracing::info!("Generating {total} audio clips in sequential batches of {batch_size}");

    let done = AtomicUsize::new(0);
    let mut paths = Vec::with_capacity(total);
    for (batch_number, batch_start) in (0..total).step_by(batch_size).enumerate() {
        let batch_end = (batch_start + batch_size).min(total);
        let batch = (batch_start..batch_end).map(|index| {
            let turn = &transcript[index];
            let voice = voices[index];
            let done = &done;
            async move {
                let model = tts.for_speaker(&turn.speaker);
                let path = synthesize_clip(index, turn, voice, model, clips_dir, &options.retry).await?;
                on_clip(done.fetch_add(1, Ordering::SeqCst) + 1, total);
                Ok::<_, PodcastError>(path)
            }
        });
        paths.extend(try_join_all(batch).await?);
        tracing::info!("Completed batch {}/{}", batch_number + 1, total.div_ceil(batch_size));

        if batch_end < total && !options.batch_pause.is_zero() {
            tokio::time::sleep(options.batch_pause).await;
        }
    }
    tracing::info!("Generated all {} audio clips", paths.len());
    Ok(paths)
}

// ---------------------------------------------------------------------------
// Combining
// ---------------------------------------------------------------------------

/// How an episode file was produced.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CombineMethod {
    /// Decoded and re-encoded to MP3 by `ffmpeg`.
    Ffmpeg,
    /// MP3 frames joined without re-encoding.
    Mp3Frames,
    /// WAV samples joined under a new header.
    WavSamples,
}

/// The result of [`combine_clips`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CombinedAudio {
    pub path: PathBuf,
    pub method: CombineMethod,
}

fn extension_of(path: &Path) -> String {
    path.extension().and_then(|e| e.to_str()).unwrap_or_default().to_ascii_lowercase()
}

fn join_without_ffmpeg(clips: &[PathBuf], gap_ms: u32) -> Result<(Vec<u8>, &'static str, CombineMethod)> {
    let mut kinds: Vec<String> = clips.iter().map(|clip| extension_of(clip)).collect();
    kinds.sort();
    kinds.dedup();

    let read_all = || -> Result<Vec<Vec<u8>>> {
        clips
            .iter()
            .map(|clip| {
                std::fs::read(clip)
                    .map_err(|e| PodcastError::io(format!("Failed to read audio clip {}", clip.display()), e))
            })
            .collect()
    };
    let failed = |detail: String| {
        PodcastError::Audio(format!(
            "Could not combine the audio clips without ffmpeg: {detail}. \
             Install ffmpeg so clips can be decoded and re-encoded."
        ))
    };

    match kinds.as_slice() {
        [kind] if kind == "mp3" => {
            let joined = mp3::concat(&read_all()?, gap_ms).map_err(failed)?;
            Ok((joined, "mp3", CombineMethod::Mp3Frames))
        }
        [kind] if kind == "wav" => {
            let joined = wav::concat(&read_all()?, gap_ms).map_err(failed)?;
            Ok((joined, "wav", CombineMethod::WavSamples))
        }
        [kind] => Err(failed(format!("'.{kind}' clips can only be joined by ffmpeg"))),
        kinds => Err(failed(format!(
            "the clips are in different formats ({})",
            kinds.iter().map(|kind| format!(".{kind}")).collect::<Vec<_>>().join(", ")
        ))),
    }
}

fn temp_path(audio_dir: &Path, file_name: &str) -> PathBuf {
    audio_dir.join(format!(".{file_name}.{}.tmp", uuid::Uuid::new_v4().simple()))
}

async fn move_into_place(temp: &Path, target: &Path) -> Result<()> {
    tokio::fs::rename(temp, target)
        .await
        .map_err(|e| PodcastError::io(format!("Failed to write {}", target.display()), e))
}

/// Join `clips`, in the order given, into `audio_dir/<file_stem>.<ext>`.
///
/// The extension is `mp3` except when WAV clips were joined without `ffmpeg`,
/// which gives `wav`. The file appears under its final name only when it is
/// complete: a failed run never leaves a partial episode behind.
pub async fn combine_clips(
    clips: &[PathBuf],
    audio_dir: &Path,
    file_stem: &str,
    options: &AudioOptions,
) -> Result<CombinedAudio> {
    if clips.is_empty() {
        return Err(PodcastError::InvalidInput("There are no audio clips to combine".into()));
    }
    tokio::fs::create_dir_all(audio_dir)
        .await
        .map_err(|e| PodcastError::io(format!("Failed to create {}", audio_dir.display()), e))?;

    let ffmpeg = match &options.ffmpeg {
        FfmpegMode::Auto => find_executable("ffmpeg"),
        FfmpegMode::Path(path) => Some(path.clone()),
        FfmpegMode::Disabled => None,
    };

    if let Some(ffmpeg) = ffmpeg {
        let file_name = format!("{file_stem}.mp3");
        let target = audio_dir.join(&file_name);
        let temp = temp_path(audio_dir, &file_name);
        let outcome = ffmpeg::combine(&ffmpeg, clips, &temp, options.gap_ms).await;
        if let Err(detail) = outcome {
            let _ = tokio::fs::remove_file(&temp).await;
            return Err(PodcastError::Audio(format!("Podcast audio generation failed: {detail}")));
        }
        move_into_place(&temp, &target).await?;
        tracing::info!("Combined audio saved to: {}", target.display());
        return Ok(CombinedAudio { path: target, method: CombineMethod::Ffmpeg });
    }

    let owned_clips = clips.to_vec();
    let gap_ms = options.gap_ms;
    let (bytes, extension, method) =
        tokio::task::spawn_blocking(move || join_without_ffmpeg(&owned_clips, gap_ms))
            .await
            .map_err(|e| PodcastError::Audio(format!("Combining the audio clips was interrupted: {e}")))??;

    let file_name = format!("{file_stem}.{extension}");
    let target = audio_dir.join(&file_name);
    let temp = temp_path(audio_dir, &file_name);
    if let Err(e) = tokio::fs::write(&temp, &bytes).await {
        let _ = tokio::fs::remove_file(&temp).await;
        return Err(PodcastError::io(format!("Failed to write {}", target.display()), e));
    }
    move_into_place(&temp, &target).await?;
    tracing::info!("Combined audio saved to: {} (joined without ffmpeg)", target.display());
    Ok(CombinedAudio { path: target, method })
}

// ---------------------------------------------------------------------------
// Duration
// ---------------------------------------------------------------------------

/// Length of an audio file in seconds: MP3 (frame scan, honouring Xing/Info
/// and VBRI headers), M4A/MP4 (movie header) or WAV. The format is recognised
/// from the content, not the file name.
///
/// This reads the file on the calling thread; from async code run it through
/// `tokio::task::spawn_blocking`.
pub fn duration_seconds(path: &Path) -> Result<f64> {
    let unreadable = |detail: String| {
        PodcastError::Audio(format!("Could not read the audio length of {}: {detail}", path.display()))
    };
    let io = |e: std::io::Error| unreadable(e.to_string());

    let mut file = std::fs::File::open(path).map_err(io)?;
    let file_len = file.metadata().map_err(io)?.len();
    let mut head = [0u8; 12];
    let head_len = file.by_ref().take(12).read(&mut head).map_err(io)?;
    let head = &head[..head_len];

    if head.get(4..8) == Some(b"ftyp") {
        return mp4::duration_seconds(&mut file, file_len).map_err(unreadable);
    }
    let data = std::fs::read(path).map_err(io)?;
    if head.starts_with(b"RIFF") && head.get(8..12) == Some(b"WAVE") {
        return wav::duration_seconds(&data).map_err(unreadable);
    }
    mp3::duration_seconds(&data).map_err(unreadable)
}

#[cfg(test)]
mod tests {
    use bytes::Bytes;

    use super::*;

    fn audio(bytes: &[u8], content_type: &str) -> AudioData {
        AudioData { bytes: Bytes::copy_from_slice(bytes), content_type: content_type.to_string() }
    }

    #[test]
    fn clip_extension_trusts_the_bytes_over_the_content_type() {
        let mp3 = mp3::testing::frames([0xFF, 0xF3, 0x84, 0xC0], 2);
        assert_eq!(clip_extension(&audio(&mp3, "audio/wav")), "mp3");
        assert_eq!(clip_extension(&audio(b"ID3\x04\x00\x00\x00\x00\x00\x00", "application/octet-stream")), "mp3");
        let wav = wav::testing::pcm16(8_000, 1, &[1, 2]);
        assert_eq!(clip_extension(&audio(&wav, "audio/mpeg")), "wav");
        assert_eq!(clip_extension(&audio(b"OggS\0\0", "audio/mpeg")), "ogg");
        assert_eq!(clip_extension(&audio(b"fLaC\0\0", "audio/mpeg")), "flac");
        assert_eq!(clip_extension(&audio(&mp4::testing::ftyp(), "audio/mpeg")), "m4a");
        // Unrecognised bytes: the content type decides.
        assert_eq!(clip_extension(&audio(&[0u8; 16], "audio/L16;rate=24000")), "pcm");
        assert_eq!(clip_extension(&audio(&[0u8; 16], "audio/aac")), "aac");
    }

    #[test]
    fn duration_is_read_by_content_for_each_format() {
        let dir = tempfile::tempdir().unwrap();
        let write = |name: &str, data: &[u8]| {
            let path = dir.path().join(name);
            std::fs::write(&path, data).unwrap();
            path
        };

        // The original's publishing fixture: mutagen and ffprobe both round it to 26.
        let mp3 = write("ep1.mp3", &mp3::testing::frames([0xFF, 0xFB, 0x90, 0x00], 1000));
        assert_eq!(duration_seconds(&mp3).unwrap().round(), 26.0);

        let m4a = write("ep1.m4a", &mp4::testing::m4a(1000, 3000));
        assert_eq!(duration_seconds(&m4a).unwrap(), 3.0);

        let wav = write("ep1.wav", &wav::testing::pcm16(24_000, 1, &vec![0i16; 36_000]));
        assert_eq!(duration_seconds(&wav).unwrap(), 1.5);

        // The name does not matter.
        let disguised = write("really-m4a.mp3", &mp4::testing::m4a(1000, 4500));
        assert_eq!(duration_seconds(&disguised).unwrap(), 4.5);
    }

    #[test]
    fn unreadable_audio_is_an_error_naming_the_file() {
        let dir = tempfile::tempdir().unwrap();
        let broken = dir.path().join("broken.mp3");
        std::fs::write(&broken, b"definitely not audio").unwrap();
        let error = duration_seconds(&broken).unwrap_err().to_string();
        assert!(error.contains("broken.mp3"), "{error}");

        let stub = dir.path().join("stub.wav");
        std::fs::write(&stub, b"RIFF....WAVE").unwrap();
        assert!(duration_seconds(&stub).is_err());
        assert!(duration_seconds(&dir.path().join("missing.mp3")).is_err());
    }

    fn write_clips(dir: &Path, clips: &[(&str, Vec<u8>)]) -> Vec<PathBuf> {
        clips
            .iter()
            .map(|(name, data)| {
                let path = dir.join(name);
                std::fs::write(&path, data).unwrap();
                path
            })
            .collect()
    }

    fn no_ffmpeg(gap_ms: u32) -> AudioOptions {
        AudioOptions { ffmpeg: FfmpegMode::Disabled, gap_ms, ..AudioOptions::default() }
    }

    const MONO_24K: [u8; 4] = [0xFF, 0xF3, 0x84, 0xC0];

    #[tokio::test]
    async fn mp3_clips_are_joined_without_ffmpeg() {
        let dir = tempfile::tempdir().unwrap();
        let mut tagged = mp3::testing::id3v2(64);
        tagged.extend(mp3::testing::xing(MONO_24K, b"Info", 50));
        tagged.extend(mp3::testing::frames(MONO_24K, 50));
        let clips = write_clips(
            dir.path(),
            &[("0000.mp3", tagged), ("0001.mp3", mp3::testing::frames(MONO_24K, 75))],
        );

        let combined = combine_clips(&clips, &dir.path().join("audio"), "episode", &no_ffmpeg(0)).await.unwrap();
        assert_eq!(combined.method, CombineMethod::Mp3Frames);
        assert_eq!(combined.path, dir.path().join("audio/episode.mp3"));
        // 125 frames of 24 ms; a naive byte join would report the first clip's 1.2 s.
        assert!((duration_seconds(&combined.path).unwrap() - 3.0).abs() < 1e-6);
        let leftovers: Vec<_> = std::fs::read_dir(dir.path().join("audio")).unwrap().collect();
        assert_eq!(leftovers.len(), 1, "no temporary file is left behind");
    }

    #[tokio::test]
    async fn wav_clips_are_joined_without_ffmpeg_with_gaps() {
        let dir = tempfile::tempdir().unwrap();
        let clips = write_clips(
            dir.path(),
            &[
                ("0000.wav", wav::testing::tone(16_000, 0.5, 440.0)),
                ("0001.wav", wav::testing::tone(16_000, 0.25, 660.0)),
            ],
        );
        let combined = combine_clips(&clips, &dir.path().join("audio"), "ep", &no_ffmpeg(400)).await.unwrap();
        assert_eq!(combined.method, CombineMethod::WavSamples);
        assert_eq!(combined.path.extension().unwrap(), "wav");
        assert!((duration_seconds(&combined.path).unwrap() - 1.15).abs() < 1e-6);
    }

    #[tokio::test]
    async fn mixed_and_other_formats_without_ffmpeg_are_a_clear_error() {
        let dir = tempfile::tempdir().unwrap();
        let mixed = write_clips(
            dir.path(),
            &[
                ("0000.mp3", mp3::testing::frames(MONO_24K, 5)),
                ("0001.wav", wav::testing::tone(16_000, 0.1, 440.0)),
            ],
        );
        let audio_dir = dir.path().join("audio");
        let error = combine_clips(&mixed, &audio_dir, "ep", &no_ffmpeg(0)).await.unwrap_err();
        let message = error.to_string();
        assert!(matches!(error, PodcastError::Audio(_)));
        assert!(message.contains("different formats (.mp3, .wav)"), "{message}");
        assert!(message.contains("Install ffmpeg"), "{message}");

        let ogg = write_clips(dir.path(), &[("0002.ogg", b"OggS....".to_vec())]);
        let message = combine_clips(&ogg, &audio_dir, "ep", &no_ffmpeg(0)).await.unwrap_err().to_string();
        assert!(message.contains("'.ogg' clips can only be joined by ffmpeg"), "{message}");

        let two_rates = write_clips(
            dir.path(),
            &[
                ("0003.mp3", mp3::testing::frames(MONO_24K, 5)),
                ("0004.mp3", mp3::testing::frames([0xFF, 0xFB, 0x90, 0x00], 5)),
            ],
        );
        let message = combine_clips(&two_rates, &audio_dir, "ep", &no_ffmpeg(0)).await.unwrap_err().to_string();
        assert!(message.contains("different MP3 formats"), "{message}");

        assert_eq!(std::fs::read_dir(&audio_dir).unwrap().count(), 0, "nothing was written");
        assert!(matches!(
            combine_clips(&[], &audio_dir, "ep", &no_ffmpeg(0)).await,
            Err(PodcastError::InvalidInput(_))
        ));
    }

    #[tokio::test]
    async fn ffmpeg_combines_and_encodes_when_available() {
        if find_executable("ffmpeg").is_none() {
            eprintln!("skipped: ffmpeg is not installed");
            return;
        }
        let dir = tempfile::tempdir().unwrap();
        let clips = write_clips(
            dir.path(),
            &[
                ("0000.wav", wav::testing::tone(24_000, 1.0, 440.0)),
                ("0001.wav", wav::testing::tone(24_000, 0.5, 660.0)),
            ],
        );
        let options = AudioOptions { gap_ms: 500, ..AudioOptions::default() };
        let combined = combine_clips(&clips, &dir.path().join("audio"), "ep", &options).await.unwrap();
        assert_eq!(combined.method, CombineMethod::Ffmpeg);
        assert_eq!(combined.path.extension().unwrap(), "mp3");
        // 1.0 + 0.5 gap + 0.5, read back from LAME's own Info header by the
        // frame scanner (encoder delay and padding add a few frames).
        let seconds = duration_seconds(&combined.path).unwrap();
        assert!((seconds - 2.0).abs() < 0.15, "{seconds}");
    }

    /// Issue #31: an ID3 title that imitates ffmpeg's stream line must not
    /// reach the filter graph. Before the fix the probe regex took the title
    /// as the channel layout and the injected `ametadata=...:file=` filter
    /// wrote `target`.
    #[tokio::test]
    async fn a_failing_ffmpeg_leaves_no_partial_file() {
        let dir = tempfile::tempdir().unwrap();
        let clips = write_clips(dir.path(), &[("0000.mp3", mp3::testing::frames(MONO_24K, 5))]);
        let options =
            AudioOptions { ffmpeg: FfmpegMode::Path(dir.path().join("no-such-ffmpeg")), ..AudioOptions::default() };
        let audio_dir = dir.path().join("audio");
        let error = combine_clips(&clips, &audio_dir, "ep", &options).await.unwrap_err().to_string();
        assert!(error.starts_with("Podcast audio generation failed: failed to run ffmpeg"), "{error}");
        assert_eq!(std::fs::read_dir(&audio_dir).unwrap().count(), 0);
    }
}
