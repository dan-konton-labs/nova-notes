//! Audio and video transcription through the caller's speech-to-text model.
//!
//! * Audio that is small enough and short enough (or whose length cannot be
//!   probed) is sent as it is.
//! * Video always needs ffmpeg, to take out the audio track.
//! * Audio longer than `AudioOptions::segment_seconds`, or larger than
//!   `AudioOptions::max_upload_bytes`, is re-encoded by ffmpeg to mono 16 kHz
//!   MP3 and split into segments that each stay under the upload limit.
//!
//! Segments are transcribed with at most `AudioOptions::concurrency` requests
//! in flight and joined in their original order. ffmpeg and ffprobe are
//! optional: inputs that need them fail with a configuration error when they
//! cannot be found.

use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::Arc;
use std::time::Duration;

use bytes::Bytes;
use futures::stream::{self, StreamExt, TryStreamExt};
use nn_llm::{AiError, SpeechToTextModel, TranscriptionRequest};
use serde_json::json;

use super::detect::{Detected, FileKind};
use super::{ExtractContext, Extracted, FfmpegLocation};
use crate::error::ContentError;

/// Bit rate of re-encoded segments. 64 kbit/s mono is ample for speech and
/// keeps a 10-minute segment near 4.8 MB.
const SEGMENT_BITRATE: u64 = 64_000;

/// Formats the common speech-to-text APIs accept as uploads.
const PROVIDER_FORMATS: &[&str] = &["mp3", "mp4", "mpeg", "mpga", "m4a", "wav", "webm", "ogg", "oga", "flac"];

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Tools {
    pub ffmpeg: PathBuf,
    pub ffprobe: Option<PathBuf>,
}

/// Locate ffmpeg and ffprobe according to `location`.
pub(crate) fn find_tools(location: &FfmpegLocation) -> Option<Tools> {
    match location {
        FfmpegLocation::Disabled => None,
        FfmpegLocation::Paths { ffmpeg, ffprobe } => {
            Some(Tools { ffmpeg: ffmpeg.clone(), ffprobe: ffprobe.clone() })
        }
        FfmpegLocation::Auto => {
            let ffmpeg = find_on_path("ffmpeg")?;
            Some(Tools { ffmpeg, ffprobe: find_on_path("ffprobe") })
        }
    }
}

fn find_on_path(program: &str) -> Option<PathBuf> {
    let path = std::env::var_os("PATH")?;
    std::env::split_paths(&path).map(|dir| dir.join(program)).find(|candidate| is_executable(candidate))
}

fn is_executable(path: &Path) -> bool {
    let Ok(metadata) = std::fs::metadata(path) else {
        return false;
    };
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        metadata.is_file() && metadata.permissions().mode() & 0o111 != 0
    }
    #[cfg(not(unix))]
    {
        metadata.is_file()
    }
}

/// One piece of audio to send to the model.
#[derive(Debug, Clone)]
struct Piece {
    path: PathBuf,
    filename: String,
    content_type: String,
}

pub(crate) async fn transcribe_file(
    path: &Path,
    detected: &Detected,
    ctx: &ExtractContext,
) -> Result<Extracted, ContentError> {
    let is_video = detected.kind == FileKind::Video;
    let kind = if is_video { "video" } else { "audio" };
    refuse_playlists(path).await?;
    let model = ctx.speech_to_text.clone().ok_or_else(|| {
        ContentError::Configuration(format!(
            "transcribing {kind} needs a speech-to-text model, and none is configured"
        ))
    })?;

    let options = &ctx.audio;
    let size = tokio::fs::metadata(path)
        .await
        .map_err(|e| ContentError::FileOperation(format!("cannot read the file: {e}")))?
        .len();
    let tools = find_tools(&options.ffmpeg);
    let duration = match &tools {
        Some(Tools { ffprobe: Some(ffprobe), .. }) => probe_duration(ffprobe, path, ctx.limits.ffmpeg_timeout).await,
        _ => None,
    };
    let extension = detected.extension.clone().unwrap_or_else(|| extension_for_mime(&detected.mime).to_string());

    let too_long = options.segment_seconds > 0 && duration.is_some_and(|d| d > options.segment_seconds as f64);
    let too_large = size > options.max_upload_bytes;
    let odd_format = !PROVIDER_FORMATS.contains(&extension.as_str());
    let needs_ffmpeg = is_video || too_long || too_large;

    let work_dir = tempfile::Builder::new()
        .prefix("nn-content-audio-")
        .tempdir()
        .map_err(|e| ContentError::FileOperation(format!("cannot create a temporary directory: {e}")))?;

    let pieces = match (&tools, needs_ffmpeg) {
        (None, true) => {
            let why = if is_video {
                "extracting the audio track of a video".to_string()
            } else if too_large {
                format!("splitting audio larger than {} bytes", options.max_upload_bytes)
            } else {
                format!("splitting audio longer than {} seconds", options.segment_seconds)
            };
            return Err(ContentError::Configuration(format!("{why} needs ffmpeg, which was not found")));
        }
        (Some(tools), _) if needs_ffmpeg || odd_format => {
            let segment_seconds = segment_length(options.segment_seconds, options.max_upload_bytes);
            split_audio(&tools.ffmpeg, path, work_dir.path(), segment_seconds, ctx.limits.ffmpeg_timeout).await?
        }
        _ => vec![Piece {
            path: path.to_path_buf(),
            filename: format!("audio.{extension}"),
            content_type: if detected.mime.starts_with("audio/") { detected.mime.clone() } else { "audio/mpeg".to_string() },
        }],
    };

    if pieces.is_empty() {
        return Err(ContentError::EmptyContent(format!("the {kind} file has no audio")));
    }
    let segment_count = pieces.len();
    let transcripts = transcribe_pieces(model.clone(), pieces, ctx).await?;
    let content = transcripts.into_iter().map(|t| t.trim().to_string()).filter(|t| !t.is_empty()).collect::<Vec<_>>().join("\n\n");

    let mut metadata = json!({
        "mime_type": detected.mime,
        "size_bytes": size,
        "segments": segment_count,
        "stt_provider": model.provider(),
        "stt_model": model.model_name(),
    });
    if let Some(duration) = duration {
        metadata["duration_seconds"] = json!((duration * 10.0).round() / 10.0);
    }
    if let Some(language) = &ctx.language {
        metadata["language"] = json!(language);
    }
    if let Some(extension) = &detected.extension {
        metadata["extension"] = json!(extension);
    }
    Ok(Extracted {
        title: None,
        content,
        identified_type: kind.to_string(),
        url: None,
        file_path: None,
        metadata,
    })
}

/// Refuse an HLS playlist (`#EXTM3U`) or an ffconcat script that reached us
/// as audio or video: `detect` falls back to the extension for binary data it
/// does not recognise, and ffmpeg would follow such a file to other files.
async fn refuse_playlists(path: &Path) -> Result<(), ContentError> {
    use tokio::io::AsyncReadExt;

    let file = tokio::fs::File::open(path)
        .await
        .map_err(|e| ContentError::FileOperation(format!("cannot read the file: {e}")))?;
    let mut head = Vec::with_capacity(512);
    file.take(512)
        .read_to_end(&mut head)
        .await
        .map_err(|e| ContentError::FileOperation(format!("cannot read the file: {e}")))?;
    if let Some(identified_type) = playlist_type(&head) {
        return Err(ContentError::UnsupportedType {
            identified_type: identified_type.to_string(),
            reason: "playlists and concat scripts are not media files".to_string(),
        });
    }
    Ok(())
}

fn playlist_type(head: &[u8]) -> Option<&'static str> {
    let head = head.strip_prefix(b"\xEF\xBB\xBF").unwrap_or(head);
    let start = head.iter().position(|b| !b.is_ascii_whitespace()).unwrap_or(head.len());
    let head = &head[start..];
    if head.starts_with(b"#EXTM3U") {
        Some("application/vnd.apple.mpegurl")
    } else if head.starts_with(b"ffconcat") {
        Some("text/x-ffconcat")
    } else {
        None
    }
}

/// Segment length in seconds: the configured length, capped so a segment at
/// [`SEGMENT_BITRATE`] stays under 90% of the upload limit. A configured `0`
/// means "do not split", so only the size cap applies.
fn segment_length(configured: u64, max_upload_bytes: u64) -> u64 {
    let by_size = (max_upload_bytes.saturating_mul(8) / SEGMENT_BITRATE) * 9 / 10;
    let by_size = by_size.max(10);
    if configured == 0 {
        by_size
    } else {
        configured.min(by_size)
    }
}

fn extension_for_mime(mime: &str) -> &'static str {
    match mime {
        "audio/wav" | "audio/x-wav" => "wav",
        "audio/mp4" | "audio/m4a" | "audio/x-m4a" => "m4a",
        "audio/ogg" => "ogg",
        "audio/flac" => "flac",
        "audio/webm" => "webm",
        "audio/aac" => "aac",
        "audio/aiff" => "aiff",
        "audio/amr" => "amr",
        "audio/x-ms-wma" => "wma",
        "video/mp4" => "mp4",
        "video/webm" => "webm",
        "video/quicktime" => "mov",
        _ => "mp3",
    }
}

async fn transcribe_pieces(
    model: Arc<dyn SpeechToTextModel>,
    pieces: Vec<Piece>,
    ctx: &ExtractContext,
) -> Result<Vec<String>, ContentError> {
    let concurrency = ctx.audio.concurrency.clamp(1, 10);
    let timeout = ctx.limits.stt_timeout;
    let language = ctx.language.clone();
    stream::iter(pieces.into_iter().map(|piece| {
        let model = model.clone();
        let language = language.clone();
        async move {
            let audio = tokio::fs::read(&piece.path)
                .await
                .map_err(|e| ContentError::FileOperation(format!("cannot read an audio segment: {e}")))?;
            let request = TranscriptionRequest {
                audio: Bytes::from(audio),
                filename: piece.filename,
                content_type: Some(piece.content_type),
                language,
                prompt: None,
            };
            match tokio::time::timeout(timeout, model.transcribe(request)).await {
                Ok(Ok(text)) => Ok(text),
                Ok(Err(error)) => Err(ContentError::Transcription(error)),
                Err(_) => Err(ContentError::Transcription(AiError::Timeout(format!(
                    "speech-to-text did not answer within {} seconds",
                    timeout.as_secs()
                )))),
            }
        }
    }))
    // `buffered` keeps results in input order however requests finish.
    .buffered(concurrency)
    .try_collect()
    .await
}

/// Seconds of media in `path`, when ffprobe can tell.
async fn probe_duration(ffprobe: &Path, path: &Path, timeout: Duration) -> Option<f64> {
    let mut command = tokio::process::Command::new(ffprobe);
    command
        .args(["-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1"])
        .args(["-protocol_whitelist", "file,pipe", "-i"])
        .arg(path)
        .stdin(Stdio::null())
        .kill_on_drop(true);
    let output = tokio::time::timeout(timeout, command.output()).await.ok()?.ok()?;
    if !output.status.success() {
        return None;
    }
    String::from_utf8_lossy(&output.stdout).trim().parse::<f64>().ok().filter(|d| d.is_finite() && *d > 0.0)
}

/// Re-encode the audio of `input` to mono MP3 segments of `segment_seconds`.
async fn split_audio(
    ffmpeg: &Path,
    input: &Path,
    dir: &Path,
    segment_seconds: u64,
    timeout: Duration,
) -> Result<Vec<Piece>, ContentError> {
    let pattern = dir.join("segment_%04d.mp3");
    let mut command = tokio::process::Command::new(ffmpeg);
    command
        .args(["-hide_banner", "-nostdin", "-loglevel", "error", "-y", "-protocol_whitelist", "file,pipe", "-i"])
        .arg(input)
        .args(["-vn", "-map", "0:a:0", "-ac", "1", "-ar", "16000", "-c:a", "libmp3lame"])
        .args(["-b:a", &format!("{}", SEGMENT_BITRATE / 1000 * 1000)])
        .args(["-f", "segment", "-segment_time", &segment_seconds.to_string(), "-reset_timestamps", "1"])
        .arg(&pattern)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .kill_on_drop(true);

    let output = match tokio::time::timeout(timeout, command.output()).await {
        Err(_) => {
            return Err(ContentError::FileOperation(format!("ffmpeg did not finish within {} seconds", timeout.as_secs())))
        }
        Ok(Err(e)) => return Err(ContentError::Configuration(format!("ffmpeg could not be started: {e}"))),
        Ok(Ok(output)) => output,
    };
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        let last = stderr.lines().rev().find(|l| !l.trim().is_empty()).unwrap_or("unknown error").trim();
        if stderr.contains("matches no streams") || stderr.contains("does not contain any stream") {
            return Err(ContentError::EmptyContent("the file has no audio track".to_string()));
        }
        return Err(ContentError::FileOperation(format!("ffmpeg could not read the audio: {last}")));
    }

    let mut entries = tokio::fs::read_dir(dir)
        .await
        .map_err(|e| ContentError::FileOperation(format!("cannot list audio segments: {e}")))?;
    let mut segments = Vec::new();
    while let Some(entry) =
        entries.next_entry().await.map_err(|e| ContentError::FileOperation(format!("cannot list audio segments: {e}")))?
    {
        let name = entry.file_name().to_string_lossy().into_owned();
        if name.starts_with("segment_") && name.ends_with(".mp3") {
            // ffmpeg can leave a zero-length final segment.
            if entry.metadata().await.map(|m| m.len()).unwrap_or(0) > 0 {
                segments.push(name);
            }
        }
    }
    segments.sort();
    Ok(segments
        .into_iter()
        .map(|name| Piece { path: dir.join(&name), filename: name, content_type: "audio/mpeg".to_string() })
        .collect())
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;

    use async_trait::async_trait;

    use super::*;
    use crate::extract::{extract_content, AudioOptions, ExtractInput};

    /// Records every request and answers with the segment's file name, so
    /// tests can see order and count.
    #[derive(Default)]
    struct MockStt {
        requests: Mutex<Vec<TranscriptionRequest>>,
        fail: bool,
    }

    #[async_trait]
    impl SpeechToTextModel for MockStt {
        fn provider(&self) -> &str {
            "mock"
        }
        fn model_name(&self) -> &str {
            "mock-whisper"
        }
        async fn transcribe(&self, request: TranscriptionRequest) -> nn_llm::AiResult<String> {
            let filename = request.filename.clone();
            self.requests.lock().unwrap().push(request);
            if self.fail {
                return Err(AiError::RateLimit("quota exhausted".to_string()));
            }
            // Later segments answer first, to prove the join keeps order.
            if let Some(number) = filename.strip_prefix("segment_").and_then(|s| s.strip_suffix(".mp3")) {
                let n: u64 = number.parse().unwrap_or(0);
                tokio::time::sleep(Duration::from_millis(40u64.saturating_sub(n * 10))).await;
            }
            Ok(format!("text of {filename}"))
        }
    }

    /// A PCM WAV file of `seconds` of silence (8 kHz mono, 16-bit).
    fn wav(seconds: u32) -> Vec<u8> {
        let rate = 8000u32;
        let data_len = rate * 2 * seconds;
        let mut out = Vec::with_capacity(44 + data_len as usize);
        out.extend_from_slice(b"RIFF");
        out.extend_from_slice(&(36 + data_len).to_le_bytes());
        out.extend_from_slice(b"WAVEfmt ");
        out.extend_from_slice(&16u32.to_le_bytes());
        out.extend_from_slice(&1u16.to_le_bytes());
        out.extend_from_slice(&1u16.to_le_bytes());
        out.extend_from_slice(&rate.to_le_bytes());
        out.extend_from_slice(&(rate * 2).to_le_bytes());
        out.extend_from_slice(&2u16.to_le_bytes());
        out.extend_from_slice(&16u16.to_le_bytes());
        out.extend_from_slice(b"data");
        out.extend_from_slice(&data_len.to_le_bytes());
        out.resize(44 + data_len as usize, 0);
        out
    }

    fn ctx(model: Option<Arc<MockStt>>, audio: AudioOptions) -> ExtractContext {
        ExtractContext {
            speech_to_text: model.map(|m| m as Arc<dyn SpeechToTextModel>),
            language: Some("en".to_string()),
            audio,
            ..ExtractContext::default()
        }
    }

    fn no_ffmpeg() -> AudioOptions {
        AudioOptions { ffmpeg: FfmpegLocation::Disabled, ..AudioOptions::default() }
    }

    fn write(dir: &tempfile::TempDir, name: &str, bytes: &[u8]) -> PathBuf {
        let path = dir.path().join(name);
        std::fs::write(&path, bytes).unwrap();
        path
    }

    #[tokio::test]
    async fn small_audio_is_sent_whole_without_ffmpeg() {
        let dir = tempfile::tempdir().unwrap();
        let path = write(&dir, "upload-1", &wav(1));
        let model = Arc::new(MockStt::default());
        let input = ExtractInput::file(&path).with_file_name("Interview.WAV");
        let out = extract_content(input, &ctx(Some(model.clone()), no_ffmpeg())).await.unwrap();
        assert_eq!(out.identified_type, "audio");
        assert_eq!(out.content, "text of audio.wav");
        assert_eq!(out.title, None);
        assert_eq!(out.metadata["segments"], 1);
        assert_eq!(out.metadata["stt_model"], "mock-whisper");
        let requests = model.requests.lock().unwrap();
        assert_eq!(requests.len(), 1);
        assert_eq!(requests[0].filename, "audio.wav");
        assert_eq!(requests[0].content_type.as_deref(), Some("audio/wav"));
        assert_eq!(requests[0].language.as_deref(), Some("en"));
        assert_eq!(requests[0].audio.len(), wav(1).len());
    }

    #[tokio::test]
    async fn configuration_errors() {
        let dir = tempfile::tempdir().unwrap();
        let audio = write(&dir, "talk.wav", &wav(1));

        let error = extract_content(ExtractInput::file(&audio), &ctx(None, no_ffmpeg())).await.unwrap_err();
        assert!(matches!(&error, ContentError::Configuration(m) if m.contains("speech-to-text")), "{error:?}");

        let video = write(&dir, "clip.mp4", b"\x00\x00\x00\x20ftypisom\x00\x00\x02\x00isomiso2mp41");
        let model = Arc::new(MockStt::default());
        let error = extract_content(ExtractInput::file(&video), &ctx(Some(model.clone()), no_ffmpeg())).await.unwrap_err();
        assert!(matches!(&error, ContentError::Configuration(m) if m.contains("ffmpeg")), "{error:?}");

        let tight = AudioOptions { max_upload_bytes: 1000, ..no_ffmpeg() };
        let error = extract_content(ExtractInput::file(&audio), &ctx(Some(model.clone()), tight)).await.unwrap_err();
        assert!(matches!(&error, ContentError::Configuration(m) if m.contains("ffmpeg")), "{error:?}");
        assert!(model.requests.lock().unwrap().is_empty());

        let missing = AudioOptions {
            ffmpeg: FfmpegLocation::Paths { ffmpeg: dir.path().join("no-ffmpeg"), ffprobe: None },
            max_upload_bytes: 1000,
            ..AudioOptions::default()
        };
        let error = extract_content(ExtractInput::file(&audio), &ctx(Some(model), missing)).await.unwrap_err();
        assert!(matches!(&error, ContentError::Configuration(m) if m.contains("ffmpeg")), "{error:?}");
    }

    /// Issue #27: a text playlist or concat script named like audio (one NUL
    /// is enough to pass as binary) is refused, not handed to ffmpeg or the
    /// speech model.
    #[tokio::test]
    async fn playlists_and_concat_scripts_are_refused() {
        let dir = tempfile::tempdir().unwrap();
        let cases: [(&str, &[u8]); 4] = [
            ("list.mp3", b"#EXTM3U\n#EXTINF:1,\nfile:///etc/passwd\n\0"),
            ("bom.wav", b"\xEF\xBB\xBF  #EXTM3U\n#EXTINF:1,\nother.mp3\n\0"),
            ("concat.mp3", b"ffconcat version 1.0\nfile '/data/other.mp3'\n\0"),
            ("video.mp4", b"\n\nffconcat version 1.0\nfile 'x.mp4'\n\0"),
        ];
        for (name, bytes) in cases {
            let path = write(&dir, name, bytes);
            let model = Arc::new(MockStt::default());
            let error = extract_content(ExtractInput::file(&path), &ctx(Some(model.clone()), no_ffmpeg()))
                .await
                .unwrap_err();
            assert!(matches!(&error, ContentError::UnsupportedType { .. }), "{name}: {error:?}");
            assert!(model.requests.lock().unwrap().is_empty(), "{name}");
        }
    }

    #[tokio::test]
    async fn stt_errors_propagate() {
        let dir = tempfile::tempdir().unwrap();
        let path = write(&dir, "talk.wav", &wav(1));
        let model = Arc::new(MockStt { fail: true, ..MockStt::default() });
        let error = extract_content(ExtractInput::file(&path), &ctx(Some(model), no_ffmpeg())).await.unwrap_err();
        assert!(matches!(error, ContentError::Transcription(AiError::RateLimit(_))), "{error:?}");
        assert!(error.user_message(None).contains("service failed"));
    }

    #[test]
    fn segment_lengths() {
        let limit = 25 * 1024 * 1024;
        assert_eq!(segment_length(600, limit), 600);
        // 25 MiB at 64 kbit/s is ~3276 s; 90% of that.
        assert_eq!(segment_length(0, limit), 2948);
        assert_eq!(segment_length(10_000, limit), 2948);
        assert_eq!(segment_length(0, 1), 10);
        assert_eq!(extension_for_mime("audio/mp4"), "m4a");
        assert_eq!(extension_for_mime("audio/unknown"), "mp3");
    }

    #[test]
    fn tool_lookup() {
        assert_eq!(find_tools(&FfmpegLocation::Disabled), None);
        let explicit = FfmpegLocation::Paths { ffmpeg: "/x/ffmpeg".into(), ffprobe: None };
        assert_eq!(find_tools(&explicit).unwrap().ffmpeg, PathBuf::from("/x/ffmpeg"));
        assert!(find_on_path("definitely-not-a-real-program-name").is_none());
    }

    /// Generate media with ffmpeg; `None` (and the test is skipped) when it is absent.
    fn ffmpeg_make(dir: &tempfile::TempDir, name: &str, args: &[&str]) -> Option<PathBuf> {
        let tools = find_tools(&FfmpegLocation::Auto)?;
        let out = dir.path().join(name);
        let status = std::process::Command::new(&tools.ffmpeg)
            .args(["-hide_banner", "-loglevel", "error", "-y"])
            .args(args)
            .arg(&out)
            .status()
            .ok()?;
        status.success().then_some(out)
    }

    #[tokio::test]
    async fn long_audio_is_split_and_joined_in_order() {
        let dir = tempfile::tempdir().unwrap();
        let Some(path) = ffmpeg_make(&dir, "tone.mp3", &["-f", "lavfi", "-i", "sine=frequency=440:duration=3.5"]) else {
            eprintln!("ffmpeg not found; skipping");
            return;
        };
        let model = Arc::new(MockStt::default());
        let options = AudioOptions { segment_seconds: 1, concurrency: 4, ..AudioOptions::default() };
        let out = extract_content(ExtractInput::file(&path), &ctx(Some(model.clone()), options)).await.unwrap();
        let segments = out.metadata["segments"].as_u64().unwrap();
        assert!((3..=5).contains(&segments), "{segments} segments");
        let expected: Vec<String> = (0..segments).map(|i| format!("text of segment_{i:04}.mp3")).collect();
        assert_eq!(out.content, expected.join("\n\n"));
        assert!(out.metadata["duration_seconds"].as_f64().unwrap() > 3.0);
        let requests = model.requests.lock().unwrap();
        assert!(requests.iter().all(|r| r.content_type.as_deref() == Some("audio/mpeg") && !r.audio.is_empty()));
    }

    #[tokio::test]
    async fn video_audio_track_is_transcribed() {
        let dir = tempfile::tempdir().unwrap();
        let Some(path) = ffmpeg_make(
            &dir,
            "clip.mp4",
            &[
                "-f", "lavfi", "-i", "testsrc=duration=2:size=64x64:rate=5",
                "-f", "lavfi", "-i", "sine=frequency=300:duration=2",
                "-shortest", "-c:v", "mpeg4", "-c:a", "aac",
            ],
        ) else {
            eprintln!("ffmpeg not found; skipping");
            return;
        };
        let model = Arc::new(MockStt::default());
        let out = extract_content(ExtractInput::file(&path), &ctx(Some(model.clone()), AudioOptions::default())).await.unwrap();
        assert_eq!(out.identified_type, "video");
        assert_eq!(out.content, "text of segment_0000.mp3");
        assert_eq!(model.requests.lock().unwrap().len(), 1);

        // A video without an audio track has nothing to transcribe.
        let Some(silent) =
            ffmpeg_make(&dir, "silent.mp4", &["-f", "lavfi", "-i", "testsrc=duration=1:size=64x64:rate=5", "-c:v", "mpeg4"])
        else {
            return;
        };
        let error = extract_content(ExtractInput::file(&silent), &ctx(Some(model), AudioOptions::default())).await.unwrap_err();
        assert!(matches!(error, ContentError::EmptyContent(_)), "{error:?}");
    }
}
