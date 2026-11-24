//! Combining clips with an `ffmpeg` binary, when one is installed.
//!
//! Same invocation as podcast-creator: one process, the `concat` filter with
//! every input's timestamps reset (so a clip is decoded to its real end even
//! when its header states a wrong length), all inputs normalised to the first
//! clip's sample rate and channel layout, optional silence between clips, and
//! MP3 at 128 kbit/s out.
//!
//! Clips come from a speech provider, so they are untrusted: the first clip's
//! format is read with `ffprobe` (JSON, never ffmpeg's log) and checked before
//! it is written into the filter graph, every clip is opened with a forced
//! demuxer and `-protocol_whitelist file`, and tags are not copied to the
//! episode.

use std::ffi::OsStr;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use tokio::process::Command;

/// Look for an executable called `name` in the directories of `PATH`.
pub fn find_executable(name: &str) -> Option<PathBuf> {
    find_in(name, std::env::var_os("PATH").as_deref()?)
}

fn find_in(name: &str, search_path: &OsStr) -> Option<PathBuf> {
    std::env::split_paths(search_path)
        .filter(|dir| !dir.as_os_str().is_empty())
        .flat_map(|dir| {
            let plain = dir.join(name);
            let exe = cfg!(windows).then(|| dir.join(format!("{name}.exe")));
            std::iter::once(plain).chain(exe)
        })
        .find(|candidate| is_executable(candidate))
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

struct Output {
    success: bool,
    code: Option<i32>,
    stdout: String,
    stderr: String,
}

async fn run(program: &Path, name: &str, args: &[String]) -> Result<Output, String> {
    let output = Command::new(program)
        .args(args)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true)
        .output()
        .await
        .map_err(|e| format!("failed to run {name} ({}): {e}", program.display()))?;
    Ok(Output {
        success: output.status.success(),
        code: output.status.code(),
        stdout: String::from_utf8_lossy(&output.stdout).into_owned(),
        stderr: String::from_utf8_lossy(&output.stderr).into_owned(),
    })
}

/// Lowest and highest sample rate accepted from a clip.
const MIN_SAMPLE_RATE: u32 = 1_000;
const MAX_SAMPLE_RATE: u32 = 192_000;

/// Sample rate and channel layout of the first clip, checked so that both are
/// safe to write into a filter graph: the rate is a number in a sane range and
/// the layout is a plain name such as `mono`, `stereo` or `5.1`. The fields
/// are private, so a value that reaches [`build_concat_filter`] has always
/// passed [`AudioFormat::new`], wherever it came from.
#[derive(Debug, Clone, PartialEq, Eq)]
struct AudioFormat {
    rate: u32,
    layout: String,
}

impl AudioFormat {
    fn new(rate: u32, layout: &str) -> Option<Self> {
        // `5.1(side)` is normalised to `5.1`, as before.
        let layout = layout.split('(').next().unwrap_or_default().trim();
        let plain = !layout.is_empty()
            && layout.len() <= 32
            && layout.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'.');
        ((MIN_SAMPLE_RATE..=MAX_SAMPLE_RATE).contains(&rate) && plain)
            .then(|| Self { rate, layout: layout.to_string() })
    }

    /// Read the first audio stream from `ffprobe -show_entries
    /// stream=sample_rate,channels,channel_layout -of json`. WAV files often
    /// state no layout; one or two channels are then `mono` / `stereo`.
    fn from_ffprobe_json(json: &str) -> Option<Self> {
        let value: serde_json::Value = serde_json::from_str(json).ok()?;
        let stream = value.get("streams")?.as_array()?.first()?;
        let rate = match stream.get("sample_rate")? {
            serde_json::Value::String(text) => text.trim().parse().ok()?,
            number => u32::try_from(number.as_u64()?).ok()?,
        };
        let layout = match stream.get("channel_layout").and_then(serde_json::Value::as_str) {
            Some(layout) => layout.to_string(),
            None => match stream.get("channels").and_then(serde_json::Value::as_u64)? {
                1 => "mono".to_string(),
                2 => "stereo".to_string(),
                _ => return None,
            },
        };
        Self::new(rate, &layout)
    }
}

/// The ffmpeg demuxer for a clip, from its extension (which
/// `synthesize_clips` chose from the clip's bytes). Forcing it keeps ffmpeg
/// from probing a clip into a playlist, concat script or other format that
/// opens further files.
fn demuxer_for(clip: &Path) -> Result<&'static str, String> {
    let extension = clip.extension().and_then(OsStr::to_str).unwrap_or_default().to_ascii_lowercase();
    Ok(match extension.as_str() {
        "mp3" => "mp3",
        "wav" => "wav",
        "ogg" => "ogg",
        "flac" => "flac",
        "m4a" | "mp4" => "mov",
        "aac" => "aac",
        _ => return Err(format!("unsupported audio clip format: {}", clip.display())),
    })
}

/// Input options for one clip: local files only, a forced demuxer, and the
/// path behind an explicit `file:` so it is never read as a protocol URL.
fn input_args(clip: &Path) -> Result<Vec<String>, String> {
    Ok(vec![
        "-protocol_whitelist".into(),
        "file".into(),
        "-f".into(),
        demuxer_for(clip)?.into(),
        "-i".into(),
        format!("file:{}", clip.to_string_lossy()),
    ])
}

/// `ffprobe` next to `ffmpeg`, else the one on `PATH`.
fn find_ffprobe(ffmpeg: &Path) -> Option<PathBuf> {
    let name = if ffmpeg.extension().is_some_and(|e| e.eq_ignore_ascii_case("exe")) { "ffprobe.exe" } else { "ffprobe" };
    ffmpeg
        .parent()
        .map(|dir| dir.join(name))
        .filter(|sibling| is_executable(sibling))
        .or_else(|| find_executable("ffprobe"))
}

async fn probe_audio_format(ffprobe: &Path, clip: &Path) -> Result<Option<AudioFormat>, String> {
    let mut args: Vec<String> = ["-v", "error", "-select_streams", "a:0"].map(String::from).to_vec();
    args.extend(["-show_entries", "stream=sample_rate,channels,channel_layout", "-of", "json"].map(String::from));
    args.extend(input_args(clip)?);
    let probe = run(ffprobe, "ffprobe", &args).await?;
    Ok(probe.success.then(|| AudioFormat::from_ffprobe_json(&probe.stdout)).flatten())
}

/// Seconds as ffmpeg's `atrim=duration=` takes them, written the way Python
/// prints the float `gap_ms / 1000`.
fn gap_seconds(gap_ms: u32) -> String {
    let seconds = f64::from(gap_ms) / 1000.0;
    if seconds.fract() == 0.0 {
        format!("{seconds:.1}")
    } else {
        seconds.to_string()
    }
}

/// Build a filter graph that resets each input's timestamps and concatenates
/// them. With a known format every input is normalised to it and `gap_ms` of
/// silence is inserted between inputs.
fn build_concat_filter(num_inputs: usize, audio_format: Option<&AudioFormat>, gap_ms: u32) -> String {
    let normalize = audio_format
        .map(|AudioFormat { rate, layout }| {
            format!("aresample={rate},aformat=sample_rates={rate}:channel_layouts={layout},")
        })
        .unwrap_or_default();
    let mut chains: Vec<String> =
        (0..num_inputs).map(|i| format!("[{i}:a]{normalize}asetpts=PTS-STARTPTS[a{i}]")).collect();
    let mut labels: Vec<String> = (0..num_inputs).map(|i| format!("[a{i}]")).collect();

    if let (true, Some(AudioFormat { rate, layout }), true) = (gap_ms > 0, audio_format, num_inputs > 1) {
        let mut interleaved = Vec::with_capacity(num_inputs * 2);
        for (i, label) in labels.iter().take(num_inputs - 1).enumerate() {
            chains.push(format!(
                "anullsrc=r={rate}:cl={layout},atrim=duration={},asetpts=PTS-STARTPTS[s{i}]",
                gap_seconds(gap_ms)
            ));
            interleaved.push(label.clone());
            interleaved.push(format!("[s{i}]"));
        }
        interleaved.push(labels[num_inputs - 1].clone());
        labels = interleaved;
    }

    format!("{};{}concat=n={}:v=0:a=1[out]", chains.join(";"), labels.concat(), labels.len())
}

/// Decode `clips` in order and encode them as one MP3 file at `output`.
///
/// `output` is written by ffmpeg directly; the caller passes a temporary path
/// and renames it into place.
pub(crate) async fn combine(ffmpeg: &Path, clips: &[PathBuf], output: &Path, gap_ms: u32) -> Result<(), String> {
    let first = clips.first().ok_or("there are no clips to combine")?;
    let audio_format = match find_ffprobe(ffmpeg) {
        Some(ffprobe) => probe_audio_format(&ffprobe, first).await?,
        None => None,
    };
    if audio_format.is_none() {
        tracing::warn!(
            "could not detect the audio format of {} with ffprobe; concatenating without normalization or gaps",
            first.display()
        );
    }

    let mut args: Vec<String> = vec!["-hide_banner".into(), "-nostdin".into(), "-y".into()];
    for clip in clips {
        args.extend(input_args(clip)?);
    }
    let filter = build_concat_filter(clips.len(), audio_format.as_ref(), gap_ms);
    let target = output.to_string_lossy();
    args.extend(
        [
            "-filter_complex",
            filter.as_str(),
            "-map",
            "[out]",
            "-map_metadata",
            "-1",
            "-c:a",
            "libmp3lame",
            "-b:a",
            "128k",
            "-nostats",
            "-f",
            "mp3",
            target.as_ref(),
        ]
        .map(String::from),
    );

    let result = run(ffmpeg, "ffmpeg", &args).await?;
    if !result.success {
        let lines: Vec<&str> = result.stderr.trim().lines().collect();
        let tail = lines[lines.len().saturating_sub(20)..].join("\n");
        let code = result.code.map_or_else(|| "a signal".to_string(), |code| format!("code {code}"));
        return Err(format!(
            "ffmpeg exited with {code} while combining {} clips:\n{tail}",
            clips.len()
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn audio_format_is_read_from_ffprobe_json() {
        let mp3 = r#"{"programs": [], "streams": [{"sample_rate": "24000", "channels": 1, "channel_layout": "mono"}]}"#;
        assert_eq!(AudioFormat::from_ffprobe_json(mp3), AudioFormat::new(24_000, "mono"));
        assert_eq!(AudioFormat::new(24_000, "mono").unwrap().layout, "mono");

        let surround = r#"{"streams": [{"sample_rate": "48000", "channels": 6, "channel_layout": "5.1(side)"}]}"#;
        assert_eq!(AudioFormat::from_ffprobe_json(surround), AudioFormat::new(48_000, "5.1"));
        assert_eq!(AudioFormat::from_ffprobe_json(surround).unwrap().layout, "5.1");

        // WAV files often state no layout: one or two channels are named.
        let wav = r#"{"streams": [{"sample_rate": "16000", "channels": 2}]}"#;
        assert_eq!(AudioFormat::from_ffprobe_json(wav), AudioFormat::new(16_000, "stereo"));
        let many = r#"{"streams": [{"sample_rate": "16000", "channels": 3}]}"#;
        assert_eq!(AudioFormat::from_ffprobe_json(many), None);

        assert_eq!(AudioFormat::from_ffprobe_json(r#"{"streams": []}"#), None);
        assert_eq!(AudioFormat::from_ffprobe_json("0000.mp3: Invalid data found when processing input"), None);
    }

    /// Issue #31: whatever the source, a value that could change the filter
    /// graph is refused before it is written into one.
    #[test]
    fn clips_are_opened_as_local_files_with_a_forced_demuxer() {
        assert_eq!(
            input_args(Path::new("/data/clips/0000.mp3")).unwrap(),
            ["-protocol_whitelist", "file", "-f", "mp3", "-i", "file:/data/clips/0000.mp3"]
        );
        assert_eq!(demuxer_for(Path::new("a/0001.WAV")), Ok("wav"));
        assert_eq!(demuxer_for(Path::new("0002.m4a")), Ok("mov"));
        assert!(demuxer_for(Path::new("0003.m3u8")).is_err());
        assert!(demuxer_for(Path::new("0004")).is_err());
    }

    #[test]
    fn filter_without_a_known_format_only_resets_timestamps() {
        assert_eq!(
            build_concat_filter(2, None, 400),
            "[0:a]asetpts=PTS-STARTPTS[a0];[1:a]asetpts=PTS-STARTPTS[a1];[a0][a1]concat=n=2:v=0:a=1[out]"
        );
    }

    #[test]
    fn filter_normalises_and_inserts_gaps_between_inputs() {
        let format = AudioFormat::new(24_000, "mono").unwrap();
        let normalize = "aresample=24000,aformat=sample_rates=24000:channel_layouts=mono,";
        assert_eq!(
            build_concat_filter(3, Some(&format), 400),
            format!(
                "[0:a]{normalize}asetpts=PTS-STARTPTS[a0];\
                 [1:a]{normalize}asetpts=PTS-STARTPTS[a1];\
                 [2:a]{normalize}asetpts=PTS-STARTPTS[a2];\
                 anullsrc=r=24000:cl=mono,atrim=duration=0.4,asetpts=PTS-STARTPTS[s0];\
                 anullsrc=r=24000:cl=mono,atrim=duration=0.4,asetpts=PTS-STARTPTS[s1];\
                 [a0][s0][a1][s1][a2]concat=n=5:v=0:a=1[out]"
            )
        );
    }

    #[test]
    fn filter_has_no_gap_for_one_input_or_a_zero_gap() {
        let format = AudioFormat::new(44_100, "stereo").unwrap();
        let normalize = "aresample=44100,aformat=sample_rates=44100:channel_layouts=stereo,";
        assert_eq!(
            build_concat_filter(1, Some(&format), 400),
            format!("[0:a]{normalize}asetpts=PTS-STARTPTS[a0];[a0]concat=n=1:v=0:a=1[out]")
        );
        assert!(!build_concat_filter(2, Some(&format), 0).contains("anullsrc"));
    }

    #[test]
    fn gap_is_written_like_a_python_float() {
        assert_eq!(gap_seconds(400), "0.4");
        assert_eq!(gap_seconds(1000), "1.0");
        assert_eq!(gap_seconds(1250), "1.25");
    }

    #[cfg(unix)]
    #[test]
    fn executables_are_found_on_a_search_path_only() {
        use std::os::unix::fs::PermissionsExt;

        let dir = tempfile::tempdir().unwrap();
        let tool = dir.path().join("fake-tool");
        std::fs::write(&tool, "#!/bin/sh\n").unwrap();
        std::fs::set_permissions(&tool, std::fs::Permissions::from_mode(0o755)).unwrap();
        let data = dir.path().join("not-a-tool");
        std::fs::write(&data, "data").unwrap();
        std::fs::set_permissions(&data, std::fs::Permissions::from_mode(0o644)).unwrap();

        let search = std::env::join_paths([Path::new("/nonexistent-dir"), dir.path()]).unwrap();
        assert_eq!(find_in("fake-tool", &search), Some(tool));
        assert_eq!(find_in("not-a-tool", &search), None);
        assert_eq!(find_in("missing", &search), None);
    }
}
