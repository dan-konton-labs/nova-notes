//! Publishing episodes into the Podcast Feed: the rules, without the database.
//!
//! [`publish_episode`] and [`unpublish_episode`] change the three publication
//! fields of a [`PodcastEpisode`] in memory. Loading the episode, looking up
//! its job status and saving it afterwards are the caller's part. A refused
//! publish changes nothing.

use std::path::{Path, PathBuf};

use chrono::{DateTime, Utc};
use serde_json::Value;

use crate::audio;
use crate::audio_paths::resolve_contained_audio_path;
use crate::error::{PodcastError, Result};
use crate::models::PodcastEpisode;

/// Formats Apple Podcasts plays, mapped to the MIME type used for the enclosure.
pub const PUBLISHABLE_AUDIO_TYPES: [(&str, &str); 2] = [(".mp3", "audio/mpeg"), (".m4a", "audio/x-m4a")];

/// Default description length (outline excerpt), in characters.
pub const SUMMARY_EXCERPT_CHARS: usize = 300;
/// Longest description an edited summary may have, in characters.
pub const SUMMARY_MAX_CHARS: usize = 4000;

/// The enclosure MIME type for a file suffix (`.mp3`, `.m4a`), if publishable.
pub fn publishable_audio_type(suffix: &str) -> Option<&'static str> {
    PUBLISHABLE_AUDIO_TYPES.iter().find(|(known, _)| *known == suffix).map(|(_, mime)| *mime)
}

/// A path's extension with its dot, as written (`.MP3`); empty when it has none.
fn suffix(path: &Path) -> String {
    path.extension().map(|ext| format!(".{}", ext.to_string_lossy())).unwrap_or_default()
}

/// A path's extension with its dot, lower-cased (`.mp3`); empty when it has none.
pub fn audio_suffix(path: &Path) -> String {
    suffix(path).to_lowercase()
}

/// Plain-text episode description built from the outline's segments.
///
/// Cut at a word boundary with an ellipsis; falls back to `fallback` (the
/// episode name) when there is no outline, as for read-script episodes.
pub fn default_feed_summary(outline: Option<&Value>, fallback: &str) -> String {
    let segments = outline.and_then(|outline| outline.get("segments")).and_then(Value::as_array);
    let parts: Vec<String> = segments
        .into_iter()
        .flatten()
        .map(|segment| match segment.get("description") {
            Some(Value::String(text)) => text.split_whitespace().collect::<Vec<_>>().join(" "),
            Some(Value::Null) | None => String::new(),
            Some(other) => other.to_string(),
        })
        .filter(|part| !part.is_empty())
        .collect();
    let text = if parts.is_empty() { fallback.to_string() } else { parts.join(" ") };

    let chars: Vec<char> = text.chars().collect();
    if chars.len() <= SUMMARY_EXCERPT_CHARS {
        return text;
    }
    let mut cut: String = chars[..SUMMARY_EXCERPT_CHARS - 1].iter().collect();
    if !chars[SUMMARY_EXCERPT_CHARS - 1].is_whitespace() {
        // The limit falls inside a word: drop the partial word.
        if let Some(space) = cut.rfind(' ') {
            cut.truncate(space);
        }
    }
    format!("{}…", cut.trim_end())
}

/// Round to the nearest integer, ties to even, like Python's `round`.
fn round_half_even(value: f64) -> u64 {
    let floor = value.floor();
    let rounded = match value - floor {
        diff if diff > 0.5 => floor + 1.0,
        diff if diff < 0.5 => floor,
        _ if floor % 2.0 == 0.0 => floor,
        _ => floor + 1.0,
    };
    rounded.max(0.0) as u64
}

/// Audio length in whole seconds, as stored on a Published Episode.
pub fn audio_duration_seconds(path: &Path) -> Result<u64> {
    audio::duration_seconds(path).map(round_half_even).map_err(|error| {
        tracing::debug!("{error}");
        PodcastError::InvalidInput(
            "Could not read the audio file; it may be corrupt or not real mp3/m4a audio.".into(),
        )
    })
}

/// Return the episode's audio path, or a specific [`PodcastError::InvalidInput`].
///
/// `job_status` is the status of the episode's background job and is only
/// looked at when the episode has one (`command` is set); an imported episode
/// with no job counts as completed.
pub fn require_publishable(
    episode: &PodcastEpisode,
    job_status: Option<&str>,
    podcasts_folder: &Path,
) -> Result<PathBuf> {
    let refuse = |message: String| Err(PodcastError::InvalidInput(message));
    if episode.command.as_deref().is_some_and(|command| !command.is_empty()) && job_status != Some("completed") {
        return refuse(format!(
            "Episode is not completed (job status: {}); only completed episodes can be published.",
            job_status.unwrap_or("unknown")
        ));
    }
    let Some(audio_file) = episode.audio_file.as_deref().filter(|value| !value.is_empty()) else {
        return refuse("Episode has no audio file.".into());
    };
    let Some(audio_path) = resolve_contained_audio_path(podcasts_folder, Some(audio_file)) else {
        return refuse("Episode audio file path is invalid (outside the podcasts folder).".into());
    };
    if !audio_path.is_file() {
        return refuse("Audio file not found on disk.".into());
    }
    if publishable_audio_type(&audio_suffix(&audio_path)).is_none() {
        return refuse(format!(
            "Unsupported audio format '{}': only .mp3 and .m4a episodes can be published.",
            suffix(&audio_path)
        ));
    }
    Ok(audio_path)
}

/// Put a completed episode in the Podcast Feed. The caller saves it.
///
/// Stores the audio length and the summary: the edited text when one is given
/// (trimmed, up to [`SUMMARY_MAX_CHARS`]), else, on a first publish, an
/// excerpt of the outline. A republish keeps `published_at`, so the feed's
/// `pubDate` does not move, and without a summary keeps the existing notes.
pub fn publish_episode(
    episode: &mut PodcastEpisode,
    summary: Option<&str>,
    job_status: Option<&str>,
    podcasts_folder: &Path,
    now: DateTime<Utc>,
) -> Result<()> {
    let audio_path = require_publishable(episode, job_status, podcasts_folder)?;
    let edited = summary.unwrap_or_default().trim();
    let edited_len = edited.chars().count();
    if edited_len > SUMMARY_MAX_CHARS {
        return Err(PodcastError::InvalidInput(format!(
            "Summary is too long ({edited_len} characters; the limit is {SUMMARY_MAX_CHARS})."
        )));
    }
    let duration = audio_duration_seconds(&audio_path)?;

    episode.duration_seconds = Some(duration);
    if !edited.is_empty() {
        episode.feed_summary = Some(edited.to_string());
    } else if episode.published_at.is_none() {
        episode.feed_summary = Some(default_feed_summary(episode.outline.as_ref(), &episode.name));
    }
    if episode.published_at.is_none() {
        episode.published_at = Some(now);
    }
    Ok(())
}

/// Take an episode out of the Podcast Feed. Returns `false`, having changed
/// nothing, when it was not in it; otherwise the caller saves the episode.
pub fn unpublish_episode(episode: &mut PodcastEpisode) -> bool {
    if episode.published_at.is_none() {
        return false;
    }
    episode.published_at = None;
    episode.duration_seconds = None;
    episode.feed_summary = None;
    true
}

#[cfg(test)]
mod tests {
    use chrono::TimeZone;
    use serde_json::json;

    use super::*;

    const DEFAULT_SUMMARY: &str = "Why offline first matters. How sync conflicts resolve.";

    /// 128 kbit/s, 44.1 kHz silent frame; 1000 of them are 26.12 s.
    fn mp3_fixture() -> Vec<u8> {
        let mut frame = vec![0xFF, 0xFB, 0x90, 0x00];
        frame.resize(417, 0);
        frame.repeat(1000)
    }

    fn outline() -> Value {
        json!({"segments": [
            {"name": "Intro", "description": "Why   offline\nfirst matters.", "size": "short"},
            {"name": "Deep dive", "description": "How sync conflicts resolve.", "size": "long"},
        ]})
    }

    fn episode(overrides: Value) -> PodcastEpisode {
        let mut base = json!({
            "id": "episode:ep1",
            "name": "Offline first",
            "episode_profile": {"name": "default"},
            "speaker_profile": {"name": "default"},
            "briefing": "INTERNAL generation prompt, never shown in the feed",
            "content": "source text",
            "audio_file": "episodes/ep1/audio/ep1.mp3",
            "outline": outline(),
            "command": null,
        });
        for (key, value) in overrides.as_object().unwrap() {
            base[key] = value.clone();
        }
        serde_json::from_value(base).unwrap()
    }

    fn write_audio(root: &Path, rel: &str, data: &[u8]) {
        let path = root.join(rel);
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(path, data).unwrap();
    }

    fn root_with_audio() -> tempfile::TempDir {
        let root = tempfile::tempdir().unwrap();
        write_audio(root.path(), "episodes/ep1/audio/ep1.mp3", &mp3_fixture());
        root
    }

    fn now() -> DateTime<Utc> {
        Utc.with_ymd_and_hms(2026, 11, 12, 10, 0, 0).unwrap()
    }

    fn first() -> DateTime<Utc> {
        Utc.with_ymd_and_hms(2026, 11, 1, 9, 30, 0).unwrap()
    }

    #[test]
    fn publish_stores_publication_state() {
        let root = root_with_audio();
        let mut ep = episode(json!({}));
        publish_episode(&mut ep, None, None, root.path(), now()).unwrap();
        assert_eq!(ep.duration_seconds, Some(26));
        assert_eq!(ep.feed_summary.as_deref(), Some(DEFAULT_SUMMARY));
        assert_eq!(ep.published_at, Some(now()));
    }

    #[test]
    fn every_refusal_has_its_own_message_and_changes_nothing() {
        let root = root_with_audio();
        write_audio(root.path(), "episodes/ep1/audio/ep1.wav", b"RIFF....WAVE");
        write_audio(root.path(), "episodes/ep1/audio/broken.mp3", b"definitely not audio");
        write_audio(root.path(), "episodes/ep1/audio/LOUD.OGG", b"OggS");

        let cases = [
            (episode(json!({"command": "command:abc"})), "not completed"),
            (episode(json!({"audio_file": null})), "no audio file"),
            (episode(json!({"audio_file": "../../etc/passwd"})), "invalid"),
            (episode(json!({"audio_file": "episodes/ep1/audio/gone.mp3"})), "not found on disk"),
            (episode(json!({"audio_file": "episodes/ep1/audio/ep1.wav"})), ".mp3"),
            (episode(json!({"audio_file": "episodes/ep1/audio/broken.mp3"})), "could not read"),
        ];
        let mut messages = Vec::new();
        for (ep, keyword) in cases {
            let mut attempt = ep.clone();
            let error = publish_episode(&mut attempt, None, Some("running"), root.path(), now()).unwrap_err();
            assert!(matches!(error, PodcastError::InvalidInput(_)), "{error:?}");
            let message = error.to_string();
            assert!(message.to_lowercase().contains(keyword), "{keyword}: {message}");
            assert_eq!(attempt, ep, "a refused publish changes nothing");
            messages.push(message);
        }
        messages.sort();
        messages.dedup();
        assert_eq!(messages.len(), 6);

        let mut ogg = episode(json!({"audio_file": "episodes/ep1/audio/LOUD.OGG"}));
        assert_eq!(
            publish_episode(&mut ogg, None, None, root.path(), now()).unwrap_err().to_string(),
            "Unsupported audio format '.OGG': only .mp3 and .m4a episodes can be published."
        );
    }

    #[test]
    fn job_status_is_named_in_the_refusal() {
        let root = root_with_audio();
        let ep = episode(json!({"command": "command:abc"}));
        assert_eq!(
            require_publishable(&ep, Some("running"), root.path()).unwrap_err().to_string(),
            "Episode is not completed (job status: running); only completed episodes can be published."
        );
        assert!(require_publishable(&ep, None, root.path()).unwrap_err().to_string().contains("unknown"));
    }

    #[test]
    fn an_episode_whose_job_completed_can_be_published() {
        let root = root_with_audio();
        let mut ep = episode(json!({"command": "command:abc"}));
        publish_episode(&mut ep, None, Some("completed"), root.path(), now()).unwrap();
        assert!(ep.is_published());
    }

    #[test]
    fn m4a_is_accepted() {
        let root = tempfile::tempdir().unwrap();
        // ftyp + mdat + moov/mvhd stating 3 s.
        let mut mvhd = vec![0u8; 12];
        mvhd.extend(1000u32.to_be_bytes());
        mvhd.extend(3000u32.to_be_bytes());
        mvhd.extend([0u8; 80]);
        let boxed = |kind: &[u8; 4], payload: &[u8]| {
            let mut out = ((payload.len() + 8) as u32).to_be_bytes().to_vec();
            out.extend(kind);
            out.extend(payload);
            out
        };
        let mut file = boxed(b"ftyp", b"M4A \0\0\0\0");
        file.extend(boxed(b"mdat", &[0u8; 64]));
        file.extend(boxed(b"moov", &boxed(b"mvhd", &mvhd)));
        write_audio(root.path(), "episodes/ep1/audio/ep1.m4a", &file);

        let mut ep = episode(json!({"audio_file": "episodes/ep1/audio/ep1.m4a"}));
        publish_episode(&mut ep, None, None, root.path(), now()).unwrap();
        assert_eq!(ep.duration_seconds, Some(3));
    }

    #[test]
    fn edited_summary_is_stored_trimmed() {
        let root = root_with_audio();
        let mut ep = episode(json!({}));
        publish_episode(&mut ep, Some("  My own show notes \n"), None, root.path(), now()).unwrap();
        assert_eq!(ep.feed_summary.as_deref(), Some("My own show notes"));
    }

    #[test]
    fn blank_summary_falls_back_to_the_outline_excerpt() {
        let root = root_with_audio();
        let mut ep = episode(json!({}));
        publish_episode(&mut ep, Some("   "), None, root.path(), now()).unwrap();
        assert_eq!(ep.feed_summary.as_deref(), Some(DEFAULT_SUMMARY));
    }

    #[test]
    fn over_long_edited_summary_is_refused() {
        let root = root_with_audio();
        let mut ep = episode(json!({}));
        let error = publish_episode(&mut ep, Some(&"x".repeat(4001)), None, root.path(), now()).unwrap_err();
        assert_eq!(error.to_string(), "Summary is too long (4001 characters; the limit is 4000).");
        assert!(!ep.is_published());
        assert_eq!(ep.duration_seconds, None);

        publish_episode(&mut ep, Some(&"é".repeat(4000)), None, root.path(), now()).unwrap();
    }

    #[test]
    fn long_outline_excerpt_is_cut_at_a_word_boundary() {
        let outline = json!({"segments": [{"name": "A", "description": "word ".repeat(200)}]});
        let summary = default_feed_summary(Some(&outline), "Name");
        assert!(summary.chars().count() <= 300);
        assert!(summary.ends_with("word…"));
        let body = summary.trim_end_matches('…');
        assert!(body.split_whitespace().all(|word| word == "word"));
    }

    #[test]
    fn excerpt_ending_exactly_at_a_word_keeps_that_word() {
        // 299 characters followed by a space: nothing is a partial word.
        let description = format!("{} tail", "a".repeat(299));
        let outline = json!({"segments": [{"description": description}]});
        assert_eq!(default_feed_summary(Some(&outline), ""), format!("{}…", "a".repeat(299)));

        // One unbroken 400-character word is cut at the limit.
        let outline = json!({"segments": [{"description": "b".repeat(400)}]});
        assert_eq!(default_feed_summary(Some(&outline), ""), format!("{}…", "b".repeat(299)));

        // Exactly at the limit: unchanged.
        let outline = json!({"segments": [{"description": "c".repeat(300)}]});
        assert_eq!(default_feed_summary(Some(&outline), ""), "c".repeat(300));
    }

    #[test]
    fn episode_without_an_outline_uses_its_name() {
        assert_eq!(default_feed_summary(None, "Offline first"), "Offline first");
        assert_eq!(default_feed_summary(Some(&json!({})), "Offline first"), "Offline first");
        assert_eq!(default_feed_summary(Some(&json!({"segments": []})), "Offline first"), "Offline first");
        assert_eq!(
            default_feed_summary(Some(&json!({"segments": [{"name": "x"}, {"description": "  "}]})), "N"),
            "N"
        );

        let root = root_with_audio();
        let mut ep = episode(json!({"outline": null}));
        publish_episode(&mut ep, None, None, root.path(), now()).unwrap();
        assert_eq!(ep.feed_summary.as_deref(), Some("Offline first"));
    }

    #[test]
    fn republishing_keeps_the_original_publish_time_and_notes() {
        let root = root_with_audio();
        let mut ep = episode(json!({
            "published_at": "2026-11-01T09:30:00Z", "duration_seconds": 20, "feed_summary": "Old notes"
        }));
        publish_episode(&mut ep, None, None, root.path(), now()).unwrap();
        assert_eq!(ep.published_at, Some(first()));
        assert_eq!(ep.feed_summary.as_deref(), Some("Old notes"));
        assert_eq!(ep.duration_seconds, Some(26));
    }

    #[test]
    fn republishing_with_a_summary_edits_only_the_summary() {
        let root = root_with_audio();
        let mut ep = episode(json!({
            "published_at": "2026-11-01T09:30:00Z", "duration_seconds": 26, "feed_summary": "Old notes"
        }));
        publish_episode(&mut ep, Some("New notes"), None, root.path(), now()).unwrap();
        assert_eq!(ep.published_at, Some(first()));
        assert_eq!(ep.feed_summary.as_deref(), Some("New notes"));
        assert_eq!(ep.duration_seconds, Some(26));
    }

    #[test]
    fn unpublish_clears_the_publication_state() {
        let mut ep = episode(json!({
            "published_at": "2026-11-01T09:30:00Z", "duration_seconds": 26, "feed_summary": "Notes"
        }));
        assert!(unpublish_episode(&mut ep));
        assert_eq!((ep.published_at, ep.duration_seconds, ep.feed_summary.clone()), (None, None, None));
        // Present and null when saved: the fields are written, not skipped.
        let saved = serde_json::to_value(&ep).unwrap();
        for field in ["published_at", "duration_seconds", "feed_summary"] {
            assert_eq!(saved.get(field), Some(&Value::Null));
        }
    }

    #[test]
    fn unpublishing_an_unpublished_episode_is_a_no_op() {
        let mut ep = episode(json!({}));
        let before = ep.clone();
        assert!(!unpublish_episode(&mut ep));
        assert_eq!(ep, before);
    }

    #[test]
    fn republishing_after_an_unpublish_gets_a_new_publish_time() {
        let root = root_with_audio();
        let mut ep = episode(json!({"published_at": "2026-11-01T09:30:00Z", "feed_summary": "Notes"}));
        unpublish_episode(&mut ep);
        publish_episode(&mut ep, None, None, root.path(), now()).unwrap();
        assert!(ep.published_at.unwrap() > first());
        assert_eq!(ep.feed_summary.as_deref(), Some(DEFAULT_SUMMARY));
    }

    #[test]
    fn durations_round_half_to_even_like_python() {
        assert_eq!(round_half_even(26.0625), 26);
        assert_eq!(round_half_even(26.5), 26);
        assert_eq!(round_half_even(27.5), 28);
        assert_eq!(round_half_even(0.4), 0);
        assert_eq!(round_half_even(2.51), 3);
    }

    #[test]
    fn suffixes_map_to_enclosure_types() {
        assert_eq!(publishable_audio_type(".mp3"), Some("audio/mpeg"));
        assert_eq!(publishable_audio_type(".m4a"), Some("audio/x-m4a"));
        assert_eq!(publishable_audio_type(".wav"), None);
        assert_eq!(audio_suffix(Path::new("a/b/EP.M4A")), ".m4a");
        assert_eq!(audio_suffix(Path::new("a/b/noext")), "");
    }
}
