//! Generation progress as a 0-100 value.
//!
//! Stages: outline done -> 5, transcript segments 5..35, audio clips 35..95.
//! 100 is never produced here: the caller sets it after the final save, as it
//! sets 1 when it creates the episode record.
//!
//! The original parsed podcast-creator's log lines because that library had no
//! progress callback. This crate reports [`ProgressEvent`]s directly; the log
//! line parser is kept ([`ProgressTracker::feed`]) so both spellings of a
//! milestone give the same number.

use std::sync::OnceLock;

use regex::Regex;

pub const OUTLINE_DONE: u8 = 5;
pub const TRANSCRIPT_END: u8 = 35;
pub const AUDIO_END: u8 = 95;

/// Which part of the job a progress update belongs to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProgressPhase {
    Outline,
    Transcript,
    Audio,
}

/// A milestone of the generation job.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProgressEvent {
    /// The outline was generated.
    OutlineGenerated { segments: usize },
    /// The transcript call for segment `index` (1-based) of `total` is starting.
    TranscriptSegmentStarted { index: usize, total: usize },
    /// The transcript is complete and `clips` audio clips are about to be made.
    AudioStarted { clips: usize },
    /// One more audio clip was written.
    ClipGenerated,
}

impl ProgressEvent {
    pub fn phase(&self) -> ProgressPhase {
        match self {
            ProgressEvent::OutlineGenerated { .. } => ProgressPhase::Outline,
            ProgressEvent::TranscriptSegmentStarted { .. } => ProgressPhase::Transcript,
            ProgressEvent::AudioStarted { .. } | ProgressEvent::ClipGenerated => ProgressPhase::Audio,
        }
    }
}

/// What [`crate::create_podcast`] hands to its progress callback: only sent
/// when `percent` has grown.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ProgressUpdate {
    /// 5 to 95.
    pub percent: u8,
    pub phase: ProgressPhase,
    pub event: ProgressEvent,
}

/// Stateful, I/O-free progress calculator.
#[derive(Debug, Clone, Default)]
pub struct ProgressTracker {
    percent: u8,
    total_clips: Option<usize>,
    clips_done: usize,
}

impl ProgressTracker {
    pub fn new() -> Self {
        Self::default()
    }

    /// Start from a value already stored on the episode.
    pub fn starting_at(percent: u8) -> Self {
        Self { percent, ..Self::default() }
    }

    pub fn percent(&self) -> u8 {
        self.percent
    }

    /// Process one milestone. Returns the new percent if it grew (progress
    /// never goes backwards), otherwise `None`.
    pub fn apply(&mut self, event: ProgressEvent) -> Option<u8> {
        let value = self.compute(event)?;
        let value = value.clamp(0.0, f64::from(AUDIO_END)) as u8;
        if value <= self.percent {
            return None;
        }
        self.percent = value;
        Some(value)
    }

    /// Process one podcast-creator style log message (see the module docs).
    pub fn feed(&mut self, message: &str) -> Option<u8> {
        self.apply(parse_message(message)?)
    }

    fn compute(&mut self, event: ProgressEvent) -> Option<f64> {
        match event {
            ProgressEvent::OutlineGenerated { .. } => Some(f64::from(OUTLINE_DONE)),
            ProgressEvent::TranscriptSegmentStarted { index, total } => {
                if total == 0 {
                    return None;
                }
                let span = f64::from(TRANSCRIPT_END - OUTLINE_DONE);
                Some(f64::from(OUTLINE_DONE) + span * (index as f64 - 1.0) / total as f64)
            }
            ProgressEvent::AudioStarted { clips } => {
                self.total_clips = Some(clips);
                self.clips_done = 0;
                Some(f64::from(TRANSCRIPT_END))
            }
            ProgressEvent::ClipGenerated => {
                self.clips_done += 1;
                let total = self.total_clips.filter(|&total| total > 0)?;
                let span = f64::from(AUDIO_END - TRANSCRIPT_END);
                Some(f64::from(TRANSCRIPT_END) + span * self.clips_done as f64 / total as f64)
            }
        }
    }
}

fn parse_message(message: &str) -> Option<ProgressEvent> {
    struct Patterns {
        outline: Regex,
        transcript: Regex,
        routing: Regex,
    }
    static PATTERNS: OnceLock<Patterns> = OnceLock::new();
    let patterns = PATTERNS.get_or_init(|| Patterns {
        outline: Regex::new(r"Generated outline with (\d+) segments").expect("valid regex"),
        transcript: Regex::new(r"Generating transcript for segment (\d+)/(\d+)").expect("valid regex"),
        routing: Regex::new(r"Routing (\d+) dialogue segments").expect("valid regex"),
    });

    if let Some(found) = patterns.outline.captures(message) {
        return Some(ProgressEvent::OutlineGenerated { segments: found[1].parse().ok()? });
    }
    if let Some(found) = patterns.transcript.captures(message) {
        return Some(ProgressEvent::TranscriptSegmentStarted {
            index: found[1].parse().ok()?,
            total: found[2].parse().ok()?,
        });
    }
    if let Some(found) = patterns.routing.captures(message) {
        return Some(ProgressEvent::AudioStarted { clips: found[1].parse().ok()? });
    }
    if message.contains("Generated audio clip") {
        return Some(ProgressEvent::ClipGenerated);
    }
    None
}

/// Progress to show for an episode: completed jobs are 100, queued or new ones
/// 0 until something was stored.
pub fn effective_progress(job_status: Option<&str>, progress: Option<i64>) -> Option<i64> {
    match job_status {
        Some("completed") => Some(100),
        Some("queued") | Some("new") if progress.is_none() => Some(0),
        _ => progress,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn outline_sets_five() {
        let mut t = ProgressTracker::new();
        assert_eq!(t.feed("Generated outline with 4 segments"), Some(5));
    }

    #[test]
    fn transcript_segments_interpolate() {
        let mut t = ProgressTracker::new();
        t.feed("Generated outline with 3 segments");
        assert_eq!(t.feed("Generating transcript for segment 1/3: Intro"), None);
        assert_eq!(t.feed("Generating transcript for segment 2/3: Mid"), Some(15));
        assert_eq!(t.feed("Generating transcript for segment 3/3: End"), Some(25));
    }

    #[test]
    fn routing_sets_thirty_five() {
        let mut t = ProgressTracker::new();
        assert_eq!(t.feed("Routing 10 dialogue segments for sequential batch"), Some(35));
    }

    #[test]
    fn audio_clips_scale_and_cap() {
        let mut t = ProgressTracker::new();
        t.feed("Routing 4 dialogue segments for sequential batch processing");
        let seen: Vec<Option<u8>> =
            (0..4).map(|_| t.feed("Generated audio clip: /tmp/x.mp3")).collect();
        assert_eq!(seen, [Some(50), Some(65), Some(80), Some(95)]);
        assert_eq!(t.feed("Generated audio clip: /tmp/y.mp3"), None);
    }

    #[test]
    fn clip_before_routing_ignored() {
        let mut t = ProgressTracker::new();
        assert_eq!(t.feed("Generated audio clip: a.mp3"), None);
    }

    #[test]
    fn only_reports_changes_and_never_goes_back() {
        let mut t = ProgressTracker::new();
        assert_eq!(t.feed("Generated outline with 2 segments"), Some(5));
        assert_eq!(t.feed("Generated outline with 2 segments"), None);
        t.feed("Routing 2 dialogue segments");
        assert_eq!(t.feed("Generating transcript for segment 1/2: x"), None);
        assert_eq!(t.percent(), 35);
    }

    #[test]
    fn unrelated_message() {
        assert_eq!(ProgressTracker::new().feed("hello"), None);
    }

    #[test]
    fn events_give_the_same_numbers_as_messages() {
        let mut t = ProgressTracker::new();
        assert_eq!(t.apply(ProgressEvent::OutlineGenerated { segments: 3 }), Some(5));
        assert_eq!(t.apply(ProgressEvent::TranscriptSegmentStarted { index: 1, total: 3 }), None);
        assert_eq!(t.apply(ProgressEvent::TranscriptSegmentStarted { index: 3, total: 3 }), Some(25));
        assert_eq!(t.apply(ProgressEvent::TranscriptSegmentStarted { index: 1, total: 0 }), None);
        assert_eq!(t.apply(ProgressEvent::AudioStarted { clips: 3 }), Some(35));
        assert_eq!(t.apply(ProgressEvent::ClipGenerated), Some(55));
        assert_eq!(t.apply(ProgressEvent::ClipGenerated), Some(75));
        assert_eq!(t.apply(ProgressEvent::ClipGenerated), Some(95));
    }

    #[test]
    fn a_stored_value_is_never_undercut() {
        let mut t = ProgressTracker::starting_at(40);
        assert_eq!(t.apply(ProgressEvent::OutlineGenerated { segments: 3 }), None);
        assert_eq!(t.apply(ProgressEvent::AudioStarted { clips: 2 }), None);
        assert_eq!(t.apply(ProgressEvent::ClipGenerated), Some(65));
    }

    #[test]
    fn phases_follow_the_events() {
        assert_eq!(ProgressEvent::OutlineGenerated { segments: 1 }.phase(), ProgressPhase::Outline);
        assert_eq!(
            ProgressEvent::TranscriptSegmentStarted { index: 1, total: 1 }.phase(),
            ProgressPhase::Transcript
        );
        assert_eq!(ProgressEvent::ClipGenerated.phase(), ProgressPhase::Audio);
    }

    #[test]
    fn completed_reports_100_and_queued_reports_0() {
        assert_eq!(effective_progress(Some("completed"), Some(70)), Some(100));
        assert_eq!(effective_progress(Some("queued"), None), Some(0));
        assert_eq!(effective_progress(Some("new"), None), Some(0));
        assert_eq!(effective_progress(Some("queued"), Some(3)), Some(3));
        assert_eq!(effective_progress(Some("running"), Some(40)), Some(40));
        assert_eq!(effective_progress(Some("running"), None), None);
        assert_eq!(effective_progress(None, Some(55)), Some(55));
    }
}
