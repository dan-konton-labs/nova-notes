//! The whole job, `create_podcast`, driven by mock models into a temp dir.

mod common;

use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use common::*;
use nn_llm::AiError;
use nn_podcast::audio::{find_executable, synthesize_clips};
use nn_podcast::models::EpisodeProfile;
use nn_podcast::{
    build_episode_output_dir, create_podcast, duration_seconds, resolve_contained_audio_path, to_relative_audio_path,
    AudioOptions, CombineMethod, Dialogue, FfmpegMode, GenerationInput, GenerationOptions, PodcastError,
    PodcastOutput, PodcastRequest, PodcastSource, ProgressPhase, ProgressUpdate, RetryConfig, TtsModels,
};
use serde_json::json;

fn audio_options(ffmpeg: FfmpegMode) -> AudioOptions {
    AudioOptions {
        batch_size: 2,
        batch_pause: Duration::ZERO,
        gap_ms: 0,
        retry: RetryConfig::immediate(3),
        ffmpeg,
    }
}

fn no_ffmpeg() -> AudioOptions {
    audio_options(FfmpegMode::Disabled)
}

fn generation(outline: Arc<ScriptedModel>, writer: Arc<ScriptedModel>) -> PodcastSource {
    PodcastSource::Generate(Box::new(GenerationInput {
        content: "Offline-first apps keep working without a network.".into(),
        briefing: "Explain offline-first design.".into(),
        num_segments: 2,
        language: None,
        outline_model: outline,
        transcript_model: writer,
        options: GenerationOptions { retry: RetryConfig::immediate(3), ..GenerationOptions::default() },
    }))
}

fn request(root: &Path, tts: TtsModels, source: PodcastSource, audio: AudioOptions) -> PodcastRequest {
    let (episode_dir_name, output_dir) = build_episode_output_dir(root);
    PodcastRequest { output_dir, episode_dir_name, speaker_profile: speaker_profile(), tts, source, audio }
}

fn writer_replies() -> Vec<nn_llm::AiResult<nn_llm::ChatResponse>> {
    vec![
        reply(&transcript_json(&[("Velvet Noir", "Good evening."), ("Max", "Hello there.")])),
        reply(&transcript_json(&[("Max", "That is all."), ("Velvet Noir", "Good night.")])),
    ]
}

async fn run(request: PodcastRequest) -> (Result<PodcastOutput, nn_podcast::PodcastFailure>, Vec<ProgressUpdate>) {
    let updates = Mutex::new(Vec::new());
    let result = create_podcast(request, &|update| updates.lock().unwrap().push(update)).await;
    (result, updates.into_inner().unwrap())
}

fn chars(text: &str) -> f64 {
    text.chars().count() as f64
}

// ---------------------------------------------------------------------------
// Generated episodes
// ---------------------------------------------------------------------------

#[tokio::test]
async fn generated_episode_end_to_end() {
    let root = tempfile::tempdir().unwrap();
    let outliner = ScriptedModel::new("outliner", vec![reply(OUTLINE_JSON)]);
    let writer = ScriptedModel::new("writer", writer_replies());
    let tts = MockTts::new("voice").shared();
    let req = request(root.path(), TtsModels::new(tts.clone()), generation(outliner.clone(), writer.clone()), no_ffmpeg());
    let output_dir = req.output_dir.clone();
    let stem = req.episode_dir_name.clone();

    let (result, updates) = run(req).await;
    let output = result.unwrap();

    // Dialogue and outline.
    let outline = output.outline.as_ref().unwrap();
    assert_eq!(outline.segments.len(), 2);
    assert_eq!(output.transcript.len(), 4);
    assert_eq!(output.transcript[0], Dialogue::new("Velvet Noir", "Good evening."));
    assert_eq!((outliner.call_count(), writer.call_count()), (1, 2));

    // Each turn spoken with its speaker's voice, in transcript order.
    let mut spoken = tts.calls();
    spoken.sort_by_key(|(_, text)| output.transcript.iter().position(|t| &t.dialogue == text));
    assert_eq!(
        spoken,
        [
            ("af_bella".to_string(), "Good evening.".to_string()),
            ("am_adam".to_string(), "Hello there.".to_string()),
            ("am_adam".to_string(), "That is all.".to_string()),
            ("af_bella".to_string(), "Good night.".to_string()),
        ]
    );

    // Files laid out as the original's output directory.
    assert_eq!(output.output_dir, output_dir);
    assert_eq!(output.audio_file, output_dir.join("audio").join(format!("{stem}.mp3")));
    assert_eq!(output.combine_method, CombineMethod::Mp3Frames);
    let clip_names: Vec<String> =
        output.clips.iter().map(|p| p.file_name().unwrap().to_string_lossy().into_owned()).collect();
    assert_eq!(clip_names, ["0000.mp3", "0001.mp3", "0002.mp3", "0003.mp3"]);
    assert!(output.clips.iter().all(|clip| clip.parent().unwrap() == output_dir.join("clips")));

    let outline_json: serde_json::Value =
        serde_json::from_slice(&std::fs::read(output_dir.join("outline.json")).unwrap()).unwrap();
    assert_eq!(outline_json, output.outline_record().unwrap());
    assert_eq!(outline_json["segments"][0]["size"], json!("short"));
    let transcript_json: serde_json::Value =
        serde_json::from_slice(&std::fs::read(output_dir.join("transcript.json")).unwrap()).unwrap();
    assert_eq!(transcript_json, json!([
        {"speaker": "Velvet Noir", "dialogue": "Good evening."},
        {"speaker": "Max", "dialogue": "Hello there."},
        {"speaker": "Max", "dialogue": "That is all."},
        {"speaker": "Velvet Noir", "dialogue": "Good night."},
    ]));
    assert_eq!(output.transcript_record()["transcript"], transcript_json);

    // Duration: one 24 ms frame per character, no gap configured.
    let expected = output.transcript.iter().map(|t| chars(&t.dialogue)).sum::<f64>() * MP3_FRAME_SECONDS;
    let measured = output.duration_seconds.unwrap();
    assert!((measured - expected).abs() < 1e-6, "{measured} vs {expected}");
    assert!((duration_seconds(&output.audio_file).unwrap() - expected).abs() < 1e-6);

    // Stored relative to the podcasts folder, and resolvable back.
    let stored = to_relative_audio_path(root.path(), &output.audio_file).unwrap();
    assert_eq!(stored, format!("episodes/{stem}/audio/{stem}.mp3"));
    assert!(resolve_contained_audio_path(root.path(), Some(&stored)).unwrap().is_file());

    // Progress: outline 5, transcript segment 2/2 at 20, audio 35, then four clips to 95.
    let percents: Vec<u8> = updates.iter().map(|u| u.percent).collect();
    assert_eq!(percents, [5, 20, 35, 50, 65, 80, 95]);
    let phases: Vec<ProgressPhase> = updates.iter().map(|u| u.phase).collect();
    assert_eq!(phases[0], ProgressPhase::Outline);
    assert_eq!(phases[1], ProgressPhase::Transcript);
    assert!(phases[2..].iter().all(|phase| *phase == ProgressPhase::Audio));
}

#[tokio::test]
async fn episode_from_an_episode_profile() {
    let profile: EpisodeProfile = serde_json::from_value(json!({
        "name": "Deep Dive", "speaker_config": "speaker_profile:sp1", "default_briefing": "Go deep.",
        "num_segments": 3, "language": "pt-BR", "max_tokens": 4000
    }))
    .unwrap();
    let outliner = ScriptedModel::new("outliner", vec![reply(OUTLINE_JSON)]);
    let writer = ScriptedModel::new("writer", writer_replies());
    let input = GenerationInput::from_profile(&profile, "Source text", Some("Mention sync."), outliner.clone(), writer)
        .unwrap();
    assert_eq!(input.briefing, "Go deep.\n\nAdditional instructions: Mention sync.");
    assert_eq!((input.num_segments, input.options.max_tokens), (3, Some(4000)));

    let root = tempfile::tempdir().unwrap();
    let tts = MockTts::new("voice").shared();
    let (result, _) =
        run(request(root.path(), TtsModels::new(tts), PodcastSource::Generate(Box::new(input)), no_ffmpeg())).await;
    result.unwrap();

    let call = &outliner.calls()[0];
    assert_eq!(call.options.max_tokens, Some(4000));
    // The BCP 47 code reaches the prompt as a language name.
    assert!(call.prompt().contains("You MUST generate ALL content in Portuguese"));
    assert!(call.prompt().contains("Create 3 distinct segments"));

    let mut bad = profile.clone();
    bad.num_segments = 2;
    let outliner = ScriptedModel::new("o", vec![]);
    let error = GenerationInput::from_profile(&bad, "x", None, outliner.clone(), outliner).unwrap_err();
    assert_eq!(error.to_string(), "Number of segments must be between 3 and 20");
}

#[tokio::test]
async fn an_invalid_language_code_fails_before_any_model_call() {
    let outliner = ScriptedModel::new("outliner", vec![reply(OUTLINE_JSON)]);
    let writer = ScriptedModel::new("writer", vec![]);
    let mut source = generation(outliner.clone(), writer);
    if let PodcastSource::Generate(input) = &mut source {
        input.language = Some("klingon".into());
    }
    let root = tempfile::tempdir().unwrap();
    let (result, _) = run(request(root.path(), TtsModels::new(MockTts::new("v").shared()), source, no_ffmpeg())).await;
    let failure = result.unwrap_err();
    assert!(failure.error.to_string().starts_with("Invalid language code: 'klingon'."));
    assert_eq!(outliner.call_count(), 0);
}

#[tokio::test]
async fn a_transcript_failure_keeps_the_outline_for_the_episode_record() {
    let root = tempfile::tempdir().unwrap();
    let outliner = ScriptedModel::new("outliner", vec![reply(OUTLINE_JSON)]);
    let invented = transcript_json(&[("Narrator", "Once upon a time.")]);
    let writer = ScriptedModel::new("writer", vec![reply(&invented), reply(&invented)]);
    let tts = MockTts::new("voice").shared();
    let (result, updates) =
        run(request(root.path(), TtsModels::new(tts.clone()), generation(outliner, writer), no_ffmpeg())).await;

    let failure = result.unwrap_err();
    assert!(failure.outline.is_some());
    assert!(failure.transcript.is_none());
    assert!(failure.error.is_permanent());
    let message = failure.message();
    assert!(message.contains("Invalid speaker name 'Narrator'"), "{message}");
    assert!(message.contains("\n\nNOTE: "), "{message}");
    assert!(tts.calls().is_empty(), "no speech is paid for after a failed transcript");
    assert_eq!(updates.last().unwrap().percent, 5);
}

#[tokio::test]
async fn an_audio_failure_keeps_outline_and_transcript() {
    let root = tempfile::tempdir().unwrap();
    let outliner = ScriptedModel::new("outliner", vec![reply(OUTLINE_JSON)]);
    let writer = ScriptedModel::new("writer", writer_replies());
    let tts = MockTts::new("voice")
        .failing_with(vec![AiError::InvalidRequest("Voice name echo is not supported".into())])
        .shared();
    let (result, _) =
        run(request(root.path(), TtsModels::new(tts), generation(outliner, writer), no_ffmpeg())).await;

    let failure = result.unwrap_err();
    assert!(failure.outline.is_some());
    assert_eq!(failure.transcript.as_ref().map(Vec::len), Some(4));
    assert!(matches!(failure.error, PodcastError::Ai(AiError::InvalidRequest(_))));
    assert!(failure.message().contains("Check the speaker profile voices in Podcasts -> Profiles"));
}

// ---------------------------------------------------------------------------
// Script episodes
// ---------------------------------------------------------------------------

#[tokio::test]
async fn script_episode_skips_the_language_models() {
    let root = tempfile::tempdir().unwrap();
    let tts = MockTts::new("voice").shared();
    let script = "Intro line.\n\nMax: Hello.\n**Velvet Noir:** Back to *me*.";
    let req = request(root.path(), TtsModels::new(tts.clone()), PodcastSource::Script(script.into()), no_ffmpeg());
    let output_dir = req.output_dir.clone();
    let (result, updates) = run(req).await;
    let output = result.unwrap();

    assert!(output.outline.is_none());
    assert_eq!(output.outline_record(), None);
    assert_eq!(
        output.transcript,
        [
            Dialogue::new("Velvet Noir", "Intro line."),
            Dialogue::new("Max", "Hello."),
            Dialogue::new("Velvet Noir", "Back to me."),
        ]
    );
    assert_eq!(
        output.transcript_record(),
        json!({"transcript": [
            {"speaker": "Velvet Noir", "dialogue": "Intro line."},
            {"speaker": "Max", "dialogue": "Hello."},
            {"speaker": "Velvet Noir", "dialogue": "Back to me."},
        ]})
    );
    let mut voices: Vec<(String, String)> = tts.calls();
    voices.sort_by_key(|(_, text)| output.transcript.iter().position(|t| &t.dialogue == text));
    assert_eq!(voices[0], ("af_bella".to_string(), "Intro line.".to_string()));
    assert_eq!(voices[1], ("am_adam".to_string(), "Hello.".to_string()));

    assert!(!output_dir.join("outline.json").exists());
    assert!(output_dir.join("transcript.json").is_file());
    assert!(output.audio_file.is_file());
    assert_eq!(updates.first().unwrap().percent, 35);
    assert_eq!(updates.last().unwrap().percent, 95);
}

#[tokio::test]
async fn an_empty_script_is_refused_before_any_speech() {
    let root = tempfile::tempdir().unwrap();
    let tts = MockTts::new("voice").shared();
    let (result, _) = run(request(
        root.path(),
        TtsModels::new(tts.clone()),
        PodcastSource::Script("  \n# \n---\n".into()),
        no_ffmpeg(),
    ))
    .await;
    let failure = result.unwrap_err();
    assert_eq!(failure.error.to_string(), "The script has no text to read");
    assert!(matches!(failure.error, PodcastError::InvalidInput(_)));
    assert!(tts.calls().is_empty());
}

#[tokio::test]
async fn an_invalid_speaker_profile_is_refused_up_front() {
    let root = tempfile::tempdir().unwrap();
    let mut req = request(
        root.path(),
        TtsModels::new(MockTts::new("v").shared()),
        PodcastSource::Script("Hi.".into()),
        no_ffmpeg(),
    );
    req.speaker_profile.speakers[1].voice_id = "af_bella".into();
    let (result, _) = run(req).await;
    assert_eq!(result.unwrap_err().error.to_string(), "Voice IDs must be unique");
}

// ---------------------------------------------------------------------------
// Speech
// ---------------------------------------------------------------------------

#[tokio::test]
async fn per_speaker_voice_models_override_the_profile_model() {
    let root = tempfile::tempdir().unwrap();
    let default = MockTts::new("default").shared();
    let max_only = MockTts::new("max-only").shared();
    let tts = TtsModels::new(default.clone()).with_speaker("Max", max_only.clone());
    let (result, _) = run(request(
        root.path(),
        tts,
        PodcastSource::Script("One.\n\nMax: Two.\n\nVelvet Noir: Three.".into()),
        no_ffmpeg(),
    ))
    .await;
    result.unwrap();
    let mut default_texts: Vec<String> = default.calls().into_iter().map(|(_, text)| text).collect();
    default_texts.sort();
    assert_eq!(default_texts, ["One.", "Three."]);
    assert_eq!(max_only.calls(), [("am_adam".to_string(), "Two.".to_string())]);
}

#[tokio::test]
async fn clips_keep_transcript_order_with_bounded_concurrency() {
    let dir = tempfile::tempdir().unwrap();
    // Earlier turns take longer, so they finish after later ones.
    let tts = MockTts::new("voice")
        .delay(|text: &str| Duration::from_millis(80u64.saturating_sub(text.len() as u64 * 5)))
        .shared();
    let transcript: Vec<Dialogue> = (0..7)
        .map(|i| Dialogue::new(if i % 2 == 0 { "Velvet Noir" } else { "Max" }, "x".repeat(i + 1)))
        .collect();
    let options = AudioOptions { batch_size: 3, ..no_ffmpeg() };
    let progress = Mutex::new(Vec::new());
    let clips = synthesize_clips(
        &transcript,
        &speaker_profile(),
        &TtsModels::new(tts.clone()),
        &dir.path().join("clips"),
        &options,
        &|done, total| progress.lock().unwrap().push((done, total)),
    )
    .await
    .unwrap();

    assert!(tts.max_in_flight() <= 3, "{}", tts.max_in_flight());
    assert!(tts.max_in_flight() >= 2, "clips of a batch run concurrently");
    for (index, clip) in clips.iter().enumerate() {
        assert_eq!(clip.file_name().unwrap().to_string_lossy(), format!("{index:04}.mp3"));
        let seconds = duration_seconds(clip).unwrap();
        assert!((seconds - (index + 1) as f64 * MP3_FRAME_SECONDS).abs() < 1e-6, "clip {index}");
    }
    assert_eq!(*progress.lock().unwrap(), (1..=7).map(|done| (done, 7)).collect::<Vec<_>>());
}

#[tokio::test]
async fn transient_speech_errors_are_retried() {
    let dir = tempfile::tempdir().unwrap();
    let tts = MockTts::new("voice")
        .failing_with(vec![AiError::RateLimit("429".into()), AiError::Network("reset".into())])
        .shared();
    let clips = synthesize_clips(
        &[Dialogue::new("Max", "Hello.")],
        &speaker_profile(),
        &TtsModels::new(tts.clone()),
        &dir.path().join("clips"),
        &no_ffmpeg(),
        &|_, _| {},
    )
    .await
    .unwrap();
    assert_eq!(clips.len(), 1);
    assert_eq!(tts.calls().len(), 3);
}

#[tokio::test]
async fn a_speaker_missing_from_the_profile_fails_before_any_speech() {
    let dir = tempfile::tempdir().unwrap();
    let tts = MockTts::new("voice").shared();
    let error = synthesize_clips(
        &[Dialogue::new("Max", "Fine."), Dialogue::new("Nobody", "Who?")],
        &speaker_profile(),
        &TtsModels::new(tts.clone()),
        &dir.path().join("clips"),
        &no_ffmpeg(),
        &|_, _| {},
    )
    .await
    .unwrap_err();
    assert_eq!(error.to_string(), "Speaker 'Nobody' not found in profile");
    assert!(tts.calls().is_empty());
}

// ---------------------------------------------------------------------------
// Audio formats (Gemini / Vertex speech returns WAV, the others MP3)
// ---------------------------------------------------------------------------

#[tokio::test]
async fn wav_only_episode_is_joined_without_ffmpeg() {
    let root = tempfile::tempdir().unwrap();
    let tts = MockTts::new("gemini-tts").format(ClipFormat::Wav).shared();
    let mut audio = no_ffmpeg();
    audio.gap_ms = 400;
    let req = request(root.path(), TtsModels::new(tts), PodcastSource::Script("One two.\n\nMax: Three.".into()), audio);
    let stem = req.episode_dir_name.clone();
    let (result, _) = run(req).await;
    let output = result.unwrap();

    assert_eq!(output.combine_method, CombineMethod::WavSamples);
    assert!(output.audio_file.ends_with(format!("audio/{stem}.wav")));
    assert!(output.clips.iter().all(|clip| clip.extension().unwrap() == "wav"));
    let expected = (chars("One two.") + chars("Three.")) * MP3_FRAME_SECONDS + 0.4;
    assert!((output.duration_seconds.unwrap() - expected).abs() < 1e-3);
}

#[tokio::test]
async fn mixed_formats_without_ffmpeg_are_a_clear_error() {
    let root = tempfile::tempdir().unwrap();
    let mp3 = MockTts::new("openai-tts").shared();
    let wav = MockTts::new("gemini-tts").format(ClipFormat::Wav).shared();
    let tts = TtsModels::new(mp3).with_speaker("Max", wav);
    let (result, _) =
        run(request(root.path(), tts, PodcastSource::Script("One.\n\nMax: Two.".into()), no_ffmpeg())).await;

    let failure = result.unwrap_err();
    assert!(matches!(failure.error, PodcastError::Audio(_)));
    let message = failure.message();
    assert!(message.contains("different formats (.mp3, .wav)"), "{message}");
    assert!(message.contains("Install ffmpeg"), "{message}");
    assert_eq!(failure.transcript.as_ref().map(Vec::len), Some(2));
}

#[tokio::test]
async fn mixed_formats_are_encoded_to_mp3_by_ffmpeg() {
    let Some(ffmpeg) = find_executable("ffmpeg") else {
        eprintln!("skipped: ffmpeg is not installed");
        return;
    };
    let root = tempfile::tempdir().unwrap();
    // Long enough clips for a real encoder: one frame (24 ms) per character.
    let mp3 = MockTts::new("openai-tts").shared();
    let wav = MockTts::new("gemini-tts").format(ClipFormat::Wav).shared();
    let tts = TtsModels::new(mp3).with_speaker("Max", wav);
    let first = "a".repeat(40);
    let second = "b".repeat(40);
    let script = format!("{first}\n\nMax: {second}");
    let req = request(root.path(), tts, PodcastSource::Script(script), audio_options(FfmpegMode::Path(ffmpeg)));
    let (result, _) = run(req).await;
    let output = result.unwrap();

    assert_eq!(output.combine_method, CombineMethod::Ffmpeg);
    assert_eq!(output.audio_file.extension().unwrap(), "mp3");
    let seconds = output.duration_seconds.unwrap();
    assert!((seconds - 80.0 * MP3_FRAME_SECONDS).abs() < 0.2, "{seconds}");
}
