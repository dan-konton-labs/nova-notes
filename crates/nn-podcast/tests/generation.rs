//! Outline and transcript generation against scripted language models.

mod common;

use std::sync::Mutex;

use common::*;
use nn_llm::{AiError, Role};
use nn_podcast::generate::{DEFAULT_OUTLINE_MAX_TOKENS, DEFAULT_TRANSCRIPT_MAX_TOKENS};
use nn_podcast::{
    generate_outline, generate_transcript, Content, Dialogue, GenerationOptions, Outline, OutlineRequest,
    PodcastError, RefusalPolicy, RetryConfig, Segment, SegmentSize, TranscriptRequest,
};

fn options() -> GenerationOptions {
    GenerationOptions { retry: RetryConfig::immediate(3), ..GenerationOptions::default() }
}

async fn outline_with(
    model: &ScriptedModel,
    options: &GenerationOptions,
    language: Option<&str>,
) -> Result<Outline, PodcastError> {
    let content = Content::from("Offline-first apps keep working without a network.");
    let speakers = speakers();
    generate_outline(
        model,
        &OutlineRequest {
            briefing: "Explain offline-first design to developers.",
            content: &content,
            speakers: &speakers,
            num_segments: 4,
            language,
        },
        options,
    )
    .await
}

async fn outline(model: &ScriptedModel) -> Result<Outline, PodcastError> {
    outline_with(model, &options(), None).await
}

fn two_segments() -> Outline {
    Outline {
        segments: vec![
            Segment { name: "Setting out".into(), description: "Why it matters.".into(), size: SegmentSize::Short },
            Segment { name: "Wrapping up".into(), description: "What to remember.".into(), size: SegmentSize::Long },
        ],
    }
}

async fn transcript_with(
    model: &ScriptedModel,
    outline: &Outline,
    on_segment: &(dyn Fn(usize, usize, &Segment) + Send + Sync),
) -> Result<Vec<Dialogue>, PodcastError> {
    let content = Content::from("Offline-first apps keep working without a network.");
    let speakers = speakers();
    generate_transcript(
        model,
        &TranscriptRequest {
            briefing: "Explain offline-first design to developers.",
            content: &content,
            speakers: &speakers,
            outline,
            language: None,
        },
        &options(),
        on_segment,
    )
    .await
}

async fn transcript(model: &ScriptedModel) -> Result<Vec<Dialogue>, PodcastError> {
    transcript_with(model, &two_segments(), &|_, _, _| {}).await
}

// ---------------------------------------------------------------------------
// Outline
// ---------------------------------------------------------------------------

#[tokio::test]
async fn outline_is_one_json_mode_call_with_the_rendered_prompt() {
    let model = ScriptedModel::new("outliner", vec![reply(OUTLINE_JSON)]);
    let outline = outline(&model).await.unwrap();

    assert_eq!(outline.segments.len(), 2);
    assert_eq!(outline.segments[0].name, "Setting out");
    assert_eq!(outline.segments[0].size, SegmentSize::Short);
    assert_eq!(outline.segments[1].size, SegmentSize::Medium);

    let calls = model.calls();
    assert_eq!(calls.len(), 1);
    assert!(calls[0].options.json);
    assert_eq!(calls[0].options.max_tokens, Some(DEFAULT_OUTLINE_MAX_TOKENS));
    assert_eq!(calls[0].options.max_tokens, Some(8192));
    assert_eq!(calls[0].options.temperature, None);
    assert_eq!(calls[0].messages.len(), 1);
    assert_eq!(calls[0].messages[0].role, Role::User);

    let prompt = calls[0].prompt();
    assert!(prompt.contains("Explain offline-first design to developers."));
    assert!(prompt.contains("Offline-first apps keep working without a network."));
    assert!(prompt.contains("- **Velvet Noir**: A late-night radio host."));
    assert!(prompt.contains("Create 4 distinct segments"));
    assert!(!prompt.contains("IMPORTANT LANGUAGE INSTRUCTION"));
}

#[tokio::test]
async fn profile_max_tokens_and_language_reach_the_model() {
    let model = ScriptedModel::new("outliner", vec![reply(OUTLINE_JSON)]);
    let options = GenerationOptions { max_tokens: Some(12_000), ..options() };
    outline_with(&model, &options, Some("Portuguese")).await.unwrap();

    let call = &model.calls()[0];
    assert_eq!(call.options.max_tokens, Some(12_000));
    assert!(call.prompt().contains("You MUST generate ALL content in Portuguese"));
    // The English sample is not shown to a non-English run.
    assert!(!call.prompt().contains("EXCERPT"));
}

#[tokio::test]
async fn thinking_fences_and_prose_around_the_json_are_tolerated() {
    let wrapped = format!("<think>\nFour segments? Let me plan.\n</think>\nHere you go:\n```json\n{OUTLINE_JSON}\n```\nDone!");
    let model = ScriptedModel::new("outliner", vec![reply(&wrapped)]);
    assert_eq!(outline(&model).await.unwrap().segments.len(), 2);
    assert_eq!(model.call_count(), 1);
}

#[tokio::test]
async fn an_unusable_reply_is_repaired_with_one_more_call() {
    let model = ScriptedModel::new(
        "outliner",
        vec![reply("Sure, I can help with an outline. Where should I start?"), reply(OUTLINE_JSON)],
    );
    assert_eq!(outline(&model).await.unwrap().segments.len(), 2);

    let calls = model.calls();
    assert_eq!(calls.len(), 2);
    let repair = &calls[1].messages;
    assert_eq!(repair.iter().map(|m| m.role).collect::<Vec<_>>(), [Role::User, Role::Assistant, Role::User]);
    assert_eq!(repair[0].content, calls[0].prompt());
    assert_eq!(repair[1].content, "Sure, I can help with an outline. Where should I start?");
    assert!(repair[2].content.contains("could not be used: the reply was not valid JSON"));
    assert!(repair[2].content.contains("only the corrected JSON object"));
    assert!(calls[1].options.json);
}

#[tokio::test]
async fn a_validation_failure_tells_the_model_what_was_wrong() {
    let bad_size = r#"{"segments": [{"name": "A", "description": "d", "size": "enorme"}]}"#;
    let model = ScriptedModel::new("outliner", vec![reply(bad_size), reply(OUTLINE_JSON)]);
    outline(&model).await.unwrap();
    let repair = &model.calls()[1].messages[2].content;
    assert!(repair.contains("segments.0.size"), "{repair}");
    assert!(repair.contains("'short', 'medium' or 'long'"), "{repair}");
}

#[tokio::test]
async fn a_second_unusable_reply_fails_with_a_hint() {
    let model = ScriptedModel::new("outliner", vec![reply("no json here"), reply("still none")]);
    let error = outline(&model).await.unwrap_err();
    assert_eq!(model.call_count(), 2, "exactly one repair attempt");
    assert!(matches!(error, PodcastError::InvalidOutput(_)), "{error:?}");
    assert_eq!(error.to_string(), "Invalid json output: still none");
    assert!(error.is_permanent());
    assert!(error.message_with_hint().contains("NOTE: The model's response could not be parsed as JSON"));
}

#[tokio::test]
async fn a_truncated_reply_is_not_asked_again() {
    let cut = &OUTLINE_JSON[..OUTLINE_JSON.len() - 40];
    let model = ScriptedModel::new("outliner", vec![reply_with(cut, "length"), reply(OUTLINE_JSON)]);
    let error = outline(&model).await.unwrap_err();
    assert_eq!(model.call_count(), 1);
    assert!(matches!(error, PodcastError::Incomplete(_)), "{error:?}");
    let message = error.to_string();
    assert!(message.starts_with("Invalid json output: {\"segments\""), "{message}");
    assert!(message.contains("cut off at the output token limit, max_tokens = 8192"), "{message}");
    assert!(error.hint().unwrap().contains("max_tokens"));
}

#[tokio::test]
async fn a_complete_reply_that_hit_the_limit_exactly_is_still_accepted() {
    let model = ScriptedModel::new("outliner", vec![reply_with(OUTLINE_JSON, "length")]);
    assert_eq!(outline(&model).await.unwrap().segments.len(), 2);
}

#[tokio::test]
async fn an_empty_reply_is_asked_again_as_the_same_question() {
    let model = ScriptedModel::new(
        "outliner",
        vec![reply("<think>all the output went into thinking"), reply(OUTLINE_JSON)],
    );
    outline(&model).await.unwrap();
    let calls = model.calls();
    assert_eq!(calls.len(), 2);
    assert_eq!(calls[1].messages.len(), 1, "no empty assistant turn is sent back");
    assert_eq!(calls[1].prompt(), calls[0].prompt());

    let model = ScriptedModel::new("outliner", vec![reply(""), reply("  ")]);
    let error = outline(&model).await.unwrap_err();
    assert!(matches!(error, PodcastError::Incomplete(_)), "{error:?}");
    assert!(error.hint().unwrap().contains("<think>"));
}

#[tokio::test]
async fn a_refusal_is_retried_before_parsing() {
    let model = ScriptedModel::new("outliner", vec![refused(), refused(), reply(OUTLINE_JSON)]);
    assert_eq!(outline(&model).await.unwrap().segments.len(), 2);
    assert_eq!(model.call_count(), 3);
    // Each retry is the same request, not a repair conversation.
    assert!(model.calls().iter().all(|call| call.messages.len() == 1));
}

#[tokio::test]
async fn a_refusal_on_every_attempt_fails_clearly_without_a_repair_round() {
    let model = ScriptedModel::new("outliner", vec![refused(), refused(), refused(), reply(OUTLINE_JSON)]);
    let error = outline(&model).await.unwrap_err();
    assert_eq!(model.call_count(), 3);
    assert!(matches!(error, PodcastError::Refusal(_)));
    assert_eq!(error.to_string(), "mock/outliner refused the request (refusal) on every attempt");
    assert!(error.message_with_hint().contains("PODCAST_REFUSAL_FALLBACK_MODEL"));
}

#[tokio::test]
async fn the_fallback_model_answers_after_the_retries_are_refused() {
    let model = ScriptedModel::new("outliner", vec![refused()]);
    let fallback = ScriptedModel::new("small-outliner", vec![reply(OUTLINE_JSON)]);
    let options =
        GenerationOptions { refusal: RefusalPolicy { retries: 0, fallback: Some(fallback.clone()) }, ..options() };
    assert_eq!(outline_with(&model, &options, None).await.unwrap().segments.len(), 2);
    assert_eq!((model.call_count(), fallback.call_count()), (1, 1));
    assert_eq!(fallback.calls()[0].prompt(), model.calls()[0].prompt());
}

#[tokio::test]
async fn transient_provider_errors_are_retried() {
    let model = ScriptedModel::new(
        "outliner",
        vec![
            Err(AiError::RateLimit("slow down".into())),
            Err(AiError::Provider { status: Some(503), message: "overloaded".into() }),
            reply(OUTLINE_JSON),
        ],
    );
    assert_eq!(outline(&model).await.unwrap().segments.len(), 2);
    assert_eq!(model.call_count(), 3);
}

#[tokio::test]
async fn transient_errors_stop_after_the_configured_attempts() {
    let model = ScriptedModel::new(
        "outliner",
        vec![Err(AiError::Timeout("1".into())), Err(AiError::Timeout("2".into())), Err(AiError::Timeout("3".into()))],
    );
    let error = outline(&model).await.unwrap_err();
    assert!(matches!(error, PodcastError::Ai(AiError::Timeout(_))));
    assert!(!error.is_permanent());
    assert_eq!(model.call_count(), 3);
}

#[tokio::test]
async fn permanent_provider_errors_are_not_retried() {
    let model = ScriptedModel::new("outliner", vec![Err(AiError::Authentication("bad key".into())), reply(OUTLINE_JSON)]);
    let error = outline(&model).await.unwrap_err();
    assert!(matches!(error, PodcastError::Ai(AiError::Authentication(_))));
    assert!(error.is_permanent());
    assert_eq!(model.call_count(), 1);
}

// ---------------------------------------------------------------------------
// Transcript
// ---------------------------------------------------------------------------

#[tokio::test]
async fn transcript_is_one_call_per_segment_with_running_context() {
    let model = ScriptedModel::new(
        "writer",
        vec![
            reply(&transcript_json(&[("Velvet Noir", "Good evening."), ("Max", "Glad to be here.")])),
            reply(&transcript_json(&[(" Max ", "That is the lot."), ("Velvet Noir", "Good night.")])),
        ],
    );
    let seen = Mutex::new(Vec::new());
    let turns = transcript_with(&model, &two_segments(), &|index, total, segment| {
        seen.lock().unwrap().push((index, total, segment.name.clone()));
    })
    .await
    .unwrap();

    assert_eq!(
        turns,
        [
            Dialogue::new("Velvet Noir", "Good evening."),
            Dialogue::new("Max", "Glad to be here."),
            Dialogue::new("Max", "That is the lot."),
            Dialogue::new("Velvet Noir", "Good night."),
        ]
    );
    assert_eq!(
        *seen.lock().unwrap(),
        [(1, 2, "Setting out".to_string()), (2, 2, "Wrapping up".to_string())]
    );

    let calls = model.calls();
    assert_eq!(calls.len(), 2);
    for call in &calls {
        assert!(call.options.json);
        assert_eq!(call.options.max_tokens, Some(DEFAULT_TRANSCRIPT_MAX_TOKENS));
        assert!(call.prompt().contains("1. Setting out (short): Why it matters."));
        assert!(call.prompt().contains("2. Wrapping up (long): What to remember."));
        assert!(call.prompt().contains("Use the actual speaker names (Velvet Noir, Max)"));
    }

    let first = calls[0].prompt();
    assert!(first.contains("<segment>\nSetting out (short): Why it matters.\n</segment>"));
    assert!(first.contains("at least 3 turns"));
    assert!(!first.contains("Here is the current transcript so far"));
    assert!(!first.contains("This is the final segment"));

    let second = calls[1].prompt();
    assert!(second.contains("<segment>\nWrapping up (long): What to remember.\n</segment>"));
    assert!(second.contains("at least 10 turns"));
    assert!(second.contains("Here is the current transcript so far"));
    assert!(second.contains("Velvet Noir: Good evening.\nMax: Glad to be here."));
    assert!(second.contains("This is the final segment"));
}

#[tokio::test]
async fn a_placeholder_speaker_is_corrected_by_the_repair_round() {
    let model = ScriptedModel::new(
        "writer",
        vec![
            reply(r#"{"transcript": [{"speaker": "...", "dialogue": "..."}]}"#),
            reply(&transcript_json(&[("Max", "Real words.")])),
            reply(&transcript_json(&[("Velvet Noir", "And the end.")])),
        ],
    );
    let turns = transcript(&model).await.unwrap();
    assert_eq!(turns.len(), 2);

    let calls = model.calls();
    assert_eq!(calls.len(), 3);
    let repair = &calls[1].messages;
    assert_eq!(repair.len(), 3);
    assert!(
        repair[2].content.contains("Invalid speaker name '...'. Must be one of: Velvet Noir, Max"),
        "{}",
        repair[2].content
    );
    // The next segment starts a fresh conversation and sees only accepted dialogue.
    assert_eq!(calls[2].messages.len(), 1);
    assert!(calls[2].prompt().contains("Max: Real words."));
    assert!(!calls[2].prompt().contains("\"speaker\": \"...\""));
}

#[tokio::test]
async fn an_invented_speaker_twice_fails_with_the_original_message_and_hint() {
    let invented = transcript_json(&[("Narrator", "Once upon a time.")]);
    let model = ScriptedModel::new("writer", vec![reply(&invented), reply(&invented)]);
    let error = transcript(&model).await.unwrap_err();
    assert_eq!(model.call_count(), 2);
    assert!(matches!(error, PodcastError::InvalidOutput(_)));
    let message = error.to_string();
    assert!(message.starts_with("Failed to parse Transcript from completion"), "{message}");
    assert!(message.contains("Invalid speaker name 'Narrator'. Must be one of: Velvet Noir, Max"), "{message}");
    assert!(error.message_with_hint().contains("Speaker names must match the profile exactly"));
}

#[tokio::test]
async fn a_failing_segment_stops_the_transcript() {
    let model = ScriptedModel::new(
        "writer",
        vec![
            reply(&transcript_json(&[("Max", "First segment is fine.")])),
            Err(AiError::ContextLength("too long".into())),
        ],
    );
    let error = transcript(&model).await.unwrap_err();
    assert!(matches!(error, PodcastError::Ai(AiError::ContextLength(_))));
    assert_eq!(model.call_count(), 2);
}

#[tokio::test]
async fn a_solo_profile_gets_the_solo_prompt() {
    let model = ScriptedModel::new("writer", vec![reply(&transcript_json(&[("Velvet Noir", "Just me tonight.")]))]);
    let content = Content::from("c");
    let solo = &speakers()[..1];
    let outline = Outline { segments: two_segments().segments[..1].to_vec() };
    let turns = generate_transcript(
        model.as_ref(),
        &TranscriptRequest { briefing: "b", content: &content, speakers: solo, outline: &outline, language: None },
        &options(),
        &|_, _, _| {},
    )
    .await
    .unwrap();
    assert_eq!(turns, [Dialogue::new("Velvet Noir", "Just me tonight.")]);
    let prompt = model.calls()[0].prompt().to_string();
    assert!(prompt.contains("SOLO podcast with only ONE speaker (Velvet Noir)"));
    assert!(prompt.contains("This is the final segment"));

    // Another name is refused even though the reply is otherwise well formed.
    let model = ScriptedModel::new("writer", vec![reply(&transcript_json(&[("Max", "Hi.")])); 2]);
    let error = generate_transcript(
        model.as_ref(),
        &TranscriptRequest { briefing: "b", content: &content, speakers: solo, outline: &outline, language: None },
        &options(),
        &|_, _, _| {},
    )
    .await
    .unwrap_err();
    assert!(error.to_string().contains("Invalid speaker name 'Max'. Must be one of: Velvet Noir"));
}

#[tokio::test]
async fn no_speakers_is_refused_before_any_call() {
    let model = ScriptedModel::new("writer", vec![]);
    let content = Content::from("c");
    let outline = two_segments();
    let error = generate_transcript(
        model.as_ref(),
        &TranscriptRequest { briefing: "b", content: &content, speakers: &[], outline: &outline, language: None },
        &options(),
        &|_, _, _| {},
    )
    .await
    .unwrap_err();
    assert!(matches!(error, PodcastError::InvalidInput(_)));
    assert_eq!(model.call_count(), 0);
}
