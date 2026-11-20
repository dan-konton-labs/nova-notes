//! Outline and transcript generation: the replacement for podcast-creator's
//! language-model nodes.
//!
//! Both steps follow the same path per model call:
//!
//! 1. render the prompt ([`crate::prompts`]);
//! 2. call the model in JSON mode through the refusal guard, repeating
//!    transient provider errors ([`RetryConfig`]);
//! 3. parse and validate the reply ([`crate::reply`]);
//! 4. if the reply cannot be used, ask once more, showing the model its reply
//!    and what was wrong with it. A reply that was cut off at the output token
//!    limit is not asked again: the same limit would cut it off again.

use nn_llm::{ChatMessage, ChatOptions, LanguageModel};

use crate::error::{PodcastError, Result};
use crate::models::{Content, Dialogue, Outline, Segment, Speaker};
use crate::prompts::{OutlinePrompt, TranscriptPrompt};
use crate::refusal_guard::{guarded_chat, model_key, RefusalPolicy};
use crate::reply::{clean_thinking_content, parse_outline, parse_transcript, ReplyError};
use crate::retry::{retry, RetryConfig};

/// Output token limit for an outline call unless the episode profile sets one.
pub const DEFAULT_OUTLINE_MAX_TOKENS: u32 = 8192;
/// Output token limit for a transcript call unless the episode profile sets one.
pub const DEFAULT_TRANSCRIPT_MAX_TOKENS: u32 = 8192;

/// Settings shared by the outline and transcript calls of one episode.
#[derive(Debug, Clone, Default)]
pub struct GenerationOptions {
    /// `EpisodeProfile::max_tokens`. `None` uses the 8192 default.
    pub max_tokens: Option<u32>,
    pub refusal: RefusalPolicy,
    /// Applied to transient provider errors (rate limit, network, timeout, 5xx).
    pub retry: RetryConfig,
}

/// Inputs of [`generate_outline`].
#[derive(Debug, Clone)]
pub struct OutlineRequest<'a> {
    pub briefing: &'a str,
    pub content: &'a Content,
    pub speakers: &'a [Speaker],
    pub num_segments: u32,
    /// Language name as shown to the model (see
    /// [`crate::language::resolve_language_name`]); `None` adds no instruction.
    pub language: Option<&'a str>,
}

/// Inputs of [`generate_transcript`].
#[derive(Debug, Clone)]
pub struct TranscriptRequest<'a> {
    pub briefing: &'a str,
    pub content: &'a Content,
    pub speakers: &'a [Speaker],
    pub outline: &'a Outline,
    /// Language name as shown to the model; `None` adds no instruction.
    pub language: Option<&'a str>,
}

/// Generate the episode outline: `num_segments` segments with a name, a
/// description and a size.
pub async fn generate_outline(
    model: &dyn LanguageModel,
    request: &OutlineRequest<'_>,
    options: &GenerationOptions,
) -> Result<Outline> {
    tracing::info!("Starting outline generation");
    let prompt = OutlinePrompt {
        briefing: request.briefing,
        content: request.content,
        speakers: request.speakers,
        num_segments: request.num_segments,
        language: request.language,
    }
    .render()?;

    let max_tokens = options.max_tokens.unwrap_or(DEFAULT_OUTLINE_MAX_TOKENS);
    let outline = structured_call(model, &prompt, max_tokens, options, &parse_outline).await?;
    tracing::info!("Generated outline with {} segments", outline.segments.len());
    Ok(outline)
}

/// Generate the dialogue, one model call per outline segment. Each call sees
/// the dialogue written so far, and every speaker name is checked against
/// `request.speakers`.
///
/// `on_segment(index, total, segment)` is called before each segment's call,
/// with a 1-based index.
pub async fn generate_transcript(
    model: &dyn LanguageModel,
    request: &TranscriptRequest<'_>,
    options: &GenerationOptions,
    on_segment: &(dyn Fn(usize, usize, &Segment) + Send + Sync),
) -> Result<Vec<Dialogue>> {
    tracing::info!("Starting transcript generation");
    if request.speakers.is_empty() {
        return Err(PodcastError::InvalidInput("The speaker profile has no speakers".into()));
    }
    let speaker_names: Vec<String> = request.speakers.iter().map(|s| s.name.trim().to_string()).collect();
    let max_tokens = options.max_tokens.unwrap_or(DEFAULT_TRANSCRIPT_MAX_TOKENS);
    let parse = |reply: &str| parse_transcript(reply, &speaker_names);

    let total = request.outline.segments.len();
    let mut transcript: Vec<Dialogue> = Vec::new();
    for (index, segment) in request.outline.segments.iter().enumerate() {
        tracing::info!("Generating transcript for segment {}/{total}: {}", index + 1, segment.name);
        on_segment(index + 1, total, segment);

        let prompt = TranscriptPrompt {
            briefing: request.briefing,
            content: request.content,
            speakers: request.speakers,
            outline: request.outline,
            segment,
            is_final: index + 1 == total,
            transcript: &transcript,
            language: request.language,
        }
        .render()?;

        let turns = structured_call(model, &prompt, max_tokens, options, &parse).await?;
        transcript.extend(turns);
    }

    tracing::info!("Generated transcript with {} dialogue segments", transcript.len());
    Ok(transcript)
}

fn was_truncated(finish_reason: Option<&str>) -> bool {
    matches!(
        finish_reason.map(|reason| reason.trim().to_ascii_lowercase()).as_deref(),
        Some("length") | Some("max_tokens")
    )
}

fn unusable(error: &ReplyError, truncated: bool, max_tokens: u32) -> PodcastError {
    if truncated {
        return PodcastError::Incomplete(format!(
            "{error} (the reply was cut off at the output token limit, max_tokens = {max_tokens})"
        ));
    }
    match error {
        ReplyError::Empty => PodcastError::Incomplete(error.to_string()),
        _ => PodcastError::InvalidOutput(error.to_string()),
    }
}

fn repair_instruction(error: &ReplyError) -> String {
    format!(
        "Your previous reply could not be used: {}\n\n\
         Reply again with only the corrected JSON object described above. \
         Write every value out in full, with no placeholders, no explanations and no code fences.",
        error.detail()
    )
}

/// One prompt in, one validated value out, with a single repair round.
async fn structured_call<T>(
    model: &dyn LanguageModel,
    prompt: &str,
    max_tokens: u32,
    options: &GenerationOptions,
    parse: &(dyn Fn(&str) -> std::result::Result<T, ReplyError> + Send + Sync),
) -> Result<T> {
    let chat_options = ChatOptions { max_tokens: Some(max_tokens), json: true, ..ChatOptions::default() };
    let key = model_key(model);
    let call = |messages: Vec<ChatMessage>| {
        let chat_options = &chat_options;
        let key = &key;
        async move {
            retry(
                &options.retry,
                key,
                |error: &PodcastError| matches!(error, PodcastError::Ai(e) if e.is_retryable()),
                || guarded_chat(model, &messages, chat_options, &options.refusal),
            )
            .await
        }
    };

    let first = call(vec![ChatMessage::user(prompt)]).await?;
    let error = match parse(&first.content) {
        Ok(value) => return Ok(value),
        Err(error) => error,
    };
    if was_truncated(first.finish_reason.as_deref()) {
        return Err(unusable(&error, true, max_tokens));
    }

    tracing::warn!("{key} returned an unusable reply ({}); asking once more", error.detail());
    let messages = match error {
        // An empty assistant turn is rejected by several providers, and there
        // is nothing to correct: ask the same question again.
        ReplyError::Empty => vec![ChatMessage::user(prompt)],
        _ => vec![
            ChatMessage::user(prompt),
            ChatMessage::assistant(clean_thinking_content(&first.content)),
            ChatMessage::user(repair_instruction(&error)),
        ],
    };
    let second = call(messages).await?;
    parse(&second.content)
        .map_err(|error| unusable(&error, was_truncated(second.finish_reason.as_deref()), max_tokens))
}
