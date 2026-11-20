//! The crate's error type and the hints attached to generation failures.

use nn_llm::AiError;

/// Everything that can go wrong in the podcast engine.
///
/// The variants mirror the classification the Python original expressed with
/// exception types, so the server can pick an HTTP status and decide whether a
/// failed job is worth retrying.
#[derive(Debug, thiserror::Error)]
pub enum PodcastError {
    /// Bad input from a user or a stored record. The original raised
    /// `ValueError` / `InvalidInputError` (HTTP 400).
    #[error("{0}")]
    InvalidInput(String),
    /// The deployment is missing configuration. The original raised
    /// `ConfigurationError` (HTTP 422).
    #[error("{0}")]
    Configuration(String),
    /// The provider refused the request on every attempt
    /// (`ModelRefusalError` in the original).
    #[error("{0}")]
    Refusal(String),
    /// The model answered, but the answer could not be parsed or validated.
    #[error("{0}")]
    InvalidOutput(String),
    /// The model answered with nothing, or the answer was cut off.
    #[error("{0}")]
    Incomplete(String),
    /// A model call failed.
    #[error(transparent)]
    Ai(#[from] AiError),
    /// Clips could not be synthesized, combined or measured.
    #[error("{0}")]
    Audio(String),
    /// A filesystem operation failed.
    #[error("{context}: {source}")]
    Io {
        context: String,
        #[source]
        source: std::io::Error,
    },
}

pub type Result<T> = std::result::Result<T, PodcastError>;

impl PodcastError {
    pub(crate) fn io(context: impl Into<String>, source: std::io::Error) -> Self {
        PodcastError::Io { context: context.into(), source }
    }

    /// True when running the same job again cannot be expected to help without
    /// a change of input or configuration. This is what `ValueError` meant to
    /// the original's command layer ("permanent failure, do not retry").
    pub fn is_permanent(&self) -> bool {
        match self {
            PodcastError::InvalidInput(_)
            | PodcastError::Configuration(_)
            | PodcastError::Refusal(_)
            | PodcastError::InvalidOutput(_)
            | PodcastError::Incomplete(_) => true,
            PodcastError::Ai(error) => !error.is_retryable(),
            PodcastError::Audio(_) | PodcastError::Io { .. } => false,
        }
    }

    /// An actionable hint for this failure, when one is known.
    pub fn hint(&self) -> Option<&'static str> {
        explain_generation_failure(&self.to_string())
    }

    /// The message to store on a failed job: the error, followed by
    /// `\n\nNOTE: <hint>` when a hint applies. Same shape as the original.
    pub fn message_with_hint(&self) -> String {
        let message = self.to_string();
        match explain_generation_failure(&message) {
            Some(hint) => format!("{message}\n\nNOTE: {hint}"),
            None => message,
        }
    }
}

/// Map a podcast-generation failure message to an actionable hint, or `None`.
///
/// Ordered most specific first. The texts are the original's, including the
/// environment variable name the server crate is expected to keep honouring.
pub fn explain_generation_failure(error_msg: &str) -> Option<&'static str> {
    if error_msg.contains("Invalid speaker name") {
        return Some(
            "The transcript model returned a speaker name that is not in the \
             speaker profile - usually a placeholder copied from the prompt \
             such as \"...\" rather than an invented person. Speaker names must \
             match the profile exactly. Retrying the episode often succeeds, \
             since each attempt is a fresh sample.",
        );
    }

    if error_msg.contains("Voice name") && error_msg.contains("not supported") {
        return Some(
            "The speaker profile's voice_id is not valid for its TTS model. \
             Check the speaker profile voices in Podcasts -> Profiles against the ones \
             your voice model provides (the profiles seeded on install use \
             OpenAI voice names).",
        );
    }

    if error_msg.contains("Requested entity was not found") {
        return Some(
            "Google returns this for any resource it cannot find, without \
             naming which one. Two candidates, likeliest first: a speaker \
             profile voice_id that its TTS model doesn't provide (the profiles \
             seeded on install use OpenAI voice names, which Gemini voice \
             models reject with exactly this message), or a model id in the \
             episode profile that doesn't exist for its provider. If the \
             transcript finished and the failure came during audio, it is the \
             voice.",
        );
    }

    if error_msg.contains("refused the request") {
        return Some(
            "The provider's safety system refused this request on every \
             attempt. This is often a false positive on long source text and \
             is intermittent, so retrying the episode can work. To make it \
             automatic, set PODCAST_REFUSAL_FALLBACK_MODEL to another model \
             from the same provider (for example claude-haiku-5-5). Using \
             fewer or shorter sources also lowers the chance.",
        );
    }

    if error_msg.contains("Invalid json output") || error_msg.contains("Expecting value") {
        return Some(
            "The model's response could not be parsed as JSON. Two common \
             causes: (1) the response was truncated - podcast-creator requests \
             up to 8192 output tokens per outline or transcript step unless the \
             episode profile sets max_tokens; if the model stops earlier (a \
             lower output limit, or a token-expensive language), set max_tokens \
             to its limit or use fewer and shorter segments; (2) a model using \
             extended thinking (e.g. \
             GPT-5) put all of its output inside <think> tags, leaving nothing \
             to parse - try gpt-4o, gpt-4o-mini or gpt-4-turbo instead.",
        );
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn placeholder_speaker_name_is_explained() {
        let hint = explain_generation_failure(
            "Failed to parse ValidatedTranscript from completion \
             {\"transcript\": [{\"speaker\": \"...\", \"dialogue\": \"...\"}]}. Got: \
             1 validation error for ValidatedTranscript transcript.0.speaker \
             Value error, Invalid speaker name '...'. Must be one of: \
             Marcus Thompson, Elena Vasquez",
        )
        .unwrap();
        assert!(hint.contains("speaker"));
        assert!(!hint.contains("gpt-4o"));
    }

    #[test]
    fn generic_not_found_names_both_candidates() {
        let hint =
            explain_generation_failure("Google API error: Requested entity was not found.").unwrap();
        assert!(hint.contains("voice_id"));
        assert!(hint.contains("model id"));
    }

    #[test]
    fn unsupported_voice_name_points_at_the_speaker_profile() {
        let hint = explain_generation_failure(
            "Google API error: Voice name echo is not supported. Allowed voice \
             names are: achernar, achird, algenib",
        )
        .unwrap();
        assert!(hint.contains("voice_id"));
        assert!(hint.contains("Podcasts -> Profiles"));
    }

    #[test]
    fn unparseable_output_mentions_truncation_and_thinking() {
        for message in [
            "Invalid json output: {'transcript': [{'speaker'",
            "Expecting value: line 1 column 1 (char 0)",
        ] {
            let hint = explain_generation_failure(message).unwrap();
            assert!(hint.contains("max_tokens"));
            assert!(hint.contains("<think>"));
        }
    }

    #[test]
    fn refusal_gets_an_actionable_hint() {
        let hint = explain_generation_failure(
            "anthropic/claude-sonnet-5-5 refused the request (reasoning_extraction)",
        )
        .unwrap();
        assert!(hint.contains("PODCAST_REFUSAL_FALLBACK_MODEL"));
    }

    #[test]
    fn unrecognised_failure_gets_no_hint() {
        assert!(explain_generation_failure("Connection reset by peer").is_none());
    }

    #[test]
    fn message_with_hint_appends_a_note_only_when_one_applies() {
        let hinted = PodcastError::InvalidOutput(
            "Failed to parse transcript. Got: Invalid speaker name '...'.".into(),
        );
        let message = hinted.message_with_hint();
        assert!(message.contains("Invalid speaker name"));
        assert!(message.contains("\n\nNOTE: "));
        assert!(message.contains("Speaker names must match the profile exactly"));

        let plain = PodcastError::InvalidInput("Episode profile 'x' not found".into());
        assert_eq!(plain.message_with_hint(), "Episode profile 'x' not found");
    }

    #[test]
    fn permanence_follows_the_original_classification() {
        assert!(PodcastError::InvalidInput("x".into()).is_permanent());
        assert!(PodcastError::Refusal("x".into()).is_permanent());
        assert!(!PodcastError::Ai(AiError::Timeout("x".into())).is_permanent());
        assert!(PodcastError::Ai(AiError::Authentication("x".into())).is_permanent());
        assert!(!PodcastError::Audio("x".into()).is_permanent());
    }
}
