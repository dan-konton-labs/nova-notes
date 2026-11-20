//! Retry, then fall back, when a provider refuses an outline or transcript call.
//!
//! Anthropic can answer a long request with a refusal and no content (a
//! classifier false positive, and intermittent). Parsing that empty reply as
//! JSON would fail with a message that hides the cause, so every outline and
//! transcript call goes through [`guarded_chat`], which per call:
//!
//! 1. repeats the same request [`RefusalPolicy::retries`] more times;
//! 2. then tries [`RefusalPolicy::fallback`], if one is set and it is a
//!    different model;
//! 3. otherwise fails with [`PodcastError::Refusal`], which is permanent.
//!
//! The original read the retry count and the fallback model name from the
//! environment (`PODCAST_REFUSAL_RETRIES`, `PODCAST_REFUSAL_FALLBACK_MODEL`).
//! This crate reads no environment: the caller fills in the policy.

use std::fmt;

use nn_llm::{ChatMessage, ChatOptions, ChatResponse, LanguageModel, SharedLanguageModel};

use crate::error::{PodcastError, Result};

/// Extra attempts after a refusal when the caller sets nothing else.
pub const DEFAULT_REFUSAL_RETRIES: u32 = 2;

/// What to do when the provider refuses.
#[derive(Clone)]
pub struct RefusalPolicy {
    /// Extra attempts with the same model. `0` disables retrying.
    pub retries: u32,
    /// A model to try once after the retries were refused too. Skipped when it
    /// has the same model name as the refusing model.
    pub fallback: Option<SharedLanguageModel>,
}

impl Default for RefusalPolicy {
    fn default() -> Self {
        Self { retries: DEFAULT_REFUSAL_RETRIES, fallback: None }
    }
}

impl fmt::Debug for RefusalPolicy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RefusalPolicy")
            .field("retries", &self.retries)
            .field("fallback", &self.fallback.as_ref().map(|m| model_key(m.as_ref())))
            .finish()
    }
}

/// `provider/model`, as used in log lines and refusal errors.
pub fn model_key(model: &dyn LanguageModel) -> String {
    format!("{}/{}", model.provider(), model.model_name())
}

/// A short reason when the reply is a refusal, else `None`.
///
/// Keys off the normalised finish reason: `refusal` (Anthropic's
/// `stop_reason`) and `content_filter` (OpenAI-compatible providers).
pub fn refusal_reason(response: &ChatResponse) -> Option<&'static str> {
    match response.finish_reason.as_deref().map(str::trim) {
        Some(reason) if reason.eq_ignore_ascii_case("refusal") => Some("refusal"),
        Some(reason) if reason.eq_ignore_ascii_case("content_filter") => Some("content_filter"),
        _ => None,
    }
}

/// One chat call with the refusal policy applied. Errors of the call itself
/// are returned unchanged for the caller's own retry handling.
pub async fn guarded_chat(
    model: &dyn LanguageModel,
    messages: &[ChatMessage],
    options: &ChatOptions,
    policy: &RefusalPolicy,
) -> Result<ChatResponse> {
    let key = model_key(model);
    let attempts = policy.retries + 1;
    let mut reason = "refusal";
    for attempt in 1..=attempts {
        let response = model.chat(messages, options).await?;
        match refusal_reason(&response) {
            None => return Ok(response),
            Some(found) => {
                reason = found;
                tracing::warn!("{key} refused the request ({reason}), attempt {attempt}/{attempts}");
            }
        }
    }

    if let Some(fallback) = policy.fallback.as_deref() {
        if fallback.model_name() != model.model_name() {
            let fallback_key = model_key(fallback);
            tracing::warn!("Falling back to {fallback_key}");
            let response = fallback.chat(messages, options).await?;
            match refusal_reason(&response) {
                None => return Ok(response),
                Some(found) => {
                    reason = found;
                    tracing::warn!("{fallback_key} refused too ({reason})");
                }
            }
        }
    }

    Err(PodcastError::Refusal(format!("{key} refused the request ({reason}) on every attempt")))
}

#[cfg(test)]
mod tests {
    use std::collections::VecDeque;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{Arc, Mutex};

    use async_trait::async_trait;
    use nn_llm::{AiError, AiResult, TextStream};

    use super::*;

    struct FakeModel {
        name: &'static str,
        replies: Mutex<VecDeque<ChatResponse>>,
        calls: AtomicUsize,
    }

    impl FakeModel {
        fn new(name: &'static str, replies: Vec<ChatResponse>) -> Arc<Self> {
            Arc::new(Self { name, replies: Mutex::new(replies.into()), calls: AtomicUsize::new(0) })
        }

        fn calls(&self) -> usize {
            self.calls.load(Ordering::SeqCst)
        }
    }

    #[async_trait]
    impl LanguageModel for FakeModel {
        fn provider(&self) -> &str {
            "anthropic"
        }
        fn model_name(&self) -> &str {
            self.name
        }
        async fn chat(&self, _: &[ChatMessage], _: &ChatOptions) -> AiResult<ChatResponse> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            self.replies
                .lock()
                .unwrap()
                .pop_front()
                .ok_or_else(|| AiError::Other("no reply scripted".into()))
        }
        async fn chat_stream(&self, _: &[ChatMessage], _: &ChatOptions) -> AiResult<TextStream> {
            Err(AiError::Unsupported("not used".into()))
        }
    }

    fn response(content: &str, finish_reason: &str) -> ChatResponse {
        ChatResponse {
            content: content.to_string(),
            model: "m".into(),
            finish_reason: Some(finish_reason.to_string()),
            usage: None,
        }
    }

    fn ok() -> ChatResponse {
        response("{\"ok\": true}", "stop")
    }

    fn refused() -> ChatResponse {
        response("", "refusal")
    }

    async fn run(model: &FakeModel, policy: &RefusalPolicy) -> Result<ChatResponse> {
        guarded_chat(model, &[ChatMessage::user("p")], &ChatOptions::default(), policy).await
    }

    #[test]
    fn refusal_reason_detects_anthropic_refusal() {
        assert_eq!(refusal_reason(&refused()), Some("refusal"));
        assert_eq!(refusal_reason(&response("", "REFUSAL")), Some("refusal"));
        assert_eq!(refusal_reason(&ok()), None);
        assert_eq!(refusal_reason(&response("x", "length")), None);
        let no_reason =
            ChatResponse { content: "x".into(), model: "m".into(), finish_reason: None, usage: None };
        assert_eq!(refusal_reason(&no_reason), None);
    }

    #[test]
    fn refusal_reason_detects_openai_content_filter() {
        assert_eq!(refusal_reason(&response("", "content_filter")), Some("content_filter"));
    }

    #[tokio::test]
    async fn normal_reply_is_returned_after_one_call() {
        let model = FakeModel::new("claude-sonnet-5-5", vec![ok()]);
        assert_eq!(run(&model, &RefusalPolicy::default()).await.unwrap(), ok());
        assert_eq!(model.calls(), 1);
    }

    #[tokio::test]
    async fn refusal_is_retried_then_succeeds() {
        let model = FakeModel::new("claude-sonnet-5-5", vec![refused(), refused(), ok()]);
        assert_eq!(run(&model, &RefusalPolicy::default()).await.unwrap(), ok());
        assert_eq!(model.calls(), 3);
    }

    #[tokio::test]
    async fn exhausted_retries_raise_clear_error() {
        let model = FakeModel::new("claude-sonnet-5-5", vec![refused(), refused(), refused()]);
        let error = run(&model, &RefusalPolicy::default()).await.unwrap_err();
        assert_eq!(model.calls(), 3);
        assert!(matches!(error, PodcastError::Refusal(_)));
        assert!(error.is_permanent());
        assert_eq!(
            error.to_string(),
            "anthropic/claude-sonnet-5-5 refused the request (refusal) on every attempt"
        );
        assert!(error.hint().unwrap().contains("PODCAST_REFUSAL_FALLBACK_MODEL"));
    }

    #[tokio::test]
    async fn retry_count_comes_from_the_policy() {
        let model = FakeModel::new("claude-sonnet-5-5", vec![refused()]);
        let policy = RefusalPolicy { retries: 0, fallback: None };
        assert!(run(&model, &policy).await.is_err());
        assert_eq!(model.calls(), 1);
    }

    #[tokio::test]
    async fn fallback_model_used_after_retries() {
        let model = FakeModel::new("claude-sonnet-5-5", vec![refused(), refused(), refused()]);
        let fallback = FakeModel::new("claude-haiku-5-5", vec![ok()]);
        let policy = RefusalPolicy { retries: 2, fallback: Some(fallback.clone()) };
        assert_eq!(run(&model, &policy).await.unwrap(), ok());
        assert_eq!(model.calls(), 3);
        assert_eq!(fallback.calls(), 1);
    }

    #[tokio::test]
    async fn fallback_also_refused_raises() {
        let model = FakeModel::new("claude-sonnet-5-5", vec![refused(), refused(), refused()]);
        let fallback = FakeModel::new("claude-haiku-5-5", vec![response("", "content_filter")]);
        let policy = RefusalPolicy { retries: 2, fallback: Some(fallback.clone()) };
        let error = run(&model, &policy).await.unwrap_err();
        assert_eq!(
            error.to_string(),
            "anthropic/claude-sonnet-5-5 refused the request (content_filter) on every attempt"
        );
        assert_eq!(fallback.calls(), 1);
    }

    #[tokio::test]
    async fn fallback_same_as_primary_is_skipped() {
        let model = FakeModel::new("claude-sonnet-5-5", vec![refused(), refused(), refused()]);
        let fallback = FakeModel::new("claude-sonnet-5-5", vec![ok()]);
        let policy = RefusalPolicy { retries: 2, fallback: Some(fallback.clone()) };
        assert!(matches!(run(&model, &policy).await, Err(PodcastError::Refusal(_))));
        assert_eq!(fallback.calls(), 0);
    }

    #[tokio::test]
    async fn call_errors_pass_through_unchanged() {
        let model = FakeModel::new("claude-sonnet-5-5", vec![]);
        let error = run(&model, &RefusalPolicy::default()).await.unwrap_err();
        assert!(matches!(error, PodcastError::Ai(AiError::Other(_))));
        assert_eq!(model.calls(), 1);
    }
}
