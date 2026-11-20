//! Retry policy for model calls (outline, transcript and speech).
//!
//! Same shape as podcast-creator's: a fixed number of attempts with
//! exponential backoff, applied only to failures that may be transient.

use std::fmt::Display;
use std::future::Future;
use std::time::Duration;

/// How often and how patiently a model call is repeated.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RetryConfig {
    /// Total attempts, including the first. `1` disables retrying.
    pub max_attempts: u32,
    /// Wait before the second attempt; doubles for each later one.
    pub wait_multiplier: Duration,
    /// Upper bound of a single wait.
    pub wait_max: Duration,
}

impl Default for RetryConfig {
    /// podcast-creator's defaults: 3 attempts, 5 s then 10 s, capped at 30 s.
    fn default() -> Self {
        Self {
            max_attempts: 3,
            wait_multiplier: Duration::from_secs(5),
            wait_max: Duration::from_secs(30),
        }
    }
}

impl RetryConfig {
    /// A single attempt, no waiting.
    pub fn none() -> Self {
        Self { max_attempts: 1, wait_multiplier: Duration::ZERO, wait_max: Duration::ZERO }
    }

    /// The configured number of attempts with no wait between them.
    pub fn immediate(max_attempts: u32) -> Self {
        Self { max_attempts, wait_multiplier: Duration::ZERO, wait_max: Duration::ZERO }
    }

    /// Wait after failed attempt number `attempt` (1-based).
    pub fn wait_after(&self, attempt: u32) -> Duration {
        let factor = 2u32.saturating_pow(attempt.saturating_sub(1));
        self.wait_multiplier.saturating_mul(factor).min(self.wait_max)
    }
}

/// Run `call` until it succeeds, fails with an error `retryable` rejects, or
/// the attempts run out.
pub(crate) async fn retry<T, E, F, Fut>(
    config: &RetryConfig,
    what: &str,
    retryable: impl Fn(&E) -> bool,
    mut call: F,
) -> std::result::Result<T, E>
where
    E: Display,
    F: FnMut() -> Fut,
    Fut: Future<Output = std::result::Result<T, E>>,
{
    let attempts = config.max_attempts.max(1);
    let mut attempt = 1;
    loop {
        match call().await {
            Ok(value) => return Ok(value),
            Err(error) if attempt < attempts && retryable(&error) => {
                let wait = config.wait_after(attempt);
                tracing::warn!(
                    "{what}: attempt {attempt}/{attempts} failed with {error}; \
                     waiting {:.1}s before the next attempt",
                    wait.as_secs_f64()
                );
                if !wait.is_zero() {
                    tokio::time::sleep(wait).await;
                }
                attempt += 1;
            }
            Err(error) => return Err(error),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicU32, Ordering};

    use nn_llm::AiError;

    use super::*;

    #[test]
    fn waits_double_and_are_capped() {
        let config = RetryConfig::default();
        assert_eq!(config.wait_after(1), Duration::from_secs(5));
        assert_eq!(config.wait_after(2), Duration::from_secs(10));
        assert_eq!(config.wait_after(3), Duration::from_secs(20));
        assert_eq!(config.wait_after(4), Duration::from_secs(30));
        assert_eq!(config.wait_after(40), Duration::from_secs(30));
    }

    #[tokio::test]
    async fn transient_errors_are_retried_until_success() {
        let calls = AtomicU32::new(0);
        let result = retry(&RetryConfig::immediate(3), "test", AiError::is_retryable, || async {
            if calls.fetch_add(1, Ordering::SeqCst) < 2 {
                Err(AiError::RateLimit("slow down".into()))
            } else {
                Ok(7)
            }
        })
        .await;
        assert_eq!(result.unwrap(), 7);
        assert_eq!(calls.load(Ordering::SeqCst), 3);
    }

    #[tokio::test]
    async fn permanent_errors_are_not_retried() {
        let calls = AtomicU32::new(0);
        let result: std::result::Result<(), _> =
            retry(&RetryConfig::immediate(3), "test", AiError::is_retryable, || async {
                calls.fetch_add(1, Ordering::SeqCst);
                Err(AiError::Authentication("bad key".into()))
            })
            .await;
        assert!(matches!(result, Err(AiError::Authentication(_))));
        assert_eq!(calls.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn attempts_run_out() {
        let calls = AtomicU32::new(0);
        let result: std::result::Result<(), _> =
            retry(&RetryConfig::immediate(2), "test", AiError::is_retryable, || async {
                calls.fetch_add(1, Ordering::SeqCst);
                Err(AiError::Timeout("slow".into()))
            })
            .await;
        assert!(matches!(result, Err(AiError::Timeout(_))));
        assert_eq!(calls.load(Ordering::SeqCst), 2);
    }
}
