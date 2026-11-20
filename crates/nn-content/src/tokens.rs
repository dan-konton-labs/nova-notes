//! Token counting and cost estimation.
//!
//! Port of `open_notebook/utils/token_utils.py`. Counts use the `o200k_base`
//! encoding, the same one the Python original asked tiktoken for, so chunk
//! boundaries and context budgets line up between the two implementations.
//! The vocabulary is compiled into the binary; nothing is downloaded.

use std::sync::OnceLock;

use tiktoken_rs::CoreBPE;

/// Default price per million tokens used by [`token_cost`] in the original.
pub const DEFAULT_COST_PER_MILLION: f64 = 0.150;

fn encoder() -> Option<&'static CoreBPE> {
    static ENCODER: OnceLock<Option<CoreBPE>> = OnceLock::new();
    ENCODER
        .get_or_init(|| match tiktoken_rs::o200k_base() {
            Ok(bpe) => Some(bpe),
            Err(e) => {
                tracing::warn!("o200k_base tokenizer unavailable, falling back to word-count estimation: {e}");
                None
            }
        })
        .as_ref()
}

/// Number of `o200k_base` tokens in `text`.
///
/// Special-token spellings such as `<|endoftext|>` are counted as ordinary
/// text. If the tokenizer cannot be built the count falls back to
/// [`estimate_token_count`].
pub fn token_count(text: &str) -> usize {
    match encoder() {
        Some(bpe) => bpe.encode_ordinary(text).len(),
        None => estimate_token_count(text),
    }
}

/// Word-count estimate (`words * 1.3`, truncated) used when no tokenizer is
/// available.
pub fn estimate_token_count(text: &str) -> usize {
    (text.split_whitespace().count() as f64 * 1.3) as usize
}

/// Cost of `token_count` tokens at `cost_per_million` per million tokens.
pub fn token_cost(token_count: usize, cost_per_million: f64) -> f64 {
    cost_per_million * (token_count as f64 / 1_000_000.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn counts_with_o200k_base() {
        assert_eq!(token_count(""), 0);
        assert_eq!(token_count("hello world"), 2);
        // Known o200k_base count; guards against silently using another encoding.
        assert_eq!(token_count("tiktoken is great!"), 6);
    }

    #[test]
    fn special_token_text_is_counted_as_ordinary_text() {
        assert!(token_count("<|endoftext|>") > 1);
    }

    #[test]
    fn fallback_estimate_is_positive() {
        // 5 words * 1.3 = 6.5 -> 6
        assert_eq!(estimate_token_count("one two three four five"), 6);
        assert_eq!(estimate_token_count(""), 0);
    }

    #[test]
    fn cost_scales_per_million() {
        assert!((token_cost(1_000_000, DEFAULT_COST_PER_MILLION) - 0.150).abs() < 1e-12);
        assert!((token_cost(500_000, 2.0) - 1.0).abs() < 1e-12);
        assert_eq!(token_cost(0, 5.0), 0.0);
    }
}
