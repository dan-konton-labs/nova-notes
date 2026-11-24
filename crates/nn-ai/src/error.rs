//! Mapping provider failures onto [`AiError`].
//!
//! Three jobs live here:
//!
//! * [`http_error`] turns an HTTP status plus response body into the right
//!   variant, using the provider's own message from its JSON error body.
//! * [`transport_error`] does the same for `reqwest` failures.
//! * [`classify_error_text`] and [`user_message`] port
//!   `open_notebook/utils/error_classifier.py`: classifying a raw error string
//!   and producing the text shown to users.
//!
//! Messages built here never contain the API key: keys travel in headers only,
//! URLs are stripped from transport errors, and anything a provider echoes
//! back is passed through [`redact`].

use nn_llm::AiError;
use serde_json::Value;

const MAX_MESSAGE_CHARS: usize = 1000;

/// Cuts `text` to at most `max_chars` characters, appending `...` when cut.
pub(crate) fn truncate(text: &str, max_chars: usize) -> String {
    match text.char_indices().nth(max_chars) {
        Some((idx, _)) => format!("{}...", &text[..idx]),
        None => text.to_string(),
    }
}

/// Replaces every occurrence of `secret` in `text`.
pub(crate) fn redact(text: &str, secret: Option<&str>) -> String {
    match secret.map(str::trim) {
        Some(secret) if secret.len() >= 4 && text.contains(secret) => text.replace(secret, "[redacted]"),
        _ => text.to_string(),
    }
}

/// The interesting parts of a provider error body.
#[derive(Debug, Default, PartialEq, Eq)]
pub(crate) struct ErrorBody {
    /// Human-readable message, empty when the body had none.
    pub message: String,
    /// Machine codes (`error.type`, `error.code`, `error.status`...) joined by
    /// spaces; used for classification only.
    pub codes: String,
}

fn scalar(value: &Value) -> Option<String> {
    match value {
        Value::String(s) if !s.trim().is_empty() => Some(s.trim().to_string()),
        Value::Number(n) => Some(n.to_string()),
        _ => None,
    }
}

/// Pulls the message and codes out of the JSON error shapes used by the
/// supported providers, falling back to the raw text.
pub(crate) fn parse_error_body(body: &str) -> ErrorBody {
    let trimmed = body.trim();
    let Ok(json) = serde_json::from_str::<Value>(trimmed) else {
        // A body that is not a JSON error object is never passed on: it may
        // be any page the configured URL serves (an internal service, an HTML
        // error page). The caller logs it and reports the status only; the
        // text still feeds classification ("context length exceeded").
        let codes = if trimmed.starts_with('<') { String::new() } else { truncate(trimmed, MAX_MESSAGE_CHARS) };
        return ErrorBody { message: String::new(), codes };
    };
    // Gemini streaming endpoints wrap the error object in an array.
    let json = match json {
        Value::Array(items) => items.into_iter().next().unwrap_or(Value::Null),
        other => other,
    };

    let mut message = None;
    let mut codes = Vec::new();
    let mut note_codes = |obj: &Value| {
        for key in ["type", "code", "status", "err_code", "param"] {
            if let Some(code) = obj.get(key).and_then(scalar) {
                codes.push(code);
            }
        }
    };

    match json.get("error") {
        // OpenAI, Anthropic, Google, Groq, xAI, OpenRouter...
        Some(err @ Value::Object(_)) => {
            message = err.get("message").and_then(scalar);
            note_codes(err);
        }
        // Ollama and several OpenAI-compatible servers.
        Some(Value::String(s)) => message = Some(s.clone()),
        _ => {}
    }
    note_codes(&json);

    if message.is_none() {
        message = match json.get("detail") {
            // ElevenLabs: {"detail": {"status": "...", "message": "..."}}
            Some(detail @ Value::Object(_)) => {
                note_codes(detail);
                detail.get("message").and_then(scalar)
            }
            // FastAPI validation errors: {"detail": [{"msg": "..."}]}
            Some(Value::Array(items)) => {
                let parts: Vec<String> =
                    items.iter().filter_map(|i| i.get("msg").and_then(scalar)).collect();
                (!parts.is_empty()).then(|| parts.join("; "))
            }
            Some(other) => scalar(other),
            None => None,
        };
    }
    if message.is_none() {
        // Cohere and Mistral use "message", Deepgram "err_msg", MiniMax
        // "base_resp.status_msg".
        message = ["message", "err_msg", "reason", "error_description", "msg"]
            .iter()
            .find_map(|k| json.get(*k).and_then(scalar))
            .or_else(|| json.pointer("/base_resp/status_msg").and_then(scalar));
    }

    ErrorBody {
        message: truncate(&message.unwrap_or_default(), MAX_MESSAGE_CHARS),
        codes: codes.join(" "),
    }
}

/// Port of the context-length substring checks in `error_classifier.py`, plus
/// the phrasings Anthropic and Gemini use for the same condition.
pub(crate) fn is_context_length(lower: &str) -> bool {
    // "Unsupported parameter: 'max_tokens'..." is a malformed request, not an
    // oversized one.
    let unsupported_param =
        lower.contains("unsupported parameter") || lower.contains("not supported with this model");
    [
        "context length",
        "token limit",
        "maximum context",
        "context_length_exceeded",
        "context window",
        "prompt is too long",
        "input is too long",
        "too many tokens",
        "exceeds the maximum number of tokens",
    ]
    .iter()
    .any(|k| lower.contains(k))
        || (lower.contains("max_tokens") && !unsupported_param)
}

fn is_model_not_found(status: u16, lower: &str) -> bool {
    if lower.contains("model_not_found") || lower.contains("model not found") {
        return true;
    }
    let names_model = lower.contains("model") || lower.contains("deployment");
    let missing = lower.contains("not found")
        || lower.contains("not_found")
        || lower.contains("does not exist")
        || lower.contains("no such")
        || lower.contains("unknown model")
        || lower.contains("invalid model");
    names_model && (status == 404 || (missing && matches!(status, 400 | 422)))
}

fn status_reason(status: u16) -> &'static str {
    match status {
        400 => "bad request",
        401 => "unauthorized",
        403 => "forbidden",
        404 => "not found",
        408 => "request timeout",
        413 => "payload too large",
        422 => "unprocessable request",
        429 => "too many requests",
        500 => "internal server error",
        502 => "bad gateway",
        503 => "service unavailable",
        504 => "gateway timeout",
        529 => "overloaded",
        _ => "request failed",
    }
}

/// Classifies a non-success HTTP response.
///
/// The message keeps the provider's own wording and ends in `(HTTP <status>)`
/// so [`http_status`] can recover the status from any variant.
pub(crate) fn http_error(provider: &str, status: u16, body: &str, secret: Option<&str>) -> AiError {
    let parsed = parse_error_body(body);
    if parsed.message.is_empty() && !body.trim().is_empty() {
        tracing::debug!(
            provider,
            status,
            body = %redact(&truncate(body.trim(), MAX_MESSAGE_CHARS), secret),
            "provider error response without a JSON message"
        );
    }
    let detail = if parsed.message.is_empty() { status_reason(status).to_string() } else { parsed.message };
    let detail = redact(&detail, secret);
    let message = format!("{provider}: {detail} (HTTP {status})");
    let haystack = format!("{} {}", detail, parsed.codes).to_lowercase();

    match status {
        401 | 403 => AiError::Authentication(message),
        429 => AiError::RateLimit(message),
        408 => AiError::Timeout(message),
        _ if is_context_length(&haystack) => AiError::ContextLength(message),
        _ if is_model_not_found(status, &haystack) => AiError::ModelNotFound(message),
        400 | 422 => AiError::InvalidRequest(message),
        _ => AiError::Provider { status: Some(status), message },
    }
}

/// Classifies a `reqwest` failure. The URL is dropped from the message, so
/// nothing that was part of the request can leak.
pub(crate) fn transport_error(provider: &str, err: reqwest::Error) -> AiError {
    let is_connect = err.is_connect();
    let is_timeout = err.is_timeout();
    let err = err.without_url();
    let mut detail = err.to_string();
    let mut source = std::error::Error::source(&err);
    while let Some(cause) = source {
        let text = cause.to_string();
        if !text.is_empty() && !detail.contains(&text) {
            detail.push_str(": ");
            detail.push_str(&text);
        }
        source = cause.source();
    }
    let detail = truncate(&detail, 300);
    if is_connect {
        // Includes connect timeouts: the provider was never reached.
        AiError::Network(format!("{provider}: could not connect ({detail})"))
    } else if is_timeout {
        AiError::Timeout(format!("{provider}: request timed out ({detail})"))
    } else {
        AiError::Network(format!("{provider}: connection error ({detail})"))
    }
}

/// The HTTP status behind an error produced by this crate, when there is one.
pub fn http_status(err: &AiError) -> Option<u16> {
    if let AiError::Provider { status: Some(status), .. } = err {
        return Some(*status);
    }
    let text = err.to_string();
    let rest = &text[text.rfind("(HTTP ")? + 6..];
    rest.strip_suffix(')')?.parse().ok()
}

const AUTH_MESSAGE: &str = "Authentication failed. Please check your API key in Manage -> Models.";
const RATE_LIMIT_MESSAGE: &str = "Rate limit exceeded. Please wait a moment and try again.";
const NETWORK_MESSAGE: &str =
    "Could not connect to the AI provider. Please check your network connection and provider URL.";
const TIMEOUT_MESSAGE: &str = "The AI provider took too long to respond. Try again, use a faster model, or raise the model call timeout (180 seconds by default).";
const CONTEXT_MESSAGE: &str = "Content too large for the selected model. Try using a smaller selection or a model with a larger context window.";
const PAYLOAD_MESSAGE: &str = "The request payload is too large for the AI provider. Try reducing the content size or using a different model.";
const UNAVAILABLE_MESSAGE: &str =
    "The AI provider is temporarily unavailable. Please try again in a few minutes.";

/// Classifies a raw error string the way `classify_error()` does in the
/// original: ordered keyword rules over the lowercased text. Use it for errors
/// that did not come out of this crate already typed.
pub fn classify_error_text(text: &str) -> AiError {
    let lower = text.to_lowercase();
    let has = |keywords: &[&str]| keywords.iter().any(|k| lower.contains(k));
    let owned = || text.to_string();

    if has(&["authentication", "unauthorized", "invalid api key", "invalid_api_key", "401"]) {
        AiError::Authentication(owned())
    } else if has(&["rate limit", "rate_limit", "429", "too many requests", "quota exceeded"]) {
        AiError::RateLimit(owned())
    } else if has(&["model not found", "does not exist", "model_not_found"]) {
        AiError::ModelNotFound(owned())
    } else if has(&["no model configured", "please go to settings", "please go to manage"]) {
        AiError::Configuration(owned())
    } else if has(&[
        "connecterror",
        "connecttimeout",
        "connection timed out",
        "connection refused",
        "connection error",
    ]) {
        AiError::Network(owned())
    } else if has(&["timeoutexception", "timed out", "timeout"]) {
        AiError::Timeout(owned())
    } else if has(&["context length", "token limit", "maximum context", "context_length_exceeded", "max_tokens"]) {
        AiError::ContextLength(owned())
    } else if has(&["413", "payload too large", "request entity too large"]) {
        AiError::Provider { status: Some(413), message: owned() }
    } else if has(&["500", "502", "503", "service unavailable", "overloaded", "internal server error"]) {
        let status = [500u16, 502, 503].into_iter().find(|s| lower.contains(&s.to_string())).unwrap_or(503);
        AiError::Provider { status: Some(status), message: owned() }
    } else {
        AiError::Other(owned())
    }
}

/// The message to show a user for `err`, ported from `error_classifier.py`.
///
/// Authentication, rate-limit, network, timeout, context-length and
/// availability errors get fixed friendly wording; model-not-found and
/// configuration errors pass through (truncated to 200 characters); anything
/// else becomes `AI service error: ...`.
pub fn user_message(err: &AiError) -> String {
    match err {
        AiError::Authentication(_) => AUTH_MESSAGE.to_string(),
        AiError::RateLimit(_) => RATE_LIMIT_MESSAGE.to_string(),
        AiError::Network(_) => NETWORK_MESSAGE.to_string(),
        AiError::Timeout(_) => TIMEOUT_MESSAGE.to_string(),
        AiError::ContextLength(_) => CONTEXT_MESSAGE.to_string(),
        AiError::ModelNotFound(m)
        | AiError::Configuration(m)
        | AiError::Unsupported(m)
        | AiError::Incomplete(m) => truncate(m, 200),
        AiError::Provider { status: Some(413), .. } => PAYLOAD_MESSAGE.to_string(),
        AiError::Provider { status: Some(status), .. } if *status >= 500 => UNAVAILABLE_MESSAGE.to_string(),
        AiError::Provider { message, .. } if message.to_lowercase().contains("overloaded") => {
            UNAVAILABLE_MESSAGE.to_string()
        }
        AiError::Provider { message: m, .. } | AiError::InvalidRequest(m) | AiError::Other(m) => {
            format!("AI service error: {}", truncate(m, 200))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn classify(status: u16, body: &str) -> AiError {
        http_error("openai", status, body, None)
    }

    #[test]
    fn extracts_messages_from_every_known_body_shape() {
        let cases = [
            (r#"{"error":{"message":"Incorrect API key","type":"invalid_request_error","code":"invalid_api_key"}}"#, "Incorrect API key"),
            (r#"{"type":"error","error":{"type":"not_found_error","message":"model: claude-x"}}"#, "model: claude-x"),
            (r#"{"error":{"code":404,"message":"models/x is not found","status":"NOT_FOUND"}}"#, "models/x is not found"),
            (r#"[{"error":{"code":400,"message":"bad","status":"INVALID_ARGUMENT"}}]"#, "bad"),
            (r#"{"error":"model 'llama9' not found"}"#, "model 'llama9' not found"),
            (r#"{"message":"invalid api token"}"#, "invalid api token"),
            (r#"{"detail":{"status":"invalid_api_key","message":"Invalid API key"}}"#, "Invalid API key"),
            (r#"{"detail":"Not authenticated"}"#, "Not authenticated"),
            (r#"{"detail":[{"loc":["body"],"msg":"field required"},{"msg":"bad value"}]}"#, "field required; bad value"),
            (r#"{"err_code":"INVALID_AUTH","err_msg":"Invalid credentials."}"#, "Invalid credentials."),
            (r#"{"base_resp":{"status_code":1004,"status_msg":"login fail"}}"#, "login fail"),
            // Plain text is never reflected (it may be any page the URL serves).
            ("upstream connect error", ""),
            ("<html><body>502 Bad Gateway</body></html>", ""),
        ];
        for (body, expected) in cases {
            assert_eq!(parse_error_body(body).message, expected, "{body}");
        }
        let parsed = parse_error_body(r#"{"error":{"message":"m","type":"t","code":"c"}}"#);
        assert_eq!(parsed.codes, "t c");
    }

    #[test]
    fn non_json_bodies_are_not_reflected_but_still_classify() {
        let err = classify(500, "root:x:0:0:internal-service-secret");
        assert_eq!(err.to_string(), "openai: internal server error (HTTP 500)");
        assert!(matches!(classify(400, "input exceeds the token limit"), AiError::ContextLength(_)));
        assert_eq!(classify(400, "input exceeds the token limit").to_string(), "openai: bad request (HTTP 400)");
    }

    #[test]
    fn auth_and_rate_limit_follow_the_status() {
        assert!(matches!(classify(401, r#"{"error":{"message":"bad key"}}"#), AiError::Authentication(_)));
        assert!(matches!(classify(403, "forbidden"), AiError::Authentication(_)));
        assert!(matches!(classify(429, r#"{"error":{"message":"slow down"}}"#), AiError::RateLimit(_)));
    }

    #[test]
    fn model_not_found_needs_model_wording() {
        let openai = r#"{"error":{"message":"The model `gpt-9` does not exist or you do not have access to it.","type":"invalid_request_error","code":"model_not_found"}}"#;
        assert!(matches!(classify(404, openai), AiError::ModelNotFound(_)));
        let anthropic = r#"{"type":"error","error":{"type":"not_found_error","message":"model: claude-3-haiku-20240307"}}"#;
        assert!(matches!(classify(404, anthropic), AiError::ModelNotFound(_)));
        let azure = r#"{"error":{"code":"DeploymentNotFound","message":"The API deployment for this resource does not exist."}}"#;
        assert!(matches!(classify(404, azure), AiError::ModelNotFound(_)));
        let ollama = r#"{"error":"model 'llama9' not found"}"#;
        assert!(matches!(classify(404, ollama), AiError::ModelNotFound(_)));
        let mistral = r#"{"message":"Invalid model: nope","type":"invalid_model"}"#;
        assert!(matches!(classify(400, mistral), AiError::ModelNotFound(_)));
        // A wrong base URL also gives 404, but says nothing about a model.
        assert!(matches!(classify(404, "<html>Not Found</html>"), AiError::Provider { status: Some(404), .. }));
        assert!(matches!(classify(404, r#"{"detail":"Not Found"}"#), AiError::Provider { status: Some(404), .. }));
    }

    #[test]
    fn context_window_messages_are_context_length() {
        let bodies = [
            r#"{"error":{"message":"This model's maximum context length is 8192 tokens. However, your messages resulted in 9000 tokens.","code":"context_length_exceeded"}}"#,
            r#"{"error":{"message":"Request too big","code":"context_length_exceeded"}}"#,
            r#"{"type":"error","error":{"type":"invalid_request_error","message":"prompt is too long: 213462 tokens > 200000 maximum"}}"#,
            r#"{"error":{"code":400,"message":"The input token count (2000000) exceeds the maximum number of tokens allowed (1048576).","status":"INVALID_ARGUMENT"}}"#,
            r#"{"error":{"message":"max_tokens is too large: 99999. This model supports at most 4096 completion tokens"}}"#,
            r#"{"error":"input exceeds the token limit"}"#,
        ];
        for body in bodies {
            assert!(matches!(classify(400, body), AiError::ContextLength(_)), "{body}");
        }
    }

    #[test]
    fn unsupported_max_tokens_parameter_is_an_invalid_request() {
        let body = r#"{"error":{"message":"Unsupported parameter: 'max_tokens' is not supported with this model. Use 'max_completion_tokens' instead.","type":"invalid_request_error","param":"max_tokens","code":"unsupported_parameter"}}"#;
        assert!(matches!(classify(400, body), AiError::InvalidRequest(_)));
    }

    #[test]
    fn everything_else_keeps_status_and_provider_message() {
        let err = classify(503, r#"{"error":{"message":"The engine is currently overloaded"}}"#);
        match &err {
            AiError::Provider { status, message } => {
                assert_eq!(*status, Some(503));
                assert_eq!(message, "openai: The engine is currently overloaded (HTTP 503)");
            }
            other => panic!("unexpected {other:?}"),
        }
        assert!(err.is_retryable());
        assert!(matches!(classify(400, r#"{"error":{"message":"bad json"}}"#), AiError::InvalidRequest(_)));
        assert!(matches!(classify(422, "{}"), AiError::InvalidRequest(_)));
        assert!(matches!(classify(408, ""), AiError::Timeout(_)));
        match classify(500, "") {
            AiError::Provider { message, .. } => assert_eq!(message, "openai: internal server error (HTTP 500)"),
            other => panic!("unexpected {other:?}"),
        }
    }

    #[test]
    fn http_status_is_recoverable_from_any_variant() {
        assert_eq!(http_status(&classify(403, "no")), Some(403));
        assert_eq!(http_status(&classify(404, r#"{"error":"model x not found"}"#)), Some(404));
        assert_eq!(http_status(&classify(502, "")), Some(502));
        assert_eq!(http_status(&AiError::Network("down".into())), None);
    }

    #[test]
    fn echoed_api_keys_are_redacted() {
        let err = http_error(
            "openai",
            401,
            r#"{"error":{"message":"Incorrect API key provided: sk-live-123456"}}"#,
            Some("sk-live-123456"),
        );
        assert!(!err.to_string().contains("sk-live-123456"));
        assert!(err.to_string().contains("[redacted]"));
    }

    #[test]
    fn truncate_respects_character_boundaries() {
        assert_eq!(truncate("héllo", 2), "hé...");
        assert_eq!(truncate("abc", 3), "abc");
    }

    // Ported from tests/test_error_classifier.py.

    #[test]
    fn read_timeout_says_the_provider_was_slow() {
        let err = classify_error_text("ReadTimeout: The read operation timed out");
        assert!(matches!(err, AiError::Timeout(_)));
        assert!(user_message(&err).contains("took too long"));
        let err = classify_error_text("APITimeoutError: Request timed out.");
        assert!(user_message(&err).contains("took too long"));
    }

    #[test]
    fn connect_failures_are_connection_problems() {
        for raw in [
            "ConnectTimeout: timed out",
            "ConnectError: [Errno 111] Connection refused",
            "OSError: [Errno 110] Connection timed out",
        ] {
            let err = classify_error_text(raw);
            assert!(matches!(err, AiError::Network(_)), "{raw}");
            assert!(user_message(&err).contains("Could not connect"));
        }
    }

    #[test]
    fn auth_error_points_to_the_models_page() {
        let err = classify_error_text("401 invalid api key");
        assert!(matches!(err, AiError::Authentication(_)));
        assert!(user_message(&err).contains("Manage -> Models"));
    }

    #[test]
    fn missing_default_model_passes_through() {
        let raw = "No model configured for default for type=chat. Please go to Manage → Models and configure a default model for 'chat'.";
        let err = classify_error_text(raw);
        assert!(matches!(err, AiError::Configuration(_)));
        assert_eq!(user_message(&err), raw);
    }

    #[test]
    fn remaining_rules_map_to_the_original_messages() {
        assert_eq!(user_message(&classify_error_text("429 Too Many Requests")), RATE_LIMIT_MESSAGE);
        assert_eq!(user_message(&classify_error_text("context_length_exceeded")), CONTEXT_MESSAGE);
        assert_eq!(user_message(&classify_error_text("413 Request Entity Too Large")), PAYLOAD_MESSAGE);
        assert_eq!(user_message(&classify_error_text("503 Service Unavailable")), UNAVAILABLE_MESSAGE);
        assert_eq!(user_message(&classify_error_text("Overloaded")), UNAVAILABLE_MESSAGE);
        let odd = classify_error_text("something odd");
        assert_eq!(user_message(&odd), "AI service error: something odd");
        let long = "x".repeat(300);
        assert_eq!(user_message(&AiError::Other(long)).len(), "AI service error: ".len() + 203);
    }
}
