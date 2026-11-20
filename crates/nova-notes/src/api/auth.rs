//! `GET /api/auth/status`, plus the helpers the password middleware and the
//! web UI's login page share.

use axum::http::{header, HeaderMap};
use axum::routing::get;
use axum::{extract::State, Json, Router};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

use crate::state::AppState;

/// Name of the cookie the web UI's login page sets.
pub const SESSION_COOKIE: &str = "nn_session";

pub fn router() -> Router<AppState> {
    Router::new().route("/auth/status", get(status))
}

async fn status(State(state): State<AppState>) -> Json<Value> {
    let auth_enabled = state.config.password.is_some();
    Json(json!({
        "auth_enabled": auth_enabled,
        "message": if auth_enabled { "Authentication is required" } else { "Authentication is disabled" },
    }))
}

/// Compare two byte strings without leaking where they differ.
pub fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    // Hash first so the comparison length does not depend on the inputs.
    let (a, b) = (Sha256::digest(a), Sha256::digest(b));
    a.iter().zip(b.iter()).fold(0u8, |acc, (x, y)| acc | (x ^ y)) == 0
}

/// The session cookie value for a given instance password. Derived, not
/// random, so sessions survive restarts and are revoked by changing the
/// password.
pub fn session_token(password: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(b"nova-notes-session-v1:");
    hasher.update(password.as_bytes());
    hex::encode(hasher.finalize())
}

/// The value of cookie `name`, if the request carries it.
pub fn cookie_value<'a>(headers: &'a HeaderMap, name: &str) -> Option<&'a str> {
    headers
        .get_all(header::COOKIE)
        .iter()
        .filter_map(|v| v.to_str().ok())
        .flat_map(|v| v.split(';'))
        .filter_map(|pair| pair.trim().split_once('='))
        .find(|(key, _)| *key == name)
        .map(|(_, value)| value)
}

pub fn has_valid_session_cookie(headers: &HeaderMap, password: &str) -> bool {
    cookie_value(headers, SESSION_COOKIE)
        .is_some_and(|value| constant_time_eq(value.as_bytes(), session_token(password).as_bytes()))
}

/// Percent-encode a value for use in a query string.
pub fn encode_query_value(value: &str) -> String {
    url::form_urlencoded::byte_serialize(value.as_bytes()).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cookies_parse() {
        let mut headers = HeaderMap::new();
        headers.insert(header::COOKIE, "a=1; nn_session=abc; b=2".parse().unwrap());
        assert_eq!(cookie_value(&headers, "nn_session"), Some("abc"));
        assert_eq!(cookie_value(&headers, "missing"), None);
    }

    #[test]
    fn session_tokens_depend_on_the_password() {
        assert_eq!(session_token("a"), session_token("a"));
        assert_ne!(session_token("a"), session_token("b"));
        assert!(constant_time_eq(b"same", b"same"));
        assert!(!constant_time_eq(b"same", b"diff"));
        assert!(!constant_time_eq(b"short", b"longer value"));
    }
}
