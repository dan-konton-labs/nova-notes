//! `GET /api/auth/status`, plus the helpers the password middleware and the
//! web UI's login page share.

use std::collections::HashMap;
use std::net::{IpAddr, Ipv4Addr, SocketAddr};
use std::sync::Mutex;
use std::time::{Duration, Instant};

use axum::extract::ConnectInfo;
use axum::http::{header, Extensions, HeaderMap};
use axum::routing::get;
use axum::{extract::State, Json, Router};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

use crate::config::Config;
use crate::services::login_sessions;
use crate::state::AppState;

/// Name of the cookie the web UI's login page sets over plain HTTP.
pub const SESSION_COOKIE: &str = "nn_session";
/// Name of the cookie set over HTTPS. The `__Host-` prefix makes browsers
/// refuse it unless it is `Secure`, `Path=/` and has no `Domain`, so a sibling
/// subdomain or a plain-HTTP page cannot plant or shadow it.
pub const SECURE_SESSION_COOKIE: &str = "__Host-nn_session";

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

/// The session ids the request carries, under either cookie name (`__Host-`
/// over HTTPS, the plain name over HTTP).
pub fn session_ids(headers: &HeaderMap) -> Vec<&str> {
    [SECURE_SESSION_COOKIE, SESSION_COOKIE].iter().filter_map(|name| cookie_value(headers, name)).collect()
}

/// Whether the request carries a live session made under `password`.
pub async fn has_valid_session(state: &AppState, headers: &HeaderMap, password: &str) -> bool {
    for id in session_ids(headers) {
        if login_sessions::is_valid(state, id, password).await {
            return true;
        }
    }
    false
}

// ---------------------------------------------------------------------------
// Failed sign-in throttling
// ---------------------------------------------------------------------------

/// Failures from one client address before it has to wait.
const FREE_FAILURES: u32 = 5;
/// The longest a client address is made to wait.
const MAX_LOCK: Duration = Duration::from_secs(15 * 60);
/// A client address with no failure for this long starts afresh.
const FORGET_AFTER: Duration = Duration::from_secs(60 * 60);
/// Addresses tracked at once; beyond this the oldest are dropped.
const MAX_TRACKED: usize = 10_000;
/// Failures from all addresses together within [`GLOBAL_WINDOW`] before every
/// password check waits for the window to end. Loose and short on purpose:
/// it slows a distributed guesser without locking the owner out for long.
const GLOBAL_MAX_FAILURES: u32 = 100;
const GLOBAL_WINDOW: Duration = Duration::from_secs(60);

#[derive(Debug)]
struct Strikes {
    failures: u32,
    last: Instant,
    locked_until: Option<Instant>,
}

#[derive(Debug)]
struct ThrottleState {
    per_client: HashMap<IpAddr, Strikes>,
    global_window_start: Instant,
    global_failures: u32,
}

/// In-memory backoff on failed password checks (the sign-in form and API
/// bearer tokens), per client address and across all clients.
///
/// After [`FREE_FAILURES`] failures an address waits 2 s, then 4 s, 8 s, ...
/// up to 15 minutes per further failure; a correct password clears its count.
/// While it waits, its password checks are answered 429 without being tried.
#[derive(Debug)]
pub struct LoginThrottle {
    state: Mutex<ThrottleState>,
}

impl Default for LoginThrottle {
    fn default() -> Self {
        Self {
            state: Mutex::new(ThrottleState {
                per_client: HashMap::new(),
                global_window_start: Instant::now(),
                global_failures: 0,
            }),
        }
    }
}

impl LoginThrottle {
    /// `Err(wait)` when `client` must wait before its next password check.
    pub fn check(&self, client: IpAddr) -> Result<(), Duration> {
        let now = Instant::now();
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        if now.duration_since(state.global_window_start) >= GLOBAL_WINDOW {
            state.global_window_start = now;
            state.global_failures = 0;
        } else if state.global_failures >= GLOBAL_MAX_FAILURES {
            return Err(GLOBAL_WINDOW - now.duration_since(state.global_window_start));
        }
        if let Some(strikes) = state.per_client.get(&client) {
            if now.duration_since(strikes.last) >= FORGET_AFTER {
                state.per_client.remove(&client);
            } else if let Some(until) = strikes.locked_until.filter(|until| *until > now) {
                return Err(until - now);
            }
        }
        Ok(())
    }

    /// Count a wrong password from `client`.
    pub fn record_failure(&self, client: IpAddr) {
        let now = Instant::now();
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        if now.duration_since(state.global_window_start) >= GLOBAL_WINDOW {
            state.global_window_start = now;
            state.global_failures = 0;
        }
        state.global_failures = state.global_failures.saturating_add(1);
        if state.per_client.len() >= MAX_TRACKED && !state.per_client.contains_key(&client) {
            state.per_client.retain(|_, s| now.duration_since(s.last) < FORGET_AFTER);
            if state.per_client.len() >= MAX_TRACKED {
                if let Some(oldest) = state.per_client.iter().min_by_key(|(_, s)| s.last).map(|(ip, _)| *ip) {
                    state.per_client.remove(&oldest);
                }
            }
        }
        let strikes = state.per_client.entry(client).or_insert(Strikes { failures: 0, last: now, locked_until: None });
        if now.duration_since(strikes.last) >= FORGET_AFTER {
            strikes.failures = 0;
        }
        strikes.failures = strikes.failures.saturating_add(1);
        strikes.last = now;
        if strikes.failures >= FREE_FAILURES {
            let doublings = (strikes.failures - FREE_FAILURES + 1).min(20);
            let wait = Duration::from_secs(1u64 << doublings).min(MAX_LOCK);
            strikes.locked_until = Some(now + wait);
        }
    }

    /// A correct password from `client` clears its failures.
    pub fn record_success(&self, client: IpAddr) {
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        state.per_client.remove(&client);
    }
}

/// The `detail` of a throttled password check.
pub fn throttled_detail(wait: Duration) -> String {
    let seconds = wait.as_secs() + u64::from(wait.subsec_nanos() > 0);
    format!("Too many failed sign-in attempts. Try again in {} seconds.", seconds.max(1))
}

/// The address a request came from: the connection's peer, or with
/// `OPEN_NOTEBOOK_TRUST_PROXY_HEADERS` the last `X-Forwarded-For` entry (the
/// one the proxy itself appended). Unknown (in-process tests) is
/// `0.0.0.0`, one shared bucket.
pub fn client_ip(config: &Config, headers: &HeaderMap, extensions: &Extensions) -> IpAddr {
    if config.trust_proxy_headers {
        let forwarded = headers
            .get_all("x-forwarded-for")
            .iter()
            .next_back()
            .and_then(|v| v.to_str().ok())
            .and_then(|v| v.rsplit(',').next())
            .and_then(|entry| entry.trim().parse::<IpAddr>().ok());
        if let Some(ip) = forwarded {
            return ip;
        }
    }
    extensions
        .get::<ConnectInfo<SocketAddr>>()
        .map(|info| info.0.ip())
        .unwrap_or(IpAddr::V4(Ipv4Addr::UNSPECIFIED))
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
    fn constant_time_comparison() {
        assert!(constant_time_eq(b"same", b"same"));
        assert!(!constant_time_eq(b"same", b"diff"));
        assert!(!constant_time_eq(b"short", b"longer value"));
    }

    #[test]
    fn failed_sign_ins_back_off_per_client() {
        let throttle = LoginThrottle::default();
        let (a, b): (IpAddr, IpAddr) = ("192.0.2.1".parse().unwrap(), "192.0.2.2".parse().unwrap());
        for _ in 0..FREE_FAILURES - 1 {
            assert!(throttle.check(a).is_ok());
            throttle.record_failure(a);
        }
        assert!(throttle.check(a).is_ok());
        throttle.record_failure(a);
        let wait = throttle.check(a).unwrap_err();
        assert!(wait <= Duration::from_secs(2) && wait > Duration::from_secs(1), "{wait:?}");
        assert!(throttle.check(b).is_ok(), "other clients are unaffected");
        throttle.record_failure(a);
        assert!(throttle.check(a).unwrap_err() > Duration::from_secs(2), "the wait doubles");
        throttle.record_success(a);
        assert!(throttle.check(a).is_ok());
        assert_eq!(throttled_detail(Duration::from_millis(1500)), "Too many failed sign-in attempts. Try again in 2 seconds.");
    }

    #[test]
    fn many_failures_from_everywhere_slow_everyone() {
        let throttle = LoginThrottle::default();
        for i in 0..GLOBAL_MAX_FAILURES {
            let ip = IpAddr::V4(Ipv4Addr::from(0x0a00_0000 + i));
            throttle.record_failure(ip);
        }
        let fresh: IpAddr = "198.51.100.7".parse().unwrap();
        assert!(throttle.check(fresh).unwrap_err() <= GLOBAL_WINDOW);
    }

    #[test]
    fn forwarded_for_counts_only_behind_a_trusted_proxy() {
        let mut config = Config::for_tests(std::path::Path::new("/tmp/nn"));
        let mut headers = HeaderMap::new();
        headers.insert("x-forwarded-for", "203.0.113.9, 198.51.100.1".parse().unwrap());
        let mut extensions = Extensions::new();
        extensions.insert(ConnectInfo::<SocketAddr>("127.0.0.1:4000".parse().unwrap()));
        assert_eq!(client_ip(&config, &headers, &extensions), "127.0.0.1".parse::<IpAddr>().unwrap());
        config.trust_proxy_headers = true;
        assert_eq!(client_ip(&config, &headers, &extensions), "198.51.100.1".parse::<IpAddr>().unwrap());
        assert_eq!(client_ip(&config, &HeaderMap::new(), &Extensions::new()), IpAddr::V4(Ipv4Addr::UNSPECIFIED));
    }
}
