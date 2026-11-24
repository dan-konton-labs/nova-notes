//! Web UI sign-in sessions.
//!
//! Each sign-in gets its own random 256-bit session id, sent to the browser
//! as the session cookie. The server keeps a row per session in
//! `login_session`, keyed by a SHA-256 of the id (so a database dump holds
//! no usable cookie), with:
//!
//! - `expires`: an absolute expiry 30 days after sign-in, checked by the
//!   server on every request (signing in again starts a new session).
//! - `binding`: a SHA-256 over the session id and the instance password.
//!   The password comes from the environment, so changing it and restarting
//!   ends every session made under the old one. Because the hash covers the
//!   random id, which is never stored, the row is no offline password oracle.
//!
//! Signing out deletes the row. Expired rows are purged at each sign-in.

use chrono::{Duration, Utc};
use rand::RngCore;
use sha2::{Digest, Sha256};

use crate::api::auth::constant_time_eq;
use crate::db::{Doc, Params};
use crate::error::AppResult;
use crate::state::AppState;

/// Table holding one row per signed-in browser.
pub const TABLE: &str = "login_session";

/// How long a session lasts from sign-in (absolute, not sliding).
pub const LIFETIME_DAYS: i64 = 30;

/// The cookie `Max-Age` matching [`LIFETIME_DAYS`].
pub const LIFETIME_SECONDS: i64 = LIFETIME_DAYS * 24 * 60 * 60;

/// A session id is 32 random bytes, hex-encoded.
const ID_LENGTH: usize = 64;

fn sha256_hex(parts: &[&[u8]]) -> String {
    let mut hasher = Sha256::new();
    for part in parts {
        hasher.update(part);
    }
    hex::encode(hasher.finalize())
}

/// Whether a cookie value has the shape of a session id; anything else is
/// refused without a database read. (The old password-derived cookie has the
/// same shape and is refused because no row matches it.)
fn well_formed(id: &str) -> bool {
    id.len() == ID_LENGTH && id.bytes().all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase())
}

fn record_id(id: &str) -> String {
    format!("{TABLE}:{}", sha256_hex(&[b"nova-notes-session-id-v2:", id.as_bytes()]))
}

fn binding(id: &str, password: &str) -> String {
    sha256_hex(&[b"nova-notes-session-binding-v2:", id.as_bytes(), b":", password.as_bytes()])
}

/// Start a session for a browser that has just given `password`. Returns
/// the session id to put in its cookie.
pub async fn create(state: &AppState, password: &str) -> AppResult<String> {
    if let Err(e) = state.db.query(&format!("DELETE {TABLE} WHERE expires < time::now()"), Params::new()).await {
        tracing::warn!("Could not purge expired sign-in sessions: {e}");
    }
    let mut bytes = [0u8; 32];
    rand::rngs::OsRng.fill_bytes(&mut bytes);
    let id = hex::encode(bytes);
    let expires = Utc::now() + Duration::days(LIFETIME_DAYS);
    state
        .db
        .create_with_id(&record_id(&id), Doc::new().set("binding", binding(&id, password)).datetime("expires", expires))
        .await?;
    Ok(id)
}

/// Whether `id` names a live session made under the current `password`.
/// A database error counts as "no" (the browser is sent to sign in).
pub async fn is_valid(state: &AppState, id: &str, password: &str) -> bool {
    if !well_formed(id) {
        return false;
    }
    let rows = state
        .db
        .query("SELECT binding FROM $id WHERE expires > time::now()", Params::new().record("id", record_id(id)))
        .await;
    match rows {
        Ok(rows) => rows.iter().any(|row| {
            row.get("binding")
                .and_then(|v| v.as_str())
                .is_some_and(|stored| constant_time_eq(stored.as_bytes(), binding(id, password).as_bytes()))
        }),
        Err(e) => {
            tracing::warn!("Could not check a sign-in session: {e}");
            false
        }
    }
}

/// End the session `id` (signing out). Unknown ids are ignored.
pub async fn revoke(state: &AppState, id: &str) -> AppResult<()> {
    if well_formed(id) {
        state.db.delete(&record_id(id)).await?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn sessions_are_random_bound_to_the_password_and_expire() {
        let (state, _dir) = AppState::for_tests().await.unwrap();
        let a = create(&state, "hunter2").await.unwrap();
        let b = create(&state, "hunter2").await.unwrap();
        assert_ne!(a, b);
        assert!(well_formed(&a));
        assert!(is_valid(&state, &a, "hunter2").await);
        assert!(!is_valid(&state, &a, "hunter3").await, "a new password ends old sessions");
        assert!(!is_valid(&state, &a.to_uppercase(), "hunter2").await);
        assert!(!is_valid(&state, "not-a-session", "hunter2").await);

        // The raw id is never stored.
        let rows = state.db.query(&format!("SELECT * FROM {TABLE}"), Params::new()).await.unwrap();
        assert_eq!(rows.len(), 2);
        assert!(!serde_json::to_string(&rows).unwrap().contains(&a));

        // Expiry is the server's, not just the cookie's.
        state
            .db
            .query("UPDATE $id SET expires = time::now() - 1s", Params::new().record("id", record_id(&a)))
            .await
            .unwrap();
        assert!(!is_valid(&state, &a, "hunter2").await);
        assert!(is_valid(&state, &b, "hunter2").await);
        // ... and expired rows are purged at the next sign-in.
        let c = create(&state, "hunter2").await.unwrap();
        let rows = state.db.query(&format!("SELECT * FROM {TABLE}"), Params::new()).await.unwrap();
        assert_eq!(rows.len(), 2);

        revoke(&state, &b).await.unwrap();
        assert!(!is_valid(&state, &b, "hunter2").await);
        assert!(is_valid(&state, &c, "hunter2").await);
    }
}
