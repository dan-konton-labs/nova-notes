//! `GET /api/config`: version and database health for clients.

use std::time::Duration;

use axum::routing::get;
use axum::{extract::State, Json, Router};
use serde_json::{json, Value};

use crate::state::AppState;

pub fn router() -> Router<AppState> {
    Router::new().route("/config", get(get_config))
}

async fn get_config(State(state): State<AppState>) -> Json<Value> {
    let db_status = match tokio::time::timeout(Duration::from_secs(2), state.db.ping()).await {
        Ok(Ok(())) => "online",
        Ok(Err(e)) => {
            tracing::warn!("Database health check failed: {e}");
            "offline"
        }
        Err(_) => {
            tracing::warn!("Database health check timed out after 2 seconds");
            "offline"
        }
    };
    // The Python build asks GitHub whether a newer release exists. This build
    // makes no such outbound call, so it never reports an update.
    Json(json!({
        "version": crate::API_COMPAT_VERSION,
        "latestVersion": Value::Null,
        "hasUpdate": false,
        "dbStatus": db_status,
        "build": format!("nova-notes-rust {}", crate::VERSION),
    }))
}
