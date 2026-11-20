//! Nova Notes: a Rust rebuild of the Open Notebook research assistant.
//!
//! Layout:
//! - [`config`], [`error`], [`state`]: process-wide plumbing.
//! - [`db`]: the SurrealDB connection, repository helpers and migrations.
//! - [`jobs`]: the in-process background worker.
//! - [`prompts`]: embedded Jinja prompt templates.
//! - [`ai`]: turning configured models and credentials into clients.
//! - [`domain`]: records and the queries that belong to them.
//! - [`services`]: feature logic shared by the JSON API and the web UI.
//! - [`api`]: the JSON API, path-compatible with the Python build.
//! - [`web`]: the server-rendered web UI.

pub mod ai;
pub mod api;
pub mod config;
pub mod crypto;
pub mod db;
pub mod domain;
pub mod error;
pub mod jobs;
pub mod prompts;
pub mod services;
pub mod state;
#[cfg(test)]
pub mod testing;
pub mod web;

pub use config::Config;
pub use error::{AppError, AppResult};
pub use state::AppState;

/// Version reported by `GET /api/config`.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// The Open Notebook release whose API this build is compatible with. Reported
/// as `version` by `GET /api/config`, because clients gate features on it.
pub const API_COMPAT_VERSION: &str = "1.15.1";
