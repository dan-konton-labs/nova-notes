//! Feature logic shared by the JSON API and the web UI.
//!
//! Each module holds one feature's operations as plain async functions taking
//! `&AppState`, with typed inputs and outputs. The routers in [`crate::api`]
//! and the pages in [`crate::web`] are thin layers over these.

pub mod chat;
pub mod chat_import;
pub mod context;
pub mod credentials;
pub mod discovery;
pub mod embedding;
pub mod explainers;
pub mod generation_jobs;
pub mod infographics;
pub mod insights;
pub mod meetings;
pub mod models;
pub mod note_docs;
pub mod notebooks;
pub mod notes;
pub mod podcast_feed;
pub mod podcasts;
pub mod search;
pub mod settings;
pub mod source_chat;
pub mod sources;
pub mod transformations;
pub mod transport;
pub mod vault;
pub mod workspace;

use crate::error::AppResult;
use crate::jobs::JobQueue;
use crate::state::AppState;

/// Register every feature's background job handlers.
pub fn register_jobs(jobs: &JobQueue) {
    chat::register_jobs(jobs);
    context::register_jobs(jobs);
    credentials::register_jobs(jobs);
    discovery::register_jobs(jobs);
    embedding::register_jobs(jobs);
    explainers::register_jobs(jobs);
    infographics::register_jobs(jobs);
    insights::register_jobs(jobs);
    meetings::register_jobs(jobs);
    models::register_jobs(jobs);
    note_docs::register_jobs(jobs);
    notebooks::register_jobs(jobs);
    notes::register_jobs(jobs);
    podcast_feed::register_jobs(jobs);
    podcasts::register_jobs(jobs);
    search::register_jobs(jobs);
    settings::register_jobs(jobs);
    source_chat::register_jobs(jobs);
    sources::register_jobs(jobs);
    transformations::register_jobs(jobs);
    transport::register_jobs(jobs);
    vault::register_jobs(jobs);
    workspace::register_jobs(jobs);
}

/// Register the job handlers, start the background worker and run the
/// one-off startup tasks. Call once, after [`AppState::init`].
pub async fn start(state: &AppState) -> AppResult<()> {
    register_jobs(&state.jobs);
    state.jobs.start(state.clone()).await?;
    podcast_feed::on_startup(state).await?;
    Ok(())
}
