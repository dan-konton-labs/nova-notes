//! Shared application state handed to every request handler and job.

use std::sync::Arc;

use crate::ai::ModelManager;
use crate::config::Config;
use crate::db::{migrate, Db};
use crate::error::AppResult;
use crate::jobs::JobQueue;
use crate::prompts::Prompter;

/// Everything a handler or job needs. Cheap to clone: every field is a handle.
#[derive(Clone)]
pub struct AppState {
    pub config: Arc<Config>,
    pub db: Db,
    pub jobs: JobQueue,
    /// Shared outbound HTTP client for fetched pages and web search
    /// (connection pool). AI provider calls use `models.clients()` instead.
    pub http: reqwest::Client,
    pub prompts: Prompter,
    /// Resolves configured models and credentials into ready-to-call clients.
    pub models: ModelManager,
    /// Backoff on failed password checks (sign-in form and API bearer).
    pub login_throttle: Arc<crate::api::auth::LoginThrottle>,
}

impl AppState {
    /// Connect to the database, apply pending migrations and assemble the state.
    /// Does not start the background worker; see [`crate::services::start`].
    pub async fn init(config: Config) -> AppResult<Self> {
        config.ensure_dirs()?;
        let db = Db::connect(&config).await?;
        migrate::run(&db).await?;
        Self::assemble(config, db)
    }

    fn assemble(config: Config, db: Db) -> AppResult<Self> {
        // Page fetches and search: bounded redirects and a total timeout.
        let http = crate::ai::net::content_client()?;
        // AI providers: no redirects, every resolved address vetted.
        let provider_http = crate::ai::net::provider_client(config.providers.url_policy)?;
        let config = Arc::new(config);
        Ok(Self {
            jobs: JobQueue::new(db.clone(), config.worker_max_tasks),
            models: ModelManager::new(db.clone(), config.clone(), provider_http),
            prompts: Prompter::new()?,
            config,
            db,
            http,
            login_throttle: Arc::default(),
        })
    }

    /// A fully migrated in-memory instance with its data in a temp directory.
    /// Keep the returned guard alive for as long as the state is used.
    pub async fn for_tests() -> AppResult<(Self, tempfile::TempDir)> {
        Self::for_tests_with(|_| {}).await
    }

    /// [`Self::for_tests`] with `edit` applied to the test configuration first.
    pub async fn for_tests_with(edit: impl FnOnce(&mut Config)) -> AppResult<(Self, tempfile::TempDir)> {
        let dir = tempfile::tempdir()?;
        let mut config = Config::for_tests(dir.path());
        edit(&mut config);
        config.ensure_dirs()?;
        let db = Db::memory().await?;
        migrate::run(&db).await?;
        Ok((Self::assemble(config, db)?, dir))
    }
}
