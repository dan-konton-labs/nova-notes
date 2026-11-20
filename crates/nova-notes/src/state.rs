//! Shared application state handed to every request handler and job.

use std::sync::Arc;

use crate::ai::ModelManager;
use crate::config::Config;
use crate::db::{migrate, Db};
use crate::error::{AppError, AppResult};
use crate::jobs::JobQueue;
use crate::prompts::Prompter;

/// Everything a handler or job needs. Cheap to clone: every field is a handle.
#[derive(Clone)]
pub struct AppState {
    pub config: Arc<Config>,
    pub db: Db,
    pub jobs: JobQueue,
    /// Shared outbound HTTP client (connection pool).
    pub http: reqwest::Client,
    pub prompts: Prompter,
    /// Resolves configured models and credentials into ready-to-call clients.
    pub models: ModelManager,
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
        let http = reqwest::Client::builder()
            .user_agent(concat!("nova-notes/", env!("CARGO_PKG_VERSION")))
            .connect_timeout(std::time::Duration::from_secs(30))
            .build()
            .map_err(|e| AppError::internal(format!("Could not build the HTTP client: {e}")))?;
        let config = Arc::new(config);
        Ok(Self {
            jobs: JobQueue::new(db.clone(), config.worker_max_tasks),
            models: ModelManager::new(db.clone(), config.clone(), http.clone()),
            prompts: Prompter::new()?,
            config,
            db,
            http,
        })
    }

    /// A fully migrated in-memory instance with its data in a temp directory.
    /// Keep the returned guard alive for as long as the state is used.
    pub async fn for_tests() -> AppResult<(Self, tempfile::TempDir)> {
        let dir = tempfile::tempdir()?;
        let config = Config::for_tests(dir.path());
        config.ensure_dirs()?;
        let db = Db::memory().await?;
        migrate::run(&db).await?;
        Ok((Self::assemble(config, db)?, dir))
    }
}
