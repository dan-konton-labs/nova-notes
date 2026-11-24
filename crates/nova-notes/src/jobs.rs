//! Background jobs.
//!
//! The Python build hands long-running work (source processing, embeddings,
//! podcasts) to a separate `surreal-commands` worker process that polls a
//! `command` table. Here the worker lives inside the server: jobs are still
//! recorded in the same `command` table with the same fields, so everything
//! that reads job status (the API, `FETCH command` in queries, another
//! build's UI) keeps working, but there is no second process to forget to
//! start.
//!
//! A job is a named async function taking JSON arguments and returning a JSON
//! result. Features register their handlers at startup with
//! [`JobQueue::register`]; anything can then [`JobQueue::submit`] by name.

use std::collections::HashMap;
use std::future::Future;
use std::panic::AssertUnwindSafe;
use std::pin::Pin;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, RwLock};
use std::time::Duration;

use chrono::{DateTime, Utc};
use futures::FutureExt;
use rand::Rng;
use serde::{Deserialize, Serialize};
use tokio::sync::{mpsc, Semaphore};
use tokio::task::AbortHandle;

use crate::db::{Db, Doc, Json, Params};
use crate::error::{AppError, AppResult};
use crate::state::AppState;

/// The `app` every command is filed under (the Python build's app name).
pub const APP_NAME: &str = "open_notebook";

/// How many jobs may be queued or running at once by default
/// (`OPEN_NOTEBOOK_MAX_PENDING_JOBS`). Past it, [`JobQueue::submit`] answers
/// 429 until the backlog drains. Generous so a full embedding rebuild of a
/// large library still fits.
pub const DEFAULT_MAX_PENDING_JOBS: usize = 10_000;

/// Finished jobs older than this many days are pruned by default
/// (`OPEN_NOTEBOOK_JOB_RETENTION_DAYS`, `0` keeps them forever).
pub const DEFAULT_JOB_RETENTION_DAYS: u64 = 30;

/// How often the worker prunes old finished jobs.
const RETENTION_SWEEP_INTERVAL: Duration = Duration::from_secs(6 * 60 * 60);

/// Rows deleted per retention statement, so one sweep never holds a huge transaction.
const RETENTION_BATCH: usize = 1_000;

/// Batches per sweep; whatever is left waits for the next sweep.
const RETENTION_MAX_BATCHES: usize = 100;

fn max_pending_from_env() -> usize {
    match crate::config::env_nonempty("OPEN_NOTEBOOK_MAX_PENDING_JOBS") {
        None => DEFAULT_MAX_PENDING_JOBS,
        Some(raw) => match raw.trim().parse::<usize>() {
            Ok(n) if n > 0 => n,
            _ => {
                tracing::warn!(
                    "OPEN_NOTEBOOK_MAX_PENDING_JOBS={raw:?} is not a positive number; using {DEFAULT_MAX_PENDING_JOBS}"
                );
                DEFAULT_MAX_PENDING_JOBS
            }
        },
    }
}

fn retention_days_from_env() -> u64 {
    match crate::config::env_nonempty("OPEN_NOTEBOOK_JOB_RETENTION_DAYS") {
        None => DEFAULT_JOB_RETENTION_DAYS,
        Some(raw) => raw.trim().parse::<u64>().unwrap_or_else(|_| {
            tracing::warn!(
                "OPEN_NOTEBOOK_JOB_RETENTION_DAYS={raw:?} is not a whole number of days; using {DEFAULT_JOB_RETENTION_DAYS}"
            );
            DEFAULT_JOB_RETENTION_DAYS
        }),
    }
}

pub mod status {
    pub const NEW: &str = "new";
    pub const RUNNING: &str = "running";
    pub const COMPLETED: &str = "completed";
    pub const FAILED: &str = "failed";
    pub const CANCELED: &str = "canceled";

    /// Whether a job in this state will never change again.
    pub fn is_terminal(status: &str) -> bool {
        matches!(status, COMPLETED | FAILED | CANCELED)
    }
}

/// How often and how patiently a failed job is retried.
///
/// Only errors for which [`AppError::is_retryable`] is true are retried.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RetryPolicy {
    pub max_attempts: u32,
    pub wait_min: Duration,
    pub wait_max: Duration,
}

impl RetryPolicy {
    /// Run once; any failure is final. Used where a second run would
    /// duplicate work (podcast generation).
    pub const fn once() -> Self {
        Self { max_attempts: 1, wait_min: Duration::ZERO, wait_max: Duration::ZERO }
    }

    /// Up to `max_attempts` runs with exponential backoff and jitter between
    /// `wait_min` and `wait_max`.
    pub const fn attempts(max_attempts: u32, wait_min: Duration, wait_max: Duration) -> Self {
        Self { max_attempts, wait_min, wait_max }
    }

    fn delay(&self, attempt: u32) -> Duration {
        let factor = 2u32.saturating_pow(attempt.saturating_sub(1));
        let base = self.wait_min.saturating_mul(factor);
        let jitter = Duration::from_millis(rand::thread_rng().gen_range(0..=1000));
        base.saturating_add(jitter).min(self.wait_max.max(self.wait_min))
    }
}

/// What a running job knows about itself.
#[derive(Clone)]
pub struct JobContext {
    pub state: AppState,
    /// `command:<id>` of the record tracking this run.
    pub command_id: String,
    /// 1 for the first run, 2 for the first retry, ...
    pub attempt: u32,
}

impl JobContext {
    /// Publish progress on the command record (shown by `GET /api/commands/jobs/{id}`).
    pub async fn set_progress(&self, progress: Json) -> AppResult<()> {
        self.state
            .db
            .query(
                "UPDATE $id SET progress = $progress, updated = time::now()",
                Params::new().record("id", &self.command_id).set("progress", progress),
            )
            .await
            .map(|_| ())
    }
}

type JobFuture = Pin<Box<dyn Future<Output = AppResult<Json>> + Send>>;
type Handler = Arc<dyn Fn(JobContext, Json) -> JobFuture + Send + Sync>;

#[derive(Clone)]
struct Registered {
    handler: Handler,
    policy: RetryPolicy,
}

/// A row of the `command` table.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CommandRecord {
    pub id: String,
    #[serde(default)]
    pub app: Option<String>,
    #[serde(default)]
    pub name: Option<String>,
    #[serde(default)]
    pub args: Option<Json>,
    #[serde(default)]
    pub status: Option<String>,
    #[serde(default)]
    pub result: Option<Json>,
    #[serde(default)]
    pub error_message: Option<String>,
    #[serde(default)]
    pub progress: Option<Json>,
    #[serde(default)]
    pub created: Option<DateTime<Utc>>,
    #[serde(default)]
    pub updated: Option<DateTime<Utc>>,
}

impl CommandRecord {
    /// The status string, `"unknown"` when the record has none.
    pub fn status_str(&self) -> &str {
        self.status.as_deref().unwrap_or("unknown")
    }

    pub fn is_terminal(&self) -> bool {
        status::is_terminal(self.status_str())
    }
}

/// A place in the queue, held from submission until the job's task ends
/// (finished, failed or canceled). Dropping it frees the place.
pub struct PendingSlot(Arc<AtomicUsize>);

impl Drop for PendingSlot {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::SeqCst);
    }
}

struct Inner {
    db: Db,
    handlers: RwLock<HashMap<String, Registered>>,
    // Unbounded, but every message carries a `PendingSlot`, so the number of
    // queued and running jobs is capped by `max_pending`.
    sender: mpsc::UnboundedSender<(String, PendingSlot)>,
    receiver: Mutex<Option<mpsc::UnboundedReceiver<(String, PendingSlot)>>>,
    running: Mutex<HashMap<String, AbortHandle>>,
    permits: Arc<Semaphore>,
    pending: Arc<AtomicUsize>,
    max_pending: AtomicUsize,
}

/// The job queue. Cheap to clone.
#[derive(Clone)]
pub struct JobQueue {
    inner: Arc<Inner>,
}

impl JobQueue {
    pub fn new(db: Db, max_concurrent: usize) -> Self {
        let (sender, receiver) = mpsc::unbounded_channel();
        Self {
            inner: Arc::new(Inner {
                db,
                handlers: RwLock::new(HashMap::new()),
                sender,
                receiver: Mutex::new(Some(receiver)),
                running: Mutex::new(HashMap::new()),
                permits: Arc::new(Semaphore::new(max_concurrent.max(1))),
                pending: Arc::new(AtomicUsize::new(0)),
                max_pending: AtomicUsize::new(max_pending_from_env()),
            }),
        }
    }

    /// Register the handler for a command name. Registering a name twice
    /// replaces the earlier handler.
    pub fn register<F, Fut>(&self, name: &str, policy: RetryPolicy, handler: F)
    where
        F: Fn(JobContext, Json) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = AppResult<Json>> + Send + 'static,
    {
        let handler: Handler = Arc::new(move |ctx, args| Box::pin(handler(ctx, args)));
        self.inner
            .handlers
            .write()
            .expect("job registry lock poisoned")
            .insert(name.to_string(), Registered { handler, policy });
    }

    /// Names of every registered command, sorted.
    pub fn registered(&self) -> Vec<String> {
        let mut names: Vec<String> =
            self.inner.handlers.read().expect("job registry lock poisoned").keys().cloned().collect();
        names.sort();
        names
    }

    fn lookup(&self, name: &str) -> AppResult<Registered> {
        self.inner
            .handlers
            .read()
            .expect("job registry lock poisoned")
            .get(name)
            .cloned()
            .ok_or_else(|| AppError::internal(format!("Unknown command: {name}")))
    }

    async fn create_record(&self, name: &str, args: &Json) -> AppResult<String> {
        self.lookup(name)?;
        let record = self
            .inner
            .db
            .create(
                "command",
                Doc::new()
                    .set("app", APP_NAME)
                    .set("name", name)
                    .set("args", args)
                    .set("status", status::NEW),
            )
            .await?;
        record
            .get("id")
            .and_then(Json::as_str)
            .map(String::from)
            .ok_or_else(|| AppError::database("Command record has no id"))
    }

    /// Change how many jobs may be queued or running at once.
    pub fn set_max_pending(&self, max: usize) {
        self.inner.max_pending.store(max.max(1), Ordering::SeqCst);
    }

    /// Jobs queued or running right now.
    pub fn pending(&self) -> usize {
        self.inner.pending.load(Ordering::SeqCst)
    }

    /// Take a place in the queue, or fail with 429 when it is full. Lets a
    /// caller check for room before doing something it cannot undo, then
    /// [`JobQueue::submit_reserved`].
    pub fn reserve(&self) -> AppResult<PendingSlot> {
        let max = self.inner.max_pending.load(Ordering::SeqCst);
        let pending = &self.inner.pending;
        let mut current = pending.load(Ordering::SeqCst);
        loop {
            if current >= max {
                tracing::warn!("Refused a background job: {max} jobs are already queued or running");
                return Err(AppError::RateLimit(format!(
                    "Too many background jobs are queued ({max}). Wait for some to finish and try again."
                )));
            }
            match pending.compare_exchange_weak(current, current + 1, Ordering::SeqCst, Ordering::SeqCst) {
                Ok(_) => return Ok(PendingSlot(pending.clone())),
                Err(actual) => current = actual,
            }
        }
    }

    /// A place in the queue that ignores the limit (jobs resumed at start-up).
    fn force_reserve(&self) -> PendingSlot {
        self.inner.pending.fetch_add(1, Ordering::SeqCst);
        PendingSlot(self.inner.pending.clone())
    }

    /// Queue a job and return its `command:<id>` immediately. Fails with 429
    /// (retryable) when too many jobs are already queued or running.
    pub async fn submit(&self, name: &str, args: Json) -> AppResult<String> {
        let slot = self.reserve()?;
        self.submit_reserved(slot, name, args).await
    }

    /// [`JobQueue::submit`] with a place taken earlier by [`JobQueue::reserve`].
    pub async fn submit_reserved(&self, slot: PendingSlot, name: &str, args: Json) -> AppResult<String> {
        let id = self.create_record(name, &args).await?;
        tracing::info!("Submitted command job: {id} for {APP_NAME}.{name}");
        self.inner
            .sender
            .send((id.clone(), slot))
            .map_err(|_| AppError::internal("The background worker is not running"))?;
        Ok(id)
    }

    /// Run a job to completion in the calling task (still recorded in the
    /// `command` table, still retried per its policy) and return its result.
    pub async fn run_inline(&self, state: &AppState, name: &str, args: Json) -> AppResult<(String, Json)> {
        let id = self.create_record(name, &args).await?;
        let result = execute(state.clone(), &id).await?;
        Ok((id, result))
    }

    /// The command record, or `None` if there is no such job.
    pub async fn get(&self, id: &str) -> AppResult<Option<CommandRecord>> {
        let id = crate::db::with_table("command", id);
        match self.inner.db.select(&id).await? {
            Some(row) => Ok(Some(serde_json::from_value(row)?)),
            None => Ok(None),
        }
    }

    /// Command records for many ids in one query, keyed by id.
    pub async fn get_many(&self, ids: &[String]) -> AppResult<HashMap<String, CommandRecord>> {
        if ids.is_empty() {
            return Ok(HashMap::new());
        }
        let ids: Vec<String> = ids.iter().map(|id| crate::db::with_table("command", id)).collect();
        let rows: Vec<CommandRecord> = self
            .inner
            .db
            .query_as("SELECT * FROM command WHERE id IN $ids", Params::new().records("ids", &ids))
            .await?;
        Ok(rows.into_iter().map(|r| (r.id.clone(), r)).collect())
    }

    /// Recent jobs, newest first, optionally filtered by command name and status.
    pub async fn list(&self, name: Option<&str>, status: Option<&str>, limit: usize) -> AppResult<Vec<CommandRecord>> {
        let mut conditions = Vec::new();
        let mut params = Params::new().set("limit", limit);
        if let Some(name) = name {
            conditions.push("name = $name");
            params = params.set("name", name);
        }
        if let Some(status) = status {
            conditions.push("status = $status");
            params = params.set("status", status);
        }
        let filter = if conditions.is_empty() { String::new() } else { format!("WHERE {}", conditions.join(" AND ")) };
        self.inner
            .db
            .query_as(&format!("SELECT * FROM command {filter} ORDER BY created DESC LIMIT $limit"), params)
            .await
    }

    /// Stop a queued or running job. Returns false when it had already finished
    /// or does not exist.
    pub async fn cancel(&self, id: &str) -> AppResult<bool> {
        let id = crate::db::with_table("command", id);
        let Some(record) = self.get(&id).await? else {
            return Ok(false);
        };
        if record.is_terminal() {
            return Ok(false);
        }
        if let Some(handle) = self.inner.running.lock().expect("job map lock poisoned").remove(&id) {
            handle.abort();
        }
        self.inner
            .db
            .query(
                "UPDATE $id SET status = $status, error_message = 'Canceled', updated = time::now()",
                Params::new().record("id", &id).set("status", status::CANCELED),
            )
            .await?;
        Ok(true)
    }

    /// Wait until a job reaches a terminal state. Mostly for tests.
    pub async fn wait(&self, id: &str, timeout: Duration) -> AppResult<CommandRecord> {
        let deadline = tokio::time::Instant::now() + timeout;
        loop {
            if let Some(record) = self.get(id).await? {
                if record.is_terminal() {
                    return Ok(record);
                }
            }
            if tokio::time::Instant::now() >= deadline {
                return Err(AppError::internal(format!("Timed out waiting for {id}")));
            }
            tokio::time::sleep(Duration::from_millis(25)).await;
        }
    }

    /// Start the worker. Jobs left `running` by a previous process are marked
    /// failed (their work is gone); jobs still `new` are picked up again.
    /// Calling this twice is a no-op.
    pub async fn start(&self, state: AppState) -> AppResult<()> {
        let Some(mut receiver) = self.inner.receiver.lock().expect("job receiver lock poisoned").take() else {
            return Ok(());
        };

        self.inner
            .db
            .query(
                "UPDATE command SET status = $failed, error_message = 'Interrupted by a server restart', updated = time::now() WHERE status = $running",
                Params::new().set("failed", status::FAILED).set("running", status::RUNNING),
            )
            .await?;
        let pending = self
            .inner
            .db
            .query(
                "SELECT id, created FROM command WHERE status = $new ORDER BY created ASC",
                Params::new().set("new", status::NEW),
            )
            .await?;
        for row in &pending {
            if let Some(id) = row.get("id").and_then(Json::as_str) {
                let _ = self.inner.sender.send((id.to_string(), self.force_reserve()));
            }
        }
        if !pending.is_empty() {
            tracing::info!("Resuming {} queued background job(s)", pending.len());
        }

        let retention_days = retention_days_from_env();
        if retention_days > 0 {
            let queue = self.clone();
            tokio::spawn(async move {
                let mut every = tokio::time::interval(RETENTION_SWEEP_INTERVAL);
                loop {
                    every.tick().await;
                    match queue.prune_finished(retention_days).await {
                        Ok(0) => {}
                        Ok(n) => tracing::info!("Pruned {n} finished background job(s) older than {retention_days} days"),
                        Err(e) => tracing::warn!("Could not prune old background jobs: {e}"),
                    }
                }
            });
        }

        let queue = self.clone();
        tokio::spawn(async move {
            while let Some((id, slot)) = receiver.recv().await {
                let inner = queue.inner.clone();
                let state = state.clone();
                let job_id = id.clone();
                let task = tokio::spawn(async move {
                    // Held until this task ends, however it ends (an abort drops it too).
                    let _slot = slot;
                    let Ok(_permit) = inner.permits.clone().acquire_owned().await else {
                        return;
                    };
                    if let Err(e) = execute(state, &job_id).await {
                        tracing::debug!("Job {job_id} ended with: {e}");
                    }
                    inner.running.lock().expect("job map lock poisoned").remove(&job_id);
                });
                let mut running = queue.inner.running.lock().expect("job map lock poisoned");
                // The task may already be done, in which case it has nothing to clean up after itself.
                if !task.is_finished() {
                    running.insert(id, task.abort_handle());
                }
            }
        });
        Ok(())
    }
}

impl JobQueue {
    /// Delete finished jobs (completed, failed, canceled) last touched more
    /// than `days` days ago. Jobs still linked from a source, an episode or a
    /// meeting are kept, since their status is still shown, but their
    /// arguments are emptied (`{}`). Queued and running jobs are never touched.
    /// Returns how many rows were deleted.
    pub async fn prune_finished(&self, days: u64) -> AppResult<usize> {
        if days == 0 {
            return Ok(0);
        }
        // `days` is a number from our own config, not user input.
        let old = format!(
            "status IN $done AND (updated ?? created) != NONE AND (updated ?? created) < time::now() - {days}d"
        );
        let done = [status::COMPLETED, status::FAILED, status::CANCELED];
        let refs = "LET $linked = array::union(
                (SELECT VALUE command FROM source WHERE command != NONE),
                (SELECT VALUE command FROM episode WHERE command != NONE)
            );
            LET $meeting_jobs = (SELECT VALUE summary_job FROM meeting WHERE type::is::string(summary_job) AND summary_job != '');";
        let mut deleted = 0;
        for _ in 0..RETENTION_MAX_BATCHES {
            let rows = self
                .inner
                .db
                .query_last(
                    &format!(
                        "{refs}
                        LET $gone = (SELECT VALUE id FROM command WHERE {old}
                            AND id NOT IN $linked AND type::string(id) NOT IN $meeting_jobs LIMIT $batch);
                        DELETE command WHERE id IN $gone RETURN NONE;
                        RETURN array::len($gone);"
                    ),
                    Params::new().set("done", done).set("batch", RETENTION_BATCH),
                )
                .await?;
            let n = rows.first().and_then(Json::as_u64).unwrap_or(0) as usize;
            deleted += n;
            if n < RETENTION_BATCH {
                break;
            }
        }
        self.inner
            .db
            .query(&format!("UPDATE command SET args = {{}} WHERE {old} AND args != NONE AND args != {{}} RETURN NONE"), Params::new().set("done", done))
            .await?;
        Ok(deleted)
    }
}

/// Run one job through its retry policy, recording every state change.
async fn execute(state: AppState, id: &str) -> AppResult<Json> {
    let db = &state.db;
    let record: CommandRecord = db.get(id).await?;
    if record.status_str() != status::NEW {
        // Canceled while queued, or already picked up.
        return Err(AppError::internal(format!("Job {id} is {} and will not be run", record.status_str())));
    }
    let name = record.name.clone().unwrap_or_default();
    let args = record.args.clone().unwrap_or(Json::Null);
    let registered = match state.jobs.lookup(&name) {
        Ok(r) => r,
        Err(e) => {
            fail(db, id, &e).await;
            return Err(e);
        }
    };

    let started = Utc::now();
    db.query(
        "UPDATE $id SET status = $status, updated = time::now()",
        Params::new().record("id", id).set("status", status::RUNNING),
    )
    .await?;

    let mut attempt = 1;
    loop {
        let ctx = JobContext { state: state.clone(), command_id: id.to_string(), attempt };
        let outcome = AssertUnwindSafe((registered.handler)(ctx, args.clone()))
            .catch_unwind()
            .await
            .unwrap_or_else(|panic| {
                let what = panic
                    .downcast_ref::<&str>()
                    .map(|s| s.to_string())
                    .or_else(|| panic.downcast_ref::<String>().cloned())
                    .unwrap_or_else(|| "unknown panic".to_string());
                Err(AppError::internal(format!("Job panicked: {what}")))
            });

        match outcome {
            Ok(mut result) => {
                let completed = Utc::now();
                if let Json::Object(map) = &mut result {
                    map.insert(
                        "execution_metadata".to_string(),
                        serde_json::json!({
                            "started_at": started.to_rfc3339(),
                            "completed_at": completed.to_rfc3339(),
                            "execution_time": (completed - started).num_milliseconds() as f64 / 1000.0,
                            "attempts": attempt,
                        }),
                    );
                }
                db.query(
                    "UPDATE $id SET status = $status, result = $result, error_message = NONE, updated = time::now()",
                    Params::new().record("id", id).set("status", status::COMPLETED).set("result", &result),
                )
                .await?;
                return Ok(result);
            }
            Err(e) if e.is_retryable() && attempt < registered.policy.max_attempts => {
                let delay = registered.policy.delay(attempt);
                tracing::debug!("Job {id} ({name}) attempt {attempt} failed, retrying in {delay:?}: {e}");
                attempt += 1;
                tokio::time::sleep(delay).await;
            }
            Err(e) => {
                tracing::error!("Job {id} ({name}) failed after {attempt} attempt(s): {e}");
                fail(db, id, &e).await;
                return Err(e);
            }
        }
    }
}

async fn fail(db: &Db, id: &str, error: &AppError) {
    let outcome = db
        .query(
            "UPDATE $id SET status = $status, error_message = $message, updated = time::now()",
            Params::new().record("id", id).set("status", status::FAILED).set("message", error.to_string()),
        )
        .await;
    if let Err(e) = outcome {
        tracing::error!("Could not record failure of job {id}: {e}");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU32, Ordering};

    use serde_json::json;

    async fn started_state() -> (AppState, tempfile::TempDir) {
        let (state, dir) = AppState::for_tests().await.unwrap();
        state.jobs.start(state.clone()).await.unwrap();
        (state, dir)
    }

    #[tokio::test]
    async fn a_job_runs_and_records_its_result() {
        let (state, _dir) = started_state().await;
        state.jobs.register("double", RetryPolicy::once(), |_ctx, args| async move {
            Ok(json!({ "value": args["n"].as_i64().unwrap_or(0) * 2 }))
        });
        let id = state.jobs.submit("double", json!({ "n": 21 })).await.unwrap();
        assert!(id.starts_with("command:"));

        let record = state.jobs.wait(&id, Duration::from_secs(5)).await.unwrap();
        assert_eq!(record.status_str(), status::COMPLETED);
        let result = record.result.unwrap();
        assert_eq!(result["value"], 42);
        assert!(result["execution_metadata"]["started_at"].is_string());
        assert_eq!(record.app.as_deref(), Some(APP_NAME));
        assert_eq!(record.name.as_deref(), Some("double"));
    }

    #[tokio::test]
    async fn retryable_errors_retry_and_permanent_ones_do_not() {
        let (state, _dir) = started_state().await;
        let flaky_runs = Arc::new(AtomicU32::new(0));
        let runs = flaky_runs.clone();
        state.jobs.register(
            "flaky",
            RetryPolicy::attempts(5, Duration::from_millis(1), Duration::from_millis(5)),
            move |ctx, _args| {
                let runs = runs.clone();
                async move {
                    runs.fetch_add(1, Ordering::SeqCst);
                    if ctx.attempt < 3 {
                        Err(AppError::RateLimit("slow down".into()))
                    } else {
                        Ok(json!({ "attempt": ctx.attempt }))
                    }
                }
            },
        );
        let bad_runs = Arc::new(AtomicU32::new(0));
        let runs = bad_runs.clone();
        state.jobs.register(
            "invalid",
            RetryPolicy::attempts(5, Duration::from_millis(1), Duration::from_millis(5)),
            move |_ctx, _args| {
                let runs = runs.clone();
                async move {
                    runs.fetch_add(1, Ordering::SeqCst);
                    Err::<Json, _>(AppError::invalid("nope"))
                }
            },
        );

        let flaky = state.jobs.submit("flaky", Json::Null).await.unwrap();
        let invalid = state.jobs.submit("invalid", Json::Null).await.unwrap();

        let record = state.jobs.wait(&flaky, Duration::from_secs(10)).await.unwrap();
        assert_eq!(record.status_str(), status::COMPLETED);
        assert_eq!(flaky_runs.load(Ordering::SeqCst), 3);

        let record = state.jobs.wait(&invalid, Duration::from_secs(10)).await.unwrap();
        assert_eq!(record.status_str(), status::FAILED);
        assert_eq!(record.error_message.as_deref(), Some("nope"));
        assert_eq!(bad_runs.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn a_panicking_job_fails_cleanly() {
        let (state, _dir) = started_state().await;
        state.jobs.register("boom", RetryPolicy::once(), |_ctx, _args| async move {
            if true {
                panic!("kaboom");
            }
            Ok(Json::Null)
        });
        let id = state.jobs.submit("boom", Json::Null).await.unwrap();
        let record = state.jobs.wait(&id, Duration::from_secs(5)).await.unwrap();
        assert_eq!(record.status_str(), status::FAILED);
        assert!(record.error_message.unwrap().contains("kaboom"));
    }

    #[tokio::test]
    async fn unknown_commands_are_rejected_at_submit() {
        let (state, _dir) = started_state().await;
        assert!(state.jobs.submit("nope", Json::Null).await.is_err());
    }

    #[tokio::test]
    async fn inline_runs_return_the_result() {
        let (state, _dir) = AppState::for_tests().await.unwrap();
        state.jobs.register("echo", RetryPolicy::once(), |ctx, args| async move {
            ctx.set_progress(json!({ "percent": 50 })).await?;
            Ok(json!({ "echo": args }))
        });
        let (id, result) = state.jobs.run_inline(&state, "echo", json!("hi")).await.unwrap();
        assert_eq!(result["echo"], "hi");
        let record = state.jobs.get(&id).await.unwrap().unwrap();
        assert_eq!(record.status_str(), status::COMPLETED);
        assert_eq!(record.progress.unwrap()["percent"], 50);
    }

    #[tokio::test]
    async fn restart_fails_interrupted_jobs_and_resumes_queued_ones() {
        let (state, _dir) = AppState::for_tests().await.unwrap();
        state.jobs.register("noop", RetryPolicy::once(), |_ctx, _args| async move { Ok(json!({})) });
        // Left behind by a previous process: one mid-run, one never started.
        let interrupted = state.db.create("command", Doc::new().set("app", APP_NAME).set("name", "noop").set("status", status::RUNNING)).await.unwrap();
        let queued = state.db.create("command", Doc::new().set("app", APP_NAME).set("name", "noop").set("status", status::NEW)).await.unwrap();

        state.jobs.start(state.clone()).await.unwrap();

        let interrupted = state.jobs.wait(interrupted["id"].as_str().unwrap(), Duration::from_secs(5)).await.unwrap();
        assert_eq!(interrupted.status_str(), status::FAILED);
        let queued = state.jobs.wait(queued["id"].as_str().unwrap(), Duration::from_secs(5)).await.unwrap();
        assert_eq!(queued.status_str(), status::COMPLETED);
    }

    /// Issue #34: the queue is capped; the cap frees as jobs finish, and jobs
    /// resumed at start-up are never refused.
    #[tokio::test]
    async fn the_queue_refuses_jobs_past_its_cap_and_frees_places() {
        let app = crate::testing::TestApp::new().await;
        let jobs = &app.state.jobs;
        jobs.register("zz_gate", RetryPolicy::once(), |_ctx, args| async move {
            tokio::time::sleep(Duration::from_millis(args["ms"].as_u64().unwrap_or(0))).await;
            Ok(json!({}))
        });
        jobs.set_max_pending(2);
        let slow = jobs.submit("zz_gate", json!({ "ms": 60_000 })).await.unwrap();
        let quick = jobs.submit("zz_gate", json!({ "ms": 0 })).await.unwrap();
        let before = app.state.db.query("SELECT id FROM command", Params::new()).await.unwrap().len();
        let refused = jobs.submit("zz_gate", json!({ "ms": 0 })).await.unwrap_err();
        assert!(matches!(refused, AppError::RateLimit(_)), "{refused:?}");
        assert!(refused.is_retryable());
        assert_eq!(app.state.db.query("SELECT id FROM command", Params::new()).await.unwrap().len(), before);

        // A finished job and a canceled one both give their place back.
        assert_eq!(app.wait_for_job(&quick).await, status::COMPLETED);
        let again = jobs.submit("zz_gate", json!({ "ms": 0 })).await.unwrap();
        assert_eq!(app.wait_for_job(&again).await, status::COMPLETED);
        assert!(jobs.cancel(&slow).await.unwrap());
        for _ in 0..200 {
            if jobs.pending() == 0 {
                break;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        assert_eq!(jobs.pending(), 0);
    }

    /// Issue #34: finished jobs past the retention window are deleted;
    /// recent, unfinished and still-linked ones are kept.
    #[tokio::test]
    async fn old_finished_jobs_are_pruned() {
        let app = crate::testing::TestApp::new().await;
        let db = &app.state.db;
        let job = |status: &str| {
            Doc::new().set("app", APP_NAME).set("name", "noop").set("status", status).set("args", json!({ "big": "x" }))
        };
        let id = |row: Json| row["id"].as_str().unwrap().to_string();
        let old_done = id(db.create("command", job(status::COMPLETED)).await.unwrap());
        let old_failed = id(db.create("command", job(status::FAILED)).await.unwrap());
        let old_canceled = id(db.create("command", job(status::CANCELED)).await.unwrap());
        let old_new = id(db.create("command", job(status::NEW)).await.unwrap());
        let old_running = id(db.create("command", job(status::RUNNING)).await.unwrap());
        let old_queued = id(db.create("command", job("queued")).await.unwrap());
        let old_linked_source = id(db.create("command", job(status::COMPLETED)).await.unwrap());
        let old_linked_meeting = id(db.create("command", job(status::COMPLETED)).await.unwrap());
        let old_linked_episode = id(db.create("command", job(status::FAILED)).await.unwrap());
        let recent_done = id(db.create("command", job(status::COMPLETED)).await.unwrap());
        for old in [&old_done, &old_failed, &old_canceled, &old_new, &old_running, &old_queued, &old_linked_source, &old_linked_meeting, &old_linked_episode] {
            db.query("UPDATE $id SET created = time::now() - 41d, updated = time::now() - 40d", Params::new().record("id", old))
                .await
                .unwrap();
        }
        db.create("source", Doc::new().record("command", &old_linked_source)).await.unwrap();
        db.create("meeting", Doc::new().set("summary_job", &old_linked_meeting)).await.unwrap();
        db.query("CREATE episode SET name = 'e', episode_profile = {}, speaker_profile = {}, command = $command", Params::new().record("command", &old_linked_episode))
            .await
            .unwrap();

        assert_eq!(app.state.jobs.prune_finished(30).await.unwrap(), 3);
        for gone in [&old_done, &old_failed, &old_canceled] {
            assert!(app.state.jobs.get(gone).await.unwrap().is_none(), "{gone} pruned");
        }
        for kept in [&old_new, &old_running, &old_queued, &recent_done] {
            let record = app.state.jobs.get(kept).await.unwrap().unwrap_or_else(|| panic!("{kept} kept"));
            assert_eq!(record.args, Some(json!({ "big": "x" })), "{kept} untouched");
        }
        for linked in [&old_linked_source, &old_linked_meeting, &old_linked_episode] {
            let record = app.state.jobs.get(linked).await.unwrap().unwrap_or_else(|| panic!("{linked} kept"));
            assert!(record.is_terminal());
            assert_eq!(record.args, Some(json!({})), "{linked} slimmed");
        }
        // Nothing left to do, and 0 days means keep forever.
        assert_eq!(app.state.jobs.prune_finished(30).await.unwrap(), 0);
        assert_eq!(app.state.jobs.prune_finished(0).await.unwrap(), 0);
    }

    #[tokio::test]
    async fn a_running_job_can_be_canceled() {
        let (state, _dir) = started_state().await;
        state.jobs.register("sleepy", RetryPolicy::once(), |_ctx, _args| async move {
            tokio::time::sleep(Duration::from_secs(60)).await;
            Ok(Json::Null)
        });
        let id = state.jobs.submit("sleepy", Json::Null).await.unwrap();
        // Let it start.
        for _ in 0..200 {
            if state.jobs.get(&id).await.unwrap().unwrap().status_str() == status::RUNNING {
                break;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        assert!(state.jobs.cancel(&id).await.unwrap());
        let record = state.jobs.get(&id).await.unwrap().unwrap();
        assert_eq!(record.status_str(), status::CANCELED);
        assert!(!state.jobs.cancel(&id).await.unwrap());
    }
}
