//! Shared plumbing of the notebook generators (explainers, infographics): the
//! Rust counterpart of `api/generation_jobs.py`, plus the parts of the two
//! Python commands they have in common.
//!
//! A generator runs as a background job whose args carry `notebook_id` and
//! `requested_at` and whose result carries `success`, `note_id`, `title` and
//! `error_message`. The notes pane lists those jobs so one that fails stays
//! visible with its reason instead of vanishing.

use std::time::Instant;

use chrono::{DateTime, Duration, NaiveDateTime, SecondsFormat, Utc};
use serde::{Deserialize, Deserializer, Serialize};
use serde_json::json;

use crate::db::{Json, Params};
use crate::domain::{DefaultModelType, Note, Notebook};
use crate::error::{AppError, AppResult};
use crate::jobs::APP_NAME;
use crate::services::notes::store_note_file;
use crate::state::AppState;

/// Finished jobs older than this (hours) drop out of the notes pane.
pub const RECENT_JOBS_WINDOW_HOURS: i64 = 24;
/// A job still queued or running after this many minutes was orphaned
/// (worker restarted or died mid-job) and is shown as failed, so the pane
/// stops polling and the row can be dismissed.
pub const STALE_JOB_AFTER_MINUTES: i64 = 30;
/// At most this many jobs are listed.
pub const MAX_JOBS: usize = 10;
/// Longest accepted `instructions` (characters), as `max_length=2000` in the
/// Python request models.
pub const MAX_INSTRUCTIONS_CHARS: usize = 2_000;
/// Longest generated note title (characters).
pub const MAX_TITLE_CHARS: usize = 200;

/// `POST /notebooks/{id}/explainers` and `/infographics` response
/// (`ExplainerJobResponse`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GenerationJobStarted {
    pub job_id: String,
    pub status: String,
    pub notebook_id: String,
}

/// One row of a generator's job list (`ExplainerJobSummary`). `status` is
/// `queued`, `running`, `completed` or `failed`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GenerationJobSummary {
    pub job_id: String,
    pub status: String,
    pub requested_at: Option<String>,
    pub note_id: Option<String>,
    pub title: Option<String>,
    pub error_message: Option<String>,
}

/// The messages a generator shows for jobs that failed without saying why.
#[derive(Debug, Clone, Copy)]
pub struct JobMessages {
    /// The job failed and recorded no reason.
    pub worker_failed: &'static str,
    /// The job was queued or running for too long.
    pub stale: &'static str,
}

/// Jobs of `command_name` for the notebook: every queued or running one, plus
/// those requested in the last 24 hours, newest first, at most [`MAX_JOBS`].
/// A job queued or running for more than 30 minutes is reported as failed.
pub async fn list_generation_jobs(
    state: &AppState,
    command_name: &str,
    notebook_id: &str,
    messages: JobMessages,
) -> AppResult<Vec<GenerationJobSummary>> {
    let rows = state
        .db
        .query(
            "SELECT id, status, args, result, error_message FROM command \
             WHERE app = $app AND name = $name AND args.notebook_id = $notebook_id",
            Params::new()
                .set("app", APP_NAME)
                .set("name", command_name)
                .set("notebook_id", notebook_id),
        )
        .await?;
    Ok(summarize_jobs(&rows, Utc::now(), messages))
}

/// The list logic of [`list_generation_jobs`] over already-loaded rows.
pub fn summarize_jobs(
    rows: &[Json],
    now: DateTime<Utc>,
    messages: JobMessages,
) -> Vec<GenerationJobSummary> {
    let window = now - Duration::hours(RECENT_JOBS_WINDOW_HOURS);
    let mut jobs: Vec<GenerationJobSummary> = rows
        .iter()
        .filter_map(|row| summarize_job(row, now, messages))
        .filter(|job| {
            matches!(job.status.as_str(), "queued" | "running")
                || after(job.requested_at.as_deref(), window)
        })
        .collect();
    // Stable, so equal stamps keep their order (as Python's sort(reverse=True)).
    jobs.sort_by(|a, b| {
        b.requested_at
            .as_deref()
            .unwrap_or("")
            .cmp(a.requested_at.as_deref().unwrap_or(""))
    });
    jobs.truncate(MAX_JOBS);
    jobs
}

/// One command row as a job summary; `None` for a job submitted before job
/// tracking existed (no `requested_at`).
pub fn summarize_job(
    row: &Json,
    now: DateTime<Utc>,
    messages: JobMessages,
) -> Option<GenerationJobSummary> {
    let requested_at = row
        .get("args")
        .and_then(|args| args.get("requested_at"))
        .and_then(Json::as_str)
        .filter(|s| !s.is_empty())?
        .to_string();
    let empty = json!({});
    let result = row
        .get("result")
        .filter(|r| r.is_object())
        .unwrap_or(&empty);
    let success = result.get("success").is_some_and(truthy);
    let raw = row.get("status").and_then(Json::as_str).unwrap_or("");
    let mut status = match raw {
        "new" => "queued",
        "running" => "running",
        "completed" if success => "completed",
        _ => "failed", // failed, canceled, or completed with success=False
    };
    let mut error = None;
    if matches!(status, "queued" | "running")
        && !after(
            Some(&requested_at),
            now - Duration::minutes(STALE_JOB_AFTER_MINUTES),
        )
    {
        status = "failed";
        error = Some(messages.stale.to_string());
    } else if status == "failed" {
        error = Some(
            non_empty_str(result.get("error_message"))
                .or_else(|| non_empty_str(row.get("error_message")))
                .unwrap_or(messages.worker_failed)
                .to_string(),
        );
    }
    let completed = status == "completed";
    let field = |name: &str| {
        if completed {
            result.get(name).and_then(Json::as_str).map(String::from)
        } else {
            None
        }
    };
    Some(GenerationJobSummary {
        job_id: match row.get("id") {
            Some(Json::String(id)) => id.clone(),
            Some(other) => other.to_string(),
            None => "None".to_string(),
        },
        status: status.to_string(),
        requested_at: Some(requested_at),
        note_id: field("note_id"),
        title: field("title"),
        error_message: error,
    })
}

fn non_empty_str(value: Option<&Json>) -> Option<&str> {
    value.and_then(Json::as_str).filter(|s| !s.is_empty())
}

/// Python truthiness of a JSON value.
fn truthy(value: &Json) -> bool {
    match value {
        Json::Null => false,
        Json::Bool(b) => *b,
        Json::Number(n) => n.as_f64().is_some_and(|f| f != 0.0),
        Json::String(s) => !s.is_empty(),
        Json::Array(a) => !a.is_empty(),
        Json::Object(o) => !o.is_empty(),
    }
}

/// Whether an ISO timestamp is at or after `cutoff`. A stamp without a zone
/// is taken as UTC; one that does not parse is never "after".
fn after(stamp: Option<&str>, cutoff: DateTime<Utc>) -> bool {
    parse_iso(stamp.unwrap_or("")).is_some_and(|when| when >= cutoff)
}

/// Parse the subset of ISO 8601 Python's `datetime.fromisoformat` accepts
/// that matters here: with or without offset, `T` or space separator.
fn parse_iso(stamp: &str) -> Option<DateTime<Utc>> {
    let stamp = stamp.trim();
    if let Ok(when) = DateTime::parse_from_rfc3339(stamp) {
        return Some(when.with_timezone(&Utc));
    }
    for format in [
        "%Y-%m-%dT%H:%M:%S%.f%:z",
        "%Y-%m-%d %H:%M:%S%.f%:z",
        "%Y-%m-%dT%H:%M%:z",
    ] {
        if let Ok(when) = DateTime::parse_from_str(stamp, format) {
            return Some(when.with_timezone(&Utc));
        }
    }
    for format in [
        "%Y-%m-%dT%H:%M:%S%.f",
        "%Y-%m-%d %H:%M:%S%.f",
        "%Y-%m-%dT%H:%M",
        "%Y-%m-%d %H:%M",
    ] {
        if let Ok(when) = NaiveDateTime::parse_from_str(stamp, format) {
            return Some(when.and_utc());
        }
    }
    chrono::NaiveDate::parse_from_str(stamp, "%Y-%m-%d")
        .ok()
        .and_then(|d| d.and_hms_opt(0, 0, 0))
        .map(|d| d.and_utc())
}

/// The current time as Python's `datetime.now(timezone.utc).isoformat()`
/// writes it (`2026-01-02T03:04:05.123456+00:00`).
pub fn requested_now() -> String {
    Utc::now().to_rfc3339_opts(SecondsFormat::Micros, false)
}

// ---------------------------------------------------------------------------
// Request helpers
// ---------------------------------------------------------------------------

/// `instructions` with FastAPI's `max_length=2000` (a longer value is a 422).
pub(crate) fn instructions_field<'de, D>(deserializer: D) -> Result<Option<String>, D::Error>
where
    D: Deserializer<'de>,
{
    let value = Option::<String>::deserialize(deserializer)?;
    if let Some(text) = &value {
        if text.chars().count() > MAX_INSTRUCTIONS_CHARS {
            return Err(serde::de::Error::custom(format!(
                "instructions: String should have at most {MAX_INSTRUCTIONS_CHARS} characters"
            )));
        }
    }
    Ok(value)
}

/// `(instructions or "").strip() or None`.
pub(crate) fn clean_instructions(instructions: Option<&str>) -> Option<String> {
    instructions
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(String::from)
}

/// Load the notebook and require at least one source (`no_sources` is the
/// 400 detail otherwise). Missing notebook → 404.
pub(crate) async fn notebook_with_sources(
    state: &AppState,
    notebook_id: &str,
    no_sources: &str,
) -> AppResult<Notebook> {
    let notebook = Notebook::get(state, notebook_id).await?;
    if notebook.sources(state, false).await?.is_empty() {
        return Err(AppError::invalid(no_sources));
    }
    Ok(notebook)
}

/// Queue a generator job. Any failure to queue is a 500 with `failed_detail`.
pub(crate) async fn submit_generation(
    state: &AppState,
    command_name: &str,
    notebook_id: &str,
    args: Json,
    failed_detail: &str,
) -> AppResult<GenerationJobStarted> {
    match state.jobs.submit(command_name, args).await {
        Ok(job_id) => Ok(GenerationJobStarted {
            job_id,
            status: "queued".into(),
            notebook_id: notebook_id.to_string(),
        }),
        // A full queue is the caller's cue to retry later: keep the 429.
        Err(e @ AppError::RateLimit(_)) => Err(e),
        Err(e) => {
            tracing::error!("Failed to submit {command_name} job: {e}");
            Err(AppError::http(
                axum::http::StatusCode::INTERNAL_SERVER_ERROR,
                failed_detail,
            ))
        }
    }
}

// ---------------------------------------------------------------------------
// Job helpers
// ---------------------------------------------------------------------------

/// A positive integer from the environment, or `default` when unset or bad.
pub(crate) fn int_env(name: &str, default: usize) -> usize {
    match std::env::var(name) {
        Ok(raw) if !raw.trim().is_empty() => match raw.trim().parse::<i64>() {
            Ok(value) if value > 0 => usize::try_from(value).unwrap_or(default),
            _ => default,
        },
        _ => default,
    }
}

/// The first `n` characters of `s`.
pub(crate) fn char_prefix(s: &str, n: usize) -> &str {
    match s.char_indices().nth(n) {
        Some((at, _)) => &s[..at],
        None => s,
    }
}

/// A download name from the note title: letters, digits, `-` and `_` kept,
/// runs of anything else turned into single dashes, at most 80 characters.
pub(crate) fn attachment_file_name(title: &str, fallback: &str, ext: &str) -> String {
    let safe: String = title
        .chars()
        .map(|c| {
            if c.is_alphanumeric() || " -_".contains(c) {
                c
            } else {
                ' '
            }
        })
        .collect();
    let joined = safe.split_whitespace().collect::<Vec<_>>().join("-");
    let safe = char_prefix(&joined, 80).trim_matches('-');
    format!("{}{ext}", if safe.is_empty() { fallback } else { safe })
}

/// One call to the default chat model (the large-context model for big
/// inputs) with thinking tags stripped from the reply.
pub(crate) async fn generate_reply(
    state: &AppState,
    system: &str,
    human: &str,
    max_tokens: usize,
) -> AppResult<String> {
    let messages = [
        nn_llm::ChatMessage::system(system),
        nn_llm::ChatMessage::user(human),
    ];
    let payload = format!("{system}\n{human}");
    let model =
        super::context::provision_language_model(state, &payload, None, DefaultModelType::Chat)
            .await?;
    let options = nn_llm::ChatOptions::default()
        .with_max_tokens(u32::try_from(max_tokens).unwrap_or(u32::MAX));
    let reply = model.chat(&messages, &options).await?;
    Ok(nn_content::clean_thinking_content(&reply.content))
}

/// A generated page or image to save as a note.
pub(crate) struct GeneratedNote<'a> {
    pub title: String,
    pub content: String,
    pub explainer: Option<bool>,
    pub file_name: String,
    pub ext: &'a str,
    pub mime: &'a str,
    pub bytes: Vec<u8>,
}

/// Store the file, save the note and file it in the notebook. If saving or
/// filing fails, the note and the stored file are removed again.
pub(crate) async fn save_generated_note(
    state: &AppState,
    notebook_id: &str,
    generated: GeneratedNote<'_>,
) -> AppResult<Note> {
    let stored = store_note_file(&state.config, &generated.bytes, generated.ext).await?;
    let mut note = Note {
        title: Some(generated.title),
        content: Some(generated.content),
        note_type: Some("ai".into()),
        explainer: generated.explainer,
        attachment_name: Some(generated.file_name),
        attachment_file: Some(stored.clone()),
        attachment_mime: Some(generated.mime.to_string()),
        attachment_size: Some(i64::try_from(generated.bytes.len()).unwrap_or(i64::MAX)),
        ..Note::default()
    };
    let outcome = match note.save(state).await {
        Ok(_) => note.add_to_notebook(state, notebook_id).await,
        Err(e) => Err(e),
    };
    if let Err(e) = outcome {
        if !note.id.is_empty() {
            if let Err(cleanup) = note.delete(state).await {
                tracing::warn!("Failed to remove orphan generated note: {cleanup}");
            }
        }
        crate::domain::note::remove_note_file(&state.config, &stored);
        return Err(e);
    }
    Ok(note)
}

/// What a generator job returns (`GenerateExplainerOutput` /
/// `GenerateInfographicOutput`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GenerationOutput {
    pub success: bool,
    pub notebook_id: String,
    pub note_id: Option<String>,
    pub title: Option<String>,
    pub processing_time: f64,
    pub error_message: Option<String>,
}

/// Turn a generator run into its job result. Failures are reported in the
/// result (`success: false`) rather than failing the job, as the Python
/// commands do; errors with no user-facing meaning get `unexpected`.
pub(crate) fn generation_output(
    notebook_id: &str,
    started: Instant,
    outcome: AppResult<Note>,
    unexpected: &str,
) -> Json {
    let processing_time = started.elapsed().as_secs_f64();
    let output = match outcome {
        Ok(note) => GenerationOutput {
            success: true,
            notebook_id: notebook_id.to_string(),
            note_id: Some(note.id),
            title: note.title,
            processing_time,
            error_message: None,
        },
        Err(e) => {
            let message = match &e {
                AppError::Internal(_) | AppError::FileOperation(_) => {
                    tracing::error!("Generation for {notebook_id} failed unexpectedly: {e}");
                    unexpected.to_string()
                }
                _ => {
                    tracing::warn!("Generation for {notebook_id} failed: {e}");
                    e.detail()
                }
            };
            GenerationOutput {
                success: false,
                notebook_id: notebook_id.to_string(),
                note_id: None,
                title: None,
                processing_time,
                error_message: Some(message),
            }
        }
    };
    serde_json::to_value(output)
        .unwrap_or_else(|_| json!({ "success": false, "notebook_id": notebook_id }))
}

#[cfg(test)]
mod tests {
    use super::*;

    const MESSAGES: JobMessages = JobMessages {
        worker_failed: "worker failed",
        stale: "stale",
    };

    fn cmd(id: &str, status: &str, requested_at: &str, result: Json, error: &str) -> Json {
        json!({
            "id": id,
            "status": status,
            "args": { "notebook_id": "notebook:n1", "requested_at": requested_at },
            "result": result,
            "error_message": error,
        })
    }

    #[test]
    fn failures_stay_visible_and_stale_jobs_fail() {
        let now = Utc::now();
        let iso = |d: Duration| (now - d).to_rfc3339_opts(SecondsFormat::Micros, false);
        let recent = iso(Duration::minutes(5));
        let old = iso(Duration::days(2));
        let rows = vec![
            cmd(
                "command:done",
                "completed",
                &recent,
                json!({ "success": true, "note_id": "note:1", "title": "Tides" }),
                "",
            ),
            cmd(
                "command:nomodel",
                "completed",
                &iso(Duration::minutes(1)),
                json!({ "success": false, "error_message": "No model configured" }),
                "",
            ),
            cmd("command:crash", "failed", &recent, Json::Null, "boom"),
            cmd(
                "command:queued",
                "new",
                &iso(Duration::zero()),
                Json::Null,
                "",
            ),
            cmd(
                "command:oldfail",
                "completed",
                &old,
                json!({ "success": false, "error_message": "old" }),
                "",
            ),
            cmd("command:oldqueued", "new", &old, Json::Null, ""),
            cmd(
                "command:stuck",
                "running",
                &iso(Duration::minutes(45)),
                Json::Null,
                "",
            ),
            cmd("command:silent", "failed", &recent, Json::Null, ""),
            json!({ "id": "command:legacy", "status": "completed", "args": { "notebook_id": "notebook:n1" }, "result": {} }),
        ];
        let jobs = summarize_jobs(&rows, now, MESSAGES);
        let ids: Vec<&str> = jobs.iter().map(|j| j.job_id.as_str()).collect();
        assert_eq!(ids[0], "command:queued", "newest first");
        let mut sorted = ids.clone();
        sorted.sort();
        assert_eq!(
            sorted,
            [
                "command:crash",
                "command:done",
                "command:nomodel",
                "command:queued",
                "command:silent",
                "command:stuck"
            ]
        );
        let get = |id: &str| jobs.iter().find(|j| j.job_id == id).unwrap();
        assert_eq!(get("command:done").status, "completed");
        assert_eq!(get("command:done").note_id.as_deref(), Some("note:1"));
        assert_eq!(get("command:done").title.as_deref(), Some("Tides"));
        assert_eq!(get("command:nomodel").status, "failed");
        assert_eq!(
            get("command:nomodel").error_message.as_deref(),
            Some("No model configured")
        );
        assert_eq!(get("command:nomodel").note_id, None);
        assert_eq!(get("command:crash").error_message.as_deref(), Some("boom"));
        assert_eq!(
            get("command:silent").error_message.as_deref(),
            Some("worker failed")
        );
        assert_eq!(get("command:queued").status, "queued");
        assert_eq!(get("command:queued").error_message, None);
        assert_eq!(get("command:stuck").status, "failed");
        assert_eq!(get("command:stuck").error_message.as_deref(), Some("stale"));
    }

    #[test]
    fn list_is_capped_and_naive_stamps_are_utc() {
        let now = Utc::now();
        let rows: Vec<Json> = (0..15)
            .map(|i| {
                let at = (now - Duration::minutes(i))
                    .naive_utc()
                    .format("%Y-%m-%dT%H:%M:%S%.f")
                    .to_string();
                cmd(&format!("command:{i}"), "new", &at, Json::Null, "")
            })
            .collect();
        let jobs = summarize_jobs(&rows, now, MESSAGES);
        assert_eq!(jobs.len(), MAX_JOBS);
        assert!(jobs.iter().all(|j| j.status == "queued"), "{jobs:?}");
        assert_eq!(jobs[0].job_id, "command:0");
        // Unparseable stamps count as old: a queued one is stale, then ages out.
        let bad = [cmd("command:bad", "new", "yesterday", Json::Null, "")];
        assert!(summarize_jobs(&bad, now, MESSAGES).is_empty());
    }

    #[test]
    fn helpers_match_python() {
        assert_eq!(
            attachment_file_name("How & why tides work", "explainer", ".html"),
            "How-why-tides-work.html"
        );
        assert_eq!(
            attachment_file_name("???", "explainer", ".html"),
            "explainer.html"
        );
        assert_eq!(
            attachment_file_name("Tides & the moon", "infographic", ".svg"),
            "Tides-the-moon.svg"
        );
        assert_eq!(
            clean_instructions(Some("  focus  ")).as_deref(),
            Some("focus")
        );
        assert_eq!(clean_instructions(Some("   ")), None);
        assert_eq!(char_prefix("héllo", 2), "hé");
        assert!(parse_iso("2020-01-01T00:00:00+00:00").is_some());
        assert!(parse_iso("2020-01-01T00:00:00Z").is_some());
        assert!(parse_iso(&requested_now()).is_some());
        assert!(requested_now().ends_with("+00:00"));
    }

    #[test]
    fn int_env_ignores_bad_values() {
        let name = "NOVA_TEST_GENERATION_INT_ENV";
        for (value, expected) in [("20000", 20_000), ("", 7), ("lots", 7), ("0", 7), ("-5", 7)] {
            std::env::set_var(name, value);
            assert_eq!(int_env(name, 7), expected, "{value:?}");
        }
        std::env::remove_var(name);
        assert_eq!(int_env(name, 7), 7);
    }
}
