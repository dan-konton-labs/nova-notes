//! Embeddings for vector search: notes, insights and source chunks, and the
//! rebuild that re-embeds everything.
//!
//! The Rust counterpart of `api/routers/embedding.py`,
//! `api/routers/embedding_rebuild.py` and `commands/embedding_commands.py`.
//!
//! As in Python, the `embed_*` jobs report a missing record or a record
//! without content as a completed job with `success: false` (a permanent
//! outcome, never retried); configuration and provider errors fail the job,
//! and transient ones are retried.

use std::future::Future;
use std::time::{Duration, Instant};

use axum::http::StatusCode;
use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};

use super::notebooks::{ensure_table, timestamp};
use crate::ai::{chunk_config, generate_embedding, generate_embeddings};
use crate::db::{Doc, Json, Params};
use crate::domain::{Note, Source, SourceEmbedding, SourceInsight};
use crate::error::{AppError, AppResult};
use crate::jobs::{status, JobContext, JobQueue, RetryPolicy};
use crate::state::AppState;

/// Rows per `INSERT` when storing source chunks.
const INSERT_BATCH_SIZE: usize = 50;

pub const NO_EMBEDDING_MODEL: &str = "No embedding model configured. Please configure one in the Models section.";

fn embed_retry() -> RetryPolicy {
    RetryPolicy::attempts(5, Duration::from_secs(1), Duration::from_secs(60))
}

pub fn register_jobs(jobs: &JobQueue) {
    jobs.register("embed_note", embed_retry(), embed_note_job);
    jobs.register("embed_insight", embed_retry(), embed_insight_job);
    jobs.register("embed_source", embed_retry(), embed_source_job);
    // A coordinator: its embed_* jobs retry on their own.
    jobs.register("rebuild_embeddings", RetryPolicy::once(), rebuild_embeddings_job);
}

// ---------------------------------------------------------------------------
// POST /embed
// ---------------------------------------------------------------------------

/// Body of `POST /embed`.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct EmbedRequest {
    pub item_id: String,
    /// `source` or `note` (any case).
    pub item_type: String,
    #[serde(default)]
    pub async_processing: bool,
}

/// Answer of `POST /embed` (`EmbedResponse`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EmbedResult {
    pub success: bool,
    pub message: String,
    pub item_id: String,
    pub item_type: String,
    pub command_id: Option<String>,
}

/// Queue the embedding of a source or a note.
pub async fn embed_item(state: &AppState, request: &EmbedRequest) -> AppResult<EmbedResult> {
    if state.models.embedding_opt(state).await?.is_none() {
        return Err(AppError::bad_request(NO_EMBEDDING_MODEL));
    }
    let item_id = request.item_id.clone();
    let item_type = request.item_type.to_lowercase();
    if item_type != "source" && item_type != "note" {
        return Err(AppError::bad_request("Item type must be either 'source' or 'note'"));
    }
    let result = |message: &str, command_id: Option<String>| EmbedResult {
        success: true,
        message: message.to_string(),
        item_id: item_id.clone(),
        item_type: item_type.clone(),
        command_id,
    };

    if request.async_processing {
        let (name, args) = if item_type == "source" {
            ("embed_source", json!({ "source_id": item_id }))
        } else {
            ("embed_note", json!({ "note_id": item_id }))
        };
        let command_id = state.jobs.submit(name, args).await.map_err(|e| match e {
            // A full queue is the caller's cue to retry later: keep the 429.
            AppError::RateLimit(_) => e,
            e => AppError::http(StatusCode::INTERNAL_SERVER_ERROR, format!("Failed to queue embedding: {e}")),
        })?;
        tracing::info!("Submitted async {name} command: {command_id}");
        return Ok(result("Embedding queued for background processing", Some(command_id)));
    }

    let missing = format!("{} not found", request.item_type);
    let not_found = |e: AppError| if e.is_not_found() { AppError::not_found(missing.clone()) } else { e };
    if item_type == "source" {
        ensure_table(&item_id, Source::TABLE, &missing)?;
        let source = Source::get(state, &item_id).await.map_err(not_found)?;
        let command_id = source.vectorize(state).await.map_err(|e| match e {
            // Python lets the "no text" ValueError reach its catch-all.
            AppError::InvalidInput(m) => {
                AppError::http(StatusCode::INTERNAL_SERVER_ERROR, format!("Error embedding content: {m}"))
            }
            other => other,
        })?;
        Ok(result("Source embedding job submitted", Some(command_id)))
    } else {
        ensure_table(&item_id, Note::TABLE, &missing)?;
        let mut note = Note::get(state, &item_id).await.map_err(not_found)?;
        let command_id = note.save(state).await?;
        if command_id.is_none() && note.content.as_deref().is_some_and(|c| !c.trim().is_empty()) {
            return Err(AppError::http(StatusCode::INTERNAL_SERVER_ERROR, "Failed to submit note embedding job"));
        }
        Ok(result("Note embedding job submitted", command_id))
    }
}

// ---------------------------------------------------------------------------
// The embed_* jobs
// ---------------------------------------------------------------------------

/// Run `work`; a permanent "record missing / nothing to embed" outcome
/// becomes `Ok(Err(message))`, everything else propagates.
async fn embed_record<F>(ctx: &JobContext, kind: &str, record_id: &str, work: F) -> AppResult<(Map<String, Value>, f64, Option<String>)>
where
    F: Future<Output = AppResult<Map<String, Value>>>,
{
    let started = Instant::now();
    tracing::info!("Starting embedding for {kind}: {record_id}");
    match work.await {
        Ok(extra) => {
            let elapsed = started.elapsed().as_secs_f64();
            tracing::info!("Successfully embedded {kind} {record_id} in {elapsed:.2}s");
            Ok((extra, elapsed, None))
        }
        Err(e @ (AppError::NotFound(_) | AppError::InvalidInput(_))) => {
            tracing::error!("Failed to embed {kind} {record_id} (command: {}): {e}", ctx.command_id);
            Ok((Map::new(), started.elapsed().as_secs_f64(), Some(e.to_string())))
        }
        Err(e) => {
            if e.is_retryable() {
                tracing::warn!("Transient error embedding {kind} {record_id} (attempt {}): {e}", ctx.attempt);
            }
            Err(e)
        }
    }
}

fn record_arg(args: &Value, field: &str) -> AppResult<String> {
    args.get(field)
        .and_then(Json::as_str)
        .filter(|id| !id.is_empty())
        .map(String::from)
        .ok_or_else(|| AppError::invalid(format!("Missing {field}")))
}

/// Embed a note's or an insight's content as one vector on the record.
async fn embed_content_field(state: &AppState, label: &str, record_id: &str, content: Option<&str>) -> AppResult<Map<String, Value>> {
    let content = content.filter(|c| !c.trim().is_empty()).ok_or_else(|| {
        AppError::invalid(format!("{label} '{record_id}' has no content to embed"))
    })?;
    let embedding = generate_embedding(state, content).await?;
    // Never overlap a user's edit of the same record (see `Db::lock_record`).
    let _write = state.db.lock_record(record_id).await;
    state
        .db
        .query(
            "UPDATE $record_id SET embedding = $embedding WHERE content = $embedded_text",
            Params::new().record("record_id", record_id).set("embedding", embedding).set("embedded_text", content),
        )
        .await?;
    Ok(Map::new())
}

async fn embed_note_job(ctx: JobContext, args: Value) -> AppResult<Value> {
    let note_id = record_arg(&args, "note_id")?;
    let state = ctx.state.clone();
    let work = async {
        let note = Note::get(&state, &note_id).await?;
        embed_content_field(&state, "Note", &note_id, note.content.as_deref()).await
    };
    let (_, processing_time, error_message) = embed_record(&ctx, "note", &note_id, work).await?;
    Ok(json!({
        "success": error_message.is_none(),
        "note_id": note_id,
        "processing_time": processing_time,
        "error_message": error_message,
    }))
}

async fn embed_insight_job(ctx: JobContext, args: Value) -> AppResult<Value> {
    let insight_id = record_arg(&args, "insight_id")?;
    let state = ctx.state.clone();
    let work = async {
        let insight = SourceInsight::get(&state, &insight_id).await?;
        embed_content_field(&state, "Insight", &insight_id, Some(&insight.content)).await
    };
    let (_, processing_time, error_message) = embed_record(&ctx, "insight", &insight_id, work).await?;
    Ok(json!({
        "success": error_message.is_none(),
        "insight_id": insight_id,
        "processing_time": processing_time,
        "error_message": error_message,
    }))
}

/// Replace a source's chunks: delete the old ones, chunk the full text by
/// content type, embed every chunk and insert them. A failed insert leaves
/// no partial set behind. Returns the number of chunks stored.
pub async fn embed_source_chunks(state: &AppState, source_id: &str) -> AppResult<usize> {
    let source = Source::get(state, source_id).await?;
    let text = source
        .full_text
        .as_deref()
        .filter(|t| !t.trim().is_empty())
        .ok_or_else(|| AppError::invalid(format!("Source '{source_id}' has no text to embed")))?;

    delete_source_chunks(state, source_id).await?;

    let file_path = source.asset.as_ref().and_then(|a| a.file_path.as_deref());
    let content_type = nn_content::detect_content_type(text, file_path);
    let chunks = nn_content::chunk_text_with(text, Some(content_type), file_path, &chunk_config());
    if chunks.is_empty() {
        return Err(AppError::invalid("No chunks created after splitting text"));
    }
    tracing::info!("Created {} chunks for source {source_id} ({} content)", chunks.len(), content_type.as_str());

    let model = state.models.embedding(state).await?;
    let embeddings = generate_embeddings(state, &model, &chunks).await?;
    if embeddings.len() != chunks.len() {
        return Err(AppError::invalid(format!(
            "Embedding count mismatch: got {} embeddings for {} chunks",
            embeddings.len(),
            chunks.len()
        )));
    }

    let docs: Vec<Doc> = chunks
        .iter()
        .zip(&embeddings)
        .enumerate()
        .map(|(order, (chunk, embedding))| {
            Doc::new().record("source", source_id).set("order", order).set("content", chunk).set("embedding", embedding)
        })
        .collect();
    if let Err(e) = state.db.insert_many(SourceEmbedding::TABLE, docs, INSERT_BATCH_SIZE).await {
        // Batches are not atomic together: a partial set would look embedded
        // while search sees only part of the source.
        if let Err(cleanup) = delete_source_chunks(state, source_id).await {
            tracing::error!("Failed to clean up partial embeddings for source {source_id}: {cleanup}");
        }
        return Err(e);
    }
    Ok(chunks.len())
}

async fn delete_source_chunks(state: &AppState, source_id: &str) -> AppResult<()> {
    state
        .db
        .query("DELETE source_embedding WHERE source = $source_id", Params::new().record("source_id", source_id))
        .await
        .map(|_| ())
}

async fn embed_source_job(ctx: JobContext, args: Value) -> AppResult<Value> {
    let source_id = record_arg(&args, "source_id")?;
    let state = ctx.state.clone();
    let work = async {
        let chunks = embed_source_chunks(&state, &source_id).await?;
        let mut extra = Map::new();
        extra.insert("chunks_created".into(), json!(chunks));
        Ok(extra)
    };
    let (extra, processing_time, error_message) = embed_record(&ctx, "source", &source_id, work).await?;
    Ok(json!({
        "success": error_message.is_none(),
        "source_id": source_id,
        "chunks_created": extra.get("chunks_created").cloned().unwrap_or(json!(0)),
        "processing_time": processing_time,
        "error_message": error_message,
    }))
}

// ---------------------------------------------------------------------------
// Rebuild
// ---------------------------------------------------------------------------

/// What a rebuild re-embeds.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum RebuildMode {
    /// Only items that already have embeddings.
    Existing,
    /// Everything with content.
    All,
}

fn default_true() -> bool {
    true
}

/// Body of `POST /embeddings/rebuild` (`RebuildRequest`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RebuildRequest {
    pub mode: RebuildMode,
    #[serde(default = "default_true")]
    pub include_sources: bool,
    #[serde(default = "default_true")]
    pub include_notes: bool,
    #[serde(default = "default_true")]
    pub include_insights: bool,
}

/// Answer of `POST /embeddings/rebuild` (`RebuildResponse`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RebuildStarted {
    pub command_id: String,
    pub total_items: i64,
    pub message: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RebuildProgress {
    pub processed: i64,
    pub total: i64,
    pub percentage: f64,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct RebuildStats {
    pub sources: i64,
    pub notes: i64,
    pub insights: i64,
    pub failed: i64,
}

/// Answer of `GET /embeddings/rebuild/{id}/status` (`RebuildStatusResponse`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RebuildStatus {
    pub command_id: String,
    pub status: String,
    pub progress: Option<RebuildProgress>,
    pub stats: Option<RebuildStats>,
    pub started_at: Option<String>,
    pub completed_at: Option<String>,
    pub error_message: Option<String>,
}

async fn count(state: &AppState, sql: &str) -> AppResult<i64> {
    let rows = state.db.query(sql, Params::new()).await?;
    Ok(rows
        .first()
        .map(|row| match row {
            Json::Object(map) => map.get("count").and_then(Json::as_i64).unwrap_or(0),
            other => other.as_i64().unwrap_or(0),
        })
        .unwrap_or(0))
}

/// Rough number of items a rebuild will touch (counted before it runs).
pub async fn estimate_rebuild(state: &AppState, request: &RebuildRequest) -> AppResult<i64> {
    let existing = request.mode == RebuildMode::Existing;
    let mut total = 0;
    if request.include_sources {
        total += if existing {
            count(
                state,
                "RETURN array::len(array::distinct(SELECT VALUE source FROM source_embedding WHERE embedding != NONE AND array::len(embedding) > 0))",
            )
            .await?
        } else {
            count(state, "SELECT count() AS count FROM source WHERE full_text != NONE GROUP ALL").await?
        };
    }
    if request.include_notes {
        total += if existing {
            count(state, "SELECT count() AS count FROM note WHERE embedding != NONE AND array::len(embedding) > 0 GROUP ALL").await?
        } else {
            count(state, "SELECT count() AS count FROM note WHERE content != NONE GROUP ALL").await?
        };
    }
    if request.include_insights {
        total += if existing {
            count(state, "SELECT count() AS count FROM source_insight WHERE embedding != NONE AND array::len(embedding) > 0 GROUP ALL")
                .await?
        } else {
            count(state, "SELECT count() AS count FROM source_insight GROUP ALL").await?
        };
    }
    Ok(total)
}

/// Queue the `rebuild_embeddings` job.
pub async fn start_rebuild(state: &AppState, request: &RebuildRequest) -> AppResult<RebuildStarted> {
    let total_items = estimate_rebuild(state, request).await?;
    let command_id = state.jobs.submit("rebuild_embeddings", serde_json::to_value(request)?).await?;
    tracing::info!("Submitted rebuild command: {command_id}");
    Ok(RebuildStarted {
        command_id,
        total_items,
        message: format!("Rebuild operation started. Estimated {total_items} items to process."),
    })
}

/// Progress of a rebuild job.
pub async fn rebuild_status(state: &AppState, command_id: &str) -> AppResult<RebuildStatus> {
    let record = state
        .jobs
        .get(command_id)
        .await
        .ok()
        .flatten()
        .ok_or_else(|| AppError::not_found("Rebuild command not found"))?;
    let result = record.result.as_ref().filter(|r| r.is_object());
    let int = |key: &str| result.and_then(|r| r.get(key)).and_then(Json::as_i64);
    let progress = match (int("total_items"), int("jobs_submitted")) {
        (Some(total), Some(submitted)) => Some(RebuildProgress {
            processed: submitted,
            total,
            percentage: if total > 0 { ((submitted as f64 / total as f64) * 10_000.0).round() / 100.0 } else { 0.0 },
        }),
        _ => None,
    };
    let stats = result.map(|_| RebuildStats {
        sources: int("sources_submitted").unwrap_or(0),
        notes: int("notes_submitted").unwrap_or(0),
        insights: int("insights_submitted").unwrap_or(0),
        failed: int("failed_submissions").unwrap_or(0),
    });
    let error_message = (record.status_str() == status::FAILED).then(|| {
        result
            .and_then(|r| r.get("error_message"))
            .and_then(Json::as_str)
            .map(String::from)
            .or_else(|| record.error_message.clone())
            .unwrap_or_else(|| "Unknown error".to_string())
    });
    Ok(RebuildStatus {
        command_id: command_id.to_string(),
        status: record.status_str().to_string(),
        progress,
        stats,
        started_at: record.created.map(|d| timestamp(Some(d))),
        completed_at: record.updated.map(|d| timestamp(Some(d))),
        error_message,
    })
}

/// Ids of the items a rebuild re-embeds.
async fn collect_ids(state: &AppState, sql: &str) -> AppResult<Vec<String>> {
    let rows = state.db.query(sql, Params::new()).await?;
    Ok(rows
        .into_iter()
        .filter_map(|row| match row {
            Json::String(id) => Some(id),
            Json::Object(map) => map.get("id").and_then(Json::as_str).map(String::from),
            _ => None,
        })
        .collect())
}

async fn submit_each(state: &AppState, kind: &str, command: &str, field: &str, ids: &[String]) -> (i64, i64) {
    tracing::info!("Submitting {} {kind} embedding jobs...", ids.len());
    let (mut submitted, mut failed) = (0, 0);
    for id in ids {
        match state.jobs.submit(command, json!({ field: id })).await {
            Ok(_) => submitted += 1,
            Err(e) => {
                tracing::error!("Failed to submit {command} for {id}: {e}");
                failed += 1;
            }
        }
    }
    (submitted, failed)
}

async fn rebuild(state: &AppState, request: &RebuildRequest) -> AppResult<Value> {
    if state.models.embedding_opt(state).await?.is_none() {
        return Err(AppError::config(NO_EMBEDDING_MODEL));
    }
    let existing = request.mode == RebuildMode::Existing;
    let sources = if !request.include_sources {
        Vec::new()
    } else if existing {
        collect_ids(
            state,
            "RETURN array::distinct(SELECT VALUE source FROM source_embedding WHERE embedding != NONE AND array::len(embedding) > 0)",
        )
        .await?
    } else {
        collect_ids(state, "SELECT id FROM source WHERE full_text != NONE AND string::trim(full_text) != ''").await?
    };
    let notes = if !request.include_notes {
        Vec::new()
    } else if existing {
        collect_ids(state, "SELECT id FROM note WHERE embedding != NONE AND array::len(embedding) > 0").await?
    } else {
        collect_ids(state, "SELECT id FROM note WHERE content != NONE AND string::trim(content) != ''").await?
    };
    let insights = if !request.include_insights {
        Vec::new()
    } else if existing {
        collect_ids(state, "SELECT id FROM source_insight WHERE embedding != NONE AND array::len(embedding) > 0").await?
    } else {
        collect_ids(state, "SELECT id FROM source_insight WHERE content != NONE AND string::trim(content) != ''").await?
    };
    let total_items = (sources.len() + notes.len() + insights.len()) as i64;

    let (sources_submitted, sources_failed) = submit_each(state, "source", "embed_source", "source_id", &sources).await;
    let (notes_submitted, notes_failed) = submit_each(state, "note", "embed_note", "note_id", &notes).await;
    let (insights_submitted, insights_failed) = submit_each(state, "insight", "embed_insight", "insight_id", &insights).await;
    Ok(json!({
        "success": true,
        "total_items": total_items,
        "jobs_submitted": sources_submitted + notes_submitted + insights_submitted,
        "failed_submissions": sources_failed + notes_failed + insights_failed,
        "sources_submitted": sources_submitted,
        "notes_submitted": notes_submitted,
        "insights_submitted": insights_submitted,
    }))
}

/// `rebuild_embeddings`: queue one embed job per item. Like Python, a
/// failure is reported in the result (`success: false`) rather than failing
/// the coordinator job.
async fn rebuild_embeddings_job(ctx: JobContext, args: Value) -> AppResult<Value> {
    let started = Instant::now();
    let request: RebuildRequest =
        serde_json::from_value(args).map_err(|e| AppError::invalid(format!("Invalid rebuild_embeddings input: {e}")))?;
    tracing::info!("Starting embedding rebuild with mode={:?}", request.mode);
    let mut result = match rebuild(&ctx.state, &request).await {
        Ok(result) => result,
        Err(e) => {
            tracing::error!("Rebuild embeddings failed: {e}");
            json!({
                "success": false,
                "total_items": 0,
                "jobs_submitted": 0,
                "failed_submissions": 0,
                "sources_submitted": 0,
                "notes_submitted": 0,
                "insights_submitted": 0,
                "error_message": e.to_string(),
            })
        }
    };
    result["processing_time"] = json!(started.elapsed().as_secs_f64());
    if result.get("error_message").is_none() {
        result["error_message"] = Value::Null;
    }
    Ok(result)
}
