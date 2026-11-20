//! Source insights: the output of transformations, stored per source.
//!
//! The Rust counterpart of `api/routers/insights.py`, the insight routes of
//! `api/routers/sources.py` and the `create_insight` command.

use std::time::{Duration, Instant};

use chrono::{DateTime, Timelike, Utc};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use super::notebooks::{ensure_table, not_found_as};
use super::notes::NoteView;
use super::sources::load_source;
use super::transformations::{load_transformation, TRANSFORMATION_NOT_FOUND};
use crate::db::{Doc, Json};
use crate::domain::SourceInsight;
use crate::error::{AppError, AppResult};
use crate::jobs::{JobContext, JobQueue, RetryPolicy};
use crate::state::AppState;

pub const INSIGHT_NOT_FOUND: &str = "Insight not found";

/// `create_insight`: retried up to 5 times (1-60 s backoff), mainly for
/// transaction conflicts when several transformations finish at once.
pub fn register_jobs(jobs: &JobQueue) {
    jobs.register(
        "create_insight",
        RetryPolicy::attempts(5, Duration::from_secs(1), Duration::from_secs(60)),
        create_insight_job,
    );
}

/// A timestamp the way Python's `datetime.isoformat()` writes it
/// (`2026-07-11T12:30:45+00:00`, microseconds only when non-zero).
pub fn iso_timestamp(value: DateTime<Utc>) -> String {
    if value.nanosecond() / 1000 == 0 {
        value.format("%Y-%m-%dT%H:%M:%S+00:00").to_string()
    } else {
        value.format("%Y-%m-%dT%H:%M:%S%.6f+00:00").to_string()
    }
}

/// An insight as the API reports it (`SourceInsightResponse`). Insights
/// created before migration 19 have no timestamps: those are `null`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct InsightView {
    pub id: String,
    pub source_id: String,
    pub insight_type: String,
    pub content: String,
    pub created: Option<String>,
    pub updated: Option<String>,
}

impl InsightView {
    pub fn new(insight: SourceInsight, source_id: &str) -> Self {
        Self {
            id: insight.id,
            source_id: source_id.to_string(),
            insight_type: insight.insight_type,
            content: insight.content,
            created: insight.created.map(iso_timestamp),
            updated: insight.updated.map(iso_timestamp),
        }
    }
}

/// Body of `POST /sources/{id}/insights`. `model_id` is accepted for
/// compatibility; the run uses the transformation's own model, as in Python.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct NewSourceInsight {
    pub transformation_id: String,
    #[serde(default)]
    pub model_id: Option<String>,
}

/// Answer of `POST /sources/{id}/insights` (`InsightCreationResponse`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct InsightCreation {
    /// Always `"pending"`.
    pub status: String,
    pub message: String,
    pub source_id: String,
    pub transformation_id: String,
    pub command_id: Option<String>,
}

async fn load_insight(state: &AppState, id: &str) -> AppResult<SourceInsight> {
    ensure_table(id, SourceInsight::TABLE, INSIGHT_NOT_FOUND)?;
    SourceInsight::get(state, id).await.map_err(not_found_as(INSIGHT_NOT_FOUND))
}

pub async fn get_insight(state: &AppState, id: &str) -> AppResult<InsightView> {
    let insight = load_insight(state, id).await?;
    let source_id = insight.source.clone().unwrap_or_default();
    Ok(InsightView::new(insight, &source_id))
}

pub async fn delete_insight(state: &AppState, id: &str) -> AppResult<()> {
    load_insight(state, id).await?.delete(state).await
}

/// Copy an insight into a new note (titled `"<type> from source <title>"`),
/// optionally filed in a notebook.
pub async fn save_insight_as_note(state: &AppState, id: &str, notebook_id: Option<&str>) -> AppResult<NoteView> {
    let insight = load_insight(state, id).await?;
    let notebook_id = notebook_id.filter(|n| !n.is_empty());
    let note = insight.save_as_note(state, notebook_id).await.map_err(not_found_as("Notebook not found"))?;
    Ok(NoteView::new(note, None))
}

/// A source's insights.
pub async fn list_source_insights(state: &AppState, source_id: &str) -> AppResult<Vec<InsightView>> {
    let source = load_source(state, source_id).await?;
    Ok(source.insights(state).await?.into_iter().map(|insight| InsightView::new(insight, source_id)).collect())
}

/// Queue the `run_transformation` job for a source. A source without text is
/// refused (the model could only invent an answer).
pub async fn create_source_insight(state: &AppState, source_id: &str, request: &NewSourceInsight) -> AppResult<InsightCreation> {
    let source = load_source(state, source_id).await?;
    if source.full_text.as_deref().is_none_or(|t| t.trim().is_empty()) {
        return Err(AppError::invalid("Source has no text content"));
    }
    load_transformation(state, &request.transformation_id).await.map_err(not_found_as(TRANSFORMATION_NOT_FOUND))?;
    let command_id = state
        .jobs
        .submit("run_transformation", json!({ "source_id": source_id, "transformation_id": request.transformation_id }))
        .await?;
    tracing::info!("Submitted run_transformation command {command_id} for source {source_id}");
    Ok(InsightCreation {
        status: "pending".to_string(),
        message: "Insight generation started".to_string(),
        source_id: source_id.to_string(),
        transformation_id: request.transformation_id.clone(),
        command_id: Some(command_id),
    })
}

#[derive(Debug, Deserialize)]
struct CreateInsightArgs {
    source_id: String,
    insight_type: String,
    content: String,
}

/// `create_insight`: store an insight and queue its embedding.
async fn create_insight_job(ctx: JobContext, args: Value) -> AppResult<Value> {
    let started = Instant::now();
    let args: CreateInsightArgs =
        serde_json::from_value(args).map_err(|e| AppError::invalid(format!("Invalid create_insight input: {e}")))?;
    let state = &ctx.state;
    tracing::info!("Creating insight for source {}: type={}", args.source_id, args.insight_type);

    let created = state
        .db
        .create(
            SourceInsight::TABLE,
            Doc::new().record("source", &args.source_id).set("insight_type", &args.insight_type).set("content", &args.content),
        )
        .await;
    let row = match created {
        Ok(row) => row,
        Err(e) if e.is_retryable() => {
            tracing::warn!("Transient error creating insight for source {} (attempt {}): {e}", args.source_id, ctx.attempt);
            return Err(e);
        }
        Err(e) => return Err(e),
    };
    let Some(insight_id) = row.get("id").and_then(Json::as_str).map(String::from) else {
        let message = "Failed to create insight - no ID in result";
        tracing::error!("Failed to create insight for source {} (command: {}): {message}", args.source_id, ctx.command_id);
        return Ok(json!({
            "success": false,
            "insight_id": null,
            "processing_time": started.elapsed().as_secs_f64(),
            "error_message": message,
        }));
    };

    state.jobs.submit("embed_insight", json!({ "insight_id": insight_id })).await?;
    let processing_time = started.elapsed().as_secs_f64();
    tracing::info!("Successfully created insight {insight_id} for source {} in {processing_time:.2}s", args.source_id);
    Ok(json!({
        "success": true,
        "insight_id": insight_id,
        "processing_time": processing_time,
        "error_message": null,
    }))
}
