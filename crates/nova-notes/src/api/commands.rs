//! Background job status: `/api/commands/...`.

use std::collections::BTreeMap;

use axum::extract::{Path, Query, State};
use axum::routing::get;
use axum::{Json, Router};
use serde::Deserialize;
use serde_json::{json, Value};

use crate::error::AppResult;
use crate::jobs::{CommandRecord, APP_NAME};
use crate::state::AppState;

pub fn router() -> Router<AppState> {
    Router::new()
        .route("/commands/jobs", get(list_jobs))
        .route("/commands/jobs/{job_id}", get(job_status).delete(cancel_job))
        .route("/commands/registry/debug", get(registry_debug))
}

/// The JSON shape of one job, shared with the web UI.
pub fn job_json(job_id: &str, record: Option<&CommandRecord>) -> Value {
    json!({
        "job_id": job_id,
        "status": record.map(|r| r.status_str()).unwrap_or("unknown"),
        "result": record.and_then(|r| r.result.clone()).filter(Value::is_object),
        "error_message": record.and_then(|r| r.error_message.clone()),
        "created": record.and_then(|r| r.created).map(|d| d.to_rfc3339()),
        "updated": record.and_then(|r| r.updated).map(|d| d.to_rfc3339()),
        "progress": record.and_then(|r| r.progress.clone()).filter(Value::is_object),
    })
}

async fn job_status(State(state): State<AppState>, Path(job_id): Path<String>) -> AppResult<Json<Value>> {
    // A malformed id is simply a job we do not know.
    let record = state.jobs.get(&job_id).await.unwrap_or(None);
    Ok(Json(job_json(&job_id, record.as_ref())))
}

#[derive(Debug, Deserialize)]
struct ListQuery {
    command_filter: Option<String>,
    status_filter: Option<String>,
    limit: Option<usize>,
}

async fn list_jobs(State(state): State<AppState>, Query(query): Query<ListQuery>) -> AppResult<Json<Vec<Value>>> {
    let records = state
        .jobs
        .list(query.command_filter.as_deref(), query.status_filter.as_deref(), query.limit.unwrap_or(50).min(500))
        .await?;
    Ok(Json(
        records
            .iter()
            .map(|r| {
                let mut job = job_json(&r.id, Some(r));
                job["command"] = json!(r.name);
                job
            })
            .collect(),
    ))
}

async fn cancel_job(State(state): State<AppState>, Path(job_id): Path<String>) -> AppResult<Json<Value>> {
    let cancelled = state.jobs.cancel(&job_id).await.unwrap_or(false);
    Ok(Json(json!({ "job_id": job_id, "cancelled": cancelled })))
}

async fn registry_debug(State(state): State<AppState>) -> Json<Value> {
    let names = state.jobs.registered();
    let items: Vec<Value> = names
        .iter()
        .map(|name| json!({ "app_id": APP_NAME, "name": name, "full_id": format!("{APP_NAME}.{name}") }))
        .collect();
    let by_app: BTreeMap<&str, &Vec<String>> = BTreeMap::from([(APP_NAME, &names)]);
    Json(json!({
        "total_commands": names.len(),
        "commands_by_app": by_app,
        "command_items": items,
    }))
}
