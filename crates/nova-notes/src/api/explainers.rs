//! Explainers: `/api/notebooks/{notebook_id}/explainers...`
//! (`api/routers/explainers.py`). Logic lives in [`crate::services::explainers`].

use axum::extract::{Path, State};
use axum::routing::{get, post};
use axum::{Json, Router};

use super::ApiJson;
use crate::error::AppResult;
use crate::services::explainers::{self, ExplainerCreateRequest};
use crate::services::generation_jobs::{GenerationJobStarted, GenerationJobSummary};
use crate::state::AppState;

pub fn router() -> Router<AppState> {
    Router::new()
        .route(
            "/notebooks/{notebook_id}/explainers",
            post(create_explainer),
        )
        .route(
            "/notebooks/{notebook_id}/explainers/jobs",
            get(list_explainer_jobs),
        )
}

/// Start generating an explainer; follow it with the jobs list.
async fn create_explainer(
    State(state): State<AppState>,
    Path(notebook_id): Path<String>,
    ApiJson(body): ApiJson<ExplainerCreateRequest>,
) -> AppResult<Json<GenerationJobStarted>> {
    Ok(Json(
        explainers::create_explainer(&state, &notebook_id, &body).await?,
    ))
}

async fn list_explainer_jobs(
    State(state): State<AppState>,
    Path(notebook_id): Path<String>,
) -> AppResult<Json<Vec<GenerationJobSummary>>> {
    Ok(Json(
        explainers::list_explainer_jobs(&state, &notebook_id).await?,
    ))
}
