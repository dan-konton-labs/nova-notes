//! Infographics: `/api/notebooks/{notebook_id}/infographics...`
//! (`api/routers/infographics.py`). Logic lives in [`crate::services::infographics`].

use axum::extract::{Path, State};
use axum::routing::{get, post};
use axum::{Json, Router};

use super::ApiJson;
use crate::error::AppResult;
use crate::services::generation_jobs::{GenerationJobStarted, GenerationJobSummary};
use crate::services::infographics::{self, InfographicCreateRequest};
use crate::state::AppState;

pub fn router() -> Router<AppState> {
    Router::new()
        .route(
            "/notebooks/{notebook_id}/infographics",
            post(create_infographic),
        )
        .route(
            "/notebooks/{notebook_id}/infographics/jobs",
            get(list_infographic_jobs),
        )
}

/// Start generating an infographic; follow it with the jobs list.
async fn create_infographic(
    State(state): State<AppState>,
    Path(notebook_id): Path<String>,
    ApiJson(body): ApiJson<InfographicCreateRequest>,
) -> AppResult<Json<GenerationJobStarted>> {
    Ok(Json(
        infographics::create_infographic(&state, &notebook_id, &body).await?,
    ))
}

async fn list_infographic_jobs(
    State(state): State<AppState>,
    Path(notebook_id): Path<String>,
) -> AppResult<Json<Vec<GenerationJobSummary>>> {
    Ok(Json(
        infographics::list_infographic_jobs(&state, &notebook_id).await?,
    ))
}
