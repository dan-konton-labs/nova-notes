//! Search and Ask: `/api/search...` (Python `api/routers/search.py`).

use axum::extract::State;
use axum::response::{IntoResponse, Response};
use axum::routing::post;
use axum::{Json, Router};

use crate::api::source_chat::sse_response;
use crate::api::{validation_error, ApiJson};
use crate::error::AppResult;
use crate::services::search::{self, AskRequest, AskResponse, SearchRequest, SearchResponse};
use crate::state::AppState;

pub fn router() -> Router<AppState> {
    Router::new()
        .route("/search", post(search_knowledge_base))
        .route("/search/ask", post(ask))
        .route("/search/ask/simple", post(ask_simple))
}

async fn search_knowledge_base(State(state): State<AppState>, ApiJson(body): ApiJson<SearchRequest>) -> AppResult<Response> {
    if let Err(message) = body.validate() {
        return Ok(validation_error("body", "value_error", message));
    }
    let response: SearchResponse = search::search(&state, &body).await?;
    Ok(Json(response).into_response())
}

async fn ask(State(state): State<AppState>, ApiJson(body): ApiJson<AskRequest>) -> AppResult<Response> {
    if let Err(message) = body.validate() {
        return Ok(validation_error("body", "value_error", message));
    }
    let plan = search::prepare_ask(&state, &body).await?;
    Ok(sse_response(search::ask_stream(state, plan)))
}

async fn ask_simple(State(state): State<AppState>, ApiJson(body): ApiJson<AskRequest>) -> AppResult<Response> {
    if let Err(message) = body.validate() {
        return Ok(validation_error("body", "value_error", message));
    }
    let response: AskResponse = search::ask_simple(&state, &body).await?;
    Ok(Json(response).into_response())
}
