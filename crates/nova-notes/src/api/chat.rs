//! Notebook chat: `/api/chat/...` (Python `api/routers/chat.py`).

use axum::extract::{Path, State};
use axum::routing::{get, post};
use axum::{Json, Router};
use serde::Deserialize;

use crate::api::{ApiJson, ApiQuery};
use crate::error::AppResult;
use crate::services::chat::{
    self, ChatSessionResponse, ChatSessionWithMessages, CreateSessionRequest, ExecuteChatRequest, ExecuteChatResponse,
    SuccessResponse, UpdateSessionRequest,
};
use crate::services::context::{self, BuildContextRequest, BuildContextResponse};
use crate::state::AppState;

pub fn router() -> Router<AppState> {
    Router::new()
        .route("/chat/sessions", get(list_sessions).post(create_session))
        .route("/chat/sessions/{session_id}", get(get_session).put(update_session).delete(delete_session))
        .route("/chat/execute", post(execute_chat))
        .route("/chat/context", post(build_context))
}

#[derive(Debug, Deserialize)]
struct SessionsQuery {
    notebook_id: String,
}

async fn list_sessions(State(state): State<AppState>, ApiQuery(query): ApiQuery<SessionsQuery>) -> AppResult<Json<Vec<ChatSessionResponse>>> {
    Ok(Json(chat::list_sessions(&state, &query.notebook_id).await?))
}

async fn create_session(State(state): State<AppState>, ApiJson(body): ApiJson<CreateSessionRequest>) -> AppResult<Json<ChatSessionResponse>> {
    Ok(Json(chat::create_session(&state, &body).await?))
}

async fn get_session(State(state): State<AppState>, Path(session_id): Path<String>) -> AppResult<Json<ChatSessionWithMessages>> {
    Ok(Json(chat::get_session(&state, &session_id).await?))
}

async fn update_session(
    State(state): State<AppState>,
    Path(session_id): Path<String>,
    ApiJson(body): ApiJson<UpdateSessionRequest>,
) -> AppResult<Json<ChatSessionResponse>> {
    Ok(Json(chat::update_session(&state, &session_id, &body).await?))
}

async fn delete_session(State(state): State<AppState>, Path(session_id): Path<String>) -> AppResult<Json<SuccessResponse>> {
    Ok(Json(chat::delete_session(&state, &session_id).await?))
}

async fn execute_chat(State(state): State<AppState>, ApiJson(body): ApiJson<ExecuteChatRequest>) -> AppResult<Json<ExecuteChatResponse>> {
    Ok(Json(chat::execute_chat(&state, &body).await?))
}

async fn build_context(State(state): State<AppState>, ApiJson(body): ApiJson<BuildContextRequest>) -> AppResult<Json<BuildContextResponse>> {
    Ok(Json(context::build_context(&state, &body).await?))
}
