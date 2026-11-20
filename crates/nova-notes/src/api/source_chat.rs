//! Source chat: `/api/sources/{source_id}/chat/...` (Python
//! `api/routers/source_chat.py`).

use std::convert::Infallible;

use axum::body::{Body, Bytes};
use axum::extract::{Path, State};
use axum::http::{header, HeaderValue};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use futures::{Stream, StreamExt};
use serde::Serialize;

use crate::api::ApiJson;
use crate::error::AppResult;
use crate::services::chat::SuccessResponse;
use crate::services::source_chat::{
    self, CreateSourceChatSessionRequest, SendMessageRequest, SourceChatSessionResponse, SourceChatSessionWithMessages,
    UpdateSourceChatSessionRequest,
};
use crate::state::AppState;

pub fn router() -> Router<AppState> {
    Router::new()
        .route("/sources/{source_id}/chat/sessions", get(list_sessions).post(create_session))
        .route(
            "/sources/{source_id}/chat/sessions/{session_id}",
            get(get_session).put(update_session).delete(delete_session),
        )
        .route("/sources/{source_id}/chat/sessions/{session_id}/messages", post(send_message))
}

/// A `text/event-stream` response: each event as `data: <json>\n\n`, with
/// the headers the Python `StreamingResponse` set.
pub(crate) fn sse_response<S, E>(events: S) -> Response
where
    S: Stream<Item = E> + Send + 'static,
    E: Serialize,
{
    let body = events.map(|event| {
        let json = serde_json::to_string(&event).unwrap_or_else(|e| {
            serde_json::json!({ "type": "error", "message": format!("Could not encode event: {e}") }).to_string()
        });
        Ok::<_, Infallible>(Bytes::from(format!("data: {json}\n\n")))
    });
    let mut response = Body::from_stream(body).into_response();
    let headers = response.headers_mut();
    headers.insert(header::CONTENT_TYPE, HeaderValue::from_static("text/event-stream"));
    headers.insert(header::CACHE_CONTROL, HeaderValue::from_static("no-cache"));
    headers.insert(header::CONNECTION, HeaderValue::from_static("keep-alive"));
    headers.insert("x-accel-buffering", HeaderValue::from_static("no"));
    response
}

async fn create_session(
    State(state): State<AppState>,
    Path(source_id): Path<String>,
    ApiJson(body): ApiJson<CreateSourceChatSessionRequest>,
) -> AppResult<Json<SourceChatSessionResponse>> {
    Ok(Json(source_chat::create_session(&state, &source_id, &body).await?))
}

async fn list_sessions(State(state): State<AppState>, Path(source_id): Path<String>) -> AppResult<Json<Vec<SourceChatSessionResponse>>> {
    Ok(Json(source_chat::list_sessions(&state, &source_id).await?))
}

async fn get_session(
    State(state): State<AppState>,
    Path((source_id, session_id)): Path<(String, String)>,
) -> AppResult<Json<SourceChatSessionWithMessages>> {
    Ok(Json(source_chat::get_session(&state, &source_id, &session_id).await?))
}

async fn update_session(
    State(state): State<AppState>,
    Path((source_id, session_id)): Path<(String, String)>,
    ApiJson(body): ApiJson<UpdateSourceChatSessionRequest>,
) -> AppResult<Json<SourceChatSessionResponse>> {
    Ok(Json(source_chat::update_session(&state, &source_id, &session_id, &body).await?))
}

async fn delete_session(State(state): State<AppState>, Path((source_id, session_id)): Path<(String, String)>) -> AppResult<Json<SuccessResponse>> {
    Ok(Json(source_chat::delete_session(&state, &source_id, &session_id).await?))
}

async fn send_message(
    State(state): State<AppState>,
    Path((source_id, session_id)): Path<(String, String)>,
    ApiJson(body): ApiJson<SendMessageRequest>,
) -> AppResult<Response> {
    let turn = source_chat::prepare_message(&state, &source_id, &session_id, &body).await?;
    Ok(sse_response(source_chat::stream_message(state, turn)))
}
