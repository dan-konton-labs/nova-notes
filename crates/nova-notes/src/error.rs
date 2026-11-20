//! The application error type and its HTTP mapping.
//!
//! Variants mirror the Python build's exception classes, and each maps to the
//! status code its FastAPI exception handler used, so API clients see the same
//! `{"detail": "..."}` bodies and codes.

use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::Json;
use nn_llm::AiError;

#[derive(Debug, Clone, thiserror::Error)]
pub enum AppError {
    /// 404
    #[error("{0}")]
    NotFound(String),
    /// 400
    #[error("{0}")]
    InvalidInput(String),
    /// 401
    #[error("{0}")]
    Authentication(String),
    /// 429
    #[error("{0}")]
    RateLimit(String),
    /// 422: something the user must configure (model, credential, key).
    #[error("{0}")]
    Configuration(String),
    /// 502: could not reach an external service.
    #[error("{0}")]
    Network(String),
    /// 502: an external service (usually an AI provider) failed.
    #[error("{0}")]
    ExternalService(String),
    /// 502: content exceeds the model's context window. Never retried.
    #[error("{0}")]
    ContextLength(String),
    /// 502: the model returned truncated or empty output. Never retried.
    #[error("{0}")]
    IncompleteGeneration(String),
    /// 415
    #[error("{0}")]
    UnsupportedType(String),
    /// 500: a database operation failed. `retryable` marks transaction conflicts.
    #[error("{message}")]
    Database { message: String, retryable: bool },
    /// 500
    #[error("{0}")]
    FileOperation(String),
    /// 500
    #[error("{0}")]
    Internal(String),
    /// An explicit status with a detail message (FastAPI's `HTTPException`).
    #[error("{detail}")]
    Http { status: StatusCode, detail: String },
}

pub type AppResult<T> = Result<T, AppError>;

impl AppError {
    pub fn not_found(msg: impl Into<String>) -> Self {
        Self::NotFound(msg.into())
    }
    pub fn invalid(msg: impl Into<String>) -> Self {
        Self::InvalidInput(msg.into())
    }
    pub fn config(msg: impl Into<String>) -> Self {
        Self::Configuration(msg.into())
    }
    pub fn internal(msg: impl Into<String>) -> Self {
        Self::Internal(msg.into())
    }
    pub fn database(msg: impl Into<String>) -> Self {
        let message = msg.into();
        let retryable = is_transaction_conflict(&message);
        Self::Database { message, retryable }
    }
    pub fn http(status: StatusCode, detail: impl Into<String>) -> Self {
        Self::Http { status, detail: detail.into() }
    }
    pub fn bad_request(detail: impl Into<String>) -> Self {
        Self::http(StatusCode::BAD_REQUEST, detail)
    }
    pub fn conflict(detail: impl Into<String>) -> Self {
        Self::http(StatusCode::CONFLICT, detail)
    }
    pub fn unprocessable(detail: impl Into<String>) -> Self {
        Self::http(StatusCode::UNPROCESSABLE_ENTITY, detail)
    }

    pub fn status(&self) -> StatusCode {
        match self {
            Self::NotFound(_) => StatusCode::NOT_FOUND,
            Self::InvalidInput(_) => StatusCode::BAD_REQUEST,
            Self::Authentication(_) => StatusCode::UNAUTHORIZED,
            Self::RateLimit(_) => StatusCode::TOO_MANY_REQUESTS,
            Self::Configuration(_) => StatusCode::UNPROCESSABLE_ENTITY,
            Self::Network(_) | Self::ExternalService(_) | Self::ContextLength(_) | Self::IncompleteGeneration(_) => {
                StatusCode::BAD_GATEWAY
            }
            Self::UnsupportedType(_) => StatusCode::UNSUPPORTED_MEDIA_TYPE,
            Self::Database { .. } | Self::FileOperation(_) | Self::Internal(_) => StatusCode::INTERNAL_SERVER_ERROR,
            Self::Http { status, .. } => *status,
        }
    }

    /// The `detail` string sent to clients.
    pub fn detail(&self) -> String {
        self.to_string()
    }

    pub fn is_not_found(&self) -> bool {
        matches!(self, Self::NotFound(_)) || matches!(self, Self::Http { status, .. } if *status == StatusCode::NOT_FOUND)
    }

    /// Whether a background job should try again after this error.
    ///
    /// Follows the Python commands' `stop_on` lists: validation, configuration,
    /// missing records, context-length and incomplete-generation errors are
    /// permanent; everything else (rate limits, network, provider 5xx, database
    /// conflicts) is worth another attempt.
    pub fn is_retryable(&self) -> bool {
        match self {
            Self::NotFound(_)
            | Self::InvalidInput(_)
            | Self::Authentication(_)
            | Self::Configuration(_)
            | Self::ContextLength(_)
            | Self::IncompleteGeneration(_)
            | Self::UnsupportedType(_)
            | Self::Http { .. } => false,
            Self::RateLimit(_)
            | Self::Network(_)
            | Self::ExternalService(_)
            | Self::Database { .. }
            | Self::FileOperation(_)
            | Self::Internal(_) => true,
        }
    }
}

/// SurrealDB reports optimistic-concurrency failures as plain error text.
pub fn is_transaction_conflict(message: &str) -> bool {
    let m = message.to_ascii_lowercase();
    m.contains("can be retried") || m.contains("transaction conflict") || m.contains("resource busy") || m.contains("write conflict")
}

impl IntoResponse for AppError {
    fn into_response(self) -> Response {
        let status = self.status();
        if status.is_server_error() {
            tracing::error!(%status, "{self}");
        }
        let mut response = (status, Json(serde_json::json!({ "detail": self.detail() }))).into_response();
        if status == StatusCode::UNAUTHORIZED {
            response
                .headers_mut()
                .insert(axum::http::header::WWW_AUTHENTICATE, axum::http::HeaderValue::from_static("Bearer"));
        }
        response
    }
}

impl From<AiError> for AppError {
    fn from(e: AiError) -> Self {
        match e {
            AiError::Authentication(m) => Self::Authentication(m),
            AiError::RateLimit(m) => Self::RateLimit(m),
            AiError::ContextLength(m) => Self::ContextLength(m),
            AiError::Configuration(m) | AiError::Unsupported(m) => Self::Configuration(m),
            AiError::Network(m) | AiError::Timeout(m) => Self::Network(m),
            AiError::Incomplete(m) => Self::IncompleteGeneration(m),
            AiError::ModelNotFound(m) | AiError::InvalidRequest(m) | AiError::Other(m) => Self::ExternalService(m),
            AiError::Provider { message, .. } => Self::ExternalService(message),
        }
    }
}

impl From<surrealdb::Error> for AppError {
    fn from(e: surrealdb::Error) -> Self {
        Self::database(e.to_string())
    }
}

impl From<serde_json::Error> for AppError {
    fn from(e: serde_json::Error) -> Self {
        Self::Internal(format!("Invalid data: {e}"))
    }
}

impl From<std::io::Error> for AppError {
    fn from(e: std::io::Error) -> Self {
        Self::FileOperation(e.to_string())
    }
}

impl From<minijinja::Error> for AppError {
    fn from(e: minijinja::Error) -> Self {
        Self::Internal(format!("Prompt template error: {e}"))
    }
}

impl From<tokio::task::JoinError> for AppError {
    fn from(e: tokio::task::JoinError) -> Self {
        Self::Internal(format!("Background task failed: {e}"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn statuses_match_the_python_handlers() {
        assert_eq!(AppError::not_found("x").status(), StatusCode::NOT_FOUND);
        assert_eq!(AppError::invalid("x").status(), StatusCode::BAD_REQUEST);
        assert_eq!(AppError::Authentication("x".into()).status(), StatusCode::UNAUTHORIZED);
        assert_eq!(AppError::RateLimit("x".into()).status(), StatusCode::TOO_MANY_REQUESTS);
        assert_eq!(AppError::config("x").status(), StatusCode::UNPROCESSABLE_ENTITY);
        assert_eq!(AppError::Network("x".into()).status(), StatusCode::BAD_GATEWAY);
        assert_eq!(AppError::ExternalService("x".into()).status(), StatusCode::BAD_GATEWAY);
        assert_eq!(AppError::UnsupportedType("x".into()).status(), StatusCode::UNSUPPORTED_MEDIA_TYPE);
        assert_eq!(AppError::internal("x").status(), StatusCode::INTERNAL_SERVER_ERROR);
        assert_eq!(AppError::conflict("x").status(), StatusCode::CONFLICT);
    }

    #[test]
    fn conflicts_are_retryable_and_validation_is_not() {
        assert!(AppError::database("Failed to commit transaction due to a read or write conflict. This transaction can be retried").is_retryable());
        assert!(matches!(AppError::database("This transaction can be retried"), AppError::Database { retryable: true, .. }));
        assert!(matches!(AppError::database("field type mismatch"), AppError::Database { retryable: false, .. }));
        assert!(!AppError::invalid("bad").is_retryable());
        assert!(!AppError::ContextLength("too long".into()).is_retryable());
        assert!(AppError::RateLimit("slow down".into()).is_retryable());
    }
}
