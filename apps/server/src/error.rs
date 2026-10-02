//! Redacted machine-readable transport errors.
use axum::{
    Json,
    body::Body,
    http::StatusCode,
    response::{IntoResponse, Response},
};
use clef_rs_core::Error;

#[derive(Debug, Clone)]
pub(crate) struct ApiError {
    pub status: StatusCode,
    pub code: &'static str,
    pub message: &'static str,
}
impl ApiError {
    pub fn unauthorized() -> Self {
        Self {
            status: StatusCode::UNAUTHORIZED,
            code: "unauthorized",
            message: "Authentication is required.",
        }
    }
    pub fn forbidden() -> Self {
        Self {
            status: StatusCode::FORBIDDEN,
            code: "forbidden",
            message: "Permission denied.",
        }
    }
}
impl From<Error> for ApiError {
    fn from(error: Error) -> Self {
        let (status, code, message) = match error {
            Error::InvalidRequest(_) | Error::Json(_) => (
                StatusCode::BAD_REQUEST,
                "invalidRequest",
                "Invalid decision request.",
            ),
            Error::LimitExceeded(_) => (
                StatusCode::PAYLOAD_TOO_LARGE,
                "limitExceeded",
                "The request exceeds a configured limit.",
            ),
            Error::UnsupportedCapability(_) => (
                StatusCode::UNPROCESSABLE_ENTITY,
                "unsupportedCapability",
                "The requested capability is not qualified.",
            ),
            Error::QueueFull => (
                StatusCode::TOO_MANY_REQUESTS,
                "queueFull",
                "Decision capacity is full.",
            ),
            Error::Cancelled => (
                StatusCode::REQUEST_TIMEOUT,
                "cancelled",
                "The decision was cancelled.",
            ),
            Error::DeadlineExceeded => (
                StatusCode::GATEWAY_TIMEOUT,
                "deadlineExceeded",
                "The decision deadline expired.",
            ),
            Error::WorkerUnavailable | Error::ShuttingDown | Error::InsufficientMemory => (
                StatusCode::SERVICE_UNAVAILABLE,
                "workerUnavailable",
                "The decision worker is unavailable.",
            ),
            _ => (
                StatusCode::INTERNAL_SERVER_ERROR,
                "inferenceFailed",
                "Decision execution failed.",
            ),
        };
        Self {
            status,
            code,
            message,
        }
    }
}
impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let mut response = (
            self.status,
            Json(serde_json::json!({"error":{"code":self.code,"message":self.message}})),
        )
            .into_response();
        response.extensions_mut().insert(self.clone());
        if self.status == StatusCode::TOO_MANY_REQUESTS {
            response
                .headers_mut()
                .insert("retry-after", axum::http::HeaderValue::from_static("1"));
        }
        response
    }
}

pub(crate) fn attach_request_id(response: &mut Response, id: &str) {
    if let Some(error) = response.extensions().get::<ApiError>() {
        let body =
            serde_json::json!({"error":{"code":error.code,"message":error.message,"requestId":id}});
        *response.body_mut() = Body::from(body.to_string());
        response.headers_mut().remove("content-length");
    }
}
