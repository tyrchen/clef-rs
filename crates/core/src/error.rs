//! Domain errors with safe messages and retained diagnostic sources.
use std::sync::Arc;

use thiserror::Error;
use yaml_rust2::scanner::ScanError;

/// A library result.
pub type Result<T> = std::result::Result<T, Error>;

/// Errors crossing acquisition, validation, and execution boundaries.
#[derive(Debug, Clone, Error)]
#[non_exhaustive]
pub enum Error {
    /// Invalid external data.
    #[error("invalid request: {0}")]
    InvalidRequest(String),
    /// A configured resource limit was exceeded.
    #[error("resource limit exceeded: {0}")]
    LimitExceeded(String),
    /// Unreviewed model architecture.
    #[error("unsupported architecture")]
    UnsupportedArchitecture,
    /// Unsupported or unqualified execution combination.
    #[error("unsupported capability: {0}")]
    UnsupportedCapability(String),
    /// Missing committed snapshot.
    #[error("verified artifact missing")]
    ArtifactMissing,
    /// Integrity verification failed.
    #[error("artifact integrity mismatch: {0}")]
    IntegrityMismatch(String),
    /// Disk budget exhausted.
    #[error("artifact storage limit exceeded")]
    StorageLimit,
    /// Host/device budget exhausted.
    #[error("insufficient configured memory")]
    InsufficientMemory,
    /// Bounded admission is saturated.
    #[error("decision queue is full")]
    QueueFull,
    /// Deadline expired.
    #[error("decision deadline exceeded")]
    DeadlineExceeded,
    /// Caller canceled work.
    #[error("decision canceled")]
    Cancelled,
    /// Worker failed or is unavailable.
    #[error("worker unavailable")]
    WorkerUnavailable,
    /// Admission closed for shutdown.
    #[error("runtime is shutting down")]
    ShuttingDown,
    /// Model execution failed.
    #[error("inference failed: {0}")]
    InferenceFailed(String),
    /// Filesystem operation failed; do not format sources in public transport errors.
    #[error("artifact IO failed")]
    Io(#[source] Arc<std::io::Error>),
    /// JSON parse failure.
    #[error("invalid JSON")]
    Json(#[source] Arc<serde_json::Error>),
    /// YAML syntax failure before configuration value construction.
    #[error("invalid YAML")]
    Yaml(#[source] Arc<ScanError>),
    /// Candle operation failed.
    #[error("model tensor operation failed")]
    Tensor(#[source] Arc<candle_core::Error>),
    /// Safetensors validation failed.
    #[error("invalid safetensors")]
    Safetensors(#[source] Arc<safetensors::SafeTensorError>),
    /// HTTP transport failure.
    #[cfg(feature = "hub")]
    #[error("artifact download failed")]
    Download(#[source] Arc<reqwest::Error>),
}

impl From<std::io::Error> for Error {
    fn from(source: std::io::Error) -> Self {
        Self::Io(Arc::new(source))
    }
}
impl From<serde_json::Error> for Error {
    fn from(source: serde_json::Error) -> Self {
        Self::Json(Arc::new(source))
    }
}
impl From<ScanError> for Error {
    fn from(source: ScanError) -> Self {
        Self::Yaml(Arc::new(source))
    }
}
impl From<candle_core::Error> for Error {
    fn from(source: candle_core::Error) -> Self {
        Self::Tensor(Arc::new(source))
    }
}
impl From<safetensors::SafeTensorError> for Error {
    fn from(source: safetensors::SafeTensorError) -> Self {
        Self::Safetensors(Arc::new(source))
    }
}
#[cfg(feature = "hub")]
impl From<reqwest::Error> for Error {
    fn from(source: reqwest::Error) -> Self {
        Self::Download(Arc::new(source.without_url()))
    }
}
