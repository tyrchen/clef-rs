//! Safe, single-owner local inference for the pinned Cloudflare CLEF Flash release.
//!
//! Acquisition is explicit. Loading and inference never perform network operations.
//! Model tensors belong to a direct engine or a dedicated managed worker thread.
#![forbid(unsafe_code)]

pub mod artifacts;
pub mod encoding;
mod error;
#[cfg(feature = "vision")]
pub mod media;
mod models;
pub mod runtime;
pub mod types;

pub use error::{Error, Result};
pub use runtime::{DecisionClient, DirectEngine, ExecutionProfile, Runtime, RuntimeConfig};
pub use types::{DecisionRequest, DecisionResult, ModelPreset};
