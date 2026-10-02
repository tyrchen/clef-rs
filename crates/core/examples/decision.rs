//! Offline embedded Flash decision using a verified cache.
use std::{env, io};

use anyhow::{Context, Result};
use clef_rs_core::{
    DecisionRequest, DirectEngine, ExecutionProfile,
    artifacts::ArtifactStore,
    runtime::{DeviceKind, Modality, Precision},
    types::ModelPreset,
};

#[tokio::main]
async fn main() -> Result<()> {
    let cache = env::args_os().nth(1).context(
        "usage: cargo run --release -p clef-rs-core --example decision -- CACHE_DIRECTORY",
    )?;
    let store = ArtifactStore::new(cache.into(), 85_899_345_920)?;
    let snapshot = store.open(ModelPreset::ClefFlash).await?;
    let profile = ExecutionProfile::builder()
        .device(DeviceKind::Cpu)
        .dtype(Precision::F32)
        .modality(Modality::Text)
        .max_context_tokens(4096)
        .device_budget_bytes(64 * 1024 * 1024 * 1024)
        .host_budget_bytes(64 * 1024 * 1024 * 1024)
        .build();
    let request = DecisionRequest::from_json(include_bytes!("../../../examples/request.json"))?;
    let mut engine = DirectEngine::load(snapshot, profile).context("load offline Flash engine")?;
    let result = engine.decide(&request).context("Flash decision")?;
    serde_json::to_writer_pretty(io::stdout().lock(), &result.systemone("clef-flash")?)?;
    store.shutdown().await?;
    Ok(())
}
