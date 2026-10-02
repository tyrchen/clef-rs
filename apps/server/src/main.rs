//! `clef` artifact administration, offline decisions, and authenticated serving.
#![forbid(unsafe_code)]
// Blocking filesystem work runs during startup, on the direct caller, or in spawn_blocking.
#![allow(
    clippy::disallowed_methods,
    clippy::disallowed_types,
    reason = "bounded synchronous IO is required for the direct engine and advisory file leases"
)]

mod auth;
mod config;
mod error;
mod listener;
mod routes;

use std::{
    collections::HashMap,
    io::{self, Write},
    path::PathBuf,
    sync::Arc,
    time::Duration,
};

use anyhow::{Context, Result, bail};
use clap::{Parser, Subcommand};
use clef_rs_core::{
    DecisionRequest, Runtime,
    artifacts::{ArtifactStore, Manifest},
    runtime::{DecisionOptions, DirectEngine},
    types::{CommitRevision, ModelPreset},
};
use metrics_exporter_prometheus::PrometheusBuilder;
use serde::Serialize;
use tokio::{net::TcpListener, sync::Semaphore};
use tracing_subscriber::EnvFilter;

use crate::{
    auth::Authenticator,
    config::Settings,
    listener::LimitedListener,
    routes::{AppState, LoadedModel, rate_limiter},
};

#[derive(Debug, Parser)]
#[command(name = "clef", about = "Pinned local CLEF decision inference", version)]
struct Cli {
    /// Emit structured JSON diagnostics on stderr.
    #[arg(long, global = true)]
    json_logs: bool,
    #[command(subcommand)]
    command: Command,
}
#[derive(Debug, Subcommand)]
enum Command {
    /// Download and verify a complete pinned snapshot.
    Fetch {
        #[arg(long)]
        model: ModelPreset,
        #[arg(long)]
        revision: Option<CommitRevision>,
        #[arg(long)]
        cache_dir: PathBuf,
        #[arg(long, default_value_t = 85_899_345_920)]
        max_bytes: u64,
    },
    /// Inspect catalog and committed cache status without allocating weights.
    Inspect {
        #[arg(long)]
        model: ModelPreset,
        #[arg(long)]
        cache_dir: PathBuf,
    },
    /// Import a complete local release into the private verified cache.
    Import {
        #[arg(long)]
        model: ModelPreset,
        #[arg(long)]
        source: PathBuf,
        #[arg(long)]
        cache_dir: PathBuf,
        #[arg(long, default_value_t = 85_899_345_920)]
        max_bytes: u64,
    },
    /// Cache verification and explicit lease-aware pruning.
    Cache {
        #[command(subcommand)]
        command: CacheCommand,
    },
    /// Load a local snapshot and answer a request through the same core as HTTP.
    Decide {
        #[arg(long)]
        config: PathBuf,
        #[arg(long)]
        request: PathBuf,
    },
    /// Start authenticated local serving after load and warmup.
    Serve {
        #[arg(long)]
        config: PathBuf,
        #[arg(long)]
        offline: bool,
    },
    /// Show the memory plan before loading weights.
    PlanMemory {
        #[arg(long)]
        config: PathBuf,
    },
}
#[derive(Debug, Subcommand)]
enum CacheCommand {
    /// List committed snapshots.
    List {
        #[arg(long)]
        cache_dir: PathBuf,
    },
    /// Verify all artifact identities without network access.
    Verify {
        #[arg(long)]
        model: ModelPreset,
        #[arg(long)]
        cache_dir: PathBuf,
    },
    /// Remove only an explicitly selected unleased snapshot; inspect with dry run.
    Prune {
        #[arg(long)]
        model: ModelPreset,
        #[arg(long)]
        cache_dir: PathBuf,
        #[arg(long)]
        dry_run: bool,
    },
}
#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    let logging = tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .with_writer(io::stderr);
    if cli.json_logs {
        logging.json().init();
    } else {
        logging.init();
    }
    dispatch(cli.command).await
}
async fn dispatch(command: Command) -> Result<()> {
    match command {
        Command::Fetch {
            model,
            revision,
            cache_dir,
            max_bytes,
        } => {
            if revision.is_some_and(|r| r.as_str() != model.revision()) {
                bail!("unreviewed revision; catalog update and qualification required");
            }
            let snapshot = ArtifactStore::new(cache_dir, max_bytes)?
                .fetch(model)
                .await
                .context("fetch verified model snapshot")?;
            output(snapshot.manifest())?;
        }
        Command::Inspect { model, cache_dir } => {
            let store = ArtifactStore::new(cache_dir, 85_899_345_920)?;
            output(
                &serde_json::json!({"catalog":Manifest::catalog(model)?,"cached":store.list().await?}),
            )?;
        }
        Command::Import {
            model,
            source,
            cache_dir,
            max_bytes,
        } => {
            let snapshot = ArtifactStore::new(cache_dir, max_bytes)?
                .import(source, model)
                .await?;
            output(snapshot.manifest())?;
        }
        Command::Cache { command } => match command {
            CacheCommand::List { cache_dir } => {
                output(
                    &ArtifactStore::new(cache_dir, 85_899_345_920)?
                        .list()
                        .await?,
                )?;
            }
            CacheCommand::Verify { model, cache_dir } => {
                let snapshot = ArtifactStore::new(cache_dir, 85_899_345_920)?
                    .open(model)
                    .await?;
                output(&serde_json::json!({"verified":true,"manifestDigest":snapshot.digest()}))?;
            }
            CacheCommand::Prune {
                model,
                cache_dir,
                dry_run,
            } => output(
                &serde_json::json!({"dryRun":dry_run,"bytes":ArtifactStore::new(cache_dir,85_899_345_920)?.prune(model,dry_run).await?}),
            )?,
        },
        Command::Decide { config, request } => {
            decide_cli(Settings::load(&config)?, request).await?;
        }
        Command::PlanMemory { config } => {
            let settings = Settings::load(&config)?;
            let store = ArtifactStore::new(settings.cache.root, settings.cache.max_bytes)?;
            for model in settings.models {
                let snapshot = store.open(model.preset).await?;
                output(&DirectEngine::memory_plan(&snapshot, &model.execution)?)?;
            }
        }
        Command::Serve { config, offline } => serve(Settings::load(&config)?, offline).await?,
    }
    Ok(())
}
async fn decide_cli(settings: Settings, request: PathBuf) -> Result<()> {
    let metadata = std::fs::metadata(&request)?;
    if metadata.len() > settings.http.max_body_bytes as u64 || !metadata.is_file() {
        bail!("request file exceeds the configured HTTP body cap");
    }
    let bytes = crate::config::read_file(&request, settings.http.max_body_bytes as u64)?;
    let validated = DecisionRequest::from_json(&bytes)?;
    let envelope: serde_json::Value = serde_json::from_slice(&bytes)?;
    let alias = envelope
        .get("model")
        .and_then(serde_json::Value::as_str)
        .context("model alias is required")?;
    let model = settings
        .models
        .iter()
        .find(|m| m.alias == alias)
        .context("model alias is not configured")?;
    let store = ArtifactStore::new(settings.cache.root, settings.cache.max_bytes)?;
    let snapshot = store.open(model.preset).await?;
    let decision_timeout_ms = settings.runtime.decision_timeout_ms;
    let runtime = Runtime::start(snapshot, model.execution.clone(), settings.runtime).await?;
    let result = runtime
        .client()
        .decide(
            validated,
            DecisionOptions::new(
                "embedded".into(),
                Duration::from_millis(decision_timeout_ms),
            )?,
        )
        .await;
    let report = runtime.shutdown().await?;
    if !report.drained {
        bail!("worker did not drain");
    }
    output(&result?.systemone(alias)?)?;
    Ok(())
}
fn output(value: &impl Serialize) -> Result<()> {
    let stdout = io::stdout();
    let mut output = stdout.lock();
    serde_json::to_writer_pretty(&mut output, value)?;
    output.write_all(b"\n")?;
    Ok(())
}
async fn serve(mut settings: Settings, offline: bool) -> Result<()> {
    settings.cache.offline |= offline;
    // Validate auth before spending resources on model load; local keys never trigger discovery.
    let auth = Arc::new(Authenticator::load(settings.auth.clone())?);
    let metrics = PrometheusBuilder::new()
        .install_recorder()
        .context("install metrics recorder")?;
    let store = ArtifactStore::new(settings.cache.root.clone(), settings.cache.max_bytes)?;
    let mut runtimes = Vec::new();
    let mut shared: HashMap<String, clef_rs_core::DecisionClient> = HashMap::new();
    let mut models = HashMap::new();
    for model in settings.models {
        let key = format!("{}:{}", model.preset.alias(), model.execution.name());
        let client = if let Some(client) = shared.get(&key) {
            client.clone()
        } else {
            let snapshot = if settings.cache.offline {
                store.open(model.preset).await?
            } else {
                store.fetch(model.preset).await?
            };
            let runtime =
                Runtime::start(snapshot, model.execution.clone(), settings.runtime.clone()).await?;
            let client = runtime.client();
            shared.insert(key, client.clone());
            runtimes.push(runtime);
            client
        };
        models.insert(
            model.alias.clone(),
            LoadedModel {
                config: model,
                client,
            },
        );
    }
    let listener = TcpListener::bind(settings.http.bind)
        .await
        .context("bind serving listener")?;
    let address = listener.local_addr()?;
    let limited = LimitedListener {
        listener,
        capacity: Arc::new(Semaphore::new(settings.http.max_connections)),
        idle: Duration::from_millis(settings.http.request_read_timeout_ms),
    };
    let (rate, rate_task) = rate_limiter(settings.http.rate_limit_per_principal_per_minute);
    let state = AppState {
        auth,
        models: Arc::new(models),
        http: settings.http,
        metrics,
        rate,
        ingress: Arc::new(Semaphore::new(settings.runtime.ingress_capacity)),
        decision_timeout: Duration::from_millis(settings.runtime.decision_timeout_ms),
    };
    tracing::info!(%address,"CLEF Flash CPU F32 server ready");
    let (stopping, stopped) = tokio::sync::oneshot::channel();
    let server = axum::serve(
        limited,
        routes::router(state).into_make_service_with_connect_info::<crate::listener::Connection>(),
    )
    .with_graceful_shutdown(async {
        let _ = stopped.await;
    });
    let serving = async { server.await };
    tokio::pin!(serving);
    let result = tokio::select! {
        result = &mut serving => result,
        () = shutdown_signal() => {
            for runtime in &runtimes { runtime.close_admission(); }
            let _ = stopping.send(());
            let deadline = Duration::from_millis(settings.runtime.shutdown_grace_ms);
            tokio::time::timeout(deadline, &mut serving).await
                .context("HTTP drain deadline expired; process replacement required")?
        }
    };
    let mut drained = true;
    for runtime in runtimes {
        drained &= runtime.shutdown().await?.drained;
    }
    rate_task.await.context("rate-limiter supervisor")?;
    result.context("serve HTTP")?;
    if !drained {
        bail!("incomplete worker drain; process replacement required");
    }
    Ok(())
}

async fn shutdown_signal() {
    #[cfg(unix)]
    {
        use tokio::signal::unix::{SignalKind, signal};
        match signal(SignalKind::terminate()) {
            Ok(mut terminate) => {
                tokio::select! { _ = tokio::signal::ctrl_c() => {}, _ = terminate.recv() => {} }
            }
            Err(error) => {
                tracing::error!(%error, "cannot register SIGTERM handler");
                let _ = tokio::signal::ctrl_c().await;
            }
        }
    }
    #[cfg(not(unix))]
    {
        let _ = tokio::signal::ctrl_c().await;
    }
}
