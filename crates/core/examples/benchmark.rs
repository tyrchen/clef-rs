//! Offline full-weight benchmark: normal-path samples and separate stage diagnostics.
#![allow(
    clippy::disallowed_types,
    clippy::disallowed_methods,
    reason = "bounded synchronous configuration/checkpoint IO runs on the direct caller thread \
              outside async contexts"
)]

use std::{
    collections::BTreeMap,
    fs::{self, File},
    io::{self, Read},
    path::{Path, PathBuf},
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    thread::{self, JoinHandle},
    time::{Duration, Instant},
};

use anyhow::{Context, Result, bail};
use clap::{Parser, ValueEnum};
use clef_rs_core::{
    DecisionRequest, DecisionResult, DirectEngine, ExecutionProfile, Runtime, RuntimeConfig,
    artifacts::{ArtifactStore, VerifiedSnapshot},
    configuration::validate_yaml,
    encoding::{Encoder, Truncation},
    runtime::{
        DecisionOptions, DeviceKind, DeviceMemory, ExecutionTimings, MemoryPlan, MemoryProbe,
        Modality, Precision, PrefixCacheConfig, PrefixCacheStats,
    },
    types::{Answer, Identifier, ModelPreset},
};
use config::{Config, File as ConfigFile, FileFormat};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};
use tokio::{runtime::Builder, task::JoinSet};

#[derive(Debug, Clone, Copy, ValueEnum)]
enum Backend {
    Cpu,
    Metal,
}
#[derive(Debug, Clone, Copy, ValueEnum)]
enum Dtype {
    F32,
    F16,
}
#[derive(Debug, Parser)]
struct Args {
    #[arg(long)]
    cache_dir: PathBuf,
    #[arg(long, default_value = "examples/clef.benchmark.yaml")]
    config: PathBuf,
    #[arg(long, value_enum)]
    device: Backend,
    #[arg(long, value_enum, default_value = "f32")]
    dtype: Dtype,
    #[arg(long)]
    output: PathBuf,
    /// Validate exact workload lengths without allocating model weights.
    #[arg(long)]
    prepare_only: bool,
    /// Rotate full-prefill, capture/miss and hit samples; verify every decision.
    #[arg(long)]
    prefix_reuse: bool,
}
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct Settings {
    schema_version: u32,
    warmup_per_case: usize,
    cases: Vec<Case>,
    concurrency: Vec<usize>,
    requests_per_worker: usize,
}
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct Case {
    name: String,
    tokens: usize,
    fields: usize,
    options: usize,
    samples: usize,
}
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct Summary {
    count: usize,
    mean_ms: f64,
    min_ms: f64,
    p50_ms: f64,
    p95_ms: f64,
    p99_ms: f64,
    max_ms: f64,
    standard_deviation_ms: f64,
}
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct CaseResult {
    workload: Case,
    actual_tokens: usize,
    latency: Summary,
    wall_ms: f64,
    requests_per_second: f64,
    input_tokens_per_second: f64,
    samples_ms: Vec<f64>,
    diagnostic: ExecutionTimings,
    #[serde(skip_serializing_if = "Option::is_none")]
    device_memory: Option<DeviceMemory>,
}
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct ConcurrentResult {
    concurrency: usize,
    attempted: usize,
    succeeded: usize,
    errors: BTreeMap<String, usize>,
    wall_ms: f64,
    successful_requests_per_second: f64,
    latency: Summary,
    samples_ms: Vec<f64>,
}
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct Report {
    schema_version: u32,
    revision: String,
    manifest_digest: String,
    profile: String,
    pid: u32,
    memory_plan: MemoryPlan,
    verification_ms: f64,
    load_ms: f64,
    first_decision_ms: f64,
    cases: Vec<CaseResult>,
    managed_startup_ms: f64,
    concurrency: Vec<ConcurrentResult>,
    shutdown_ms: f64,
    complete: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    gpu_sampled_peak_bytes: Option<u64>,
}
#[derive(Debug)]
struct MemoryMonitor {
    stop: Arc<AtomicBool>,
    worker: Option<JoinHandle<u64>>,
}
impl MemoryMonitor {
    fn start(probe: MemoryProbe) -> Result<Option<Self>> {
        if probe.snapshot().is_none() {
            return Ok(None);
        }
        let stop = Arc::new(AtomicBool::new(false));
        let signal = stop.clone();
        let worker = thread::Builder::new()
            .name("clef-gpu-memory".into())
            .spawn(move || {
                let mut peak = 0;
                while !signal.load(Ordering::Acquire) {
                    if let Some(memory) = probe.snapshot() {
                        peak = peak.max(memory.allocated_bytes);
                    }
                    thread::sleep(Duration::from_millis(100));
                }
                peak
            })?;
        Ok(Some(Self {
            stop,
            worker: Some(worker),
        }))
    }
    fn finish(mut self) -> Result<u64> {
        self.join()
    }
    fn join(&mut self) -> Result<u64> {
        self.stop.store(true, Ordering::Release);
        let worker = self
            .worker
            .take()
            .context("memory observer already joined")?;
        worker
            .join()
            .map_err(|_| anyhow::anyhow!("memory observer panicked"))
    }
}
impl Drop for MemoryMonitor {
    fn drop(&mut self) {
        if self.worker.is_some()
            && let Err(error) = self.join()
        {
            tracing::error!(%error, "memory observer failed");
        }
    }
}
fn load_settings(path: &Path) -> Result<Settings> {
    let metadata = fs::metadata(path).context("benchmark configuration metadata")?;
    if !metadata.is_file() || metadata.len() > 65536 {
        bail!("benchmark configuration must be a regular file of at most 64 KiB");
    }
    let mut text = String::new();
    File::open(path)?.take(65537).read_to_string(&mut text)?;
    if text.len() > 65536 {
        bail!("benchmark YAML exceeds 64 KiB");
    }
    validate_yaml(&text)?;
    let settings: Settings = Config::builder()
        .add_source(ConfigFile::from_str(&text, FileFormat::Yaml))
        .build()?
        .try_deserialize()?;
    if settings.schema_version != 1
        || settings.cases.is_empty()
        || settings.cases.len() > 16
        || settings.warmup_per_case > 10
        || settings.concurrency.is_empty()
        || settings.concurrency.len() > 8
        || !(1..=100).contains(&settings.requests_per_worker)
        || settings.concurrency.iter().any(|n| !(1..=32).contains(n))
    {
        bail!("benchmark configuration limits");
    }
    for case in &settings.cases {
        let _: Identifier = case.name.parse()?;
        if !(139..=4096).contains(&case.tokens)
            || !(1..=32).contains(&case.fields)
            || !(2..=64).contains(&case.options)
            || case
                .fields
                .checked_mul(case.options)
                .is_none_or(|n| n > 512)
            || !(1..=1000).contains(&case.samples)
        {
            bail!("benchmark workload limits");
        }
    }
    Ok(settings)
}
fn request(case: &Case, words: usize) -> Result<DecisionRequest> {
    let mut questions = Map::new();
    for i in 0..case.fields {
        let field = if case.options == 2 && i % 3 == 0 {
            json!({"type":"noul"})
        } else if i % 3 == 2 && case.options < 64 {
            json!({"type":"score", "criteria": (0..case.options).collect::<Vec<_>>()})
        } else {
            let options: Map<String, Value> = (0..case.options)
                .map(|n| (format!("o{n}"), Value::Null))
                .collect();
            json!({"type":"choice","criteria":options})
        };
        questions.insert(format!("q{i}"), field);
    }
    Ok(DecisionRequest::from_json(&serde_json::to_vec(
        &json!({"state":"normal ".repeat(words),"questions":questions}),
    )?)?)
}
fn exact_request(case: &Case, encoder: &Encoder) -> Result<DecisionRequest> {
    let mut low = 0;
    let mut high = case.tokens;
    while low < high {
        let middle = low + (high - low).div_ceil(2);
        let input = request(case, middle)?;
        if encoder
            .encode(&input, 16384, Truncation::Reject, None)?
            .token_count()
            <= case.tokens
        {
            low = middle;
        } else {
            high = middle - 1;
        }
    }
    let input = request(case, low)?;
    let actual = encoder
        .encode(&input, 4096, Truncation::Reject, None)?
        .token_count();
    if actual != case.tokens {
        bail!(
            "{}: requested {} tokens, got {actual}",
            case.name,
            case.tokens
        );
    }
    Ok(input)
}
#[allow(
    clippy::cast_precision_loss,
    reason = "bounded benchmark sample counts are exactly representable as f64"
)]
fn summarize(samples: &[f64]) -> Summary {
    let mut sorted = samples.to_vec();
    sorted.sort_by(f64::total_cmp);
    let count = sorted.len();
    let mean = samples.iter().sum::<f64>() / count.max(1) as f64;
    let percentile = |numerator: usize| {
        let rank = count
            .saturating_mul(numerator)
            .div_ceil(100)
            .saturating_sub(1);
        sorted.get(rank).copied().unwrap_or_default()
    };
    let variance = samples.iter().map(|n| (n - mean).powi(2)).sum::<f64>() / count.max(1) as f64;
    Summary {
        count,
        mean_ms: mean,
        min_ms: sorted.first().copied().unwrap_or_default(),
        p50_ms: percentile(50),
        p95_ms: percentile(95),
        p99_ms: percentile(99),
        max_ms: sorted.last().copied().unwrap_or_default(),
        standard_deviation_ms: variance.sqrt(),
    }
}
fn save(path: &Path, report: &Report) -> Result<()> {
    let temporary = path.with_extension("partial.json");
    serde_json::to_writer_pretty(File::create(&temporary)?, report)?;
    fs::rename(temporary, path)?;
    Ok(())
}
fn finish_direct_phase(
    monitor: Option<MemoryMonitor>,
    report: &mut Report,
    path: &Path,
) -> Result<()> {
    report.gpu_sampled_peak_bytes = monitor.map(MemoryMonitor::finish).transpose()?;
    save(path, report)?;
    if report
        .gpu_sampled_peak_bytes
        .is_some_and(|peak| peak > report.memory_plan.device_bytes)
    {
        bail!("measured Metal allocation peak exceeds the memory plan");
    }
    Ok(())
}
#[allow(
    clippy::cast_precision_loss,
    reason = "bounded sample/token counts are exactly representable as f64"
)]
fn measure(
    engine: &mut DirectEngine,
    case: &Case,
    input: &DecisionRequest,
    warmups: usize,
) -> Result<CaseResult> {
    for _ in 0..warmups {
        engine.decide(input)?;
    }
    let wall = Instant::now();
    let mut samples = Vec::with_capacity(case.samples);
    for index in 0..case.samples {
        let started = Instant::now();
        let result = engine.decide(input)?;
        if result.input_tokens != case.tokens || result.answers.len() != case.fields {
            bail!("benchmark result invariant");
        }
        samples.push(started.elapsed().as_secs_f64() * 1000.);
        tracing::info!(
            case = case.name,
            sample = index + 1,
            milliseconds = samples.last().copied(),
            "normal inference measured"
        );
    }
    let wall_ms = wall.elapsed().as_secs_f64() * 1000.;
    let (_, diagnostic) = engine.decide_profiled(input)?;
    Ok(CaseResult {
        workload: case.clone(),
        actual_tokens: case.tokens,
        latency: summarize(&samples),
        wall_ms,
        requests_per_second: case.samples as f64 * 1000. / wall_ms,
        input_tokens_per_second: (case.tokens * case.samples) as f64 * 1000. / wall_ms,
        samples_ms: samples,
        diagnostic,
        device_memory: engine.device_memory(),
    })
}
#[allow(
    clippy::cast_precision_loss,
    reason = "bounded request counts are exactly representable as f64"
)]
async fn concurrent(
    runtime: &Runtime,
    input: &DecisionRequest,
    concurrency: usize,
    requests: usize,
) -> Result<ConcurrentResult> {
    let started = Instant::now();
    let mut tasks = JoinSet::new();
    for worker in 0..concurrency {
        let client = runtime.client();
        let request = input.clone();
        tasks.spawn(async move {
            let mut samples = Vec::new();
            let mut errors = BTreeMap::new();
            for _ in 0..requests {
                let start = Instant::now();
                let options =
                    DecisionOptions::new(format!("benchmark-{worker}"), Duration::from_secs(300))?;
                match client.decide(request.clone(), options).await {
                    Ok(_) => samples.push(start.elapsed().as_secs_f64() * 1000.),
                    Err(error) => {
                        let code = match error {
                            clef_rs_core::Error::QueueFull => "queueFull",
                            clef_rs_core::Error::DeadlineExceeded => "deadlineExceeded",
                            _ => return Err(anyhow::Error::new(error)),
                        };
                        *errors.entry(code.to_owned()).or_insert(0_usize) += 1;
                    }
                }
            }
            Ok::<_, anyhow::Error>((samples, errors))
        });
    }
    let mut samples = Vec::new();
    let mut errors = BTreeMap::new();
    while let Some(joined) = tasks.join_next().await {
        let (latencies, rejected) = joined.context("benchmark worker panicked")??;
        samples.extend(latencies);
        for (code, count) in rejected {
            *errors.entry(code).or_insert(0_usize) += count;
        }
    }
    let wall_ms = started.elapsed().as_secs_f64() * 1000.;
    Ok(ConcurrentResult {
        concurrency,
        attempted: concurrency * requests,
        succeeded: samples.len(),
        successful_requests_per_second: samples.len() as f64 * 1000. / wall_ms,
        latency: summarize(&samples),
        wall_ms,
        errors,
        samples_ms: samples,
    })
}
fn save_prepared(
    path: &Path,
    settings: &Settings,
    inputs: &[DecisionRequest],
    encoder: &Encoder,
) -> Result<()> {
    let cases = settings.cases.iter().zip(inputs).map(|(case, input)| {
            Ok(json!({"name":case.name,"tokens":encoder.encode(input,4096,Truncation::Reject,None)?.token_count(),"fields":case.fields,"options":case.options}))
        }).collect::<Result<Vec<_>>>()?;
    serde_json::to_writer_pretty(File::create(path)?, &cases)?;
    Ok(())
}
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct ReuseCase {
    workload: Case,
    uncached: Summary,
    capture: Summary,
    hit: Summary,
    uncached_ms: Vec<f64>,
    capture_ms: Vec<f64>,
    hit_ms: Vec<f64>,
    reused_tokens: usize,
    maximum_probability_drift: f64,
    mean_probability_drift: f64,
    stats: PrefixCacheStats,
}
fn probabilities(answer: &Answer) -> Result<BTreeMap<String, f64>> {
    match answer {
        Answer::Noul { noul } => Ok(BTreeMap::from([
            ("true".into(), f64::from(noul.get())),
            ("false".into(), 1.0 - f64::from(noul.get())),
        ])),
        Answer::Choice { probabilities, .. } | Answer::Score { probabilities, .. } => probabilities
            .iter()
            .map(|(key, value)| {
                Ok((
                    key.clone(),
                    value.as_f64().context("invalid unrounded probability")?,
                ))
            })
            .collect(),
        _ => bail!("unsupported benchmark answer type"),
    }
}
fn probability_drift(reference: &DecisionResult, actual: &DecisionResult) -> Result<Vec<f64>> {
    if reference.input_tokens != actual.input_tokens
        || reference.truncated_state_tokens != actual.truncated_state_tokens
        || reference.revision != actual.revision
        || reference.manifest_digest != actual.manifest_digest
        || reference.execution_profile != actual.execution_profile
        || reference.answers.len() != actual.answers.len()
    {
        bail!("prefix reuse result provenance invariant");
    }
    let mut errors = Vec::new();
    for ((id, reference), (actual_id, actual)) in reference.answers.iter().zip(&actual.answers) {
        if id != actual_id {
            bail!("prefix reuse field identity");
        }
        let reference = probabilities(reference)?;
        let actual = probabilities(actual)?;
        if reference.len() != actual.len() {
            bail!("prefix reuse distribution length");
        }
        for (option, expected) in reference {
            errors.push(
                (expected - actual.get(&option).context("missing reused probability")?).abs(),
            );
        }
    }
    Ok(errors)
}
#[allow(
    clippy::cast_precision_loss,
    reason = "bounded benchmark counts are exactly representable"
)]
fn measure_reuse(
    engine: &mut DirectEngine,
    case: &Case,
    input: &DecisionRequest,
    warmups: usize,
) -> Result<ReuseCase> {
    engine.clear_prefix_cache();
    let expected = engine.decide_uncached(input)?;
    for _ in 0..warmups {
        engine.clear_prefix_cache();
        engine.decide(input)?;
        engine.decide(input)?;
    }
    let mut samples = [Vec::new(), Vec::new(), Vec::new()];
    let mut errors = Vec::new();
    let mut reused_tokens = 0;
    let mut hit_stats = PrefixCacheStats::default();
    for iteration in 0..case.samples {
        // Rotate mode order to avoid systematically assigning thermal drift to a mode.
        for mode in (0..3).map(|n| (n + iteration) % 3) {
            engine.clear_prefix_cache();
            if mode == 2 {
                engine.decide(input)?;
            }
            let started = Instant::now();
            let actual = if mode == 0 {
                engine.decide_uncached(input)?
            } else {
                engine.decide(input)?
            };
            samples
                .get_mut(mode)
                .context("benchmark mode")?
                .push(started.elapsed().as_secs_f64() * 1000.);
            errors.extend(probability_drift(&expected, &actual)?);
            let stats = engine.prefix_cache_stats();
            if mode == 2 {
                reused_tokens = usize::try_from(stats.reused_tokens)?;
                hit_stats = stats;
                if (case.tokens >= 1024 && stats.hits != 1) || stats.hits > 1 {
                    bail!("prefix reuse workload did not hit exactly once");
                }
            }
        }
        tracing::info!(
            case = case.name,
            sample = iteration + 1,
            "rotated prefix reuse modes measured"
        );
    }
    let maximum = errors.iter().copied().fold(0.0_f64, f64::max);
    let mean = errors.iter().sum::<f64>() / errors.len().max(1) as f64;
    if maximum > 1e-3 || mean > 1e-4 {
        bail!("reuse drift maximum={maximum}, mean={mean}");
    }
    let [uncached_ms, capture_ms, hit_ms] = samples;
    Ok(ReuseCase {
        workload: case.clone(),
        uncached: summarize(&uncached_ms),
        capture: summarize(&capture_ms),
        hit: summarize(&hit_ms),
        uncached_ms,
        capture_ms,
        hit_ms,
        reused_tokens,
        maximum_probability_drift: maximum,
        mean_probability_drift: mean,
        stats: hit_stats,
    })
}
#[allow(
    clippy::too_many_arguments,
    reason = "offline report provenance is explicit and immutable"
)]
fn save_reuse(
    engine: &mut DirectEngine,
    settings: &Settings,
    inputs: &[DecisionRequest],
    output: &Path,
    snapshot: &VerifiedSnapshot,
    profile: &ExecutionProfile,
    plan: &MemoryPlan,
    load_ms: f64,
) -> Result<()> {
    let mut cases = Vec::new();
    let write = |cases: &[ReuseCase], complete: bool, engine: &DirectEngine| -> Result<()> {
        serde_json::to_writer_pretty(
            File::create(output)?,
            &json!({"schemaVersion":1,"revision":ModelPreset::ClefFlash.revision(),
            "manifestDigest":snapshot.digest(), "profile":profile.name(), "memoryPlan":plan, "loadMs":load_ms,
            "cases":cases, "deviceMemory":engine.device_memory(), "complete":complete}),
        )?;
        Ok(())
    };
    for (case, input) in settings.cases.iter().zip(inputs) {
        cases.push(measure_reuse(
            engine,
            case,
            input,
            settings.warmup_per_case,
        )?);
        write(&cases, false, engine)?;
    }
    write(&cases, true, engine)
}
fn benchmark_profile(device: Backend, dtype: Dtype) -> ExecutionProfile {
    ExecutionProfile::builder()
        .device(match device {
            Backend::Cpu => DeviceKind::Cpu,
            Backend::Metal => DeviceKind::Metal,
        })
        .dtype(match dtype {
            Dtype::F32 => Precision::F32,
            Dtype::F16 => Precision::F16,
        })
        .modality(Modality::Text)
        .max_context_tokens(4096)
        .device_budget_bytes(64 * 1024 * 1024 * 1024)
        .host_budget_bytes(64 * 1024 * 1024 * 1024)
        .build()
}
#[allow(
    clippy::too_many_arguments,
    reason = "explicit offline benchmark provenance and independent lifecycle timings"
)]
fn initial_report(
    profile: String,
    manifest_digest: String,
    plan: MemoryPlan,
    verification_ms: f64,
    load_ms: f64,
    first_decision_ms: f64,
) -> Report {
    Report {
        schema_version: 1,
        revision: ModelPreset::ClefFlash.revision().into(),
        manifest_digest,
        profile,
        pid: std::process::id(),
        memory_plan: plan,
        verification_ms,
        load_ms,
        first_decision_ms,
        cases: Vec::new(),
        managed_startup_ms: 0.,
        concurrency: Vec::new(),
        shutdown_ms: 0.,
        complete: false,
        gpu_sampled_peak_bytes: None,
    }
}
fn main() -> Result<()> {
    let executor = Builder::new_multi_thread().enable_all().build()?;
    tracing_subscriber::fmt().with_writer(io::stderr).init();
    let args = Args::parse();
    let settings = load_settings(&args.config)?;
    let profile = benchmark_profile(args.device, args.dtype);
    profile.validate()?;
    let started = Instant::now();
    let store = ArtifactStore::new(args.cache_dir, 85_899_345_920)?;
    let snapshot = executor.block_on(store.open(ModelPreset::ClefFlash))?;
    let verification_ms = started.elapsed().as_secs_f64() * 1000.;
    let encoder = Encoder::from_snapshot(&snapshot)?;
    let inputs: Vec<_> = settings
        .cases
        .iter()
        .map(|case| exact_request(case, &encoder))
        .collect::<Result<_>>()?;
    if args.prepare_only {
        return save_prepared(&args.output, &settings, &inputs, &encoder);
    }
    let first = inputs.first().context("benchmark needs a workload")?;
    let cache = PrefixCacheConfig::new(if args.prefix_reuse {
        512 * 1024 * 1024
    } else {
        0
    })?;
    let plan = DirectEngine::memory_plan_with_prefix_cache(&snapshot, &profile, &cache)?;
    let started = Instant::now();
    let mut engine =
        DirectEngine::load_with_prefix_cache(snapshot.clone(), profile.clone(), cache)?;
    let load_ms = started.elapsed().as_secs_f64() * 1000.;
    if args.prefix_reuse {
        save_reuse(
            &mut engine,
            &settings,
            &inputs,
            &args.output,
            &snapshot,
            &profile,
            &plan,
            load_ms,
        )?;
        executor.block_on(store.shutdown())?;
        return Ok(());
    }
    let monitor = MemoryMonitor::start(engine.memory_probe())?;
    let started = Instant::now();
    engine.decide(first)?;
    let first_decision_ms = started.elapsed().as_secs_f64() * 1000.;
    let mut report = initial_report(
        profile.name(),
        snapshot.digest().into(),
        plan,
        verification_ms,
        load_ms,
        first_decision_ms,
    );
    save(&args.output, &report)?;
    for (case, input) in settings.cases.iter().zip(&inputs) {
        tracing::info!(
            case = case.name,
            tokens = case.tokens,
            "benchmark workload started"
        );
        report
            .cases
            .push(measure(&mut engine, case, input, settings.warmup_per_case)?);
        save(&args.output, &report)?;
    }
    finish_direct_phase(monitor, &mut report, &args.output)?;
    drop(engine);
    let config = RuntimeConfig::builder()
        .preparation_workers(8)
        .queue_timeout_ms(300_000)
        .decision_timeout_ms(300_000)
        .shutdown_grace_ms(300_000)
        .build();
    let started = Instant::now();
    let runtime = executor.block_on(Runtime::start(snapshot, profile, config))?;
    report.managed_startup_ms = started.elapsed().as_secs_f64() * 1000.;
    for concurrency in settings.concurrency {
        report.concurrency.push(executor.block_on(concurrent(
            &runtime,
            first,
            concurrency,
            settings.requests_per_worker,
        ))?);
        save(&args.output, &report)?;
    }
    let started = Instant::now();
    if !executor.block_on(runtime.shutdown())?.drained {
        bail!("benchmark worker did not drain");
    }
    report.shutdown_ms = started.elapsed().as_secs_f64() * 1000.;
    executor.block_on(store.shutdown())?;
    report.complete = true;
    save(&args.output, &report)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use clef_rs_core::{Error as CoreError, types::Probability};
    use rstest::rstest;
    use tempfile::tempdir;

    use super::*;
    #[test]
    fn test_should_compare_unrounded_probabilities_below_serialization_precision() -> Result<()> {
        let a = Answer::Noul {
            noul: Probability::try_from(0.12344)?,
        };
        let b = Answer::Noul {
            noul: Probability::try_from(0.12343)?,
        };
        let a = probabilities(&a)?;
        let b = probabilities(&b)?;
        let drift = (a.get("true").context("true probability")?
            - b.get("true").context("true probability")?)
        .abs();
        assert!(drift > 9e-6 && drift < 11e-6);
        Ok(())
    }
    #[test]
    fn test_should_report_order_statistics_without_hiding_small_sample_count() {
        let result = summarize(&[3., 1., 2., 4.]);
        assert_eq!(result.count, 4);
        assert!((result.p50_ms - 2.).abs() < f64::EPSILON);
        assert!((result.p99_ms - 4.).abs() < f64::EPSILON);
    }
    #[test]
    fn test_should_reject_duplicate_yaml_before_configuration_overwrites_values() -> Result<()> {
        let directory = tempdir()?;
        let path = directory.path().join("benchmark.yaml");
        let mut source = include_str!("../../../examples/clef.benchmark.yaml").to_owned();
        source.push_str("\nwarmupPerCase: 0\n");
        fs::write(&path, source)?;
        let error = load_settings(&path).err().context("duplicate must fail")?;
        assert!(
            matches!(error.downcast_ref::<CoreError>(), Some(CoreError::InvalidRequest(message)) if message.contains("duplicate"))
        );
        Ok(())
    }
    #[rstest]
    #[case("../../examples/clef.benchmark.yaml", 6)]
    #[case("../../examples/clef.mbp-benchmark.yaml", 8)]
    #[case("../../examples/clef.profile.yaml", 1)]
    fn test_should_validate_checked_in_benchmark_configuration(
        #[case] path: &str,
        #[case] cases: usize,
    ) -> Result<()> {
        let settings = load_settings(Path::new(path))?;
        assert_eq!(settings.cases.len(), cases);
        Ok(())
    }
    #[rstest]
    #[case(138)]
    #[case(4097)]
    fn test_should_reject_out_of_range_workload_tokens(#[case] tokens: usize) -> Result<()> {
        let directory = tempdir()?;
        let path = directory.path().join("benchmark.yaml");
        fs::write(
            &path,
            include_str!("../../../examples/clef.profile.yaml")
                .replace("tokens: 256", &format!("tokens: {tokens}")),
        )?;
        assert!(load_settings(&path).is_err());
        Ok(())
    }
}
