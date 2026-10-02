//! Blocking direct inference and bounded managed execution on a dedicated OS thread.
use std::{
    collections::{HashMap, VecDeque},
    sync::{
        Arc,
        atomic::{AtomicBool, AtomicU8, Ordering},
    },
    thread::{self, JoinHandle},
    time::{Duration, Instant},
};

#[cfg(feature = "vision")]
use candle_core::Tensor;
use candle_core::{DType, Device};
use serde::{Deserialize, Serialize};
use tokio::{
    sync::{OwnedSemaphorePermit, Semaphore, mpsc, oneshot, watch},
    task,
    time::{interval, timeout, timeout_at},
};
use typed_builder::TypedBuilder;

use crate::{
    Error, Result,
    artifacts::VerifiedSnapshot,
    encoding::{EncodedRecord, Encoder, Truncation},
    models::{
        Control,
        head::{Head, HeadConfig},
        qwen::{Backbone, ModelConfig},
        weights::Weights,
    },
    types::{DecisionRequest, DecisionResult, answers},
};

#[derive(Debug, Clone, Copy)]
#[repr(u8)]
enum WorkerState {
    Loading,
    Ready,
    Draining,
    Stopped,
}

/// Explicit compute device; no silent fallback.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum DeviceKind {
    /// Portable CPU reference execution.
    Cpu,
    /// CUDA with an explicit device ordinal.
    Cuda,
    /// Metal is separately qualified.
    Metal,
}
/// Execution precision.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum Precision {
    /// Float32 CPU reference.
    F32,
    /// `BFloat16` CUDA execution.
    Bf16,
    /// Float16 Metal execution.
    F16,
}
/// Modality requires a separate qualification contract.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum Modality {
    /// Text and bounded JSON.
    Text,
    /// Images and text.
    Image,
}
/// Checked profile and explicit memory limits.
#[derive(Debug, Clone, Serialize, Deserialize, TypedBuilder)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
#[non_exhaustive]
pub struct ExecutionProfile {
    /// Compute backend.
    pub device: DeviceKind,
    /// Device index; CPU requires zero.
    #[serde(default)]
    #[builder(default)]
    pub ordinal: usize,
    /// Explicit precision.
    pub dtype: Precision,
    /// Qualified modality.
    pub modality: Modality,
    /// Total prompt token ceiling.
    pub max_context_tokens: usize,
    /// Total target device allocation allowance.
    pub device_budget_bytes: u64,
    /// Host weights, staging, and preparation allowance.
    pub host_budget_bytes: u64,
}
impl ExecutionProfile {
    /// Validate a profile before allocating any tensor.
    ///
    /// # Errors
    /// Rejects unsupported combinations, excessive context, or zero budgets.
    pub fn validate(&self) -> Result<()> {
        if !(1..=4096).contains(&self.max_context_tokens)
            || self.host_budget_bytes == 0
            || self.device_budget_bytes == 0
        {
            return Err(Error::InvalidRequest("execution limits".into()));
        }
        if self.modality != Modality::Text && !cfg!(feature = "vision") {
            return Err(Error::UnsupportedCapability(
                "image execution is not qualified".into(),
            ));
        }
        match (self.device, self.dtype) {
            (DeviceKind::Cpu, Precision::F32) if self.ordinal == 0 => Ok(()),
            _ => Err(Error::UnsupportedCapability(
                "device/dtype combination is unavailable or unqualified".into(),
            )),
        }
    }
    fn device(&self) -> Result<Device> {
        match self.device {
            DeviceKind::Cpu => Ok(Device::Cpu),
            DeviceKind::Cuda => Ok(Device::new_cuda(self.ordinal)?),
            DeviceKind::Metal => Err(Error::UnsupportedCapability(
                "Metal is not qualified".into(),
            )),
        }
    }
    fn dtype(&self) -> DType {
        match self.dtype {
            Precision::F32 => DType::F32,
            Precision::Bf16 => DType::BF16,
            Precision::F16 => DType::F16,
        }
    }
    /// Provenance identifier; does not imply numerical qualification.
    #[must_use]
    pub fn name(&self) -> String {
        format!(
            "{:?}:{}:{:?}:{:?}:{}",
            self.device, self.ordinal, self.dtype, self.modality, self.max_context_tokens
        )
        .to_ascii_lowercase()
    }
}
/// Planner estimate including shard staging, tiled attention, recurrence and 20% reserve.
#[derive(Debug, Clone, Copy, Serialize)]
#[serde(rename_all = "camelCase")]
#[non_exhaustive]
pub struct MemoryPlan {
    /// Required peak host bytes.
    pub host_bytes: u64,
    /// Required peak device bytes.
    pub device_bytes: u64,
}
/// Direct engine owns every model tensor on its caller's thread.
#[derive(Debug)]
pub struct DirectEngine {
    snapshot: VerifiedSnapshot,
    profile: ExecutionProfile,
    encoder: Encoder,
    backbone: Backbone,
    head: Head,
    device: Device,
    truncation: Truncation,
    #[cfg(feature = "vision")]
    vision: Option<crate::models::vision::Vision>,
}
impl DirectEngine {
    /// Conservative plan before allocation.
    ///
    /// # Errors
    /// Returns errors for invalid metadata or arithmetic overflow.
    pub fn memory_plan(
        snapshot: &VerifiedSnapshot,
        profile: &ExecutionProfile,
    ) -> Result<MemoryPlan> {
        profile.validate()?;
        let config: ModelConfig = serde_json::from_slice(
            &snapshot.read_verified("config.json", Instant::now() + Duration::from_secs(30))?,
        )?;
        config.validate()?;
        let c = &config.text_config;
        let payload: u64 = snapshot
            .manifest()
            .files
            .iter()
            .filter(|f| f.name.ends_with(".safetensors"))
            .map(|f| f.size)
            .sum();
        let largest = snapshot
            .manifest()
            .files
            .iter()
            .filter(|f| f.name.ends_with(".safetensors"))
            .map(|f| f.size)
            .max()
            .ok_or(Error::ArtifactMissing)?;
        let scalar = if profile.dtype == Precision::F32 {
            4_u64
        } else {
            2
        };
        let weights = payload
            .checked_mul(scalar)
            .and_then(|n| n.checked_div(2))
            .ok_or(Error::InsufficientMemory)?;
        let t = profile.max_context_tokens as u64;
        let h = c.hidden_size as u64;
        // Includes live hidden states, Q/K/V/gates, FFN intermediates and head evidence.
        let activations = t
            .checked_mul(h)
            .and_then(|n| n.checked_mul(scalar * 32))
            .ok_or(Error::InsufficientMemory)?;
        let attention = (c.num_attention_heads as u64)
            .checked_mul(128 * 256 * 4 * 4)
            .ok_or(Error::InsufficientMemory)?;
        let recurrence = (c.linear_num_value_heads as u64)
            .checked_mul(128 * 128 * 4 * 4)
            .ok_or(Error::InsufficientMemory)?;
        let media_scratch = if profile.modality == Modality::Image {
            1024 * 1024 * 1024
        } else {
            0
        };
        let scratch = activations
            .checked_add(attention)
            .and_then(|n| n.checked_add(media_scratch))
            .and_then(|n| n.checked_add(recurrence))
            .ok_or(Error::InsufficientMemory)?;
        let reserve = |n: u64| {
            n.checked_mul(6)
                .and_then(|n| n.checked_div(5))
                .ok_or(Error::InsufficientMemory)
        };
        let device_bytes = reserve(
            weights
                .checked_add(scratch)
                .ok_or(Error::InsufficientMemory)?,
        )?;
        let host_bytes = if profile.device == DeviceKind::Cpu {
            reserve(
                weights
                    .checked_add(scratch)
                    .and_then(|n| n.checked_add(largest * 2))
                    .ok_or(Error::InsufficientMemory)?,
            )?
        } else {
            reserve(
                largest
                    .checked_mul(2)
                    .and_then(|n| n.checked_add(64 * 1024 * 1024))
                    .ok_or(Error::InsufficientMemory)?,
            )?
        };
        Ok(MemoryPlan {
            host_bytes,
            device_bytes,
        })
    }
    /// Blocking offline load. May take minutes; the caller owns cancellation of its thread.
    ///
    /// # Errors
    /// Returns integrity, architecture, memory or tensor errors; never falls back.
    pub fn load(snapshot: VerifiedSnapshot, profile: ExecutionProfile) -> Result<Self> {
        profile.validate()?;
        let plan = Self::memory_plan(&snapshot, &profile)?;
        if plan.host_bytes > profile.host_budget_bytes
            || plan.device_bytes > profile.device_budget_bytes
        {
            return Err(Error::InsufficientMemory);
        }
        snapshot.verify()?;
        let deadline = Instant::now() + Duration::from_secs(600);
        let config: ModelConfig =
            serde_json::from_slice(&snapshot.read_verified("config.json", deadline)?)?;
        config.validate()?;
        let head_config: HeadConfig =
            serde_json::from_slice(&snapshot.read_verified("joint_head_config.json", deadline)?)?;
        head_config.validate(config.text_config.hidden_size)?;
        let encoder = Encoder::load(&snapshot.file("tokenizer.json")?)?;
        let device = profile.device()?;
        let mut weights = Weights::load(
            &snapshot,
            &device,
            profile.dtype(),
            profile.modality == Modality::Image,
        )?;
        let backbone = Backbone::load(&mut weights, config.text_config)?;
        let head = Head::load(&mut weights, &head_config)?;
        #[cfg(feature = "vision")]
        let vision = if profile.modality == Modality::Image {
            Some(crate::models::vision::Vision::load(
                &mut weights,
                config.vision_config,
            )?)
        } else {
            None
        };
        weights.finish()?;
        device.synchronize()?;
        Ok(Self {
            snapshot,
            profile,
            encoder,
            backbone,
            head,
            device,
            truncation: Truncation::Reject,
            #[cfg(feature = "vision")]
            vision,
        })
    }
    /// Set explicit state-prefix truncation.
    pub fn set_truncation(&mut self, policy: Truncation) {
        self.truncation = policy;
    }
    /// Blocking one-pass inference; no state is retained between decisions.
    ///
    /// # Errors
    /// Returns validation, deadline or inference errors. Kernel interruption is cooperative.
    pub fn decide(&mut self, request: &DecisionRequest) -> Result<DecisionResult> {
        #[cfg(feature = "vision")]
        if self.profile.modality == Modality::Text && !request.images.is_empty() {
            return Err(Error::UnsupportedCapability(
                "text worker does not accept images".into(),
            ));
        }
        let control = Control {
            cancel: Arc::new(AtomicBool::new(false)),
            deadline: Instant::now() + Duration::from_secs(300),
        };
        let encoded = self.encoder.encode(
            request,
            self.profile.max_context_tokens,
            self.truncation,
            None,
        )?;
        self.execute(request, &encoded, &control)
    }
    fn execute(
        &mut self,
        request: &DecisionRequest,
        encoded: &EncodedRecord,
        control: &Control,
    ) -> Result<DecisionResult> {
        let result = (|| {
            control.check()?;
            #[cfg(feature = "vision")]
            let hidden = if encoded.images.is_empty() {
                self.backbone.forward(&encoded.ids, control)?
            } else {
                let vision = self.vision.as_ref().ok_or_else(|| {
                    Error::UnsupportedCapability("text worker does not accept images".into())
                })?;
                let embeddings = self.backbone.embeddings(&encoded.ids)?;
                let mut pieces = Vec::new();
                let mut offset = 0;
                for (image, span) in encoded.images.iter().zip(&encoded.media_spans) {
                    if span.start() > offset {
                        pieces.push(embeddings.narrow(0, offset, span.start() - offset)?);
                    }
                    let visual = vision.forward(image, control)?;
                    if visual.dim(0)? != span.len() {
                        return Err(Error::InferenceFailed("visual embedding count".into()));
                    }
                    pieces.push(visual);
                    offset = span.end();
                }
                if offset < encoded.ids.len() {
                    pieces.push(embeddings.narrow(0, offset, encoded.ids.len() - offset)?);
                }
                self.backbone.forward_embedded(
                    Tensor::cat(&pieces, 0)?,
                    &encoded.positions,
                    control,
                )?
            };
            #[cfg(not(feature = "vision"))]
            let hidden = self.backbone.forward(&encoded.ids, control)?;
            let logits =
                self.head
                    .forward(&hidden, &self.backbone.output_embeddings, encoded, control)?;
            let answers = answers(request, &logits)?;
            Ok(DecisionResult {
                answers,
                input_tokens: encoded.ids.len(),
                truncated_state_tokens: encoded.truncated,
                revision: self.snapshot.manifest().revision.as_str().into(),
                manifest_digest: self.snapshot.digest().into(),
                execution_profile: self.profile.name(),
            })
        })();
        // Synchronize even after cooperative cancellation before reservations can drop.
        self.device.synchronize()?;
        result
    }
    fn warmup(&mut self) -> Result<()> {
        let request = warmup_request()?;
        self.decide(&request)?;
        Ok(())
    }
}
fn warmup_request() -> Result<DecisionRequest> {
    DecisionRequest::from_json(br#"{"state":"warmup","questions":{"n":{"type":"noul"},"c":{"type":"choice","criteria":{"a":null,"b":null}},"s":{"type":"score","criteria":[0,1]}}}"#)
}
/// Runtime admission, queue, deadline, and preparation limits.
#[derive(Debug, Clone, Deserialize, Serialize, TypedBuilder)]
#[serde(rename_all = "camelCase", deny_unknown_fields, default)]
#[non_exhaustive]
pub struct RuntimeConfig {
    /// Only one weight instance per model/device in this implementation.
    #[builder(default = 1)]
    pub workers_per_model: usize,
    /// Includes preparation, queued and active requests.
    #[builder(default = 8)]
    pub ingress_capacity: usize,
    /// Maximum queued records.
    #[builder(default = 8)]
    pub queue_capacity: usize,
    /// Aggregate encoded token reservations.
    #[builder(default = 32768)]
    pub max_queued_tokens: usize,
    /// Aggregate raw/prepared payload bytes.
    #[builder(default = 8_388_608)]
    pub max_payload_bytes: usize,
    /// Maximum concurrent tokenization work.
    #[builder(default = 2)]
    pub preparation_workers: usize,
    /// Batching is deliberately disabled.
    #[builder(default = 1)]
    pub max_batch_size: usize,
    /// Maximum waiting time before execution.
    #[builder(default = 5000)]
    pub queue_timeout_ms: u64,
    /// End-to-end request deadline.
    #[builder(default = 60000)]
    pub decision_timeout_ms: u64,
    /// Graceful shutdown deadline.
    #[builder(default = 30000)]
    pub shutdown_grace_ms: u64,
    /// Planner reserve, at least 20 percent.
    #[builder(default = 20)]
    pub memory_reserve_percent: u64,
    /// Explicit truncation policy.
    #[builder(default)]
    pub truncation: Truncation,
}
impl Default for RuntimeConfig {
    fn default() -> Self {
        Self::builder().build()
    }
}
impl RuntimeConfig {
    /// Validate all simultaneous hard caps.
    ///
    /// # Errors
    /// Returns an error for unsupported batch/replica counts or excessive budgets.
    pub fn validate(&self) -> Result<()> {
        if self.workers_per_model != 1
            || self.max_batch_size != 1
            || !(1..=256).contains(&self.ingress_capacity)
            || !(1..=256).contains(&self.queue_capacity)
            || !(1..=16).contains(&self.preparation_workers)
            || !(1..=262_144).contains(&self.max_queued_tokens)
            || !(1..=512 * 1024 * 1024).contains(&self.max_payload_bytes)
            || !(1..=300_000).contains(&self.decision_timeout_ms)
            || !(1..=300_000).contains(&self.queue_timeout_ms)
            || !(1..=300_000).contains(&self.shutdown_grace_ms)
            || self.memory_reserve_percent != 20
        {
            return Err(Error::InvalidRequest("runtime configuration".into()));
        }
        Ok(())
    }
}
/// Per-call principal and bounded deadline; dropped calls cancel queued/active work.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct DecisionOptions {
    principal: String,
    timeout: Duration,
}
impl DecisionOptions {
    /// Checked principal and deadline; deadline also cannot exceed the runtime ceiling.
    ///
    /// # Errors
    /// Rejects empty/oversized principal or a deadline over 300 seconds.
    pub fn new(principal: String, timeout: Duration) -> Result<Self> {
        if principal.is_empty()
            || principal.len() > 256
            || timeout.is_zero()
            || timeout > Duration::from_secs(300)
        {
            return Err(Error::InvalidRequest("decision options".into()));
        }
        Ok(Self { principal, timeout })
    }
}
impl Default for DecisionOptions {
    fn default() -> Self {
        Self {
            principal: "embedded".into(),
            timeout: Duration::from_secs(60),
        }
    }
}
#[derive(Debug)]
struct Job {
    request: DecisionRequest,
    record: EncodedRecord,
    control: Control,
    queue_deadline: Instant,
    principal: String,
    reply: oneshot::Sender<Result<DecisionResult>>,
    _ingress: OwnedSemaphorePermit,
    _bytes: OwnedSemaphorePermit,
    _tokens: OwnedSemaphorePermit,
}
#[derive(Debug)]
struct CancelOnDrop(Arc<AtomicBool>);
impl Drop for CancelOnDrop {
    fn drop(&mut self) {
        self.0.store(true, Ordering::Release);
    }
}
/// Cloneable bounded admission handle; does not own model tensors.
#[derive(Debug, Clone)]
pub struct DecisionClient {
    jobs: mpsc::Sender<Job>,
    encoder: Arc<Encoder>,
    config: Arc<RuntimeConfig>,
    context: usize,
    #[cfg(feature = "vision")]
    modality: Modality,
    ingress: Arc<Semaphore>,
    bytes: Arc<Semaphore>,
    tokens: Arc<Semaphore>,
    preparation: Arc<Semaphore>,
    state: Arc<AtomicU8>,
}
impl DecisionClient {
    /// Prepare/admit one record and await its typed answer.
    ///
    /// # Errors
    /// Returns queue, deadline, cancellation, validation or execution errors.
    #[tracing::instrument(skip_all)]
    pub async fn decide(
        &self,
        request: DecisionRequest,
        options: DecisionOptions,
    ) -> Result<DecisionResult> {
        if self.state.load(Ordering::Acquire) != WorkerState::Ready as u8 {
            return Err(Error::ShuttingDown);
        }
        #[cfg(feature = "vision")]
        if self.modality == Modality::Text && !request.images.is_empty() {
            return Err(Error::UnsupportedCapability(
                "text worker does not accept images".into(),
            ));
        }
        let started = Instant::now();
        let duration = options
            .timeout
            .min(Duration::from_millis(self.config.decision_timeout_ms));
        let deadline = started + duration;
        let ingress = self
            .ingress
            .clone()
            .try_acquire_owned()
            .map_err(|_| Error::QueueFull)?;
        let bytes = u32::try_from(request.payload_bytes).map_err(|_| Error::QueueFull)?;
        let bytes = self
            .bytes
            .clone()
            .try_acquire_many_owned(bytes)
            .map_err(|_| Error::QueueFull)?;
        let prep = self
            .preparation
            .clone()
            .try_acquire_owned()
            .map_err(|_| Error::QueueFull)?;
        let cancel = Arc::new(AtomicBool::new(false));
        let _cancel = CancelOnDrop(cancel.clone());
        let tokenizer = self.encoder.clone();
        let context = self.context;
        let truncation = self.config.truncation;
        let c = Control {
            cancel: cancel.clone(),
            deadline,
        };
        let preparation_control = c.clone();
        let preparation = task::spawn_blocking(move || {
            let _permit = prep;
            preparation_control.check()?;
            let encoded = tokenizer.encode(&request, context, truncation, None)?;
            Ok::<_, Error>((request, encoded, ingress, bytes))
        });
        let (request, encoded, ingress, bytes) =
            timeout_at(tokio::time::Instant::from_std(deadline), preparation)
                .await
                .map_err(|_| Error::DeadlineExceeded)?
                .map_err(|_| Error::WorkerUnavailable)??;
        c.check()?;
        let tokens = self
            .tokens
            .clone()
            .try_acquire_many_owned(
                u32::try_from(encoded.token_count()).map_err(|_| Error::QueueFull)?,
            )
            .map_err(|_| Error::QueueFull)?;
        let (reply, receiver) = oneshot::channel();
        let queue_deadline =
            deadline.min(Instant::now() + Duration::from_millis(self.config.queue_timeout_ms));
        self.jobs
            .try_send(Job {
                request,
                record: encoded,
                control: c,
                queue_deadline,
                principal: options.principal,
                reply,
                _ingress: ingress,
                _bytes: bytes,
                _tokens: tokens,
            })
            .map_err(|e| match e {
                mpsc::error::TrySendError::Full(_) => Error::QueueFull,
                mpsc::error::TrySendError::Closed(_) => Error::ShuttingDown,
            })?;
        let result = timeout_at(tokio::time::Instant::from_std(deadline), receiver)
            .await
            .map_err(|_| Error::DeadlineExceeded)?
            .map_err(|_| Error::WorkerUnavailable)??;
        Ok(result)
    }
    /// Ready means warmed and accepting work, independently of saturation.
    #[must_use]
    pub fn is_ready(&self) -> bool {
        self.state.load(Ordering::Acquire) == WorkerState::Ready as u8
    }
}
/// Completion of a graceful stop.
#[derive(Debug, Clone, Copy)]
#[non_exhaustive]
pub struct ShutdownReport {
    /// False if a blocking device call remains active after the grace deadline.
    pub drained: bool,
}
/// Owner of the scheduler and dedicated model worker.
#[derive(Debug)]
pub struct Runtime {
    client: DecisionClient,
    stop: watch::Sender<bool>,
    scheduler: Option<task::JoinHandle<()>>,
    worker: Option<JoinHandle<()>>,
    config: RuntimeConfig,
}
impl Runtime {
    /// Load on a dedicated thread, warm all question types, then open admission.
    ///
    /// # Errors
    /// Returns load/warmup errors before readiness, without network activity.
    #[tracing::instrument(skip_all)]
    pub async fn start(
        snapshot: VerifiedSnapshot,
        profile: ExecutionProfile,
        config: RuntimeConfig,
    ) -> Result<Self> {
        config.validate()?;
        profile.validate()?;
        let tokenizer_path = snapshot.file("tokenizer.json")?;
        let encoder = Arc::new(
            task::spawn_blocking(move || Encoder::load(&tokenizer_path))
                .await
                .map_err(|_| Error::WorkerUnavailable)??,
        );
        let context = profile.max_context_tokens;
        encoder.encode(&warmup_request()?, context, config.truncation, None)?;
        #[cfg(feature = "vision")]
        let modality = profile.modality;
        let (jobs, receiver) = mpsc::channel(config.queue_capacity);
        let (work, worker_jobs) = mpsc::channel::<Job>(1);
        let (events, event_receiver) = mpsc::channel::<bool>(8);
        let (ready, readiness) = oneshot::channel();
        let state = Arc::new(AtomicU8::new(WorkerState::Loading as u8));
        let worker_state = state.clone();
        let truncation = config.truncation;
        let worker = thread::Builder::new()
            .name("clef-device-worker".into())
            .spawn(move || {
                worker_loop(
                    snapshot,
                    profile,
                    truncation,
                    worker_jobs,
                    events,
                    ready,
                    worker_state,
                );
            })?;
        let loaded = timeout(Duration::from_secs(600), readiness).await;
        match loaded {
            Ok(Ok(Ok(()))) => {}
            Ok(Ok(Err(e))) => {
                task::spawn_blocking(move || worker.join())
                    .await
                    .map_err(|_| Error::WorkerUnavailable)?
                    .map_err(|_| Error::WorkerUnavailable)?;
                return Err(e);
            }
            _ => {
                drop(work);
                state.store(WorkerState::Draining as u8, Ordering::Release);
                task::spawn_blocking(move || worker.join())
                    .await
                    .map_err(|_| Error::WorkerUnavailable)?
                    .map_err(|_| Error::WorkerUnavailable)?;
                return Err(Error::DeadlineExceeded);
            }
        }
        let (stop, stop_rx) = watch::channel(false);
        let scheduler_state = state.clone();
        let capacity = config.queue_capacity;
        let scheduler = task::spawn(async move {
            schedule(
                receiver,
                work,
                event_receiver,
                stop_rx,
                scheduler_state,
                capacity,
            )
            .await;
        });
        tracing::info!("Flash worker loaded and warmed; admission open");
        let client = DecisionClient {
            jobs,
            encoder,
            config: Arc::new(config.clone()),
            context,
            #[cfg(feature = "vision")]
            modality,
            ingress: Arc::new(Semaphore::new(config.ingress_capacity)),
            bytes: Arc::new(Semaphore::new(config.max_payload_bytes)),
            tokens: Arc::new(Semaphore::new(config.max_queued_tokens)),
            preparation: Arc::new(Semaphore::new(config.preparation_workers)),
            state,
        };
        Ok(Self {
            client,
            stop,
            scheduler: Some(scheduler),
            worker: Some(worker),
            config,
        })
    }
    /// Clone the tensor-free client.
    #[must_use]
    pub fn client(&self) -> DecisionClient {
        self.client.clone()
    }
    /// Close admission immediately and ask the scheduler to drain active execution.
    /// Calling this repeatedly is harmless; [`Self::shutdown`] still joins both owners.
    pub fn close_admission(&self) {
        self.client
            .state
            .store(WorkerState::Draining as u8, Ordering::Release);
        let _ = self.stop.send(true);
    }
    /// Close admission, cancel queued jobs, and join the scheduler/worker.
    ///
    /// # Errors
    /// Returns errors for failed supervisor joins. A timed-out kernel cannot be killed.
    pub async fn shutdown(mut self) -> Result<ShutdownReport> {
        self.close_admission();
        let deadline =
            tokio::time::Instant::now() + Duration::from_millis(self.config.shutdown_grace_ms);
        if let Some(mut scheduler) = self.scheduler.take() {
            match timeout_at(deadline, &mut scheduler).await {
                Ok(Ok(())) => {}
                Ok(Err(_)) => return Err(Error::WorkerUnavailable),
                Err(_) => {
                    self.scheduler = Some(scheduler);
                    return Ok(ShutdownReport { drained: false });
                }
            }
        }
        if let Some(worker) = self.worker.take() {
            let joined = task::spawn_blocking(move || worker.join());
            match timeout_at(deadline, joined).await {
                Ok(Ok(Ok(()))) => {}
                Ok(_) => return Err(Error::WorkerUnavailable),
                Err(_) => return Ok(ShutdownReport { drained: false }),
            }
        }
        Ok(ShutdownReport { drained: true })
    }
}
impl Drop for Runtime {
    fn drop(&mut self) {
        self.client
            .state
            .store(WorkerState::Draining as u8, Ordering::Release);
        let _ = self.stop.send(true);
    }
}
#[allow(
    clippy::needless_pass_by_value,
    reason = "the worker owns these handles for its entire thread lifetime"
)]
fn worker_loop(
    snapshot: VerifiedSnapshot,
    profile: ExecutionProfile,
    truncation: Truncation,
    mut jobs: mpsc::Receiver<Job>,
    events: mpsc::Sender<bool>,
    ready: oneshot::Sender<Result<()>>,
    state: Arc<AtomicU8>,
) {
    let startup = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let mut engine = DirectEngine::load(snapshot.clone(), profile.clone())?;
        engine.set_truncation(truncation);
        engine.warmup()?;
        Ok::<_, Error>(engine)
    }));
    let mut engine = match startup {
        Ok(Ok(engine)) => engine,
        Ok(Err(e)) => {
            let _ = ready.send(Err(e));
            return;
        }
        Err(_) => {
            let _ = ready.send(Err(Error::WorkerUnavailable));
            return;
        }
    };
    state.store(WorkerState::Ready as u8, Ordering::Release);
    if ready.send(Ok(())).is_err() {
        state.store(WorkerState::Stopped as u8, Ordering::Release);
        return;
    }
    let mut restarts = VecDeque::new();
    while let Some(job) = jobs.blocking_recv() {
        let Job {
            request,
            record,
            control,
            reply,
            _ingress: ingress_reservation,
            _bytes: byte_reservation,
            _tokens: token_reservation,
            ..
        } = job;
        let execution = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            engine.execute(&request, &record, &control)
        }));
        drop((
            ingress_reservation,
            byte_reservation,
            token_reservation,
            request,
            record,
            control,
        ));
        if let Ok(result) = execution {
            let _ = reply.send(result);
            let _ = events.blocking_send(true);
        } else {
            let _ = reply.send(Err(Error::WorkerUnavailable));
            tracing::warn!("device worker panicked; active decision failed without replay");
            drop(engine);
            let recovered = recover_worker(&state, || {
                let now = Instant::now();
                while restarts
                    .front()
                    .is_some_and(|t| now.duration_since(*t) > Duration::from_secs(600))
                {
                    restarts.pop_front();
                }
                if restarts.len() >= 3 {
                    return Err(Error::WorkerUnavailable);
                }
                restarts.push_back(now);
                thread::sleep(Duration::from_millis(250 * (1 << restarts.len())));
                let mut engine = DirectEngine::load(snapshot.clone(), profile.clone())?;
                engine.set_truncation(truncation);
                engine.warmup()?;
                Ok(engine)
            });
            engine = if let Ok(engine) = recovered {
                engine
            } else {
                let _ = events.blocking_send(false);
                return;
            };
            let _ = events.blocking_send(true);
        }
    }
    state.store(WorkerState::Stopped as u8, Ordering::Release);
}
// Recovery may only transition ready -> loading -> ready. A concurrent drain
// owns the terminal transition and must never be overwritten by a reload.
fn recover_worker<T>(state: &AtomicU8, reload: impl FnOnce() -> Result<T>) -> Result<T> {
    state
        .compare_exchange(
            WorkerState::Ready as u8,
            WorkerState::Loading as u8,
            Ordering::AcqRel,
            Ordering::Acquire,
        )
        .map_err(|_| Error::ShuttingDown)?;
    let engine = std::panic::catch_unwind(std::panic::AssertUnwindSafe(reload))
        .map_err(|_| Error::WorkerUnavailable)??;
    state
        .compare_exchange(
            WorkerState::Loading as u8,
            WorkerState::Ready as u8,
            Ordering::AcqRel,
            Ordering::Acquire,
        )
        .map_err(|_| Error::ShuttingDown)?;
    Ok(engine)
}
async fn schedule(
    mut jobs: mpsc::Receiver<Job>,
    work: mpsc::Sender<Job>,
    mut events: mpsc::Receiver<bool>,
    mut stop: watch::Receiver<bool>,
    state: Arc<AtomicU8>,
    capacity: usize,
) {
    let mut queues: HashMap<String, VecDeque<Job>> = HashMap::new();
    let mut round_robin = VecDeque::new();
    let mut count = 0;
    let mut busy = false;
    let mut draining = false;
    let mut tick = interval(Duration::from_millis(10));
    loop {
        tokio::select! {
            biased;
            changed=stop.changed(),if !draining=>{if changed.is_err() || *stop.borrow() {draining=true;state.store(WorkerState::Draining as u8,Ordering::Release);jobs.close();}},
            event=events.recv(),if busy=>{busy=false;if event!=Some(true){draining=true;state.store(WorkerState::Draining as u8,Ordering::Release);jobs.close();}},
            next=jobs.recv(),if !draining=>{
                let Some(job)=next else{draining=true;continue;};
                if count>=capacity || (!queues.contains_key(&job.principal) && queues.len()>=256) {let _=job.reply.send(Err(Error::QueueFull));}
                else {if !queues.contains_key(&job.principal){round_robin.push_back(job.principal.clone());}queues.entry(job.principal.clone()).or_default().push_back(job);count+=1;}
            },
            _=tick.tick()=>{},
        }
        for queue in queues.values_mut() {
            let mut retained = VecDeque::new();
            while let Some(job) = queue.pop_front() {
                if draining
                    || job.reply.is_closed()
                    || job.control.cancel.load(Ordering::Acquire)
                    || Instant::now() >= job.queue_deadline
                {
                    let error = if draining {
                        Error::ShuttingDown
                    } else if job.reply.is_closed() {
                        Error::Cancelled
                    } else {
                        Error::DeadlineExceeded
                    };
                    let _ = job.reply.send(Err(error));
                    count -= 1;
                } else {
                    retained.push_back(job);
                }
            }
            *queue = retained;
        }
        queues.retain(|_, q| !q.is_empty());
        round_robin.retain(|p| queues.contains_key(p));
        if draining {
            while let Ok(job) = jobs.try_recv() {
                let _ = job.reply.send(Err(Error::ShuttingDown));
            }
            if !busy {
                break;
            }
        } else if !busy
            && let Some(principal) = round_robin.pop_front()
            && let Some(queue) = queues.get_mut(&principal)
        {
            if let Some(job) = queue.pop_front() {
                count -= 1;
                match work.try_send(job) {
                    Ok(()) => busy = true,
                    Err(e) => {
                        let _ = e.into_inner().reply.send(Err(Error::WorkerUnavailable));
                        draining = true;
                    }
                }
            }
            if queue.is_empty() {
                queues.remove(&principal);
            } else {
                round_robin.push_back(principal);
            }
        }
    }
    drop(work);
    state.store(WorkerState::Stopped as u8, Ordering::Release);
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_should_validate_runtime_caps_and_explicit_profiles() -> Result<()> {
        RuntimeConfig::default().validate()?;
        assert!(
            RuntimeConfig {
                max_batch_size: 2,
                ..RuntimeConfig::default()
            }
            .validate()
            .is_err()
        );
        let profile = ExecutionProfile::builder()
            .device(DeviceKind::Cpu)
            .dtype(Precision::F32)
            .modality(Modality::Text)
            .max_context_tokens(4096)
            .device_budget_bytes(64 * 1024 * 1024 * 1024)
            .host_budget_bytes(64 * 1024 * 1024 * 1024)
            .build();
        profile.validate()?;
        assert!(
            ExecutionProfile {
                dtype: Precision::Bf16,
                ..profile
            }
            .validate()
            .is_err()
        );
        assert!("../../model".parse::<crate::types::Identifier>().is_err());
        Ok(())
    }
}

#[cfg(test)]
mod release_tests {
    use super::*;
    use crate::{artifacts::ArtifactStore, types::ModelPreset};
    #[tokio::test]
    #[ignore = "requires the pinned Flash cache and 100-record Python oracle"]
    async fn test_should_match_full_flash_python_probabilities() -> Result<()> {
        flash_parity(100, Modality::Text).await
    }
    #[cfg(feature = "vision")]
    #[tokio::test]
    #[ignore = "requires pinned Flash weights and extended image/context oracle"]
    async fn test_should_match_full_flash_image_and_context_probabilities() -> Result<()> {
        flash_parity(3, Modality::Image).await
    }
    #[cfg(feature = "vision")]
    #[tokio::test]
    #[ignore = "requires full Flash weights and the JPEG oracle"]
    async fn test_should_match_full_flash_jpeg_probabilities() -> Result<()> {
        flash_parity(2, Modality::Image).await
    }
    #[allow(
        clippy::disallowed_methods,
        reason = "verification reads a bounded offline oracle"
    )]
    async fn flash_parity(expected_count: usize, modality: Modality) -> Result<()> {
        let root = std::env::var_os("CLEF_RELEASE_CACHE").ok_or(Error::ArtifactMissing)?;
        let oracle = std::env::var_os("CLEF_RELEASE_ORACLE").ok_or(Error::ArtifactMissing)?;
        let data: serde_json::Value = serde_json::from_slice(&std::fs::read(oracle)?)?;
        let store = ArtifactStore::new(root.into(), 85_899_345_920)?;
        let snapshot = store.open(ModelPreset::ClefFlash).await?;
        let profile = ExecutionProfile::builder()
            .device(DeviceKind::Cpu)
            .dtype(Precision::F32)
            .modality(modality)
            .max_context_tokens(4096)
            .device_budget_bytes(64 * 1024 * 1024 * 1024)
            .host_budget_bytes(64 * 1024 * 1024 * 1024)
            .build();
        task::spawn_blocking(move || {
            let mut engine = DirectEngine::load(snapshot, profile)?;
            let records = data
                .get("records")
                .and_then(serde_json::Value::as_array)
                .ok_or_else(|| Error::InvalidRequest("oracle records".into()))?;
            assert_eq!(records.len(), expected_count);
            assert_eq!(
                data["revision"].as_str(),
                Some(ModelPreset::ClefFlash.revision())
            );
            let mut maximum = 0.0_f64;
            let mut total = 0.0_f64;
            let mut count = 0_u32;
            for (index, record) in records.iter().enumerate() {
                let bytes = serde_json::to_vec(&record["request"])?;
                let request = DecisionRequest::from_json(&bytes)?;
                let actual = engine.decide(&request)?;
                assert_eq!(
                    Some(actual.input_tokens as u64),
                    record["inputTokens"].as_u64()
                );
                for (id, answer) in &actual.answers {
                    let probabilities = match answer {
                        crate::types::Answer::Noul { noul } => {
                            serde_json::json!({"true":noul.get(),"false":1.0-noul.get()})
                        }
                        crate::types::Answer::Choice { probabilities, .. }
                        | crate::types::Answer::Score { probabilities, .. } => {
                            serde_json::Value::Object(probabilities.clone())
                        }
                    };
                    let expected = record["probabilities"][id]
                        .as_object()
                        .ok_or_else(|| Error::InvalidRequest("oracle probabilities".into()))?;
                    for (option, value) in expected {
                        let expected = value
                            .as_f64()
                            .ok_or_else(|| Error::InvalidRequest("oracle probability".into()))?;
                        let actual = probabilities[option]
                            .as_f64()
                            .ok_or_else(|| Error::InferenceFailed("missing probability".into()))?;
                        let error = (actual - expected).abs();
                        maximum = maximum.max(error);
                        total += error;
                        count += 1;
                        assert!(
                            error <= 1e-3,
                            "record {index} {id}/{option}: {actual} != {expected}; error={error}"
                        );
                    }
                }
                eprintln!(
                    "Flash parity {}/{} maximum_error={maximum}",
                    index + 1,
                    records.len()
                );
            }
            let mean = total / f64::from(count);
            assert!(mean <= 1e-4, "mean error {mean}");
            eprintln!(
                "Flash F32 qualification: {count} probabilities, maximum={maximum}, mean={mean}"
            );
            Ok(())
        })
        .await
        .map_err(|_| Error::WorkerUnavailable)?
    }
    #[tokio::test]
    #[ignore = "requires complete pinned weights and a large-memory runner"]
    async fn test_should_execute_full_flash_release_offline() -> Result<()> {
        let root = std::env::var_os("CLEF_RELEASE_CACHE").ok_or(Error::ArtifactMissing)?;
        let store = ArtifactStore::new(root.into(), 85_899_345_920)?;
        let snapshot = store.open(ModelPreset::ClefFlash).await?;
        let profile = ExecutionProfile::builder()
            .device(DeviceKind::Cpu)
            .dtype(Precision::F32)
            .modality(Modality::Text)
            .max_context_tokens(512)
            .device_budget_bytes(64 * 1024 * 1024 * 1024)
            .host_budget_bytes(64 * 1024 * 1024 * 1024)
            .build();
        let request=DecisionRequest::from_json(br#"{"state":"Checkout has failed for every customer for the last hour.","questions":{"urgent":{"type":"noul","instructions":"Is this urgent?"},"team":{"type":"choice","criteria":{"billing":"invoices","technical":"outages"}},"severity":{"type":"score","criteria":["minor","critical"]}}}"#)?;
        task::spawn_blocking(move || {
            let mut engine = DirectEngine::load(snapshot, profile)?;
            let first = engine.decide(&request)?;
            let second = engine.decide(&request)?;
            assert_eq!(
                first.systemone("clef-flash")?,
                second.systemone("clef-flash")?
            );
            assert_eq!(first.answers.len(), 3);
            assert_eq!(first.truncated_state_tokens, 0);
            Ok(())
        })
        .await
        .map_err(|_| Error::WorkerUnavailable)?
    }
}

#[cfg(test)]
mod scheduler_tests {
    use super::*;
    #[test]
    fn test_should_keep_shutdown_terminal_during_worker_recovery() -> Result<()> {
        let state = AtomicU8::new(WorkerState::Ready as u8);
        let resource = Arc::new(());
        let engine = recover_worker(&state, || Ok(resource.clone()))?;
        assert_eq!(state.load(Ordering::Acquire), WorkerState::Ready as u8);
        drop(engine);
        let interrupted = recover_worker(&state, || {
            assert_eq!(state.load(Ordering::Acquire), WorkerState::Loading as u8);
            state.store(WorkerState::Draining as u8, Ordering::Release);
            Ok(resource.clone())
        });
        assert!(matches!(interrupted, Err(Error::ShuttingDown)));
        assert_eq!(state.load(Ordering::Acquire), WorkerState::Draining as u8);
        assert_eq!(Arc::strong_count(&resource), 1);
        let reload_started = AtomicBool::new(false);
        let stopped = recover_worker(&state, || {
            reload_started.store(true, Ordering::Release);
            Ok(())
        });
        assert!(matches!(stopped, Err(Error::ShuttingDown)));
        assert!(!reload_started.load(Ordering::Acquire));
        Ok(())
    }
    #[test]
    fn test_should_fail_reload_panics_without_reopening_admission() {
        let state = AtomicU8::new(WorkerState::Ready as u8);
        let result = recover_worker::<()>(&state, || panic!("injected worker reload panic"));
        assert!(matches!(result, Err(Error::WorkerUnavailable)));
        assert_eq!(state.load(Ordering::Acquire), WorkerState::Loading as u8);
    }
    fn job(
        principal: &str,
        ingress: Arc<Semaphore>,
        deadline: Instant,
    ) -> Result<(Job, oneshot::Receiver<Result<DecisionResult>>)> {
        let request =
            DecisionRequest::from_json(br#"{"state":null,"questions":{"q":{"type":"noul"}}}"#)?;
        let record = EncodedRecord {
            ids: vec![0; 20],
            questions: Vec::new(),
            truncated: 0,
            #[cfg(feature = "vision")]
            images: Vec::new(),
            #[cfg(feature = "vision")]
            media_spans: Vec::new(),
            #[cfg(feature = "vision")]
            positions: vec![[0; 3]; 20],
        };
        let (reply, receiver) = oneshot::channel();
        let permit = ingress.try_acquire_owned().map_err(|_| Error::QueueFull)?;
        let bytes = Arc::new(Semaphore::new(1))
            .try_acquire_owned()
            .map_err(|_| Error::QueueFull)?;
        let tokens = Arc::new(Semaphore::new(1))
            .try_acquire_owned()
            .map_err(|_| Error::QueueFull)?;
        Ok((
            Job {
                request,
                record,
                control: Control {
                    cancel: Arc::new(AtomicBool::new(false)),
                    deadline,
                },
                queue_deadline: deadline,
                principal: principal.into(),
                reply,
                _ingress: permit,
                _bytes: bytes,
                _tokens: tokens,
            },
            receiver,
        ))
    }
    #[tokio::test]
    async fn test_should_hold_active_reservations_and_cancel_queued_work() -> Result<()> {
        let ingress = Arc::new(Semaphore::new(2));
        let (send, recv) = mpsc::channel(8);
        let (worker, mut work) = mpsc::channel(1);
        let (events, received) = mpsc::channel(1);
        let (stop, stopped) = watch::channel(false);
        let state = Arc::new(AtomicU8::new(WorkerState::Ready as u8));
        let task = tokio::spawn(schedule(recv, worker, received, stopped, state.clone(), 8));
        let (first, first_reply) = job(
            "one",
            ingress.clone(),
            Instant::now() + Duration::from_secs(60),
        )?;
        send.send(first)
            .await
            .map_err(|_| Error::WorkerUnavailable)?;
        let active = work.recv().await.ok_or(Error::WorkerUnavailable)?;
        let (second, second_reply) = job(
            "two",
            ingress.clone(),
            Instant::now() + Duration::from_secs(60),
        )?;
        send.send(second)
            .await
            .map_err(|_| Error::WorkerUnavailable)?;
        assert_eq!(ingress.available_permits(), 0);
        drop(second_reply);
        tokio::time::sleep(Duration::from_millis(30)).await;
        assert_eq!(ingress.available_permits(), 1);
        drop(first_reply);
        assert_eq!(ingress.available_permits(), 1);
        stop.send(true).map_err(|_| Error::WorkerUnavailable)?;
        tokio::time::sleep(Duration::from_millis(20)).await;
        assert!(!task.is_finished());
        drop(active);
        events
            .send(true)
            .await
            .map_err(|_| Error::WorkerUnavailable)?;
        task.await.map_err(|_| Error::WorkerUnavailable)?;
        assert_eq!(ingress.available_permits(), 2);
        assert_eq!(state.load(Ordering::Acquire), WorkerState::Stopped as u8);
        Ok(())
    }
    #[tokio::test]
    async fn test_should_expire_queued_jobs_and_schedule_principals_fairly() -> Result<()> {
        let ingress = Arc::new(Semaphore::new(8));
        let (send, recv) = mpsc::channel(8);
        let (worker, mut work) = mpsc::channel(1);
        let (events, received) = mpsc::channel(1);
        let (stop, stopped) = watch::channel(false);
        let state = Arc::new(AtomicU8::new(WorkerState::Ready as u8));
        let task = tokio::spawn(schedule(recv, worker, received, stopped, state, 8));
        let (first, _reply) = job(
            "active",
            ingress.clone(),
            Instant::now() + Duration::from_secs(60),
        )?;
        send.send(first)
            .await
            .map_err(|_| Error::WorkerUnavailable)?;
        let active = work.recv().await.ok_or(Error::WorkerUnavailable)?;
        let (expired, reply) = job(
            "expired",
            ingress.clone(),
            Instant::now()
                .checked_sub(Duration::from_secs(1))
                .ok_or(Error::DeadlineExceeded)?,
        )?;
        send.send(expired)
            .await
            .map_err(|_| Error::WorkerUnavailable)?;
        assert!(matches!(
            reply.await.map_err(|_| Error::WorkerUnavailable)?,
            Err(Error::DeadlineExceeded)
        ));
        let mut replies = Vec::new();
        for principal in ["a", "a", "b", "b"] {
            let (next, reply) = job(
                principal,
                ingress.clone(),
                Instant::now() + Duration::from_secs(60),
            )?;
            send.send(next)
                .await
                .map_err(|_| Error::WorkerUnavailable)?;
            replies.push(reply);
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
        drop(active);
        events
            .send(true)
            .await
            .map_err(|_| Error::WorkerUnavailable)?;
        for expected in ["a", "b", "a", "b"] {
            let next = work.recv().await.ok_or(Error::WorkerUnavailable)?;
            assert_eq!(next.principal, expected);
            drop(next);
            events
                .send(true)
                .await
                .map_err(|_| Error::WorkerUnavailable)?;
        }
        stop.send(true).map_err(|_| Error::WorkerUnavailable)?;
        task.await.map_err(|_| Error::WorkerUnavailable)?;
        drop(replies);
        Ok(())
    }
}
