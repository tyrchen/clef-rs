//! Checked safe dispatch for the fused F32 gated-DeltaNet recurrence.
//!
//! A group owns four SIMD vectors of value columns for one head. Each lane keeps key states
//! in registers. SIMD reductions combine dot products without shared memory or
//! cross-group barriers. Every output is written once. Validated dimensions, contiguous
//! offsets and buffer lengths bound every shader address. No raw FFI or unsafe
//! Rust is used; Candle owns allocation, command lifetime and hazard tracking.

use std::sync::OnceLock;

use candle_core::{
    CpuStorage, CustomOp1, CustomOp2, CustomOp3, DType, Device, Error as CandleError, Layout,
    MetalDevice, MetalStorage, Result as CandleResult, Shape, Tensor, backend::BackendStorage,
};
use candle_metal_kernels::{
    metal::{ComputePipeline, ConstantValues, Value},
    source::SDPA,
};
use objc2_metal::{MTLComputePipelineState, MTLDevice, MTLGPUFamily, MTLSize};

use super::{Control, MAX_SEQUENCE_TOKENS, get_or_init_fallible, pointwise::DeltaPrepareKernel};
use crate::{Error, Result};

/// String replacement that fails unless the pattern occurs exactly once, so a
/// shader patch can never silently become a no-op after an upstream edit.
fn replace_once(source: &str, from: &str, to: &str) -> Result<String> {
    if source.matches(from).count() != 1 {
        return Err(Error::InferenceFailed(
            "shader patch pattern matched zero or multiple times".into(),
        ));
    }
    Ok(source.replacen(from, to, 1))
}

/// Retain exact-sized storage rather than an oversized best-fit scratch allocation.
/// The pinned allocator may reuse a large free projection buffer for a tiny snapshot.
/// Uploading bounded zero bytes obtains a fresh tracked buffer, then a safe device copy
/// fills it without GPU readback. Residency and reclamation remain Candle-owned.
pub(super) fn compact_copy(x: &Tensor) -> CandleResult<Tensor> {
    x.apply_op1_no_bwd(&CompactCopy)
}
#[derive(Debug)]
struct CompactCopy;
impl CustomOp1 for CompactCopy {
    fn name(&self) -> &'static str {
        "clef-compact-prefix-copy"
    }
    fn cpu_fwd(&self, _: &CpuStorage, _: &Layout) -> CandleResult<(CpuStorage, Shape)> {
        Err(CandleError::Msg(
            "compact prefix copy requires Metal".into(),
        ))
    }
    fn metal_fwd(
        &self,
        input: &MetalStorage,
        layout: &Layout,
    ) -> CandleResult<(MetalStorage, Shape)> {
        let count = layout.shape().elem_count();
        if !matches!(input.dtype(), DType::F32 | DType::F16) || !(1..=16_777_216).contains(&count) {
            return Err(CandleError::Msg(
                "compact prefix copy dimensions or dtype".into(),
            ));
        }
        let scalar = input.dtype().size_in_bytes();
        let last = layout.dims().iter().zip(layout.stride()).try_fold(
            layout.start_offset(),
            |offset, (dim, stride)| {
                dim.checked_sub(1)
                    .and_then(|n| n.checked_mul(*stride))
                    .and_then(|n| offset.checked_add(n))
                    .ok_or_else(|| CandleError::Msg("compact prefix source offset overflow".into()))
            },
        )?;
        let end = last
            .checked_add(1)
            .and_then(|n| n.checked_mul(scalar))
            .ok_or_else(|| CandleError::Msg("compact prefix source byte overflow".into()))?;
        if end > input.buffer().length() {
            return Err(CandleError::Msg("compact prefix source bounds".into()));
        }
        let bytes = count
            .checked_mul(scalar)
            .ok_or_else(|| CandleError::Msg("compact prefix byte overflow".into()))?;
        let device = input.device();
        // The destination is fully overwritten by `copy_strided_src` below,
        // so skip the zeroed staging allocation.
        let buffer = device.new_buffer(bytes, DType::U8, "clef_compact_copy")?;
        let mut output = MetalStorage::new(buffer, device.clone(), count, input.dtype());
        input.copy_strided_src(&mut output, 0, layout)?;
        Ok((output, layout.shape().clone()))
    }
}

#[derive(Debug, Clone)]
pub(super) struct DeltaKernel {
    /// Compiled on first dispatch, not at model load: shader compilation is
    /// the dominant load-latency cost and the cached variant is only needed
    /// when prefix capture/resume is actually used.
    pipeline: OnceLock<ComputePipeline>,
    cached_pipeline: OnceLock<ComputePipeline>,
    preparation: Option<DeltaPrepareKernel>,
    key_dim: usize,
    value_dim: usize,
    values: usize,
}
impl DeltaKernel {
    pub fn new(device: &Device, key_dim: usize, value_dim: usize) -> Result<Option<Self>> {
        let values = match device {
            Device::Metal(metal)
                if key_dim == 128
                    && value_dim == 128
                    && metal
                        .device()
                        .as_ref()
                        .supportsFamily(MTLGPUFamily::Apple10) =>
            {
                4
            }
            _ => 1,
        };
        Self::new_variant(device, key_dim, value_dim, values)
    }
    fn new_variant(
        device: &Device,
        key_dim: usize,
        value_dim: usize,
        values: usize,
    ) -> Result<Option<Self>> {
        if ![1, 2, 4, 8, 16].contains(&values) {
            return Err(Error::InvalidRequest("DeltaNet vector width".into()));
        }
        let Device::Metal(device) = device else {
            return Ok(None);
        };
        dimensions(1, 1, key_dim, value_dim)?;
        Ok(Some(Self {
            pipeline: OnceLock::new(),
            cached_pipeline: OnceLock::new(),
            preparation: if key_dim == 128 && value_dim == 128 {
                DeltaPrepareKernel::new(&Device::Metal(device.clone()))?
            } else {
                None
            },
            key_dim,
            value_dim,
            values,
        }))
    }
    fn compile(
        device: &MetalDevice,
        key_dim: usize,
        values: usize,
        cached: bool,
    ) -> Result<ComputePipeline> {
        let library = device
            .device()
            .new_library_with_source(
                &format!(
                    "#define CLEF_VALUES {values}\n#define CLEF_CACHE {}\n{}",
                    u8::from(cached),
                    include_str!("delta.metal")
                ),
                None,
            )
            .map_err(|e| Error::InferenceFailed(format!("compile DeltaNet kernel: {e}")))?;
        let constants = ConstantValues::new(vec![(0, Value::USize(key_dim))]);
        let function = library
            .get_function("clef_delta", Some(&constants))
            .map_err(|e| Error::InferenceFailed(format!("load DeltaNet kernel: {e}")))?;
        let pipeline = device
            .device()
            .new_compute_pipeline_state_with_function(&function)
            .map_err(|e| Error::InferenceFailed(format!("create DeltaNet pipeline: {e}")))?;
        if pipeline.max_total_threads_per_threadgroup() < 128
            || pipeline.as_ref().threadExecutionWidth() != 32
        {
            return Err(Error::UnsupportedCapability(
                "DeltaNet requires 32-lane SIMD and 128-thread groups".into(),
            ));
        }
        Ok(pipeline)
    }
    fn pipeline(&self, device: &MetalDevice) -> Result<&ComputePipeline> {
        get_or_init_fallible(&self.pipeline, || {
            Self::compile(device, self.key_dim, self.values, false)
        })
    }
    fn cached_pipeline(&self, device: &MetalDevice) -> Result<&ComputePipeline> {
        get_or_init_fallible(&self.cached_pipeline, || {
            Self::compile(device, self.key_dim, self.values, true)
        })
    }
    pub fn forward(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        g: &Tensor,
        beta: &Tensor,
        control: &Control,
    ) -> Result<Tensor> {
        control.check()?;
        let (tokens, heads, width) = q.dims3()?;
        dimensions(tokens, heads, self.key_dim, self.value_dim)?;
        if width != self.key_dim
            || k.dims() != q.dims()
            || v.dims() != [tokens, heads, self.value_dim]
            || g.dims() != [tokens, heads]
            || beta.dims() != g.dims()
        {
            return Err(Error::InvalidRequest("DeltaNet tensor shapes".into()));
        }
        // Decay depends only on token/head; compute it once, not in every state lane.
        let decay = g.exp()?;
        let packed =
            Tensor::cat(&[q, k, v, &decay.unsqueeze(2)?, &beta.unsqueeze(2)?], 2)?.contiguous()?;
        let out = packed.apply_op1_no_bwd(self)?;
        control.check()?;
        Ok(out)
    }
    pub fn can_prepare(&self) -> bool {
        self.preparation.is_some()
    }
    pub fn prepare(&self, mixed: &Tensor, g: &Tensor, beta: &Tensor) -> Result<Tensor> {
        Ok(self
            .preparation
            .as_ref()
            .ok_or_else(|| Error::UnsupportedCapability("DeltaNet preparation dimensions".into()))?
            .forward(mixed, g, beta)?)
    }
    pub fn prepared_prefill(
        &self,
        packed: &Tensor,
        control: &Control,
        initial: Option<&Tensor>,
        capture: Option<usize>,
    ) -> Result<(Tensor, Option<Tensor>)> {
        control.check()?;
        let (tokens, heads, width) = packed.dims3()?;
        dimensions(tokens, heads, self.key_dim, self.value_dim)?;
        if width != 2 * self.key_dim + self.value_dim + 2
            || capture.is_some_and(|n| n == 0 || n > tokens)
            || initial.is_some_and(|state| state.dims() != [heads, self.key_dim, self.value_dim])
        {
            return Err(Error::InvalidRequest(
                "prepared DeltaNet continuation shape".into(),
            ));
        }
        if initial.is_none() && capture.is_none() {
            return Ok((packed.apply_op1_no_bwd(self)?, None));
        }
        let operation = CachedDelta {
            kernel: self.clone(),
            capture: capture.unwrap_or_default(),
            resume: initial.is_some(),
        };
        // Without a prior state the shader never reads buffer 5 (`resume` is
        // false), so aliasing the second operand to `packed` is safe and
        // avoids a dummy allocation.
        let output = packed.apply_op2_no_bwd(initial.unwrap_or(packed), &operation)?;
        let count = tokens * heads * self.value_dim;
        let saved = if capture.is_some() {
            Some(compact_copy(
                &output
                    .narrow(0, count, heads * self.key_dim * self.value_dim)?
                    .reshape((heads, self.key_dim, self.value_dim))?,
            )?)
        } else {
            None
        };
        control.check()?;
        Ok((
            output
                .narrow(0, 0, count)?
                .reshape((tokens, heads, self.value_dim))?,
            saved,
        ))
    }
    #[allow(
        clippy::too_many_arguments,
        reason = "tensor equation with explicit immutable resume and capture boundaries"
    )]
    pub fn prefill(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        g: &Tensor,
        beta: &Tensor,
        control: &Control,
        initial: Option<&Tensor>,
        capture: Option<usize>,
    ) -> Result<(Tensor, Option<Tensor>)> {
        control.check()?;
        let (tokens, heads, width) = q.dims3()?;
        dimensions(tokens, heads, self.key_dim, self.value_dim)?;
        if width != self.key_dim
            || k.dims() != q.dims()
            || v.dims() != [tokens, heads, self.value_dim]
            || g.dims() != [tokens, heads]
            || beta.dims() != g.dims()
            || capture.is_some_and(|n| n == 0 || n > tokens)
            || initial.is_some_and(|state| state.dims() != [heads, width, self.value_dim])
        {
            return Err(Error::InvalidRequest(
                "cached DeltaNet tensor shapes".into(),
            ));
        }
        let decay = g.exp()?;
        // `cat` already returns a contiguous tensor; the hot 128/128 path
        // avoids this eager pack entirely via the fused `prepare` kernel.
        let packed = Tensor::cat(&[q, k, v, &decay.unsqueeze(2)?, &beta.unsqueeze(2)?], 2)?;
        self.prepared_prefill(&packed, control, initial, capture)
    }
}
pub(super) fn dimensions(tokens: usize, heads: usize, key: usize, value: usize) -> Result<()> {
    if !(1..=MAX_SEQUENCE_TOKENS).contains(&tokens)
        || !(1..=32).contains(&heads)
        || !(1..=128).contains(&key)
        || !key.is_power_of_two()
        || !(1..=128).contains(&value)
    {
        return Err(Error::InvalidRequest("fused DeltaNet dimensions".into()));
    }
    Ok(())
}
impl CustomOp1 for DeltaKernel {
    fn name(&self) -> &'static str {
        "clef-fused-delta"
    }
    fn cpu_fwd(&self, _: &CpuStorage, _: &Layout) -> CandleResult<(CpuStorage, Shape)> {
        Err(CandleError::Msg("fused DeltaNet requires Metal".into()))
    }
    fn metal_fwd(
        &self,
        input: &MetalStorage,
        layout: &Layout,
    ) -> CandleResult<(MetalStorage, Shape)> {
        let (tokens, heads, width) = layout.shape().dims3()?;
        dimensions(tokens, heads, self.key_dim, self.value_dim).map_err(CandleError::wrap)?;
        if width != 2 * self.key_dim + self.value_dim + 2 {
            return Err(CandleError::Msg("fused DeltaNet packed width".into()));
        }
        let offset = checked_f32_buffer(input, layout)?;
        let count = tokens * heads * self.value_dim;
        let device = input.device();
        let output = device.new_buffer(count, DType::F32, "clef_delta_output")?;
        let guard = device.command_encoder()?;
        let encoder = guard.as_ref();
        let pipeline = self.pipeline(device).map_err(CandleError::wrap)?;
        encoder.set_compute_pipeline_state(pipeline);
        encoder.set_input_buffer(0, Some(input.buffer()), offset);
        encoder.set_output_buffer(1, Some(&output), 0);
        for (index, value) in [tokens, heads, self.value_dim].into_iter().enumerate() {
            let value = u32::try_from(value).map_err(CandleError::wrap)?;
            encoder.set_bytes(index + 2, &value);
        }
        encoder.dispatch_thread_groups(
            MTLSize {
                width: heads * self.value_dim.div_ceil(4 * self.values),
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: 128,
                height: 1,
                depth: 1,
            },
        );
        Ok((
            MetalStorage::new(output, device.clone(), count, DType::F32),
            Shape::from((tokens, heads, self.value_dim)),
        ))
    }
}

#[derive(Debug)]
struct CachedDelta {
    kernel: DeltaKernel,
    capture: usize,
    resume: bool,
}
impl CustomOp2 for CachedDelta {
    fn name(&self) -> &'static str {
        "clef-cached-delta"
    }
    fn cpu_fwd(
        &self,
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
    ) -> CandleResult<(CpuStorage, Shape)> {
        Err(CandleError::Msg("cached DeltaNet requires Metal".into()))
    }
    fn metal_fwd(
        &self,
        input: &MetalStorage,
        layout: &Layout,
        initial: &MetalStorage,
        initial_layout: &Layout,
    ) -> CandleResult<(MetalStorage, Shape)> {
        let (tokens, heads, width) = layout.shape().dims3()?;
        let kernel = &self.kernel;
        dimensions(tokens, heads, kernel.key_dim, kernel.value_dim).map_err(CandleError::wrap)?;
        if width != 2 * kernel.key_dim + kernel.value_dim + 2
            || self.capture > tokens
            || (self.resume && initial_layout.dims() != [heads, kernel.key_dim, kernel.value_dim])
        {
            return Err(CandleError::Msg("cached DeltaNet dimensions".into()));
        }
        let offset = checked_f32_buffer(input, layout)?;
        let initial_offset = checked_f32_buffer(initial, initial_layout)?;
        let count = tokens * heads * kernel.value_dim
            + if self.capture > 0 {
                heads * kernel.key_dim * kernel.value_dim
            } else {
                0
            };
        let device = input.device();
        let output = device.new_buffer(count, DType::F32, "clef_cached_delta")?;
        let guard = device.command_encoder()?;
        let encoder = guard.as_ref();
        let cached = kernel.cached_pipeline(device).map_err(CandleError::wrap)?;
        encoder.set_compute_pipeline_state(cached);
        encoder.set_input_buffer(0, Some(input.buffer()), offset);
        encoder.set_output_buffer(1, Some(&output), 0);
        for (index, value) in [tokens, heads, kernel.value_dim].into_iter().enumerate() {
            encoder.set_bytes(index + 2, &u32::try_from(value).map_err(CandleError::wrap)?);
        }
        encoder.set_input_buffer(5, Some(initial.buffer()), initial_offset);
        encoder.set_bytes(6, &u32::try_from(self.capture).map_err(CandleError::wrap)?);
        encoder.set_bytes(7, &self.resume);
        encoder.dispatch_thread_groups(
            MTLSize {
                width: heads * kernel.value_dim.div_ceil(4 * kernel.values),
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: 128,
                height: 1,
                depth: 1,
            },
        );
        Ok((
            MetalStorage::new(output, device.clone(), count, DType::F32),
            Shape::from(count),
        ))
    }
}

/// F32 Flash attention with a 16-query / 8-key tile (28,928 shared bytes).
/// The upstream 32-query / 16-key tile needs 53,760 bytes at head width 256.
#[derive(Debug, Clone)]
pub(super) struct AttentionKernel {
    /// Compiled on first dispatch, not at model load: four shader variants
    /// are the single biggest chunk of startup compile time.
    pipelines: OnceLock<[ComputePipeline; 4]>,
}
impl AttentionKernel {
    pub fn new(device: &Device, width: usize) -> Result<Option<Self>> {
        let Device::Metal(_) = device else {
            return Ok(None);
        };
        if width != 256 {
            return Ok(None);
        }
        Ok(Some(Self {
            pipelines: OnceLock::new(),
        }))
    }
    fn pipelines(&self, device: &MetalDevice) -> Result<&[ComputePipeline; 4]> {
        get_or_init_fallible(&self.pipelines, || Self::compile(device))
    }
    fn compile(device: &MetalDevice) -> Result<[ComputePipeline; 4]> {
        // Reuse the dependency's exact algorithm and license; only instantiate
        // a smaller tile, avoiding a fork or a copied shader implementation.
        // A partial last query tile must not scan past the final KV tile.
        // The upstream causal bound rounds queries up to BQ, which can exceed keys.
        //
        // `NK` here is the key element count passed from Rust; the clamp is
        // only valid if the upstream shader's `params->NK` carries that same
        // meaning, so re-verify against candle-metal-kernels 0.11.0 before
        // touching this expression.
        let sdpa = replace_once(
            SDPA,
            "kb_lim = (q_max + BK - 1) / BK;",
            "kb_lim = min(params->NK, (q_max + BK - 1) / BK);",
        )?;
        let source =
            format!("{sdpa}\ninstantiate_attn(float32, float, 16, 8, 256, 2, 1, float32, float)\n");
        let library = device
            .device()
            .new_library_with_source(&source, None)
            .map_err(|e| Error::InferenceFailed(format!("compile Flash attention: {e}")))?;
        let mut pipelines = Vec::with_capacity(4);
        for variant in 0..4 {
            let constants = ConstantValues::new(vec![
                (200, Value::Bool(variant & 1 != 0)),
                (201, Value::Bool(variant & 2 != 0)),
                (300, Value::Bool(false)),
                (301, Value::Bool(true)),
            ]);
            let function = library
                .get_function(
                    "steel_attention_float32_bq16_bk8_bd256_wm2_wn1_maskfloat32",
                    Some(&constants),
                )
                .map_err(|e| Error::InferenceFailed(format!("load Flash attention: {e}")))?;
            pipelines.push(
                device
                    .device()
                    .new_compute_pipeline_state_with_function(&function)
                    .map_err(|e| {
                        Error::InferenceFailed(format!("create Flash attention pipeline: {e}"))
                    })?,
            );
        }
        pipelines
            .try_into()
            .map_err(|_| Error::InferenceFailed("attention pipeline count".into()))
    }
    pub fn forward(&self, q: &Tensor, k: &Tensor, v: &Tensor, control: &Control) -> Result<Tensor> {
        control.check()?;
        let out = q.to_dtype(DType::F32)?.contiguous()?.apply_op3_no_bwd(
            &k.to_dtype(DType::F32)?.contiguous()?,
            &v.to_dtype(DType::F32)?.contiguous()?,
            self,
        )?;
        control.check()?;
        Ok(out)
    }
}
/// Explicit padding matches the upstream Metal ABI without implicit Rust padding.
#[derive(Debug, Clone, Copy)]
#[repr(C)]
struct AttentionParams {
    batch: i32,
    heads: i32,
    width: i32,
    queries: i32,
    keys: i32,
    gqa: i32,
    scale: f32,
    softcap: f32,
    query_tiles: i32,
    key_tiles: i32,
    aligned_query_tiles: i32,
    aligned_key_tiles: i32,
    query_tail: i32,
    key_tail: i32,
    query_offset: i32,
    padding: i32,
    query_strides: [i64; 3],
    key_strides: [i64; 3],
    value_strides: [i64; 3],
    output_strides: [i64; 3],
}
pub(super) fn checked_f32_buffer(input: &MetalStorage, layout: &Layout) -> CandleResult<usize> {
    if input.dtype() != DType::F32 || !layout.is_contiguous() {
        return Err(CandleError::Msg(
            "custom Metal operation requires contiguous F32".into(),
        ));
    }
    let offset = layout
        .start_offset()
        .checked_mul(4)
        .ok_or_else(|| CandleError::Msg("custom Metal offset overflow".into()))?;
    let end = layout
        .shape()
        .elem_count()
        .checked_mul(4)
        .and_then(|bytes| offset.checked_add(bytes))
        .ok_or_else(|| CandleError::Msg("custom Metal buffer overflow".into()))?;
    if end > input.buffer().length() {
        return Err(CandleError::Msg("custom Metal input buffer bounds".into()));
    }
    Ok(offset)
}
impl CustomOp3 for AttentionKernel {
    fn name(&self) -> &'static str {
        "clef-f32-flash-attention"
    }
    fn cpu_fwd(
        &self,
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
    ) -> CandleResult<(CpuStorage, Shape)> {
        Err(CandleError::Msg("Flash attention requires Metal".into()))
    }
    fn metal_fwd(
        &self,
        q: &MetalStorage,
        ql: &Layout,
        k: &MetalStorage,
        kl: &Layout,
        v: &MetalStorage,
        vl: &Layout,
    ) -> CandleResult<(MetalStorage, Shape)> {
        let (heads, tokens, width) = ql.shape().dims3()?;
        let (kv_heads, keys, kw) = kl.shape().dims3()?;
        if !(1..=32).contains(&heads)
            || !(1..=heads).contains(&kv_heads)
            || heads % kv_heads != 0
            || !(1..=MAX_SEQUENCE_TOKENS).contains(&tokens)
            || !(tokens..=MAX_SEQUENCE_TOKENS).contains(&keys)
            || width != 256
            || kw != width
            || vl.dims() != kl.dims()
        {
            return Err(CandleError::Msg("Flash attention tensor dimensions".into()));
        }
        let offsets = [
            checked_f32_buffer(q, ql)?,
            checked_f32_buffer(k, kl)?,
            checked_f32_buffer(v, vl)?,
        ];
        let count = heads * tokens * width;
        let device = q.device();
        let output = device.new_buffer(count, DType::F32, "clef_flash_attention")?;
        let variant = usize::from(tokens % 16 == 0) | (usize::from(keys % 8 == 0) << 1);
        let pipeline = self
            .pipelines(q.device())
            .map_err(CandleError::wrap)?
            .get(variant)
            .ok_or_else(|| CandleError::Msg("attention tile variant".into()))?;
        let to_i32 = |value| i32::try_from(value).map_err(CandleError::wrap);
        let strides = |n, length| -> CandleResult<[i64; 3]> {
            Ok([
                i64::try_from(n * length * width).map_err(CandleError::wrap)?,
                i64::try_from(length * width).map_err(CandleError::wrap)?,
                256,
            ])
        };
        let params = AttentionParams {
            batch: 1,
            heads: to_i32(heads)?,
            width: 256,
            queries: to_i32(tokens)?,
            keys: to_i32(keys)?,
            gqa: to_i32(heads / kv_heads)?,
            scale: 0.0625,
            softcap: 1.,
            query_tiles: to_i32(tokens.div_ceil(16))?,
            key_tiles: to_i32(keys.div_ceil(8))?,
            aligned_query_tiles: to_i32(tokens / 16)?,
            aligned_key_tiles: to_i32(keys / 8)?,
            query_tail: to_i32(tokens % 16)?,
            key_tail: to_i32(keys % 8)?,
            query_offset: to_i32(keys - tokens)?,
            padding: 0,
            query_strides: strides(heads, tokens)?,
            key_strides: strides(kv_heads, keys)?,
            value_strides: strides(kv_heads, keys)?,
            output_strides: strides(heads, tokens)?,
        };
        let guard = device.command_encoder()?;
        let encoder = guard.as_ref();
        encoder.set_compute_pipeline_state(pipeline);
        for (index, (input, offset)) in [q, k, v].into_iter().zip(offsets).enumerate() {
            encoder.set_input_buffer(index, Some(input.buffer()), offset);
        }
        encoder.set_output_buffer(3, Some(&output), 0);
        encoder.set_bytes(4, &params);
        encoder.dispatch_thread_groups(
            MTLSize {
                width: tokens.div_ceil(16),
                height: heads,
                depth: 1,
            },
            MTLSize {
                width: 32,
                height: 2,
                depth: 1,
            },
        );
        Ok((
            MetalStorage::new(output, device.clone(), count, DType::F32),
            Shape::from((heads, tokens, width)),
        ))
    }
}

#[cfg(test)]
mod tests {
    use std::{
        sync::{Arc, atomic::AtomicBool},
        time::{Duration, Instant},
    };

    use super::*;
    use crate::models::qwen::delta_recurrence;

    #[test]
    #[ignore = "diagnostic microprofile requires a real Apple Metal device"]
    fn test_should_profile_full_width_delta_kernel() -> Result<()> {
        let device = Device::new_metal(0)?;
        let reference =
            DeltaKernel::new_variant(&device, 128, 128, 1)?.ok_or(Error::ArtifactMissing)?;
        for values in [1, 2, 4, 8, 16] {
            let kernel = DeltaKernel::new_variant(&device, 128, 128, values)?
                .ok_or_else(|| Error::InferenceFailed("missing DeltaNet kernel".into()))?;
            for tokens in [256, 1024, 4096] {
                let control = Control {
                    cancel: Arc::new(AtomicBool::new(false)),
                    deadline: Instant::now() + Duration::from_secs(300),
                };
                let q = Tensor::full(0.01f32, (tokens, 32, 128), &device)?;
                let k = Tensor::full(0.088f32, (tokens, 32, 128), &device)?;
                let v = Tensor::full(0.25f32, (tokens, 32, 128), &device)?;
                let g = Tensor::full(-0.03f32, (tokens, 32), &device)?;
                let beta = Tensor::full(0.65f32, (tokens, 32), &device)?;
                let expected = reference.forward(&q, &k, &v, &g, &beta, &control)?;
                let actual = kernel.forward(&q, &k, &v, &g, &beta, &control)?;
                let drift = (&actual - &expected)?
                    .abs()?
                    .flatten_all()?
                    .max(0)?
                    .to_scalar::<f32>()?;
                assert!(drift <= 1e-7, "Delta vector {values} drift={drift}");
                device.synchronize()?;
                let started = Instant::now();
                for _ in 0..3 {
                    kernel.forward(&q, &k, &v, &g, &beta, &control)?;
                    device.synchronize()?;
                }
                eprintln!(
                    "DeltaNet {tokens} tokens values={values}: {:.3} ms per layer",
                    started.elapsed().as_secs_f64() * 1000. / 3.
                );
            }
        }
        Ok(())
    }
    #[test]
    #[ignore = "requires a real Apple Metal device"]
    #[allow(
        clippy::cast_precision_loss,
        reason = "bounded deterministic test indices"
    )]
    fn test_should_match_f32_attention_with_small_tiles_and_gqa() -> Result<()> {
        use crate::models::ops::attention;
        let device = Device::new_metal(0)?;
        let kernel = AttentionKernel::new(&device, 256)?
            .ok_or_else(|| Error::InferenceFailed("missing attention kernel".into()))?;
        assert_eq!(size_of::<AttentionParams>(), 160);
        for tokens in [1, 8, 16, 17, 257] {
            let control = Control {
                cancel: Arc::new(AtomicBool::new(false)),
                deadline: Instant::now() + Duration::from_secs(60),
            };
            let q = Tensor::from_vec(
                (0..2 * tokens * 256)
                    .map(|i| (i as f32 * 0.017).sin())
                    .collect(),
                (2, tokens, 256),
                &Device::Cpu,
            )?;
            let k = Tensor::from_vec(
                (0..tokens * 256)
                    .map(|i| (i as f32 * 0.031).cos())
                    .collect(),
                (1, tokens, 256),
                &Device::Cpu,
            )?;
            let expected = attention(&q, &k, &k, true, &control)?;
            let actual = kernel
                .forward(
                    &q.to_device(&device)?,
                    &k.to_device(&device)?,
                    &k.to_device(&device)?,
                    &control,
                )?
                .to_device(&Device::Cpu)?;
            let error = (&actual - expected)?
                .abs()?
                .flatten_all()?
                .max(0)?
                .to_scalar::<f32>()?;
            assert!(error < 1e-5, "Flash attention {tokens}: error {error}");
        }
        Ok(())
    }
    #[test]
    #[ignore = "requires a real Apple Metal device"]
    #[allow(
        clippy::cast_precision_loss,
        reason = "small deterministic test indices"
    )]
    fn test_should_match_cpu_recurrence_with_segment_and_column_tails() -> Result<()> {
        let device = Device::new_metal(0)?;
        for (tokens, heads, key, value) in [
            (1, 1, 1, 1),
            (33, 2, 4, 5),
            (33, 2, 32, 13),
            (257, 2, 64, 13),
            (257, 2, 128, 13),
            (17, 2, 128, 128),
        ] {
            let control = Control {
                cancel: Arc::new(AtomicBool::new(false)),
                deadline: Instant::now() + Duration::from_secs(60),
            };
            let q = Tensor::from_vec(
                (0..tokens * heads * key)
                    .map(|i| (i as f32 * 0.017).sin())
                    .collect(),
                (tokens, heads, key),
                &Device::Cpu,
            )?;
            let k = ((&q * 0.7)? + 0.1)?;
            let v = Tensor::from_vec(
                (0..tokens * heads * value)
                    .map(|i| (i as f32 * 0.031).cos())
                    .collect(),
                (tokens, heads, value),
                &Device::Cpu,
            )?;
            let g = Tensor::full(-0.03f32, (tokens, heads), &Device::Cpu)?;
            let beta = Tensor::full(0.65f32, (tokens, heads), &Device::Cpu)?;
            let expected = delta_recurrence(&q, &k, &v, &g, &beta, &control, None)?;
            for values in [1, 2, 4, 8, 16] {
                let kernel = DeltaKernel::new_variant(&device, key, value, values)?;
                let actual = delta_recurrence(
                    &q.to_device(&device)?,
                    &k.to_device(&device)?,
                    &v.to_device(&device)?,
                    &g.to_device(&device)?,
                    &beta.to_device(&device)?,
                    &control,
                    kernel.as_ref(),
                )?
                .to_device(&Device::Cpu)?;
                if key == 128 && tokens == 17 {
                    let reference = delta_recurrence(
                        &q.to_device(&device)?,
                        &k.to_device(&device)?,
                        &v.to_device(&device)?,
                        &g.to_device(&device)?,
                        &beta.to_device(&device)?,
                        &control,
                        None,
                    )?
                    .to_device(&Device::Cpu)?;
                    let drift = (&actual - reference)?
                        .abs()?
                        .flatten_all()?
                        .max(0)?
                        .to_scalar::<f32>()?;
                    assert!(drift < 1e-7, "Metal reduction-order drift {drift}");
                }
                let error = (&actual - &expected)?
                    .abs()?
                    .flatten_all()?
                    .max(0)?
                    .to_scalar::<f32>()?;
                assert!(
                    error < 1e-5,
                    "dimensions {tokens}/{heads}/{key}/{value} values={values}: error {error}"
                );
            }
        }
        Ok(())
    }

    #[test]
    #[ignore = "requires a real Apple Metal device"]
    fn test_should_resume_f32_delta_bit_exactly_at_capture_boundary() -> Result<()> {
        let device = Device::new_metal(0)?;
        let control = Control {
            cancel: Arc::new(AtomicBool::new(false)),
            deadline: Instant::now() + Duration::from_secs(60),
        };
        for (key, value, values) in [(4, 5, 1), (128, 128, 4), (128, 13, 2), (128, 13, 8)] {
            let kernel = DeltaKernel::new_variant(&device, key, value, values)?
                .ok_or(Error::ArtifactMissing)?;
            let q = Tensor::full(0.01f32, (257, 2, key), &device)?;
            let k = Tensor::full(0.088f32, (257, 2, key), &device)?;
            let v = Tensor::full(0.25f32, (257, 2, value), &device)?;
            let g = Tensor::full(-0.03f32, (257, 2), &device)?;
            let beta = Tensor::full(0.65f32, (257, 2), &device)?;
            let expected = kernel.forward(&q, &k, &v, &g, &beta, &control)?;
            for boundary in [1, 16, 256] {
                let (captured, state) =
                    kernel.prefill(&q, &k, &v, &g, &beta, &control, None, Some(boundary))?;
                let error = (&captured - &expected)?
                    .abs()?
                    .flatten_all()?
                    .max(0)?
                    .to_scalar::<f32>()?;
                assert_eq!(error, 0.);
                let state = state.ok_or(Error::ArtifactMissing)?;
                let tail = |tensor: &Tensor| -> Result<Tensor> {
                    Ok(tensor.narrow(0, boundary, 257 - boundary)?.contiguous()?)
                };
                let actual = kernel
                    .prefill(
                        &tail(&q)?,
                        &tail(&k)?,
                        &tail(&v)?,
                        &tail(&g)?,
                        &tail(&beta)?,
                        &control,
                        Some(&state),
                        None,
                    )?
                    .0;
                let error = (actual - expected.narrow(0, boundary, 257 - boundary)?)?
                    .abs()?
                    .flatten_all()?
                    .max(0)?
                    .to_scalar::<f32>()?;
                assert_eq!(error, 0.);
            }
        }
        Ok(())
    }
    #[test]
    #[ignore = "requires a real Apple Metal device"]
    #[allow(
        clippy::cast_precision_loss,
        reason = "bounded deterministic test indices"
    )]
    fn test_should_apply_lower_right_causal_mask_with_cached_kv_and_tails() -> Result<()> {
        let device = Device::new_metal(0)?;
        let kernel = AttentionKernel::new(&device, 256)?.ok_or(Error::ArtifactMissing)?;
        let control = Control {
            cancel: Arc::new(AtomicBool::new(false)),
            deadline: Instant::now() + Duration::from_secs(60),
        };
        for (tokens, prefix) in [(1, 256), (8, 256), (16, 256), (17, 256), (257, 256)] {
            let q = Tensor::from_vec(
                (0..2 * tokens * 256)
                    .map(|i| (i as f32 * 0.017).sin())
                    .collect(),
                (2, tokens, 256),
                &Device::Cpu,
            )?;
            let k = Tensor::from_vec(
                (0..(tokens + prefix) * 256)
                    .map(|i| (i as f32 * 0.031).cos())
                    .collect(),
                (1, tokens + prefix, 256),
                &Device::Cpu,
            )?;
            let expected = crate::models::ops::attention(&q, &k, &k, true, &control)?;
            let actual = kernel.forward(
                &q.to_device(&device)?,
                &k.to_device(&device)?,
                &k.to_device(&device)?,
                &control,
            )?;
            let error = (actual.to_device(&Device::Cpu)? - expected)?
                .abs()?
                .flatten_all()?
                .max(0)?
                .to_scalar::<f32>()?;
            assert!(
                error < 1e-5,
                "cached Flash attention {tokens}/{prefix}: {error}"
            );
        }
        Ok(())
    }
    #[test]
    #[ignore = "requires Metal; validates physical buffer lengths and strided snapshots"]
    fn test_should_retain_exact_sized_prefix_buffers_without_large_pool_reuse() -> Result<()> {
        use candle_core::Storage;
        let device = Device::new_metal(0)?;
        for dtype in [DType::F32, DType::F16] {
            let source = Tensor::arange(0f32, 4096f32, &device)?
                .reshape((4, 8, 128))?
                .to_dtype(dtype)?;
            let view = source.narrow(1, 1, 3)?;
            let output = compact_copy(&view)?;
            let bytes = output.elem_count() * dtype.size_in_bytes();
            let (storage, layout) = output.storage_and_layout();
            let Storage::Metal(storage) = &*storage else {
                return Err(Error::ArtifactMissing);
            };
            assert_eq!(storage.buffer().length(), bytes);
            assert_eq!(layout.start_offset(), 0);
            assert!(layout.is_contiguous());
            let drift = (output.to_dtype(DType::F32)? - view.to_dtype(DType::F32)?)?
                .abs()?
                .flatten_all()?
                .max(0)?
                .to_scalar::<f32>()?;
            assert_eq!(drift.to_bits(), 0);
        }
        Ok(())
    }
    #[test]
    fn test_should_reject_unsupported_dispatch_shapes_before_allocation() {
        for shape in [
            (0, 32, 128, 128),
            (4097, 32, 128, 128),
            (1, 33, 128, 128),
            (1, 1, 127, 128),
            (1, 1, 128, 129),
        ] {
            assert!(dimensions(shape.0, shape.1, shape.2, shape.3).is_err());
        }
        assert!(dimensions(4096, 32, 128, 128).is_ok());
    }
}
