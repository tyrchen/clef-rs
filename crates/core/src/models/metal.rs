//! Checked safe dispatch for the fused F32 gated-DeltaNet recurrence.
//!
//! A group owns four value columns of one head. Each SIMD lane keeps up to four key states
//! in registers. SIMD reductions combine dot products without shared memory or
//! cross-group barriers. Every output is written once. Validated dimensions, contiguous
//! offsets and buffer lengths bound every shader address. No raw FFI or unsafe
//! Rust is used; Candle owns allocation, command lifetime and hazard tracking.

use candle_core::{
    CpuStorage, CustomOp1, CustomOp3, DType, Device, Error as CandleError, Layout, MetalStorage,
    Result as CandleResult, Shape, Tensor, backend::BackendStorage,
};
use candle_metal_kernels::{
    metal::{ComputePipeline, ConstantValues, Value},
    source::SDPA,
};
use objc2_metal::{MTLComputePipelineState, MTLSize};

use super::Control;
use crate::{Error, Result};

#[derive(Debug, Clone)]
pub(super) struct DeltaKernel {
    pipeline: ComputePipeline,
    key_dim: usize,
    value_dim: usize,
}
impl DeltaKernel {
    pub fn new(device: &Device, key_dim: usize, value_dim: usize) -> Result<Option<Self>> {
        let Device::Metal(device) = device else {
            return Ok(None);
        };
        dimensions(1, 1, key_dim, value_dim)?;
        let library = device
            .device()
            .new_library_with_source(include_str!("delta.metal"), None)
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
        Ok(Some(Self {
            pipeline,
            key_dim,
            value_dim,
        }))
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
}
fn dimensions(tokens: usize, heads: usize, key: usize, value: usize) -> Result<()> {
    if !(1..=4096).contains(&tokens)
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
        encoder.set_compute_pipeline_state(&self.pipeline);
        encoder.set_input_buffer(0, Some(input.buffer()), offset);
        encoder.set_output_buffer(1, Some(&output), 0);
        for (index, value) in [tokens, heads, self.value_dim].into_iter().enumerate() {
            let value = u32::try_from(value).map_err(CandleError::wrap)?;
            encoder.set_bytes(index + 2, &value);
        }
        encoder.dispatch_thread_groups(
            MTLSize {
                width: heads * self.value_dim.div_ceil(4),
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

/// F32 Flash attention with a 16-query / 8-key tile (28,928 shared bytes).
/// The upstream 32-query / 16-key tile needs 53,760 bytes at head width 256.
#[derive(Debug, Clone)]
pub(super) struct AttentionKernel {
    pipelines: [ComputePipeline; 4],
}
impl AttentionKernel {
    pub fn new(device: &Device, width: usize) -> Result<Option<Self>> {
        let Device::Metal(device) = device else {
            return Ok(None);
        };
        if width != 256 {
            return Ok(None);
        }
        // Reuse the dependency's exact algorithm and license; only instantiate
        // a smaller tile, avoiding a fork or a copied shader implementation.
        let source =
            format!("{SDPA}\ninstantiate_attn(float32, float, 16, 8, 256, 2, 1, float32, float)\n");
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
        let pipelines = pipelines
            .try_into()
            .map_err(|_| Error::InferenceFailed("attention pipeline count".into()))?;
        Ok(Some(Self { pipelines }))
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
            || !(1..=4096).contains(&tokens)
            || tokens != keys
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
        let variant = usize::from(tokens % 16 == 0) | (usize::from(tokens % 8 == 0) << 1);
        let pipeline = self
            .pipelines
            .get(variant)
            .ok_or_else(|| CandleError::Msg("attention tile variant".into()))?;
        let to_i32 = |value| i32::try_from(value).map_err(CandleError::wrap);
        let strides = |n| -> CandleResult<[i64; 3]> {
            Ok([
                i64::try_from(n * tokens * width).map_err(CandleError::wrap)?,
                i64::try_from(tokens * width).map_err(CandleError::wrap)?,
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
            key_tiles: to_i32(tokens.div_ceil(8))?,
            aligned_query_tiles: to_i32(tokens / 16)?,
            aligned_key_tiles: to_i32(tokens / 8)?,
            query_tail: to_i32(tokens % 16)?,
            key_tail: to_i32(tokens % 8)?,
            query_offset: 0,
            padding: 0,
            query_strides: strides(heads)?,
            key_strides: strides(kv_heads)?,
            value_strides: strides(kv_heads)?,
            output_strides: strides(heads)?,
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
        let kernel = DeltaKernel::new(&device, 128, 128)?
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
            kernel.forward(&q, &k, &v, &g, &beta, &control)?;
            device.synchronize()?;
            let started = Instant::now();
            for _ in 0..3 {
                kernel.forward(&q, &k, &v, &g, &beta, &control)?;
                device.synchronize()?;
            }
            eprintln!(
                "DeltaNet {tokens} tokens: {:.3} ms per layer",
                started.elapsed().as_secs_f64() * 1000. / 3.
            );
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
            let kernel = DeltaKernel::new(&device, key, value)?;
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
            let error = (&actual - expected)?
                .abs()?
                .flatten_all()?
                .max(0)?
                .to_scalar::<f32>()?;
            assert!(
                error < 1e-5,
                "dimensions {tokens}/{heads}/{key}/{value}: error {error}"
            );
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
