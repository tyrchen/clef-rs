//! Test-only chunkwise gated-DeltaNet candidate with stable local decay products.
//! Correctness is checked here; measured regressions keep it out of serving builds.

use candle_core::{
    CpuStorage, CustomOp1, DType, Error as CandleError, Layout, MetalDevice, MetalStorage,
    Result as CandleResult, Shape, Tensor, backend::BackendStorage,
};
use candle_metal_kernels::metal::ComputePipeline;
use objc2_metal::{
    MTLCompileOptions, MTLComputePipelineState, MTLLanguageVersion, MTLMathMode, MTLSize,
};

use super::metal::{checked_f32_buffer, dimensions};
use crate::{Error, Result};

const CHUNK: usize = 32;

#[derive(Debug, Clone)]
pub(super) struct ChunkKernel {
    pipelines: [ComputePipeline; 5],
    key_dim: usize,
    value_dim: usize,
}
impl ChunkKernel {
    pub fn new(device: &MetalDevice, key_dim: usize, value_dim: usize) -> Result<Self> {
        dimensions(1, 1, key_dim, value_dim)?;
        let options = MTLCompileOptions::new();
        options.setLanguageVersion(MTLLanguageVersion::Version4_0);
        options.setMathMode(MTLMathMode::Safe);
        let source = format!(
            "#define CLEF_CHUNK_KEY {}\n{}",
            key_dim.max(16),
            include_str!("chunk.metal")
        );
        let library = device
            .device()
            .new_library_with_source(&source, Some(&options))
            .map_err(|e| Error::InferenceFailed(format!("compile chunkwise DeltaNet: {e}")))?;
        let mut pipelines = Vec::with_capacity(5);
        for name in [
            "clef_chunk_gram",
            "clef_chunk_inverse",
            "clef_chunk_transform",
            "clef_chunk_carry",
            "clef_chunk_output",
        ] {
            let function = library
                .get_function(name, None)
                .map_err(|e| Error::InferenceFailed(format!("load {name}: {e}")))?;
            let pipeline = device
                .device()
                .new_compute_pipeline_state_with_function(&function)
                .map_err(|e| Error::InferenceFailed(format!("create {name}: {e}")))?;
            if pipeline.max_total_threads_per_threadgroup() < 256
                || pipeline.as_ref().threadExecutionWidth() != 32
            {
                return Err(Error::UnsupportedCapability(
                    "chunkwise DeltaNet thread geometry".into(),
                ));
            }
            pipelines.push(pipeline);
        }
        Ok(Self {
            pipelines: pipelines
                .try_into()
                .map_err(|_| Error::InferenceFailed("chunk pipeline count".into()))?,
            key_dim,
            value_dim,
        })
    }
    pub fn forward(&self, packed: &Tensor) -> CandleResult<Tensor> {
        packed.apply_op1_no_bwd(self)
    }
}
impl CustomOp1 for ChunkKernel {
    fn name(&self) -> &'static str {
        "clef-chunkwise-delta"
    }
    fn cpu_fwd(&self, _: &CpuStorage, _: &Layout) -> CandleResult<(CpuStorage, Shape)> {
        Err(CandleError::Msg("chunkwise DeltaNet requires Metal".into()))
    }
    fn metal_fwd(
        &self,
        input: &MetalStorage,
        layout: &Layout,
    ) -> CandleResult<(MetalStorage, Shape)> {
        let (tokens, heads, width) = layout.shape().dims3()?;
        dimensions(tokens, heads, self.key_dim, self.value_dim).map_err(CandleError::wrap)?;
        if width != 2 * self.key_dim + self.value_dim + 2 {
            return Err(CandleError::Msg("chunkwise DeltaNet packed width".into()));
        }
        let offset = checked_f32_buffer(input, layout)?;
        let blocks = tokens.div_ceil(CHUNK) * heads;
        let count = tokens * heads * self.value_dim;
        let feature_count = blocks * CHUNK * (2 * self.key_dim + self.value_dim);
        let device = input.device();
        let matrices =
            device.new_buffer(blocks * 2 * CHUNK * CHUNK, DType::F32, "chunk matrices")?;
        let factors = device.new_buffer(blocks * 2 * CHUNK, DType::F32, "chunk decays")?;
        let inverse = device.new_buffer(blocks * CHUNK * CHUNK, DType::F32, "chunk inverse")?;
        let transformed = device.new_buffer(feature_count, DType::F32, "chunk transforms")?;
        let innovation = device.new_buffer(count, DType::F32, "chunk innovations")?;
        let prefix = device.new_buffer(count, DType::F32, "chunk prefix")?;
        let output = device.new_buffer(count, DType::F32, "chunk output")?;
        let guard = device.command_encoder()?;
        let encoder = guard.as_ref();
        let [gram, solve, transform, carry, finish] = &self.pipelines;
        for (index, value) in [tokens, heads, self.key_dim, self.value_dim]
            .into_iter()
            .enumerate()
        {
            encoder.set_bytes(
                10 + index,
                &u32::try_from(value).map_err(CandleError::wrap)?,
            );
        }
        encoder.set_compute_pipeline_state(gram);
        encoder.set_input_buffer(0, Some(input.buffer()), offset);
        encoder.set_output_buffer(1, Some(&matrices), 0);
        encoder.set_output_buffer(2, Some(&factors), 0);
        encoder.dispatch_thread_groups(size(blocks * 8), size(128));
        encoder.set_compute_pipeline_state(solve);
        encoder.set_input_buffer(0, Some(&matrices), 0);
        encoder.set_output_buffer(1, Some(&inverse), 0);
        encoder.dispatch_thread_groups(size(blocks * 8), size(128));
        encoder.set_compute_pipeline_state(transform);
        encoder.set_input_buffer(0, Some(input.buffer()), offset);
        encoder.set_input_buffer(1, Some(&inverse), 0);
        encoder.set_input_buffer(2, Some(&factors), 0);
        encoder.set_output_buffer(3, Some(&transformed), 0);
        encoder.dispatch_thread_groups(size(feature_count.div_ceil(256)), size(256));
        encoder.set_compute_pipeline_state(carry);
        encoder.set_input_buffer(0, Some(input.buffer()), offset);
        encoder.set_input_buffer(1, Some(&transformed), 0);
        encoder.set_input_buffer(2, Some(&factors), 0);
        encoder.set_output_buffer(3, Some(&innovation), 0);
        encoder.set_output_buffer(4, Some(&prefix), 0);
        encoder.dispatch_thread_groups(size(heads * self.value_dim.div_ceil(32)), size(128));
        encoder.set_compute_pipeline_state(finish);
        encoder.set_input_buffer(0, Some(&matrices), 0);
        encoder.set_input_buffer(1, Some(&innovation), 0);
        encoder.set_input_buffer(2, Some(&prefix), 0);
        encoder.set_output_buffer(3, Some(&output), 0);
        encoder.dispatch_thread_groups(size(count.div_ceil(256)), size(256));
        Ok((
            MetalStorage::new(output, device.clone(), count, DType::F32),
            Shape::from((tokens, heads, self.value_dim)),
        ))
    }
}
fn size(width: usize) -> MTLSize {
    MTLSize {
        width,
        height: 1,
        depth: 1,
    }
}

#[cfg(test)]
mod tests {
    use std::{
        sync::{Arc, atomic::AtomicBool},
        time::{Duration, Instant},
    };

    use candle_core::{D, Device};

    use super::*;
    use crate::models::{Control, metal::DeltaKernel, qwen::delta_recurrence};

    #[test]
    #[ignore = "requires Metal; chunk boundaries, underflow, and full-width numerical checks"]
    #[allow(
        clippy::cast_precision_loss,
        reason = "bounded deterministic test indices"
    )]
    fn test_should_match_chunkwise_delta_and_profile_long_prefill() -> Result<()> {
        let device = Device::new_metal(0)?;
        let Device::Metal(metal) = &device else {
            return Err(Error::ArtifactMissing);
        };
        for (tokens, heads, key, value) in [
            (1, 2, 4, 13),
            (31, 2, 32, 17),
            (32, 1, 128, 128),
            (33, 2, 64, 3),
            (139, 2, 128, 13),
            (512, 2, 128, 17),
            (1024, 1, 128, 32),
        ] {
            let q = Tensor::from_vec(
                (0..tokens * heads * key)
                    .map(|i| (i as f32 * 0.071).sin() * 0.02)
                    .collect(),
                (tokens, heads, key),
                &Device::Cpu,
            )?;
            let k = Tensor::from_vec(
                (0..tokens * heads * key)
                    .map(|i| (i as f32 * 0.037).cos())
                    .collect(),
                (tokens, heads, key),
                &Device::Cpu,
            )?;
            let k = k.broadcast_div(&(k.sqr()?.sum_keepdim(D::Minus1)? + 1e-6)?.sqrt()?)?;
            let v = Tensor::from_vec(
                (0..tokens * heads * value)
                    .map(|i| (i as f32 * 0.033).sin())
                    .collect(),
                (tokens, heads, value),
                &Device::Cpu,
            )?;
            for decay in [-0.03f32, -1000.] {
                let g = Tensor::full(decay, (tokens, heads), &Device::Cpu)?;
                let beta = Tensor::full(0.65f32, (tokens, heads), &Device::Cpu)?;
                let control = Control {
                    cancel: Arc::new(AtomicBool::new(false)),
                    deadline: Instant::now() + Duration::from_secs(300),
                };
                // The public recurrence normalizes its inputs; mirror those boundaries.
                let expected = delta_recurrence(&q, &k, &v, &g, &beta, &control, None)?;
                let qn = q.broadcast_div(&(q.sqr()?.sum_keepdim(D::Minus1)? + 1e-6)?.sqrt()?)?;
                let qn = (qn / (key as f64).sqrt())?;
                let kn = k.broadcast_div(&(k.sqr()?.sum_keepdim(D::Minus1)? + 1e-6)?.sqrt()?)?;
                let packed = Tensor::cat(
                    &[&qn, &kn, &v, &g.exp()?.unsqueeze(2)?, &beta.unsqueeze(2)?],
                    2,
                )?
                .to_device(&device)?;
                let kernel = ChunkKernel::new(metal, key, value)?;
                let actual = kernel.forward(&packed)?.to_device(&Device::Cpu)?;
                let error = (&actual - expected)?
                    .abs()?
                    .flatten_all()?
                    .max(0)?
                    .to_scalar::<f32>()?;
                assert!(
                    error < 1e-5,
                    "chunkwise {tokens}/{heads}/{key}/{value} decay={decay}: {error}"
                );
            }
        }
        let recurrent = DeltaKernel::new(&device, 128, 128)?.ok_or(Error::ArtifactMissing)?;
        let chunk = ChunkKernel::new(metal, 128, 128)?;
        for tokens in [256, 1024, 4096] {
            let q = Tensor::full(0.01f32, (tokens, 32, 128), &device)?;
            let k = Tensor::full(0.088f32, (tokens, 32, 128), &device)?;
            let v = Tensor::full(0.25f32, (tokens, 32, 128), &device)?;
            let decay = Tensor::full((-0.03f32).exp(), (tokens, 32, 1), &device)?;
            let beta = Tensor::full(0.65f32, (tokens, 32, 1), &device)?;
            let packed = Tensor::cat(&[q, k, v, decay, beta], 2)?;
            let expected = packed.apply_op1_no_bwd(&recurrent)?;
            let actual = chunk.forward(&packed)?;
            let error = (&actual - &expected)?
                .abs()?
                .flatten_all()?
                .max(0)?
                .to_scalar::<f32>()?;
            assert!(error < 1e-5, "full-width chunk drift {tokens}: {error}");
            for native in [false, true] {
                device.synchronize()?;
                let started = Instant::now();
                for _ in 0..3 {
                    if native {
                        chunk.forward(&packed)?;
                    } else {
                        packed.apply_op1_no_bwd(&recurrent)?;
                    }
                    device.synchronize()?;
                }
                eprintln!(
                    "CHUNK {tokens} chunkwise={native}: {:.3} ms",
                    started.elapsed().as_secs_f64() * 1000. / 3.
                );
            }
        }
        Ok(())
    }
}
