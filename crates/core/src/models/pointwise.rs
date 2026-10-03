//! Checked F32 pointwise kernels preserving the reference arithmetic tree.

use candle_core::{
    CpuStorage, CustomOp2, CustomOp3, DType, Device, Error as CandleError, Layout, MetalDevice,
    MetalStorage, Result as CandleResult, Shape, Tensor, backend::BackendStorage,
};
use candle_metal_kernels::metal::ComputePipeline;
use objc2_metal::{MTLComputePipelineState, MTLSize};

use super::metal::checked_f32_buffer;
use crate::{Error, Result};

#[derive(Debug, Clone)]
pub(super) struct RmsKernel {
    pipeline: ComputePipeline,
    eps: f32,
}
impl RmsKernel {
    #[allow(
        clippy::cast_possible_truncation,
        reason = "validated epsilon uses the same F32 conversion as the reference"
    )]
    pub fn new(device: &Device, eps: f64) -> Result<Option<Self>> {
        let Device::Metal(device) = device else {
            return Ok(None);
        };
        let pipeline = compile_pipeline(device, "clef_rms")?;
        if pipeline.max_total_threads_per_threadgroup() < 1024
            || pipeline.as_ref().threadExecutionWidth() != 32
        {
            return Ok(None);
        }
        Ok(Some(Self {
            pipeline,
            eps: eps as f32,
        }))
    }
    pub fn forward(&self, x: &Tensor, weight: &Tensor) -> CandleResult<Tensor> {
        x.apply_op2_no_bwd(weight, self)
    }
}
impl CustomOp2 for RmsKernel {
    fn name(&self) -> &'static str {
        "clef-rms"
    }
    fn cpu_fwd(
        &self,
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
    ) -> CandleResult<(CpuStorage, Shape)> {
        Err(CandleError::Msg("fused RMS requires Metal".into()))
    }
    fn metal_fwd(
        &self,
        x: &MetalStorage,
        xl: &Layout,
        w: &MetalStorage,
        wl: &Layout,
    ) -> CandleResult<(MetalStorage, Shape)> {
        let width = wl.shape().dims1()?;
        let count = xl.shape().elem_count();
        if !(1..=4096).contains(&width)
            || !width.is_power_of_two()
            || xl.dims().last().copied() != Some(width)
            || !(1..=16_777_216).contains(&count)
        {
            return Err(CandleError::Msg("fused RMS dimensions".into()));
        }
        let xo = checked_f32_buffer(x, xl)?;
        let wo = checked_f32_buffer(w, wl)?;
        let device = x.device();
        let output = device.new_buffer(count, DType::F32, "clef_rms")?;
        let guard = device.command_encoder()?;
        let encoder = guard.as_ref();
        encoder.set_compute_pipeline_state(&self.pipeline);
        encoder.set_input_buffer(0, Some(x.buffer()), xo);
        encoder.set_input_buffer(1, Some(w.buffer()), wo);
        encoder.set_output_buffer(2, Some(&output), 0);
        encoder.set_bytes(3, &u32::try_from(width).map_err(CandleError::wrap)?);
        encoder.set_bytes(4, &self.eps);
        encoder.dispatch_thread_groups(
            MTLSize {
                width: count / width,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: (width / 2).max(1).next_power_of_two().min(1024),
                height: 1,
                depth: 1,
            },
        );
        Ok((
            MetalStorage::new(output, device.clone(), count, DType::F32),
            xl.shape().clone(),
        ))
    }
}
fn compile_pipeline(device: &MetalDevice, name: &str) -> Result<ComputePipeline> {
    let library = device
        .device()
        .new_library_with_source(include_str!("pointwise.metal"), None)
        .map_err(|e| Error::InferenceFailed(format!("compile pointwise kernels: {e}")))?;
    let function = library
        .get_function(name, None)
        .map_err(|e| Error::InferenceFailed(format!("load {name}: {e}")))?;
    device
        .device()
        .new_compute_pipeline_state_with_function(&function)
        .map_err(|e| Error::InferenceFailed(format!("create {name} pipeline: {e}")))
}
#[derive(Debug, Clone)]
pub(super) struct ConvKernel {
    pipeline: ComputePipeline,
}
impl ConvKernel {
    pub fn new(device: &Device) -> Result<Option<Self>> {
        let Device::Metal(device) = device else {
            return Ok(None);
        };
        let pipeline = compile_pipeline(device, "clef_conv")?;
        if pipeline.max_total_threads_per_threadgroup() < 256 {
            return Ok(None);
        }
        Ok(Some(Self { pipeline }))
    }
    pub fn forward(&self, x: &Tensor, w: &Tensor) -> CandleResult<Tensor> {
        x.apply_op2_no_bwd(w, self)
    }
}
impl CustomOp2 for ConvKernel {
    fn name(&self) -> &'static str {
        "clef-conv"
    }
    fn cpu_fwd(
        &self,
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
    ) -> CandleResult<(CpuStorage, Shape)> {
        Err(CandleError::Msg("fused convolution requires Metal".into()))
    }
    fn metal_fwd(
        &self,
        x: &MetalStorage,
        xl: &Layout,
        w: &MetalStorage,
        wl: &Layout,
    ) -> CandleResult<(MetalStorage, Shape)> {
        let (tokens, width) = xl.shape().dims2()?;
        if !(1..=4096).contains(&tokens)
            || !(1..=8192).contains(&width)
            || wl.shape().dims3()? != (width, 1, 4)
        {
            return Err(CandleError::Msg("fused convolution dimensions".into()));
        }
        let xo = checked_f32_buffer(x, xl)?;
        let wo = checked_f32_buffer(w, wl)?;
        let count = tokens
            .checked_mul(width)
            .ok_or_else(|| CandleError::Msg("convolution element count overflow".into()))?;
        let device = x.device();
        let output = device.new_buffer(count, DType::F32, "clef_conv")?;
        let guard = device.command_encoder()?;
        let encoder = guard.as_ref();
        encoder.set_compute_pipeline_state(&self.pipeline);
        encoder.set_input_buffer(0, Some(x.buffer()), xo);
        encoder.set_input_buffer(1, Some(w.buffer()), wo);
        encoder.set_output_buffer(2, Some(&output), 0);
        encoder.set_bytes(3, &u32::try_from(tokens).map_err(CandleError::wrap)?);
        encoder.set_bytes(4, &u32::try_from(width).map_err(CandleError::wrap)?);
        encoder.dispatch_thread_groups(
            MTLSize {
                width: count.div_ceil(256),
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: 256,
                height: 1,
                depth: 1,
            },
        );
        Ok((
            MetalStorage::new(output, device.clone(), count, DType::F32),
            xl.shape().clone(),
        ))
    }
}

#[derive(Debug, Clone)]
pub(super) struct DeltaPrepareKernel {
    pipeline: ComputePipeline,
}
impl DeltaPrepareKernel {
    pub fn new(device: &Device) -> Result<Option<Self>> {
        let Device::Metal(device) = device else {
            return Ok(None);
        };
        let pipeline = compile_pipeline(device, "clef_delta_prepare")?;
        if pipeline.max_total_threads_per_threadgroup() < 64
            || pipeline.as_ref().threadExecutionWidth() != 32
        {
            return Ok(None);
        }
        Ok(Some(Self { pipeline }))
    }
    pub fn forward(&self, mixed: &Tensor, g: &Tensor, beta: &Tensor) -> CandleResult<Tensor> {
        mixed.apply_op3_no_bwd(g, beta, self)
    }
}
impl CustomOp3 for DeltaPrepareKernel {
    fn name(&self) -> &'static str {
        "clef-delta-prepare"
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
        Err(CandleError::Msg(
            "DeltaNet preparation requires Metal".into(),
        ))
    }
    fn metal_fwd(
        &self,
        mixed: &MetalStorage,
        ml: &Layout,
        g: &MetalStorage,
        gl: &Layout,
        beta: &MetalStorage,
        bl: &Layout,
    ) -> CandleResult<(MetalStorage, Shape)> {
        let (tokens, width) = ml.shape().dims2()?;
        if !(1..=4096).contains(&tokens)
            || width != 8192
            || gl.dims() != [tokens, 32]
            || bl.dims() != gl.dims()
        {
            return Err(CandleError::Msg("DeltaNet preparation dimensions".into()));
        }
        let offsets = [
            checked_f32_buffer(mixed, ml)?,
            checked_f32_buffer(g, gl)?,
            checked_f32_buffer(beta, bl)?,
        ];
        let count = tokens * 32 * 386;
        let device = mixed.device();
        let output = device.new_buffer(count, DType::F32, "clef_delta_prepared")?;
        let guard = device.command_encoder()?;
        let encoder = guard.as_ref();
        encoder.set_compute_pipeline_state(&self.pipeline);
        for (index, (input, offset)) in [mixed, g, beta].into_iter().zip(offsets).enumerate() {
            encoder.set_input_buffer(index, Some(input.buffer()), offset);
        }
        encoder.set_output_buffer(3, Some(&output), 0);
        encoder.dispatch_thread_groups(
            MTLSize {
                width: tokens * 32,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: 64,
                height: 1,
                depth: 1,
            },
        );
        Ok((
            MetalStorage::new(output, device.clone(), count, DType::F32),
            Shape::from((tokens, 32, 386)),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::{ops::rms, qwen::causal_convolution};
    #[test]
    #[ignore = "requires Metal; validates custom dispatch boundaries"]
    fn test_should_reject_invalid_pointwise_inputs() -> Result<()> {
        let device = Device::new_metal(0)?;
        let rms = RmsKernel::new(&device, 1e-6)?.ok_or(Error::ArtifactMissing)?;
        let conv = ConvKernel::new(&device)?.ok_or(Error::ArtifactMissing)?;
        let x = Tensor::zeros((2, 4), DType::F32, &device)?;
        let w = Tensor::ones(4, DType::F32, &device)?;
        let cw = Tensor::ones((4, 1, 4), DType::F32, &device)?;
        for (input, weight) in [
            (x.to_dtype(DType::F16)?, w.clone()),
            (x.t()?, Tensor::ones(2, DType::F32, &device)?),
            (x.clone(), Tensor::ones(3, DType::F32, &device)?),
            (
                Tensor::zeros((2, 3), DType::F32, &device)?,
                Tensor::ones(3, DType::F32, &device)?,
            ),
        ] {
            assert!(matches!(
                rms.forward(&input, &weight),
                Err(CandleError::Msg(_))
            ));
        }
        for (input, weight) in [
            (x.to_dtype(DType::F16)?, cw.clone()),
            (x.t()?, Tensor::ones((2, 1, 4), DType::F32, &device)?),
            (x.clone(), Tensor::ones((4, 1, 3), DType::F32, &device)?),
            (Tensor::zeros((4097, 4), DType::F32, &device)?, cw),
        ] {
            assert!(matches!(
                conv.forward(&input, &weight),
                Err(CandleError::Msg(_))
            ));
        }
        device.synchronize()?;
        Ok(())
    }
    #[test]
    #[ignore = "requires Metal; causal tails and full Flash convolution width"]
    #[allow(
        clippy::cast_precision_loss,
        reason = "bounded deterministic test indices"
    )]
    fn test_should_match_fused_causal_convolution() -> Result<()> {
        let device = Device::new_metal(0)?;
        let kernel = ConvKernel::new(&device)?.ok_or(Error::ArtifactMissing)?;
        for width in [1, 3, 128, 8192] {
            for tokens in [1, 2, 3, 4, 17, 139] {
                let x = Tensor::from_vec(
                    (0..tokens * width)
                        .map(|i| (i as f32 * 0.073).sin())
                        .collect(),
                    (tokens, width),
                    &device,
                )?;
                let w = Tensor::from_vec(
                    (0..width * 4).map(|i| (i as f32 * 0.027).cos()).collect(),
                    (width, 1, 4),
                    &device,
                )?;
                let expected = causal_convolution(&x, &w, 4)?;
                let actual = kernel.forward(&x, &w)?;
                let error = (&actual - expected)?
                    .abs()?
                    .flatten_all()?
                    .max(0)?
                    .to_scalar::<f32>()?;
                assert_eq!(error.to_bits(), 0, "convolution {tokens}/{width}: {error}");
            }
        }
        Ok(())
    }
    #[test]
    #[ignore = "requires Metal; checks the complete reference preparation graph"]
    #[allow(
        clippy::cast_precision_loss,
        reason = "bounded deterministic test indices"
    )]
    fn test_should_match_fused_delta_preparation_bit_exactly() -> Result<()> {
        use candle_core::D;
        let device = Device::new_metal(0)?;
        let kernel = DeltaPrepareKernel::new(&device)?.ok_or(Error::ArtifactMissing)?;
        for tokens in [1, 17, 139, 1024] {
            for zero in [false, true] {
                let mixed = Tensor::from_vec(
                    (0..tokens * 8192)
                        .map(|i| if zero { 0.0 } else { (i as f32 * 0.017).sin() })
                        .collect(),
                    (tokens, 8192),
                    &device,
                )?;
                let g = Tensor::full(-0.03f32, (tokens, 32), &device)?;
                let beta = Tensor::full(0.65f32, (tokens, 32), &device)?;
                let repeat = Tensor::new((0..32u32).map(|i| i / 2).collect::<Vec<_>>(), &device)?;
                let l2 = |x: Tensor| -> Result<Tensor> {
                    let root = (x.sqr()?.sum_keepdim(D::Minus1)? + 1e-6)?.sqrt()?;
                    Ok(x.broadcast_div(&root)?)
                };
                let q = (l2(mixed
                    .narrow(1, 0, 2048)?
                    .reshape((tokens, 16, 128))?
                    .index_select(&repeat, 1)?)?
                    / 128f64.sqrt())?;
                let k = l2(mixed
                    .narrow(1, 2048, 2048)?
                    .reshape((tokens, 16, 128))?
                    .index_select(&repeat, 1)?)?;
                let v = mixed.narrow(1, 4096, 4096)?.reshape((tokens, 32, 128))?;
                let expected = Tensor::cat(
                    &[&q, &k, &v, &g.exp()?.unsqueeze(2)?, &beta.unsqueeze(2)?],
                    2,
                )?
                .contiguous()?;
                let actual = kernel.forward(&mixed, &g, &beta)?;
                let error = (actual - expected)?
                    .abs()?
                    .flatten_all()?
                    .max(0)?
                    .to_scalar::<f32>()?;
                assert_eq!(
                    error.to_bits(),
                    0,
                    "preparation {tokens} zero={zero}: {error}"
                );
            }
        }
        assert!(
            kernel
                .forward(
                    &Tensor::zeros((1, 8191), DType::F32, &device)?,
                    &Tensor::zeros((1, 32), DType::F32, &device)?,
                    &Tensor::zeros((1, 32), DType::F32, &device)?
                )
                .is_err()
        );
        Ok(())
    }
    #[test]
    #[ignore = "requires Metal; checks exact reference reduction and dtype rounding"]
    #[allow(
        clippy::cast_precision_loss,
        reason = "bounded deterministic test indices"
    )]
    fn test_should_match_fused_rms_reference() -> Result<()> {
        let device = Device::new_metal(0)?;
        let kernel = RmsKernel::new(&device, 1e-6)?.ok_or(Error::ArtifactMissing)?;
        for width in [1, 4, 128, 256, 4096] {
            for rows in [1, 17, 139] {
                let x = Tensor::from_vec(
                    (0..rows * width)
                        .map(|i| (i as f32 * 0.073).sin())
                        .collect(),
                    (rows, width),
                    &device,
                )?;
                let w = Tensor::full(0.7f32, width, &device)?;
                let expected = rms(&x, &w, 1e-6, false)?;
                let actual = kernel.forward(&x, &w)?;
                let error = (&actual - &expected)?
                    .abs()?
                    .flatten_all()?
                    .max(0)?
                    .to_scalar::<f32>()?;
                assert!(error <= 1e-6, "RMS {rows}/{width}: {error}");
                let expected = expected.to_dtype(DType::F16)?.to_dtype(DType::F32)?;
                let actual = actual.to_dtype(DType::F16)?.to_dtype(DType::F32)?;
                let rounded_error = (&actual - expected)?
                    .abs()?
                    .flatten_all()?
                    .max(0)?
                    .to_scalar::<f32>()?;
                assert_eq!(
                    rounded_error.to_bits(),
                    0,
                    "RMS half rounding {rows}/{width}: {rounded_error}"
                );
            }
        }
        Ok(())
    }
}
