//! Safe F16 projections through Metal 4 tensor operations on Apple GPU family 10.

use candle_core::{
    CpuStorage, CustomOp2, DType, Device, Error as CandleError, Layout, MetalDevice, MetalStorage,
    Result as CandleResult, Shape, Tensor, backend::BackendStorage,
};
use candle_metal_kernels::metal::ComputePipeline;
use objc2_metal::{
    MTLCompileOptions, MTLComputePipelineState, MTLDevice, MTLGPUFamily, MTLLanguageVersion,
    MTLMathMode, MTLSize,
};

use crate::{Error, Result};

#[derive(Debug, Clone)]
struct TileKernel {
    pipeline: ComputePipeline,
    tile_m: usize,
    tile_n: usize,
}
#[derive(Debug, Clone)]
pub(super) struct GemmKernel {
    standard: TileKernel,
    wide: TileKernel,
}
impl GemmKernel {
    pub fn new(device: &Device, dtype: DType) -> Result<Option<Self>> {
        let Device::Metal(device) = device else {
            return Ok(None);
        };
        if dtype != DType::F16
            || !device
                .device()
                .as_ref()
                .supportsFamily(MTLGPUFamily::Apple10)
        {
            return Ok(None);
        }
        Ok(Some(Self {
            standard: TileKernel::compile(device, 64, 64)?,
            wide: TileKernel::compile(device, 64, 128)?,
        }))
    }
    pub fn forward(&self, x: &Tensor, weight: &Tensor) -> CandleResult<Tensor> {
        let (rows, inner) = x.dims2()?;
        let columns = weight.dim(0)?;
        let tile = if rows >= 2048 || inner == 12288 || (rows >= 256 && columns <= 8192) {
            &self.wide
        } else {
            &self.standard
        };
        tile.forward(x, weight)
    }
}
impl TileKernel {
    fn compile(device: &MetalDevice, tile_m: usize, tile_n: usize) -> Result<Self> {
        let options = MTLCompileOptions::new();
        options.setLanguageVersion(MTLLanguageVersion::Version4_0);
        options.setMathMode(MTLMathMode::Safe);
        let source = format!(
            "#define CLEF_TILE_M {tile_m}\n#define CLEF_TILE_N {tile_n}\n{}",
            include_str!("neural.metal")
        );
        let library = device
            .device()
            .new_library_with_source(&source, Some(&options))
            .map_err(|e| Error::InferenceFailed(format!("compile Metal 4 projection: {e}")))?;
        let function = library
            .get_function("clef_neural_gemm", None)
            .map_err(|e| Error::InferenceFailed(format!("load Metal 4 projection: {e}")))?;
        let pipeline = device
            .device()
            .new_compute_pipeline_state_with_function(&function)
            .map_err(|e| Error::InferenceFailed(format!("create Metal 4 projection: {e}")))?;
        if pipeline.max_total_threads_per_threadgroup() < 128
            || pipeline.as_ref().threadExecutionWidth() != 32
        {
            return Err(Error::UnsupportedCapability(
                "Metal 4 projection thread geometry".into(),
            ));
        }
        Ok(Self {
            pipeline,
            tile_m,
            tile_n,
        })
    }
    pub fn forward(&self, x: &Tensor, weight: &Tensor) -> CandleResult<Tensor> {
        x.apply_op2_no_bwd(weight, self)
    }
}
fn buffer_offset(input: &MetalStorage, layout: &Layout) -> CandleResult<usize> {
    if input.dtype() != DType::F16 || !layout.is_contiguous() {
        return Err(CandleError::Msg(
            "Metal 4 projection needs contiguous F16".into(),
        ));
    }
    let offset = layout
        .start_offset()
        .checked_mul(2)
        .ok_or_else(|| CandleError::Msg("projection offset overflow".into()))?;
    let end = layout
        .shape()
        .elem_count()
        .checked_mul(2)
        .and_then(|n| offset.checked_add(n))
        .ok_or_else(|| CandleError::Msg("projection buffer overflow".into()))?;
    if end > input.buffer().length() {
        return Err(CandleError::Msg("projection buffer bounds".into()));
    }
    Ok(offset)
}
impl CustomOp2 for TileKernel {
    fn name(&self) -> &'static str {
        "clef-metal4-gemm"
    }
    fn cpu_fwd(
        &self,
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
    ) -> CandleResult<(CpuStorage, Shape)> {
        Err(CandleError::Msg("Metal 4 projection requires Metal".into()))
    }
    fn metal_fwd(
        &self,
        x: &MetalStorage,
        xl: &Layout,
        w: &MetalStorage,
        wl: &Layout,
    ) -> CandleResult<(MetalStorage, Shape)> {
        let (rows, inner) = xl.shape().dims2()?;
        let (columns, wi) = wl.shape().dims2()?;
        if !(1..=4096).contains(&rows)
            || !(1..=16384).contains(&columns)
            || !(1..=12288).contains(&inner)
            || wi != inner
        {
            return Err(CandleError::Msg("Metal 4 projection dimensions".into()));
        }
        let offsets = [buffer_offset(x, xl)?, buffer_offset(w, wl)?];
        let count = rows * columns;
        let device = x.device();
        let output = device.new_buffer(count, DType::F16, "clef_neural_gemm")?;
        let guard = device.command_encoder()?;
        let encoder = guard.as_ref();
        encoder.set_compute_pipeline_state(&self.pipeline);
        for (index, (input, offset)) in [x, w].into_iter().zip(offsets).enumerate() {
            encoder.set_input_buffer(index, Some(input.buffer()), offset);
        }
        encoder.set_output_buffer(2, Some(&output), 0);
        for (index, value) in [rows, columns, inner].into_iter().enumerate() {
            encoder.set_bytes(index + 3, &u32::try_from(value).map_err(CandleError::wrap)?);
        }
        encoder.dispatch_thread_groups(
            MTLSize {
                width: columns.div_ceil(self.tile_n),
                height: rows.div_ceil(self.tile_m),
                depth: 1,
            },
            MTLSize {
                width: 128,
                height: 1,
                depth: 1,
            },
        );
        Ok((
            MetalStorage::new(output, device.clone(), count, DType::F16),
            Shape::from((rows, columns)),
        ))
    }
}

#[cfg(test)]
mod tests {
    use std::time::Instant;

    use super::*;
    #[test]
    #[ignore = "requires M5; synchronized full-shape tile sweep"]
    fn test_should_profile_metal4_projection_tiles() -> Result<()> {
        let device = Device::new_metal(0)?;
        let Device::Metal(metal) = &device else {
            return Err(Error::ArtifactMissing);
        };
        for (tile_m, tile_n) in [
            (32, 64),
            (32, 128),
            (64, 64),
            (64, 128),
            (128, 64),
            (128, 128),
        ] {
            let kernel = TileKernel::compile(metal, tile_m, tile_n)?;
            for (n, k) in [
                (12288, 4096),
                (4096, 12288),
                (8192, 4096),
                (4096, 4096),
                (1024, 4096),
            ] {
                let w = Tensor::full(0.01f32, (n, k), &device)?.to_dtype(DType::F16)?;
                for rows in [139, 256, 1024, 4096] {
                    let x = Tensor::full(0.01f32, (rows, k), &device)?.to_dtype(DType::F16)?;
                    kernel.forward(&x, &w)?;
                    device.synchronize()?;
                    let started = Instant::now();
                    for _ in 0..3 {
                        kernel.forward(&x, &w)?;
                        device.synchronize()?;
                    }
                    eprintln!(
                        "TILE {tile_m}/{tile_n} {rows}/{n}/{k}: {:.3} ms",
                        started.elapsed().as_secs_f64() * 1000. / 3.
                    );
                }
            }
        }
        Ok(())
    }
    #[test]
    #[ignore = "requires Apple GPU family 10 and Metal 4"]
    #[allow(
        clippy::cast_precision_loss,
        reason = "bounded deterministic test indices"
    )]
    fn test_should_match_metal4_projections_and_profile_flash_shapes() -> Result<()> {
        let device = Device::new_metal(0)?;
        let kernel = GemmKernel::new(&device, DType::F16)?
            .ok_or_else(|| Error::UnsupportedCapability("Metal 4 test device".into()))?;
        for tile in [&kernel.standard, &kernel.wide] {
            for (m, n, k) in [(1, 1, 1), (17, 73, 65), (139, 129, 128), (17, 73, 4096)] {
                let x = Tensor::from_vec(
                    (0..m * k).map(|i| (i as f32 * 0.013).sin()).collect(),
                    (m, k),
                    &Device::Cpu,
                )?
                .to_dtype(DType::F16)?;
                let w = Tensor::from_vec(
                    (0..n * k)
                        .map(|i| (i as f32 * 0.029).cos() * 0.015_625)
                        .collect(),
                    (n, k),
                    &Device::Cpu,
                )?
                .to_dtype(DType::F16)?;
                let expected = x
                    .to_dtype(DType::F32)?
                    .matmul(&w.to_dtype(DType::F32)?.t()?)?
                    .to_dtype(DType::F16)?
                    .to_dtype(DType::F32)?;
                let actual = tile
                    .forward(&x.to_device(&device)?, &w.to_device(&device)?)?
                    .to_dtype(DType::F32)?
                    .to_device(&Device::Cpu)?;
                let error = (&actual - expected)?
                    .abs()?
                    .flatten_all()?
                    .max(0)?
                    .to_scalar::<f32>()?;
                assert!(error < 0.001, "Metal 4 GEMM {m}/{n}/{k}: error {error}");
            }
        }
        let w = Tensor::full(0.01f32, (12288, 4096), &device)?.to_dtype(DType::F16)?;
        for rows in [139, 256, 1024] {
            let x = Tensor::full(0.01f32, (rows, 4096), &device)?.to_dtype(DType::F16)?;
            for native in [false, true] {
                if native {
                    kernel.forward(&x, &w)?;
                } else {
                    x.matmul(&w.t()?)?;
                }
                device.synchronize()?;
                let started = Instant::now();
                for _ in 0..3 {
                    if native {
                        kernel.forward(&x, &w)?;
                    } else {
                        x.matmul(&w.t()?)?;
                    }
                    device.synchronize()?;
                }
                eprintln!(
                    "GEMM {rows}/12288/4096 native={native}: {:.3} ms",
                    started.elapsed().as_secs_f64() * 1000. / 3.
                );
            }
        }
        Ok(())
    }
}
