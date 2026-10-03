//! Safe F16 projections through Metal 4 tensor operations on Apple GPU family 10.

use candle_core::{
    CpuStorage, CustomOp2, CustomOp3, DType, Device, Error as CandleError, Layout, MetalDevice,
    MetalStorage, Result as CandleResult, Shape, Tensor, backend::BackendStorage,
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
    gated_pipeline: ComputePipeline,
    tile_m: usize,
    tile_n: usize,
    walk: usize,
}
#[derive(Debug, Clone)]
pub(super) struct GemmKernel {
    standard: TileKernel,
    wide: TileKernel,
    #[cfg(test)]
    local: TileKernel,
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
            #[cfg(test)]
            local: TileKernel::compile_variant(device, 64, 128, 2, 0)?,
        }))
    }
    pub fn forward(&self, x: &Tensor, weight: &Tensor) -> CandleResult<Tensor> {
        let (rows, inner) = x.dims2()?;
        let columns = weight.dim(0)?;
        let tile = self.select(rows, inner, columns);
        tile.forward(x, weight)
    }
    pub fn gated(&self, x: &Tensor, gate: &Tensor, up: &Tensor) -> CandleResult<Tensor> {
        x.apply_op3_no_bwd(gate, up, &self.wide)
    }
    fn select(&self, rows: usize, inner: usize, columns: usize) -> &TileKernel {
        if rows >= 2048 || inner == 12288 || (rows >= 256 && columns <= 8192) {
            &self.wide
        } else {
            &self.standard
        }
    }
}
impl TileKernel {
    fn compile(device: &MetalDevice, tile_m: usize, tile_n: usize) -> Result<Self> {
        Self::compile_variant(device, tile_m, tile_n, 0, 0)
    }
    fn compile_variant(
        device: &MetalDevice,
        tile_m: usize,
        tile_n: usize,
        walk: usize,
        k_block: usize,
    ) -> Result<Self> {
        let options = MTLCompileOptions::new();
        options.setLanguageVersion(MTLLanguageVersion::Version4_0);
        options.setMathMode(MTLMathMode::Safe);
        let source = format!(
            "#define CLEF_TILE_M {tile_m}\n#define CLEF_TILE_N {tile_n}\n#define CLEF_WALK \
             {walk}\n#define CLEF_K_BLOCK {k_block}\n{}",
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
        let gated_function = library
            .get_function("clef_neural_gated", None)
            .map_err(|e| Error::InferenceFailed(format!("load gated projection: {e}")))?;
        let gated_pipeline = device
            .device()
            .new_compute_pipeline_state_with_function(&gated_function)
            .map_err(|e| Error::InferenceFailed(format!("create gated projection: {e}")))?;
        if gated_pipeline.max_total_threads_per_threadgroup() < 128
            || gated_pipeline.as_ref().threadExecutionWidth() != 32
        {
            return Err(Error::UnsupportedCapability(
                "gated projection thread geometry".into(),
            ));
        }
        Ok(Self {
            pipeline,
            gated_pipeline,
            tile_m,
            tile_n,
            walk,
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
        self.dispatch(x, xl, w, wl, None)
    }
}
impl CustomOp3 for TileKernel {
    fn name(&self) -> &'static str {
        "clef-metal4-gated"
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
        Err(CandleError::Msg("gated projection requires Metal".into()))
    }
    fn metal_fwd(
        &self,
        x: &MetalStorage,
        xl: &Layout,
        gate: &MetalStorage,
        gl: &Layout,
        up: &MetalStorage,
        ul: &Layout,
    ) -> CandleResult<(MetalStorage, Shape)> {
        self.dispatch(x, xl, gate, gl, Some((up, ul)))
    }
}
impl TileKernel {
    fn dispatch(
        &self,
        x: &MetalStorage,
        xl: &Layout,
        w: &MetalStorage,
        wl: &Layout,
        up: Option<(&MetalStorage, &Layout)>,
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
        let up_offset = match up {
            Some((storage, layout)) => {
                if layout.shape() != wl.shape() {
                    return Err(CandleError::Msg(
                        "gated projection weight dimensions".into(),
                    ));
                }
                Some(buffer_offset(storage, layout)?)
            }
            None => None,
        };
        let count = rows * columns;
        let device = x.device();
        let output = device.new_buffer(count, DType::F16, "clef_neural_gemm")?;
        let guard = device.command_encoder()?;
        let encoder = guard.as_ref();
        encoder.set_compute_pipeline_state(if up.is_some() {
            &self.gated_pipeline
        } else {
            &self.pipeline
        });
        if let (Some((storage, _)), Some(offset)) = (up, up_offset) {
            encoder.set_input_buffer(6, Some(storage.buffer()), offset);
        }
        for (index, (input, offset)) in [x, w].into_iter().zip(offsets).enumerate() {
            encoder.set_input_buffer(index, Some(input.buffer()), offset);
        }
        encoder.set_output_buffer(2, Some(&output), 0);
        for (index, value) in [rows, columns, inner].into_iter().enumerate() {
            encoder.set_bytes(index + 3, &u32::try_from(value).map_err(CandleError::wrap)?);
        }
        encoder.dispatch_thread_groups(
            MTLSize {
                width: if self.walk == 1 {
                    columns.div_ceil(self.tile_n).div_ceil(8)
                        * rows.div_ceil(self.tile_m).div_ceil(8)
                        * 64
                } else if self.walk == 0 {
                    columns.div_ceil(self.tile_n)
                } else {
                    columns.div_ceil(self.tile_n) * rows.div_ceil(self.tile_m)
                },
                height: if self.walk == 0 {
                    rows.div_ceil(self.tile_m)
                } else {
                    1
                },
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
    #[ignore = "requires M5; rejects malformed custom projection inputs"]
    fn test_should_reject_invalid_metal4_projection_inputs() -> Result<()> {
        let device = Device::new_metal(0)?;
        let kernel = GemmKernel::new(&device, DType::F16)?.ok_or(Error::ArtifactMissing)?;
        let x = Tensor::ones((2, 4), DType::F16, &device)?;
        let w = Tensor::ones((3, 4), DType::F16, &device)?;
        for (input, weight) in [
            (x.to_dtype(DType::F32)?, w.clone()),
            (x.clone(), w.to_dtype(DType::F32)?),
            (x.t()?, Tensor::ones((3, 2), DType::F16, &device)?),
            (x.clone(), w.t()?),
            (Tensor::ones((4097, 4), DType::F16, &device)?, w.clone()),
            (x.clone(), Tensor::ones((3, 5), DType::F16, &device)?),
        ] {
            assert!(kernel.forward(&input, &weight).is_err());
        }
        for up in [
            w.to_dtype(DType::F32)?,
            w.t()?,
            Tensor::ones((2, 4), DType::F16, &device)?,
        ] {
            assert!(kernel.gated(&x, &w, &up).is_err());
        }
        let offset_x = Tensor::ones((3, 4), DType::F16, &device)?.narrow(0, 1, 2)?;
        let offset_w = Tensor::ones((4, 4), DType::F16, &device)?.narrow(0, 1, 3)?;
        assert_eq!(
            kernel.gated(&offset_x, &offset_w, &offset_w)?.dims(),
            &[2, 3]
        );
        Ok(())
    }
    #[test]
    #[ignore = "requires M5; profiles fused FFN register occupancy"]
    fn test_should_profile_metal4_gated_tiles() -> Result<()> {
        let device = Device::new_metal(0)?;
        let Device::Metal(metal) = &device else {
            return Err(Error::ArtifactMissing);
        };
        let weight = Tensor::full(0.01f32, (12288, 4096), &device)?.to_dtype(DType::F16)?;
        for (m, n) in [
            (16, 256),
            (32, 128),
            (32, 256),
            (64, 64),
            (64, 128),
            (128, 64),
            (128, 128),
        ] {
            let tile = TileKernel::compile(metal, m, n)?;
            for rows in [1024, 4096] {
                let x = Tensor::full(0.01f32, (rows, 4096), &device)?.to_dtype(DType::F16)?;
                for sample in 0..4 {
                    device.synchronize()?;
                    let started = Instant::now();
                    x.apply_op3_no_bwd(&weight, &weight, &tile)?;
                    device.synchronize()?;
                    if sample > 0 {
                        eprintln!(
                            "GATED_TILE {m}/{n} {rows} sample={sample}: {:.3} ms",
                            started.elapsed().as_secs_f64() * 1000.
                        );
                    }
                }
            }
        }
        Ok(())
    }
    #[test]
    #[ignore = "requires M5; long-prefill GEMM locality and K synchronization sweep"]
    fn test_should_profile_metal4_long_prefill_variants() -> Result<()> {
        let device = Device::new_metal(0)?;
        let Device::Metal(metal) = &device else {
            return Err(Error::ArtifactMissing);
        };
        for tile_n in [64, 128] {
            for walk in [0, 1, 2] {
                for k_block in [0, 128, 512, 1024] {
                    let kernel = TileKernel::compile_variant(metal, 64, tile_n, walk, k_block)?;
                    for (n, k) in [(12288, 4096), (4096, 12288), (8192, 4096), (4096, 4096)] {
                        let w = Tensor::full(0.01f32, (n, k), &device)?.to_dtype(DType::F16)?;
                        for rows in [1024, 4096] {
                            let x =
                                Tensor::full(0.01f32, (rows, k), &device)?.to_dtype(DType::F16)?;
                            kernel.forward(&x, &w)?;
                            device.synchronize()?;
                            let started = Instant::now();
                            for _ in 0..3 {
                                kernel.forward(&x, &w)?;
                                device.synchronize()?;
                            }
                            eprintln!(
                                "LONG_GEMM 64/{tile_n} walk={walk} bk={k_block} {rows}/{n}/{k}: \
                                 {:.3} ms",
                                started.elapsed().as_secs_f64() * 1000. / 3.
                            );
                        }
                    }
                }
            }
        }
        Ok(())
    }
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
        clippy::float_cmp,
        reason = "bounded test indices and exact half-rounding regression checks"
    )]
    fn test_should_match_metal4_projections_and_profile_flash_shapes() -> Result<()> {
        let device = Device::new_metal(0)?;
        let kernel = GemmKernel::new(&device, DType::F16)?
            .ok_or_else(|| Error::UnsupportedCapability("Metal 4 test device".into()))?;
        for tile in [&kernel.standard, &kernel.wide, &kernel.local] {
            for (m, n, k) in [
                (1, 1, 1),
                (17, 73, 65),
                (139, 129, 128),
                (17, 73, 4096),
                (513, 137, 65),
                (1025, 257, 128),
            ] {
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
                let xm = x.to_device(&device)?;
                let wm = w.to_device(&device)?;
                let um = (&wm * 0.7)?;
                let expected_gated = (tile.forward(&xm, &wm)?.silu()? * tile.forward(&xm, &um)?)?;
                let actual_gated = xm.apply_op3_no_bwd(&wm, &um, tile)?;
                let drift = (actual_gated.to_dtype(DType::F32)?
                    - expected_gated.to_dtype(DType::F32)?)?
                .abs()?
                .flatten_all()?
                .max(0)?
                .to_scalar::<f32>()?;
                assert!(drift == 0., "gated GEMM {m}/{n}/{k}: half drift {drift}");
            }
        }
        profile_projections(&device, &kernel)
    }
    fn profile_projections(device: &Device, kernel: &GemmKernel) -> Result<()> {
        let w = Tensor::full(0.01f32, (12288, 4096), device)?.to_dtype(DType::F16)?;
        for rows in [139, 256, 1024, 4096] {
            let x = Tensor::full(0.01f32, (rows, 4096), device)?.to_dtype(DType::F16)?;
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
        for rows in [1024, 4096] {
            let x = Tensor::full(0.01f32, (rows, 4096), device)?.to_dtype(DType::F16)?;
            for (name, tile) in [("wide-2d", &kernel.wide), ("local-1d", &kernel.local)] {
                for sample in 0..4 {
                    device.synchronize()?;
                    let started = Instant::now();
                    x.apply_op3_no_bwd(&w, &w, tile)?;
                    device.synchronize()?;
                    if sample > 0 {
                        eprintln!(
                            "GATED_LAYOUT {rows} {name} sample={sample}: {:.3} ms",
                            started.elapsed().as_secs_f64() * 1000.
                        );
                    }
                }
            }
        }
        for rows in [1024, 4096] {
            let x = Tensor::full(0.01f32, (rows, 4096), device)?.to_dtype(DType::F16)?;
            for fused in [false, true] {
                for sample in 0..4 {
                    device.synchronize()?;
                    let started = Instant::now();
                    if fused {
                        kernel.gated(&x, &w, &w)?;
                    } else {
                        (kernel.forward(&x, &w)?.silu()? * kernel.forward(&x, &w)?)?;
                    }
                    device.synchronize()?;
                    if sample > 0 {
                        eprintln!(
                            "GATED {rows} fused={fused} sample={sample}: {:.3} ms",
                            started.elapsed().as_secs_f64() * 1000.
                        );
                    }
                }
            }
        }
        Ok(())
    }
}
