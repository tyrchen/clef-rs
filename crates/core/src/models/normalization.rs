//! Loaded RMS parameters and shared Metal normalization pipeline.

use candle_core::{DType, Device, Tensor};

#[cfg(all(feature = "metal", target_os = "macos"))]
use super::pointwise::{ConvKernel, RmsKernel};
use super::{ops::rms, weights::Weights};
use crate::{Error, Result};

#[derive(Debug)]
pub(super) struct Norms {
    eps: f64,
    #[cfg(all(feature = "metal", target_os = "macos"))]
    kernel: Option<RmsKernel>,
    #[cfg(all(feature = "metal", target_os = "macos"))]
    convolution: Option<ConvKernel>,
}
impl Norms {
    pub fn new(device: &Device, eps: f64) -> Result<Self> {
        if !eps.is_finite() || !(1e-12..=1.).contains(&eps) {
            return Err(Error::InvalidRequest("normalization epsilon".into()));
        }
        #[cfg(not(all(feature = "metal", target_os = "macos")))]
        let _ = device;
        Ok(Self {
            eps,
            #[cfg(all(feature = "metal", target_os = "macos"))]
            kernel: RmsKernel::new(device, eps)?,
            #[cfg(all(feature = "metal", target_os = "macos"))]
            convolution: ConvKernel::new(device)?,
        })
    }
    #[cfg(all(feature = "metal", target_os = "macos"))]
    pub fn convolution(&self) -> Option<ConvKernel> {
        self.convolution.clone()
    }
    pub fn load(
        &self,
        weights: &mut Weights,
        name: &str,
        width: usize,
        offset: bool,
    ) -> Result<Rms> {
        let weight = weights.take(name, &[width])?.to_dtype(DType::F32)?;
        let weight = if offset { (weight + 1.)? } else { weight };
        Ok(Rms {
            weight,
            eps: self.eps,
            #[cfg(all(feature = "metal", target_os = "macos"))]
            kernel: self.kernel.clone(),
        })
    }
}
#[derive(Debug)]
pub(super) struct Rms {
    weight: Tensor,
    eps: f64,
    #[cfg(all(feature = "metal", target_os = "macos"))]
    kernel: Option<RmsKernel>,
}
impl Rms {
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        measured!("normalization", x.device(), self.forward_inner(x))
    }
    fn forward_inner(&self, x: &Tensor) -> Result<Tensor> {
        #[cfg(all(feature = "metal", target_os = "macos"))]
        if let Some(kernel) = &self.kernel {
            return Ok(kernel
                .forward(&x.to_dtype(DType::F32)?.contiguous()?, &self.weight)?
                .to_dtype(x.dtype())?);
        }
        rms(x, &self.weight, self.eps, false)
    }
}
