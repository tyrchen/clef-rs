//! Backbone projections with a qualified device-specific matrix kernel.

#[cfg(all(feature = "metal", target_os = "macos"))]
use candle_core::Device;
use candle_core::{DType, Result as CandleResult, Tensor};
use candle_nn::{Linear, Module};

#[cfg(all(feature = "metal", target_os = "macos"))]
use super::neural::GemmKernel;
use super::weights::Weights;
use crate::{Error, Result};

#[derive(Debug)]
pub(super) struct Projections {
    dtype: DType,
    #[cfg(all(feature = "metal", target_os = "macos"))]
    kernel: Option<GemmKernel>,
}
impl Default for Projections {
    fn default() -> Self {
        Self {
            dtype: DType::F32,
            #[cfg(all(feature = "metal", target_os = "macos"))]
            kernel: None,
        }
    }
}
impl Projections {
    #[cfg(all(feature = "metal", target_os = "macos"))]
    pub fn new(device: &Device, dtype: DType) -> Result<Self> {
        Ok(Self {
            dtype,
            kernel: GemmKernel::new(device, dtype)?,
        })
    }
    pub fn load(
        &self,
        weights: &mut Weights,
        name: &str,
        input: usize,
        output: usize,
        bias: bool,
    ) -> Result<Projection> {
        let linear = weights.linear(name, input, output, bias)?;
        if linear.weight().dtype() != self.dtype {
            return Err(Error::InferenceFailed(
                "backbone projection dtype mismatch".into(),
            ));
        }
        Ok(Projection {
            linear,
            #[cfg(all(feature = "metal", target_os = "macos"))]
            kernel: self.kernel.clone(),
        })
    }
}
#[derive(Debug)]
pub(super) struct Projection {
    linear: Linear,
    #[cfg(all(feature = "metal", target_os = "macos"))]
    kernel: Option<GemmKernel>,
}
impl Projection {
    pub fn weight(&self) -> &Tensor {
        self.linear.weight()
    }
}
impl Module for Projection {
    fn forward(&self, input: &Tensor) -> CandleResult<Tensor> {
        #[cfg(all(feature = "metal", target_os = "macos"))]
        if let Some(kernel) = &self.kernel {
            // Tiny decay/beta projections are faster on the existing kernel.
            if input.rank() == 2 && self.linear.weight().dim(0)? >= 64 {
                let output = kernel.forward(input, self.linear.weight())?;
                return match self.linear.bias() {
                    Some(bias) => output.broadcast_add(bias),
                    None => Ok(output),
                };
            }
        }
        self.linear.forward(input)
    }
}
