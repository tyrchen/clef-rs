//! Strict safe slice loading with approved tensor shapes and bounded shard staging.
use std::{
    collections::{HashMap, HashSet},
    time::{Duration, Instant},
};

use candle_core::{DType, Device, Tensor};
use candle_nn::Linear;
use safetensors::{Dtype, SafeTensors};
use serde::Deserialize;

use super::ops::{MultiAttention, Norm};
use crate::{Error, Result, artifacts::VerifiedSnapshot};

#[derive(Debug, Deserialize)]
struct TensorSpec {
    shard: String,
    shape: Vec<usize>,
    dtype: String,
}
#[derive(Debug)]
pub(crate) struct Weights {
    tensors: HashMap<String, Tensor>,
}
impl Weights {
    pub fn load(
        snapshot: &VerifiedSnapshot,
        device: &Device,
        dtype: DType,
        load_vision: bool,
    ) -> Result<Self> {
        let bytes = include_bytes!("../artifacts/clef-flash-tensors.json");
        let specs: HashMap<String, TensorSpec> = serde_json::from_slice(bytes)?;
        let mut tensors = HashMap::new();
        let mut seen = HashSet::new();
        let deadline = Instant::now() + Duration::from_secs(600);
        for file in snapshot
            .manifest()
            .files
            .iter()
            .filter(|f| f.name.ends_with(".safetensors"))
        {
            let data = snapshot.read_verified(&file.name, deadline)?;
            validate_header(&data)?;
            let safe = SafeTensors::deserialize(&data)?;
            let expected = specs.values().filter(|s| s.shard == file.name).count();
            if safe.len() != expected {
                return Err(Error::IntegrityMismatch("shard tensor count".into()));
            }
            for (name, view) in safe.tensors() {
                if Instant::now() >= deadline {
                    return Err(Error::DeadlineExceeded);
                }
                let spec = specs
                    .get(&name)
                    .ok_or_else(|| Error::IntegrityMismatch("unknown tensor".into()))?;
                if spec.shard != file.name
                    || view.shape() != spec.shape
                    || spec.dtype != "BF16"
                    || view.dtype() != Dtype::BF16
                    || !seen.insert(name.clone())
                {
                    return Err(Error::IntegrityMismatch(
                        "tensor architecture/shape/dtype".into(),
                    ));
                }
                let elements = spec
                    .shape
                    .iter()
                    .try_fold(1_usize, |n, d| n.checked_mul(*d))
                    .ok_or_else(|| Error::IntegrityMismatch("tensor dimension overflow".into()))?;
                if elements.checked_mul(2) != Some(view.data().len()) {
                    return Err(Error::IntegrityMismatch("tensor bytes".into()));
                }
                // Text profiles deliberately skip the validated vision namespace on-device.
                if name.starts_with("model.visual.") && !load_vision {
                    continue;
                }
                let tensor =
                    Tensor::from_raw_buffer(view.data(), DType::BF16, view.shape(), device)?
                        .to_dtype(dtype)?;
                tensors.insert(name, tensor);
            }
        }
        if seen.len() != specs.len() {
            return Err(Error::IntegrityMismatch("missing required tensor".into()));
        }
        Ok(Self { tensors })
    }
    pub fn take(&mut self, name: &str, shape: &[usize]) -> Result<Tensor> {
        let tensor = self
            .tensors
            .remove(name)
            .ok_or_else(|| Error::IntegrityMismatch(format!("missing tensor {name}")))?;
        if tensor.dims() != shape {
            return Err(Error::IntegrityMismatch(format!("tensor shape {name}")));
        }
        Ok(tensor)
    }
    pub fn linear(
        &mut self,
        prefix: &str,
        input: usize,
        output: usize,
        bias: bool,
    ) -> Result<Linear> {
        let weight = self.take(&format!("{prefix}.weight"), &[output, input])?;
        let bias = if bias {
            Some(self.take(&format!("{prefix}.bias"), &[output])?)
        } else {
            None
        };
        Ok(Linear::new(weight, bias))
    }
    pub fn norm(&mut self, prefix: &str, width: usize, eps: f64) -> Result<Norm> {
        Ok(Norm {
            weight: self.take(&format!("{prefix}.weight"), &[width])?,
            bias: self.take(&format!("{prefix}.bias"), &[width])?,
            eps,
        })
    }
    pub fn scalar(&mut self, name: &str) -> Result<f64> {
        let v = self
            .take(name, &[])?
            .to_dtype(DType::F32)?
            .to_scalar::<f32>()?;
        if !v.is_finite() {
            return Err(Error::IntegrityMismatch("nonfinite scalar".into()));
        }
        Ok(f64::from(v))
    }
    pub fn attention(
        &mut self,
        prefix: &str,
        width: usize,
        heads: usize,
    ) -> Result<MultiAttention> {
        let weight = self.take(&format!("{prefix}.in_proj_weight"), &[3 * width, width])?;
        let bias = self.take(&format!("{prefix}.in_proj_bias"), &[3 * width])?;
        let project = |index| -> Result<Linear> {
            Ok(Linear::new(
                weight.narrow(0, index * width, width)?,
                Some(bias.narrow(0, index * width, width)?),
            ))
        };
        Ok(MultiAttention {
            q: project(0)?,
            k: project(1)?,
            v: project(2)?,
            out: self.linear(&format!("{prefix}.out_proj"), width, width, true)?,
            heads,
        })
    }
    pub fn finish(self) -> Result<()> {
        if !self.tensors.is_empty() {
            return Err(Error::IntegrityMismatch("unconsumed model tensors".into()));
        }
        Ok(())
    }
}
fn validate_header(data: &[u8]) -> Result<()> {
    let prefix: [u8; 8] = data
        .get(..8)
        .ok_or_else(|| Error::IntegrityMismatch("short safetensors".into()))?
        .try_into()
        .map_err(|_| Error::IntegrityMismatch("safetensors length".into()))?;
    let len = usize::try_from(u64::from_le_bytes(prefix))
        .map_err(|_| Error::IntegrityMismatch("header overflow".into()))?;
    if len > 1024 * 1024 {
        return Err(Error::LimitExceeded("safetensors header".into()));
    }
    let end = len
        .checked_add(8)
        .ok_or_else(|| Error::IntegrityMismatch("header length".into()))?;
    let header = data
        .get(8..end)
        .ok_or_else(|| Error::IntegrityMismatch("truncated header".into()))?;
    // Duplicate names must fail before the safetensors map can overwrite them.
    crate::types::parse_json_with_collection_cap(header, 1024 * 1024, 2048)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_should_reject_corrupt_and_duplicate_headers() {
        assert!(validate_header(&[0; 3]).is_err());
        let header = br#"{"x":{},"x":{}}"#;
        let mut data = (header.len() as u64).to_le_bytes().to_vec();
        data.extend(header);
        assert!(validate_header(&data).is_err());
    }
}

#[cfg(test)]
impl Weights {
    pub(crate) fn fixture(data: &[u8]) -> Result<Self> {
        Ok(Self {
            tensors: candle_core::safetensors::load_buffer(data, &Device::Cpu)?,
        })
    }
}

#[cfg(test)]
mod header_limits {
    use super::*;
    #[test]
    fn test_should_accept_reviewed_header_counts_without_raising_request_limits() -> Result<()> {
        let entries: serde_json::Map<String, serde_json::Value> = (0..300)
            .map(|i| {
                (
                    format!("tensor.{i}"),
                    serde_json::json!({"dtype":"BF16","shape":[1],"data_offsets":[i*2,i*2+2]}),
                )
            })
            .collect();
        let header = serde_json::to_vec(&entries)?;
        let mut data = (header.len() as u64).to_le_bytes().to_vec();
        data.extend(&header);
        validate_header(&data)?;
        assert!(crate::types::parse_json(&header, 1024 * 1024).is_err());
        Ok(())
    }
}
