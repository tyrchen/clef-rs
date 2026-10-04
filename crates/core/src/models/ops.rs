//! Stable normalization, tiled attention, and rotary operations.
#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::many_single_char_names,
    reason = "reviewed model dimensions are bounded; tensor equations retain reference \
              mathematical names"
)]

use candle_core::{D, DType, Tensor};
use candle_nn::{Linear, Module};

use super::Control;
use crate::{Error, Result};

pub(crate) fn rms(x: &Tensor, weight: &Tensor, eps: f64, offset: bool) -> Result<Tensor> {
    measured!(
        "normalization",
        x.device(),
        rms_inner(x, weight, eps, offset)
    )
}
fn rms_inner(x: &Tensor, weight: &Tensor, eps: f64, offset: bool) -> Result<Tensor> {
    let f = x.to_dtype(DType::F32)?;
    let inverse = (f.sqr()?.mean_keepdim(D::Minus1)? + eps)?.sqrt()?.recip()?;
    let w = weight.to_dtype(DType::F32)?;
    let w = if offset { (w + 1.0)? } else { w };
    Ok(f.broadcast_mul(&inverse)?
        .broadcast_mul(&w)?
        .to_dtype(x.dtype())?)
}
pub(crate) fn normalize(x: &Tensor, eps: f64) -> Result<Tensor> {
    measured!("normalization", x.device(), normalize_inner(x, eps))
}
fn normalize_inner(x: &Tensor, eps: f64) -> Result<Tensor> {
    let f = x.to_dtype(DType::F32)?;
    let n = f
        .sqr()?
        .sum_keepdim(D::Minus1)?
        .sqrt()?
        .clamp(eps, f64::INFINITY)?;
    Ok(f.broadcast_div(&n)?.to_dtype(x.dtype())?)
}
#[derive(Debug)]
pub(crate) struct Norm {
    pub weight: Tensor,
    pub bias: Tensor,
    pub eps: f64,
}
impl Norm {
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let f = x.to_dtype(DType::F32)?;
        let mean = f.mean_keepdim(D::Minus1)?;
        let centered = f.broadcast_sub(&mean)?;
        let variance = centered.sqr()?.mean_keepdim(D::Minus1)?;
        let normalized = centered.broadcast_div(&(variance + self.eps)?.sqrt()?)?;
        Ok(normalized
            .to_dtype(x.dtype())?
            .broadcast_mul(&self.weight)?
            .broadcast_add(&self.bias)?)
    }
}
/// Queries, keys, values have [heads, tokens, width]. Online accumulation is F32.
///
/// When `causal` and the query block is shorter than the key block (`qn < kn`),
/// queries are positioned at the tail of the key sequence: query `i` may attend
/// to keys `j <= kn - qn + i`. This is the decode/KV-cache calling convention;
/// a full square causal mask needs `qn == kn`.
pub(crate) fn attention(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    causal: bool,
    control: &Control,
) -> Result<Tensor> {
    let (heads, qn, width) = q.dims3()?;
    let kn = k.dim(1)?;
    let vw = v.dim(2)?;
    let kv_heads = k.dim(0)?;
    if heads == 0
        || kv_heads == 0
        || heads % kv_heads != 0
        || width == 0
        || k.dim(2)? != width
        || v.dim(0)? != kv_heads
        || v.dim(1)? != kn
        || qn == 0
        || kn == 0
        || (causal && qn > kn)
    {
        return Err(Error::InvalidRequest("attention tensor shapes".into()));
    }
    // `kn - qn` cannot underflow: the guard above rejects `causal && qn > kn`,
    // and the offset is only read on the causal path.
    debug_assert!(!causal || qn <= kn);
    let groups = heads / kv_heads;
    let offset = if causal { kn - qn } else { 0 };
    let mut outputs = Vec::new();
    for qs in (0..qn).step_by(128) {
        control.check()?;
        let count = (qn - qs).min(128);
        // Convert only the resident query tile; the full Q never materializes
        // in F32.
        let query = q.narrow(1, qs, count)?.to_dtype(DType::F32)?.contiguous()?;
        let mut group_outputs = Vec::with_capacity(kv_heads);
        // Heads sharing one KV head are processed as a group on narrow views
        // of the original K/V: the cache is never expanded to `heads`.
        for g in 0..kv_heads {
            let queries = query.narrow(0, g * groups, groups)?;
            let mut maximum = Tensor::full(f32::NEG_INFINITY, (groups, count, 1), q.device())?;
            let mut denominator = Tensor::zeros((groups, count, 1), DType::F32, q.device())?;
            let mut accumulator = Tensor::zeros((groups, count, vw), DType::F32, q.device())?;
            for ks in (0..kn).step_by(256) {
                control.check()?;
                if causal && ks >= offset + qs + count {
                    break;
                }
                let keys = (kn - ks).min(256);
                // Convert only the resident key/value tiles.
                let keys_t = k
                    .narrow(0, g, 1)?
                    .narrow(1, ks, keys)?
                    .to_dtype(DType::F32)?;
                let values_t = v
                    .narrow(0, g, 1)?
                    .narrow(1, ks, keys)?
                    .to_dtype(DType::F32)?;
                let mut scores = (queries
                    .broadcast_matmul(&keys_t.transpose(1, 2)?.contiguous()?)?
                    / (width as f64).sqrt())?;
                // At most one key tile per query tile straddles the causal
                // boundary; earlier tiles are fully visible and need no mask.
                if causal && ks + keys > offset + qs + 1 {
                    let mask: Vec<f32> = (0..count)
                        .flat_map(|i| {
                            (0..keys).map(move |j| {
                                if ks + j > offset + qs + i {
                                    f32::NEG_INFINITY
                                } else {
                                    0.0
                                }
                            })
                        })
                        .collect();
                    scores = scores.broadcast_add(&Tensor::from_vec(
                        mask,
                        (1, count, keys),
                        q.device(),
                    )?)?;
                }
                let new_maximum = maximum.maximum(&scores.max_keepdim(2)?)?;
                let rescale = (&maximum - &new_maximum)?.exp()?;
                let probabilities = scores.broadcast_sub(&new_maximum)?.exp()?;
                accumulator = (accumulator.broadcast_mul(&rescale)?
                    + probabilities.broadcast_matmul(&values_t.contiguous()?)?)?;
                denominator =
                    (denominator.broadcast_mul(&rescale)? + probabilities.sum_keepdim(2)?)?;
                maximum = new_maximum;
            }
            group_outputs.push(accumulator.broadcast_div(&denominator)?);
        }
        outputs.push(Tensor::cat(&group_outputs, 0)?);
    }
    Ok(Tensor::cat(&outputs, 1)?)
}
/// Rotary embedding with per-axis positions (mRoPE).
///
/// `inv_freq[i]` is the inverse frequency for pair `i`
/// (`theta.powf(-(2 * i) / dim)`); it depends only on the model config, so it
/// is precomputed once at load instead of once per token.
pub(crate) fn rotary(
    x: &Tensor,
    positions: &[[usize; 3]],
    inv_freq: &[f32],
    sections: [usize; 3],
) -> Result<Tensor> {
    let (tokens, _, width) = x.dims3()?;
    let half = inv_freq.len();
    let dim = half * 2;
    let mut angles = Vec::with_capacity(tokens * dim);
    for p in positions {
        let row: Vec<f32> = (0..half)
            .map(|i| {
                let axis = if i % 3 == 1 && i < sections[1] * 3 {
                    1
                } else if i % 3 == 2 && i < sections[2] * 3 {
                    2
                } else {
                    0
                };
                p.get(axis).copied().unwrap_or_default() as f32 * inv_freq[i]
            })
            .collect();
        angles.extend(row.iter().copied());
        angles.extend(row);
    }
    let angles = Tensor::from_vec(angles, (tokens, 1, dim), x.device())?;
    let rotation = x.narrow(2, 0, dim)?.to_dtype(DType::F32)?;
    let rotated = Tensor::cat(
        &[
            rotation.narrow(2, half, half)?.neg()?,
            rotation.narrow(2, 0, half)?,
        ],
        2,
    )?;
    let rotation = (rotation.broadcast_mul(&angles.cos()?)?
        + rotated.broadcast_mul(&angles.sin()?)?)?
    .to_dtype(x.dtype())?;
    if dim == width {
        Ok(rotation)
    } else {
        Ok(Tensor::cat(&[rotation, x.narrow(2, dim, width - dim)?], 2)?)
    }
}
#[derive(Debug)]
pub(crate) struct MultiAttention {
    pub q: Linear,
    pub k: Linear,
    pub v: Linear,
    pub out: Linear,
    pub heads: usize,
}
impl MultiAttention {
    pub fn forward(&self, queries: &Tensor, memory: &Tensor, control: &Control) -> Result<Tensor> {
        let width = queries.dim(1)?;
        // No `.contiguous()` after the transposes: `attention` converts each
        // tile itself, so materializing the full [heads, tokens, width]
        // tensors here would only add a pass.
        let q = self
            .q
            .forward(queries)?
            .reshape((queries.dim(0)?, self.heads, width / self.heads))?
            .transpose(0, 1)?;
        let k = self
            .k
            .forward(memory)?
            .reshape((memory.dim(0)?, self.heads, width / self.heads))?
            .transpose(0, 1)?;
        let v = self
            .v
            .forward(memory)?
            .reshape((memory.dim(0)?, self.heads, width / self.heads))?
            .transpose(0, 1)?;
        let out = attention(&q, &k, &v, false, control)?
            .to_dtype(queries.dtype())?
            .transpose(0, 1)?
            .contiguous()?
            .reshape((queries.dim(0)?, width))?;
        Ok(self.out.forward(&out)?)
    }
}

#[cfg(test)]
mod tests {
    use std::{
        sync::{Arc, atomic::AtomicBool},
        time::{Duration, Instant},
    };

    use candle_core::Device;

    use super::*;
    #[test]
    fn test_should_match_dense_causal_attention_across_tiles() -> Result<()> {
        let d = Device::Cpu;
        let q = Tensor::from_vec(
            (0..300 * 4).map(|i| (i as f32 * 0.013).sin()).collect(),
            (1, 300, 4),
            &d,
        )?;
        let c = Control {
            cancel: Arc::new(AtomicBool::new(false)),
            deadline: Instant::now() + Duration::from_secs(60),
        };
        let tiled = attention(&q, &q, &q, true, &c)?;
        let mask: Vec<f32> = (0..300)
            .flat_map(|i| (0..300).map(move |j| if j > i { f32::NEG_INFINITY } else { 0. }))
            .collect();
        let scores = (q.matmul(&q.transpose(1, 2)?.contiguous()?)? / 2.)?
            .broadcast_add(&Tensor::from_vec(mask, (1, 300, 300), &d)?)?;
        let dense = candle_nn::ops::softmax_last_dim(&scores)?.matmul(&q)?;
        let error = (&tiled - &dense)?
            .abs()?
            .flatten_all()?
            .max(0)?
            .to_scalar::<f32>()?;
        assert!(error < 1e-5, "error {error}");
        Ok(())
    }
    #[test]
    fn test_should_match_dense_attention_with_grouped_query_heads() -> Result<()> {
        let d = Device::Cpu;
        let (heads, kv_heads, tokens, width) = (4, 2, 300, 8);
        let q = Tensor::from_vec(
            (0..heads * tokens * width)
                .map(|i| (i as f32 * 0.013).sin())
                .collect(),
            (heads, tokens, width),
            &d,
        )?;
        let kv = Tensor::from_vec(
            (0..kv_heads * tokens * width)
                .map(|i| (i as f32 * 0.031).cos())
                .collect(),
            (kv_heads, tokens, width),
            &d,
        )?;
        let c = Control {
            cancel: Arc::new(AtomicBool::new(false)),
            deadline: Instant::now() + Duration::from_secs(60),
        };
        for causal in [false, true] {
            let tiled = attention(&q, &kv, &kv, causal, &c)?;
            // Dense reference: expand the grouped heads, then plain softmax.
            let groups = heads / kv_heads;
            let ids: Vec<u32> = (0..heads).map(|i| (i / groups) as u32).collect();
            let ids = Tensor::new(ids, &d)?;
            let k = kv.index_select(&ids, 0)?;
            let v = kv.index_select(&ids, 0)?;
            let mut scores =
                (q.matmul(&k.transpose(1, 2)?.contiguous()?)? / (width as f64).sqrt())?;
            if causal {
                let mask: Vec<f32> = (0..tokens)
                    .flat_map(|i| {
                        (0..tokens).map(move |j| if j > i { f32::NEG_INFINITY } else { 0.0 })
                    })
                    .collect();
                scores = scores.broadcast_add(&Tensor::from_vec(mask, (1, tokens, tokens), &d)?)?;
            }
            let dense = candle_nn::ops::softmax_last_dim(&scores)?.matmul(&v)?;
            let error = (&tiled - &dense)?
                .abs()?
                .flatten_all()?
                .max(0)?
                .to_scalar::<f32>()?;
            assert!(error < 1e-4, "causal {causal}: error {error}");
        }
        Ok(())
    }
}
