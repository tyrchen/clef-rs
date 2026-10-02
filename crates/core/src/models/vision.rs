//! Qwen3.5 vision patch embedding, spatial interpolation, rotary attention and merger.
#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::many_single_char_names,
    reason = "reviewed vision dimensions are bounded; equations follow the reference"
)]
use candle_core::{DType, Device, Tensor};
use candle_nn::{Linear, Module};
use serde::Deserialize;

use super::{
    Control,
    ops::{Norm, attention},
    weights::Weights,
};
use crate::{Error, Result, media::PreparedImage};

#[derive(Debug, Clone, Deserialize)]
pub(crate) struct VisionConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub depth: usize,
    pub num_heads: usize,
    pub num_position_embeddings: usize,
    pub out_hidden_size: usize,
    pub patch_size: usize,
    pub temporal_patch_size: usize,
    pub spatial_merge_size: usize,
    pub in_channels: usize,
}
#[derive(Debug)]
struct Block {
    norm1: Norm,
    norm2: Norm,
    qkv: Linear,
    out: Linear,
    ff1: Linear,
    ff2: Linear,
}
#[derive(Debug)]
pub(crate) struct Vision {
    patch: Linear,
    position: Tensor,
    blocks: Vec<Block>,
    merger_norm: Norm,
    merger1: Linear,
    merger2: Linear,
    config: VisionConfig,
}
impl Vision {
    pub fn load(w: &mut Weights, c: VisionConfig) -> Result<Self> {
        let h = c.hidden_size;
        let input = c.in_channels * c.temporal_patch_size * c.patch_size * c.patch_size;
        let weight = w
            .take(
                "model.visual.patch_embed.proj.weight",
                &[
                    h,
                    c.in_channels,
                    c.temporal_patch_size,
                    c.patch_size,
                    c.patch_size,
                ],
            )?
            .reshape((h, input))?;
        let patch = Linear::new(
            weight,
            Some(w.take("model.visual.patch_embed.proj.bias", &[h])?),
        );
        let position = w.take(
            "model.visual.pos_embed.weight",
            &[c.num_position_embeddings, h],
        )?;
        let mut blocks = Vec::new();
        for i in 0..c.depth {
            let p = format!("model.visual.blocks.{i}");
            blocks.push(Block {
                norm1: w.norm(&format!("{p}.norm1"), h, 1e-6)?,
                norm2: w.norm(&format!("{p}.norm2"), h, 1e-6)?,
                qkv: w.linear(&format!("{p}.attn.qkv"), h, 3 * h, true)?,
                out: w.linear(&format!("{p}.attn.proj"), h, h, true)?,
                ff1: w.linear(&format!("{p}.mlp.linear_fc1"), h, c.intermediate_size, true)?,
                ff2: w.linear(&format!("{p}.mlp.linear_fc2"), c.intermediate_size, h, true)?,
            });
        }
        let merged = h * c.spatial_merge_size * c.spatial_merge_size;
        Ok(Self {
            patch,
            position,
            blocks,
            merger_norm: w.norm("model.visual.merger.norm", h, 1e-6)?,
            merger1: w.linear("model.visual.merger.linear_fc1", merged, merged, true)?,
            merger2: w.linear(
                "model.visual.merger.linear_fc2",
                merged,
                c.out_hidden_size,
                true,
            )?,
            config: c,
        })
    }
    pub fn forward(&self, image: &PreparedImage, control: &Control) -> Result<Tensor> {
        control.check()?;
        let c = &self.config;
        let t = image.coordinates.len();
        let h = c.hidden_size;
        let d = h / c.num_heads;
        let device = self.position.device();
        let patches = Tensor::new(image.patches.as_slice(), device)?
            .reshape((
                t,
                c.in_channels * c.temporal_patch_size * c.patch_size * c.patch_size,
            ))?
            .to_dtype(self.position.dtype())?;
        let mut hidden = self.patch.forward(&patches)?;
        let side = (c.num_position_embeddings as f64).sqrt() as usize;
        let gh = *image
            .grid
            .get(1)
            .ok_or_else(|| Error::InvalidRequest("vision grid".into()))?;
        let gw = *image
            .grid
            .get(2)
            .ok_or_else(|| Error::InvalidRequest("vision grid".into()))?;
        let mut positional = Tensor::zeros((t, h), self.position.dtype(), device)?;
        for corner in 0..4 {
            let mut ids = Vec::with_capacity(t);
            let mut weights = Vec::with_capacity(t);
            for &[row, col] in &image.coordinates {
                let rh = row as f32 * (side - 1) as f32 / (gh - 1) as f32;
                let rw = col as f32 * (side - 1) as f32 / (gw - 1) as f32;
                let floor_h = rh.floor() as usize;
                let floor_w = rw.floor() as usize;
                let y = if corner >= 2 {
                    (floor_h + 1).min(side - 1)
                } else {
                    floor_h
                };
                let x = if corner % 2 == 1 {
                    (floor_w + 1).min(side - 1)
                } else {
                    floor_w
                };
                ids.push((y * side + x) as u32);
                weights.push(
                    if corner >= 2 {
                        rh - floor_h as f32
                    } else {
                        1. - (rh - floor_h as f32)
                    } * if corner % 2 == 1 {
                        rw - floor_w as f32
                    } else {
                        1. - (rw - floor_w as f32)
                    },
                );
            }
            positional = (positional
                + self
                    .position
                    .index_select(&Tensor::new(ids, device)?, 0)?
                    .broadcast_mul(
                        &Tensor::new(weights, device)?
                            .reshape((t, 1))?
                            .to_dtype(self.position.dtype())?,
                    )?)?;
        }
        hidden = (hidden + positional)?;
        let (cos, sin) = vision_angles(&image.coordinates, d, device)?;
        for block in &self.blocks {
            control.check()?;
            let qkv = block
                .qkv
                .forward(&block.norm1.forward(&hidden)?)?
                .reshape((t, 3, c.num_heads, d))?;
            let q = vision_rotate(&qkv.narrow(1, 0, 1)?.squeeze(1)?, &cos, &sin)?;
            let k = vision_rotate(&qkv.narrow(1, 1, 1)?.squeeze(1)?, &cos, &sin)?;
            let v = qkv.narrow(1, 2, 1)?.squeeze(1)?;
            let attended = attention(
                &q.transpose(0, 1)?.contiguous()?,
                &k.transpose(0, 1)?.contiguous()?,
                &v.transpose(0, 1)?.contiguous()?,
                false,
                control,
            )?
            .to_dtype(hidden.dtype())?
            .transpose(0, 1)?
            .contiguous()?
            .reshape((t, h))?;
            hidden = (hidden + block.out.forward(&attended)?)?;
            let ff = block
                .ff2
                .forward(&block.ff1.forward(&block.norm2.forward(&hidden)?)?.gelu()?)?;
            hidden = (hidden + ff)?;
        }
        let merged = self.merger_norm.forward(&hidden)?.reshape((
            t / (c.spatial_merge_size * c.spatial_merge_size),
            h * c.spatial_merge_size * c.spatial_merge_size,
        ))?;
        Ok(self
            .merger2
            .forward(&self.merger1.forward(&merged)?.gelu_erf()?)?)
    }
}
fn vision_angles(
    coordinates: &[[usize; 2]],
    d: usize,
    device: &Device,
) -> Result<(Tensor, Tensor)> {
    let half = d / 2;
    let mut angles = Vec::with_capacity(coordinates.len() * d);
    for p in coordinates {
        let mut row = Vec::with_capacity(half);
        for coordinate in p {
            for i in (0..half).step_by(2) {
                row.push(*coordinate as f32 * 10000_f32.powf(-(i as f32) / half as f32));
            }
        }
        angles.extend(row.iter().copied());
        angles.extend(row);
    }
    let angles = Tensor::new(angles, device)?.reshape((coordinates.len(), 1, d))?;
    Ok((angles.cos()?, angles.sin()?))
}
fn vision_rotate(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    let x32 = x.to_dtype(DType::F32)?;
    let d = x.dim(2)?;
    let half = d / 2;
    let rotated = Tensor::cat(
        &[x32.narrow(2, half, half)?.neg()?, x32.narrow(2, 0, half)?],
        2,
    )?;
    Ok((x32.broadcast_mul(cos)? + rotated.broadcast_mul(sin)?)?.to_dtype(x.dtype())?)
}

#[cfg(test)]
mod tests {
    use std::{
        sync::{Arc, atomic::AtomicBool},
        time::{Duration, Instant},
    };

    use bytes::Bytes;

    use super::*;
    use crate::media::Image;
    #[test]
    fn test_should_match_reference_image_patches_and_vision_tower() -> Result<()> {
        let config: VisionConfig = serde_json::from_slice(include_bytes!(
            "../../fixtures/synthetic/vision-config.json"
        ))?;
        let mut weights = Weights::fixture(include_bytes!(
            "../../fixtures/synthetic/vision.safetensors"
        ))?;
        let vision = Vision::load(&mut weights, config)?;
        weights.finish()?;
        for (width, height, pixels, expected) in [
            (
                256,
                256,
                include_bytes!("../../fixtures/synthetic/image-256-256.rgb").as_slice(),
                include_bytes!("../../fixtures/synthetic/image-256-256.safetensors").as_slice(),
            ),
            (
                61,
                37,
                include_bytes!("../../fixtures/synthetic/image-37-61.rgb").as_slice(),
                include_bytes!("../../fixtures/synthetic/image-37-61.safetensors").as_slice(),
            ),
        ] {
            let image =
                Image::from_rgb(width, height, Bytes::copy_from_slice(pixels))?.prepare()?;
            let expected = candle_core::safetensors::load_buffer(expected, &Device::Cpu)?;
            let patches = Tensor::new(image.patches.as_slice(), &Device::Cpu)?
                .reshape((image.coordinates.len(), 1536))?;
            let difference = (&patches - expected.get("patches").ok_or(Error::ArtifactMissing)?)?
                .abs()?
                .flatten_all()?
                .max(0)?
                .to_scalar::<f32>()?;
            assert!(
                difference < 1e-5,
                "pixel/patch error {difference} for {width}x{height}"
            );
            let control = Control {
                cancel: Arc::new(AtomicBool::new(false)),
                deadline: Instant::now() + Duration::from_secs(60),
            };
            let output = vision.forward(&image, &control)?;
            let reference = expected.get("output").ok_or(Error::ArtifactMissing)?;
            let difference = (&output - reference)?
                .abs()?
                .flatten_all()?
                .max(0)?
                .to_scalar::<f32>()?;
            assert!(
                difference < 1e-5,
                "vision error {difference} for {width}x{height}"
            );
        }
        Ok(())
    }
}
