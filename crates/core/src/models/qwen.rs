//! Qwen3.5 hybrid prefill graph; no decoding cache is retained.
#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::many_single_char_names,
    reason = "reviewed model dimensions are bounded; tensor equations retain reference \
              mathematical names"
)]

use candle_core::{D, DType, Tensor};
use candle_nn::{
    Module,
    ops::{sigmoid, silu},
};
use serde::Deserialize;

#[cfg(all(feature = "metal", target_os = "macos"))]
use super::metal::{AttentionKernel, DeltaKernel};
use super::{
    Control,
    ops::{attention, rms, rotary},
    projection::{Projection, Projections},
    weights::Weights,
};
use crate::{Error, Result};

#[derive(Debug, Clone, Deserialize)]
pub(crate) struct ModelConfig {
    pub model_type: String,
    pub text_config: TextConfig,
    #[cfg(feature = "vision")]
    pub vision_config: super::vision::VisionConfig,
}
#[derive(Debug, Clone, Deserialize)]
pub(crate) struct TextConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub vocab_size: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub rms_norm_eps: f64,
    pub layer_types: Vec<String>,
    pub linear_num_key_heads: usize,
    pub linear_num_value_heads: usize,
    pub linear_key_head_dim: usize,
    pub linear_value_head_dim: usize,
    pub linear_conv_kernel_dim: usize,
    pub rope_parameters: RopeConfig,
}
#[derive(Debug, Clone, Deserialize)]
pub(crate) struct RopeConfig {
    pub rope_theta: f64,
    pub partial_rotary_factor: f64,
    pub mrope_section: [usize; 3],
}
impl ModelConfig {
    #[allow(
        clippy::float_cmp,
        reason = "architecture fingerprints require exact reference constants"
    )]
    pub fn validate(&self) -> Result<()> {
        let c = &self.text_config;
        let flash = c.hidden_size == 4096
            && c.num_hidden_layers == 32
            && c.intermediate_size == 12288
            && c.num_attention_heads == 16
            && c.linear_num_value_heads == 32;
        if self.model_type != "qwen3_5"
            || !flash
            || c.vocab_size != 248_320
            || c.num_key_value_heads != 4
            || c.head_dim != 256
            || c.linear_num_key_heads != 16
            || c.linear_key_head_dim != 128
            || c.linear_value_head_dim != 128
            || c.linear_conv_kernel_dim != 4
            || c.rms_norm_eps != 1e-6
            || c.rope_parameters.rope_theta != 10_000_000.
            || c.rope_parameters.partial_rotary_factor != 0.25
            || c.rope_parameters.mrope_section != [11, 11, 10]
            || c.layer_types.len() != c.num_hidden_layers
            || c.layer_types.iter().enumerate().any(|(i, t)| {
                t != if i % 4 == 3 {
                    "full_attention"
                } else {
                    "linear_attention"
                }
            })
        {
            return Err(Error::UnsupportedArchitecture);
        }
        Ok(())
    }
}
#[derive(Debug)]
struct FullAttention {
    #[cfg(all(feature = "metal", target_os = "macos"))]
    kernel: Option<AttentionKernel>,
    q: Projection,
    k: Projection,
    v: Projection,
    out: Projection,
    qn: Tensor,
    kn: Tensor,
}
#[derive(Debug)]
struct Delta {
    #[cfg(all(feature = "metal", target_os = "macos"))]
    kernel: Option<DeltaKernel>,
    qkv: Projection,
    z: Projection,
    a: Projection,
    b: Projection,
    out: Projection,
    conv: Tensor,
    dt: Tensor,
    a_log: Tensor,
    norm: Tensor,
}
#[derive(Debug)]
enum Mixer {
    Full(FullAttention),
    Linear(Delta),
}
#[derive(Debug)]
struct Layer {
    input_norm: Tensor,
    post_norm: Tensor,
    mixer: Mixer,
    gate: Projection,
    up: Projection,
    down: Projection,
}
#[derive(Debug)]
pub(crate) struct Backbone {
    embeddings: Tensor,
    pub output_embeddings: Tensor,
    norm: Tensor,
    layers: Vec<Layer>,
    config: TextConfig,
}
impl Backbone {
    pub fn load(w: &mut Weights, c: TextConfig) -> Result<Self> {
        let h = c.hidden_size;
        let embeddings = w.take(
            "model.language_model.embed_tokens.weight",
            &[c.vocab_size, h],
        )?;
        #[cfg(all(feature = "metal", target_os = "macos"))]
        let projections = Projections::new(embeddings.device(), embeddings.dtype())?;
        #[cfg(not(all(feature = "metal", target_os = "macos")))]
        let projections = Projections::default();
        #[cfg(all(feature = "metal", target_os = "macos"))]
        let delta_kernel = DeltaKernel::new(
            embeddings.device(),
            c.linear_key_head_dim,
            c.linear_value_head_dim,
        )?;
        #[cfg(all(feature = "metal", target_os = "macos"))]
        let attention_kernel = AttentionKernel::new(embeddings.device(), c.head_dim)?;
        let output_embeddings = w.take("lm_head.weight", &[c.vocab_size, h])?;
        let norm = w.take("model.language_model.norm.weight", &[h])?;
        let mut layers = Vec::with_capacity(c.num_hidden_layers);
        for (index, kind) in c.layer_types.iter().enumerate() {
            let p = format!("model.language_model.layers.{index}");
            let mixer = if kind == "full_attention" {
                Mixer::Full(FullAttention::load(
                    w,
                    &projections,
                    &c,
                    &p,
                    #[cfg(all(feature = "metal", target_os = "macos"))]
                    attention_kernel.clone(),
                )?)
            } else {
                Mixer::Linear(Delta::load(
                    w,
                    &projections,
                    &c,
                    &p,
                    #[cfg(all(feature = "metal", target_os = "macos"))]
                    delta_kernel.clone(),
                )?)
            };
            layers.push(Layer {
                mixer,
                input_norm: w.take(&format!("{p}.input_layernorm.weight"), &[h])?,
                post_norm: w.take(&format!("{p}.post_attention_layernorm.weight"), &[h])?,
                gate: projections.load(
                    w,
                    &format!("{p}.mlp.gate_proj"),
                    h,
                    c.intermediate_size,
                    false,
                )?,
                up: projections.load(
                    w,
                    &format!("{p}.mlp.up_proj"),
                    h,
                    c.intermediate_size,
                    false,
                )?,
                down: projections.load(
                    w,
                    &format!("{p}.mlp.down_proj"),
                    c.intermediate_size,
                    h,
                    false,
                )?,
            });
        }
        Ok(Self {
            embeddings,
            output_embeddings,
            norm,
            layers,
            config: c,
        })
    }
    pub fn forward(&self, ids: &[u32], control: &Control) -> Result<Tensor> {
        control.check()?;
        if ids.iter().any(|id| *id as usize >= self.config.vocab_size) {
            return Err(Error::InvalidRequest("token outside vocabulary".into()));
        }
        let indices = Tensor::new(ids, self.embeddings.device())?;
        let hidden = self.embeddings.index_select(&indices, 0)?;
        let positions: Vec<_> = (0..ids.len()).map(|i| [i, i, i]).collect();
        self.forward_embedded(hidden, &positions, control)
    }
    #[cfg(feature = "vision")]
    pub fn embeddings(&self, ids: &[u32]) -> Result<Tensor> {
        Ok(self
            .embeddings
            .index_select(&Tensor::new(ids, self.embeddings.device())?, 0)?)
    }
    pub fn forward_embedded(
        &self,
        mut hidden: Tensor,
        positions: &[[usize; 3]],
        control: &Control,
    ) -> Result<Tensor> {
        // Preserve residual additions in F32 while the large projection weights
        // and their matrix multiplications use the selected profile precision.
        hidden = hidden.to_dtype(DType::F32)?;
        for layer in &self.layers {
            control.check()?;
            let input = rms(&hidden, &layer.input_norm, self.config.rms_norm_eps, true)?
                .to_dtype(layer.gate.weight().dtype())?;
            let mixed = match &layer.mixer {
                Mixer::Full(a) => a.forward(&input, &self.config, positions, control)?,
                Mixer::Linear(a) => a.forward(&input, &self.config, control)?,
            };
            hidden = (hidden + mixed.to_dtype(DType::F32)?)?;
            let norm = rms(&hidden, &layer.post_norm, self.config.rms_norm_eps, true)?
                .to_dtype(layer.gate.weight().dtype())?;
            let ff = layer
                .down
                .forward(&(silu(&layer.gate.forward(&norm)?)? * layer.up.forward(&norm)?)?)?;
            hidden = (hidden + ff.to_dtype(DType::F32)?)?;
        }
        rms(&hidden, &self.norm, self.config.rms_norm_eps, true)
    }
}
impl FullAttention {
    fn load(
        w: &mut Weights,
        projections: &Projections,
        c: &TextConfig,
        p: &str,
        #[cfg(all(feature = "metal", target_os = "macos"))] kernel: Option<AttentionKernel>,
    ) -> Result<Self> {
        let h = c.hidden_size;
        let n = c.num_attention_heads * c.head_dim;
        let kv = c.num_key_value_heads * c.head_dim;
        Ok(Self {
            #[cfg(all(feature = "metal", target_os = "macos"))]
            kernel,
            q: projections.load(w, &format!("{p}.self_attn.q_proj"), h, n * 2, false)?,
            k: projections.load(w, &format!("{p}.self_attn.k_proj"), h, kv, false)?,
            v: projections.load(w, &format!("{p}.self_attn.v_proj"), h, kv, false)?,
            out: projections.load(w, &format!("{p}.self_attn.o_proj"), n, h, false)?,
            qn: w.take(&format!("{p}.self_attn.q_norm.weight"), &[c.head_dim])?,
            kn: w.take(&format!("{p}.self_attn.k_norm.weight"), &[c.head_dim])?,
        })
    }
    fn forward(
        &self,
        x: &Tensor,
        c: &TextConfig,
        positions: &[[usize; 3]],
        control: &Control,
    ) -> Result<Tensor> {
        let t = x.dim(0)?;
        let d = c.head_dim;
        let heads = c.num_attention_heads;
        let projected = self.q.forward(x)?.reshape((t, heads, 2 * d))?;
        let gate = projected
            .narrow(2, d, d)?
            .contiguous()?
            .reshape((t, heads * d))?;
        let q = rms(&projected.narrow(2, 0, d)?, &self.qn, c.rms_norm_eps, true)?;
        let k = rms(
            &self.k.forward(x)?.reshape((t, c.num_key_value_heads, d))?,
            &self.kn,
            c.rms_norm_eps,
            true,
        )?;
        let rotary_dim = (d as f64 * c.rope_parameters.partial_rotary_factor) as usize;
        let q = rotary(
            &q,
            positions,
            rotary_dim,
            c.rope_parameters.rope_theta,
            c.rope_parameters.mrope_section,
        )?
        .transpose(0, 1)?
        .contiguous()?;
        let k = rotary(
            &k,
            positions,
            rotary_dim,
            c.rope_parameters.rope_theta,
            c.rope_parameters.mrope_section,
        )?
        .transpose(0, 1)?
        .contiguous()?;
        let v = self
            .v
            .forward(x)?
            .reshape((t, c.num_key_value_heads, d))?
            .transpose(0, 1)?
            .contiguous()?;
        #[cfg(all(feature = "metal", target_os = "macos"))]
        let attended = match &self.kernel {
            Some(kernel) => kernel.forward(&q, &k, &v, control)?,
            None => attention(&q, &k, &v, true, control)?,
        };
        #[cfg(not(all(feature = "metal", target_os = "macos")))]
        let attended = attention(&q, &k, &v, true, control)?;
        let out = attended
            .to_dtype(x.dtype())?
            .transpose(0, 1)?
            .contiguous()?
            .reshape((t, heads * d))?;
        Ok(self.out.forward(&(out * sigmoid(&gate)?)?)?)
    }
}
impl Delta {
    fn load(
        w: &mut Weights,
        projections: &Projections,
        c: &TextConfig,
        p: &str,
        #[cfg(all(feature = "metal", target_os = "macos"))] kernel: Option<DeltaKernel>,
    ) -> Result<Self> {
        let h = c.hidden_size;
        let key = c.linear_num_key_heads * c.linear_key_head_dim;
        let value = c.linear_num_value_heads * c.linear_value_head_dim;
        Ok(Self {
            #[cfg(all(feature = "metal", target_os = "macos"))]
            kernel,
            qkv: projections.load(
                w,
                &format!("{p}.linear_attn.in_proj_qkv"),
                h,
                2 * key + value,
                false,
            )?,
            z: projections.load(w, &format!("{p}.linear_attn.in_proj_z"), h, value, false)?,
            a: projections.load(
                w,
                &format!("{p}.linear_attn.in_proj_a"),
                h,
                c.linear_num_value_heads,
                false,
            )?,
            b: projections.load(
                w,
                &format!("{p}.linear_attn.in_proj_b"),
                h,
                c.linear_num_value_heads,
                false,
            )?,
            out: projections.load(w, &format!("{p}.linear_attn.out_proj"), value, h, false)?,
            conv: w.take(
                &format!("{p}.linear_attn.conv1d.weight"),
                &[2 * key + value, 1, c.linear_conv_kernel_dim],
            )?,
            dt: w.take(
                &format!("{p}.linear_attn.dt_bias"),
                &[c.linear_num_value_heads],
            )?,
            a_log: w.take(
                &format!("{p}.linear_attn.A_log"),
                &[c.linear_num_value_heads],
            )?,
            norm: w.take(
                &format!("{p}.linear_attn.norm.weight"),
                &[c.linear_value_head_dim],
            )?,
        })
    }
    fn forward(&self, x: &Tensor, c: &TextConfig, control: &Control) -> Result<Tensor> {
        let t = x.dim(0)?;
        let heads = c.linear_num_value_heads;
        let kd = c.linear_key_head_dim;
        let vd = c.linear_value_head_dim;
        let key = c.linear_num_key_heads * kd;
        let value = heads * vd;
        let width = 2 * key + value;
        let projected = self.qkv.forward(x)?.to_dtype(DType::F32)?;
        // Explicit causal depthwise convolution: oldest kernel tap sees t-(K-1).
        let mut convolved = Tensor::zeros((t, width), DType::F32, x.device())?;
        for tap in 0..c.linear_conv_kernel_dim {
            let delay = c.linear_conv_kernel_dim - 1 - tap;
            if delay >= t {
                continue;
            }
            let weight = self
                .conv
                .narrow(2, tap, 1)?
                .reshape((1, width))?
                .to_dtype(DType::F32)?;
            let part = projected.narrow(0, 0, t - delay)?.broadcast_mul(&weight)?;
            let part = if delay == 0 {
                part
            } else {
                Tensor::cat(
                    &[Tensor::zeros((delay, width), DType::F32, x.device())?, part],
                    0,
                )?
            };
            convolved = (convolved + part)?;
        }
        let mixed = silu(&convolved)?;
        let repeat: Vec<u32> = (0..heads)
            .map(|i| (i / (heads / c.linear_num_key_heads)) as u32)
            .collect();
        let repeat = Tensor::new(repeat, x.device())?;
        let q = mixed
            .narrow(1, 0, key)?
            .reshape((t, c.linear_num_key_heads, kd))?
            .index_select(&repeat, 1)?;
        let k = mixed
            .narrow(1, key, key)?
            .reshape((t, c.linear_num_key_heads, kd))?
            .index_select(&repeat, 1)?;
        let v = mixed.narrow(1, 2 * key, value)?.reshape((t, heads, vd))?;
        let a = self
            .a
            .forward(x)?
            .to_dtype(DType::F32)?
            .broadcast_add(&self.dt.to_dtype(DType::F32)?)?;
        // Stable softplus, preserving the reference's float32 decay path.
        let softplus = (a.clamp(0., f64::INFINITY)? + ((a.abs()?.neg()?.exp()? + 1.)?.log()?))?;
        let g = softplus.broadcast_mul(&self.a_log.to_dtype(DType::F32)?.exp()?.neg()?)?;
        let beta = sigmoid(&self.b.forward(x)?.to_dtype(DType::F32)?)?;
        let out = delta_recurrence(
            &q,
            &k,
            &v,
            &g,
            &beta,
            control,
            #[cfg(all(feature = "metal", target_os = "macos"))]
            self.kernel.as_ref(),
        )?
        .to_dtype(x.dtype())?;
        let z = self.z.forward(x)?.reshape((t, heads, vd))?;
        let gated = (rms(&out, &self.norm, c.rms_norm_eps, false)?.to_dtype(DType::F32)?
            * silu(&z.to_dtype(DType::F32)?)?)?
        .to_dtype(x.dtype())?
        .reshape((t, value))?;
        Ok(self.out.forward(&gated)?)
    }
}
pub(super) fn delta_recurrence(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    g: &Tensor,
    beta: &Tensor,
    control: &Control,
    #[cfg(all(feature = "metal", target_os = "macos"))] kernel: Option<&DeltaKernel>,
) -> Result<Tensor> {
    fn l2(x: &Tensor) -> Result<Tensor> {
        let n = (x.sqr()?.sum_keepdim(D::Minus1)? + 1e-6)?.sqrt()?;
        Ok(x.broadcast_div(&n)?)
    }
    let (t, heads, kd) = q.dims3()?;
    let vd = v.dim(2)?;
    // Normalize and accumulate in F32, including the mixed F16 profile.
    let q = (l2(&q.to_dtype(DType::F32)?)? / (kd as f64).sqrt())?;
    let k = l2(&k.to_dtype(DType::F32)?)?;
    let v = v.to_dtype(DType::F32)?;
    let beta = beta.to_dtype(DType::F32)?;
    let g = g.to_dtype(DType::F32)?;
    #[cfg(all(feature = "metal", target_os = "macos"))]
    if let Some(kernel) = kernel {
        return kernel.forward(&q, &k, &v, &g, &beta, control);
    }
    let mut state = Tensor::zeros((heads, kd, vd), DType::F32, q.device())?;
    let mut outputs = Vec::with_capacity(t);
    for i in 0..t {
        control.check()?;
        let key = k.narrow(0, i, 1)?.squeeze(0)?.unsqueeze(2)?;
        let query = q.narrow(0, i, 1)?.squeeze(0)?.unsqueeze(2)?;
        state = state.broadcast_mul(&g.narrow(0, i, 1)?.reshape((heads, 1, 1))?.exp()?)?;
        let memory = state.broadcast_mul(&key)?.sum(1)?;
        let delta = (v.narrow(0, i, 1)?.squeeze(0)? - memory)?
            .broadcast_mul(&beta.narrow(0, i, 1)?.reshape((heads, 1))?)?;
        state = (state + key.broadcast_mul(&delta.unsqueeze(1)?)?)?;
        outputs.push(state.broadcast_mul(&query)?.sum(1)?.unsqueeze(0)?);
    }
    Ok(Tensor::cat(&outputs, 0)?)
}
