//! Qwen3.5 hybrid prefill with immutable, exact-prefix continuation states.
#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::many_single_char_names,
    reason = "reviewed model dimensions are bounded; tensor equations retain reference \
              mathematical names"
)]

use std::fmt::{self, Debug, Formatter};

use candle_core::{D, DType, Tensor};
use candle_nn::{
    Module,
    ops::{sigmoid, silu},
};
use serde::Deserialize;

use super::{
    Control, MAX_SEQUENCE_TOKENS,
    normalization::{Norms, Rms},
    ops::{attention, rotary},
    projection::{Projection, Projections},
    weights::Weights,
};
#[cfg(all(feature = "metal", target_os = "macos"))]
use super::{
    metal::{AttentionKernel, DeltaKernel, compact_copy},
    pointwise::ConvKernel,
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
    qn: Rms,
    kn: Rms,
    /// Rotary inverse frequencies, precomputed once at load.
    inv_freq: Vec<f32>,
}
#[derive(Debug)]
struct Delta {
    #[cfg(all(feature = "metal", target_os = "macos"))]
    convolution: Option<ConvKernel>,
    #[cfg(all(feature = "metal", target_os = "macos"))]
    kernel: Option<DeltaKernel>,
    qkv: Projection,
    z: Projection,
    /// Fused `[h -> 2 * heads]` decay/beta projection; split after the GEMM.
    ab: Projection,
    out: Projection,
    conv: Tensor,
    dt: Tensor,
    a_log: Tensor,
    norm: Rms,
}
#[derive(Debug)]
enum Mixer {
    Full(FullAttention),
    Linear(Delta),
}
#[derive(Debug)]
struct Layer {
    input_norm: Rms,
    post_norm: Rms,
    mixer: Mixer,
    gate: Projection,
    up: Projection,
    down: Projection,
}
#[derive(Debug)]
pub(crate) struct Backbone {
    embeddings: Tensor,
    pub output_embeddings: Tensor,
    norm: Rms,
    layers: Vec<Layer>,
    config: TextConfig,
}
/// Immutable state at one exact text-token boundary. Tensors own compact storage.
#[derive(Clone)]
pub(crate) struct PrefixState {
    pub hidden: Tensor,
    layers: Vec<MixerState>,
}
#[derive(Debug, Clone)]
enum MixerState {
    Full { key: Tensor, value: Tensor },
    Linear { recurrent: Tensor, history: Tensor },
}
impl Debug for PrefixState {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_struct("PrefixState")
            .field("hidden_shape", &self.hidden.dims())
            .field("layers", &self.layers.len())
            .finish_non_exhaustive()
    }
}
impl PrefixState {
    pub fn bytes(&self) -> Result<u64> {
        let mut tensors = self
            .layers
            .iter()
            .flat_map(|state| match state {
                MixerState::Full { key, value } => [key, value],
                MixerState::Linear { recurrent, history } => [recurrent, history],
            })
            .chain([&self.hidden]);
        tensors.try_fold(0u64, |total, tensor| {
            let bytes = tensor
                .elem_count()
                .checked_mul(tensor.dtype().size_in_bytes())
                .and_then(|n| u64::try_from(n).ok())
                .ok_or(Error::InsufficientMemory)?;
            total.checked_add(bytes).ok_or(Error::InsufficientMemory)
        })
    }
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
        let norms = Norms::new(embeddings.device(), c.rms_norm_eps)?;
        let norm = norms.load(w, "model.language_model.norm.weight", h, true)?;
        let mut layers = Vec::with_capacity(c.num_hidden_layers);
        for (index, kind) in c.layer_types.iter().enumerate() {
            let p = format!("model.language_model.layers.{index}");
            let mixer = if kind == "full_attention" {
                Mixer::Full(FullAttention::load(
                    w,
                    &projections,
                    &norms,
                    &c,
                    &p,
                    #[cfg(all(feature = "metal", target_os = "macos"))]
                    attention_kernel.clone(),
                )?)
            } else {
                Mixer::Linear(Delta::load(
                    w,
                    &projections,
                    &norms,
                    &c,
                    &p,
                    #[cfg(all(feature = "metal", target_os = "macos"))]
                    delta_kernel.clone(),
                )?)
            };
            layers.push(Layer {
                mixer,
                input_norm: norms.load(w, &format!("{p}.input_layernorm.weight"), h, true)?,
                post_norm: norms.load(
                    w,
                    &format!("{p}.post_attention_layernorm.weight"),
                    h,
                    true,
                )?,
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
    /// Upper-bound estimate of the device bytes needed to retain a
    /// `tokens`-long prefix, for capacity planning.
    ///
    /// The result is conservative, not exact: callers must treat it as an
    /// upper bound (short prefixes are rounded up), never as a precise size.
    pub fn prefix_bytes_upper_bound(&self, tokens: usize) -> Result<u64> {
        let c = &self.config;
        let scalar = self.embeddings.dtype().size_in_bytes();
        let mut bytes = tokens
            .checked_mul(c.hidden_size)
            .and_then(|n| n.checked_mul(4))
            .ok_or(Error::InsufficientMemory)?;
        for layer in &self.layers {
            let layer_bytes = match layer.mixer {
                Mixer::Full(_) => tokens
                    .checked_mul(c.num_key_value_heads)
                    .and_then(|n| n.checked_mul(c.head_dim))
                    .and_then(|n| n.checked_mul(2 * scalar)),
                Mixer::Linear(_) => c
                    .linear_num_value_heads
                    .checked_mul(c.linear_key_head_dim)
                    .and_then(|n| n.checked_mul(c.linear_value_head_dim))
                    .and_then(|n| {
                        n.checked_add(
                            (c.linear_conv_kernel_dim - 1)
                                * (2 * c.linear_num_key_heads * c.linear_key_head_dim
                                    + c.linear_num_value_heads * c.linear_value_head_dim),
                        )
                    })
                    .and_then(|n| n.checked_mul(4)),
            }
            .ok_or(Error::InsufficientMemory)?;
            bytes = bytes
                .checked_add(layer_bytes)
                .ok_or(Error::InsufficientMemory)?;
        }
        u64::try_from(bytes).map_err(|_| Error::InsufficientMemory)
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
        hidden: Tensor,
        positions: &[[usize; 3]],
        control: &Control,
    ) -> Result<Tensor> {
        if positions.len() != hidden.dim(0)? {
            return Err(Error::InvalidRequest(
                "positions/hidden length mismatch".into(),
            ));
        }
        Ok(self
            .prefill_embedded(hidden, positions, control, None, None)?
            .0)
    }
    /// Run a prefill over `ids`, optionally resuming from and extending a
    /// retained prefix.
    ///
    /// `initial` resumes from a previously captured [`PrefixState`]; positions
    /// continue after its tokens. `capture` retains the state after the first
    /// `capture` tokens *of this call* and must be a strict prefix of `ids`
    /// (`0 < capture < ids.len()`); capturing the whole request is rejected
    /// because there would be no suffix left to score.
    ///
    /// Passing both extends a retained prefix: the returned state covers the
    /// initial tokens plus the first `capture` new tokens, so a cache hit can
    /// lengthen the entry it came from instead of recomputing the suffix from
    /// the old boundary every time.
    ///
    /// # Errors
    /// Rejects empty inputs, sequences longer than [`MAX_SEQUENCE_TOKENS`],
    /// out-of-vocabulary ids, degenerate captures, and layer-count mismatches
    /// between `initial` and this backbone.
    pub fn prefill(
        &self,
        ids: &[u32],
        control: &Control,
        initial: Option<&PrefixState>,
        capture: Option<usize>,
    ) -> Result<(Tensor, Option<PrefixState>)> {
        control.check()?;
        let offset = initial
            .map(|state| state.hidden.dim(0))
            .transpose()?
            .unwrap_or_default();
        if ids.is_empty()
            || ids
                .len()
                .checked_add(offset)
                .is_none_or(|n| n > MAX_SEQUENCE_TOKENS)
            || ids.iter().any(|id| *id as usize >= self.config.vocab_size)
            || capture.is_some_and(|n| n == 0 || n >= ids.len())
            || initial.is_some_and(|state| state.layers.len() != self.layers.len())
        {
            return Err(Error::InvalidRequest(
                "prefill boundary or vocabulary".into(),
            ));
        }
        let indices = Tensor::new(ids, self.embeddings.device())?;
        let hidden = self.embeddings.index_select(&indices, 0)?;
        let positions: Vec<_> = (offset..offset + ids.len()).map(|i| [i, i, i]).collect();
        self.prefill_embedded(hidden, &positions, control, initial, capture)
    }
    fn prefill_embedded(
        &self,
        mut hidden: Tensor,
        positions: &[[usize; 3]],
        control: &Control,
        initial: Option<&PrefixState>,
        capture: Option<usize>,
    ) -> Result<(Tensor, Option<PrefixState>)> {
        hidden = hidden.to_dtype(DType::F32)?;
        let mut states = Vec::with_capacity(if capture.is_some() {
            self.layers.len()
        } else {
            0
        });
        for (index, layer) in self.layers.iter().enumerate() {
            control.check()?;
            let input = layer
                .input_norm
                .forward(&hidden)?
                .to_dtype(layer.gate.weight().dtype())?;
            let state = initial.and_then(|state| state.layers.get(index));
            let (mixed, saved) = match &layer.mixer {
                Mixer::Full(a) => measured!(
                    "fullAttention",
                    input.device(),
                    a.forward(&input, &self.config, positions, control, state, capture)
                )?,
                Mixer::Linear(a) => measured!(
                    "deltaNet",
                    input.device(),
                    a.forward(&input, &self.config, control, state, capture)
                )?,
            };
            if let Some(saved) = saved {
                states.push(saved);
            }
            hidden = (hidden + mixed.to_dtype(DType::F32)?)?;
            let norm = layer
                .post_norm
                .forward(&hidden)?
                .to_dtype(layer.gate.weight().dtype())?;
            let ff = layer
                .down
                .forward(&layer.gate.gated_forward(&norm, &layer.up)?)?;
            hidden = (hidden + ff.to_dtype(DType::F32)?)?;
        }
        let hidden = self.norm.forward(&hidden)?;
        // With `initial`, the retained state extends the previous prefix: the
        // hidden part covers `[0, offset + tokens)` and each layer state was
        // already extended by its mixer forward.
        let saved = if let Some(tokens) = capture {
            let prefix = hidden.narrow(0, 0, tokens)?;
            let hidden = match initial {
                Some(initial) => Tensor::cat(&[&initial.hidden, &prefix], 0)?,
                None => prefix,
            };
            Some(PrefixState {
                hidden: retain(&hidden)?,
                layers: states,
            })
        } else {
            None
        };
        let hidden = if let Some(initial) = initial {
            Tensor::cat(&[&initial.hidden, &hidden], 0)?
        } else {
            hidden
        };
        Ok((hidden, saved))
    }
}
impl FullAttention {
    fn load(
        w: &mut Weights,
        projections: &Projections,
        norms: &Norms,
        c: &TextConfig,
        p: &str,
        #[cfg(all(feature = "metal", target_os = "macos"))] kernel: Option<AttentionKernel>,
    ) -> Result<Self> {
        let h = c.hidden_size;
        let n = c.num_attention_heads * c.head_dim;
        let kv = c.num_key_value_heads * c.head_dim;
        let rotary_dim = (c.head_dim as f64 * c.rope_parameters.partial_rotary_factor) as usize;
        let inv_freq: Vec<f32> = (0..rotary_dim / 2)
            .map(|i| {
                c.rope_parameters
                    .rope_theta
                    .powf(-((2 * i) as f64) / rotary_dim as f64) as f32
            })
            .collect();
        Ok(Self {
            #[cfg(all(feature = "metal", target_os = "macos"))]
            kernel,
            q: projections.load(w, &format!("{p}.self_attn.q_proj"), h, n * 2, false)?,
            k: projections.load(w, &format!("{p}.self_attn.k_proj"), h, kv, false)?,
            v: projections.load(w, &format!("{p}.self_attn.v_proj"), h, kv, false)?,
            out: projections.load(w, &format!("{p}.self_attn.o_proj"), n, h, false)?,
            qn: norms.load(w, &format!("{p}.self_attn.q_norm.weight"), c.head_dim, true)?,
            kn: norms.load(w, &format!("{p}.self_attn.k_norm.weight"), c.head_dim, true)?,
            inv_freq,
        })
    }
    fn forward(
        &self,
        x: &Tensor,
        c: &TextConfig,
        positions: &[[usize; 3]],
        control: &Control,
        initial: Option<&MixerState>,
        capture: Option<usize>,
    ) -> Result<(Tensor, Option<MixerState>)> {
        let t = x.dim(0)?;
        let d = c.head_dim;
        let heads = c.num_attention_heads;
        let projected = self.q.forward(x)?.reshape((t, heads, 2 * d))?;
        let gate = projected
            .narrow(2, d, d)?
            .contiguous()?
            .reshape((t, heads * d))?;
        let q = self.qn.forward(&projected.narrow(2, 0, d)?)?;
        let k = self
            .kn
            .forward(&self.k.forward(x)?.reshape((t, c.num_key_value_heads, d))?)?;
        let q = rotary(
            &q,
            positions,
            &self.inv_freq,
            c.rope_parameters.mrope_section,
        )?
        .transpose(0, 1)?
        .contiguous()?;
        let k = rotary(
            &k,
            positions,
            &self.inv_freq,
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
        let (k, v, initial_len) = match initial {
            Some(MixerState::Full { key, value }) => {
                // Validate a resumed KV cache before concatenating: the head
                // count, width and dtype must match this call's tensors.
                let valid = match key.dims() {
                    [kh, _, kd] => {
                        *kh == c.num_key_value_heads
                            && *kd == d
                            && key.dtype() == k.dtype()
                            && value.dims() == key.dims()
                            && value.dtype() == key.dtype()
                    }
                    _ => false,
                };
                if !valid {
                    return Err(Error::InvalidRequest("full-attention prefix state".into()));
                }
                let len = key.dim(1)?;
                (
                    Tensor::cat(&[key, &k], 1)?,
                    Tensor::cat(&[value, &v], 1)?,
                    len,
                )
            }
            None => (k, v, 0),
            _ => return Err(Error::InferenceFailed("full-attention prefix state".into())),
        };
        // With `initial`, the concatenated K/V already cover the previous
        // prefix, so the retained window extends past it instead of
        // re-capturing the old tokens.
        let saved = if let Some(tokens) = capture {
            let kept = initial_len + tokens;
            Some(MixerState::Full {
                key: retain(&k.narrow(1, 0, kept)?)?,
                value: retain(&v.narrow(1, 0, kept)?)?,
            })
        } else {
            None
        };
        #[cfg(all(feature = "metal", target_os = "macos"))]
        let attended = match &self.kernel {
            Some(kernel) => measured!("sdpa", x.device(), kernel.forward(&q, &k, &v, control))?,
            None => attention(&q, &k, &v, true, control)?,
        };
        #[cfg(not(all(feature = "metal", target_os = "macos")))]
        let attended = attention(&q, &k, &v, true, control)?;
        let out = attended
            .to_dtype(x.dtype())?
            .transpose(0, 1)?
            .contiguous()?
            .reshape((t, heads * d))?;
        Ok((self.out.forward(&(out * sigmoid(&gate)?)?)?, saved))
    }
}
impl Delta {
    fn load(
        w: &mut Weights,
        projections: &Projections,
        norms: &Norms,
        c: &TextConfig,
        p: &str,
        #[cfg(all(feature = "metal", target_os = "macos"))] kernel: Option<DeltaKernel>,
    ) -> Result<Self> {
        let h = c.hidden_size;
        let key = c.linear_num_key_heads * c.linear_key_head_dim;
        let value = c.linear_num_value_heads * c.linear_value_head_dim;
        Ok(Self {
            #[cfg(all(feature = "metal", target_os = "macos"))]
            convolution: norms.convolution(),
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
            ab: projections.load_fused(
                w,
                &format!("{p}.linear_attn.in_proj_a"),
                &format!("{p}.linear_attn.in_proj_b"),
                h,
                c.linear_num_value_heads,
            )?,
            out: projections.load(w, &format!("{p}.linear_attn.out_proj"), value, h, false)?,
            conv: w
                .take(
                    &format!("{p}.linear_attn.conv1d.weight"),
                    &[2 * key + value, 1, c.linear_conv_kernel_dim],
                )?
                .to_dtype(DType::F32)?,
            dt: w.take(
                &format!("{p}.linear_attn.dt_bias"),
                &[c.linear_num_value_heads],
            )?,
            a_log: w.take(
                &format!("{p}.linear_attn.A_log"),
                &[c.linear_num_value_heads],
            )?,
            norm: norms.load(
                w,
                &format!("{p}.linear_attn.norm.weight"),
                c.linear_value_head_dim,
                false,
            )?,
        })
    }
    fn forward(
        &self,
        x: &Tensor,
        c: &TextConfig,
        control: &Control,
        initial: Option<&MixerState>,
        capture: Option<usize>,
    ) -> Result<(Tensor, Option<MixerState>)> {
        let t = x.dim(0)?;
        let heads = c.linear_num_value_heads;
        let kd = c.linear_key_head_dim;
        let vd = c.linear_value_head_dim;
        let key = c.linear_num_key_heads * kd;
        let value = heads * vd;
        let projected = self.qkv.forward(x)?.to_dtype(DType::F32)?;
        let (recurrent, history) = match initial {
            Some(MixerState::Linear { recurrent, history }) => (Some(recurrent), Some(history)),
            None => (None, None),
            _ => return Err(Error::InferenceFailed("DeltaNet prefix state".into())),
        };
        let projected = if let Some(history) = history {
            Tensor::cat(&[history, &projected], 0)?
        } else {
            projected
        };
        let history_elems = history.map_or(0, Tensor::elem_count);
        debug_assert_eq!(
            history_elems % (2 * key + value),
            0,
            "convolution history must hold whole tokens"
        );
        let history_tokens = history_elems / (2 * key + value);

        let convolved = self.convolve(&projected, c.linear_conv_kernel_dim)?;
        let convolved = convolved.narrow(0, history_tokens, t)?;
        let mixed = silu(&convolved)?;
        // One GEMM computes both decay and beta projections; `x` is read once.
        let ab = self.ab.forward(x)?.to_dtype(DType::F32)?;
        let a = ab
            .narrow(1, 0, heads)?
            .broadcast_add(&self.dt.to_dtype(DType::F32)?)?;
        // Stable softplus, preserving the reference's float32 decay path.
        let softplus = (a.clamp(0., f64::INFINITY)? + ((a.abs()?.neg()?.exp()? + 1.)?.log()?))?;
        let g = softplus.broadcast_mul(&self.a_log.to_dtype(DType::F32)?.exp()?.neg()?)?;
        let beta = sigmoid(&ab.narrow(1, heads, heads)?)?;
        #[cfg(all(feature = "metal", target_os = "macos"))]
        let prepared = self
            .kernel
            .as_ref()
            .filter(|kernel| kernel.can_prepare())
            .map(|kernel| {
                kernel
                    .prepare(&mixed, &g, &beta)
                    .map(|packed| (kernel, packed))
            })
            .transpose()?;
        #[cfg(all(feature = "metal", target_os = "macos"))]
        let (out, saved_recurrent) = if let Some((kernel, packed)) = prepared {
            measured!(
                "recurrence",
                x.device(),
                kernel.prepared_prefill(&packed, control, recurrent, capture)
            )?
        } else {
            Self::recur_portable(
                &mixed,
                &g,
                &beta,
                c,
                control,
                recurrent,
                capture,
                #[cfg(all(feature = "metal", target_os = "macos"))]
                self.kernel.as_ref(),
            )?
        };
        #[cfg(not(all(feature = "metal", target_os = "macos")))]
        let (out, saved_recurrent) = Self::recur_portable(
            &mixed,
            &g,
            &beta,
            c,
            control,
            recurrent,
            capture,
            #[cfg(all(feature = "metal", target_os = "macos"))]
            self.kernel.as_ref(),
        )?;
        let z = self.z.forward(x)?.reshape((t, heads, vd))?;
        // `out` is already F32: skip the F32 -> model dtype -> F32 round trip.
        // `Rms` converts non-F32 inputs to F32 internally, so feeding it F32
        // directly is a no-op conversion and avoids extra quantization noise.
        let gated = (self.norm.forward(&out)? * silu(&z.to_dtype(DType::F32)?)?)?
            .to_dtype(x.dtype())?
            .reshape((t, value))?;
        let saved = if let (Some(tokens), Some(recurrent)) = (capture, saved_recurrent) {
            let end = history_tokens + tokens;
            let count = end.min(c.linear_conv_kernel_dim - 1);
            Some(MixerState::Linear {
                recurrent,
                history: retain(&projected.narrow(0, end - count, count)?)?,
            })
        } else {
            None
        };
        Ok((self.out.forward(&gated)?, saved))
    }
    fn convolve(&self, projected: &Tensor, taps: usize) -> Result<Tensor> {
        #[cfg(all(feature = "metal", target_os = "macos"))]
        let convolved = measured!(
            "convolution",
            projected.device(),
            match &self.convolution {
                Some(kernel) => Ok::<_, Error>(kernel.forward(projected, &self.conv)?),
                None => causal_convolution(projected, &self.conv, taps),
            }
        )?;
        #[cfg(not(all(feature = "metal", target_os = "macos")))]
        let convolved = measured!(
            "convolution",
            projected.device(),
            causal_convolution(projected, &self.conv, taps)
        )?;
        Ok(convolved)
    }
    #[allow(
        clippy::too_many_arguments,
        reason = "explicit tensors and continuation state implement the reference equation"
    )]
    fn recur_portable(
        mixed: &Tensor,
        g: &Tensor,
        beta: &Tensor,
        c: &TextConfig,
        control: &Control,
        recurrent: Option<&Tensor>,
        capture: Option<usize>,
        #[cfg(all(feature = "metal", target_os = "macos"))] kernel: Option<&DeltaKernel>,
    ) -> Result<(Tensor, Option<Tensor>)> {
        let t = mixed.dim(0)?;
        let heads = c.linear_num_value_heads;
        let kd = c.linear_key_head_dim;
        let vd = c.linear_value_head_dim;
        let key = c.linear_num_key_heads * kd;
        let value = heads * vd;
        let repeat: Vec<u32> = (0..heads)
            .map(|i| (i / (heads / c.linear_num_key_heads)) as u32)
            .collect();
        let repeat = Tensor::new(repeat, mixed.device())?;
        let q = mixed
            .narrow(1, 0, key)?
            .reshape((t, c.linear_num_key_heads, kd))?
            .index_select(&repeat, 1)?;
        let k = mixed
            .narrow(1, key, key)?
            .reshape((t, c.linear_num_key_heads, kd))?
            .index_select(&repeat, 1)?;
        let v = mixed.narrow(1, 2 * key, value)?.reshape((t, heads, vd))?;
        measured!(
            "recurrence",
            mixed.device(),
            delta_recurrence_prefill(
                &q,
                &k,
                &v,
                g,
                beta,
                control,
                recurrent,
                capture,
                #[cfg(all(feature = "metal", target_os = "macos"))]
                kernel,
            )
        )
    }
}
fn retain(tensor: &Tensor) -> Result<Tensor> {
    #[cfg(all(feature = "metal", target_os = "macos"))]
    if tensor.device().is_metal() {
        return Ok(compact_copy(tensor)?);
    }
    Ok(tensor.force_contiguous()?)
}
pub(super) fn causal_convolution(
    projected: &Tensor,
    weights: &Tensor,
    taps: usize,
) -> Result<Tensor> {
    let (t, width) = projected.dims2()?;
    if taps == 0 {
        return Ok(Tensor::zeros((t, width), DType::F32, projected.device())?);
    }
    let projected = projected.to_dtype(DType::F32)?;
    // Pad once so every tap becomes a plain narrow view of the padded input.
    // Tap `tap` reads the input delayed by `taps - 1 - tap` rows. Accumulate
    // per tap over views: materializing a stacked [taps, t, width] tensor
    // costs an extra full-size copy plus strided reads, which measured slower
    // than the loop on CPU.
    let padded = Tensor::cat(
        &[
            Tensor::zeros((taps - 1, width), DType::F32, projected.device())?,
            projected,
        ],
        0,
    )?;
    let kernel = weights
        .narrow(2, 0, taps)?
        .squeeze(1)?
        .to_dtype(DType::F32)?
        .transpose(0, 1)?
        .contiguous()?;
    let mut acc = Tensor::zeros((t, width), DType::F32, padded.device())?;
    for tap in 0..taps {
        // `kernel` is contiguous, so each row is a dense `[1, width]` view.
        let shifted = padded.narrow(0, tap, t)?;
        let weight = kernel.narrow(0, tap, 1)?;
        acc = (acc + shifted.broadcast_mul(&weight)?)?;
    }
    Ok(acc)
}

#[cfg(all(test, feature = "metal", target_os = "macos"))]
pub(super) fn delta_recurrence(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    g: &Tensor,
    beta: &Tensor,
    control: &Control,
    #[cfg(all(feature = "metal", target_os = "macos"))] kernel: Option<&DeltaKernel>,
) -> Result<Tensor> {
    Ok(delta_recurrence_prefill(
        q,
        k,
        v,
        g,
        beta,
        control,
        None,
        None,
        #[cfg(all(feature = "metal", target_os = "macos"))]
        kernel,
    )?
    .0)
}
#[allow(
    clippy::too_many_arguments,
    reason = "explicit tensors and immutable continuation state mirror the recurrence equation"
)]
fn delta_recurrence_prefill(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    g: &Tensor,
    beta: &Tensor,
    control: &Control,
    initial: Option<&Tensor>,
    capture: Option<usize>,
    #[cfg(all(feature = "metal", target_os = "macos"))] kernel: Option<&DeltaKernel>,
) -> Result<(Tensor, Option<Tensor>)> {
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
        if initial.is_none() && capture.is_none() {
            return Ok((kernel.forward(&q, &k, &v, &g, &beta, control)?, None));
        }
        return kernel.prefill(&q, &k, &v, &g, &beta, control, initial, capture);
    }
    let mut state = match initial {
        Some(state) if state.dims() == [heads, kd, vd] => state.clone(),
        Some(_) => {
            return Err(Error::InvalidRequest(
                "DeltaNet initial state dimensions".into(),
            ));
        }
        None => Tensor::zeros((heads, kd, vd), DType::F32, q.device())?,
    };
    let mut saved = None;
    let mut outputs = Vec::with_capacity(t);
    // The decay depends only on `g`: compute all exps once instead of once per
    // loop iteration.
    let decay = g.exp()?;
    for i in 0..t {
        control.check()?;
        let key = k.narrow(0, i, 1)?.squeeze(0)?.unsqueeze(2)?;
        let query = q.narrow(0, i, 1)?.squeeze(0)?.unsqueeze(2)?;
        state = state.broadcast_mul(&decay.narrow(0, i, 1)?.reshape((heads, 1, 1))?)?;
        let memory = state.broadcast_mul(&key)?.sum(1)?;
        let delta = (v.narrow(0, i, 1)?.squeeze(0)? - memory)?
            .broadcast_mul(&beta.narrow(0, i, 1)?.reshape((heads, 1))?)?;
        state = (state + key.broadcast_mul(&delta.unsqueeze(1)?)?)?;
        outputs.push(state.broadcast_mul(&query)?.sum(1)?.unsqueeze(0)?);
        if capture == Some(i + 1) {
            saved = Some(retain(&state)?);
        }
    }
    Ok((Tensor::cat(&outputs, 0)?, saved))
}

#[cfg(test)]
mod tests {
    use candle_core::{DType, Device, Tensor};

    use super::*;

    fn reference_convolution(projected: &Tensor, weights: &Tensor, taps: usize) -> Result<Tensor> {
        let (t, width) = projected.dims2()?;
        let projected = projected.to_dtype(DType::F32)?;
        let mut convolved = Tensor::zeros((t, width), DType::F32, projected.device())?;
        for tap in 0..taps {
            let delay = taps - 1 - tap;
            if delay >= t {
                continue;
            }
            let weight = weights
                .narrow(2, tap, 1)?
                .reshape((1, width))?
                .to_dtype(DType::F32)?;
            let part = projected.narrow(0, 0, t - delay)?.broadcast_mul(&weight)?;
            let part = if delay == 0 {
                part
            } else {
                Tensor::cat(
                    &[
                        Tensor::zeros((delay, width), DType::F32, projected.device())?,
                        part,
                    ],
                    0,
                )?
            };
            convolved = (convolved + part)?;
        }
        Ok(convolved)
    }

    #[test]
    fn test_causal_convolution_matches_reference() -> Result<()> {
        let device = Device::Cpu;
        for (tokens, width) in [(1, 7), (3, 5), (17, 64), (139, 128)] {
            let x = Tensor::randn(0f32, 1., (tokens, width), &device)?;
            let w = Tensor::randn(0f32, 1., (width, 1, 4), &device)?;
            let expected = reference_convolution(&x, &w, 4)?;
            let actual = causal_convolution(&x, &w, 4)?;
            let error = (&actual - &expected)?
                .abs()?
                .flatten_all()?
                .max(0)?
                .to_scalar::<f32>()?;
            assert!(
                error < 1e-4,
                "causal_convolution mismatch at {tokens}/{width}: {error}"
            );
        }
        // taps == 0 returns zeros without touching the weights.
        let zeros = causal_convolution(
            &Tensor::randn(0f32, 1., (8, 16), &device)?,
            &Tensor::randn(0f32, 1., (16, 1, 4), &device)?,
            0,
        )?;
        assert_eq!(zeros.sum_all()?.to_scalar::<f32>()?, 0.0);
        Ok(())
    }
}
