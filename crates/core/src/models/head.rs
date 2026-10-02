//! Learned joint schema head with packed `PyTorch` attention projection ordering.
#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::many_single_char_names,
    reason = "reviewed model dimensions are bounded; tensor equations retain reference \
              mathematical names"
)]

use candle_core::{D, Tensor};
use candle_nn::{Linear, Module};
use serde::Deserialize;

use super::{
    Control,
    ops::{MultiAttention, Norm, normalize},
    weights::Weights,
};
use crate::{
    Error, Result,
    encoding::{EncodedRecord, TokenSpan},
};

#[derive(Debug, Clone, Deserialize)]
pub(crate) struct HeadConfig {
    pub hidden_size: usize,
    pub width: usize,
    pub routing_layers: usize,
    pub layers: usize,
    pub heads: usize,
    pub feedforward: usize,
}
impl HeadConfig {
    pub fn validate(&self, h: usize) -> Result<()> {
        if self.hidden_size != h
            || self.width != 1024
            || self.routing_layers != 2
            || self.layers != 4
            || self.heads != 16
            || self.feedforward != 4096
        {
            return Err(Error::UnsupportedArchitecture);
        }
        Ok(())
    }
}
#[derive(Debug)]
struct Routing {
    query_norm: Norm,
    memory_norm: Norm,
    attention: MultiAttention,
    ff_norm: Norm,
    ff1: Linear,
    ff2: Linear,
}
#[derive(Debug)]
struct Decoder {
    norm1: Norm,
    norm2: Norm,
    norm3: Norm,
    self_attn: MultiAttention,
    cross_attn: MultiAttention,
    ff1: Linear,
    ff2: Linear,
}
#[derive(Debug)]
pub(crate) struct Head {
    hidden_norm: Norm,
    memory: Linear,
    question: Linear,
    option_question: Linear,
    global: Linear,
    context: Linear,
    lexical: Linear,
    types: Tensor,
    routes: Vec<Routing>,
    summary_norm: Norm,
    decoders: Vec<Decoder>,
    field_norm: Norm,
    option_norm: Norm,
    residual1: Linear,
    residual2: Linear,
    prior_scale: f64,
    joint_scale: f64,
    residual_gate: f64,
    width: usize,
}
impl Head {
    pub fn load(w: &mut Weights, c: &HeadConfig) -> Result<Self> {
        let h = c.hidden_size;
        let width = c.width;
        let mut routes = Vec::new();
        for i in 0..c.routing_layers {
            let p = format!("evidence_layers.{i}");
            routes.push(Routing {
                query_norm: w.norm(&format!("{p}.query_norm"), width, 1e-5)?,
                memory_norm: w.norm(&format!("{p}.memory_norm"), width, 1e-5)?,
                attention: w.attention(&format!("{p}.attention"), width, c.heads)?,
                ff_norm: w.norm(&format!("{p}.feedforward_norm"), width, 1e-5)?,
                ff1: w.linear(&format!("{p}.feedforward.0"), width, c.feedforward, true)?,
                ff2: w.linear(&format!("{p}.feedforward.3"), c.feedforward, width, true)?,
            });
        }
        let mut decoders = Vec::new();
        for i in 0..c.layers {
            let p = format!("layers.{i}");
            decoders.push(Decoder {
                norm1: w.norm(&format!("{p}.norm1"), width, 1e-5)?,
                norm2: w.norm(&format!("{p}.norm2"), width, 1e-5)?,
                norm3: w.norm(&format!("{p}.norm3"), width, 1e-5)?,
                self_attn: w.attention(&format!("{p}.self_attn"), width, c.heads)?,
                cross_attn: w.attention(&format!("{p}.multihead_attn"), width, c.heads)?,
                ff1: w.linear(&format!("{p}.linear1"), width, c.feedforward, true)?,
                ff2: w.linear(&format!("{p}.linear2"), c.feedforward, width, true)?,
            });
        }
        let prior_scale = w.scalar("prior_logit_scale")?.min(100_f64.ln()).exp();
        let joint_scale = w.scalar("joint_logit_scale")?.min(100_f64.ln()).exp();
        let gate = w.scalar("residual_gate")?;
        Ok(Self {
            hidden_norm: w.norm("hidden_norm", h, 1e-5)?,
            memory: w.linear("memory_projection", h, width, false)?,
            question: w.linear("question_projection", h, width, false)?,
            option_question: w.linear("option_question_projection", h, width, false)?,
            global: w.linear("global_projection", h, width, false)?,
            context: w.linear("option_context_projection", h, width, false)?,
            lexical: w.linear("option_lexical_projection", h, width, false)?,
            types: w.take("type_embedding.weight", &[3, width])?,
            routes,
            summary_norm: w.norm("option_summary_norm", width, 1e-5)?,
            decoders,
            field_norm: w.norm("field_norm", width, 1e-5)?,
            option_norm: w.norm("option_norm", width, 1e-5)?,
            residual1: w.linear("residual_scorer.0", 4 * width, width, true)?,
            residual2: w.linear("residual_scorer.3", width, 1, true)?,
            prior_scale,
            joint_scale,
            residual_gate: 1. / (1. + (-gate).exp()),
            width,
        })
    }
    #[allow(
        clippy::too_many_lines,
        reason = "single reference head graph stays within the repository 150-line ceiling"
    )]
    pub fn forward(
        &self,
        hidden: &Tensor,
        output_embeddings: &Tensor,
        record: &EncodedRecord,
        control: &Control,
    ) -> Result<Vec<Vec<f32>>> {
        control.check()?;
        let hidden = hidden.to_dtype(self.hidden_norm.weight.dtype())?;
        let normalized = self.hidden_norm.forward(&hidden)?;
        let memory = self.memory.forward(&normalized)?;
        let global = normalized.narrow(0, record.ids.len() - 1, 1)?;
        let questions: Vec<Tensor> = record
            .questions
            .iter()
            .map(|q| mean_span(&normalized, q.instruction))
            .collect::<Result<_>>()?;
        let questions = Tensor::cat(&questions, 0)?;
        let input_ids = Tensor::new(record.ids.as_slice(), hidden.device())?;
        let mut lexical = Vec::new();
        let mut queries = Vec::new();
        let mut counts = Vec::new();
        for (i, q) in record.questions.iter().enumerate() {
            let contexts: Vec<Tensor> = q
                .options
                .iter()
                .map(|span| mean_span(&normalized, *span))
                .collect::<Result<_>>()?;
            let contexts = Tensor::cat(&contexts, 0)?;
            let lex: Vec<Tensor> = q
                .options
                .iter()
                .map(|span| {
                    let ids = input_ids.narrow(0, span.start(), span.len())?;
                    Ok(output_embeddings
                        .index_select(&ids, 0)?
                        .to_dtype(hidden.dtype())?
                        .mean_keepdim(0)?)
                })
                .collect::<Result<_>>()?;
            let lex = Tensor::cat(&lex, 0)?;
            let query = (self.context.forward(&contexts)? + self.lexical.forward(&lex)?)?
                .broadcast_add(&self.option_question.forward(&questions.narrow(0, i, 1)?)?)?;
            queries.push(query);
            lexical.push(lex);
            counts.push(q.options.len());
        }
        let mut routed = Tensor::cat(&queries, 0)?;
        for route in &self.routes {
            control.check()?;
            let normalized_queries = route.query_norm.forward(&routed)?;
            let normalized_memory = route.memory_norm.forward(&memory)?;
            routed = (routed
                + route
                    .attention
                    .forward(&normalized_queries, &normalized_memory, control)?)?;
            let ff = route.ff2.forward(
                &route
                    .ff1
                    .forward(&route.ff_norm.forward(&routed)?)?
                    .gelu_erf()?,
            )?;
            routed = (routed + ff)?;
        }
        let base = self.question.forward(&questions)?;
        let mut summaries = Vec::new();
        let mut split = Vec::new();
        let mut offset = 0;
        for (i, count) in counts.iter().copied().enumerate() {
            let options = routed.narrow(0, offset, count)?;
            let weights = (options
                .broadcast_mul(&base.narrow(0, i, 1)?)?
                .sum_keepdim(1)?
                / (self.width as f64).sqrt())?;
            let weights = candle_nn::ops::softmax(&weights, 0)?;
            summaries.push(options.broadcast_mul(&weights)?.sum_keepdim(0)?);
            split.push(options);
            offset += count;
        }
        let type_ids: Vec<u32> = record.questions.iter().map(|q| q.kind).collect();
        let type_vectors = self
            .types
            .index_select(&Tensor::new(type_ids, hidden.device())?, 0)?;
        let mut fields = (base + self.summary_norm.forward(&Tensor::cat(&summaries, 0)?)?)?
            .broadcast_add(&self.global.forward(&global)?)?;
        fields = (fields + type_vectors)?;
        for layer in &self.decoders {
            control.check()?;
            let normalized = layer.norm1.forward(&fields)?;
            fields = (fields + layer.self_attn.forward(&normalized, &normalized, control)?)?;
            let normalized = layer.norm2.forward(&fields)?;
            fields = (fields + layer.cross_attn.forward(&normalized, &memory, control)?)?;
            let ff = layer.ff2.forward(
                &layer
                    .ff1
                    .forward(&layer.norm3.forward(&fields)?)?
                    .gelu_erf()?,
            )?;
            fields = (fields + ff)?;
        }
        let fields = self.field_norm.forward(&fields)?;
        let mut logits = Vec::new();
        for (i, (lex, options)) in lexical.iter().zip(&split).enumerate() {
            control.check()?;
            let anchor = normalize(&questions.narrow(0, i, 1)?.broadcast_add(&global)?, 1e-12)?;
            let prior =
                (normalize(lex, 1e-12)?.broadcast_mul(&anchor)?.sum(1)? * self.prior_scale)?;
            let options = self.option_norm.forward(options)?;
            let field = fields.narrow(0, i, 1)?.broadcast_as(options.shape())?;
            let cosine = (normalize(&field, 1e-8)? * normalize(&options, 1e-8)?)?.sum(1)?;
            let product = (&field * &options)?;
            let difference = (&field - &options)?.abs()?;
            let features = Tensor::cat(&[&field, &options, &product, &difference], 1)?;
            let residual = self
                .residual2
                .forward(&self.residual1.forward(&features)?.gelu_erf()?)?
                .squeeze(1)?;
            let joint = ((cosine * self.joint_scale)? + residual)?;
            let logit = (prior + (joint * self.residual_gate)?)?
                .to_dtype(candle_core::DType::F32)?
                .to_vec1::<f32>()?;
            if logit.iter().any(|v| !v.is_finite()) {
                return Err(Error::InferenceFailed("nonfinite head output".into()));
            }
            logits.push(logit);
        }
        Ok(logits)
    }
}
fn mean_span(x: &Tensor, span: TokenSpan) -> Result<Tensor> {
    Ok(x.narrow(0, span.start(), span.len())?
        .mean_keepdim(D::Minus2)?)
}
