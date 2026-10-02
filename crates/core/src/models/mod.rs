//! Private safe tensor implementation of the pinned Qwen3.5/CLEF graph.
pub(crate) mod head;
pub(crate) mod ops;
pub(crate) mod qwen;
#[cfg(feature = "vision")]
pub(crate) mod vision;
pub(crate) mod weights;

use std::{
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    time::Instant,
};

use crate::{Error, Result};

#[derive(Debug, Clone)]
pub(crate) struct Control {
    pub cancel: Arc<AtomicBool>,
    pub deadline: Instant,
}
impl Control {
    pub fn check(&self) -> Result<()> {
        if self.cancel.load(Ordering::Acquire) {
            return Err(Error::Cancelled);
        }
        if Instant::now() >= self.deadline {
            return Err(Error::DeadlineExceeded);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use candle_core::{Device, Tensor};

    use super::*;
    use crate::encoding::{EncodedQuestion, EncodedRecord, TokenSpan};
    fn control() -> Control {
        Control {
            cancel: Arc::new(AtomicBool::new(false)),
            deadline: Instant::now() + Duration::from_secs(60),
        }
    }
    fn assert_close(actual: &Tensor, expected: &Tensor) -> Result<()> {
        let a = actual.flatten_all()?.to_vec1::<f32>()?;
        let e = expected.flatten_all()?.to_vec1::<f32>()?;
        assert_eq!(a.len(), e.len());
        for (i, (a, e)) in a.into_iter().zip(e).enumerate() {
            assert!(
                a.is_finite() && (a - e).abs() <= 1e-5 + 1e-4 * e.abs(),
                "element {i}: {a} vs {e}"
            );
        }
        Ok(())
    }
    #[test]
    fn test_should_match_pinned_transformers_hybrid_backbone() -> Result<()> {
        let config: qwen::TextConfig = serde_json::from_slice(include_bytes!(
            "../../fixtures/synthetic/backbone-config.json"
        ))?;
        let mut weights = weights::Weights::fixture(include_bytes!(
            "../../fixtures/synthetic/backbone.safetensors"
        ))?;
        let backbone = qwen::Backbone::load(&mut weights, config)?;
        weights.finish()?;
        let ids: Vec<u32> = (0..20).collect();
        let actual = backbone.forward(&ids, &control())?;
        let expected = candle_core::safetensors::load_buffer(
            include_bytes!("../../fixtures/synthetic/backbone-output.safetensors"),
            &Device::Cpu,
        )?;
        assert_close(
            &actual,
            expected.get("hidden").ok_or(Error::ArtifactMissing)?,
        )
    }
    #[test]
    fn test_should_match_pinned_joint_head_all_question_types() -> Result<()> {
        let config: head::HeadConfig =
            serde_json::from_slice(include_bytes!("../../fixtures/synthetic/head-config.json"))?;
        let mut weights =
            weights::Weights::fixture(include_bytes!("../../fixtures/synthetic/head.safetensors"))?;
        let head = head::Head::load(&mut weights, &config)?;
        weights.finish()?;
        let input = candle_core::safetensors::load_buffer(
            include_bytes!("../../fixtures/synthetic/head-input.safetensors"),
            &Device::Cpu,
        )?;
        let span = |s, e| TokenSpan::new(s, e, 20);
        let questions = vec![
            EncodedQuestion {
                kind: 0,
                instruction: span(1, 3)?,
                options: vec![span(3, 5)?, span(5, 7)?],
            },
            EncodedQuestion {
                kind: 1,
                instruction: span(7, 9)?,
                options: vec![span(9, 11)?, span(11, 13)?, span(13, 15)?],
            },
            EncodedQuestion {
                kind: 2,
                instruction: span(15, 17)?,
                options: vec![span(17, 18)?, span(18, 19)?],
            },
        ];
        let record = EncodedRecord {
            ids: (0..20).collect(),
            questions,
            truncated: 0,
            #[cfg(feature = "vision")]
            images: Vec::new(),
            #[cfg(feature = "vision")]
            media_spans: Vec::new(),
            #[cfg(feature = "vision")]
            positions: (0..20).map(|i| [i, i, i]).collect(),
        };
        let actual = head.forward(
            input.get("hidden").ok_or(Error::ArtifactMissing)?,
            input
                .get("output_embeddings")
                .ok_or(Error::ArtifactMissing)?,
            &record,
            &control(),
        )?;
        for (i, l) in actual.into_iter().enumerate() {
            assert_close(
                &Tensor::new(l, &Device::Cpu)?,
                input
                    .get(&format!("logits.{i}"))
                    .ok_or(Error::ArtifactMissing)?,
            )?;
        }
        Ok(())
    }
}
