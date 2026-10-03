//! Private safe tensor implementation of the pinned Qwen3.5/CLEF graph.
// Diagnostic barriers exist only in test binaries, never normal inference.
macro_rules! measured {
    ($name:expr, $device:expr, $operation:expr) => {{
        #[cfg(test)]
        {
            $crate::models::diagnostics::measure($name, $device, || $operation)
        }
        #[cfg(not(test))]
        {
            $operation
        }
    }};
}
#[cfg(all(test, feature = "metal", target_os = "macos"))]
mod chunk;
#[cfg(test)]
pub(crate) mod diagnostics;
pub(crate) mod head;
#[cfg(all(feature = "metal", target_os = "macos"))]
mod metal;
#[cfg(all(feature = "metal", target_os = "macos"))]
mod neural;
mod normalization;
pub(crate) mod ops;
#[cfg(all(feature = "metal", target_os = "macos"))]
mod pointwise;
mod projection;
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
        backbone_parity(&Device::Cpu)
    }
    pub(super) fn backbone_parity(device: &Device) -> Result<()> {
        let config: qwen::TextConfig = serde_json::from_slice(include_bytes!(
            "../../fixtures/synthetic/backbone-config.json"
        ))?;
        let mut weights = weights::Weights::fixture_on(
            include_bytes!("../../fixtures/synthetic/backbone.safetensors"),
            device,
        )?;
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
    fn test_should_resume_exact_prefix_with_changed_suffix_and_short_history() -> Result<()> {
        prefix_parity(&Device::Cpu)
    }
    pub(super) fn prefix_parity(device: &Device) -> Result<()> {
        let config: qwen::TextConfig = serde_json::from_slice(include_bytes!(
            "../../fixtures/synthetic/backbone-config.json"
        ))?;
        let mut weights = weights::Weights::fixture_on(
            include_bytes!("../../fixtures/synthetic/backbone.safetensors"),
            device,
        )?;
        let backbone = qwen::Backbone::load(&mut weights, config)?;
        let ids: Vec<u32> = (0..20).collect();
        let expected = backbone.forward(&ids, &control())?;
        for boundary in [1, 2, 3, 8, 19] {
            let (captured, state) = backbone.prefill(&ids, &control(), None, Some(boundary))?;
            assert_close(&captured, &expected)?;
            let state = state.ok_or(Error::ArtifactMissing)?;
            assert_eq!(
                state.bytes()?,
                backbone.prefix_bytes(boundary)?
                    - if boundary < 3 {
                        u64::try_from((3 - boundary) * 3 * 16 * 4)
                            .map_err(|_| Error::InsufficientMemory)?
                    } else {
                        0
                    }
            );
            let suffix = ids.get(boundary..).ok_or(Error::ArtifactMissing)?;
            assert!(
                backbone
                    .prefill(suffix, &control(), Some(&state), Some(1))
                    .is_err()
            );
            let resumed = backbone.prefill(suffix, &control(), Some(&state), None)?.0;
            assert_close(&resumed, &expected)?;
            let mut changed = ids.clone();
            for id in changed.iter_mut().skip(boundary) {
                *id += 20;
            }
            let suffix = changed.get(boundary..).ok_or(Error::ArtifactMissing)?;
            let resumed = backbone.prefill(suffix, &control(), Some(&state), None)?.0;
            assert_close(&resumed, &backbone.forward(&changed, &control())?)?;
            // Resuming never mutates the shared state.
            assert_close(
                &backbone
                    .prefill(
                        ids.get(boundary..).ok_or(Error::ArtifactMissing)?,
                        &control(),
                        Some(&state),
                        None,
                    )?
                    .0,
                &expected,
            )?;
        }
        assert!(backbone.prefill(&ids, &control(), None, Some(20)).is_err());
        Ok(())
    }
    #[test]
    fn test_should_match_pinned_joint_head_all_question_types() -> Result<()> {
        head_parity(&Device::Cpu)
    }
    pub(super) fn head_parity(device: &Device) -> Result<()> {
        let config: head::HeadConfig =
            serde_json::from_slice(include_bytes!("../../fixtures/synthetic/head-config.json"))?;
        let mut weights = weights::Weights::fixture_on(
            include_bytes!("../../fixtures/synthetic/head.safetensors"),
            device,
        )?;
        let head = head::Head::load(&mut weights, &config)?;
        weights.finish()?;
        let input = candle_core::safetensors::load_buffer(
            include_bytes!("../../fixtures/synthetic/head-input.safetensors"),
            device,
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
            state_end: 0,
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

#[cfg(all(test, feature = "metal", target_os = "macos"))]
mod metal_tests {
    use candle_core::Device;

    use super::tests::{backbone_parity, head_parity, prefix_parity};
    use crate::Result;

    #[test]
    #[ignore = "requires a real Apple Metal device"]
    fn test_should_resume_hybrid_prefix_on_metal() -> Result<()> {
        prefix_parity(&Device::new_metal(0)?)
    }
    #[test]
    #[ignore = "requires a real Apple Metal device"]
    fn test_should_match_hybrid_backbone_on_metal() -> Result<()> {
        backbone_parity(&Device::new_metal(0)?)
    }
    #[test]
    #[ignore = "requires a real Apple Metal device"]
    fn test_should_match_joint_head_on_metal() -> Result<()> {
        head_parity(&Device::new_metal(0)?)
    }
}
