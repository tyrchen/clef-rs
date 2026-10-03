//! Test-only, owner-thread GPU timing with nested exclusive categories.

use std::{
    cell::RefCell,
    collections::BTreeMap,
    result::Result as StandardResult,
    time::{Duration, Instant},
};

use candle_core::{Device, Error as CandleError};
use serde::Serialize;

use crate::{Error, Result};

#[derive(Debug, Default, Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct Timing {
    calls: usize,
    exclusive_ms: f64,
    inclusive_ms: f64,
}
#[derive(Debug, Default)]
struct Profile {
    frames: Vec<Frame>,
    timings: BTreeMap<&'static str, Timing>,
}
#[derive(Debug)]
struct Frame {
    children: Duration,
}
thread_local! {
    static ACTIVE: RefCell<Option<Profile>> = const { RefCell::new(None) };
}
#[derive(Debug)]
struct Session;
impl Drop for Session {
    fn drop(&mut self) {
        ACTIVE.with(|slot| {
            slot.replace(None);
        });
    }
}
pub(crate) fn capture<T>(
    operation: impl FnOnce() -> Result<T>,
) -> Result<(T, BTreeMap<&'static str, Timing>)> {
    ACTIVE.with(|slot| {
        let mut slot = slot.borrow_mut();
        if slot.is_some() {
            return Err(Error::InferenceFailed("nested diagnostic session".into()));
        }
        *slot = Some(Profile::default());
        Ok(())
    })?;
    let _session = Session;
    let value = operation()?;
    let profile = ACTIVE
        .with(RefCell::take)
        .ok_or_else(|| Error::InferenceFailed("missing diagnostic session".into()))?;
    Ok((value, profile.timings))
}
pub(crate) fn measure<T, E: From<CandleError>>(
    name: &'static str,
    device: &Device,
    operation: impl FnOnce() -> StandardResult<T, E>,
) -> StandardResult<T, E> {
    if !ACTIVE.with(|slot| slot.borrow().is_some()) {
        return operation();
    }
    device.synchronize().map_err(E::from)?;
    ACTIVE.with(|slot| {
        if let Some(profile) = slot.borrow_mut().as_mut() {
            profile.frames.push(Frame {
                children: Duration::ZERO,
            });
        }
    });
    let started = Instant::now();
    let result = operation();
    let completion = device.synchronize();
    let elapsed = started.elapsed();
    ACTIVE.with(|slot| {
        if let Some(profile) = slot.borrow_mut().as_mut()
            && let Some(frame) = profile.frames.pop()
        {
            let timing = profile.timings.entry(name).or_default();
            timing.calls += 1;
            timing.exclusive_ms += elapsed.saturating_sub(frame.children).as_secs_f64() * 1000.;
            timing.inclusive_ms += elapsed.as_secs_f64() * 1000.;
            if let Some(parent) = profile.frames.last_mut() {
                parent.children += elapsed;
            }
        }
    });
    completion.map_err(E::from)?;
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_should_record_nested_categories_and_clear_failed_sessions() -> Result<()> {
        let (value, timings) = capture(|| {
            measure("parent", &Device::Cpu, || {
                measure("child", &Device::Cpu, || Ok::<_, Error>(7))
            })
        })?;
        assert_eq!(value, 7);
        for name in ["parent", "child"] {
            let timing = timings.get(name).ok_or(Error::ArtifactMissing)?;
            assert_eq!(timing.calls, 1);
            assert!(timing.exclusive_ms <= timing.inclusive_ms);
        }
        let failed = capture(|| Err::<(), _>(Error::Cancelled));
        assert!(matches!(failed, Err(Error::Cancelled)));
        assert!(capture(|| Ok(())).is_ok());
        Ok(())
    }
}
