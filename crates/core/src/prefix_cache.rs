//! Bounded, principal-scoped immutable prefill states owned by the device actor.

use std::{
    collections::VecDeque,
    fmt::{self, Debug, Formatter},
    time::{Duration, Instant},
};

use serde::{Deserialize, Serialize};

use crate::{Error, Result, models::qwen::PrefixState};

/// Block granularity for prefix capture; requests reuse whole blocks only.
const PREFIX_BLOCK: usize = 128;
const MIN_PREFIX: usize = 512;

/// Exact text-prefix reuse. Zero capacity disables retention and capture overhead.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields, default)]
#[non_exhaustive]
pub struct PrefixCacheConfig {
    /// Live tensor/token budget, at most 4 GiB. Admission also reserves staging space.
    pub capacity_bytes: u64,
    /// Maximum retained prefixes, from 1 to 16.
    pub max_entries: usize,
    /// Absolute lifetime of a state, from 1 to 3600 seconds.
    pub ttl_seconds: u64,
}
impl Default for PrefixCacheConfig {
    fn default() -> Self {
        Self {
            capacity_bytes: 0,
            max_entries: 4,
            ttl_seconds: 300,
        }
    }
}
impl PrefixCacheConfig {
    /// Configure up to four prefixes with a five-minute lifetime.
    ///
    /// # Errors
    /// Rejects capacities over 4 GiB.
    ///
    /// ```
    /// # use clef_rs_core::runtime::PrefixCacheConfig;
    /// let cache = PrefixCacheConfig::new(512 * 1024 * 1024)?;
    /// assert_eq!(cache.max_entries, 4);
    /// # Ok::<(), clef_rs_core::Error>(())
    /// ```
    pub fn new(capacity_bytes: u64) -> Result<Self> {
        let config = Self {
            capacity_bytes,
            ..Self::default()
        };
        config.validate()?;
        Ok(config)
    }
    /// Set the maximum retained prefixes (1..=16).
    ///
    /// # Errors
    /// Rejects values outside 1..=16.
    pub fn with_max_entries(mut self, max_entries: usize) -> Result<Self> {
        self.max_entries = max_entries;
        self.validate()?;
        Ok(self)
    }
    /// Set the absolute state lifetime in seconds (1..=3600).
    ///
    /// # Errors
    /// Rejects values outside 1..=3600.
    pub fn with_ttl_seconds(mut self, ttl_seconds: u64) -> Result<Self> {
        self.ttl_seconds = ttl_seconds;
        self.validate()?;
        Ok(self)
    }
    pub(crate) fn validate(&self) -> Result<()> {
        if self.capacity_bytes > 4 * 1024 * 1024 * 1024
            || !(1..=16).contains(&self.max_entries)
            || !(1..=3600).contains(&self.ttl_seconds)
        {
            return Err(Error::InvalidRequest("prefix cache configuration".into()));
        }
        Ok(())
    }
    pub(crate) fn reservation(&self) -> Result<u64> {
        self.validate()?;
        // Existing entries plus a staged replacement, each with the planner's 20% reserve.
        self.capacity_bytes
            .checked_mul(12)
            .map(|n| n / 5)
            .ok_or(Error::InsufficientMemory)
    }
}
/// Aggregate observations; never expose principals, tokens, or tensor contents.
#[derive(Debug, Clone, Copy, Default, Serialize)]
#[serde(rename_all = "camelCase")]
#[non_exhaustive]
pub struct PrefixCacheStats {
    /// Successful decisions that reused a prefix.
    pub hits: u64,
    /// Successful decisions that captured a prefix.
    pub captures: u64,
    /// Total tokens skipped by successful decisions.
    pub reused_tokens: u64,
    /// Prefix writes dropped after a failed admission check; the request itself
    /// still succeeded.
    pub publish_errors: u64,
    /// Currently retained entries.
    pub entries: usize,
    /// Retained tensor and key bytes.
    pub bytes: u64,
}
struct Entry {
    principal: String,
    tokens: Vec<u32>,
    state: PrefixState,
    bytes: u64,
    created: Instant,
}
/// Debug output deliberately excludes token and principal data.
impl Debug for Entry {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_struct("PrefixEntry")
            .field("tokens", &self.tokens.len())
            .field("bytes", &self.bytes)
            .finish_non_exhaustive()
    }
}
#[derive(Debug, Default)]
pub(crate) struct PrefixCache {
    config: PrefixCacheConfig,
    entries: VecDeque<Entry>,
    stats: PrefixCacheStats,
}
#[derive(Debug)]
pub(crate) enum Reuse {
    Bypass,
    Capture(usize),
    /// A usable prefix was found. `extend` is the number of additional tokens
    /// to capture (relative to this request's suffix) so the retained entry
    /// grows toward the next block boundary; `None` means the entry already
    /// covers the cacheable prefix.
    Hit {
        state: PrefixState,
        extend: Option<usize>,
    },
}
impl PrefixCache {
    pub fn configure(&mut self, config: PrefixCacheConfig) {
        if let Err(error) = config.validate() {
            tracing::warn!(%error, "rejecting invalid prefix cache configuration");
            return;
        }
        self.config = config;
        self.clear();
    }
    pub fn clear(&mut self) {
        self.entries.clear();
        self.stats = PrefixCacheStats::default();
    }
    pub fn stats(&self) -> PrefixCacheStats {
        PrefixCacheStats {
            entries: self.entries.len(),
            bytes: self.entries.iter().map(|e| e.bytes).sum(),
            ..self.stats
        }
    }
    pub fn select(
        &mut self,
        principal: &str,
        ids: &[u32],
        state_end: usize,
        estimate: impl FnOnce(usize) -> Result<u64>,
    ) -> Result<Reuse> {
        if self.config.capacity_bytes == 0 {
            return Ok(Reuse::Bypass);
        }
        let now = Instant::now();
        self.entries.retain(|entry| {
            now.duration_since(entry.created) < Duration::from_secs(self.config.ttl_seconds)
        });
        if state_end < MIN_PREFIX {
            return Ok(Reuse::Bypass);
        }
        let hit = self
            .entries
            .iter()
            .enumerate()
            .filter(|(_, entry)| {
                entry.principal == principal
                    && entry.tokens.len() <= state_end
                    && ids.get(..entry.tokens.len()) == Some(entry.tokens.as_slice())
            })
            .max_by_key(|(_, entry)| entry.tokens.len())
            .map(|(index, _)| index);
        if let Some(index) = hit {
            // The index was produced by enumerating `self.entries` just above,
            // so removal is infallible in practice; fall through to capture on
            // the impossible `None` instead of inventing an error variant.
            if let Some(entry) = self.entries.remove(index) {
                let covered = entry.tokens.len();
                let state = entry.state.clone();
                self.entries.push_back(entry);
                // Extend the retained prefix toward the next block boundary when
                // the request carries enough new tokens; the lengthened entry
                // subsumes the old one on publish.
                let extend = if ids.is_empty() {
                    None
                } else {
                    let target = (state_end / PREFIX_BLOCK * PREFIX_BLOCK).min(ids.len() - 1);
                    (target > covered).then_some(target - covered)
                };
                return Ok(Reuse::Hit { state, extend });
            }
            debug_assert!(false, "prefix cache hit index became invalid");
        }
        let tokens = state_end / PREFIX_BLOCK * PREFIX_BLOCK;
        if tokens >= ids.len() {
            // `prefill` rejects capturing the whole request, so there is nothing
            // worth retaining; skip the cache instead of erroring later.
            return Ok(Reuse::Bypass);
        }
        let estimated = estimate(tokens)?
            .checked_add(Self::key_bytes(principal, tokens)?)
            .ok_or(Error::InsufficientMemory)?;
        Ok(if estimated <= self.config.capacity_bytes {
            Reuse::Capture(tokens)
        } else {
            Reuse::Bypass
        })
    }
    fn key_bytes(principal: &str, tokens: usize) -> Result<u64> {
        tokens
            .checked_mul(4)
            .and_then(|n| n.checked_add(principal.len()))
            .and_then(|n| n.checked_add(1024))
            .and_then(|n| u64::try_from(n).ok())
            .ok_or(Error::InsufficientMemory)
    }
    pub fn record_hit(&mut self, tokens: usize) {
        self.stats.hits = self.stats.hits.saturating_add(1);
        self.stats.reused_tokens = self.stats.reused_tokens.saturating_add(tokens as u64);
    }
    pub fn record_publish_error(&mut self) {
        self.stats.publish_errors = self.stats.publish_errors.saturating_add(1);
    }
    pub fn publish(&mut self, principal: &str, ids: &[u32], state: PrefixState) -> Result<()> {
        let tokens = state.hidden.dim(0)?;
        let bytes = state
            .bytes()?
            .checked_add(Self::key_bytes(principal, tokens)?)
            .ok_or(Error::InsufficientMemory)?;
        if bytes > self.config.capacity_bytes {
            return Err(Error::InsufficientMemory);
        }
        let entry_tokens: Vec<u32> = ids
            .get(..tokens)
            .ok_or_else(|| Error::InvalidRequest("prefix boundary".into()))?
            .to_vec();
        // An extended entry subsumes shorter entries for the same principal
        // whose tokens match its prefix; drop those (and exact duplicates) so
        // repeated publishes do not accumulate stale entries.
        self.entries.retain(|entry| {
            !(entry.principal == principal
                && entry.tokens.len() <= entry_tokens.len()
                && entry_tokens.starts_with(&entry.tokens))
        });
        let entry = Entry {
            principal: principal.into(),
            tokens: entry_tokens,
            state,
            bytes,
            created: Instant::now(),
        };
        while self.entries.len() >= self.config.max_entries
            || self.stats().bytes > self.config.capacity_bytes - bytes
        {
            self.entries.pop_front();
        }
        self.entries.push_back(entry);
        self.stats.captures = self.stats.captures.saturating_add(1);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, atomic::AtomicBool};

    use candle_core::Device;

    use super::*;
    use crate::models::{
        Control,
        qwen::{Backbone, TextConfig},
        weights::Weights,
    };

    fn fixture() -> Result<PrefixState> {
        let config: TextConfig =
            serde_json::from_slice(include_bytes!("../fixtures/synthetic/backbone-config.json"))?;
        let mut weights = Weights::fixture_on(
            include_bytes!("../fixtures/synthetic/backbone.safetensors"),
            &Device::Cpu,
        )?;
        let backbone = Backbone::load(&mut weights, config)?;
        let control = Control {
            cancel: Arc::new(AtomicBool::new(false)),
            deadline: Instant::now() + Duration::from_secs(60),
        };
        backbone
            .prefill(&vec![1; 513], &control, None, Some(512))?
            .1
            .ok_or(Error::ArtifactMissing)
    }
    #[test]
    fn test_should_isolate_exact_tokens_principals_and_eviction() -> Result<()> {
        let state = fixture()?;
        let bytes = state.bytes()?;
        let mut cache = PrefixCache::default();
        cache.configure(PrefixCacheConfig::new(bytes + 4096)?.with_max_entries(1)?);
        let ids = vec![1; 600];
        assert!(matches!(
            cache.select("alice", &ids, 550, |_| Ok(bytes))?,
            Reuse::Capture(512)
        ));
        assert_eq!(cache.stats().entries, 0); // Selecting a miss never publishes unfinished work.
        cache.publish("alice", &ids, state.clone())?;
        assert!(matches!(
            cache.select("alice", &ids, 550, |_| Ok(bytes))?,
            Reuse::Hit { .. }
        ));
        assert!(matches!(
            cache.select("bob", &ids, 550, |_| Ok(bytes))?,
            Reuse::Capture(512)
        ));
        let changed = vec![2; 600];
        assert!(matches!(
            cache.select("alice", &changed, 550, |_| Ok(bytes))?,
            Reuse::Capture(512)
        ));
        assert!(matches!(
            cache.select("alice", &ids, 511, |_| Ok(bytes))?,
            Reuse::Bypass
        ));
        assert!(matches!(
            cache.select("alice", &ids, 550, |_| Ok(bytes * 2))?,
            Reuse::Hit { .. }
        ));
        assert!(cache.publish("bob", &[], state.clone()).is_err());
        assert_eq!(cache.stats().entries, 1);
        assert!(matches!(
            cache.select("alice", &ids, 550, |_| Ok(bytes))?,
            Reuse::Hit { .. }
        ));
        cache.publish("bob", &ids, state.clone())?;
        assert_eq!(cache.stats().entries, 1);
        assert!(cache.stats().bytes <= cache.config.capacity_bytes);
        assert!(matches!(
            cache.select("alice", &ids, 550, |_| Ok(bytes))?,
            Reuse::Capture(512)
        ));
        assert!(matches!(
            cache.select("bob", &ids, 550, |_| Ok(bytes))?,
            Reuse::Hit { .. }
        ));
        cache.configure(PrefixCacheConfig::new(bytes - 1)?);
        assert!(matches!(
            cache.select("alice", &ids, 550, |_| Ok(bytes))?,
            Reuse::Bypass
        ));
        assert!(cache.publish("alice", &ids, state).is_err());
        assert_eq!(cache.stats().entries, 0);
        Ok(())
    }
    #[test]
    fn test_should_evict_the_least_recently_used_principal_with_two_entries() -> Result<()> {
        let state = fixture()?;
        let bytes = state.bytes()?;
        let ids = vec![1; 600];
        let mut cache = PrefixCache::default();
        cache.configure(PrefixCacheConfig {
            max_entries: 2,
            ..PrefixCacheConfig::new(2 * bytes + 8192)?
        });
        cache.publish("alice", &ids, state.clone())?;
        cache.publish("bob", &ids, state.clone())?;
        assert!(matches!(
            cache.select("alice", &ids, 550, |_| Ok(bytes))?,
            Reuse::Hit { .. }
        ));
        cache.publish("carol", &ids, state)?;
        assert_eq!(cache.stats().entries, 2);
        assert!(cache.stats().bytes <= cache.config.capacity_bytes);
        assert!(matches!(
            cache.select("bob", &ids, 550, |_| Ok(bytes))?,
            Reuse::Capture(512)
        ));
        assert!(matches!(
            cache.select("alice", &ids, 550, |_| Ok(bytes))?,
            Reuse::Hit { .. }
        ));
        assert!(matches!(
            cache.select("carol", &ids, 550, |_| Ok(bytes))?,
            Reuse::Hit { .. }
        ));
        Ok(())
    }
    #[test]
    fn test_should_expire_clear_bound_and_redact_retained_states() -> Result<()> {
        let state = fixture()?;
        let bytes = state.bytes()?;
        let mut cache = PrefixCache::default();
        let ids = vec![1; 600];
        assert!(matches!(
            cache.select("secret-principal", &ids, 550, |_| Ok(bytes))?,
            Reuse::Bypass
        ));
        cache.configure(PrefixCacheConfig::new(bytes + 4096)?);
        cache.publish("secret-principal", &ids, state)?;
        let debug = format!("{cache:?}");
        assert!(!debug.contains("secret-principal"));
        assert!(!debug.contains("Tensor["));
        cache
            .entries
            .front_mut()
            .ok_or(Error::ArtifactMissing)?
            .created = Instant::now()
            .checked_sub(Duration::from_secs(301))
            .ok_or(Error::DeadlineExceeded)?;
        assert!(matches!(
            cache.select("secret-principal", &ids, 550, |_| Ok(bytes))?,
            Reuse::Capture(512)
        ));
        assert_eq!(cache.stats().entries, 0);
        cache.clear();
        assert_eq!(cache.stats().captures, 0);
        assert!(PrefixCacheConfig::new(u64::MAX).is_err());
        for (entries, ttl) in [(0, 300), (17, 300), (1, 0), (1, 3601)] {
            let config = PrefixCacheConfig {
                max_entries: entries,
                ttl_seconds: ttl,
                ..PrefixCacheConfig::default()
            };
            assert!(config.validate().is_err());
        }
        assert_eq!(PrefixCacheConfig::new(1024)?.reservation()?, 2457);
        Ok(())
    }
    #[test]
    fn test_should_build_cache_config_with_builders() -> Result<()> {
        let config = PrefixCacheConfig::new(1024)?
            .with_max_entries(8)?
            .with_ttl_seconds(60)?;
        assert_eq!(config.max_entries, 8);
        assert_eq!(config.ttl_seconds, 60);
        assert!(PrefixCacheConfig::new(1024)?.with_max_entries(0).is_err());
        assert!(PrefixCacheConfig::new(1024)?.with_max_entries(17).is_err());
        assert!(PrefixCacheConfig::new(1024)?.with_ttl_seconds(0).is_err());
        assert!(
            PrefixCacheConfig::new(1024)?
                .with_ttl_seconds(3601)
                .is_err()
        );
        // Invalid configurations are rejected instead of applied.
        let mut cache = PrefixCache::default();
        cache.configure(PrefixCacheConfig::new(1024)?);
        cache.configure(PrefixCacheConfig {
            max_entries: 0,
            ..PrefixCacheConfig::new(1024)?
        });
        assert_eq!(cache.config.max_entries, 4);
        Ok(())
    }
    #[test]
    fn test_should_extend_hits_and_bypass_full_captures() -> Result<()> {
        let state = fixture()?;
        let bytes = state.bytes()?;
        let mut cache = PrefixCache::default();
        cache.configure(PrefixCacheConfig::new(4 * bytes + 16384)?);
        let ids = vec![1; 700];
        cache.publish("alice", &ids, state.clone())?;
        // 512 tokens covered, 640 cacheable: extend by one 128-token block.
        match cache.select("alice", &ids, 640, |_| Ok(bytes))? {
            Reuse::Hit {
                extend: Some(128), ..
            } => {}
            other => panic!("expected extend by 128, got {other:?}"),
        }
        // Nothing new to capture: no extension.
        match cache.select("alice", &ids, 600, |_| Ok(bytes))? {
            Reuse::Hit { extend: None, .. } => {}
            other => panic!("expected no extension, got {other:?}"),
        }
        // Capturing the whole request is rejected by prefill: bypass instead.
        // (Different content so the existing entry does not hit.)
        let changed = vec![2; 512];
        assert!(matches!(
            cache.select("alice", &changed, 512, |_| Ok(bytes))?,
            Reuse::Bypass
        ));
        // Re-publishing for the same principal and token prefix replaces the
        // old entry instead of accumulating duplicates.
        cache.publish("alice", &ids, state)?;
        assert_eq!(cache.stats().entries, 1);
        Ok(())
    }
}
