//! Strict bounded YAML configuration and cross-field validation.
// Blocking filesystem work runs during startup, on the direct caller, or in spawn_blocking.
#![allow(
    clippy::disallowed_methods,
    clippy::disallowed_types,
    reason = "bounded synchronous IO is required for the direct engine and advisory file leases"
)]

use std::{
    collections::{HashMap, HashSet},
    fs::File as FsFile,
    io::Read,
    net::SocketAddr,
    path::{Path, PathBuf},
    time::{SystemTime, UNIX_EPOCH},
};

use anyhow::{Context, Result, bail};
use clef_rs_core::{
    runtime::{ExecutionProfile, RuntimeConfig},
    types::{CommitRevision, Identifier, ModelPreset},
};
use config::{Config, File, FileFormat};
use serde::Deserialize;
use yaml_rust2::parser::{Event, Parser};

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub(crate) struct Settings {
    pub schema_version: u32,
    pub cache: Cache,
    pub models: Vec<Model>,
    pub runtime: RuntimeConfig,
    pub http: Http,
    pub auth: Auth,
}
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub(crate) struct Cache {
    pub root: PathBuf,
    pub offline: bool,
    pub max_bytes: u64,
}
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub(crate) struct Model {
    pub alias: String,
    pub preset: ModelPreset,
    pub revision: CommitRevision,
    pub required: bool,
    pub execution: ExecutionProfile,
}
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub(crate) struct Http {
    pub bind: SocketAddr,
    pub max_body_bytes: usize,
    pub request_read_timeout_ms: u64,
    pub rate_limit_per_principal_per_minute: u32,
    #[serde(default)]
    pub authenticated_tls_ingress: bool,
    #[serde(default = "default_connections")]
    pub max_connections: usize,
}
const fn default_connections() -> usize {
    128
}
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub(crate) struct Auth {
    pub mode: String,
    pub issuer: String,
    pub audience: String,
    pub jwks_file: PathBuf,
    pub keys_valid_until_unix_seconds: u64,
    pub allowed_algorithms: Vec<String>,
    pub required_scope: String,
    pub observation_scope: String,
    #[serde(default = "default_type")]
    pub access_token_type: String,
}
fn default_type() -> String {
    "at+jwt".into()
}
impl Settings {
    pub fn load(path: &Path) -> Result<Self> {
        let metadata = std::fs::metadata(path).context("read configuration metadata")?;
        if !metadata.is_file() || metadata.len() > 65536 {
            bail!("configuration must be a regular file of at most 64 KiB");
        }
        let source = String::from_utf8(read_file(path, 65536)?).context("configuration UTF-8")?;
        preflight(&source)?;
        let mut builder = Config::builder().add_source(File::from_str(&source, FileFormat::Yaml));
        // Only these explicit operator overrides are accepted; no ambient Hub/proxy settings.
        if let Ok(bind) = std::env::var("CLEF_HTTP_BIND") {
            builder = builder.set_override("http.bind", bind)?;
        }
        if let Ok(root) = std::env::var("CLEF_CACHE_ROOT") {
            builder = builder.set_override("cache.root", root)?;
        }
        let settings: Self = builder
            .build()?
            .try_deserialize()
            .context("decode strict configuration")?;
        settings.validate()?;
        Ok(settings)
    }
    fn validate(&self) -> Result<()> {
        if self.schema_version != 1
            || self.models.is_empty()
            || self.models.len() > 8
            || self.cache.max_bytes == 0
        {
            bail!("invalid schema version, model count, or cache budget");
        }
        let mut aliases = HashSet::new();
        let mut devices = HashMap::new();
        for model in &self.models {
            let _: Identifier = model.alias.parse()?;
            if !model.required {
                bail!("Flash v1 requires every configured model at startup");
            }
            if !aliases.insert(&model.alias) || model.revision.as_str() != model.preset.revision() {
                bail!("duplicate alias or unreviewed revision");
            }
            model.execution.validate()?;
            let device = format!("{:?}:{}", model.execution.device, model.execution.ordinal);
            let identity = format!(
                "{}:{}",
                model.preset.alias(),
                serde_json::to_string(&model.execution)?
            );
            if devices
                .insert(device, identity.clone())
                .is_some_and(|old| old != identity)
            {
                bail!("distinct execution profiles on one device require a combined capacity plan");
            }
        }
        self.runtime.validate()?;
        if self.http.max_body_bytes == 0
            || self.http.max_body_bytes
                > if cfg!(feature = "vision") {
                    16 * 1024 * 1024
                } else {
                    1024 * 1024
                }
            || !(1..=60000).contains(&self.http.request_read_timeout_ms)
            || !(1..=10000).contains(&self.http.rate_limit_per_principal_per_minute)
            || !(1..=4096).contains(&self.http.max_connections)
        {
            bail!("invalid HTTP limits");
        }
        if !self.http.bind.ip().is_loopback() && !self.http.authenticated_tls_ingress {
            bail!("non-loopback listener requires an authenticated TLS ingress");
        }
        if self
            .models
            .iter()
            .any(|m| m.execution.modality == clef_rs_core::runtime::Modality::Image)
            && (self.http.max_body_bytes < 16 * 1024 * 1024
                || self.runtime.max_payload_bytes < 24 * 1024 * 1024)
        {
            bail!(
                "image serving requires explicit 16 MiB bodies and 24 MiB preparation payload \
                 budget"
            );
        }
        let a = &self.auth;
        if a.mode != "oidc"
            || a.allowed_algorithms != ["RS256"]
            || !a.issuer.starts_with("https://")
            || a.issuer.len() > 256
            || a.audience.is_empty()
            || a.audience.len() > 256
            || a.required_scope.is_empty()
            || a.observation_scope.is_empty()
            || a.required_scope.len() > 256
            || a.observation_scope.len() > 256
            || a.access_token_type.len() > 64
            || a.access_token_type.is_empty()
            || a.keys_valid_until_unix_seconds <= unix_seconds()?
        {
            bail!("invalid or expired authentication configuration");
        }
        Ok(())
    }
}
pub(crate) fn read_file(path: &Path, cap: u64) -> Result<Vec<u8>> {
    let file = FsFile::open(path).context("open bounded local file")?;
    if !file.metadata()?.is_file() {
        bail!("regular file required");
    }
    let mut bytes = Vec::new();
    file.take(cap.checked_add(1).context("file cap overflow")?)
        .read_to_end(&mut bytes)?;
    if bytes.len() as u64 > cap {
        bail!("local file byte limit");
    }
    Ok(bytes)
}
pub(crate) fn unix_seconds() -> Result<u64> {
    Ok(SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs())
}
#[derive(Debug)]
enum Container {
    Map {
        keys: HashSet<String>,
        expect_key: bool,
    },
    Sequence,
}
fn preflight(source: &str) -> Result<()> {
    if source.len() > 65536 {
        bail!("YAML byte limit");
    }
    let mut parser = Parser::new_from_str(source);
    let mut stack = Vec::new();
    let mut docs = 0;
    for _ in 0..8192 {
        let (event, _) = parser.next_token().context("parse bounded YAML events")?;
        match event {
            Event::Alias(_) => bail!("YAML aliases are forbidden"),
            Event::DocumentStart => {
                docs += 1;
                if docs > 1 {
                    bail!("multiple YAML documents are forbidden");
                }
            }
            Event::Scalar(value, style, anchor, tag) => {
                if anchor != 0 || tag.is_some() || value.len() > 8192 {
                    bail!("YAML anchor/tag/scalar limit");
                }
                if let Some(Container::Map { keys, expect_key }) = stack.last_mut() {
                    if *expect_key {
                        if value.is_empty() || !keys.insert(value.clone()) {
                            bail!("duplicate or empty YAML key");
                        }
                        if matches!(style, yaml_rust2::scanner::TScalarStyle::Plain)
                            && (value.parse::<f64>().is_ok()
                                || matches!(value.as_str(), "true" | "false" | "null" | "~"))
                        {
                            bail!("YAML keys must be strings");
                        }
                    }
                    *expect_key = !*expect_key;
                }
            }
            Event::MappingStart(anchor, ref tag) | Event::SequenceStart(anchor, ref tag) => {
                if anchor != 0 || tag.is_some() || stack.len() >= 16 {
                    bail!("YAML anchor/tag/depth limit");
                }
                if let Some(Container::Map { expect_key, .. }) = stack.last_mut() {
                    if *expect_key {
                        bail!("YAML keys must be scalars");
                    }
                    *expect_key = true;
                }
                if matches!(event, Event::MappingStart(..)) {
                    stack.push(Container::Map {
                        keys: HashSet::new(),
                        expect_key: true,
                    });
                } else {
                    stack.push(Container::Sequence);
                }
            }
            Event::MappingEnd | Event::SequenceEnd => {
                stack.pop();
            }
            Event::StreamEnd => return Ok(()),
            _ => {}
        }
    }
    bail!("YAML event limit")
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_should_reject_yaml_expansion_duplicates_and_tags() {
        for source in [
            "x: &a [1]\ny: *a",
            "x: 1\nx: 2",
            "x: !custom value",
            "---\nx: 1\n---\nx: 2",
            "? [a, b]\n: c",
        ] {
            assert!(preflight(source).is_err(), "{source}");
        }
        assert!(preflight("a:\n  b: 1\n  c: [2,3]").is_ok());
    }
    #[test]
    fn test_should_load_example_and_reject_conflicting_device_budgets() -> Result<()> {
        let mut settings = Settings::load(Path::new("../../examples/clef.cpu.yaml"))?;
        let model = settings.models.first_mut().context("example model")?;
        model.required = false;
        assert!(settings.validate().is_err());
        settings
            .models
            .first_mut()
            .context("example model")?
            .required = true;
        let mut alias = settings.models.first().context("example model")?.clone();
        alias.alias = "another-alias".into();
        settings.models.push(alias.clone());
        settings.validate()?;
        alias.alias = "conflicting-alias".into();
        alias.execution.device_budget_bytes = 1;
        settings.models.push(alias);
        assert!(settings.validate().is_err());
        Ok(())
    }
}
