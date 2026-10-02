//! Bounded YAML event validation shared by server and benchmark configuration.
//!
//! Call this before a configuration loader constructs or expands a YAML value tree.

use std::collections::HashSet;

use yaml_rust2::{
    parser::{Event, Parser},
    scanner::TScalarStyle,
};

use crate::{Error, Result};

#[derive(Debug)]
enum Container {
    Map {
        keys: HashSet<String>,
        expect_key: bool,
    },
    Sequence,
}
/// Reject YAML expansion, ambiguous keys, excessive depth and oversized input.
/// This checks syntax and resource policy; typed deserialization checks the schema.
///
/// # Errors
/// Returns malformed-YAML, invalid-configuration or resource-limit errors.
///
/// ```
/// use clef_rs_core::configuration::validate_yaml;
/// validate_yaml("schemaVersion: 1\nworkers: [1, 2]")?;
/// assert!(validate_yaml("value: &a [1]\ncopy: *a").is_err());
/// # Ok::<(), clef_rs_core::Error>(())
/// ```
pub fn validate_yaml(source: &str) -> Result<()> {
    if source.len() > 65536 {
        return Err(Error::LimitExceeded("YAML bytes".into()));
    }
    let mut parser = Parser::new_from_str(source);
    let mut stack = Vec::new();
    let mut docs = 0_u8;
    for _ in 0..8192 {
        let (event, _) = parser.next_token()?;
        match event {
            Event::Alias(_) => {
                return Err(Error::InvalidRequest("YAML aliases are forbidden".into()));
            }
            Event::DocumentStart => {
                docs = docs.saturating_add(1);
                if docs > 1 {
                    return Err(Error::InvalidRequest(
                        "multiple YAML documents are forbidden".into(),
                    ));
                }
            }
            Event::Scalar(value, style, anchor, tag) => {
                if anchor != 0 || tag.is_some() || value.len() > 8192 {
                    return Err(Error::InvalidRequest("YAML anchor/tag/scalar limit".into()));
                }
                if let Some(Container::Map { keys, expect_key }) = stack.last_mut() {
                    if *expect_key {
                        if value.is_empty() || !keys.insert(value.clone()) {
                            return Err(Error::InvalidRequest(
                                "duplicate or empty YAML key".into(),
                            ));
                        }
                        if matches!(style, TScalarStyle::Plain)
                            && (value.parse::<f64>().is_ok()
                                || matches!(value.as_str(), "true" | "false" | "null" | "~"))
                        {
                            return Err(Error::InvalidRequest("YAML keys must be strings".into()));
                        }
                    }
                    *expect_key = !*expect_key;
                }
            }
            Event::MappingStart(anchor, ref tag) | Event::SequenceStart(anchor, ref tag) => {
                if anchor != 0 || tag.is_some() || stack.len() >= 16 {
                    return Err(Error::InvalidRequest("YAML anchor/tag/depth limit".into()));
                }
                if let Some(Container::Map { expect_key, .. }) = stack.last_mut() {
                    if *expect_key {
                        return Err(Error::InvalidRequest("YAML keys must be scalars".into()));
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
    Err(Error::LimitExceeded("YAML event count".into()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_should_reject_ambiguous_or_expanding_yaml_before_value_construction() {
        for source in [
            "x: &a [1]\ny: *a",
            "x: 1\nx: 2",
            "x: !custom value",
            "---\nx: 1\n---\nx: 2",
            "? [a, b]\n: c",
            "true: 1",
            "x: [1,",
        ] {
            assert!(validate_yaml(source).is_err(), "{source}");
        }
        assert!(validate_yaml("a:\n  b: 1\n  c: [2,3]").is_ok());
    }

    #[test]
    fn test_should_bound_yaml_bytes_depth_scalars_and_events() {
        assert!(matches!(
            validate_yaml(&" ".repeat(65537)),
            Err(Error::LimitExceeded(_))
        ));
        let nested = format!("x: {}0{}", "[".repeat(17), "]".repeat(17));
        assert!(validate_yaml(&nested).is_err());
        let scalar = format!("x: {}", "a".repeat(8193));
        assert!(validate_yaml(&scalar).is_err());
        let events = format!("x: [{}]", vec!["0"; 8200].join(","));
        assert!(matches!(
            validate_yaml(&events),
            Err(Error::LimitExceeded(_))
        ));
    }
}
