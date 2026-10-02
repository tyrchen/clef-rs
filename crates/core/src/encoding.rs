//! Versioned reference renderer and independently tokenized prompt segments.
use std::{
    fmt::{self, Debug, Formatter},
    path::Path,
    time::{Duration, Instant},
};

use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use tokenizers::Tokenizer;

use crate::{
    Error, Result,
    artifacts::{VerifiedSnapshot, hex, read_bounded},
    types::DecisionRequest,
};

/// Exact prompt contract version.
pub const ENCODING_VERSION: &str = "clef-reference-v1";
const SYSTEM: &str = "Read the complete state and schema. Decide every field jointly. Each answer \
                      must be exactly one of that field's allowed options.";

/// State truncation is an explicit policy.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum Truncation {
    /// Reject any overflow.
    #[default]
    Reject,
    /// Retain only the reference state token prefix.
    ReferencePrefix,
}
/// Checked half-open token span.
#[derive(Debug, Clone, Copy, Serialize)]
pub struct TokenSpan {
    start: usize,
    end: usize,
}
impl TokenSpan {
    pub(crate) fn new(start: usize, end: usize, len: usize) -> Result<Self> {
        if start >= end || end > len {
            return Err(Error::InvalidRequest(
                "empty or out-of-range token span".into(),
            ));
        }
        Ok(Self { start, end })
    }
    /// Inclusive start token.
    #[must_use]
    pub const fn start(self) -> usize {
        self.start
    }
    /// Exclusive end token.
    #[must_use]
    pub const fn end(self) -> usize {
        self.end
    }
    pub(crate) const fn len(self) -> usize {
        self.end - self.start
    }
}
#[derive(Debug, Clone)]
pub(crate) struct EncodedQuestion {
    pub kind: u32,
    pub instruction: TokenSpan,
    pub options: Vec<TokenSpan>,
}
/// Encoded record, without exposing mutable spans or tokens.
#[derive(Clone)]
pub struct EncodedRecord {
    pub(crate) ids: Vec<u32>,
    pub(crate) questions: Vec<EncodedQuestion>,
    pub(crate) truncated: usize,
    #[cfg(feature = "vision")]
    pub(crate) images: Vec<crate::media::PreparedImage>,
    #[cfg(feature = "vision")]
    pub(crate) media_spans: Vec<TokenSpan>,
    #[cfg(feature = "vision")]
    pub(crate) positions: Vec<[usize; 3]>,
}
impl Debug for EncodedRecord {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_struct("EncodedRecord")
            .field("tokens", &self.ids.len())
            .field("fields", &self.questions.len())
            .finish_non_exhaustive()
    }
}
impl EncodedRecord {
    /// Full prompt length.
    #[must_use]
    pub fn token_count(&self) -> usize {
        self.ids.len()
    }
}
/// Tokenizer owned by the engine; never interprets downloaded templates.
#[derive(Clone)]
pub struct Encoder {
    tokenizer: Tokenizer,
}
impl Debug for Encoder {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_struct("Encoder").finish_non_exhaustive()
    }
}
impl Encoder {
    /// Create an encoder from the tokenizer in a verified pinned snapshot.
    ///
    /// # Errors
    /// Returns an error for missing, changed or malformed tokenizer bytes.
    pub fn from_snapshot(snapshot: &VerifiedSnapshot) -> Result<Self> {
        Self::load(&snapshot.file("tokenizer.json")?)
    }
    /// Load a reviewed tokenizer JSON from a verified local snapshot.
    ///
    /// # Errors
    /// Returns an error when tokenizer loading fails.
    pub fn load(path: &Path) -> Result<Self> {
        let bytes = read_bounded(
            path,
            32 * 1024 * 1024,
            Instant::now() + Duration::from_secs(30),
        )?;
        if hex(&Sha256::digest(&bytes))
            != "06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523"
        {
            return Err(Error::IntegrityMismatch("tokenizer identity".into()));
        }
        let tokenizer = Tokenizer::from_bytes(bytes)
            .map_err(|_| Error::InvalidRequest("invalid tokenizer".into()))?;
        Ok(Self { tokenizer })
    }
    fn tokens(&self, s: &str) -> Result<Vec<u32>> {
        let e = self
            .tokenizer
            .encode(s, false)
            .map_err(|_| Error::InvalidRequest("tokenization failed".into()))?;
        Ok(e.get_ids().to_vec())
    }
    /// Encode one complete record; question order remains meaningful.
    ///
    /// # Errors
    /// Rejects context overflow, empty spans, or tokenizer errors.
    #[allow(
        clippy::too_many_lines,
        reason = "complete independently-tokenized reference record remains below the repository \
                  150-line ceiling"
    )]
    pub fn encode(
        &self,
        request: &DecisionRequest,
        max_tokens: usize,
        truncation: Truncation,
        max_state_tokens: Option<usize>,
    ) -> Result<EncodedRecord> {
        if max_tokens == 0 || max_tokens > 16384 {
            return Err(Error::UnsupportedCapability(
                "context must be 1..16384 tokens".into(),
            ));
        }
        let mut schema = self.tokens("\n\nSCHEMA FIELDS:\n")?;
        let mut raw_questions = Vec::new();
        for (index, q) in request.questions.iter().enumerate() {
            schema.extend(self.tokens(&format!(
                "\nFIELD {}\nID: {}\nTYPE: {}\nINSTRUCTION: ",
                index + 1,
                q.id.as_str(),
                q.kind.name()
            ))?);
            let start = schema.len();
            schema.extend(self.tokens(&render(&q.instructions)?)?);
            let instruction = TokenSpan::new(start, schema.len(), schema.len())?;
            schema.extend(self.tokens("\nALLOWED OPTIONS:\n")?);
            let mut options = Vec::new();
            for (index, (id, description)) in q.encoded_options().iter().enumerate() {
                schema.extend(self.tokens(&format!("OPTION {}: ", index + 1))?);
                let start = schema.len();
                let mut semantic = serde_json::Map::new();
                semantic.insert("option_id".into(), Value::String(id.clone()));
                if !description.is_null() {
                    semantic.insert("description".into(), description.clone());
                }
                schema.extend(self.tokens(&render(&Value::Object(semantic))?)?);
                options.push(TokenSpan::new(start, schema.len(), schema.len())?);
                schema.extend(self.tokens("\n")?);
            }
            schema.extend(self.tokens("END FIELD\n")?);
            raw_questions.push(EncodedQuestion {
                kind: q.kind.index(),
                instruction,
                options,
            });
        }
        let mut ids = self.tokens(&format!(
            "<|im_start|>system\n{SYSTEM}<|im_end|>\n<|im_start|>user\nSTATE:\n"
        ))?;
        #[cfg(feature = "vision")]
        let images = request
            .images
            .iter()
            .map(crate::media::Image::prepare)
            .collect::<Result<Vec<_>>>()?;
        #[cfg(feature = "vision")]
        let mut media_spans = Vec::new();
        #[cfg(feature = "vision")]
        if !images.is_empty() {
            let mut placeholder = String::new();
            let mut visual_tokens = 0;
            for image in &images {
                let count = image.coordinates.len() / 4;
                visual_tokens += count;
                if visual_tokens > 4096 {
                    return Err(Error::LimitExceeded("aggregate visual tokens".into()));
                }
                placeholder.push_str("<|vision_start|>");
                placeholder.push_str(&"<|image_pad|>".repeat(count));
                placeholder.push_str("<|vision_end|>");
            }
            placeholder.push('\n');
            let media_ids = self.tokens(&placeholder)?;
            let prefix = ids.len();
            let mut cursor = 0;
            for image in &images {
                while media_ids.get(cursor).is_some_and(|id| *id != 248_056) {
                    cursor += 1;
                }
                let count = image.coordinates.len() / 4;
                if media_ids
                    .get(cursor..cursor + count)
                    .is_none_or(|tokens| tokens.iter().any(|id| *id != 248_056))
                {
                    return Err(Error::InvalidRequest("media token expansion".into()));
                }
                media_spans.push(TokenSpan::new(
                    prefix + cursor,
                    prefix + cursor + count,
                    prefix + media_ids.len(),
                )?);
                cursor += count;
            }
            ids.extend(media_ids);
        }
        let suffix = self.tokens(
            "\n<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\nJOINT SCHEMA DECISIONS:",
        )?;
        let fixed = ids
            .len()
            .checked_add(schema.len())
            .and_then(|n| n.checked_add(suffix.len()))
            .ok_or_else(|| Error::LimitExceeded("token overflow".into()))?;
        if fixed > max_tokens {
            return Err(Error::LimitExceeded("schema exceeds context".into()));
        }
        let mut state = self.tokens(&render(request.state.value())?)?;
        let original = state.len();
        let available = max_tokens - fixed;
        let allowed = available.min(max_state_tokens.unwrap_or(usize::MAX));
        if original > allowed {
            if matches!(truncation, Truncation::Reject) {
                return Err(Error::LimitExceeded("encoded context".into()));
            }
            state.truncate(allowed);
        }
        ids.extend(state);
        let offset = ids.len();
        ids.extend(schema);
        ids.extend(suffix);
        for q in &mut raw_questions {
            q.instruction = shifted(q.instruction, offset, ids.len())?;
            for span in &mut q.options {
                *span = shifted(*span, offset, ids.len())?;
            }
        }
        #[cfg(feature = "vision")]
        let positions = media_positions(ids.len(), &images, &media_spans)?;
        Ok(EncodedRecord {
            ids,
            questions: raw_questions,
            truncated: original - allowed.min(original),
            #[cfg(feature = "vision")]
            images,
            #[cfg(feature = "vision")]
            media_spans,
            #[cfg(feature = "vision")]
            positions,
        })
    }
}
fn shifted(s: TokenSpan, offset: usize, len: usize) -> Result<TokenSpan> {
    let start = s
        .start
        .checked_add(offset)
        .ok_or_else(|| Error::LimitExceeded("span overflow".into()))?;
    let end = s
        .end
        .checked_add(offset)
        .ok_or_else(|| Error::LimitExceeded("span overflow".into()))?;
    TokenSpan::new(start, end, len)
}
/// Python-compatible semantic rendering with recursively sorted object keys.
///
/// Strings render directly; nested strings use JSON quoting.
///
/// # Errors
/// Returns an error for nonrepresentable numeric values.
///
/// ```
/// use clef_rs_core::encoding::render;
/// assert_eq!(render(&serde_json::json!({"z":1,"a":"中文"}))?, "{\"a\":\"中文\",\"z\":1}");
/// # Ok::<(), clef_rs_core::Error>(())
/// ```
pub fn render(v: &Value) -> Result<String> {
    fn json(v: &Value, out: &mut String) -> Result<()> {
        match v {
            Value::Object(o) => {
                out.push('{');
                let mut entries: Vec<_> = o.iter().collect();
                entries.sort_by(|a, b| a.0.cmp(b.0));
                for (i, (k, v)) in entries.into_iter().enumerate() {
                    if i > 0 {
                        out.push(',');
                    }
                    out.push_str(&serde_json::to_string(k)?);
                    out.push(':');
                    json(v, out)?;
                }
                out.push('}');
            }
            Value::Array(a) => {
                out.push('[');
                for (i, v) in a.iter().enumerate() {
                    if i > 0 {
                        out.push(',');
                    }
                    json(v, out)?;
                }
                out.push(']');
            }
            Value::Number(n) if n.is_f64() => out.push_str(&python_float(
                n.as_f64()
                    .ok_or_else(|| Error::InvalidRequest("invalid float".into()))?,
            )),
            _ => out.push_str(&serde_json::to_string(v)?),
        }
        Ok(())
    }
    if let Value::String(s) = v {
        return Ok(s.clone());
    }
    let mut out = String::new();
    json(v, &mut out)?;
    Ok(out)
}
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    clippy::many_single_char_names,
    reason = "binary64 decimal representations contain fewer than 400 characters and bounded \
              exponents"
)]
fn python_float(v: f64) -> String {
    if v == 0.0 {
        return if v.is_sign_negative() {
            "-0.0".into()
        } else {
            "0.0".into()
        };
    }
    let s = format!("{v:?}");
    let (mantissa, exp) = s.split_once('e').map_or((s.as_str(), 0), |(m, e)| {
        (m, e.parse::<i32>().unwrap_or_default())
    });
    let negative = mantissa.starts_with('-');
    let m = mantissa.trim_start_matches('-');
    let point = m.find('.').unwrap_or(m.len()) as i32;
    let mut digits: String = m.chars().filter(|c| *c != '.').collect();
    let leading = digits.bytes().take_while(|c| *c == b'0').count();
    digits.drain(..leading);
    while digits.ends_with('0') && digits.len() > 1 {
        digits.pop();
    }
    let exponent = exp + point - leading as i32 - 1;
    let sign = if negative { "-" } else { "" };
    if (-4..16).contains(&exponent) {
        let decimal = exponent + 1;
        if decimal <= 0 {
            format!(
                "{sign}0.{}{digits}",
                "0".repeat(decimal.unsigned_abs() as usize)
            )
        } else if decimal as usize >= digits.len() {
            format!(
                "{sign}{digits}{}.0",
                "0".repeat(decimal as usize - digits.len())
            )
        } else {
            let (a, b) = digits.split_at(decimal as usize);
            format!("{sign}{a}.{b}")
        }
    } else {
        let mut d = digits.chars();
        let first = d.next().unwrap_or('0');
        let rest: String = d.collect();
        let fraction = if rest.is_empty() {
            String::new()
        } else {
            format!(".{rest}")
        };
        format!(
            "{sign}{first}{fraction}e{}{:02}",
            if exponent >= 0 { "+" } else { "-" },
            exponent.unsigned_abs()
        )
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;
    #[rstest]
    #[case(1e-5, "1e-05")]
    #[case(1e16, "1e+16")]
    #[case(1e-4, "0.0001")]
    #[case(-0.0,"-0.0")]
    #[case(1.5, "1.5")]
    fn test_should_render_python_floats(#[case] n: f64, #[case] expected: &str) {
        assert_eq!(python_float(n), expected);
    }
    #[test]
    fn test_should_reject_empty_spans() {
        assert!(TokenSpan::new(1, 1, 3).is_err());
        assert!(TokenSpan::new(1, 4, 3).is_err());
    }
}

#[cfg(test)]
mod parity {
    use super::*;
    #[test]
    #[ignore = "requires the pinned tokenizer; run make verify-parity"]
    fn test_should_match_reference_token_ids_and_spans() -> Result<()> {
        let path = std::env::var_os("CLEF_TOKENIZER").ok_or(Error::ArtifactMissing)?;
        let encoder = Encoder::load(Path::new(&path))?;
        let cases: Vec<Value> =
            serde_json::from_slice(include_bytes!("../fixtures/synthetic/encoding.json"))?;
        for case in cases {
            let request = DecisionRequest::from_json(&serde_json::to_vec(
                case.get("request").ok_or(Error::ArtifactMissing)?,
            )?)?;
            let record = encoder.encode(&request, 4096, Truncation::Reject, None)?;
            let expected: Vec<u32> =
                serde_json::from_value(case.get("ids").ok_or(Error::ArtifactMissing)?.clone())?;
            assert_eq!(record.ids, expected);
            let qs = case
                .get("questions")
                .and_then(Value::as_array)
                .ok_or(Error::ArtifactMissing)?;
            for (actual, expected) in record.questions.iter().zip(qs) {
                let instruction: [usize; 2] = serde_json::from_value(
                    expected
                        .get("instruction")
                        .ok_or(Error::ArtifactMissing)?
                        .clone(),
                )?;
                assert_eq!(
                    [actual.instruction.start, actual.instruction.end],
                    instruction
                );
                let options: Vec<[usize; 2]> = serde_json::from_value(
                    expected
                        .get("options")
                        .ok_or(Error::ArtifactMissing)?
                        .clone(),
                )?;
                assert_eq!(
                    actual
                        .options
                        .iter()
                        .map(|s| [s.start, s.end])
                        .collect::<Vec<_>>(),
                    options
                );
            }
            assert!(
                encoder
                    .encode(&request, 10, Truncation::ReferencePrefix, None)
                    .is_err()
            );
            let truncated = encoder.encode(&request, 4096, Truncation::ReferencePrefix, Some(0))?;
            assert!(truncated.truncated > 0);
        }
        Ok(())
    }
}

#[cfg(feature = "vision")]
fn media_positions(
    tokens: usize,
    images: &[crate::media::PreparedImage],
    spans: &[TokenSpan],
) -> Result<Vec<[usize; 3]>> {
    let mut result = Vec::with_capacity(tokens);
    let mut position = 0;
    let mut cursor = 0;
    for (image, span) in images.iter().zip(spans) {
        while cursor < span.start() {
            result.push([position; 3]);
            position += 1;
            cursor += 1;
        }
        let height = image
            .grid
            .get(1)
            .copied()
            .ok_or_else(|| Error::InvalidRequest("media grid".into()))?
            / 2;
        let width = image
            .grid
            .get(2)
            .copied()
            .ok_or_else(|| Error::InvalidRequest("media grid".into()))?
            / 2;
        for row in 0..height {
            for col in 0..width {
                result.push([position, position + row, position + col]);
                cursor += 1;
            }
        }
        if cursor != span.end() {
            return Err(Error::InvalidRequest("media position count".into()));
        }
        position += height.max(width);
    }
    while cursor < tokens {
        result.push([position; 3]);
        position += 1;
        cursor += 1;
    }
    Ok(result)
}

#[cfg(all(test, feature = "vision"))]
mod media_parity {
    use serde::Deserialize;

    use super::*;
    use crate::media::PreparedImage;
    #[derive(Deserialize)]
    struct Fixture {
        tokens: usize,
        grids: Vec<[usize; 3]>,
        spans: Vec<[usize; 2]>,
        positions: Vec<[usize; 3]>,
    }
    #[test]
    fn test_should_match_four_image_rotary_positions_without_resetting_text_offset() -> Result<()> {
        let fixture: Fixture =
            serde_json::from_slice(include_bytes!("../fixtures/synthetic/media-positions.json"))?;
        let images: Vec<_> = fixture
            .grids
            .into_iter()
            .map(|grid| PreparedImage {
                grid,
                patches: Vec::new(),
                coordinates: Vec::new(),
            })
            .collect();
        let spans = fixture
            .spans
            .into_iter()
            .map(|[start, end]| TokenSpan::new(start, end, fixture.tokens))
            .collect::<Result<Vec<_>>>()?;
        assert_eq!(
            media_positions(fixture.tokens, &images, &spans)?,
            fixture.positions
        );
        Ok(())
    }
}
