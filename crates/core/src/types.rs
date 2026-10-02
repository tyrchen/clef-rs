//! Validated ordered requests, bounded dynamic JSON, and typed decision answers.
use std::{
    collections::HashSet,
    fmt::{self, Debug, Formatter},
    str::FromStr,
};

use serde::{
    Deserialize, Serialize,
    de::{DeserializeSeed, Error as DeError, MapAccess, SeqAccess, Visitor},
};
use serde_json::{Map, Number, Value};

use crate::{Error, Result};

/// Pinned official releases.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ModelPreset {
    /// 9B CLEF Flash release.
    ClefFlash,
}
impl ModelPreset {
    /// Reviewed Hugging Face repository.
    #[must_use]
    pub const fn repository(self) -> &'static str {
        match self {
            Self::ClefFlash => "Cloudflare/clef-flash",
        }
    }
    /// Immutable reviewed commit.
    #[must_use]
    pub const fn revision(self) -> &'static str {
        match self {
            Self::ClefFlash => "17f0b0ad64efb65d273590632833508766b2aae6",
        }
    }
    /// Canonical alias.
    #[must_use]
    pub const fn alias(self) -> &'static str {
        match self {
            Self::ClefFlash => "clef-flash",
        }
    }
}
impl FromStr for ModelPreset {
    type Err = Error;
    fn from_str(s: &str) -> Result<Self> {
        match s {
            "clef-flash" => Ok(Self::ClefFlash),
            _ => Err(Error::InvalidRequest("unknown model preset".into())),
        }
    }
}

/// Validated ASCII identifier, shared by question, option, and model aliases.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize)]
#[serde(transparent)]
pub struct Identifier(String);
impl Identifier {
    /// Identifier text.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}
impl FromStr for Identifier {
    type Err = Error;
    fn from_str(s: &str) -> Result<Self> {
        if s.is_empty()
            || s.len() > 64
            || !s
                .bytes()
                .all(|b| b.is_ascii_alphanumeric() || b == b'_' || b == b'-')
        {
            return Err(Error::InvalidRequest(
                "identifier must contain 1..64 ASCII letters, digits, underscores or hyphens"
                    .into(),
            ));
        }
        Ok(Self(s.to_owned()))
    }
}

/// Immutable full commit revision.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(try_from = "String", into = "String")]
pub struct CommitRevision(String);
impl CommitRevision {
    /// Commit text.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}
impl TryFrom<String> for CommitRevision {
    type Error = Error;
    fn try_from(s: String) -> Result<Self> {
        if s.len() != 40
            || !s
                .bytes()
                .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
        {
            return Err(Error::InvalidRequest(
                "revision must be 40 lowercase hexadecimal bytes".into(),
            ));
        }
        Ok(Self(s))
    }
}
impl From<CommitRevision> for String {
    fn from(r: CommitRevision) -> Self {
        r.0
    }
}
impl FromStr for CommitRevision {
    type Err = Error;
    fn from_str(s: &str) -> Result<Self> {
        Self::try_from(s.to_owned())
    }
}

/// Bounded semantic JSON; debug output omits customer content.
#[derive(Clone)]
pub struct BoundedJson(Value);
impl Debug for BoundedJson {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.write_str("BoundedJson([REDACTED])")
    }
}
impl BoundedJson {
    /// Parse bounded JSON, rejecting duplicate keys during parsing.
    ///
    /// # Errors
    /// Returns an error for malformed, excessive, or invalid semantic JSON.
    ///
    /// ```
    /// use clef_rs_core::types::BoundedJson;
    /// let state = BoundedJson::parse(br#"{"status":"failed"}"#)?;
    /// # Ok::<(), clef_rs_core::Error>(())
    /// ```
    pub fn parse(bytes: &[u8]) -> Result<Self> {
        let value = parse_json(bytes, 512 * 1024)?;
        validate_semantic(&value, 512 * 1024)?;
        Ok(Self(value))
    }
    /// Borrow the validated value.
    #[must_use]
    pub fn value(&self) -> &Value {
        &self.0
    }
}
impl TryFrom<Value> for BoundedJson {
    type Error = Error;
    fn try_from(value: Value) -> Result<Self> {
        let bytes = serde_json::to_vec(&value)?;
        Self::parse(&bytes)
    }
}

struct JsonSeed<'a> {
    depth: usize,
    nodes: &'a mut usize,
    collection_cap: usize,
}
impl<'de> DeserializeSeed<'de> for JsonSeed<'_> {
    type Value = Value;
    fn deserialize<D: serde::Deserializer<'de>>(
        self,
        deserializer: D,
    ) -> std::result::Result<Value, D::Error> {
        *self.nodes += 1;
        if self.depth > 20 || *self.nodes > 16384 {
            return Err(D::Error::custom("JSON nesting/node limit"));
        }
        deserializer.deserialize_any(self)
    }
}
impl<'de> Visitor<'de> for JsonSeed<'_> {
    type Value = Value;
    fn expecting(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.write_str("bounded JSON")
    }
    fn visit_bool<E: DeError>(self, v: bool) -> std::result::Result<Value, E> {
        Ok(Value::Bool(v))
    }
    fn visit_unit<E: DeError>(self) -> std::result::Result<Value, E> {
        Ok(Value::Null)
    }
    fn visit_i64<E: DeError>(self, v: i64) -> std::result::Result<Value, E> {
        if v.unsigned_abs() > 9_007_199_254_740_991 {
            return Err(E::custom("integer outside exact binary64 range"));
        }
        Ok(Value::Number(v.into()))
    }
    fn visit_u64<E: DeError>(self, v: u64) -> std::result::Result<Value, E> {
        if v > 9_007_199_254_740_991 {
            return Err(E::custom("integer outside exact binary64 range"));
        }
        Ok(Value::Number(v.into()))
    }
    fn visit_f64<E: DeError>(self, v: f64) -> std::result::Result<Value, E> {
        if !v.is_finite() || v.abs() > 1e100 {
            return Err(E::custom("float outside allowed range"));
        }
        Number::from_f64(v)
            .map(Value::Number)
            .ok_or_else(|| E::custom("nonfinite number"))
    }
    fn visit_str<E: DeError>(self, v: &str) -> std::result::Result<Value, E> {
        validate_text(v, 3 * 1024 * 1024).map_err(E::custom)?;
        Ok(Value::String(v.into()))
    }
    fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> std::result::Result<Value, A::Error> {
        let mut values = Vec::new();
        while let Some(v) = seq.next_element_seed(JsonSeed {
            depth: self.depth + 1,
            nodes: self.nodes,
            collection_cap: self.collection_cap,
        })? {
            if values.len() == self.collection_cap {
                return Err(A::Error::custom("array length limit"));
            }
            values.push(v);
        }
        Ok(Value::Array(values))
    }
    fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> std::result::Result<Value, A::Error> {
        let mut values = Map::new();
        while let Some(key) = map.next_key::<String>()? {
            validate_text(&key, 256).map_err(A::Error::custom)?;
            if values.len() == self.collection_cap || values.contains_key(&key) {
                return Err(A::Error::custom("duplicate key or object length limit"));
            }
            let value = map.next_value_seed(JsonSeed {
                depth: self.depth + 1,
                nodes: self.nodes,
                collection_cap: self.collection_cap,
            })?;
            values.insert(key, value);
        }
        Ok(Value::Object(values))
    }
}
pub(crate) fn parse_json(bytes: &[u8], cap: usize) -> Result<Value> {
    parse_json_with_collection_cap(bytes, cap, 256)
}
pub(crate) fn parse_json_with_collection_cap(
    bytes: &[u8],
    cap: usize,
    collection_cap: usize,
) -> Result<Value> {
    if bytes.len() > cap {
        return Err(Error::LimitExceeded("JSON bytes".into()));
    }
    let mut d = serde_json::Deserializer::from_slice(bytes);
    let mut nodes = 0;
    let v = JsonSeed {
        depth: 0,
        nodes: &mut nodes,
        collection_cap,
    }
    .deserialize(&mut d)?;
    d.end()?;
    Ok(v)
}
fn validate_text(s: &str, cap: usize) -> Result<()> {
    if s.len() > cap {
        return Err(Error::LimitExceeded("string bytes".into()));
    }
    if s.chars()
        .any(|c| c.is_control() && !matches!(c, '\n' | '\r' | '\t'))
    {
        return Err(Error::InvalidRequest("disallowed control character".into()));
    }
    Ok(())
}
fn validate_semantic(v: &Value, cap: usize) -> Result<()> {
    fn walk(v: &Value, depth: usize, nodes: &mut usize) -> Result<()> {
        *nodes += 1;
        if depth > 16 || *nodes > 4096 {
            return Err(Error::LimitExceeded("semantic JSON structure".into()));
        }
        match v {
            Value::Array(a) => {
                for child in a {
                    walk(child, depth + 1, nodes)?;
                }
            }
            Value::Object(o) => {
                for child in o.values() {
                    walk(child, depth + 1, nodes)?;
                }
            }
            Value::String(s) => validate_text(s, 65536)?,
            _ => {}
        }
        Ok(())
    }
    walk(v, 0, &mut 0)?;
    if crate::encoding::render(v)?.len() > cap {
        return Err(Error::LimitExceeded("semantic bytes".into()));
    }
    Ok(())
}

/// Validated question type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum QuestionType {
    /// Truth probability.
    Noul,
    /// Categorical selection.
    Choice,
    /// Ordinal expected value.
    Score,
}
impl QuestionType {
    pub(crate) const fn index(self) -> u32 {
        match self {
            Self::Noul => 0,
            Self::Choice => 1,
            Self::Score => 2,
        }
    }
    pub(crate) const fn name(self) -> &'static str {
        match self {
            Self::Noul => "noul",
            Self::Choice => "choice",
            Self::Score => "score",
        }
    }
}
/// An ordered validated question; fields are private to prevent invariant bypass.
#[derive(Clone)]
pub struct Question {
    pub(crate) id: Identifier,
    pub(crate) kind: QuestionType,
    pub(crate) instructions: Value,
    pub(crate) options: Vec<(String, Value)>,
}
impl Debug for Question {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_struct("Question")
            .field("id", &self.id)
            .field("kind", &self.kind)
            .finish_non_exhaustive()
    }
}
impl Question {
    /// Stable field ID.
    #[must_use]
    pub fn id(&self) -> &str {
        self.id.as_str()
    }
    /// Answer semantics.
    #[must_use]
    pub fn kind(&self) -> QuestionType {
        self.kind
    }
    pub(crate) fn encoded_options(&self) -> Vec<(String, Value)> {
        let mut result = self.options.clone();
        if self.kind == QuestionType::Choice {
            result.sort_by(|a, b| a.0.cmp(&b.0));
        }
        result
    }
}

/// Validated record with ordered questions and redacted debug output.
#[derive(Clone)]
pub struct DecisionRequest {
    pub(crate) state: BoundedJson,
    pub(crate) questions: Vec<Question>,
    pub(crate) payload_bytes: usize,
    #[cfg(feature = "vision")]
    pub(crate) images: Vec<crate::media::Image>,
}
impl Debug for DecisionRequest {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_struct("DecisionRequest")
            .field("questions", &self.questions.len())
            .field("payload_bytes", &self.payload_bytes)
            .finish_non_exhaustive()
    }
}
impl DecisionRequest {
    /// Attach validated RGB images without a file decoder or base64 envelope.
    ///
    /// # Errors
    /// Rejects more than four images or more than eight megapixels in total.
    #[cfg(feature = "vision")]
    pub fn with_images(mut self, images: Vec<crate::media::Image>) -> Result<Self> {
        if images.len() > 4
            || images
                .iter()
                .map(crate::media::Image::pixels)
                .sum::<usize>()
                > 8_000_000
        {
            return Err(Error::LimitExceeded("image count/pixels".into()));
        }
        let old = self.images.iter().try_fold(0_usize, |n, image| {
            n.checked_add(image.reservation_bytes()?)
                .ok_or_else(|| Error::LimitExceeded("image reservations".into()))
        })?;
        let new = images.iter().try_fold(0_usize, |n, image| {
            n.checked_add(image.reservation_bytes()?)
                .ok_or_else(|| Error::LimitExceeded("image reservations".into()))
        })?;
        self.payload_bytes = self
            .payload_bytes
            .checked_sub(old)
            .and_then(|n| n.checked_add(new))
            .ok_or_else(|| Error::LimitExceeded("image reservations".into()))?;
        self.images = images;
        Ok(self)
    }
    /// Parse a library record, or the `SystemOne` envelope with its model alias.
    ///
    /// # Errors
    /// Rejects malformed/duplicate JSON, unknown fields, invalid questions and limits.
    ///
    /// ```
    /// use clef_rs_core::DecisionRequest;
    /// let request = DecisionRequest::from_json(br#"{"state":"failure","questions":{"urgent":{"type":"noul"}}}"#)?;
    /// # Ok::<(), clef_rs_core::Error>(())
    /// ```
    #[allow(
        clippy::too_many_lines,
        reason = "boundary validation stays together, within the repository 150-line ceiling"
    )]
    pub fn from_json(bytes: &[u8]) -> Result<Self> {
        let v = parse_json(
            bytes,
            if cfg!(feature = "vision") {
                16 * 1024 * 1024
            } else {
                1024 * 1024
            },
        )?;
        let o = v
            .as_object()
            .ok_or_else(|| Error::InvalidRequest("record must be an object".into()))?;
        if o.keys().any(|k| {
            !matches!(
                k.as_str(),
                "model" | "state" | "questions" | "images" | "videos"
            )
        }) {
            return Err(Error::InvalidRequest("unknown envelope field".into()));
        }
        if let Some(model) = o.get("model") {
            let _: Identifier = model
                .as_str()
                .ok_or_else(|| Error::InvalidRequest("model must be a string".into()))?
                .parse()?;
        }
        if o.contains_key("videos") {
            return Err(Error::UnsupportedCapability(
                "video frames are not qualified".into(),
            ));
        }
        #[cfg(not(feature = "vision"))]
        if o.contains_key("images") {
            return Err(Error::UnsupportedCapability(
                "vision feature is disabled".into(),
            ));
        }
        #[cfg(feature = "vision")]
        let images = parse_images(o.get("images"))?;
        if bytes.len() > 1024 * 1024 && !o.contains_key("images") {
            return Err(Error::LimitExceeded("text request bytes".into()));
        }
        let state = o
            .get("state")
            .ok_or_else(|| Error::InvalidRequest("state is required".into()))?;
        validate_semantic(state, 512 * 1024)?;
        let questions = o
            .get("questions")
            .and_then(Value::as_object)
            .ok_or_else(|| Error::InvalidRequest("questions must be an object".into()))?;
        if questions.is_empty() || questions.len() > 32 {
            return Err(Error::LimitExceeded(
                "questions must contain 1..32 fields".into(),
            ));
        }
        let mut entries = Vec::with_capacity(questions.len());
        let mut total_options: usize = 0;
        for (id, raw) in questions {
            let id: Identifier = id.parse()?;
            let q = raw
                .as_object()
                .ok_or_else(|| Error::InvalidRequest("question must be an object".into()))?;
            if q.keys()
                .any(|k| !matches!(k.as_str(), "type" | "instructions" | "criteria"))
            {
                return Err(Error::InvalidRequest("unknown question field".into()));
            }
            let kind = match q.get("type").and_then(Value::as_str) {
                Some("noul") => QuestionType::Noul,
                Some("choice") => QuestionType::Choice,
                Some("score") => QuestionType::Score,
                _ => return Err(Error::InvalidRequest("unknown question type".into())),
            };
            let instructions = match q.get("instructions") {
                None | Some(Value::Null) => Value::String(id.0.clone()),
                Some(Value::String(s)) if s.is_empty() => Value::String(id.0.clone()),
                Some(v) => v.clone(),
            };
            validate_semantic(&instructions, 4096)?;
            let criteria = q.get("criteria");
            let options = match kind {
                QuestionType::Noul => {
                    let mut options = vec![
                        (
                            "true".into(),
                            Value::String("The proposition is true or the answer is yes.".into()),
                        ),
                        (
                            "false".into(),
                            Value::String("The proposition is false or the answer is no.".into()),
                        ),
                    ];
                    if let Some(c) = criteria.filter(|c| !c.is_null()) {
                        let o = c.as_object().ok_or_else(|| {
                            Error::InvalidRequest("noul criteria must be an object".into())
                        })?;
                        if o.keys().any(|k| k != "true" && k != "false") {
                            return Err(Error::InvalidRequest(
                                "noul criteria permit only true and false".into(),
                            ));
                        }
                        for (id, v) in &mut options {
                            if let Some(description) = o.get(id) {
                                *v = description.clone();
                            }
                        }
                    }
                    options
                }
                QuestionType::Choice => {
                    let o = criteria.and_then(Value::as_object).ok_or_else(|| {
                        Error::InvalidRequest("choice criteria must be an object".into())
                    })?;
                    o.iter()
                        .map(|(id, v)| {
                            let _: Identifier = id.parse()?;
                            Ok((id.clone(), v.clone()))
                        })
                        .collect::<Result<Vec<_>>>()?
                }
                QuestionType::Score => criteria
                    .and_then(Value::as_array)
                    .ok_or_else(|| Error::InvalidRequest("score criteria must be an array".into()))?
                    .iter()
                    .enumerate()
                    .map(|(i, v)| (i.to_string(), v.clone()))
                    .collect(),
            };
            if !(2..=64).contains(&options.len()) {
                return Err(Error::LimitExceeded(
                    "options must contain 2..64 entries".into(),
                ));
            }
            for (_, description) in &options {
                validate_semantic(description, 4096)?;
            }
            total_options = total_options
                .checked_add(options.len())
                .ok_or_else(|| Error::LimitExceeded("option count overflow".into()))?;
            if total_options > 512 {
                return Err(Error::LimitExceeded("total option count".into()));
            }
            entries.push(Question {
                id,
                kind,
                instructions,
                options,
            });
        }
        Ok(Self {
            state: BoundedJson(state.clone()),
            questions: entries,
            payload_bytes: {
                #[cfg(feature = "vision")]
                {
                    images.iter().try_fold(bytes.len(), |total, image| {
                        total
                            .checked_add(image.reservation_bytes()?)
                            .ok_or_else(|| Error::LimitExceeded("image reservations".into()))
                    })?
                }
                #[cfg(not(feature = "vision"))]
                {
                    bytes.len()
                }
            },
            #[cfg(feature = "vision")]
            images,
        })
    }
    /// Ordered fields.
    #[must_use]
    pub fn questions(&self) -> &[Question] {
        &self.questions
    }
}

/// Finite checked probability.
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
#[serde(transparent)]
pub struct Probability(f32);
impl Probability {
    /// Unrounded model score.
    #[must_use]
    pub const fn get(self) -> f32 {
        self.0
    }
}
impl TryFrom<f32> for Probability {
    type Error = Error;
    fn try_from(p: f32) -> Result<Self> {
        if !p.is_finite() || !(0.0..=1.0).contains(&p) {
            return Err(Error::InferenceFailed("invalid probability".into()));
        }
        Ok(Self(p))
    }
}

/// One typed answer with unrounded probabilities.
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type", rename_all = "lowercase")]
#[non_exhaustive]
pub enum Answer {
    /// Truth score.
    Noul {
        /// Probability of true.
        noul: Probability,
    },
    /// Categorical answer.
    Choice {
        /// Winner in caller order for ties.
        choice: String,
        /// Winner probability.
        confidence: Probability,
        /// Ordered complete distribution.
        probabilities: Map<String, Value>,
    },
    /// Ordinal answer.
    Score {
        /// Expected ordinal index.
        score: f64,
        /// Largest option probability.
        confidence: Probability,
        /// Caller legend.
        legend: Map<String, Value>,
        /// Complete distribution.
        probabilities: Map<String, Value>,
    },
}
/// Immutable provenance and unrounded ordered decisions.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct DecisionResult {
    /// Ordered answers.
    pub answers: Vec<(String, Answer)>,
    /// Entire encoded input length.
    pub input_tokens: usize,
    /// Dropped state prefix-tail token count.
    pub truncated_state_tokens: usize,
    /// Immutable model revision.
    pub revision: String,
    /// Manifest identity.
    pub manifest_digest: String,
    /// Explicit execution combination.
    pub execution_profile: String,
}
impl DecisionResult {
    /// SystemOne-compatible four-decimal response.
    ///
    /// # Errors
    /// Returns a serialization error if an invariant has been violated.
    pub fn systemone(&self, alias: &str) -> Result<Value> {
        let _: Identifier = alias.parse()?;
        let mut answers = Map::new();
        for (id, answer) in &self.answers {
            let mut value = serde_json::to_value(answer)?;
            round_value(&mut value);
            answers.insert(id.clone(), value);
        }
        Ok(
            serde_json::json!({"model":alias,"answers":answers,"usage":{"input_tokens":self.input_tokens,"output_tokens":0}}),
        )
    }
}
fn round_value(v: &mut Value) {
    match v {
        Value::Number(n) if n.is_f64() => {
            if let Some(x) = n
                .as_f64()
                .and_then(|x| format!("{x:.4}").parse::<f64>().ok())
                .and_then(Number::from_f64)
            {
                *n = x;
            }
        }
        Value::Array(a) => {
            for child in a {
                round_value(child);
            }
        }
        Value::Object(o) => {
            for (key, child) in o {
                if key != "legend" {
                    round_value(child);
                }
            }
        }
        _ => {}
    }
}
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    reason = "lookup entries originate from f32 probabilities; ordinal indices are bounded to 64"
)]
pub(crate) fn answers(
    request: &DecisionRequest,
    logits: &[Vec<f32>],
) -> Result<Vec<(String, Answer)>> {
    if logits.len() != request.questions.len() {
        return Err(Error::InferenceFailed("field count mismatch".into()));
    }
    request
        .questions
        .iter()
        .zip(logits)
        .map(|(q, l)| {
            let opts = q.encoded_options();
            if opts.len() != l.len() || l.iter().any(|x| !x.is_finite()) {
                return Err(Error::InferenceFailed("invalid logits".into()));
            }
            let max = l.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut ps: Vec<f32> = l.iter().map(|v| (v - max).exp()).collect();
            let sum: f32 = ps.iter().sum();
            if !sum.is_finite() || sum <= 0.0 {
                return Err(Error::InferenceFailed("invalid softmax".into()));
            }
            for p in &mut ps {
                *p /= sum;
            }
            let lookup: Map<String, Value> = opts
                .iter()
                .zip(&ps)
                .map(|((id, _), p)| (id.clone(), serde_json::json!(p)))
                .collect();
            let mut probabilities = Map::new();
            let mut seen = HashSet::new();
            let mut winner = (String::new(), -1.0_f32);
            let mut score = 0.0;
            let mut legend = Map::new();
            for (index, (id, desc)) in q.options.iter().enumerate() {
                if !seen.insert(id) {
                    return Err(Error::InferenceFailed("duplicate output option".into()));
                }
                let p = lookup
                    .get(id)
                    .and_then(Value::as_f64)
                    .ok_or_else(|| Error::InferenceFailed("missing option".into()))?
                    as f32;
                Probability::try_from(p)?;
                if p > winner.1 {
                    winner = (id.clone(), p);
                }
                probabilities.insert(id.clone(), serde_json::json!(p));
                score += index as f64 * f64::from(p);
                legend.insert(id.clone(), desc.clone());
            }
            let answer = match q.kind {
                QuestionType::Noul => Answer::Noul {
                    noul: Probability::try_from(
                        *ps.first()
                            .ok_or_else(|| Error::InferenceFailed("missing true option".into()))?,
                    )?,
                },
                QuestionType::Choice => Answer::Choice {
                    choice: winner.0,
                    confidence: Probability::try_from(winner.1)?,
                    probabilities,
                },
                QuestionType::Score => Answer::Score {
                    score,
                    confidence: Probability::try_from(winner.1)?,
                    legend,
                    probabilities,
                },
            };
            Ok((q.id.0.clone(), answer))
        })
        .collect()
}

#[cfg(feature = "vision")]
fn parse_images(value: Option<&Value>) -> Result<Vec<crate::media::Image>> {
    use base64::{Engine, engine::general_purpose::STANDARD};
    use bytes::Bytes;

    use crate::media::Image;
    let Some(value) = value else {
        return Ok(Vec::new());
    };
    let array = value
        .as_array()
        .ok_or_else(|| Error::InvalidRequest("images must be an array".into()))?;
    if array.is_empty() || array.len() > 4 {
        return Err(Error::LimitExceeded("image count".into()));
    }
    let mut images = Vec::with_capacity(array.len());
    let mut file_bytes = 0;
    let mut pixels = 0;
    for value in array {
        let o = value
            .as_object()
            .ok_or_else(|| Error::InvalidRequest("image object".into()))?;
        if o.len() != 2 || !o.contains_key("mediaType") || !o.contains_key("data") {
            return Err(Error::InvalidRequest("image fields".into()));
        }
        let media_type = o
            .get("mediaType")
            .and_then(Value::as_str)
            .ok_or_else(|| Error::InvalidRequest("image MIME".into()))?;
        let data = o
            .get("data")
            .and_then(Value::as_str)
            .ok_or_else(|| Error::InvalidRequest("image base64".into()))?;
        if data.len() > 2_796_204 {
            return Err(Error::LimitExceeded("base64 image bytes".into()));
        }
        let data = STANDARD
            .decode(data)
            .map_err(|_| Error::InvalidRequest("image base64 encoding".into()))?;
        file_bytes += data.len();
        if file_bytes > 8 * 1024 * 1024 {
            return Err(Error::LimitExceeded("image aggregate bytes".into()));
        }
        let image = Image::decode(media_type, &Bytes::from(data))?;
        pixels += image.pixels();
        if pixels > 8_000_000 {
            return Err(Error::LimitExceeded("image aggregate pixels".into()));
        }
        images.push(image);
    }
    Ok(images)
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;

    use super::*;
    #[test]
    fn test_should_reject_duplicate_and_excessive_json() {
        assert!(BoundedJson::parse(br#"{"x":1,"x":2}"#).is_err());
        assert!(BoundedJson::parse(b"9007199254740992").is_err());
        assert!(BoundedJson::parse(br#""\u0000""#).is_err());
        assert!(BoundedJson::parse(br#""\ud800""#).is_err());
    }
    #[test]
    fn test_should_preserve_choice_ties_and_score_expectation() -> Result<()> {
        let r = DecisionRequest::from_json(br#"{"state":null,"questions":{"x":{"type":"choice","criteria":{"z":null,"a":null}},"s":{"type":"score","criteria":["low","high"]}}}"#)?;
        let a = answers(&r, &[vec![0., 0.], vec![0., 0.]])?;
        assert!(
            matches!(a.first().map(|x| &x.1), Some(Answer::Choice { choice, .. }) if choice == "z")
        );
        assert!(
            matches!(a.get(1).map(|x| &x.1), Some(Answer::Score { score, .. }) if (*score - 0.5).abs() < f64::EPSILON)
        );
        Ok(())
    }
    proptest! {
        #[test]
        fn test_should_accept_only_finite_probabilities(p in any::<f32>()) {
            prop_assert_eq!(Probability::try_from(p).is_ok(), p.is_finite() && (0.0..=1.0).contains(&p));
        }
    }
}
