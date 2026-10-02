//! Local-key RS256 resource-server authentication. No discovery or token URLs.
// Blocking filesystem work runs during startup, on the direct caller, or in spawn_blocking.
#![allow(
    clippy::disallowed_methods,
    clippy::disallowed_types,
    reason = "bounded synchronous IO is required for the direct engine and advisory file leases"
)]

use std::{
    collections::{HashMap, HashSet},
    fmt::{self, Debug, Formatter},
};

use anyhow::{Context, Result, bail};
use base64::{Engine, engine::general_purpose::URL_SAFE_NO_PAD};
use clef_rs_core::types::{BoundedJson, Identifier};
use jsonwebtoken::{
    Algorithm, DecodingKey, Validation, decode, decode_header,
    jwk::{AlgorithmParameters, JwkSet, KeyAlgorithm, PublicKeyUse},
};
use serde::Deserialize;

use crate::config::{Auth, unix_seconds};

#[derive(Debug, Clone, Deserialize)]
pub(crate) struct Claims {
    pub sub: String,
    pub scope: String,
    #[serde(default)]
    pub models: Vec<String>,
}
pub(crate) struct Authenticator {
    keys: HashMap<String, DecodingKey>,
    validation: Validation,
    config: Auth,
}
impl Debug for Authenticator {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_struct("Authenticator")
            .field("keys", &self.keys.len())
            .finish_non_exhaustive()
    }
}
impl Authenticator {
    pub fn load(config: Auth) -> Result<Self> {
        let meta = std::fs::metadata(&config.jwks_file).context("read JWKS metadata")?;
        if !meta.is_file() || meta.len() > 65536 {
            bail!("JWKS must be a regular file of at most 64 KiB");
        }
        let data = crate::config::read_file(&config.jwks_file, 65536)?;
        // Reuse duplicate-rejecting bounded parsing before typed JWK decoding.
        clef_rs_core::types::BoundedJson::parse(&data)?;
        let set: JwkSet = serde_json::from_slice(&data).context("decode local JWKS")?;
        if set.keys.is_empty() || set.keys.len() > 16 {
            bail!("JWKS requires 1..16 keys");
        }
        let mut keys = HashMap::new();
        for jwk in set.keys {
            let AlgorithmParameters::RSA(parameters) = &jwk.algorithm else {
                bail!("RSA key required");
            };
            let modulus = URL_SAFE_NO_PAD
                .decode(&parameters.n)
                .context("RSA modulus")?;
            let leading = modulus.first().copied().unwrap_or_default().leading_zeros();
            let bits = modulus
                .len()
                .saturating_mul(8)
                .saturating_sub(usize::try_from(leading)?);
            if !(2048..=8192).contains(&bits) || parameters.e.len() > 16 {
                bail!("RSA key size limit");
            }
            if !matches!(jwk.algorithm, AlgorithmParameters::RSA(_))
                || jwk.common.key_algorithm != Some(KeyAlgorithm::RS256)
                || jwk
                    .common
                    .public_key_use
                    .as_ref()
                    .is_some_and(|u| *u != PublicKeyUse::Signature)
            {
                bail!("only signature RS256 RSA keys are allowed");
            }
            let kid = jwk.common.key_id.as_deref().context("JWKS kid required")?;
            if kid.is_empty() || kid.len() > 64 || keys.contains_key(kid) {
                bail!("invalid/duplicate JWKS kid");
            }
            keys.insert(kid.to_owned(), DecodingKey::from_jwk(&jwk)?);
        }
        let mut validation = Validation::new(Algorithm::RS256);
        validation.leeway = 30;
        validation.validate_exp = true;
        validation.validate_nbf = true;
        validation.set_issuer(&[&config.issuer]);
        validation.set_audience(&[&config.audience]);
        validation.set_required_spec_claims(&["exp", "iss", "aud", "sub"]);
        Ok(Self {
            keys,
            validation,
            config,
        })
    }
    pub fn authenticate(&self, token: &str) -> Result<Claims> {
        if token.len() > 8192
            || token.is_empty()
            || unix_seconds()? >= self.config.keys_valid_until_unix_seconds
        {
            bail!("invalid credentials");
        }
        let parts: Vec<_> = token.split('.').collect();
        if parts.len() != 3 {
            bail!("invalid JWT structure");
        }
        for part in parts.iter().take(2) {
            let decoded = URL_SAFE_NO_PAD.decode(part).context("JWT encoding")?;
            let parsed = BoundedJson::parse(&decoded).context("bounded JWT JSON")?;
            if parsed.value().get("crit").is_some() || parsed.value().get("b64").is_some() {
                bail!("unsupported JWT extension");
            }
            if !parsed.value().is_object() {
                bail!("JWT JSON object required");
            }
        }
        let header = decode_header(token).context("access token header")?;
        if header.alg != Algorithm::RS256
            || header.typ.as_deref() != Some(self.config.access_token_type.as_str())
            || header.jku.is_some()
            || header.x5u.is_some()
        {
            bail!("invalid access token header");
        }
        let kid = header.kid.as_deref().context("access token kid required")?;
        let key = self.keys.get(kid).context("unknown access token kid")?;
        let data = decode::<Claims>(token, key, &self.validation).context("verify access token")?;
        let claims = data.claims;
        if claims.sub.is_empty()
            || claims.sub.len() > 256
            || claims.sub.chars().any(char::is_control)
            || claims.scope.len() > 4096
            || claims.models.len() > 64
            || claims
                .models
                .iter()
                .any(|m| m.len() > 64 || m.parse::<Identifier>().is_err())
        {
            bail!("access token claim limits");
        }
        let scopes: HashSet<_> = claims.scope.split_ascii_whitespace().collect();
        if scopes.len() > 64 || scopes.iter().any(|s| s.len() > 256) {
            bail!("scope limits");
        }
        Ok(claims)
    }
    pub fn permits_decision(&self, claims: &Claims, model: &str) -> bool {
        claims
            .scope
            .split_ascii_whitespace()
            .any(|s| s == self.config.required_scope)
            && claims.models.iter().any(|m| m == model)
    }
    pub fn permits_observation(&self, claims: &Claims) -> bool {
        claims
            .scope
            .split_ascii_whitespace()
            .any(|s| s == self.config.observation_scope)
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use std::fs;

    use jsonwebtoken::{EncodingKey, Header, encode, jwk::Jwk};
    use serde_json::{Value, json};

    use super::*;
    pub(crate) fn authenticator() -> Result<(Authenticator, EncodingKey)> {
        let private = EncodingKey::from_rsa_der(include_bytes!("../fixtures/test-rsa-private.der"));
        let mut jwk = Jwk::from_encoding_key(&private, Algorithm::RS256)?;
        jwk.common.key_id = Some("test-key".into());
        jwk.common.public_key_use = Some(PublicKeyUse::Signature);
        let directory = tempfile::tempdir()?;
        let path = directory.path().join("jwks.json");
        fs::write(&path, serde_json::to_vec(&JwkSet { keys: vec![jwk] })?)?;
        let auth = Authenticator::load(Auth {
            mode: "oidc".into(),
            issuer: "https://issuer.example".into(),
            audience: "clef-rs".into(),
            jwks_file: path,
            keys_valid_until_unix_seconds: unix_seconds()? + 3600,
            allowed_algorithms: vec!["RS256".into()],
            required_scope: "clef:decide".into(),
            observation_scope: "clef:observe".into(),
            access_token_type: "at+jwt".into(),
        })?;
        Ok((auth, private))
    }
    pub(crate) fn claims() -> Result<Value> {
        Ok(
            json!({"iss":"https://issuer.example","aud":"clef-rs","sub":"principal","scope":"clef:decide clef:observe","models":["clef-flash"],"exp":unix_seconds()?+600}),
        )
    }
    pub(crate) fn token(private: &EncodingKey, claims: &Value) -> Result<String> {
        let mut header = Header::new(Algorithm::RS256);
        header.typ = Some("at+jwt".into());
        header.kid = Some("test-key".into());
        Ok(encode(&header, claims, private)?)
    }
    #[test]
    fn test_should_validate_access_token_and_enforce_model_permission() -> Result<()> {
        let (auth, key) = authenticator()?;
        let claims = auth.authenticate(&token(&key, &claims()?)?)?;
        assert!(auth.permits_decision(&claims, "clef-flash"));
        assert!(!auth.permits_decision(&claims, "clef"));
        assert!(auth.permits_observation(&claims));
        Ok(())
    }
    #[test]
    fn test_should_reject_wrong_issuer_audience_expiry_and_type() -> Result<()> {
        let (auth, key) = authenticator()?;
        for (field, value) in [
            ("iss", json!("https://wrong.example")),
            ("aud", json!("wrong")),
            ("exp", json!(unix_seconds()?.saturating_sub(60))),
            ("nbf", json!(unix_seconds()? + 120)),
            ("sub", json!("")),
        ] {
            let mut claims = claims()?;
            claims[field] = value;
            assert!(auth.authenticate(&token(&key, &claims)?).is_err());
        }
        let mut header = Header::new(Algorithm::RS256);
        header.kid = Some("test-key".into());
        assert!(
            auth.authenticate(&encode(&header, &claims()?, &key)?)
                .is_err()
        );
        header.typ = Some("at+jwt".into());
        header.kid = Some("unknown-key".into());
        assert!(
            auth.authenticate(&encode(&header, &claims()?, &key)?)
                .is_err()
        );
        assert!(auth.authenticate(&"x".repeat(8193)).is_err());
        Ok(())
    }
    #[test]
    fn test_should_reject_duplicate_jwt_fields_before_signature_verification() -> Result<()> {
        let (auth, _) = authenticator()?;
        let header = URL_SAFE_NO_PAD
            .encode(br#"{"alg":"RS256","alg":"RS256","typ":"at+jwt","kid":"test-key"}"#);
        let payload = URL_SAFE_NO_PAD.encode(br#"{"sub":"principal"}"#);
        let error = auth
            .authenticate(&format!("{header}.{payload}.AAAA"))
            .err()
            .context("duplicate JWT accepted")?;
        assert!(error.to_string().contains("bounded JWT JSON"));
        Ok(())
    }
}
