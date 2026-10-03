//! Authenticated transport adapter; encoding and answers remain in the core.
use std::{
    collections::HashMap,
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
    time::{Duration, Instant},
};

use axum::{
    Extension, Json, Router,
    body::to_bytes,
    extract::{Request, State},
    http::{HeaderValue, StatusCode},
    middleware::{self, Next},
    response::{IntoResponse, Response},
    routing::{get, post},
};
use clef_rs_core::{
    DecisionClient, DecisionRequest, encoding::ENCODING_VERSION, runtime::DecisionOptions,
};
use metrics_exporter_prometheus::PrometheusHandle;
use serde_json::Value;
use tokio::{
    sync::{Semaphore, mpsc, oneshot},
    task::JoinHandle,
    time::timeout,
};

use crate::{
    auth::{Authenticator, Claims},
    config::{Http, Model},
    error::ApiError,
};

static REQUESTS: AtomicU64 = AtomicU64::new(1);
#[derive(Debug, Clone)]
pub(crate) struct LoadedModel {
    pub config: Model,
    pub client: DecisionClient,
}
#[derive(Debug, Clone)]
pub(crate) struct AppState {
    pub auth: Arc<Authenticator>,
    pub models: Arc<HashMap<String, LoadedModel>>,
    pub http: Http,
    pub metrics: PrometheusHandle,
    pub rate: mpsc::Sender<RateRequest>,
    pub ingress: Arc<Semaphore>,
    pub decision_timeout: Duration,
}
#[derive(Debug)]
pub(crate) struct RateRequest {
    principal: String,
    reply: oneshot::Sender<bool>,
}
pub(crate) fn rate_limiter(limit: u32) -> (mpsc::Sender<RateRequest>, JoinHandle<()>) {
    let (sender, mut receiver) = mpsc::channel::<RateRequest>(128);
    let task = tokio::spawn(async move {
        let mut windows: HashMap<String, (Instant, u32)> = HashMap::new();
        while let Some(request) = receiver.recv().await {
            let now = Instant::now();
            windows.retain(|_, (start, _)| now.duration_since(*start) < Duration::from_secs(60));
            let allowed = if windows.len() >= 256 && !windows.contains_key(&request.principal) {
                false
            } else {
                let entry = windows.entry(request.principal).or_insert((now, 0));
                if entry.1 >= limit {
                    false
                } else {
                    entry.1 = entry.1.saturating_add(1);
                    true
                }
            };
            let _ = request.reply.send(allowed);
        }
    });
    (sender, task)
}
pub(crate) fn router(state: AppState) -> Router {
    Router::new()
        .route("/v1/systemone", post(decide))
        .route("/v1/models", get(models))
        .route("/livez", get(livez))
        .route("/readyz", get(readyz))
        .route("/metrics", get(metrics))
        .layer(middleware::from_fn_with_state(state.clone(), authenticate))
        .with_state(state)
}
async fn authenticate(State(state): State<AppState>, mut request: Request, next: Next) -> Response {
    let _active = request
        .extensions()
        .get::<axum::extract::ConnectInfo<crate::listener::Connection>>()
        .map(|connection| connection.0.active());
    let id = format!("clef-{:016x}", REQUESTS.fetch_add(1, Ordering::Relaxed));
    let token = request
        .headers()
        .get("authorization")
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.strip_prefix("Bearer "));
    let mut response = match token.and_then(|t| state.auth.authenticate(t).ok()) {
        Some(claims) => {
            let observation = request.uri().path() != "/v1/systemone";
            if (observation && !state.auth.permits_observation(&claims))
                || (!observation
                    && !claims
                        .models
                        .iter()
                        .any(|model| state.auth.permits_decision(&claims, model)))
            {
                ApiError::forbidden().into_response()
            } else if request.headers().len() > 32
                || request.uri().to_string().len() > 2048
                || request.headers().iter().any(|(name, value)| {
                    value.as_bytes().len() > if name == "authorization" { 8192 } else { 256 }
                })
            {
                ApiError::from(clef_rs_core::Error::LimitExceeded("HTTP headers".into()))
                    .into_response()
            } else if request.headers().contains_key("content-encoding") {
                ApiError {
                    status: StatusCode::BAD_REQUEST,
                    code: "compressedRequest",
                    message: "Compressed requests are unsupported.",
                }
                .into_response()
            } else {
                let (reply, receiver) = oneshot::channel();
                if state
                    .rate
                    .try_send(RateRequest {
                        principal: claims.sub.clone(),
                        reply,
                    })
                    .is_err()
                    || timeout(Duration::from_secs(1), receiver)
                        .await
                        .ok()
                        .and_then(Result::ok)
                        != Some(true)
                {
                    ApiError {
                        status: StatusCode::TOO_MANY_REQUESTS,
                        code: "rateLimited",
                        message: "Request rate exceeded.",
                    }
                    .into_response()
                } else {
                    request.extensions_mut().insert(claims);
                    next.run(request).await
                }
            }
        }
        None => ApiError::unauthorized().into_response(),
    };
    if let Ok(value) = HeaderValue::from_str(&id) {
        response.headers_mut().insert("x-request-id", value);
    }
    crate::error::attach_request_id(&mut response, &id);
    metrics::counter!("clef_http_requests_total","status"=>response.status().as_u16().to_string())
        .increment(1);
    response
}
#[tracing::instrument(skip_all)]
async fn decide(
    State(state): State<AppState>,
    Extension(claims): Extension<Claims>,
    request: Request,
) -> Result<Response, ApiError> {
    let started = Instant::now();
    let permit = state
        .ingress
        .clone()
        .try_acquire_owned()
        .map_err(|_| ApiError::from(clef_rs_core::Error::QueueFull))?;
    if request
        .headers()
        .get("content-type")
        .and_then(|v| v.to_str().ok())
        .is_none_or(|v| v.split(';').next() != Some("application/json"))
    {
        return Err(ApiError {
            status: StatusCode::BAD_REQUEST,
            code: "contentType",
            message: "Content-Type must be application/json.",
        });
    }
    let bytes = timeout(
        Duration::from_millis(state.http.request_read_timeout_ms),
        to_bytes(request.into_body(), state.http.max_body_bytes),
    )
    .await
    .map_err(|_| ApiError::from(clef_rs_core::Error::DeadlineExceeded))?
    .map_err(|_| ApiError {
        status: StatusCode::PAYLOAD_TOO_LARGE,
        code: "bodyTooLarge",
        message: "Request body exceeds the configured limit.",
    })?;
    let (validated, alias, _permit) = tokio::task::spawn_blocking(move || {
        let validated = DecisionRequest::from_json(&bytes)?;
        let envelope: Value = serde_json::from_slice(&bytes)?;
        let alias = envelope
            .get("model")
            .and_then(Value::as_str)
            .ok_or_else(|| clef_rs_core::Error::InvalidRequest("model is required".into()))?
            .to_owned();
        Ok::<_, clef_rs_core::Error>((validated, alias, permit))
    })
    .await
    .map_err(|_| ApiError::from(clef_rs_core::Error::WorkerUnavailable))??;
    let model = state
        .models
        .get(&alias)
        .filter(|_| claims.models.iter().any(|m| m == &alias))
        .ok_or(ApiError {
            status: StatusCode::NOT_FOUND,
            code: "modelNotFound",
            message: "The model alias is unavailable.",
        })?;
    if !state.auth.permits_decision(&claims, &alias) {
        return Err(ApiError::forbidden());
    }
    let remaining = state
        .decision_timeout
        .checked_sub(started.elapsed())
        .ok_or_else(|| ApiError::from(clef_rs_core::Error::DeadlineExceeded))?;
    let options = DecisionOptions::new(claims.sub, remaining)?;
    let decision = model.client.decide(validated, options).await?;
    metrics::histogram!("clef_decision_seconds","model"=>model.config.preset.alias())
        .record(started.elapsed().as_secs_f64());
    let mut response = Json(decision.systemone(&alias)?).into_response();
    for (key, value) in [
        ("x-clef-revision", decision.revision),
        ("x-clef-encoding-version", ENCODING_VERSION.into()),
        ("x-clef-execution-profile", decision.execution_profile),
        (
            "x-clef-truncated-state-tokens",
            decision.truncated_state_tokens.to_string(),
        ),
    ] {
        if let Ok(value) = HeaderValue::from_str(&value) {
            response.headers_mut().insert(key, value);
        }
    }
    Ok(response)
}
async fn models(
    State(state): State<AppState>,
    Extension(claims): Extension<Claims>,
) -> Json<Value> {
    let mut models: Vec<_> = state.models.iter()
        .filter(|(alias,_)| claims.models.iter().any(|allowed| allowed == *alias))
        .map(|(alias, model)| serde_json::json!({
            "id": alias, "revision": model.config.revision.as_str(),
            "profile": model.config.execution.name(), "modality": model.config.execution.modality,
            "ready": model.client.is_ready(), "qualification": format!("flash-{:?}-{:?}-v1", model.config.execution.device, model.config.execution.dtype).to_ascii_lowercase(),
            "maxContextTokens": model.config.execution.max_context_tokens,
        })).collect();
    models.sort_by(|a, b| {
        a.get("id")
            .and_then(Value::as_str)
            .cmp(&b.get("id").and_then(Value::as_str))
    });
    Json(serde_json::json!({"data":models}))
}
async fn livez() -> Json<Value> {
    Json(serde_json::json!({"alive":true}))
}
async fn readyz(State(state): State<AppState>) -> Response {
    let ready = state
        .models
        .values()
        .filter(|m| m.config.required)
        .all(|m| m.client.is_ready());
    (
        if ready {
            StatusCode::OK
        } else {
            StatusCode::SERVICE_UNAVAILABLE
        },
        Json(serde_json::json!({"ready":ready})),
    )
        .into_response()
}
async fn metrics(State(state): State<AppState>) -> Response {
    (
        [("content-type", "text/plain; version=0.0.4")],
        state.metrics.render(),
    )
        .into_response()
}

#[cfg(test)]
mod tests {
    use axum::{body::Body, http::Request};
    use tower::ServiceExt;

    use super::*;
    use crate::auth::tests::{authenticator, claims, token};
    #[tokio::test]
    async fn test_should_authenticate_every_endpoint_and_protect_discovery() -> anyhow::Result<()> {
        let (auth, key) = authenticator()?;
        let recorder = metrics_exporter_prometheus::PrometheusBuilder::new().build_recorder();
        let (rate, task) = rate_limiter(100);
        let http = Http {
            bind: "127.0.0.1:8080".parse()?,
            max_body_bytes: 1024 * 1024,
            request_read_timeout_ms: 1000,
            rate_limit_per_principal_per_minute: 100,
            authenticated_tls_ingress: false,
            max_connections: 8,
        };
        let app = router(AppState {
            auth: Arc::new(auth),
            models: Arc::new(HashMap::new()),
            http,
            metrics: recorder.handle(),
            rate,
            ingress: Arc::new(Semaphore::new(8)),
            decision_timeout: Duration::from_secs(60),
        });
        for path in ["/v1/models", "/livez", "/readyz", "/metrics"] {
            let response = app
                .clone()
                .oneshot(Request::builder().uri(path).body(Body::empty())?)
                .await?;
            assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
            assert!(response.headers().contains_key("x-request-id"));
            let response = app
                .clone()
                .oneshot(
                    Request::builder()
                        .uri(path)
                        .header(
                            "authorization",
                            format!("Bearer {}", token(&key, &claims()?)?),
                        )
                        .body(Body::empty())?,
                )
                .await?;
            assert_eq!(response.status(), StatusCode::OK);
        }
        let mut unauthorized = claims()?;
        unauthorized["scope"] = serde_json::json!("clef:decide");
        let response = app
            .clone()
            .oneshot(
                Request::builder()
                    .uri("/metrics")
                    .header(
                        "authorization",
                        format!("Bearer {}", token(&key, &unauthorized)?),
                    )
                    .body(Body::empty())?,
            )
            .await?;
        assert_eq!(response.status(), StatusCode::FORBIDDEN);
        drop(app);
        task.await?;
        Ok(())
    }
    #[tokio::test]
    async fn test_should_bound_rate_limiter_and_expire_sender() -> anyhow::Result<()> {
        let (rate, task) = rate_limiter(1);
        for expected in [true, false] {
            let (reply, receiver) = oneshot::channel();
            rate.send(RateRequest {
                principal: "one".into(),
                reply,
            })
            .await?;
            assert_eq!(receiver.await?, expected);
        }
        drop(rate);
        task.await?;
        Ok(())
    }
    #[tokio::test]
    #[ignore = "requires pinned Flash weights and a large-memory CPU runner"]
    #[allow(
        clippy::too_many_lines,
        reason = "end-to-end fixture setup and lifecycle assertions remain below 150 lines"
    )]
    async fn test_should_match_embedded_decisions_over_authenticated_http() -> anyhow::Result<()> {
        use std::future::IntoFuture;

        use clef_rs_core::{
            Runtime, RuntimeConfig,
            artifacts::ArtifactStore,
            runtime::{DeviceKind, ExecutionProfile, Modality, Precision, PrefixCacheConfig},
            types::{CommitRevision, ModelPreset},
        };
        use tokio::{
            io::{AsyncReadExt, AsyncWriteExt},
            net::{TcpListener, TcpStream},
        };

        use crate::listener::{Connection, LimitedListener};
        let root =
            std::env::var_os("CLEF_RELEASE_CACHE").ok_or(clef_rs_core::Error::ArtifactMissing)?;
        let store = ArtifactStore::new(root.into(), 85_899_345_920)?;
        let device = if std::env::var("CLEF_RELEASE_DEVICE").as_deref() == Ok("metal") {
            DeviceKind::Metal
        } else {
            DeviceKind::Cpu
        };
        let dtype = if std::env::var("CLEF_RELEASE_DTYPE").as_deref() == Ok("f16") {
            Precision::F16
        } else {
            Precision::F32
        };
        let profile = ExecutionProfile::builder()
            .device(device)
            .dtype(dtype)
            .modality(Modality::Text)
            .max_context_tokens(1024)
            .device_budget_bytes(64 * 1024 * 1024 * 1024)
            .host_budget_bytes(64 * 1024 * 1024 * 1024)
            .build();
        let runtime = Runtime::start(
            store.open(ModelPreset::ClefFlash).await?,
            profile.clone(),
            RuntimeConfig::builder()
                .prefix_cache(PrefixCacheConfig::new(512 * 1024 * 1024)?)
                .build(),
        )
        .await?;
        let client = runtime.client();
        let (auth, key) = authenticator()?;
        let recorder = metrics_exporter_prometheus::PrometheusBuilder::new().build_recorder();
        let (rate, task) = rate_limiter(100);
        let http = Http {
            bind: "127.0.0.1:8080".parse()?,
            max_body_bytes: 1024 * 1024,
            request_read_timeout_ms: 1000,
            rate_limit_per_principal_per_minute: 100,
            authenticated_tls_ingress: false,
            max_connections: 8,
        };
        let model = Model {
            alias: "clef-flash".into(),
            preset: ModelPreset::ClefFlash,
            revision: ModelPreset::ClefFlash
                .revision()
                .parse::<CommitRevision>()?,
            required: true,
            execution: profile,
        };
        let app = router(AppState {
            auth: Arc::new(auth),
            models: Arc::new(HashMap::from([(
                "clef-flash".into(),
                LoadedModel {
                    config: model,
                    client: client.clone(),
                },
            )])),
            http,
            metrics: recorder.handle(),
            rate,
            ingress: Arc::new(Semaphore::new(8)),
            decision_timeout: Duration::from_secs(60),
        });
        let mut input: Value = serde_json::from_slice( br#"{"model":"clef-flash","state":"Checkout has failed for every customer.","questions":{"urgent":{"type":"noul"},"team":{"type":"choice","criteria":{"billing":"invoices","technical":"outages"}},"severity":{"type":"score","criteria":["minor","critical"]}}}"#)?;
        *input
            .get_mut("state")
            .ok_or_else(|| anyhow::anyhow!("missing fixture state"))? =
            Value::String("normal ".repeat(600));
        let bytes = serde_json::to_vec(&input)?;
        let expected = client
            .decide(
                DecisionRequest::from_json(&bytes)?,
                DecisionOptions::default(),
            )
            .await?
            .systemone("clef-flash")?;
        let response = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/systemone")
                    .header("content-type", "application/json")
                    .header(
                        "authorization",
                        format!("Bearer {}", token(&key, &claims()?)?),
                    )
                    .body(Body::from(bytes.clone()))?,
            )
            .await?;
        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(
            response
                .headers()
                .get("x-clef-revision")
                .and_then(|v| v.to_str().ok()),
            Some(ModelPreset::ClefFlash.revision())
        );
        let actual: Value = serde_json::from_slice(&to_bytes(response.into_body(), 65536).await?)?;
        assert_eq!(actual, expected);
        let listener = TcpListener::bind("127.0.0.1:0").await?;
        let address = listener.local_addr()?;
        let limited = LimitedListener {
            listener,
            capacity: Arc::new(Semaphore::new(4)),
            idle: Duration::from_millis(1000),
        };
        let (stop, stopped) = oneshot::channel();
        let serving = tokio::spawn(
            axum::serve(
                limited,
                app.clone()
                    .into_make_service_with_connect_info::<Connection>(),
            )
            .with_graceful_shutdown(async {
                let _ = stopped.await;
            })
            .into_future(),
        );
        let bearer = token(&key, &claims()?)?;
        let headers = format!(
            "POST /v1/systemone HTTP/1.1\r\nHost: localhost\r\nAuthorization: Bearer \
             {bearer}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: \
             close\r\n\r\n",
            bytes.len()
        );
        let mut stream = TcpStream::connect(address).await?;
        stream.write_all(headers.as_bytes()).await?;
        stream.write_all(&bytes).await?;
        let mut received = Vec::new();
        timeout(
            Duration::from_secs(60),
            stream.take(65536).read_to_end(&mut received),
        )
        .await??;
        let response = String::from_utf8(received)?;
        assert!(response.starts_with("HTTP/1.1 200"), "{response}");
        let (_, body) = response
            .split_once("\r\n\r\n")
            .ok_or_else(|| anyhow::anyhow!("missing HTTP body"))?;
        assert_eq!(serde_json::from_str::<Value>(body)?, expected);
        assert!(store.prune(ModelPreset::ClefFlash, true).await.is_err());
        let _ = stop.send(());
        serving.await??;
        runtime.close_admission();
        assert!(!client.is_ready());
        assert!(matches!(
            client
                .decide(
                    DecisionRequest::from_json(&bytes)?,
                    DecisionOptions::default()
                )
                .await,
            Err(clef_rs_core::Error::ShuttingDown)
        ));
        drop(app);
        drop(client);
        assert!(runtime.shutdown().await?.drained);
        task.await?;
        Ok(())
    }
}
