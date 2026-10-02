# Flash v1 usage and operations

This implementation supports the pinned CLEF Flash release. Read the [verification report](clef-flash-verification.md) for numerical evidence and the current device/modality/context qualification. The larger CLEF release is intentionally absent from the catalog and preset parser.

## Acquisition and offline execution

Use `make build-release` to build the `clef` binary. Rust 1.99.0 is pinned in `rust-toolchain.toml`. The default `hub` feature enables explicit acquisition; `--no-default-features` builds a core/server that rejects fetch without linking the outbound HTTP client.

```sh
clef fetch --model clef-flash --cache-dir ./model-cache
clef inspect --model clef-flash --cache-dir ./model-cache
clef cache verify --model clef-flash --cache-dir ./model-cache
clef cache list --cache-dir ./model-cache
clef plan-memory --config examples/clef.cpu.yaml
clef decide --config examples/clef.cpu.yaml --request examples/request.json
```

If `clef` is not installed, prefix each command with `cargo run --release -p clef-rs-server --`. `cargo install --path apps/server --locked` installs it locally.

Fetch acquires all 14 approved files, including the model license and reference source; the runtime never executes that source. The shipped catalog binds filenames, exact lengths, SHA-256 digests, the immutable revision, and all 882 tensor names/shapes/dtypes/shard assignments. Downloads allow only reviewed HTTPS origins, pin resolved public IP addresses, disable ambient proxies and redirects, and send `HF_TOKEN` only to `huggingface.co`. Resumable transfers validate ranges and rehash the completed file before publication. Retryable failures use bounded retries/backoff and capped numeric Retry-After delays.

The private cache has content-addressed `blobs/sha256`, immutable `snapshots`, `staging`, and advisory `locks`. Acquisition is owned by a bounded actor and coalesces simultaneous Flash fetches. A global mutation lease serializes acquisition/import/prune across processes. Publication uses safe copies, no-overwrite hard links, fsync, and atomic snapshot rename. Cache mutations require a writable operator-owned cache; inference can open a complete prefetched read-only cache with existing lease files. A lease protects every loaded snapshot from pruning.

`clef import --model clef-flash --source /path/to/release --cache-dir ./model-cache` accepts a flat directory containing exactly the 14 reviewed regular files. It copies and verifies bytes into private storage and rejects symlinks or extra entries. `clef cache prune --model clef-flash --cache-dir ./model-cache --dry-run` reports reclaimable bytes, and refuses a leased snapshot. Remove `--dry-run` only when deliberately deleting that release. V1 has no automatic eviction or background GC.

## Configuration and resource policy

[examples/clef.cpu.yaml](../examples/clef.cpu.yaml) is the complete CPU configuration; [examples/request.json](../examples/request.json) exercises all answer types. The schema uses camelCase keys, rejects unknown fields, and preflights YAML duplicates, aliases, tags, depth, and collection limits. Every configured alias must use `required: true`; optional startup models are unsupported in v1. Only `CLEF_CACHE_ROOT` and `CLEF_HTTP_BIND` are explicit environment overrides.

CPU F32 uses approximately 35 GB of materialized text weights plus conservative shard staging and activation allowances. A 64 GiB host budget is appropriate for the tested machine; `plan-memory` computes the actual context-specific estimate before loading. Budgets are admission limits, not physical-memory reservations. The operator must ensure that other processes leave sufficient RAM. Insufficient budgets fail before tensor allocation. Multiple aliases with an identical execution profile share one runtime; conflicting profiles on one device are rejected rather than silently duplicating weights.

Each runtime owns one blocking device thread and one scheduler. Defaults cap preparation to two workers, ingress/queued records to eight, total reserved tokens to 32,768, and payload reservations to 8 MiB. Queues use FIFO within a principal and round-robin between principals. No cross-principal result cache, generation cache, or microbatching exists. Dropping a caller cancels its work cooperatively; reservations remain owned until preparation or execution actually ends. Queued deadlines expire independently. Device synchronization precedes result publication and resource release.

The example allows a 300-second decision deadline for CPU prefill. Queue timeout is separate and defaults to five seconds, so an overloaded CPU server can return `deadlineExceeded` instead of waiting behind a long prefill. Raise the queue timeout deliberately if your workload needs it. There is no universal latency guarantee across hardware or request lengths.

Context includes the system prompt, state, schema, suffix, and image tokens. Overflow is rejected by default. `runtime.truncation: referencePrefix` explicitly retains the reference state-token prefix; schema/media are never silently truncated. Responses report dropped state tokens in `x-clef-truncated-state-tokens`.

## Request and response contract

A request has a configured `model` alias, any bounded JSON `state`, and an ordered `questions` object. The core accepts the same envelope; embedded callers can omit `model` because they already chose the engine.

| Type | Criteria | Output |
| --- | --- | --- |
| `noul` | Optional `true`/`false` description object | Unrounded truth probability in the library; four-decimal `noul` on the wire |
| `choice` | Ordered object of option identifiers and JSON descriptions | Winner, confidence, and complete caller-ordered distribution |
| `score` | Ordered array of JSON legend entries | Expected zero-based ordinal index, confidence, legend, and distribution |

Missing/null/empty instructions fall back to the question ID. Choice options are encoded in lexical order, then mapped back to caller order; exact ties choose the first caller option. Scores use ordinal indices even when legend entries are numbers. Library probabilities remain unrounded; only `systemone()` and transport serialization apply four-decimal rounding. JSON legend values remain unchanged.

Identifiers allow 1–64 ASCII letters, digits, underscores, or hyphens. Requests permit 1–32 questions, 2–64 options per field, and at most 512 options total. State is capped at 512 KiB, semantic strings at 64 KiB, instruction/description renderings at 4 KiB, depth at 16, and semantic nodes at 4,096. Collections and JSON keys also have explicit caps. Duplicate keys, non-finite numbers, unsafe integers outside ±(2^53−1), control characters, and malformed Unicode are rejected. No URLs or paths in state are fetched or executed.

The HTTP response preserves the SystemOne `model`, `answers`, and `usage` envelope (`output_tokens` is zero). Provenance headers contain revision, encoding version, execution profile, truncation count, and a bounded request ID. Errors expose a stable `error.code`, redacted `error.message`, and matching `error.requestId`; internal diagnostics and user data are not returned. Saturation returns 429 with Retry-After; malformed input returns 400, resource limits 413, unsupported capabilities 422, deadlines 504, and unavailable workers 503.

## Authentication and lifecycle

Configure your OIDC issuer and audience, a local public RS256 JWKS file, and the date until which that key snapshot is valid. Obtain access tokens from your own identity provider. The header must specify `alg: RS256`, `typ: at+jwt`, and a known `kid`; the payload requires issuer, audience, expiry, subject, space-separated scopes, and a `models` alias allowlist. Signature/issuer/audience/expiry/not-before are verified with at most 30 seconds of skew. Duplicate JWT/JWKS JSON and critical JWT extensions are rejected. Multiple local keys support overlap during key rotation; refreshing keys requires an operator restart in v1.

```json
{"iss":"https://your-issuer.example","aud":"clef-rs","sub":"your-principal","exp":1893456000,"scope":"clef:decide clef:observe","models":["clef-flash"]}
```

That is a claims shape, not a usable signed token. Never use the public test signing key as a deployment credential.

`serve --offline` verifies local artifacts, loads weights, warms all three question types, then binds the listener. Every endpoint requires authentication:

| Endpoint | Permission |
| --- | --- |
| `POST /v1/systemone` | `clef:decide` and the requested model alias |
| `GET /v1/models` | `clef:observe`; only permitted aliases are listed |
| `GET /livez`, `GET /readyz`, `GET /metrics` | `clef:observe` |

Default HTTP limits are a 1 MiB body, 128 connections, 32 bounded headers, and an explicit body/header read deadline. Slow idle connections expire while active inference remains protected. Rate-limit state is bounded to 256 principals and 128 pending checks. A non-loopback bind requires an explicit authenticated TLS ingress. The Rust listener itself serves HTTP and relies on that ingress for external TLS.

SIGINT/SIGTERM close admission immediately, reject queued work, allow active execution to drain within the grace period, and join scheduler/device owners. A panic fails the active request and attempts at most three reload/warmup cycles in ten minutes; requests are never replayed. Cancellation is cooperative between layers and attention tiles. A stuck native kernel cannot be forcibly killed safely; an incomplete drain is an explicit process-replacement error.

Use the global `--json-logs` flag for structured production diagnostics on stderr.

Metrics expose request counts by status and decision latency by pinned model through the authenticated endpoint. Tracing records lifecycle events, never raw states, images, bearer tokens, or full requests. V1 requires restart for configuration changes; it does not implement runtime configuration/revision switching.

## Still images

Install the pinned native codec with `make native-jpeg` (requires CMake, a C compiler, curl, and shasum), then build with `--features vision`, set `modality: image`, raise `http.maxBodyBytes` to 16,777,216, and configure preparation payload capacity explicitly (up to 536,870,912 bytes). The payload reservation includes encoded data, decoded RGB pixels, and a conservative resize/patch estimate before preparation. Use a 512 MiB budget when supporting the maximum advertised image sizes. The memory planner adds a 1 GiB media allowance.

`images` is an array of `{"mediaType":"image/png","data":"<base64>"}` or JPEG objects. Limits are four images, 2 MiB encoded bytes per image, 8 MiB encoded bytes total, four megapixels per image, eight megapixels total, and dimensions at most 4,096. PNG animation, 16-bit PNG, compressed/extended PNG metadata, multi-picture JPEG, non-8-bit/non-RGB-or-grayscale JPEG, and excessive metadata are rejected. JPEG decoding caps progressive scans at 32. Decode probes and allocation limits run before full decode. The processor matches the released RGB normalization, smart resize, uint8 antialiased bicubic interpolation, temporal duplication, patch ordering, learned vision positions, merger, placeholder expansion, and multimodal rotary positions.

Embedded callers can use `Image::from_rgb` and `DecisionRequest::with_images` to bypass file decode while retaining validation and processor behavior. Text workers reject images before expensive preparation. Video is unsupported and never interpreted as image frames.

JPEG uses the current libjpeg-turbo 3.2.0 release through a bounded safe Rust wrapper because other Rust JPEG decoders changed reference pixels and exceeded the existing final-probability tolerance. The wrapper validates the header and output length before allocation, caps dimensions/components/scans/metadata, uses a private synchronous decoder, and exposes no raw pointers. `make fuzz-media` runs the mutation harness. Project Rust remains `forbid(unsafe_code)`; native code is confined to this dependency boundary. Include [the native notices](licenses/index.md) with distributed vision binaries.

## Apple Silicon Metal

Build on macOS with `cargo build --release -p clef-rs-server --features metal`, then use `examples/clef.metal.yaml` for `plan-memory`, `decide`, or `serve --offline`. Optional image serving requires `make native-jpeg` and `--features metal,vision`, plus the image body/payload limits described above. The same pinned Flash snapshot works for CPU and Metal; no second model download is required.

Set `execution.dtype` to `f32` for the same-precision CPU comparison, or `f16` for mixed precision. F16 keeps large text projections and their matmuls in F16; the classifier head, vision tower, residual accumulation, convolution accumulation, normalization, attention accumulation, and recurrent state use F32. On M5 / Apple GPU family 10, large F16 backbone projections automatically use Metal 4 GPU neural accelerators with F32 accumulation; older GPUs use the existing Metal matrix kernel. Both profiles use the original probability qualification limits. Weight conversion occurs in bounded CPU staging before upload; model inference runs on the selected Metal device, without a CPU fallback.

Metal requires a feature-enabled native macOS build and an available device ordinal. Startup checks the operator's host/device budgets and the GPU's recommended working set before loading. CPU and GPU allocations share physical unified RAM: these are overlapping limits, not separate physical pools. Distinct CPU/Metal models are rejected in one server process because that configuration requires a combined capacity plan; identical aliases share the existing weight owner. Backend tensor errors fail the current request without replay and trigger bounded reload. Cancellation cannot interrupt a native GPU kernel; completion is synchronized before its reservation is released.

Use `make verify-metal`, `make verify-metal-release`, `make verify-metal-media`, and `make verify-metal-serving` to reproduce the GPU qualification with a complete cache. See the [benchmark guide](clef-flash-benchmarks.md) for CPU/Metal measurement.
