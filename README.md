# clef-rs

Local Rust inference for **Cloudflare CLEF Flash**, powered by Candle. Version 1 pins `Cloudflare/clef-flash` at `17f0b0ad64efb65d273590632833508766b2aae6`. It implements the Qwen3.5 hybrid backbone and the released learned joint schema head, with `noul`, `choice`, and `score` answers through an embedded library, CLI, and authenticated HTTP server.

The model graph runs in Rust. Model acquisition is explicit; loading and decisions never download files or execute model Python. Artifacts are checked against shipped sizes, SHA-256 digests, and tensor shapes. Read [usage and operations](docs/clef-flash-usage.md) and the [verification report](docs/clef-flash-verification.md) for the supported execution combinations and measured parity.

## Run Flash

CPU F32 requires a large-memory machine. This implementation was tested on a 64 GiB Apple Silicon system. The complete snapshot is approximately 19 GB; allow at least 21 GB of free disk space. Memory estimates include safe shard staging, activations, and a 20% reserve. Use a release build for inference.

```sh
make build-release
cargo run --release -p clef-rs-server -- fetch --model clef-flash --cache-dir ./model-cache
cargo run --release -p clef-rs-server -- plan-memory --config examples/clef.cpu.yaml
cargo run --release -p clef-rs-server -- decide --config examples/clef.cpu.yaml --request examples/request.json
```

For HTTP serving, configure your issuer, audience, public RS256 JWKS file, and key-validity deadline in the YAML file, then:

```sh
cargo run --release -p clef-rs-server -- serve --config examples/clef.cpu.yaml --offline
curl http://127.0.0.1:8080/v1/systemone \
  -H "Authorization: Bearer $CLEF_ACCESS_TOKEN" \
  -H 'Content-Type: application/json' \
  --data-binary @examples/request.json
```

Tokens require `iss`, `aud`, `exp`, `sub`, `scope`, and a `models` allowlist. Decisions require `clef:decide`; discovery, health, and metrics require `clef:observe`. Every route requires authentication. No key discovery or token URL fetch occurs during serving. Non-loopback listeners require an explicitly configured authenticated TLS ingress.

The core exposes `ArtifactStore`, `DirectEngine`, and the bounded actor `Runtime`/`DecisionClient`; see the [embedded example](crates/core/examples/decision.rs). An optional `vision` feature provides bounded PNG/JPEG and RGB image inputs; run `make native-jpeg` first to install the pinned reference-compatible JPEG codec. Apple Silicon Metal supports F32 and mixed F16 profiles with `--features metal`; use [the Metal configuration](examples/clef.metal.yaml). CUDA, video, the larger CLEF model, and unqualified contexts fail explicitly in this first version.

## Verify

```sh
make verify
make verify-parity CLEF_TOKENIZER=/path/to/verified/tokenizer.json
make verify-release CLEF_RELEASE_CACHE=/path/to/model-cache
make verify-media-release verify-jpeg-release CLEF_RELEASE_CACHE=/path/to/model-cache
```

Fast CI uses CPU and image operator fixtures and needs no model download. Full-release tests are explicitly ignored in ordinary `cargo test` because they require the complete weights and substantial RAM. The checked-in unrounded Python oracle covers 100 real Flash requests; Python is needed only to regenerate verification fixtures (`make reference-env`, `make reference-fixtures`, `make reference-release`).

Rust 2024 and stable Rust 1.99.0 are pinned. Source is MIT licensed; the model's Apache-2.0 license and notices remain separate and are preserved in the cache.

Real-weight CPU/Metal benchmarks and domain microbenchmarks are available through `make bench-inference` and `make bench-domain`. See [benchmark usage](docs/clef-flash-benchmarks.md) for workloads, reproduction, and measured results.
