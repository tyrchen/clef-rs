# CLEF Flash v1 verification

Verified on 2026-10-01/02 against the actual pinned Flash weights. The supported v1 execution profile is **CPU F32, text or optional still images, up to 4,096 total tokens**. The library, CLI, and authenticated HTTP adapter use the same encoder, model graph, learned joint head, and answer conversion. The larger CLEF model, CUDA, Metal, video, and longer contexts are rejected explicitly.

## Reference and environment

| Item | Value |
| --- | --- |
| Model | `Cloudflare/clef-flash` |
| Revision | `17f0b0ad64efb65d273590632833508766b2aae6` |
| Reviewed reference source SHA-256 | `0e304cf7c6500e8bb59bef7e2afd2c6373f82596dfb3b57d1aa93c175e2dc3a3` |
| Tokenizer SHA-256 | `06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523` |
| Python oracle | Python 3.12.13, Torch 2.11.0, Transformers 5.10.2, torchvision 0.26.0, Pillow 12.3.0 |
| Rust | Stable 1.99.0, edition 2024; Candle 0.11.0 |
| Runner | macOS, Apple Silicon ARM64, 18 CPU cores, 64 GiB RAM |
| Native JPEG codec | Checksum-pinned libjpeg-turbo 3.2.0, static build without SIMD; turbojpeg 1.5.1 safe API |

The [reference harness](../tests/reference/qualify_flash.py) verifies all snapshot files before generating unrounded probabilities. Its [dependency lock](../tests/reference/requirements.lock) fixes the verification environment. Python is an independent test oracle, never a production backend; downloaded model source is not executed during Rust acquisition, load, or inference.

## Numerical results

The CPU release gate remains the design's original maximum absolute probability error of **0.001**. Comparisons use unrounded library probabilities, cover `noul`, `choice`, and `score`, and check the winning option when the reference margin exceeds the error bound. No tolerance was widened to accept these results.

| Full-weight corpus | Records | Probabilities | Maximum absolute error | Mean absolute error | Rust test elapsed |
| --- | ---: | ---: | ---: | ---: | ---: |
| [Mixed text/JSON schemas](../crates/core/fixtures/release/flash-f32.json) | 100 | 800 | 0.00000226498 | 0.000000191524 | 1,804.98 s |
| [4,096-token text and two PNG shapes](../crates/core/fixtures/release/flash-extended-f32.json) | 3 | 24 | 0.00000333786 | 0.000000907991 | 281.05 s |
| [Baseline and progressive JPEG](../crates/core/fixtures/release/flash-jpeg-f32.json) | 2 | 16 | 0.00000452995 | 0.000000939617 | 68.33 s |

Text fixtures include ordered/reordered schemas, missing instructions, Unicode, bounded JSON, and all three question types. The long-input fixture has exactly 4,096 encoded tokens without truncation. PNG cases include a 256×256 image and an asymmetric 37×61 image. JPEG cases exercise baseline 4:4:4 and progressive 4:2:0 input. Repeated full-model decisions are deterministic on the tested CPU profile.

Synthetic fixtures independently check hybrid full/linear attention, causal convolution and recurrent state, norms, gates, partial/multimodal rotary positions, the learned joint schema head, and the entire vision tower/merger. Tiled attention includes a sequence spanning multiple tiles. The released tokenizer's token IDs and half-open question/option spans match exactly. Image resize/patch fixtures match Torch's uint8 antialiased bicubic path; a four-image fixture checks all multimodal position coordinates against Transformers.

JPEG RGB output matches Pillow exactly on the checked pixel fixture. Other tested Rust JPEG decoders differed by up to two pixel levels and caused a full-model probability error above the unchanged gate. The pinned native codec fixes that difference; its private safe wrapper bounds encoded bytes, dimensions, components, progressive scans, metadata, and output allocation. A 10,000-case mutation fuzz run passed. That campaign is regression evidence, not a proof that the native library has no defects. Distribution notices are [preserved separately](licenses/index.md).

## Resource and lifecycle evidence

For text at 4,096 tokens, `plan-memory` reported 60,319,553,990 host bytes and 48,348,934,963 device/accounted execution bytes, including safe shard staging and a 20% reserve. Images add a 1 GiB allowance. Both budgets must pass before model allocation; these estimates are conservative admission policy, not a physical-memory reservation.

The managed-runtime authenticated HTTP qualification observed maximum resident size of 31,851,823,104 bytes. The extended full-context/image qualification observed 33,702,543,360 bytes. Both fit the configured 64 GiB budget. These macOS process measurements include allocator retention and should not be treated as portable memory requirements.

Fast scheduler tests verify principal fairness, bounded admission, queued expiry, caller cancellation, reservation ownership, and shutdown. Recovery tests catch a reload panic and verify that shutdown wins a concurrent reload without reopening admission or retaining the discarded engine. Artifact tests verify digest/manifest integrity, incomplete and malformed snapshots, safe paths, leases, and refusal to prune a loaded snapshot. A complete real snapshot was fetched, verified, opened offline, and used for every full-weight test.

The full-weight HTTP integration test compares direct library answers with authenticated router and real TCP answers exactly, verifies provenance, checks the active snapshot lease, closes admission, and joins runtime owners. Listener regression tests expire partial slow headers while allowing active inference to outlive the idle socket deadline.

The release CLI/server smoke test uses a temporary local JWKS and the clearly identified public test key. It starts `serve --offline`, waits for authenticated readiness, rejects unauthenticated health requests, verifies model discovery, runs PNG/JPEG decisions over TCP against the reference, reads protected metrics, sends SIGTERM, requires successful process exit, then runs offline CLI inference and requires exact CLI/HTTP response equality. No external identity provider or runtime key discovery is needed.

## Build and policy gates

`make verify` passed on the final Rust source and lockfile:

- Workspace `cargo build` and `cargo test`: 24 core tests, eight server tests, and three documentation examples passed with default features.
- Vision build/tests: 28 core tests, eight server tests, and three documentation examples passed. Large-weight tests and the explicit fuzz campaign remain ignored in ordinary tests and were run separately as recorded above.
- Nightly formatting, pedantic Clippy for all workspace targets with default and vision features, and `--no-default-features` offline compilation passed.
- Public API documentation built with `RUSTDOCFLAGS='-D warnings'`; artifact and tensor-loader checks passed.
- `cargo audit --ignore RUSTSEC-2024-0436` and `cargo deny check` passed. The one documented exception is the unmaintained compile-time `paste` macro required transitively by Candle/gemm/tokenizers; no vulnerability advisory is exempted. Dependency duplicates and unused allowed-license entries remain non-failing policy warnings.

Boundary Clippy checks additionally cover production core/server code with `unwrap_used`, `expect_used`, `indexing_slicing`, and `panic` warnings denied. Project Rust forbids unsafe code. Optional JPEG decoding confines native code to the reviewed dependency boundary.

## Reproduction

The native codec requires CMake and a C compiler. Default text builds need no native JPEG SDK. A complete verified model cache and a large-RAM runner are required for release gates; ordinary CI does not download weights.

```sh
make verify
make verify-parity CLEF_TOKENIZER=/path/to/verified/tokenizer.json
make verify-release CLEF_RELEASE_CACHE=/path/to/model-cache
make verify-media-release verify-jpeg-release CLEF_RELEASE_CACHE=/path/to/model-cache
make fuzz-media
make verify-serving PYTHON=/path/to/reference/python CLEF_RELEASE_CACHE=/path/to/model-cache
```

To regenerate the independent fixtures, use `make reference-env`, then set `PYTHON=.venv-reference/bin/python` for `reference-fixtures`, `reference-release`, `reference-extended`, and `reference-jpeg`. The harness validates the reference source fingerprint and release manifest. [Usage and operations](clef-flash-usage.md) describe production configuration, limits, local OIDC keys, cache commands, and image input.

## Qualification limits

This report establishes functional and numerical compatibility for Flash v1 on the tested CPU F32 profile. It does not establish a workload latency SLO, a sustained 10,000-decision leak/throughput qualification, or accelerator performance. The 10,000-case campaign above tests the image boundary, not 10,000 model decisions. CPU timing varies materially with thread count, input length, available RAM, and contention.

CUDA/Metal qualification was skipped because no CUDA runner was available and neither backend is enabled as a supported v1 profile. No container image was built or tested because Docker was unavailable; this release supplies the native Cargo binary and embedded library. A production container/SBOM/provenance pipeline, broader platform performance qualification, hot configuration/revision switching, additional stage metrics, and the larger model/video milestones from the target design remain outside the advertised Flash v1 capabilities. Offline operation was exercised with local weights and keys and the outbound client can be compiled out; an operating-system egress-blocked deployment test was not run.
