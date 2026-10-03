# CLEF Flash v1 verification

Verified on 2026-10-01/02 (latest follow-up 2026-10-03 UTC) against the actual pinned Flash weights. Supported v1 execution profiles are **CPU F32 and native macOS Metal F32/mixed F16, text or optional still images, up to 4,096 total tokens**. The library, CLI, and authenticated HTTP adapter use the same encoder, model graph, learned joint head, and answer conversion. The larger CLEF model, CUDA, video, and longer contexts are rejected explicitly.

## Reference and environment

| Item | Value |
| --- | --- |
| Model | `Cloudflare/clef-flash` |
| Revision | `17f0b0ad64efb65d273590632833508766b2aae6` |
| Reviewed reference source SHA-256 | `0e304cf7c6500e8bb59bef7e2afd2c6373f82596dfb3b57d1aa93c175e2dc3a3` |
| Tokenizer SHA-256 | `06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523` |
| Python oracle | Python 3.12.13, Torch 2.11.0, Transformers 5.10.2, torchvision 0.26.0, Pillow 12.3.0 |
| Rust | Stable 1.99.0, edition 2024; Candle 0.11.0 |
| Runner | macOS 26.5.2, Apple M5 Pro, 18 CPU cores, 20 GPU cores, Metal 4, 64 GiB unified RAM |
| Native JPEG codec | Checksum-pinned libjpeg-turbo 3.2.0, static build without SIMD; turbojpeg 1.5.1 safe API |

The [reference harness](../tests/reference/qualify_flash.py) verifies all snapshot files before generating unrounded probabilities. Its [dependency lock](../tests/reference/requirements.lock) fixes the verification environment. Python is an independent test oracle, never a production backend; downloaded model source is not executed during Rust acquisition, load, or inference.

## Numerical results

Every qualified CPU/Metal profile retains maximum absolute probability error **0.001** and mean absolute error **0.0001** against the same CPU F32 oracle. Comparisons use unrounded library probabilities, cover `noul`, `choice`, and `score`, and check the winning option when the reference margin exceeds the error bound. No tolerance was widened to accept these results.

| Full-weight corpus | Records | Probabilities | Maximum absolute error | Mean absolute error | Rust test elapsed |
| --- | ---: | ---: | ---: | ---: | ---: |
| [Mixed text/JSON schemas](../crates/core/fixtures/release/flash-f32.json) | 100 | 800 | 0.00000226498 | 0.000000191524 | 1,804.98 s |
| [4,096-token text and two PNG shapes](../crates/core/fixtures/release/flash-extended-f32.json) | 3 | 24 | 0.00000333786 | 0.000000907991 | 281.05 s |
| [Baseline and progressive JPEG](../crates/core/fixtures/release/flash-jpeg-f32.json) | 2 | 16 | 0.00000452995 | 0.000000939617 | 68.33 s |

Pre-optimization Metal full-weight qualification on the same runner:

| Profile | Corpus | Probabilities | Maximum absolute error | Mean absolute error | Rust test elapsed |
| --- | --- | ---: | ---: | ---: | ---: |
| F32 | 100 mixed text requests | 800 | 0.00000166893 | 0.000000161918 | 564.87 s |
| F32 | 4,096-token text and two PNG shapes | 24 | 0.00000488758 | 0.00000149102 | 191.55 s |
| F32 | Baseline and progressive JPEG | 16 | 0.00000405312 | 0.00000102358 | 51.58 s |
| Mixed F16 | 100 mixed text requests | 800 | 0.00071564317 | 0.00007115226 | 558.60 s |
| Mixed F16 | 4,096-token text and two PNG shapes | 24 | 0.00016713142 | 0.00006244767 | 181.14 s |
| Mixed F16 | Baseline and progressive JPEG | 16 | 0.00019443035 | 0.00007643865 | 46.20 s |


Optimized Metal qualification (`5df02eb`, unchanged thresholds):

| Profile | Corpus | Probabilities | Maximum absolute error | Mean absolute error | Rust test elapsed |
| --- | --- | ---: | ---: | ---: | ---: |
| F32 | 100 mixed text requests | 800 | 0.000003114343 | 0.000000202775 | 116.56 s |
| F32 | 4,096-token text and two PNG shapes | 24 | 0.000005424023 | 0.000001789847 | 56.19 s |
| F32 | Baseline and progressive JPEG | 16 | 0.000004351139 | 0.000001026667 | 34.39 s |
| Mixed F16 | 100 mixed text requests | 800 | 0.000503331423 | 0.000047634602 | 104.42 s |
| Mixed F16 | 4,096-token text and two PNG shapes | 24 | 0.000213682652 | 0.000052259798 | 47.41 s |
| Mixed F16 | Baseline and progressive JPEG | 16 | 0.000249028206 | 0.000047095004 | 27.90 s |

Metal 4 M5 mixed-F16 projections additionally passed the same full-model gates:

| Corpus | Probabilities | Maximum absolute error | Mean absolute error | Rust test elapsed |
| --- | ---: | ---: | ---: | ---: |
| 100 mixed text requests | 800 | 0.000443369150 | 0.000037016605 | 62.20 s |
| 4,096-token text and two PNG shapes | 24 | 0.000328183174 | 0.000090070253 | 38.80 s |
| Baseline and progressive JPEG | 16 | 0.000418752432 | 0.000095413183 | 25.72 s |

F32 text qualification was repeated after the projection wrapper change: 800 probabilities, maximum 0.000003114343, mean 0.000000202775 (116.45 s), identical to the preceding optimized F32 result. Its image attention/projection math is unchanged and the preceding F32 PNG/JPEG qualification is retained.

After the M5 projection change, authenticated real HTTP qualification passed again on F32 (33.81 s) and mixed F16 (25.80 s). Both release CLI/server PNG/JPEG smoke tests passed again, including oracle probabilities, exact CLI/HTTP answers, provenance, metrics and SIGTERM drain.

After completing the final 158-sample performance matrix, `make verify-metal-neural` passed again. `make profile-metal` also completed both direct and managed benchmark phases under Instruments and exited normally. The final raw statistics, executable/configuration hashes, exact token/sample counts and device/host capacity bounds were checked independently.

The M5 matrix pipeline uses F16 operands with F32 accumulation, disables relaxed precision and applies the existing GPU half conversion. CPU comparison tests cover complete tiles, M/N tails and unaligned K. Full default/vision/Metal build, tests, nightly format, pedantic Clippy, audit/deny, documentation and production boundary lints passed again with this projection implementation.

The fused recurrence preserves the previous Metal reduction tree and is independently checked against it within 1e-7. CPU checks cover key widths 1/4/32/64/128, padded value widths, multiple heads and sequence lengths. Native F32 GQA attention checks aligned/unaligned causal lengths against the CPU reference. The optimization retains the original classifier/vision attention graph after an experimental generic SDPA route exceeded the mixed-F16 image mean gate. These test elapsed times include verification/loading and varying inputs; use [the controlled MBP measurements](clef-flash-metal-performance.md) for speedups.

F16 retains large text projection matmuls in F16, with an F32 classifier head, vision tower, residuals, normalization, convolution/attention accumulation, and recurrent state. Qualification caught pure-F16 classifier/vision errors and a visual-feature downcast; these were fixed by preserving sensitive paths in F32. The original limits were not relaxed. Normal and separately profiled decisions produce exactly the same rounded response on the checked first corpus records. Qualification elapsed times have different thread/workload conditions and must not be used as controlled speedups; use [the performance measurements](clef-flash-benchmarks.md).

Text fixtures include ordered/reordered schemas, missing instructions, Unicode, bounded JSON, and all three question types. The long-input fixture has exactly 4,096 encoded tokens without truncation. PNG cases include a 256×256 image and an asymmetric 37×61 image. JPEG cases exercise baseline 4:4:4 and progressive 4:2:0 input. Repeated full-model decisions are deterministic on the tested CPU profile.

Synthetic fixtures independently check hybrid full/linear attention, causal convolution and recurrent state, norms, gates, partial/multimodal rotary positions, the learned joint schema head, and the entire vision tower/merger. Tiled attention includes a sequence spanning multiple tiles. The released tokenizer's token IDs and half-open question/option spans match exactly. Image resize/patch fixtures match Torch's uint8 antialiased bicubic path; a four-image fixture checks all multimodal position coordinates against Transformers.

JPEG RGB output matches Pillow exactly on the checked pixel fixture. Other tested Rust JPEG decoders differed by up to two pixel levels and caused a full-model probability error above the unchanged gate. The pinned native codec fixes that difference; its private safe wrapper bounds encoded bytes, dimensions, components, progressive scans, metadata, and output allocation. A 10,000-case mutation fuzz run passed. That campaign is regression evidence, not a proof that the native library has no defects. Distribution notices are [preserved separately](licenses/index.md).

## Projection and pointwise tuning qualification

The `190a459` implementation adds shape-selected MPP tiles, direct cooperative half output, and fused backbone RMSNorm/four-tap convolution. Full-model checks retain the original gates and reproduce the preceding M5 maximum/mean errors exactly in both precisions:

| Profile | Corpus | Probabilities | Maximum absolute error | Mean absolute error | Rust test elapsed |
| --- | --- | ---: | ---: | ---: | ---: |
| F32 | 100 mixed text requests | 800 | 0.000003114343 | 0.000000202775 | 108.18 s |
| F32 | 4,096-token text and two PNG shapes | 24 | 0.000005424023 | 0.000001789847 | 56.13 s |
| F32 | Baseline and progressive JPEG | 16 | 0.000004351139 | 0.000001026667 | 36.04 s |
| Mixed F16 | 100 mixed text requests | 800 | 0.000443369150 | 0.000037016605 | 53.61 s |
| Mixed F16 | 4,096-token text and two PNG shapes | 24 | 0.000328183174 | 0.000090070253 | 33.36 s |
| Mixed F16 | Baseline and progressive JPEG | 16 | 0.000418752432 | 0.000095413183 | 25.73 s |

Operator checks exercise both matrix tile shapes against CPU accumulation with half rounding, convolution causal tails with exact F32 equality, and RMSNorm with at most 1e-6 F32 error and exact half rounding. Invalid dtype, shape, sequence bounds and strided pointwise inputs are rejected before dispatch. Test-only category instrumentation adds no barriers or timing state to production inference.

Authenticated full-weight HTTP regression passed on F32 (34.05 s) and mixed F16 (25.46 s), requiring exact direct/router/TCP results, provenance and owner shutdown. Both release CLI/server smoke tests passed with offline PNG/JPEG inference, oracle probabilities, protected discovery/metrics, SIGTERM drain and exact CLI/HTTP answers. Default/vision/Metal workspace builds/tests, nightly formatting, all-target pedantic Clippy, production boundary Clippy, no-default-feature compilation, warning-free public docs, artifact checks and seven benchmark tests passed. The current counts are 29 core/eight server/nine doc tests by default, 33/eight/nine with vision, and 34/nine/nine with Metal+vision. Audit and deny passed with the existing unmaintained `paste` advisory exception; dependencies and the lockfile did not change.

After the commit, the final Metal build/test/pedantic gate, five recurrence/attention diagnostics, both projection tiles and three pointwise checks passed again. The backbone category diagnostic completed all four lengths. The [normal benchmark](benchmarks/flash-metal-tuned/report.md) then completed 79 samples per precision and 54 managed admissions per precision. Independent checks verified completion flags, exact source/executable/configuration identity, unchanged YAML workloads, raw sample counts/statistics, expected overload errors and device/host capacity bounds. No Rust/MSL source changed after the measured implementation commit. The [qualification JSON](benchmarks/flash-metal-tuned/qualification.json) retains unrounded maximum/mean errors and oracle fixture hashes; category/geometry measurements are separate diagnostics.

## Resource and lifecycle evidence

For CPU F32 text at 4,096 tokens, `plan-memory` reported 60,319,553,990 host bytes and 48,348,934,963 device/accounted execution bytes, including safe shard staging and a 20% reserve. Images add a 1 GiB allowance. Both budgets must pass before model allocation; these estimates are conservative admission policy, not a physical-memory reservation.

The managed-runtime authenticated HTTP qualification observed maximum resident size of 31,851,823,104 bytes. The extended full-context/image qualification observed 33,702,543,360 bytes. Both fit the configured 64 GiB budget. These macOS process measurements include allocator retention and should not be treated as portable memory requirements.

Metal loading converts bounded host tensors before final-dtype GPU upload, avoiding a second model-sized temporary allocation. Its host plan includes GPU weights, scratch and shard staging because these share physical unified RAM; the GPU recommended working set is checked independently. Mixed F16 accounts for F32-promoted parameter bytes and budgets activation storage as F32 at the larger FFN width.

The full benchmark exposed an initial F16 admission underestimate (23.996 GiB planned versus 26.661 GiB sampled). The final 4,096-token text plans are approximately 28.796 GiB device / 39.944 GiB unified host for F16, and 49.828 GiB device / 60.977 GiB unified host for F32. Both observed peaks, text/image budgets and arithmetic overflow are covered by model-metadata-bound regression tests. Original measured estimates remain in the [raw benchmark report](benchmarks/flash-cpu-metal/report.md).

Distinct CPU/Metal profiles in one process fail configuration validation until a combined capacity plan exists. Tensor backend errors and panics fail the active request without replay, then use the existing bounded recovery path.

Fast scheduler tests verify principal fairness, bounded admission, queued expiry, caller cancellation, reservation ownership, and shutdown. Recovery tests catch a reload panic and verify that shutdown wins a concurrent reload without reopening admission or retaining the discarded engine. Artifact tests verify digest/manifest integrity, incomplete and malformed snapshots, safe paths, leases, and refusal to prune a loaded snapshot. A complete real snapshot was fetched, verified, opened offline, and used for every full-weight test.

The full-weight HTTP integration test compares direct library answers with authenticated router and real TCP answers exactly, verifies provenance, checks the active snapshot lease, closes admission, and joins runtime owners. Listener regression tests expire partial slow headers while allowing active inference to outlive the idle socket deadline.

The final optimized release CLI/server smoke tests passed on both Metal precisions with offline PNG/JPEG inference, reference probabilities, authenticated readiness/discovery/metrics, provenance, SIGTERM drain and exact CLI/HTTP response equality. The CLI harness used the existing reference Python environment; the system Python lacked its PyYAML test dependency. `verify-metal-cli` is independently runnable to avoid repeating passed Rust HTTP checks when correcting only the test interpreter.

The optimized full-weight HTTP integration tests also passed on Metal F32 (32.85 s) and mixed F16 (27.01 s), including exact direct/router/TCP answer equivalence, provenance and owner shutdown.

The same full-weight HTTP integration test passed on Metal F32 (49.56 s) and mixed F16 (42.83 s). Both release Metal CLI/server smoke tests passed with actual GPU PNG/JPEG inference and exact CLI/HTTP response equality. After the shared YAML guard and memory-plan correction, both real HTTP tests were repeated successfully (F32 50.03 s, F16 42.42 s), and both final release CLI/server PNG/JPEG smoke tests passed again.

The release CLI/server smoke test uses a temporary local JWKS and the clearly identified public test key. It starts `serve --offline`, waits for authenticated readiness, rejects unauthenticated health requests, verifies model discovery, runs PNG/JPEG decisions over TCP against the reference, reads protected metrics, sends SIGTERM, requires successful process exit, then runs offline CLI inference and requires exact CLI/HTTP response equality. No external identity provider or runtime key discovery is needed.

## Build and policy gates

`make verify` passed on the final Rust source and lockfile:

- Workspace `cargo build` and `cargo test`: 28 core tests, eight server tests, and nine documentation examples passed with default features.
- Vision build/tests: 32 core tests, eight server tests, and nine documentation examples passed. Large-weight tests and the explicit fuzz campaign remain ignored in ordinary tests and were run separately as recorded above.
- Nightly formatting, pedantic Clippy for all workspace targets with default and vision features, and `--no-default-features` offline compilation passed.
- Metal feature build/tests, real GPU synthetic backbone/head fixtures, and pedantic all-target Clippy passed; the Metal configuration and shared-RAM rejection test passed. The Metal matrix passed 32 core tests, nine server tests and nine documentation examples. The original benchmark example passed three statistics/configuration tests, including duplicate YAML rejection. The optimized matrix passed 33 core/9 server/9 documentation tests with Metal+vision, five real-device kernel/backbone/head diagnostics, and seven benchmark configuration/statistics cases. The full default/vision/offline/build/format/pedantic gates and production boundary lints were repeated on the final optimization source. Audit and deny passed with the same documented advisory exception.
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
make verify-metal verify-metal-release verify-metal-media CLEF_RELEASE_CACHE=/path/to/model-cache
make verify-metal-serving PYTHON=/path/to/reference/python CLEF_RELEASE_CACHE=/path/to/model-cache
make bench-domain bench-inference CLEF_RELEASE_CACHE=/path/to/model-cache
```

To regenerate the independent fixtures, use `make reference-env`, then set `PYTHON=.venv-reference/bin/python` for `reference-fixtures`, `reference-release`, `reference-extended`, and `reference-jpeg`. The harness validates the reference source fingerprint and release manifest. [Usage and operations](clef-flash-usage.md) describe production configuration, limits, local OIDC keys, cache commands, and image input.

## Qualification limits

This report establishes functional and numerical compatibility for Flash v1 on the tested CPU F32 and Metal F32/mixed F16 profiles. It does not establish a workload latency SLO, a sustained 10,000-decision leak/throughput qualification, or performance on other accelerator generations. The finite performance matrix and raw measurements are reported separately. The 10,000-case campaign above tests the image boundary, not 10,000 model decisions. CPU timing varies materially with thread count, input length, available RAM, and contention.

CUDA qualification was skipped because no CUDA runner was available; CUDA remains unsupported. Metal was qualified on the actual Apple M5 Pro GPU. No container image was built or tested because Docker was unavailable; this release supplies the native Cargo binary and embedded library. A production container/SBOM/provenance pipeline, broader platform performance qualification, hot configuration/revision switching, additional production stage metrics, and the larger model/video milestones from the target design remain outside the advertised Flash v1 capabilities. Offline operation was exercised with local weights and keys and the outbound client can be compiled out; an operating-system egress-blocked deployment test was not run.
