# CLEF Flash performance on Apple M5 Pro

The implementation replaces token-by-token Metal DeltaNet submission with a fused F32 recurrence and uses a bounded native F32 GQA attention tile for the backbone. Large text projections remain selectable F32 or mixed F16; M5 mixed F16 uses Metal 4 GPU neural accelerators. The CPU reference, pinned weights/encoder, vision tower and classifier semantics are preserved.

## Implementation

One 128-thread group contains four independent SIMD groups, each owning one value column. A lane retains up to four key states in registers and processes the bounded sequence on GPU. Precomputed decay avoids repeating exponentials across state columns; key-width specialization allows register allocation and loop unrolling. The exact reference sum tree protects F16 rounding boundaries. Each of the 24 linear layers issues one recurrence launch instead of a sequence-length-dependent tensor loop. No recurrent state persists across requests.

The eight full-attention layers use GQA directly, keeping four KV heads rather than copying them to 16. The pinned upstream SDPA shader is instantiated with 16-query/8-key tiles: 28,928 shared bytes fit the Apple GPU limit, whereas the default F32 width-256 tile needs 53,760 bytes and fails pipeline creation. Classifier and vision attention retain the qualified tensor graph. Pipeline creation happens during model loading, and inference adds no host readback or synchronization.

On Apple GPU family 10, large F16 backbone projections use a shared Metal 4 MPP pipeline with 64×64 tiles, F16 inputs and F32 accumulation. Relaxed precision is disabled. Complete tiles use static extents; partial tiles use checked tensor bounds. The F32 result is converted to half on GPU before the existing normalization/residual graph. Tiny decay/beta projections and older GPUs retain Candle’s Metal GEMM. Classifier and vision projections remain on their qualified F32 path.

Rust dispatch validates shapes, dtype, contiguous byte ranges and integer conversions before submitting work. The safe Candle wrappers own allocation, command lifetime and hazard tracking; this project adds no unsafe Rust. The existing owner-thread cancellation, deadline, recovery and shutdown behavior remains in effect.

## Reproduction

```sh
make bench-mbp CLEF_RELEASE_CACHE=/path/to/model-cache
make profile-metal CLEF_RELEASE_CACHE=/path/to/model-cache
make profile-metal-kernels
make verify-metal-neural # M5 only
make verify verify-metal verify-benchmark
make verify-metal-release verify-metal-media verify-metal-serving PYTHON=.venv-reference/bin/python
```

The [MBP YAML](../examples/clef.mbp-benchmark.yaml) retains the original six workloads and adds exact 139/192-token inputs. The harness’s minimum binary question with an empty state is 139 tokens under the pinned protocol; a 64-token synthetic decision would omit required protocol content. Models run sequentially with eight fixed Rayon/Candle CPU threads. Each profile has 79 normal-path samples, excluded warmups, eight separate stage diagnostics and bounded concurrency 1/2/8/16. The normal timer includes encoding, GPU completion and answer conversion, but excludes JSON parsing, HTTP/authentication and transport. Small-n percentiles are descriptive observations, not production tail guarantees.

The previous [CPU/Metal campaign](benchmarks/flash-cpu-metal/report.md) remains the baseline. CPU equations are unchanged and its 82-minute full-weight performance run is reused; CPU unit/operator regression checks still run. Traces and isolated operator measurements are diagnostic only. Raw Instruments traces stay outside the repository because they can include unrelated system process information. The existing conservative memory plan is retained rather than lowered from finite peak samples.

Implementation rationale and numerical pitfalls are recorded in [the research evidence](research/clef-metal-performance-research.md) and [design contract](../specs/clef-metal-performance-design.md).

## Measured results

The final sequential campaign used the committed `988691f` implementation on the 64 GiB M5 Pro. Both precisions completed all 79 direct samples and 54 managed admission attempts. Every case has one excluded warmup. The earlier fused-only campaign used `5df02eb`; the original baseline used the pre-optimization graph. Raw reports retain executable SHA-256, configuration SHA-256, exact model revision, environment, individual samples and process metrics. The dirty-tree flags include documentation and measurement artifacts; the measured Rust source corresponds to the stated commits.

| Workload | n | Original Metal F16 ms | Fused recurrence/attention ms | Final M5 F16 ms | Original / final |
| --- | ---: | ---: | ---: | ---: | ---: |
| short-256 | 10 | 4380.099 | 637.118 | 306.912 | 14.27× |
| medium-1024 | 10 | 17560.557 | 2544.382 | 1226.342 | 14.32× |
| long-4096 | 3 | 71434.042 | 10581.239 | 6091.856 | 11.73× |
| mixed-8-fields | 10 | 17700.851 | 2557.617 | 1238.903 | 14.29× |
| wide-32-fields | 3 | 54052.874 | 7912.541 | 4388.024 | 12.32× |
| large-option-schema | 3 | 72355.850 | 10613.320 | 6125.369 | 11.81× |
| minimal-139 | 20 | — | 448.033 | 209.566 | — |
| compact-192 | 20 | — | 486.072 | 250.308 | — |

M5 F16 cuts the original six workload means by 11.73–14.32×. Relative to the fused-only implementation, the M5 matrix path adds a further 1.73–2.14× full-model improvement across the eight workloads. At 256 tokens the final p95 is 307.601 ms (n=10); the 139-token p95 is 210.539 ms (n=20). These are warm embedded decisions, including encoding and GPU completion, with HTTP/authentication excluded. The shortest harness workload averages 209.566 ms: **this implementation has not demonstrated 100 ms**. The 4,096-token cases have three samples and do not establish a production tail bound.

The F32 path keeps the original projection precision. The following compares it with the unchanged pure-Rust CPU baseline on matching requests:

| Workload | n | CPU F32 ms | Original Metal F32 ms | Final Metal F32 ms | Metal original / final |
| --- | ---: | ---: | ---: | ---: | ---: |
| short-256 | 10 | 13271.713 | 4434.517 | 682.777 | 6.49× |
| medium-1024 | 10 | 51347.788 | 17826.264 | 2773.547 | 6.43× |
| long-4096 | 3 | 212415.691 | 72481.962 | 11517.158 | 6.29× |
| mixed-8-fields | 10 | 51335.671 | 17805.576 | 2802.787 | 6.35× |
| wide-32-fields | 3 | 156584.235 | 54143.759 | 8647.113 | 6.26× |
| large-option-schema | 3 | 213301.310 | 72484.770 | 11598.797 | 6.25× |
| minimal-139 | 20 | — | — | 479.022 | — |
| compact-192 | 20 | — | — | 536.868 | — |

F32 improves 6.25–6.49× over its original Metal graph. The mixed-F16 versus CPU comparison changes precision; use the F32 table for a same-precision comparison. CPU still uses Candle without Accelerate/MKL, and its unchanged 82-minute full-weight campaign was not repeated. CPU correctness/build gates were repeated.

At concurrency 1/2/8/16, M5 F16 achieved 3.261/3.256/3.252/3.246 successful 256-token decisions per second with one device owner. Each precision completed 38 managed decisions out of 54 attempted admissions. The 16-worker overload case rejected 16 requests with `queueFull`, as expected from the bounded eight-slot ingress/queue; no inference error occurred. Queued p95 rises to 2,467.508 ms at concurrency 16 and is reported separately from unqueued inference. This is embedded-runtime load, not an HTTP throughput claim.

| Precision | Original sampled GPU peak GiB | Final sampled GPU peak GiB | Original OS peak footprint GiB | Final OS peak footprint GiB |
| --- | ---: | ---: | ---: | ---: |
| f32 | 44.276 | 36.077 | 49.016 | 42.300 |
| f16 | 26.661 | 18.712 | 27.027 | 26.826 |

The F16 sampled GPU peak falls about 29.8%; the OS footprint high-water changes much less because it includes loading and retained allocations. GPU samples are every 100 ms after direct loading and may miss short peaks; they exclude managed reload. These overlapping unified-memory metrics must not be added. Both final sampled GPU peaks and both OS footprint high-water marks fit the retained device/host plans. F16 snapshot verification took 6.401 s and model loading 17.515 s in the final new process; warm latency excludes these phases. OS/driver caches were not cleared.

The [full numerical verification](clef-flash-verification.md) passed the original maximum/mean probability gates for text, 4,096-token context, PNG and JPEG, plus exact CLI/authenticated HTTP results and shutdown. All default/vision/Metal Rust gates, pedantic and boundary lints, documentation, audit and deny passed. The M5 matrix operator check passed again after the performance campaign. This is qualified mixed F16 with F32 accumulation; FP8 is not implemented or qualified.

Raw evidence: [original CPU/Metal](benchmarks/flash-cpu-metal/report.md), [fused-only Metal](benchmarks/flash-metal-optimized/report.md), [final M5 matrix](benchmarks/flash-metal-m5/report.md), and [final metadata](benchmarks/flash-metal-m5/metadata.json).

The final Metal System Trace ended normally when the benchmark exited (45.792 s), including direct and managed phases. Its [sanitized summary](benchmarks/flash-metal-m5/instruments-summary.json) and [instrumented diagnostic](benchmarks/flash-metal-m5/instruments-diagnostic.json) are separate from normal samples. The earlier 60-second baseline trace stopped before managed completion. Neither trace interval counts nor traced timings are used to establish the reported speedups.
