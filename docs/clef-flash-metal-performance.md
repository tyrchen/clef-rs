# CLEF Flash performance on Apple M5 Pro

The implementation replaces token-by-token Metal DeltaNet submission with a fused F32 recurrence and uses a bounded native F32 GQA attention tile for the backbone. Large text projections remain selectable F32 or mixed F16; M5 mixed F16 uses Metal 4 GPU neural accelerators. The CPU reference, pinned weights/encoder, vision tower and classifier semantics are preserved.

The subsequent [exact-prefix and preparation-fusion campaign](clef-flash-prefix-reuse.md) adds an automatic cold-path optimization and optional bounded CPU/Metal prefix retention. It records separate complete, capture and hit measurements; the campaigns below remain historical baselines.

## Implementation

One 128-thread group contains four independent SIMD groups. On M5 with Flash’s 128-key/128-value geometry, each SIMD group processes four value columns together, sharing q/k/decay loads while retaining independent F32 state. Other GPU families/shapes keep one column per SIMD group. Each lane retains up to four vectors of key states in registers and processes the bounded sequence on GPU. Precomputed decay avoids repeating exponentials across state columns; key-width specialization allows register allocation and loop unrolling. The exact reference sum tree protects F16 rounding boundaries. Each of the 24 linear layers issues one recurrence launch instead of a sequence-length-dependent tensor loop. With default configuration, no recurrent state persists across requests; explicitly configured prefix reuse retains immutable snapshots.

The eight full-attention layers use GQA directly, keeping four KV heads rather than copying them to 16. The pinned upstream SDPA shader is instantiated with 16-query/8-key tiles: 28,928 shared bytes fit the Apple GPU limit, whereas the default F32 width-256 tile needs 53,760 bytes and fails pipeline creation. Classifier and vision attention retain the qualified tensor graph. Pipeline creation happens during model loading, and inference adds no host readback or synchronization.

On Apple GPU family 10, large F16 backbone projections use shared Metal 4 MPP pipelines with shape-selected 64×64/64×128 tiles, F16 inputs and F32 accumulation. Relaxed precision is disabled. Complete tiles use static extents; partial tiles use checked tensor bounds. The cooperative F32 result is converted to half and stored directly, avoiding an intermediate buffer and conversion dispatch. Tiny decay/beta projections and older GPUs retain Candle’s Metal GEMM. Classifier and vision projections remain on their qualified F32 path.

Backbone RMSNorm now uses one F32 reduction/scaling kernel, preserving the reference reduction tree and separate square-root/reciprocal rounding before the existing dtype boundary. Offset weights are computed once at load. Four causal convolution taps use one kernel in the original accumulation order, with SiLU still applied by its qualified operation. Both pointwise pipelines are shared across layers. These optimizations apply to both Metal precisions; the CPU equations remain unchanged.

For M5 mixed-F16 requests with at least 512 tokens, one 64×128 MPP dispatch computes the FFN gate/up projections, SiLU and product. It reuses one F32 accumulator and stores only the final half intermediate. Separate projection, activation and product half-rounding boundaries are preserved; tests require exact equality with the previous GPU operations. Short requests and other profiles keep their existing operations. Explicit launch-order/K-block and chunkwise DeltaNet experiments remain diagnostic; the chunkwise candidate is compiled only into test binaries because its local performance regresses.

Rust dispatch validates shapes, dtype, contiguous byte ranges and integer conversions before submitting work. The safe Candle wrappers own allocation, command lifetime and hazard tracking; this project adds no unsafe Rust. The existing owner-thread cancellation, deadline, recovery and shutdown behavior remains in effect.

## Reproduction

```sh
make bench-mbp CLEF_RELEASE_CACHE=/path/to/model-cache
make profile-metal CLEF_RELEASE_CACHE=/path/to/model-cache
make profile-metal-kernels
make profile-metal-tiles
make profile-metal-backbone
make profile-metal-ffn
make profile-metal-long-gemm
make profile-metal-chunk # M5 test-only candidate; excluded from serving
make verify-metal-neural # M5 only
make verify-metal-pointwise
make verify verify-metal verify-benchmark
make verify-metal-release verify-metal-media verify-metal-serving PYTHON=.venv-reference/bin/python
```

The [MBP YAML](../examples/clef.mbp-benchmark.yaml) retains the original six workloads and adds exact 139/192-token inputs. The harness’s minimum binary question with an empty state is 139 tokens under the pinned protocol; a 64-token synthetic decision would omit required protocol content. Models run sequentially with eight fixed Rayon/Candle CPU threads. The current matrix gives each expensive case ten samples and has 100 normal-path samples per profile, excluded warmups, eight separate stage diagnostics and bounded concurrency 1/2/8/16. The normal timer includes encoding, GPU completion and answer conversion, but excludes JSON parsing, HTTP/authentication and transport. Small-n percentiles are descriptive observations, not production tail guarantees.

The previous [CPU/Metal campaign](benchmarks/flash-cpu-metal/report.md) remains the baseline. CPU equations are unchanged and its 82-minute full-weight performance run is reused; CPU unit/operator regression checks still run. Traces and isolated operator measurements are diagnostic only. Raw Instruments traces stay outside the repository because they can include unrelated system process information. The existing conservative memory plan is retained rather than lowered from finite peak samples.

Implementation rationale and numerical pitfalls are recorded in [the research evidence](research/clef-metal-performance-research.md) and [design contract](../specs/clef-metal-performance-design.md).

## Latest long-prefill campaign

Committed `874255d` completed all eight workloads in separate sequential F32/F16 processes on this 64 GiB M5 Pro. Each precision has **100 warm normal-path samples**, one excluded warmup per case, eight separate diagnostic decisions and the existing bounded managed-runtime workload. The three expensive cases now have ten samples instead of three. Actual tokens/schemas, thread counts and concurrency construction match the prior campaign. The [raw report](benchmarks/flash-metal-prefill/report.md) preserves every sample and clean-tree source/executable/workload identity.

| Workload | Previous n | Current n | Previous F16 mean ms | Current F16 mean ms | Observed reduction |
| --- | ---: | ---: | ---: | ---: | ---: |
| short-256 | 10 | 10 | 225.560 | 222.366 | 1.4% |
| medium-1024 | 10 | 10 | 880.720 | 820.664 | 6.8% |
| long-4096 | 3 | 10 | 3902.149 | 3718.966 | 4.7% |
| mixed-8-fields | 10 | 10 | 894.514 | 835.621 | 6.6% |
| wide-32-fields | 3 | 10 | 2867.769 | 2726.028 | 4.9% |
| large-option-schema | 3 | 10 | 3921.047 | 3755.468 | 4.2% |
| minimal-139 | 20 | 20 | 169.785 | 163.715 | 3.6% |
| compact-192 | 20 | 20 | 193.051 | 188.491 | 2.4% |

This round yields a modest 1.4–6.8% observed F16 reduction across the eight workload means. The requested 1,024/4,096-token cases average **820.664/3718.966 ms**, with p95 821.901/3742.234 ms (ten samples each). The shortest 139-token input averages 163.715 ms (n=20); **100 ms has not been demonstrated**. Timings include encoding, GPU completion and answer conversion, excluding model load, HTTP/authentication and transport. These finite samples do not establish production-tail performance.

| Workload | Current Metal n | Reused CPU n | CPU F32 mean ms | Previous Metal F32 mean ms | Current Metal F32 mean ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| short-256 | 10 | 10 | 13271.713 | 622.549 | 610.332 |
| medium-1024 | 10 | 10 | 51347.788 | 2485.774 | 2446.849 |
| long-4096 | 10 | 3 | 212415.691 | 10266.398 | 10105.794 |
| mixed-8-fields | 10 | 10 | 51335.671 | 2503.339 | 2460.319 |
| wide-32-fields | 10 | 3 | 156584.235 | 7677.114 | 7559.214 |
| large-option-schema | 10 | 3 | 213301.310 | 10307.017 | 10135.600 |
| minimal-139 | 20 | — | — | 449.957 | 441.842 |
| compact-192 | 20 | — | — | 493.134 | 484.187 |

F32 improves about 1.5–2.0% versus its prior means while retaining projection precision. The CPU comparison uses the unchanged pure-Rust Candle backend without Accelerate/MKL and reuses its existing samples. Full-weight performance was not repeated on CPU; its current unit/operator/build gates passed.

| Profile | Sampled GPU peak GiB | OS peak footprint GiB | Concurrency 1 / 2 / 8 / 16 successful req/s |
| --- | ---: | ---: | --- |
| metal:f32 | 35.920 | 40.336 | 1.635 / 1.639 / 1.637 / 1.636 |
| metal:f16 | 18.527 | 26.773 | 4.469 / 4.497 / 4.489 / 4.472 |

Each profile completed 38 of 54 managed admissions with 16 expected `queueFull` rejections and no inference errors. Concurrency uses the 256-token workload and one device owner; it queues requests rather than adding GPU throughput. Both peaks fit the unchanged conservative device/host plans. GPU samples and process footprint overlap and must not be added; memory remains broadly unchanged versus the prior campaign. F16 snapshot verification/model loading took 6.356/17.446 s; these phases are excluded from warm latency.

All six full-model text/context/PNG/JPEG qualifications reproduce the previous maximum/mean errors exactly under the unchanged 1e-3/1e-4 gates. Both real authenticated HTTP and CLI/server checks passed, including exact answers, image probabilities, provenance, metrics and shutdown. Default/vision/Metal builds/tests, formatting, pedantic and production boundary lints, artifacts, benchmark tests and warning-free docs passed. See [the verification report](clef-flash-verification.md) and [qualification JSON](benchmarks/flash-metal-prefill/qualification.json).

The [operator evidence](benchmarks/flash-metal-prefill/operator-diagnostics.json) records 192 GEMM sweep means, 42 FFN tile samples, five SIMD value widths, the complete test-only chunkwise algorithm and the raw full-model rejection data for the combined one-dimensional-walk/initial-FFN prototype. Four-column SIMD reuse and FFN fusion are enabled. The original two-dimensional grid/dynamic-K projection path is retained; the slower chunkwise candidate has no serving route and is absent from both release executables. Synthetic/barrier timings are excluded from normal-path claims. This remains mixed F16 with F32-sensitive computation; FP8 is not implemented.

## Previous projection and pointwise campaign

The committed `190a459` implementation completed the same eight workloads on this MBP, sequentially in F32 and mixed F16, with 79 normal samples and 54 managed admission attempts per precision. Each case has one excluded warmup. The [raw report](benchmarks/flash-metal-tuned/report.md) preserves all samples and environment/executable/configuration hashes. The implementation adds shape-selected matrix tiles, direct half output, fused RMSNorm and fused causal convolution; the prior campaigns below remain intact.

| Workload | n | Previous M5 F16 ms | Tuned F16 ms | Mean latency reduction |
| --- | ---: | ---: | ---: | ---: |
| short-256 | 10 | 306.912 | 225.560 | 26.5% |
| medium-1024 | 10 | 1226.342 | 880.720 | 28.2% |
| long-4096 | 3 | 6091.856 | 3902.149 | 35.9% |
| mixed-8-fields | 10 | 1238.903 | 894.514 | 27.8% |
| wide-32-fields | 3 | 4388.024 | 2867.769 | 34.6% |
| large-option-schema | 3 | 6125.369 | 3921.047 | 36.0% |
| minimal-139 | 20 | 209.566 | 169.785 | 19.0% |
| compact-192 | 20 | 250.308 | 193.051 | 22.9% |

F16 mean latency falls 19.0–36.0% across all eight workloads versus the previous M5 implementation. Matching the original six Metal F16 workloads gives 18.31–19.94× speedups. The 139-token request averages 169.785 ms with p95 170.951 ms (n=20); 256 tokens averages 225.560 ms with p95 226.767 ms (n=10). **100 ms has not been demonstrated.** These timings include encoding and GPU completion, with HTTP/authentication excluded. Three-sample long cases do not establish production tail latency.

| Workload | n | Reused CPU F32 ms | Previous M5 F32 ms | Tuned F32 ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| short-256 | 10 | 13271.713 | 682.777 | 622.549 |
| medium-1024 | 10 | 51347.788 | 2773.547 | 2485.774 |
| long-4096 | 3 | 212415.691 | 11517.158 | 10266.398 |
| mixed-8-fields | 10 | 51335.671 | 2802.787 | 2503.339 |
| wide-32-fields | 3 | 156584.235 | 8647.113 | 7677.114 |
| large-option-schema | 3 | 213301.310 | 11598.797 | 10307.017 |
| minimal-139 | 20 | — | 479.022 | 449.957 |
| compact-192 | 20 | — | 536.868 | 493.134 |

F32 mean latency falls 6.1–11.2% versus the previous M5 campaign, without changing projection precision. The CPU reference uses Candle's pure-Rust backend without Accelerate/MKL; its full-weight performance data is reused, and CPU build/operator/regression gates were repeated.

At concurrency 1/2/8/16, tuned F16 completed 4.391/4.402/4.387/4.373 successful 256-token decisions/s. Each precision completed 38 of 54 admissions, with 16 expected `queueFull` rejections under overload and no inference errors. One owner executes GPU work, so concurrency queues requests rather than multiplying device throughput. The concurrency-16 successful p95 is 1,831.713 ms. These are finite embedded-runtime measurements, not HTTP load or sustained-throughput qualification.

| Precision | Previous sampled GPU peak GiB | Tuned sampled GPU peak GiB | Previous OS peak footprint GiB | Tuned OS peak footprint GiB |
| --- | ---: | ---: | ---: | ---: |
| f32 | 36.077 | 35.920 | 42.300 | 40.531 |
| f16 | 18.712 | 18.527 | 26.826 | 26.756 |

Both sampled GPU peaks and OS footprint high-water marks fit the retained device/host plans. The memory metrics overlap and must not be added; GPU sampling every 100 ms may miss short peaks and excludes managed reload. F16 snapshot verification/model loading took 6.428/17.459 s in a new process; warm latency excludes those phases and driver/OS caches were not cleared.

All six full-model text/context/PNG/JPEG qualifications passed the existing probability gates, reproducing the previous M5 maximum/mean errors exactly. Both precision profiles passed real CLI/authenticated HTTP results, provenance, image inference and shutdown checks. Default/vision/Metal build/test/format/pedantic and boundary lint gates, documentation and dependency policy checks passed. The same [verification report](clef-flash-verification.md) and [machine-readable qualification](benchmarks/flash-metal-tuned/qualification.json) record the evidence. This remains mixed F16 with F32 accumulation; FP8 is not implemented.

The [operator diagnostics](benchmarks/flash-metal-tuned/operator-diagnostics.json) retain the tile sweep and before/after backbone categories. They use synthetic tensors and explicit barriers, with incomplete category coverage, and are excluded from the normal latency statistics. The workload, hashes, completion flags, sample counts, statistics, overload outcomes and capacity bounds were independently checked after both processes exited successfully.

## Previous M5 campaign

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
