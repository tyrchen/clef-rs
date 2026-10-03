# CLEF Flash CPU and Metal benchmarks

The project now has two benchmark layers: Criterion domain microbenchmarks and full-weight CPU/Metal inference measurements. Before this change, verification logs contained elapsed test times, but there was no controlled performance benchmark suite.

## Run

```sh
make bench-domain
make bench-inference CLEF_RELEASE_CACHE=/path/to/model-cache
# Individual backend matrices:
make bench-cpu CLEF_RELEASE_CACHE=/path/to/model-cache
make bench-metal CLEF_RELEASE_CACHE=/path/to/model-cache
```

Metal targets require native macOS and an available GPU. The default inference target builds a release example with the `metal` feature and sequentially runs CPU F32, Metal F32, and Metal mixed F16. Models never coexist during measurement. `BENCHMARK_THREADS` defaults to eight and fixes both `RAYON_NUM_THREADS` and `CANDLE_NUM_THREADS`. CPU uses Candle's default pure-Rust backend without Accelerate/MKL; the comparison does not represent every possible optimized CPU implementation.

`BENCHMARK_CONFIG` selects bounded YAML workloads; [the checked-in matrix](../examples/clef.benchmark.yaml) controls token counts, schema shape, warmup, sample counts, and concurrency. The runner shares the server's YAML event preflight, rejecting aliases, tags, duplicate/ambiguous keys, excessive depth and event counts before constructing a configuration tree. `BENCHMARK_RESULTS` selects the artifact directory. A complete checksum-verified model cache is required. The runner fails if it cannot construct exactly the requested token count, and fails on unsuccessful direct decisions. It records admission errors separately in concurrency tests. The performance workload is text-only; image correctness is independently qualified with full weights.

## Measurements

The default matrix includes exact 256/1,024/4,096-token inputs, mixed answer types, eight/32 fields, and 256 total options. Short cases use ten warm samples, expensive cases three, with one excluded warmup each. Managed concurrency is 1/2/8/16 with two requests per worker and one device owner. The benchmark sets eight preparation workers, keeps ingress/queue capacities at eight, and uses 300-second queue/request/shutdown limits to measure queued work. These differ from production defaults; rejected admissions and successful queued latency are reported separately.

Normal-path wall time measures `DirectEngine::decide` on an already validated request object and includes encoding, backbone, head, typed answer conversion, and final device completion. It excludes JSON envelope parsing, HTTP/authentication and transport serialization. Concurrency measures the embedded `Runtime`, not HTTP load generation. Reports contain every raw sample, mean/standard deviation/min/max, nearest-rank p50/p95/p99, successful request throughput, and encoded input tokens per second. There are no generated output tokens. Stage diagnostics add explicit GPU barriers and are measured separately; their sum is not used as the normal-path throughput estimate.

Cold process phases report snapshot verification, model loading, and first decision. OS file caches and Metal driver/shader caches are not cleared; cold means a new process, not a cache-empty machine. Process RSS high-water covers direct and managed phases. On macOS, OS-reported peak memory footprint and average active CPU cores from process user/system time are also retained. These are distinct accounting metrics; the RSS, GPU allocation and footprint columns are not summed. Metal allocator usage is sampled every 100 ms after loading in the direct phase, and current/recommended allocator snapshots are recorded per case. A sampled peak can miss shorter spikes and excludes initial loading and managed reload. Private GPU allocations can reside outside process RSS; CPU/GPU memory on Apple Silicon must not be summed as independent physical pools.

The external harness records hardware, OS/Rust versions, base commit/dirty state, executable SHA-256, workload SHA-256, environment settings, exact command lines, and raw process logs. It atomically checkpoints the JSON after each complete case; `complete` becomes true only after managed shutdown drains. Reports are updated after each completed backend; `make bench-report` validates and regenerates a report from completed raw measurements without rerunning inference.

## Results and interpretation

The pre-optimization measured matrix is published in [the CPU/Metal report](benchmarks/flash-cpu-metal/report.md), with machine-readable raw results beside it. F32 compares the same precision across backends. F16 is a mixed-precision comparison and must be read alongside [the numerical qualification](clef-flash-verification.md).

Small sample counts make p95/p99 descriptive order statistics, often equal to the maximum observation; they do not establish a production tail latency or SLO. Queue time under concurrency is intentional. Rejected work is excluded from successful latency/throughput and shown separately. The suite does not establish a sustained 10,000-decision memory guarantee.

Criterion measures request parsing, bounded JSON validation, and reference-compatible JSON rendering without loading weights. The exported [domain results](benchmarks/flash-cpu-metal/domain.md) include confidence intervals and raw samples. Local artifacts are also stored under Cargo's target directory in `criterion`; `make verify-benchmark` checks the inference runner's statistics and configuration rules.

## Pre-optimization campaign

The [complete measured report](benchmarks/flash-cpu-metal/report.md) contains CPU F32, Metal F32 and mixed Metal F16 results on the 64 GiB Apple M5 Pro, with 39 normal-path samples and four concurrency groups per backend. The full campaign took approximately 138 minutes (82 CPU, 29 Metal F32, 28 Metal F16); reduce the bounded YAML sample counts for a shorter local experiment. Metal is approximately 2.9–3.0× faster than the default pure-Rust CPU path. Mixed F16 saves about 40% of the sampled direct-phase GPU peak (26.661 versus 44.276 GiB) while its mean latencies are only about 0.2–1.5% lower than Metal F32 across these workloads. The large-context cases have only three samples; percentiles are descriptive, not a production SLO.

The campaign found and corrected an F16 memory-planning underestimate. The final planner budgets Metal FFN temporary width and covers both measured peaks; the report preserves the original executable's estimates and documents the follow-up correction. The configuration guard, admission estimate and documentation changed after measurement, while the measured tensor graph remains unchanged. Raw artifacts identify the exact measured executable from `0900a46` by SHA-256.

## Optimized MBP campaign

The fused Metal execution graph and its detailed measurements are described in [the MBP performance report](clef-flash-metal-performance.md). The original CPU/Metal raw baseline above is preserved. The extended [MBP workload configuration](../examples/clef.mbp-benchmark.yaml) retains all six original cases and adds exact 139/192-token one-field requests (20 samples each). The current MBP matrix gives all six original workloads ten samples each and has 100 normal-path samples per precision; earlier M5/tuned campaigns used 79. The pinned encoding protocol requires 139 tokens for the harness's minimal binary question with an empty state; requested 64/128-token cases cannot be constructed and are rejected rather than reported as fictitious timings.

```sh
make bench-mbp CLEF_RELEASE_CACHE=/path/to/model-cache
make profile-metal CLEF_RELEASE_CACHE=/path/to/model-cache
make profile-metal-kernels
make verify-metal-neural # M5 matrix correctness and operator diagnostics
```

`bench-mbp` runs only Metal F32/mixed F16, sequentially, and writes to `docs/benchmarks/flash-metal-prefill` by default (`MBP_BENCHMARK_RESULTS` overrides it), preserving the previous M5 campaign. `METAL_BENCHMARK_PROFILES=metal:f16` selects just F16 for an experiment. CPU tensor equations are unchanged; their earlier full-weight campaign is the reference rather than an unnecessary repeat of the 82-minute CPU run. `profile-metal` requires Instruments/Xcode and records a timestamped trace and diagnostic benchmark under Cargo's target directory; traces can contain unrelated system information and remain outside the repository. `profile-metal-tiles` sweeps matrix geometry, and `profile-metal-backbone` adds nested category barriers in a test-only build. `profile-metal-ffn` measures fused gate/up/SiLU tiles; `profile-metal-long-gemm` measures launch-order and K-block candidates; `profile-metal-chunk` verifies and times a test-only chunkwise DeltaNet candidate. These diagnostics use synthetic data and are separate from normal-path latency measurements.

The previous [M5 Metal 4 matrix](benchmarks/flash-metal-m5/report.md) records 158 normal-path samples across both precisions. Mixed F16 averages 209.566/306.912/1,226.342/6,091.856 ms at 139/256/1,024/4,096 tokens. Matching original workloads improve 11.73–14.32×. The [campaign comparison](clef-flash-metal-performance.md) preserves the fused-only results and gives the same-precision CPU/F32 comparison, memory accounting and numerical qualification.

The preceding [projection/pointwise matrix](benchmarks/flash-metal-tuned/report.md), from committed `190a459`, repeats all 158 normal samples and both managed matrices. F16 averages 169.785/225.560/880.720/3,902.149 ms at 139/256/1,024/4,096 tokens, reducing all eight workload means by 19.0–36.0% versus the previous M5 campaign. F32 also improves by 6.1–11.2%. Both profiles complete 38 of 54 managed admissions, with 16 expected bounded-queue rejections and no inference errors. Sampled GPU peaks are 35.920/18.527 GiB and OS footprint peaks 40.531/26.756 GiB for F32/F16; both fit the retained capacity plans. All six full-model probability gates and real CLI/authenticated HTTP checks pass. The normal measurements exclude HTTP and instrumented diagnostics; 100 ms has not been demonstrated.

The latest [long-prefill matrix](benchmarks/flash-metal-prefill/report.md), measured from clean committed `874255d`, completes 100 warm samples per precision; the expensive cases now each have n=10. Mixed F16 averages 820.664 / 3,718.966 ms at 1,024 / 4,096 tokens, reducing the preceding means by 6.8% / 4.7%. All eight F16 means improve by 1.4–6.8%, with unchanged numerical gates and 38 of 54 successful managed admissions per precision. These are modest observed gains from fused FFN and four-column SIMD recurrence. The [detailed comparison](clef-flash-metal-performance.md) preserves CPU/F32 comparisons, previous sample counts, limits and rejected optimization evidence.

The newer [exact-prefix reuse campaign](clef-flash-prefix-reuse.md) separately measures complete calculation, miss/capture and matching-prefix hit. It also records a matched previous executable for cold comparisons and a scoped CPU continuation campaign. The original complete CPU/Metal and M5 campaigns above remain archived; a prefix hit is not a fresh-state latency measurement.
