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

`BENCHMARK_CONFIG` selects bounded YAML workloads; [the checked-in matrix](../examples/clef.benchmark.yaml) controls token counts, schema shape, warmup, sample counts, and concurrency. `BENCHMARK_RESULTS` selects the artifact directory. A complete checksum-verified model cache is required. The runner fails if it cannot construct exactly the requested token count, and fails on unsuccessful direct decisions. It records admission errors separately in concurrency tests. The performance workload is text-only; image correctness is independently qualified with full weights.

## Measurements

The default matrix includes exact 256/1,024/4,096-token inputs, mixed answer types, eight/32 fields, and 256 total options. Short cases use ten warm samples, expensive cases three, with one excluded warmup each. Managed concurrency is 1/2/8/16 with two requests per worker, bounded admission, and one device owner.

Normal-path wall time includes encoding, backbone, head, typed answer conversion, and final device completion. Reports contain every raw sample, mean/standard deviation/min/max, nearest-rank p50/p95/p99, successful request throughput, and encoded input tokens per second. There are no generated output tokens. Stage diagnostics add explicit GPU barriers and are measured separately; their sum is not used as the normal-path throughput estimate.

Cold process phases report snapshot verification, model loading, and first decision. The OS file cache is not flushed. Process RSS high-water covers direct and managed phases. Metal allocator usage is sampled every 100 ms after loading in the direct phase, and current/recommended allocator snapshots are recorded per case. A sampled peak can miss shorter spikes and excludes initial loading and managed reload. Private GPU allocations can reside outside process RSS; CPU/GPU memory on Apple Silicon must not be summed as independent physical pools.

The external harness records hardware, OS/Rust versions, base commit/dirty state, executable SHA-256, workload SHA-256, environment settings, exact command lines, and raw process logs. It atomically checkpoints the JSON after each complete case; `complete` becomes true only after managed shutdown drains. Reports are updated after each completed backend.

## Results and interpretation

The measured matrix is published in [the CPU/Metal report](benchmarks/flash-cpu-metal/report.md), with machine-readable raw results beside it. F32 compares the same precision across backends. F16 is a mixed-precision comparison and must be read alongside [the numerical qualification](clef-flash-verification.md).

Small sample counts make p95/p99 descriptive order statistics, often equal to the maximum observation; they do not establish a production tail latency or SLO. Queue time under concurrency is intentional. Rejected work is excluded from successful latency/throughput and shown separately. The suite does not establish a sustained 10,000-decision memory guarantee.

Criterion measures request parsing, bounded JSON validation, and reference-compatible JSON rendering without loading weights. The exported [domain results](benchmarks/flash-cpu-metal/domain.md) include confidence intervals and raw samples. Local artifacts are also stored under Cargo's target directory in `criterion`; `make verify-benchmark` checks the inference runner's statistics and configuration rules.
