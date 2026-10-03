# Flash exact prefix reuse and preparation fusion

The implementation preserves the complete input, the released model and the qualified CPU F32 / Metal F32 / mixed-F16 precision profiles. It adds exact text-prefix continuation on CPU and Metal, and a bitwise-checked Metal DeltaNet preparation kernel. Distillation, input compression, relaxed precision and quantization are not used.

Prefix reuse is useful when a principal repeatedly evaluates a long state with different questions or appends text while retaining the captured token prefix. The encoder places state before schema; changing only the schema can therefore reuse most state computation. A completely different state runs the full backbone. The classifier still receives every token's final feature, including retained prefix evidence.

## Enable and inspect

The library defaults to zero cache capacity. [The Metal serving example](../examples/clef.metal.yaml) explicitly enables:

```yaml
runtime:
  prefixCache:
    capacityBytes: 536870912
    maxEntries: 1
    ttlSeconds: 300
```

The cache is owned by one engine/device worker. Keys require an exact token-prefix match and the same principal; the snapshot, precision and device remain fixed for that owner. Captures align to 256 tokens before the schema and skip prefixes shorter than 512 tokens. Capacity is limited to 4 GiB, entries to 16, and lifetime to one hour. Expiration removes entries on the next cache-enabled text lookup. Images bypass reuse.

Embedded code can use `DirectEngine::load_with_prefix_cache` and `PrefixCacheConfig::new`. `decide_with_options` supplies a principal/deadline, `prefix_cache_stats` reports aggregate hits/captures/skipped tokens/live bytes, and `clear_prefix_cache` drops entries. Ordinary direct `decide` uses the embedded principal. Managed HTTP calls use their existing authenticated principal. Worker recovery creates an empty cache. `decide_uncached` performs a complete prefill without reading or changing retained entries.

The admission planner includes existing entries plus a staged replacement and 20% reserve. `plan-memory` uses that same cache-aware plan. A 3,840-token prefix retains approximately 230.25 MiB in mixed F16 and 350.25 MiB in F32, plus bounded keys. Full-attention K/V remain in their projection dtype; DeltaNet state, convolution history and final head evidence stay F32. Exact-sized tracked Metal buffers prevent best-fit scratch reuse from retaining much larger allocations. This fixed a reproduced F32 GPU OOM in the long-state/scope-change qualification.

A cache miss captures during one original full prefill and publishes only after successful head execution, device synchronization and cancellation/deadline checks. Cache hits do not mutate saved tensors. Changed prefixes, principals or expired entries cannot reuse mismatched state. Debug output excludes request content and principal data.

## Measurements and reproduction

The benchmark rotates three modes: complete computation, miss/capture, and prepared hit. Each timed sample includes encoding, execution, typed conversion and GPU completion. Hit preparation, model loading, JSON parsing and network/authentication are excluded. Every mode checks unrounded probability drift against an independent complete prefill; the original 1e-3 maximum / 1e-4 mean limits remain fixed. The archived executable enables a separate, matched cold comparison.

Measured on the 64 GiB Apple M5 Pro with eight fixed CPU threads. Values below are means in milliseconds; F16 uses 20 samples per mode and the scoped F32 campaign uses five. The old cold phase uses the preserved executable under the same workload. Separate processes leave normal system/thermal variation in that comparison.

| Profile | Tokens | Old complete | New complete | Capture | Prefix hit | Reused tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Metal mixed F16 | 139 | 162.262 | 147.606 | 147.342 | 147.575 | 0 |
| Metal mixed F16 | 1024 | 825.652 | 692.986 | 701.887 | 226.081 | 768 |
| Metal mixed F16 | 4096 | 3724.350 | 3111.542 | 3176.597 | 321.937 | 3840 |
| Metal F32 | 139 | 440.222 | 425.562 | 426.379 | 426.101 | 0 |
| Metal F32 | 1024 | 2450.983 | 2324.003 | 2334.686 | 614.527 | 768 |
| Metal F32 | 4096 | 10115.267 | 9555.857 | 9623.365 | 711.034 | 3840 |
| CPU F32 | 1024 | — | 50816.350 | 50916.374 | 13887.560 | 768 |

F16 complete computation is approximately 16% faster at 1024/4096 tokens. A 4096-token F16 hit is 9.67× faster than the new complete path, or 11.57× faster than the old complete path. Capturing adds 1.28%/2.09% to the new 1024/4096-token complete path; both remain faster than the old executable. These large hit gains require a shared prefix. Short inputs bypass retention and independent states receive the cold-path improvement. All benchmark maximum/mean **unrounded** reuse errors are zero.

The [F16 report](benchmarks/flash-prefix-reuse/report.md), [F32 report](benchmarks/flash-prefix-reuse-f32/report.md) and [CPU report](benchmarks/flash-prefix-reuse-cpu/report.md) preserve every sample, descriptive percentiles/standard deviations, retained bytes and source/executable/workload hashes. CPU uses three samples per mode, without an old executable phase: capture overhead is 0.20%, and prefix hits are 3.66× faster than the new complete path. CPU execution remains much slower than Metal on this MBP. Synthetic repeated-word timing workloads measure latency; decision-quality qualification uses the independent release corpus below.

```sh
make verify-cpu
make verify-metal verify-metal-pointwise verify-prefix-reuse
make verify-metal-release verify-metal-media
make verify-metal-serving PYTHON=/path/to/reference-environment/bin/python
make bench-prefix-reuse
make bench-prefix-reuse PREFIX_REUSE_PROFILES=cpu:f32 PREFIX_REUSE_CONFIG=examples/clef.prefix-reuse-cpu-benchmark.yaml PREFIX_REUSE_RESULTS=docs/benchmarks/flash-prefix-reuse-cpu
```

The serving smoke test needs the existing pinned reference environment's PyYAML. Benchmark/report rendering uses Python's standard library. An optional `PREFIX_REUSE_BASELINE=/path/to/previous/benchmark` adds an old cold phase; the previous executable must support the same original benchmark arguments. The completed local comparison used the preserved `874255d` implementation executable with SHA-256 `2c239aa584fe685c2cadfc2db280bdd4f9493378cd249318fb29e80e272aea40`.

The normal MBP campaign uses [20 samples per mode at 139/1024/4096 tokens](../examples/clef.prefix-reuse-benchmark.yaml). `PREFIX_REUSE_PROFILES=metal:f32` selects full F32. The [scoped CPU campaign](../examples/clef.prefix-reuse-cpu-benchmark.yaml) uses three 1024-token samples per mode; it is separate from the previously completed 82-minute CPU matrix. `PREFIX_REUSE_CONFIG`, `PREFIX_REUSE_RESULTS`, and `BENCHMARK_THREADS` select bounded workloads, output location and fixed thread counts. `make prefix-reuse-report` validates and renders archived samples without executing the model.

## Qualification

CPU/vision and Metal/vision builds, tests, nightly formatting, pedantic Clippy and warning-free documentation passed. Synthetic continuation covers changed suffixes, one/two/three-token convolution histories, immutable recurrence state, byte accounting, exact token/principal mismatch, expiration, clear, single/multiple-entry LRU eviction and rejected configuration bounds. Real Metal tests verify bitwise preparation equality, bitwise register-state capture/resume, exact physical snapshot lengths, strided F32/F16 copying, GQA and partial causal tiles.

The [structured qualification record](benchmarks/flash-prefix-reuse/qualification.json) includes source hashes, executed gates, exact oracle comparisons and verification scope.

Full-weight Metal F32 and F16 continuation passed seven unrounded comparisons per precision: 4096-token capture/hit, changed schema, a different principal, appended state, changed prefix and a subsequent hit after cancellation. All seven had zero maximum and mean drift against complete execution in the same profile. Authenticated HTTP/TCP now exercises a cache-enabled long-state workload, including principal separation and a repeated HTTP hit. CLI, PNG/JPEG, provenance, metrics and SIGTERM drain passed for both precisions.

| Python F32 oracle corpus | Metal F32 max / mean | Mixed F16 max / mean |
| --- | ---: | ---: |
| Text: 100 records, 800 probabilities | 3.114343e-6 / 2.027748e-7 | 4.433692e-4 / 3.701660e-5 |
| 4096-token state + PNG: 24 probabilities | 5.424023e-6 / 1.789847e-6 | 3.281832e-4 / 9.007025e-5 |
| JPEG: 16 probabilities | 4.351139e-6 / 1.026667e-6 | 4.187524e-4 / 9.541318e-5 |

The JPEG mean remains close to the existing limit; the limit was not relaxed. No new dependency or lockfile change was required, so supply-chain gates were not repeated for this feature; their preceding verified results remain recorded in the existing reports. The unchanged full CPU oracle/matrix was not rerun mechanically: CPU continuation is covered by operator/backbone tests and the scoped full-weight campaign.

See the [design contract](../specs/clef-prefix-reuse-design.md) and [underlying research](research/clef-metal-latency-research.md) for invariants and rejected alternatives.
