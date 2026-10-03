# Exact prefix reuse: normal-path measurements

Measured 2026-10-03T06:41:13.947274+00:00.

Model loading, JSON parsing, HTTP/authentication, and hit preparation are excluded from each latency sample. Every sample includes encoding, inference, unrounded answer conversion and device completion. Full/capture/hit order rotates; capture runs one full prefill and saves state, while hit computes the suffix over retained complete context. Short inputs bypass caching. No input compression, distillation or precision change is used.

| Profile | Tokens / fields | n per mode | Reused tokens | Old cold ms | New cold ms | Capture ms | Hit ms | Cold→hit speedup |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| metal:f32 | 139 / 1 | 5 | 0 | 440.222 | 425.562 | 426.379 | 426.101 | 1.00× |
| metal:f32 | 1024 / 3 | 5 | 768 | 2450.983 | 2324.003 | 2334.686 | 614.527 | 3.78× |
| metal:f32 | 4096 / 3 | 5 | 3840 | 10115.267 | 9555.857 | 9623.365 | 711.034 | 13.44× |

[metal:f32 raw samples](metal-f32.json); [matched old cold samples](metal-f32-baseline.json).

| Profile | Case | Hit p50 / p95 / p99 ms | Hit SD ms | Capture overhead | Unrounded max / mean drift | Retained MiB |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| metal:f32 | short-139 | 425.691 / 427.966 / 427.966 | 0.978 | 0.19% | 0 / 0 | 0.00 |
| metal:f32 | long-1024 | 614.205 / 615.892 / 615.892 | 0.905 | 0.46% | 0 / 0 | 110.25 |
| metal:f32 | max-4096 | 711.134 / 711.600 / 711.600 | 0.476 | 0.71% | 0 / 0 | 350.27 |

A hit requires the same principal and an exactly matching long state prefix. Independent states pay the cold/capture cost. Library caching defaults to disabled; the MBP serving YAML explicitly enables it. Cache staging is included in the admission plan, and all head evidence is retained. Synthetic repeated-word workload timings are not application decision-quality evidence; full-weight release and changed-schema/state qualification is reported separately. Percentiles from small samples do not establish an SLO.

Hardware, compiler, executable/workload/source hashes and per-process elapsed time are in [metadata.json](metadata.json). Processes are sequential; OS/driver caches are not cleared. Baseline and new runs are different processes, so their cold means also include normal thermal and system variation. Tensor memory is live retained storage, not total process or GPU allocator peak.

```json
{
  "system": "Darwin",
  "release": "25.5.0",
  "machine": "arm64",
  "cpu": "Apple M5 Pro",
  "ramBytes": 68719476736,
  "cpuCores": 18,
  "osVersion": "26.5.2",
  "gpus": [
    {
      "model": "Apple M5 Pro",
      "cores": "20",
      "metal": "spdisplays_metal4"
    }
  ]
}
```
