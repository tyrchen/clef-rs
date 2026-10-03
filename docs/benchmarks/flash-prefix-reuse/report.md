# Exact prefix reuse: normal-path measurements

Measured 2026-10-03T06:33:49.010180+00:00.

Model loading, JSON parsing, HTTP/authentication, and hit preparation are excluded from each latency sample. Every sample includes encoding, inference, unrounded answer conversion and device completion. Full/capture/hit order rotates; capture runs one full prefill and saves state, while hit computes the suffix over retained complete context. Short inputs bypass caching. No input compression, distillation or precision change is used.

| Profile | Tokens / fields | n per mode | Reused tokens | Old cold ms | New cold ms | Capture ms | Hit ms | Cold→hit speedup |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| metal:f16 | 139 / 1 | 20 | 0 | 162.262 | 147.606 | 147.342 | 147.575 | 1.00× |
| metal:f16 | 1024 / 3 | 20 | 768 | 825.652 | 692.986 | 701.887 | 226.081 | 3.07× |
| metal:f16 | 4096 / 3 | 20 | 3840 | 3724.350 | 3111.542 | 3176.597 | 321.937 | 9.67× |

[metal:f16 raw samples](metal-f16.json); [matched old cold samples](metal-f16-baseline.json).

| Profile | Case | Hit p50 / p95 / p99 ms | Hit SD ms | Capture overhead | Unrounded max / mean drift | Retained MiB |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| metal:f16 | short-139 | 147.690 / 148.634 / 148.818 | 0.681 | -0.18% | 0 / 0 | 0.00 |
| metal:f16 | long-1024 | 226.016 / 228.449 / 228.524 | 1.362 | 1.28% | 0 / 0 | 86.25 |
| metal:f16 | max-4096 | 321.755 / 323.914 / 324.627 | 1.345 | 2.09% | 0 / 0 | 230.27 |

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
