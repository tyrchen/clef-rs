# Exact prefix reuse: normal-path measurements

Measured 2026-10-03T06:47:48.862723+00:00.

Model loading, JSON parsing, HTTP/authentication, and hit preparation are excluded from each latency sample. Every sample includes encoding, inference, unrounded answer conversion and device completion. Full/capture/hit order rotates; capture runs one full prefill and saves state, while hit computes the suffix over retained complete context. Short inputs bypass caching. No input compression, distillation or precision change is used.

| Profile | Tokens / fields | n per mode | Reused tokens | Old cold ms | New cold ms | Capture ms | Hit ms | Cold→hit speedup |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| cpu:f32 | 1024 / 3 | 3 | 768 | — | 50816.350 | 50916.374 | 13887.560 | 3.66× |

[cpu:f32 raw samples](cpu-f32.json); No old executable comparison was requested.

| Profile | Case | Hit p50 / p95 / p99 ms | Hit SD ms | Capture overhead | Unrounded max / mean drift | Retained MiB |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| cpu:f32 | long-1024 | 13894.888 / 13898.634 / 13898.634 | 13.102 | 0.20% | 0 / 0 | 110.25 |

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
