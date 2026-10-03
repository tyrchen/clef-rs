# CPU / Metal Flash performance measurements

Measured 2026-10-03T01:33:11.774955+00:00.

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

All backends ran in separate sequential processes using the same YAML workload and fixed thread counts. Normal-path timings measure DirectEngine on prevalidated requests and include encoding, model execution, answer conversion and GPU completion. JSON parsing, HTTP/authentication and transport serialization are excluded; concurrency uses the embedded Runtime. Cold process means a new process: OS file caches and Metal driver/shader caches were not cleared.

| Profile | Snapshot verify s | Model load s | First decision s | Managed load/warmup s | Process peak RSS GiB | Post-load direct GPU peak GiB | OS peak footprint GiB | Average active CPU cores |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| metal:f32 | 6.029 | 22.760 | 0.650 | 24.644 | 30.422 | 35.920 | 40.531 | 0.163 |
| metal:f16 | 6.428 | 17.459 | 0.240 | 18.617 | 25.572 | 18.527 | 26.756 | 0.329 |

| Profile | Workload | Tokens | Fields | Options/field | n | Mean ms | p50 ms | p95 ms | p99 ms | req/s | input tokens/s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| metal:f32 | short-256 | 256 | 1 | 2 | 10 | 622.549 | 622.500 | 624.217 | 624.217 | 1.6062 | 411.19 |
| metal:f32 | medium-1024 | 1024 | 1 | 2 | 10 | 2485.774 | 2485.072 | 2491.262 | 2491.262 | 0.4023 | 411.94 |
| metal:f32 | long-4096 | 4096 | 1 | 2 | 3 | 10266.398 | 10266.193 | 10272.571 | 10272.571 | 0.0974 | 398.97 |
| metal:f32 | mixed-8-fields | 1024 | 8 | 4 | 10 | 2503.339 | 2504.076 | 2511.337 | 2511.337 | 0.3995 | 409.05 |
| metal:f32 | wide-32-fields | 3072 | 32 | 2 | 3 | 7677.114 | 7672.019 | 7687.592 | 7687.592 | 0.1303 | 400.15 |
| metal:f32 | large-option-schema | 4096 | 4 | 64 | 3 | 10307.017 | 10307.442 | 10309.563 | 10309.563 | 0.0970 | 397.40 |
| metal:f32 | minimal-139 | 139 | 1 | 2 | 20 | 449.957 | 449.876 | 450.974 | 451.049 | 2.2223 | 308.90 |
| metal:f32 | compact-192 | 192 | 1 | 2 | 20 | 493.134 | 493.078 | 495.099 | 495.230 | 2.0277 | 389.33 |
| metal:f16 | short-256 | 256 | 1 | 2 | 10 | 225.560 | 225.458 | 226.767 | 226.767 | 4.4329 | 1134.82 |
| metal:f16 | medium-1024 | 1024 | 1 | 2 | 10 | 880.720 | 880.568 | 881.959 | 881.959 | 1.1354 | 1162.65 |
| metal:f16 | long-4096 | 4096 | 1 | 2 | 3 | 3902.149 | 3902.137 | 3912.952 | 3912.952 | 0.2563 | 1049.66 |
| metal:f16 | mixed-8-fields | 1024 | 8 | 4 | 10 | 894.514 | 894.391 | 896.698 | 896.698 | 1.1179 | 1144.71 |
| metal:f16 | wide-32-fields | 3072 | 32 | 2 | 3 | 2867.769 | 2867.148 | 2874.153 | 2874.153 | 0.3487 | 1071.20 |
| metal:f16 | large-option-schema | 4096 | 4 | 64 | 3 | 3921.047 | 3924.167 | 3925.402 | 3925.402 | 0.2550 | 1044.61 |
| metal:f16 | minimal-139 | 139 | 1 | 2 | 20 | 169.785 | 169.559 | 170.951 | 171.451 | 5.8888 | 818.54 |
| metal:f16 | compact-192 | 192 | 1 | 2 | 20 | 193.051 | 192.913 | 194.343 | 195.888 | 5.1794 | 994.44 |

Diagnostic stages add explicit barriers and are measured separately from the latency samples above.

| Profile | Workload | Encoding ms | Backbone/vision ms | Head/readback ms | Conversion ms | Final sync ms | Diagnostic total ms | Metal allocated GiB |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| metal:f32 | short-256 | 0.212 | 615.490 | 8.056 | 0.005 | 0.003 | 623.766 | 33.810 |
| metal:f32 | medium-1024 | 0.644 | 2473.651 | 21.864 | 0.006 | 0.003 | 2496.168 | 33.810 |
| metal:f32 | long-4096 | 2.207 | 10201.060 | 75.025 | 0.007 | 0.003 | 10278.302 | 33.810 |
| metal:f32 | mixed-8-fields | 0.719 | 2469.821 | 33.042 | 0.014 | 0.003 | 2503.599 | 33.810 |
| metal:f32 | wide-32-fields | 1.924 | 7593.848 | 101.120 | 0.024 | 0.003 | 7696.920 | 33.810 |
| metal:f32 | large-option-schema | 2.884 | 10191.280 | 108.192 | 0.062 | 0.003 | 10302.422 | 33.810 |
| metal:f32 | minimal-139 | 0.160 | 443.689 | 6.695 | 0.005 | 0.002 | 450.551 | 33.810 |
| metal:f32 | compact-192 | 0.175 | 485.593 | 7.516 | 0.004 | 0.002 | 493.291 | 33.810 |
| metal:f16 | short-256 | 0.196 | 218.168 | 8.121 | 0.004 | 0.002 | 226.493 | 17.135 |
| metal:f16 | medium-1024 | 0.652 | 862.199 | 21.069 | 0.005 | 0.003 | 883.928 | 17.135 |
| metal:f16 | long-4096 | 2.194 | 3804.562 | 76.858 | 0.006 | 0.003 | 3883.624 | 17.135 |
| metal:f16 | mixed-8-fields | 0.688 | 864.311 | 32.061 | 0.013 | 0.003 | 897.076 | 17.135 |
| metal:f16 | wide-32-fields | 1.909 | 2778.104 | 98.838 | 0.022 | 0.003 | 2878.877 | 17.135 |
| metal:f16 | large-option-schema | 2.696 | 3826.682 | 103.642 | 0.061 | 0.003 | 3933.086 | 17.135 |
| metal:f16 | minimal-139 | 0.142 | 163.930 | 6.696 | 0.004 | 0.003 | 170.775 | 17.135 |
| metal:f16 | compact-192 | 0.168 | 186.437 | 7.595 | 0.004 | 0.003 | 194.208 | 17.135 |

Managed-runtime concurrency uses the same fixed short workload, FIFO within a principal and round-robin across principals. The harness has one device worker, eight ingress/queue slots, eight preparation workers, and 300-second queue/request/shutdown limits; these differ from production defaults. Admission errors are counted separately.

| Profile | Concurrency | Attempts | Successes | Errors | Successful req/s | Successful p50 ms | Successful p95 ms |
| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| metal:f32 | 1 | 2 | 2 | {} | 1.5997 | 624.479 | 625.720 |
| metal:f32 | 2 | 4 | 4 | {} | 1.6017 | 1248.433 | 1248.908 |
| metal:f32 | 8 | 16 | 16 | {} | 1.6002 | 4992.951 | 5004.952 |
| metal:f32 | 16 | 32 | 16 | {"queueFull": 16} | 1.6001 | 4999.081 | 5001.289 |
| metal:f16 | 1 | 2 | 2 | {} | 4.3911 | 227.382 | 228.058 |
| metal:f16 | 2 | 4 | 4 | {} | 4.4015 | 452.990 | 455.770 |
| metal:f16 | 8 | 16 | 16 | {} | 4.3866 | 1820.113 | 1824.645 |
| metal:f16 | 16 | 32 | 16 | {"queueFull": 16} | 4.3730 | 1827.791 | 1831.713 |

Percentiles use nearest-rank order statistics. With small n, p95/p99 often equal the maximum observation; these are descriptive samples, not production-tail estimates or SLOs. The report preserves every raw timing, error count and environment setting. Metal allocator snapshots include retained buffers. The direct phase also samples allocator usage every 100 ms; its reported peak can miss shorter spikes and excludes initial loading and managed-runtime reload. Metal private allocations can appear as wired system memory outside process RSS, so RSS alone is not a physical-memory comparison. OS process RSS/footprint high-water includes both direct and managed phases. Footprint is macOS accounting, not a sum of the RSS and GPU columns; unified CPU/GPU allocations must not be summed as independent physical-memory pools.

CPU F32 and Metal F32 are a same-precision comparison. Metal F16 uses F16 text backbone with an F32 classifier head/vision tower, residuals/convolution, norms/attention accumulation and recurrent state; compare its numerical qualification separately. CPU uses the default Candle pure-Rust backend, without Accelerate/MKL. Thread settings and the actual executable SHA-256 are recorded in metadata.json.

The measured Rust/MSL implementation is committed as `190a459c52c3389c79ed0ad05ebf532c365fd81c`. The dirty-tree flag reflects documentation/measurement artifacts and unrelated local documentation; no Rust/MSL source changed after that commit.

See [the previous M5 campaign](../flash-metal-m5/report.md) and [the original CPU/Metal baseline](../flash-cpu-metal/report.md). Exact YAML workloads and sample counts match the previous M5 campaign. CPU equations did not change; CPU correctness gates were repeated and the prior 82-minute full-weight CPU performance campaign is reused.

Additional evidence: [six full-model probability qualifications](qualification.json), [synthetic tile/backbone diagnostics](operator-diagnostics.json), [implementation and before/after analysis](../../clef-flash-metal-performance.md), and [executed correctness gates](../../clef-flash-verification.md). Operator diagnostics use synchronization barriers and are excluded from normal samples.
