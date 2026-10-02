# CPU / Metal Flash performance measurements

Measured 2026-10-02T21:25:39.199752+00:00.

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
| metal:f32 | 5.962 | 21.060 | 0.724 | 23.917 | 31.309 | 36.077 | 42.479 | 0.150 |
| metal:f16 | 6.474 | 17.822 | 0.652 | 18.788 | 25.528 | 18.712 | 25.731 | 0.153 |

| Profile | Workload | Tokens | Fields | Options/field | n | Mean ms | p50 ms | p95 ms | p99 ms | req/s | input tokens/s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| metal:f32 | short-256 | 256 | 1 | 2 | 10 | 682.610 | 682.519 | 684.534 | 684.534 | 1.4649 | 375.02 |
| metal:f32 | medium-1024 | 1024 | 1 | 2 | 10 | 2772.414 | 2770.898 | 2783.790 | 2783.790 | 0.3607 | 369.35 |
| metal:f32 | long-4096 | 4096 | 1 | 2 | 3 | 11509.453 | 11505.843 | 11519.948 | 11519.948 | 0.0869 | 355.88 |
| metal:f32 | mixed-8-fields | 1024 | 8 | 4 | 10 | 2789.303 | 2788.203 | 2794.255 | 2794.255 | 0.3585 | 367.11 |
| metal:f32 | wide-32-fields | 3072 | 32 | 2 | 3 | 8602.194 | 8600.883 | 8605.184 | 8605.184 | 0.1162 | 357.12 |
| metal:f32 | large-option-schema | 4096 | 4 | 64 | 3 | 11534.664 | 11536.522 | 11539.525 | 11539.525 | 0.0867 | 355.10 |
| metal:f32 | minimal-139 | 139 | 1 | 2 | 20 | 479.019 | 478.992 | 479.670 | 479.746 | 2.0875 | 290.17 |
| metal:f32 | compact-192 | 192 | 1 | 2 | 20 | 536.875 | 536.799 | 537.370 | 537.419 | 1.8626 | 357.61 |
| metal:f16 | short-256 | 256 | 1 | 2 | 10 | 637.118 | 637.160 | 638.244 | 638.244 | 1.5695 | 401.80 |
| metal:f16 | medium-1024 | 1024 | 1 | 2 | 10 | 2544.382 | 2544.268 | 2546.635 | 2546.635 | 0.3930 | 402.45 |
| metal:f16 | long-4096 | 4096 | 1 | 2 | 3 | 10581.239 | 10581.219 | 10581.361 | 10581.361 | 0.0945 | 387.10 |
| metal:f16 | mixed-8-fields | 1024 | 8 | 4 | 10 | 2557.617 | 2557.327 | 2559.874 | 2559.874 | 0.3910 | 400.37 |
| metal:f16 | wide-32-fields | 3072 | 32 | 2 | 3 | 7912.541 | 7912.923 | 7912.966 | 7912.966 | 0.1264 | 388.24 |
| metal:f16 | large-option-schema | 4096 | 4 | 64 | 3 | 10613.320 | 10614.792 | 10614.886 | 10614.886 | 0.0942 | 385.93 |
| metal:f16 | minimal-139 | 139 | 1 | 2 | 20 | 448.033 | 448.026 | 448.563 | 448.722 | 2.2319 | 310.23 |
| metal:f16 | compact-192 | 192 | 1 | 2 | 20 | 486.072 | 485.967 | 486.836 | 487.430 | 2.0572 | 394.99 |

Diagnostic stages add explicit barriers and are measured separately from the latency samples above.

| Profile | Workload | Encoding ms | Backbone/vision ms | Head/readback ms | Conversion ms | Final sync ms | Diagnostic total ms | Metal allocated GiB |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| metal:f32 | short-256 | 0.219 | 674.531 | 8.107 | 0.004 | 0.003 | 682.864 | 33.810 |
| metal:f32 | medium-1024 | 0.677 | 2746.810 | 21.379 | 0.005 | 0.003 | 2768.873 | 33.810 |
| metal:f32 | long-4096 | 2.222 | 11434.476 | 75.677 | 0.006 | 0.003 | 11512.384 | 33.810 |
| metal:f32 | mixed-8-fields | 0.738 | 2762.290 | 33.691 | 0.014 | 0.003 | 2796.736 | 33.810 |
| metal:f32 | wide-32-fields | 1.951 | 8510.590 | 101.508 | 0.027 | 0.003 | 8614.079 | 33.810 |
| metal:f32 | large-option-schema | 2.761 | 11429.049 | 106.081 | 0.061 | 0.770 | 11538.722 | 33.810 |
| metal:f32 | minimal-139 | 0.144 | 473.031 | 6.802 | 0.004 | 0.002 | 479.983 | 33.810 |
| metal:f32 | compact-192 | 0.181 | 529.595 | 7.638 | 0.006 | 0.003 | 537.422 | 33.810 |
| metal:f16 | short-256 | 0.199 | 630.969 | 8.057 | 0.004 | 0.002 | 639.232 | 17.132 |
| metal:f16 | medium-1024 | 0.642 | 2526.237 | 21.681 | 0.005 | 0.003 | 2548.569 | 17.132 |
| metal:f16 | long-4096 | 2.217 | 10514.494 | 71.982 | 0.006 | 0.003 | 10588.702 | 17.132 |
| metal:f16 | mixed-8-fields | 0.712 | 2527.027 | 32.936 | 0.013 | 0.003 | 2560.691 | 17.132 |
| metal:f16 | wide-32-fields | 1.957 | 7825.487 | 100.267 | 0.023 | 0.002 | 7927.738 | 17.132 |
| metal:f16 | large-option-schema | 2.988 | 10506.030 | 104.167 | 0.061 | 0.674 | 10613.919 | 17.132 |
| metal:f16 | minimal-139 | 0.144 | 442.237 | 6.717 | 0.004 | 0.003 | 449.105 | 17.132 |
| metal:f16 | compact-192 | 0.189 | 480.195 | 7.453 | 0.005 | 0.003 | 487.845 | 17.132 |

Managed-runtime concurrency uses the same fixed short workload, FIFO within a principal and round-robin across principals. The harness has one device worker, eight ingress/queue slots, eight preparation workers, and 300-second queue/request/shutdown limits; these differ from production defaults. Admission errors are counted separately.

| Profile | Concurrency | Attempts | Successes | Errors | Successful req/s | Successful p50 ms | Successful p95 ms |
| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| metal:f32 | 1 | 2 | 2 | {} | 1.4672 | 681.497 | 681.638 |
| metal:f32 | 2 | 4 | 4 | {} | 1.4680 | 1361.156 | 1363.884 |
| metal:f32 | 8 | 16 | 16 | {} | 1.4644 | 5457.927 | 5465.744 |
| metal:f32 | 16 | 32 | 16 | {"queueFull": 16} | 1.4635 | 5461.518 | 5468.193 |
| metal:f16 | 1 | 2 | 2 | {} | 1.5669 | 637.693 | 638.703 |
| metal:f16 | 2 | 4 | 4 | {} | 1.5689 | 1274.552 | 1275.279 |
| metal:f16 | 8 | 16 | 16 | {} | 1.5681 | 5094.517 | 5106.173 |
| metal:f16 | 16 | 32 | 16 | {"queueFull": 16} | 1.5622 | 5111.757 | 5130.782 |

Percentiles use nearest-rank order statistics. With small n, p95/p99 often equal the maximum observation; these are descriptive samples, not production-tail estimates or SLOs. The report preserves every raw timing, error count and environment setting. Metal allocator snapshots include retained buffers. The direct phase also samples allocator usage every 100 ms; its reported peak can miss shorter spikes and excludes initial loading and managed-runtime reload. Metal private allocations can appear as wired system memory outside process RSS, so RSS alone is not a physical-memory comparison. OS process RSS/footprint high-water includes both direct and managed phases. Footprint is macOS accounting, not a sum of the RSS and GPU columns; unified CPU/GPU allocations must not be summed as independent physical-memory pools.

CPU F32 and Metal F32 are a same-precision comparison. Metal F16 uses F16 text backbone with an F32 classifier head/vision tower, residuals/convolution, norms/attention accumulation and recurrent state; compare its numerical qualification separately. CPU uses the default Candle pure-Rust backend, without Accelerate/MKL. Thread settings and the actual executable SHA-256 are recorded in metadata.json.
