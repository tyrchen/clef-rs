# CPU / Metal Flash performance measurements

Measured 2026-10-02T21:58:38.187319+00:00.

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
| metal:f32 | 5.964 | 22.813 | 0.714 | 25.282 | 30.157 | 36.077 | 42.300 | 0.154 |
| metal:f16 | 6.401 | 17.515 | 0.323 | 18.592 | 25.531 | 18.712 | 26.826 | 0.255 |

| Profile | Workload | Tokens | Fields | Options/field | n | Mean ms | p50 ms | p95 ms | p99 ms | req/s | input tokens/s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| metal:f32 | short-256 | 256 | 1 | 2 | 10 | 682.777 | 682.539 | 684.864 | 684.864 | 1.4646 | 374.93 |
| metal:f32 | medium-1024 | 1024 | 1 | 2 | 10 | 2773.547 | 2773.986 | 2777.559 | 2777.559 | 0.3605 | 369.20 |
| metal:f32 | long-4096 | 4096 | 1 | 2 | 3 | 11517.158 | 11518.473 | 11520.398 | 11520.398 | 0.0868 | 355.64 |
| metal:f32 | mixed-8-fields | 1024 | 8 | 4 | 10 | 2802.787 | 2791.280 | 2908.817 | 2908.817 | 0.3568 | 365.35 |
| metal:f32 | wide-32-fields | 3072 | 32 | 2 | 3 | 8647.113 | 8650.296 | 8650.657 | 8650.657 | 0.1156 | 355.26 |
| metal:f32 | large-option-schema | 4096 | 4 | 64 | 3 | 11598.797 | 11591.491 | 11626.050 | 11626.050 | 0.0862 | 353.14 |
| metal:f32 | minimal-139 | 139 | 1 | 2 | 20 | 479.022 | 479.057 | 479.601 | 479.609 | 2.0875 | 290.16 |
| metal:f32 | compact-192 | 192 | 1 | 2 | 20 | 536.868 | 537.102 | 537.655 | 537.661 | 1.8626 | 357.62 |
| metal:f16 | short-256 | 256 | 1 | 2 | 10 | 306.912 | 306.873 | 307.601 | 307.601 | 3.2581 | 834.06 |
| metal:f16 | medium-1024 | 1024 | 1 | 2 | 10 | 1226.342 | 1225.600 | 1230.129 | 1230.129 | 0.8154 | 834.99 |
| metal:f16 | long-4096 | 4096 | 1 | 2 | 3 | 6091.856 | 6093.700 | 6097.432 | 6097.432 | 0.1642 | 672.37 |
| metal:f16 | mixed-8-fields | 1024 | 8 | 4 | 10 | 1238.903 | 1238.714 | 1240.754 | 1240.754 | 0.8071 | 826.52 |
| metal:f16 | wide-32-fields | 3072 | 32 | 2 | 3 | 4388.024 | 4390.485 | 4396.966 | 4396.966 | 0.2279 | 700.08 |
| metal:f16 | large-option-schema | 4096 | 4 | 64 | 3 | 6125.369 | 6124.262 | 6136.318 | 6136.318 | 0.1633 | 668.69 |
| metal:f16 | minimal-139 | 139 | 1 | 2 | 20 | 209.566 | 209.515 | 210.539 | 210.949 | 4.7712 | 663.19 |
| metal:f16 | compact-192 | 192 | 1 | 2 | 20 | 250.308 | 250.277 | 251.608 | 251.930 | 3.9948 | 767.00 |

Diagnostic stages add explicit barriers and are measured separately from the latency samples above.

| Profile | Workload | Encoding ms | Backbone/vision ms | Head/readback ms | Conversion ms | Final sync ms | Diagnostic total ms | Metal allocated GiB |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| metal:f32 | short-256 | 0.203 | 673.878 | 8.178 | 0.005 | 0.002 | 682.267 | 33.810 |
| metal:f32 | medium-1024 | 0.643 | 2757.902 | 21.320 | 0.005 | 0.002 | 2779.872 | 33.810 |
| metal:f32 | long-4096 | 2.259 | 11457.989 | 73.850 | 0.006 | 0.003 | 11534.107 | 33.810 |
| metal:f32 | mixed-8-fields | 0.717 | 2764.472 | 34.506 | 0.014 | 0.003 | 2799.713 | 33.810 |
| metal:f32 | wide-32-fields | 1.961 | 8549.088 | 105.167 | 0.023 | 0.003 | 8656.243 | 33.810 |
| metal:f32 | large-option-schema | 2.729 | 11491.131 | 105.354 | 0.061 | 0.762 | 11600.038 | 33.810 |
| metal:f32 | minimal-139 | 0.153 | 474.698 | 6.681 | 0.005 | 0.003 | 481.540 | 33.810 |
| metal:f32 | compact-192 | 0.185 | 529.921 | 7.406 | 0.005 | 0.003 | 537.520 | 33.810 |
| metal:f16 | short-256 | 0.196 | 299.734 | 7.978 | 0.004 | 0.002 | 307.915 | 17.132 |
| metal:f16 | medium-1024 | 0.630 | 1208.278 | 20.685 | 0.005 | 0.003 | 1229.601 | 17.132 |
| metal:f16 | long-4096 | 2.241 | 6013.809 | 73.209 | 0.006 | 0.003 | 6089.269 | 17.132 |
| metal:f16 | mixed-8-fields | 0.708 | 1209.627 | 32.397 | 0.013 | 0.003 | 1242.749 | 17.132 |
| metal:f16 | wide-32-fields | 1.972 | 4309.750 | 98.788 | 0.023 | 0.003 | 4410.537 | 17.132 |
| metal:f16 | large-option-schema | 2.675 | 6030.524 | 104.773 | 0.061 | 0.688 | 6138.722 | 17.132 |
| metal:f16 | minimal-139 | 0.143 | 204.051 | 6.742 | 0.004 | 0.002 | 210.942 | 17.132 |
| metal:f16 | compact-192 | 0.171 | 243.285 | 7.317 | 0.004 | 0.002 | 250.779 | 17.132 |

Managed-runtime concurrency uses the same fixed short workload, FIFO within a principal and round-robin across principals. The harness has one device worker, eight ingress/queue slots, eight preparation workers, and 300-second queue/request/shutdown limits; these differ from production defaults. Admission errors are counted separately.

| Profile | Concurrency | Attempts | Successes | Errors | Successful req/s | Successful p50 ms | Successful p95 ms |
| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| metal:f32 | 1 | 2 | 2 | {} | 1.4652 | 682.350 | 682.613 |
| metal:f32 | 2 | 4 | 4 | {} | 1.4657 | 1364.460 | 1365.131 |
| metal:f32 | 8 | 16 | 16 | {} | 1.4627 | 5464.240 | 5474.225 |
| metal:f32 | 16 | 32 | 16 | {"queueFull": 16} | 1.4610 | 5473.189 | 5476.395 |
| metal:f16 | 1 | 2 | 2 | {} | 3.2607 | 306.003 | 307.344 |
| metal:f16 | 2 | 4 | 4 | {} | 3.2556 | 614.059 | 614.708 |
| metal:f16 | 8 | 16 | 16 | {} | 3.2516 | 2457.519 | 2463.049 |
| metal:f16 | 16 | 32 | 16 | {"queueFull": 16} | 3.2461 | 2463.243 | 2467.508 |

Percentiles use nearest-rank order statistics. With small n, p95/p99 often equal the maximum observation; these are descriptive samples, not production-tail estimates or SLOs. The report preserves every raw timing, error count and environment setting. Metal allocator snapshots include retained buffers. The direct phase also samples allocator usage every 100 ms; its reported peak can miss shorter spikes and excludes initial loading and managed-runtime reload. Metal private allocations can appear as wired system memory outside process RSS, so RSS alone is not a physical-memory comparison. OS process RSS/footprint high-water includes both direct and managed phases. Footprint is macOS accounting, not a sum of the RSS and GPU columns; unified CPU/GPU allocations must not be summed as independent physical-memory pools.

CPU F32 and Metal F32 are a same-precision comparison. Metal F16 uses F16 text backbone with an F32 classifier head/vision tower, residuals/convolution, norms/attention accumulation and recurrent state; compare its numerical qualification separately. CPU uses the default Candle pure-Rust backend, without Accelerate/MKL. Thread settings and the actual executable SHA-256 are recorded in metadata.json.
