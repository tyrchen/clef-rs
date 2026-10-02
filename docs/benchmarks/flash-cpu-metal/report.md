# CPU / Metal Flash performance measurements

Measured 2026-10-02T15:17:57.622865+00:00.

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
| cpu:f32 | 5.993 | 19.435 | 15.124 | 35.949 | 29.786 | — | 40.875 | 3.554 |
| metal:f32 | 6.512 | 25.082 | 11.694 | 28.754 | 27.814 | 44.276 | 49.016 | 0.097 |
| metal:f16 | 6.530 | 17.640 | 4.513 | 22.663 | 25.651 | 26.661 | 27.027 | 0.094 |

| Profile | Workload | Tokens | Fields | Options/field | n | Mean ms | p50 ms | p95 ms | p99 ms | req/s | input tokens/s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| cpu:f32 | short-256 | 256 | 1 | 2 | 10 | 13271.713 | 13316.746 | 13428.674 | 13428.674 | 0.0753 | 19.29 |
| cpu:f32 | medium-1024 | 1024 | 1 | 2 | 10 | 51347.788 | 51348.169 | 51523.971 | 51523.971 | 0.0195 | 19.94 |
| cpu:f32 | long-4096 | 4096 | 1 | 2 | 3 | 212415.691 | 212344.209 | 212859.296 | 212859.296 | 0.0047 | 19.28 |
| cpu:f32 | mixed-8-fields | 1024 | 8 | 4 | 10 | 51335.671 | 51290.112 | 51771.671 | 51771.671 | 0.0195 | 19.95 |
| cpu:f32 | wide-32-fields | 3072 | 32 | 2 | 3 | 156584.235 | 156675.240 | 156807.951 | 156807.951 | 0.0064 | 19.62 |
| cpu:f32 | large-option-schema | 4096 | 4 | 64 | 3 | 213301.310 | 213107.176 | 214760.678 | 214760.678 | 0.0047 | 19.20 |
| metal:f32 | short-256 | 256 | 1 | 2 | 10 | 4434.517 | 4434.421 | 4435.828 | 4435.828 | 0.2255 | 57.73 |
| metal:f32 | medium-1024 | 1024 | 1 | 2 | 10 | 17826.264 | 17823.970 | 17834.256 | 17834.256 | 0.0561 | 57.44 |
| metal:f32 | long-4096 | 4096 | 1 | 2 | 3 | 72481.962 | 72458.523 | 72534.222 | 72534.222 | 0.0138 | 56.51 |
| metal:f32 | mixed-8-fields | 1024 | 8 | 4 | 10 | 17805.576 | 17804.574 | 17814.355 | 17814.355 | 0.0562 | 57.51 |
| metal:f32 | wide-32-fields | 3072 | 32 | 2 | 3 | 54143.759 | 54144.962 | 54150.361 | 54150.361 | 0.0185 | 56.74 |
| metal:f32 | large-option-schema | 4096 | 4 | 64 | 3 | 72484.770 | 72488.908 | 72490.159 | 72490.159 | 0.0138 | 56.51 |
| metal:f16 | short-256 | 256 | 1 | 2 | 10 | 4380.099 | 4380.024 | 4381.709 | 4381.709 | 0.2283 | 58.45 |
| metal:f16 | medium-1024 | 1024 | 1 | 2 | 10 | 17560.557 | 17560.440 | 17562.891 | 17562.891 | 0.0569 | 58.31 |
| metal:f16 | long-4096 | 4096 | 1 | 2 | 3 | 71434.042 | 71433.480 | 71436.654 | 71436.654 | 0.0140 | 57.34 |
| metal:f16 | mixed-8-fields | 1024 | 8 | 4 | 10 | 17700.851 | 17582.890 | 18066.408 | 18066.408 | 0.0565 | 57.85 |
| metal:f16 | wide-32-fields | 3072 | 32 | 2 | 3 | 54052.874 | 54054.588 | 54058.374 | 54058.374 | 0.0185 | 56.83 |
| metal:f16 | large-option-schema | 4096 | 4 | 64 | 3 | 72355.850 | 72360.039 | 72368.089 | 72368.089 | 0.0138 | 56.61 |

| Workload | Comparison | CPU mean / Metal mean |
| --- | --- | ---: |
| short-256 | cpu:f32 / metal:f32 | 2.99× |
| medium-1024 | cpu:f32 / metal:f32 | 2.88× |
| long-4096 | cpu:f32 / metal:f32 | 2.93× |
| mixed-8-fields | cpu:f32 / metal:f32 | 2.88× |
| wide-32-fields | cpu:f32 / metal:f32 | 2.89× |
| large-option-schema | cpu:f32 / metal:f32 | 2.94× |
| short-256 | cpu:f32 / metal:f16 | 3.03× |
| medium-1024 | cpu:f32 / metal:f16 | 2.92× |
| long-4096 | cpu:f32 / metal:f16 | 2.97× |
| mixed-8-fields | cpu:f32 / metal:f16 | 2.90× |
| wide-32-fields | cpu:f32 / metal:f16 | 2.90× |
| large-option-schema | cpu:f32 / metal:f16 | 2.95× |

Diagnostic stages add explicit barriers and are measured separately from the latency samples above.

| Profile | Workload | Encoding ms | Backbone/vision ms | Head/readback ms | Conversion ms | Final sync ms | Diagnostic total ms | Metal allocated GiB |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| cpu:f32 | short-256 | 0.265 | 13400.931 | 31.563 | 0.007 | 0.000 | 13432.767 | — |
| cpu:f32 | medium-1024 | 0.683 | 51249.951 | 78.670 | 0.008 | 0.000 | 51329.313 | — |
| cpu:f32 | long-4096 | 2.567 | 211911.791 | 287.584 | 0.008 | 0.000 | 212201.951 | — |
| cpu:f32 | mixed-8-fields | 0.841 | 51130.112 | 107.159 | 0.021 | 0.000 | 51238.134 | — |
| cpu:f32 | wide-32-fields | 2.109 | 155920.542 | 369.151 | 0.026 | 0.000 | 156291.829 | — |
| cpu:f32 | large-option-schema | 2.941 | 213925.692 | 561.437 | 0.067 | 0.000 | 214490.138 | — |
| metal:f32 | short-256 | 0.262 | 4430.378 | 8.944 | 0.006 | 0.003 | 4439.593 | 33.810 |
| metal:f32 | medium-1024 | 0.651 | 17813.425 | 21.483 | 0.006 | 0.003 | 17835.569 | 33.810 |
| metal:f32 | long-4096 | 2.363 | 72386.757 | 69.939 | 0.007 | 0.004 | 72459.070 | 33.810 |
| metal:f32 | mixed-8-fields | 0.748 | 17774.113 | 32.244 | 0.015 | 0.003 | 17807.124 | 33.810 |
| metal:f32 | wide-32-fields | 2.071 | 54049.302 | 95.020 | 0.022 | 0.004 | 54146.420 | 33.810 |
| metal:f32 | large-option-schema | 2.927 | 72388.881 | 100.918 | 0.061 | 0.836 | 72493.623 | 33.810 |
| metal:f16 | short-256 | 0.328 | 4372.885 | 8.157 | 0.006 | 0.003 | 4381.379 | 17.132 |
| metal:f16 | medium-1024 | 0.646 | 17538.557 | 20.763 | 0.007 | 0.003 | 17559.976 | 17.132 |
| metal:f16 | long-4096 | 2.245 | 71366.106 | 69.172 | 0.008 | 0.004 | 71437.535 | 17.132 |
| metal:f16 | mixed-8-fields | 0.723 | 17756.009 | 32.302 | 0.015 | 0.004 | 17789.053 | 17.132 |
| metal:f16 | wide-32-fields | 1.986 | 53948.566 | 95.985 | 0.023 | 0.006 | 54046.566 | 17.132 |
| metal:f16 | large-option-schema | 2.981 | 72270.317 | 102.044 | 0.065 | 0.743 | 72376.151 | 17.132 |

Managed-runtime concurrency uses the same fixed short workload, FIFO within a principal and round-robin across principals. The harness has one device worker, eight ingress/queue slots, eight preparation workers, and 300-second queue/request/shutdown limits; these differ from production defaults. Admission errors are counted separately.

| Profile | Concurrency | Attempts | Successes | Errors | Successful req/s | Successful p50 ms | Successful p95 ms |
| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| cpu:f32 | 1 | 2 | 2 | {} | 0.0734 | 13613.801 | 13618.368 |
| cpu:f32 | 2 | 4 | 4 | {} | 0.0729 | 27381.352 | 27511.900 |
| cpu:f32 | 8 | 16 | 16 | {} | 0.0732 | 108743.406 | 109967.239 |
| cpu:f32 | 16 | 32 | 16 | {"queueFull": 16} | 0.0727 | 109764.987 | 110534.032 |
| metal:f32 | 1 | 2 | 2 | {} | 0.2261 | 4422.161 | 4424.565 |
| metal:f32 | 2 | 4 | 4 | {} | 0.2262 | 8842.565 | 8844.842 |
| metal:f32 | 8 | 16 | 16 | {} | 0.2261 | 35383.006 | 35390.546 |
| metal:f32 | 16 | 32 | 16 | {"queueFull": 16} | 0.2261 | 35388.312 | 35392.006 |
| metal:f16 | 1 | 2 | 2 | {} | 0.2258 | 4427.492 | 4428.549 |
| metal:f16 | 2 | 4 | 4 | {} | 0.2258 | 8857.954 | 8858.822 |
| metal:f16 | 8 | 16 | 16 | {} | 0.2258 | 35417.363 | 35433.484 |
| metal:f16 | 16 | 32 | 16 | {"queueFull": 16} | 0.2259 | 35410.382 | 35425.906 |

Percentiles use nearest-rank order statistics. With small n, p95/p99 often equal the maximum observation; these are descriptive samples, not production-tail estimates or SLOs. The report preserves every raw timing, error count and environment setting. Metal allocator snapshots include retained buffers. The direct phase also samples allocator usage every 100 ms; its reported peak can miss shorter spikes and excludes initial loading and managed-runtime reload. Metal private allocations can appear as wired system memory outside process RSS, so RSS alone is not a physical-memory comparison. OS process RSS/footprint high-water includes both direct and managed phases. Footprint is macOS accounting, not a sum of the RSS and GPU columns; unified CPU/GPU allocations must not be summed as independent physical-memory pools.

CPU F32 and Metal F32 are a same-precision comparison. Metal F16 uses F16 text backbone with an F32 classifier head/vision tower, residuals/convolution, norms/attention accumulation and recurrent state; compare its numerical qualification separately. CPU uses the default Candle pure-Rust backend, without Accelerate/MKL. Thread settings and the actual executable SHA-256 are recorded in metadata.json.

See [the domain microbenchmarks](domain.md) for parsing/rendering estimates and [numerical qualification](../../clef-flash-verification.md) for the precision gates.

The measured executable is from commit `0900a46`, identified by its recorded SHA-256. Follow-up changes share bounded YAML validation between server and benchmark, add API examples, and correct the admission memory estimate; these do not change the measured tensor execution graph.

The recorded F16 device plan (23.996 GiB) underestimated its observed sampled peak (26.661 GiB). The final planner budgets Metal temporary tensors at the larger FFN width, retaining the 20% reserve: approximately 28.796 GiB device / 39.944 GiB unified host for F16, and 49.828 GiB device / 60.977 GiB unified host for F32 at 4,096 text tokens. Regression tests bind the reviewed model metadata and cover both measured peaks, arithmetic overflow and the qualified runner budgets. The raw JSON keeps the original measured executable’s estimates.
