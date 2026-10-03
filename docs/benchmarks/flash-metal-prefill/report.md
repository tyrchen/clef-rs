# CPU / Metal Flash performance measurements

Measured 2026-10-03T03:21:42.177039+00:00.

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
| metal:f32 | 5.938 | 22.673 | 0.640 | 24.751 | 29.558 | 35.920 | 40.336 | 0.104 |
| metal:f16 | 6.356 | 17.446 | 0.236 | 18.419 | 25.569 | 18.527 | 26.773 | 0.226 |

| Profile | Workload | Tokens | Fields | Options/field | n | Mean ms | p50 ms | p95 ms | p99 ms | req/s | input tokens/s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| metal:f32 | short-256 | 256 | 1 | 2 | 10 | 610.332 | 610.283 | 611.278 | 611.278 | 1.6384 | 419.43 |
| metal:f32 | medium-1024 | 1024 | 1 | 2 | 10 | 2446.849 | 2447.093 | 2456.946 | 2456.946 | 0.4087 | 418.49 |
| metal:f32 | long-4096 | 4096 | 1 | 2 | 10 | 10105.794 | 10104.784 | 10116.402 | 10116.402 | 0.0990 | 405.31 |
| metal:f32 | mixed-8-fields | 1024 | 8 | 4 | 10 | 2460.319 | 2459.828 | 2467.605 | 2467.605 | 0.4064 | 416.20 |
| metal:f32 | wide-32-fields | 3072 | 32 | 2 | 10 | 7559.214 | 7558.238 | 7572.642 | 7572.642 | 0.1323 | 406.39 |
| metal:f32 | large-option-schema | 4096 | 4 | 64 | 10 | 10135.600 | 10131.469 | 10149.809 | 10149.809 | 0.0987 | 404.12 |
| metal:f32 | minimal-139 | 139 | 1 | 2 | 20 | 441.842 | 441.854 | 442.487 | 442.825 | 2.2632 | 314.58 |
| metal:f32 | compact-192 | 192 | 1 | 2 | 20 | 484.187 | 484.083 | 485.231 | 485.403 | 2.0652 | 396.53 |
| metal:f16 | short-256 | 256 | 1 | 2 | 10 | 222.366 | 222.655 | 223.670 | 223.670 | 4.4966 | 1151.13 |
| metal:f16 | medium-1024 | 1024 | 1 | 2 | 10 | 820.664 | 820.877 | 821.901 | 821.901 | 1.2185 | 1247.75 |
| metal:f16 | long-4096 | 4096 | 1 | 2 | 10 | 3718.966 | 3713.226 | 3742.234 | 3742.234 | 0.2689 | 1101.37 |
| metal:f16 | mixed-8-fields | 1024 | 8 | 4 | 10 | 835.621 | 835.872 | 837.815 | 837.815 | 1.1967 | 1225.41 |
| metal:f16 | wide-32-fields | 3072 | 32 | 2 | 10 | 2726.028 | 2726.512 | 2733.942 | 2733.942 | 0.3668 | 1126.90 |
| metal:f16 | large-option-schema | 4096 | 4 | 64 | 10 | 3755.468 | 3749.110 | 3771.963 | 3771.963 | 0.2663 | 1090.67 |
| metal:f16 | minimal-139 | 139 | 1 | 2 | 20 | 163.715 | 163.783 | 164.362 | 164.696 | 6.1076 | 848.95 |
| metal:f16 | compact-192 | 192 | 1 | 2 | 20 | 188.491 | 188.430 | 189.573 | 189.998 | 5.3048 | 1018.53 |

Diagnostic stages add explicit barriers and are measured separately from the latency samples above.

| Profile | Workload | Encoding ms | Backbone/vision ms | Head/readback ms | Conversion ms | Final sync ms | Diagnostic total ms | Metal allocated GiB |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| metal:f32 | short-256 | 0.204 | 602.249 | 8.042 | 0.004 | 0.002 | 610.501 | 33.810 |
| metal:f32 | medium-1024 | 0.627 | 2433.099 | 22.510 | 0.004 | 0.002 | 2456.243 | 33.810 |
| metal:f32 | long-4096 | 2.151 | 10029.866 | 75.950 | 0.005 | 0.003 | 10107.975 | 33.810 |
| metal:f32 | mixed-8-fields | 0.744 | 2426.102 | 34.234 | 0.014 | 0.003 | 2461.097 | 33.810 |
| metal:f32 | wide-32-fields | 1.898 | 7461.076 | 102.710 | 0.023 | 0.003 | 7565.710 | 33.810 |
| metal:f32 | large-option-schema | 2.757 | 10040.123 | 108.157 | 0.062 | 0.003 | 10151.103 | 33.810 |
| metal:f32 | minimal-139 | 0.145 | 436.236 | 6.715 | 0.004 | 0.002 | 443.103 | 33.810 |
| metal:f32 | compact-192 | 0.172 | 477.370 | 7.381 | 0.004 | 0.002 | 484.930 | 33.810 |
| metal:f16 | short-256 | 0.196 | 214.952 | 8.192 | 0.004 | 0.002 | 223.346 | 17.135 |
| metal:f16 | medium-1024 | 0.621 | 800.918 | 20.662 | 0.004 | 0.002 | 822.208 | 17.135 |
| metal:f16 | long-4096 | 2.197 | 3652.695 | 74.662 | 0.006 | 0.003 | 3729.564 | 17.135 |
| metal:f16 | mixed-8-fields | 0.683 | 805.224 | 33.827 | 0.012 | 0.003 | 839.750 | 17.135 |
| metal:f16 | wide-32-fields | 1.887 | 2635.450 | 100.589 | 0.023 | 0.003 | 2737.954 | 17.135 |
| metal:f16 | large-option-schema | 2.619 | 3657.011 | 105.422 | 0.059 | 0.003 | 3765.115 | 17.135 |
| metal:f16 | minimal-139 | 0.135 | 156.644 | 6.572 | 0.004 | 0.002 | 163.357 | 17.136 |
| metal:f16 | compact-192 | 0.166 | 181.899 | 7.241 | 0.004 | 0.002 | 189.313 | 17.136 |

Managed-runtime concurrency uses the same fixed short workload, FIFO within a principal and round-robin across principals. The harness has one device worker, eight ingress/queue slots, eight preparation workers, and 300-second queue/request/shutdown limits; these differ from production defaults. Admission errors are counted separately.

| Profile | Concurrency | Attempts | Successes | Errors | Successful req/s | Successful p50 ms | Successful p95 ms |
| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| metal:f32 | 1 | 2 | 2 | {} | 1.6346 | 609.004 | 614.351 |
| metal:f32 | 2 | 4 | 4 | {} | 1.6389 | 1219.600 | 1220.958 |
| metal:f32 | 8 | 16 | 16 | {} | 1.6368 | 4886.047 | 4889.937 |
| metal:f32 | 16 | 32 | 16 | {"queueFull": 16} | 1.6356 | 4889.560 | 4892.637 |
| metal:f16 | 1 | 2 | 2 | {} | 4.4691 | 222.599 | 224.894 |
| metal:f16 | 2 | 4 | 4 | {} | 4.4968 | 444.459 | 445.067 |
| metal:f16 | 8 | 16 | 16 | {} | 4.4890 | 1780.428 | 1785.532 |
| metal:f16 | 16 | 32 | 16 | {"queueFull": 16} | 4.4724 | 1787.379 | 1791.555 |

Percentiles use nearest-rank order statistics. With small n, p95/p99 often equal the maximum observation; these are descriptive samples, not production-tail estimates or SLOs. The report preserves every raw timing, error count and environment setting. Metal allocator snapshots include retained buffers. The direct phase also samples allocator usage every 100 ms; its reported peak can miss shorter spikes and excludes initial loading and managed-runtime reload. Metal private allocations can appear as wired system memory outside process RSS, so RSS alone is not a physical-memory comparison. OS process RSS/footprint high-water includes both direct and managed phases. Footprint is macOS accounting, not a sum of the RSS and GPU columns; unified CPU/GPU allocations must not be summed as independent physical-memory pools.

CPU F32 and Metal F32 are a same-precision comparison. Metal F16 uses F16 text backbone with an F32 classifier head/vision tower, residuals/convolution, norms/attention accumulation and recurrent state; compare its numerical qualification separately. CPU uses the default Candle pure-Rust backend, without Accelerate/MKL. Thread settings and the actual executable SHA-256 are recorded in metadata.json.

The measured implementation is commit 874255d32e759a8a72fbf931e9366a97799beb50 with a clean source tree at launch. Independent checks verified the executable/workload SHA-256, unchanged measured source, 100 normal samples per precision, exact encoded lengths, statistics, 54/38/16 managed admissions and device/host capacity bounds.

The preceding flash-metal-tuned campaign is preserved. Actual token counts, field/option schemas, warmups, thread settings and concurrency construction are unchanged; long-4096, wide-32-fields and large-option-schema increase from n=3 to n=10. The current matrix totals 100 direct samples per precision instead of 79.

CPU tensor equations are unchanged. The existing 82-minute CPU full-weight performance campaign is reused, with current CPU build/parity/regression gates repeated; no new CPU timing is implied by this Metal-only campaign.

See [numerical qualification](qualification.json), [operator and rejected-candidate diagnostics](operator-diagnostics.json), and [the campaign comparison](../../clef-flash-metal-performance.md). Chunkwise DeltaNet is a test-only candidate and is absent from the measured benchmark and serving executables.
