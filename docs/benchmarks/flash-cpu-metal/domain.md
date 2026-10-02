# Domain microbenchmarks

Criterion 0.8.2, release build, 50 samples, one-second warmup and three-second measurement per case. These single-thread CPU measurements do not load model weights. Confidence intervals are Criterion bootstrap estimates, not inference latency percentiles.

| Operation | Mean µs | 95% CI lower µs | 95% CI upper µs |
| --- | ---: | ---: | ---: |
| request_parse/256_bytes | 1.748 | 1.743 | 1.753 |
| request_parse/4096_bytes | 4.493 | 4.477 | 4.510 |
| request_parse/65536_bytes | 47.806 | 47.562 | 48.107 |
| bounded_json | 0.920 | 0.919 | 0.922 |
| reference_json_render | 0.458 | 0.457 | 0.459 |

The `request_parse/*_bytes` labels describe state UTF-8 byte length, not the entire JSON envelope. Actual request sizes are 408, 4,248, 65,688 bytes; byte throughput uses those full sizes.

Raw iteration counts, observed nanoseconds, estimates and workload byte throughput are preserved in [domain.json](domain.json). Hardware for the full measurement campaign is recorded in [metadata.json](metadata.json).
