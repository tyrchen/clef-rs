# Alternative Metal operator measurements

Measured 2026-10-03T05:26:15.434780+00:00 on Apple M5 Pro, macOS 26.5.2, MLX 0.32.3.

This is a synthetic operator experiment, not CLEF inference. Each candidate has 20 retained samples, ten excluded warmups and rotating execution order. Weight conversion/quantization and compilation are excluded; dynamic activation quantization is included for the W8A8 candidate. All invocations complete GPU work. [Raw samples and identity](operators.json) are authoritative. The [research record](../../research/clef-metal-latency-research.md) explains scope, candidate selection and integration gates.

## Projection means (ms)

| M / N / K | F16 | affine W4A16 | affine W8A16 | MXFP4 weights | MXFP8 weights | MXFP8 W8A8 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1024 / 12288 / 4096 | 3.467 | 3.784 | 3.925 | 3.768 | 4.061 | 40.835 |
| 4096 / 12288 / 4096 | 13.396 | 14.331 | 14.891 | 14.347 | 15.460 | 163.071 |
| 1024 / 4096 / 12288 | 3.977 | 4.041 | 4.361 | 3.955 | 4.271 | 44.876 |
| 4096 / 4096 / 12288 | 15.141 | 14.994 | 16.238 | 14.765 | 16.132 | 178.836 |
| 1024 / 8192 / 4096 | 2.423 | 2.573 | 2.696 | 2.596 | 2.754 | 27.290 |
| 4096 / 8192 / 4096 | 9.032 | 9.768 | 10.102 | 9.644 | 10.478 | 108.904 |
| 1024 / 4096 / 4096 | 1.344 | 1.408 | 1.469 | 1.393 | 1.507 | 13.785 |
| 4096 / 4096 / 4096 | 4.669 | 5.050 | 5.216 | 4.961 | 5.360 | 54.550 |

The four N/K shapes cover FFN expansion/contraction, combined QKV and hidden-width projections. Weights and activations are seeded synthetic normal distributions, not loaded Flash activations. Numerical tensor drift versus F16 is retained in JSON and does not qualify classifier probabilities. Affine quantization uses groups of 64; MX formats use groups of 32.

## Causal grouped-query attention

| Tokens | F32 ms | F16 ms | F16 output max absolute drift |
| --- | ---: | ---: | ---: |
| 1024 | 1.758 | 0.734 | 0.00358510 |
| 4096 | 20.822 | 6.698 | 0.00466430 |

Batch one, 16 query / four KV heads, width 256, causal mask, scale 1/16. This compares MLX candidates internally; it is not an A/B against the deployed Candle attention kernel.

## DeltaNet preparation

| Tokens | Eager ms | Compiled ms | Output max absolute drift |
| --- | ---: | ---: | ---: |
| 1024 | 1.667 | 1.552 | 0.00000000 |
| 4096 | 6.361 | 5.882 | 0.00000000 |

Preparation expands 16 q/k heads to 32, normalizes in F32, applies query scale, exponentiates decay and concatenates q/k/v/decay/beta. It omits convolution, projection and other mixer operations. Equality here is against MLX eager equations, not a full-model Candle oracle.

## Prepared F32 DeltaNet recurrence

| Tokens | Current shader mean ± stddev ms | MLX fused chunk mean ± stddev ms | Ratio | Output max absolute drift |
| --- | ---: | ---: | ---: | ---: |
| 1024 | 2.071 ± 0.029 | 0.850 ± 0.049 | 2.44× | 0.00017041 |
| 4096 | 7.654 ± 0.136 | 2.812 ± 0.036 | 2.72× | 0.00021671 |

Identical normalized F32 q/k, v, gamma and beta; zero initial state. The current shader body is read from the repository and dispatched through an MLX diagnostic wrapper with runtime dimensions. This does not measure Candle dispatch or the complete recurrence preparation. The current shader returns output only; the MLX candidate also computes final state. The M5 chunk implementation changes arithmetic and is not serving-qualified.

| Tokens | Zero-decay max absolute drift | 1e-12-decay max absolute drift |
| --- | ---: | ---: |
| 1024 | 0.00012469 | 0.00012469 |
| 4096 | 0.00013127 | 0.00013127 |

The isolated 4,096-token difference multiplied by 24 layers is 116.2 ms. This is a budgeting calculation, not measured whole-model latency; packing, synchronization, real activations and the execution wrapper differ from serving.

## Reproduction

```sh
make metal-candidate-env
make profile-metal-candidates METAL_CANDIDATE_RESULTS=/tmp/clef-candidates-reproduction.json
make profile-metal-backbone
```

The Makefile target writes raw JSON. This table records the checked-in campaign; compare a new run using its own JSON rather than silently relabeling these results. The original qualified normal-path reports remain unchanged. No real-model parity is claimed for these candidates.
