# CLEF Flash performance on Apple M5 Pro

The implementation replaces token-by-token Metal DeltaNet submission with a fused F32 recurrence and uses a bounded native F32 GQA attention tile for the backbone. Large text projections remain selectable F32 or mixed F16; M5 mixed F16 uses Metal 4 GPU neural accelerators. The CPU reference, pinned weights/encoder, vision tower and classifier semantics are preserved.

## Implementation

One 128-thread group contains four independent SIMD groups, each owning one value column. A lane retains up to four key states in registers and processes the bounded sequence on GPU. Precomputed decay avoids repeating exponentials across state columns; key-width specialization allows register allocation and loop unrolling. The exact reference sum tree protects F16 rounding boundaries. Each of the 24 linear layers issues one recurrence launch instead of a sequence-length-dependent tensor loop. No recurrent state persists across requests.

The eight full-attention layers use GQA directly, keeping four KV heads rather than copying them to 16. The pinned upstream SDPA shader is instantiated with 16-query/8-key tiles: 28,928 shared bytes fit the Apple GPU limit, whereas the default F32 width-256 tile needs 53,760 bytes and fails pipeline creation. Classifier and vision attention retain the qualified tensor graph. Pipeline creation happens during model loading, and inference adds no host readback or synchronization.

On Apple GPU family 10, large F16 backbone projections use a shared Metal 4 MPP pipeline with 64×64 tiles, F16 inputs and F32 accumulation. Relaxed precision is disabled. Complete tiles use static extents; partial tiles use checked tensor bounds. The F32 result is converted to half on GPU before the existing normalization/residual graph. Tiny decay/beta projections and older GPUs retain Candle’s Metal GEMM. Classifier and vision projections remain on their qualified F32 path.

Rust dispatch validates shapes, dtype, contiguous byte ranges and integer conversions before submitting work. The safe Candle wrappers own allocation, command lifetime and hazard tracking; this project adds no unsafe Rust. The existing owner-thread cancellation, deadline, recovery and shutdown behavior remains in effect.

## Reproduction

```sh
make bench-mbp CLEF_RELEASE_CACHE=/path/to/model-cache
make profile-metal CLEF_RELEASE_CACHE=/path/to/model-cache
make profile-metal-kernels
make verify-metal-neural # M5 only
make verify verify-metal verify-benchmark
make verify-metal-release verify-metal-media verify-metal-serving PYTHON=.venv-reference/bin/python
```

The [MBP YAML](../examples/clef.mbp-benchmark.yaml) retains the original six workloads and adds exact 139/192-token inputs. The minimum binary question with an empty state is 139 tokens under the pinned protocol; a 64-token synthetic decision would omit required protocol content. Models run sequentially with eight fixed Rayon/Candle CPU threads. Each profile has 79 normal-path samples, excluded warmups, eight separate stage diagnostics and bounded concurrency 1/2/8/16. The normal timer includes encoding, GPU completion and answer conversion, but excludes JSON parsing, HTTP/authentication and transport. Small-n percentiles are descriptive observations, not production tail guarantees.

The previous [CPU/Metal campaign](benchmarks/flash-cpu-metal/report.md) remains the baseline. CPU equations are unchanged and its 82-minute full-weight performance run is reused; CPU unit/operator regression checks still run. Traces and isolated operator measurements are diagnostic only. Raw Instruments traces stay outside the repository because they can include unrelated system process information. The existing conservative memory plan is retained rather than lowered from finite peak samples.

Implementation rationale and numerical pitfalls are recorded in [the research evidence](research/clef-metal-performance-research.md) and [design contract](../specs/clef-metal-performance-design.md).
