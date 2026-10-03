# Flash: larger latency reductions on the M5 Pro

Checked 2026-10-02 on the user's M5 Pro, 20 GPU cores, 64 GiB unified memory,
macOS 26.5.2 and SDK 26.5. Read the [previous performance research](clef-metal-performance-research.md)
before this investigation. The deployed model, precision profiles, encoding and
probability gates are unchanged. This record investigates alternatives; it does
not announce a faster serving implementation.

This is the pre-implementation research record. The implemented lossless paths and subsequent measurements are described in [exact-prefix reuse](../clef-flash-prefix-reuse.md).

## Decision

The next large reduction needs to remove substantial work or change how a whole
backbone block executes. Another recurrence or attention kernel alone is unlikely
to halve the current 4,096-token latency. Exact reuse of a long state prefix is
the strongest candidate when the application has shared state. For independent
long states, input selection or a smaller trained decision model offers a larger
structural reduction, with a separate task-quality evaluation. Native low-bit
compute remains worth testing, but the locally measured MLX quantization paths
do not establish that benefit.

There is no measured 2× speedup for a first, unique 4,096-token request with the
same Flash weights and unchanged numerical gates. Candidate integration must
earn that claim with full-model results.

## Evidence and cost

The [qualified normal-path matrix](../benchmarks/flash-metal-prefill/report.md)
measures mixed F16 at 820.664 ms for 1,024 tokens and 3,718.966 ms for 4,096,
with ten samples each. Encoding takes about 0.6/2.2 ms and the F32 head about
20.7/74.7 ms in separate stage diagnostics. Those are not the main latency cost.

The [new backbone capture](../benchmarks/flash-metal-candidates/backbone.json)
uses the committed implementation, synthetic token ID 42, one excluded warmup
and one synchronized capture per length. At 4,096 tokens it records 2,072 ms
in projections, 722 ms in recurrence including its preparation, 316 ms in
normalization, 405 ms in SDPA and 758 ms of otherwise uncategorized DeltaNet
work. Barriers change scheduling, categories omit some work, and parent
inclusive values overlap their children. These numbers identify investigation
targets; they are not percentages of the 3,719 ms normal latency.

Counting the released two-dimensional layer projection weights in the reviewed
tensor catalog gives 6,918,504,448 densely multiplied parameters per token:
4,831,838,208 FFN, 1,616,904,192 DeltaNet and 469,762,048 full-attention weights.
Using two FLOPs per multiply-add gives 14.169 trillion FLOPs at 1,024 tokens
and 56.676 trillion at 4,096, before attention, recurrence, norms and the head.
Embedding lookup and the vocabulary output matrix are excluded: this classifier
does not run vocabulary decoding. Reaching 100 ms from the current 4,096-token
measurement requires a 37.2× reduction in elapsed time, much larger than a
local epilogue improvement. This calculation is a work count, not a hardware
peak or an assertion that every hardware execution has identical cost.

## New local experiments

[Raw operator results](../benchmarks/flash-metal-candidates/operators.json)
contain 20 samples per candidate, ten excluded warmups, rotated candidate order,
seed 42, device/OS identity and script/shader hashes. All arrays are synthetic.
Weight quantization is outside timing. The activation-quantized FP8 candidate
includes activation quantization inside timing. Every invocation creates and
completes GPU work; evaluated outputs are discarded between samples.

The [operator report](../benchmarks/flash-metal-candidates/report.md) provides
the exact measured tables. Important findings:

- Across four representative projection geometries and two lengths, affine
  4-bit/8-bit and weight-only MXFP8 do not yield a large latency improvement
  over MLX F16. Their smaller weight buffers are a memory benefit, not evidence
  of a faster complete decision. Quantization tensor errors here are not
  classifier probability errors.
- The separate activation-quantized MXFP8 `qqmm` path is substantially slower
  in this environment. The [pinned implementation](https://github.com/ml-explore/mlx/blob/v0.32.3/mlx/backend/metal/quantized.cpp)
  quantizes/dequantizes activations and dispatches a different path. This
  experiment does not benchmark a hypothetical native FP8 tensor pipeline on
  macOS 27.
- MLX's fused chunkwise DeltaNet is roughly 3× faster than the current recurrence
  shader on identical prepared F32 arrays. The comparison runs our shader body
  through a research MLX wrapper with runtime dimensions, not through Candle's
  production dispatch. It excludes q/k normalization and packing; MLX also
  returns final state. Multiplying the per-layer difference by 24 gives an
  approximately 116 ms investigation budget at 4,096 tokens, **not** a measured
  full-model saving or a 3× model speedup.
- This fast DeltaNet is not an exact replacement. The [upstream M5 kernel](https://github.com/ml-explore/mlx/blob/v0.32.3/mlx/backend/metal/kernels/gated_delta_update_nax.h)
  uses relaxed-precision matrix operations and clamps decay before its logarithm.
  Our synthetic output comparison has nonzero drift, and separate zero/tiny-decay
  probes are retained. It has not passed Flash's text/context/image/JPEG gates.
- F16 attention is substantially faster than F32 attention within MLX, with
  nonzero output drift. An optimized F32 MPP attention implementation also
  deserves comparison with our current 16-query/8-key shader. Cross-framework
  operator timings do not qualify that replacement.
- Simply compiling the q/k-normalization-and-packing graph gives a modest
  improvement. Eliminating packing and repeated key heads entirely requires
  changing the mixer data flow, rather than just enabling graph compilation.

[MLX 0.32.3](https://github.com/ml-explore/mlx/releases/tag/v0.32.3) is pinned in
an optional research environment, separate from Rust/serving dependencies.
The [quantization API](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.quantize.html)
defines different affine and microscaling formats; their names do not imply
equal accuracy or native execution. A complete backend migration is not justified
by these operator results alone.

## Ranked alternatives

| Path | Applicability | Potential scale and evidence | Required change or qualification |
| --- | --- | --- | --- |
| Long-prefix state reuse | Shared long state; changed schema or appended suffix | Can avoid most 32-layer prefix work on a hit; several-fold improvement is plausible, unmeasured | Retain full-attention KV, DeltaNet state, convolution history and head evidence; bounded owner-thread cache; hit/miss parity |
| Complete-result cache | Exactly repeated decisions | Avoids model inference on a hit; no cold-miss improvement | Key complete semantic request, images, model/profile and principal; fresh response metadata; explicit bounded cache policy |
| Application state projection / retrieval | Large independent state containing irrelevant material | The existing 256-token workload is 16.7× faster than the 4,096-token workload; this is a different input workload, not a measured compression speedup | Include selection cost and verify decision quality on representative difficult cases; preserve the complete-state default |
| Distilled/pruned decision model; optional fast-path cascade | Repeated domain/tasks and available training/evaluation data | Removes actual layer/width/FFN work; potentially much larger than dtype storage changes; no local model exists | New trained artifact and explicit task-quality contract; calibrated fallback coverage; original Flash still available |
| Whole-mixer fusion plus MPP attention | First unique requests using existing Flash | Broadens coverage beyond one kernel; isolated experiments identify opportunities, but no 2× result is established | Fuse conv/SiLU/qk normalization/gating with direct strided recurrence inputs and output norm/gate; preserve rounding or pass full parity |
| Native block-scaled low-bit tensor compute | First unique requests; supported OS/toolchain | Could change matrix arithmetic, unlike shrinking storage alone; local weight-only quantization does not prove it | New quantization/calibration/packing implementation, newer APIs where required, end-to-end latency and quality gates |
| Core ML/ANE or CPU+GPU partition | Exploratory backend alternative | No CLEF-local speedup evidence; moving a subset of work cannot remove the remaining serial backbone | Export/partition hybrid recurrent graph, count crossings and device fallback, compare the complete decision |

The [MPP prefill study](https://arxiv.org/html/2607.19438v1) supports investigating
matrix accelerators and attention as a combined execution path. Its models,
quantization, runtime and comparison baselines differ from this already MPP-enabled
Flash engine, so its published speedup ratios are not transferable estimates.
[Apple's newer tensor guidance](https://developer.apple.com/videos/play/wwdc2026/330/)
introduces block-scale planes and MX formats with macOS 27. That is a distinct
development/test route from the current SDK 26.5; updating the OS by itself is
not an inference optimization. No OS/toolchain upgrade was performed.

## What exact prefix reuse entails

The encoder orders the input as system instruction, images if present, **state**,
schema and assistant suffix. Therefore, a shared schema with different states
usually shares only the short system prefix. Moving schema before state would
change the trained/reference encoding and is not an exact caching optimization.

For a shared state prefix, both mixers are causal. A resumable backbone needs
eight layers of F32 K/V, 24 layers of F32 recurrent state, three projected tokens
of convolution history per linear layer, the absolute/multimodal positions, and
the prefix's final hidden evidence. The joint head must still see the complete
sequence. Caching only the last hidden vector or only attention KV is insufficient.

For a 3,840-token text prefix, base tensor storage is approximately:

| Cached item | Size |
| --- | ---: |
| Eight full-attention K/V pairs, four KV heads, width 256, F32 | 240 MiB |
| 24 DeltaNet states, 32 heads, 128×128, F32 | 48 MiB |
| Three convolution input tokens per DeltaNet layer | 2.25 MiB |
| Final prefix evidence, width 4,096, F32 | 60 MiB |
| Total before allocator overhead / optional head caches | 350.25 MiB |

A 4,096-token request with a matching 3,840-token prefix still executes its
256-token suffix, attends to retained prefix KV and runs the head over full
evidence. A **research target** of 0.3–0.7 seconds on such hits is plausible from
the existing short/long stage observations; it is not an implemented or measured
latency. Cold misses retain full-prefill cost. At a 0.4-second hypothetical hit
and current 3.719-second miss, 90% hit rate gives 0.732-second arithmetic mean,
whereas 50% gives 2.059 seconds. Hit probability is as important as hit speed;
neither formula supplies a p95 estimate.

Use exact token/position identity, verified model/profile identity and principal
isolation. Image identity and preprocessing must be included when applicable.
The GPU owner actor owns cached tensors, capacity/eviction and shutdown. Bound
both bytes and entries, account them in admission, and do not share mutable
recurrent state between requests. Compare suffix results against uncached full
prefill across partial prefix boundaries and changed schemas. Different matmul
and attention tiling can change floating-point rounding even when the mathematics
is equivalent; the existing 1e-3 maximum / 1e-4 mean gates still apply.

## First-unique-request route

For a first unique request, combine larger mixer/attention changes in one
measured execution experiment. Keep conv and normalized q/k in their original
head layout, read v and gates directly, avoid the expanded packed tensor, and
fuse output normalization/SiLU/gating where rounding can be preserved. Compare
a strict F32 chunk implementation and MPP attention before considering relaxed
arithmetic. A 30% improvement is an experiment acceptance target, not a forecast.
Measure normal 1,024/4,096-token decisions with pinned real weights, then run all
existing probability oracles; fail the candidate if accuracy or whole-model
latency does not improve.

If the target remains below one second for every independent 4,096-token state,
the stronger structural routes are reduced input work, a smaller trained model,
or independently demonstrated native low-bit compute. A cascade needs explicit
coverage evidence: its mean cost is `T_fast + (1 - coverage) * T_Flash`, including
fallback work. Confidence alone does not guarantee that its accepted decisions
match Flash. No model substitution, token dropping, prompt reordering or
probability-gate relaxation was enabled during this research.

## Reproduction and verification

```sh
make metal-candidate-env
make profile-metal-candidates
make profile-metal-backbone
```

The candidate environment is ignored by Git. `METAL_CANDIDATE_PYTHON` and
`METAL_CANDIDATE_RESULTS` select an already installed research interpreter and a
separate output file. Run GPU experiments sequentially. The recorded operator
runner completed with finite outputs and 1,200 retained timing samples; zero and
tiny decay are numerical probes, not extra latency samples. Validate sample
statistics and script/shader identity before reusing these results.

The Rust serving graph, shader files and Cargo manifests/lockfile are unchanged.
The full Rust/release-oracle gates were not rerun for this Python research runner
and documentation change. The existing full-weight backbone diagnostic was
executed, Python syntax and recorded-data integrity were checked, and Cargo
audit/deny passed with the repository's existing paste advisory exception. This
verification qualifies the research artifacts, not any candidate serving backend.

## Follow-up: retained buffer allocation

Implementation found that the pinned Candle Metal allocator uses best-fit scratch reuse: [`new_buffer`, `find_available_buffer`, and `new_buffer_with_data`](https://github.com/huggingface/candle/blob/31f35b147389700ed2a178ee66a91c3cc25cc80d/candle-core/src/metal_backend/device.rs). A logical small snapshot can therefore own a much larger projection buffer after `force_contiguous`. The F32 changed-schema/principal qualification reproduced GPU OOM with this approach. Exact-sized tracked snapshot storage removed the failure; full-F32 continuation then completed with zero unrounded probability drift in all seven comparisons. Allocation bounds include staging, and a real Metal test verifies exact physical buffer lengths for strided F32/F16 snapshots.

Keep residency management in Candle rather than introducing unmanaged native buffers. Apple's [residency-set contract](https://developer.apple.com/documentation/metal/mtlresidencyset) requires registering and removing allocations over their GPU lifetime. The implementation uses the existing tracked upload/copy API with checked layouts and no unsafe Rust.
