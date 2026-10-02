# Flash Metal support and performance benchmark design

Date: 2026-10-02. Scope: pinned Flash only, CPU F32 and feature-gated macOS Metal F32/F16, text and existing optional still images, maximum 4,096 tokens.

Metal device creation is explicit, with no CPU inference fallback. All model tensors belong to the existing direct engine/device thread. Loading casts on bounded host storage before uploading final weights, retains the classifier head and vision tower in F32, and accounts for unified RAM and the Metal working-set limit. Unavailable devices/features/dtypes fail before inference. Backend errors fail the active request without replay and trigger bounded worker recovery. Shutdown remains terminal during recovery.

Qualification reuses exactly the frozen encoder/operator/head/full-model corpus. F32 retains maximum probability error <= 0.001 and mean <= 0.0001 against the CPU F32 oracle. The mixed F16 candidate is initially tested against those same strict limits; failure requires investigating operations/precision or independent matching-dtype reference evidence, not silently relaxing the F32 gate. Text, full 4,096-token context, PNG/JPEG, repeated decisions, authenticated HTTP equivalence, and shutdown are tested on real Metal hardware before advertisement.

Performance has three separate layers:

1. Criterion microbenchmarks for bounded request parsing and answer conversion, independent of model downloads.
2. A Rust large-weight runner with YAML workloads, exact tokenizer-counted lengths, all answer types, varied field/option count, normal-path warm latency samples, separate synchronized stage diagnostics, cold snapshot verification/load/first inference, planner/device memory snapshots, and bounded managed-runtime concurrency/overload measurements.
3. An external sequential CPU/Metal process harness that records hardware/toolchain/thread settings, process RSS high-water, raw per-request samples and machine-readable JSON, then renders a comparison report. CPU and GPU models never coexist during measurement.

Default workload matrix includes 256/1,024/4,096-token prompts, eight fields, 32 fields, and a large-option schema within context. Requested token count is an invariant: generation fails if a schema cannot fit or the tokenizer cannot produce that exact count. Defaults use ten samples for short cases and three for expensive long cases, one warmup per case, plus concurrency 1/2/8/16 at a fixed short request. Samples and concurrency are bounded/configurable in YAML. Every result reports sample count; p95/p99 for small samples are descriptive order statistics, not production-tail estimates. Overload reports successful/rejected/deadline counts separately and never treats errors as fast successful inference.

Normal timers include encoding, model execution, typed conversion and final GPU synchronization. Stage diagnostics add explicit barriers and must not be conflated with normal-path latency. Throughput is successful decisions divided by measured wall time; encoded input-token throughput is reported separately. There is no output-token generation. Raw samples permit independent aggregation. Measurements fix thread counts and run backend processes serially. Cold process load may use a warm operating-system file cache, which is documented; privileged cache flushing is not performed.

No latency SLO, 10,000-decision leak guarantee, or statistical p99 claim is inferred from this finite matrix. Reports include observed memory, error counts, numerical qualification, sampling limitations, command lines, software versions and hardware identity.
