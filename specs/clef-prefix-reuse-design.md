# Exact Flash prefix reuse and Metal preparation fusion

## Contract

Preserve the released model, complete input, positions, decision-head evidence, projection precision, and probability gates (maximum 1e-3, mean 1e-4). No distillation, automatic input selection, quantization, relaxed matrix precision, or approximate recurrence is introduced. The existing CPU and Metal profiles remain supported.

Fusion is automatic for the Flash 128-key/128-value Metal mixer. Prefix retention is explicitly configured and disabled by default in the library. The MBP Metal YAML enables a 512 MiB, one-entry, five-minute cache. Images bypass prefix reuse; their qualified graph still benefits from preparation fusion.

## Exact continuation

The encoder records the state/schema boundary without modifying the token stream. A capture boundary is the largest multiple of 256 not exceeding that boundary, with a minimum of 512 tokens. Aligning to 256 preserves existing CPU attention, Metal attention and MPP projection tile boundaries. Matching is an exact comparison of token IDs, including the fixed system/state wrapper. Schema identity alone never enables reuse. The model snapshot, modality, precision and device are immutable properties of the owning engine.

For each of eight full-attention layers, retain the prefix K and V in their existing projection dtype. For each of 24 DeltaNet layers, retain the F32 recurrent state and the last three projected, pre-convolution rows. Retain the full F32 final prefix features: the joint head continues to receive the complete sequence, including previously computed state evidence. Absolute rotary positions of suffix tokens start at the captured prefix length.

A miss executes one original full prefill. The recurrent kernel snapshots registers at the capture token without altering arithmetic; attention and convolution states are copied from that same prefill. A hit projects only the suffix, prepends the convolution history, resumes recurrence, concatenates retained KV, and uses lower-right causal attention. The causal KV scan is capped at the actual last key tile even when the last query tile is partial. The portable CPU path implements the same continuation and masking.

Tensor views do not suffice for retention. CPU snapshots use `force_contiguous`. Metal snapshots use a checked custom copy into a fresh, exact-sized, Candle-tracked buffer. In the pinned Metal backend, `copy` shares storage; `contiguous` may retain a view; and `force_contiguous` may reuse a much larger scratch buffer through best-fit allocation. Retaining those oversized buffers caused a real F32 long-state OOM during qualification. A bounded zero-byte upload allocates exact storage, then a device-side strided copy fills it without GPU readback. Buffer lifetime, residency and reclamation stay in the existing Candle allocator. This cost applies only to capture, and physical buffer lengths are tested.

## Ownership and admission

A direct engine or one device actor exclusively owns a bounded LRU; no cross-thread mutable tensor cache is introduced. Keys include the bounded principal and exact prefix token IDs. Direct calls use the embedded scope; `decide_with_options` supplies explicit principal isolation. Managed requests pass their existing authenticated principal through execution. Worker reload drops all states.

Configuration bounds capacity to 4 GiB, entries to 1–16 and absolute lifetime to 1–3600 seconds. Expired entries are removed on the next cache-enabled text lookup. Short inputs and oversized captures bypass retention. Admission adds twice the configured capacity plus the existing 20% reserve to both host/device plans, covering old entries and a staged replacement before any model allocation. The CLI memory planner uses the same cache-aware calculation.

Only successful, synchronized, non-cancelled decisions publish captured states or count hits. Failed continuation never mutates retained tensors. Each capture is staged until head execution and device completion succeed. Debug output and aggregate statistics exclude principal/token/tensor contents. Explicit clear drops entries and resets counters.

## Preparation fusion

One checked 64-thread Metal operation replaces q/k extraction, duplicated-head materialization, square/reduce/add/sqrt/division/query scaling, decay exponentiation, and concatenation of recurrence inputs. Input layouts are the unchanged F32 convolution/SiLU result and F32 gates. The output is the existing recurrence's packed format, so both normal and resumed recurrence share it.

Retain the pinned 128-element reduction tree: per-lane `(0+64)`, then `(0+64)+(32+96)` and SIMD sum. Retain separate square-root rounding, division and query multiplication. Disable contraction/reassociation; do not substitute reciprocal square root. Shape, contiguous dtype, offsets and buffer bounds are checked before dispatch. Other geometries and CPU use the portable graph.

## Verification

Run CPU and Metal build/test/fmt/pedantic-Clippy gates, compact-state byte accounting and synthetic hybrid continuation, changed suffixes and 1/2/3-token convolution histories. Run real Metal bitwise preparation/recurrence capture-resume tests, GQA and partial query/key tile tests. Check cache scope, exact mismatch, LRU eviction, expiration, clear, configuration limits and failed publication.

Qualify full-weight text, 4096-token state, PNG and JPEG against pinned Python F32 probabilities in both Metal precisions. Verify unrounded full versus resumed distributions with changed schema, appended state, changed prefix, different principals and cancellation. Exercise configured reuse through the managed worker and authenticated serving lifecycle.

Measure unchanged full-prefill, miss/capture and hit separately. Rotate these modes across iterations, exclude hit preparation from timed hit samples, preserve raw measurements and compare **unrounded** probabilities for every mode. Keep warmup/model loading distinct. Report the additional retained memory, full/capture overhead and the shared-prefix requirement. Keep a matched previous executable and its hash for cold-path comparisons; never compare a cache-hit mean to a cold mean without labeling both.
