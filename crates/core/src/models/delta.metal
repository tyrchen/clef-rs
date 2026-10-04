#include <metal_stdlib>
using namespace metal;
// Deliberately *not* a performance choice: disabling FP contraction and
// reassociation keeps every multiply-add rounded exactly like the portable
// CPU reference, so Metal and portable results stay bit-identical. Removing
// these pragmas would let the compiler fuse FMAs and reorder reductions,
// silently changing numerics.
#pragma clang fp contract(off)
#pragma clang fp reassociate(off)
constant ulong CLEF_KEY_DIM [[function_constant(0)]];
// Vector width per value-column group: 1, 2 or 4, chosen by the Rust side
// from the GPU family (wider on Apple10+). The `Values` branches below keep
// the per-lane register footprint bounded while the float4 paths maximize
// vector throughput; `CLEF_SECTIONS`/`CLEF_ACTIVE` derive the section count
// and the live lanes from it so every branch stays in sync.
#if CLEF_VALUES == 1
using Values = float;
#elif CLEF_VALUES == 2
using Values = float2;
#else
using Values = float4;
#endif
#define CLEF_SECTIONS ((CLEF_VALUES + 3) / 4)
#define CLEF_ACTIVE min(4u, uint(CLEF_VALUES))

// Each SIMD group owns a bounded vector of value columns. Each lane holds
// up to four key-state vectors, avoiding cross-group memory and barriers.
kernel void clef_delta(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& tokens [[buffer(2)]],
    constant uint& heads [[buffer(3)]],
    constant uint& value_dim [[buffer(4)]],
#if CLEF_CACHE
    device const float* initial [[buffer(5)]],
    constant uint& capture [[buffer(6)]],
    constant bool& resume [[buffer(7)]],
#endif
    uint group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]]) {
    const uint key_dim = uint(CLEF_KEY_DIM);
    uint tiles = (value_dim + 4 * CLEF_VALUES - 1) / (4 * CLEF_VALUES);
    uint head = group / tiles;
    uint value = ((group % tiles) * 4 + lane / 32) * CLEF_VALUES;
    uint key_lane = lane % 32;
    // Precondition: key_dim <= 128, so parts <= 4 and the fixed-size
    // `keys`/`queries`/`products` register arrays below never overflow.
    // Enforced on the Rust side (`dimensions` + `CLEF_KEY_DIM` constant).
    uint parts = (key_dim + 31) / 32;
    uint packed_width = 2 * key_dim + value_dim + 2;
    Values state[4 * CLEF_SECTIONS];
    #pragma clang loop unroll(full)
    for (uint i = 0; i < 4 * CLEF_SECTIONS; ++i) state[i] = Values(0);
#if CLEF_CACHE
    if (resume) {
        for (uint section = 0; section < CLEF_SECTIONS; ++section) {
            uint first = value + section * 4;
            for (uint p = 0; p < parts; ++p) {
                uint index = key_lane + p * 32;
                float4 loaded = float4(0);
                for (uint v = 0; v < CLEF_ACTIVE; ++v)
                    if (index < key_dim && first + v < value_dim)
                        loaded[v] = initial[(head * key_dim + index) * value_dim + first + v];
#if CLEF_VALUES == 1
                state[section * 4 + p] = loaded.x;
#elif CLEF_VALUES == 2
                state[section * 4 + p] = loaded.xy;
#else
                state[section * 4 + p] = loaded;
#endif
            }
        }
    }
#endif
    for (uint t = 0; t < tokens; ++t) {
        uint base = (t * heads + head) * packed_width;
        float decay = input[base + 2 * key_dim + value_dim];
        float beta = input[base + 2 * key_dim + value_dim + 1];
        float keys[4], queries[4];
        #pragma clang loop unroll(full)
        for (uint p = 0; p < parts; ++p) {
            uint index = key_lane + p * 32;
            keys[p] = index < key_dim ? input[base + key_dim + index] : 0.0f;
            queries[p] = index < key_dim ? input[base + index] : 0.0f;
        }
        #pragma clang loop unroll(full)
        for (uint section = 0; section < CLEF_SECTIONS; ++section) {
            uint first = value + section * 4;
            Values products[4] = {Values(0), Values(0), Values(0), Values(0)};
            #pragma clang loop unroll(full)
            for (uint p = 0; p < parts; ++p) {
                state[section * 4 + p] *= decay;
                products[p] = state[section * 4 + p] * keys[p];
            }
            // Preserve the reference (0+64)+(32+96) reduction per value column.
            Values dot = (products[0] + products[2]) + (products[1] + products[3]);
            Values memory = simd_sum(dot);
            // The v values depend only on the subgroup (`first` is constant
            // across the 32 lanes of a subgroup), so one lane loads and the
            // rest take it via broadcast instead of 32 redundant loads.
            float4 loaded = float4(0);
            if (key_lane == 0) {
                #pragma clang loop unroll(full)
                for (uint v = 0; v < CLEF_ACTIVE; ++v)
                    if (first + v < value_dim) loaded[v] = input[base + 2 * key_dim + first + v];
            }
            loaded = float4(
                simd_broadcast(loaded.x, 0),
                simd_broadcast(loaded.y, 0),
                simd_broadcast(loaded.z, 0),
                simd_broadcast(loaded.w, 0));
#if CLEF_VALUES == 1
            Values actual = loaded.x;
#elif CLEF_VALUES == 2
            Values actual = loaded.xy;
#else
            Values actual = loaded;
#endif
            Values delta = (actual - memory) * beta;
            #pragma clang loop unroll(full)
            for (uint p = 0; p < parts; ++p) {
                state[section * 4 + p] += keys[p] * delta;
                products[p] = state[section * 4 + p] * queries[p];
            }
#if CLEF_CACHE
            // Capture writes the recurrent state after the first `capture`
            // tokens to the tail of the output buffer, so a later call can
            // resume from exactly this point. `capture == 0` disables it;
            // the Rust side rejects `capture > tokens`.
            if (t + 1 == capture) {
                for (uint p = 0; p < parts; ++p) {
                    uint index = key_lane + p * 32;
#if CLEF_VALUES == 2
                    float4 saved = float4(state[section * 4 + p], 0.0f, 0.0f);
#else
                    float4 saved = float4(state[section * 4 + p]);
#endif
                    for (uint v = 0; v < CLEF_ACTIVE; ++v)
                        if (index < key_dim && first + v < value_dim)
                            output[tokens * heads * value_dim + (head * key_dim + index) * value_dim + first + v] = saved[v];
                }
            }
#endif
            dot = (products[0] + products[2]) + (products[1] + products[3]);
#if CLEF_VALUES == 2
            float4 result = float4(simd_sum(dot), 0.0f, 0.0f);
#else
            float4 result = float4(simd_sum(dot));
#endif
            if (key_lane == 0) {
                #pragma clang loop unroll(full)
                for (uint v = 0; v < CLEF_ACTIVE; ++v)
                    if (first + v < value_dim)
                        output[(t * heads + head) * value_dim + first + v] = result[v];
            }
        }
    }
}
