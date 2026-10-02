#include <metal_stdlib>
using namespace metal;
#pragma clang fp contract(off)
#pragma clang fp reassociate(off)
constant ulong CLEF_KEY_DIM [[function_constant(0)]];

// Each SIMD group owns one value column, with up to four key-state values
// per lane. SIMD collectives eliminate cross-group memory and barriers.
kernel void clef_delta(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& tokens [[buffer(2)]],
    constant uint& heads [[buffer(3)]],
    constant uint& value_dim [[buffer(4)]],
    uint group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]]) {
    const uint key_dim = uint(CLEF_KEY_DIM);
    uint tiles = (value_dim + 3) / 4;
    uint head = group / tiles;
    uint value = (group % tiles) * 4 + lane / 32;
    uint key_lane = lane % 32;
    uint parts = (key_dim + 31) / 32;
    uint packed_width = 2 * key_dim + value_dim + 2;
    float state[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (uint t = 0; t < tokens; ++t) {
        uint base = (t * heads + head) * packed_width;
        float decay = input[base + 2 * key_dim + value_dim];
        float keys[4];
        float products[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        #pragma clang loop unroll(full)
        for (uint p = 0; p < parts; ++p) {
            uint index = key_lane + p * 32;
            keys[p] = index < key_dim ? input[base + key_dim + index] : 0.0f;
            state[p] *= decay;
            products[p] = state[p] * keys[p];
        }
        // Match Candle's 128-key block reduction: (0+64)+(32+96).
        float dot = (products[0] + products[2]) + (products[1] + products[3]);
        float memory = simd_sum(dot);
        float actual = value < value_dim ? input[base + 2 * key_dim + value] : 0.0f;
        float delta = (actual - memory) * input[base + 2 * key_dim + value_dim + 1];
        #pragma clang loop unroll(full)
        for (uint p = 0; p < parts; ++p) {
            uint index = key_lane + p * 32;
            float query = index < key_dim ? input[base + index] : 0.0f;
            state[p] += keys[p] * delta;
            products[p] = state[p] * query;
        }
        dot = (products[0] + products[2]) + (products[1] + products[3]);
        float result = simd_sum(dot);
        if (key_lane == 0 && value < value_dim)
            output[(t * heads + head) * value_dim + value] = result;
    }
}
