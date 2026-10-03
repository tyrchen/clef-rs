#include <metal_stdlib>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;
#pragma clang fp contract(off)
#pragma clang fp reassociate(off)
constant uint CHUNK = 32;

// Bounded products of decay avoid dividing by a possibly underflowed prefix.
kernel void clef_chunk_gram(
    device const float* x [[buffer(0)]], device float* matrices [[buffer(1)]],
    device float* factors [[buffer(2)]],
    constant uint& tokens [[buffer(10)]], constant uint& heads [[buffer(11)]],
    constant uint& key [[buffer(12)]], constant uint& value [[buffer(13)]],
    uint group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    uint ch = group / 8, head = ch % heads, block = ch / heads;
    uint row = (group % 8) * 4 + tid / 32, lane = tid % 32;
    uint count = min(CHUNK, tokens - block * CHUNK);
    if (row >= count) return;
    uint width = 2 * key + value + 2;
    uint base = ((block * CHUNK + row) * heads + head) * width;
    float decay = 1.0f;
    for (int col = int(row); col >= 0; --col) {
        uint other = ((block * CHUNK + uint(col)) * heads + head) * width;
        float kk[4] = {0,0,0,0}, qk[4] = {0,0,0,0};
        for (uint p = 0; p < 4; ++p) {
            uint k = lane + p * 32;
            if (k < key) {
                float v = x[other + key + k];
                kk[p] = x[base + key + k] * v;
                qk[p] = x[base + k] * v;
            }
        }
        float gram = simd_sum((kk[0]+kk[2])+(kk[1]+kk[3]));
        float attention = simd_sum((qk[0]+qk[2])+(qk[1]+qk[3]));
        if (lane == 0) {
            uint index = ch * 2 * CHUNK * CHUNK + row * CHUNK + uint(col);
            matrices[index] = uint(col) == row ? 0.0f : (gram * decay) * x[base + width - 1];
            matrices[index + CHUNK * CHUNK] = attention * decay;
        }
        decay *= x[other + width - 2];
    }
    if (lane == 0) {
        float suffix = 1.0f;
        for (uint j = row + 1; j < count; ++j)
            suffix *= x[((block * CHUNK + j) * heads + head) * width + width - 2];
        factors[ch * 2 * CHUNK + row] = decay;
        factors[ch * 2 * CHUNK + CHUNK + row] = suffix;
    }
}

// Each SIMD group independently solves one row of the unit lower inverse.
kernel void clef_chunk_inverse(
    device const float* matrices [[buffer(0)]], device float* inverse [[buffer(1)]],
    constant uint& tokens [[buffer(10)]], constant uint& heads [[buffer(11)]],
    uint group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    uint ch = group / 8, row = (group % 8) * 4 + tid / 32, lane = tid % 32;
    uint count = min(CHUNK, tokens - (ch / heads) * CHUNK);
    if (row >= count) return;
    uint base = ch * 2 * CHUNK * CHUNK;
    float v = lane == row ? 1.0f : 0.0f;
    for (int col = int(row) - 1; col >= 0; --col) {
        float product = lane > uint(col) && lane < row ? v * matrices[base + lane * CHUNK + uint(col)] : 0.0f;
        float sum = simd_sum(product);
        if (lane == uint(col)) v = -matrices[base + row * CHUNK + uint(col)] - sum;
    }
    inverse[ch * CHUNK * CHUNK + row * CHUNK + lane] = v;
}

kernel void clef_chunk_transform(
    device const float* x [[buffer(0)]], device const float* inverse [[buffer(1)]],
    device const float* factors [[buffer(2)]], device float* transformed [[buffer(3)]],
    constant uint& tokens [[buffer(10)]], constant uint& heads [[buffer(11)]],
    constant uint& key [[buffer(12)]], constant uint& value [[buffer(13)]],
    uint index [[thread_position_in_grid]]) {
    uint feature_width = 2 * key + value, chunks = (tokens + CHUNK - 1) / CHUNK;
    if (index >= chunks * heads * CHUNK * feature_width) return;
    uint feature = index % feature_width, row = (index / feature_width) % CHUNK;
    uint ch = index / (feature_width * CHUNK), head = ch % heads, block = ch / heads;
    uint count = min(CHUNK, tokens - block * CHUNK), width = 2 * key + value + 2;
    float sum = 0.0f;
    if (row < count) {
        for (uint j = 0; j <= row; ++j) {
            uint src = ((block * CHUNK + j) * heads + head) * width;
            float coefficient = inverse[ch * CHUNK * CHUNK + row * CHUNK + j] * x[src + width - 1];
            if (feature < key) coefficient *= factors[ch * 2 * CHUNK + j];
            if (feature < key + value) sum += coefficient * x[src + key + feature];
        }
    }
    if (feature >= key + value && row < count) {
        uint src = ((block * CHUNK + row) * heads + head) * width;
        sum = x[src + feature - value] * factors[ch * 2 * CHUNK + CHUNK + row];
    }
    transformed[index] = sum;
}

// Matrix operations process an entire 32-token / 32-value tile per chunk.
kernel void clef_chunk_carry(
    device float* x [[buffer(0)]], device float* transformed [[buffer(1)]],
    device const float* factors [[buffer(2)]], device float* innovation [[buffer(3)]],
    device float* prefix [[buffer(4)]],
    constant uint& tokens [[buffer(10)]], constant uint& heads [[buffer(11)]],
    constant uint& key [[buffer(12)]], constant uint& value [[buffer(13)]],
    uint group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float state[CLEF_CHUNK_KEY * 32], delta[32 * 32], projected[32 * 32];
    uint tiles = (value + 31) / 32, head = group / tiles, vbase = (group % tiles) * 32;
    uint width = 2 * key + value + 2, features = 2 * key + value;
    for (uint i = tid; i < key * 32; i += 128) state[i] = 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    auto state_tensor = tensor(state, dextents<int,2>{32,int(key)}, array<int,2>{1,32});
    constexpr auto dot_descriptor = matmul2d_descriptor(32,32,CLEF_CHUNK_KEY,false,false,false);
    matmul2d<dot_descriptor,execution_simdgroups<4>> dot_op;
    constexpr auto update_descriptor = matmul2d_descriptor(CLEF_CHUNK_KEY,32,32,true,false,false,
        matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<update_descriptor,execution_simdgroups<4>> update_op;
    for (uint block = 0; block < (tokens + CHUNK - 1) / CHUNK; ++block) {
        uint ch = block * heads + head, count = min(CHUNK, tokens - block * CHUNK);
        auto w = tensor(transformed + ch * CHUNK * features,
            dextents<int,2>{int(key),int(count)}, array<int,2>{1,int(features)});
        auto q = tensor(x + (block * CHUNK * heads + head) * width,
            dextents<int,2>{int(key),int(count)}, array<int,2>{1,int(heads * width)});
        auto d = tensor(delta, dextents<int,2>{32,32}, array<int,2>{1,32});
        auto p = tensor(projected, dextents<int,2>{32,32}, array<int,2>{1,32});
        dot_op.run(w, state_tensor, d);
        dot_op.run(q, state_tensor, p);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i = tid; i < CHUNK * 32; i += 128) {
            uint row = i / 32, v = vbase + i % 32;
            float correction = 0.0f;
            if (row < count && v < value) {
                correction = transformed[(ch * CHUNK + row) * features + key + v] - delta[i];
                uint dst = ((block * CHUNK + row) * heads + head) * value + v;
                innovation[dst] = correction;
                prefix[dst] = projected[i] * factors[ch * 2 * CHUNK + row];
            }
            delta[i] = correction;
        }
        float decay = factors[ch * 2 * CHUNK + count - 1];
        for (uint i = tid; i < key * 32; i += 128) state[i] *= decay;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        auto keys = tensor(transformed + ch * CHUNK * features + key + value,
            dextents<int,2>{int(key),int(count)}, array<int,2>{1,int(features)});
        auto result = update_op.get_destination_cooperative_tensor<decltype(keys),decltype(d),float>();
        result.load(state_tensor);
        update_op.run(keys, d, result);
        result.store(state_tensor);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

kernel void clef_chunk_output(
    device const float* matrices [[buffer(0)]], device const float* innovation [[buffer(1)]],
    device const float* prefix [[buffer(2)]], device float* output [[buffer(3)]],
    constant uint& tokens [[buffer(10)]], constant uint& heads [[buffer(11)]],
    constant uint& value [[buffer(13)]], uint index [[thread_position_in_grid]]) {
    if (index >= tokens * heads * value) return;
    uint v = index % value, head = (index / value) % heads, token = index / (value * heads);
    uint block = token / CHUNK, row = token % CHUNK, ch = block * heads + head;
    uint base = ch * 2 * CHUNK * CHUNK + CHUNK * CHUNK + row * CHUNK;
    float result = prefix[index];
    for (uint j = 0; j <= row; ++j)
        result += matrices[base + j] * innovation[((block * CHUNK + j) * heads + head) * value + v];
    output[index] = result;
}
