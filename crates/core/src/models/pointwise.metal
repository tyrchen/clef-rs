#include <metal_stdlib>
using namespace metal;
#pragma clang fp contract(off)
#pragma clang fp reassociate(off)

// Preserve the reference loader, block tree, mean scaling, sqrt and reciprocal.
kernel void clef_rms(
    device const float* input [[buffer(0)]],
    device const float* weight [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& width [[buffer(3)]],
    constant float& eps [[buffer(4)]],
    uint row [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]],
    uint threads [[threads_per_threadgroup]]) {
    threadgroup float partial[1024];
    threadgroup float inverse;
    uint base = row * width;
    float sum = 0.0f;
    for (uint i = lane; i < width; i += threads) {
        float value = input[base + i];
        sum += value * value;
    }
    partial[lane] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint step = threads / 2; step >= 64; step /= 2) {
        if (lane < step) partial[lane] += partial[lane + step];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lane < 32) {
        float value = threads >= 64 ? partial[lane] + partial[lane + 32] : sum;
        float total = simd_sum(value);
        if (lane == 0) {
            volatile float root = sqrt(total * (1.0f / float(width)) + eps);
            inverse = 1.0f / root;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = lane; i < width; i += threads) {
        float scaled = input[base + i] * inverse;
        output[base + i] = scaled * weight[i];
    }
}

// Four causal taps, in the original oldest-to-newest accumulation order.
kernel void clef_conv(
    device const float* input [[buffer(0)]],
    device const float* weight [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& tokens [[buffer(3)]],
    constant uint& width [[buffer(4)]],
    uint index [[thread_position_in_grid]]) {
    if (index >= tokens * width) return;
    uint row = index / width;
    uint channel = index % width;
    float value = 0.0f;
    for (uint tap = 0; tap < 4; ++tap) {
        uint delay = 3 - tap;
        if (row >= delay) {
            float product = input[(row - delay) * width + channel] * weight[channel * 4 + tap];
            value += product;
        }
    }
    output[index] = value;
}
