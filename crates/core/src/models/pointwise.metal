#include <metal_stdlib>
using namespace metal;
// Same intent as delta.metal: disabling FP contraction and reassociation
// keeps the kernel's arithmetic rounded exactly as written, so numerics stay
// predictable across compiler versions. The RMS kernel is not bit-identical
// with the portable reference (different reduction order); it only needs to
// match within the test tolerance.
#pragma clang fp contract(off)
#pragma clang fp reassociate(off)

// Preserve the reference loader, block tree, mean scaling, sqrt and reciprocal.
//
// Preconditions (enforced on the Rust side): `width` is a power of two in
// 1..=4096, and the threadgroup holds `(width / 2).max(1).next_power_of_two()`
// threads, so the thread count is always a power of two as well. Small widths
// degrade gracefully: with fewer than 32 threads only the first SIMD group is
// partially filled and `simd_sum` covers exactly the active lanes.
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
    // Vector loads: four consecutive elements per iteration. The compiler
    // coalesces these into 128-bit loads; the tail loop covers widths that
    // are not a multiple of four.
    uint vec_width = width / 4;
    for (uint i = lane; i < vec_width; i += threads) {
        float4 values = float4(
            input[base + 4 * i], input[base + 4 * i + 1],
            input[base + 4 * i + 2], input[base + 4 * i + 3]);
        sum += values.x * values.x + values.y * values.y
             + values.z * values.z + values.w * values.w;
    }
    for (uint i = vec_width * 4 + lane; i < width; i += threads) {
        float value = input[base + i];
        sum += value * value;
    }
    partial[lane] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint step = threads / 2; step >= 32; step /= 2) {
        if (lane < step) partial[lane] += partial[lane + step];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lane < 32) {
        float total = simd_sum(partial[lane]);
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

// Fuse the reference L2 reductions, query scaling, head repetition, decay and packing.
// 64 lanes load (0+64), then reduce (0+64)+(32+96) through the same SIMD tree.
kernel void clef_delta_prepare(
    device const float* mixed [[buffer(0)]],
    device const float* g [[buffer(1)]],
    device const float* beta [[buffer(2)]],
    device float* packed [[buffer(3)]],
    uint row [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]]) {
    threadgroup float query_sum[64];
    threadgroup float key_sum[64];
    threadgroup float query_root;
    threadgroup float key_root;
    uint token = row / 32;
    uint head = row % 32;
    uint base = token * 8192 + (head / 2) * 128;
    float q0 = mixed[base + lane];
    float q1 = mixed[base + lane + 64];
    float k0 = mixed[base + 2048 + lane];
    float k1 = mixed[base + 2048 + lane + 64];
    query_sum[lane] = q0 * q0 + q1 * q1;
    key_sum[lane] = k0 * k0 + k1 * k1;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane < 32) {
        float qs = simd_sum(query_sum[lane] + query_sum[lane + 32]);
        float ks = simd_sum(key_sum[lane] + key_sum[lane + 32]);
        if (lane == 0) {
            volatile float qr = sqrt(qs + 1e-6f);
            volatile float kr = sqrt(ks + 1e-6f);
            query_root = qr;
            key_root = kr;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint output = row * 386;
    float query_scale = 0.08838834764831845f;
    packed[output + lane] = (q0 / query_root) * query_scale;
    packed[output + lane + 64] = (q1 / query_root) * query_scale;
    packed[output + 128 + lane] = k0 / key_root;
    packed[output + 128 + lane + 64] = k1 / key_root;
    uint value_base = token * 8192 + 4096 + head * 128;
    packed[output + 256 + lane] = mixed[value_base + lane];
    packed[output + 256 + lane + 64] = mixed[value_base + lane + 64];
    if (lane == 0) {
        packed[output + 384] = exp(g[row]);
        packed[output + 385] = beta[row];
    }
}
