#include <metal_stdlib>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;

template <typename Left, typename Right, typename Destination>
inline void project(Left left, Right right, Destination destination, int inner) {
    constexpr auto descriptor = matmul2d_descriptor(CLEF_TILE_M, CLEF_TILE_N, CLEF_K_BLOCK == 0 ? dynamic_length_v<int> : CLEF_K_BLOCK, false, true, false,
        CLEF_K_BLOCK == 0 ? matmul2d_descriptor::mode::multiply : matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<descriptor, execution_simdgroups<4>> operation;
    auto product = operation.get_destination_cooperative_tensor<Left, Right, float>();
#if CLEF_K_BLOCK == 0
    operation.run(left, right, product);
#else
    for (ushort i = 0; i < product.get_capacity(); ++i) product[i] = 0.0f;
    for (int k = 0; k < inner; k += CLEF_K_BLOCK) {
        threadgroup_barrier(mem_flags::mem_none);
        auto a = left.slice(k, 0);
        auto b = right.slice(k, 0);
        operation.run(a, b, product);
    }
#endif
    auto rounded = operation.get_destination_cooperative_tensor<Left, Right, half>();
    for (ushort i = 0; i < rounded.get_capacity(); ++i) rounded[i] = half(product[i]);
    rounded.store(destination);
}

template <typename Left, typename Right, typename Destination>
inline void project_gated(Left left, Right gate, Right up, Destination destination, int inner) {
    constexpr auto descriptor = matmul2d_descriptor(CLEF_TILE_M, CLEF_TILE_N, dynamic_length_v<int>, false, true, false);
    matmul2d<descriptor, execution_simdgroups<4>> operation;
    auto product = operation.get_destination_cooperative_tensor<Left, Right, float>();
    auto rounded = operation.get_destination_cooperative_tensor<Left, Right, half>();
    operation.run(left, gate, product);
    for (ushort i = 0; i < rounded.get_capacity(); ++i) {
        half g = half(product[i]);
        rounded[i] = half(g / (1 + exp(-g)));
    }
    threadgroup_barrier(mem_flags::mem_none);
    operation.run(left, up, product);
    for (ushort i = 0; i < rounded.get_capacity(); ++i)
        rounded[i] = half(rounded[i] * half(product[i]));
    rounded.store(destination);
}

inline uint2 tile_position(uint2 position, uint rows, uint columns) {
    uint group = position.x;
    uint mt = (rows + CLEF_TILE_M - 1) / CLEF_TILE_M;
    uint nt = (columns + CLEF_TILE_N - 1) / CLEF_TILE_N;
    uint r, c;
#if CLEF_WALK == 1
    uint block = group / 64;
    uint code = group % 64;
    uint block_cols = (nt + 7) / 8;
    r = (block / block_cols) * 8 + ((code >> 1) & 1) + ((code >> 2) & 2) + ((code >> 3) & 4);
    c = (block % block_cols) * 8 + (code & 1) + ((code >> 1) & 2) + ((code >> 2) & 4);
#elif CLEF_WALK == 2
    uint block = group / (8 * nt);
    uint in_block = group % (8 * nt);
    uint height = min(8u, mt - block * 8);
    r = block * 8 + in_block % height;
    c = in_block / height;
#else
    r = position.y;
    c = position.x;
#endif
    return uint2(r,c);
}

// Device tensor views preserve row-major x and transposed row-major weights.
// Dynamic extents guard partial M/N tiles; all 128 threads participate together.
kernel void clef_neural_gemm(
    device half* x [[buffer(0)]],
    device half* weight [[buffer(1)]],
    device half* output [[buffer(2)]],
    constant uint& rows [[buffer(3)]],
    constant uint& columns [[buffer(4)]],
    constant uint& inner [[buffer(5)]],
    uint2 group [[threadgroup_position_in_grid]]) {
    uint2 pos = tile_position(group, rows, columns);
    uint r = pos.x, c = pos.y;
    if (r * CLEF_TILE_M >= rows || c * CLEF_TILE_N >= columns) return;
    int row = int(r) * CLEF_TILE_M;
    int column = int(c) * CLEF_TILE_N;
    auto left = tensor(x, dextents<int, 2>{int(inner), int(rows)}, array<int, 2>{1, int(inner)});
    auto right = tensor(weight, dextents<int, 2>{int(inner), int(columns)}, array<int, 2>{1, int(inner)});
    auto destination = tensor(output, dextents<int, 2>{int(columns), int(rows)}, array<int, 2>{1, int(columns)});
    if (row + CLEF_TILE_M <= int(rows) && column + CLEF_TILE_N <= int(columns)) {
        project(left.slice<dynamic_extent, CLEF_TILE_M>(0, row),
                right.slice<dynamic_extent, CLEF_TILE_N>(0, column),
                destination.slice<CLEF_TILE_N, CLEF_TILE_M>(column, row), int(inner));
    } else {
        project(left.slice(0, row), right.slice(0, column), destination.slice(column, row), int(inner));
    }
}

// Device tensor views preserve row-major x and transposed row-major weights.
// Dynamic extents guard partial M/N tiles; all 128 threads participate together.
kernel void clef_neural_gated(
    device half* x [[buffer(0)]],
    device half* weight [[buffer(1)]],
    device half* output [[buffer(2)]],
    device half* up [[buffer(6)]],
    constant uint& rows [[buffer(3)]],
    constant uint& columns [[buffer(4)]],
    constant uint& inner [[buffer(5)]],
    uint2 group [[threadgroup_position_in_grid]]) {
    uint2 pos = tile_position(group, rows, columns);
    uint r = pos.x, c = pos.y;
    if (r * CLEF_TILE_M >= rows || c * CLEF_TILE_N >= columns) return;
    int row = int(r) * CLEF_TILE_M;
    int column = int(c) * CLEF_TILE_N;
    auto left = tensor(x, dextents<int, 2>{int(inner), int(rows)}, array<int, 2>{1, int(inner)});
    auto right = tensor(weight, dextents<int, 2>{int(inner), int(columns)}, array<int, 2>{1, int(inner)});
    auto other = tensor(up, dextents<int, 2>{int(inner), int(columns)}, array<int, 2>{1, int(inner)});
    auto destination = tensor(output, dextents<int, 2>{int(columns), int(rows)}, array<int, 2>{1, int(columns)});
    if (row + CLEF_TILE_M <= int(rows) && column + CLEF_TILE_N <= int(columns)) {
        project_gated(left.slice<dynamic_extent, CLEF_TILE_M>(0, row),
                right.slice<dynamic_extent, CLEF_TILE_N>(0, column),
                other.slice<dynamic_extent, CLEF_TILE_N>(0, column),
                destination.slice<CLEF_TILE_N, CLEF_TILE_M>(column, row), int(inner));
    } else {
        project_gated(left.slice(0, row), right.slice(0, column), other.slice(0, column), destination.slice(column, row), int(inner));
    }
}
