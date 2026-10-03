#include <metal_stdlib>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;

template <typename Left, typename Right, typename Destination>
inline void project(Left left, Right right, Destination destination) {
    constexpr auto descriptor = matmul2d_descriptor(CLEF_TILE_M, CLEF_TILE_N, dynamic_length_v<int>, false, true, false);
    matmul2d<descriptor, execution_simdgroups<4>> operation;
    auto product = operation.get_destination_cooperative_tensor<Left, Right, float>();
    operation.run(left, right, product);
    auto rounded = operation.get_destination_cooperative_tensor<Left, Right, half>();
    for (ushort i = 0; i < rounded.get_capacity(); ++i) rounded[i] = half(product[i]);
    rounded.store(destination);
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
    int row = int(group.y) * CLEF_TILE_M;
    int column = int(group.x) * CLEF_TILE_N;
    auto left = tensor(x, dextents<int, 2>{int(inner), int(rows)}, array<int, 2>{1, int(inner)});
    auto right = tensor(weight, dextents<int, 2>{int(inner), int(columns)}, array<int, 2>{1, int(inner)});
    auto destination = tensor(output, dextents<int, 2>{int(columns), int(rows)}, array<int, 2>{1, int(columns)});
    if (row + CLEF_TILE_M <= int(rows) && column + CLEF_TILE_N <= int(columns)) {
        project(left.slice<dynamic_extent, CLEF_TILE_M>(0, row),
                right.slice<dynamic_extent, CLEF_TILE_N>(0, column),
                destination.slice<CLEF_TILE_N, CLEF_TILE_M>(column, row));
    } else {
        project(left.slice(0, row), right.slice(0, column), destination.slice(column, row));
    }
}
