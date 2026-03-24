#include <metal_stdlib>
using namespace metal;

kernel void sgemm_32x32_unrolled(
                        const device float* A,
                        const device float* B,
                        device float* C,
                        constant const uint64_t& M,
                        constant const uint64_t& N,
                        constant const uint64_t& P,
                        //                  threadgroup float* As,
                        //                  threadgroup float* Bs,
                        uint2 gid [[thread_position_in_grid]],
                        uint2 lid [[thread_position_in_threadgroup]],
                        uint2 group_id [[threadgroup_position_in_grid]],
                        uint2 group_size [[threads_per_threadgroup]]
                        )
{
    constexpr auto dim = 4;
    constexpr auto dK = dim * 8;
    // threadgroup float As[dK * dK];
    // threadgroup float Bs[dK * dK];

    // C += lid.y * 16 * N;
    A += group_id.y * (dK * 2) * N;
    B += group_id.x * dK;
    C += group_id.y * (dK * 2) * P + group_id.x * dK;

    A += lid.y * dK * N;
    C += lid.y * dK * P;

    simdgroup_float8x8 Am[dim];
    simdgroup_float8x8 Bm[dim];
    simdgroup_float8x8 acc[dim][dim];

    // --- Unrolled initialization of acc matrix ---
    acc[0][0] = simdgroup_float8x8(0);
    acc[0][1] = simdgroup_float8x8(0);
    acc[0][2] = simdgroup_float8x8(0);
    acc[0][3] = simdgroup_float8x8(0);

    acc[1][0] = simdgroup_float8x8(0);
    acc[1][1] = simdgroup_float8x8(0);
    acc[1][2] = simdgroup_float8x8(0);
    acc[1][3] = simdgroup_float8x8(0);

    acc[2][0] = simdgroup_float8x8(0);
    acc[2][1] = simdgroup_float8x8(0);
    acc[2][2] = simdgroup_float8x8(0);
    acc[2][3] = simdgroup_float8x8(0);

    acc[3][0] = simdgroup_float8x8(0);
    acc[3][1] = simdgroup_float8x8(0);
    acc[3][2] = simdgroup_float8x8(0);
    acc[3][3] = simdgroup_float8x8(0);

    for (uint k = 0; k < N; k += 8) {
        // --- Unrolled loads from A into Am[0..3] ---
        simdgroup_load(Am[0], A + (k) + (0 * 8 * N), N);
        simdgroup_load(Am[1], A + (k) + (1 * 8 * N), N);
        simdgroup_load(Am[2], A + (k) + (2 * 8 * N), N);
        simdgroup_load(Am[3], A + (k) + (3 * 8 * N), N);

        // --- Unrolled loads from B into Bm[0..3] ---
        simdgroup_load(Bm[0], B + (k * P) + (0 * 8), P);
        simdgroup_load(Bm[1], B + (k * P) + (1 * 8), P);
        simdgroup_load(Bm[2], B + (k * P) + (2 * 8), P);
        simdgroup_load(Bm[3], B + (k * P) + (3 * 8), P);

        // --- Unrolled multiply-accumulate for all 4x4 combinations ---
        // y = 0
        simdgroup_multiply_accumulate(acc[0][0], Am[0], Bm[0], acc[0][0]);
        simdgroup_multiply_accumulate(acc[0][1], Am[0], Bm[1], acc[0][1]);
        simdgroup_multiply_accumulate(acc[0][2], Am[0], Bm[2], acc[0][2]);
        simdgroup_multiply_accumulate(acc[0][3], Am[0], Bm[3], acc[0][3]);

        // y = 1
        simdgroup_multiply_accumulate(acc[1][0], Am[1], Bm[0], acc[1][0]);
        simdgroup_multiply_accumulate(acc[1][1], Am[1], Bm[1], acc[1][1]);
        simdgroup_multiply_accumulate(acc[1][2], Am[1], Bm[2], acc[1][2]);
        simdgroup_multiply_accumulate(acc[1][3], Am[1], Bm[3], acc[1][3]);

        // y = 2
        simdgroup_multiply_accumulate(acc[2][0], Am[2], Bm[0], acc[2][0]);
        simdgroup_multiply_accumulate(acc[2][1], Am[2], Bm[1], acc[2][1]);
        simdgroup_multiply_accumulate(acc[2][2], Am[2], Bm[2], acc[2][2]);
        simdgroup_multiply_accumulate(acc[2][3], Am[2], Bm[3], acc[2][3]);

        // y = 3
        simdgroup_multiply_accumulate(acc[3][0], Am[3], Bm[0], acc[3][0]);
        simdgroup_multiply_accumulate(acc[3][1], Am[3], Bm[1], acc[3][1]);
        simdgroup_multiply_accumulate(acc[3][2], Am[3], Bm[2], acc[3][2]);
        simdgroup_multiply_accumulate(acc[3][3], Am[3], Bm[3], acc[3][3]);
    }

    // --- Unrolled stores from acc to C ---
    simdgroup_store(acc[0][0], C + (0 * 8 * P) + (0 * 8), P);
    simdgroup_store(acc[0][1], C + (0 * 8 * P) + (1 * 8), P);
    simdgroup_store(acc[0][2], C + (0 * 8 * P) + (2 * 8), P);
    simdgroup_store(acc[0][3], C + (0 * 8 * P) + (3 * 8), P);

    simdgroup_store(acc[1][0], C + (1 * 8 * P) + (0 * 8), P);
    simdgroup_store(acc[1][1], C + (1 * 8 * P) + (1 * 8), P);
    simdgroup_store(acc[1][2], C + (1 * 8 * P) + (2 * 8), P);
    simdgroup_store(acc[1][3], C + (1 * 8 * P) + (3 * 8), P);

    simdgroup_store(acc[2][0], C + (2 * 8 * P) + (0 * 8), P);
    simdgroup_store(acc[2][1], C + (2 * 8 * P) + (1 * 8), P);
    simdgroup_store(acc[2][2], C + (2 * 8 * P) + (2 * 8), P);
    simdgroup_store(acc[2][3], C + (2 * 8 * P) + (3 * 8), P);

    simdgroup_store(acc[3][0], C + (3 * 8 * P) + (0 * 8), P);
    simdgroup_store(acc[3][1], C + (3 * 8 * P) + (1 * 8), P);
    simdgroup_store(acc[3][2], C + (3 * 8 * P) + (2 * 8), P);
    simdgroup_store(acc[3][3], C + (3 * 8 * P) + (3 * 8), P);
}

kernel void sgemv(
                  const device float* A,
                  const device float* B,
                  device float* C,
                  constant const uint& H,
                  constant const uint& W,
//                  threadgroup float* shared,
                  uint2 gid [[thread_position_in_grid]],
                  uint2 lid [[thread_position_in_threadgroup]],
                  uint2 group_id [[threadgroup_position_in_grid]],
                  uint2 group_size [[threads_per_threadgroup]]
                  )
{
    threadgroup float shared[64];
    threadgroup float acc[2][32];
    C[gid.y] = 0;
    acc[lid.y][lid.x] = 0;

    for (uint i = 0; i < W; i += 64) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        shared[lid.y * 32 + lid.x] = B[i + lid.y * 32 + lid.x];
        
        threadgroup_barrier(mem_flags::mem_threadgroup);
        acc[lid.y][lid.x] += shared[lid.x * 2 + 0] * A[gid.y * W + i + lid.x * 2 + 0] + shared[lid.x * 2 + 1] * A[gid.y * W + i + lid.x * 2 + 1];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid.x == 0) {
        for (uint i = 0; i < 32; i++) {
            C[gid.y] += acc[lid.y][i];
        }
    }
}

kernel void dot_reduce0(
                   const device float4* X,
                   const device float4* Y,
                   device float* output,
                   constant const uint64_t& N,
                   threadgroup float* shared,
                   uint gid [[thread_position_in_grid]],
                   uint lid [[thread_position_in_threadgroup]],
                   uint group_id [[threadgroup_position_in_grid]],
                   uint group_size [[threads_per_threadgroup]],
                   uint groups [[threadgroups_per_grid]]
                   )
{
    shared[lid] = dot(X[(group_id * group_size + lid)], Y[(group_id * group_size + lid)]);
    X += groups * group_size;
    Y += groups * group_size;
    shared[group_size + lid] = dot(X[(group_id * group_size + lid)], Y[(group_id * group_size + lid)]);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint offset = group_size; offset > 0; offset = offset >> 1) {
        if (lid < offset) {
            shared[lid] += shared[lid + offset];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    if (lid == 0) output[group_id] = shared[0];
}

kernel void dot_reduce1(
                          const device float* X,
                          device float* output,
                          constant const uint& N,
                          threadgroup float* shared,
                          uint gid [[thread_position_in_grid]],
                          uint lid [[thread_position_in_threadgroup]],
                          uint group_size [[threads_per_threadgroup]]
                          )
{
    shared[lid] = X[lid];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    for (uint offset = group_size / 2; offset > 0; offset >>= 1) {
        if (lid < offset) {
            shared[lid] += shared[lid + offset];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    output[0] = shared[0];
}

kernel void sum_reduce0(
                   const device float4* X,
                   constant const float& y,
                   device float* output,
                   constant const uint64_t& N,
                   threadgroup float* shared,
                   uint gid [[thread_position_in_grid]],
                   uint lid [[thread_position_in_threadgroup]],
                   uint group_id [[threadgroup_position_in_grid]],
                   uint group_size [[threads_per_threadgroup]],
                   uint groups [[threadgroups_per_grid]]
                   )
{
    // sim_sum???
    shared[lid] = dot(X[(group_id * group_size + lid)], y);
    X += groups * group_size;
    shared[group_size + lid] = dot(X[(group_id * group_size + lid)], y);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint offset = group_size; offset > 0; offset = offset >> 1) {
        if (lid < offset) {
            shared[lid] += shared[lid + offset];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    if (lid == 0) output[group_id] = shared[0];
}

