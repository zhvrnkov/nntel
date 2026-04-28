#include <metal_stdlib>
using namespace metal;

kernel void sgemm_unrolled_dim4(
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
  A += group_id.y * (dK * group_size.y) * N;
  B += group_id.x * dK;
  C += group_id.y * (dK * group_size.y) * P + group_id.x * dK;
  
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

kernel void sgemm_unrolled_dim2(
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
  constexpr auto dim = 2;
  constexpr auto dK = dim * 8;
  // threadgroup float As[dK * dK];
  // threadgroup float Bs[dK * dK];
  
  // C += lid.y * 16 * N;
  A += group_id.y * (dK * group_size.y) * N;
  B += group_id.x * dK;
  C += group_id.y * (dK * group_size.y) * P + group_id.x * dK;
  
  A += lid.y * dK * N;
  C += lid.y * dK * P;
  
  simdgroup_float8x8 Am[dim];
  simdgroup_float8x8 Bm[dim];
  simdgroup_float8x8 acc[dim][dim];
  
  // --- Unrolled initialization of acc matrix ---
  acc[0][0] = simdgroup_float8x8(0);
  acc[0][1] = simdgroup_float8x8(0);
  
  acc[1][0] = simdgroup_float8x8(0);
  acc[1][1] = simdgroup_float8x8(0);
  
  for (uint k = 0; k < N; k += 8) {
    // --- Unrolled loads from A into Am[0..3] ---
    simdgroup_load(Am[0], A + (k) + (0 * 8 * N), N);
    simdgroup_load(Am[1], A + (k) + (1 * 8 * N), N);
    
    // --- Unrolled loads from B into Bm[0..3] ---
    simdgroup_load(Bm[0], B + (k * P) + (0 * 8), P);
    simdgroup_load(Bm[1], B + (k * P) + (1 * 8), P);
    
    // --- Unrolled multiply-accumulate for all 4x4 combinations ---
    // y = 0
    simdgroup_multiply_accumulate(acc[0][0], Am[0], Bm[0], acc[0][0]);
    simdgroup_multiply_accumulate(acc[0][1], Am[0], Bm[1], acc[0][1]);
    
    // y = 1
    simdgroup_multiply_accumulate(acc[1][0], Am[1], Bm[0], acc[1][0]);
    simdgroup_multiply_accumulate(acc[1][1], Am[1], Bm[1], acc[1][1]);
    
  }
  
  // --- Unrolled stores from acc to C ---
  simdgroup_store(acc[0][0], C + (0 * 8 * P) + (0 * 8), P);
  simdgroup_store(acc[0][1], C + (0 * 8 * P) + (1 * 8), P);
  
  simdgroup_store(acc[1][0], C + (1 * 8 * P) + (0 * 8), P);
  simdgroup_store(acc[1][1], C + (1 * 8 * P) + (1 * 8), P);
}

kernel void sgemm_unrolled_dim1(
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
  constexpr auto dim = 1;
  constexpr auto dK = dim * 8;
  // threadgroup float As[dK * dK];
  // threadgroup float Bs[dK * dK];
  
  // C += lid.y * 16 * N;
  A += group_id.y * (dK * group_size.y) * N;
  B += group_id.x * dK;
  C += group_id.y * (dK * group_size.y) * P + group_id.x * dK;
  
  A += lid.y * dK * N;
  C += lid.y * dK * P;
  
  simdgroup_float8x8 Am[dim];
  simdgroup_float8x8 Bm[dim];
  simdgroup_float8x8 acc[dim][dim];
  
  // --- Unrolled initialization of acc matrix ---
  acc[0][0] = simdgroup_float8x8(0);
  
  for (uint k = 0; k < N; k += 8) {
    // --- Unrolled loads from A into Am[0..3] ---
    simdgroup_load(Am[0], A + (k) + (0 * 8 * N), N);
    
    // --- Unrolled loads from B into Bm[0..3] ---
    simdgroup_load(Bm[0], B + (k * P) + (0 * 8), P);
    
    // --- Unrolled multiply-accumulate for all 4x4 combinations ---
    // y = 0
    simdgroup_multiply_accumulate(acc[0][0], Am[0], Bm[0], acc[0][0]);
  }
  
  // --- Unrolled stores from acc to C ---
  simdgroup_store(acc[0][0], C + (0 * 8 * P) + (0 * 8), P);
}

kernel void sgemm_na_unrolled_dim4(
                                           const device float* A,
                                           const device float* B,
                                           device float* C,
                                           constant const uint64_t& M,
                                           constant const uint64_t& N,
                                           constant const uint64_t& P,
                                           threadgroup float* tA,
                                           threadgroup float* tB,
                                           threadgroup float* tC,
                                           uint2 gid [[thread_position_in_grid]],
                                           uint2 lid [[thread_position_in_threadgroup]],
                                           uint2 group_id [[threadgroup_position_in_grid]],
                                           uint2 group_size [[threads_per_threadgroup]]
                                           )
{
    constexpr uint dim = 4;
    constexpr uint dK = dim * 8;              // 32
    constexpr uint tileK = 8;                 // inner tile size for simdgroup
    constexpr uint thread_tile_rows = dim;    // 4
    constexpr uint thread_tile_cols = dim;    // 4
    constexpr uint thread_tile_size = thread_tile_rows * thread_tile_cols * tileK * tileK; // 4*4*64=1024

    // Precompute thread‑local indices
    uint lid_div8 = lid.x / 8;
    uint lid_mod8 = lid.x % 8;

    uint tgidyA = group_id.y * (dK * group_size.y) + lid.y * dK;
    uint tgidxB = group_id.x * dK;

    // Simdgroup accumulators
    simdgroup_float8x8 acc[dim][dim];
    for (uint y = 0; y < dim; ++y)
        for (uint x = 0; x < dim; ++x)
            acc[y][x] = simdgroup_float8x8(0);

    // Threadgroup pointers for this thread
    tA += lid.y * (dim * 8 * 8);
    tB += lid.y * (dim * 8 * 8);
    tC += lid.y * (dim * dim * 8 * 8);

    uint Nk = ((N + tileK - 1) / tileK) * tileK;

    // Main reduction loop over K
    for (uint k = 0; k < Nk; k += tileK) {
        // ----- Load A and B tiles into threadgroup memory (fully unrolled over d=0..3) -----
        // d = 0
        {
            uint didyA = 0 * tileK;
            uint didxB = 0 * tileK;

            // A row 0
            uint rowA0 = tgidyA + 0*4 + didyA + lid_div8;
            uint colA0 = k + lid_mod8;
            tA[((0*4) + lid_div8) * (dim * tileK) + (0*tileK + lid_mod8)] = (rowA0 < M && colA0 < N) ? A[rowA0 * N + colA0] : 0.0f;
            // A row 1
            uint rowA1 = tgidyA + 1*4 + didyA + lid_div8;
            uint colA1 = k + lid_mod8;
            tA[((1*4) + lid_div8) * (dim * tileK) + (0*tileK + lid_mod8)] = (rowA1 < M && colA1 < N) ? A[rowA1 * N + colA1] : 0.0f;

            // B row 0
            uint rowB0 = 0*4 + k + lid_div8;
            uint colB0 = tgidxB + didxB + lid_mod8;
            tB[((0*4) + lid_div8) * (dim * tileK) + (0*tileK + lid_mod8)] = (rowB0 < N && colB0 < P) ? B[rowB0 * P + colB0] : 0.0f;
            // B row 1
            uint rowB1 = 1*4 + k + lid_div8;
            uint colB1 = tgidxB + didxB + lid_mod8;
            tB[((1*4) + lid_div8) * (dim * tileK) + (0*tileK + lid_mod8)] = (rowB1 < N && colB1 < P) ? B[rowB1 * P + colB1] : 0.0f;
        }
        // d = 1
        {
            uint didyA = 1 * tileK;
            uint didxB = 1 * tileK;

            uint rowA0 = tgidyA + 0*4 + didyA + lid_div8;
            uint colA0 = k + lid_mod8;
            tA[((0*4) + lid_div8) * (dim * tileK) + (1*tileK + lid_mod8)] = (rowA0 < M && colA0 < N) ? A[rowA0 * N + colA0] : 0.0f;
            uint rowA1 = tgidyA + 1*4 + didyA + lid_div8;
            uint colA1 = k + lid_mod8;
            tA[((1*4) + lid_div8) * (dim * tileK) + (1*tileK + lid_mod8)] = (rowA1 < M && colA1 < N) ? A[rowA1 * N + colA1] : 0.0f;

            uint rowB0 = 0*4 + k + lid_div8;
            uint colB0 = tgidxB + didxB + lid_mod8;
            tB[((0*4) + lid_div8) * (dim * tileK) + (1*tileK + lid_mod8)] = (rowB0 < N && colB0 < P) ? B[rowB0 * P + colB0] : 0.0f;
            uint rowB1 = 1*4 + k + lid_div8;
            uint colB1 = tgidxB + didxB + lid_mod8;
            tB[((1*4) + lid_div8) * (dim * tileK) + (1*tileK + lid_mod8)] = (rowB1 < N && colB1 < P) ? B[rowB1 * P + colB1] : 0.0f;
        }
        // d = 2
        {
            uint didyA = 2 * tileK;
            uint didxB = 2 * tileK;

            uint rowA0 = tgidyA + 0*4 + didyA + lid_div8;
            uint colA0 = k + lid_mod8;
            tA[((0*4) + lid_div8) * (dim * tileK) + (2*tileK + lid_mod8)] = (rowA0 < M && colA0 < N) ? A[rowA0 * N + colA0] : 0.0f;
            uint rowA1 = tgidyA + 1*4 + didyA + lid_div8;
            uint colA1 = k + lid_mod8;
            tA[((1*4) + lid_div8) * (dim * tileK) + (2*tileK + lid_mod8)] = (rowA1 < M && colA1 < N) ? A[rowA1 * N + colA1] : 0.0f;

            uint rowB0 = 0*4 + k + lid_div8;
            uint colB0 = tgidxB + didxB + lid_mod8;
            tB[((0*4) + lid_div8) * (dim * tileK) + (2*tileK + lid_mod8)] = (rowB0 < N && colB0 < P) ? B[rowB0 * P + colB0] : 0.0f;
            uint rowB1 = 1*4 + k + lid_div8;
            uint colB1 = tgidxB + didxB + lid_mod8;
            tB[((1*4) + lid_div8) * (dim * tileK) + (2*tileK + lid_mod8)] = (rowB1 < N && colB1 < P) ? B[rowB1 * P + colB1] : 0.0f;
        }
        // d = 3
        {
            uint didyA = 3 * tileK;
            uint didxB = 3 * tileK;

            uint rowA0 = tgidyA + 0*4 + didyA + lid_div8;
            uint colA0 = k + lid_mod8;
            tA[((0*4) + lid_div8) * (dim * tileK) + (3*tileK + lid_mod8)] = (rowA0 < M && colA0 < N) ? A[rowA0 * N + colA0] : 0.0f;
            uint rowA1 = tgidyA + 1*4 + didyA + lid_div8;
            uint colA1 = k + lid_mod8;
            tA[((1*4) + lid_div8) * (dim * tileK) + (3*tileK + lid_mod8)] = (rowA1 < M && colA1 < N) ? A[rowA1 * N + colA1] : 0.0f;

            uint rowB0 = 0*4 + k + lid_div8;
            uint colB0 = tgidxB + didxB + lid_mod8;
            tB[((0*4) + lid_div8) * (dim * tileK) + (3*tileK + lid_mod8)] = (rowB0 < N && colB0 < P) ? B[rowB0 * P + colB0] : 0.0f;
            uint rowB1 = 1*4 + k + lid_div8;
            uint colB1 = tgidxB + didxB + lid_mod8;
            tB[((1*4) + lid_div8) * (dim * tileK) + (3*tileK + lid_mod8)] = (rowB1 < N && colB1 < P) ? B[rowB1 * P + colB1] : 0.0f;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Load A tiles (4 of them)
        simdgroup_float8x8 Am0, Am1, Am2, Am3;
        simdgroup_load(Am0, &tA[0 * tileK], dim * tileK);
        simdgroup_load(Am1, &tA[1 * tileK], dim * tileK);
        simdgroup_load(Am2, &tA[2 * tileK], dim * tileK);
        simdgroup_load(Am3, &tA[3 * tileK], dim * tileK);

        // Load B tiles (4 of them)
        simdgroup_float8x8 Bm0, Bm1, Bm2, Bm3;
        simdgroup_load(Bm0, &tB[0 * tileK], dim * tileK);
        simdgroup_load(Bm1, &tB[1 * tileK], dim * tileK);
        simdgroup_load(Bm2, &tB[2 * tileK], dim * tileK);
        simdgroup_load(Bm3, &tB[3 * tileK], dim * tileK);

        // Multiply-accumulate (4x4 outer product)
        simdgroup_multiply_accumulate(acc[0][0], Am0, Bm0, acc[0][0]);
        simdgroup_multiply_accumulate(acc[0][1], Am0, Bm1, acc[0][1]);
        simdgroup_multiply_accumulate(acc[0][2], Am0, Bm2, acc[0][2]);
        simdgroup_multiply_accumulate(acc[0][3], Am0, Bm3, acc[0][3]);

        simdgroup_multiply_accumulate(acc[1][0], Am1, Bm0, acc[1][0]);
        simdgroup_multiply_accumulate(acc[1][1], Am1, Bm1, acc[1][1]);
        simdgroup_multiply_accumulate(acc[1][2], Am1, Bm2, acc[1][2]);
        simdgroup_multiply_accumulate(acc[1][3], Am1, Bm3, acc[1][3]);

        simdgroup_multiply_accumulate(acc[2][0], Am2, Bm0, acc[2][0]);
        simdgroup_multiply_accumulate(acc[2][1], Am2, Bm1, acc[2][1]);
        simdgroup_multiply_accumulate(acc[2][2], Am2, Bm2, acc[2][2]);
        simdgroup_multiply_accumulate(acc[2][3], Am2, Bm3, acc[2][3]);

        simdgroup_multiply_accumulate(acc[3][0], Am3, Bm0, acc[3][0]);
        simdgroup_multiply_accumulate(acc[3][1], Am3, Bm1, acc[3][1]);
        simdgroup_multiply_accumulate(acc[3][2], Am3, Bm2, acc[3][2]);
        simdgroup_multiply_accumulate(acc[3][3], Am3, Bm3, acc[3][3]);

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Store accumulators back to threadgroup memory
    simdgroup_store(acc[0][0], tC + 0 * (dim * tileK * tileK) + 0 * tileK, dim * tileK);
    simdgroup_store(acc[0][1], tC + 0 * (dim * tileK * tileK) + 1 * tileK, dim * tileK);
    simdgroup_store(acc[0][2], tC + 0 * (dim * tileK * tileK) + 2 * tileK, dim * tileK);
    simdgroup_store(acc[0][3], tC + 0 * (dim * tileK * tileK) + 3 * tileK, dim * tileK);

    simdgroup_store(acc[1][0], tC + 1 * (dim * tileK * tileK) + 0 * tileK, dim * tileK);
    simdgroup_store(acc[1][1], tC + 1 * (dim * tileK * tileK) + 1 * tileK, dim * tileK);
    simdgroup_store(acc[1][2], tC + 1 * (dim * tileK * tileK) + 2 * tileK, dim * tileK);
    simdgroup_store(acc[1][3], tC + 1 * (dim * tileK * tileK) + 3 * tileK, dim * tileK);

    simdgroup_store(acc[2][0], tC + 2 * (dim * tileK * tileK) + 0 * tileK, dim * tileK);
    simdgroup_store(acc[2][1], tC + 2 * (dim * tileK * tileK) + 1 * tileK, dim * tileK);
    simdgroup_store(acc[2][2], tC + 2 * (dim * tileK * tileK) + 2 * tileK, dim * tileK);
    simdgroup_store(acc[2][3], tC + 2 * (dim * tileK * tileK) + 3 * tileK, dim * tileK);

    simdgroup_store(acc[3][0], tC + 3 * (dim * tileK * tileK) + 0 * tileK, dim * tileK);
    simdgroup_store(acc[3][1], tC + 3 * (dim * tileK * tileK) + 1 * tileK, dim * tileK);
    simdgroup_store(acc[3][2], tC + 3 * (dim * tileK * tileK) + 2 * tileK, dim * tileK);
    simdgroup_store(acc[3][3], tC + 3 * (dim * tileK * tileK) + 3 * tileK, dim * tileK);

    // Write back to global memory (scatter loop remains)
    uint gidyC = group_id.y * (dK * group_size.y) + lid.y * dK;
    uint gidxC = group_id.x * dK;
    uint tile_total_elements = thread_tile_size;  // 1024

    for (uint i = lid.x; i < tile_total_elements; i += group_size.x) {
        uint x = i % (dim * tileK);
        uint y = i / (dim * tileK);
        uint idyC = gidyC + y;
        uint idxC = gidxC + x;
        if (idyC < M && idxC < P) {
            C[idyC * P + idxC] = tC[y * dim * tileK + x];
        }
    }
}

kernel void sgemm_na_unrolled_dim2(
    const device float* A,
    const device float* B,
    device float* C,
    constant const uint64_t& M,
    constant const uint64_t& N,
    constant const uint64_t& P,
    threadgroup float* tA,
    threadgroup float* tB,
    threadgroup float* tC,
    uint2 gid [[thread_position_in_grid]],
    uint2 lid [[thread_position_in_threadgroup]],
    uint2 group_id [[threadgroup_position_in_grid]],
    uint2 group_size [[threads_per_threadgroup]]
)
{
    constexpr uint dim = 2;
    constexpr uint dK = dim * 8;              // 16
    constexpr uint tileK = 8;
    constexpr uint thread_tile_rows = dim;    // 2
    constexpr uint thread_tile_cols = dim;    // 2
    constexpr uint thread_tile_size = thread_tile_rows * thread_tile_cols * tileK * tileK; // 2*2*64=256

    uint lid_div8 = lid.x / 8;
    uint lid_mod8 = lid.x % 8;

    uint tgidyA = group_id.y * (dK * group_size.y) + lid.y * dK;
    uint tgidxB = group_id.x * dK;

    // Accumulators (2x2)
    simdgroup_float8x8 acc00, acc01, acc10, acc11;
    acc00 = simdgroup_float8x8(0);
    acc01 = simdgroup_float8x8(0);
    acc10 = simdgroup_float8x8(0);
    acc11 = simdgroup_float8x8(0);

    tA += lid.y * (dim * 8 * 8);
    tB += lid.y * (dim * 8 * 8);
    tC += lid.y * (dim * dim * 8 * 8);

    uint Nk = ((N + tileK - 1) / tileK) * tileK;

    for (uint k = 0; k < Nk; k += tileK) {
        // ----- d = 0 -----
        {
            uint didyA = 0 * tileK;
            uint didxB = 0 * tileK;

            uint rowA0 = tgidyA + 0*4 + didyA + lid_div8;
            uint colA0 = k + lid_mod8;
            tA[((0*4) + lid_div8) * (dim * tileK) + (0*tileK + lid_mod8)] = (rowA0 < M && colA0 < N) ? A[rowA0 * N + colA0] : 0.0f;
            uint rowA1 = tgidyA + 1*4 + didyA + lid_div8;
            uint colA1 = k + lid_mod8;
            tA[((1*4) + lid_div8) * (dim * tileK) + (0*tileK + lid_mod8)] = (rowA1 < M && colA1 < N) ? A[rowA1 * N + colA1] : 0.0f;

            uint rowB0 = 0*4 + k + lid_div8;
            uint colB0 = tgidxB + didxB + lid_mod8;
            tB[((0*4) + lid_div8) * (dim * tileK) + (0*tileK + lid_mod8)] = (rowB0 < N && colB0 < P) ? B[rowB0 * P + colB0] : 0.0f;
            uint rowB1 = 1*4 + k + lid_div8;
            uint colB1 = tgidxB + didxB + lid_mod8;
            tB[((1*4) + lid_div8) * (dim * tileK) + (0*tileK + lid_mod8)] = (rowB1 < N && colB1 < P) ? B[rowB1 * P + colB1] : 0.0f;
        }
        // ----- d = 1 -----
        {
            uint didyA = 1 * tileK;
            uint didxB = 1 * tileK;

            uint rowA0 = tgidyA + 0*4 + didyA + lid_div8;
            uint colA0 = k + lid_mod8;
            tA[((0*4) + lid_div8) * (dim * tileK) + (1*tileK + lid_mod8)] = (rowA0 < M && colA0 < N) ? A[rowA0 * N + colA0] : 0.0f;
            uint rowA1 = tgidyA + 1*4 + didyA + lid_div8;
            uint colA1 = k + lid_mod8;
            tA[((1*4) + lid_div8) * (dim * tileK) + (1*tileK + lid_mod8)] = (rowA1 < M && colA1 < N) ? A[rowA1 * N + colA1] : 0.0f;

            uint rowB0 = 0*4 + k + lid_div8;
            uint colB0 = tgidxB + didxB + lid_mod8;
            tB[((0*4) + lid_div8) * (dim * tileK) + (1*tileK + lid_mod8)] = (rowB0 < N && colB0 < P) ? B[rowB0 * P + colB0] : 0.0f;
            uint rowB1 = 1*4 + k + lid_div8;
            uint colB1 = tgidxB + didxB + lid_mod8;
            tB[((1*4) + lid_div8) * (dim * tileK) + (1*tileK + lid_mod8)] = (rowB1 < N && colB1 < P) ? B[rowB1 * P + colB1] : 0.0f;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Load A tiles (2 of them)
        simdgroup_float8x8 Am0, Am1;
        simdgroup_load(Am0, &tA[0 * tileK], dim * tileK);
        simdgroup_load(Am1, &tA[1 * tileK], dim * tileK);

        // Load B tiles (2 of them)
        simdgroup_float8x8 Bm0, Bm1;
        simdgroup_load(Bm0, &tB[0 * tileK], dim * tileK);
        simdgroup_load(Bm1, &tB[1 * tileK], dim * tileK);

        // Multiply-accumulate (2x2)
        simdgroup_multiply_accumulate(acc00, Am0, Bm0, acc00);
        simdgroup_multiply_accumulate(acc01, Am0, Bm1, acc01);
        simdgroup_multiply_accumulate(acc10, Am1, Bm0, acc10);
        simdgroup_multiply_accumulate(acc11, Am1, Bm1, acc11);

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Store to threadgroup memory
    simdgroup_store(acc00, tC + 0 * (dim * tileK * tileK) + 0 * tileK, dim * tileK);
    simdgroup_store(acc01, tC + 0 * (dim * tileK * tileK) + 1 * tileK, dim * tileK);
    simdgroup_store(acc10, tC + 1 * (dim * tileK * tileK) + 0 * tileK, dim * tileK);
    simdgroup_store(acc11, tC + 1 * (dim * tileK * tileK) + 1 * tileK, dim * tileK);

    // Write back to global memory
    uint gidyC = group_id.y * (dK * group_size.y) + lid.y * dK;
    uint gidxC = group_id.x * dK;
    uint tile_total_elements = thread_tile_size;

    for (uint i = lid.x; i < tile_total_elements; i += group_size.x) {
        uint x = i % (dim * tileK);
        uint y = i / (dim * tileK);
        uint idyC = gidyC + y;
        uint idxC = gidxC + x;
        if (idyC < M && idxC < P) {
            C[idyC * P + idxC] = tC[y * dim * tileK + x];
        }
    }
}

kernel void sgemm_na_unrolled_dim1(
    const device float* A,
    const device float* B,
    device float* C,
    constant const uint64_t& M,
    constant const uint64_t& N,
    constant const uint64_t& P,
    threadgroup float* tA,
    threadgroup float* tB,
    threadgroup float* tC,
    uint2 gid [[thread_position_in_grid]],
    uint2 lid [[thread_position_in_threadgroup]],
    uint2 group_id [[threadgroup_position_in_grid]],
    uint2 group_size [[threads_per_threadgroup]]
)
{
    constexpr uint dim = 1;
    constexpr uint dK = dim * 8;              // 8
    constexpr uint tileK = 8;
    constexpr uint thread_tile_rows = dim;    // 1
    constexpr uint thread_tile_cols = dim;    // 1
    constexpr uint thread_tile_size = thread_tile_rows * thread_tile_cols * tileK * tileK; // 1*1*64=64

    uint lid_div8 = lid.x / 8;
    uint lid_mod8 = lid.x % 8;

    uint tgidyA = group_id.y * (dK * group_size.y) + lid.y * dK;
    uint tgidxB = group_id.x * dK;

    // Single accumulator
    simdgroup_float8x8 acc;
    acc = simdgroup_float8x8(0);

    tA += lid.y * (dim * 8 * 8);
    tB += lid.y * (dim * 8 * 8);
    tC += lid.y * (dim * dim * 8 * 8);

    uint Nk = ((N + tileK - 1) / tileK) * tileK;

    for (uint k = 0; k < Nk; k += tileK) {
        // Only d = 0 (since dim=1)
        {
            uint didyA = 0 * tileK;
            uint didxB = 0 * tileK;

            uint rowA0 = tgidyA + 0*4 + didyA + lid_div8;
            uint colA0 = k + lid_mod8;
            tA[((0*4) + lid_div8) * (dim * tileK) + (0*tileK + lid_mod8)] = (rowA0 < M && colA0 < N) ? A[rowA0 * N + colA0] : 0.0f;
            uint rowA1 = tgidyA + 1*4 + didyA + lid_div8;
            uint colA1 = k + lid_mod8;
            tA[((1*4) + lid_div8) * (dim * tileK) + (0*tileK + lid_mod8)] = (rowA1 < M && colA1 < N) ? A[rowA1 * N + colA1] : 0.0f;

            uint rowB0 = 0*4 + k + lid_div8;
            uint colB0 = tgidxB + didxB + lid_mod8;
            tB[((0*4) + lid_div8) * (dim * tileK) + (0*tileK + lid_mod8)] = (rowB0 < N && colB0 < P) ? B[rowB0 * P + colB0] : 0.0f;
            uint rowB1 = 1*4 + k + lid_div8;
            uint colB1 = tgidxB + didxB + lid_mod8;
            tB[((1*4) + lid_div8) * (dim * tileK) + (0*tileK + lid_mod8)] = (rowB1 < N && colB1 < P) ? B[rowB1 * P + colB1] : 0.0f;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Load A tile (only one)
        simdgroup_float8x8 Am;
        simdgroup_load(Am, &tA[0 * tileK], dim * tileK);

        // Load B tile (only one)
        simdgroup_float8x8 Bm;
        simdgroup_load(Bm, &tB[0 * tileK], dim * tileK);

        // Multiply-accumulate
        simdgroup_multiply_accumulate(acc, Am, Bm, acc);

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Store result to threadgroup memory
    simdgroup_store(acc, tC + 0 * (dim * tileK * tileK) + 0 * tileK, dim * tileK);

    // Write back to global memory
    uint gidyC = group_id.y * (dK * group_size.y) + lid.y * dK;
    uint gidxC = group_id.x * dK;
    uint tile_total_elements = thread_tile_size;

    for (uint i = lid.x; i < tile_total_elements; i += group_size.x) {
        uint x = i % (dim * tileK);
        uint y = i / (dim * tileK);
        uint idyC = gidyC + y;
        uint idxC = gidxC + x;
        if (idyC < M && idxC < P) {
            C[idyC * P + idxC] = tC[y * dim * tileK + x];
        }
    }
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

kernel void dot(
                const device float4* X,
                const device float4* Y,
                device atomic_float* output,
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
  
  if (lid == 0) {
    atomic_fetch_add_explicit(output, shared[0], memory_order_relaxed);
  }
}

kernel void sum(
                const device float4* X,
                constant const float& y,
                device atomic_float* output,
                constant const uint64_t& N,
                threadgroup float* shared,
                uint gid [[thread_position_in_grid]],
                uint lid [[thread_position_in_threadgroup]],
                uint group_id [[threadgroup_position_in_grid]],
                uint group_size [[threads_per_threadgroup]],
                uint groups [[threadgroups_per_grid]]
                )
{
  uint64_t grp = (uint64_t)group_id * (uint64_t)group_size;
  // sim_sum???
  shared[lid] = dot(X[grp + lid], y);
  X += groups * group_size;
  shared[group_size + lid] = dot(X[grp + lid], y);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  
  for (uint offset = group_size; offset > 0; offset = offset >> 1) {
    if (lid < offset) {
      shared[lid] += shared[lid + offset];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  
  if (lid == 0) {
    atomic_fetch_add_explicit(output, shared[0], memory_order_relaxed);
  }
}

kernel void axpby2(
                  const device float* X, const device float* Y, device float* out,
                  constant const uint& N,
                  constant const float& fX,
                  constant const float& fY,
                  constant const float& A,
                  uint gid [[thread_position_in_grid]],
                  uint threads [[threads_per_grid]]
                  )
{
    for (uint i = gid; i < N; i += threads) {
        out[i] = X[i] * fX + Y[i] * fY + A;
    }
}

kernel void axpby1(
                  const device float* X, device float* out,
                  constant const uint& N,
                  constant const float& fX,
                  constant const float& A,
                  uint gid [[thread_position_in_grid]],
                  uint threads [[threads_per_grid]]
                  )
{
    for (uint i = gid; i < N; i += threads) {
        out[i] = X[i] * fX + A;
    }
}

kernel void addcmul(
                  const device float* X, const device float* Y, device float* out,
                  constant const uint& N,
                  constant const float& a,
                  constant const float& b,
                  uint gid [[thread_position_in_grid]],
                  uint threads [[threads_per_grid]]
                  )
{
    for (uint i = gid; i < N; i += threads) {
        out[i] = X[i] * Y[i] * a + b;
    }
}

kernel void addcdiv(
                  const device float* X, const device float* Y, device float* out,
                  constant const uint& N,
                  constant const float& a,
                  constant const float& b,
                  uint gid [[thread_position_in_grid]],
                  uint threads [[threads_per_grid]]
                  )
{
    for (uint i = gid; i < N; i += threads) {
        out[i] = X[i] / Y[i] * a + b;
    }
}

kernel void axpy(
                  const device float* X, const device float* Y, device float* out,
                  constant const uint& N,
                  constant const float& a,
                  uint gid [[thread_position_in_grid]],
                  uint threads [[threads_per_grid]]
                  )
{
    for (uint i = gid; i < N; i += threads) {
        out[i] = X[i] * a + Y[i];
    }
}

kernel void axpby2dBcol(
                  const device float* X, const device float* Y, device float* out,
                  constant const uint& N,
                  constant const float& fX,
                  constant const float& fY,
                  constant const float& A,
                  constant const uint& strideB,
                  uint gid [[thread_position_in_grid]],
                  uint threads [[threads_per_grid]]
                  )
{
    for (uint i = gid; i < N; i += threads) {
        out[i] = X[i] * fX + Y[i / strideB] * fY + A;
    }
}

kernel void sum_dim0(
                  const device float* X,
                  device float* out,
                  constant const uint& Nrows,
                  constant const uint& Ncols,
                  constant const uint& stride,
                  constant const float& scale,
                  uint gid [[thread_position_in_grid]]
                  )
{
    if (gid >= Ncols) return;
    float acc = 0;
    for (uint i = 0; i < Nrows; i++) acc += X[i * stride + gid];
    out[gid] = acc * scale;
}

kernel void sum_dim1(
                  const device float* X,
                  device float* out,
                  constant const uint& Nrows,
                  constant const uint& Ncols,
                  constant const uint& stride,
                  constant const float& scale,
                  uint gid [[thread_position_in_grid]]
                  )
{
    if (gid >= Nrows) return;
    float acc = 0;
    for (uint j = 0; j < Ncols; j++) acc += X[gid * stride + j];
    out[gid] = acc * scale;
}

kernel void transpose(
                  const device float* X,
                  device float* out,
                  constant const uint& M,
                  constant const uint& N,
                  uint gid [[thread_position_in_grid]]
                  )
{
    uint y = gid / N;
    uint x = gid % N;
    if (y >= M) return;
    out[x * M + y] = X[y * N + x];
}

