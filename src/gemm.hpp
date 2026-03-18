#pragma once
// clang++ -ffast-math -O3 -std=c++20 gemm.mm -framework Accelerate

#include <cstdint>
#include <memory>
#include "gpu.hpp"

namespace gemm {
  template<int64_t N, int64_t BLOCK_X=4, int64_t BLOCK_Y = 4>
  void cpu(const float* A, const float* B, float* C, bool transposeB = true);

  template<int length, int block>
  void transpose(const float* m, float* mT);

  void gpu(id<MTLCommandBuffer> cmd, id<MTLBuffer> A, id<MTLBuffer> B, id<MTLBuffer> C, 
            uint64_t m, uint64_t n, uint64_t p);
}

template<int length, int block>
void gemm::transpose(const float* m, float* mT)
{
  // transpose in blocks
  for (int y = 0; y < length; y += block) {
    for (int x = 0; x < length; x += block) {
      for (int yb = 0; yb < block; yb++) {
        for (int xb = 0; xb < block; xb++) {
          mT[y + x * length + yb * length + xb] = m[y * length + x + yb * length + xb];
        }
      }
    }
  }
}

template<int64_t N, int64_t BLOCK_X, int64_t BLOCK_Y>
void gemm::cpu(const float* A, const float* B, float* C, bool transposeB)
{
  const float* Bt;
  std::unique_ptr<const float> Btptr;
  if (transposeB) {
    float* tmpBt = new float[N * N];
    transpose<N, 1>(B, tmpBt);
    Btptr = std::unique_ptr<const float>(tmpBt);
    Bt = Btptr.get();
  } else {
    Bt = B;
  }

  for (int y = 0; y < N; y += BLOCK_Y) {
    for (int x = 0; x < N; x += BLOCK_X) {
      // y and x is the block
      // to compute the whole block we need to go through BLOCK_Y A rows and BLOCK_X B cols

      // block_C is in cache since its relatively small and local
      float block_C[BLOCK_X * BLOCK_Y] = {0};
      const float* block_A = &A[y * N];
      const float* block_B = &Bt[x * N];

      // after this loop (kernel) block_C is complete
      // iterating over ks, this is outer loop, so block_C contain partial dot products
      for (int k = 0; k < N; k += 1) {
        for (int yb = 0; yb < BLOCK_Y; yb++) {
          // A[y * N + yb * N + k]... is in cache now, but yb is strided by N.
          // is that a problem for L1 cache?
          // if BLOCK_Y is 4, then 32KB of A is used for this K loop
          float tA = block_A[yb * N + k];
          for (int xb = 0; xb < BLOCK_X; xb++) {
            // B[x * N + xb * N + k] we go through xb with stride N
            // so whole B for 0<=xb<4 is in cache?
            block_C[yb * BLOCK_X + xb] += tA * block_B[xb * N + k];     
          }
        }
      }

      for (int yb = 0; yb < BLOCK_Y; yb++) {
        for (int xb = 0; xb < BLOCK_X; xb++) {
          C[y * N + yb * N + x + xb] = block_C[yb * BLOCK_X + xb];
        }
      }
    }
  }
}
void gemm::gpu(id<MTLCommandBuffer> cmd, id<MTLBuffer> A, id<MTLBuffer> B, id<MTLBuffer> C,
    uint64_t m, uint64_t n, uint64_t p)
{
    assert(m == n);
    assert(n == p);

    static id<MTLComputePipelineState> kernel;
    static dispatch_once_t onceToken;
    dispatch_once(&onceToken, ^{
        auto kernelFunc = [gpu::lib newFunctionWithName:@"sgemm_32x32"];
        kernel = [gpu::device newComputePipelineStateWithFunction:kernelFunc error:nil];
        });
    if (!kernel) {
      NSLog(@"got error during pipeline creation");
      return;
    }

    constexpr auto dim = 4;
    MTLSize tgroupSize;
    tgroupSize.width = 32;
    tgroupSize.height = 2;
    tgroupSize.depth = 1;
    auto encoder = [cmd computeCommandEncoder];

    [encoder setBuffer:A offset:0 atIndex:0];
    [encoder setBuffer:B offset:0 atIndex:1];
    [encoder setBuffer:C offset:0 atIndex:2];
    [encoder setBytes:(void*)&n length:sizeof(n) atIndex:3];
    [encoder setComputePipelineState:kernel];
    [encoder dispatchThreadgroups:MTLSizeMake(n / (dim * 8), m / (dim * 8 * 2), 1) threadsPerThreadgroup:tgroupSize];

    [encoder endEncoding];
}

