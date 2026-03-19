#pragma once

#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include <cassert>

#ifdef GPU
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

namespace gpu {
  auto device = MTLCreateSystemDefaultDevice();
  auto queue = [device newCommandQueue];
  auto lib = [device newDefaultLibrary];

  namespace compute {
    inline void dispatch1d(id<MTLComputeCommandEncoder> encoder,
        id<MTLComputePipelineState> kernel,
        uint64_t size)
    {
      uint64_t simdgroupSize = kernel.threadExecutionWidth;
      uint64_t simdgroupsInSize = (size + simdgroupSize - 1) / simdgroupSize;
      auto threadsPerThreadgroup = std::min(simdgroupSize * simdgroupsInSize, (uint64_t)encoder.device.maxThreadsPerThreadgroup.width);
      [encoder setComputePipelineState:kernel];
      [encoder dispatchThreads:MTLSizeMake(size, 1, 1) threadsPerThreadgroup:MTLSizeMake(threadsPerThreadgroup, 1, 1)];
    }
  }
}
#endif

namespace nn {
  namespace gemm {
    template<int64_t BLOCK_X=4, int64_t BLOCK_Y = 4>
      void cpu(const float* A, const float* B, float* C, 
          int64_t M, int64_t N, int64_t P, 
          bool transposeB = true);

    template<int BLOCK>
      void transpose(const float* m, float* mT, int64_t M, int64_t N);

#ifdef GPU
    void gpu(id<MTLCommandBuffer> cmd, id<MTLBuffer> A, id<MTLBuffer> B, id<MTLBuffer> C, 
        uint64_t m, uint64_t n, uint64_t p);
#endif
  }

  namespace gemv {
    void cpu(const float* mat, const float* vec, float* output, uint64_t M, uint64_t N);
#ifdef GPU
  void gpu(id<MTLCommandBuffer> cmd, id<MTLBuffer> mat, id<MTLBuffer> vec, id<MTLBuffer> out,
      uint64_t m, uint64_t n);
#endif
  }

  namespace utils {
    template <typename xs_t>
      static typename xs_t::value_type area(const xs_t& xs) {
        typename xs_t::value_type out = 1;
        for (auto x : xs) {
          out *= x;
        }
        return out;
      }
  }

  namespace tensor {
    template<typename val_t>
    struct data_t {
      std::vector<int64_t> dims;
      val_t* xs;

      data_t(std::initializer_list<int64_t> dims) : dims(dims), xs(new val_t[utils::area(dims)]) {}

      ~data_t() {
        delete[] xs;
      }

      static data_t random_int(std::initializer_list<int64_t> dims) {
        data_t out{dims};
        auto area = utils::area(out.dims);
        for (auto i = 0; i < area; i++) {
          out.xs[i] = (float)(rand() % 100);
        }
        return out;
      }

      static data_t zero(std::initializer_list<int64_t> dims) {
        data_t out{dims};
        memset(out.xs, 0, utils::area(out.dims) * sizeof(out.xs[0]));
        return out;
      }

      int64_t size() const {
        return utils::area(dims);
      }
    };

    void matmul(const data_t<float>& A, const data_t<float>& B, data_t<float>& C);
  }
}

#ifdef NN_IMPL

namespace nn::gemm {
  template<int BLOCK>
    void transpose(const float* m, float* mT, int64_t M, int64_t N)
    {
      // transpose in blocks
      for (int y = 0; y < M; y += BLOCK) {
        for (int x = 0; x < N; x += BLOCK) {
          for (int yb = 0; yb < BLOCK; yb++) {
            for (int xb = 0; xb < BLOCK; xb++) {
              mT[y + x * M + yb * M + xb] = m[y * N + x + yb * N + xb];
            }
          }
        }
      }
    }

  template<int64_t BLOCK_X, int64_t BLOCK_Y>
    void cpu(const float* A, const float* B, float* C, 
        int64_t M, int64_t N, int64_t P, 
        bool transposeB)
  {
    const float* Bt;
    std::unique_ptr<const float> Btptr;
    if (transposeB) {
      float* tmpBt = new float[N * P];
      transpose<1>(B, tmpBt, N, P);
      Btptr = std::unique_ptr<const float>(tmpBt);
      Bt = Btptr.get();
    } else {
      Bt = B;
    }

    for (int y = 0; y < M; y += BLOCK_Y) {
      for (int x = 0; x < P; x += BLOCK_X) {
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
            C[y * P + yb * P + x + xb] = block_C[yb * BLOCK_X + xb];
          }
        }
      }
    }
  }

#ifdef GPU
  void gpu(id<MTLCommandBuffer> cmd, id<MTLBuffer> A, id<MTLBuffer> B, id<MTLBuffer> C, uint64_t m, uint64_t n, uint64_t p)
  {
    constexpr auto dim = 4;
    constexpr auto block_size = dim * 8;
    constexpr auto block_size_m = dim * 8 * 2;

    assert(m >= block_size_m && "M must be at least 64");
    assert(n >= 8 && "N must be at least 8");
    assert(p >= block_size && "P must be at least 32");
    assert(m % block_size_m == 0 && "M must be divisible by 64");
    assert(n % 8 == 0 && "N must be divisible by 8");
    assert(p % block_size == 0 && "P must be divisible by 32");

    static id<MTLComputePipelineState> kernel;
    static dispatch_once_t onceToken;
    dispatch_once(&onceToken, ^{
        auto kernelFunc = [gpu::lib newFunctionWithName:@"sgemm_32x32_unrolled"];
        kernel = [gpu::device newComputePipelineStateWithFunction:kernelFunc error:nil];
        });
    if (!kernel) {
      NSLog(@"got error during pipeline creation");
      return;
    }

    MTLSize tgroupSize;
    tgroupSize.width = 32;
    tgroupSize.height = 2;
    tgroupSize.depth = 1;
    auto encoder = [cmd computeCommandEncoder];

    [encoder setBuffer:A offset:0 atIndex:0];
    [encoder setBuffer:B offset:0 atIndex:1];
    [encoder setBuffer:C offset:0 atIndex:2];
    [encoder setBytes:(void*)&m length:sizeof(m) atIndex:3];
    [encoder setBytes:(void*)&n length:sizeof(n) atIndex:4];
    [encoder setBytes:(void*)&p length:sizeof(p) atIndex:5];
    [encoder setComputePipelineState:kernel];
    [encoder dispatchThreadgroups:MTLSizeMake(p / block_size, m / block_size_m, 1) threadsPerThreadgroup:tgroupSize];

    [encoder endEncoding];
  }
#endif

}
namespace nn::gemv {
    void cpu(const float* mat, const float* vec, float* output, uint64_t M, uint64_t N)
    {
      for (uint32_t y = 0; y < M; y++) {
        output[y] = 0;
        for (uint32_t x = 0; x < N; x++) {
          output[y] += mat[y * N + x] * vec[x];
        }
      }
    }
#ifdef GPU
  void gpu(id<MTLCommandBuffer> cmd, id<MTLBuffer> mat, id<MTLBuffer> vec, id<MTLBuffer> output,
      uint64_t m, uint64_t n)
  {
    assert(m >= 2 && "M (height) must be at least 2");
    assert(n >= 64 && "N (width) must be at least 64");
    assert(m % 2 == 0 && "M (height) must be divisible by 2");
    assert(n % 64 == 0 && "N (width) must be divisible by 64");

    static id<MTLComputePipelineState> kernel;
    static dispatch_once_t onceToken;
    dispatch_once(&onceToken, ^{
        auto kernelFunc = [gpu::lib newFunctionWithName:@"sgemv"];
        kernel = [gpu::device newComputePipelineStateWithFunction:kernelFunc error:nil];
    });
    if (!kernel) {
        NSLog(@"got error during pipeline creation");
        return;
    }

    MTLSize tgroupSize;
    tgroupSize.width = 32;
    tgroupSize.height = 2;
    tgroupSize.depth = 1;

    uint32_t H = (uint32_t)m;
    uint32_t W = (uint32_t)n;
    auto encoder = [cmd computeCommandEncoder];
    [encoder setBuffer:mat offset:0 atIndex:0];
    [encoder setBuffer:vec offset:0 atIndex:1];
    [encoder setBuffer:output offset:0 atIndex:2];
    [encoder setBytes:(void*)&H length:sizeof(H) atIndex:3];
    [encoder setBytes:(void*)&W length:sizeof(W) atIndex:4];
//    [encoder setThreadgroupMemoryLength:tgroupSize.width * sizeof(float) atIndex:0];
    [encoder setComputePipelineState:kernel];

    [encoder dispatchThreadgroups:MTLSizeMake(1, H / tgroupSize.height, 1) threadsPerThreadgroup:tgroupSize];
    [encoder endEncoding];
  }
#endif
}

namespace nn::tensor {
  void matmul(const data_t<float>& A, const data_t<float>& B, data_t<float>& C)
  {
    assert(A.dims.size() == 2);
    assert(B.dims.size() == 2);
    assert(C.dims.size() == 2);

    nn::gemm::cpu(A.xs, B.xs, C.xs, C.dims[0], A.dims[1], C.dims[1]);
  }
}

namespace nn {
} // namespace nn

#endif
