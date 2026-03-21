#pragma once

#include <MacTypes.h>
#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include <cassert>

#define NN_IMPL

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

namespace nn {
  namespace cpu {
    dispatch_queue_t queue = dispatch_queue_create("nntel", NULL);

    template<int64_t BLOCK_X=4, int64_t BLOCK_Y = 4>
    void gemm(const float* A, const float* B, float* C, 
          int64_t M, int64_t N, int64_t P, 
          bool transposeB = true);

    void gemv(const float* mat, const float* vec, float* output, uint64_t M, uint64_t N);

    template<int BLOCK>
    void transpose(const float* m, float* mT, int64_t M, int64_t N);

    void dot(const float* x, const float* y, float* output, int64_t N);

    void add(const float* x, const float* y, float* output, int64_t N);
  }

  namespace gpu {

    void gemm(id<MTLCommandBuffer> cmd, id<MTLBuffer> A, id<MTLBuffer> B, id<MTLBuffer> C, 
        uint64_t m, uint64_t n, uint64_t p);
    void gemv(id<MTLCommandBuffer> cmd, id<MTLBuffer> mat, id<MTLBuffer> vec, id<MTLBuffer> out,
        uint64_t m, uint64_t n);
    void dot(id<MTLCommandBuffer> cmd, id<MTLBuffer> X, id<MTLBuffer> Y, id<MTLBuffer> output);

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

  namespace stream {
    struct ctx_t {
      ctx_t() 
        : cmd([gpu::queue commandBuffer])
        , event([gpu::device newSharedEvent])
        , listener([[MTLSharedEventListener alloc] initWithDispatchQueue:cpu::queue])
        , last_id(0) {}

      id<MTLCommandBuffer> cmd;
      id<MTLSharedEvent> event;
      MTLSharedEventListener* listener;
      uint64_t last_id = 0;

      void synchronize()
      {
        [cmd commit];
        [event waitUntilSignaledValue:last_id timeoutMS:-1];
        // std::cout << "sync w8=" << event.signaledValue << " sig=" << last_id << std::endl;

        cmd = [gpu::queue commandBuffer];
        event = [gpu::device newSharedEvent];
        listener = [[MTLSharedEventListener alloc] initWithDispatchQueue:cpu::queue];
        last_id = 0;
      }

      void cpu_dispatch(void (^block)())
      {
        // printf("cpu_dispatch w8=%llu curr=%llu sig=%llu\n", last_id, event.signaledValue, last_id + 1);
        [event notifyListener:listener atValue:last_id block:^(id<MTLSharedEvent> _Nonnull _event, uint64_t _value) {
          block();
          event.signaledValue = _value + 1;
        }];
        last_id += 1;
      }

      void gpu_dispatch(void (^block)())
      {
        // printf("gpu_dispatch w8=%llu curr=%llu sig=%llu\n", last_id, event.signaledValue, last_id + 1);
        [cmd encodeWaitForEvent:event value:last_id];
        block();
        [cmd encodeSignalEvent:event value:last_id + 1];
        last_id += 1;
      }
    };

    ctx_t global{};
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
    struct data_t {
      std::vector<int64_t> dims;
      id<MTLBuffer> xs;

      data_t(std::initializer_list<int64_t> dims, id<MTLBuffer> xs) : dims(dims), xs(xs) {}
      data_t(std::vector<int64_t> dims, id<MTLBuffer> xs) : dims(dims), xs(xs) {}

      static data_t random_int(std::initializer_list<int64_t> dims) {
        auto size = utils::area(dims) * sizeof(float);
        auto storage = [gpu::device newBufferWithLength:size options:MTLResourceStorageModeShared];

        data_t out{dims, storage};
        auto area = utils::area(out.dims);
        for (auto i = 0; i < area; i++) {
          out.data()[i] = (float)(rand() % 100);
        }
        return out;
      }

      static data_t zero(std::initializer_list<int64_t> dims) {
        auto size = utils::area(dims) * sizeof(float);
        auto storage = [gpu::device newBufferWithLength:size options:MTLResourceStorageModeShared];

        data_t out{dims, storage};
        memset(out.data(), 0, utils::area(out.dims) * sizeof(out.data()[0]));
        return out;
      }

      static data_t fill(std::initializer_list<int64_t> dims, float x) {
        auto size = utils::area(dims) * sizeof(float);
        auto storage = [gpu::device newBufferWithLength:size options:MTLResourceStorageModeShared];

        data_t out{dims, storage};
        for (int64_t i = 0; i < out.size(); i++) {
          out.data()[i] = x;
        }
        return out;
      }

      float* data() const {
        return (float*)[xs contents];
      }

      int64_t size() const {
        return utils::area(dims);
      }
    };

    enum class device_type {
      cpu,
      gpu
    };
    void mul(const data_t& A, const data_t& B, data_t& C, device_type dev=device_type::cpu);
    void add(const data_t& A, const data_t& B, data_t& C, device_type dev=device_type::cpu);
  }
}






























































#ifdef NN_IMPL

namespace nn::tensor {
  void cpu_mul(const data_t& A, const data_t& B, data_t& C);
  void gpu_mul(const data_t& A, const data_t& B, data_t& C);

  void cpu_add(const data_t& A, const data_t& B, data_t& C);
  void gpu_add(const data_t& A, const data_t& B, data_t& C);


  void mul(const data_t& A, const data_t& B, data_t& C, device_type dev)
  {
    if (dev == device_type::cpu) {
      nn::stream::global.cpu_dispatch(^() {
          cpu_mul(A, B, C);
      });
    }
    else if (dev == device_type::gpu) {
      nn::stream::global.gpu_dispatch(^() {
          gpu_mul(A, B, C);
      });
    }
  }

  void add(const data_t& A, const data_t& B, data_t& C, device_type dev)
  {
    if (dev == device_type::cpu) {
      nn::stream::global.cpu_dispatch(^() {
          cpu_add(A, B, C);
          });
    }
    else if (dev == device_type::gpu) {
      nn::stream::global.gpu_dispatch(^() {
          gpu_add(A, B, C);
          });
    }
  }

  void cpu_add(const data_t& A, const data_t& B, data_t& C)
  {
    if (A.dims.size() == B.dims.size()) {
      assert(A.dims.size() == C.dims.size());
      for (int64_t i = 0; i < A.size(); i++) {
        C.data()[i] = A.data()[i] + B.data()[i];
      }
    } else if (B.dims.size() == 1) {
      assert(A.dims.size() == C.dims.size());
      assert(B.dims[0] == 1);
      for (int64_t i = 0; i < A.size(); i++) {
        C.data()[i] = A.data()[i] + *B.data();
      }
    } else if (A.dims.size() == 1) {
      assert(B.dims.size() == C.dims.size());
      assert(A.dims[0] == 1);
      for (int64_t i = 0; i < A.size(); i++) {
        C.data()[i] = B.data()[i] + *A.data();
      }
    } else {
    std::cout << B.dims.size() << std::endl;
      assert(false);
    }
  }

  void gpu_add(const data_t& A, const data_t& B, data_t& C)
  {
    printf("no gpu add\n");
    assert(false);
  }

  void cpu_mul(const data_t& A, const data_t& B, data_t& C)
  {
    if (A.dims.size() == 2 && B.dims.size() == 2) {
      assert(C.dims.size() == 2);
      assert(A.dims[0] == C.dims[0]);
      assert(A.dims[1] == B.dims[0]);
      assert(B.dims[0] == C.dims[1]);
      nn::cpu::gemm(A.data(), B.data(), C.data(), C.dims[0], A.dims[1], C.dims[1]);
    } else if (A.dims.size() == 1 && B.dims.size() == 1) {
      assert(C.dims.size() == 1);
      assert(A.dims[0] == B.dims[0]);
      assert(1 == C.dims[0]);
      nn::cpu::dot(A.data(), B.data(), C.data(), A.dims[0]);
    } else if (A.dims.size() == 2 && B.dims.size() == 1) {
      assert(C.dims.size() == 1);
      assert(C.dims[0] == A.dims[0]);
      assert(B.dims[0] == A.dims[1]);
      nn::cpu::gemv(A.data(), B.data(), C.data(), A.dims[0], A.dims[1]);
    } else {
      assert(false);
    }
  }

  void gpu_mul(const data_t& A, const data_t& B, data_t& C)
  {
    if (A.dims.size() == 2 && B.dims.size() == 2) {
      assert(C.dims.size() == 2);
      assert(A.dims[0] == C.dims[0]);
      assert(A.dims[1] == B.dims[0]);
      assert(B.dims[0] == C.dims[1]);
      nn::gpu::gemm(stream::global.cmd, A.xs, B.xs, C.xs, C.dims[0], A.dims[1], C.dims[1]);
    } else if (A.dims.size() == 1 && B.dims.size() == 1) {
      assert(C.dims.size() == 1);
      assert(A.dims[0] == B.dims[0]);
      assert(1 == C.dims[0]);
      nn::gpu::dot(stream::global.cmd, A.xs, B.xs, C.xs);
    } else if (A.dims.size() == 2 && B.dims.size() == 1) {
      assert(C.dims.size() == 1);
      assert(C.dims[0] == A.dims[0]);
      assert(B.dims[0] == A.dims[1]);
      nn::gpu::gemv(stream::global.cmd, A.xs, B.xs, C.xs, A.dims[0], A.dims[1]);
    } else {
      assert(false);
    }
  }
}






























































namespace nn::cpu {
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
    void gemm(const float* A, const float* B, float* C, 
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

  void gemv(const float* mat, const float* vec, float* output, uint64_t M, uint64_t N)
  {
    for (uint32_t y = 0; y < M; y++) {
      output[y] = 0;
      for (uint32_t x = 0; x < N; x++) {
        output[y] += mat[y * N + x] * vec[x];
      }
    }
  }

  void dot(const float* x, const float* y, float* output, int64_t N)
  {
    output = 0;
    for (int64_t i = 0; i < N; i++) {
      *output += x[i] * y[i];
    }
  }

  void add(const float* x, const float* y, float* output, int64_t N)
  {
    for (int64_t i = 0; i < N; i++) {
      output[i] = x[i] + y[i];
    }
  }
}

namespace nn::gpu {
  void gemm(id<MTLCommandBuffer> cmd, id<MTLBuffer> A, id<MTLBuffer> B, id<MTLBuffer> C, uint64_t m, uint64_t n, uint64_t p)
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
        auto kernelFunc = [lib newFunctionWithName:@"sgemm_32x32_unrolled"];
        kernel = [device newComputePipelineStateWithFunction:kernelFunc error:nil];
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

  void gemv(id<MTLCommandBuffer> cmd, id<MTLBuffer> mat, id<MTLBuffer> vec, id<MTLBuffer> output,
      uint64_t m, uint64_t n)
  {
    assert(m >= 2 && "M (height) must be at least 2");
    assert(n >= 64 && "N (width) must be at least 64");
    assert(m % 2 == 0 && "M (height) must be divisible by 2");
    assert(n % 64 == 0 && "N (width) must be divisible by 64");

    static id<MTLComputePipelineState> kernel;
    static dispatch_once_t onceToken;
    dispatch_once(&onceToken, ^{
        auto kernelFunc = [lib newFunctionWithName:@"sgemv"];
        kernel = [device newComputePipelineStateWithFunction:kernelFunc error:nil];
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

  void dot(id<MTLCommandBuffer> cmd, id<MTLBuffer> x, id<MTLBuffer> y, id<MTLBuffer> output)
  {
    assert(x.length == y.length);
    const uint64_t N = x.length / sizeof(float);

    static id<MTLComputePipelineState> kernel0;
    static id<MTLComputePipelineState> kernel1;
    static dispatch_once_t onceToken;
    dispatch_once(&onceToken, ^{
        auto kernel0Func = [lib newFunctionWithName:@"dot_reduce0"];
        auto kernel1Func = [lib newFunctionWithName:@"dot_reduce1"];
        kernel0 = [device newComputePipelineStateWithFunction:kernel0Func error:nil];
        kernel1 = [device newComputePipelineStateWithFunction:kernel1Func error:nil];
        });
    if (!kernel0 || !kernel1) {
      NSLog(@"got error during pipeline creation");
      return;
    }

    auto threadsPerThreadgroup = MTLSizeMake(1024, 1, 1);
    auto threadgroupMemFloats = 1024 * 2;
    auto floatsPerThreadgroup = threadgroupMemFloats * 4;
    auto threadgroupsWidth = N / floatsPerThreadgroup;
    auto threadgroups = MTLSizeMake(threadgroupsWidth, 1, 1);
    id<MTLBuffer> interm = [device newBufferWithLength:threadgroups.width * sizeof(float) options:MTLResourceStorageModePrivate];

    auto encoder = [cmd computeCommandEncoder];
    [encoder setBuffer:x offset:0 atIndex:0];
    [encoder setBuffer:y offset:0 atIndex:1];
    [encoder setBuffer:interm offset:0 atIndex:2];
    [encoder setBytes:(void*)&N length:sizeof(N) atIndex:3];
    [encoder setThreadgroupMemoryLength:threadgroupMemFloats * sizeof(float) atIndex:0];
    [encoder setComputePipelineState:kernel0];
    [encoder dispatchThreadgroups:threadgroups threadsPerThreadgroup:threadsPerThreadgroup];

    [encoder setBuffer:interm offset:0 atIndex:0];
    [encoder setBuffer:output offset:0 atIndex:1];
    [encoder setBytes:(void*)&threadgroupsWidth length:sizeof(threadgroupsWidth) atIndex:2];
    [encoder setThreadgroupMemoryLength:threadgroupsWidth * sizeof(float) atIndex:0];
    [encoder setComputePipelineState:kernel1];
    [encoder dispatchThreads:MTLSizeMake(threadgroupsWidth, 1, 1) threadsPerThreadgroup:MTLSizeMake(threadgroupsWidth, 1, 1)];

    [encoder endEncoding];
  }
}

#endif
