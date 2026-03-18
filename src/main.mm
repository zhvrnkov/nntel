#include <iostream>
#include <Metal/Metal.h>
#include "gpu.hpp"
#include "gemm.hpp"

template<int64_t N>
void naive_gemm(const float* A, const float* B, float* C)
{
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      for (int k = 0; k < N; k++) {
        C[i * N + j] += A[i * N + k] * B[k * N + j];
      }
    }
  }
}

int main()
{
  constexpr auto N = 128;
  std::array<float, N*N> A;
  std::array<float, N*N> B;
  std::array<float, N*N> naiveC;
  std::array<float, N*N> cpuC;

  // cpu
  for (int i = 0; i < N * N; i++) {
    A[i] = (float)(rand() % 100);
    B[i] = (float)(rand() % 100);
  }

  naive_gemm<N>(A.data(), B.data(), naiveC.data());
  gemm::cpu<N>(A.data(), B.data(), cpuC.data());

  for (uint i = 0; i < N * N; i++) {
    if (cpuC[i] != naiveC[i]) {
      printf("missmatch at %d : %f != %f\n", i, cpuC[i], naiveC[i]);
      assert(false);
    }
  }

  // gpu
  auto buffA = [gpu::device newBufferWithBytes:A.data() length:A.size() * sizeof(A[0]) options:MTLResourceStorageModeShared];
  auto buffB = [gpu::device newBufferWithBytes:B.data() length:B.size() * sizeof(B[0]) options:MTLResourceStorageModeShared];
  auto buffC = [gpu::device newBufferWithLength:N * N * sizeof(float) options:MTLResourceStorageModeShared];
  memset(buffC.contents, 0, buffC.length);

  auto cmd = [gpu::queue commandBuffer];
  gemm::gpu(cmd, buffA, buffB, buffC, N, N, N);

  [cmd commit];
  [cmd waitUntilCompleted];

  for (uint i = 0; i < N * N; i++) {
    auto x = ((float*)buffC.contents)[i];
    if (x != naiveC[i]) {
      printf("missmatch at %d : %f != %f\n", i, x, naiveC[i]);
      assert(false);
    }
  }

  std::cout << "hello nntel" << std::endl;
}
