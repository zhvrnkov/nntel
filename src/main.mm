#include <array>
#include <iostream>
#include <cassert>

#define GPU
#define NN_IMPL
#include "nn.hpp"

namespace test {
struct TestSample {
    std::vector<float> input;
    std::vector<float> output;
    
    TestSample(std::vector<float> input, std::vector<float> output) : input(input), output(output) {}
};
using TestData = std::vector<TestSample>;

}

// mat of size (rows x cols)
// A of size (M x N)
// B of size (N x P)
// C of size (M x P)
void naive_gemm(const float* A, const float* B, float* C, int64_t M, int64_t N, int64_t P)
{
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < P; j++) {
      for (int k = 0; k < N; k++) {
        C[i * P + j] += A[i * N + k] * B[k * P + j];
      }
    }
  }
}

int main0()
{
  nn::tensor::data_t<float> src{{3, 4}};
  for (int y = 0; y < src.dims[0]; y++) {
    for (int x = 0; x < src.dims[1]; x++) {
      src.xs[y * src.dims[1] + x] = y;
    }
  }

  nn::tensor::data_t<float> dst{{4, 3}};
  nn::gemm::transpose<1>(src.xs, dst.xs, src.dims[0], src.dims[1]);

  for (int y = 0; y < src.dims[0]; y++) {
    for (int x = 0; x < src.dims[1]; x++) {
      std::cout << src.xs[y * src.dims[1] + x] << " ";
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;

  for (int y = 0; y < dst.dims[0]; y++) {
    for (int x = 0; x < dst.dims[1]; x++) {
      std::cout << dst.xs[y * dst.dims[1] + x] << " ";
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;

  return 0;
}

int main()
{
  constexpr auto M = 128;
  constexpr auto N = 128;
  constexpr auto P = 128;
  auto A = nn::tensor::data_t<float>::random_int({M, N});
  auto B = nn::tensor::data_t<float>::random_int({N, P});
  auto naiveC = nn::tensor::data_t<float>::zero({M, P});
  auto cpuC = nn::tensor::data_t<float>::zero({M, P});

  naive_gemm(A.xs, B.xs, naiveC.xs, naiveC.dims[0], A.dims[1], naiveC.dims[1]);
  nn::tensor::matmul(A, B, cpuC);

  for (uint i = 0; i < M * P; i++) {
    if (cpuC.xs[i] != naiveC.xs[i]) {
      printf("missmatch at %d : %f != %f\n", i, cpuC.xs[i], naiveC.xs[i]);
      assert(false);
    }
  }

  // gpu
  auto buffA = [gpu::device newBufferWithBytes:A.xs length:A.size() * sizeof(A.xs[0]) options:MTLResourceStorageModeShared];
  auto buffB = [gpu::device newBufferWithBytes:B.xs length:B.size() * sizeof(B.xs[0]) options:MTLResourceStorageModeShared];
  auto buffC = [gpu::device newBufferWithLength:cpuC.size() * sizeof(float) options:MTLResourceStorageModeShared];
  memset(buffC.contents, 0, buffC.length);

  auto cmd = [gpu::queue commandBuffer];
  nn::gemm::gpu(cmd, buffA, buffB, buffC, M, N, P);

  [cmd commit];
  [cmd waitUntilCompleted];

  for (uint i = 0; i < M * P; i++) {
    auto x = ((float*)buffC.contents)[i];
    if (x != naiveC.xs[i]) {
      printf("missmatch at %d : %f != %f\n", i, x, naiveC.xs[i]);
      assert(false);
    }
  }
  return 0;
}
