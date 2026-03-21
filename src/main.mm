#include <array>
#include <iostream>
#include <cassert>

#define GPU
#define NN_IMPL
#include "nn.mm"

namespace test {
struct TestSample {
    std::vector<float> input;
    std::vector<float> output;
    
    TestSample(std::vector<float> input, std::vector<float> output) : input(input), output(output) {}
};
using TestData = std::vector<TestSample>;

}

int main()
{
  constexpr auto M = 2048;
  constexpr auto N = 2048;
  constexpr auto P = 2048;
  auto A = nn::tensor::data_t::random_int({M, N});
  auto B = nn::tensor::data_t::random_int({N, P});
  auto naiveC = nn::tensor::data_t::zero({M, P});
  auto C = nn::tensor::data_t::zero({M, P});

  nn::cpu::gemm(A.data(), B.data(), naiveC.data(), naiveC.dims[0], A.dims[1], naiveC.dims[1]);
  nn::tensor::mul(A, B, C, nn::tensor::device_type::cpu);
  nn::stream::global.synchronize();

  for (uint i = 0; i < M * P; i++) {
    if (C.data()[i] != naiveC.data()[i]) {
      printf("missmatch at %d : %f != %f\n", i, C.data()[i], naiveC.data()[i]);
      assert(false);
    }
  }

  // gpu
  auto d = nn::tensor::data_t::fill({1}, 1);
  nn::tensor::mul(A, B, C, nn::tensor::device_type::gpu);
  nn::tensor::add(C, d, C);
  nn::stream::global.synchronize();

  for (uint i = 0; i < M * P; i++) {
    auto x = C.data()[i];
    auto eps = fabsf(x - (naiveC.data()[i] + 1));
    if (eps > 0.001) {
      printf("missmatch at %d : %f != %f (%f)\n", i, x, naiveC.data()[i], eps);
      assert(false);
    }
  }
  return 0;
}

