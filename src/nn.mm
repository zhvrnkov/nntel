#pragma once

#include <MacTypes.h>
#include <iostream>
#include <sstream>
#include <type_traits>
#include <vector>
#include <random>
#include <algorithm>
#include <cassert>

#define NN_IMPL

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

namespace nn {

  namespace utils {
    template <typename xs_t>
      static typename xs_t::value_type area(const xs_t& xs) {
        typename xs_t::value_type out = 1;
        for (auto x : xs) {
          out *= x;
        }
        return out;
      }

      template<typename xs_t>
      std::string xs2str(xs_t xs) {
        std::ostringstream ss;
        ss << "[";
        for (auto x : xs) {
          ss << x << " ";
        }
        ss << "]";
        return ss.str();
      }
  }

  namespace tensor {
    struct data_t {
      std::vector<int64_t> dims;
      id<MTLBuffer> xs;

      data_t(std::initializer_list<int64_t> dims, id<MTLBuffer> xs) : dims(dims), xs(xs) {}
      data_t(std::vector<int64_t> dims, id<MTLBuffer> xs) : dims(dims), xs(xs) {}

      static data_t value(std::initializer_list<float> list);

      static data_t random(std::initializer_list<int64_t> dims);
      template<typename dims_t>
      static data_t copy(dims_t dims, const float* data);
      static data_t copy(std::initializer_list<int64_t> dims, const float* data);

      static data_t zero(std::initializer_list<int64_t> dims);
      template<typename dims_t>
      static data_t zero(dims_t dims);

      static data_t fill(std::initializer_list<int64_t> dims, float x);

      template<typename tensors>
      static data_t concat(tensors xs);

      float* data() const {
        return (float*)[xs contents];
      }

      int64_t size() const {
        return utils::area(dims);
      }

      template<typename dims_t>
      void resize(dims_t dims) {
        this->dims = dims;
      }

      void flatten() {
        this->dims = {size()};
      }

      void transpose();

      data_t copy() const;
    };

    enum class device_type {
      cpu,
      gpu
    };
    void matmul(const data_t A, const data_t B, data_t C, device_type dev=device_type::cpu);
    void add(const data_t A, const data_t B, data_t C, float a=1.0, float b=1.0, device_type dev=device_type::cpu);
    void sub(const data_t A, const data_t B, data_t C, device_type dev=device_type::cpu);
    void mul(const data_t A, const data_t B, data_t C, device_type dev=device_type::cpu);
    void div(const data_t A, const data_t B, data_t C, device_type dev=device_type::cpu);
    void sigmoid(const data_t A, data_t C, device_type=device_type::cpu);

    void sum(const data_t A, data_t C, device_type dev=device_type::cpu);
  }

  namespace layer {
    struct linear {
      tensor::data_t weights;
      tensor::data_t biases;
      tensor::data_t output;

      linear(int64_t inputsCount, int64_t outputsCount)
        : weights(tensor::data_t::random({outputsCount, inputsCount}))
        , biases(tensor::data_t::random({outputsCount}))
        , output(tensor::data_t::zero({outputsCount}))
        {}

      linear(tensor::data_t weights, tensor::data_t biases)
        : weights(weights)
          , biases(biases)
          , output(tensor::data_t::zero({biases.size()}))
          {
            assert(weights.dims.size() == 2);
            assert(biases.dims.size() == 1);
            assert(weights.dims[0] == biases.dims[0]);
          }
      
      tensor::data_t& forward(const tensor::data_t& input)
      {
        if (input.dims.size() != output.dims.size()) {
          std::vector<int64_t> dims;
          dims.push_back(biases.size());
          if (input.dims.size() == 2) {
            dims.push_back(input.dims.back());
          }
          output = tensor::data_t::zero(dims);
        }
        matmul(weights, input, output, tensor::device_type::cpu);
        add(output, biases, output);
        sigmoid(output, output);
        return output;
      }

    };
  }

  namespace helpers {
    tensor::data_t& forward(std::vector<nn::layer::linear>& model, const tensor::data_t& input);

    template<typename dims_t>
      std::vector<nn::layer::linear> buildModel(dims_t dims);
  }

  namespace cost {
    tensor::data_t quadratic(
        std::vector<nn::layer::linear>& model, 
        tensor::data_t& inputs, 
        const tensor::data_t& outputs
        );
  }

  namespace allocator {
    struct Buffer {
      id<MTLBuffer> mtl;
      std::vector<int64_t> shape;
    };
    // in bytes
    constexpr uint64_t alignment = 64;

    template<typename shape_t>
    Buffer aligned_alloc(shape_t shape);
    void free(Buffer buff);
  }

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
    void sub(const float* x, const float* y, float* output, int64_t N);
  }

  namespace gpu {

    void gemm(id<MTLCommandBuffer> cmd, id<MTLBuffer> A, id<MTLBuffer> B, id<MTLBuffer> C, 
        uint64_t m, uint64_t n, uint64_t p);
    void gemv(id<MTLCommandBuffer> cmd, id<MTLBuffer> mat, id<MTLBuffer> vec, id<MTLBuffer> out,
        uint64_t m, uint64_t n);
    void dot(id<MTLCommandBuffer> cmd, id<MTLBuffer> X, id<MTLBuffer> Y, id<MTLBuffer> output);
    void sum(id<MTLCommandBuffer> cmd, id<MTLBuffer> X, float y, id<MTLBuffer> output);

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

      void cpu_dispatch(std::function<void()> block)
      {
        // printf("cpu_dispatch w8=%llu curr=%llu sig=%llu\n", last_id, event.signaledValue, last_id + 1);
        [event notifyListener:listener atValue:last_id block:^(id<MTLSharedEvent> _Nonnull _event, uint64_t _value) {
          block();
          event.signaledValue = _value + 1;
        }];
        last_id += 1;
      }

      void gpu_dispatch(std::function<void()> block)
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


}






























































#ifdef NN_IMPL

namespace nn::tensor {

  std::mt19937 gen{};

  void cpu_matmul(const data_t A, const data_t B, data_t C);
  void gpu_matmul(const data_t A, const data_t B, data_t C);

  void cpu_add(const data_t A, const data_t B, data_t C, float a, float b);
  void gpu_add(const data_t A, const data_t B, data_t C, float a, float b);

  void cpu_mul(const data_t A, const data_t B, data_t C);
  void gpu_mul(const data_t A, const data_t B, data_t C);

  void cpu_div(const data_t A, const data_t B, data_t C);
  void gpu_div(const data_t A, const data_t B, data_t C);

  void cpu_sigmoid(const data_t A, data_t C);
  void gpu_sigmoid(const data_t A, data_t C);

  void cpu_sum(const data_t A, data_t C);
  void gpu_sum(const data_t A, data_t C);

  data_t data_t::value(std::initializer_list<float> list)
  {
    auto size = list.size() * sizeof(float);
    auto storage = [gpu::device newBufferWithBytes:(const void*)list.begin() length:size options:MTLResourceStorageModeShared];

    data_t out {{(int64_t)list.size()}, storage};
    return out;
  }

  template<typename dims_t>
  data_t data_t::copy(dims_t dims, const float* data) {
    auto size = utils::area(dims) * sizeof(float);
    auto storage = [gpu::device newBufferWithBytes:(const void*)data length:size options:MTLResourceStorageModeShared];

    data_t out{dims, storage};
    return out;
  }

  data_t data_t::copy(std::initializer_list<int64_t> dims, const float* data) {
    auto size = utils::area(dims) * sizeof(float);
    auto storage = [gpu::device newBufferWithBytes:(const void*)data length:size options:MTLResourceStorageModeShared];

    data_t out{dims, storage};
    return out;
  }

  data_t data_t::zero(std::initializer_list<int64_t> dims) {
    auto size = utils::area(dims) * sizeof(float);
    auto storage = [gpu::device newBufferWithLength:size options:MTLResourceStorageModeShared];

    data_t out{dims, storage};
    memset(out.data(), 0, utils::area(out.dims) * sizeof(out.data()[0]));
    return out;
  }

  template<typename dims_t>
    data_t data_t::zero(dims_t dims) {
      auto size = utils::area(dims) * sizeof(float);
      auto storage = [gpu::device newBufferWithLength:size options:MTLResourceStorageModeShared];

      data_t out{dims, storage};
      memset(out.data(), 0, utils::area(out.dims) * sizeof(out.data()[0]));
      return out;
    }

  data_t data_t::fill(std::initializer_list<int64_t> dims, float x) {
    auto size = utils::area(dims) * sizeof(float);
    auto storage = [gpu::device newBufferWithLength:size options:MTLResourceStorageModeShared];

    data_t out{dims, storage};
    for (int64_t i = 0; i < out.size(); i++) {
      out.data()[i] = x;
    }
    return out;
  }

  data_t data_t::random(std::initializer_list<int64_t> dims) {
    auto size = utils::area(dims) * sizeof(float);
    auto storage = [gpu::device newBufferWithLength:size options:MTLResourceStorageModeShared];

    std::normal_distribution<float> dstr(0.0, 1.0);

    data_t out{dims, storage};
    auto area = utils::area(out.dims);
    for (auto i = 0; i < area; i++) {
      out.data()[i] = dstr(gen);
    }
    return out;
  }

  template<typename tensors>
  data_t data_t::concat(tensors xs) {
    auto singleSize = xs.begin()->size();
    uint64_t size = 0;
    for (auto& x : xs) {
      assert(x.size() == singleSize);
      size += utils::area(x.dims);
    }

    auto storage = [gpu::device newBufferWithLength:size * sizeof(float) options:MTLResourceStorageModeShared];
    uint64_t i = 0;
    for (auto& x : xs) {
      memcpy(((float*)[storage contents]) + singleSize * i, x.data(), x.size() * sizeof(float));
      i += 1;
    }

    data_t out{{(int64_t)xs.size(), singleSize}, storage};
    return out;
  }

  void matmul(const data_t A, const data_t B, data_t C, device_type dev)
  {
    if (dev == device_type::cpu) {
      nn::stream::global.cpu_dispatch([=] {
          cpu_matmul(A, B, C);
      });
    }
    else if (dev == device_type::gpu) {
      nn::stream::global.gpu_dispatch([=] {
          gpu_matmul(A, B, C);
      });
    }
  }

  void add(const data_t A, const data_t B, data_t C, float a, float b, device_type dev)
  {
    if (dev == device_type::cpu) {
      nn::stream::global.cpu_dispatch([=] {
          cpu_add(A, B, C, a, b);
          });
    }
    else if (dev == device_type::gpu) {
      nn::stream::global.gpu_dispatch([=] {
          gpu_add(A, B, C, a, b);
          });
    }
  }

  void mul(const data_t A, const data_t B, data_t C, device_type dev)
  {
    if (dev == device_type::cpu) {
      nn::stream::global.cpu_dispatch([=] {
          cpu_mul(A, B, C);
          });
    }
    else if (dev == device_type::gpu) {
      nn::stream::global.gpu_dispatch([=] {
          gpu_mul(A, B, C);
          });
    }
  }

  void div(const data_t A, const data_t B, data_t C, device_type dev)
  {
    if (dev == device_type::cpu) {
      nn::stream::global.cpu_dispatch([=] {
          cpu_div(A, B, C);
          });
    }
    else if (dev == device_type::gpu) {
      nn::stream::global.gpu_dispatch([=] {
          gpu_div(A, B, C);
          });
    }
  }

  void sigmoid(const data_t A, data_t C, device_type dev)
  {
    if (dev == device_type::cpu) {
      nn::stream::global.cpu_dispatch([=] {
          cpu_sigmoid(A, C);
          });
    }
    else if (dev == device_type::gpu) {
      nn::stream::global.gpu_dispatch([=] {
          gpu_sigmoid(A, C);
          });
    }
  }

  void sub(const data_t A, const data_t B, data_t C, device_type dev)
  {
    nn::tensor::add(A, B, C, 1.0, -1.0, dev);
  }

  void sum(const data_t A, data_t C, device_type dev)
  {
    if (dev == device_type::cpu) {
      nn::stream::global.cpu_dispatch([=] {
          cpu_sum(A, C);
          });
    }
    else if (dev == device_type::gpu) {
      nn::stream::global.gpu_dispatch([=] {
          gpu_sum(A, C);
          });
    }
  }

  void cpu_sum(const data_t A, data_t C)
  {
    assert(C.dims.size() == 1);
    assert(C.dims[0] == 1);

    for (int64_t i = 0; i < A.size(); i++) {
      *C.data() += A.data()[i];
    }
  }

  void gpu_sum(const data_t A, data_t C)
  {
    assert(C.dims.size() == 1);
    assert(C.dims[0] == 1);
  }

  void cpu_add(const data_t A, const data_t B, data_t C, float a, float b)
  {
    if (A.dims.size() == B.dims.size()) {
      assert(A.dims.size() == C.dims.size());
      for (int64_t i = 0; i < A.size(); i++) {
        C.data()[i] = A.data()[i] * a + B.data()[i] * b;
      }
    } else if (B.dims.size() == 1) {
      assert(A.dims.size() == C.dims.size());
      if (B.dims[0] == 1) {
        for (int64_t i = 0; i < A.size(); i++) {
          C.data()[i] = A.data()[i] * a + *B.data() * b;
        }
      } else {
        assert(B.dims[0] == A.dims[0]);
        assert(A.dims.size() == 2);

        for (int64_t i = 0; i < A.size(); i++) {
          C.data()[i] = A.data()[i] * a + B.data()[i / A.dims[1]] * b;
        }
      }
    } else if (A.dims.size() == 1) {
      assert(B.dims.size() == C.dims.size());
      assert(A.dims[0] == 1);
      for (int64_t i = 0; i < C.size(); i++) {
        C.data()[i] = B.data()[i] * b + *A.data() * a;
      }
    } else {
      assert(false);
    }
  }

  void gpu_add(const data_t A, const data_t B, data_t C, float a, float b)
  {
    printf("no gpu add\n");
    assert(false);
  }

  void cpu_matmul(const data_t A, const data_t B, data_t C)
  {
    if (A.dims.size() == 2 && B.dims.size() == 2) {
      assert(C.dims.size() == 2);
      assert(A.dims[0] == C.dims[0]);
      assert(A.dims[1] == B.dims[0]);
      assert(B.dims[1] == C.dims[1]);
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

  void gpu_matmul(const data_t A, const data_t B, data_t C)
  {
    if (A.dims.size() == 2 && B.dims.size() == 2) {
      assert(C.dims.size() == 2);
      assert(A.dims[0] == C.dims[0]);
      assert(A.dims[1] == B.dims[0]);
      assert(B.dims[1] == C.dims[1]);
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

  void cpu_sigmoid(const data_t A, data_t C)
  {
    assert(A.size() == C.size());
    for (auto i = 0; i < A.size(); i++) {
      C.data()[i] = 1.0 / (1.0 + exp(-A.data()[i]));
    }
  }

  void gpu_sigmoid(const data_t A, data_t C)
  {
    printf("no gpu add\n");
    assert(false);
  }

  void cpu_mul(const data_t A, const data_t B, data_t C)
  {
    // Check for scalar multiplication first
    if (B.dims.size() == 1 && B.size() == 1) {
      assert(A.size() == C.size());
      assert(A.dims.size() == C.dims.size());
      for (uint i = 0; i < A.size(); i++) {
        C.data()[i] = A.data()[i] * B.data()[0];
      }
    } else if (A.dims.size() == B.dims.size()) {
      assert(A.dims.size() == C.dims.size());
      assert(A.size() == C.size());
      assert(A.size() == B.size());
      for (uint i = 0; i < A.size(); i++) {
        C.data()[i] = A.data()[i] * B.data()[i];
      }
    } else {
      assert(false);
    }
  }

  void gpu_mul(const data_t A, const data_t B, data_t C)
  {
    printf("no gpu add\n");
    assert(false);
  }

  void cpu_div(const data_t A, const data_t B, data_t C)
  {
    // Check for scalar multiplication first
    if (B.dims.size() == 1 && B.size() == 1) {
      assert(A.size() == C.size());
      assert(A.dims.size() == C.dims.size());
      for (uint i = 0; i < A.size(); i++) {
        C.data()[i] = A.data()[i] / B.data()[0];
      }
    } else if (A.dims.size() == B.dims.size()) {
      assert(A.dims.size() == C.dims.size());
      assert(A.size() == C.size());
      assert(A.size() == B.size());
      for (uint i = 0; i < A.size(); i++) {
        C.data()[i] = A.data()[i] / B.data()[i];
      }
    } else {
      assert(false);
    }
  }

  void gpu_div(const data_t A, const data_t B, data_t C)
  {
    printf("no gpu add\n");
    assert(false);
  }
}

namespace nn::tensor {
void data_t::transpose() {
  assert(dims.size() == 2);
  auto tstorage = [gpu::device newBufferWithLength:size() * sizeof(float) options:MTLResourceStorageModeShared];
  auto oldData = xs;
  auto old_M = dims[0];
  auto old_N = dims[1];
  std::swap(dims[0], dims[1]);
  xs = tstorage;
  nn::stream::global.cpu_dispatch(^() {
    nn::cpu::transpose<1>((float*)[oldData contents], (float*)[tstorage contents], old_M, old_N);
  });
}

data_t data_t::copy() const {
  auto storage = [gpu::device newBufferWithLength:[xs length] options:MTLResourceStorageModeShared];
  data_t out{dims, storage};
  nn::stream::global.gpu_dispatch(^() {
    id<MTLBlitCommandEncoder> blit = [nn::stream::global.cmd blitCommandEncoder];
    [blit copyFromBuffer:xs sourceOffset:0 toBuffer:storage destinationOffset:0 size:[storage length]];
    [blit endEncoding];
  });
  return out;
}
}

namespace nn::helpers {
  tensor::data_t& forward(std::vector<nn::layer::linear>& model, tensor::data_t& input)
  {
    auto output = &input;
    for (auto& l : model) {
      output = &l.forward(*output);
    }
    return *output; 
  }

  template<typename dims_t>
    // 784 100 10
    std::vector<nn::layer::linear> buildModel(dims_t dims)
    {
      std::vector<nn::layer::linear> model;
      for (uint i = 0; i < dims.size() - 1; i++) {
        model.emplace_back(dims.at(i), dims.at(i + 1));
      }
      return model;
    }
}

namespace nn::allocator {
  // in bytes
  template<typename shape_t>
  Buffer aligned_alloc(shape_t shape)
  {
    std::vector<int64_t> realShape = shape;
    for (auto& s : realShape) {
      s = alignment * (s + (alignment - 1)) / alignment;
    }

    auto length = utils::area(realShape) * sizeof(float);
    auto mtlBuff = [gpu::device newBufferWithLength:length options:MTLResourceStorageModeShared];
    return Buffer { mtlBuff, realShape };
  }

  void free(Buffer buff)
  {
    // no-op
  }
}

namespace nn::cost {
  tensor::data_t quadratic(
    std::vector<nn::layer::linear>& model, 
    tensor::data_t& inputs, 
    const tensor::data_t& outputs
  )
  {
    auto modelOutput = nn::helpers::forward(model, inputs);
    modelOutput.transpose();

    auto diff = nn::tensor::data_t::zero(modelOutput.dims);
    auto cost = nn::tensor::data_t::value({0.0});

    nn::tensor::sub(modelOutput, outputs, diff);
    nn::tensor::mul(diff, diff, diff);
    nn::tensor::sum(diff, cost);
    nn::tensor::div(cost, nn::tensor::data_t::value({(float)2 * outputs.dims[0]}), cost);

    return cost;
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
              if (y + yb >= M) break;
              // A[y * N + yb * N + k]... is in cache now, but yb is strided by N.
              // is that a problem for L1 cache?
              // if BLOCK_Y is 4, then 32KB of A is used for this K loop
              float tA = block_A[yb * N + k];
              for (int xb = 0; xb < BLOCK_X; xb++) {
                if (x + xb >= P) break;
                // B[x * N + xb * N + k] we go through xb with stride N
                // so whole B for 0<=xb<4 is in cache?
                block_C[yb * BLOCK_X + xb] += tA * block_B[xb * N + k];
              }
            }
          }

          for (int yb = 0; yb < BLOCK_Y; yb++) {
            if (y + yb >= M) break;
            for (int xb = 0; xb < BLOCK_X; xb++) {
              if (x + xb >= P) break;
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
    *output = 0;
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

  void sub(const float* x, const float* y, float* output, int64_t N)
  {
    for (int64_t i = 0; i < N; i++) {
      output[i] = x[i] - y[i];
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

  void sum(id<MTLCommandBuffer> cmd, id<MTLBuffer> X, float y, id<MTLBuffer> output)
  {
    assert(X.length == output.length);
    const uint64_t N = X.length / sizeof(float);

    static id<MTLComputePipelineState> kernel0;
    static id<MTLComputePipelineState> kernel1;
    static dispatch_once_t onceToken;
    dispatch_once(&onceToken, ^{
        auto kernel0Func = [lib newFunctionWithName:@"sum_reduce0"];
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
    [encoder setBuffer:X offset:0 atIndex:0];
    [encoder setBytes:(void*)&y length:sizeof(y) atIndex:1];
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
