#pragma once

#include <MacTypes.h>
#include <cstring>
#include <iostream>
#include <optional>
#include <simd/packed.h>
#include <simd/simd.h>
#include <simd/math.h>
#include <sstream>
#include <type_traits>
#include <filesystem>
#include <vector>
#include <span>
#include <random>
#include <algorithm>
#include <cassert>
#include <print>
#include <functional>
#define ACCELERATE_NEW_LAPACK
#include <Accelerate/Accelerate.h>

#define NN_IMPL

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

namespace nn {
namespace gpu {

void gemm(id<MTLCommandBuffer> cmd, id<MTLBuffer> A, id<MTLBuffer> B, id<MTLBuffer> C,
          uint64_t m, uint64_t n, uint64_t p);
void gemv(id<MTLCommandBuffer> cmd, id<MTLBuffer> mat, id<MTLBuffer> vec, id<MTLBuffer> out,
          uint64_t m, uint64_t n);
void dot(id<MTLCommandBuffer> cmd, id<MTLBuffer> X, id<MTLBuffer> Y, id<MTLBuffer> output);
void sum(id<MTLCommandBuffer> cmd, id<MTLBuffer> X, float y, id<MTLBuffer> output);
void sigmoid(id<MTLCommandBuffer> cmd, id<MTLBuffer> X, id<MTLBuffer> output);
void sigmoidDerivative(id<MTLCommandBuffer> cmd, id<MTLBuffer> X, id<MTLBuffer> output);
void axpby(id<MTLCommandBuffer> cmd, id<MTLBuffer> X, id<MTLBuffer> Y, id<MTLBuffer> output, float fX, float fY, float A);
void axpby2dBcol(id<MTLCommandBuffer> cmd, id<MTLBuffer> X, id<MTLBuffer> Y, id<MTLBuffer> output, float fX, float fY, float A);
void sum_dim0(id<MTLCommandBuffer> cmd, id<MTLBuffer> X, id<MTLBuffer> out, uint32_t Nrows, uint32_t Ncols, uint32_t stride);
void sum_dim1(id<MTLCommandBuffer> cmd, id<MTLBuffer> X, id<MTLBuffer> out, uint32_t Nrows, uint32_t Ncols, uint32_t stride);
void transpose(id<MTLCommandBuffer> cmd, id<MTLBuffer> X, id<MTLBuffer> out, uint32_t M, uint32_t N);
void addcmul(id<MTLCommandBuffer> cmd, id<MTLBuffer> X, id<MTLBuffer> Y, id<MTLBuffer> output, float a, float b);
void addcdiv(id<MTLCommandBuffer> cmd, id<MTLBuffer> X, id<MTLBuffer> Y, id<MTLBuffer> output, float a, float b);
void axpy(id<MTLCommandBuffer> cmd, id<MTLBuffer> X, id<MTLBuffer> Y, id<MTLBuffer> output, float a);

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

namespace cpu {
dispatch_queue_attr_t attr = dispatch_queue_attr_make_with_autorelease_frequency(DISPATCH_QUEUE_SERIAL, DISPATCH_AUTORELEASE_FREQUENCY_WORK_ITEM);

dispatch_queue_t queue = dispatch_queue_create("nntel", attr);

template<int64_t BLOCK_X=4, int64_t BLOCK_Y = 4>
void gemm(const float* A, const float* B, float* C,
          int64_t M, int64_t N, int64_t P,
          bool transposeB = true);

void gemv(const float* mat, const float* vec, float* output, uint64_t M, uint64_t N);

template<int BLOCK>
void transpose(const float* m, float* mT, int64_t M, int64_t N);

void dot(const float* x, const float* y, float* output, int64_t N);

void axpby(const float* x, const float* y, float* output, int64_t N, float fX, float fY, float a);
void axpby2dBcol(const float* x, const float* y, float* output, int64_t N, int64_t NB, float fX, float fY, float a);
void axpy(const float* x, const float* y, float* output, int64_t N, float a);
void addcmul(const float* x, const float* y, float* output, int64_t N, float a, float b);
void addcdiv(const float* x, const float* y, float* output, int64_t N, float a, float b);
void sigmoid(const float* x, float* output, int64_t N);
void sigmoidDerivative(const float* x, float* output, int64_t N);
void sum(const float* x, float* output, int64_t N);
void sum_dim0(const float* x, float* output, int64_t Nrows, int64_t Ncols, int64_t stride);
void sum_dim1(const float* x, float* output, int64_t Nrows, int64_t Ncols, int64_t stride);
}

namespace accelerate {
void gemm(const float* A, const float* B, float* C, int64_t M, int64_t N, int64_t P, bool transposeB = false);
void gemv(const float* mat, const float* vec, float* output, uint64_t M, uint64_t N);
void dot(const float* x, const float* y, float* output, int64_t N);
void axpby(const float* x, const float* y, float* output, int64_t N, float fX, float fY, float a);
void axpby2dBcol(const float* x, const float* y, float* output, int64_t N, int64_t NB, float fX, float fY, float a);
void axpy(const float* x, const float* y, float* output, int64_t N, float a);
void addcmul(const float* x, const float* y, float* output, int64_t N, float a, float b);
void addcdiv(const float* x, const float* y, float* output, int64_t N, float a, float b);
void sigmoid(const float* x, float* output, int64_t N);
void sigmoidDerivative(const float* x, float* output, int64_t N);
void sum(const float* x, float* output, int64_t N);
void sum_dim0(const float* x, float* output, int64_t Nrows, int64_t Ncols, int64_t stride);
void sum_dim1(const float* x, float* output, int64_t Nrows, int64_t Ncols, int64_t stride);
void transpose(const float* m, float* mT, int64_t M, int64_t N);
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

namespace stream {
struct ctx_t {
  ctx_t()
  : cmd([gpu::queue commandBuffer])
  , event([gpu::device newSharedEvent])
  , listener([[MTLSharedEventListener alloc] initWithDispatchQueue:cpu::queue])
  , last_id(0)
  , lastCommandCpu(std::nullopt)
  {}

  id<MTLCommandBuffer> cmd;
  id<MTLSharedEvent> event;
  MTLSharedEventListener* listener;
  uint64_t last_id = 0;
  std::optional<bool> lastCommandCpu;
  uint64_t gpu_pending_commands_count = 0;
  
  inline void gpuFlush();
  inline void synchronize();
  template<typename block_t>
  inline void cpu_dispatch(block_t block);
  inline void gpu_dispatch(std::function<void()> block);
};

ctx_t global = ctx_t();
}

namespace allocator {
std::vector<int64_t> computeStrides(const std::vector<int64_t>& shape) {
  std::vector<int64_t> strides;
  strides.resize(shape.size());
  int64_t stride = 1;
  for (int64_t i = shape.size() - 1; i >= 0; --i) {
    strides[i] = stride;
    stride *= shape[i];
  }
  return strides;
}
struct ndspan {
  float* data;
  std::vector<int64_t> shape;
  std::vector<int64_t> strides;
  
  ndspan(float* data, std::vector<int64_t> shape)
  : data(data)
  , shape(shape)
  , strides(computeStrides(shape))
  {}

  int64_t size() const {
    return utils::area(shape) * sizeof(float);
  }
  
  int64_t idx_dot(const std::vector<int64_t>& idxs) const {
    int64_t idx = 0;
    for (uint64_t i = 0; i < idxs.size(); i++) {
      idx += idxs[i] * strides[i];
    }
    return idx;
  }
  
  
  float& operator[](std::vector<int64_t> idxs) {
    assert(idxs.size() == strides.size());
    return data[idx_dot(idxs)];
  }
  
  std::span<float> row(std::vector<int64_t> idxs) {
    assert((idxs.size() + 1) == strides.size());
    return {data + idx_dot(idxs), (size_t)shape.back()};
  }
  
  std::vector<int64_t> idxs(const std::vector<int64_t>& shape, int64_t flatIdx) const {
    auto strides = computeStrides(shape);
    std::vector<int64_t> output(shape.size());
    int64_t remaining = flatIdx;
    for (uint64_t i = 0; i < shape.size(); i++) {
      output[i] = remaining / strides[i];
      remaining %= strides[i];
    }
    return output;
  }
  
  void rowsIter(std::vector<int64_t> shape, std::function<void(int64_t idx)> action) const {
    uint idx = 0;
    auto a = utils::area(shape);
    
    while (idx < a) {
      auto rowIdx = idxs(shape, idx);
      rowIdx.pop_back();
      auto fidx = idx_dot(rowIdx);
      for (auto i = 0; i < shape.back(); i++) action(fidx + i);
      idx += shape.back();
    }
  }
  
  void zero_validate(const std::vector<int64_t>& logical_shape) const {
    assert(logical_shape.size() == shape.size());
    auto realTotal = utils::area(shape);
    auto realLastDim = shape.back();
    auto logicalLastDim = logical_shape.back();
    
    for (int64_t flat = 0; flat < realTotal; flat += realLastDim) {
      auto ri = idxs(shape, flat);
      bool rowInLogical = true;
      for (uint64_t d = 0; d + 1 < ri.size(); d++) {
        if (ri[d] >= logical_shape[d]) {
          rowInLogical = false;
          break;
        }
      }
      
      int64_t start = rowInLogical ? logicalLastDim : 0;
      for (int64_t i = start; i < realLastDim; i++) {
        assert(data[flat + i] == 0.0f && "padding is not zero");
      }
    }
  }
  
  void spanIter(std::vector<int64_t> shape, std::function<void(int64_t idx, int64_t sz)> action) const {
    uint idx = 0;
    auto a = utils::area(shape);
    
    while (idx < a) {
      auto rowIdx = idxs(shape, idx);
      rowIdx.pop_back();
      auto fidx = idx_dot(rowIdx);
      action(fidx, this->shape.back());
      idx += shape.back();
    }
  }
};

struct Buffer;
template<typename shape_t>
Buffer aligned_alloc(shape_t shape);
void free(Buffer buff);

struct Buffer {
  float* data;
  ndspan dspan;
  
  Buffer(id<MTLBuffer> mtl, std::vector<int64_t> shape)
  : data((float*)[mtl contents])
  , dspan(ndspan{data, shape})
  , mtl(mtl)
  {
  }

  Buffer(std::shared_ptr<float[]> sptr, std::vector<int64_t> shape)
  : data(sptr.get())
  , dspan(ndspan{data, shape})
  , sptr(sptr)
  {
  }

  float& operator[](std::vector<int64_t> idxs) {
    return dspan[idxs];
  }
  
  std::span<float> row(std::vector<int64_t> idxs) {
    return dspan.row(idxs);
  }

  id<MTLBuffer> buff() const {
    return mtl;
  }
  
  uint64_t size() const {
    return dspan.size();
  }
  private:
  id<MTLBuffer> mtl;
  std::shared_ptr<float[]> sptr;
  
};
constexpr uint64_t alignment = 1;
}


namespace tensor {
enum class device_type {
  cpu,
  gpu,
  accelerate
};

struct data_t {
  std::vector<int64_t> dims;
  allocator::Buffer xs;
  
  data_t(std::initializer_list<int64_t> dims, allocator::Buffer xs)
  : dims(dims)
  , xs(xs) {}
  
  data_t(std::vector<int64_t> dims, allocator::Buffer xs)
  : dims(dims)
  , xs(xs) {}

  data_t(std::vector<int64_t> dims)
  : dims(dims)
  , xs(allocator::aligned_alloc(dims)) {}

  data_t(std::initializer_list<int64_t> dims)
  : dims(dims)
  , xs(allocator::aligned_alloc(dims)) {}

  ~data_t() {
    allocator::free(xs);
  }
  
  static data_t value(std::initializer_list<float> list);
  
  static data_t random(std::initializer_list<int64_t> dims, float stddev = 1.0f);
  template<typename dims_t>
  static data_t copy(dims_t dims, const float* data);
  static data_t copy(std::initializer_list<int64_t> dims, const float* data);
  
  static data_t zero(std::initializer_list<int64_t> dims);
  template<typename dims_t>
  static data_t zero(dims_t dims);
  
  static data_t fill(std::initializer_list<int64_t> dims, float x);
  
  template<typename tensors>
  static data_t concat(const tensors& xs);

  id<MTLBuffer> buff() const {
    return xs.buff();
  }
  
  float* data() const {
    return xs.data;
  }
  
  int64_t size() const {
    return utils::area(dims);
  }
  
  int64_t rsize() const {
    return utils::area(xs.dspan.shape);
  }
  
  const std::vector<int64_t>& rshape() const {
    return xs.dspan.shape;
  }
  
  template<typename dims_t>
  void resize(dims_t dims) {
    this->dims = dims;
  }
  
  void flatten();
  
  void transpose(device_type dev = device_type::cpu);
  
  data_t copy(tensor::device_type dev) const;
  
  void rowsIter(std::function<void(int64_t)> f) const {
    xs.dspan.rowsIter(dims, f);
  }
  
  void spanIter(std::function<void(int64_t, int64_t)> f) const {
    xs.dspan.spanIter(dims, f);
  }
};

void matmul(const data_t& A, const data_t& B, data_t& C, device_type dev=device_type::cpu);
void add(const data_t& A, const data_t& B, data_t& C, float a=1.0, float b=1.0, device_type dev=device_type::cpu);
void sub(const data_t& A, const data_t& B, data_t& C, device_type dev=device_type::cpu);
void mul(const data_t& A, const data_t& B, data_t& C, device_type dev=device_type::cpu);
void div(const data_t& A, const data_t& B, data_t& C, device_type dev=device_type::cpu);
void sigmoid(const data_t& A, data_t& C, device_type=device_type::cpu);
void sigmoidDerivative(const data_t& A, data_t& C, device_type=device_type::cpu);

void sum(const data_t& A, data_t& C, int64_t dim=-1, device_type dev=device_type::cpu);
void transpose(const data_t& A, data_t& C, device_type dev=device_type::cpu);
}

namespace layer {
struct base {
virtual tensor::data_t forward(const tensor::data_t& input, tensor::device_type dev) = 0;
virtual tensor::data_t backward(const tensor::data_t& input, tensor::device_type dev) = 0;
virtual void applyGrad(tensor::device_type) {}
virtual ~base() = default;
};

struct linear : public base {
  tensor::data_t weights;
  tensor::data_t biases;
  tensor::data_t zs;
  
  std::optional<tensor::data_t> dweights;
  std::optional<tensor::data_t> dbiases;
  std::optional<tensor::data_t> input;
  tensor::data_t tweights;
  
  linear(int64_t inputsCount, int64_t outputsCount)
  : weights(tensor::data_t::random({outputsCount, inputsCount}, std::sqrt(2.0f / (inputsCount + outputsCount))))
  , biases(tensor::data_t::zero({outputsCount}))
  , zs(tensor::data_t::zero({outputsCount}))
  , tweights(tensor::data_t{inputsCount, outputsCount})
  {}
  
  linear(tensor::data_t weights, tensor::data_t biases)
  : weights(weights)
  , biases(biases)
  , zs(tensor::data_t::zero({biases.size()}))
  , tweights(tensor::data_t{weights.dims[1], weights.dims[0]})
  {
    assert(weights.dims.size() == 2);
    assert(biases.dims.size() == 1);
    assert(weights.dims[0] == biases.dims[0]);
  }
  
  tensor::data_t forward(const tensor::data_t& input, tensor::device_type dev) override
  {
    std::vector<int64_t> dims;
    dims.push_back(biases.size());
    if (input.dims.size() == 2) {
      dims.push_back(input.dims.back());
    }
    if (dims != zs.dims) {
      zs = tensor::data_t(dims);
    }
    this->input = input;
    matmul(weights, input, zs, dev);
    add(zs, biases, zs, 1.0, 1.0, dev);
    // nn::stream::global.gpuFlush();
    return zs;
  }
  
  tensor::data_t backward(const tensor::data_t& input, tensor::device_type dev) override
  {
    tensor::transpose(weights, tweights, dev);
    tensor::data_t output = tensor::data_t::zero({tweights.dims[0], input.dims[1]});
    nn::tensor::matmul(tweights, input, output, dev);
    
    if (dweights.has_value() == false) {
      dweights = tensor::data_t::zero({weights.dims[0], weights.dims[1]});
    }
    if (dbiases.has_value() == false) {
      dbiases = tensor::data_t({biases.dims[0]});
    }
//    memset(dbiases->data(), 0, dbiases->rsize() * sizeof(float));
    
    (*(this->input)).transpose(dev);
    
    tensor::matmul(input, *(this->input), *dweights, dev);
    tensor::sum(input, *dbiases, 1, dev);
    
    return output;
  }
  
  void applyGrad(tensor::device_type dev) override
  {
    tensor::add(weights, *dweights, weights, 1.0, -1.0 / (2.0 * zs.dims[1]), dev);
    tensor::add(biases, *dbiases, biases, 1.0, -1.0 / (2.0 * zs.dims[1]), dev);
  }
};

struct sigmoid : public base {
  sigmoid(int64_t size)
  : size(size)
  {}

  tensor::data_t forward(const tensor::data_t& input, tensor::device_type dev) override
  {
    std::vector<int64_t> dims;
    dims.push_back(size);
    if (input.dims.size() == 2) {
      dims.push_back(input.dims.back());
    }
    if (output.has_value() == false || output.value().dims != dims) {
      output = tensor::data_t(dims);
    }
    tensor::sigmoid(input, *output, dev);
    this->input = input;
    return *output;
  }

  tensor::data_t backward(const tensor::data_t& input, tensor::device_type dev) override
  {
    nn::tensor::sigmoidDerivative(*(this->input), *(this->input), dev);

    error = tensor::data_t{input.dims};
    nn::tensor::mul(input, *(this->input), *error, dev);

    return *error;
  }

  private:
  int64_t size;
  std::optional<tensor::data_t> output;
  std::optional<tensor::data_t> input;
  std::optional<tensor::data_t> error;
};
}

namespace helpers {
const tensor::data_t forward(std::vector<nn::layer::linear>& model, const tensor::data_t& input, tensor::device_type);

template<typename dims_t>
std::vector<std::unique_ptr<nn::layer::base>> buildModel(dims_t dims);
}

namespace cost {
tensor::data_t quadratic(
                         std::vector<std::unique_ptr<nn::layer::base>>& model,
                         tensor::data_t& inputs,
                         const tensor::data_t& outputs,
                         tensor::data_t* backwardInput,
                         tensor::device_type dev
                         );
}

namespace train {
struct data_loader {
  data_loader(std::string path, int64_t batchSize, bool shuffle)
  : path(path)
  , batchSize(batchSize)
  , shuffle(shuffle)
  {
    load();
  }
  
  std::optional<std::pair<tensor::data_t, tensor::data_t>> nextBatch()
  {
    std::vector<float> output_vectors;
    std::vector<int64_t> output_vectors_dims = {0, 10};
    std::vector<nn::tensor::data_t> input_images;
    
    if (idx >= entries.size()) return std::nullopt;
    
    int64_t count = batchSize < 0 ? entries.size() : batchSize;
    for (auto i = 0; i < count; i++) {
      input_images.push_back(entries.at(idx + i).second);
      output_vectors.insert(output_vectors.end(), entries.at(idx + i).first.begin(), entries.at(idx + i).first.end());
      output_vectors_dims.at(0) += 1;
    }
    idx += count;
    
    auto output = std::make_pair(nn::tensor::data_t::concat(input_images), nn::tensor::data_t::copy(output_vectors_dims, output_vectors.data()));
    output.first.transpose();
    output.second.transpose();
    return output;
  }
  
  void load()
  {
    idx = 0;
    namespace fs = std::filesystem;
    std::vector<fs::path> dirs;
    for (const auto& entry : fs::directory_iterator(path)) {
      if (fs::is_directory(entry.path())) {
        dirs.push_back(entry.path());
      }
    }
    std::sort(dirs.begin(), dirs.end());
    
    for (const auto& dirPath : dirs) {
      auto dirName = dirPath.filename().string();
      int64_t dirNameNumber = std::atoi(dirName.data());
      
      for (const auto& imageEntry : fs::directory_iterator(dirPath)) {
        std::vector<float> imageOutput(10);
        imageOutput[dirNameNumber] = 1.0;
        entries.push_back(std::make_pair(imageOutput, grayscale_image(imageEntry.path())));
      }
    }
    
    if (shuffle) {
      std::shuffle(entries.begin(), entries.end(), gen);
    }
  }
  
  void reset() {
    idx = 0;
    if (shuffle) {
      std::shuffle(entries.begin(), entries.end(), gen);
    }
  }
  
  nn::tensor::data_t grayscale_image(std::string path)
  {
    int width, height, channels;
    auto data = stbi_loadf(path.data(), &width, &height, &channels, 0);
    
    auto output = nn::tensor::data_t::copy({height, width}, data);
    
    stbi_image_free(data);
    return output;
  }
  
  
private:
  std::vector<std::pair<std::vector<float>, nn::tensor::data_t>> entries;
  std::string path;
  uint64_t idx = 0;
  int64_t batchSize;
  bool shuffle;
  std::mt19937 gen{};
};

void train(tensor::data_t trainingData, size_t epochs_count, size_t mini_batch_size, float learning_rate, std::optional<tensor::data_t> testData);
}
}






























































#ifdef NN_IMPL

namespace nn::tensor {

std::mt19937 gen{};

void copy(allocator::Buffer dst, float* src, const std::vector<int64_t>& shape) {
  assert(shape.size() == dst.dspan.shape.size());
  
  auto srcSpan = allocator::ndspan(src, shape);
  
  auto rowSize = shape.back();
  uint idx = 0;
  auto a = utils::area(shape);
  while (idx < a) {
    auto rowIdx = srcSpan.idxs(shape, idx);
    rowIdx.pop_back();
    auto drow = dst.row(rowIdx);
    auto srow = srcSpan.row(rowIdx);
    for (auto i = 0; i < rowSize; i++) {
      drow[i] = srow[i];
    }
    idx += rowSize;
  }
}

data_t data_t::value(std::initializer_list<float> list)
{
  std::vector<int64_t> dims = {(int64_t)list.size()};
  auto storage = allocator::aligned_alloc(dims);
  nn::tensor::copy(storage, (float*)list.begin(), dims);
  
  data_t out {dims, storage};
  return out;
}

template<typename dims_t>
data_t data_t::copy(dims_t dims, const float* data) {
  auto storage = allocator::aligned_alloc(dims);
  nn::tensor::copy(storage, (float*)data, std::vector<int64_t>(dims.begin(), dims.end()));
  
  data_t out{dims, storage};
  return out;
}

data_t data_t::copy(std::initializer_list<int64_t> dims, const float* data) {
  auto storage = allocator::aligned_alloc(dims);
  nn::tensor::copy(storage, (float*)data, std::vector<int64_t>(dims));
  
  data_t out{dims, storage};
  return out;
}

data_t data_t::zero(std::initializer_list<int64_t> dims) {
  auto storage = allocator::aligned_alloc(dims);
  
  data_t out{dims, storage};
  memset(out.data(), 0, utils::area(storage.dspan.shape) * sizeof(float));
  return out;
}

template<typename dims_t>
data_t data_t::zero(dims_t dims) {
  auto storage = allocator::aligned_alloc(dims);
  
  data_t out{dims, storage};
  memset(out.data(), 0, utils::area(storage.dspan.shape) * sizeof(float));
  return out;
}

data_t data_t::fill(std::initializer_list<int64_t> dims, float x) {
  auto storage = allocator::aligned_alloc(dims);
  
  // Zero the entire buffer first (including padding)
  memset(storage.data, 0, utils::area(storage.dspan.shape) * sizeof(float));
  
  // Fill only the logical shape
  auto area = utils::area(dims);
  std::vector<float> tmp(area, x);
  nn::tensor::copy(storage, tmp.data(), std::vector<int64_t>(dims));
  
  data_t out{dims, storage};
  return out;
}

data_t data_t::random(std::initializer_list<int64_t> dims, float stddev) {
  auto storage = allocator::aligned_alloc(dims);
  
  // Zero the entire buffer first (including padding)
  memset(storage.data, 0, utils::area(storage.dspan.shape) * sizeof(float));
  
  std::normal_distribution<float> dstr(0.0, stddev);
  
  // Generate random data only for logical shape
  auto area = utils::area(dims);
  std::vector<float> tmp(area);
  for (auto i = 0; i < area; i++) {
    tmp[i] = dstr(gen);
  }
  nn::tensor::copy(storage, tmp.data(), std::vector<int64_t>(dims));
  
  data_t out{dims, storage};
  return out;
}

template<typename tensors>
data_t data_t::concat(const tensors& xs) {
  auto singleSize = xs.begin()->size();
  uint64_t count = 0;
  for (auto& x : xs) {
    assert(x.size() == singleSize);
    count++;
  }
  
  std::vector<int64_t> dims = {(int64_t)count, singleSize};
  auto storage = allocator::aligned_alloc(dims);
  
  // Zero the entire buffer first (including padding)
  memset(storage.data, 0, utils::area(storage.dspan.shape) * sizeof(float));
  
  uint64_t i = 0;
  for (auto& x : xs) {
    memcpy(storage.data + x.rsize() * i, x.data(), x.rsize() * sizeof(float));
    i += 1;
  }
  
  data_t out{dims, storage};
  return out;
}

void matmul(const data_t& A, const data_t& B, data_t& C, device_type dev)
{
  if (A.dims.size() == 2 && B.dims.size() == 2) {
    assert(C.dims.size() == 2);
    assert(A.dims[0] == C.dims[0]);
    assert(A.dims[1] == B.dims[0]);
    assert(B.dims[1] == C.dims[1]);
    if (dev == device_type::cpu) {
      nn::stream::global.cpu_dispatch([=] {
        nn::cpu::gemm(A.data(), B.data(), C.data(), C.rshape()[0], A.rshape()[1], C.rshape()[1]);
      });
    } else if (dev == device_type::accelerate) {
      nn::stream::global.cpu_dispatch([=] {
        nn::accelerate::gemm(A.data(), B.data(), C.data(), C.dims[0], A.dims[1], C.dims[1]);
      });
    } else if (dev == device_type::gpu) {
      nn::stream::global.gpu_dispatch([=] {
        nn::gpu::gemm(stream::global.cmd, A.buff(), B.buff(), C.buff(), C.rshape()[0], A.rshape()[1], C.rshape()[1]);
      });
    }
  } else if (A.dims.size() == 1 && B.dims.size() == 1) {
    assert(C.dims.size() == 1);
    assert(A.dims[0] == B.dims[0]);
    assert(1 == C.dims[0]);
    if (dev == device_type::cpu) {
      nn::stream::global.cpu_dispatch([=] {
        nn::cpu::dot(A.data(), B.data(), C.data(), A.rsize());
      });
    } else if (dev == device_type::accelerate) {
      nn::stream::global.cpu_dispatch([=] {
        nn::accelerate::dot(A.data(), B.data(), C.data(), A.dims[0]);
      });
    } else if (dev == device_type::gpu) {
      nn::stream::global.gpu_dispatch([=] {
        nn::gpu::dot(stream::global.cmd, A.buff(), B.buff(), C.buff());
      });
    }
  } else if (A.dims.size() == 2 && B.dims.size() == 1) {
    assert(C.dims.size() == 1);
    assert(C.dims[0] == A.dims[0]);
    assert(B.dims[0] == A.dims[1]);
    if (dev == device_type::cpu) {
      nn::stream::global.cpu_dispatch([=] {
        nn::cpu::gemv(A.data(), B.data(), C.data(), A.rshape()[0], A.rshape()[1]);
      });
    } else if (dev == device_type::accelerate) {
      nn::stream::global.cpu_dispatch([=] {
        nn::accelerate::gemv(A.data(), B.data(), C.data(), A.dims[0], A.dims[1]);
      });
    } else if (dev == device_type::gpu) {
      nn::stream::global.gpu_dispatch([=] {
        nn::gpu::gemv(stream::global.cmd, A.buff(), B.buff(), C.buff(), A.rshape()[0], A.rshape()[1]);
      });
    }
  } else {
    assert(false);
  }
}

void add(const data_t& A, const data_t& B, data_t& C, float a, float b, device_type dev)
{
  if (A.dims.size() == B.dims.size()) {
    assert(A.dims.size() == C.dims.size());
    if (dev == device_type::cpu || dev == device_type::accelerate) {
      auto axpbyFn = dev == device_type::accelerate ? nn::accelerate::axpby : nn::cpu::axpby;
      nn::stream::global.cpu_dispatch([=] {
        axpbyFn(A.data(), B.data(), C.data(), A.rsize(), a, b, 0.0);
      });
    } else if (dev == device_type::gpu) {
      nn::stream::global.gpu_dispatch([=] {
        nn::gpu::axpby(stream::global.cmd, A.buff(), B.buff(), C.buff(), a, b, 0.0);
      });
    }
  } else if (B.dims.size() == 1) {
    assert(A.dims.size() == C.dims.size());
    if (B.dims[0] == 1) {
      if (dev == device_type::cpu || dev == device_type::accelerate) {
        auto axpbyFn = dev == device_type::accelerate ? nn::accelerate::axpby : nn::cpu::axpby;
        nn::stream::global.cpu_dispatch([=] {
          axpbyFn(A.data(), A.data(), C.data(), A.rsize(), a, 0.0, *B.data() * b);
        });
      } else if (dev == device_type::gpu) {
        nn::stream::global.gpu_dispatch([=] {
          nn::gpu::axpby(stream::global.cmd, A.buff(), A.buff(), C.buff(), a, 0.0, *B.data() * b);
        });
      }
    } else {
      assert(B.dims[0] == A.dims[0]);
      assert(A.dims.size() == 2);
      if (dev == device_type::cpu || dev == device_type::accelerate) {
        auto axpby2dBcolFn = dev == device_type::accelerate ? nn::accelerate::axpby2dBcol : nn::cpu::axpby2dBcol;
        nn::stream::global.cpu_dispatch([=] {
          axpby2dBcolFn(A.data(), B.data(), C.data(), A.rsize(), B.rsize(), a, b, 0.0f);
        });
      } else if (dev == device_type::gpu) {
        nn::stream::global.gpu_dispatch([=] {
          nn::gpu::axpby2dBcol(stream::global.cmd, A.buff(), B.buff(), C.buff(), a, b, 0.0f);
        });
      }
    }
  } else if (A.dims.size() == 1) {
    assert(B.dims.size() == C.dims.size());
    assert(A.dims[0] == 1);
    if (dev == device_type::cpu || dev == device_type::accelerate) {
      auto axpbyFn = dev == device_type::accelerate ? nn::accelerate::axpby : nn::cpu::axpby;
      nn::stream::global.cpu_dispatch([=] {
        axpbyFn(B.data(), B.data(), C.data(), C.rsize(), 0.0, b, *A.data() * a);
      });
    } else if (dev == device_type::gpu) {
      nn::stream::global.gpu_dispatch([=] {
        nn::gpu::axpby(stream::global.cmd, B.buff(), B.buff(), C.buff(), 0.0, b, *A.data() * a);
      });
    }
  } else {
    assert(false);
  }
}

void sub(const data_t& A, const data_t& B, data_t& C, device_type dev)
{
  nn::tensor::add(A, B, C, 1.0, -1.0, dev);
}

void mul(const data_t& A, const data_t& B, data_t& C, device_type dev)
{
  if (B.dims.size() == 1 && B.size() == 1) {
    assert(A.size() == C.size());
    assert(A.dims.size() == C.dims.size());
    if (dev == device_type::cpu || dev == device_type::accelerate) {
      auto axpbyFn = dev == device_type::accelerate ? nn::accelerate::axpby : nn::cpu::axpby;
      nn::stream::global.cpu_dispatch([=] {
        axpbyFn(A.data(), A.data(), C.data(), A.rsize(), B.data()[0], 0.0, 0.0);
      });
    } else if (dev == device_type::gpu) {
      nn::stream::global.gpu_dispatch([=] {
        nn::gpu::axpby(stream::global.cmd, A.buff(), A.buff(), C.buff(), B.data()[0], 0.0, 0.0);
      });
    }
  } else if (A.dims.size() == B.dims.size()) {
    assert(A.dims.size() == C.dims.size());
    assert(A.size() == C.size());
    assert(A.size() == B.size());
    assert(A.dims == B.dims);
    assert(A.dims == C.dims);
    if (dev == device_type::cpu || dev == device_type::accelerate) {
      auto addcmulFn = dev == device_type::accelerate ? nn::accelerate::addcmul : nn::cpu::addcmul;
      nn::stream::global.cpu_dispatch([=] {
        addcmulFn(A.data(), B.data(), C.data(), A.rsize(), 1.0, 0.0);
      });
    } else if (dev == device_type::gpu) {
      nn::stream::global.gpu_dispatch([=] {
        nn::gpu::addcmul(stream::global.cmd, A.buff(), B.buff(), C.buff(), 1.0, 0.0);
      });
    }
  } else {
    assert(false);
  }
}

void div(const data_t& A, const data_t& B, data_t& C, device_type dev)
{
  if (B.dims.size() == 1 && B.size() == 1) {
    assert(A.size() == C.size());
    assert(A.dims.size() == C.dims.size());
    if (dev == device_type::cpu || dev == device_type::accelerate) {
      auto axpbyFn = dev == device_type::accelerate ? nn::accelerate::axpby : nn::cpu::axpby;
      nn::stream::global.cpu_dispatch([=] {
        axpbyFn(A.data(), A.data(), C.data(), A.rsize(), 1.0f / B.data()[0], 0.0, 0.0);
      });
    } else if (dev == device_type::gpu) {
      nn::stream::global.gpu_dispatch([=] {
        nn::gpu::axpby(stream::global.cmd, A.buff(), A.buff(), C.buff(), 1.0f / B.data()[0], 0.0, 0.0);
      });
    }
  } else if (A.dims.size() == B.dims.size()) {
    assert(A.dims.size() == C.dims.size());
    assert(A.size() == C.size());
    assert(A.size() == B.size());
    if (dev == device_type::cpu || dev == device_type::accelerate) {
      auto addcdivFn = dev == device_type::accelerate ? nn::accelerate::addcdiv : nn::cpu::addcdiv;
      nn::stream::global.cpu_dispatch([=] {
        addcdivFn(A.data(), B.data(), C.data(), A.rsize(), 1.0, 0.0);
      });
    } else if (dev == device_type::gpu) {
      nn::stream::global.gpu_dispatch([=] {
        nn::gpu::addcdiv(stream::global.cmd, A.buff(), B.buff(), C.buff(), 1.0, 0.0);
      });
    }
  } else {
    assert(false);
  }
}

void sigmoid(const data_t& A, data_t& C, device_type dev)
{
  assert(A.size() == C.size());
  if (dev == device_type::cpu || dev == device_type::accelerate) {
    auto sigmoidFn = dev == device_type::accelerate ? nn::accelerate::sigmoid : nn::cpu::sigmoid;
    nn::stream::global.cpu_dispatch([=] {
      sigmoidFn(A.data(), C.data(), A.rsize());
    });
  } else if (dev == device_type::gpu) {
    nn::stream::global.gpu_dispatch([=] {
      nn::gpu::sigmoid(nn::stream::global.cmd, A.buff(), C.buff());
    });
  }
}

void sigmoidDerivative(const data_t& A, data_t& C, device_type dev)
{
  assert(A.size() == C.size());
  if (dev == device_type::cpu || dev == device_type::accelerate) {
    auto sigmoidDerivFn = dev == device_type::accelerate ? nn::accelerate::sigmoidDerivative : nn::cpu::sigmoidDerivative;
    nn::stream::global.cpu_dispatch([=] {
      sigmoidDerivFn(A.data(), C.data(), A.rsize());
    });
  } else if (dev == device_type::gpu) {
    nn::stream::global.gpu_dispatch([=] {
      nn::gpu::sigmoidDerivative(nn::stream::global.cmd, A.buff(), C.buff());
    });
  }
}

void sum(const data_t& A, data_t& C, int64_t dim, device_type dev)
{
  if (dim == -1) {
    assert(C.dims.size() == 1);
    assert(C.dims[0] == 1);
    if (dev == device_type::cpu || dev == device_type::accelerate) {
      auto sumFn = dev == device_type::accelerate ? nn::accelerate::sum : nn::cpu::sum;
      nn::stream::global.cpu_dispatch([=] {
        sumFn(A.data(), C.data(), A.rsize());
      });
    } else if (dev == device_type::gpu) {
      nn::stream::global.gpu_dispatch([=] {
        nn::gpu::sum(stream::global.cmd, A.buff(), 1.0, C.buff());
      });
    }
  } else {
    assert(A.dims.size() == 2);
    assert(C.dims.size() == 1);
    if (dim == 0) {
      assert(C.dims[0] == A.dims[1]);
      if (dev == device_type::cpu || dev == device_type::accelerate) {
        auto sumDim0Fn = dev == device_type::accelerate ? nn::accelerate::sum_dim0 : nn::cpu::sum_dim0;
        nn::stream::global.cpu_dispatch([=] {
          sumDim0Fn(A.data(), C.data(), A.dims[0], A.dims[1], A.rshape()[1]);
        });
      } else if (dev == device_type::gpu) {
        nn::stream::global.gpu_dispatch([=] {
          nn::gpu::sum_dim0(stream::global.cmd, A.buff(), C.buff(),
                            (uint32_t)A.dims[0], (uint32_t)A.dims[1],
                            (uint32_t)A.rshape()[1]);
        });
      }
    } else if (dim == 1) {
      assert(C.dims[0] == A.dims[0]);
      if (dev == device_type::cpu || dev == device_type::accelerate) {
        auto sumDim1Fn = dev == device_type::accelerate ? nn::accelerate::sum_dim1 : nn::cpu::sum_dim1;
        nn::stream::global.cpu_dispatch([=] {
          sumDim1Fn(A.data(), C.data(), A.dims[0], A.dims[1], A.rshape()[1]);
        });
      } else if (dev == device_type::gpu) {
        nn::stream::global.gpu_dispatch([=] {
          nn::gpu::sum_dim1(stream::global.cmd, A.buff(), C.buff(),
                            (uint32_t)A.dims[0], (uint32_t)A.dims[1],
                            (uint32_t)A.rshape()[1]);
        });
      }
    } else {
      assert(false);
    }
  }
}

void transpose(const data_t& A, data_t& C, device_type dev)
{
  assert(A.dims.size() == 2);
  assert(A.dims.size() == C.dims.size());
  assert(A.dims[0] == C.dims[1]);
  assert(A.dims[1] == C.dims[0]);
  auto rshape = A.rshape();
  if (dev == device_type::cpu) {
    nn::stream::global.cpu_dispatch([=]() {
      nn::cpu::transpose<1>(A.xs.data, C.xs.data, rshape[0], rshape[1]);
    });
  } else if (dev == device_type::accelerate) {
    nn::stream::global.cpu_dispatch([=]() {
      nn::accelerate::transpose(A.xs.data, C.xs.data, rshape[0], rshape[1]);
    });
  } else if (dev == device_type::gpu) {
    nn::stream::global.gpu_dispatch([=]() {
      nn::gpu::transpose(stream::global.cmd, A.buff(), C.buff(),
                         (uint32_t)rshape[0], (uint32_t)rshape[1]);
    });
  }

}

}

namespace nn::tensor {
void data_t::transpose(device_type dev) {
  assert(dims.size() == 2);

  std::vector<int64_t> newDims = {dims[1], dims[0]};
  auto tself = tensor::data_t(newDims, allocator::aligned_alloc(newDims));
  tensor::transpose(*this, tself, dev);
  xs = tself.xs;
  dims = tself.dims;
}

void data_t::flatten() {
  if (dims.size() <= 1) {
    this->dims = {size()};
    return;
  }
  auto newDims = std::vector<int64_t>{size()};
  auto newStorage = allocator::aligned_alloc(newDims);
  int64_t destIdx = 0;
  rowsIter([&](int64_t srcIdx) {
    newStorage.data[destIdx++] = xs.data[srcIdx];
  });
  this->dims = newDims;
  this->xs = newStorage;
}

data_t data_t::copy(tensor::device_type dev) const {
  auto storage = allocator::aligned_alloc(dims);
  data_t out{dims, storage};
  auto srcBuffer = xs;
  if (dev == tensor::device_type::cpu || dev == tensor::device_type::accelerate) {
    nn::stream::global.cpu_dispatch([=]() {
      memcpy(storage.data, srcBuffer.data, storage.size());
    });
  } else {
    nn::stream::global.gpu_dispatch([=]() {
      id<MTLBlitCommandEncoder> blit = [nn::stream::global.cmd blitCommandEncoder];
      [blit copyFromBuffer:srcBuffer.buff() sourceOffset:0 toBuffer:storage.buff() destinationOffset:0 size:storage.size()];
      [blit endEncoding];
    });
  }
  return out;
}
}

namespace nn::helpers {
const tensor::data_t forward(std::vector<std::unique_ptr<nn::layer::base>>& model, tensor::data_t& input, tensor::device_type dev)
{
  auto output = input;
  for (auto& l : model) {
    output = l->forward(output, dev);
  }
  return output;
}

template<typename dims_t>
// 784 100 10
std::vector<std::unique_ptr<nn::layer::base>> buildModel(dims_t dims)
{
  std::vector<std::unique_ptr<nn::layer::base>> model;
  for (uint i = 0; i < dims.size() - 1; i++) {
    model.emplace_back(std::make_unique<nn::layer::linear>(dims.at(i), dims.at(i + 1)));
    model.emplace_back(std::make_unique<nn::layer::sigmoid>(dims.at(i + 1)));
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
    s = alignment * ((s + (alignment - 1)) / alignment);
  }
  
  auto area = utils::area(realShape);
  if (true) {
    auto mtlBuff = [gpu::device newBufferWithLength:area * sizeof(float) options:MTLResourceStorageModeShared];
    memset(mtlBuff.contents, 0, area * sizeof(float));
    return Buffer { mtlBuff, realShape };
  } else {
    std::shared_ptr<float[]> sptr{new float[area]{0.0f}};
    return Buffer { sptr, realShape };
  }
}

void free(Buffer buff)
{
  // delete[] buff.data;
}
}

namespace nn::cost {
tensor::data_t quadratic(
                         std::vector<std::unique_ptr<nn::layer::base>>& model,
                         tensor::data_t& inputs,
                         const tensor::data_t& outputs,
                         tensor::data_t* backwardInput,
                         tensor::device_type dev
                         )
{
  const auto modelOutput = nn::helpers::forward(model, inputs, dev);
  
  auto diff = nn::tensor::data_t::zero(modelOutput.dims);
  auto cost = nn::tensor::data_t::value({0.0});
  auto diffSquared = diff;
  
  nn::tensor::sub(modelOutput, outputs, diff, dev);
  if (backwardInput) {
    *backwardInput = diff;
    diffSquared = nn::tensor::data_t{diffSquared.dims};
  }
  nn::tensor::mul(diff, diff, diffSquared, dev);
  
  nn::tensor::sum(diffSquared, cost, -1, dev);
  nn::tensor::div(cost, nn::tensor::data_t::value({(float)2 * outputs.dims[1]}), cost, tensor::device_type::cpu);
  // nn::stream::global.gpuFlush();
  
  return cost;
}
}

namespace nn::stream {


inline void ctx_t::gpuFlush()
{
  gpu_pending_commands_count = 0;
  [cmd commit];
  cmd = [gpu::queue commandBuffer];
}

inline void ctx_t::synchronize()
{
  gpuFlush();
  [event waitUntilSignaledValue:last_id timeoutMS:-1];
  // std::cout << "sync w8=" << event.signaledValue << " sig=" << last_id << std::endl;
  
  event = [gpu::device newSharedEvent];
  listener = [[MTLSharedEventListener alloc] initWithDispatchQueue:cpu::queue];
  last_id = 0;
}

template<typename block_t>
inline void ctx_t::cpu_dispatch(block_t block)
{
  assert(!dispatch_queue_get_label(DISPATCH_CURRENT_QUEUE_LABEL) ||
         strcmp(dispatch_queue_get_label(DISPATCH_CURRENT_QUEUE_LABEL),
                dispatch_queue_get_label(cpu::queue)) != 0
         && "cpu_dispatch called from within a cpu worker callback — this causes non-monotonic event signaling");
  auto val = last_id;
  if (lastCommandCpu.has_value() && *lastCommandCpu == false) {
    gpuFlush();
    dispatch_async(cpu::queue, ^() {
      [event waitUntilSignaledValue:val timeoutMS:-1];
    });
  }
  lastCommandCpu = true;
  dispatch_async(cpu::queue, ^() {
    block();
    event.signaledValue = val + 1;
  });
  last_id += 1;
}

inline void ctx_t::gpu_dispatch(std::function<void()> block)
{
  assert(!dispatch_queue_get_label(DISPATCH_CURRENT_QUEUE_LABEL) ||
         strcmp(dispatch_queue_get_label(DISPATCH_CURRENT_QUEUE_LABEL),
                dispatch_queue_get_label(cpu::queue)) != 0
         && "gpu_dispatch called from within a cpu worker callback — this causes non-monotonic event signaling");
  if (lastCommandCpu.has_value() && *lastCommandCpu == true) {
    [cmd encodeWaitForEvent:event value:last_id];
  }
  gpu_pending_commands_count += 1;
  lastCommandCpu = false;
  block();
  [cmd encodeSignalEvent:event value:last_id + 1];
  last_id += 1;
  if (gpu_pending_commands_count >= 20) {
    gpuFlush();
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
  assert(N % 16 == 0);
  *output = 0;
  for (int64_t i = 0; i < N; i++) {
    *output += x[i] * y[i];
  }
}

void axpby(const float* x, const float* y, float* output, int64_t N, float fX, float fY, float a)
{
  for (int64_t i = 0; i < N; i++) {
    output[i] = x[i] * fX + y[i] * fY + a;
  }
}

void sigmoid(const float* x, float* output, int64_t N)
{
  if (N % 16 == 0) {
    simd_packed_float16* simdX = (simd_packed_float16*)x;
    simd_packed_float16* simdOut = (simd_packed_float16*)output;
    for (int64_t i = 0; i < (N / 16); i++) {
      simdOut[i] = 1.0 / (1.0 + simd::exp(-simdX[i]));
    }
  } else if (N % 8 == 0) {
    simd_packed_float8* simdX = (simd_packed_float8*)x;
    simd_packed_float8* simdOut = (simd_packed_float8*)output;
    for (int64_t i = 0; i < (N / 8); i++) {
      simdOut[i] = 1.0 / (1.0 + simd::exp(-simdX[i]));
    }
  } else if (N % 4 == 0) {
    simd_packed_float4* simdX = (simd_packed_float4*)x;
    simd_packed_float4* simdOut = (simd_packed_float4*)output;
    for (int64_t i = 0; i < (N / 4); i++) {
      simdOut[i] = 1.0 / (1.0 + simd::exp(-simdX[i]));
    }
  } else if (N % 2 == 0) {
    simd_packed_float2* simdX = (simd_packed_float2*)x;
    simd_packed_float2* simdOut = (simd_packed_float2*)output;
    for (int64_t i = 0; i < (N / 2); i++) {
      simdOut[i] = 1.0 / (1.0 + simd::exp(-simdX[i]));
    }
  } else {
    for (int64_t i = 0; i < N; i++) {
      output[i] = 1.0 / (1.0 + exp(-x[i]));
    }
  }
}

void sigmoidDerivative(const float* x, float* output, int64_t N)
{
  if (N % 16 == 0) {
    simd_packed_float16* simdX = (simd_packed_float16*)x;
    simd_packed_float16* simdOut = (simd_packed_float16*)output;
    for (int64_t i = 0; i < (N / 16); i++) {
      simd_packed_float16 s = 1.0 / (1.0 + simd::exp(-simdX[i]));
      simdOut[i] = s * (1.0 - s);
    }
  } else if (N % 8 == 0) {
    simd_packed_float8* simdX = (simd_packed_float8*)x;
    simd_packed_float8* simdOut = (simd_packed_float8*)output;
    for (int64_t i = 0; i < (N / 8); i++) {
      simd_packed_float8 s = 1.0 / (1.0 + simd::exp(-simdX[i]));
      simdOut[i] = s * (1.0 - s);
    }
  } else if (N % 4 == 0) {
    simd_packed_float4* simdX = (simd_packed_float4*)x;
    simd_packed_float4* simdOut = (simd_packed_float4*)output;
    for (int64_t i = 0; i < (N / 4); i++) {
      simd_packed_float4 s = 1.0 / (1.0 + simd::exp(-simdX[i]));
      simdOut[i] = s * (1.0 - s);
    }
  } else if (N % 2 == 0) {
    simd_packed_float2* simdX = (simd_packed_float2*)x;
    simd_packed_float2* simdOut = (simd_packed_float2*)output;
    for (int64_t i = 0; i < (N / 2); i++) {
      simd_packed_float2 s = 1.0 / (1.0 + simd::exp(-simdX[i]));
      simdOut[i] = s * (1.0 - s);
    }
  } else {
    for (int64_t i = 0; i < N; i++) {
      float s = 1.0 / (1.0 + exp(-x[i]));
      output[i] = s * (1.0 - s);
    }
  }
}

void axpy(const float* x, const float* y, float* output, int64_t N, float a)
{
  for (int64_t i = 0; i < N; i++) {
    output[i] = x[i] * a + y[i];
  }
}

void addcmul(const float* x, const float* y, float* output, int64_t N, float a, float b)
{
  for (int64_t i = 0; i < N; i++) {
    output[i] = x[i] * y[i] * a + b;
  }
}

void addcdiv(const float* x, const float* y, float* output, int64_t N, float a, float b)
{
  for (int64_t i = 0; i < N; i++) {
    output[i] = x[i] / y[i] * a + b;
  }
}

void axpby2dBcol(const float* x, const float* y, float* output, int64_t N, int64_t NB,
                 float fX, float fY, float a)
{
  int64_t strideB = N / NB;
  for (int64_t i = 0; i < N; i++) {
    output[i] = x[i] * fX + y[i / strideB] * fY + a;
  }
}

void sum(const float* x, float* output, int64_t N)
{
  float acc = 0;
  int64_t i = 0;
  for (; i + 16 <= N; i += 16) {
    acc += simd_dot(*(simd_packed_float16*)(x + i), 1.0f);
  }
  for (; i + 8 <= N; i += 8) {
    acc += simd_dot(*(simd_packed_float8*)(x + i), 1.0f);
  }
  for (; i + 4 <= N; i += 4) {
    acc += simd_dot(*(simd_packed_float4*)(x + i), 1.0f);
  }
  for (; i < N; i++) {
    acc += x[i];
  }
  *output = acc;
}

void sum_dim0(const float* x, float* output, int64_t Nrows, int64_t Ncols, int64_t stride)
{
  for (int64_t row = 0; row < Nrows; row++) {
    for (int64_t col = 0; col < Ncols; col++) {
      output[col] += x[row * stride + col];
    }
  }
}

void sum_dim1(const float* x, float* output, int64_t Nrows, int64_t Ncols, int64_t stride)
{
  for (int64_t row = 0; row < Nrows; row++) {
    for (int64_t col = 0; col < Ncols; col++) {
      output[row] += x[row * stride + col];
    }
  }
}
}

namespace nn::accelerate {

void gemm(const float* A, const float* B, float* C, int64_t M, int64_t N, int64_t P, bool transposeB)
{
  cblas_sgemm(CblasRowMajor,
              CblasNoTrans, transposeB ? CblasTrans : CblasNoTrans,
              (int)M, (int)P, (int)N,
              1.0f, A, (int)N, B, transposeB ? (int)N : (int)P,
              0.0f, C, (int)P);
}

void gemv(const float* mat, const float* vec, float* output, uint64_t M, uint64_t N)
{
  cblas_sgemv(CblasRowMajor, CblasNoTrans,
              (int)M, (int)N,
              1.0f, mat, (int)N, vec, 1,
              0.0f, output, 1);
}

void dot(const float* x, const float* y, float* output, int64_t N)
{
  *output = cblas_sdot((int)N, x, 1, y, 1);
}

void axpby(const float* x, const float* y, float* output, int64_t N, float fX, float fY, float a)
{
  if (output != x && output != y) {
    memset(output, 0, N * sizeof(float));
    catlas_saxpby((int)N, fX, x, 1, 0.0f, output, 1);
    cblas_saxpy((int)N, fY, y, 1, output, 1);
  } else if (output == y) {
    catlas_saxpby((int)N, fX, x, 1, fY, output, 1);
  } else {
    // output == x
    catlas_saxpby((int)N, fY, y, 1, fX, output, 1);
  }
  if (a != 0.0f) {
    vDSP_vsadd(output, 1, &a, output, 1, (vDSP_Length)N);
  }
}

void axpby2dBcol(const float* x, const float* y, float* output, int64_t N, int64_t NB,
                 float fX, float fY, float a)
{
  // No direct Accelerate equivalent for broadcast pattern
  nn::cpu::axpby2dBcol(x, y, output, N, NB, fX, fY, a);
}

void axpy(const float* x, const float* y, float* output, int64_t N, float a)
{
  if (output != y) {
    memcpy(output, y, N * sizeof(float));
  }
  cblas_saxpy((int)N, a, x, 1, output, 1);
}

void addcmul(const float* x, const float* y, float* output, int64_t N, float a, float b)
{
  vDSP_vmul(x, 1, y, 1, output, 1, (vDSP_Length)N);
  if (a != 1.0f) {
    vDSP_vsmul(output, 1, &a, output, 1, (vDSP_Length)N);
  }
  if (b != 0.0f) {
    vDSP_vsadd(output, 1, &b, output, 1, (vDSP_Length)N);
  }
}

void addcdiv(const float* x, const float* y, float* output, int64_t N, float a, float b)
{
  // vDSP_vdiv: note reversed arg order — divides B by A
  vDSP_vdiv(y, 1, x, 1, output, 1, (vDSP_Length)N);
  if (a != 1.0f) {
    vDSP_vsmul(output, 1, &a, output, 1, (vDSP_Length)N);
  }
  if (b != 0.0f) {
    vDSP_vsadd(output, 1, &b, output, 1, (vDSP_Length)N);
  }
}

void sigmoid(const float* x, float* output, int64_t N)
{
  // No direct vDSP sigmoid; reuse SIMD implementation
  nn::cpu::sigmoid(x, output, N);
}

void sigmoidDerivative(const float* x, float* output, int64_t N)
{
  nn::cpu::sigmoidDerivative(x, output, N);
}

void sum(const float* x, float* output, int64_t N)
{
  vDSP_sve(x, 1, output, (vDSP_Length)N);
}

void sum_dim0(const float* x, float* output, int64_t Nrows, int64_t Ncols, int64_t stride)
{
  memcpy(output, x, Ncols * sizeof(float));
  for (int64_t row = 1; row < Nrows; row++) {
    vDSP_vadd(x + row * stride, 1, output, 1, output, 1, (vDSP_Length)Ncols);
  }
}

void sum_dim1(const float* x, float* output, int64_t Nrows, int64_t Ncols, int64_t stride)
{
  for (int64_t row = 0; row < Nrows; row++) {
    vDSP_sve(x + row * stride, 1, &output[row], (vDSP_Length)Ncols);
  }
}

void transpose(const float* m, float* mT, int64_t M, int64_t N)
{
  vDSP_mtrans(m, 1, mT, 1, (vDSP_Length)N, (vDSP_Length)M);
}
}

namespace nn::gpu {
void gemm(id<MTLCommandBuffer> cmd, id<MTLBuffer> A, id<MTLBuffer> B, id<MTLBuffer> C, uint64_t M, uint64_t N, uint64_t P)
{
  int dim = -1;
  int simdim = 1;
  bool aligned = false;
  
  aligned = M % 8 == 0 && P % 8 == 0 && N % 8 == 0;
  if (aligned) {
    if (M % 8 == 0 && P % 8 == 0) dim = 1;
    if (M % 16 == 0 && P % 16 == 0) dim = 2;
    if (M % 32 == 0 && P % 32 == 0) dim = 4;
    
    if (M % (dim * 8 * 2) == 0) simdim = 2;
  } else {
    dim = 1;
    if (M > 16 && P > 16) dim = 2;
    if (M > 32 && P > 32) dim = 4;
    
    // simdim = 2 is slower
    //    if (M > (dim * 8 * 2)) simdim = 2;
  }
  
  static NSMutableArray* aligned_kernels = [NSMutableArray new];
  static NSMutableArray* noaligned_kernels = [NSMutableArray new];
  static dispatch_once_t onceToken;
  dispatch_once(&onceToken, ^{
    auto func1 = [gpu::lib newFunctionWithName:@"sgemm_unrolled_dim1"];
    auto func2 = [gpu::lib newFunctionWithName:@"sgemm_unrolled_dim2"];
    auto func4 = [gpu::lib newFunctionWithName:@"sgemm_unrolled_dim4"];
    [aligned_kernels addObject:[gpu::device newComputePipelineStateWithFunction:func1 error:nil]];
    [aligned_kernels addObject:[gpu::device newComputePipelineStateWithFunction:func2 error:nil]];
    [aligned_kernels addObject:[gpu::device newComputePipelineStateWithFunction:func4 error:nil]];
    
    auto func0 = [gpu::lib newFunctionWithName:@"sgemm"];
    func1 = [gpu::lib newFunctionWithName:@"sgemm_na_unrolled_dim1"];
    func2 = [gpu::lib newFunctionWithName:@"sgemm_na_unrolled_dim2"];
    func4 = [gpu::lib newFunctionWithName:@"sgemm_na_unrolled_dim4"];
    [noaligned_kernels addObject:[gpu::device newComputePipelineStateWithFunction:func1 error:nil]];
    [noaligned_kernels addObject:[gpu::device newComputePipelineStateWithFunction:func2 error:nil]];
    [noaligned_kernels addObject:[gpu::device newComputePipelineStateWithFunction:func4 error:nil]];
  });
  if (aligned_kernels.count <= 0 || noaligned_kernels.count <= 0) {
    NSLog(@"got error during pipeline creation");
    return;
  }
  
  auto xdim = dim * 8;
  auto ydim = xdim * simdim;
  MTLSize tgroupSize;
  tgroupSize.width = 32;
  tgroupSize.height = simdim;
  tgroupSize.depth = 1;
  auto encoder = [cmd computeCommandEncoder];
  
  [encoder setBuffer:A offset:0 atIndex:0];
  [encoder setBuffer:B offset:0 atIndex:1];
  [encoder setBuffer:C offset:0 atIndex:2];
  [encoder setBytes:(void*)&M length:sizeof(M) atIndex:3];
  [encoder setBytes:(void*)&N length:sizeof(N) atIndex:4];
  [encoder setBytes:(void*)&P length:sizeof(P) atIndex:5];
  if (aligned == false) {
    auto matsize = 8 * 8 * sizeof(float);
    [encoder setThreadgroupMemoryLength:(dim * simdim * matsize) atIndex:0];
    [encoder setThreadgroupMemoryLength:(dim * simdim * matsize) atIndex:1];
    [encoder setThreadgroupMemoryLength:(dim * dim * simdim * matsize) atIndex:2];
  }
  [encoder setComputePipelineState:(aligned ? aligned_kernels : noaligned_kernels)[(int)log2(dim)]];
  [encoder dispatchThreadgroups:MTLSizeMake((P + xdim - 1) / xdim, (M + ydim - 1) / ydim, 1) threadsPerThreadgroup:tgroupSize];
  
  [encoder endEncoding];
}

void gemv(id<MTLCommandBuffer> cmd, id<MTLBuffer> mat, id<MTLBuffer> vec, id<MTLBuffer> output,
          uint64_t m, uint64_t n)
{
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
  
  [encoder dispatchThreadgroups:MTLSizeMake(1, (H + tgroupSize.height - 1) / tgroupSize.height, 1) threadsPerThreadgroup:tgroupSize];
  [encoder endEncoding];
}

void dot(id<MTLCommandBuffer> cmd, id<MTLBuffer> x, id<MTLBuffer> y, id<MTLBuffer> output)
{
  assert(x.length == y.length);
  const uint64_t N = x.length / sizeof(float);
  assert(N % 4 == 0);
  
  static id<MTLComputePipelineState> kernel0;
  static dispatch_once_t onceToken;
  dispatch_once(&onceToken, ^{
    auto kernel0Func = [gpu::lib newFunctionWithName:@"dot"];
    kernel0 = [gpu::device newComputePipelineStateWithFunction:kernel0Func error:nil];
  });
  if (!kernel0) {
    NSLog(@"got error during pipeline creation");
    return;
  }
  
  auto totalThreads = (N + 7) / 8;
  auto threadsPerThreadgroup = MTLSizeMake(totalThreads > 1024 ? 1024 : 1 << ((int)ceil(log2(totalThreads))), 1, 1);
  auto threadgroupMemFloats = threadsPerThreadgroup.width * 2;
  auto threadgroupsWidth = (totalThreads + threadsPerThreadgroup.width - 1) / threadsPerThreadgroup.width;
  auto threadgroups = MTLSizeMake(threadgroupsWidth, 1, 1);
  
  auto tgroupmemSize = threadgroupMemFloats * sizeof(float);
  tgroupmemSize = ((tgroupmemSize + 15) / 16) * 16;
  auto encoder = [cmd computeCommandEncoder];
  [encoder setBuffer:x offset:0 atIndex:0];
  [encoder setBuffer:y offset:0 atIndex:1];
  [encoder setBuffer:output offset:0 atIndex:2];
  [encoder setBytes:(void*)&N length:sizeof(N) atIndex:3];
  [encoder setThreadgroupMemoryLength:tgroupmemSize atIndex:0];
  [encoder setComputePipelineState:kernel0];
  [encoder dispatchThreadgroups:threadgroups threadsPerThreadgroup:threadsPerThreadgroup];
  
  [encoder endEncoding];
}

void sum(id<MTLCommandBuffer> cmd, id<MTLBuffer> X, float y, id<MTLBuffer> output)
{
  const uint64_t N = X.length / sizeof(float);
  
  static id<MTLComputePipelineState> kernel0;
  static dispatch_once_t onceToken;
  dispatch_once(&onceToken, ^{
    auto kernel0Func = [gpu::lib newFunctionWithName:@"sum"];
    kernel0 = [gpu::device newComputePipelineStateWithFunction:kernel0Func error:nil];
  });
  if (!kernel0) {
    NSLog(@"got error during pipeline creation");
    return;
  }
  
  auto totalThreads = (N + 7) / 8;
  auto threadsPerThreadgroup = MTLSizeMake(totalThreads > 1024 ? 1024 : 1 << ((int)ceil(log2(totalThreads))), 1, 1);
  auto threadgroupMemFloats = threadsPerThreadgroup.width * 2;
  auto threadgroupsWidth = (totalThreads + threadsPerThreadgroup.width - 1) / threadsPerThreadgroup.width;
  auto threadgroups = MTLSizeMake(threadgroupsWidth, 1, 1);
  
  auto encoder = [cmd computeCommandEncoder];
  auto tgroupmemSize = threadgroupMemFloats * sizeof(float);
  tgroupmemSize = ((tgroupmemSize + 15) / 16) * 16;
  
  [encoder setBuffer:X offset:0 atIndex:0];
  [encoder setBytes:(void*)&y length:sizeof(y) atIndex:1];
  [encoder setBuffer:output offset:0 atIndex:2];
  [encoder setBytes:(void*)&N length:sizeof(N) atIndex:3];
  [encoder setThreadgroupMemoryLength:tgroupmemSize atIndex:0];
  [encoder setComputePipelineState:kernel0];
  [encoder dispatchThreadgroups:threadgroups threadsPerThreadgroup:threadsPerThreadgroup];
  
  [encoder endEncoding];
}

void sigmoid(id<MTLCommandBuffer> cmd, id<MTLBuffer> X, id<MTLBuffer> output)
{
  static id<MTLComputePipelineState> kernel;
  static dispatch_once_t onceToken;
  dispatch_once(&onceToken, ^{
    auto kernelFunc = [gpu::lib newFunctionWithName:@"sigmoid"];
    kernel = [gpu::device newComputePipelineStateWithFunction:kernelFunc error:nil];
  });
  if (!kernel) { NSLog(@"got error during pipeline creation"); return; }
  
  assert(X.length == output.length);
  const uint32_t N = ((uint32_t)X.length) / sizeof(float);
  auto encoder = [cmd computeCommandEncoder];
  
  [encoder setBuffer:X offset:0 atIndex:0];
  [encoder setBuffer:output offset:0 atIndex:1];
  [encoder setBytes:(void*)&N length:sizeof(N) atIndex:2];
  [encoder setComputePipelineState:kernel];
  [encoder dispatchThreads:MTLSizeMake(N / 2, 1, 1) threadsPerThreadgroup:MTLSizeMake(32 * 2, 1, 1)];
  
  [encoder endEncoding];
}

void sigmoidDerivative(id<MTLCommandBuffer> cmd, id<MTLBuffer> X, id<MTLBuffer> output)
{
  static id<MTLComputePipelineState> kernel;
  static dispatch_once_t onceToken;
  dispatch_once(&onceToken, ^{
    auto kernelFunc = [gpu::lib newFunctionWithName:@"sigmoidDerivative"];
    kernel = [gpu::device newComputePipelineStateWithFunction:kernelFunc error:nil];
  });
  if (!kernel) { NSLog(@"got error during pipeline creation"); return; }
  
  assert(X.length == output.length);
  const uint32_t N = ((uint32_t)X.length) / sizeof(float);
  auto encoder = [cmd computeCommandEncoder];
  
  [encoder setBuffer:X offset:0 atIndex:0];
  [encoder setBuffer:output offset:0 atIndex:1];
  [encoder setBytes:(void*)&N length:sizeof(N) atIndex:2];
  [encoder setComputePipelineState:kernel];
  [encoder dispatchThreads:MTLSizeMake(N / 2, 1, 1) threadsPerThreadgroup:MTLSizeMake(32 * 2, 1, 1)];
  
  [encoder endEncoding];
}

void axpby(id<MTLCommandBuffer> cmd, id<MTLBuffer> X, id<MTLBuffer> Y, id<MTLBuffer> output, float fX, float fY, float A)
{
  static id<MTLComputePipelineState> kernel1;
  static id<MTLComputePipelineState> kernel2;
  static dispatch_once_t onceToken;
  dispatch_once(&onceToken, ^{
    auto kernelFunc1 = [gpu::lib newFunctionWithName:@"axpby1"];
    auto kernelFunc2 = [gpu::lib newFunctionWithName:@"axpby2"];
    kernel1 = [gpu::device newComputePipelineStateWithFunction:kernelFunc1 error:nil];
    kernel2 = [gpu::device newComputePipelineStateWithFunction:kernelFunc2 error:nil];
  });
  if (!kernel1 || !kernel2) {
    NSLog(@"got error during pipeline creation");
    return;
  }
  
  if (X) assert(X.length == output.length);
  if (Y) assert(Y.length == output.length);
  
  const uint32_t N = ((uint32_t)X.length) / sizeof(float);
  auto encoder = [cmd computeCommandEncoder];
  
  [encoder setBuffer:X offset:0 atIndex:0];
  if (Y) {
    [encoder setBuffer:Y offset:0 atIndex:1];
    [encoder setBuffer:output offset:0 atIndex:2];
    [encoder setBytes:(void*)&N length:sizeof(N) atIndex:3];
    [encoder setBytes:(void*)&fX length:sizeof(fX) atIndex:4];
    [encoder setBytes:(void*)&fY length:sizeof(fY) atIndex:5];
    [encoder setBytes:(void*)&A length:sizeof(A) atIndex:6];
    [encoder setComputePipelineState:kernel2];
  } else {
    [encoder setBuffer:output offset:0 atIndex:1];
    [encoder setBytes:(void*)&N length:sizeof(N) atIndex:2];
    [encoder setBytes:(void*)&fX length:sizeof(fX) atIndex:3];
    [encoder setBytes:(void*)&A length:sizeof(A) atIndex:4];
    [encoder setComputePipelineState:kernel1];
  }
  [encoder dispatchThreads:MTLSizeMake(N / 2, 1, 1) threadsPerThreadgroup:MTLSizeMake(32 * 2, 1, 1)];
  
  [encoder endEncoding];
}

void addcmul(id<MTLCommandBuffer> cmd, id<MTLBuffer> X, id<MTLBuffer> Y, id<MTLBuffer> output, float a, float b)
{
  static id<MTLComputePipelineState> kernel;
  static dispatch_once_t onceToken;
  dispatch_once(&onceToken, ^{
    auto kernelFunc = [gpu::lib newFunctionWithName:@"addcmul"];
    kernel = [gpu::device newComputePipelineStateWithFunction:kernelFunc error:nil];
  });
  if (!kernel) { NSLog(@"got error during pipeline creation"); return; }
  
  assert(X.length == output.length);
  assert(Y.length == output.length);
  const uint32_t N = ((uint32_t)X.length) / sizeof(float);
  auto encoder = [cmd computeCommandEncoder];
  [encoder setBuffer:X offset:0 atIndex:0];
  [encoder setBuffer:Y offset:0 atIndex:1];
  [encoder setBuffer:output offset:0 atIndex:2];
  [encoder setBytes:(void*)&N length:sizeof(N) atIndex:3];
  [encoder setBytes:(void*)&a length:sizeof(a) atIndex:4];
  [encoder setBytes:(void*)&b length:sizeof(b) atIndex:5];
  [encoder setComputePipelineState:kernel];
  [encoder dispatchThreads:MTLSizeMake(N / 2, 1, 1) threadsPerThreadgroup:MTLSizeMake(32 * 2, 1, 1)];
  [encoder endEncoding];
}

void addcdiv(id<MTLCommandBuffer> cmd, id<MTLBuffer> X, id<MTLBuffer> Y, id<MTLBuffer> output, float a, float b)
{
  static id<MTLComputePipelineState> kernel;
  static dispatch_once_t onceToken;
  dispatch_once(&onceToken, ^{
    auto kernelFunc = [gpu::lib newFunctionWithName:@"addcdiv"];
    kernel = [gpu::device newComputePipelineStateWithFunction:kernelFunc error:nil];
  });
  if (!kernel) { NSLog(@"got error during pipeline creation"); return; }
  
  assert(X.length == output.length);
  assert(Y.length == output.length);
  const uint32_t N = ((uint32_t)X.length) / sizeof(float);
  auto encoder = [cmd computeCommandEncoder];
  [encoder setBuffer:X offset:0 atIndex:0];
  [encoder setBuffer:Y offset:0 atIndex:1];
  [encoder setBuffer:output offset:0 atIndex:2];
  [encoder setBytes:(void*)&N length:sizeof(N) atIndex:3];
  [encoder setBytes:(void*)&a length:sizeof(a) atIndex:4];
  [encoder setBytes:(void*)&b length:sizeof(b) atIndex:5];
  [encoder setComputePipelineState:kernel];
  [encoder dispatchThreads:MTLSizeMake(N / 2, 1, 1) threadsPerThreadgroup:MTLSizeMake(32 * 2, 1, 1)];
  [encoder endEncoding];
}

void axpy(id<MTLCommandBuffer> cmd, id<MTLBuffer> X, id<MTLBuffer> Y, id<MTLBuffer> output, float a)
{
  static id<MTLComputePipelineState> kernel;
  static dispatch_once_t onceToken;
  dispatch_once(&onceToken, ^{
    auto kernelFunc = [gpu::lib newFunctionWithName:@"axpy"];
    kernel = [gpu::device newComputePipelineStateWithFunction:kernelFunc error:nil];
  });
  if (!kernel) { NSLog(@"got error during pipeline creation"); return; }
  
  assert(X.length == output.length);
  assert(Y.length == output.length);
  const uint32_t N = ((uint32_t)X.length) / sizeof(float);
  auto encoder = [cmd computeCommandEncoder];
  [encoder setBuffer:X offset:0 atIndex:0];
  [encoder setBuffer:Y offset:0 atIndex:1];
  [encoder setBuffer:output offset:0 atIndex:2];
  [encoder setBytes:(void*)&N length:sizeof(N) atIndex:3];
  [encoder setBytes:(void*)&a length:sizeof(a) atIndex:4];
  [encoder setComputePipelineState:kernel];
  [encoder dispatchThreads:MTLSizeMake(N / 2, 1, 1) threadsPerThreadgroup:MTLSizeMake(32 * 2, 1, 1)];
  [encoder endEncoding];
}

void axpby2dBcol(id<MTLCommandBuffer> cmd, id<MTLBuffer> X, id<MTLBuffer> Y, id<MTLBuffer> output,
                 float fX, float fY, float A)
{
  static id<MTLComputePipelineState> kernel;
  static dispatch_once_t onceToken;
  dispatch_once(&onceToken, ^{
    auto kernelFunc = [gpu::lib newFunctionWithName:@"axpby2dBcol"];
    kernel = [gpu::device newComputePipelineStateWithFunction:kernelFunc error:nil];
  });
  if (!kernel) { NSLog(@"got error during pipeline creation"); return; }
  
  const uint32_t N = (uint32_t)(output.length / sizeof(float));
  const uint32_t strideB = (uint32_t)(output.length / Y.length);
  auto encoder = [cmd computeCommandEncoder];
  [encoder setComputePipelineState:kernel];
  [encoder setBuffer:X offset:0 atIndex:0];
  [encoder setBuffer:Y offset:0 atIndex:1];
  [encoder setBuffer:output offset:0 atIndex:2];
  [encoder setBytes:(void*)&N length:sizeof(N) atIndex:3];
  [encoder setBytes:(void*)&fX length:sizeof(fX) atIndex:4];
  [encoder setBytes:(void*)&fY length:sizeof(fY) atIndex:5];
  [encoder setBytes:(void*)&A length:sizeof(A) atIndex:6];
  [encoder setBytes:(void*)&strideB length:sizeof(strideB) atIndex:7];
  NSUInteger tg = kernel.maxTotalThreadsPerThreadgroup;
  [encoder dispatchThreads:MTLSizeMake(N, 1, 1) threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
  [encoder endEncoding];
}

void sum_dim0(id<MTLCommandBuffer> cmd, id<MTLBuffer> X, id<MTLBuffer> out,
              uint32_t Nrows, uint32_t Ncols, uint32_t stride)
{
  static id<MTLComputePipelineState> kernel;
  static dispatch_once_t onceToken;
  dispatch_once(&onceToken, ^{
    auto fn = [gpu::lib newFunctionWithName:@"sum_dim0"];
    kernel = [gpu::device newComputePipelineStateWithFunction:fn error:nil];
  });
  if (!kernel) { NSLog(@"got error during pipeline creation"); return; }
  auto enc = [cmd computeCommandEncoder];
  [enc setComputePipelineState:kernel];
  [enc setBuffer:X offset:0 atIndex:0];
  [enc setBuffer:out offset:0 atIndex:1];
  [enc setBytes:&Nrows length:sizeof(Nrows) atIndex:2];
  [enc setBytes:&Ncols length:sizeof(Ncols) atIndex:3];
  [enc setBytes:&stride length:sizeof(stride) atIndex:4];
  NSUInteger tg = kernel.maxTotalThreadsPerThreadgroup;
  [enc dispatchThreads:MTLSizeMake(Ncols, 1, 1) threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
  [enc endEncoding];
}

void sum_dim1(id<MTLCommandBuffer> cmd, id<MTLBuffer> X, id<MTLBuffer> out,
              uint32_t Nrows, uint32_t Ncols, uint32_t stride)
{
  static id<MTLComputePipelineState> kernel;
  static dispatch_once_t onceToken;
  dispatch_once(&onceToken, ^{
    auto fn = [gpu::lib newFunctionWithName:@"sum_dim1"];
    kernel = [gpu::device newComputePipelineStateWithFunction:fn error:nil];
  });
  if (!kernel) { NSLog(@"got error during pipeline creation"); return; }
  auto enc = [cmd computeCommandEncoder];
  [enc setComputePipelineState:kernel];
  [enc setBuffer:X offset:0 atIndex:0];
  [enc setBuffer:out offset:0 atIndex:1];
  [enc setBytes:&Nrows length:sizeof(Nrows) atIndex:2];
  [enc setBytes:&Ncols length:sizeof(Ncols) atIndex:3];
  [enc setBytes:&stride length:sizeof(stride) atIndex:4];
  NSUInteger tg = kernel.maxTotalThreadsPerThreadgroup;
  [enc dispatchThreads:MTLSizeMake(Nrows, 1, 1) threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
  [enc endEncoding];
}

void transpose(id<MTLCommandBuffer> cmd, id<MTLBuffer> X, id<MTLBuffer> out,
               uint32_t M, uint32_t N)
{
  static id<MTLComputePipelineState> kernel;
  static dispatch_once_t onceToken;
  dispatch_once(&onceToken, ^{
    auto fn = [gpu::lib newFunctionWithName:@"transpose"];
    kernel = [gpu::device newComputePipelineStateWithFunction:fn error:nil];
  });
  if (!kernel) { NSLog(@"got error during pipeline creation"); return; }
  const uint32_t total = M * N;
  auto enc = [cmd computeCommandEncoder];
  [enc setComputePipelineState:kernel];
  [enc setBuffer:X offset:0 atIndex:0];
  [enc setBuffer:out offset:0 atIndex:1];
  [enc setBytes:&M length:sizeof(M) atIndex:2];
  [enc setBytes:&N length:sizeof(N) atIndex:3];
  NSUInteger tg = kernel.maxTotalThreadsPerThreadgroup;
  [enc dispatchThreads:MTLSizeMake(total, 1, 1) threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
  [enc endEncoding];
}
}

#endif
