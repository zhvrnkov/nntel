#include <array>
#include <cstdio>
#include <iostream>
#include <fstream>
#include <cassert>
#include <simd/simd.h>

#define GPU
#define NN_IMPL
#include "nn.mm"

namespace test {
struct TestSample {
    nn::tensor::data_t input;
    nn::tensor::data_t output;
    
    TestSample(nn::tensor::data_t input, nn::tensor::data_t output) : input(input), output(output) {}
};
using TestData = std::vector<TestSample>;

TestData xorTestData() {
  TestData output;
  output.push_back({nn::tensor::data_t::value({0.0f, 0.0f}), nn::tensor::data_t::value({0.0f})});
  output.push_back({nn::tensor::data_t::value({1.0f, 0.0f}), nn::tensor::data_t::value({1.0f})});
  output.push_back({nn::tensor::data_t::value({0.0f, 1.0f}), nn::tensor::data_t::value({1.0f})});
  output.push_back({nn::tensor::data_t::value({1.0f, 1.0f}), nn::tensor::data_t::value({0.0f})});
  return output;
}
nn::tensor::data_t grayscale_image(std::string path)
{
  int width, height, channels;
  auto data = stbi_loadf(path.data(), &width, &height, &channels, 0);

  auto output = nn::tensor::data_t::copy({height, width}, data);

  stbi_image_free(data);
  return output;
}

std::pair<nn::tensor::data_t, nn::tensor::data_t> mnistData(const std::string path) {

  namespace fs = std::filesystem;

  std::vector<float> output_vectors;
  std::vector<int64_t> output_vectors_dims = {0, 10};
  std::vector<nn::tensor::data_t> input_images;

  // Sort directories to ensure consistent ordering
  std::vector<fs::path> directories;
  for (const auto& entry : fs::directory_iterator(path)) {
    if (fs::is_directory(entry.path())) {
      directories.push_back(entry.path());
    }
  }
  std::sort(directories.begin(), directories.end());

  for (const auto& dirPath : directories) {
    auto dirName = dirPath.filename().string();
    auto dirNameNumber = std::atoi(dirName.data());

    std::vector<float> netOutput(10);
    netOutput[dirNameNumber] = 1.0;

    // Sort files within each directory to ensure consistent ordering
    std::vector<fs::path> files;
    for (const auto& imageEntry : fs::directory_iterator(dirPath)) {
      files.push_back(imageEntry.path());
    }
    std::sort(files.begin(), files.end());

    for (const auto& imagePath : files) {
      auto image = grayscale_image(imagePath.string());
      input_images.push_back(image);

      output_vectors.insert(output_vectors.end(), netOutput.begin(), netOutput.end());
      output_vectors_dims.at(0) += 1;
    }
  }

  auto output = std::make_pair(nn::tensor::data_t::concat(input_images), nn::tensor::data_t::copy(output_vectors_dims, output_vectors.data()));
  std::cout << nn::utils::xs2str(output.first.dims) << std::endl;
  return output;
}
}


bool load_model(std::vector<nn::layer::linear>& layers, const std::string& filename) {
    std::ifstream ifs(filename, std::ios::binary);
    if (!ifs) return false;

    // Check magic
    char magic[4];
    ifs.read(magic, 4);
    if (std::string(magic, 4) != "NNET") return false;

    // Check version
    uint8_t version;
    ifs.read(reinterpret_cast<char*>(&version), 1);
    if (version != 1) return false;  // future versions can be handled

    // Read number of layers
    size_t num_layers;
    ifs.read(reinterpret_cast<char*>(&num_layers), sizeof(num_layers));

    for (uint i = 0; i < num_layers; i++) {
        // Activation
        uint8_t act;
        ifs.read(reinterpret_cast<char*>(&act), 1);

        // Weights
        size_t w_rows, w_cols;
        ifs.read(reinterpret_cast<char*>(&w_rows), sizeof(w_rows));
        ifs.read(reinterpret_cast<char*>(&w_cols), sizeof(w_cols));

        // Read weights into temporary buffer first
        std::vector<float> weight_data(w_rows * w_cols);
        ifs.read(reinterpret_cast<char*>(weight_data.data()), weight_data.size() * sizeof(float));

        // Create tensor with proper padding using copy constructor
        auto weights = nn::tensor::data_t::copy({(int64_t)w_rows, (int64_t)w_cols}, weight_data.data());

        // Biases
        size_t b_rows, b_cols;
        ifs.read(reinterpret_cast<char*>(&b_rows), sizeof(b_rows));
        ifs.read(reinterpret_cast<char*>(&b_cols), sizeof(b_cols));

        // Read biases into temporary buffer first
        std::vector<float> bias_data(b_rows * b_cols);
        ifs.read(reinterpret_cast<char*>(bias_data.data()), bias_data.size() * sizeof(float));

        // Create tensor with proper padding using copy constructor
        auto biases = nn::tensor::data_t::copy({(int64_t)(b_rows * b_cols)}, bias_data.data());

        layers.push_back(nn::layer::linear{weights, biases});
    }

    return true;
}

int main()
{
  auto device = nn::tensor::device_type::accelerate;
  auto train_loader = nn::train::data_loader{"/Users/vz/Developer/learn/informatics/ml/nntel/assets/mnist_png/training", 20, true};
  auto test_loader = nn::train::data_loader{"/Users/vz/Developer/learn/informatics/ml/nntel/assets/mnist_png/testing", -1, false};

  auto model = nn::helpers::buildModel({784, 100, 10});

  auto testdata = test_loader.nextBatch();
  test_loader.reset();
  auto cost = nn::cost::quadratic(model, testdata->first, testdata->second, nullptr, device);
  nn::stream::global.synchronize();
  std::cout << "initial cost = " << *cost.data() << std::endl;

  for (int64_t epoch = 0; epoch < 30; epoch++) @autoreleasepool {
    train_loader.reset();
    std::optional<std::pair<nn::tensor::data_t, nn::tensor::data_t>> batch;
    auto bi = nn::tensor::data_t::zero({1});
    while ((batch = train_loader.nextBatch())) @autoreleasepool {
      // std::println("{} {}", nn::utils::xs2str(batch->first.dims), nn::utils::xs2str(batch->second.dims));
      auto cost = nn::cost::quadratic(model, batch->first, batch->second, &bi, device);
      for (int64_t i = model.size() - 1; i >= 0; i--) {
        bi = model[i].backward(bi, device);
      }
      for (int64_t i = model.size() - 1; i >= 0; i--) {
        model[i].applyGrad(device);
      }
      nn::stream::global.synchronize();
      // std::cout << "cost = " << *cost.data() << std::endl;
    }
    auto cost = nn::cost::quadratic(model, testdata->first, testdata->second, nullptr, device);
    nn::stream::global.synchronize();
    std::cout << "cost = " << *cost.data() << std::endl;
  }
  return 0;
}
