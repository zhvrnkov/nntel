#include <array>
#include <cstdio>
#include <iostream>
#include <fstream>
#include <cassert>

#define GPU
#define NN_IMPL
#include "nn.mm"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

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

  for (const auto& entry : fs::directory_iterator(path)) {
    if (fs::is_directory(entry.path())) {
      auto dirName = entry.path().filename().string();
      auto dirNameNumber = std::atoi(dirName.data());

      std::vector<float> netOutput(10);
      netOutput[dirNameNumber] = 1.0;

      for (const auto& imageEntry : fs::directory_iterator(entry.path())) {
        const auto& path = imageEntry.path();
        auto image = grayscale_image(path.string());
        input_images.push_back(image);

        output_vectors.insert(output_vectors.end(), netOutput.begin(), netOutput.end());
        output_vectors_dims.at(0) += 1;
      }
    }
  }

  return std::make_pair(nn::tensor::data_t::concat(input_images), nn::tensor::data_t::copy(output_vectors_dims, output_vectors.data()));
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
        auto weights = nn::tensor::data_t::zero({(int64_t)w_rows, (int64_t)w_cols});
        
        ifs.read(reinterpret_cast<char*>(weights.data()), weights.size() * sizeof(float));

        // Biases
        size_t b_rows, b_cols;
        ifs.read(reinterpret_cast<char*>(&b_rows), sizeof(b_rows));
        ifs.read(reinterpret_cast<char*>(&b_cols), sizeof(b_cols));
        auto biases = nn::tensor::data_t::zero({(int64_t)(b_rows * b_cols)});

        ifs.read(reinterpret_cast<char*>(biases.data()), biases.size() * sizeof(float));

        layers.push_back(nn::layer::linear{weights, biases});
    }

    return true;
}

int main()
{
  std::vector<nn::layer::linear> layers;
  load_model(layers, "/Users/vz/Developer/learn/informatics/ml/nn-sandbox/ml-logic-gates/mnist.net");

  std::cout << layers.size() << std::endl;

  auto image = test::grayscale_image("./assets/mnist_png/testing/3/1020.png");
  image.flatten();

  using namespace nn::tensor;

  nn::tensor::data_t& output = nn::helpers::forward(layers, image);

  nn::stream::global.synchronize();
  auto max_idx = 0;
  for (auto i = 1; i < output.size(); i++) {
    if (output.data()[i] > output.data()[max_idx]) {
      max_idx = i;
    }
  }

  std::cout << max_idx << std::endl;

  {
    auto samples = test::mnistData("./assets/mnist_png/training");
    samples.first.transpose();
    std::cout << "inputs = " << nn::utils::xs2str(samples.first.dims) << std::endl;
    std::cout << "outputs = " << nn::utils::xs2str(samples.second.dims) << std::endl;

    auto output = nn::helpers::forward(layers, samples.first);
    nn::stream::global.synchronize();
    std::cout << "model output = " << nn::utils::xs2str(output.dims) << std::endl;
    // output.transpose();




    // auto model = nn::helpers::buildModel(std::vector<int64_t>{784, 128, 10});
    // nn::tensor::data_t& output = nn::helpers::forward(model, image);
    // nn::stream::global.synchronize();
    // auto max_idx = 0;
    // for (auto i = 1; i < output.size(); i++) {
    //   if (output.data()[i] > output.data()[max_idx]) {
    //     max_idx = i;
    //   }
    // }
    // std::cout << max_idx << std::endl;
  }
}

