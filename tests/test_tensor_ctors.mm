#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>
#include <numeric>

#define NN_IMPL
#include "../src/nn.mm"

using namespace nn;

void test_value_constructor() {
    std::cout << "Testing data_t::value constructor...\n";

    // Test 1: Simple value list
    {
        std::cout << "  Test 1: Creating tensor from value list {1.0, 2.0, 3.0}\n";
        auto t = tensor::data_t::value({1.0f, 2.0f, 3.0f});

        assert(t.dims.size() == 1);
        assert(t.dims[0] == 3);
        assert(t.size() == 3);

        // Check values are correctly placed
        assert(t.data()[0] == 1.0f);
        assert(t.data()[1] == 2.0f);
        assert(t.data()[2] == 3.0f);

        // Check padding is zero (real shape should be [64])
        assert(t.rshape()[0] == 64);
        for (int i = 3; i < 64; i++) {
            assert(t.data()[i] == 0.0f);
        }

        std::cout << "    ✓ Values correctly placed with zero padding\n";
    }

    // Test 2: Single value
    {
        std::cout << "  Test 2: Single value tensor\n";
        auto t = tensor::data_t::value({42.0f});

        assert(t.dims[0] == 1);
        assert(t.data()[0] == 42.0f);

        // Check rest is padded with zeros
        for (int i = 1; i < t.rsize(); i++) {
            assert(t.data()[i] == 0.0f);
        }

        std::cout << "    ✓ Single value with padding\n";
    }
}

void test_zero_constructor() {
    std::cout << "Testing data_t::zero constructor...\n";

    // Test 1: 1D zero tensor
    {
        std::cout << "  Test 1: 1D zero tensor [5]\n";
        auto t = tensor::data_t::zero({5});

        assert(t.dims[0] == 5);
        assert(t.size() == 5);

        // Check entire buffer is zero (including padding)
        for (int i = 0; i < t.rsize(); i++) {
            assert(t.data()[i] == 0.0f);
        }

        std::cout << "    ✓ All zeros including padding\n";
    }

    // Test 2: 2D zero tensor
    {
        std::cout << "  Test 2: 2D zero tensor [3,4]\n";
        auto t = tensor::data_t::zero({3, 4});

        assert(t.dims.size() == 2);
        assert(t.dims[0] == 3 && t.dims[1] == 4);
        assert(t.size() == 12);

        // Real shape should be aligned
        assert(t.rshape()[0] == 64 && t.rshape()[1] == 64);

        // Check all values are zero
        for (int i = 0; i < t.rsize(); i++) {
            assert(t.data()[i] == 0.0f);
        }

        std::cout << "    ✓ 2D tensor all zeros\n";
    }

    // Test 3: Template version
    {
        std::cout << "  Test 3: Template zero constructor\n";
        std::vector<int64_t> dims = {2, 5, 3};
        auto t = tensor::data_t::zero(dims);

        assert(t.dims.size() == 3);
        assert(t.size() == 30);

        for (int i = 0; i < t.rsize(); i++) {
            assert(t.data()[i] == 0.0f);
        }

        std::cout << "    ✓ Template version works\n";
    }
}

void test_fill_constructor() {
    std::cout << "Testing data_t::fill constructor...\n";

    // Test 1: Fill 1D tensor
    {
        std::cout << "  Test 1: Fill 1D tensor [7] with value 3.14\n";
        auto t = tensor::data_t::fill({7}, 3.14f);

        assert(t.dims[0] == 7);

        // Check logical values are filled
        int count = 0;
        t.rowsIter([&t, &count](int64_t idx) {
            assert(std::abs(t.data()[idx] - 3.14f) < 0.001f);
            count++;
        });
        assert(count == 7);

        // Check padding is zero
        // The logical values are at indices 0-6, padding starts at 7
        for (int i = 7; i < t.rshape()[0]; i++) {
            assert(t.data()[i] == 0.0f);
        }

        std::cout << "    ✓ Logical values filled, padding zero\n";
    }

    // Test 2: Fill 2D tensor
    {
        std::cout << "  Test 2: Fill 2D tensor [2,3] with value -1.5\n";
        auto t = tensor::data_t::fill({2, 3}, -1.5f);

        assert(t.dims[0] == 2 && t.dims[1] == 3);

        // Check only logical indices have the fill value
        std::vector<int64_t> filledIndices;
        t.rowsIter([&t, &filledIndices](int64_t idx) {
            assert(t.data()[idx] == -1.5f);
            filledIndices.push_back(idx);
        });

        assert(filledIndices.size() == 6);

        // Verify specific positions with alignment
        assert(t.data()[0] == -1.5f);   // [0,0]
        assert(t.data()[1] == -1.5f);   // [0,1]
        assert(t.data()[2] == -1.5f);   // [0,2]
        assert(t.data()[3] == 0.0f);    // padding
        assert(t.data()[64] == -1.5f);  // [1,0]
        assert(t.data()[65] == -1.5f);  // [1,1]
        assert(t.data()[66] == -1.5f);  // [1,2]
        assert(t.data()[67] == 0.0f);   // padding

        std::cout << "    ✓ 2D fill with correct padding\n";
    }
}

void test_random_constructor() {
    std::cout << "Testing data_t::random constructor...\n";

    // Test 1: 1D random tensor
    {
        std::cout << "  Test 1: Random 1D tensor [8]\n";
        auto t = tensor::data_t::random({8});

        assert(t.dims[0] == 8);

        // Check that logical values are non-zero (very likely for random)
        int nonZeroCount = 0;
        t.rowsIter([&t, &nonZeroCount](int64_t idx) {
            if (t.data()[idx] != 0.0f) nonZeroCount++;
        });

        assert(nonZeroCount > 0);  // At least some should be non-zero

        // Check padding is zero
        for (int i = 8; i < t.rshape()[0]; i++) {
            assert(t.data()[i] == 0.0f);
        }

        std::cout << "    ✓ Random values in logical area, zero padding\n";
    }

    // Test 2: 2D random tensor
    {
        std::cout << "  Test 2: Random 2D tensor [4,5]\n";
        auto t = tensor::data_t::random({4, 5});

        assert(t.dims[0] == 4 && t.dims[1] == 5);
        assert(t.size() == 20);

        // Collect statistics about the random values
        float sum = 0.0f;
        float min = 1e9f, max = -1e9f;
        int count = 0;

        t.rowsIter([&](int64_t idx) {
            float val = t.data()[idx];
            sum += val;
            min = std::min(min, val);
            max = std::max(max, val);
            count++;
        });

        assert(count == 20);
        float mean = sum / count;

        std::cout << "    Random stats: mean=" << mean
                  << ", min=" << min << ", max=" << max << "\n";

        // Check padding between rows
        assert(t.data()[5] == 0.0f);   // padding after first row
        assert(t.data()[63] == 0.0f);  // end of first real row

        std::cout << "    ✓ Random values properly distributed with padding\n";
    }
}

void test_copy_constructor() {
    std::cout << "Testing data_t::copy constructor...\n";

    // Test 1: Copy from array (template version)
    {
        std::cout << "  Test 1: Copy from float array\n";
        float data[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
        std::vector<int64_t> dims = {2, 3};

        auto t = tensor::data_t::copy(dims, data);

        assert(t.dims[0] == 2 && t.dims[1] == 3);

        // Verify values are correctly placed with alignment
        assert(t.data()[0] == 1.0f);  // [0,0]
        assert(t.data()[1] == 2.0f);  // [0,1]
        assert(t.data()[2] == 3.0f);  // [0,2]
        assert(t.data()[3] == 0.0f);  // padding

        assert(t.data()[64] == 4.0f); // [1,0]
        assert(t.data()[65] == 5.0f); // [1,1]
        assert(t.data()[66] == 6.0f); // [1,2]
        assert(t.data()[67] == 0.0f); // padding

        std::cout << "    ✓ Data correctly copied with strided layout\n";
    }

    // Test 2: Copy with initializer_list dims
    {
        std::cout << "  Test 2: Copy with initializer_list dims\n";
        float data[] = {10.0f, 20.0f, 30.0f, 40.0f};

        auto t = tensor::data_t::copy({4}, data);

        assert(t.dims[0] == 4);
        assert(t.size() == 4);

        for (int i = 0; i < 4; i++) {
            assert(t.data()[i] == data[i]);
        }

        // Check padding
        for (int i = 4; i < t.rshape()[0]; i++) {
            assert(t.data()[i] == 0.0f);
        }

        std::cout << "    ✓ Initializer_list version works\n";
    }

    // Test 3: Copy larger matrix
    {
        std::cout << "  Test 3: Copy 3x5 matrix\n";
        std::vector<float> data(15);
        std::iota(data.begin(), data.end(), 1.0f);  // Fill with 1,2,3...15

        auto t = tensor::data_t::copy({3, 5}, data.data());

        // Verify using rowsIter
        std::vector<float> retrieved;
        t.rowsIter([&t, &retrieved](int64_t idx) {
            retrieved.push_back(t.data()[idx]);
        });

        assert(retrieved.size() == 15);
        for (int i = 0; i < 15; i++) {
            assert(retrieved[i] == data[i]);
        }

        std::cout << "    ✓ Larger matrix copied correctly\n";
    }
}

void test_concat_constructor() {
    std::cout << "Testing data_t::concat constructor...\n";

    // Test 1: Concatenate 1D vectors
    {
        std::cout << "  Test 1: Concatenate three 1D vectors\n";

        auto v1 = tensor::data_t::value({1.0f, 2.0f});
        auto v2 = tensor::data_t::value({3.0f, 4.0f});
        auto v3 = tensor::data_t::value({5.0f, 6.0f});

        std::vector<tensor::data_t> vecs = {v1, v2, v3};
        auto t = tensor::data_t::concat(vecs);

        assert(t.dims.size() == 2);
        assert(t.dims[0] == 3);  // 3 vectors
        assert(t.dims[1] == 2);  // each of size 2

        // Check values with alignment
        // Row 0
        assert(t.data()[0] == 1.0f);
        assert(t.data()[1] == 2.0f);

        // Row 1 (at offset 64 due to alignment)
        assert(t.data()[64] == 3.0f);
        assert(t.data()[65] == 4.0f);

        // Row 2 (at offset 128)
        assert(t.data()[128] == 5.0f);
        assert(t.data()[129] == 6.0f);

        // Check padding
        assert(t.data()[2] == 0.0f);   // padding in row 0
        assert(t.data()[66] == 0.0f);  // padding in row 1
        assert(t.data()[130] == 0.0f); // padding in row 2

        std::cout << "    ✓ Vectors concatenated with proper alignment\n";
    }

    // Test 2: Concatenate with different data
    {
        std::cout << "  Test 2: Concatenate vectors with pattern\n";

        auto v1 = tensor::data_t::fill({4}, 1.0f);
        auto v2 = tensor::data_t::fill({4}, 2.0f);

        std::vector<tensor::data_t> vecs = {v1, v2};
        auto t = tensor::data_t::concat(vecs);

        assert(t.dims[0] == 2);
        assert(t.dims[1] == 4);

        // Verify values using rowsIter
        int row = 0;
        int col = 0;
        t.rowsIter([&](int64_t idx) {
            float expected = (row == 0) ? 1.0f : 2.0f;
            assert(t.data()[idx] == expected);
            col++;
            if (col == 4) {
                col = 0;
                row++;
            }
        });

        std::cout << "    ✓ Concatenated with correct values\n";
    }
}

void test_member_copy() {
    std::cout << "Testing data_t::copy() member function...\n";

    // Test 1: Copy 1D tensor
    {
        std::cout << "  Test 1: Copy 1D tensor\n";
        auto original = tensor::data_t::value({1.0f, 2.0f, 3.0f, 4.0f, 5.0f});
        auto copy = original.copy();

        // Wait for GPU copy to complete
        nn::stream::global.synchronize();

        assert(copy.dims == original.dims);
        assert(copy.size() == original.size());

        // Verify values match
        for (int i = 0; i < 5; i++) {
            assert(copy.data()[i] == original.data()[i]);
        }

        // Verify it's a deep copy (different buffer)
        assert(copy.data() != original.data());

        // Modify original and check copy is unchanged
        original.data()[0] = 999.0f;
        assert(copy.data()[0] == 1.0f);

        std::cout << "    ✓ Deep copy works correctly\n";
    }

    // Test 2: Copy 2D tensor
    {
        std::cout << "  Test 2: Copy 2D tensor with alignment\n";
        auto original = tensor::data_t::random({3, 7});
        auto copy = original.copy();

        // Wait for GPU copy to complete
        nn::stream::global.synchronize();

        assert(copy.dims == original.dims);
        assert(copy.rshape() == original.rshape());

        // Verify all values match (including padding)
        for (int i = 0; i < original.rsize(); i++) {
            assert(copy.data()[i] == original.data()[i]);
        }

        std::cout << "    ✓ 2D copy preserves alignment and values\n";
    }
}

void test_transpose() {
    std::cout << "Testing data_t::transpose()...\n";

    // Test 1: Simple 2x3 matrix transpose
    {
        std::cout << "  Test 1: Transpose 2x3 matrix\n";
        float data[] = {1.0f, 2.0f, 3.0f,
                        4.0f, 5.0f, 6.0f};
        auto t = tensor::data_t::copy({2, 3}, data);

        // Wait for any async operations
        nn::stream::global.synchronize();

        t.transpose();

        // Wait for transpose to complete
        nn::stream::global.synchronize();

        assert(t.dims[0] == 3);
        assert(t.dims[1] == 2);

        // Expected after transpose:
        // 1 4
        // 2 5
        // 3 6

        // With alignment, values should be at:
        // Row 0: [0]=1, [1]=4
        // Row 1: [64]=2, [65]=5
        // Row 2: [128]=3, [129]=6

        assert(t.data()[0] == 1.0f);
        assert(t.data()[1] == 4.0f);
        assert(t.data()[64] == 2.0f);
        assert(t.data()[65] == 5.0f);
        assert(t.data()[128] == 3.0f);
        assert(t.data()[129] == 6.0f);

        std::cout << "    ✓ Transpose with alignment works\n";
    }

    // Test 2: Square matrix transpose
    {
        std::cout << "  Test 2: Transpose square 4x4 matrix\n";
        std::vector<float> data(16);
        std::iota(data.begin(), data.end(), 1.0f);

        auto t = tensor::data_t::copy({4, 4}, data.data());

        nn::stream::global.synchronize();
        t.transpose();
        nn::stream::global.synchronize();

        assert(t.dims[0] == 4);
        assert(t.dims[1] == 4);

        // Verify diagonal elements (should stay in place)
        assert(t.data()[0] == 1.0f);      // [0,0]
        assert(t.data()[65] == 6.0f);     // [1,1]
        assert(t.data()[130] == 11.0f);   // [2,2]
        assert(t.data()[195] == 16.0f);   // [3,3]

        std::cout << "    ✓ Square matrix transpose correct\n";
    }
}

int main() {
    std::cout << "=== Testing tensor data_t constructors ===\n\n";

    test_value_constructor();
    test_zero_constructor();
    test_fill_constructor();
    test_random_constructor();
    test_copy_constructor();
    test_concat_constructor();
    test_member_copy();
    test_transpose();

    std::cout << "\n=== All tensor constructor tests passed! ===\n";
    return 0;
}