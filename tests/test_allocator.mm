#include <iostream>
#include <vector>
#include <cassert>
#include <functional>
#define NN_IMPL
#include "../src/nn.mm"

// Copy the relevant functions from nn.mm for testing
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

void test_idxs() {
    std::cout << "Testing idxs function...\n";
    using namespace nn::allocator;

    // Test 1: 1D array
    {
        std::cout << "  Test 1: 1D array with logical shape [5], real shape [64]\n";
        std::vector<int64_t> logicalShape = {5};
        std::vector<int64_t> realShape = {64};

        ndspan logical(nullptr, logicalShape);
        ndspan real(nullptr, realShape);

        for (int i = 0; i < 5; i++) {
            auto logicalIdx = logical.idxs(logicalShape, i);
            assert(logicalIdx[0] == i);
            std::cout << "    logical.idxs(" << i << ") = [" << logicalIdx[0] << "] ✓\n";
        }
    }

    // Test 2: 2D matrix
    {
        std::cout << "  Test 2: 2D matrix with logical shape [3,4], real shape [64,64]\n";
        std::vector<int64_t> logicalShape = {3, 4};
        std::vector<int64_t> realShape = {64, 64};

        ndspan logical(nullptr, logicalShape);

        // Expected mappings for logical shape [3,4]
        std::vector<std::pair<int, std::vector<int64_t>>> expected = {
            {0, {0, 0}}, {1, {0, 1}}, {2, {0, 2}}, {3, {0, 3}},
            {4, {1, 0}}, {5, {1, 1}}, {6, {1, 2}}, {7, {1, 3}},
            {8, {2, 0}}, {9, {2, 1}}, {10, {2, 2}}, {11, {2, 3}}
        };

        for (auto& [flatIdx, expectedIdx] : expected) {
            auto actualIdx = logical.idxs(logicalShape, flatIdx);
            assert(actualIdx[0] == expectedIdx[0] && actualIdx[1] == expectedIdx[1]);
            std::cout << "    logical.idxs(" << flatIdx << ") = ["
                      << actualIdx[0] << ", " << actualIdx[1] << "] ✓\n";
        }
    }

    // Test 3: 3D tensor
    {
        std::cout << "  Test 3: 3D tensor with logical shape [2,3,4]\n";
        std::vector<int64_t> logicalShape = {2, 3, 4};

        ndspan logical(nullptr, logicalShape);

        // Test some key indices
        auto idx0 = logical.idxs(logicalShape, 0);
        assert(idx0[0] == 0 && idx0[1] == 0 && idx0[2] == 0);
        std::cout << "    logical.idxs(0) = [0, 0, 0] ✓\n";

        auto idx4 = logical.idxs(logicalShape, 4);  // Should be [0, 1, 0]
        assert(idx4[0] == 0 && idx4[1] == 1 && idx4[2] == 0);
        std::cout << "    logical.idxs(4) = [0, 1, 0] ✓\n";

        auto idx12 = logical.idxs(logicalShape, 12);  // Should be [1, 0, 0]
        assert(idx12[0] == 1 && idx12[1] == 0 && idx12[2] == 0);
        std::cout << "    logical.idxs(12) = [1, 0, 0] ✓\n";

        auto idx23 = logical.idxs(logicalShape, 23);  // Should be [1, 2, 3]
        assert(idx23[0] == 1 && idx23[1] == 2 && idx23[2] == 3);
        std::cout << "    logical.idxs(23) = [1, 2, 3] ✓\n";
    }

    std::cout << "  All idxs tests passed!\n\n";
}

void test_idx_dot() {
    std::cout << "Testing idx_dot function...\n";
    using namespace nn::allocator;

    // Test with 2D matrix
    {
        std::cout << "  Test: 2D matrix with real shape [64,64]\n";
        std::vector<int64_t> realShape = {64, 64};
        ndspan real(nullptr, realShape);

        // Test converting multi-dim indices to flat index
        assert(real.idx_dot({0, 0}) == 0);
        assert(real.idx_dot({0, 1}) == 1);
        assert(real.idx_dot({1, 0}) == 64);
        assert(real.idx_dot({1, 1}) == 65);
        assert(real.idx_dot({2, 3}) == 2 * 64 + 3);

        std::cout << "    idx_dot([0,0]) = 0 ✓\n";
        std::cout << "    idx_dot([1,0]) = 64 ✓\n";
        std::cout << "    idx_dot([2,3]) = 131 ✓\n";
    }

    std::cout << "  All idx_dot tests passed!\n\n";
}

void test_rowsIter() {
    std::cout << "Testing rowsIter with alignment...\n";
    using namespace nn;

    // Test 1: 2D matrix with padding
    {
        std::cout << "  Test 1: 2D matrix [3,4] with alignment\n";
        auto buff = tensor::data_t::zero({3, 4});

        // Fill with sequential values to verify iteration order
        int counter = 0;
        buff.rowsIter([&buff, &counter](int64_t idx) {
            buff.data()[idx] = counter++;
        });

        // Verify the values are in correct positions
        std::vector<int64_t> visitedIndices;
        buff.rowsIter([&visitedIndices](int64_t idx) {
            visitedIndices.push_back(idx);
        });

        std::cout << "    Visited " << visitedIndices.size() << " indices (expected 12)\n";
        assert(visitedIndices.size() == 12);

        // Check that indices are correctly spaced for aligned buffer
        // With alignment=64, shape [3,4] becomes [64,64]
        // Row 0 should start at 0, Row 1 at 64, Row 2 at 128
        assert(visitedIndices[0] == 0);   // [0,0]
        assert(visitedIndices[3] == 3);   // [0,3]
        assert(visitedIndices[4] == 64);  // [1,0]
        assert(visitedIndices[7] == 67);  // [1,3]
        assert(visitedIndices[8] == 128); // [2,0]
        assert(visitedIndices[11] == 131);// [2,3]

        std::cout << "    Row iteration pattern correct ✓\n";
    }

    // Test 2: 1D vector with padding
    {
        std::cout << "  Test 2: 1D vector [10] with alignment\n";
        auto buff = tensor::data_t::zero({10});

        std::vector<int64_t> indices;
        buff.rowsIter([&indices](int64_t idx) {
            indices.push_back(idx);
        });

        assert(indices.size() == 10);
        for (int i = 0; i < 10; i++) {
            assert(indices[i] == i);
        }
        std::cout << "    1D iteration correct ✓\n";
    }

    // Test 3: Larger matrix to verify pattern
    {
        std::cout << "  Test 3: Matrix [5,7] with alignment\n";
        auto buff = tensor::data_t::zero({5, 7});

        std::vector<int64_t> indices;
        buff.rowsIter([&indices](int64_t idx) {
            indices.push_back(idx);
        });

        assert(indices.size() == 35);

        // Check first element of each row
        // With alignment=64, real shape is [64,64]
        for (int row = 0; row < 5; row++) {
            assert(indices[row * 7] == row * 64);
        }
        std::cout << "    Large matrix iteration correct ✓\n";
    }

    std::cout << "  All rowsIter tests passed!\n\n";
}

int main() {
    std::cout << "=== Testing allocator functions ===\n\n";

    test_idxs();
    test_idx_dot();
    test_rowsIter();

    std::cout << "=== All allocator tests passed! ===\n";
    return 0;
}
