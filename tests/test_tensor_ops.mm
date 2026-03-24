#include "../src/nn.mm"
#include <cmath>
#include <iostream>
#include <iomanip>

// Simple test framework
int tests_passed = 0;
int tests_failed = 0;

#define TOLERANCE 1e-5f

void assert_float_equal(float actual, float expected, const char* test_name) {
    if (std::abs(actual - expected) < TOLERANCE) {
        tests_passed++;
    } else {
        tests_failed++;
        std::cerr << "FAIL: " << test_name << " - Expected " << expected
                  << " but got " << actual << std::endl;
    }
}

void assert_tensor_equal(const nn::tensor::data_t& actual, const float* expected, int64_t size, const char* test_name) {
    bool passed = true;
    for (int64_t i = 0; i < size; i++) {
        if (std::abs(actual.data()[i] - expected[i]) >= TOLERANCE) {
            passed = false;
            std::cerr << "FAIL: " << test_name << " - At index " << i
                      << " expected " << expected[i] << " but got " << actual.data()[i] << std::endl;
            break;
        }
    }
    if (passed) {
        tests_passed++;
    } else {
        tests_failed++;
    }
}

void assert_true(bool condition, const char* test_name) {
    if (condition) {
        tests_passed++;
    } else {
        tests_failed++;
        std::cerr << "FAIL: " << test_name << std::endl;
    }
}

// Test: tensor::add - same dimensions
void test_add_same_dims() {
    auto A = nn::tensor::data_t::copy({3}, (float[]){1.0f, 2.0f, 3.0f});
    auto B = nn::tensor::data_t::copy({3}, (float[]){4.0f, 5.0f, 6.0f});
    auto C = nn::tensor::data_t::zero({3});

    nn::tensor::add(A, B, C, 1.0f, 1.0f, nn::tensor::device_type::cpu);
    nn::stream::global.synchronize();

    float expected[] = {5.0f, 7.0f, 9.0f};
    assert_tensor_equal(C, expected, 3, "add_same_dims");
}

// Test: tensor::add - same dimensions 2D
void test_add_same_dims_2d() {
    auto A = nn::tensor::data_t::copy({2, 3}, (float[]){1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
    auto B = nn::tensor::data_t::copy({2, 3}, (float[]){0.5f, 1.0f, 1.5f, 2.0f, 2.5f, 3.0f});
    auto C = nn::tensor::data_t::zero({2, 3});

    nn::tensor::add(A, B, C, 1.0f, 1.0f, nn::tensor::device_type::cpu);
    nn::stream::global.synchronize();

    float expected[] = {1.5f, 3.0f, 4.5f, 6.0f, 7.5f, 9.0f};
    assert_tensor_equal(C, expected, 6, "add_same_dims_2d");
}

// Test: tensor::add - with scalar factors
void test_add_with_factors() {
    auto A = nn::tensor::data_t::copy({3}, (float[]){1.0f, 2.0f, 3.0f});
    auto B = nn::tensor::data_t::copy({3}, (float[]){4.0f, 5.0f, 6.0f});
    auto C = nn::tensor::data_t::zero({3});

    nn::tensor::add(A, B, C, 2.0f, 3.0f, nn::tensor::device_type::cpu);
    nn::stream::global.synchronize();

    float expected[] = {14.0f, 19.0f, 24.0f}; // 2*1+3*4, 2*2+3*5, 2*3+3*6
    assert_tensor_equal(C, expected, 3, "add_with_factors");
}

// Test: tensor::add - broadcast B (1D tensor with size 1)
void test_add_broadcast_scalar() {
    auto A = nn::tensor::data_t::copy({2, 2}, (float[]){1.0f, 2.0f, 3.0f, 4.0f});
    auto B = nn::tensor::data_t::copy({1}, (float[]){10.0f});
    auto C = nn::tensor::data_t::zero({2, 2});

    nn::tensor::add(A, B, C, 1.0f, 1.0f, nn::tensor::device_type::cpu);
    nn::stream::global.synchronize();

    float expected[] = {11.0f, 12.0f, 13.0f, 14.0f};
    assert_tensor_equal(C, expected, 4, "add_broadcast_scalar");
}

// Test: tensor::add - broadcast B (vector to 2D matrix)
void test_add_broadcast_vector_to_2d() {
    auto A = nn::tensor::data_t::copy({3, 4}, (float[]){
        1.0f, 2.0f, 3.0f, 4.0f,
        5.0f, 6.0f, 7.0f, 8.0f,
        9.0f, 10.0f, 11.0f, 12.0f
    });
    auto B = nn::tensor::data_t::copy({3}, (float[]){100.0f, 200.0f, 300.0f});
    auto C = nn::tensor::data_t::zero({3, 4});

    nn::tensor::add(A, B, C, 1.0f, 1.0f, nn::tensor::device_type::cpu);
    nn::stream::global.synchronize();

    float expected[] = {
        101.0f, 102.0f, 103.0f, 104.0f,
        205.0f, 206.0f, 207.0f, 208.0f,
        309.0f, 310.0f, 311.0f, 312.0f
    };
    assert_tensor_equal(C, expected, 12, "add_broadcast_vector_to_2d");
}

// Test: tensor::add - A is scalar (size 1), B is 2D tensor
void test_add_scalar_to_vector() {
    auto A = nn::tensor::data_t::copy({1}, (float[]){5.0f});
    auto B = nn::tensor::data_t::copy({2, 2}, (float[]){1.0f, 2.0f, 3.0f, 4.0f});
    auto C = nn::tensor::data_t::zero({2, 2});

    nn::tensor::add(A, B, C, 1.0f, 1.0f, nn::tensor::device_type::cpu);
    nn::stream::global.synchronize();

    float expected[] = {6.0f, 7.0f, 8.0f, 9.0f};
    assert_tensor_equal(C, expected, 4, "add_scalar_to_vector");
}

// Test: tensor::sub - basic subtraction
void test_sub_basic() {
    auto A = nn::tensor::data_t::copy({3}, (float[]){10.0f, 20.0f, 30.0f});
    auto B = nn::tensor::data_t::copy({3}, (float[]){1.0f, 2.0f, 3.0f});
    auto C = nn::tensor::data_t::zero({3});

    nn::tensor::sub(A, B, C, nn::tensor::device_type::cpu);
    nn::stream::global.synchronize();

    float expected[] = {9.0f, 18.0f, 27.0f};
    assert_tensor_equal(C, expected, 3, "sub_basic");
}

// Test: tensor::mul - element-wise same dims
void test_mul_elementwise_same_dims() {
    auto A = nn::tensor::data_t::copy({4}, (float[]){1.0f, 2.0f, 3.0f, 4.0f});
    auto B = nn::tensor::data_t::copy({4}, (float[]){2.0f, 3.0f, 4.0f, 5.0f});
    auto C = nn::tensor::data_t::zero({4});

    nn::tensor::mul(A, B, C, nn::tensor::device_type::cpu);
    nn::stream::global.synchronize();

    float expected[] = {2.0f, 6.0f, 12.0f, 20.0f};
    assert_tensor_equal(C, expected, 4, "mul_elementwise_same_dims");
}

// Test: tensor::mul - element-wise same dims 2D
void test_mul_elementwise_2d() {
    auto A = nn::tensor::data_t::copy({2, 3}, (float[]){1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
    auto B = nn::tensor::data_t::copy({2, 3}, (float[]){2.0f, 2.0f, 2.0f, 3.0f, 3.0f, 3.0f});
    auto C = nn::tensor::data_t::zero({2, 3});

    nn::tensor::mul(A, B, C, nn::tensor::device_type::cpu);
    nn::stream::global.synchronize();

    float expected[] = {2.0f, 4.0f, 6.0f, 12.0f, 15.0f, 18.0f};
    assert_tensor_equal(C, expected, 6, "mul_elementwise_2d");
}

// Test: tensor::mul - scalar multiplication (1D array)
void test_mul_scalar() {
    auto A = nn::tensor::data_t::copy({5}, (float[]){1.0f, 2.0f, 3.0f, 4.0f, 5.0f});
    auto B = nn::tensor::data_t::copy({1}, (float[]){10.0f});
    auto C = nn::tensor::data_t::zero({5});

    nn::tensor::mul(A, B, C, nn::tensor::device_type::cpu);
    nn::stream::global.synchronize();

    float expected[] = {10.0f, 20.0f, 30.0f, 40.0f, 50.0f};
    assert_tensor_equal(C, expected, 5, "mul_scalar");
}

// Test: tensor::mul - scalar multiplication 2D
void test_mul_scalar_2d() {
    auto A = nn::tensor::data_t::copy({2, 2}, (float[]){1.0f, 2.0f, 3.0f, 4.0f});
    auto B = nn::tensor::data_t::copy({1}, (float[]){0.5f});
    auto C = nn::tensor::data_t::zero({2, 2});

    nn::tensor::mul(A, B, C, nn::tensor::device_type::cpu);
    nn::stream::global.synchronize();

    float expected[] = {0.5f, 1.0f, 1.5f, 2.0f};
    assert_tensor_equal(C, expected, 4, "mul_scalar_2d");
}

// Test: tensor::matmul - 2D x 2D
void test_matmul_2d_2d() {
    auto A = nn::tensor::data_t::copy({2, 3}, (float[]){
        1.0f, 2.0f, 3.0f,
        4.0f, 5.0f, 6.0f
    });
    auto B = nn::tensor::data_t::copy({3, 2}, (float[]){
        1.0f, 2.0f,
        3.0f, 4.0f,
        5.0f, 6.0f
    });
    auto C = nn::tensor::data_t::zero({2, 2});

    nn::tensor::matmul(A, B, C, nn::tensor::device_type::cpu);
    nn::stream::global.synchronize();

    // [1 2 3] [1 2]   [1*1+2*3+3*5  1*2+2*4+3*6]   [22  28]
    // [4 5 6] [3 4] = [4*1+5*3+6*5  4*2+5*4+6*6] = [49  64]
    //         [5 6]
    float expected[] = {22.0f, 28.0f, 49.0f, 64.0f};
    assert_tensor_equal(C, expected, 4, "matmul_2d_2d");
}

// Test: tensor::matmul - 2D x 2D (square matrices)
void test_matmul_2d_2d_square() {
    auto A = nn::tensor::data_t::copy({2, 2}, (float[]){
        1.0f, 2.0f,
        3.0f, 4.0f
    });
    auto B = nn::tensor::data_t::copy({2, 2}, (float[]){
        5.0f, 6.0f,
        7.0f, 8.0f
    });
    auto C = nn::tensor::data_t::zero({2, 2});

    nn::tensor::matmul(A, B, C, nn::tensor::device_type::cpu);
    nn::stream::global.synchronize();

    // [1 2] [5 6]   [1*5+2*7  1*6+2*8]   [19  22]
    // [3 4] [7 8] = [3*5+4*7  3*6+4*8] = [43  50]
    float expected[] = {19.0f, 22.0f, 43.0f, 50.0f};
    assert_tensor_equal(C, expected, 4, "matmul_2d_2d_square");
}

// Test: tensor::matmul - 2D x 1D (matrix-vector)
void test_matmul_2d_1d() {
    auto A = nn::tensor::data_t::copy({3, 4}, (float[]){
        1.0f, 2.0f, 3.0f, 4.0f,
        5.0f, 6.0f, 7.0f, 8.0f,
        9.0f, 10.0f, 11.0f, 12.0f
    });
    auto B = nn::tensor::data_t::copy({4}, (float[]){1.0f, 2.0f, 3.0f, 4.0f});
    auto C = nn::tensor::data_t::zero({3});

    nn::tensor::matmul(A, B, C, nn::tensor::device_type::cpu);
    nn::stream::global.synchronize();

    // Row 0: 1*1 + 2*2 + 3*3 + 4*4 = 1 + 4 + 9 + 16 = 30
    // Row 1: 5*1 + 6*2 + 7*3 + 8*4 = 5 + 12 + 21 + 32 = 70
    // Row 2: 9*1 + 10*2 + 11*3 + 12*4 = 9 + 20 + 33 + 48 = 110
    float expected[] = {30.0f, 70.0f, 110.0f};
    assert_tensor_equal(C, expected, 3, "matmul_2d_1d");
}

// Test: tensor::matmul - 1D x 1D (dot product)
void test_matmul_1d_1d() {
    auto A = nn::tensor::data_t::copy({4}, (float[]){1.0f, 2.0f, 3.0f, 4.0f});
    auto B = nn::tensor::data_t::copy({4}, (float[]){5.0f, 6.0f, 7.0f, 8.0f});
    auto C = nn::tensor::data_t::zero({1});

    nn::tensor::matmul(A, B, C, nn::tensor::device_type::cpu);
    nn::stream::global.synchronize();

    // 1*5 + 2*6 + 3*7 + 4*8 = 5 + 12 + 21 + 32 = 70
    float expected[] = {70.0f};
    assert_tensor_equal(C, expected, 1, "matmul_1d_1d");
}

// Test: tensor::sigmoid - basic
void test_sigmoid_basic() {
    auto A = nn::tensor::data_t::copy({5}, (float[]){-2.0f, -1.0f, 0.0f, 1.0f, 2.0f});
    auto C = nn::tensor::data_t::zero({5});

    nn::tensor::sigmoid(A, C, nn::tensor::device_type::cpu);
    nn::stream::global.synchronize();

    float expected[5];
    for (int i = 0; i < 5; i++) {
        expected[i] = 1.0f / (1.0f + exp(-A.data()[i]));
    }
    assert_tensor_equal(C, expected, 5, "sigmoid_basic");
}

// Test: tensor::sigmoid - 2D
void test_sigmoid_2d() {
    auto A = nn::tensor::data_t::copy({2, 3}, (float[]){
        -1.0f, 0.0f, 1.0f,
        -0.5f, 0.5f, 2.0f
    });
    auto C = nn::tensor::data_t::zero({2, 3});

    nn::tensor::sigmoid(A, C, nn::tensor::device_type::cpu);
    nn::stream::global.synchronize();

    float expected[6];
    for (int i = 0; i < 6; i++) {
        expected[i] = 1.0f / (1.0f + exp(-A.data()[i]));
    }
    assert_tensor_equal(C, expected, 6, "sigmoid_2d");
}

// Test: tensor::sigmoid - large positive/negative values
void test_sigmoid_extreme_values() {
    auto A = nn::tensor::data_t::copy({4}, (float[]){-10.0f, -100.0f, 10.0f, 100.0f});
    auto C = nn::tensor::data_t::zero({4});

    nn::tensor::sigmoid(A, C, nn::tensor::device_type::cpu);
    nn::stream::global.synchronize();

    // sigmoid(-100) ≈ 0, sigmoid(100) ≈ 1
    assert_true(C.data()[0] < 0.001f, "sigmoid_extreme_neg_10");
    assert_true(C.data()[1] < 0.001f, "sigmoid_extreme_neg_100");
    assert_true(C.data()[2] > 0.999f, "sigmoid_extreme_pos_10");
    assert_true(C.data()[3] > 0.999f, "sigmoid_extreme_pos_100");
}

// Test: tensor::sum - basic 1D
void test_sum_basic() {
    auto A = nn::tensor::data_t::copy({5}, (float[]){1.0f, 2.0f, 3.0f, 4.0f, 5.0f});
    auto C = nn::tensor::data_t::zero({1});

    nn::tensor::sum(A, C, nn::tensor::device_type::cpu);
    nn::stream::global.synchronize();

    float expected[] = {15.0f}; // 1+2+3+4+5
    assert_tensor_equal(C, expected, 1, "sum_basic");
}

// Test: tensor::sum - 2D tensor
void test_sum_2d() {
    auto A = nn::tensor::data_t::copy({2, 3}, (float[]){
        1.0f, 2.0f, 3.0f,
        4.0f, 5.0f, 6.0f
    });
    auto C = nn::tensor::data_t::zero({1});

    nn::tensor::sum(A, C, nn::tensor::device_type::cpu);
    nn::stream::global.synchronize();

    float expected[] = {21.0f}; // 1+2+3+4+5+6
    assert_tensor_equal(C, expected, 1, "sum_2d");
}

// Test: tensor::sum - with negative values
void test_sum_negative() {
    auto A = nn::tensor::data_t::copy({4}, (float[]){-1.0f, 2.0f, -3.0f, 4.0f});
    auto C = nn::tensor::data_t::zero({1});

    nn::tensor::sum(A, C, nn::tensor::device_type::cpu);
    nn::stream::global.synchronize();

    float expected[] = {2.0f}; // -1+2-3+4
    assert_tensor_equal(C, expected, 1, "sum_negative");
}

// Test: data_t constructors and helpers
void test_data_constructors() {
    // Test value constructor
    auto t1 = nn::tensor::data_t::value({1.0f, 2.0f, 3.0f});
    assert_true(t1.dims.size() == 1 && t1.dims[0] == 3, "value_constructor_dims");
    assert_float_equal(t1.data()[0], 1.0f, "value_constructor_data_0");
    assert_float_equal(t1.data()[1], 2.0f, "value_constructor_data_1");
    assert_float_equal(t1.data()[2], 3.0f, "value_constructor_data_2");

    // Test zero constructor
    auto t2 = nn::tensor::data_t::zero({2, 3});
    assert_true(t2.dims.size() == 2 && t2.dims[0] == 2 && t2.dims[1] == 3, "zero_constructor_dims");
    for (int i = 0; i < 6; i++) {
        assert_float_equal(t2.data()[i], 0.0f, "zero_constructor_data");
    }

    // Test fill constructor
    auto t3 = nn::tensor::data_t::fill({4}, 7.5f);
    assert_true(t3.dims.size() == 1 && t3.dims[0] == 4, "fill_constructor_dims");
    for (int i = 0; i < 4; i++) {
        assert_float_equal(t3.data()[i], 7.5f, "fill_constructor_data");
    }

    // Test size method
    assert_true(t1.size() == 3, "size_1d");
    assert_true(t2.size() == 6, "size_2d");
}

// Test: edge cases for matmul
void test_matmul_edge_cases() {
    // 1x1 matrices
    auto A = nn::tensor::data_t::copy({1, 1}, (float[]){5.0f});
    auto B = nn::tensor::data_t::copy({1, 1}, (float[]){3.0f});
    auto C = nn::tensor::data_t::zero({1, 1});

    nn::tensor::matmul(A, B, C, nn::tensor::device_type::cpu);
    nn::stream::global.synchronize();

    float expected[] = {15.0f};
    assert_tensor_equal(C, expected, 1, "matmul_1x1");

    // Larger matrix-vector
    auto A2 = nn::tensor::data_t::copy({1, 5}, (float[]){1.0f, 2.0f, 3.0f, 4.0f, 5.0f});
    auto B2 = nn::tensor::data_t::copy({5}, (float[]){1.0f, 1.0f, 1.0f, 1.0f, 1.0f});
    auto C2 = nn::tensor::data_t::zero({1});

    nn::tensor::matmul(A2, B2, C2, nn::tensor::device_type::cpu);
    nn::stream::global.synchronize();

    float expected2[] = {15.0f}; // 1+2+3+4+5
    assert_tensor_equal(C2, expected2, 1, "matmul_1xN_Nx1");
}

// Test: multiple operations in sequence
void test_combined_operations() {
    auto A = nn::tensor::data_t::copy({3}, (float[]){1.0f, 2.0f, 3.0f});
    auto B = nn::tensor::data_t::copy({3}, (float[]){2.0f, 2.0f, 2.0f});
    auto C = nn::tensor::data_t::zero({3});
    auto D = nn::tensor::data_t::zero({3});

    // First multiply: C = A * B (element-wise)
    nn::tensor::mul(A, B, C, nn::tensor::device_type::cpu);
    nn::stream::global.synchronize();

    // Then add: D = C + A
    nn::tensor::add(C, A, D, 1.0f, 1.0f, nn::tensor::device_type::cpu);
    nn::stream::global.synchronize();

    // C should be [2, 4, 6] (A * B element-wise)
    // D should be [3, 6, 9] (C + A)
    float expected[] = {3.0f, 6.0f, 9.0f};
    assert_tensor_equal(D, expected, 3, "combined_mul_add");
}

// Test: zero tensor operations
void test_zero_operations() {
    auto A = nn::tensor::data_t::zero({3});
    auto B = nn::tensor::data_t::copy({3}, (float[]){1.0f, 2.0f, 3.0f});
    auto C = nn::tensor::data_t::zero({3});

    nn::tensor::add(A, B, C, 1.0f, 1.0f, nn::tensor::device_type::cpu);
    nn::stream::global.synchronize();

    float expected[] = {1.0f, 2.0f, 3.0f};
    assert_tensor_equal(C, expected, 3, "add_with_zero");

    // Multiply with zero
    nn::tensor::mul(A, B, C, nn::tensor::device_type::cpu);
    nn::stream::global.synchronize();

    float expected_zero[] = {0.0f, 0.0f, 0.0f};
    assert_tensor_equal(C, expected_zero, 3, "mul_with_zero");
}

void run_all_tests() {
    std::cout << "Running CPU tensor operations tests..." << std::endl;
    std::cout << "======================================" << std::endl;

    // Addition tests
    test_add_same_dims();
    test_add_same_dims_2d();
    test_add_with_factors();
    test_add_broadcast_scalar();
    test_add_broadcast_vector_to_2d();
    test_add_scalar_to_vector();

    // Subtraction tests
    test_sub_basic();

    // Multiplication tests
    test_mul_elementwise_same_dims();
    test_mul_elementwise_2d();
    test_mul_scalar();
    test_mul_scalar_2d();

    // Matrix multiplication tests
    test_matmul_2d_2d();
    test_matmul_2d_2d_square();
    test_matmul_2d_1d();
    test_matmul_1d_1d();
    test_matmul_edge_cases();

    // Sigmoid tests
    test_sigmoid_basic();
    test_sigmoid_2d();
    test_sigmoid_extreme_values();

    // Sum tests
    test_sum_basic();
    test_sum_2d();
    test_sum_negative();

    // Constructor tests
    test_data_constructors();

    // Combined operations
    test_combined_operations();
    test_zero_operations();

    std::cout << "======================================" << std::endl;
    std::cout << "Tests passed: " << tests_passed << std::endl;
    std::cout << "Tests failed: " << tests_failed << std::endl;

    if (tests_failed == 0) {
        std::cout << "All tests passed!" << std::endl;
    } else {
        std::cout << "Some tests failed." << std::endl;
    }
}

int main() {
    run_all_tests();
    return tests_failed > 0 ? 1 : 0;
}
