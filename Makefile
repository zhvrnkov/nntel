# Build everything
all: main

# Test target
test: test_ops test_allocator test_ctors

test_ops: tests/test_tensor_ops
	./tests/test_tensor_ops

test_allocator: tests/test_allocator
	./tests/test_allocator

test_ctors: tests/test_tensor_ctors
	./tests/test_tensor_ctors

tests/test_tensor_ops: tests/test_tensor_ops.mm src/nn.mm default.metallib
	clang++ -std=c++23 -O3 -ffast-math -Wall -Wextra tests/test_tensor_ops.mm -framework Metal -framework Foundation -o tests/test_tensor_ops

tests/test_allocator: tests/test_allocator.mm src/nn.mm default.metallib
	clang++ -std=c++23 -O3 -ffast-math -Wall -Wextra tests/test_allocator.mm -framework Metal -framework Foundation -o tests/test_allocator

tests/test_tensor_ctors: tests/test_tensor_ctors.mm src/nn.mm default.metallib
	clang++ -std=c++23 -O3 -ffast-math -Wall -Wextra tests/test_tensor_ctors.mm -framework Metal -framework Foundation -o tests/test_tensor_ctors

# Metal library target
default.metallib: gpugemm.air
	xcrun -sdk macosx metallib gpugemm.air -o default.metallib

gpugemm.air: src/gpugemm.metal
	xcrun -sdk macosx metal -c -O3 -ffast-math src/gpugemm.metal -o gpugemm.air

# Executable target
main: src/nn.mm src/main.mm default.metallib
	clang++ -std=c++23 -O3 -ffast-math -Wall -Wextra src/main.mm -framework Metal -framework Foundation -o main

# Clean build artifacts
clean:
	rm -f gpugemm.air default.metallib main tests/test_tensor_ops tests/test_allocator tests/test_tensor_ctors

.PHONY: all clean test test_ops test_allocator test_ctors
