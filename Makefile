# Build everything
all: main

# Metal library target
default.metallib: gpugemm.air
	xcrun -sdk macosx metallib gpugemm.air -o default.metallib

gpugemm.air: src/gpugemm.metal
	xcrun -sdk macosx metal -c -O3 -ffast-math src/gpugemm.metal -o gpugemm.air

# Executable target
main: src/nn.mm src/main.mm default.metallib
	clang++ -std=c++20 -O3 -ffast-math -Wall -Wextra src/main.mm -framework Metal -framework Foundation -o main

# Clean build artifacts
clean:
	rm -f gpugemm.air default.metallib main

.PHONY: all clean
