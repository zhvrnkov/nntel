#!/bin/bash
set -e

# Compile Metal shaders with all optimizations
echo "Compiling Metal shaders..."
xcrun -sdk macosx metal -c -O3 -ffast-math src/gpugemm.metal -o gpugemm.air

# Create Metal library
echo "Creating Metal library..."
xcrun -sdk macosx metallib gpugemm.air -o default.metallib

# Compile main
echo "Compiling main..."
FRAMEWORKS="-framework Metal -framework Foundation"
FLAGS="-std=c++20 -O3 -ffast-math -Wall -Wextra"
clang++ $FLAGS src/main.mm $FRAMEWORKS -o main

echo "Build complete!"
