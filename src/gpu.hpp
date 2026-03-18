#pragma once

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <algorithm>

namespace gpu {
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
