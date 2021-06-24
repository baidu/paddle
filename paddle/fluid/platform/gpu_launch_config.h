// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Used for compute gpu launch parameter

#pragma once

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)

#ifdef PADDLE_WITH_CUDA
#include <cuda_runtime.h>
#else
#include <hip/hip_runtime.h>
#endif
#include <stddef.h>
#include <algorithm>
#include <string>
#include <vector>

namespace paddle {
namespace platform {

inline int DivUp(int a, int b) { return (a + b - 1) / b; }

struct GpuLaunchConfig {
  dim3 theory_thread_count = dim3(1, 1, 1);
  dim3 thread_per_block = dim3(1, 1, 1);
  dim3 block_per_grid = dim3(1, 1, 1);
  int compute_capability = 0;
};

inline GpuLaunchConfig GetGpuLaunchConfig1D(
    const platform::CUDADeviceContext& context, int element_count,
#ifdef PADDLE_WITH_HIP
    // HIP will throw GPU memory access fault if threads > 256
    int max_threads = 256) {
#else
    int max_threads = 1024) {
#endif
  PADDLE_ENFORCE_GT(element_count, 0,
                    platform::errors::InvalidArgument(
                        "element count should be greater than 0,"
                        " but received value is: %d.",
                        element_count));

  const int theory_thread_count = element_count;
  // Get Max threads in all SM
  int max_physical_threads = context.GetMaxPhysicalThreadCount();
  int sm = context.GetSMCount();

  // Compute physical threads we need, should small than max sm threads
  const int physical_thread_count =
      std::min(max_physical_threads, theory_thread_count);

  // Need get from device
  const int thread_per_block =
      std::min(max_threads, context.GetMaxThreadsPerBlock());
  const int block_count =
      std::min(DivUp(physical_thread_count, thread_per_block), sm);
  // Get compute_capability
  const int capability = context.GetComputeCapability();

  GpuLaunchConfig config;
  config.theory_thread_count.x = theory_thread_count;
  config.thread_per_block.x = thread_per_block;
  config.block_per_grid.x = block_count;
  config.compute_capability = capability;
  return config;
}

inline GpuLaunchConfig GetGpuLaunchConfig2D(
    const platform::CUDADeviceContext& context, int x_dim, int y_dim) {
  PADDLE_ENFORCE_GT(x_dim, 0, platform::errors::InvalidArgument(
                                  "x dim number should greater than 0,"
                                  " but received value is: %d",
                                  x_dim));
  PADDLE_ENFORCE_GT(y_dim, 0, platform::errors::InvalidArgument(
                                  "y dim number should greater than 0,"
                                  " but received value is: %d",
                                  y_dim));

  const int kThreadsPerBlock = 256;
  int block_cols = std::min(x_dim, kThreadsPerBlock);
  int block_rows = std::max(kThreadsPerBlock / block_cols, 1);

  int max_physical_threads = context.GetMaxPhysicalThreadCount();
  const int max_blocks = std::max(max_physical_threads / kThreadsPerBlock, 1);

  GpuLaunchConfig config;
  // Noticed, block size is not align to 32, if needed do it yourself.
  config.theory_thread_count = dim3(x_dim, y_dim, 1);
  config.thread_per_block = dim3(block_cols, block_rows, 1);

  int grid_x = std::min(DivUp(x_dim, block_cols), max_blocks);
  int grid_y = std::min(max_blocks / grid_x, std::max(y_dim / block_rows, 1));

  config.block_per_grid = dim3(grid_x, grid_y, 1);
  return config;
}

// TODO(wangchaochaohu): 3D will add later

}  // namespace platform
}  // namespace paddle

#endif
