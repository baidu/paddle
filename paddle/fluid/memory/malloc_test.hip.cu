// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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
#ifdef PADDLE_WITH_HIP
#include <hip/hip_runtime.h>
#endif

#ifdef PADDLE_WITH_CUDA
#include <cuda.h>
#include <cuda_runtime.h>
#endif

#include <thread>  // NOLINT
#include <vector>

#include "gtest/gtest.h"
#include "paddle/fluid/memory/malloc.h"
#include "paddle/fluid/platform/device_context.h"

namespace paddle {
namespace memory {

const int NUM_STREAMS = 8;
const int N = 2;
const float DELTA = 1e-1;

using CudaDevCtxVec = std::vector<std::unique_ptr<platform::CUDADeviceContext>>;

__global__ void kernel(float *x, int n) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  for (int i = tid; i < n; i += blockDim.x * gridDim.x) {
    x[i] = 3.14159 * i;
  }
}

void CheckKernelOutput(float *x, int n) {
  auto host_x = std::unique_ptr<float[]>(new float[n]);
  for (int i = 0; i < n; ++i) {
#ifdef PADDLE_WITH_CUDA
    EXPECT_TRUE(cudaSuccess == cudaMemcpy(host_x.get(), x, n * sizeof(float),
                                          cudaMemcpyDeviceToHost));
#endif
#ifdef PADDLE_WITH_HIP
    EXPECT_TRUE(hipSuccess == hipMemcpy(host_x.get(), x, n * sizeof(float),
                                        hipMemcpyDeviceToHost));
#endif
    EXPECT_GE(host_x[i] + DELTA, 3.14159f * i);
    EXPECT_LE(host_x[i] - DELTA, 3.14159f * i);
  }
}

void MultiStreamCompute(float **data, float **second_data,
                        const platform::CUDADeviceContext &ctx) {
  // multi-streams
  AllocationPtr allocation_ptr = Alloc(ctx, N * sizeof(float));
  EXPECT_GE(allocation_ptr->size(), N * sizeof(float));
  *data = reinterpret_cast<float *>(allocation_ptr->ptr());
  hipLaunchKernelGGL((kernel), dim3(1), dim3(64), 0, ctx.stream(), *data, N);

  // allocate and compute on same stream again
  allocation_ptr = Alloc(ctx, N * sizeof(float));
  EXPECT_GE(allocation_ptr->size(), N * sizeof(float));
  *second_data = reinterpret_cast<float *>(allocation_ptr->ptr());
  hipLaunchKernelGGL((kernel), dim3(1), dim3(64), 0, ctx.stream(), *second_data,
                     N);
}

TEST(Malloc, CUDADeviceContextMultiStream) {
  auto place = platform::CUDAPlace(0);
#ifdef PADDLE_WITH_CUDA
  EXPECT_TRUE(cudaSuccess == cudaDeviceSynchronize());
#else  // PADDLE_WITH_HIP
  EXPECT_TRUE(hipSuccess == hipSetDevice(0));
#endif
  AllocationPtr main_stream_alloc_ptr = Alloc(place, N * sizeof(float));
  EXPECT_GE(main_stream_alloc_ptr->size(), N * sizeof(float));
  float *main_stream_data =
      reinterpret_cast<float *>(main_stream_alloc_ptr->ptr());

  float *data[NUM_STREAMS];
  float *second_data[NUM_STREAMS];
  CudaDevCtxVec dev_ctx;

  // default stream
  hipLaunchKernelGGL((kernel), dim3(1), dim3(64), 0, 0, main_stream_data, N);
  main_stream_alloc_ptr.reset();

  for (int i = 0; i < NUM_STREAMS; ++i) {
    dev_ctx.push_back(std::unique_ptr<platform::CUDADeviceContext>(
        new platform::CUDADeviceContext(place)));
    MultiStreamCompute(&data[i], &second_data[i], *dev_ctx[i]);
  }
#ifdef PADDLE_WITH_CUDA
  EXPECT_TRUE(cudaSuccess == cudaDeviceSynchronize());
#else  // PADDLE_WITH_HIP
  EXPECT_TRUE(hipSuccess == hipDeviceSynchronize());
#endif
  for (int i = 0; i < NUM_STREAMS; ++i) {
    CheckKernelOutput(data[i], N);
    CheckKernelOutput(second_data[i], N);
  }
}

TEST(Malloc, CUDADeviceContextMultiThreadMultiStream) {
  auto place = platform::CUDAPlace(0);
#ifdef PADDLE_WITH_CUDA
  EXPECT_TRUE(cudaSuccess == cudaSetDevice(0));
#else  // PADDLE_WITH_HIP
  EXPECT_TRUE(hipSuccess == hipSetDevice(0));
#endif
  AllocationPtr main_stream_alloc_ptr = Alloc(place, N * sizeof(float));
  EXPECT_GE(main_stream_alloc_ptr->size(), N * sizeof(float));
  float *main_stream_data =
      reinterpret_cast<float *>(main_stream_alloc_ptr->ptr());

  float *data[NUM_STREAMS];
  float *second_data[NUM_STREAMS];
  CudaDevCtxVec dev_ctx;
  std::vector<std::thread> threads;

  // default stream
  hipLaunchKernelGGL((kernel), dim3(1), dim3(64), 0, 0, main_stream_data, N);
  main_stream_alloc_ptr.reset();

  for (int i = 0; i < NUM_STREAMS; ++i) {
    dev_ctx.push_back(std::unique_ptr<platform::CUDADeviceContext>(
        new platform::CUDADeviceContext(place)));
    threads.push_back(std::thread(MultiStreamCompute, &data[i], &second_data[i],
                                  std::cref(*dev_ctx[i])));
  }

  for (int i = 0; i < NUM_STREAMS; ++i) {
    threads[i].join();
  }
#ifdef PADDLE_WITH_CUDA
  EXPECT_TRUE(cudaSuccess == cudaDeviceSynchronize());
#else  // PADDLE_WITH_HIP
  EXPECT_TRUE(hipSuccess == hipDeviceSynchronize());
#endif
  for (int i = 0; i < NUM_STREAMS; ++i) {
    CheckKernelOutput(data[i], N);
    CheckKernelOutput(second_data[i], N);
  }
}

TEST(Malloc, AllocZero) {
  auto place = platform::CUDAPlace(0);
  AllocationPtr allocation_ptr = Alloc(place, 0);
  EXPECT_GE(allocation_ptr->size(), 0);
}
}  // namespace memory
}  // namespace paddle
