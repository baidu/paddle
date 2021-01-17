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

#include <gtest/gtest.h>
#include <algorithm>
#include <iostream>
#include "hip/hip_runtime.h"
#ifdef _WIN32
#include <numeric>
#endif
#include <random>

#define PADDLE_CUDA_FP16
#include "paddle/fluid/platform/cuda_device_function.h"
#include "paddle/fluid/platform/cuda_primitives.h"
#include "paddle/fluid/platform/float16.h"

using paddle::platform::PADDLE_CUDA_NUM_THREADS;
using paddle::platform::float16;

template <typename T>
__global__ void AddKernel(const T* data_a, T* data_b, size_t num) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < num;
       i += blockDim.x * gridDim.x) {
    paddle::platform::CudaAtomicAdd(&data_b[i], data_a[i]);
  }
}

template <typename T>
struct AddFunctor {
  T operator()(const T& a, const T& b) { return a + b; }
};

template <typename T>
void TestCase(size_t num) {
  T *in1, *in2, *out;
  T *d_in1, *d_in2;
  size_t size = sizeof(T) * num;
  hipMalloc(reinterpret_cast<void**>(&d_in1), size);
  hipMalloc(reinterpret_cast<void**>(&d_in2), size);
  in1 = reinterpret_cast<T*>(malloc(size));
  in2 = reinterpret_cast<T*>(malloc(size));
  out = reinterpret_cast<T*>(malloc(size));
  std::minstd_rand engine;
  std::uniform_real_distribution<double> dist(0.0, 1.0);
  for (size_t i = 0; i < num; ++i) {
    in1[i] = static_cast<T>(dist(engine));
    in2[i] = static_cast<T>(dist(engine));
  }
  hipMemcpy(d_in1, in1, size, hipMemcpyHostToDevice);
  hipMemcpy(d_in2, in2, size, hipMemcpyHostToDevice);
  hipLaunchKernelGGL(HIP_KERNEL_NAME(AddKernel<T>), dim3(1),
                     dim3(PADDLE_CUDA_NUM_THREADS), 0, 0, d_in1, d_in2, num);
  hipDeviceSynchronize();
  hipMemcpy(out, d_in2, size, hipMemcpyDeviceToHost);
  hipDeviceSynchronize();
  for (size_t i = 0; i < num; ++i) {
    // NOTE(dzhwinter): the float16 add has small underflow/overflow
    // so we use EXPECT_NEAR to check the result.
    EXPECT_NEAR(static_cast<float>(out[i]),
                static_cast<float>(AddFunctor<T>()(in1[i], in2[i])), 0.001);
  }
  free(in1);
  free(in2);
  free(out);
  hipFree(d_in1);
  hipFree(d_in2);
}

// cuda primitives
TEST(CudaAtomic, Add) {
  TestCase<float>(static_cast<size_t>(10));
  TestCase<float>(static_cast<size_t>(1024 * 1024));

  TestCase<double>(static_cast<size_t>(10));
  TestCase<double>(static_cast<size_t>(1024 * 1024));
}

TEST(CudaAtomic, float16) {
  TestCase<float16>(static_cast<size_t>(1));
  TestCase<float16>(static_cast<size_t>(2));
  TestCase<float16>(static_cast<size_t>(3));

  TestCase<float16>(static_cast<size_t>(10));
  TestCase<float16>(static_cast<size_t>(1024 * 1024));
}

// unalignment of uint8
void TestUnalign(size_t num, const int shift_bit) {
  ASSERT_EQ(num % 2, 0);
  float16 *in1, *in2, *out;
  float16 *d_in1, *d_in2;
  size_t size = sizeof(uint8_t) * (num + shift_bit);
  size_t array_size = sizeof(float16) * (num / 2);

  hipMalloc(reinterpret_cast<void**>(&d_in1), size);
  hipMalloc(reinterpret_cast<void**>(&d_in2), size);
  in1 = reinterpret_cast<float16*>(malloc(size));
  in2 = reinterpret_cast<float16*>(malloc(size));
  out = reinterpret_cast<float16*>(malloc(size));

  // right shift 1, mimic the unalignment of address
  float16* r_in1 =
      reinterpret_cast<float16*>(reinterpret_cast<uint8_t*>(in1) + shift_bit);
  float16* r_in2 =
      reinterpret_cast<float16*>(reinterpret_cast<uint8_t*>(in2) + shift_bit);

  std::minstd_rand engine;
  std::uniform_real_distribution<double> dist(0.0, 1.0);
  for (size_t i = 0; i < num / 2; ++i) {
    r_in1[i] = static_cast<float16>(dist(engine));
    r_in2[i] = static_cast<float16>(dist(engine));
  }
  hipMemcpy(d_in1, r_in1, array_size, hipMemcpyHostToDevice);
  hipMemcpy(d_in2, r_in2, array_size, hipMemcpyHostToDevice);
  hipLaunchKernelGGL(HIP_KERNEL_NAME(AddKernel<float16>), dim3(1),
                     dim3(PADDLE_CUDA_NUM_THREADS), 0, 0, d_in1, d_in2,
                     num / 2);
  hipDeviceSynchronize();
  hipMemcpy(out, d_in2, array_size, hipMemcpyDeviceToHost);
  hipDeviceSynchronize();
  for (size_t i = 0; i < num / 2; ++i) {
    // NOTE(dzhwinter): the float16 add has small truncate error.
    // so we use EXPECT_NEAR to check the result.
    EXPECT_NEAR(static_cast<float>(out[i]),
                static_cast<float>(AddFunctor<float16>()(r_in1[i], r_in2[i])),
                0.001);
  }
  free(in1);
  free(in2);
  free(out);
  hipFree(d_in1);
  hipFree(d_in2);
}

TEST(CudaAtomic, float16Unalign) {
  // same with float16 testcase
  TestUnalign(static_cast<size_t>(2), /*shift_bit*/ 2);
  TestUnalign(static_cast<size_t>(1024), /*shift_bit*/ 2);
  TestUnalign(static_cast<size_t>(1024 * 1024), /*shift_bit*/ 2);

  // shift the address.
  TestUnalign(static_cast<size_t>(2), /*shift_bit*/ 1);
  TestUnalign(static_cast<size_t>(1024), /*shift_bit*/ 1);
  TestUnalign(static_cast<size_t>(1024 * 1024), /*shift_bit*/ 1);

  TestUnalign(static_cast<size_t>(2), /*shift_bit*/ 3);
  TestUnalign(static_cast<size_t>(1024), /*shift_bit*/ 3);
  TestUnalign(static_cast<size_t>(1024 * 1024), /*shift_bit*/ 3);
}

// https://devblogs.nvidia.com/faster-parallel-reductions-kepler/
template <typename T>
static __forceinline__ __device__ T WarpReduceSum(T val) {
  unsigned mask = 0u;
  CREATE_SHFL_MASK(mask, true);
  for (int offset = warpSize / 2; offset > 0; offset /= 2) {
    val += paddle::platform::CudaShuffleDownSync(mask, val, offset);
  }
  return val;
}

template <typename T>
__forceinline__ __device__ T BlockReduce(T val) {
  static __shared__ T shared[32];  // Shared mem for 32 partial sums
  int lane = threadIdx.x % warpSize;
  int wid = threadIdx.x / warpSize;

  val = WarpReduceSum(val);  // Each warp performs partial reduction

  if (lane == 0) shared[wid] = val;  // Write reduced value to shared memory

  __syncthreads();  // Wait for all partial reductions

  // read from shared memory only if that warp existed
  val =
      (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : static_cast<T>(0);

  if (wid == 0) val = WarpReduceSum(val);  // Final reduce within first warp

  return val;
}

template <typename T>
__global__ void DeviceReduceSum(T* in, T* out, size_t N) {
  T sum(0);
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N;
       i += blockDim.x * gridDim.x) {
    sum += in[i];
  }
  sum = BlockReduce<T>(sum);
  __syncthreads();
  if (threadIdx.x == 0) out[blockIdx.x] = sum;
}

template <typename T>
void TestReduce(size_t num, float atol = 0.01) {
  T* in1;
  T *d_in1, *d_in2;
  size_t size = sizeof(T) * num;
  hipMalloc(reinterpret_cast<void**>(&d_in1), size);
  hipMalloc(reinterpret_cast<void**>(&d_in2), sizeof(T));
  in1 = reinterpret_cast<T*>(malloc(size));
  std::minstd_rand engine;
  std::uniform_real_distribution<double> dist(0.0, 1.0);
  for (size_t i = 0; i < num; ++i) {
    in1[i] = static_cast<T>(dist(engine));
  }
  auto out = std::accumulate(in1, in1 + num, static_cast<T>(0));
  hipMemcpy(d_in1, in1, size, hipMemcpyHostToDevice);
  hipDeviceSynchronize();
  hipLaunchKernelGGL(HIP_KERNEL_NAME(DeviceReduceSum<T>), dim3(1),
                     dim3(PADDLE_CUDA_NUM_THREADS), 0, 0, d_in1, d_in2, num);
  hipMemcpy(in1, d_in2, sizeof(T), hipMemcpyDeviceToHost);
  hipDeviceSynchronize();
  // NOTE(dzhwinter): the float16 add has small underflow/overflow
  // so we use EXPECT_NEAR to check the result.
  EXPECT_NEAR(static_cast<float>(in1[0]), static_cast<float>(out), atol);
  free(in1);
  hipFree(d_in1);
  hipFree(d_in2);
}

TEST(CudaShuffleSync, float16) {
  TestReduce<float>(10);
  TestReduce<float>(1000);

  // float16 will overflow or accumulate truncate errors in big size.
  TestReduce<float16>(10);
  TestReduce<float16>(100, /*atol error*/ 1.0);
}
