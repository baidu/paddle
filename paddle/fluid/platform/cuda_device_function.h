/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once

#include <cuda.h>
// NOTE(): support float16 to half in header file.
#define PADDLE_CUDA_FP16
#include <cuda_fp16.h>
#include "paddle/fluid/platform/complex128.h"
#include "paddle/fluid/platform/complex64.h"
#include "paddle/fluid/platform/float16.h"

namespace paddle {
namespace platform {

#if CUDA_VERSION < 9000
#define CREATE_SHFL_MASK(mask, predicate) mask = 0u;
#else
#define FULL_WARP_MASK 0xFFFFFFFF
#define CREATE_SHFL_MASK(mask, predicate) \
  mask = __ballot_sync(FULL_WARP_MASK, (predicate))
#endif

inline static int RoundToPowerOfTwo(int dim) {
  if (dim > 512) {
    return 1024;
  } else if (dim > 256) {
    return 512;
  } else if (dim > 128) {
    return 256;
  } else if (dim > 64) {
    return 128;
  } else if (dim > 32) {
    return 64;
  } else {
    return 32;
  }
}

#define CUDA_LAUNCH_KERNEL_BASE(dim, ...)  \
  case (dim): {                            \
    constexpr auto kPowerOfTwoDim = (dim); \
    __VA_ARGS__;                           \
  } break

#define CUDA_LAUNCH_KERNEL_HELPER(...)          \
  CUDA_LAUNCH_KERNEL_BASE(1024, ##__VA_ARGS__); \
  CUDA_LAUNCH_KERNEL_BASE(512, ##__VA_ARGS__);  \
  CUDA_LAUNCH_KERNEL_BASE(256, ##__VA_ARGS__);  \
  CUDA_LAUNCH_KERNEL_BASE(128, ##__VA_ARGS__);  \
  CUDA_LAUNCH_KERNEL_BASE(64, ##__VA_ARGS__);   \
  CUDA_LAUNCH_KERNEL_BASE(32, ##__VA_ARGS__);

template <typename T>
__forceinline__ __device__ T CudaShuffleDownSync(unsigned mask, T val,
                                                 int delta,
                                                 int width = warpSize) {
#if CUDA_VERSION < 9000
  return __shfl_down(val, delta, width);
#else
  return __shfl_down_sync(mask, val, static_cast<unsigned>(delta), width);
#endif
}

template <typename T>
__forceinline__ __device__ T CudaShuffleXorSync(unsigned mask, T val,
                                                int width = warpSize) {
#if CUDA_VERSION < 9000
  return __shfl_xor(val, width);
#else
  return __shfl_xor_sync(mask, val, width);
#endif
}

// CUDA 9.0 have native compatible float16 shfl_down
#if CUDA_VERSION < 9000
template <>
__forceinline__ __device__ float16 CudaShuffleDownSync(unsigned mask,
                                                       float16 val, int delta,
                                                       int width) {
  return float16(
      __shfl_down(static_cast<half>(val), static_cast<unsigned>(delta), width));
}
template <>
__forceinline__ __device__ float16 CudaShuffleXorSync(unsigned mask,
                                                      float16 val, int width) {
  return float16(__shfl_xor(static_cast<half>(val), width));
}
#else
template <>
__forceinline__ __device__ float16 CudaShuffleDownSync(unsigned mask,
                                                       float16 val, int delta,
                                                       int width) {
  return float16(__shfl_down_sync(mask, static_cast<half>(val),
                                  static_cast<unsigned>(delta), width));
}

template <>
__forceinline__ __device__ paddle::platform::complex64 CudaShuffleDownSync(
    unsigned mask, paddle::platform::complex64 val, int delta, int width) {
  float real = static_cast<float>(__shfl_down_sync(
      mask, static_cast<float>(val.real), static_cast<unsigned>(delta), width));
  float imag = static_cast<float>(__shfl_down_sync(
      mask, static_cast<float>(val.imag), static_cast<unsigned>(delta), width));
  return paddle::platform::complex64(real, imag);
}

template <>
__forceinline__ __device__ paddle::platform::complex128 CudaShuffleDownSync(
    unsigned mask, paddle::platform::complex128 val, int delta, int width) {
  double real = static_cast<double>(
      __shfl_down_sync(mask, static_cast<double>(val.real),
                       static_cast<unsigned>(delta), width));
  double imag = static_cast<double>(
      __shfl_down_sync(mask, static_cast<double>(val.imag),
                       static_cast<unsigned>(delta), width));
  return paddle::platform::complex128(real, imag);
}

template <>
__forceinline__ __device__ float16 CudaShuffleXorSync(unsigned mask,
                                                      float16 val, int width) {
  return float16(__shfl_xor_sync(mask, static_cast<half>(val), width));
}

template <>
__forceinline__ __device__ paddle::platform::complex64 CudaShuffleXorSync(
    unsigned mask, paddle::platform::complex64 val, int width) {
  float real = static_cast<float>(
      __shfl_xor_sync(mask, static_cast<float>(val.real), width));
  float imag = static_cast<float>(
      __shfl_xor_sync(mask, static_cast<float>(val.imag), width));
  return paddle::platform::complex64(real, imag);
}

template <>
__forceinline__ __device__ paddle::platform::complex128 CudaShuffleXorSync(
    unsigned mask, paddle::platform::complex128 val, int width) {
  double real = static_cast<double>(
      __shfl_xor_sync(mask, static_cast<double>(val.real), width));
  double imag = static_cast<double>(
      __shfl_xor_sync(mask, static_cast<double>(val.imag), width));
  return paddle::platform::complex128(real, imag);
}
#endif

template <typename T>
__forceinline__ __device__ T CudaShuffleSync(unsigned mask, T val, int src_line,
                                             int width = 32) {
#if CUDA_VERSION < 9000
  return __shfl(val, src_line, width);
#else
  return __shfl_sync(mask, val, src_line, width);
#endif
}

template <typename T>
HOSTDEVICE T Infinity() {
  return INFINITY;
}

template <typename T>
__device__ T reduceSum(T val, int tid, int len) {
  // NOTE(zcd): The warp size should be taken from the
  // parameters of the GPU but not specified as 32 simply.
  // To make the reduceSum more efficiently,
  // I use Warp-Level Parallelism and assume the Warp size
  // is 32 which may be different for different GPU,
  // but most card's warp size is 32.
  const int warpSize = 32;
  __shared__ T shm[warpSize];
  unsigned mask = 0u;
  CREATE_SHFL_MASK(mask, tid < len);

  for (int offset = warpSize / 2; offset > 0; offset /= 2)
    val += platform::CudaShuffleDownSync(mask, val, offset);

  if (tid < warpSize) shm[tid] = 0;
  __syncthreads();

  if (tid % warpSize == 0) {
    shm[tid / warpSize] = val;
  }
  __syncthreads();

  CREATE_SHFL_MASK(mask, tid < warpSize);

  if (tid < warpSize) {
    val = shm[tid];
    for (int offset = warpSize / 2; offset > 0; offset /= 2)
      val += platform::CudaShuffleDownSync(mask, val, offset);
  }
  return val;
}

}  // namespace platform
}  // namespace paddle
