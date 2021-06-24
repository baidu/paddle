/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

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

#ifdef PADDLE_WITH_CUDA
#include <cuda_fp16.h>
#endif
#ifdef PADDLE_WITH_HIP
#include <hip/hip_fp16.h>
#endif

#include <algorithm>

namespace paddle {
namespace operators {
namespace math {

template <typename T>
__device__ __forceinline__ T FromFloat(float a);

template <typename T>
__device__ __forceinline__ float ToFloat(T a);

template <typename T>
__device__ __forceinline__ float2 ToFloat2(T a);

template <typename T>
__device__ __forceinline__ T exp_func(T a);

template <typename T>
__device__ __forceinline__ T FloatsToPair(const float a, const float b);

template <typename T>
struct KeyValuePair;

template <typename T>
using kvp = KeyValuePair<T>;

// from_float
template <>
__device__ __forceinline__ float FromFloat<float>(float a) {
  return a;
}

template <>
__device__ __forceinline__ half FromFloat<half>(float a) {
  return __float2half(a);
}

// to_float
template <>
__device__ __forceinline__ float ToFloat<float>(float a) {
  return a;
}

template <>
__device__ __forceinline__ float2 ToFloat2<float2>(float2 a) {
  return a;
}

template <>
__device__ __forceinline__ float2 FloatsToPair<float2>(const float a,
                                                       const float b) {
  return make_float2(a, b);
}

__inline__ __device__ float2 operator+(const float2 &a, const float2 &b) {
  return make_float2(a.x + b.x, a.y + b.y);
}

template <>
__device__ __forceinline__ float ToFloat<half>(half a) {
  return __half2float(a);
}

template <>
__device__ __forceinline__ float2 ToFloat2<__half2>(__half2 a) {
  return __half22float2(a);
}

template <>
__device__ __forceinline__ __half2 FloatsToPair<__half2>(const float a,
                                                         const float b) {
  return __floats2half2_rn(a, b);
}

template <>
__device__ __forceinline__ float exp_func<float>(float a) {
  return expf(a);
}

template <>
__device__ __forceinline__ half exp_func<half>(half a) {
#if defined(__HIPCC__) || CUDA_ARCH_FP16_SUPPORTED(__CUDA_ARCH__)
  return hexp(a);
#else
  return FromFloat<half>(expf(ToFloat<half>(a)));
#endif
}

template <>
struct KeyValuePair<float> {
  __device__ __forceinline__ KeyValuePair() {}
  __device__ __forceinline__ KeyValuePair(float k, float v)
      : key(k), value(v) {}
  __device__ __forceinline__ KeyValuePair(const KeyValuePair &a) {
    key = a.key;
    value = a.value;
  }
  float key;
  float value;
  __device__ __forceinline__ KeyValuePair
  operator+(const KeyValuePair &a) const {
    KeyValuePair tmp;
    tmp.key = key + a.key;
    tmp.value = value + a.value;
    return tmp;
  }
};

template <>
struct KeyValuePair<half> {
  __device__ __forceinline__ KeyValuePair() {}
  __device__ __forceinline__ KeyValuePair(half k, half v) : key(k), value(v) {}
  __device__ __forceinline__ KeyValuePair(const KeyValuePair &a) {
    key = a.key;
    value = a.value;
  }
  half key;
  half value;
  __device__ __forceinline__ KeyValuePair
  operator+(const KeyValuePair &a) const {
    const half2 a2 = __halves2half2(key, value);
    const half2 b2 = __halves2half2(a.key, a.value);
#ifdef PADDLE_WITH_CUDA
#if CUDA_ARCH_FP16_SUPPORTED(__CUDA_ARCH__)
    const half2 res = __hadd2(a2, b2);
#else
    float a2_1 = __low2float(a2);
    float a2_2 = __high2float(a2);
    float b2_1 = __low2float(b2);
    float b2_2 = __high2float(b2);
    float r1 = a2_1 + b2_1;
    float r2 = a2_2 + b2_2;
    const half2 res = __floats2half2_rn(r1, r2);
#endif
    return KeyValuePair(res.x, res.y);
#else  // PADDLE_WITH_HIP
    const half2 res = __hadd2(a2, b2);
    return KeyValuePair(__low2half(res), __high2half(res));
#endif
  }
};

#define FINAL_MASK 0xffffffff
#define HALF_WARP 16
#define WARP_SIZE 32

template <typename T>
__inline__ __device__ T warpReduceSum(T val, unsigned lane_mask) {
  for (int mask = HALF_WARP; mask > 0; mask >>= 1)
#if defined(PADDLE_WITH_CUDA) && (__CUDA_ARCH__ >= 350 && CUDA_VERSION >= 9000)
    val += __shfl_xor_sync(lane_mask, val, mask, warpSize);
#else
    val += __shfl_xor(val, mask, warpSize);
#endif
  return val;
}

/* Calculate the sum of all elements in a block */
template <typename T>
__inline__ __device__ T blockReduceSum(T val, unsigned mask) {
  static __shared__ T shared[WARP_SIZE];
  int lane = threadIdx.x & 0x1f;
  int wid = threadIdx.x >> 5;

  val = warpReduceSum<T>(val, mask);

  __syncthreads();
  if (lane == 0) shared[wid] = val;

  __syncthreads();

  // align block_span to warpSize
  int block_span = (blockDim.x + warpSize - 1) >> 5;
  val = (lane < block_span) ? shared[lane] : static_cast<T>(0.0f);
  val = warpReduceSum<T>(val, mask);

  return val;
}

template <typename T>
__inline__ __device__ T warpReduceMax(T val, unsigned lane_mask) {
  for (int mask = HALF_WARP; mask > 0; mask >>= 1)
#if defined(PADDLE_WITH_CUDA) && (__CUDA_ARCH__ >= 350 && CUDA_VERSION >= 9000)
    val = max(val, __shfl_xor_sync(lane_mask, val, mask, warpSize));
#else
    val = max(val, __shfl_xor(val, mask, warpSize));
#endif
  return val;
}

template <typename T>
__inline__ __device__ T warpReduceMin(T val, unsigned lane_mask) {
  for (int mask = HALF_WARP; mask > 0; mask >>= 1)
#if defined(PADDLE_WITH_CUDA) && (__CUDA_ARCH__ >= 350 && CUDA_VERSION >= 9000)
    val = min(val, __shfl_xor_sync(lane_mask, val, mask, warpSize));
#else
    val = min(val, __shfl_xor(val, mask, warpSize));
#endif
  return val;
}

/* Calculate the minimum of all elements in a warp when actual quantity of
 * threads are less than warpSize.*/
template <typename T>
__inline__ __device__ T PartialWarpReduceMin(T val, unsigned lane_mask) {
#if defined(PADDLE_WITH_CUDA) && (__CUDA_ARCH__ >= 350 && CUDA_VERSION >= 9000)
  T warp_val = __shfl_sync(lane_mask, val, 0, warpSize);
#else
  T warp_val = __shfl(
      val, 0, warpSize);  // To fullfill the data in each thread of this warp.
#endif
  warp_val = val;

  for (int offset = HALF_WARP; offset > 0; offset >>= 1)
#if defined(PADDLE_WITH_CUDA) && (__CUDA_ARCH__ >= 350 && CUDA_VERSION >= 9000)
    warp_val =
        min(warp_val, __shfl_down_sync(lane_mask, warp_val, offset, warpSize));
#else
    warp_val = min(warp_val, __shfl_down(warp_val, offset, warpSize));
#endif
  return warp_val;
}

/* Calculate the maximum of all elements in a block */
template <typename T>
__inline__ __device__ T blockReduceMax(T val, unsigned mask) {
  static __shared__ T shared[WARP_SIZE];
  int lane = threadIdx.x & 0x1f;
  int wid = threadIdx.x >> 5;

  val = warpReduceMax(val, mask);

  if (lane == 0) shared[wid] = val;

  __syncthreads();

  // align block_span to warpSize
  int block_span = (blockDim.x + warpSize - 1) >> 5;
  val = (lane < block_span) ? shared[lane] : -1e10f;
  val = warpReduceMax(val, mask);

  return val;
}

/* Calculate the minimum of all elements in a block */
template <typename T>
__inline__ __device__ T blockReduceMin(T val, unsigned mask) {
  static __shared__ T shared[WARP_SIZE];
  int lane = threadIdx.x & 0x1f;
  int wid = threadIdx.x >> 5;

  val = warpReduceMin(val, mask);
  if (lane == 0) shared[wid] = val;
  __syncthreads();

  // align block_span to warpSize
  int block_span = (blockDim.x + warpSize - 1) >> 5;
  val = (lane < block_span) ? shared[lane] : 1e10f;
  val = warpReduceMin(val, mask);

  return val;
}

/* Calculate the minimum of all elements in a warp when actual quantity of
 * threads are less than warpSize.*/
template <typename T>
__inline__ __device__ T PartialBlockReduceMin(T val, unsigned mask) {
  static __shared__ T shared[WARP_SIZE];
  static __shared__ T min_value;
  int lane = threadIdx.x & 0x1f;
  int wid = threadIdx.x >> 5;

  val = PartialWarpReduceMin(val, mask);
  if (lane == 0) shared[wid] = val;
  __syncthreads();

  shared[lane] = PartialWarpReduceMin(shared[lane], mask);
#if defined(PADDLE_WITH_HIP)
  // HIP do not support __syncwarp, using __syncthreads() instead is ok,
  // although bringing a few performance decrease.
  __syncthreads();
#else
  __syncwarp();
#endif

#if defined(PADDLE_WITH_CUDA) && (__CUDA_ARCH__ >= 350 && CUDA_VERSION >= 9000)
  val = __shfl_sync(mask, shared[lane], 0, warpSize);
#else
  val = __shfl(shared[lane], 0, warpSize);
#endif
  return val;
}

}  // namespace math
}  // namespace operators
}  // namespace paddle
