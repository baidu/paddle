/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

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
#include <cuda_fp16.h>
#include <math.h>
#include <iostream>
#include <vector>

namespace paddle {
namespace operators {
namespace module {

// for reduce
enum ReduceMode { Global_Mode = 0x00, Local_Mode = 0x01 };

// for Vec load or store
template <typename T, int VecSize>
struct alignas(sizeof(T) * VecSize) AlignVec {
  T val[VecSize];
};

/**
 * @brief compute functor for elementwise_two, in1 and in2 has the same shape
 * @param：
 * T : the type of in1 and in2
 * NX: the row of in1 and in2
 * NY: the col of in1 and in2
 * BlockSize: the strid of col
 * OpFunc: compute functor eg: ADD, SUB, XOR, OR, MUL
 */
template <typename T, int NX, int NY, int BlockSize, class OpFunc>
__device__ void elementwise_binary(const T* __restrict__ in1,
                                   const T* __restrict__ in2,
                                   T* __restrict__ out) {
  OpFunc compute;
#pragma unroll
  for (int idx = 0; idx < NX * NY; ++idx) {
    compute(in1[idx], in2[idx], &out[idx]);
  }
}

/**
 * @brief fma eg: a * b + c, in1 in2, in3 and out has the same shape
 * @param：
 * T : the type of in1 and in2, in3
 * NX: the row of in1, in2 and in3
 * NY: the col of in1, in2 and in3
 * BlockSize: the strid of col
 */
template <typename T, int NX, int NY, int BlockSize, class OpFunc>
__device__ void elementwise_fma(const T* in1, const T* in2, const T* in3,
                                T* out) {
#pragma unroll
  for (int idx = 0; idx < NX * NY; ++idx) {
    out[idx] = in1[idx] * in2[idx] + out[idx];
  }
}

/**
 * @brief compute functor for elementwise_two, in1 is [1, NY], in2 is [NX, NY]
 * @param：
 * T : the type of in1 and in2
 * NX: the row of in1 and in2
 * NY: the col of in2
 * BlockSize: the strid of col
 * OpFunc: compute functor eg: ADD, SUB, XOR, OR, MUL
 */
template <typename T, int NX, int NY, int BlockSize, class OpFunc>
__device__ void cycle_binary(const T* in1, const T* in2, T* out) {
  OpFunc compute;
#pragma unroll
  for (int idx = 0; idx < NX; idx++) {
#pragma unroll
    for (int idy = 0; idy < NY; idy++) {
      compute(in1[idx], in2[idx + idy * NX], out[idx + idy * NX]);
    }
  }
}

/**
 * @brief compute functor for unary, in1 is [NX, NY]
 * @param：
 * T : the type of in
 * NX: the row of in
 * NY: the col of in
 * BlockSize: the strid of col
 * OpFunc: compute functor eg: relu, sigmoid, exp
 */
template <typename T, int NX, int NY, int BlockSize, class OpFunc>
__device__ void elementwise_unary(const T* in, T* out) {
  OpFunc compute;
#pragma unroll
  for (int idx = 0; idx < NX * NY; idx++) {
    compute(in[idx], out[idx]);
  }
}

/** @brief: load
 *
 */
template <typename T, int NX, int NY, int BlockSize>
__device__ void load(const T* in, T* out, int strid_in) {
#pragma unroll
  for (int idx = 0; idx < NX; idx++) {
#pragma unroll
    for (int idy = 0; idy < NY; idy++) {
      out[idx + idy * NX] = in[idy + strid_in * idx];
    }
  }
}

/** @brief: store
 *
 */
template <typename T, int NX, int NY, int BlockSize>
__device__ void store(const T* in, T* out, int strid_out) {
#pragma unroll
  for (int idx = 0; idx < NX; idx++) {
#pragma unroll
    for (int idy = 0; idy < NY; idy++) {
      out[idy + strid_out * idx] = in[idx + idy * NX];
    }
  }
}

// memcopy
template <typename T, int NX, int NY, int BlockSize>
__device__ void memcopy(const T* in, T* out) {
#pragma unroll
  for (int idx = 0; idx < NX * NY; idx++) {
    out[idx] = in[idx];
  }
}

// transformer_t(x)
template <typename Tx, typename Ty, int NX, int NY, typename TransformOp>
__device__ __forceinline__ void transformer(const Tx* in, Ty* out,
                                            TransformOp trans) {
#pragma unroll NY
  for (int idx = 0; idx < NX * NY; idx++) {
    out[idx] = static_cast<Ty>(trans(in[idx]));
  }
}

// ReduceMode == Local_Mode
template <typename T, int NX, int NY, typename OpFunc>
__device__ __forceinline__ void reduceNY(const T* in, T* out, OpFunc reducer) {
#pragma unroll NX
  for (int idx = 0; idx < NX; ++idx) {
#pragma unroll NY
    for (int idy = 0; idy < NY; ++idy) {
      out[idx] = reducer(out[idx], in[idx * NY + idy]);
    }
  }
}

// ReduceMode == Global_Mode
template <typename T, int NX, int NY, typename OpFunc>
__device__ __forceinline__ void reduceNX(const T* in, T* out, OpFunc reducer) {
  for (int idy = 0; idy < NY; ++idy) {
    for (int idx = 0; idx < NX; ++idx) {
      out[idy] = reducer(out[idy], in[idy * NX + idx]);
    }
  }
}

// reduce = reduce higher + reduce_lastDim
/**
 * @brief compute functor for unary, in1 is [NX, NY]
 * @param：
 * T : data type of in and out
 * ReduceType: the type of reduce can be Global_Mode and Local_Mode
 * OpFunc: can be SUM, MEAN, MAX, MIN, OR, AND
 *
 */
// reduce higher
template <typename T, int NX, int NY, typename OpFunc, typename ReduceType>
__device__ __forceinline__ void reduce(const T* in, T* out, OpFunc reducer,
                                       ReduceType reduce_mode) {
  if (reduce_mode == ReduceMode::Global_Mode) {
    reduceNX<T, NX, NY, OpFunc>(in, out, reducer);
  } else {  // reduce_mode == Local_Mode
    reduceNY<T, NX, NY, OpFunc>(in, out, reducer);
  }
}

}  // namespace module
}  // namespace operators
}  // namespace paddle
