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

#include <glog/logging.h>

#include <algorithm>
#include <functional>  // for multiplies
#include <iterator>
#include <vector>

#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/memory/malloc.h"
#include "paddle/fluid/operators/amp/fp16_type_traits.h"
#include "paddle/fluid/operators/elementwise/elementwise_op_function.cu.h"
#include "paddle/fluid/platform/gpu_info.h"
#include "paddle/fluid/platform/transform.h"

#if defined(__NVCC__) || defined(__HIPCC__)
#ifdef __NVCC__
#include <cuda.h>
#elif defined(__HIPCC__)
#include <hip/hip_runtime.h>
#endif
#include <thrust/iterator/iterator_adaptor.h>

#include "paddle/fluid/platform/cuda_device_function.h"
#include "paddle/fluid/platform/cuda_primitives.h"
#ifdef __HIPCC__
constexpr int ELEMWISE_MAX_BLOCK_DIM = 256;
#else
constexpr int ELEMWISE_MAX_BLOCK_DIM = 1024;
#endif
#define BLOCK_X 32
#define BLOCK_Y 32
#endif

#include "paddle/fluid/operators/math/math_function.h"
#include "paddle/fluid/platform/for_range.h"
#define GetDivMod(dividend, divisor, div, mod) \
  do {                                         \
    const auto dividend_copy = dividend;       \
    *div = dividend_copy / divisor;            \
    *mod = dividend_copy % divisor;            \
  } while (0)

#define DIVUP(x, y) (((x) + (y)-1) / (y))

#define ROUNDUP(x, y) (DIVUP((x), (y)) * (y))

namespace paddle {
namespace operators {

/*
*  Pack input and output tensors into respective vectors with
*  consideration of varible X`s class type.
*  Input variable X is supported to be whether LoDTensor or
*  SelectedRows class type in this package function, once X
*  was SelectedRows type, a valid pointer x_for_selectedrows
*  is excepted to be passed in from op kernel for acquisition
*  of the valid address of LoDTensor created ahead in the function.
*/
template <typename OutT>
int PackTensorsIntoVector(const framework::ExecutionContext &ctx,
                          std::vector<const framework::Tensor *> *ins,
                          std::vector<framework::Tensor *> *outs,
                          framework::Tensor *x_for_selectedrows = nullptr) {
  int axis = -1;
  auto x_var = ctx.InputVar("X");
  PADDLE_ENFORCE_NOT_NULL(
      x_var, platform::errors::InvalidArgument(
                 "Unable to get input Variable X, Variable name is %s.\n",
                 ctx.InputName("X")));
  auto *y = ctx.Input<framework::LoDTensor>("Y");
  framework::Tensor *z;

  if (x_var->IsType<framework::LoDTensor>()) {
    auto *x = ctx.Input<framework::LoDTensor>("X");
    z = ctx.Output<framework::LoDTensor>("Out");
    ins->emplace_back(x);
  } else if (x_var->IsType<framework::SelectedRows>()) {
    PADDLE_ENFORCE_EQ(y->dims().size() == 1 && y->dims()[0] == 1, true,
                      platform::errors::InvalidArgument(
                          "For elementwise_op, if X is Sparse, Y must be "
                          "scalar. But reveived the size of Y = %d.",
                          y->dims().size()));
    PADDLE_ENFORCE_NOT_NULL(
        x_for_selectedrows,
        platform::errors::InvalidArgument(
            "The parameter x_for_selectedrows is excepted to "
            "be valid, once input varible X`s class type is "
            "SelectedRows.\n"));
    auto &x_sele = x_var->Get<framework::SelectedRows>();
    auto out_sele = ctx.Output<framework::SelectedRows>("Out");
    *x_for_selectedrows = x_sele.value();
    out_sele->set_rows(x_sele.rows());
    out_sele->set_height(x_sele.height());
    out_sele->mutable_value()->Resize(x_sele.value().dims());
    out_sele->mutable_value()->mutable_data(ctx.GetPlace(),
                                            x_for_selectedrows->type());
    z = ctx.Output<framework::SelectedRows>("Out")->mutable_value();
    ins->emplace_back(x_for_selectedrows);
  } else {
    PADDLE_THROW(platform::errors::InvalidArgument(
        "X's type[%s] is not supported by elementwise_op. X's type should be "
        "LoDTensor or SelectedRows.",
        framework::ToTypeName(x_var->Type())));
  }
  z->mutable_data<OutT>(ctx.GetPlace());
  outs->emplace_back(z);

  if (y != nullptr) {
    ins->emplace_back(y);
    axis = ctx.HasAttr("axis") ? ctx.Attr<int>("axis") : -1;
  }
  return axis;
}

/*
 * Out = X ⊙ Y
 * If Y's shape does not match X' shape, they will be reshaped.
 * For example:
 * 1. shape(X) = (2, 3, 4, 5), shape(Y) = (3, 4), with axis=1
 *    pre=2, n=3*4, post=5
 *    x.shape(2, 12, 5) * y.shape(1, 12, 1).broadcast(2, 12, 5)
 * 2. shape(X) = (2, 3, 4, 5), shape(Y) = (4,5)
 *    pre=2*3, n=4*5, post=1
 *    x.shape(6, 20, 1) * y.shape(1, 20, 1).broadcast(6, 20, 1)
 *
 * New parameter: *is_run_common_broadcast* is a flag to record whether to run
 * common broadcast code.
 */
inline void get_mid_dims(const framework::DDim &x_dims,
                         const framework::DDim &y_dims, const int axis,
                         int *pre, int *n, int *post,
                         int *is_run_common_broadcast) {
  *pre = 1;
  *n = 1;
  *post = 1;
  *is_run_common_broadcast = 0;
  for (int i = 0; i < axis; ++i) {
    (*pre) *= x_dims[i];
  }
  for (int i = 0; i < y_dims.size(); ++i) {
    if (x_dims[i + axis] != y_dims[i]) {
      PADDLE_ENFORCE_EQ(y_dims[i] == 1 || x_dims[i + axis] == 1, true,
                        platform::errors::InvalidArgument(
                            "Broadcast dimension mismatch. Operands "
                            "could not be broadcast together with the shape of "
                            "X = [%s] and the shape of Y = [%s]. Received [%d] "
                            "in X is not equal to [%d] in Y.",
                            x_dims, y_dims, x_dims[i + axis], y_dims[i]));
      *is_run_common_broadcast = 1;
      return;
    }
    (*n) *= y_dims[i];
  }
  for (int i = axis + y_dims.size(); i < x_dims.size(); ++i) {
    (*post) *= x_dims[i];
  }
}

inline int GetElementwiseIndex(const int *x_dims_array, const int max_dim,
                               const int *index_array) {
  int index_ = 0;
  for (int i = 0; i < max_dim; i++) {
    if (x_dims_array[i] > 1) {
      index_ = index_ * x_dims_array[i] + index_array[i];
    }
  }
  return index_;
}

inline void UpdateElementwiseIndexArray(const int *out_dims_array,
                                        const int max_dim, int *index_array) {
  for (int i = max_dim - 1; i >= 0; --i) {
    ++index_array[i];
    if (index_array[i] >= out_dims_array[i]) {
      index_array[i] -= out_dims_array[i];
    } else {
      break;
    }
  }
}

inline void GetBroadcastDimsArrays(const framework::DDim &x_dims,
                                   const framework::DDim &y_dims,
                                   int *x_dims_array, int *y_dims_array,
                                   int *out_dims_array, const int max_dim,
                                   const int axis) {
  PADDLE_ENFORCE_GE(
      axis, 0,
      platform::errors::InvalidArgument(
          "Axis should be great than or equal to 0, but received axis is %d.",
          axis));
  PADDLE_ENFORCE_LT(axis, max_dim,
                    platform::errors::InvalidArgument(
                        "Axis should be less than %d, but received axis is %d.",
                        max_dim, axis));
  if (x_dims.size() > y_dims.size()) {
    std::fill(y_dims_array, y_dims_array + axis, 1);
    if (axis + y_dims.size() < max_dim) {
      std::fill(y_dims_array + axis + y_dims.size(), y_dims_array + max_dim, 1);
    }
    std::copy(x_dims.Get(), x_dims.Get() + x_dims.size(), x_dims_array);
    std::copy(y_dims.Get(), y_dims.Get() + y_dims.size(), y_dims_array + axis);
  } else {
    std::fill(x_dims_array, x_dims_array + axis, 1);
    if (axis + x_dims.size() < max_dim) {
      std::fill(x_dims_array + axis + x_dims.size(), x_dims_array + max_dim, 1);
    }
    std::copy(x_dims.Get(), x_dims.Get() + x_dims.size(), x_dims_array + axis);
    std::copy(y_dims.Get(), y_dims.Get() + y_dims.size(), y_dims_array);
  }

  for (int i = 0; i < max_dim; i++) {
    PADDLE_ENFORCE_EQ(
        x_dims_array[i] == y_dims_array[i] || x_dims_array[i] <= 1 ||
            y_dims_array[i] <= 1,
        true, platform::errors::InvalidArgument(
                  "Broadcast dimension mismatch. Operands could "
                  "not be broadcast together with the shape of X = [%s] and "
                  "the shape of Y = [%s]. Received [%d] in X is not equal to "
                  "[%d] in Y at i:%d.",
                  x_dims, y_dims, x_dims_array[i], y_dims_array[i], i));
    if ((x_dims_array[i] > 1 || y_dims_array[i] > 1) ||
        (x_dims_array[i] == 1 && y_dims_array[i] == 1)) {
      out_dims_array[i] = std::max(x_dims_array[i], y_dims_array[i]);
    } else {
      out_dims_array[i] = -1;
    }
  }
}

template <typename Functor, typename T, typename OutType = T>
void CommonForwardBroadcastCPU(const framework::Tensor *x,
                               const framework::Tensor *y, framework::Tensor *z,
                               int *x_dims_array, int *y_dims_array,
                               int *out_dims_array, int max_dim,
                               const platform::CPUDeviceContext &ctx,
                               Functor func,
                               const bool is_xsize_larger = true) {
  std::vector<int> index_array(max_dim, 0);
  const T *x_data = x->data<T>();
  const T *y_data = y->data<T>();
  PADDLE_ENFORCE_NOT_NULL(x_data, platform::errors::InvalidArgument(
                                      "The input X should not be empty."));
  PADDLE_ENFORCE_NOT_NULL(y_data, platform::errors::InvalidArgument(
                                      "The input Y should not be empty."));
  OutType *out_data = z->mutable_data<OutType>(ctx.GetPlace());

  const int out_size = std::accumulate(out_dims_array, out_dims_array + max_dim,
                                       1, std::multiplies<int>());
  int x_index, y_index;
  for (int out_index = 0; out_index < out_size; ++out_index) {
    x_index = GetElementwiseIndex(x_dims_array, max_dim, index_array.data());
    y_index = GetElementwiseIndex(y_dims_array, max_dim, index_array.data());
    if (is_xsize_larger) {
      out_data[out_index] = func(x_data[x_index], y_data[y_index]);
    } else {
      out_data[out_index] = func(y_data[y_index], x_data[x_index]);
    }

    UpdateElementwiseIndexArray(out_dims_array, max_dim, index_array.data());
  }
}

#if defined(__NVCC__) || defined(__HIPCC__)
template <typename Functor, typename T, typename OutType>
__global__ void ElementwiseKernel(const T *__restrict__ x_data,
                                  const T *__restrict__ y_data,
                                  OutType *__restrict__ out_data, int n,
                                  int post, const size_t total, Functor func) {
  int tid = threadIdx.x + blockDim.x * blockIdx.x;
  int stride = blockDim.x * gridDim.x;

  for (int i = tid; i < total; i += stride) {
    int idx = i / post % n;
    out_data[i] = func(x_data[i], y_data[idx]);
  }
}

template <typename Functor, typename T, typename OutType>
void ComputeElementwiseCUDA(const framework::Tensor *x,
                            const framework::Tensor *y, framework::Tensor *z,
                            int pre, int n, int post,
                            const platform::CUDADeviceContext &ctx,
                            Functor func, const bool is_xsize_larger = true) {
  const T *x_data = x->data<T>();
  const T *y_data = y->data<T>();
  OutType *out_data = z->mutable_data<OutType>(ctx.GetPlace());

  int numel = pre * n * post;
  int threads = 256;
  int blocks = (numel + threads - 1) / threads;

  if (is_xsize_larger) {
    ElementwiseKernel<Functor, T,
                      OutType><<<blocks, threads, 0, ctx.stream()>>>(
        x_data, y_data, out_data, n, post, numel, func);

  } else {
    ElementwiseKernel<Functor, T,
                      OutType><<<blocks, threads, 0, ctx.stream()>>>(
        y_data, x_data, out_data, n, post, numel, func);
  }
}

template <typename Functor, typename T, typename OutType = T>
__global__ void CommonForwardBroadcastCUDAKernel(
    const int *x_strides_array, const int *y_strides_array,
    const int *out_dims_array, const T *x, const T *y, OutType *out,
    int out_size, int max_dim, Functor func, const bool is_xsize_larger) {
  for (int out_index = blockIdx.x * blockDim.x + threadIdx.x;
       out_index < out_size; out_index += blockDim.x * gridDim.x) {
    int x_index = 0;
    int y_index = 0;
    int out_index_quotient = out_index;
    int remainder = 0;
#pragma unroll
    for (int i = max_dim - 1; i >= 0; --i) {
      GetDivMod(out_index_quotient, out_dims_array[i], &out_index_quotient,
                &remainder);
      x_index += remainder * x_strides_array[i];
      y_index += remainder * y_strides_array[i];
    }
    if (is_xsize_larger) {
      out[out_index] = func(x[x_index], y[y_index]);
    } else {
      out[out_index] = func(y[y_index], x[x_index]);
    }
  }
}

template <typename Functor, typename T, typename OutType = T>
void CommonForwardBroadcastCUDA(
    const framework::Tensor *x, const framework::Tensor *y,
    framework::Tensor *z, int *x_dims_array, int *y_dims_array,
    int *out_dims_array, int max_dim, const platform::CUDADeviceContext &ctx,
    Functor func, const bool is_xsize_larger = true) {
  const auto gplace = BOOST_GET_CONST(platform::CUDAPlace, ctx.GetPlace());
  auto cplace = platform::CPUPlace();
  const T *x_data = x->data<T>();
  const T *y_data = y->data<T>();
  OutType *out_data = z->mutable_data<OutType>(ctx.GetPlace());

  std::vector<int> x_strides_array(max_dim);
  std::vector<int> y_strides_array(max_dim);
  int x_stride = 1;
  int y_stride = 1;
  for (int i = max_dim - 1; i >= 0; i--) {
    x_strides_array[i] = x_dims_array[i] == 1 ? 0 : x_stride;
    y_strides_array[i] = y_dims_array[i] == 1 ? 0 : y_stride;
    x_stride *= x_dims_array[i];
    y_stride *= y_dims_array[i];
  }

  int bytes = max_dim * sizeof(int);
  auto x_strides_array_tmp = memory::Alloc(ctx, bytes);
  int *x_strides_array_gpu =
      reinterpret_cast<int *>(x_strides_array_tmp->ptr());
  memory::Copy(gplace, x_strides_array_gpu, cplace, x_strides_array.data(),
               bytes, ctx.stream());

  auto y_strides_array_tmp = memory::Alloc(ctx, bytes);
  int *y_strides_array_gpu =
      reinterpret_cast<int *>(y_strides_array_tmp->ptr());
  memory::Copy(gplace, y_strides_array_gpu, cplace, y_strides_array.data(),
               bytes, ctx.stream());

  auto out_dims_array_tmp = memory::Alloc(ctx, bytes);
  int *out_dims_array_gpu = reinterpret_cast<int *>(out_dims_array_tmp->ptr());
  memory::Copy(gplace, out_dims_array_gpu, cplace, out_dims_array, bytes,
               ctx.stream());

  const int out_size = std::accumulate(out_dims_array, out_dims_array + max_dim,
                                       1, std::multiplies<int>());
  dim3 gird_size = dim3(
      (out_size + PADDLE_CUDA_THREAD_SIZE - 1) / PADDLE_CUDA_THREAD_SIZE, 1);
  dim3 block_size = dim3(PADDLE_CUDA_THREAD_SIZE, 1);

  CommonForwardBroadcastCUDAKernel<
      Functor, T, OutType><<<gird_size, block_size, 0, ctx.stream()>>>(
      x_strides_array_gpu, y_strides_array_gpu, out_dims_array_gpu, x_data,
      y_data, out_data, out_size, max_dim, func, is_xsize_larger);
}

#endif  // __NVCC__ or __HIPCC__

template <typename T, typename DX_OP, typename DY_OP>
void CommonGradBroadcastCPU(
    const framework::Tensor &x, const framework::Tensor &y,
    const framework::Tensor &out, const framework::Tensor &dout,
    framework::Tensor *dx, framework::Tensor *dy, int *x_dims_array,
    int *y_dims_array, int *out_dims_array, int max_dim,
    const platform::CPUDeviceContext &ctx, DX_OP dx_op, DY_OP dy_op) {
  std::vector<int> index_array(max_dim, 0);
  const T *x_data = x.data<T>();
  const T *y_data = y.data<T>();
  const T *out_data = out.data<T>();
  const T *dout_data = dout.data<T>();
  T *dx_data = dx == nullptr ? nullptr : dx->mutable_data<T>(ctx.GetPlace());
  T *dy_data = dy == nullptr ? nullptr : dy->mutable_data<T>(ctx.GetPlace());
  if (dx_data != nullptr) {
    memset(dx_data, 0, dx->numel() * sizeof(T));
  }
  if (dy_data != nullptr) {
    memset(dy_data, 0, dy->numel() * sizeof(T));
  }
  const int out_size = std::accumulate(out_dims_array, out_dims_array + max_dim,
                                       1, std::multiplies<int>());
  int x_index, y_index;
  for (int out_index = 0; out_index < out_size; ++out_index) {
    x_index = GetElementwiseIndex(x_dims_array, max_dim, index_array.data());
    y_index = GetElementwiseIndex(y_dims_array, max_dim, index_array.data());
    if (dx_data != nullptr) {
      dx_data[x_index] += dx_op(x_data[x_index], y_data[y_index],
                                out_data[out_index], dout_data[out_index]);
    }
    if (dy_data != nullptr) {
      dy_data[y_index] += dy_op(x_data[x_index], y_data[y_index],
                                out_data[out_index], dout_data[out_index]);
    }

    UpdateElementwiseIndexArray(out_dims_array, max_dim, index_array.data());
  }
}

inline void ComputeBroadcastKernelSize(int *x_dims_array, int *out_dims_array,
                                       int *x_blocks, int *x_threads,
                                       int max_dim) {
  *x_blocks = 1;
  *x_threads = 1;
  for (int i = 0; i < max_dim; i++) {
    if (x_dims_array[i] == out_dims_array[i]) {
      *x_blocks *= x_dims_array[i];
    } else {
      *x_threads *= out_dims_array[i];
    }
  }
}

inline void ComputeBroadcastTranspositionArray(const int *x_one_indexs,
                                               int *x_trans_indexs,
                                               const int max_dim,
                                               const int x_one_size) {
  int diff = max_dim - x_one_size;
  std::copy_n(x_one_indexs, x_one_size, x_trans_indexs + diff);
  int p = 0;
  int q = diff;
  for (int i = 0; i < max_dim; ++i) {
    if (q < max_dim && i == x_trans_indexs[q]) {
      ++q;
    } else {
      x_trans_indexs[p++] = i;
    }
  }
}

#if defined(__NVCC__) || defined(__HIPCC__)
template <typename T, typename DX_OP, typename DY_OP>
static __global__ void ElemwiseGradBroadcast1CUDAKernel(
    const T *x, const T *y, const T *out, const T *dout, int h, int w,
    bool is_xsize_larger, DX_OP dx_op, DY_OP dy_op, T *dx, T *dy) {
  int j = blockIdx.x;
  int i = threadIdx.x;
  int tid = threadIdx.x;
  T val(0);
  if (is_xsize_larger) {
    do {
      int x_offset = i * w + j;
      if (dx) {
        dx[x_offset] = dx_op(x[x_offset], y[j], out[x_offset], dout[x_offset]);
      }
      if (dy) {
        val += dy_op(x[x_offset], y[j], out[x_offset], dout[x_offset]);
      }
      i += ELEMWISE_MAX_BLOCK_DIM;
    } while (i < h);

    if (dy) {
      h = h > ELEMWISE_MAX_BLOCK_DIM ? ELEMWISE_MAX_BLOCK_DIM : h;
      val = paddle::platform::reduceSum(val, tid, h);
      if (threadIdx.x == 0) {
        dy[j] = val;
      }
    }
  } else {  // x.dims < y.dims, broadcast for x.
    do {
      int y_offset = i * w + j;
      if (dy) {
        dy[y_offset] = dy_op(x[j], y[y_offset], out[y_offset], dout[y_offset]);
      }
      if (dx) {
        val += dx_op(x[j], y[y_offset], out[y_offset], dout[y_offset]);
      }
      i += ELEMWISE_MAX_BLOCK_DIM;
    } while (i < h);

    if (dx) {
      h = h > ELEMWISE_MAX_BLOCK_DIM ? ELEMWISE_MAX_BLOCK_DIM : h;
      val = paddle::platform::reduceSum(val, tid, h);
      if (threadIdx.x == 0) {
        dx[j] = val;
      }
    }
  }
}

// suppose use 2D block is fast because more parallel
// and memory coalesced
template <typename T, typename DX_OP, typename DY_OP>
static __global__ void FastElemwiseGradBroadcast1CUDAKernel(
    const T *x, const T *y, const T *out, const T *dout, int h, int w,
    bool is_xsize_larger, DX_OP dx_op, DY_OP dy_op, T *dx, T *dy) {
  __shared__ T sdata[BLOCK_Y][BLOCK_X + 1];

  T val(0);
  size_t width_stride = gridDim.x * blockDim.x;
  size_t idx = threadIdx.x + blockDim.x * blockIdx.x;
  size_t full_width =
      (w & (~((uint64_t)(BLOCK_X - 1)))) + ((w & (BLOCK_X - 1)) ? BLOCK_X : 0);
  size_t full_height =
      (h & (~((uint64_t)(BLOCK_Y - 1)))) + ((h & (BLOCK_Y - 1)) ? BLOCK_Y : 0);
  if (is_xsize_larger) {
    for (int m = idx; m < full_width; m += width_stride) {
      sdata[threadIdx.y][threadIdx.x] = 0;
      for (int n = threadIdx.y; n < full_height; n += BLOCK_Y) {
        int x_offset = n * w + m;
        if (dx && m < w && n < h) {
          dx[x_offset] =
              dx_op(x[x_offset], y[m], out[x_offset], dout[x_offset]);
        }
        if (dy) {
          if (m < w && n < h) {
            T val = dy_op(x[x_offset], y[m], out[x_offset], dout[x_offset]);
            sdata[threadIdx.y][threadIdx.x] += val;
          }
          __syncthreads();
        }
      }
      if (dy) {
        T my_val = sdata[threadIdx.x][threadIdx.y];
        for (int i = warpSize >> 1; i > 0; i >>= 1)
          my_val += platform::CudaShuffleXorSync(0xFFFFFFFF, my_val, i);
        __syncthreads();
        if ((threadIdx.x == 0)) {
          sdata[0][threadIdx.y] = my_val;
        }
        __syncthreads();
        if (threadIdx.y == 0 && m < w) {
          dy[m] = sdata[0][threadIdx.x];
        }
      }
    }
  } else {  // x.dims < y.dims, broadcast for x.
    for (int m = idx; m < full_width; m += width_stride) {
      sdata[threadIdx.y][threadIdx.x] = 0;
      for (int n = threadIdx.y; n < full_height; n += BLOCK_Y) {
        int y_offset = n * w + m;
        if (dy && m < w && n < h) {
          dy[y_offset] =
              dy_op(x[m], y[y_offset], out[y_offset], dout[y_offset]);
        }
        if (dx) {
          if (m < w && n < h) {
            T val = dx_op(x[m], y[y_offset], out[y_offset], dout[y_offset]);
            sdata[threadIdx.y][threadIdx.x] += val;
          }
          __syncthreads();
        }
      }
      if (dx) {
        T my_val = sdata[threadIdx.x][threadIdx.y];
        for (int i = warpSize >> 1; i > 0; i >>= 1)
          my_val += platform::CudaShuffleXorSync(0xFFFFFFFF, my_val, i);
        __syncthreads();
        if ((threadIdx.x == 0)) {
          sdata[0][threadIdx.y] = my_val;
        }
        __syncthreads();
        if (threadIdx.y == 0 && m < w) {
          dx[m] = sdata[0][threadIdx.x];
        }
      }
    }
  }
}

template <typename T, typename DX_OP>
__global__ void CommonGradBroadcastCUDAKernel(
    const int *x_strides_array, const int *y_strides_array,
    const int *out_dims_array, const int *y_strides_order,
    const int *y_dims_order, const T *x, const T *y, const T *out,
    const T *dout, T *dx, int out_size, int max_dim, int thread_num,
    DX_OP dx_op) {
  T val(0);
  int i = blockIdx.x;
  int tid = threadIdx.x;
  for (int j = tid; j < thread_num; j += blockDim.x) {
    const int X_index = i * thread_num + j;
    int out_index = X_index;
    int C_index = 0;
    int B_index = i * thread_num + j;
    int remainder = 0;
#pragma unroll
    for (int d = max_dim - 1; d >= 0; --d) {
      GetDivMod(B_index, y_dims_order[d], &B_index, &remainder);
      C_index += remainder * y_strides_order[d];
    }
    int x_index = 0;
    int y_index = 0;
    int C_index_val = C_index;
#pragma unroll
    for (int d = max_dim - 1; d >= 0; --d) {
      GetDivMod(C_index_val, out_dims_array[d], &C_index_val, &remainder);
      x_index += remainder * x_strides_array[d];
      y_index += remainder * y_strides_array[d];
    }
    out_index = C_index;
    val += dx_op(x[x_index], y[y_index], out[out_index], dout[out_index]);
  }
  val = paddle::platform::reduceSum(val, tid, thread_num);
  if (threadIdx.x == 0) {
    dx[i] = val;
  }
}

template <typename T, typename DY_OP>
static __global__ void CommonGradBroadcast1CUDAKernelHeight(
    const T *x, const T *y, const T *out, const T *dout, int h, int w,
    DY_OP dy_op, T *dy, int x_h, int x_w, bool is_y) {
  int j = blockIdx.x;
  int i = threadIdx.x;
  int tid = threadIdx.x;
  T val(0);

  if (is_y) {
    do {
      int out_offset = i * w + j;
      int x_offset = (i % x_h) * x_w + j % x_w;
      if (dy) {
        val += dy_op(x[x_offset], y[j], out[out_offset], dout[out_offset]);
      }
      i += ELEMWISE_MAX_BLOCK_DIM;
    } while (i < h);

    if (dy) {
      h = h > ELEMWISE_MAX_BLOCK_DIM ? ELEMWISE_MAX_BLOCK_DIM : h;
      val = paddle::platform::reduceSum(val, tid, h);
      if (threadIdx.x == 0) {
        dy[j] = val;
      }
    }
  } else {
    do {
      int out_offset = i * w + j;
      int y_offset = (i % x_h) * x_w + j % x_w;
      if (dy) {
        val += dy_op(x[j], y[y_offset], out[out_offset], dout[out_offset]);
      }
      i += ELEMWISE_MAX_BLOCK_DIM;
    } while (i < h);

    if (dy) {
      h = h > ELEMWISE_MAX_BLOCK_DIM ? ELEMWISE_MAX_BLOCK_DIM : h;
      val = paddle::platform::reduceSum(val, tid, h);
      if (threadIdx.x == 0) {
        dy[j] = val;
      }
    }
  }
}

template <typename T, typename DY_OP>
static __global__ void FastCommonGradBroadcastCUDAKernelHeight(
    const T *x, const T *y, const T *out, const T *dout, int h, int w,
    DY_OP dy_op, T *dy, int x_h, int x_w, bool is_y) {
  __shared__ T sdata[BLOCK_Y][BLOCK_X + 1];

  T val(0);
  size_t width_stride = gridDim.x * blockDim.x;
  size_t idx = threadIdx.x + blockDim.x * blockIdx.x;
  size_t full_width =
      (w & (~((uint64_t)(BLOCK_X - 1)))) + ((w & (BLOCK_X - 1)) ? BLOCK_X : 0);
  size_t full_height =
      (h & (~((uint64_t)(BLOCK_Y - 1)))) + ((h & (BLOCK_Y - 1)) ? BLOCK_Y : 0);
  if (is_y) {
    for (int m = idx; m < full_width; m += width_stride) {
      sdata[threadIdx.y][threadIdx.x] = 0;
      for (int n = threadIdx.y; n < full_height; n += BLOCK_Y) {
        int out_offset = n * w + m;
        int x_offset = (n % x_h) * x_w + m % x_w;
        if (dy) {
          if (m < w && n < h) {
            T val = dy_op(x[x_offset], y[m], out[out_offset], dout[out_offset]);
            sdata[threadIdx.y][threadIdx.x] += val;
          }
          __syncthreads();
        }
      }
      if (dy) {
        T my_val = sdata[threadIdx.x][threadIdx.y];
        for (int i = warpSize >> 1; i > 0; i >>= 1) {
          my_val += platform::CudaShuffleXorSync(0xFFFFFFFF, my_val, i);
        }
        __syncthreads();
        if ((threadIdx.x == 0)) {
          sdata[0][threadIdx.y] = my_val;
        }
        __syncthreads();
        if (threadIdx.y == 0 && m < w) {
          dy[m] = sdata[0][threadIdx.x];
        }
      }
    }
  } else {
    for (int m = idx; m < full_width; m += width_stride) {
      sdata[threadIdx.y][threadIdx.x] = 0;
      for (int n = threadIdx.y; n < full_height; n += BLOCK_Y) {
        int out_offset = n * w + m;
        int y_offset = (n % x_h) * x_w + m % x_w;
        if (dy) {
          if (m < w && n < h) {
            T val = dy_op(x[m], y[y_offset], out[out_offset], dout[out_offset]);
            sdata[threadIdx.y][threadIdx.x] += val;
          }
          __syncthreads();
        }
      }
      if (dy) {
        T my_val = sdata[threadIdx.x][threadIdx.y];
        for (int i = warpSize >> 1; i > 0; i >>= 1) {
          my_val += platform::CudaShuffleXorSync(0xFFFFFFFF, my_val, i);
        }
        __syncthreads();
        if ((threadIdx.x == 0)) {
          sdata[0][threadIdx.y] = my_val;
        }
        __syncthreads();
        if (threadIdx.y == 0 && m < w) {
          dy[m] = sdata[0][threadIdx.x];
        }
      }
    }
  }
}

template <typename T, typename DY_OP, typename DX_OP>
static __global__ void FastCommonGradBroadcastAllCUDAKernel(
    const T *x, const T *y, const T *out, const T *dout, int pre, int n,
    int post, bool is_xsize_larger, DX_OP dx_op, DY_OP dy_op, T *dx, T *dy) {
  int tid = threadIdx.x;
  int bid = blockIdx.x;

  T val(0);
  if (is_xsize_larger) {
    for (int i = tid; i < n; i += ELEMWISE_MAX_BLOCK_DIM) {
      int b_i = bid / post;
      int b_j = bid % post;
      int x_offset = b_i * n * post + i * post + b_j;
      int y_offset = b_i * post + b_j;
      if (dx) {
        dx[x_offset] =
            dx_op(x[x_offset], y[y_offset], out[x_offset], dout[x_offset]);
      }
      if (dy) {
        val += dy_op(x[x_offset], y[y_offset], out[x_offset], dout[x_offset]);
      }
    }
    if (dy) {
      int h = n > ELEMWISE_MAX_BLOCK_DIM ? ELEMWISE_MAX_BLOCK_DIM : n;
      val = paddle::platform::reduceSum(val, tid, h);
      if (tid == 0) {
        dy[bid] = val;
      }
    }
  } else {
    for (int i = tid; i < n; i += ELEMWISE_MAX_BLOCK_DIM) {
      int b_i = bid / post;
      int b_j = bid % post;
      int y_offset = b_i * n * post + i * post + b_j;
      int x_offset = b_i * post + b_j;
      if (dy) {
        dy[y_offset] =
            dy_op(x[x_offset], y[y_offset], out[y_offset], dout[y_offset]);
      }
      if (dx) {
        val += dx_op(x[x_offset], y[y_offset], out[y_offset], dout[y_offset]);
      }
    }
    if (dx) {
      int h = n > ELEMWISE_MAX_BLOCK_DIM ? ELEMWISE_MAX_BLOCK_DIM : n;
      val = paddle::platform::reduceSum(val, tid, h);
      if (tid == 0) {
        dx[bid] = val;
      }
    }
  }
}

template <typename T, typename OP>
static __global__ void FastCommonGradBroadcastOneCUDAKernel(
    const T *x, const T *y, const T *out, const T *dout, int pre, int n,
    int post, int y_pre, int y_n, int y_post, bool is_xsize, OP op, T *dd) {
  int tid = threadIdx.x;
  int bid = blockIdx.x;

  T val(0);
  if (is_xsize) {
    // do reduce for x
    for (int i = tid; i < n; i += ELEMWISE_MAX_BLOCK_DIM) {
      int b_i = bid / post;
      int b_j = bid % post;
      int x_offset = b_i * n * post + b_j;
      int out_offset = b_i * n * post + i * post + b_j;

      // Get y pre rows id with x post and y_pre.
      int b_yi = bid / (post * y_pre);
      int b_yj = bid % y_post;
      int y_offset = b_yi * y_n + i * y_post + b_yj;

      if (dd) {
        val += op(x[x_offset], y[y_offset], out[out_offset], dout[out_offset]);
      }
    }
    if (dd) {
      int h = n > ELEMWISE_MAX_BLOCK_DIM ? ELEMWISE_MAX_BLOCK_DIM : n;
      val = paddle::platform::reduceSum(val, tid, h);
      if (tid == 0) {
        dd[bid] = val;
      }
    }
  } else {
    // do reduce for y
    for (int i = tid; i < n; i += ELEMWISE_MAX_BLOCK_DIM) {
      int b_i = bid / post;
      int b_j = bid % post;
      int y_offset = b_i * n * post + b_j;
      int out_offset = b_i * n * post + i * post + b_j;

      int b_yi = bid / (post * y_pre);
      int b_yj = bid % y_post;
      int x_offset = b_yi * y_n + i * y_post + b_yj;

      if (dd) {
        val += op(x[x_offset], y[y_offset], out[out_offset], dout[out_offset]);
      }
    }
    if (dd) {
      int h = n > ELEMWISE_MAX_BLOCK_DIM ? ELEMWISE_MAX_BLOCK_DIM : n;
      val = paddle::platform::reduceSum(val, tid, h);
      if (tid == 0) {
        dd[bid] = val;
      }
    }
  }
}

// Check input can be split into 2 parts
static inline bool SplitDims(const std::vector<int> &y_broadcast_pos,
                             int max_dim) {
  bool can_split_dim2 = true;
  // must at start or end.
  if (y_broadcast_pos[0] != 0 &&
      y_broadcast_pos[y_broadcast_pos.size() - 1] != max_dim - 1) {
    can_split_dim2 = false;
  } else {
    for (int i = 1; i < y_broadcast_pos.size(); ++i) {
      // dim must be continue
      if (y_broadcast_pos[i] != y_broadcast_pos[i - 1] + 1) {
        can_split_dim2 = false;
        break;
      }
    }
  }
  return can_split_dim2;
}

// Suppose only has contiguous dims
static inline bool CheckContiguousDims(const std::vector<int> &broadcast_pos) {
  for (int i = 1; i < broadcast_pos.size(); ++i) {
    if (broadcast_pos[i] != broadcast_pos[i - 1] + 1) {
      return false;
    }
  }
  return true;
}

template <typename T, typename DX_OP, typename DY_OP>
void CommonGradBroadcastCUDA(
    const framework::Tensor &x, const framework::Tensor &y,
    const framework::Tensor &out, const framework::Tensor &dout,
    framework::Tensor *dx, framework::Tensor *dy, int *x_dims_array,
    int *y_dims_array, int *out_dims_array, int max_dim,
    const platform::CUDADeviceContext &ctx, DX_OP dx_op, DY_OP dy_op) {
  const auto gplace = BOOST_GET_CONST(platform::CUDAPlace, ctx.GetPlace());
  auto cplace = platform::CPUPlace();
  const T *x_data = x.data<T>();
  const T *y_data = y.data<T>();
  const T *out_data = out.data<T>();
  const T *dout_data = dout.data<T>();
  T *dx_data = dx == nullptr ? nullptr : dx->mutable_data<T>(ctx.GetPlace());
  T *dy_data = dy == nullptr ? nullptr : dy->mutable_data<T>(ctx.GetPlace());

  std::vector<int> x_one_indexs;
  std::vector<int> y_one_indexs;
  for (int i = 0; i < max_dim; i++) {
    if (x_dims_array[i] != y_dims_array[i]) {
      if (x_dims_array[i] == 1) {
        x_one_indexs.push_back(i);
      }
      if (y_dims_array[i] == 1) {
        y_one_indexs.push_back(i);
      }
    }
  }

  std::vector<int> x_trans_indexs(max_dim);
  std::vector<int> y_trans_indexs(max_dim);
  ComputeBroadcastTranspositionArray(x_one_indexs.data(), x_trans_indexs.data(),
                                     max_dim, x_one_indexs.size());
  ComputeBroadcastTranspositionArray(y_one_indexs.data(), y_trans_indexs.data(),
                                     max_dim, y_one_indexs.size());

  // compute array stride for cuda kernel;
  // e.g. x.dims=[2,3,4], x_stride=[12,4,1]
  std::vector<int> x_strides_array(max_dim);
  std::vector<int> y_strides_array(max_dim);
  std::vector<int> out_strides_array(max_dim);
  int x_stride = 1;
  int y_stride = 1;
  int z_stride = 1;
  for (int i = max_dim - 1; i >= 0; i--) {
    x_strides_array[i] = x_dims_array[i] == 1 ? 0 : x_stride;
    y_strides_array[i] = y_dims_array[i] == 1 ? 0 : y_stride;
    out_strides_array[i] = z_stride;
    x_stride *= x_dims_array[i];
    y_stride *= y_dims_array[i];
    z_stride *= out_dims_array[i];
  }

  std::vector<int> x_strides_order(max_dim);
  std::vector<int> y_strides_order(max_dim);
  std::vector<int> x_dims_order(max_dim);
  std::vector<int> y_dims_order(max_dim);
  for (int i = 0; i < max_dim; ++i) {
    x_strides_order[i] = out_strides_array[x_trans_indexs[i]];
    y_strides_order[i] = out_strides_array[y_trans_indexs[i]];
    x_dims_order[i] = out_dims_array[x_trans_indexs[i]];
    y_dims_order[i] = out_dims_array[y_trans_indexs[i]];
  }
  std::vector<int> x_broadcast_pos;
  std::vector<int> y_broadcast_pos;

  int bytes = max_dim * sizeof(int);

  for (int i = 0; i < max_dim; ++i) {
    if (x_dims_array[i] != out_dims_array[i] && x_dims_array[i] == 1) {
      x_broadcast_pos.emplace_back(i);
    }
    if (y_dims_array[i] != out_dims_array[i] && y_dims_array[i] == 1) {
      y_broadcast_pos.emplace_back(i);
    }
  }

  auto stream = ctx.stream();
  bool can_split_x = false;
  bool can_split_y = false;

  auto FastCommonCUDAF = [&](const std::vector<int> &broadcast_pos, bool is_y) {
    int h =
        std::accumulate(out_dims_array, out_dims_array + broadcast_pos.size(),
                        1, std::multiplies<int>());
    int w =
        std::accumulate(out_dims_array + broadcast_pos.size(),
                        out_dims_array + max_dim, 1, std::multiplies<int>());

    VLOG(3) << "FastCommonCUDAF elementwise w:" << w << " h:" << h
            << " is_y:" << is_y;

    int split_h;
    int split_w;
    int kh = h;
    int kw = w;

    if (is_y) {
      split_h =
          std::accumulate(x_dims_array, x_dims_array + broadcast_pos.size(), 1,
                          std::multiplies<int>());
      split_w =
          std::accumulate(x_dims_array + broadcast_pos.size(),
                          x_dims_array + max_dim, 1, std::multiplies<int>());

    } else {
      split_h =
          std::accumulate(y_dims_array, y_dims_array + broadcast_pos.size(), 1,
                          std::multiplies<int>());
      split_w =
          std::accumulate(y_dims_array + broadcast_pos.size(),
                          y_dims_array + max_dim, 1, std::multiplies<int>());
    }

    if (h > split_h) kh = split_h;
    if (w > split_w) kw = split_w;

    if (is_y) {
      if (w < 16 || h < 16) {
        int block_size = std::min(ELEMWISE_MAX_BLOCK_DIM, h);
        int grid_size = w;
        CommonGradBroadcast1CUDAKernelHeight<<<grid_size, block_size, 0,
                                               stream>>>(
            x_data, y_data, out_data, dout_data, h, w, dy_op, dy_data, kh, kw,
            is_y);
      } else {
        dim3 block_size = dim3(BLOCK_X, BLOCK_Y);
        int grid_size = (w + BLOCK_X - 1) / BLOCK_X;
        FastCommonGradBroadcastCUDAKernelHeight<<<grid_size, block_size, 0,
                                                  stream>>>(
            x_data, y_data, out_data, dout_data, h, w, dy_op, dy_data, kh, kw,
            is_y);
      }
    } else {
      if (w < 16 || h < 16) {
        int block_size = std::min(ELEMWISE_MAX_BLOCK_DIM, h);
        int grid_size = w;
        CommonGradBroadcast1CUDAKernelHeight<<<grid_size, block_size, 0,
                                               stream>>>(
            x_data, y_data, out_data, dout_data, h, w, dx_op, dx_data, kh, kw,
            is_y);
      } else {
        dim3 block_size = dim3(BLOCK_X, BLOCK_Y);
        int grid_size = (w + BLOCK_X - 1) / BLOCK_X;
        FastCommonGradBroadcastCUDAKernelHeight<<<grid_size, block_size, 0,
                                                  stream>>>(
            x_data, y_data, out_data, dout_data, h, w, dx_op, dx_data, kh, kw,
            is_y);
      }
    }
  };

  auto FastBroadCastHeightCUDAF = [&](const std::vector<int> &broadcast_pos,
                                      bool x_large) {
    int h =
        std::accumulate(out_dims_array, out_dims_array + broadcast_pos.size(),
                        1, std::multiplies<int>());
    int w =
        std::accumulate(out_dims_array + broadcast_pos.size(),
                        out_dims_array + max_dim, 1, std::multiplies<int>());

    VLOG(3) << "FastBroadCastHeightCUDAF w:" << w << " h:" << h;

    if (w < 16 || h < 16) {
      int block_size = std::min(ELEMWISE_MAX_BLOCK_DIM, h);
      int grid_size = w;
      ElemwiseGradBroadcast1CUDAKernel<<<grid_size, block_size, 0, stream>>>(
          x_data, y_data, out_data, dout_data, h, w, x_large, dx_op, dy_op,
          dx_data, dy_data);
    } else {
      dim3 block_size = dim3(BLOCK_X, BLOCK_Y);
      int grid_size = (w + BLOCK_X - 1) / BLOCK_X;
      FastElemwiseGradBroadcast1CUDAKernel<<<grid_size, block_size, 0,
                                             stream>>>(
          x_data, y_data, out_data, dout_data, h, w, x_large, dx_op, dy_op,
          dx_data, dy_data);
    }
  };

  auto FastBroadCastAllCUDAF = [&](const std::vector<int> &broadcast_pos,
                                   int max_dim, bool is_x_large) {
    int axis = broadcast_pos[0];
    int pre = std::accumulate(out_dims_array, out_dims_array + axis, 1,
                              std::multiplies<int>());
    int mid = 1;
    int post = 1;

    if (broadcast_pos.size() == 1) {
      mid = out_dims_array[axis];
      post =
          std::accumulate(out_dims_array + axis + 1, out_dims_array + max_dim,
                          1, std::multiplies<int>());
    } else {
      mid = std::accumulate(out_dims_array + axis,
                            out_dims_array + broadcast_pos.back() + 1, 1,
                            std::multiplies<int>());
      post =
          std::accumulate(out_dims_array + broadcast_pos.back() + 1,
                          out_dims_array + max_dim, 1, std::multiplies<int>());
    }

    VLOG(3) << "FastBroadCastAllCUDAF pre:" << pre << " mid:" << mid
            << " post:" << post;

    int block_size = std::min(ELEMWISE_MAX_BLOCK_DIM, mid);
    int grid_size = pre * post;

    FastCommonGradBroadcastAllCUDAKernel<<<grid_size, block_size, 0, stream>>>(
        x_data, y_data, out_data, dout_data, pre, mid, post, is_x_large, dx_op,
        dy_op, dx_data, dy_data);
  };

  auto FastBroadCastOneCUDAF = [&](const std::vector<int> &broadcast_pos,
                                   int max_dim, bool is_x) {
    int axis = broadcast_pos[0];
    int pre = std::accumulate(out_dims_array, out_dims_array + axis, 1,
                              std::multiplies<int>());
    int mid = out_dims_array[axis];
    int post =
        std::accumulate(out_dims_array + axis + 1, out_dims_array + max_dim, 1,
                        std::multiplies<int>());

    int k_pre;
    int k_mid;
    int k_post;

    if (is_x) {
      k_pre = std::accumulate(y_dims_array, y_dims_array + axis, 1,
                              std::multiplies<int>());
      k_mid = y_dims_array[axis];
      k_post = std::accumulate(y_dims_array + axis + 1, y_dims_array + max_dim,
                               1, std::multiplies<int>());
      int block_size = std::min(ELEMWISE_MAX_BLOCK_DIM, mid);
      int grid_size = pre * post;
      // we need to calc y offset with blockid, so do x_pre/y_pre to get left
      // size.
      if (k_pre != pre) k_pre = pre / k_pre;

      FastCommonGradBroadcastOneCUDAKernel<<<grid_size, block_size, 0,
                                             stream>>>(
          x_data, y_data, out_data, dout_data, pre, mid, post, k_pre, k_mid,
          k_post, true, dx_op, dx_data);
    } else {
      k_pre = std::accumulate(x_dims_array, x_dims_array + axis, 1,
                              std::multiplies<int>());
      k_mid = x_dims_array[axis];
      k_post = std::accumulate(x_dims_array + axis + 1, x_dims_array + max_dim,
                               1, std::multiplies<int>());
      int block_size = std::min(ELEMWISE_MAX_BLOCK_DIM, mid);
      int grid_size = pre * post;
      if (k_pre != pre) k_pre = pre / k_pre;

      FastCommonGradBroadcastOneCUDAKernel<<<grid_size, block_size, 0,
                                             stream>>>(
          x_data, y_data, out_data, dout_data, pre, mid, post, k_pre, k_mid,
          k_post, false, dy_op, dy_data);
    }
    VLOG(3) << "FastBroadCastOneCUDAF pre:" << pre << " mid:" << mid
            << " post:" << post;
  };

  // do fast elementwise if: 1. only one input need to do broadcast, we can
  // fallback
  // to old fast path.
  // 2. if both x and y need broadcast, then do it one by one.
  bool fast_broadcast = false;
  if (x_broadcast_pos.empty() && !y_broadcast_pos.empty()) {
    can_split_y = SplitDims(y_broadcast_pos, max_dim);
    if (can_split_y) {
      // only y need to do broadcast on h
      if (y_broadcast_pos[0] == 0) {
        FastBroadCastHeightCUDAF(y_broadcast_pos, true);
        fast_broadcast = true;
      }
    } else if (y_broadcast_pos.size() == 1 ||
               CheckContiguousDims(y_broadcast_pos)) {  // for only one dim and
                                                        // contiguous broadcast.
      // If cannot split,  which means input has 3 parts
      FastBroadCastAllCUDAF(y_broadcast_pos, max_dim, true);
      fast_broadcast = true;
    }
  } else if (y_broadcast_pos.empty() && !x_broadcast_pos.empty()) {
    // only x need broadcast
    can_split_x = SplitDims(x_broadcast_pos, max_dim);
    if (can_split_x) {
      if (x_broadcast_pos[0] == 0) {
        FastBroadCastHeightCUDAF(x_broadcast_pos, false);
        fast_broadcast = true;
      }
    } else if (x_broadcast_pos.size() == 1 ||
               CheckContiguousDims(x_broadcast_pos)) {
      FastBroadCastAllCUDAF(x_broadcast_pos, max_dim, false);
      fast_broadcast = true;
    }
  } else if (!x_broadcast_pos.empty() && !y_broadcast_pos.empty()) {
    // do x and y broadcast each.
    can_split_y = SplitDims(y_broadcast_pos, max_dim);
    bool fast_broadcast_x = false;
    bool fast_broadcast_y = false;
    if (can_split_y) {
      // begin at start.
      if (y_broadcast_pos[0] == 0) {
        FastCommonCUDAF(y_broadcast_pos, true);
        fast_broadcast_y = true;
      }
    } else if (y_broadcast_pos.size() == 1) {
      FastBroadCastOneCUDAF(y_broadcast_pos, max_dim, false);
      can_split_y = true;
      fast_broadcast_y = true;
    }
    can_split_x = SplitDims(x_broadcast_pos, max_dim);
    if (can_split_x) {
      if (x_broadcast_pos[0] == 0) {
        FastCommonCUDAF(x_broadcast_pos, false);
        fast_broadcast_x = true;
      }
    } else if (x_broadcast_pos.size() == 1) {
      FastBroadCastOneCUDAF(x_broadcast_pos, max_dim, true);
      can_split_x = true;
      fast_broadcast_x = true;
    }
    VLOG(3) << "CommonBroadcast can_split_y:" << can_split_y
            << " can_split_x:" << can_split_x;
    // if both x and y into fast path then return
    if (fast_broadcast_x && fast_broadcast_y) {
      fast_broadcast = true;
    }
    if (can_split_y && can_split_x && fast_broadcast) return;
  }

  // Should remove memory copy, use reg instead.
  if (fast_broadcast) {
    return;
  }
  int x_blocks = 0;
  int x_threads = 0;
  ComputeBroadcastKernelSize(x_dims_array, out_dims_array, &x_blocks,
                             &x_threads, max_dim);
  int y_blocks = 0;
  int y_threads = 0;
  ComputeBroadcastKernelSize(y_dims_array, out_dims_array, &y_blocks,
                             &y_threads, max_dim);

  auto x_strides_array_tmp = memory::Alloc(ctx, bytes);
  int *x_strides_array_gpu =
      reinterpret_cast<int *>(x_strides_array_tmp->ptr());
  memory::Copy(gplace, x_strides_array_gpu, cplace, x_strides_array.data(),
               bytes, ctx.stream());

  auto y_strides_array_tmp = memory::Alloc(ctx, bytes);
  int *y_strides_array_gpu =
      reinterpret_cast<int *>(y_strides_array_tmp->ptr());
  memory::Copy(gplace, y_strides_array_gpu, cplace, y_strides_array.data(),
               bytes, ctx.stream());

  auto out_dims_array_tmp = memory::Alloc(ctx, bytes);
  int *out_dims_array_gpu = reinterpret_cast<int *>(out_dims_array_tmp->ptr());
  memory::Copy(gplace, out_dims_array_gpu, cplace, out_dims_array, bytes,
               ctx.stream());

  const int out_size = std::accumulate(out_dims_array, out_dims_array + max_dim,
                                       1, std::multiplies<int>());
  int x_block_size = std::min(ELEMWISE_MAX_BLOCK_DIM, x_threads);
  int y_block_size = std::min(ELEMWISE_MAX_BLOCK_DIM, y_threads);
  if (dx) {
    auto x_strides_order_tmp = memory::Alloc(ctx, bytes);
    int *x_strides_order_gpu =
        reinterpret_cast<int *>(x_strides_order_tmp->ptr());
    memory::Copy(gplace, x_strides_order_gpu, cplace, x_strides_order.data(),
                 bytes, ctx.stream());

    auto x_dims_order_tmp = memory::Alloc(ctx, bytes);
    int *x_dims_order_gpu = reinterpret_cast<int *>(x_dims_order_tmp->ptr());
    memory::Copy(gplace, x_dims_order_gpu, cplace, x_dims_order.data(), bytes,
                 ctx.stream());
    CommonGradBroadcastCUDAKernel<
        T, DX_OP><<<x_blocks, x_block_size, 0, ctx.stream()>>>(
        x_strides_array_gpu, y_strides_array_gpu, out_dims_array_gpu,
        x_strides_order_gpu, x_dims_order_gpu, x_data, y_data, out_data,
        dout_data, dx_data, out_size, max_dim, x_threads, dx_op);
  }
  if (dy) {
    auto y_strides_order_tmp = memory::Alloc(ctx, bytes);
    int *y_strides_order_gpu =
        reinterpret_cast<int *>(y_strides_order_tmp->ptr());
    memory::Copy(gplace, y_strides_order_gpu, cplace, y_strides_order.data(),
                 bytes, ctx.stream());

    auto y_dims_order_tmp = memory::Alloc(ctx, bytes);
    int *y_dims_order_gpu = reinterpret_cast<int *>(y_dims_order_tmp->ptr());
    memory::Copy(gplace, y_dims_order_gpu, cplace, y_dims_order.data(), bytes,
                 ctx.stream());
    CommonGradBroadcastCUDAKernel<
        T, DY_OP><<<y_blocks, y_block_size, 0, ctx.stream()>>>(
        x_strides_array_gpu, y_strides_array_gpu, out_dims_array_gpu,
        y_strides_order_gpu, y_dims_order_gpu, x_data, y_data, out_data,
        dout_data, dy_data, out_size, max_dim, y_threads, dy_op);
  }
}

#endif  // __NVCC__ or __HIPCC__

inline framework::DDim trim_trailing_singular_dims(
    const framework::DDim &dims) {
  // Remove trailing dimensions of size 1 for y
  auto actual_dims_size = dims.size();
  for (; actual_dims_size != 0; --actual_dims_size) {
    if (dims[actual_dims_size - 1] != 1) break;
  }
  if (actual_dims_size == dims.size()) return dims;
  std::vector<int> trim_dims;
  trim_dims.resize(actual_dims_size);
  for (int i = 0; i < actual_dims_size; ++i) {
    trim_dims[i] = dims[i];
  }
  if (trim_dims.size() == 0) {
    return framework::DDim(framework::make_dim());
  }
  framework::DDim actual_dims = framework::make_ddim(trim_dims);
  return actual_dims;
}

template <typename T, typename DeviceContext>
class RowwiseTransformIterator;

template <typename T, typename DeviceContext>
class MidWiseTransformIterator;

// NOTE(dzhwinter): ptrdiff_t in iterator is deperecated in c++17
template <typename T>
class RowwiseTransformIterator<T, platform::CPUDeviceContext>
    : public std::iterator<std::random_access_iterator_tag, T, std::ptrdiff_t,
                           T *, T &> {
 public:
  RowwiseTransformIterator(const T *ptr, int n) : ptr_(ptr), i_(0), n_(n) {}

  RowwiseTransformIterator<T, platform::CPUDeviceContext> &operator++() {
    ++i_;
    if (UNLIKELY(i_ == n_)) {
      i_ = 0;
    }
    return *this;
  }

  RowwiseTransformIterator<T, platform::CPUDeviceContext> &operator+(int n) {
    while (n-- > 0) {
      ++i_;
      if (UNLIKELY(i_ == n_)) {
        i_ = 0;
      }
    }

    return *this;
  }

  bool operator==(const RowwiseTransformIterator<T, platform::CPUDeviceContext>
                      &rhs) const {
    return (ptr_ + i_) == &(*rhs);
  }

  bool operator!=(const RowwiseTransformIterator<T, platform::CPUDeviceContext>
                      &rhs) const {
    return (ptr_ + i_) != &(*rhs);
  }

  const T &operator*() { return ptr_[i_]; }

 private:
  const T *ptr_;
  int i_;
  int64_t n_;
};

template <typename T>
class MidWiseTransformIterator<T, platform::CPUDeviceContext>
    : public std::iterator<std::random_access_iterator_tag, T, std::ptrdiff_t,
                           T *, T &> {
 public:
  MidWiseTransformIterator(const T *ptr, int n, int post)
      : ptr_(ptr), i_(0), j_(0), n_(n), post_(post) {}

  MidWiseTransformIterator<T, platform::CPUDeviceContext> &operator++() {
    ++j_;
    if (UNLIKELY(j_ == post_)) {
      ++i_;
      j_ = 0;
      if (UNLIKELY(i_ == n_)) {
        i_ = 0;
      }
    }
    return *this;
  }

  MidWiseTransformIterator<T, platform::CPUDeviceContext> &operator+(int n) {
    while (n-- > 0) {
      ++j_;
      if (UNLIKELY(j_ == post_)) {
        ++i_;
        j_ = 0;
        if (UNLIKELY(i_ == n_)) {
          i_ = 0;
        }
      }
    }
    return *this;
  }

  bool operator==(const MidWiseTransformIterator<T, platform::CPUDeviceContext>
                      &rhs) const {
    return (ptr_ + i_) == &(*rhs);
  }

  bool operator!=(const MidWiseTransformIterator<T, platform::CPUDeviceContext>
                      &rhs) const {
    return (ptr_ + i_) != &(*rhs);
  }

  const T &operator*() { return ptr_[i_]; }

 private:
  const T *ptr_;
  int64_t i_;
  int64_t j_;
  int64_t n_;
  int64_t post_;
};

#if defined(__NVCC__) || defined(__HIPCC__)
template <typename T>
class RowwiseTransformIterator<T, platform::CUDADeviceContext>
    : public thrust::iterator_adaptor<
          RowwiseTransformIterator<T, platform::CUDADeviceContext>, const T *> {
 public:
  typedef thrust::iterator_adaptor<
      RowwiseTransformIterator<T, platform::CUDADeviceContext>, const T *>
      super_t;
  HOSTDEVICE RowwiseTransformIterator(const T *x, int n)
      : super_t(x), begin_(x), n_(n) {}
  friend class thrust::iterator_core_access;

 private:
  unsigned int n_;
  const T *begin_;
  HOSTDEVICE typename super_t::reference dereference() const {
    return *(begin_ + (this->base() - begin_) % n_);
  }
};

template <typename T>
class MidWiseTransformIterator<T, platform::CUDADeviceContext>
    : public thrust::iterator_adaptor<
          MidWiseTransformIterator<T, platform::CUDADeviceContext>, const T *> {
 public:
  typedef thrust::iterator_adaptor<
      MidWiseTransformIterator<T, platform::CUDADeviceContext>, const T *>
      super_t;
  HOSTDEVICE MidWiseTransformIterator(const T *x, int n, int post)
      : super_t(x), begin_(x), n_(n), post_(post) {}
  friend class thrust::iterator_core_access;

 private:
  unsigned int post_;
  unsigned int n_;
  const T *begin_;
  HOSTDEVICE typename super_t::reference dereference() const {
    return *(begin_ + (((this->base() - begin_) / post_) % n_));
  }
};
#endif

template <typename Functor, typename T, typename DeviceContext,
          typename OutType = T>
class TransformFunctor {
 public:
  TransformFunctor(const framework::Tensor *x, const framework::Tensor *y,
                   framework::Tensor *z, const DeviceContext &ctx, Functor func,
                   const bool is_xsize_larger = true)
      : x_(x->data<T>()),
        y_(y->data<T>()),
        z_(z->mutable_data<OutType>(ctx.GetPlace())),
        nx_(x->numel()),
        ctx_(ctx),
        func_(func),
        is_xsize_larger_(is_xsize_larger) {
    if (is_xsize_larger_ == false) {
      nx_ = y->numel();
    }
  }

  inline void Run() const {
    platform::Transform<DeviceContext> trans;
    trans(ctx_, x_, x_ + nx_, y_, z_, func_);
  }

  inline void RunRowWise(int n, int pre) const {
    platform::Transform<DeviceContext> trans;
    if (is_xsize_larger_) {
      trans(ctx_, x_, x_ + nx_,
            RowwiseTransformIterator<T, DeviceContext>(y_, n), z_, func_);
    } else {
      trans(ctx_, y_, y_ + nx_,
            RowwiseTransformIterator<T, DeviceContext>(x_, n), z_, func_);
    }
  }

  inline void RunMidWise(int n, int pre, int post) const {
    platform::Transform<DeviceContext> trans;
    if (is_xsize_larger_) {
      trans(ctx_, x_, x_ + nx_,
            MidWiseTransformIterator<T, DeviceContext>(y_, n, post), z_, func_);
    } else {
      trans(ctx_, y_, y_ + nx_,
            MidWiseTransformIterator<T, DeviceContext>(x_, n, post), z_, func_);
    }
  }

 private:
  const T *x_;
  const T *y_;
  OutType *z_;
  int64_t nx_;
  const DeviceContext &ctx_;
  Functor func_;
  bool is_xsize_larger_;
};

template <typename T, typename DX_OP, typename DY_OP>
struct ElemwiseGradNoBroadcast {
  const T *x_;
  const T *y_;
  const T *out_;
  const T *dout_;

  HOSTDEVICE void operator()(size_t i) {
    if (dx_ != nullptr) {
      dx_[i] = dx_op_(x_[i], y_[i], out_[i], dout_[i]);
    }
    if (dy_ != nullptr) {
      dy_[i] = dy_op_(x_[i], y_[i], out_[i], dout_[i]);
    }
  }

  DX_OP dx_op_;
  DY_OP dy_op_;
  T *dx_;
  T *dy_;
};

template <typename T, typename DX_OP, typename DY_OP>
static void ElemwiseGradBroadcast1CPU(const T *x, const T *y, const T *out,
                                      const T *dout, int h, int w,
                                      bool is_xsize_larger, DX_OP dx_op,
                                      DY_OP dy_op, T *dx, T *dy) {
  if (is_xsize_larger) {
    for (int i = 0; i < h; ++i) {
      for (int j = 0; j < w; ++j) {
        int x_offset = i * w + j;
        if (dx != nullptr) {
          dx[x_offset] =
              dx_op(x[x_offset], y[j], out[x_offset], dout[x_offset]);
        }
        if (dy != nullptr) {
          T tmp = dy_op(x[x_offset], y[j], out[x_offset], dout[x_offset]);
          if (i == 0) {
            dy[j] = tmp;
          } else {
            dy[j] += tmp;
          }
        }
      }
    }
  } else {  // x.dims < y.dims, broadcast for x.
    for (int i = 0; i < h; ++i) {
      for (int j = 0; j < w; ++j) {
        int y_offset = i * w + j;
        if (dy != nullptr) {
          dy[y_offset] =
              dy_op(x[j], y[y_offset], out[y_offset], dout[y_offset]);
        }
        if (dx != nullptr) {
          T tmp = dx_op(x[j], y[y_offset], out[y_offset], dout[y_offset]);
          if (i == 0) {
            dx[j] = tmp;
          } else {
            dx[j] += tmp;
          }
        }
      }
    }
  }
}

#if defined(__NVCC__) || defined(__HIPCC__)

template <typename T, typename DX_OP, typename DY_OP>
static void ElemwiseGradBroadcast1CUDA(gpuStream_t stream, const T *x,
                                       const T *y, const T *out, const T *dout,
                                       int h, int w, bool is_xsize_larger,
                                       DX_OP dx_op, DY_OP dy_op, T *dx, T *dy) {
  // For small case use 1D block
  constexpr int half_walf = 16;
  if (w < half_walf || h < half_walf) {
    int block_size = std::min(ELEMWISE_MAX_BLOCK_DIM, h);
    int gird_size = w;
    ElemwiseGradBroadcast1CUDAKernel<<<gird_size, block_size, 0, stream>>>(
        x, y, out, dout, h, w, is_xsize_larger, dx_op, dy_op, dx, dy);
  } else {
    // suppose perfoemance improves with h increased.
    dim3 block_size = dim3(BLOCK_X, BLOCK_Y);
    int grid_size = (w + BLOCK_X - 1) / BLOCK_X;
    FastElemwiseGradBroadcast1CUDAKernel<<<grid_size, block_size, 0, stream>>>(
        x, y, out, dout, h, w, is_xsize_larger, dx_op, dy_op, dx, dy);
  }
}

#endif

template <typename T, typename DX_OP, typename DY_OP>
static void ElemwiseGradBroadcast2CPU(const T *x, const T *y, const T *out,
                                      const T *dout, int pre, int n, int post,
                                      bool is_xsize_larger, DX_OP dx_op,
                                      DY_OP dy_op, T *dx, T *dy) {
  if (is_xsize_larger) {
    for (int i = 0; i < pre; ++i) {
      for (int j = 0; j < n; ++j) {
        for (int k = 0; k < post; ++k) {
          int x_offset = i * n * post + j * post + k;
          if (dx != nullptr) {
            dx[x_offset] =
                dx_op(x[x_offset], y[j], out[x_offset], dout[x_offset]);
          }
          if (dy != nullptr) {
            T tmp = dy_op(x[x_offset], y[j], out[x_offset], dout[x_offset]);
            if (i == 0 && k == 0) {
              dy[j] = tmp;
            } else {
              dy[j] += tmp;
            }
          }
        }
      }
    }
  } else {  // x.dims < y.dims, broadcast for x.
    for (int i = 0; i < pre; ++i) {
      for (int j = 0; j < n; ++j) {
        for (int k = 0; k < post; ++k) {
          int y_offset = i * n * post + j * post + k;
          if (dy != nullptr) {
            dy[y_offset] =
                dy_op(x[j], y[y_offset], out[y_offset], dout[y_offset]);
          }
          if (dx != nullptr) {
            T tmp = dx_op(x[j], y[y_offset], out[y_offset], dout[y_offset]);
            if (i == 0 && k == 0) {
              dx[j] = tmp;
            } else {
              dx[j] += tmp;
            }
          }
        }
      }
    }
  }
}

#if defined(__NVCC__) || defined(__HIPCC__)
template <typename T, typename DX_OP, typename DY_OP>
static __global__ void ElemwiseGradBroadcast2CUDAKernel(
    const T *x, const T *y, const T *out, const T *dout, int pre, int n,
    int post, bool is_xsize_larger, DX_OP dx_op, DY_OP dy_op, T *dx, T *dy) {
  int tid = threadIdx.x;
  int j = blockIdx.x;

  T val(0);
  int ttid = tid;

  if (is_xsize_larger) {
    T local_y = y[j];
    while (true) {
      int i = ttid / post;
      int k = ttid % post;
      if (i >= pre) break;

      int x_offset = i * n * post + j * post + k;

      if (dx != nullptr) {
        dx[x_offset] =
            dx_op(x[x_offset], local_y, out[x_offset], dout[x_offset]);
      }

      if (dy != nullptr) {
        val += dy_op(x[x_offset], local_y, out[x_offset], dout[x_offset]);
      }

      ttid += ELEMWISE_MAX_BLOCK_DIM;
    }

    if (dy) {
      int h = pre * post;
      h = h > ELEMWISE_MAX_BLOCK_DIM ? ELEMWISE_MAX_BLOCK_DIM : h;
      val = paddle::platform::reduceSum(val, tid, h);
      if (threadIdx.x == 0) {
        dy[j] = val;
      }
    }
  } else {  // x.dims < y.dims, broadcast for x.
    T local_x = x[j];
    while (true) {
      int i = ttid / post;
      int k = ttid % post;
      if (i >= pre) break;

      int y_offset = i * n * post + j * post + k;

      if (dy != nullptr) {
        dy[y_offset] =
            dy_op(local_x, y[y_offset], out[y_offset], dout[y_offset]);
      }

      if (dx != nullptr) {
        val += dx_op(local_x, y[y_offset], out[y_offset], dout[y_offset]);
      }

      ttid += ELEMWISE_MAX_BLOCK_DIM;
    }

    if (dx) {
      int h = pre * post;
      h = h > ELEMWISE_MAX_BLOCK_DIM ? ELEMWISE_MAX_BLOCK_DIM : h;
      val = paddle::platform::reduceSum(val, tid, h);
      if (threadIdx.x == 0) {
        dx[j] = val;
      }
    }
  }
}

template <typename T, int N>
struct GetVecType;

template <typename T>
struct GetVecType<T, 1> {
  using Type = T;
};

template <>
struct GetVecType<paddle::platform::float16, 2> {
  using Type = half2;
};

template <>
struct GetVecType<paddle::platform::float16, 4> {
  using Type = float2;
};

template <>
struct GetVecType<float, 2> {
  using Type = float2;
};

template <>
struct GetVecType<float, 4> {
  using Type = float4;
};

template <>
struct GetVecType<double, 2> {
  using Type = double2;
};

template <>
struct GetVecType<double, 4> {
  using Type = double4;
};

// each matrix is divided into "tile_num" tiles by dimension "n"
// each tile compute "pre * post" elemwise_grad result
// post number continuous threads deal with a tile
// each thread loop "pre" outside assure read and write coalescing
template <typename T, typename DX_OP, typename DY_OP, int vec_size>
static __global__ void ElemwiseGradBroadcast2ThreadVecCUDAKernel(
    const T *__restrict__ x, const T *__restrict__ y, const T *__restrict__ out,
    const T *__restrict__ dout, int pre, int mid, int post,
    bool is_xsize_larger, DX_OP dx_op, DY_OP dy_op, T *__restrict__ dx,
    T *__restrict__ dy) {
  assert((mid * post) % vec_size == 0);

  const int elem_per_block = (blockDim.x * vec_size / post) * post;
  const int elem_thread_start = threadIdx.x * vec_size;
  if (elem_thread_start >= elem_per_block) return;

  const int total_elem_num = mid * post;
  const int elem_global_start = blockIdx.x * elem_per_block + elem_thread_start;
  if (elem_global_start >= total_elem_num) return;

  // shared memory
  using AccT = typename details::MPTypeTrait<T>::Type;
  __shared__ __align__(sizeof(AccT)) unsigned char
      s_mem[ELEMWISE_MAX_BLOCK_DIM * sizeof(AccT) * vec_size];
  AccT *s_data = reinterpret_cast<AccT *>(s_mem);

#pragma unroll
  for (int k = 0; k < vec_size; k++) s_data[elem_thread_start + k] = (AccT)0;

  // vectorize data read and write
  using VecT = typename GetVecType<T, vec_size>::Type;
  VecT vec_out, vec_dout;
  T *buf_out = reinterpret_cast<T *>(&vec_out);
  T *buf_dout = reinterpret_cast<T *>(&vec_dout);

  if (is_xsize_larger) {
    VecT vec_x, vec_dx;
    T *buf_x = reinterpret_cast<T *>(&vec_x);
    T *buf_dx = reinterpret_cast<T *>(&vec_dx);

    for (int i = 0; i < pre; i++) {
      // x_offset is consecutive for all thread in block
      int x_offset = i * total_elem_num + elem_global_start;

      // vectorize read x to ensure coalescing
      vec_x = reinterpret_cast<const VecT *>(&x[x_offset])[0];
      vec_out = reinterpret_cast<const VecT *>(&out[x_offset])[0];
      vec_dout = reinterpret_cast<const VecT *>(&dout[x_offset])[0];
      if (dx != nullptr) {
#pragma unroll
        for (int k = 0; k < vec_size; k++) {
          const int mid_id = (elem_global_start + k) / post;
          buf_dx[k] = dx_op(buf_x[k], y[mid_id], buf_out[k], buf_dout[k]);
        }
        // vectorize write dx
        reinterpret_cast<VecT *>(&dx[x_offset])[0] = vec_dx;
      }

      if (dy != nullptr) {
#pragma unroll
        for (int k = 0; k < vec_size; k++) {
          const int mid_id = (elem_global_start + k) / post;
          s_data[elem_thread_start + k] += static_cast<AccT>(
              dy_op(buf_x[k], y[mid_id], buf_out[k], buf_dout[k]));
        }
      }
    }
    __syncthreads();

    if (dy != nullptr) {
// calculate total "dy[nid]" value
#pragma unroll
      for (int k = 0; k < vec_size; k++) {
        const int mid_id = (elem_global_start + k) / post;
        const int mid_id_in_block = (elem_thread_start + k) / post;
        const int tile_beg = mid_id_in_block * post;
        AccT val(0);
        for (int l = 0; l < post; l++) {
          val += s_data[tile_beg + l];
        }
        dy[mid_id] = static_cast<T>(val);
      }
    }
  } else {  // x.dims < y.dims, broadcast for x.
    VecT vec_y, vec_dy;
    T *buf_y = reinterpret_cast<T *>(&vec_y);
    T *buf_dy = reinterpret_cast<T *>(&vec_dy);

    for (int i = 0; i < pre; i++) {
      // y_offset is consecutive for all thread in block
      int y_offset = i * total_elem_num + elem_global_start;

      // vectorize read x
      vec_y = reinterpret_cast<const VecT *>(&y[y_offset])[0];
      vec_out = reinterpret_cast<const VecT *>(&out[y_offset])[0];
      vec_dout = reinterpret_cast<const VecT *>(&dout[y_offset])[0];
      if (dy != nullptr) {
#pragma unroll
        for (int k = 0; k < vec_size; k++) {
          const int mid_id = (elem_global_start + k) / post;
          buf_dy[k] = dy_op(x[mid_id], buf_y[k], buf_out[k], buf_dout[k]);
        }
        // vectorize write dx
        reinterpret_cast<VecT *>(&dy[y_offset])[0] = vec_dy;
      }

      if (dx != nullptr) {
#pragma unroll
        for (int k = 0; k < vec_size; k++) {
          const int mid_id = (elem_global_start + k) / post;
          s_data[elem_thread_start + k] += static_cast<AccT>(
              dx_op(x[mid_id], buf_y[k], buf_out[k], buf_dout[k]));
        }
      }
    }
    __syncthreads();

    if (dx != nullptr) {
// calculate total "dx[nid]" value
#pragma unroll
      for (int k = 0; k < vec_size; k++) {
        const int mid_id = (elem_global_start + k) / post;
        const int mid_id_in_block = (elem_thread_start + k) / post;
        const int tile_beg = mid_id_in_block * post;
        AccT val(0);
        for (int l = 0; l < post; l++) {
          val += s_data[tile_beg + l];
        }
        dx[mid_id] = static_cast<T>(val);
      }
    }
  }
}

#define LAUNCH_ELEMWISE_GRAD_KERNEL(vec_size)                                 \
  int post_per_block = std::min(ELEMWISE_MAX_BLOCK_DIM * vec_size / post, n); \
  int elem_per_block = post_per_block * post;                                 \
  while (elem_per_block % vec_size != 0) elem_per_block -= post;              \
  post_per_block = elem_per_block / post;                                     \
  int thread_per_block = elem_per_block / vec_size;                           \
  int block_per_grid = (n + post_per_block - 1) / post_per_block;             \
  ElemwiseGradBroadcast2ThreadVecCUDAKernel<                                  \
      T, DX_OP, DY_OP,                                                        \
      vec_size><<<block_per_grid, thread_per_block, 0, stream>>>(             \
      x, y, out, dout, pre, n, post, is_xsize_larger, dx_op, dy_op, dx, dy);

template <typename T, typename DX_OP, typename DY_OP>
static typename std::enable_if<
    !std::is_same<T, paddle::platform::float16>::value &&
        !std::is_same<T, float>::value && !std::is_same<T, double>::value,
    void>::type
LaunchElemwiseGradBroadcast2ThreadVecCUDAKernel(gpuStream_t stream, const T *x,
                                                const T *y, const T *out,
                                                const T *dout, int pre, int n,
                                                int post, bool is_xsize_larger,
                                                DX_OP dx_op, DY_OP dy_op, T *dx,
                                                T *dy) {
  LAUNCH_ELEMWISE_GRAD_KERNEL(1)
}

template <typename T, typename DX_OP, typename DY_OP>
static
    typename std::enable_if<std::is_same<T, paddle::platform::float16>::value ||
                                std::is_same<T, float>::value ||
                                std::is_same<T, double>::value,
                            void>::type
    LaunchElemwiseGradBroadcast2ThreadVecCUDAKernel(
        gpuStream_t stream, const T *x, const T *y, const T *out, const T *dout,
        int pre, int n, int post, bool is_xsize_larger, DX_OP dx_op,
        DY_OP dy_op, T *dx, T *dy) {
  const int last_num = n * post;

  if (last_num % 4 == 0) {
    LAUNCH_ELEMWISE_GRAD_KERNEL(4)
  } else if (last_num % 2 == 0) {
    LAUNCH_ELEMWISE_GRAD_KERNEL(2)
  } else {
    LAUNCH_ELEMWISE_GRAD_KERNEL(1)
  }
}

#undef LAUNCH_ELEMWISE_GRAD_KERNEL

template <typename T, typename DX_OP, typename DY_OP>
static void ElemwiseGradBroadcast2CUDA(gpuStream_t stream, const T *x,
                                       const T *y, const T *out, const T *dout,
                                       int pre, int n, int post,
                                       bool is_xsize_larger, DX_OP dx_op,
                                       DY_OP dy_op, T *dx, T *dy) {
  int num = pre * post;
  if (num <= 64 && n >= ELEMWISE_MAX_BLOCK_DIM * 3) {
    LaunchElemwiseGradBroadcast2ThreadVecCUDAKernel<T, DX_OP, DY_OP>(
        stream, x, y, out, dout, pre, n, post, is_xsize_larger, dx_op, dy_op,
        dx, dy);
  } else {
    int block_size = std::min(ELEMWISE_MAX_BLOCK_DIM, pre * post);
    int gird_size = n;
    ElemwiseGradBroadcast2CUDAKernel<<<gird_size, block_size, 0, stream>>>(
        x, y, out, dout, pre, n, post, is_xsize_larger, dx_op, dy_op, dx, dy);
  }
}

#endif

template <typename DeviceContext, typename T, typename DX_OP, typename DY_OP>
void CommonElementwiseBroadcastBackward(
    const framework::ExecutionContext &ctx, const framework::DDim &x_dims,
    const framework::DDim &y_dims, const framework::Tensor &x,
    const framework::Tensor &y, const framework::Tensor &out,
    const framework::Tensor &dout, int axis, framework::Tensor *dx,
    framework::Tensor *dy, DX_OP dx_op, DY_OP dy_op) {
  int max_dim = std::max(x_dims.size(), y_dims.size());
  axis = (axis == -1 ? std::abs(x_dims.size() - y_dims.size()) : axis);
  std::vector<int> x_dims_array(max_dim);
  std::vector<int> y_dims_array(max_dim);
  std::vector<int> out_dims_array(max_dim);
  GetBroadcastDimsArrays(x_dims, y_dims, x_dims_array.data(),
                         y_dims_array.data(), out_dims_array.data(), max_dim,
                         axis);
  // for inplace strategy. memset will make dx and dout clear and get wrong
  // result.
  if (dx && dx->IsSharedBufferWith(dout)) {
    dx->clear();
    dx->mutable_data<T>(x_dims, ctx.GetPlace());
  }

  VLOG(3) << "CommonElementwiseBroadcastBackward xdims:"
          << framework::make_ddim(x_dims_array)
          << " ydim:" << framework::make_ddim(y_dims_array);

  if (platform::is_gpu_place(ctx.GetPlace())) {
#if defined(__NVCC__) || defined(__HIPCC__)
    CommonGradBroadcastCUDA<T, DX_OP, DY_OP>(
        x, y, out, dout, dx, dy, x_dims_array.data(), y_dims_array.data(),
        out_dims_array.data(), max_dim,
        ctx.template device_context<platform::CUDADeviceContext>(), dx_op,
        dy_op);
#endif
  } else {
    CommonGradBroadcastCPU<T, DX_OP, DY_OP>(
        x, y, out, dout, dx, dy, x_dims_array.data(), y_dims_array.data(),
        out_dims_array.data(), max_dim,
        ctx.template device_context<platform::CPUDeviceContext>(), dx_op,
        dy_op);
  }
}

template <typename DeviceContext, typename T, typename DX_OP, typename DY_OP>
void ElemwiseGradComputeNoBroadcast(
    const framework::ExecutionContext &ctx, const framework::DDim &x_dim,
    const framework::DDim &y_dim, const framework::Tensor &x,
    const framework::Tensor &y, const framework::Tensor &out,
    const framework::Tensor &dout, int axis, framework::Tensor *dx,
    framework::Tensor *dy, DX_OP dx_op, DY_OP dy_op) {
  size_t N = static_cast<size_t>(framework::product(x_dim));
#if !defined(_WIN32)
  platform::ForRange<DeviceContext> for_range(
      ctx.template device_context<DeviceContext>(), N);
#else
  platform::ForRange<DeviceContext> for_range(
      ctx.device_context<DeviceContext>(), N);
#endif  // !_WIN32
  for_range(ElemwiseGradNoBroadcast<T, DX_OP, DY_OP>{
      x.data<T>(), y.data<T>(), out.data<T>(), dout.data<T>(), dx_op, dy_op,
      dx == nullptr ? nullptr : dx->mutable_data<T>(ctx.GetPlace()),
      dy == nullptr ? nullptr : dy->mutable_data<T>(ctx.GetPlace())});
}

template <typename DeviceContext, typename T, typename DX_OP, typename DY_OP>
void ElemwiseGradComputeWithBroadcast(
    const framework::ExecutionContext &ctx, const framework::DDim &x_dims,
    const framework::DDim &y_dims, const framework::Tensor &x,
    const framework::Tensor &y, const framework::Tensor &out,
    const framework::Tensor &dout, int axis, framework::Tensor *dx,
    framework::Tensor *dy, DX_OP dx_op, DY_OP dy_op) {
  bool is_xsize_larger = true;

  int max_dim = x_dims.size();
  if (x_dims.size() < y_dims.size()) {
    is_xsize_larger = false;
    max_dim = y_dims.size();
  }

  axis = (axis == -1 ? std::abs(x_dims.size() - y_dims.size()) : axis);
  PADDLE_ENFORCE_GE(
      axis, 0,
      platform::errors::InvalidArgument(
          "Axis should be great than or equal to 0, but received axis is %d.",
          axis));
  PADDLE_ENFORCE_LT(axis, max_dim,
                    platform::errors::InvalidArgument(
                        "Axis should be less than %d, but received axis is %d.",
                        max_dim, axis));

  int pre, n, post, is_run_common_broadcast, axis_trim = 0;
  if (is_xsize_larger) {
    auto y_dims_trimed = trim_trailing_singular_dims(y_dims);
    axis_trim = (y_dims_trimed.size() == 0) ? x_dims.size() : axis;
    get_mid_dims(x_dims, y_dims_trimed, axis_trim, &pre, &n, &post,
                 &is_run_common_broadcast);
  } else {
    auto x_dims_trimed = trim_trailing_singular_dims(x_dims);
    axis_trim = (x_dims_trimed.size() == 0) ? y_dims.size() : axis;
    get_mid_dims(y_dims, x_dims_trimed, axis_trim, &pre, &n, &post,
                 &is_run_common_broadcast);
  }
  // special case for common backward implementation.
  if (is_run_common_broadcast) {
    CommonElementwiseBroadcastBackward<DeviceContext, T, DX_OP, DY_OP>(
        ctx, x_dims, y_dims, x, y, out, dout, axis, dx, dy, dx_op, dy_op);
    return;
  }
  if (post == 1) {
    if (platform::is_gpu_place(ctx.GetPlace())) {
#if defined(__NVCC__) || defined(__HIPCC__)
      ElemwiseGradBroadcast1CUDA(
          ctx.template device_context<DeviceContext>().stream(), x.data<T>(),
          y.data<T>(), out.data<T>(), dout.data<T>(), pre, n, is_xsize_larger,
          dx_op, dy_op,
          dx == nullptr ? nullptr : dx->mutable_data<T>(ctx.GetPlace()),
          dy == nullptr ? nullptr : dy->mutable_data<T>(ctx.GetPlace()));
#endif
    } else {
      ElemwiseGradBroadcast1CPU(
          x.data<T>(), y.data<T>(), out.data<T>(), dout.data<T>(), pre, n,
          is_xsize_larger, dx_op, dy_op,
          dx == nullptr ? nullptr : dx->mutable_data<T>(ctx.GetPlace()),
          dy == nullptr ? nullptr : dy->mutable_data<T>(ctx.GetPlace()));
    }
  } else {
    if (platform::is_gpu_place(ctx.GetPlace())) {
#if defined(__NVCC__) || defined(__HIPCC__)
      ElemwiseGradBroadcast2CUDA(
          ctx.template device_context<DeviceContext>().stream(), x.data<T>(),
          y.data<T>(), out.data<T>(), dout.data<T>(), pre, n, post,
          is_xsize_larger, dx_op, dy_op,
          dx == nullptr ? nullptr : dx->mutable_data<T>(ctx.GetPlace()),
          dy == nullptr ? nullptr : dy->mutable_data<T>(ctx.GetPlace()));
#endif
    } else {
      ElemwiseGradBroadcast2CPU(
          x.data<T>(), y.data<T>(), out.data<T>(), dout.data<T>(), pre, n, post,
          is_xsize_larger, dx_op, dy_op,
          dx == nullptr ? nullptr : dx->mutable_data<T>(ctx.GetPlace()),
          dy == nullptr ? nullptr : dy->mutable_data<T>(ctx.GetPlace()));
    }
  }
}

template <typename Functor, typename DeviceContext, typename T,
          typename OutType = T>
void CommonElementwiseBroadcastForward(
    const framework::ExecutionContext &ctx, const framework::Tensor *x,
    const framework::Tensor *y, framework::Tensor *z,
    const framework::DDim &x_dims, const framework::DDim &y_dims, Functor func,
    int axis, const bool is_xsize_larger = true) {
  int max_dim = std::max(x_dims.size(), y_dims.size());
  axis = (axis == -1 ? std::abs(x_dims.size() - y_dims.size()) : axis);
  PADDLE_ENFORCE_GE(
      axis, 0,
      platform::errors::InvalidArgument(
          "Axis should be great than or equal to 0, but received axis is %d.",
          axis));
  PADDLE_ENFORCE_LT(axis, max_dim,
                    platform::errors::InvalidArgument(
                        "Axis should be less than %d, but received axis is %d.",
                        max_dim, axis));
  std::vector<int> x_dims_array(max_dim);
  std::vector<int> y_dims_array(max_dim);
  std::vector<int> out_dims_array(max_dim);
  GetBroadcastDimsArrays(x_dims, y_dims, x_dims_array.data(),
                         y_dims_array.data(), out_dims_array.data(), max_dim,
                         axis);

  if (platform::is_gpu_place(ctx.GetPlace())) {
#if defined(__NVCC__) || defined(__HIPCC__)
    CommonForwardBroadcastCUDA<Functor, T, OutType>(
        x, y, z, x_dims_array.data(), y_dims_array.data(),
        out_dims_array.data(), max_dim,
        ctx.template device_context<platform::CUDADeviceContext>(), func,
        is_xsize_larger);
#endif
  } else {
    CommonForwardBroadcastCPU<Functor, T, OutType>(
        x, y, z, x_dims_array.data(), y_dims_array.data(),
        out_dims_array.data(), max_dim,
        ctx.template device_context<platform::CPUDeviceContext>(), func,
        is_xsize_larger);
  }
}

template <typename DeviceContext, typename T, typename DX_OP, typename DY_OP>
void ElemwiseGradCompute(const framework::ExecutionContext &ctx,
                         const framework::Tensor &x, const framework::Tensor &y,
                         const framework::Tensor &out,
                         const framework::Tensor &dout, int axis,
                         framework::Tensor *dx, framework::Tensor *dy,
                         DX_OP dx_op, DY_OP dy_op) {
  const framework::DDim &x_dim = x.dims();
  const framework::DDim &y_dim = y.dims();
  if (x.dims() == y.dims()) {
    ElemwiseGradComputeNoBroadcast<DeviceContext, T, DX_OP, DY_OP>(
        ctx, x_dim, y_dim, x, y, out, dout, axis, dx, dy, dx_op, dy_op);
  } else {
    ElemwiseGradComputeWithBroadcast<DeviceContext, T, DX_OP, DY_OP>(
        ctx, x_dim, y_dim, x, y, out, dout, axis, dx, dy, dx_op, dy_op);
  }
}

// NOTE(dzhwinter): Only used in elementwise_add, elementwise_sub.
// explicit gradient can cut off X, Y, Out from gradient op
// In elementwise_add, elementwise_sub, we use dout as fake X, Y, Out to reuse
// elementwise code.
template <typename DeviceContext, typename T, typename DX_OP, typename DY_OP>
void ElemwiseExplicitGradCompute(const framework::ExecutionContext &ctx,
                                 const framework::Tensor &x,
                                 const framework::Tensor &y,
                                 const framework::Tensor &out,
                                 const framework::Tensor &dout, int axis,
                                 framework::Tensor *dx, framework::Tensor *dy,
                                 DX_OP dx_op, DY_OP dy_op) {
  const framework::DDim &x_dim = x.dims();
  const framework::DDim &y_dim = y.dims();
  if (x.dims() == y.dims()) {
    ElemwiseGradComputeNoBroadcast<DeviceContext, T, DX_OP, DY_OP>(
        ctx, x_dim, y_dim, dout, dout, out, dout, axis, dx, dy, dx_op, dy_op);
  } else {
    ElemwiseGradComputeWithBroadcast<DeviceContext, T, DX_OP, DY_OP>(
        ctx, x_dim, y_dim, dout, dout, out, dout, axis, dx, dy, dx_op, dy_op);
  }
}

template <typename Functor, typename DeviceContext, typename T,
          typename OutType = T>
void ElementwiseComputeEx(const framework::ExecutionContext &ctx,
                          const framework::Tensor *x,
                          const framework::Tensor *y, int axis, Functor func,
                          framework::Tensor *z) {
  auto x_dims = x->dims();
  auto y_dims = y->dims();
  bool is_xsize_larger = true;
  int max_dim = x_dims.size();
  if (x_dims.size() < y_dims.size()) {
    is_xsize_larger = false;
    max_dim = y_dims.size();
  }
  TransformFunctor<Functor, T, DeviceContext, OutType> functor(
      x, y, z, ctx.template device_context<DeviceContext>(), func,
      is_xsize_larger);
  if (x_dims == y_dims) {
    functor.Run();
    return;
  }

  axis = (axis == -1 ? std::abs(x_dims.size() - y_dims.size()) : axis);
  PADDLE_ENFORCE_GE(
      axis, 0,
      platform::errors::InvalidArgument(
          "Axis should be great than or equal to 0, but received axis is %d.",
          axis));
  PADDLE_ENFORCE_LT(axis, max_dim,
                    platform::errors::InvalidArgument(
                        "Axis should be less than %d, but received axis is %d.",
                        max_dim, axis));

  int pre, n, post, is_run_common_broadcast, axis_trim = 0;
  if (is_xsize_larger) {
    auto y_dims_trimed = trim_trailing_singular_dims(y_dims);
    axis_trim = (y_dims_trimed.size() == 0) ? x_dims.size() : axis;
    get_mid_dims(x_dims, y_dims_trimed, axis_trim, &pre, &n, &post,
                 &is_run_common_broadcast);
  } else {
    auto x_dims_trimed = trim_trailing_singular_dims(x_dims);
    axis_trim = (x_dims_trimed.size() == 0) ? y_dims.size() : axis;
    get_mid_dims(y_dims, x_dims_trimed, axis_trim, &pre, &n, &post,
                 &is_run_common_broadcast);
  }
  // special case for common implementation.
  // case 1: x=[2,3,1,5], y=[2,1,4,1]
  // case 2: x=[2,3,4], y=[1,1,4]
  if (is_run_common_broadcast == 1) {
    CommonElementwiseBroadcastForward<Functor, DeviceContext, T, OutType>(
        ctx, x, y, z, x_dims, y_dims, func, axis, is_xsize_larger);
    return;
  }

  if (platform::is_gpu_place(ctx.GetPlace())) {
#if defined(__NVCC__) || defined(__HIPCC__)
    ComputeElementwiseCUDA<Functor, T, OutType>(
        x, y, z, pre, n, post,
        ctx.template device_context<platform::CUDADeviceContext>(), func,
        is_xsize_larger);
#endif
    return;
  }
  if (post == 1) {
    functor.RunRowWise(n, pre);
    return;
  } else {
    functor.RunMidWise(n, pre, post);
    return;
  }
}

// FusedElemwiseAndAct
// --- forward
template <typename T, typename CompoundFunctor, bool KeepIntermediateOut>
struct FusedElemwiseAndActNoBroadcast {
  HOSTDEVICE void operator()(size_t i) {
    T y_val = y_[i];
    T x_val = x_[i];
    if (KeepIntermediateOut) {
      T intermeidiate_out = compound_functor_.GetIntermediateOut(x_val, y_val);
      intermediate_out_[i] = intermeidiate_out;
      out_[i] =
          compound_functor_.GetOutUseIntermediateOut(x_val, intermeidiate_out);
    } else {
      out_[i] = compound_functor_.GetOut(x_val, y_val);
    }
  }

  const T *x_;
  const T *y_;
  CompoundFunctor compound_functor_;
  T *out_;
  T *intermediate_out_;
};

// FusedElemwiseAndActBroadcast1:
// In this case, X and Y can be reshaped to a matrix.
// For example shape(X) = (2, 3, 4, 5), shape(Y) = (4, 5) and axis = -1 or 2,
// X can be reshaped to (6, 20) and Y can be reshaped to (1, 20)
template <typename T, typename CompoundFunctor, bool BcastY,
          bool KeepIntermediateOut, bool SameShapeOfIntermediateOutAndOut>
static void FusedElemwiseAndActBroadcast1CPU(const T *x, const T *y,
                                             CompoundFunctor compound_functor,
                                             int h, int w, T *out,
                                             T *intermediate_out) {
  for (int i = 0; i < h; ++i) {
    for (int j = 0; j < w; ++j) {
      int offset = i * w + j;

      T y_val = BcastY ? y[j] : y[offset];
      T x_val = BcastY ? x[offset] : x[j];
      int64_t intermediate_out_offset;
      if (KeepIntermediateOut) {
        T intermeidiate_out = compound_functor.GetIntermediateOut(x_val, y_val);

        if (SameShapeOfIntermediateOutAndOut) {
          // for the case of f1(f2(x, y))
          intermediate_out_offset = offset;
        } else if (BcastY) {
          intermediate_out_offset = j;
        } else {
          intermediate_out_offset = offset;
        }

        intermediate_out[intermediate_out_offset] = intermeidiate_out;
        out[offset] =
            compound_functor.GetOutUseIntermediateOut(x_val, intermeidiate_out);
      } else {
        out[offset] = compound_functor.GetOut(x_val, y_val);
      }
    }
  }
}

// FusedElemwiseAndActBroadcast2
// In this case, X and Y can be reshaped to a matrix.
// For example shape(X) = (2, 3, 4, 5), shape(Y) = (3, 4) and axis = 1,
// X can be reshaped to (2, 12, 5) and Y can be reshaped to (1, 12, 1)
// pre = 2, n = 12, post = 5
template <typename T, typename CompoundFunctor, bool BcastY,
          bool KeepIntermediateOut, bool SameShapeOfIntermediateOutAndOut>
static void FusedElemwiseAndActBroadcast2CPU(const T *x, const T *y, int pre,
                                             int n, int post,
                                             CompoundFunctor compound_functor,
                                             T *out, T *intermediate_out) {
  for (int i = 0; i < pre; ++i) {
    for (int j = 0; j < n; ++j) {
      for (int k = 0; k < post; ++k) {
        int offset = i * n * post + j * post + k;

        T y_val = BcastY ? y[j] : y[offset];
        T x_val = BcastY ? x[offset] : x[j];
        int64_t intermediate_out_offset;

        if (KeepIntermediateOut) {
          T intermeidiate_out =
              compound_functor.GetIntermediateOut(x_val, y_val);

          if (SameShapeOfIntermediateOutAndOut) {
            // for the case of f1(f2(x, y))
            intermediate_out_offset = offset;
          } else if (BcastY) {
            intermediate_out_offset = j;
          } else {
            intermediate_out_offset = offset;
          }

          intermediate_out[intermediate_out_offset] = intermeidiate_out;
          out[offset] = compound_functor.GetOutUseIntermediateOut(
              x_val, intermeidiate_out);
        } else {
          out[offset] = compound_functor.GetOut(x_val, y_val);
        }
      }
    }
  }
}

#if defined(__NVCC__) || defined(__HIPCC__)
template <typename T, typename CompoundFunctor, bool BcastY,
          bool KeepIntermediateOut, bool SameShapeOfIntermediateOutAndOut>
static __global__ void FusedElemwiseAndActBroadcast1CUDAKernel(
    const T *x, const T *y, int h, int w, CompoundFunctor compound_functor,
    T *out, T *intermediate_out) {
  int i = blockIdx.x;
  int j = threadIdx.x;

  while (j < w) {
    int offset = i * w + j;

    T y_val = BcastY ? y[j] : y[offset];
    T x_val = BcastY ? x[offset] : x[j];
    int64_t intermediate_out_offset;

    if (KeepIntermediateOut) {
      T intermeidiate_out = compound_functor.GetIntermediateOut(x_val, y_val);

      if (SameShapeOfIntermediateOutAndOut) {
        // for the case of f1(f2(x, y))
        intermediate_out_offset = offset;
      } else if (BcastY) {
        intermediate_out_offset = j;
      } else {
        intermediate_out_offset = offset;
      }

      intermediate_out[intermediate_out_offset] = intermeidiate_out;
      out[offset] =
          compound_functor.GetOutUseIntermediateOut(x_val, intermeidiate_out);
    } else {
      out[offset] = compound_functor.GetOut(x_val, y_val);
    }

    j += ELEMWISE_MAX_BLOCK_DIM;
  }
}

template <typename T, typename CompoundFunctor, bool BcastY,
          bool KeepIntermediateOut, bool SameShapeOfIntermediateOutAndOut>
static void FusedElemwiseAndActBroadcast1CUDA(gpuStream_t stream, const T *x,
                                              const T *y,
                                              CompoundFunctor compound_functor,
                                              int h, int w, T *out,
                                              T *intermediate_out) {
  int block_size = std::min(ELEMWISE_MAX_BLOCK_DIM, w);
  int gird_size = h;
  FusedElemwiseAndActBroadcast1CUDAKernel<
      T, CompoundFunctor, BcastY, KeepIntermediateOut,
      SameShapeOfIntermediateOutAndOut><<<gird_size, block_size, 0, stream>>>(
      x, y, h, w, compound_functor, out, intermediate_out);
}

template <typename T, typename CompoundFunctor, bool BcastY,
          bool KeepIntermediateOut, bool SameShapeOfIntermediateOutAndOut>
static __global__ void FusedElemwiseAndActBroadcast2CUDAKernel(
    const T *x, const T *y, CompoundFunctor compound_functor, int pre, int n,
    int post, T *out, T *intermediate_out) {
  int tid = threadIdx.x;
  int j = blockIdx.x;

  while (true) {
    int i = tid / post;
    int k = tid % post;
    if (i >= pre) break;

    int offset = i * n * post + j * post + k;

    T y_val = BcastY ? y[j] : y[offset];
    T x_val = BcastY ? x[offset] : x[j];
    int64_t intermediate_out_offset;

    if (KeepIntermediateOut) {
      T intermeidiate_out = compound_functor.GetIntermediateOut(x_val, y_val);

      if (SameShapeOfIntermediateOutAndOut) {
        // for the case of f1(f2(x, y))
        intermediate_out_offset = offset;
      } else if (BcastY) {
        intermediate_out_offset = j;
      } else {
        intermediate_out_offset = offset;
      }

      intermediate_out[intermediate_out_offset] = intermeidiate_out;
      out[offset] =
          compound_functor.GetOutUseIntermediateOut(x_val, intermeidiate_out);
    } else {
      out[offset] = compound_functor.GetOut(x_val, y_val);
    }

    tid += ELEMWISE_MAX_BLOCK_DIM;
  }
}

template <typename T, typename CompoundFunctor, bool BcastY,
          bool KeepIntermediateOut, bool SameShapeOfIntermediateOutAndOut>
static void FusedElemwiseAndActBroadcast2CUDA(gpuStream_t stream, const T *x,
                                              const T *y, int pre, int n,
                                              int post,
                                              CompoundFunctor compound_functor,
                                              T *out, T *intermediate_out) {
  int block_size = std::min(ELEMWISE_MAX_BLOCK_DIM, pre * post);
  int gird_size = n;

  FusedElemwiseAndActBroadcast2CUDAKernel<
      T, CompoundFunctor, BcastY, KeepIntermediateOut,
      SameShapeOfIntermediateOutAndOut><<<gird_size, block_size, 0, stream>>>(
      x, y, compound_functor, pre, n, post, out, intermediate_out);
}

#endif

template <typename DeviceContext, typename T, typename CompoundFunctor,
          bool KeepIntermediateOut>
void FusedElemwiseAndActComputeNoBroadcast(
    const framework::ExecutionContext &ctx, const framework::DDim &x_dim,
    const framework::Tensor &x, const framework::Tensor &y,
    CompoundFunctor compound_functor, framework::Tensor *out,
    framework::Tensor *intermediate_out) {
  size_t N = static_cast<size_t>(framework::product(x_dim));

  platform::ForRange<DeviceContext> for_range(
      ctx.template device_context<DeviceContext>(), N);

  for_range(
      FusedElemwiseAndActNoBroadcast<T, CompoundFunctor, KeepIntermediateOut>{
          x.data<T>(), y.data<T>(), compound_functor,
          out->mutable_data<T>(ctx.GetPlace()),
          intermediate_out == nullptr
              ? nullptr
              : intermediate_out->mutable_data<T>(ctx.GetPlace())});
}

template <typename DeviceContext, typename T, typename CompoundFunctor,
          bool BcastY, bool KeepIntermediateOut,
          bool SameShapeOfIntermediateOutAndOut>
void FusedElemwiseAndActComputeWithBroadcast(
    const framework::ExecutionContext &ctx, const framework::DDim &x_dim,
    const framework::DDim &y_dim_untrimed, const framework::Tensor &x,
    const framework::Tensor &y, CompoundFunctor compound_functor, int axis,
    framework::Tensor *out, framework::Tensor *intermediate_out) {
  axis = (axis == -1 ? x_dim.size() - y_dim_untrimed.size() : axis);
  auto y_dim = trim_trailing_singular_dims(y_dim_untrimed);
  axis = (y_dim.size() == 0) ? x_dim.size() : axis;

  int pre, n, post, is_run_common_broadcast;
  get_mid_dims(x_dim, y_dim, axis, &pre, &n, &post, &is_run_common_broadcast);
  if (post == 1) {
    int h = pre;
    int w = n;
    if (platform::is_gpu_place(ctx.GetPlace())) {
#if defined(__NVCC__) || defined(__HIPCC__)
      FusedElemwiseAndActBroadcast1CUDA<T, CompoundFunctor, BcastY,
                                        KeepIntermediateOut,
                                        SameShapeOfIntermediateOutAndOut>(
          ctx.template device_context<DeviceContext>().stream(), x.data<T>(),
          y.data<T>(), compound_functor, h, w,
          out->mutable_data<T>(ctx.GetPlace()),
          intermediate_out == nullptr
              ? nullptr
              : intermediate_out->mutable_data<T>(ctx.GetPlace()));
#endif
    } else {
      FusedElemwiseAndActBroadcast1CPU<T, CompoundFunctor, BcastY,
                                       KeepIntermediateOut,
                                       SameShapeOfIntermediateOutAndOut>(
          x.data<T>(), y.data<T>(), compound_functor, h, w,
          out->mutable_data<T>(ctx.GetPlace()),
          intermediate_out == nullptr
              ? nullptr
              : intermediate_out->mutable_data<T>(ctx.GetPlace()));
    }
  } else {
    if (platform::is_gpu_place(ctx.GetPlace())) {
#if defined(__NVCC__) || defined(__HIPCC__)
      FusedElemwiseAndActBroadcast2CUDA<T, CompoundFunctor, BcastY,
                                        KeepIntermediateOut,
                                        SameShapeOfIntermediateOutAndOut>(
          ctx.template device_context<DeviceContext>().stream(), x.data<T>(),
          y.data<T>(), pre, n, post, compound_functor,
          out->mutable_data<T>(ctx.GetPlace()),
          intermediate_out == nullptr
              ? nullptr
              : intermediate_out->mutable_data<T>(ctx.GetPlace()));
#endif
    } else {
      FusedElemwiseAndActBroadcast2CPU<T, CompoundFunctor, BcastY,
                                       KeepIntermediateOut,
                                       SameShapeOfIntermediateOutAndOut>(
          x.data<T>(), y.data<T>(), pre, n, post, compound_functor,
          out->mutable_data<T>(ctx.GetPlace()),
          intermediate_out == nullptr
              ? nullptr
              : intermediate_out->mutable_data<T>(ctx.GetPlace()));
    }
  }
}

// --- backward
template <typename T, typename DX_OP, typename DY_OP, typename DIntermediate_OP,
          bool UseIntermediateOut>
struct FusedElemwiseAndActGradNoBroadcast {
  HOSTDEVICE void operator()(size_t i) {
    T zero = static_cast<T>(0);
    T x_val = (x_ == nullptr) ? zero : x_[i];
    T y_val = (y_ == nullptr) ? zero : y_[i];
    T out_val = out_[i];
    T dout_val = dout_[i];
    T intermediate_out_val = UseIntermediateOut
                                 ? intermediate_out_[i]
                                 : dx_op_.GetIntermediateOut(x_val, y_val);
    if (dx_ != nullptr) {
      dx_[i] = dx_op_.UseIntermediateOut(x_val, y_val, intermediate_out_val,
                                         out_val, dout_val);
    }
    if (dy_ != nullptr) {
      dy_[i] = dy_op_.UseIntermediateOut(x_val, y_val, intermediate_out_val,
                                         out_val, dout_val);
    }
    if (dintermediate_ != nullptr) {
      dintermediate_[i] = dintermediate_op_.UseIntermediateOut(
          x_val, intermediate_out_val, out_val, dout_val);
    }
  }

  const T *x_;
  const T *y_;
  const T *intermediate_out_;
  const T *out_;
  const T *dout_;
  DX_OP dx_op_;
  DY_OP dy_op_;
  DIntermediate_OP dintermediate_op_;
  T *dx_;
  T *dy_;
  T *dintermediate_;
};

template <typename DeviceContext, typename T, typename DX_OP, typename DY_OP,
          typename DIntermediate_OP, bool UseIntermediateOut>
void FusedElemwiseAndActGradComputeNoBroadcast(
    const framework::ExecutionContext &ctx, const framework::DDim &x_dim,
    const framework::DDim &y_dim, const framework::Tensor *x,
    const framework::Tensor *y, const framework::Tensor *intermediate_out,
    const framework::Tensor *out, const framework::Tensor *dout, int axis,
    framework::Tensor *dx, framework::Tensor *dy,
    framework::Tensor *dintermediate, DX_OP dx_op, DY_OP dy_op,
    DIntermediate_OP dintermediate_op) {
  size_t N = static_cast<size_t>(framework::product(x_dim));
  platform::ForRange<DeviceContext> for_range(
      ctx.template device_context<DeviceContext>(), N);
  const T *x_data = nullptr;
  const T *y_data = nullptr;
  if (x->IsInitialized()) x_data = x->data<T>();
  if (y->IsInitialized()) y_data = y->data<T>();

  for_range(FusedElemwiseAndActGradNoBroadcast<
            T, DX_OP, DY_OP, DIntermediate_OP, UseIntermediateOut>{
      x_data, y_data, intermediate_out ? intermediate_out->data<T>() : nullptr,
      out->data<T>(), dout->data<T>(), dx_op, dy_op, dintermediate_op,
      dx == nullptr ? nullptr : dx->mutable_data<T>(ctx.GetPlace()),
      dy == nullptr ? nullptr : dy->mutable_data<T>(ctx.GetPlace()),
      dintermediate == nullptr ? nullptr : dintermediate->mutable_data<T>(
                                               ctx.GetPlace())});
}

template <typename T, typename DX_OP, typename DY_OP, typename DIntermediate_OP,
          bool UseIntermediateOut, bool BcastY,
          bool SameShapeOfIntermediateOutAndOut>
static void FusedElemwiseAndActGradBroadcast1CPU(
    const T *x, const T *y, const T *intermediate_out, const T *out,
    const T *dout, int h, int w, DX_OP dx_op, DY_OP dy_op,
    DIntermediate_OP dintermediate_op, T *dx, T *dy, T *d_intermediate) {
  int64_t tmp_out_idx, x_idx, y_idx;
  T zero = static_cast<T>(0);
  for (int i = 0; i < h; ++i) {
    for (int j = 0; j < w; ++j) {
      int offset = i * w + j;

      tmp_out_idx = BcastY ? j : offset;
      y_idx = BcastY ? j : offset;
      x_idx = BcastY ? offset : j;
      T x_val = (x == nullptr) ? zero : x[x_idx];
      T y_val = (y == nullptr) ? zero : y[y_idx];

      if (SameShapeOfIntermediateOutAndOut) {
        tmp_out_idx = offset;
      }

      if (dx != nullptr) {
        T tmp = UseIntermediateOut
                    ? dx_op.UseIntermediateOut(x_val, y_val,
                                               intermediate_out[tmp_out_idx],
                                               out[offset], dout[offset])
                    : dx_op.Recompute(x_val, y_val, out[offset], dout[offset]);

        if (BcastY) {
          dx[x_idx] = tmp;
        } else {
          if (i == 0) {
            dx[x_idx] = tmp;
          } else {
            dx[x_idx] += tmp;
          }
        }
      }
      if (dy != nullptr) {
        T tmp = UseIntermediateOut
                    ? dy_op.UseIntermediateOut(x_val, y_val,
                                               intermediate_out[tmp_out_idx],
                                               out[offset], dout[offset])
                    : dy_op.Recompute(x_val, y_val, out[offset], dout[offset]);
        if (BcastY) {
          if (i == 0) {
            dy[y_idx] = tmp;
          } else {
            dy[y_idx] += tmp;
          }
        } else {
          dy[y_idx] = tmp;
        }
      }
      if (d_intermediate != nullptr) {
        T tmp = UseIntermediateOut
                    ? dintermediate_op.UseIntermediateOut(
                          x_val, intermediate_out[tmp_out_idx], out[offset],
                          dout[offset])
                    : dintermediate_op.Recompute(x_val, y_val, out[offset],
                                                 dout[i]);
        if (SameShapeOfIntermediateOutAndOut) {
          d_intermediate[tmp_out_idx] = tmp;
        } else {
          if (i == 0) {
            d_intermediate[tmp_out_idx] = tmp;
          } else {
            d_intermediate[tmp_out_idx] += tmp;
          }
        }
      }
    }
  }
}

template <typename T, typename DX_OP, typename DY_OP, typename DIntermediate_OP,
          bool UseIntermediateOut, bool BcastY,
          bool SameShapeOfIntermediateOutAndOut>
static void FusedElemwiseAndActGradBroadcast2CPU(
    const T *x, const T *y, const T *intermediate_out, const T *out,
    const T *dout, int pre, int n, int post, DX_OP dx_op, DY_OP dy_op,
    DIntermediate_OP dintermediate_op, T *dx, T *dy, T *d_intermediate) {
  int64_t tmp_out_idx, x_idx, y_idx;
  T zero = static_cast<T>(0);
  for (int i = 0; i < pre; ++i) {
    for (int j = 0; j < n; ++j) {
      for (int k = 0; k < post; ++k) {
        int offset = i * n * post + j * post + k;

        tmp_out_idx = BcastY ? j : offset;
        y_idx = BcastY ? j : offset;
        x_idx = BcastY ? offset : j;

        T x_val = (x == nullptr) ? zero : x[x_idx];
        T y_val = (y == nullptr) ? zero : y[y_idx];

        if (SameShapeOfIntermediateOutAndOut) {
          tmp_out_idx = offset;
        }

        if (dx != nullptr) {
          T tmp =
              UseIntermediateOut
                  ? dx_op.UseIntermediateOut(x_val, y_val,
                                             intermediate_out[tmp_out_idx],
                                             out[offset], dout[offset])
                  : dx_op.Recompute(x_val, y_val, out[offset], dout[offset]);

          if (BcastY) {
            dx[x_idx] = tmp;
          } else {
            if (i == 0 && k == 0) {
              dx[x_idx] = tmp;
            } else {
              dx[x_idx] += tmp;
            }
          }
        }
        if (dy != nullptr) {
          T tmp =
              UseIntermediateOut
                  ? dy_op.UseIntermediateOut(x_val, y_val,
                                             intermediate_out[tmp_out_idx],
                                             out[offset], dout[offset])
                  : dy_op.Recompute(x_val, y_val, out[offset], dout[offset]);
          if (BcastY) {
            if (i == 0 && k == 0) {
              dy[y_idx] = tmp;
            } else {
              dy[y_idx] += tmp;
            }
          } else {
            dy[y_idx] = tmp;
          }
        }
        if (d_intermediate != nullptr) {
          T tmp = UseIntermediateOut
                      ? dintermediate_op.UseIntermediateOut(
                            x_val, intermediate_out[tmp_out_idx], out[offset],
                            dout[offset])
                      : dintermediate_op.Recompute(x_val, y_val, out[offset],
                                                   dout[i]);
          if (SameShapeOfIntermediateOutAndOut) {
            d_intermediate[tmp_out_idx] = tmp;
          } else {
            if (i == 0) {
              d_intermediate[tmp_out_idx] = tmp;
            } else {
              d_intermediate[tmp_out_idx] += tmp;
            }
          }
        }
      }
    }
  }
}

#if defined(__NVCC__) || defined(__HIPCC__)
template <typename T, typename DX_OP, typename DY_OP, typename DIntermediate_OP,
          bool UseIntermediateOut, bool BcastY,
          bool SameShapeOfIntermediateOutAndOut>
static __global__ void FusedElemwiseAndActGradBroadcast1CUDAKernel(
    const T *x, const T *y, const T *intermediate_out, const T *out,
    const T *dout, int h, int w, DX_OP dx_op, DY_OP dy_op,
    DIntermediate_OP dintermediate_op, T *dx, T *dy, T *d_intermediate) {
  __shared__ T sdata[BLOCK_Y][BLOCK_X];
  size_t idx = threadIdx.x + BLOCK_X * blockIdx.x;
  size_t width_stride = gridDim.x * BLOCK_X;

  size_t full_w = ROUNDUP(w, BLOCK_X);

  T zero = static_cast<T>(0);

  for (size_t j = idx; j < full_w; j += width_stride) {
    T val(0), inter_val(0);
    if (j < w) {
      for (size_t i = threadIdx.y; i < h; i += BLOCK_Y) {
        size_t offset = i * w + j;

        size_t tmp_out_idx = BcastY ? j : offset;
        size_t y_idx = BcastY ? j : offset;
        size_t x_idx = BcastY ? offset : j;
        T x_val = (x == nullptr) ? zero : x[x_idx];
        T y_val = (y == nullptr) ? zero : y[y_idx];

        if (SameShapeOfIntermediateOutAndOut) {
          tmp_out_idx = offset;
        }

        if (dx != nullptr) {
          T tmp =
              UseIntermediateOut
                  ? dx_op.UseIntermediateOut(x_val, y_val,
                                             intermediate_out[tmp_out_idx],
                                             out[offset], dout[offset])
                  : dx_op.Recompute(x_val, y_val, out[offset], dout[offset]);

          if (BcastY) {
            dx[x_idx] = tmp;
          } else {
            val += tmp;
          }
        }
        if (dy != nullptr) {
          T tmp =
              UseIntermediateOut
                  ? dy_op.UseIntermediateOut(x_val, y_val,
                                             intermediate_out[tmp_out_idx],
                                             out[offset], dout[offset])
                  : dy_op.Recompute(x_val, y_val, out[offset], dout[offset]);
          if (BcastY) {
            val += tmp;
          } else {
            dy[y_idx] = tmp;
          }
        }
        if (d_intermediate != nullptr) {
          T tmp = UseIntermediateOut
                      ? dintermediate_op.UseIntermediateOut(
                            y[y_idx], intermediate_out[tmp_out_idx],
                            out[offset], dout[offset])
                      : dintermediate_op.Recompute(x_val, y_val, out[offset],
                                                   dout[offset]);
          if (SameShapeOfIntermediateOutAndOut) {
            d_intermediate[tmp_out_idx] = tmp;
          } else {
            inter_val += tmp;
          }
        }
      }
    }

    // transpose, for ReduceSum with wrap
    sdata[threadIdx.y][threadIdx.x] = val;
    __syncthreads();
    val = sdata[threadIdx.x][threadIdx.y];
#pragma unroll
    for (int i = BLOCK_X >> 1; i > 0; i >>= 1) {
      // reduce sum with wrap
      val += platform::CudaShuffleXorSync(0xFFFFFFFF, val, i);
    }

    size_t idx_j = j + threadIdx.y;
    if (BcastY) {
      if (dy) {
        if (threadIdx.x == 0 && (idx_j < w)) dy[idx_j] = val;
      }
    } else {
      if (dx) {
        if (threadIdx.x == 0 && (idx_j < w)) dx[idx_j] = val;
      }
    }

    if (!SameShapeOfIntermediateOutAndOut) {
      if (d_intermediate) {
        sdata[threadIdx.y][threadIdx.x] = inter_val;
        __syncthreads();
        inter_val = sdata[threadIdx.x][threadIdx.y];
#pragma unroll
        for (int i = BLOCK_X >> 1; i > 0; i >>= 1) {
          // reduce sum with wrap
          inter_val += platform::CudaShuffleXorSync(0xFFFFFFFF, inter_val, i);
        }
        if (threadIdx.x == 0 && (idx_j < w)) d_intermediate[idx_j] = inter_val;
      }
    }
  }  // end for
}

template <typename T, typename DX_OP, typename DY_OP, typename DIntermediate_OP,
          bool UseIntermediateOut, bool BcastY,
          bool SameShapeOfIntermediateOutAndOut>
static void FusedElemwiseAndActGradBroadcast1CUDA(
    const framework::ExecutionContext &ctx, const T *x, const T *y,
    const T *intermediate_out, const T *out, const T *dout, int h, int w,
    DX_OP dx_op, DY_OP dy_op, DIntermediate_OP dintermediate_op, T *dx, T *dy,
    T *d_intermediate) {
  gpuStream_t stream = ctx.cuda_device_context().stream();

  dim3 blocks(BLOCK_X, BLOCK_Y);
  int max_gpu_threads = ctx.cuda_device_context().GetMaxPhysicalThreadCount();
  int max_blocks = std::max(max_gpu_threads / (BLOCK_X * BLOCK_Y), 1);
  int theory_block = (w + BLOCK_X - 1) / BLOCK_X;
  dim3 grids(std::min(theory_block, max_blocks));

  FusedElemwiseAndActGradBroadcast1CUDAKernel<
      T, DX_OP, DY_OP, DIntermediate_OP, UseIntermediateOut, BcastY,
      SameShapeOfIntermediateOutAndOut><<<grids, blocks, 0, stream>>>(
      x, y, intermediate_out, out, dout, h, w, dx_op, dy_op, dintermediate_op,
      dx, dy, d_intermediate);
}

template <typename T, typename DX_OP, typename DY_OP, typename DIntermediate_OP,
          bool UseIntermediateOut, bool BcastY,
          bool SameShapeOfIntermediateOutAndOut>
static __global__ void FusedElemwiseAndActGradBroadcast2CUDAKernel(
    const T *x, const T *y, const T *intermediate_out, const T *out,
    const T *dout, int pre, int n, int post, DX_OP dx_op, DY_OP dy_op,
    DIntermediate_OP dintermediate_op, T *dx, T *dy, T *d_intermediate) {
  int tid = threadIdx.x;
  int j = blockIdx.x;

  T val(0), inter_val(0);
  int ttid = tid;
  int64_t tmp_out_idx, x_idx, y_idx;
  T zero = static_cast<T>(0);
  while (true) {
    int i = ttid / post;
    int k = ttid % post;
    if (i >= pre) break;

    int offset = i * n * post + j * post + k;

    tmp_out_idx = BcastY ? j : offset;
    y_idx = BcastY ? j : offset;
    x_idx = BcastY ? offset : j;
    T x_val = (x == nullptr) ? zero : x[x_idx];
    T y_val = (y == nullptr) ? zero : y[y_idx];

    if (SameShapeOfIntermediateOutAndOut) {
      tmp_out_idx = offset;
    }

    if (dx != nullptr) {
      T tmp = UseIntermediateOut
                  ? dx_op.UseIntermediateOut(x_val, y_val,
                                             intermediate_out[tmp_out_idx],
                                             out[offset], dout[offset])
                  : dx_op.Recompute(x_val, y_val, out[offset], dout[offset]);

      if (BcastY) {
        dx[x_idx] = tmp;
      } else {
        val += tmp;
      }
    }
    if (dy != nullptr) {
      T tmp = UseIntermediateOut
                  ? dy_op.UseIntermediateOut(x_val, y_val,
                                             intermediate_out[tmp_out_idx],
                                             out[offset], dout[offset])
                  : dy_op.Recompute(x_val, y_val, out[offset], dout[offset]);
      if (BcastY) {
        val += tmp;
      } else {
        dy[y_idx] = tmp;
      }
    }
    if (d_intermediate != nullptr) {
      T tmp = UseIntermediateOut
                  ? dintermediate_op.UseIntermediateOut(
                        y_val, intermediate_out[tmp_out_idx], out[offset],
                        dout[offset])
                  : dintermediate_op.Recompute(x_val, y_val, out[offset],
                                               dout[offset]);
      if (SameShapeOfIntermediateOutAndOut) {
        d_intermediate[tmp_out_idx] = tmp;
      } else {
        inter_val += tmp;
      }
    }
    ttid += ELEMWISE_MAX_BLOCK_DIM;
  }

  int h = pre * post;
  h = h > ELEMWISE_MAX_BLOCK_DIM ? ELEMWISE_MAX_BLOCK_DIM : h;
  if (BcastY) {
    if (dy) {
      val = paddle::platform::reduceSum(val, tid, h);
      if (threadIdx.x == 0) {
        dy[j] = val;
      }
    }
  } else {
    if (dx) {
      val = paddle::platform::reduceSum(val, tid, h);
      if (threadIdx.x == 0) {
        dx[j] = val;
      }
    }
  }
  if (!SameShapeOfIntermediateOutAndOut) {
    if (d_intermediate) {
      inter_val = paddle::platform::reduceSum(inter_val, tid, h);
      if (threadIdx.x == 0) {
        d_intermediate[j] = inter_val;
      }
    }
  }
}

template <typename T, typename DX_OP, typename DY_OP, typename DIntermediate_OP,
          bool UseIntermediateOut, bool BcastY,
          bool SameShapeOfIntermediateOutAndOut>
static void FusedElemwiseAndActGradBroadcast2CUDA(
    gpuStream_t stream, const T *x, const T *y, const T *intermediate_out,
    const T *out, const T *dout, int pre, int n, int post, DX_OP dx_op,
    DY_OP dy_op, DIntermediate_OP dintermediate_op, T *dx, T *dy,
    T *dintermediate) {
  int block_size = std::min(ELEMWISE_MAX_BLOCK_DIM, pre * post);
  int gird_size = n;
  FusedElemwiseAndActGradBroadcast2CUDAKernel<
      T, DX_OP, DY_OP, DIntermediate_OP, UseIntermediateOut, BcastY,
      SameShapeOfIntermediateOutAndOut><<<gird_size, block_size, 0, stream>>>(
      x, y, intermediate_out, out, dout, pre, n, post, dx_op, dy_op,
      dintermediate_op, dx, dy, dintermediate);
}
#endif

template <typename DeviceContext, typename T, typename DX_OP, typename DY_OP,
          typename DIntermediate_OP, bool UseIntermediateOut, bool BcastY,
          bool SameShapeOfIntermediateOutAndOut>
void FusedElemwiseAndActGradComputeWithBroadcast(
    const framework::ExecutionContext &ctx, const framework::DDim &x_dim,
    const framework::DDim &y_dim_untrimed, const framework::Tensor *x,
    const framework::Tensor *y, const framework::Tensor *intermediate_out,
    const framework::Tensor *out, const framework::Tensor *dout, int axis,
    framework::Tensor *dx, framework::Tensor *dy,
    framework::Tensor *dintermediate, DX_OP dx_op, DY_OP dy_op,
    DIntermediate_OP dintermediate_op) {
  axis = (axis == -1 ? x_dim.size() - y_dim_untrimed.size() : axis);
  auto y_dim = trim_trailing_singular_dims(y_dim_untrimed);
  axis = (y_dim.size() == 0) ? x_dim.size() : axis;

  int pre, n, post, is_run_common_broadcast;
  get_mid_dims(x_dim, y_dim, axis, &pre, &n, &post, &is_run_common_broadcast);
  const T *x_data = nullptr;
  const T *y_data = nullptr;
  if (x->IsInitialized()) x_data = x->data<T>();
  if (y->IsInitialized()) y_data = y->data<T>();
  if (post == 1) {
    int h = pre;
    int w = n;

    if (platform::is_gpu_place(ctx.GetPlace())) {
#if defined(__NVCC__) || defined(__HIPCC__)
      FusedElemwiseAndActGradBroadcast1CUDA<T, DX_OP, DY_OP, DIntermediate_OP,
                                            UseIntermediateOut, BcastY,
                                            SameShapeOfIntermediateOutAndOut>(
          ctx, x_data, y_data,
          intermediate_out == nullptr ? nullptr : intermediate_out->data<T>(),
          out->data<T>(), dout->data<T>(), h, w, dx_op, dy_op, dintermediate_op,
          dx == nullptr ? nullptr : dx->mutable_data<T>(ctx.GetPlace()),
          dy == nullptr ? nullptr : dy->mutable_data<T>(ctx.GetPlace()),
          dintermediate == nullptr ? nullptr : dintermediate->mutable_data<T>(
                                                   ctx.GetPlace()));
#endif
    } else {
      FusedElemwiseAndActGradBroadcast1CPU<T, DX_OP, DY_OP, DIntermediate_OP,
                                           UseIntermediateOut, BcastY,
                                           SameShapeOfIntermediateOutAndOut>(
          x_data, y_data,
          intermediate_out == nullptr ? nullptr : intermediate_out->data<T>(),
          out->data<T>(), dout->data<T>(), h, w, dx_op, dy_op, dintermediate_op,
          dx == nullptr ? nullptr : dx->mutable_data<T>(ctx.GetPlace()),
          dy == nullptr ? nullptr : dy->mutable_data<T>(ctx.GetPlace()),
          dintermediate == nullptr ? nullptr : dintermediate->mutable_data<T>(
                                                   ctx.GetPlace()));
    }
  } else {
    if (platform::is_gpu_place(ctx.GetPlace())) {
#if defined(__NVCC__) || defined(__HIPCC__)
      FusedElemwiseAndActGradBroadcast2CUDA<T, DX_OP, DY_OP, DIntermediate_OP,
                                            UseIntermediateOut, BcastY,
                                            SameShapeOfIntermediateOutAndOut>(
          ctx.template device_context<DeviceContext>().stream(), x_data, y_data,
          intermediate_out == nullptr ? nullptr : intermediate_out->data<T>(),
          out->data<T>(), dout->data<T>(), pre, n, post, dx_op, dy_op,
          dintermediate_op,
          dx == nullptr ? nullptr : dx->mutable_data<T>(ctx.GetPlace()),
          dy == nullptr ? nullptr : dy->mutable_data<T>(ctx.GetPlace()),
          dintermediate == nullptr ? nullptr : dintermediate->mutable_data<T>(
                                                   ctx.GetPlace()));
#endif
    } else {
      FusedElemwiseAndActGradBroadcast2CPU<T, DX_OP, DY_OP, DIntermediate_OP,
                                           UseIntermediateOut, BcastY,
                                           SameShapeOfIntermediateOutAndOut>(
          x_data, y_data,
          intermediate_out == nullptr ? nullptr : intermediate_out->data<T>(),
          out->data<T>(), dout->data<T>(), pre, n, post, dx_op, dy_op,
          dintermediate_op,
          dx == nullptr ? nullptr : dx->mutable_data<T>(ctx.GetPlace()),
          dy == nullptr ? nullptr : dy->mutable_data<T>(ctx.GetPlace()),
          dintermediate == nullptr ? nullptr : dintermediate->mutable_data<T>(
                                                   ctx.GetPlace()));
    }
  }
}

template <typename DeviceContext, typename T, typename DX_OP, typename DY_OP,
          typename DIntermediate_OP, bool UseIntermediateOut,
          bool SameShapeOfIntermediateOutAndOut>
void FusedElemwiseAndActGradComputeEx(
    const framework::ExecutionContext &ctx, const framework::Tensor *x,
    const framework::Tensor *y, const framework::Tensor *out,
    const framework::Tensor *intermediate_out, const framework::Tensor *dout,
    int axis, framework::Tensor *dx, framework::Tensor *dy,
    framework::Tensor *dintermediate, DX_OP dx_op, DY_OP dy_op,
    DIntermediate_OP dintermediate_op) {
  const framework::DDim &x_dim = x->dims();
  const framework::DDim &y_dim = y->dims();
  if (UseIntermediateOut) {
    PADDLE_ENFORCE_NOT_NULL(
        intermediate_out,
        platform::errors::InvalidArgument("Intermediate out is null pointer."));
  }
  if (x_dim == y_dim) {
    FusedElemwiseAndActGradComputeNoBroadcast<
        DeviceContext, T, DX_OP, DY_OP, DIntermediate_OP, UseIntermediateOut>(
        ctx, x_dim, y_dim, x, y, intermediate_out, out, dout, axis, dx, dy,
        dintermediate, dx_op, dy_op, dintermediate_op);
  } else {  // Y is a scalar
    bool bcast_y = x_dim.size() >= y_dim.size();
    if (x_dim.size() == y_dim.size()) {
      for (int i = 0; i < x_dim.size(); ++i) {
        if (x_dim[i] < y_dim[i]) {
          bcast_y = false;
          break;
        }
      }
    }

    // z = f1(x, f2(y))
    // z = f1(f2(x, y))
    if (bcast_y) {  // Y should be broadcast.
      FusedElemwiseAndActGradComputeWithBroadcast<
          DeviceContext, T, DX_OP, DY_OP, DIntermediate_OP, UseIntermediateOut,
          true /*BcastY*/, SameShapeOfIntermediateOutAndOut>(
          ctx, x_dim, y_dim, x, y, intermediate_out, out, dout, axis, dx, dy,
          dintermediate, dx_op, dy_op, dintermediate_op);
    } else {
      FusedElemwiseAndActGradComputeWithBroadcast<
          DeviceContext, T, DX_OP, DY_OP, DIntermediate_OP, UseIntermediateOut,
          false /*BcastY*/, SameShapeOfIntermediateOutAndOut>(
          ctx, y_dim, x_dim, x, y, intermediate_out, out, dout, axis, dx, dy,
          dintermediate, dx_op, dy_op, dintermediate_op);
    }
  }
}

template <typename DeviceContext, typename T, typename CompoundFunctor,
          bool KeepIntermediateOut, bool SameShapeOfIntermediateOutAndOut>
void FusedElemwiseAndActComputeEx(const framework::ExecutionContext &ctx,
                                  const framework::Tensor &x,
                                  const framework::Tensor &y, int axis,
                                  CompoundFunctor compound_functor,
                                  framework::Tensor *out,
                                  framework::Tensor *intermediate_out) {
  if (KeepIntermediateOut) {
    PADDLE_ENFORCE_NOT_NULL(
        intermediate_out,
        platform::errors::InvalidArgument(
            "The save_intermediate_out is opened, intermediate "
            "out is null pointer."));
  }

  const framework::DDim &x_dim = x.dims();
  const framework::DDim &y_dim = y.dims();
  if (x.dims() == y.dims()) {
    FusedElemwiseAndActComputeNoBroadcast<DeviceContext, T, CompoundFunctor,
                                          KeepIntermediateOut>(
        ctx, x_dim, x, y, compound_functor, out, intermediate_out);
  } else {
    // Whether the shape of Y is a continuous subsequence of X,
    // For more information please refer to the op's introduction.
    bool bcast_y = x.numel() >= y.numel();
    // z = f1(x, f2(y))
    // z = f1(f2(x, y))
    if (bcast_y) {  // Y should be broadcast.
      // In this case,
      // for 'f2(y)', the shape of intermediate_out should be equal to the
      // shape
      // of Y.
      // for 'f2(x, y)', the shape of intermediate_out should be equal to the
      // shape of Out.
      // the shape of Out should be equal to the shape of X.
      FusedElemwiseAndActComputeWithBroadcast<
          DeviceContext, T, CompoundFunctor, true /*BcastY*/,
          KeepIntermediateOut, SameShapeOfIntermediateOutAndOut>(
          ctx, x_dim /*OutShape*/, y_dim, x, y, compound_functor, axis, out,
          intermediate_out);
    } else {
      // In this case,
      // for 'f2(y)', the shape of intermediate_out should be equal to the
      // shape
      // of Out.
      // for 'f2(x, y)', the shape of intermediate_out should be equal to the
      // shape of Out.
      // the shape of Out should be equal to the shape of Y.
      FusedElemwiseAndActComputeWithBroadcast<
          DeviceContext, T, CompoundFunctor, false /*BcastY*/,
          KeepIntermediateOut, SameShapeOfIntermediateOutAndOut>(
          ctx, y_dim /*OutShape*/, x_dim, x, y, compound_functor, axis, out,
          intermediate_out);
    }
  }
}

template <typename DeviceContext, typename T>
static inline void GetDoubleGradSafeTensor(
    const framework::ExecutionContext &ctx, const framework::Tensor *x,
    const framework::Tensor *ddx, framework::Tensor *ddx_safe) {
  if (ddx) {
    *ddx_safe = *ddx;
  } else {
    auto &dev_ctx = ctx.template device_context<DeviceContext>();
    *ddx_safe = ctx.AllocateTmpTensor<T, DeviceContext>(x->dims(), dev_ctx);
    math::SetConstant<DeviceContext, T> set_zero;
    set_zero(ctx.template device_context<DeviceContext>(), ddx_safe,
             static_cast<T>(0));
  }
}

}  // namespace operators
}  // namespace paddle
