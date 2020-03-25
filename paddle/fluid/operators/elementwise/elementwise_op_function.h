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
#include "paddle/fluid/operators/elementwise/elementwise_op_function.cu.h"
#include "paddle/fluid/platform/gpu_info.h"
#include "paddle/fluid/platform/transform.h"

#ifdef __NVCC__
#include <cuda.h>
#include <thrust/iterator/iterator_adaptor.h>
#include "paddle/fluid/platform/cuda_device_function.h"
#include "paddle/fluid/platform/cuda_primitives.h"
constexpr int ELEMWISE_MAX_BLOCK_DIM = 1024;
#endif

#include "paddle/fluid/operators/math/math_function.h"
#include "paddle/fluid/platform/for_range.h"
#define GetDivMod(dividend, divisor, div, mod) \
  do {                                         \
    const auto dividend_copy = dividend;       \
    *div = dividend_copy / divisor;            \
    *mod = dividend_copy % divisor;            \
  } while (0)

namespace paddle {
namespace operators {

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
      PADDLE_ENFORCE(y_dims[i] == 1 || x_dims[i + axis] == 1,
                     "ShapeError: broadcast dimension mismatch. Operands "
                     "could not be broadcast together with the shape of "
                     "X = [%s] and the shape of Y = [%s]. Received [%d] "
                     "in X is not equal to [%d] in Y",
                     x_dims, y_dims, x_dims[i + axis], y_dims[i]);
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
  PADDLE_ENFORCE_GE(axis, 0, "Axis should be in range [0, %d)", axis);
  PADDLE_ENFORCE_LT(axis, max_dim, "Axis should be in range [0, %d)", axis);
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
    PADDLE_ENFORCE(x_dims_array[i] == y_dims_array[i] || x_dims_array[i] <= 1 ||
                       y_dims_array[i] <= 1,
                   "ShapeError: broadcast dimension mismatch. Operands could "
                   "not be broadcast together with the shape of X = [%s] and "
                   "the shape of Y = [%s]. Received [%d] in X is not equal to "
                   "[%d] in Y",
                   x_dims, y_dims, x_dims_array[i], y_dims_array[i]);
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

#ifdef __NVCC__
template <typename Functor, typename T>
__global__ void CommonForwardBroadcastCUDAKernel(
    const int *x_strides_array, const int *y_strides_array,
    const int *out_dims_array, const T *x, const T *y, T *out, int out_size,
    int max_dim, Functor func, const bool is_xsize_larger) {
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

template <typename Functor, typename T>
void CommonForwardBroadcastCUDA(
    const framework::Tensor *x, const framework::Tensor *y,
    framework::Tensor *z, int *x_dims_array, int *y_dims_array,
    int *out_dims_array, int max_dim, const platform::CUDADeviceContext &ctx,
    Functor func, const bool is_xsize_larger = true) {
  const auto gplace = boost::get<platform::CUDAPlace>(ctx.GetPlace());
  auto cplace = platform::CPUPlace();
  const T *x_data = x->data<T>();
  const T *y_data = y->data<T>();
  T *out_data = z->mutable_data<T>(ctx.GetPlace());

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
      Functor, T><<<gird_size, block_size, 0, ctx.stream()>>>(
      x_strides_array_gpu, y_strides_array_gpu, out_dims_array_gpu, x_data,
      y_data, out_data, out_size, max_dim, func, is_xsize_larger);
}

#endif  // __NVCC__

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

#ifdef __NVCC__
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

template <typename T, typename DX_OP, typename DY_OP>
void CommonGradBroadcastCUDA(
    const framework::Tensor &x, const framework::Tensor &y,
    const framework::Tensor &out, const framework::Tensor &dout,
    framework::Tensor *dx, framework::Tensor *dy, int *x_dims_array,
    int *y_dims_array, int *out_dims_array, int max_dim,
    const platform::CUDADeviceContext &ctx, DX_OP dx_op, DY_OP dy_op) {
  const auto gplace = boost::get<platform::CUDAPlace>(ctx.GetPlace());
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

  int x_blocks = 0;
  int x_threads = 0;
  ComputeBroadcastKernelSize(x_dims_array, out_dims_array, &x_blocks,
                             &x_threads, max_dim);
  int y_blocks = 0;
  int y_threads = 0;
  ComputeBroadcastKernelSize(y_dims_array, out_dims_array, &y_blocks,
                             &y_threads, max_dim);

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

#endif  // __NVCC__

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

#ifdef __NVCC__
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
class TransformFunctor;

template <typename Functor, typename T, typename OutType>
class TransformFunctor<Functor, T, platform::CPUDeviceContext, OutType> {
 public:
  TransformFunctor(const framework::Tensor *x, const framework::Tensor *y,
                   framework::Tensor *z, const platform::CPUDeviceContext &ctx,
                   Functor func, const bool is_xsize_larger = true)
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
    platform::Transform<platform::CPUDeviceContext> trans;
    trans(ctx_, x_, x_ + nx_, y_, z_, func_);
  }

  inline void RunRowWise(int n, int pre) const {
    platform::Transform<platform::CPUDeviceContext> trans;
    if (is_xsize_larger_) {
      trans(ctx_, x_, x_ + nx_,
            RowwiseTransformIterator<T, platform::CPUDeviceContext>(y_, n), z_,
            func_);
    } else {
      trans(ctx_, y_, y_ + nx_,
            RowwiseTransformIterator<T, platform::CPUDeviceContext>(x_, n), z_,
            func_);
    }
  }

  inline void RunMidWise(int n, int pre, int post) const {
    platform::Transform<platform::CPUDeviceContext> trans;
    if (is_xsize_larger_) {
      trans(
          ctx_, x_, x_ + nx_,
          MidWiseTransformIterator<T, platform::CPUDeviceContext>(y_, n, post),
          z_, func_);
    } else {
      trans(
          ctx_, y_, y_ + nx_,
          MidWiseTransformIterator<T, platform::CPUDeviceContext>(x_, n, post),
          z_, func_);
    }
  }

 private:
  const T *x_;
  const T *y_;
  OutType *z_;
  int64_t nx_;
  const platform::CPUDeviceContext &ctx_;
  Functor func_;
  bool is_xsize_larger_;
};

#ifdef __NVCC__

template <typename Functor, typename T, typename OutType>
__global__ void NormalElementwiseKernel(const size_t total, const T *x_data,
                                        const T *y_data, OutType *out_data,
                                        int pre, int n, int post,
                                        Functor func) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < total) {
    int idx = tid / post % n;
    out_data[tid] = func(x_data[tid], y_data[idx]);
  }
}

template <typename Functor, typename T, typename OutType>
class TransformFunctor<Functor, T, platform::CUDADeviceContext, OutType> {
 public:
  TransformFunctor(const framework::Tensor *x, const framework::Tensor *y,
                   framework::Tensor *z, const platform::CUDADeviceContext &ctx,
                   Functor func, const bool is_xsize_larger = true)
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
    int thread = 512;
    int block = (nx_ + thread - 1) / thread;
    NormalElementwiseKernel<<<block, thread, 0, ctx_.stream()>>>(
        nx_, x_, y_, z_, nx_, 1, 1, func_);
  }

  inline void RunRowWise(int n, int pre) const {
    int thread = 512;
    int block = (nx_ + thread - 1) / thread;

    if (is_xsize_larger_) {
      NormalElementwiseKernel<<<block, thread, 0, ctx_.stream()>>>(
          nx_, x_, y_, z_, pre, n, 1, func_);
    } else {
      NormalElementwiseKernel<<<block, thread, 0, ctx_.stream()>>>(
          nx_, y_, x_, z_, pre, n, 1, func_);
    }
  }

  inline void RunMidWise(int n, int pre, int post) const {
    int thread = 512;
    int block = (nx_ + thread - 1) / thread;
    if (is_xsize_larger_) {
      NormalElementwiseKernel<<<block, thread, 0, ctx_.stream()>>>(
          nx_, x_, y_, z_, pre, n, post, func_);
    } else {
      NormalElementwiseKernel<<<block, thread, 0, ctx_.stream()>>>(
          nx_, y_, x_, z_, pre, n, post, func_);
    }
  }

 private:
  const T *x_;
  const T *y_;
  OutType *z_;
  int64_t nx_;
  const platform::CUDADeviceContext &ctx_;
  Functor func_;
  bool is_xsize_larger_;
};
#endif

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

#ifdef __NVCC__
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

#define BLOCK_X 32
#define BLOCK_Y 32

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
            T val = dy_op(x[m], y[y_offset], out[y_offset], dout[y_offset]);
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

template <typename T, typename DX_OP, typename DY_OP>
static void ElemwiseGradBroadcast1CUDA(cudaStream_t stream, const T *x,
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

#ifdef __NVCC__
template <typename T, typename DX_OP, typename DY_OP>
static __global__ void ElemwiseGradBroadcast2CUDAKernel(
    const T *x, const T *y, const T *out, const T *dout, int pre, int n,
    int post, bool is_xsize_larger, DX_OP dx_op, DY_OP dy_op, T *dx, T *dy) {
  int tid = threadIdx.x;
  int j = blockIdx.x;

  T val(0);
  int ttid = tid;

  if (is_xsize_larger) {
    while (true) {
      int i = ttid / post;
      int k = ttid % post;
      if (i >= pre) break;

      int x_offset = i * n * post + j * post + k;

      if (dx != nullptr) {
        dx[x_offset] = dx_op(x[x_offset], y[j], out[x_offset], dout[x_offset]);
      }

      if (dy != nullptr) {
        val += dy_op(x[x_offset], y[j], out[x_offset], dout[x_offset]);
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
    while (true) {
      int i = ttid / post;
      int k = ttid % post;
      if (i >= pre) break;

      int y_offset = i * n * post + j * post + k;

      if (dy != nullptr) {
        dy[y_offset] = dy_op(x[j], y[y_offset], out[y_offset], dout[y_offset]);
      }

      if (dx != nullptr) {
        val += dx_op(x[j], y[y_offset], out[y_offset], dout[y_offset]);
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

template <typename T, typename DX_OP, typename DY_OP>
static void ElemwiseGradBroadcast2CUDA(cudaStream_t stream, const T *x,
                                       const T *y, const T *out, const T *dout,
                                       int pre, int n, int post,
                                       bool is_xsize_larger, DX_OP dx_op,
                                       DY_OP dy_op, T *dx, T *dy) {
  int block_size = std::min(ELEMWISE_MAX_BLOCK_DIM, pre * post);
  int gird_size = n;
  ElemwiseGradBroadcast2CUDAKernel<<<gird_size, block_size, 0, stream>>>(
      x, y, out, dout, pre, n, post, is_xsize_larger, dx_op, dy_op, dx, dy);
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
  if (dx && dout.Holder() == dx->Holder()) {
    dx->clear();
    dx->mutable_data<T>(x_dims, ctx.GetPlace());
  }

  if (platform::is_gpu_place(ctx.GetPlace())) {
#ifdef __NVCC__
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
  PADDLE_ENFORCE_GE(axis, 0, "Axis should be in range [0, %d)", axis);
  PADDLE_ENFORCE_LT(axis, max_dim, "Axis should be in range [0, %d)", axis);

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
#ifdef __NVCC__
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
#ifdef __NVCC__
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
  PADDLE_ENFORCE_GE(axis, 0, "Axis should be in range [0, %d)", axis);
  PADDLE_ENFORCE_LT(axis, max_dim, "Axis should be in range [0, %d)", axis);
  std::vector<int> x_dims_array(max_dim);
  std::vector<int> y_dims_array(max_dim);
  std::vector<int> out_dims_array(max_dim);
  GetBroadcastDimsArrays(x_dims, y_dims, x_dims_array.data(),
                         y_dims_array.data(), out_dims_array.data(), max_dim,
                         axis);

  if (platform::is_gpu_place(ctx.GetPlace())) {
#ifdef __NVCC__
    CommonForwardBroadcastCUDA<Functor, T>(
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
  PADDLE_ENFORCE_GE(axis, 0, "Axis should be in range [0, %d)", axis);
  PADDLE_ENFORCE_LT(axis, max_dim, "Axis should be in range [0, %d)", axis);

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

#ifdef __NVCC__
template <typename T, typename CompoundFunctor, bool BcastY,
          bool KeepIntermediateOut, bool SameShapeOfIntermediateOutAndOut>
static __global__ void FusedElemwiseAndActBroadcast1CUDAKernel(
    const T *x, const T *y, int h, int w, CompoundFunctor compound_functor,
    T *out, T *intermediate_out) {
  int j = blockIdx.x;
  int i = threadIdx.x;

  while (i < h) {
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

    i += ELEMWISE_MAX_BLOCK_DIM;
  }
}

template <typename T, typename CompoundFunctor, bool BcastY,
          bool KeepIntermediateOut, bool SameShapeOfIntermediateOutAndOut>
static void FusedElemwiseAndActBroadcast1CUDA(cudaStream_t stream, const T *x,
                                              const T *y,
                                              CompoundFunctor compound_functor,
                                              int h, int w, T *out,
                                              T *intermediate_out) {
  int block_size = std::min(ELEMWISE_MAX_BLOCK_DIM, h);
  int gird_size = w;
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
static void FusedElemwiseAndActBroadcast2CUDA(cudaStream_t stream, const T *x,
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
#ifdef __NVCC__
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
#ifdef __NVCC__
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
    T x_val = x_[i];
    T y_val = y_[i];
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
  for_range(
      FusedElemwiseAndActGradNoBroadcast<T, DX_OP, DY_OP, DIntermediate_OP,
                                         UseIntermediateOut>{
          x->data<T>(), y->data<T>(),
          intermediate_out ? intermediate_out->data<T>() : nullptr,
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
  for (int i = 0; i < h; ++i) {
    for (int j = 0; j < w; ++j) {
      int offset = i * w + j;

      tmp_out_idx = BcastY ? j : offset;
      y_idx = BcastY ? j : offset;
      x_idx = BcastY ? offset : j;

      if (SameShapeOfIntermediateOutAndOut) {
        tmp_out_idx = offset;
      }

      if (dx != nullptr) {
        T tmp = UseIntermediateOut
                    ? dx_op.UseIntermediateOut(x[x_idx], y[y_idx],
                                               intermediate_out[tmp_out_idx],
                                               out[offset], dout[offset])
                    : dx_op.Recompute(x[x_idx], y[y_idx], out[offset],
                                      dout[offset]);

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
                    ? dy_op.UseIntermediateOut(x[x_idx], y[y_idx],
                                               intermediate_out[tmp_out_idx],
                                               out[offset], dout[offset])
                    : dy_op.Recompute(x[x_idx], y[y_idx], out[offset],
                                      dout[offset]);
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
                          x[x_idx], intermediate_out[tmp_out_idx], out[offset],
                          dout[offset])
                    : dintermediate_op.Recompute(x[x_idx], y[y_idx],
                                                 out[offset], dout[i]);
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
  for (int i = 0; i < pre; ++i) {
    for (int j = 0; j < n; ++j) {
      for (int k = 0; k < post; ++k) {
        int offset = i * n * post + j * post + k;

        tmp_out_idx = BcastY ? j : offset;
        y_idx = BcastY ? j : offset;
        x_idx = BcastY ? offset : j;

        if (SameShapeOfIntermediateOutAndOut) {
          tmp_out_idx = offset;
        }

        if (dx != nullptr) {
          T tmp = UseIntermediateOut
                      ? dx_op.UseIntermediateOut(x[x_idx], y[y_idx],
                                                 intermediate_out[tmp_out_idx],
                                                 out[offset], dout[offset])
                      : dx_op.Recompute(x[x_idx], y[y_idx], out[offset],
                                        dout[offset]);

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
          T tmp = UseIntermediateOut
                      ? dy_op.UseIntermediateOut(x[x_idx], y[y_idx],
                                                 intermediate_out[tmp_out_idx],
                                                 out[offset], dout[offset])
                      : dy_op.Recompute(x[x_idx], y[y_idx], out[offset],
                                        dout[offset]);
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
                            x[x_idx], intermediate_out[tmp_out_idx],
                            out[offset], dout[offset])
                      : dintermediate_op.Recompute(x[x_idx], y[y_idx],
                                                   out[offset], dout[i]);
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

#ifdef __NVCC__
template <typename T, typename DX_OP, typename DY_OP, typename DIntermediate_OP,
          bool UseIntermediateOut, bool BcastY,
          bool SameShapeOfIntermediateOutAndOut>
static __global__ void FusedElemwiseAndActGradBroadcast1CUDAKernel(
    const T *x, const T *y, const T *intermediate_out, const T *out,
    const T *dout, int h, int w, DX_OP dx_op, DY_OP dy_op,
    DIntermediate_OP dintermediate_op, T *dx, T *dy, T *d_intermediate) {
  int j = blockIdx.x;
  int i = threadIdx.x;
  int tid = threadIdx.x;
  T val(0), inter_val(0);
  int64_t tmp_out_idx, x_idx, y_idx;

  do {
    int offset = i * w + j;

    tmp_out_idx = BcastY ? j : offset;
    y_idx = BcastY ? j : offset;
    x_idx = BcastY ? offset : j;

    if (SameShapeOfIntermediateOutAndOut) {
      tmp_out_idx = offset;
    }

    if (dx != nullptr) {
      T tmp =
          UseIntermediateOut
              ? dx_op.UseIntermediateOut(x[x_idx], y[y_idx],
                                         intermediate_out[tmp_out_idx],
                                         out[offset], dout[offset])
              : dx_op.Recompute(x[x_idx], y[y_idx], out[offset], dout[offset]);

      if (BcastY) {
        dx[x_idx] = tmp;
      } else {
        val += tmp;
      }
    }
    if (dy != nullptr) {
      T tmp =
          UseIntermediateOut
              ? dy_op.UseIntermediateOut(x[x_idx], y[y_idx],
                                         intermediate_out[tmp_out_idx],
                                         out[offset], dout[offset])
              : dy_op.Recompute(x[x_idx], y[y_idx], out[offset], dout[offset]);
      if (BcastY) {
        val += tmp;
      } else {
        dy[y_idx] = tmp;
      }
    }
    if (d_intermediate != nullptr) {
      T tmp = UseIntermediateOut
                  ? dintermediate_op.UseIntermediateOut(
                        y[y_idx], intermediate_out[tmp_out_idx], out[offset],
                        dout[offset])
                  : dintermediate_op.Recompute(x[x_idx], y[y_idx], out[offset],
                                               dout[offset]);
      if (SameShapeOfIntermediateOutAndOut) {
        d_intermediate[tmp_out_idx] = tmp;
      } else {
        inter_val += tmp;
      }
    }

    i += ELEMWISE_MAX_BLOCK_DIM;
  } while (i < h);

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
static void FusedElemwiseAndActGradBroadcast1CUDA(
    cudaStream_t stream, const T *x, const T *y, const T *intermediate_out,
    const T *out, const T *dout, int h, int w, DX_OP dx_op, DY_OP dy_op,
    DIntermediate_OP dintermediate_op, T *dx, T *dy, T *d_intermediate) {
  int block_size = std::min(ELEMWISE_MAX_BLOCK_DIM, h);
  int gird_size = w;
  FusedElemwiseAndActGradBroadcast1CUDAKernel<
      T, DX_OP, DY_OP, DIntermediate_OP, UseIntermediateOut, BcastY,
      SameShapeOfIntermediateOutAndOut><<<gird_size, block_size, 0, stream>>>(
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
  while (true) {
    int i = ttid / post;
    int k = ttid % post;
    if (i >= pre) break;

    int offset = i * n * post + j * post + k;

    tmp_out_idx = BcastY ? j : offset;
    y_idx = BcastY ? j : offset;
    x_idx = BcastY ? offset : j;

    if (SameShapeOfIntermediateOutAndOut) {
      tmp_out_idx = offset;
    }

    if (dx != nullptr) {
      T tmp =
          UseIntermediateOut
              ? dx_op.UseIntermediateOut(x[x_idx], y[y_idx],
                                         intermediate_out[tmp_out_idx],
                                         out[offset], dout[offset])
              : dx_op.Recompute(x[x_idx], y[y_idx], out[offset], dout[offset]);

      if (BcastY) {
        dx[x_idx] = tmp;
      } else {
        val += tmp;
      }
    }
    if (dy != nullptr) {
      T tmp =
          UseIntermediateOut
              ? dy_op.UseIntermediateOut(x[x_idx], y[y_idx],
                                         intermediate_out[tmp_out_idx],
                                         out[offset], dout[offset])
              : dy_op.Recompute(x[x_idx], y[y_idx], out[offset], dout[offset]);
      if (BcastY) {
        val += tmp;
      } else {
        dy[y_idx] = tmp;
      }
    }
    if (d_intermediate != nullptr) {
      T tmp = UseIntermediateOut
                  ? dintermediate_op.UseIntermediateOut(
                        y[y_idx], intermediate_out[tmp_out_idx], out[offset],
                        dout[offset])
                  : dintermediate_op.Recompute(x[x_idx], y[y_idx], out[offset],
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
    cudaStream_t stream, const T *x, const T *y, const T *intermediate_out,
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
  if (post == 1) {
    int h = pre;
    int w = n;
    if (platform::is_gpu_place(ctx.GetPlace())) {
#ifdef __NVCC__
      FusedElemwiseAndActGradBroadcast1CUDA<T, DX_OP, DY_OP, DIntermediate_OP,
                                            UseIntermediateOut, BcastY,
                                            SameShapeOfIntermediateOutAndOut>(
          ctx.template device_context<DeviceContext>().stream(), x->data<T>(),
          y->data<T>(),
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
          x->data<T>(), y->data<T>(),
          intermediate_out == nullptr ? nullptr : intermediate_out->data<T>(),
          out->data<T>(), dout->data<T>(), h, w, dx_op, dy_op, dintermediate_op,
          dx == nullptr ? nullptr : dx->mutable_data<T>(ctx.GetPlace()),
          dy == nullptr ? nullptr : dy->mutable_data<T>(ctx.GetPlace()),
          dintermediate == nullptr ? nullptr : dintermediate->mutable_data<T>(
                                                   ctx.GetPlace()));
    }
  } else {
    if (platform::is_gpu_place(ctx.GetPlace())) {
#ifdef __NVCC__
      FusedElemwiseAndActGradBroadcast2CUDA<T, DX_OP, DY_OP, DIntermediate_OP,
                                            UseIntermediateOut, BcastY,
                                            SameShapeOfIntermediateOutAndOut>(
          ctx.template device_context<DeviceContext>().stream(), x->data<T>(),
          y->data<T>(),
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
          x->data<T>(), y->data<T>(),
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
    PADDLE_ENFORCE(intermediate_out, "intermediate_out should not be nullptr");
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
    PADDLE_ENFORCE(intermediate_out,
                   "The save_intermediate_out is opened, "
                   "intermediate_out should not be nullptr.");
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
