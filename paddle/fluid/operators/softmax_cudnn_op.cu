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

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/amp/fp16_type_traits.h"
#include "paddle/fluid/operators/math/math_cuda_utils.h"
#include "paddle/fluid/operators/softmax_impl.cuh"
#include "paddle/fluid/operators/softmax_op.h"
#include "paddle/fluid/platform/cuda_device_function.h"
#ifdef PADDLE_WITH_HIP
#include "paddle/fluid/platform/miopen_helper.h"
#else
#include "paddle/fluid/platform/cudnn_helper.h"
#endif

namespace paddle {
namespace platform {
struct CUDAPlace;
struct float16;
}  // namespace platform
}  // namespace paddle

namespace paddle {
namespace operators {

using ScopedTensorDescriptor = platform::ScopedTensorDescriptor;
using DataLayout = platform::CudnnDataLayout;
using Tensor = framework::Tensor;

// Vectorization trait 4 * sizeof(T)
template <typename T>
class VecT4 {};
template <>
class VecT4<double> {
 public:
  using Type = long4;
};
template <>
class VecT4<float> {
 public:
  using Type = int4;
};
template <>
class VecT4<platform::float16> {
 public:
  using Type = int2;
};

// Vectorization trait 2 * sizeof(T)
template <typename T>
class VecT2 {};
template <>
class VecT2<double> {
 public:
  using Type = int4;
};
template <>
class VecT2<float> {
 public:
  using Type = int2;
};
template <>
class VecT2<platform::float16> {
 public:
  using Type = int;
};

int static inline log2_ceil(int value) {
  int log2_value = 0;
  while ((1 << log2_value) < value) ++log2_value;
  return log2_value;
}

/*
Core function of computing softmax forward for axis=-1.
The computation includes
  - Compute maximum of batch: maxvalue_{i} = max_j src_{i,j}
  - Compute sum of exp batch: s_{i} = sum_{j}{ exp(src_{i,j} - maxvalue_{i} }
  - Compute: (a_{i,j} - maxvalue_{i}) / s_{i}
One warp (32 threads) is used to compute 1 or 2 batch (kBatchSize).
For reduction max (sum), firstly compute max (sum) to one warp, then use shuffle
api to compute max (sum) in one warp.
*/
template <typename T, typename VecT, typename AccT, int Log2Elements,
          bool LogMode = false>
__global__ void WarpSoftmaxForward(T* softmax, const T* src,
                                   const int batch_size, const int stride,
                                   const int element_count) {
  constexpr int kDimCeil = 1 << Log2Elements;
  constexpr int kWarpSize = (kDimCeil < 32) ? kDimCeil : 32;
  constexpr int kVSize = sizeof(VecT) / sizeof(T);
  constexpr int kIterations = kDimCeil / kWarpSize;
  constexpr int kIterationsV =
      (kIterations >= kVSize) ? (kIterations / kVSize) : 1;
  constexpr int kBatchSize = (kDimCeil <= 32) ? 2 : 1;

  int first_batch = (blockDim.y * blockIdx.x + threadIdx.y) * kBatchSize;

  // max index to read
  int idx_max_v[kBatchSize];
#pragma unroll
  for (int i = 0; i < kBatchSize; i++) {
    int idx_max = ((i + first_batch) < batch_size) ? element_count : 0;
    idx_max_v[i] = idx_max / kVSize;
  }

  // read data from global memory
  AccT srcdata[kBatchSize][kIterationsV][kVSize];

#pragma unroll
  for (int i = 0; i < kBatchSize; ++i) {
// read data
#pragma unroll
    for (int it = 0; it < kIterationsV; ++it) {
      int src_idx = threadIdx.x + it * kWarpSize;
      if (kVSize == 1) {
        if (src_idx < idx_max_v[i]) {
          srcdata[i][it][0] =
              static_cast<AccT>(src[(first_batch + i) * stride + src_idx]);
        } else {
          srcdata[i][it][0] = -std::numeric_limits<AccT>::infinity();
        }
      } else {
        const VecT* src_v =
            reinterpret_cast<const VecT*>(&src[(first_batch + i) * stride]);
        if (src_idx < idx_max_v[i]) {
          VecT srctmp = src_v[src_idx];
          const T* srcinptr = reinterpret_cast<const T*>(&srctmp);
#pragma unroll
          for (int s = 0; s < kVSize; s++) {
            srcdata[i][it][s] = static_cast<AccT>(srcinptr[s]);
          }
        } else {
#pragma unroll
          for (int s = 0; s < kVSize; s++) {
            srcdata[i][it][s] = -std::numeric_limits<AccT>::infinity();
          }
        }
      }
    }
  }

  // compute max value
  AccT max_value[kBatchSize];
#pragma unroll
  for (int i = 0; i < kBatchSize; ++i) {
    // it = 0
    AccT valmax = srcdata[i][0][0];
#pragma unroll
    for (int s = 1; s < kVSize; ++s) {
      valmax = (valmax > srcdata[i][0][s]) ? valmax : srcdata[i][0][s];
    }
    max_value[i] = valmax;

// it = 1, 2, ...
#pragma unroll
    for (int it = 1; it < kIterationsV; ++it) {
      AccT valmax = srcdata[i][it][0];
#pragma unroll
      for (int s = 1; s < kVSize; ++s) {
        valmax = (valmax > srcdata[i][it][s]) ? valmax : srcdata[i][it][s];
      }
      max_value[i] = (max_value[i] > valmax) ? max_value[i] : valmax;
    }
  }
  WarpReduceMax<AccT, kBatchSize, kWarpSize>(max_value);

  // compute sum
  AccT sum[kBatchSize];
#pragma unroll
  for (int i = 0; i < kBatchSize; ++i) {
    // it = 0
    if (LogMode) {
      sum[i] = std::exp(srcdata[i][0][0] - max_value[i]);
    } else {
      srcdata[i][0][0] = std::exp(srcdata[i][0][0] - max_value[i]);
      sum[i] = srcdata[i][0][0];
    }
#pragma unroll
    for (int s = 1; s < kVSize; ++s) {
      if (LogMode) {
        sum[i] += std::exp(srcdata[i][0][s] - max_value[i]);
      } else {
        srcdata[i][0][s] = std::exp(srcdata[i][0][s] - max_value[i]);
        sum[i] += srcdata[i][0][s];
      }
    }

// it = 1, 2, ...
#pragma unroll
    for (int it = 1; it < kIterationsV; ++it) {
#pragma unroll
      for (int s = 0; s < kVSize; ++s) {
        if (LogMode) {
          sum[i] += std::exp(srcdata[i][it][s] - max_value[i]);
        } else {
          srcdata[i][it][s] = std::exp(srcdata[i][it][s] - max_value[i]);
          sum[i] += srcdata[i][it][s];
        }
      }
    }
  }
  WarpReduceSum<AccT, kBatchSize, kWarpSize>(sum);

// write result to global memory
#pragma unroll
  for (int i = 0; i < kBatchSize; ++i) {
    if (LogMode) {
      sum[i] = std::log(sum[i]);
    }

#pragma unroll
    for (int it = 0; it < kIterationsV; ++it) {
      int idx = threadIdx.x + it * kWarpSize;
      if (kVSize == 1) {
        if (idx < idx_max_v[i]) {
          if (LogMode) {
            softmax[(first_batch + i) * stride + idx] =
                srcdata[i][it][0] - max_value[i] - sum[i];
          } else {
            softmax[(first_batch + i) * stride + idx] =
                srcdata[i][it][0] / sum[i];
          }
        } else {
          break;
        }
      } else {
        VecT* softmax_v =
            reinterpret_cast<VecT*>(&softmax[(first_batch + i) * stride]);
        VecT tmpdata;
        T* tmpptr = reinterpret_cast<T*>(&tmpdata);
#pragma unroll
        for (int s = 0; s < kVSize; ++s) {
          if (LogMode) {
            tmpptr[s] = srcdata[i][it][s] - max_value[i] - sum[i];
          } else {
            tmpptr[s] = srcdata[i][it][s] / sum[i];
          }
        }

        if (idx < idx_max_v[i]) {
          softmax_v[idx] = tmpdata;
        } else {
          break;
        }
      }
    }
  }
}

/*
Core function of computing softmax backward for axis=-1.
The computation includes
  - Compute sum of exp batch: s_{i} = sum_{j} {src_{i,j} * grad_{i,j}
  - Compute src_{i,j} * ( grad_{i,j}) - s_{i} )
One warp (32 threads) is used to compute 1 or 2 batch (kBatchSize).
For reduction max (sum), firstly compute max (sum) to one warp, then use shuffle
api to compute max (sum) in one warp.
*/
template <typename T, typename VecT, typename AccT, int Log2Elements,
          bool LogMode = false>
__global__ void WarpSoftmaxBackward(T* dst, const T* grad, const T* src,
                                    int batch_size, int stride,
                                    int element_count) {
  constexpr int kVSize = sizeof(VecT) / sizeof(T);
  constexpr int kDimCeil = 1 << Log2Elements;
  constexpr int kWarpSize = (kDimCeil < 32) ? kDimCeil : 32;
  constexpr int kIterations = kDimCeil / kWarpSize;
  constexpr int kBatchSize = (kDimCeil <= 128) ? 2 : 1;
  constexpr int kIterationsV =
      (kIterations >= kVSize) ? (kIterations / kVSize) : 1;
  int element_count_v = element_count / kVSize;

  int first_batch = (blockDim.y * blockIdx.x + threadIdx.y) * kBatchSize;
  int local_batches = batch_size - first_batch;
  if (local_batches > kBatchSize) {
    local_batches = kBatchSize;
  }

  // read data from global memory
  VecT src_reg[kBatchSize][kIterationsV];
  VecT grad_reg[kBatchSize][kIterationsV];

  for (int i = 0; i < kBatchSize; ++i) {
    const VecT* src_v =
        reinterpret_cast<const VecT*>(&src[(first_batch + i) * stride]);
    const VecT* grad_v =
        reinterpret_cast<const VecT*>(&grad[(first_batch + i) * stride]);

    // max index to read
    int idx_max = (i < local_batches) ? element_count : 0;
    int idx_max_v = idx_max / kVSize;

    // read data
    for (int it = 0; it < kIterationsV; ++it) {
      int src_idx = threadIdx.x + it * kWarpSize;
      if (src_idx < idx_max_v) {
        src_reg[i][it] = src_v[src_idx];
        grad_reg[i][it] = grad_v[src_idx];
      } else {
#pragma unroll
        for (int s = 0; s < kVSize; s++) {
          reinterpret_cast<T*>(&src_reg[i][it])[s] = 0.0;
          reinterpret_cast<T*>(&grad_reg[i][it])[s] = 0.0;
        }
      }
    }
  }

  // compute sum
  AccT sum[kBatchSize]{0.0};
#pragma unroll
  for (int i = 0; i < kBatchSize; ++i) {
#pragma unroll
    for (int it = 0; it < kIterationsV; ++it) {
      T* gradptr = reinterpret_cast<T*>(&grad_reg[i][it]);
      T* srcptr = reinterpret_cast<T*>(&src_reg[i][it]);
#pragma unroll
      for (int s = 0; s < kVSize; ++s) {
        if (LogMode) {
          sum[i] += static_cast<AccT>(gradptr[s]);
        } else {
          sum[i] += static_cast<AccT>(gradptr[s] * srcptr[s]);
        }
      }
    }
  }
  WarpReduceSum<AccT, kBatchSize, kWarpSize>(sum);

// write result
#pragma unroll
  for (int i = 0; i < kBatchSize; ++i) {
    if (i >= local_batches) break;

    VecT* dst_v = reinterpret_cast<VecT*>(&dst[(first_batch + i) * stride]);

    // max index to write
    int idx_max = (i < local_batches) ? element_count : 0;
    int idx_max_v = idx_max / kVSize;

#pragma unroll
    for (int it = 0; it < kIterationsV; ++it) {
      VecT tmpdata;
      T* tmpptr = reinterpret_cast<T*>(&tmpdata);
      T* gradptr = reinterpret_cast<T*>(&grad_reg[i][it]);
      T* srcptr = reinterpret_cast<T*>(&src_reg[i][it]);
#pragma unroll
      for (int s = 0; s < kVSize; ++s) {
        if (LogMode) {
          tmpptr[s] = static_cast<AccT>(gradptr[s]) -
                      std::exp(static_cast<AccT>(srcptr[s])) * sum[i];
        } else {
          tmpptr[s] = static_cast<AccT>(srcptr[s]) *
                      (static_cast<AccT>(gradptr[s]) - sum[i]);
        }
      }

      int idx = threadIdx.x + it * kWarpSize;
      if (idx < idx_max_v) {
        dst_v[idx] = tmpdata;
      }
    }
  }
}

#define SOFTMAX_WARP_FORWARD_CASE(Log2Elements, AccT)                         \
  case Log2Elements:                                                          \
    WarpSoftmaxForward<                                                       \
        T, VecT, AccT, Log2Elements,                                          \
        LogMode><<<blocks, threads, 0, ctx.cuda_device_context().stream()>>>( \
        dst, src, batch_size, stride, element_count);                         \
    break;

/*
  Wrapper of softmax formward with template instantiation on size of input.
*/
template <typename T, typename VecT, bool LogMode>
void SwitchWarpSoftmaxForward(const int blocks, const dim3 threads,
                              const framework::ExecutionContext& ctx, T* dst,
                              const T* src, const int batch_size,
                              const int stride, const int element_count,
                              int Log2Elements) {
  using AccT = typename details::MPTypeTrait<T>::Type;
  switch (Log2Elements) {
    SOFTMAX_WARP_FORWARD_CASE(0, AccT);
    SOFTMAX_WARP_FORWARD_CASE(1, AccT);
    SOFTMAX_WARP_FORWARD_CASE(2, AccT);
    SOFTMAX_WARP_FORWARD_CASE(3, AccT);
    SOFTMAX_WARP_FORWARD_CASE(4, AccT);
    SOFTMAX_WARP_FORWARD_CASE(5, AccT);
    SOFTMAX_WARP_FORWARD_CASE(6, AccT);
    SOFTMAX_WARP_FORWARD_CASE(7, AccT);
    SOFTMAX_WARP_FORWARD_CASE(8, AccT);
    SOFTMAX_WARP_FORWARD_CASE(9, AccT);
    default:
      break;
  }
}

#define SOFTMAX_WARP_BACKWARD_CASE(Log2Elements, AccT)                        \
  case Log2Elements:                                                          \
    WarpSoftmaxBackward<                                                      \
        T, VecT, AccT, Log2Elements,                                          \
        LogMode><<<blocks, threads, 0, ctx.cuda_device_context().stream()>>>( \
        dst, grad, src, batch_size, stride, element_count);                   \
    break;

/*
Wrapper of softmax backward with template instantiation on size of input.
*/
template <typename T, typename VecT, bool LogMode>
void SwitchWarpSoftmaxBackward(const int blocks, const dim3 threads,
                               const framework::ExecutionContext& ctx, T* dst,
                               const T* grad, const T* src,
                               const int batch_size, const int stride,
                               const int element_count, int Log2Elements) {
  using AccT = typename details::MPTypeTrait<T>::Type;
  switch (Log2Elements) {
    SOFTMAX_WARP_BACKWARD_CASE(0, AccT);
    SOFTMAX_WARP_BACKWARD_CASE(1, AccT);
    SOFTMAX_WARP_BACKWARD_CASE(2, AccT);
    SOFTMAX_WARP_BACKWARD_CASE(3, AccT);
    SOFTMAX_WARP_BACKWARD_CASE(4, AccT);
    SOFTMAX_WARP_BACKWARD_CASE(5, AccT);
    SOFTMAX_WARP_BACKWARD_CASE(6, AccT);
    SOFTMAX_WARP_BACKWARD_CASE(7, AccT);
    SOFTMAX_WARP_BACKWARD_CASE(8, AccT);
    SOFTMAX_WARP_BACKWARD_CASE(9, AccT);
    default:
      break;
  }
}

#undef SOFTMAX_WARP_FORWARD_CASE
#undef SOFTMAX_WARP_BACKWARD_CASE

template <typename T, bool LogMode = false>
class SoftmaxCUDNNKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* x = ctx.Input<Tensor>("X");
    auto* out = ctx.Output<Tensor>("Out");
    out->mutable_data<T>(ctx.GetPlace());
    auto* out_data = out->data<T>();

    auto dims = x->dims();
    const int rank = dims.size();
    const int axis = CanonicalAxis(ctx.Attr<int>("axis"), rank);
    const int dim = dims[axis];
    const int N = SizeToAxis(axis, dims);
    const int D = SizeOutAxis(axis, dims);

    constexpr int max_dim = 320;
    constexpr int warps_per_block = 4;

    if (D == 1 && dim <= max_dim && sizeof(T) <= 4) {
      const int kDimLog2 = static_cast<int>(log2_ceil(dim));
      const int kDimCeil = 1 << kDimLog2;
      int kWarpSize = (kDimCeil < 32) ? kDimCeil : 32;
      int batches_per_warp = (kDimCeil <= 32) ? 2 : 1;

      // use 128 threads per block to maximimize gpu utilization
      constexpr int threads_per_block = 128;

      int warps_per_block = (threads_per_block / kWarpSize);
      int batches_per_block = warps_per_block * batches_per_warp;
      int blocks = (N + batches_per_block - 1) / batches_per_block;
      dim3 threads(kWarpSize, warps_per_block, 1);

      // vectorization read/write
      using T4 = typename VecT4<T>::Type;
      using T2 = typename VecT2<T>::Type;
      if (dim % 4 == 0) {
        SwitchWarpSoftmaxForward<T, T4, LogMode>(blocks, threads, ctx, out_data,
                                                 x->data<T>(), N, dim, dim,
                                                 kDimLog2);
      } else if (dim % 2 == 0) {
        SwitchWarpSoftmaxForward<T, T2, LogMode>(blocks, threads, ctx, out_data,
                                                 x->data<T>(), N, dim, dim,
                                                 kDimLog2);
      } else {
        SwitchWarpSoftmaxForward<T, T, LogMode>(blocks, threads, ctx, out_data,
                                                x->data<T>(), N, dim, dim,
                                                kDimLog2);
      }
    } else {
      ScopedTensorDescriptor desc;
      std::vector<int> tensor_dims = {N, dim, D, 1};
      DataLayout layout = DataLayout::kNCHW;
#ifdef PADDLE_WITH_HIP
      miopenTensorDescriptor_t desc_ = desc.descriptor<T>(layout, tensor_dims);
#else
      cudnnTensorDescriptor_t desc_ = desc.descriptor<T>(layout, tensor_dims);
#endif

      auto& dev_ctx =
          ctx.template device_context<platform::CUDADeviceContext>();
      auto handle = dev_ctx.cudnn_handle();

#ifdef PADDLE_WITH_HIP
      auto mode = axis == rank - 1 ? MIOPEN_SOFTMAX_MODE_INSTANCE
                                   : MIOPEN_SOFTMAX_MODE_CHANNEL;
      if (LogMode) {
        PADDLE_ENFORCE_CUDA_SUCCESS(platform::dynload::miopenSoftmaxForward_V2(
            handle, platform::CudnnDataType<T>::kOne(), desc_, x->data<T>(),
            platform::CudnnDataType<T>::kZero(), desc_, out_data,
            MIOPEN_SOFTMAX_LOG, mode));
      } else {
        PADDLE_ENFORCE_CUDA_SUCCESS(platform::dynload::miopenSoftmaxForward_V2(
            handle, platform::CudnnDataType<T>::kOne(), desc_, x->data<T>(),
            platform::CudnnDataType<T>::kZero(), desc_, out_data,
            MIOPEN_SOFTMAX_ACCURATE, mode));
      }
#else
      auto mode = axis == rank - 1 ? CUDNN_SOFTMAX_MODE_INSTANCE
                                   : CUDNN_SOFTMAX_MODE_CHANNEL;
      if (LogMode) {
        PADDLE_ENFORCE_CUDA_SUCCESS(platform::dynload::cudnnSoftmaxForward(
            handle, CUDNN_SOFTMAX_LOG, mode, platform::CudnnDataType<T>::kOne(),
            desc_, x->data<T>(), platform::CudnnDataType<T>::kZero(), desc_,
            out_data));
      } else {
        PADDLE_ENFORCE_CUDA_SUCCESS(platform::dynload::cudnnSoftmaxForward(
            handle, CUDNN_SOFTMAX_ACCURATE, mode,
            platform::CudnnDataType<T>::kOne(), desc_, x->data<T>(),
            platform::CudnnDataType<T>::kZero(), desc_, out_data));
      }
#endif
    }
  }
};

template <typename T, bool LogMode = false>
class SoftmaxGradCUDNNKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* out = ctx.Input<Tensor>("Out");
    auto* dout = ctx.Input<Tensor>(framework::GradVarName("Out"));
    auto* dx = ctx.Output<Tensor>(framework::GradVarName("X"));
    dx->mutable_data<T>(ctx.GetPlace());
    auto* dx_data = dx->data<T>();

    auto dims = out->dims();
    const int rank = dims.size();
    const int axis = CanonicalAxis(ctx.Attr<int>("axis"), rank);
    const int dim = dims[axis];
    const int N = SizeToAxis(axis, dims);
    const int D = SizeOutAxis(axis, dims);

    constexpr int max_dim = 320;
    constexpr int warps_per_block = 4;

    if (D == 1 && dim <= max_dim && sizeof(T) <= 4) {
      const int kDimLog2 = log2_ceil(dim);
      const int kDimCeil = 1 << kDimLog2;
      int kWarpSize = (kDimCeil < 32) ? kDimCeil : 32;
      int batches_per_warp = (kDimCeil <= 128) ? 2 : 1;
      constexpr int threads_per_block = 128;

      int warps_per_block = (threads_per_block / kWarpSize);
      int batches_per_block = warps_per_block * batches_per_warp;
      int blocks = (N + batches_per_block - 1) / batches_per_block;
      dim3 threads(kWarpSize, warps_per_block, 1);

      // vectorization read/write
      using T4 = typename VecT4<T>::Type;
      using T2 = typename VecT2<T>::Type;
      if (dim % 4 == 0) {
        SwitchWarpSoftmaxBackward<T, T4, LogMode>(
            blocks, threads, ctx, dx_data, dout->data<T>(), out->data<T>(), N,
            dim, dim, kDimLog2);
      } else if (dim % 2 == 0) {
        SwitchWarpSoftmaxBackward<T, T2, LogMode>(
            blocks, threads, ctx, dx_data, dout->data<T>(), out->data<T>(), N,
            dim, dim, kDimLog2);
      } else {
        SwitchWarpSoftmaxBackward<T, T, LogMode>(
            blocks, threads, ctx, dx_data, dout->data<T>(), out->data<T>(), N,
            dim, dim, kDimLog2);
      }
    } else {
      ScopedTensorDescriptor desc;
      std::vector<int> tensor_dims = {N, dim, D, 1};
      DataLayout layout = DataLayout::kNCHW;
#ifdef PADDLE_WITH_HIP
      miopenTensorDescriptor_t desc_ = desc.descriptor<T>(layout, tensor_dims);
#else
      cudnnTensorDescriptor_t desc_ = desc.descriptor<T>(layout, tensor_dims);
#endif

      auto& dev_ctx =
          ctx.template device_context<platform::CUDADeviceContext>();
      auto handle = dev_ctx.cudnn_handle();

#ifdef PADDLE_WITH_HIP
      auto mode = axis == rank - 1 ? MIOPEN_SOFTMAX_MODE_INSTANCE
                                   : MIOPEN_SOFTMAX_MODE_CHANNEL;
      if (LogMode) {
        PADDLE_ENFORCE_CUDA_SUCCESS(platform::dynload::miopenSoftmaxBackward_V2(
            handle, platform::CudnnDataType<T>::kOne(), desc_, out->data<T>(),
            desc_, dout->data<T>(), platform::CudnnDataType<T>::kZero(), desc_,
            dx_data, MIOPEN_SOFTMAX_LOG, mode));
      } else {
        PADDLE_ENFORCE_CUDA_SUCCESS(platform::dynload::miopenSoftmaxBackward_V2(
            handle, platform::CudnnDataType<T>::kOne(), desc_, out->data<T>(),
            desc_, dout->data<T>(), platform::CudnnDataType<T>::kZero(), desc_,
            dx_data, MIOPEN_SOFTMAX_ACCURATE, mode));
      }
#else
      auto mode = axis == rank - 1 ? CUDNN_SOFTMAX_MODE_INSTANCE
                                   : CUDNN_SOFTMAX_MODE_CHANNEL;
      if (LogMode) {
        PADDLE_ENFORCE_CUDA_SUCCESS(platform::dynload::cudnnSoftmaxBackward(
            handle, CUDNN_SOFTMAX_LOG, mode, platform::CudnnDataType<T>::kOne(),
            desc_, out->data<T>(), desc_, dout->data<T>(),
            platform::CudnnDataType<T>::kZero(), desc_, dx_data));
      } else {
        PADDLE_ENFORCE_CUDA_SUCCESS(platform::dynload::cudnnSoftmaxBackward(
            handle, CUDNN_SOFTMAX_ACCURATE, mode,
            platform::CudnnDataType<T>::kOne(), desc_, out->data<T>(), desc_,
            dout->data<T>(), platform::CudnnDataType<T>::kZero(), desc_,
            dx_data));
      }
#endif
    }
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;
#ifdef PADDLE_WITH_HIP
// MIOPEN do not support double
REGISTER_OP_KERNEL(softmax, CUDNN, plat::CUDAPlace,
                   ops::SoftmaxCUDNNKernel<float>,
                   ops::SoftmaxCUDNNKernel<plat::float16>);
REGISTER_OP_KERNEL(softmax_grad, CUDNN, plat::CUDAPlace,
                   ops::SoftmaxGradCUDNNKernel<float>,
                   ops::SoftmaxGradCUDNNKernel<plat::float16>);
#else
REGISTER_OP_KERNEL(softmax, CUDNN, plat::CUDAPlace,
                   ops::SoftmaxCUDNNKernel<float>,
                   ops::SoftmaxCUDNNKernel<double>,
                   ops::SoftmaxCUDNNKernel<plat::float16>);
REGISTER_OP_KERNEL(softmax_grad, CUDNN, plat::CUDAPlace,
                   ops::SoftmaxGradCUDNNKernel<float>,
                   ops::SoftmaxGradCUDNNKernel<double>,
                   ops::SoftmaxGradCUDNNKernel<plat::float16>);
#endif
