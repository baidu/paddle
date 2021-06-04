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
#ifdef __NVCC__
#include "cub/cub.cuh"
#endif
#ifdef __HIPCC__
#include <hipcub/hipcub.hpp>
namespace cub = hipcub;
#endif
#include "paddle/fluid/operators/amp/fp16_type_traits.h"
#include "paddle/fluid/operators/math/cross_entropy.h"
#include "paddle/fluid/operators/math/math_function.h"
#include "paddle/fluid/operators/softmax_impl.cuh"
#include "paddle/fluid/operators/softmax_with_cross_entropy_op.h"
#include "paddle/fluid/platform/for_range.h"
#ifdef PADDLE_WITH_HIP
#include "paddle/fluid/platform/miopen_helper.h"
#else
#include "paddle/fluid/platform/cudnn_helper.h"
#endif

namespace paddle {
namespace operators {

using ScopedTensorDescriptor = platform::ScopedTensorDescriptor;
using DataLayout = platform::DataLayout;
using Tensor = framework::Tensor;

// Wrapper of log function. Use log(float32) for float16
template <typename T>
static __device__ __forceinline__ T Log(T x) {
  using AccT = typename details::MPTypeTrait<T>::Type;
  AccT logx = std::log(static_cast<AccT>(x));
  return math::TolerableValue<T>()(static_cast<T>(logx));
}

// Wrapper of exp function. Use exp(float32) for float16
template <typename T>
static __device__ __forceinline__ T Exp(T x) {
  using AccT = typename details::MPTypeTrait<T>::Type;
  AccT expx = std::exp(static_cast<AccT>(x));
  return math::TolerableValue<T>()(static_cast<T>(expx));
}

// log2(value)
static inline int Log2Ceil(int value) {
  int log2_value = 0;
  while ((1 << log2_value) < value) ++log2_value;
  return log2_value;
}

enum class SoftmaxMode { kSoftmax, kLogSoftmax, kCrossEntropy };

/*
  Hard label cross entropy.
*/
template <typename T, bool IgnoreIndex>
__global__ void CrossEntropyHardLabel(T* loss, const T* softmax,
                                      const int64_t* labels, const int n,
                                      const int dim, const int d,
                                      const int ignore_idx) {
  int64_t ids = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t idx_n = ids / d;
  int64_t idx_d = ids % d;

  // thread ids compute loss[ids] using softmax[idx]
  if (ids < n * d) {
    int64_t idx = idx_n * dim * d + labels[ids] * d + idx_d;
    if (IgnoreIndex == true) {
      // IgnoreIndex is true
      if (labels[ids] == ignore_idx) {
        loss[ids] = static_cast<T>(0.0);
      } else {
        loss[ids] = -Log(softmax[idx]);
      }
    } else {
      // IgnoreIndex is false
      loss[ids] = -Log(softmax[idx]);
    }
  }
}

/*
  Hard label cross entropy with exp.
  Input: log softmax
  Output: loss and exp(input)
*/
template <typename T, bool IgnoreIndex>
__global__ void CrossEntropyExpHardLabel(T* loss, T* softmax,
                                         const int64_t* labels, const int n,
                                         const int dim, const int d,
                                         const int ignore_idx) {
  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t idx_n = idx / (d * dim);
  int64_t idx_dim = (idx / d) % dim;
  int64_t idx_d = idx % d;
  int64_t ids = idx_n * d + idx_d;

  if (idx < n * dim * d) {
    if (IgnoreIndex == true) {
      // IgnoreIndex is true
      if (idx_dim == labels[ids]) {
        if (labels[ids] == ignore_idx) {
          loss[ids] = static_cast<T>(0.0);
        } else {
          loss[ids] = -softmax[idx];
        }
      }
    } else {
      // IgnoreIndex is false
      if (labels[ids] >= 0 && labels[ids] < dim) {
        if (labels[ids] == idx_dim) {
          loss[ids] = -softmax[idx];
        }
      } else {
        loss[ids] = static_cast<T>(0.0);
      }
    }
    softmax[idx] = Exp(softmax[idx]);
  }
}

/*
  Core function of softmax with cross entropy forward
    - softmax, SoftmaxMode=kSoftmax
    - log softmax, SoftmaxMode=kLogSoftmax
    - softmax with cross entropy hard label, SoftmaxMode=kCrossEntropy
  The computation includes
    - Compute max value: maxvalue_{i} = max_j src_{i,j}
    - Compute sum of exp: s_{i} = sum_{j}{e^{src_{i,j} - maxvalue_{i}}}
    - Compute: softmax_{i,j} = e^{src_{i,j} - maxvalue_{i}} / s_{i}
    - Compute: logsoftmax_{i,j} = src_{i,j} - maxvalue_{i} - log(s_{i})
    - Compute: loss_{i} = -logsoftmax[i,label[i]] (Hard label)
  This computation results from following formula:
    softmax_{i,j} = e^{src_{i,j}} / sum_{j}{e^{src_{i,j}}}
                  = e^{src_{i,j} - maxvalue_{i}}
                    / sum_{j}{e^{src_{i,j} - maxvalue_{i}}}
                  = e^{src_{i,j} - maxvalue_{i}} / s_{i}
    logsoftmax_{i,j} = log(softmax_{i,j})
                     = src_{i,j} - maxvalue_{i} - log(s_{i})
  One warp (32 threads) is used to compute 1 or 2 batch (kBatchSize).
  For reduction max (sum), firstly compute max (sum) to one warp, then use
  shuffle api to compute max (sum) in one warp.
*/
template <typename T, typename VecT, typename AccT, int Log2Elements,
          SoftmaxMode mode, bool IgnoreIndex>
__global__ void WarpSoftmaxForwardHardLabel(T* loss, T* softmax, const T* src,
                                            const int64_t* label,
                                            const int batch_size,
                                            const int stride,
                                            const int element_count,
                                            const int ignore_index) {
  constexpr int kDimCeil = 1 << Log2Elements;
  constexpr int kWarpSize = (kDimCeil < 32) ? kDimCeil : 32;
  constexpr int kVSize = sizeof(VecT) / sizeof(T);
  constexpr int kIterations = kDimCeil / kWarpSize;
  constexpr int kIterationsV =
      (kIterations >= kVSize) ? (kIterations / kVSize) : 1;
  constexpr int kBatchSize = (kDimCeil <= 128) ? 2 : 1;

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
// read data to srcdata: - KVSize==1, - KVSize>1
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

  // compute max value: maxvalue_{i} = max_j src_{i,j}
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

  // compute sum: s_{i} = sum_{j}{ exp(src_{i,j} - maxvalue_{i} }
  AccT sum[kBatchSize];
#pragma unroll
  for (int i = 0; i < kBatchSize; ++i) {
    // it = 0
    if (mode == SoftmaxMode::kLogSoftmax ||
        mode == SoftmaxMode::kCrossEntropy) {
      sum[i] = std::exp(srcdata[i][0][0] - max_value[i]);
    } else {
      srcdata[i][0][0] = std::exp(srcdata[i][0][0] - max_value[i]);
      sum[i] = srcdata[i][0][0];
    }
#pragma unroll
    for (int s = 1; s < kVSize; ++s) {
      if (mode == SoftmaxMode::kLogSoftmax ||
          mode == SoftmaxMode::kCrossEntropy) {
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
        if (mode == SoftmaxMode::kLogSoftmax ||
            mode == SoftmaxMode::kCrossEntropy) {
          sum[i] += std::exp(srcdata[i][it][s] - max_value[i]);
        } else {
          srcdata[i][it][s] = std::exp(srcdata[i][it][s] - max_value[i]);
          sum[i] += srcdata[i][it][s];
        }
      }
    }
  }
  WarpReduceSum<AccT, kBatchSize, kWarpSize>(sum);

// write data
#pragma unroll
  for (int i = 0; i < kBatchSize; ++i) {
    if (mode == SoftmaxMode::kLogSoftmax ||
        mode == SoftmaxMode::kCrossEntropy) {
      sum[i] = std::log(sum[i]);
    }

#pragma unroll
    for (int it = 0; it < kIterationsV; ++it) {
      int idx = threadIdx.x + it * kWarpSize;
      if (kVSize == 1) {  // kVSize==1
        if (idx < idx_max_v[i]) {
          if (mode == SoftmaxMode::kLogSoftmax) {  // log softmax
            softmax[(first_batch + i) * stride + idx] =
                srcdata[i][it][0] - max_value[i] - sum[i];
            // softmax with cross entropy hard label
          } else if (mode == SoftmaxMode::kCrossEntropy) {
            AccT logsoftmax = srcdata[i][it][0] - max_value[i] - sum[i];
            // softmax
            softmax[(first_batch + i) * stride + idx] = std::exp(logsoftmax);
            // label
            int loss_idx = (threadIdx.x + it * kWarpSize) * kVSize;
            if (IgnoreIndex == true) {
              // IgnoreIndex is true
              if (label[first_batch + i] == loss_idx) {
                if (label[first_batch + i] != ignore_index) {
                  loss[first_batch + i] = -logsoftmax;
                } else {
                  loss[first_batch + i] = static_cast<T>(0.0);
                }
              }
            } else {
              // IgnoreIndex is false
              if (label[first_batch + i] >= 0 &&
                  label[first_batch + i] < element_count) {
                if (label[first_batch + i] == loss_idx) {
                  loss[first_batch + i] = -logsoftmax;
                }
              } else {
                loss[first_batch + i] = static_cast<T>(0.0);
              }
            }
          } else {  // softmax
            softmax[(first_batch + i) * stride + idx] =
                srcdata[i][it][0] / sum[i];
          }
        } else {
          break;
        }
      } else {  // KVSize>1
        VecT* softmax_v =
            reinterpret_cast<VecT*>(&softmax[(first_batch + i) * stride]);
        VecT tmpdata;
        T* tmpptr = reinterpret_cast<T*>(&tmpdata);
#pragma unroll
        for (int s = 0; s < kVSize; ++s) {
          if (mode == SoftmaxMode::kLogSoftmax) {  // log softmax
            tmpptr[s] = srcdata[i][it][s] - max_value[i] - sum[i];
            // softmax with cross entropy hard label
          } else if (mode == SoftmaxMode::kCrossEntropy) {
            AccT logsoftmax = srcdata[i][it][s] - max_value[i] - sum[i];
            // softmax
            tmpptr[s] = std::exp(logsoftmax);
            // label
            int loss_idx = (threadIdx.x + it * kWarpSize) * kVSize + s;
            if (IgnoreIndex == true) {
              // IgnoreIndex is true
              if (label[first_batch + i] == loss_idx &&
                  label[first_batch + i] != ignore_index) {
                loss[first_batch + i] = -logsoftmax;
              }
            } else {
              // IgnoreIndex is false
              if (label[first_batch + i] >= 0 &&
                  label[first_batch + i] < element_count) {
                if (label[first_batch + i] == loss_idx) {
                  loss[first_batch + i] = -logsoftmax;
                }
              } else {
                loss[first_batch + i] = static_cast<T>(0.0);
              }
            }
          } else {  // softmax
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

#define SOFTMAX_WARP_FORWARD_CASE(Log2Elements, VecT, AccT)                   \
  case Log2Elements:                                                          \
    WarpSoftmaxForwardHardLabel<T, VecT, AccT, Log2Elements, mode,            \
                                IgnoreIndex><<<blocks, threads, 0, stream>>>( \
        loss, softmax, src, label, batch_size, stride, element_count,         \
        ignore_index);                                                        \
    break;

/*
  Wrapper of softmax with cross entropy forward hard label.
*/
template <typename T, SoftmaxMode mode, bool IgnoreIndex>
void SwitchWarpSoftmaxForwardHardLabel(T* loss, T* softmax, const T* src,
                                       const int64_t* label,
                                       const int batch_size, const int stride,
                                       const int element_count,
                                       const int ignore_index,
                                       gpuStream_t stream) {
  using AccT = typename details::MPTypeTrait<T>::Type;

  // use 128 threads per block to maximimize gpu utilization
  const int Log2Elements = static_cast<int>(Log2Ceil(element_count));
  const int kDimCeil = 1 << Log2Elements;
  int kWarpSize = (kDimCeil < 32) ? kDimCeil : 32;
  int batches_per_warp = (kDimCeil <= 128) ? 2 : 1;
  constexpr int threads_per_block = 128;
  int warps_per_block = (threads_per_block / kWarpSize);
  int batches_per_block = warps_per_block * batches_per_warp;
  int blocks = (batch_size + batches_per_block - 1) / batches_per_block;
  dim3 threads(kWarpSize, warps_per_block, 1);

  switch (Log2Elements) {
    SOFTMAX_WARP_FORWARD_CASE(0, T, AccT);
    SOFTMAX_WARP_FORWARD_CASE(1, T, AccT);
    SOFTMAX_WARP_FORWARD_CASE(2, T, AccT);
    SOFTMAX_WARP_FORWARD_CASE(3, T, AccT);
    SOFTMAX_WARP_FORWARD_CASE(4, T, AccT);
    SOFTMAX_WARP_FORWARD_CASE(5, T, AccT);
    SOFTMAX_WARP_FORWARD_CASE(6, T, AccT);
    SOFTMAX_WARP_FORWARD_CASE(7, T, AccT);
    SOFTMAX_WARP_FORWARD_CASE(8, T, AccT);
    SOFTMAX_WARP_FORWARD_CASE(9, T, AccT);
    default:
      break;
  }
}

/*
  Wrapper of softmax with cross entropy hard label.
  - SwitchWarpSoftmaxForwardHardLabel for small size
  - cudnn function for large size
*/
template <typename T, bool IgnoreIndex>
static void SoftmaxWithCrossEntropyHardLabel(
    const platform::CUDADeviceContext& ctx, int rank, int axis,
    const T* logits_data, const int64_t* labels_data, T* loss_data,
    T* softmax_data, int N, int dim, int D, const int ignore_index) {
  auto stream = ctx.stream();
  constexpr int max_dim = 320;
  if (D == 1 && dim <= max_dim) {  // small size
    const SoftmaxMode mode = SoftmaxMode::kCrossEntropy;
    SwitchWarpSoftmaxForwardHardLabel<T, mode, IgnoreIndex>(
        loss_data, softmax_data, logits_data, labels_data, N, dim, dim,
        ignore_index, stream);
  } else {
    ScopedTensorDescriptor desc;
    std::vector<int> tensor_dims = {N, dim, D, 1};
    DataLayout layout = DataLayout::kNCHW;
#ifdef PADDLE_WITH_HIP
    miopenTensorDescriptor_t descp = desc.descriptor<T>(layout, tensor_dims);
#else
    cudnnTensorDescriptor_t descp = desc.descriptor<T>(layout, tensor_dims);
#endif

    auto handle = ctx.cudnn_handle();

#ifdef PADDLE_WITH_HIP
    auto mode = axis == rank - 1 ? MIOPEN_SOFTMAX_MODE_INSTANCE
                                 : MIOPEN_SOFTMAX_MODE_CHANNEL;
    PADDLE_ENFORCE_CUDA_SUCCESS(platform::dynload::miopenSoftmaxForward_V2(
        handle, platform::CudnnDataType<T>::kOne(), descp, logits_data,
        platform::CudnnDataType<T>::kZero(), descp, softmax_data,
        MIOPEN_SOFTMAX_LOG, mode));
#else
    auto mode = axis == rank - 1 ? CUDNN_SOFTMAX_MODE_INSTANCE
                                 : CUDNN_SOFTMAX_MODE_CHANNEL;
    PADDLE_ENFORCE_CUDA_SUCCESS(platform::dynload::cudnnSoftmaxForward(
        handle, CUDNN_SOFTMAX_LOG, mode, platform::CudnnDataType<T>::kOne(),
        descp, logits_data, platform::CudnnDataType<T>::kZero(), descp,
        softmax_data));
#endif
    int threads = 128;
    int blocks = (N * dim * D + threads - 1) / threads;
    // compute cross entropy, input is log softmax
    CrossEntropyExpHardLabel<T, IgnoreIndex><<<blocks, threads, 0, stream>>>(
        loss_data, softmax_data, labels_data, N, dim, D, ignore_index);
  }
}

/*
  Wrapper of softmax with cross entropy grad hard label.
*/
template <typename T>
__global__ void SoftmaxWithCrossEntropyGradHardLabel(
    T* logits_grad, const T* loss_grad, const int64_t* labels, const int64_t n,
    const int64_t dim, const int64_t d, const int ignore_index) {
  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t idx_n = idx / (d * dim);
  int64_t idx_dim = (idx / d) % dim;
  int64_t idx_d = idx % d;
  int64_t ids = idx_n * d + idx_d;

  if (idx < n * dim * d) {
    if (labels[ids] == ignore_index) {
      logits_grad[idx] = static_cast<T>(0.0);
    } else if (labels[ids] == idx_dim) {
      logits_grad[idx] =
          (logits_grad[idx] - static_cast<T>(1.0)) * loss_grad[ids];
    } else {
      logits_grad[idx] *= loss_grad[ids];
    }
  }
}

/*
  Cross entropy soft label with dynamic size on axis (Log2Elements is varibale).
*/
template <typename T, typename VecT, bool InLogMode = false>
__global__ void CrossEntropySoftLabel(T* loss, T* softmaxwrt, const T* softmax,
                                      const T* labels, const int n,
                                      const int dim, const int d,
                                      int Log2Elements) {
  const int kDimCeil = 1 << Log2Elements;
  const int kVSize = sizeof(VecT) / sizeof(T);

  const int kThreadPerBlock = 512;
  const int kBatchPerBlock = 1;
  const int kWarpSize = 32;  // (dim < 32) ? dim : 32;
  const int kBatchSize = 1;
  const int kThreadPerBatch = kThreadPerBlock / kBatchPerBlock;
  const int kWarpPerBatch = kThreadPerBatch / kWarpSize;

  const int kIterations = (dim + kThreadPerBatch - 1) / kThreadPerBatch;
  const int kIterationsV = (kIterations >= kVSize) ? (kIterations / kVSize) : 1;

  const int first_batch = (blockDim.y * blockIdx.x + threadIdx.y) * kBatchSize;

  T sum[kBatchSize]{static_cast<T>(0.0)};
#pragma unroll
  for (int i = 0; i < kBatchSize; ++i) {
    int ids = first_batch + i;
    if (ids >= n * d) break;
    int idx_n = ids / d;
    int idx_d = ids % d;
#pragma unroll
    for (int it = 0; it < kIterations; ++it) {
      int idx_dim = it * kThreadPerBatch + threadIdx.x;
      int idx = idx_n * dim * d + idx_dim * d + idx_d;

      if (idx_n < n && idx_dim < dim) {
        VecT softmaxdata;
        if (InLogMode == true) {
          softmaxdata = reinterpret_cast<VecT*>(&softmaxwrt[idx])[0];
        } else {
          softmaxdata = reinterpret_cast<const VecT*>(&softmax[idx])[0];
        }
        VecT labelsdata = reinterpret_cast<const VecT*>(&labels[idx])[0];
        T* softmaxptr = reinterpret_cast<T*>(&softmaxdata);
        T* labelsptr = reinterpret_cast<T*>(&labelsdata);
#pragma unroll
        for (int s = 0; s < kVSize; s++) {
          if (InLogMode == true) {
            sum[i] -= softmaxptr[s] * labelsptr[s];
            softmaxptr[s] = Exp(softmaxptr[s]);
          } else {
            sum[i] -= Log(softmaxptr[s]) * labelsptr[s];
          }
        }
        if (InLogMode == true) {
          reinterpret_cast<VecT*>(&softmaxwrt[idx])[0] = softmaxdata;
        }
      }
    }
  }
  WarpReduceSum<T, kBatchSize, kWarpSize>(sum);
  __syncthreads();

  __shared__ T sumshare[kWarpPerBatch][kBatchPerBlock][kBatchSize];
  if (threadIdx.x % kWarpSize == 0) {
#pragma unroll
    for (int i = 0; i < kBatchSize; i++) {
      sumshare[threadIdx.x / kWarpSize][threadIdx.y][i] = sum[i];
    }
  }
  __syncthreads();

  // write
  if (threadIdx.x == 0) {
    for (int i = 0; i < kBatchSize; i++) {
      int ids = first_batch + i;
      if (ids < n * d) {
        loss[ids] = sumshare[0][threadIdx.y][i];
        for (int s = 1; s < kWarpPerBatch; s++) {
          loss[ids] += sumshare[s][threadIdx.y][i];
        }
      }
    }
  }
}

/*
Core function of softmax with cross entropy forward soft label.
The computation includes
  - Compute maximum of batch: maxvalue_{i} = max_j src_{i,j}
  - Compute sum of exp batch: s_{i} = sum_{j}{ exp(src_{i,j} - maxvalue_{i} }
  - Compute: sum of - sum_{j}{ label_{i,j} * (src_{i,j} - maxvalue_{i} -
log(sum[i]))}
One warp (32 threads) is used to compute 1 or 2 batch (kBatchSize).
For reduction max (sum), firstly compute max (sum) to one warp, then use shuffle
api to compute max (sum) in one warp.
*/
template <typename T, typename VecT, typename AccT, int Log2Elements>
__global__ void WarpSoftmaxForwardSoftLabel(T* loss, T* softmax, const T* src,
                                            const T* label,
                                            const int batch_size,
                                            const int stride,
                                            const int element_count) {
  const bool LogMode = true;

  constexpr int kDimCeil = 1 << Log2Elements;
  constexpr int kWarpSize = (kDimCeil < 32) ? kDimCeil : 32;
  constexpr int kVSize = sizeof(VecT) / sizeof(T);
  constexpr int kIterations = kDimCeil / kWarpSize;
  constexpr int kIterationsV =
      (kIterations >= kVSize) ? (kIterations / kVSize) : 1;
  constexpr int kBatchSize = (kDimCeil <= 128) ? 2 : 1;

  int first_batch = (blockDim.y * blockIdx.x + threadIdx.y) * kBatchSize;
  int local_batches = batch_size - first_batch;
  if (local_batches > kBatchSize) {
    local_batches = kBatchSize;
  }

  // read data from global memory
  VecT srcdata[kBatchSize][kIterationsV];
  VecT labeldata[kBatchSize][kIterationsV];

  for (int i = 0; i < kBatchSize; ++i) {
    const VecT* src_v =
        reinterpret_cast<const VecT*>(&src[(first_batch + i) * stride]);
    const VecT* label_v =
        reinterpret_cast<const VecT*>(&label[(first_batch + i) * stride]);

    // max index to read
    int idx_max = (i < local_batches) ? element_count : 0;
    int idx_max_v = idx_max / kVSize;

    // read data
    for (int it = 0; it < kIterationsV; ++it) {
      int src_idx = threadIdx.x + it * kWarpSize;
      if (src_idx < idx_max_v) {
        srcdata[i][it] = src_v[src_idx];
        labeldata[i][it] = label_v[src_idx];
      } else {
#pragma unroll
        for (int s = 0; s < kVSize; s++) {
          reinterpret_cast<T*>(&srcdata[i][it])[s] =
              -std::numeric_limits<AccT>::max();
          reinterpret_cast<T*>(&labeldata[i][it])[s] = 0.0;
        }
      }
    }
  }

  // compute max value
  AccT max_value[kBatchSize];
#pragma unroll
  for (int i = 0; i < kBatchSize; ++i) {
    max_value[i] = -std::numeric_limits<AccT>::infinity();
#pragma unroll
    for (int it = 0; it < kIterationsV; ++it) {
      T* srcptr_v = reinterpret_cast<T*>(&srcdata[i][it]);
      T valmax = srcptr_v[0];
#pragma unroll
      for (int s = 1; s < kVSize; ++s) {
        valmax = (valmax > srcptr_v[s]) ? valmax : srcptr_v[s];
      }
      max_value[i] = (max_value[i] > static_cast<AccT>(valmax))
                         ? max_value[i]
                         : static_cast<AccT>(valmax);
    }
  }
  WarpReduceMax<AccT, kBatchSize, kWarpSize>(max_value);

  // compute sum
  AccT sum[kBatchSize]{0.0};
#pragma unroll
  for (int i = 0; i < kBatchSize; ++i) {
#pragma unroll
    for (int it = 0; it < kIterationsV; ++it) {
      T* srcptr_v = reinterpret_cast<T*>(&srcdata[i][it]);
#pragma unroll
      for (int s = 0; s < kVSize; ++s) {
        if (LogMode) {
          sum[i] += std::exp(static_cast<AccT>(srcptr_v[s]) - max_value[i]);
        } else {
          srcptr_v[s] = std::exp(static_cast<AccT>(srcptr_v[s]) - max_value[i]);
          sum[i] += static_cast<AccT>(srcptr_v[s]);
        }
      }
    }
  }
  WarpReduceSum<AccT, kBatchSize, kWarpSize>(sum);

  // log_softmax and loss
  AccT sumloss[kBatchSize]{0.0};
#pragma unroll
  for (int i = 0; i < kBatchSize; ++i) {
    if (i >= local_batches) break;

    VecT* softmax_v =
        reinterpret_cast<VecT*>(&softmax[(first_batch + i) * stride]);

    // max index to write
    int idx_max = (i < local_batches) ? element_count : 0;
    int idx_max_v = idx_max / kVSize;

    if (LogMode) {
      sum[i] = std::log(sum[i]);
    }
#pragma unroll
    for (int it = 0; it < kIterationsV; ++it) {
      T* srcvp = reinterpret_cast<T*>(&srcdata[i][it]);
      T* labelvp = reinterpret_cast<T*>(&labeldata[i][it]);
      VecT tmpv;
      T* tmpvp = reinterpret_cast<T*>(&tmpv);
#pragma unroll
      for (int s = 0; s < kVSize; ++s) {
        if (LogMode) {
          AccT logsoftmax = static_cast<AccT>(srcvp[s]) - max_value[i] - sum[i];
          sumloss[i] -= logsoftmax * static_cast<AccT>(labelvp[s]);
          tmpvp[s] = std::exp(logsoftmax);
        } else {
          tmpvp[s] = static_cast<AccT>(srcvp[s]) / sum[i];
        }
      }

      int idx = threadIdx.x + it * kWarpSize;
      if (idx < idx_max_v) {
        softmax_v[idx] = tmpv;
      }
    }
  }

  // loss
  WarpReduceSum<AccT, kBatchSize, kWarpSize>(sumloss);

  for (int i = 0; i < kBatchSize; i++) {
    if (i >= local_batches) break;
    loss[first_batch + i] = sumloss[i];
  }
}

#define SOFTMAX_WARP_FORWARD_SOFT_CASE(Log2Elements, VecT, AccT)               \
  case Log2Elements:                                                           \
    WarpSoftmaxForwardSoftLabel<T, VecT, AccT,                                 \
                                Log2Elements><<<blocks, threads, 0, stream>>>( \
        loss, softmax, src, label, batch_size, stride, element_count);         \
    break;

/*
  Wrapper of softmax with cross entropy forward soft label.
*/
template <typename T>
void SwitchWarpSoftmaxForwardSoftLabel(const int blocks, const dim3 threads,
                                       gpuStream_t stream, T* loss, T* softmax,
                                       const T* src, const T* label,
                                       const int batch_size, const int stride,
                                       const int element_count,
                                       const int Log2Elements) {
  using AccT = typename details::MPTypeTrait<T>::Type;
  switch (Log2Elements) {
    SOFTMAX_WARP_FORWARD_SOFT_CASE(0, T, AccT);
    SOFTMAX_WARP_FORWARD_SOFT_CASE(1, T, AccT);
    SOFTMAX_WARP_FORWARD_SOFT_CASE(2, T, AccT);
    SOFTMAX_WARP_FORWARD_SOFT_CASE(3, T, AccT);
    SOFTMAX_WARP_FORWARD_SOFT_CASE(4, T, AccT);
    SOFTMAX_WARP_FORWARD_SOFT_CASE(5, T, AccT);
    SOFTMAX_WARP_FORWARD_SOFT_CASE(6, T, AccT);
    SOFTMAX_WARP_FORWARD_SOFT_CASE(7, T, AccT);
    SOFTMAX_WARP_FORWARD_SOFT_CASE(8, T, AccT);
    SOFTMAX_WARP_FORWARD_SOFT_CASE(9, T, AccT);
    default:
      break;
  }
}

template <typename T>
static void SoftmaxWithCrossEntropySoftLabel(
    const platform::CUDADeviceContext& ctx, const int rank, const int axis,
    const T* logits_data, const T* labels_data, T* softmax_data, T* loss_data,
    int N, int dim, int D) {
#ifdef __HIPCC__
  constexpr int kMaxBlockDim = 256;
#else
  constexpr int kMaxBlockDim = 512;
#endif
  int64_t block_dim = dim >= kMaxBlockDim
                          ? kMaxBlockDim
                          : (1 << static_cast<int>(std::log2(dim)));

  int64_t grid_dim = N * D;
  constexpr int max_dim = 320;

  const int kDimLog2 = static_cast<int>(Log2Ceil(dim));
  const int kDimCeil = 1 << kDimLog2;
  auto stream = ctx.stream();

  if (D == 1 && dim <= max_dim) {
    int kWarpSize = (kDimCeil < 32) ? kDimCeil : 32;
    int batches_per_warp = (kDimCeil <= 128) ? 2 : 1;

    // use 128 threads per block to maximimize gpu utilization
    constexpr int threads_per_block = 128;
    int warps_per_block = (threads_per_block / kWarpSize);
    int batches_per_block = warps_per_block * batches_per_warp;
    int blocks = (N + batches_per_block - 1) / batches_per_block;
    dim3 threads(kWarpSize, warps_per_block, 1);

    SwitchWarpSoftmaxForwardSoftLabel<T>(blocks, threads, stream, loss_data,
                                         softmax_data, logits_data, labels_data,
                                         N, dim, dim, kDimLog2);

  } else {
    ScopedTensorDescriptor desc;
    std::vector<int> tensor_dims = {N, dim, D, 1};
    DataLayout layout = DataLayout::kNCHW;
#ifdef PADDLE_WITH_HIP
    miopenTensorDescriptor_t descp = desc.descriptor<T>(layout, tensor_dims);
#else
    cudnnTensorDescriptor_t descp = desc.descriptor<T>(layout, tensor_dims);
#endif

    auto handle = ctx.cudnn_handle();

#ifdef PADDLE_WITH_HIP
    auto mode = axis == rank - 1 ? MIOPEN_SOFTMAX_MODE_INSTANCE
                                 : MIOPEN_SOFTMAX_MODE_CHANNEL;
    PADDLE_ENFORCE_CUDA_SUCCESS(platform::dynload::miopenSoftmaxForward_V2(
        handle, platform::CudnnDataType<T>::kOne(), descp, logits_data,
        platform::CudnnDataType<T>::kZero(), descp, softmax_data,
        MIOPEN_SOFTMAX_LOG, mode));
#else
    auto mode = axis == rank - 1 ? CUDNN_SOFTMAX_MODE_INSTANCE
                                 : CUDNN_SOFTMAX_MODE_CHANNEL;
    PADDLE_ENFORCE_CUDA_SUCCESS(platform::dynload::cudnnSoftmaxForward(
        handle, CUDNN_SOFTMAX_LOG, mode, platform::CudnnDataType<T>::kOne(),
        descp, logits_data, platform::CudnnDataType<T>::kZero(), descp,
        softmax_data));
#endif

    const int kDimLog2 = static_cast<int>(Log2Ceil(dim));
    const int kDimCeil = 1 << kDimLog2;

    int kThreadPerBlock = 512;
    int kBatchPerBlock = 1;
    int blocks = (N * D + kBatchPerBlock - 1) / kBatchPerBlock;
    dim3 threads(kThreadPerBlock / kBatchPerBlock, kBatchPerBlock, 1);

    CrossEntropySoftLabel<T, T, true><<<blocks, threads, 0, stream>>>(
        loss_data, softmax_data, NULL, labels_data, N, dim, D, kDimLog2);
  }
}

template <typename T>
__global__ void SoftCrossEntropyGradientKernel(T* logit_grad,
                                               const T* loss_grad,
                                               const T* labels, const int64_t n,
                                               const int64_t d,
                                               const int64_t remain) {
  int64_t ids = blockIdx.x * blockDim.x + threadIdx.x;
  if (ids < n * d) {
    int64_t idx_n = ids / d;
    int64_t idx_remain = ids % remain;
    int64_t idx_loss = idx_n * remain + idx_remain;
    logit_grad[ids] = loss_grad[idx_loss] * (logit_grad[ids] - labels[ids]);
  }
}

template <typename T>
__global__ void SoftLabelCrossEntropyGradientKernel(T* logit_grad,
                                                    const T* loss_grad,
                                                    const T* labels,
                                                    const int n, const int d,
                                                    const int remain) {
  int ids = blockIdx.x * blockDim.x + threadIdx.x;
  if (ids < n * d) {
    int idx_n = ids / d;
    int idx_remain = ids % remain;
    int idx_loss = idx_n * remain + idx_remain;
    logit_grad[ids] = loss_grad[idx_loss] * (-labels[ids] / logit_grad[ids]);
  }
}

template <typename T>
__global__ void HardLabelCrossEntropyGradientKernel(T* logit_grad,
                                                    const int64_t* labels,
                                                    const int n, const int d,
                                                    const int remain,
                                                    const int ignore_index) {
  CUDA_KERNEL_LOOP(index, n * remain) {
    int idx_n = index / remain;
    int idx_remain = index % remain;
    int tmp = labels[index];
    int idx = idx_n * d + tmp * remain + idx_remain;
    if (ignore_index != tmp) {
      logit_grad[idx] = -static_cast<T>(1.) / logit_grad[idx];
    }
  }
}

template <typename T>
__global__ void ScaleCrossEntropyGradient(T* logit_grad, const T* loss_grad,
                                          const int num, const int d,
                                          const int remain,
                                          const int64_t* labels,
                                          const int ignore_index) {
  CUDA_KERNEL_LOOP(index, num) {
    int idx_n = index / d;
    int idx_remain = index % remain;
    int idx_lbl = idx_n * remain + idx_remain;
    int k = (index % d) / remain;
    if (labels[idx_lbl] == ignore_index || labels[idx_lbl] != k) {
      logit_grad[index] = static_cast<T>(0.);
    } else {
      logit_grad[index] *= loss_grad[idx_lbl];
    }
  }
}

template <typename T>
class SoftmaxWithCrossEntropyCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    PADDLE_ENFORCE_EQ(
        platform::is_gpu_place(context.GetPlace()), true,
        platform::errors::Unavailable("softmax_with_cross_entropy operator's "
                                      "CUDA kernel only runs on GPU device."));
    const bool use_softmax = context.Attr<bool>("use_softmax");

    // do not with softmax op, and input is softmax
    if (!use_softmax) {
      const Tensor* softmax = context.Input<Tensor>("Logits");
      const Tensor* labels = context.Input<Tensor>("Label");
      Tensor* softmax_out = context.Output<Tensor>("Softmax");
      Tensor* loss = context.Output<Tensor>("Loss");

      const int rank = softmax->dims().size();
      const int axis = CanonicalAxis(context.Attr<int>("axis"), rank);
      const int axis_dim = softmax->dims()[axis];

      const int n = SizeToAxis(axis, softmax->dims());
      const int d = SizeFromAxis(axis, softmax->dims());

      auto* softmax_out_data = softmax_out->mutable_data<T>(context.GetPlace());
      auto* loss_data = loss->mutable_data<T>(context.GetPlace());

      math::SetConstant<platform::CUDADeviceContext, T> set_constant;
      set_constant(context.cuda_device_context(), loss, static_cast<T>(0));
      if (axis_dim == 1) {
        set_constant(context.cuda_device_context(), softmax_out,
                     static_cast<T>(1));
        return;
      }

      auto soft_label = context.Attr<bool>("soft_label");
      auto ignore_index = context.Attr<int>("ignore_index");

      Tensor softmax_2d, labels_2d, loss_2d, softmax_out_2d;
      softmax_2d.ShareDataWith(*softmax).Resize({n, d});
      labels_2d.ShareDataWith(*labels).Resize({n, labels->numel() / n});
      loss_2d.ShareDataWith(*loss).Resize({n, 1});
      softmax_out_2d.ShareDataWith(*softmax_out).Resize({n, d});

      // math::CrossEntropyFunctor support axis is the last
      if (axis == -1) {
        math::CrossEntropyFunctor<platform::CUDADeviceContext, T>()(
            context.cuda_device_context(), &loss_2d, &softmax_2d, &labels_2d,
            soft_label, ignore_index, axis_dim);
        return;
      }

      // if axis is not the last, we need a new impliment
      if (soft_label) {
        auto* logits_data = softmax->data<T>();
        auto* labels_data = labels->data<T>();

        const int kDimLog2 = static_cast<int>(Log2Ceil(axis_dim));
        const int kDimCeil = 1 << kDimLog2;
        int kThreadPerBlock = 512;
        int kBatchPerBlock = 1;
        int blocks = (n * d + kBatchPerBlock - 1) / kBatchPerBlock;
        dim3 threads(kThreadPerBlock / kBatchPerBlock, kBatchPerBlock, 1);

        CrossEntropySoftLabel<T, T, false><<<
            blocks, threads, 0, context.cuda_device_context().stream()>>>(
            loss_data, NULL, logits_data, labels_data, n, axis_dim,
            d / axis_dim, kDimLog2);
      } else {  // HardLabel
        auto* logits_data = softmax->data<T>();
        auto* labels_data = labels->data<int64_t>();
        int threads = 128;
        int blocks = (n * d / axis_dim + threads - 1) / threads;
        if (ignore_index >= 0 && ignore_index < axis_dim) {
          CrossEntropyHardLabel<T, true><<<
              blocks, threads, 0, context.cuda_device_context().stream()>>>(
              loss_data, logits_data, labels_data, n, axis_dim, d / axis_dim,
              ignore_index);
        } else {
          CrossEntropyHardLabel<T, false><<<
              blocks, threads, 0, context.cuda_device_context().stream()>>>(
              loss_data, logits_data, labels_data, n, axis_dim, d / axis_dim,
              ignore_index);
        }
      }

      // cause of input is softmax
      // copy to output softmax, directly
      framework::TensorCopy(*softmax, context.GetPlace(),
                            context.device_context(), softmax_out);

      return;
    }

    const Tensor* logits = context.Input<Tensor>("Logits");
    const Tensor* labels = context.Input<Tensor>("Label");
    Tensor* softmax = context.Output<Tensor>("Softmax");
    Tensor* loss = context.Output<Tensor>("Loss");

    const int rank = logits->dims().size();
    const int axis = CanonicalAxis(context.Attr<int>("axis"), rank);
    int axis_dim = logits->dims()[axis];

    const int64_t n = SizeToAxis(axis, logits->dims());
    const int64_t d = SizeFromAxis(axis, logits->dims());

    auto* softmax_data = softmax->mutable_data<T>(context.GetPlace());
    auto* loss_data = loss->mutable_data<T>(context.GetPlace());

    if (axis_dim == 1) {
      math::SetConstant<platform::CUDADeviceContext, T> set_constant;
      set_constant(context.cuda_device_context(), softmax, static_cast<T>(1));
      set_constant(context.cuda_device_context(), loss, static_cast<T>(0));
      return;
    }

    auto soft_label = context.Attr<bool>("soft_label");
    auto ignore_index = context.Attr<int>("ignore_index");

    if (soft_label) {
      auto* logits_data = logits->data<T>();
      auto* labels_data = labels->data<T>();
      SoftmaxWithCrossEntropySoftLabel<T>(
          context.cuda_device_context(), rank, axis, logits_data, labels_data,
          softmax_data, loss_data, n, axis_dim, d / axis_dim);
    } else {
      if (!context.Attr<bool>("numeric_stable_mode")) {
        // CUDNN kernel only suppoer 2-D tensor and perfome softmax on last dim
        Tensor logits_2d, softmax_2d, labels_2d, loss_2d;
        logits_2d.ShareDataWith(*logits).Resize({n, d});
        softmax_2d.ShareDataWith(*softmax).Resize({n, d});
        labels_2d.ShareDataWith(*labels).Resize({n, labels->numel() / n});
        loss_2d.ShareDataWith(*loss).Resize({n, 1});
        math::SoftmaxCUDNNFunctor<T>()(context.cuda_device_context(),
                                       &logits_2d, &softmax_2d);
        math::CrossEntropyFunctor<platform::CUDADeviceContext, T>()(
            context.cuda_device_context(), &loss_2d, &softmax_2d, &labels_2d,
            false, ignore_index, axis_dim);
      } else {
        auto* logits_data = logits->data<T>();
        auto* labels_data = labels->data<int64_t>();
        if (ignore_index >= 0 && ignore_index < axis_dim) {
          SoftmaxWithCrossEntropyHardLabel<T, true>(
              context.cuda_device_context(), rank, axis, logits_data,
              labels_data, loss_data, softmax_data, n, axis_dim, d / axis_dim,
              ignore_index);
        } else {
          SoftmaxWithCrossEntropyHardLabel<T, false>(
              context.cuda_device_context(), rank, axis, logits_data,
              labels_data, loss_data, softmax_data, n, axis_dim, d / axis_dim,
              ignore_index);
        }
      }
    }
  }
};

template <typename T>
class SoftmaxWithCrossEntropyGradCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    PADDLE_ENFORCE_EQ(
        platform::is_gpu_place(context.GetPlace()), true,
        platform::errors::Unavailable("softmax_with_cross_entropy operator's "
                                      "CUDA kernel only runs on GPU device."));
    const Tensor* labels = context.Input<Tensor>("Label");
    const T* loss_grad_data =
        context.Input<Tensor>(framework::GradVarName("Loss"))->data<T>();
    Tensor* logit_grad =
        context.Output<Tensor>(framework::GradVarName("Logits"));
    const Tensor* softmax = context.Input<Tensor>("Softmax");
    if (logit_grad != softmax) {
      framework::TensorCopy(*softmax, context.GetPlace(),
                            context.device_context(), logit_grad);
    }
    T* logit_grad_data = logit_grad->data<T>();

    const int rank = logit_grad->dims().size();
    const int axis = CanonicalAxis(context.Attr<int>("axis"), rank);
    int axis_dim = logit_grad->dims()[axis];

    const int64_t n = SizeToAxis(axis, logit_grad->dims());
    const int64_t d = SizeFromAxis(axis, logit_grad->dims());
    const int64_t remain = d / axis_dim;

    int block = 512;
    auto stream = context.cuda_device_context().stream();
    auto ignore_index = context.Attr<int>("ignore_index");
    auto use_softmax = context.Attr<bool>("use_softmax");

    // do not with softmax op, and input is softmax
    if (!use_softmax) {
      if (context.Attr<bool>("soft_label")) {
        int grid = (n * d + block - 1) / block;
        const T* label_data = labels->data<T>();
        SoftLabelCrossEntropyGradientKernel<T><<<grid, block, 0, stream>>>(
            logit_grad_data, loss_grad_data, label_data, n, d, remain);
      } else {
        Tensor logits_grad_2d;
        logits_grad_2d.ShareDataWith(*logit_grad).Resize({n, d});
        int grid = (n * remain + block - 1) / block;
        const int64_t* label_data = labels->data<int64_t>();
        HardLabelCrossEntropyGradientKernel<T><<<grid, block, 0, stream>>>(
            logit_grad_data, label_data, n, d, remain, ignore_index);
        int num = n * d;
        grid = (num + block - 1) / block;
        ScaleCrossEntropyGradient<T><<<grid, block, 0, stream>>>(
            logit_grad_data, loss_grad_data, num, d, remain, label_data,
            ignore_index);
      }

      return;
    }

    // with softmax, continue

    if (context.Attr<bool>("soft_label")) {
      int64_t grid = (n * d + block - 1) / block;
      const T* label_data = labels->data<T>();
      SoftCrossEntropyGradientKernel<T><<<grid, block, 0, stream>>>(
          logit_grad_data, loss_grad_data, label_data, n, d, remain);
    } else {
      const int64_t* label_data = labels->data<int64_t>();
      int grid = (n * d + block - 1) / block;
      SoftmaxWithCrossEntropyGradHardLabel<T><<<grid, block, 0, stream>>>(
          logit_grad_data, loss_grad_data, label_data, n, d / remain, remain,
          ignore_index);
    }
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
#ifdef PADDLE_WITH_HIP
// MIOPEN do not support double
REGISTER_OP_CUDA_KERNEL(
    softmax_with_cross_entropy, ops::SoftmaxWithCrossEntropyCUDAKernel<float>,
    ops::SoftmaxWithCrossEntropyCUDAKernel<paddle::platform::float16>);
REGISTER_OP_CUDA_KERNEL(
    softmax_with_cross_entropy_grad,
    ops::SoftmaxWithCrossEntropyGradCUDAKernel<float>,
    ops::SoftmaxWithCrossEntropyGradCUDAKernel<paddle::platform::float16>);
#else
REGISTER_OP_CUDA_KERNEL(
    softmax_with_cross_entropy, ops::SoftmaxWithCrossEntropyCUDAKernel<float>,
    ops::SoftmaxWithCrossEntropyCUDAKernel<paddle::platform::float16>,
    ops::SoftmaxWithCrossEntropyCUDAKernel<double>);
REGISTER_OP_CUDA_KERNEL(
    softmax_with_cross_entropy_grad,
    ops::SoftmaxWithCrossEntropyGradCUDAKernel<float>,
    ops::SoftmaxWithCrossEntropyGradCUDAKernel<paddle::platform::float16>,
    ops::SoftmaxWithCrossEntropyGradCUDAKernel<double>);
#endif