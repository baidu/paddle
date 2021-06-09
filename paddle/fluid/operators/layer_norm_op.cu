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
#include <memory>
#include <vector>

#include "paddle/fluid/framework/ddim.h"
#include "paddle/fluid/operators/layer_norm_op.h"
#include "paddle/fluid/platform/float16.h"
#ifdef PADDLE_WITH_CUDA
#include "paddle/fluid/platform/cudnn_helper.h"
#endif
#ifdef PADDLE_WITH_HIP
#include "paddle/fluid/platform/miopen_helper.h"
#endif

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
using DataLayout = framework::DataLayout;
template <typename T>
using CudnnDataType = platform::CudnnDataType<T>;
template <typename T>
using LayerNormParamType = typename CudnnDataType<T>::BatchNormParamType;

inline static int GetDesiredBlockDim(int block_dim) {
#ifdef __HIPCC__
  const int kMaxBlockDim = 256;
#else
  const int kMaxBlockDim = 512;
#endif
  return block_dim >= kMaxBlockDim
             ? kMaxBlockDim
             : (1 << (static_cast<int>(std::log2f(block_dim))));
}

#define FIXED_BLOCK_DIM_CASE_BASE(log2_block_dim, ...)  \
  case (1 << (log2_block_dim)): {                       \
    constexpr auto kBlockDim = (1 << (log2_block_dim)); \
    __VA_ARGS__;                                        \
  } break

#define FIXED_BLOCK_DIM_CASE(...)              \
  FIXED_BLOCK_DIM_CASE_BASE(9, ##__VA_ARGS__); \
  FIXED_BLOCK_DIM_CASE_BASE(8, ##__VA_ARGS__); \
  FIXED_BLOCK_DIM_CASE_BASE(7, ##__VA_ARGS__); \
  FIXED_BLOCK_DIM_CASE_BASE(6, ##__VA_ARGS__); \
  FIXED_BLOCK_DIM_CASE_BASE(5, ##__VA_ARGS__); \
  FIXED_BLOCK_DIM_CASE_BASE(4, ##__VA_ARGS__); \
  FIXED_BLOCK_DIM_CASE_BASE(3, ##__VA_ARGS__); \
  FIXED_BLOCK_DIM_CASE_BASE(2, ##__VA_ARGS__); \
  FIXED_BLOCK_DIM_CASE_BASE(1, ##__VA_ARGS__)

#define FIXED_BLOCK_DIM_FIXED_BLOCK_NUM_CASE_BASE(                             \
    log2_block_dim, feature_size, kMaxBlockNum, ...)                           \
  case (1 << (log2_block_dim)): {                                              \
    for (int i = 0; i < std::ceil(feature_size / (1.0 * kMaxBlockNum)); i++) { \
      int col_offset = i * kMaxBlockNum;                                       \
      int block_num = std::min(feature_size - col_offset, kMaxBlockNum);       \
      constexpr auto kBlockDim = (1 << (log2_block_dim));                      \
      __VA_ARGS__;                                                             \
    }                                                                          \
  } break

#define FIXED_BLOCK_DIM_FIXED_BLOCK_NUM_CASE(feature_size, kMaxBlockNum, ...) \
  FIXED_BLOCK_DIM_FIXED_BLOCK_NUM_CASE_BASE(9, feature_size, kMaxBlockNum,    \
                                            ##__VA_ARGS__);                   \
  FIXED_BLOCK_DIM_FIXED_BLOCK_NUM_CASE_BASE(8, feature_size, kMaxBlockNum,    \
                                            ##__VA_ARGS__);                   \
  FIXED_BLOCK_DIM_FIXED_BLOCK_NUM_CASE_BASE(7, feature_size, kMaxBlockNum,    \
                                            ##__VA_ARGS__);                   \
  FIXED_BLOCK_DIM_FIXED_BLOCK_NUM_CASE_BASE(6, feature_size, kMaxBlockNum,    \
                                            ##__VA_ARGS__);                   \
  FIXED_BLOCK_DIM_FIXED_BLOCK_NUM_CASE_BASE(5, feature_size, kMaxBlockNum,    \
                                            ##__VA_ARGS__);                   \
  FIXED_BLOCK_DIM_FIXED_BLOCK_NUM_CASE_BASE(4, feature_size, kMaxBlockNum,    \
                                            ##__VA_ARGS__);                   \
  FIXED_BLOCK_DIM_FIXED_BLOCK_NUM_CASE_BASE(3, feature_size, kMaxBlockNum,    \
                                            ##__VA_ARGS__);                   \
  FIXED_BLOCK_DIM_FIXED_BLOCK_NUM_CASE_BASE(2, feature_size, kMaxBlockNum,    \
                                            ##__VA_ARGS__);                   \
  FIXED_BLOCK_DIM_FIXED_BLOCK_NUM_CASE_BASE(1, feature_size, kMaxBlockNum,    \
                                            ##__VA_ARGS__)

static __device__ __forceinline__ float real_sqrt(float x) { return sqrtf(x); }
static __device__ __forceinline__ double real_sqrt(double x) { return sqrt(x); }

template <typename T>
struct PairForLayerNorm {
  __device__ __forceinline__ PairForLayerNorm() {}
  __device__ __forceinline__ PairForLayerNorm(const T &first, const T &second)
      : first_(first), second_(second) {}

  T first_;
  T second_;
};

template <typename T>
struct PairForLayerNormAddFunctor {
  __device__ __forceinline__ PairForLayerNorm<T> operator()(
      const PairForLayerNorm<T> &p1, const PairForLayerNorm<T> &p2) {
    return PairForLayerNorm<T>(p1.first_ + p2.first_, p1.second_ + p2.second_);
  }
};

template <typename T>
__inline__ __device__ T rsqrt_(const T val) {
  return static_cast<T>(1) / sqrt(val);
}

template <>
__inline__ __device__ float rsqrt_(const float val) {
  return rsqrtf(val);
}

template <>
__inline__ __device__ double rsqrt_(const double val) {
  return rsqrt(val);
}

#if CUDA_ARCH_FP16_SUPPORTED(__CUDA_ARCH__)
template <>
__inline__ __device__ half rsqrt_(const half val) {
  return hrsqrt(val);
}
#endif

template <typename T, typename U, int BlockDim>
__global__ void LayerNormForward(const T *x, const U *scale, const U *bias,
                                 T *y, U *mean, U *var, float epsilon,
                                 int feature_size) {
  using BlockReduce = cub::BlockReduce<PairForLayerNorm<double>, BlockDim>;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  __shared__ U mean_share;
  __shared__ U var_share;

  int beg_idx = blockIdx.x * feature_size + threadIdx.x;
  int end_idx = (blockIdx.x + 1) * feature_size;

  // Step 1: Reduce to calculate mean and var
  double mean_val = 0;
  double var_val = 0;
  for (int i = beg_idx; i < end_idx; i += BlockDim) {
    U tmp = static_cast<U>(x[i]);
    mean_val += tmp;
    var_val += (tmp * tmp);
  }
  auto pair = BlockReduce(temp_storage)
                  .Reduce(PairForLayerNorm<double>(mean_val, var_val),
                          PairForLayerNormAddFunctor<double>());
  if (threadIdx.x == 0) {
    auto tmp = pair.first_ / feature_size;
    mean[blockIdx.x] = mean_share = static_cast<U>(tmp);
    var[blockIdx.x] = var_share =
        static_cast<U>(pair.second_ / feature_size - tmp * tmp);
  }
  __syncthreads();

  mean_val = mean_share;
  U invvar = rsqrt_<U>(var_share + static_cast<U>(epsilon));

  // Step 2: Calculate y
  if (scale != nullptr) {
    if (bias != nullptr) {
      for (int i = beg_idx, j = threadIdx.x; i < end_idx;
           i += BlockDim, j += BlockDim) {
        y[i] = static_cast<T>(
            scale[j] * (static_cast<U>(x[i]) - mean_val) * invvar + bias[j]);
      }
    } else {
      for (int i = beg_idx, j = threadIdx.x; i < end_idx;
           i += BlockDim, j += BlockDim) {
        y[i] = static_cast<T>(scale[j] * (static_cast<U>(x[i]) - mean_val) *
                              invvar);
      }
    }
  } else {  // scale == nullptr
    if (bias != nullptr) {
      for (int i = beg_idx, j = threadIdx.x; i < end_idx;
           i += BlockDim, j += BlockDim) {
        y[i] = static_cast<T>((static_cast<U>(x[i]) - mean_val) * invvar +
                              bias[j]);
      }
    } else {
      for (int i = beg_idx, j = threadIdx.x; i < end_idx;
           i += BlockDim, j += BlockDim) {
        y[i] = static_cast<T>((static_cast<U>(x[i]) - mean_val) * invvar);
      }
    }
  }
}

template <typename T, typename U, int BlockDim>
__global__ void LayerNormForwardFP16(const T *x, const U *scale, const U *bias,
                                     T *y, U *mean, U *var, float epsilon,
                                     int feature_size) {
#if CUDA_ARCH_FP16_SUPPORTED(__CUDA_ARCH__)
  using BlockReduce = cub::BlockReduce<PairForLayerNorm<U>, BlockDim>;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  __shared__ U mean_share;
  __shared__ U var_share;

  int beg_idx = blockIdx.x * feature_size + threadIdx.x;
  int end_idx = (blockIdx.x + 1) * feature_size;

  // Step 1: Reduce to calculate mean and var
  U mean_val = 0;
  U var_val = 0;
  for (int i = beg_idx; i < end_idx; i += BlockDim) {
    U tmp = static_cast<U>(x[i]);
    mean_val += tmp;
    var_val += (tmp * tmp);
  }
  auto pair = BlockReduce(temp_storage)
                  .Reduce(PairForLayerNorm<U>(mean_val, var_val),
                          PairForLayerNormAddFunctor<U>());
  if (threadIdx.x == 0) {
    auto tmp = pair.first_ / static_cast<U>(feature_size);
    mean[blockIdx.x] = mean_share = static_cast<U>(tmp);
    var[blockIdx.x] = var_share =
        static_cast<U>(pair.second_ / static_cast<U>(feature_size) - tmp * tmp);
  }
  __syncthreads();

  mean_val = mean_share;
  U invvar = rsqrt_<U>(var_share + static_cast<U>(epsilon));

  // Step 2: Calculate y
  if (scale != nullptr) {
    if (bias != nullptr) {
      for (int i = beg_idx, j = threadIdx.x; i < end_idx;
           i += BlockDim, j += BlockDim) {
        y[i] = static_cast<T>(
            scale[j] * (static_cast<U>(x[i]) - mean_val) * invvar + bias[j]);
      }
    } else {
      for (int i = beg_idx, j = threadIdx.x; i < end_idx;
           i += BlockDim, j += BlockDim) {
        y[i] = static_cast<T>(scale[j] * (static_cast<U>(x[i]) - mean_val) *
                              invvar);
      }
    }
  } else {  // scale == nullptr
    if (bias != nullptr) {
      for (int i = beg_idx, j = threadIdx.x; i < end_idx;
           i += BlockDim, j += BlockDim) {
        y[i] = static_cast<T>((static_cast<U>(x[i]) - mean_val) * invvar +
                              bias[j]);
      }
    } else {
      for (int i = beg_idx, j = threadIdx.x; i < end_idx;
           i += BlockDim, j += BlockDim) {
        y[i] = static_cast<T>((static_cast<U>(x[i]) - mean_val) * invvar);
      }
    }
  }
#endif
}

template <typename T, typename U, int VPT>
__inline__ __device__ void cuLoadAddStridedInputs(
    const int i1_block, const int thr_load_row_off, const int thr_load_col_off,
    const int i2_off, const int row_stride, U *warp_buf1, U *warp_buf2,
    const T *input, const T *dout, const int i1_end, const int n2,
    const U *__restrict__ mean, const U *__restrict__ var,
    const float epsilon) {
  const int i1 = i1_block + thr_load_row_off;
  if (i1 >= i1_end) return;
  U curr_mean = mean[i1];
  U curr_invvar = rsqrt_<U>(var[i1] + epsilon);
  for (int k = 0; k < VPT; ++k) {
    const int i2 = i2_off + k;
    const int load_idx = i1 * n2 + i2;
    const int write_idx = thr_load_row_off * row_stride + thr_load_col_off + k;
    if (i2 < n2) {
      U curr_input = static_cast<U>(input[load_idx]);
      U curr_dout = static_cast<U>(dout[load_idx]);
      warp_buf1[write_idx] += curr_dout;
      warp_buf2[write_idx] +=
          curr_dout * (curr_input - curr_mean) * curr_invvar;
    }
  }
}

template <typename T, typename U, int BDIMX, int BDIMY, int VPTX>
__global__ void LayerNormBackwardPartGradGammaBeta(
    const T *__restrict__ dout, const T *__restrict__ input, const int n1,
    const int n2, const U *__restrict__ mean, const U *__restrict__ var,
    float epsilon, U *part_grad_gamma, U *part_grad_beta) {
  // VPTX -> value per thread.x, BDIMX -> blockDim.x, BDIMY -> blockDim.y, BDIMX
  // -> blockDim.x
  // template for compile time optimizations

  constexpr int row_stride = BDIMX + 1;
  const int thr_load_col_off = (threadIdx.x * VPTX) & (BDIMX - 1);
  const int thr_load_row_off =
      (threadIdx.x * VPTX) / BDIMX + threadIdx.y * BDIMY;
  const int i2_off = blockIdx.x * BDIMX + thr_load_col_off;

  constexpr int shared_cap = (BDIMX * BDIMY > 2 * VPTX * BDIMY * row_stride)
                                 ? BDIMX * BDIMY
                                 : 2 * VPTX * BDIMY * row_stride;
  __shared__ U buf[shared_cap];

  U *warp_buf1 = reinterpret_cast<U *>(buf);
  U *warp_buf2 = warp_buf1 + VPTX * BDIMY * row_stride;

  for (int idx = threadIdx.y * blockDim.x + threadIdx.x;
       idx < 2 * VPTX * BDIMY * row_stride; idx += BDIMX * BDIMY) {
    buf[idx] = U(0);
  }
  __syncthreads();

  for (int i1_block = blockIdx.y * BDIMY * VPTX; i1_block < n1;
       i1_block += VPTX * BDIMY * gridDim.y) {
    cuLoadAddStridedInputs<T, U, VPTX>(
        i1_block, thr_load_row_off, thr_load_col_off, i2_off, row_stride,
        warp_buf1, warp_buf2, input, dout, n1, n2, mean, var, epsilon);
  }
  __syncthreads();

  // inter-warp reductions
  // sum within each warp
  U acc1 = U(0);
  U acc2 = U(0);
  for (int k = 0; k < VPTX; ++k) {
    int row1 = threadIdx.y + k * VPTX;
    int idx1 = row1 * row_stride + threadIdx.x;
    acc1 += warp_buf1[idx1];
    acc2 += warp_buf2[idx1];
  }
  warp_buf1[threadIdx.y * row_stride + threadIdx.x] = acc1;
  warp_buf2[threadIdx.y * row_stride + threadIdx.x] = acc2;
  __syncthreads();
  // sum all warps
  for (int offset = VPTX >> 1; offset > 1; offset >>= 1) {
    if (threadIdx.y < offset) {
      int row1 = threadIdx.y;
      int row2 = threadIdx.y + offset;
      int idx1 = row1 * row_stride + threadIdx.x;
      int idx2 = row2 * row_stride + threadIdx.x;
      warp_buf1[idx1] += warp_buf1[idx2];
      warp_buf2[idx1] += warp_buf2[idx2];
    }
    __syncthreads();
  }
  int i2 = blockIdx.x * blockDim.x + threadIdx.x;
  if (threadIdx.y == 0 && i2 < n2) {
    int row1 = threadIdx.y;
    int row2 = threadIdx.y + 1;
    int idx1 = row1 * row_stride + threadIdx.x;
    int idx2 = row2 * row_stride + threadIdx.x;
    part_grad_beta[blockIdx.y * n2 + i2] = warp_buf1[idx1] + warp_buf1[idx2];
    part_grad_gamma[blockIdx.y * n2 + i2] = warp_buf2[idx1] + warp_buf2[idx2];
  }
}

template <typename T, typename U, int BDIMX, int BDIMY>
__global__ void LayerNormBackwardSumGradGammaBeta(
    const U *part_grad_gamma, const U *part_grad_beta, const int part_size,
    // const int n1, const int n2, T* grad_gamma, T* grad_beta) {
    const int n1, const int n2, U *grad_gamma, U *grad_beta) {
  // sum partial gradients for gamma and beta
  __shared__ U buf[BDIMX * BDIMY];
  int i2 = blockIdx.x * BDIMX + threadIdx.x;
  if (i2 < n2) {
    // each warp does sequential reductions until reduced part_size is num_warps
    int num_warp_reductions = part_size / BDIMY;
    U sum_gamma = U(0);
    U sum_beta = U(0);
    const U *part_grad_gamma_ptr =
        part_grad_gamma + threadIdx.y * num_warp_reductions * n2 + i2;
    const U *part_grad_beta_ptr =
        part_grad_beta + threadIdx.y * num_warp_reductions * n2 + i2;
    for (int warp_offset = 0; warp_offset < num_warp_reductions;
         ++warp_offset) {
      sum_gamma += part_grad_gamma_ptr[warp_offset * n2];
      sum_beta += part_grad_beta_ptr[warp_offset * n2];
    }
    // inter-warp reductions
    constexpr int nbsize3 = BDIMX * BDIMY / 2;
    for (int offset = BDIMY / 2; offset >= 1; offset /= 2) {
      // top half write to shared memory
      if (threadIdx.y >= offset && threadIdx.y < 2 * offset) {
        const int write_idx = (threadIdx.y - offset) * blockDim.x + threadIdx.x;
        buf[write_idx] = sum_gamma;
        buf[write_idx + nbsize3] = sum_beta;
      }
      __syncthreads();
      // bottom half sums
      if (threadIdx.y < offset) {
        const int read_idx = threadIdx.y * BDIMX + threadIdx.x;
        sum_gamma += buf[read_idx];
        sum_beta += buf[read_idx + nbsize3];
      }
      __syncthreads();
    }
    // write out fully summed gradients
    if (threadIdx.y == 0) {
      grad_gamma[i2] = sum_gamma;
      grad_beta[i2] = sum_beta;
    }
  }
}

template <typename T, typename U, int BDIMX, int BDIMY>
__global__ void LayerNormBackwardComputeGradInput(
    const T *__restrict__ dout, const T *__restrict__ input, const int n1,
    const int n2,
    // const U* __restrict__ mean, const U* __restrict__ var, const float
    // epsilon, const T* gamma,
    const U *__restrict__ mean, const U *__restrict__ var, const float epsilon,
    const U *gamma, T *grad_input) {
#ifdef __HIPCC__
  for (auto i1 = hipBlockIdx_y; i1 < n1; i1 += hipGridDim_y) {
#else
  for (auto i1 = blockIdx.y; i1 < n1; i1 += gridDim.y) {
#endif
    U sum_loss1 = U(0);
    U sum_loss2 = U(0);
    const U c_mean = mean[i1];
    const U c_invvar = rsqrt_<U>(var[i1] + epsilon);
    const T *k_input = input + i1 * n2;
    const T *k_dout = dout + i1 * n2;
    constexpr int numx = BDIMX * BDIMY;
    const int thrx = threadIdx.x + threadIdx.y * BDIMX;
    if (gamma != NULL) {
      int l = 4 * thrx;
      for (; l + 3 < n2; l += 4 * numx) {
        for (int k = 0; k < 4; ++k) {
          const U c_h = static_cast<U>(k_input[l + k]);
          const U c_loss = static_cast<U>(k_dout[l + k]);
          sum_loss1 += c_loss * gamma[l + k];
          sum_loss2 += c_loss * gamma[l + k] * (c_h - c_mean) * c_invvar;
        }
      }
      for (; l < n2; ++l) {
        const U c_h = static_cast<U>(k_input[l]);
        const U c_loss = static_cast<U>(k_dout[l]);
        sum_loss1 += c_loss * gamma[l];
        sum_loss2 += c_loss * gamma[l] * (c_h - c_mean) * c_invvar;
      }
    } else {
      int l = 4 * thrx;
      for (; l + 3 < n2; l += 4 * numx) {
        for (int k = 0; k < 4; ++k) {
          const U c_h = static_cast<U>(k_input[l + k]);
          const U c_loss = static_cast<U>(k_dout[l + k]);
          sum_loss1 += c_loss;
          sum_loss2 += c_loss * (c_h - c_mean) * c_invvar;
        }
      }
      for (; l < n2; ++l) {
        const U c_h = static_cast<U>(k_input[l]);
        const U c_loss = static_cast<U>(k_dout[l]);
        sum_loss1 += c_loss;
        sum_loss2 += c_loss * (c_h - c_mean) * c_invvar;
      }
    }
    // intra-warp reductions
    for (int mask = BDIMX / 2; mask > 0; mask /= 2) {
#ifdef PADDLE_WITH_HIP
      sum_loss1 += __shfl_xor(sum_loss1, mask,
                              warpSize);  // WARP_SHFL_XOR(sum_loss1, mask);
      sum_loss2 += __shfl_xor(sum_loss2, mask,
                              warpSize);  // WARP_SHFL_XOR(sum_loss2, mask);
#else
      sum_loss1 +=
          __shfl_xor_sync(0xffffffff, sum_loss1, mask,
                          warpSize);  // WARP_SHFL_XOR(sum_loss1, mask);
      sum_loss2 +=
          __shfl_xor_sync(0xffffffff, sum_loss2, mask,
                          warpSize);  // WARP_SHFL_XOR(sum_loss2, mask);
#endif
    }
    // inter-warp reductions
    if (BDIMY > 1) {
      __shared__ U buf[BDIMX * BDIMY];
      for (int offset = BDIMY / 2; offset > 0; offset /= 2) {
        // upper half of warps write to shared
        if (threadIdx.y >= offset && threadIdx.y < 2 * offset) {
          const int wrt_i = (threadIdx.y - offset) * BDIMX + threadIdx.x;
          buf[2 * wrt_i] = sum_loss1;
          buf[2 * wrt_i + 1] = sum_loss2;
        }
        __syncthreads();
        // lower half merges
        if (threadIdx.y < offset) {
          const int read_i = threadIdx.y * blockDim.x + threadIdx.x;
          sum_loss1 += buf[2 * read_i];
          sum_loss2 += buf[2 * read_i + 1];
        }
        __syncthreads();
      }
      if (threadIdx.y == 0) {
        buf[2 * threadIdx.x] = sum_loss1;
        buf[2 * threadIdx.x + 1] = sum_loss2;
      }
      __syncthreads();
      if (threadIdx.y != 0) {
        sum_loss1 = buf[2 * threadIdx.x];
        sum_loss2 = buf[2 * threadIdx.x + 1];
      }
    }
    // all threads now have the two sums over l
    U fH = (U)n2;
    U term1 = (U(1) / fH) * c_invvar;
    T *k_grad_input = grad_input + i1 * n2;
    if (gamma != NULL) {
      for (int l = thrx; l < n2; l += numx) {
        const U c_h = static_cast<U>(k_input[l]);
        const U c_loss = static_cast<U>(k_dout[l]);
        U f_grad_input = fH * c_loss * gamma[l];
        f_grad_input -= sum_loss1;
        f_grad_input -= (c_h - c_mean) * c_invvar * sum_loss2;
        f_grad_input *= term1;
        k_grad_input[l] = static_cast<T>(f_grad_input);
      }
    } else {
      for (int l = thrx; l < n2; l += numx) {
        const U c_h = static_cast<U>(k_input[l]);
        const U c_loss = static_cast<U>(k_dout[l]);
        U f_grad_input = fH * c_loss;
        f_grad_input -= sum_loss1;
        f_grad_input -= (c_h - c_mean) * c_invvar * sum_loss2;
        f_grad_input *= term1;
        k_grad_input[l] = static_cast<T>(f_grad_input);
      }
    }
  }
}

// Make sure that d_scale != nullptr && d_bias != nullptr
// Since d_scale != nullptr, scale would not be nullptr
template <typename T, typename U, int BlockDim, bool HasDx>
__global__ void LayerNormBackwardGradientAll(const T *x, const T *d_y,
                                             U *d_scale, U *d_bias, T *d_x,
                                             const U *mean, const U *var,
                                             const U *scale, float epsilon,
                                             int batch_size, int feature_size,
                                             int col_offset) {
  using BlockReduce = cub::BlockReduce<PairForLayerNorm<U>, BlockDim>;
  __shared__ typename BlockReduce::TempStorage temp_storage;

  int beg_idx = threadIdx.x * feature_size + (blockIdx.x + col_offset);
  int end_idx = batch_size * feature_size + (blockIdx.x + col_offset);
  int stride = BlockDim * feature_size;

  U d_scale_partial = static_cast<U>(0), d_bias_partial = static_cast<U>(0);

  for (int i = beg_idx; i < end_idx; i += stride) {
    int row_idx = i / feature_size;
    auto var_val = real_sqrt(static_cast<U>(var[row_idx]) + epsilon);
    d_scale_partial += static_cast<U>(d_y[i]) *
                       (static_cast<U>(x[i]) - mean[row_idx]) / var_val;
    d_bias_partial += static_cast<U>(d_y[i]);
    if (HasDx) {
      d_x[i] = static_cast<T>(static_cast<U>(d_y[i]) *
                              scale[blockIdx.x + col_offset] / var_val);
    }
  }

  auto pair = BlockReduce(temp_storage)
                  .Reduce(PairForLayerNorm<U>(d_scale_partial, d_bias_partial),
                          PairForLayerNormAddFunctor<U>());

  if (threadIdx.x == 0) {
    d_scale[blockIdx.x + col_offset] = pair.first_;
    d_bias[blockIdx.x + col_offset] = pair.second_;
  }
}

// Make sure that there is only one true expression: d_scale != nullptr
// or d_bias != nullptr
// Notice: scale may be nullptr
template <typename T, typename U, int BlockDim, bool HasDx, bool HasDScale>
__global__ void LayerNormBackwardGradientScaleOrBias(
    const T *x, const T *d_y, U *d_scale, U *d_bias, T *d_x, const U *mean,
    const U *var, const U *scale, float epsilon, int batch_size,
    int feature_size, int col_offset) {
  using BlockReduce = cub::BlockReduce<U, BlockDim>;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  int beg_idx = threadIdx.x * feature_size + blockIdx.x + col_offset;
  int end_idx = batch_size * feature_size + blockIdx.x + col_offset;
  int stride = BlockDim * feature_size;
  U d_scale_or_d_bias_partial = static_cast<U>(0);

  for (int i = beg_idx; i < end_idx; i += stride) {
    int row_idx = i / feature_size;
    auto var_val =
        static_cast<U>(real_sqrt(static_cast<float>(var[row_idx]) + epsilon));
    if (HasDScale) {
      d_scale_or_d_bias_partial += static_cast<U>(d_y[i]) *
                                   (static_cast<U>(x[i]) - mean[row_idx]) /
                                   var_val;
    } else {  // d_bias != nullptr
      d_scale_or_d_bias_partial += static_cast<U>(d_y[i]);
    }

    if (HasDx) {
      if (scale != nullptr) {
        d_x[i] = static_cast<T>(static_cast<U>(d_y[i]) *
                                scale[blockIdx.x + col_offset] / var_val);
      } else {
        d_x[i] = static_cast<T>(static_cast<U>(d_y[i]) / var_val);
      }
    }
  }

  d_scale_or_d_bias_partial =
      BlockReduce(temp_storage).Reduce(d_scale_or_d_bias_partial, cub::Sum());

  if (threadIdx.x == 0) {
    if (HasDScale) {
      d_scale[blockIdx.x + col_offset] = d_scale_or_d_bias_partial;
    } else {
      d_bias[blockIdx.x + col_offset] = d_scale_or_d_bias_partial;
    }
  }
}

template <typename T, typename U, int BlockDim>
__global__ void LayerNormBackwardPostProcessToCalculateDX(const T *x, T *d_x,
                                                          const U *mean,
                                                          const U *var,
                                                          float epsilon,
                                                          int feature_size) {
  using BlockReduce = cub::BlockReduce<PairForLayerNorm<U>, BlockDim>;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  __shared__ U d_x_reduce_tmp[2];

  int beg_idx = blockIdx.x * feature_size + threadIdx.x;
  int end_idx = (blockIdx.x + 1) * feature_size;

  U block_mean = mean[blockIdx.x];
  U block_var = var[blockIdx.x];
  U d_x_mean_partial = static_cast<U>(0), d_x_var_partial = static_cast<U>(0);
  for (int i = beg_idx; i < end_idx; i += BlockDim) {
    d_x_mean_partial += static_cast<U>(d_x[i]);
    d_x_var_partial +=
        static_cast<U>(d_x[i]) * (static_cast<U>(x[i]) - block_mean);
  }

  auto pair =
      BlockReduce(temp_storage)
          .Reduce(PairForLayerNorm<U>(d_x_mean_partial, d_x_var_partial),
                  PairForLayerNormAddFunctor<U>());

  if (threadIdx.x == 0) {
    d_x_reduce_tmp[0] = static_cast<float>(pair.first_) / feature_size;
    d_x_reduce_tmp[1] =
        static_cast<float>(pair.second_) /
        (feature_size * (static_cast<float>(block_var) + epsilon));
  }
  __syncthreads();

  d_x_mean_partial = d_x_reduce_tmp[0];
  d_x_var_partial = d_x_reduce_tmp[1];
  for (int i = beg_idx; i < end_idx; i += BlockDim) {
    d_x[i] -= static_cast<T>(d_x_mean_partial);
    d_x[i] -=
        static_cast<T>((static_cast<U>(x[i]) - block_mean) * d_x_var_partial);
  }
}

// Here, we only calculate d_x
template <typename T, typename U, int BlockDim>
__global__ void LayerNormBackwardGradientOnlyDX(const T *x, const T *d_y,
                                                T *d_x, const U *mean,
                                                const U *var, const U *scale,
                                                float epsilon,
                                                int feature_size) {
  using BlockReduce = cub::BlockReduce<PairForLayerNorm<U>, BlockDim>;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  __shared__ U d_x_reduce_tmp[2];

  int beg_idx = blockIdx.x * feature_size + threadIdx.x;
  int end_idx = (blockIdx.x + 1) * feature_size;

  U block_mean = mean[blockIdx.x], block_var = var[blockIdx.x];
  U d_x_mean_partial = static_cast<U>(0), d_x_var_partial = static_cast<U>(0);
  for (int i = beg_idx; i < end_idx; i += BlockDim) {
    auto var_val =
        static_cast<U>(real_sqrt(static_cast<float>(block_var) + epsilon));
    if (scale != nullptr) {
      int col_idx = i % feature_size;
      d_x[i] =
          static_cast<T>(static_cast<U>(d_y[i]) * scale[col_idx] / var_val);
    } else {
      d_x[i] = static_cast<T>(static_cast<U>(d_y[i]) / var_val);
    }
    d_x_mean_partial += static_cast<U>(d_x[i]);
    d_x_var_partial +=
        static_cast<U>(d_x[i]) * (static_cast<U>(x[i]) - block_mean);
  }

  auto pair =
      BlockReduce(temp_storage)
          .Reduce(PairForLayerNorm<U>(d_x_mean_partial, d_x_var_partial),
                  PairForLayerNormAddFunctor<U>());

  if (threadIdx.x == 0) {
    d_x_reduce_tmp[0] = static_cast<float>(pair.first_) / feature_size;
    d_x_reduce_tmp[1] =
        static_cast<float>(pair.second_) /
        (feature_size * (static_cast<float>(block_var) + epsilon));
  }
  __syncthreads();

  d_x_mean_partial = d_x_reduce_tmp[0];
  d_x_var_partial = d_x_reduce_tmp[1];
  for (int i = beg_idx; i < end_idx; i += BlockDim) {
    d_x[i] -= static_cast<T>(d_x_mean_partial);
    d_x[i] -=
        static_cast<T>((static_cast<U>(x[i]) - block_mean) * d_x_var_partial);
  }
}

template <typename T, typename U>
__global__ void LayerNormBackwardWhenBatchSizeIsOne(
    const T *x, const T *d_y, T *d_x, U *d_scale, U *d_bias, const U *mean,
    const U *var, const U *scale, float epsilon, int feature_size) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < feature_size) {
    auto var_val =
        static_cast<U>(real_sqrt(static_cast<float>(var[idx]) + epsilon));
    if (d_x != nullptr) {
      if (d_scale == nullptr) {
        d_x[idx] = static_cast<T>(static_cast<U>(d_y[idx]) / var_val);
      } else {
        d_x[idx] =
            static_cast<T>(static_cast<U>(d_y[idx]) * scale[idx] / var_val);
      }
    }

    if (d_scale != nullptr) {
      d_scale[idx] = static_cast<U>(d_y[idx]) *
                     (static_cast<U>(x[idx]) - mean[idx]) / var_val;
    }

    if (d_bias != nullptr) d_bias[idx] = static_cast<U>(d_y[idx]);
  }
}

template <typename T, typename U>
static void LayerNormBackward(const T *x, const T *d_y, const U *scale,
                              const U *mean, const U *var, T *d_x, U *d_scale,
                              U *d_bias, float epsilon, int batch_size,
                              int feature_size,
                              const framework::ExecutionContext &ctx) {
  auto &dev_ctx = ctx.cuda_device_context();
  auto stream = dev_ctx.stream();
#ifdef __HIPCC__
  const int kMaxBlockDim = 256;
#else
  const int kMaxBlockDim = 512;
#endif
  const int kMaxBlockNum = 128;
  int gradient_flag = ((d_x != nullptr ? 1 : 0) << 2) |
                      ((d_scale != nullptr ? 1 : 0) << 1) |
                      ((d_bias != nullptr ? 1 : 0));
  if (gradient_flag == 0) return;

  if (batch_size == 1) {
    LayerNormBackwardWhenBatchSizeIsOne<
        T, U><<<(feature_size + kMaxBlockDim - 1) / kMaxBlockDim, kMaxBlockDim,
                0, stream>>>(x, d_y, d_x, d_scale, d_bias, mean, var, scale,
                             epsilon, feature_size);

    if (d_x != nullptr) {
      switch (GetDesiredBlockDim(feature_size)) {
        FIXED_BLOCK_DIM_CASE(LayerNormBackwardPostProcessToCalculateDX<
                             T, U, kBlockDim><<<1, kBlockDim, 0, stream>>>(
            x, d_x, mean, var, epsilon, feature_size));
      }
    }
    return;
  }

  auto block_dim = GetDesiredBlockDim(batch_size);
  switch (gradient_flag) {
    case 1:  // d_x == nulptr, d_scale == nullptr, d_bias != nullptr
      switch (block_dim) {
        FIXED_BLOCK_DIM_FIXED_BLOCK_NUM_CASE(
            feature_size, kMaxBlockNum,
            LayerNormBackwardGradientScaleOrBias<
                T, U, kBlockDim, false,
                false><<<block_num, kBlockDim, 0, stream>>>(
                x, d_y, d_scale, d_bias, d_x, mean, var, scale, epsilon,
                batch_size, feature_size, col_offset));
      }
      break;
    case 2:  // d_x == nullptr, d_scale != nullptr, d_bias == nullptr
      switch (block_dim) {
        FIXED_BLOCK_DIM_FIXED_BLOCK_NUM_CASE(
            feature_size, kMaxBlockNum,
            LayerNormBackwardGradientScaleOrBias<
                T, U, kBlockDim, false,
                true><<<block_num, kBlockDim, 0, stream>>>(
                x, d_y, d_scale, d_bias, d_x, mean, var, scale, epsilon,
                batch_size, feature_size, col_offset));
      }
      break;
    case 3:  // d_x == nullptr, d_scale != nulptr, d_bias != nullptr
      switch (block_dim) {
        FIXED_BLOCK_DIM_FIXED_BLOCK_NUM_CASE(
            feature_size, kMaxBlockNum,
            LayerNormBackwardGradientAll<
                T, U, kBlockDim, false><<<block_num, kBlockDim, 0, stream>>>(
                x, d_y, d_scale, d_bias, d_x, mean, var, scale, epsilon,
                batch_size, feature_size, col_offset));
      }
      break;
    case 4:  // d_x != nullptr, d_scale == nullptr, d_bias == nullptr
      switch (GetDesiredBlockDim(feature_size)) {
        FIXED_BLOCK_DIM_CASE(
            LayerNormBackwardGradientOnlyDX<
                T, U, kBlockDim><<<batch_size, kBlockDim, 0, stream>>>(
                x, d_y, d_x, mean, var, scale, epsilon, feature_size));
      }
      break;
    case 5:  // d_x != nulptr, d_scale == nullptr, d_bias != nullptr
      switch (block_dim) {
        FIXED_BLOCK_DIM_FIXED_BLOCK_NUM_CASE(
            feature_size, kMaxBlockNum,
            LayerNormBackwardGradientScaleOrBias<
                T, U, kBlockDim, true,
                false><<<block_num, kBlockDim, 0, stream>>>(
                x, d_y, d_scale, d_bias, d_x, mean, var, scale, epsilon,
                batch_size, feature_size, col_offset));
      }
      switch (GetDesiredBlockDim(feature_size)) {
        FIXED_BLOCK_DIM_CASE(
            LayerNormBackwardPostProcessToCalculateDX<
                T, U, kBlockDim><<<batch_size, kBlockDim, 0, stream>>>(
                x, d_x, mean, var, epsilon, feature_size));
      }
      break;
    case 6:  // d_x != nullptr, d_scale != nullptr, d_bias == nullptr
      switch (block_dim) {
        FIXED_BLOCK_DIM_FIXED_BLOCK_NUM_CASE(
            feature_size, kMaxBlockNum,
            LayerNormBackwardGradientScaleOrBias<
                T, U, kBlockDim, true,
                true><<<block_num, kBlockDim, 0, stream>>>(
                x, d_y, d_scale, d_bias, d_x, mean, var, scale, epsilon,
                batch_size, feature_size, col_offset));
      }
      switch (GetDesiredBlockDim(feature_size)) {
        FIXED_BLOCK_DIM_CASE(
            LayerNormBackwardPostProcessToCalculateDX<
                T, U, kBlockDim><<<batch_size, kBlockDim, 0, stream>>>(
                x, d_x, mean, var, epsilon, feature_size));
      }
      break;
    case 7:  // d_x != nullptr, d_scale != nullptr, d_bias != nullptr
    {
      constexpr int VPT = 4;
      constexpr int BDIMX2 = 32;
      constexpr int BDIMY2 = 4;
      dim3 threads2(BDIMX2, BDIMY2, 1);
      constexpr int part_size = BDIMY2 * VPT;
      const dim3 blocks2((feature_size + BDIMX2 - 1) / BDIMX2, part_size, 1);

      auto part_grad_gamma_ptr =
          memory::Alloc(dev_ctx, part_size * feature_size * sizeof(U));
      auto part_grad_beta_ptr =
          memory::Alloc(dev_ctx, part_size * feature_size * sizeof(U));
      U *part_grad_gamma = reinterpret_cast<U *>(part_grad_gamma_ptr->ptr());
      U *part_grad_beta = reinterpret_cast<U *>(part_grad_beta_ptr->ptr());

      LayerNormBackwardPartGradGammaBeta<T, U, BDIMX2, BDIMY2,
                                         VPT><<<blocks2, threads2, 0, stream>>>(
          d_y, x, batch_size, feature_size, mean, var, epsilon, part_grad_gamma,
          part_grad_beta);  // compute part_grad_gamma, beta

      constexpr int BDIMX3 = 32;
      constexpr int BDIMY3 = 8;
      dim3 threads3(BDIMX3, BDIMY3, 1);
      const dim3 blocks3((feature_size + BDIMX2 - 1) / BDIMX2, 1, 1);
      LayerNormBackwardSumGradGammaBeta<
          T, U, BDIMX3, BDIMY3><<<blocks3, threads3, 0, stream>>>(
          part_grad_gamma, part_grad_beta, part_size, batch_size, feature_size,
          d_scale, d_bias);

      constexpr int BDIMX1 = 32;
      constexpr int BDIMY1 = 4;
      dim3 threads1(BDIMX1, BDIMY1, 1);
      const dim3 blocks1(1, batch_size, 1);
      LayerNormBackwardComputeGradInput<
          T, U, BDIMX1, BDIMY1><<<blocks1, threads1, 0, stream>>>(
          d_y, x, batch_size, feature_size, mean, var, epsilon, scale, d_x);
      break;
    }
    default:
      break;
  }
}

template <typename T>
void LayerNormDirectCUDAFunctor<T>::operator()(gpuStream_t stream,
                                               const T *input,
                                               std::vector<int> input_shape,
                                               const T *bias, const T *scale,
                                               T *output, T *mean, T *variance,
                                               int begin_norm_axis, float eps) {
  const auto x_dims = framework::make_ddim(input_shape);
  auto matrix_dim = framework::flatten_to_2d(x_dims, begin_norm_axis);
  int batch_size = static_cast<int>(matrix_dim[0]);
  int feature_size = static_cast<int>(matrix_dim[1]);
  switch (GetDesiredBlockDim(feature_size)) {
    FIXED_BLOCK_DIM_CASE(
        LayerNormForward<T, T, kBlockDim><<<batch_size, kBlockDim, 0, stream>>>(
            input, scale, bias, output, mean, variance, eps, feature_size));
    default:
      PADDLE_THROW(platform::errors::InvalidArgument(
          "Product from begin_norm_axis to end in layer_norm must be larger "
          "than 1"));
      break;
  }
}

template <>
void LayerNormDirectCUDAFunctor<half>::operator()(
    gpuStream_t stream, const half *input, std::vector<int> input_shape,
    const half *bias, const half *scale, half *output, half *mean,
    half *variance, int begin_norm_axis, float eps) {
  const auto x_dims = framework::make_ddim(input_shape);
  auto matrix_dim = framework::flatten_to_2d(x_dims, begin_norm_axis);
  int batch_size = static_cast<int>(matrix_dim[0]);
  int feature_size = static_cast<int>(matrix_dim[1]);
  switch (GetDesiredBlockDim(feature_size)) {
    FIXED_BLOCK_DIM_CASE(
        LayerNormForwardFP16<half, half,
                             kBlockDim><<<batch_size, kBlockDim, 0, stream>>>(
            input, scale, bias, output, mean, variance, eps, feature_size));
    default:
      PADDLE_THROW(platform::errors::InvalidArgument(
          "Product from begin_norm_axis to end in layer_norm must be larger "
          "than 1"));
      break;
  }
}

template <typename T>
class LayerNormKernel<platform::CUDADeviceContext, T>
    : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    using U = LayerNormParamType<T>;
    const float epsilon = ctx.Attr<float>("epsilon");
    auto *scale = ctx.Input<Tensor>("Scale");
    auto *bias = ctx.Input<Tensor>("Bias");
    auto *x = ctx.Input<Tensor>("X");

    auto *y = ctx.Output<Tensor>("Y");
    auto *mean = ctx.Output<Tensor>("Mean");
    auto *var = ctx.Output<Tensor>("Variance");
    const auto begin_norm_axis = ctx.Attr<int>("begin_norm_axis");

    const auto x_dims = x->dims();
    auto *x_data = x->data<T>();
    auto *y_data = y->mutable_data<T>(ctx.GetPlace());
    auto *mean_data = mean->mutable_data<U>(ctx.GetPlace());
    auto *var_data = var->mutable_data<U>(ctx.GetPlace());
    auto *scale_data = (scale == nullptr ? nullptr : scale->data<U>());
    auto *bias_data = (bias == nullptr ? nullptr : bias->data<U>());

    auto matrix_dim = framework::flatten_to_2d(x_dims, begin_norm_axis);
    int batch_size = static_cast<int>(matrix_dim[0]);
    int feature_size = static_cast<int>(matrix_dim[1]);

    auto stream = ctx.cuda_device_context().stream();

    switch (GetDesiredBlockDim(feature_size)) {
      FIXED_BLOCK_DIM_CASE(
          LayerNormForward<T, U,
                           kBlockDim><<<batch_size, kBlockDim, 0, stream>>>(
              x_data, scale_data, bias_data, y_data, mean_data, var_data,
              epsilon, feature_size));
      default:
        PADDLE_THROW(platform::errors::InvalidArgument(
            "Product from begin_norm_axis to end must be larger than 1"));
        break;
    }
  }
};

template <typename T>
class LayerNormGradKernel<platform::CUDADeviceContext, T>
    : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    using U = LayerNormParamType<T>;
    const float epsilon = ctx.Attr<float>("epsilon");
    // d_x, d_scale, d_bias may be nullptr
    auto *d_x = ctx.Output<Tensor>(framework::GradVarName("X"));
    auto *d_scale = ctx.Output<Tensor>(framework::GradVarName("Scale"));
    auto *d_bias = ctx.Output<Tensor>(framework::GradVarName("Bias"));

    auto *x = ctx.Input<Tensor>("X");
    auto *mean = ctx.Input<Tensor>("Mean");
    auto *var = ctx.Input<Tensor>("Variance");
    auto *scale = ctx.Input<Tensor>("Scale");
    auto *d_y = ctx.Input<Tensor>(framework::GradVarName("Y"));

    auto *x_data = x->data<T>();
    auto *d_y_data = d_y->data<T>();
    auto *mean_data = mean->data<U>();
    auto *var_data = var->data<U>();

    auto *scale_data = (scale == nullptr ? nullptr : scale->data<U>());
    auto *d_scale_data =
        (d_scale == nullptr ? nullptr
                            : d_scale->mutable_data<U>(ctx.GetPlace()));
    auto *d_bias_data =
        (d_bias == nullptr ? nullptr : d_bias->mutable_data<U>(ctx.GetPlace()));
    auto *d_x_data =
        (d_x == nullptr ? nullptr : d_x->mutable_data<T>(ctx.GetPlace()));

    const auto &x_dims = x->dims();
    const auto begin_norm_axis = ctx.Attr<int>("begin_norm_axis");
    auto matrix_dim = framework::flatten_to_2d(x_dims, begin_norm_axis);
    int batch_size = static_cast<int>(matrix_dim[0]);
    int feature_size = static_cast<int>(matrix_dim[1]);

    LayerNormBackward<T, U>(x_data, d_y_data, scale_data, mean_data, var_data,
                            d_x_data, d_scale_data, d_bias_data, epsilon,
                            batch_size, feature_size, ctx);
  }
};

template class LayerNormDirectCUDAFunctor<float>;
#ifdef TRT_PLUGIN_FP16_AVALIABLE
template class LayerNormDirectCUDAFunctor<half>;
#endif

#undef FIXED_BLOCK_DIM_FIXED_BLOCK_NUM_CASE_BASE
#undef FIXED_BLOCK_DIM_FIXED_BLOCK_NUM_CASE
#undef FIXED_BLOCK_DIM_CASE_BASE
#undef FIXED_BLOCK_DIM_CASE
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;
#ifdef PADDLE_WITH_HIP
// MIOPEN do not support double
REGISTER_OP_CUDA_KERNEL(
    layer_norm,
    ops::LayerNormKernel<paddle::platform::CUDADeviceContext, float>,
    ops::LayerNormKernel<paddle::platform::CUDADeviceContext, plat::float16>);
REGISTER_OP_CUDA_KERNEL(
    layer_norm_grad,
    ops::LayerNormGradKernel<paddle::platform::CUDADeviceContext, float>,
    ops::LayerNormGradKernel<paddle::platform::CUDADeviceContext,
                             plat::float16>);
#else
REGISTER_OP_CUDA_KERNEL(
    layer_norm,
    ops::LayerNormKernel<paddle::platform::CUDADeviceContext, float>,
    ops::LayerNormKernel<paddle::platform::CUDADeviceContext, double>,
    ops::LayerNormKernel<paddle::platform::CUDADeviceContext, plat::float16>);
REGISTER_OP_CUDA_KERNEL(
    layer_norm_grad,
    ops::LayerNormGradKernel<paddle::platform::CUDADeviceContext, float>,
    ops::LayerNormGradKernel<paddle::platform::CUDADeviceContext, double>,
    ops::LayerNormGradKernel<paddle::platform::CUDADeviceContext,
                             plat::float16>);
#endif
