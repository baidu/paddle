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

#include <cub/cub.cuh>
#include <memory>
#include <vector>
#include "paddle/fluid/framework/ddim.h"
#include "paddle/fluid/operators/layer_norm_op.h"
#include "paddle/fluid/platform/float16.h"

namespace paddle {
namespace operators {

inline static int GetDesiredBlockDim(int block_dim) {
  const int kMaxBlockDim = 512;
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

template <typename T, int BlockDim>
__global__ void LayerNormForward(const T *x, const T *scale, const T *bias,
                                 T *y, T *mean, T *var, float epsilon,
                                 int feature_size) {
  using BlockReduce = cub::BlockReduce<PairForLayerNorm<double>, BlockDim>;
  __shared__ typename BlockReduce::TempStorage temp_storage;

  int beg_idx = blockIdx.x * feature_size + threadIdx.x;
  int end_idx = (blockIdx.x + 1) * feature_size;

  // Step 1: Reduce to calculate mean and var
  double mean_val = 0;
  double var_val = 0;
  for (int i = beg_idx; i < end_idx; i += BlockDim) {
    float tmp = static_cast<float>(x[i]);
    mean_val += tmp;
    var_val += (tmp * tmp);
  }
  auto pair = BlockReduce(temp_storage)
                  .Reduce(PairForLayerNorm<double>(mean_val, var_val),
                          PairForLayerNormAddFunctor<double>());
  if (threadIdx.x == 0) {
    auto tmp = pair.first_ / feature_size;
    mean[blockIdx.x] = static_cast<T>(tmp);
    var[blockIdx.x] = static_cast<T>(pair.second_ / feature_size - tmp * tmp);
  }
  __syncthreads();
  mean_val = static_cast<float>(mean[blockIdx.x]);
  var_val = static_cast<float>(
      static_cast<T>(real_sqrt(static_cast<float>(var[blockIdx.x]) + epsilon)));

  // Step 2: Calculate y
  if (scale != nullptr) {
    if (bias != nullptr) {
      for (int i = beg_idx, j = threadIdx.x; i < end_idx;
           i += BlockDim, j += BlockDim) {
        y[i] = static_cast<float>(scale[j]) *
                   (static_cast<float>(x[i]) - mean_val) / var_val +
               static_cast<float>(bias[j]);
      }
    } else {
      for (int i = beg_idx, j = threadIdx.x; i < end_idx;
           i += BlockDim, j += BlockDim) {
        y[i] = static_cast<float>(scale[j]) *
               (static_cast<float>(x[i]) - mean_val) / var_val;
      }
    }
  } else {  // scale == nullptr
    if (bias != nullptr) {
      for (int i = beg_idx, j = threadIdx.x; i < end_idx;
           i += BlockDim, j += BlockDim) {
        y[i] = (static_cast<float>(x[i]) - mean_val) / var_val +
               static_cast<float>(bias[j]);
      }
    } else {
      for (int i = beg_idx, j = threadIdx.x; i < end_idx;
           i += BlockDim, j += BlockDim) {
        y[i] = (static_cast<float>(x[i]) - mean_val) / var_val;
      }
    }
  }
}

// Make sure that d_scale != nullptr && d_bias != nullptr
// Since d_scale != nullptr, scale would not be nullptr
template <typename T, int BlockDim, bool HasDx>
__global__ void LayerNormBackwardGradientAll(const T *x, const T *d_y,
                                             T *d_scale, T *d_bias, T *d_x,
                                             const T *mean, const T *var,
                                             const T *scale, float epsilon,
                                             int batch_size, int feature_size,
                                             int col_offset) {
  using BlockReduce = cub::BlockReduce<PairForLayerNorm<T>, BlockDim>;
  __shared__ typename BlockReduce::TempStorage temp_storage;

  int beg_idx = threadIdx.x * feature_size + (blockIdx.x + col_offset);
  int end_idx = batch_size * feature_size + (blockIdx.x + col_offset);
  int stride = BlockDim * feature_size;

  T d_scale_partial = static_cast<T>(0), d_bias_partial = static_cast<T>(0);

  for (int i = beg_idx; i < end_idx; i += stride) {
    int row_idx = i / feature_size;
    auto var_val =
        static_cast<T>(real_sqrt(static_cast<float>(var[row_idx]) + epsilon));
    d_scale_partial += d_y[i] * (x[i] - mean[row_idx]) / var_val;
    d_bias_partial += d_y[i];
    if (HasDx) {
      d_x[i] = d_y[i] * scale[blockIdx.x + col_offset] / var_val;
    }
  }

  auto pair = BlockReduce(temp_storage)
                  .Reduce(PairForLayerNorm<T>(d_scale_partial, d_bias_partial),
                          PairForLayerNormAddFunctor<T>());

  if (threadIdx.x == 0) {
    d_scale[blockIdx.x + col_offset] = pair.first_;
    d_bias[blockIdx.x + col_offset] = pair.second_;
  }
}

// Make sure that there is only one true expression: d_scale != nullptr
// or d_bias != nullptr
// Notice: scale may be nullptr
template <typename T, int BlockDim, bool HasDx, bool HasDScale>
__global__ void LayerNormBackwardGradientScaleOrBias(
    const T *x, const T *d_y, T *d_scale, T *d_bias, T *d_x, const T *mean,
    const T *var, const T *scale, float epsilon, int batch_size,
    int feature_size, int col_offset) {
  using BlockReduce = cub::BlockReduce<T, BlockDim>;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  int beg_idx = threadIdx.x * feature_size + blockIdx.x + col_offset;
  int end_idx = batch_size * feature_size + blockIdx.x + col_offset;
  int stride = BlockDim * feature_size;
  T d_scale_or_d_bias_partial = static_cast<T>(0);

  for (int i = beg_idx; i < end_idx; i += stride) {
    int row_idx = i / feature_size;
    auto var_val =
        static_cast<T>(real_sqrt(static_cast<float>(var[row_idx]) + epsilon));
    if (HasDScale) {
      d_scale_or_d_bias_partial += d_y[i] * (x[i] - mean[row_idx]) / var_val;
    } else {  // d_bias != nullptr
      d_scale_or_d_bias_partial += d_y[i];
    }

    if (HasDx) {
      if (scale != nullptr) {
        d_x[i] = d_y[i] * scale[blockIdx.x + col_offset] / var_val;
      } else {
        d_x[i] = d_y[i] / var_val;
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

template <typename T, int BlockDim>
__global__ void LayerNormBackwardPostProcessToCalculateDX(const T *x, T *d_x,
                                                          const T *mean,
                                                          const T *var,
                                                          float epsilon,
                                                          int feature_size) {
  using BlockReduce = cub::BlockReduce<PairForLayerNorm<T>, BlockDim>;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  __shared__ T d_x_reduce_tmp[2];

  int beg_idx = blockIdx.x * feature_size + threadIdx.x;
  int end_idx = (blockIdx.x + 1) * feature_size;

  T block_mean = mean[blockIdx.x];
  T block_var = var[blockIdx.x];
  T d_x_mean_partial = static_cast<T>(0), d_x_var_partial = static_cast<T>(0);
  for (int i = beg_idx; i < end_idx; i += BlockDim) {
    d_x_mean_partial += d_x[i];
    d_x_var_partial += d_x[i] * (x[i] - block_mean);
  }

  auto pair =
      BlockReduce(temp_storage)
          .Reduce(PairForLayerNorm<T>(d_x_mean_partial, d_x_var_partial),
                  PairForLayerNormAddFunctor<T>());

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
    d_x[i] -= d_x_mean_partial;
    d_x[i] -= (x[i] - block_mean) * d_x_var_partial;
  }
}

// Here, we only calculate d_x
template <typename T, int BlockDim>
__global__ void LayerNormBackwardGradientOnlyDX(const T *x, const T *d_y,
                                                T *d_x, const T *mean,
                                                const T *var, const T *scale,
                                                float epsilon,
                                                int feature_size) {
  using BlockReduce = cub::BlockReduce<PairForLayerNorm<T>, BlockDim>;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  __shared__ T d_x_reduce_tmp[2];

  int beg_idx = blockIdx.x * feature_size + threadIdx.x;
  int end_idx = (blockIdx.x + 1) * feature_size;

  T block_mean = mean[blockIdx.x], block_var = var[blockIdx.x];
  T d_x_mean_partial = static_cast<T>(0), d_x_var_partial = static_cast<T>(0);
  for (int i = beg_idx; i < end_idx; i += BlockDim) {
    auto var_val =
        static_cast<T>(real_sqrt(static_cast<float>(block_var) + epsilon));
    if (scale != nullptr) {
      int col_idx = i % feature_size;
      d_x[i] = d_y[i] * scale[col_idx] / var_val;
    } else {
      d_x[i] = d_y[i] / var_val;
    }
    d_x_mean_partial += d_x[i];
    d_x_var_partial += d_x[i] * (x[i] - block_mean);
  }

  auto pair =
      BlockReduce(temp_storage)
          .Reduce(PairForLayerNorm<T>(d_x_mean_partial, d_x_var_partial),
                  PairForLayerNormAddFunctor<T>());

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
    d_x[i] -= d_x_mean_partial;
    d_x[i] -= (x[i] - block_mean) * d_x_var_partial;
  }
}

template <typename T>
__global__ void LayerNormBackwardWhenBatchSizeIsOne(
    const T *x, const T *d_y, T *d_x, T *d_scale, T *d_bias, const T *mean,
    const T *var, const T *scale, float epsilon, int feature_size) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < feature_size) {
    auto var_val =
        static_cast<T>(real_sqrt(static_cast<float>(var[idx]) + epsilon));
    if (d_x != nullptr) {
      if (d_scale == nullptr) {
        d_x[idx] = d_y[idx] / var_val;
      } else {
        d_x[idx] = d_y[idx] * scale[idx] / var_val;
      }
    }

    if (d_scale != nullptr) {
      d_scale[idx] = d_y[idx] * (x[idx] - mean[idx]) / var_val;
    }

    if (d_bias != nullptr) d_bias[idx] = d_y[idx];
  }
}

template <typename T>
static void LayerNormBackward(const T *x, const T *d_y, const T *scale,
                              const T *mean, const T *var, T *d_x, T *d_scale,
                              T *d_bias, float epsilon, int batch_size,
                              int feature_size, cudaStream_t stream) {
  const int kMaxBlockDim = 512;
  const int kMaxBlockNum = 128;
  int gradient_flag = ((d_x != nullptr ? 1 : 0) << 2) |
                      ((d_scale != nullptr ? 1 : 0) << 1) |
                      ((d_bias != nullptr ? 1 : 0));
  if (gradient_flag == 0) return;

  if (batch_size == 1) {
    LayerNormBackwardWhenBatchSizeIsOne<
        T><<<(feature_size + kMaxBlockDim - 1) / kMaxBlockDim, kMaxBlockDim, 0,
             stream>>>(x, d_y, d_x, d_scale, d_bias, mean, var, scale, epsilon,
                       feature_size);

    if (d_x != nullptr) {
      switch (GetDesiredBlockDim(feature_size)) {
        FIXED_BLOCK_DIM_CASE(LayerNormBackwardPostProcessToCalculateDX<
                             T, kBlockDim><<<1, kBlockDim, 0, stream>>>(
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
                T, kBlockDim, false,
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
                T, kBlockDim, false, true><<<block_num, kBlockDim, 0, stream>>>(
                x, d_y, d_scale, d_bias, d_x, mean, var, scale, epsilon,
                batch_size, feature_size, col_offset));
      }
      break;
    case 3:  // d_x == nullptr, d_scale != nulptr, d_bias != nullptr
      switch (block_dim) {
        FIXED_BLOCK_DIM_FIXED_BLOCK_NUM_CASE(
            feature_size, kMaxBlockNum,
            LayerNormBackwardGradientAll<
                T, kBlockDim, false><<<block_num, kBlockDim, 0, stream>>>(
                x, d_y, d_scale, d_bias, d_x, mean, var, scale, epsilon,
                batch_size, feature_size, col_offset));
      }
      break;
    case 4:  // d_x != nullptr, d_scale == nullptr, d_bias == nullptr
      switch (GetDesiredBlockDim(feature_size)) {
        FIXED_BLOCK_DIM_CASE(
            LayerNormBackwardGradientOnlyDX<
                T, kBlockDim><<<batch_size, kBlockDim, 0, stream>>>(
                x, d_y, d_x, mean, var, scale, epsilon, feature_size));
      }
      break;
    case 5:  // d_x != nulptr, d_scale == nullptr, d_bias != nullptr
      switch (block_dim) {
        FIXED_BLOCK_DIM_FIXED_BLOCK_NUM_CASE(
            feature_size, kMaxBlockNum,
            LayerNormBackwardGradientScaleOrBias<
                T, kBlockDim, true, false><<<block_num, kBlockDim, 0, stream>>>(
                x, d_y, d_scale, d_bias, d_x, mean, var, scale, epsilon,
                batch_size, feature_size, col_offset));
      }
      switch (GetDesiredBlockDim(feature_size)) {
        FIXED_BLOCK_DIM_CASE(
            LayerNormBackwardPostProcessToCalculateDX<
                T, kBlockDim><<<batch_size, kBlockDim, 0, stream>>>(
                x, d_x, mean, var, epsilon, feature_size));
      }
      break;
    case 6:  // d_x != nullptr, d_scale != nullptr, d_bias == nullptr
      switch (block_dim) {
        FIXED_BLOCK_DIM_FIXED_BLOCK_NUM_CASE(
            feature_size, kMaxBlockNum,
            LayerNormBackwardGradientScaleOrBias<
                T, kBlockDim, true, true><<<block_num, kBlockDim, 0, stream>>>(
                x, d_y, d_scale, d_bias, d_x, mean, var, scale, epsilon,
                batch_size, feature_size, col_offset));
      }
      switch (GetDesiredBlockDim(feature_size)) {
        FIXED_BLOCK_DIM_CASE(
            LayerNormBackwardPostProcessToCalculateDX<
                T, kBlockDim><<<batch_size, kBlockDim, 0, stream>>>(
                x, d_x, mean, var, epsilon, feature_size));
      }
      break;
    case 7:  // d_x != nullptr, d_scale != nullptr, d_bias != nullptr
      switch (block_dim) {
        FIXED_BLOCK_DIM_FIXED_BLOCK_NUM_CASE(
            feature_size, kMaxBlockNum,
            LayerNormBackwardGradientAll<
                T, kBlockDim, true><<<block_num, kBlockDim, 0, stream>>>(
                x, d_y, d_scale, d_bias, d_x, mean, var, scale, epsilon,
                batch_size, feature_size, col_offset));
      }
      switch (GetDesiredBlockDim(feature_size)) {
        FIXED_BLOCK_DIM_CASE(
            LayerNormBackwardPostProcessToCalculateDX<
                T, kBlockDim><<<batch_size, kBlockDim, 0, stream>>>(
                x, d_x, mean, var, epsilon, feature_size));
      }
      break;
    default:
      break;
  }
}

template <typename T>
void LayerNormDirectCUDAFunctor<T>::operator()(cudaStream_t stream,
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
        LayerNormForward<T, kBlockDim><<<batch_size, kBlockDim, 0, stream>>>(
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
    auto *mean_data = mean->mutable_data<T>(ctx.GetPlace());
    auto *var_data = var->mutable_data<T>(ctx.GetPlace());
    auto *scale_data = (scale == nullptr ? nullptr : scale->data<T>());
    auto *bias_data = (bias == nullptr ? nullptr : bias->data<T>());

    auto matrix_dim = framework::flatten_to_2d(x_dims, begin_norm_axis);
    int batch_size = static_cast<int>(matrix_dim[0]);
    int feature_size = static_cast<int>(matrix_dim[1]);

    auto stream = ctx.cuda_device_context().stream();

    switch (GetDesiredBlockDim(feature_size)) {
      FIXED_BLOCK_DIM_CASE(
          LayerNormForward<T, kBlockDim><<<batch_size, kBlockDim, 0, stream>>>(
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
    auto *mean_data = mean->data<T>();
    auto *var_data = var->data<T>();
    auto *scale_data = (scale == nullptr ? nullptr : scale->data<T>());
    auto *d_scale_data =
        (d_scale == nullptr ? nullptr
                            : d_scale->mutable_data<T>(ctx.GetPlace()));
    auto *d_bias_data =
        (d_bias == nullptr ? nullptr : d_bias->mutable_data<T>(ctx.GetPlace()));
    auto *d_x_data =
        (d_x == nullptr ? nullptr : d_x->mutable_data<T>(ctx.GetPlace()));

    const auto &x_dims = x->dims();
    const auto begin_norm_axis = ctx.Attr<int>("begin_norm_axis");
    auto matrix_dim = framework::flatten_to_2d(x_dims, begin_norm_axis);
    int batch_size = static_cast<int>(matrix_dim[0]);
    int feature_size = static_cast<int>(matrix_dim[1]);

    auto stream = ctx.cuda_device_context().stream();

    LayerNormBackward<T>(x_data, d_y_data, scale_data, mean_data, var_data,
                         d_x_data, d_scale_data, d_bias_data, epsilon,
                         batch_size, feature_size, stream);
  }
};
template class LayerNormDirectCUDAFunctor<float>;
#undef FIXED_BLOCK_DIM_FIXED_BLOCK_NUM_CASE_BASE
#undef FIXED_BLOCK_DIM_FIXED_BLOCK_NUM_CASE
#undef FIXED_BLOCK_DIM_CASE_BASE
#undef FIXED_BLOCK_DIM_CASE
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;
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
