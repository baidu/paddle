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
#include "paddle/fluid/operators/math/cross_entropy.h"
#include "paddle/fluid/operators/math/math_function.h"
#include "paddle/fluid/operators/softmax_with_cross_entropy_op.h"
#include "paddle/fluid/platform/for_range.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;

namespace {
template <typename T>
__global__ void CrossEntropyGrad(T* logit_grad, const int64_t* labels,
                                 const int n, const int d, const int remain,
                                 const int ignore_index) {
  CUDA_KERNEL_LOOP(index, n * remain) {
    int idx_n = index / remain;
    int idx_remain = index % remain;
    int tmp = labels[index];
    if (ignore_index != tmp) {
      int idx = idx_n * d + tmp * remain + idx_remain;
      logit_grad[idx] -= static_cast<T>(1.);
    }
  }
}

template <typename T>
__global__ void Scale(T* logit_grad, const T* loss_grad, const int num,
                      const int d, const int remain) {
  CUDA_KERNEL_LOOP(index, num) {
    int idx_n = index / d;
    int idx_remain = index % remain;
    logit_grad[index] *= loss_grad[idx_n * remain + idx_remain];
  }
}

template <typename T>
__global__ void SoftCrossEntropyGradientKernel(T* logit_grad,
                                               const T* loss_grad,
                                               const T* labels, const int n,
                                               const int d, const int remain) {
  int ids = blockIdx.x * blockDim.x + threadIdx.x;
  if (ids < n * d) {
    int idx_n = ids / d;
    int idx_remain = ids % remain;
    int idx_loss = idx_n * remain + idx_remain;
    logit_grad[ids] = loss_grad[idx_loss] * (logit_grad[ids] - labels[ids]);
  }
}

}  // namespace

static __device__ __forceinline__ platform::float16 exp_on_device(
    platform::float16 x) {
  return ::Eigen::numext::exp(x);
}
static __device__ __forceinline__ float exp_on_device(float x) {
  return expf(x);
}
static __device__ __forceinline__ double exp_on_device(double x) {
  return exp(x);
}
static __device__ __forceinline__ platform::float16 log_on_device(
    platform::float16 x) {
  return math::TolerableValue<platform::float16>()(::Eigen::numext::log(x));
}
static __device__ __forceinline__ float log_on_device(float x) {
  return math::TolerableValue<float>()(logf(x));
}
static __device__ __forceinline__ double log_on_device(double x) {
  return math::TolerableValue<double>()(log(x));
}

/** In the following codes, 3 CUDA kernels are implemented to calculate softmax
 * and loss **/
/*
  Supposing the x is `logits` and y is `labels`, the equations are as
followings:
  cross\_entropy_i = \sum_{j}[- y_i_j * log({e^{x_i_j}/\sum_{j}e^{x_i_j}})]
        = \sum_{j}[- y_i_j * log({e^{x_i_j - max_i}/\sum_{j}e^{x_i_j-max_i}})]
        = \sum_{j}[-y_i_j * (x_i_j - max_i - log\sum_{j}e^{x_i_j - max_i})]
        = \sum_{j}[-y_i_j * (x_i_j - max_i - logDiffMaxSum_i)]
        = \sum_{j}(-y_i_j * tmp_i_j)
  softmax_i_j = e^{tmp_i_j}
where:
  max_i = \max_{j}{x_i_j}
  logDiffMaxSum_i = log\sum_{j}e^{x_i_j - max_i}
  tmp_i_j = x_i_j - max_i - logDiffMaxSum_i
Therefore, the calculation can be separated into 3 steps:
Step 1: row-wise operation to calculate max_i
Step 2: row-wise operation to calculate logDiffMaxSum_i
Step 3: calculate tmp_i_j, and finally get softmax_i_j and cross\_entropy_i
To save memory, we can share memory among max_i, logDiffMaxSum_i and
cross\_entropy_i.
In this way, the 3 steps should be changed to:
Step 1 (RowReductionForMax): row-wise operation to calculate max_i
Step 2 (RowReductionForDiffMaxSum): calculate immediate result of softmax'_i_j =
x_i_j - max_i, and row-wise operation to calculate logDiffMaxSum_i
Step 3 (RowReductionForSoftmaxAndCrossEntropy): calculate tmp_i_j = softmax'_i_j
- logDiffMaxSum_i, and finally get softmax_i_j and cross\_entropy_i
*/

// There are 3 kinds of reduce algorithms in cub:
// BLOCK_REDUCE_RAKING_COMMUTATIVE_ONLY
// BLOCK_REDUCE_RAKING
// BLOCK_REDUCE_WARP_REDUCTIONS (default)
template <typename T, int BlockDim>
using BlockReduce =
    cub::BlockReduce<T, BlockDim /*, cub::BLOCK_REDUCE_WARP_REDUCTIONS*/>;

template <typename T, int BlockDim>
using BlockReduceTempStorage = typename BlockReduce<T, BlockDim>::TempStorage;

// Make sure that BlockDim <= axis_dim
// This kernel is used to calculate the max element of each row
template <typename T, int BlockDim>
static __global__ void RowReductionForMax(const T* logits_data, T* max_data,
                                          int d, int axis_dim) {
  __shared__ BlockReduceTempStorage<T, BlockDim> temp_storage;

  // logits_data view as [n, axis_dim, remain]
  // max_data view as [n, 1, remain]
  // blockDim = n * remain, split blockIdx to idx_n and idx_remain
  int remain = d / axis_dim;
  int idx_n = blockIdx.x / remain;
  int idx_remain = blockIdx.x % remain;
  int beg_idx = idx_n * d + threadIdx.x * remain + idx_remain;
  int end_idx = (idx_n + 1) * d;

  int step = BlockDim * remain;
  T cur_max = logits_data[beg_idx];
  beg_idx += step;
  while (beg_idx < end_idx) {
    if (cur_max < logits_data[beg_idx]) {
      cur_max = logits_data[beg_idx];
    }
    beg_idx += step;
  }

  cur_max = BlockReduce<T, BlockDim>(temp_storage).Reduce(cur_max, cub::Max());

  if (threadIdx.x == 0) max_data[blockIdx.x] = cur_max;
}

// Make sure that BlockDim <= axis_dim
template <typename T, int BlockDim, bool CalculateLogSoftmax = false>
static __global__ void RowReductionForDiffMaxSum(const T* logits_data,
                                                 T* max_data, T* softmax, int d,
                                                 int axis_dim) {
  __shared__ BlockReduceTempStorage<T, BlockDim> temp_storage;

  // logits, softmax data view as [n, axis_dim, remain]
  // max_data view as [n, 1, remain]
  // blockDim = n * remain, split blockIdx to idx_n and idx_remain
  int remain = d / axis_dim;
  int idx_n = blockIdx.x / remain;
  int idx_remain = blockIdx.x % remain;
  int beg_idx = idx_n * d + threadIdx.x * remain + idx_remain;
  int end_idx = (idx_n + 1) * d;

  auto block_max = max_data[blockIdx.x];
  int step = BlockDim * remain;

  // In numeric stable mode softmax_with_loss, we calc loss with
  // tmp_i_j = x_i_j - max_i - logDiffMaxSum_i, instead of
  // log(exp(x_i_j - max_i)/DiffMaxSum_i). Therefore, log(0) will not occur.
  // Also we calc softmax_i_j = e^{tmp_i_j}, the maximum and minimum value will
  // be 1.0 and 0.0, represent prob is 1.0 and 0.0.
  // So there is no need to clip on shift_softmax.
  softmax[beg_idx] = logits_data[beg_idx] - block_max;
  T diff_max_sum = exp_on_device(softmax[beg_idx]);
  auto idx = beg_idx + step;
  while (idx < end_idx) {
    softmax[idx] = logits_data[idx] - block_max;
    diff_max_sum += exp_on_device(softmax[idx]);
    idx += step;
  }

  diff_max_sum =
      BlockReduce<T, BlockDim>(temp_storage).Reduce(diff_max_sum, cub::Sum());
  if (threadIdx.x == 0) max_data[blockIdx.x] = log_on_device(diff_max_sum);

  if (!CalculateLogSoftmax) return;
  __syncthreads();
  diff_max_sum = max_data[blockIdx.x];
  softmax[beg_idx] -= diff_max_sum;
  beg_idx += step;
  while (beg_idx < end_idx) {
    softmax[beg_idx] -= diff_max_sum;
    beg_idx += step;
  }

  // Note(zhiqiu): since different threads may use max_data[blockIdx.x] to
  // calculate diff_max_sum, __syncthreads() is needed here.
  __syncthreads();
  if (threadIdx.x == 0) max_data[blockIdx.x] = 0;
}

// Make sure that BlockDim <= axis_dim
template <typename T, int BlockDim>
static __global__ void RowReductionForSoftmaxAndCrossEntropy(
    const T* logits_data, const T* labels_data, T* loss_data, T* softmax, int d,
    int axis_dim) {
  __shared__ BlockReduceTempStorage<T, BlockDim> temp_storage;

  // logits, softmax, labels data view as [n, axis_dim, remain]
  // loss_data view as [n, 1, remain]
  // blockDim = n * remain, split blockIdx to idx_n and idx_remain
  int remain = d / axis_dim;
  int idx_n = blockIdx.x / remain;
  int idx_remain = blockIdx.x % remain;
  int beg_idx = idx_n * d + threadIdx.x * remain + idx_remain;
  int end_idx = (idx_n + 1) * d;

  // log_diff_max_sum shares memory with loss
  auto block_log_diff_max_sum = loss_data[blockIdx.x];
  auto tmp = softmax[beg_idx] - block_log_diff_max_sum;
  softmax[beg_idx] = exp_on_device(tmp);
  auto loss = -labels_data[beg_idx] * tmp;
  int step = BlockDim * remain;
  beg_idx += step;
  while (beg_idx < end_idx) {
    tmp = softmax[beg_idx] - block_log_diff_max_sum;
    softmax[beg_idx] = exp_on_device(tmp);
    loss -= (labels_data[beg_idx] * tmp);
    beg_idx += step;
  }

  loss = BlockReduce<T, BlockDim>(temp_storage).Reduce(loss, cub::Sum());
  if (threadIdx.x == 0) loss_data[blockIdx.x] = loss;
}

template <typename T>
struct HardLabelSoftmaxWithCrossEntropyFunctor {
 public:
  HardLabelSoftmaxWithCrossEntropyFunctor(const int64_t* labels, T* loss,
                                          T* log_softmax, int d, int axis_dim)
      : labels_(labels),
        loss_(loss),
        log_softmax_(log_softmax),
        d_(d),
        axis_dim_(axis_dim) {}

  __device__ void operator()(int idx) const {
    // logits view as [n, axis_dim, remain], where d = axis_dim * remain
    int remain = d_ / axis_dim_;
    int idx_n = idx / d_;
    int idx_axis = (idx % d_) / remain;
    int idx_remain = idx % remain;
    // labels, loss view as [n, remain]
    int idx_lbl = idx_n * remain + idx_remain;
    if (idx_axis != labels_[idx_lbl]) {
      log_softmax_[idx] = exp_on_device(log_softmax_[idx]);
    } else {
      auto softmax = log_softmax_[idx];
      log_softmax_[idx] = exp_on_device(softmax);
      loss_[idx_lbl] = -softmax;
    }
  }

 private:
  const int64_t* labels_;
  T* loss_;
  T* log_softmax_;
  int d_;
  int axis_dim_;
};

template <typename T>
struct HardLabelSoftmaxWithCrossEntropyFunctorWithIgnoreIdx {
 public:
  HardLabelSoftmaxWithCrossEntropyFunctorWithIgnoreIdx(const int64_t* labels,
                                                       T* loss, T* log_softmax,
                                                       int d, int axis_dim,
                                                       int ignore_idx)
      : labels_(labels),
        loss_(loss),
        log_softmax_(log_softmax),
        d_(d),
        axis_dim_(axis_dim),
        ignore_idx_(ignore_idx) {}

  __device__ void operator()(int idx) const {
    // logits view as [n, axis_dim, remain], where d = axis_dim * remain
    int remain = d_ / axis_dim_;
    int idx_n = idx / d_;
    int idx_axis = (idx % d_) / remain;
    int idx_remain = idx % remain;
    // labels, loss view as [n, remain]
    int idx_lbl = idx_n * remain + idx_remain;
    if (idx_axis != labels_[idx_lbl] || idx_axis == ignore_idx_) {
      log_softmax_[idx] = exp_on_device(log_softmax_[idx]);
    } else {
      auto softmax = log_softmax_[idx];
      log_softmax_[idx] = exp_on_device(softmax);
      loss_[idx_lbl] = -softmax;
    }
  }

 private:
  const int64_t* labels_;
  T* loss_;
  T* log_softmax_;
  int d_;
  int axis_dim_;
  int ignore_idx_;
};

template <typename T>
static void HardLabelSoftmaxWithCrossEntropy(
    const platform::CUDADeviceContext& ctx, const T* logits_data,
    const int64_t* labels_data, T* loss_data, T* softmax_data, int n, int d,
    int axis_dim, int ignore_idx) {
  constexpr int kMaxBlockDim = 512;
  int block_dim = axis_dim >= kMaxBlockDim
                      ? kMaxBlockDim
                      : (1 << static_cast<int>(std::log2(axis_dim)));
  int grid_dim = n * d / axis_dim;
  auto stream = ctx.stream();

#define CALL_HARD_LABEL_SOFTMAX_WITH_CROSS_ENTROPY_FUSED_KERNEL(BlockDim)  \
  case BlockDim: {                                                         \
    RowReductionForMax<T, BlockDim><<<grid_dim, BlockDim, 0, stream>>>(    \
        logits_data, loss_data, d, axis_dim);                              \
    RowReductionForDiffMaxSum<T, BlockDim,                                 \
                              true><<<grid_dim, BlockDim, 0, stream>>>(    \
        logits_data, loss_data, softmax_data, d, axis_dim);                \
    platform::ForRange<platform::CUDADeviceContext> for_range(ctx, n* d);  \
    if (ignore_idx >= 0 && ignore_idx < axis_dim) {                        \
      for_range(HardLabelSoftmaxWithCrossEntropyFunctorWithIgnoreIdx<T>(   \
          labels_data, loss_data, softmax_data, d, axis_dim, ignore_idx)); \
    } else {                                                               \
      for_range(HardLabelSoftmaxWithCrossEntropyFunctor<T>(                \
          labels_data, loss_data, softmax_data, d, axis_dim));             \
    }                                                                      \
  } break

  switch (block_dim) {
    CALL_HARD_LABEL_SOFTMAX_WITH_CROSS_ENTROPY_FUSED_KERNEL(512);
    CALL_HARD_LABEL_SOFTMAX_WITH_CROSS_ENTROPY_FUSED_KERNEL(256);
    CALL_HARD_LABEL_SOFTMAX_WITH_CROSS_ENTROPY_FUSED_KERNEL(128);
    CALL_HARD_LABEL_SOFTMAX_WITH_CROSS_ENTROPY_FUSED_KERNEL(64);
    CALL_HARD_LABEL_SOFTMAX_WITH_CROSS_ENTROPY_FUSED_KERNEL(32);
    CALL_HARD_LABEL_SOFTMAX_WITH_CROSS_ENTROPY_FUSED_KERNEL(16);
    CALL_HARD_LABEL_SOFTMAX_WITH_CROSS_ENTROPY_FUSED_KERNEL(8);
    CALL_HARD_LABEL_SOFTMAX_WITH_CROSS_ENTROPY_FUSED_KERNEL(4);
    CALL_HARD_LABEL_SOFTMAX_WITH_CROSS_ENTROPY_FUSED_KERNEL(2);
    default:
      PADDLE_THROW(platform::errors::Unavailable(
          "Block Dimension must be 2^n in softmax_with_cross_entropy_op."));
      break;
  }
#undef CALL_HARD_LABEL_SOFTMAX_WITH_CROSS_ENTROPY_FUSED_KERNEL
}

template <typename T>
static void SoftmaxWithCrossEntropyFusedKernel(const T* logits_data,
                                               const T* labels_data,
                                               T* softmax_data, T* loss_data,
                                               int n, int d, int axis_dim,
                                               cudaStream_t stream) {
  constexpr int kMaxBlockDim = 512;
  int block_dim = axis_dim >= kMaxBlockDim
                      ? kMaxBlockDim
                      : (1 << static_cast<int>(std::log2(axis_dim)));
  int grid_dim = n * d / axis_dim;

#define CALL_SOFTMAX_WITH_CROSS_ENTROPY_FUSED_KERNEL(BlockDim)                 \
  case BlockDim:                                                               \
    RowReductionForMax<T, BlockDim><<<grid_dim, BlockDim, 0, stream>>>(        \
        logits_data, loss_data, d, axis_dim);                                  \
    RowReductionForDiffMaxSum<T, BlockDim><<<grid_dim, BlockDim, 0, stream>>>( \
        logits_data, loss_data, softmax_data, d, axis_dim);                    \
    RowReductionForSoftmaxAndCrossEntropy<                                     \
        T, BlockDim><<<grid_dim, BlockDim, 0, stream>>>(                       \
        logits_data, labels_data, loss_data, softmax_data, d, axis_dim);       \
    break

  switch (block_dim) {
    CALL_SOFTMAX_WITH_CROSS_ENTROPY_FUSED_KERNEL(512);
    CALL_SOFTMAX_WITH_CROSS_ENTROPY_FUSED_KERNEL(256);
    CALL_SOFTMAX_WITH_CROSS_ENTROPY_FUSED_KERNEL(128);
    CALL_SOFTMAX_WITH_CROSS_ENTROPY_FUSED_KERNEL(64);
    CALL_SOFTMAX_WITH_CROSS_ENTROPY_FUSED_KERNEL(32);
    CALL_SOFTMAX_WITH_CROSS_ENTROPY_FUSED_KERNEL(16);
    CALL_SOFTMAX_WITH_CROSS_ENTROPY_FUSED_KERNEL(8);
    CALL_SOFTMAX_WITH_CROSS_ENTROPY_FUSED_KERNEL(4);
    CALL_SOFTMAX_WITH_CROSS_ENTROPY_FUSED_KERNEL(2);
    default:
      PADDLE_THROW(platform::errors::Unavailable(
          "Block Dimension must be 2^n in softmax_with_cross_entropy_op."));
      break;
  }

#undef CALL_SOFTMAX_WITH_CROSS_ENTROPY_FUSED_KERNEL
}

template <typename T>
class SoftmaxWithCrossEntropyCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    PADDLE_ENFORCE_EQ(
        platform::is_gpu_place(context.GetPlace()), true,
        platform::errors::Unavailable("softmax_with_cross_entropy operator's "
                                      "CUDA kernel only runs on GPU device."));
    const Tensor* logits = context.Input<Tensor>("Logits");
    const Tensor* labels = context.Input<Tensor>("Label");
    Tensor* softmax = context.Output<Tensor>("Softmax");
    Tensor* loss = context.Output<Tensor>("Loss");

    const int rank = logits->dims().size();
    const int axis = CanonicalAxis(context.Attr<int>("axis"), rank);
    int axis_dim = logits->dims()[axis];

    const int n = SizeToAxis(axis, logits->dims());
    const int d = SizeFromAxis(axis, logits->dims());

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
      SoftmaxWithCrossEntropyFusedKernel(
          logits_data, labels_data, softmax_data, loss_data, n, d, axis_dim,
          context.cuda_device_context().stream());
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
        HardLabelSoftmaxWithCrossEntropy<T>(
            context.cuda_device_context(), logits_data, labels_data, loss_data,
            softmax_data, n, d, axis_dim, ignore_index);
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

    const int n = SizeToAxis(axis, logit_grad->dims());
    const int d = SizeFromAxis(axis, logit_grad->dims());
    const int remain = d / axis_dim;

    int block = 512;
    auto stream = context.cuda_device_context().stream();
    auto ignore_index = context.Attr<int>("ignore_index");
    if (context.Attr<bool>("soft_label")) {
      int grid = (n * d + block - 1) / block;
      const T* label_data = labels->data<T>();
      SoftCrossEntropyGradientKernel<T><<<grid, block, 0, stream>>>(
          logit_grad_data, loss_grad_data, label_data, n, d, remain);
    } else {
      int grid = (n * remain + block - 1) / block;
      const int64_t* label_data = labels->data<int64_t>();
      CrossEntropyGrad<T><<<grid, block, 0, stream>>>(
          logit_grad_data, label_data, n, d, remain, ignore_index);
      int num = n * d;
      grid = (num + block - 1) / block;
      Scale<T><<<grid, block, 0, stream>>>(logit_grad_data, loss_grad_data, num,
                                           d, remain);
    }
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_CUDA_KERNEL(
    softmax_with_cross_entropy, ops::SoftmaxWithCrossEntropyCUDAKernel<float>,
    ops::SoftmaxWithCrossEntropyCUDAKernel<paddle::platform::float16>,
    ops::SoftmaxWithCrossEntropyCUDAKernel<double>);
REGISTER_OP_CUDA_KERNEL(
    softmax_with_cross_entropy_grad,
    ops::SoftmaxWithCrossEntropyGradCUDAKernel<float>,
    ops::SoftmaxWithCrossEntropyGradCUDAKernel<paddle::platform::float16>,
    ops::SoftmaxWithCrossEntropyGradCUDAKernel<double>);
