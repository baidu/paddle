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

#ifdef PADDLE_WITH_HIP
#include <hipcub/hipcub.hpp>
namespace cub = hipcub;
#else
#include <cub/cub.cuh>
#endif

#include <vector>
#include "paddle/fluid/operators/amp/fp16_type_traits.h"
#include "paddle/fluid/operators/margin_softmax_with_cross_entropy_op.h"
#include "paddle/fluid/operators/math/math_function.h"
#include "paddle/fluid/operators/math/softmax_impl.h"
#include "paddle/fluid/string/string_helper.h"

#if defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL)
#include "paddle/fluid/platform/collective_helper.h"
#include "paddle/fluid/platform/nccl_helper.h"
#endif

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;

static constexpr int kNumCUDAThreads = 512;
static constexpr int kNumMaxinumNumBlocks = 4096;

static inline int NumBlocks(const int N) {
  return std::min((N + kNumCUDAThreads - 1) / kNumCUDAThreads,
                  kNumMaxinumNumBlocks);
}

void GetClassInterval(const gpuStream_t& stream, const platform::Place& place,
                      const platform::DeviceContext& ctx, const int rid,
                      const int rank, const int nranks, const int D,
                      Tensor* class_interval) {
  std::vector<int> shard_dim_vec(nranks + 1, 0);
  shard_dim_vec[rank + 1] = D;
  if (nranks <= 1) {
    framework::TensorFromVector(shard_dim_vec, ctx, class_interval);
    return;
  }

#if defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL)
  Tensor num_classes_per_device;
  framework::TensorFromVector(shard_dim_vec, ctx, &num_classes_per_device);
  int* num_classes_per_device_ptr = num_classes_per_device.data<int>();

  const auto& comm = platform::NCCLCommContext::Instance().Get(rid, place);
  // use global calculate stream
  const auto calcu_stream =
      static_cast<platform::CUDADeviceContext*>(
          platform::DeviceContextPool::Instance().Get(place))
          ->stream();

  PADDLE_ENFORCE_CUDA_SUCCESS(platform::dynload::ncclAllReduce(
      num_classes_per_device_ptr, num_classes_per_device_ptr,
      num_classes_per_device.numel(),
      platform::ToNCCLDataType(num_classes_per_device.type()), ncclSum,
      comm->comm(), calcu_stream));

  auto class_interval_ptr =
      class_interval->mutable_data<int>({nranks + 1}, place);
  size_t cub_temp_storage_bytes = 0;
  cub::DeviceScan::InclusiveSum<int*, int*>(
      nullptr, cub_temp_storage_bytes, nullptr, nullptr, nranks + 1, stream);
  auto cub_temp_storage = memory::Alloc(place, cub_temp_storage_bytes);
  cub::DeviceScan::InclusiveSum<int*, int*>(
      cub_temp_storage->ptr(), cub_temp_storage_bytes,
      num_classes_per_device_ptr, class_interval_ptr, nranks + 1, stream);
  return;
#endif
}

template <typename T, typename IndexT>
__global__ void AddMarginToPositiveLogits(
    T* logit, const IndexT* label, const float margin1, const float margin2,
    const float margin3, const int rank, const int nranks, const int64_t N,
    const int64_t D, const int* class_interval_ptr) {
  using MPType = typename details::MPTypeTrait<T>::Type;
  int start_index = class_interval_ptr[rank];
  int end_index = class_interval_ptr[rank + 1];
  int num_classes = class_interval_ptr[nranks];
  CUDA_KERNEL_LOOP(i, N) {
    auto real_label = label[i];
    PADDLE_ENFORCE((real_label < num_classes) && (real_label >= 0),
                   "The index is out of bounds, "
                   "please check whether the value of label and "
                   "input meet the number of class. It should "
                   "be less than [%d], but received [%d]",
                   num_classes, real_label);

    if (real_label >= start_index && real_label < end_index) {
      int64_t offset = i * D + real_label - start_index;
      if (fabs(margin1 - 1.0) > 1e-8 || fabs(margin2) > 1e-8) {
        MPType x = static_cast<MPType>(logit[offset]);
        MPType theta = acos(x);
        if (fabs(margin1 - 1.0) > 1e-8) {
          theta *= static_cast<MPType>(margin1);
        }
        if (fabs(margin2) > 1e-8) {
          theta += static_cast<MPType>(margin2);
        }
        logit[offset] = static_cast<T>(cos(theta));
      }
      if (fabs(margin3) > 1e-8) {
        MPType y = static_cast<MPType>(logit[offset]);
        y -= static_cast<MPType>(margin3);
        logit[offset] = static_cast<T>(y);
      }
    }
  }
}

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

template <typename T, typename IndexT>
__global__ void HardLabelSoftmaxWithCrossEntropyKernel(
    T* loss, T* log_softmax, const IndexT* labels, const int rank,
    const int64_t N, const int64_t D, const int* class_interval_ptr) {
  int start_index = class_interval_ptr[rank];
  CUDA_KERNEL_LOOP(i, N * D) {
    auto row = i / D;
    auto col = i % D;
    if ((col + start_index) == labels[row]) {
      auto softmax = log_softmax[i];
      loss[row] = -softmax;
      log_softmax[i] = exp_on_device(softmax);
    } else {
      log_softmax[i] = exp_on_device(log_softmax[i]);
    }
  }
}

template <typename T, typename IndexT>
__global__ void CalculateGrad(T* logits_grad, const T* loss_grad,
                              const T* logits, const IndexT* labels,
                              const float margin1, const float margin2,
                              const float scale, const int rank,
                              const int64_t N, const int64_t D,
                              const int* class_interval_ptr) {
  using MPType = typename details::MPTypeTrait<T>::Type;
  int start_index = class_interval_ptr[rank];
  CUDA_KERNEL_LOOP(i, N * D) {
    auto row = i / D;
    auto col = i % D;
    if ((col + start_index) == labels[row]) {
      logits_grad[i] = (logits_grad[i] - static_cast<T>(1.0)) * loss_grad[row];
      if (fabs(margin1 - 1.0) > 1e-8 || fabs(margin2) > 1e-8) {
        MPType dout = static_cast<MPType>(logits_grad[i]);
        MPType one = static_cast<MPType>(1.0f);
        MPType x = static_cast<MPType>(logits[i]);
        MPType m1 = static_cast<MPType>(margin1);
        MPType m2 = static_cast<MPType>(margin2);

        MPType d = m1 * sin(m1 * acos(x) + m2) / sqrt(one - x * x);
        logits_grad[i] = static_cast<T>(dout * d);
      }
    } else {
      logits_grad[i] *= loss_grad[row];
    }
    if (fabs(scale - 1.0) > 1e-8) {
      logits_grad[i] *= static_cast<T>(scale);
    }
  }
}

template <typename T>
class MarginSoftmaxWithCrossEntropyOpCUDAKernel
    : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    const Tensor* logits = ctx.Input<Tensor>("Logits");
    const Tensor* labels = ctx.Input<Tensor>("Label");
    Tensor* softmax = ctx.Output<Tensor>("Softmax");
    Tensor* loss = ctx.Output<Tensor>("Loss");

    const int rid = ctx.Attr<int>("ring_id");
    const int nranks = ctx.Attr<int>("nranks");
    const int rank = ctx.Attr<int>("rank");

    const float margin1 = ctx.Attr<float>("margin1");
    const float margin2 = ctx.Attr<float>("margin2");
    const float margin3 = ctx.Attr<float>("margin3");
    const float scale = ctx.Attr<float>("scale");

    const auto& place = ctx.GetPlace();
    auto& dev_ctx = ctx.template device_context<platform::CUDADeviceContext>();

#if defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL)
    platform::NCCLComm* comm;
    gpuStream_t stream;
    if (nranks > 1) {
      comm = platform::NCCLCommContext::Instance().Get(rid, place);

      // use global calculate stream
      stream = static_cast<platform::CUDADeviceContext*>(
                   platform::DeviceContextPool::Instance().Get(place))
                   ->stream();
    }
#endif

    // allocate memory on device.
    softmax->mutable_data<T>(place);
    loss->mutable_data<T>(place);

    const auto& logits_dims = logits->dims();
    const auto& labels_dims = labels->dims();

    const int axis = logits_dims.size() - 1;
    const int N = SizeToAxis(axis, logits_dims);
    const int D = SizeFromAxis(axis, logits_dims);

    Eigen::DSizes<int, 2> batch_by_one(N, 1);
    Eigen::DSizes<int, 2> one_by_class(1, D);

    int blocks = NumBlocks(N);
    int threads = kNumCUDAThreads;
    const auto& label_type = labels->type();

    Tensor class_interval;
    GetClassInterval(dev_ctx.stream(), place, ctx.cuda_device_context(), rid,
                     rank, nranks, D, &class_interval);

    // step 0, copy logits to softmax variable since we need logits
    // when calculate grad
    framework::TensorCopy(*logits, ctx.GetPlace(), ctx.device_context(),
                          softmax);

    Tensor softmax_2d, loss_2d;
    softmax_2d.ShareDataWith(*softmax).Resize({N, D});
    loss_2d.ShareDataWith(*loss).Resize({N, 1});

    auto eigen_softmax = math::EigenMatrix<T>::From(softmax_2d);

    // step 1, add margin for positive elements
    // theta = acos(x_i)
    // s * (cos(m1 * theta + m2) - m3)
    if (label_type == framework::proto::VarType::INT32) {
      AddMarginToPositiveLogits<
          T, int32_t><<<blocks, threads, 0, dev_ctx.stream()>>>(
          softmax_2d.data<T>(), labels->data<int32_t>(), margin1, margin2,
          margin3, rank, nranks, N, D, class_interval.data<int>());
    } else if (label_type == framework::proto::VarType::INT64) {
      AddMarginToPositiveLogits<
          T, int64_t><<<blocks, threads, 0, dev_ctx.stream()>>>(
          softmax_2d.data<T>(), labels->data<int64_t>(), margin1, margin2,
          margin3, rank, nranks, N, D, class_interval.data<int>());
    }
    if (fabs(scale - 1.0) > 1e-8) {
      Tensor scale_t;
      scale_t.mutable_data<T>({N, D}, ctx.GetPlace());
      math::SetConstant<platform::CUDADeviceContext, T>()(
          dev_ctx, &scale_t, static_cast<T>(scale));
      auto eigen_scale = math::EigenMatrix<T>::From(scale_t);

      eigen_softmax.device(*dev_ctx.eigen_device()) =
          eigen_softmax * eigen_scale;
    }

    // step 2, obtain logit_max
    Tensor logits_max;
    logits_max =
        ctx.AllocateTmpTensor<T, platform::CUDADeviceContext>({N, 1}, dev_ctx);
    void* logits_max_buff = logits_max.mutable_data<T>(place);

    auto eigen_logits_max = math::EigenMatrix<T>::From(logits_max);
    Eigen::DSizes<int, 1> along_axis(1);
    eigen_logits_max.device(*dev_ctx.eigen_device()) =
        eigen_softmax.maximum(along_axis);
#if defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL)
    if (nranks > 1) {
      PADDLE_ENFORCE_CUDA_SUCCESS(platform::dynload::ncclAllReduce(
          logits_max_buff, logits_max_buff, logits_max.numel(),
          platform::ToNCCLDataType(logits_max.type()), ncclMax, comm->comm(),
          stream));
    }
#endif

    // step 3, logit - logit_max
    eigen_softmax.device(*dev_ctx.eigen_device()) =
        (eigen_softmax -
         eigen_logits_max.reshape(batch_by_one).broadcast(one_by_class));

    // step 4, sum(exp(logit - logit_max))
    Tensor sum_exp_logits;
    sum_exp_logits =
        ctx.AllocateTmpTensor<T, platform::CUDADeviceContext>({N, 1}, dev_ctx);
    void* sum_exp_logits_buff = sum_exp_logits.mutable_data<T>(place);

    auto eigen_sum_exp_logits = math::EigenMatrix<T>::From(sum_exp_logits);
    eigen_sum_exp_logits.device(*dev_ctx.eigen_device()) =
        eigen_softmax.exp().sum(along_axis);

#if defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL)
    if (nranks > 1) {
      PADDLE_ENFORCE_CUDA_SUCCESS(platform::dynload::ncclAllReduce(
          sum_exp_logits_buff, sum_exp_logits_buff, sum_exp_logits.numel(),
          platform::ToNCCLDataType(sum_exp_logits.type()), ncclSum,
          comm->comm(), stream));
    }
#endif

    // step 5, (logit - logit_max) - log(sum(exp(logit - logit_max)))
    eigen_softmax.device(*dev_ctx.eigen_device()) =
        (eigen_softmax -
         eigen_sum_exp_logits.log()
             .reshape(batch_by_one)
             .broadcast(one_by_class));

    // step 6, prob = exp((logit - logit_max) - log(sum(exp(logit -
    // logit_max))))
    // loss = -((logit_i - logit_max) - log(sum(exp(logit - logit_max))))
    void* loss_buff = loss_2d.mutable_data<T>(ctx.GetPlace());
    math::SetConstant<platform::CUDADeviceContext, T>()(dev_ctx, &loss_2d,
                                                        static_cast<T>(0.0));
    if (label_type == framework::proto::VarType::INT32) {
      HardLabelSoftmaxWithCrossEntropyKernel<
          T, int32_t><<<blocks, threads, 0, dev_ctx.stream()>>>(
          loss_2d.data<T>(), softmax_2d.data<T>(), labels->data<int32_t>(),
          rank, N, D, class_interval.data<int>());
    } else if (label_type == framework::proto::VarType::INT64) {
      HardLabelSoftmaxWithCrossEntropyKernel<
          T, int64_t><<<blocks, threads, 0, dev_ctx.stream()>>>(
          loss_2d.data<T>(), softmax_2d.data<T>(), labels->data<int64_t>(),
          rank, N, D, class_interval.data<int>());
    }
#if defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL)
    if (nranks > 1) {
      PADDLE_ENFORCE_CUDA_SUCCESS(platform::dynload::ncclAllReduce(
          loss_buff, loss_buff, loss_2d.numel(),
          platform::ToNCCLDataType(loss_2d.type()), ncclSum, comm->comm(),
          stream));
    }
#endif
  }
};

template <typename T>
class MarginSoftmaxWithCrossEntropyGradCUDAKernel
    : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    const Tensor* labels = context.Input<Tensor>("Label");
    const Tensor* logits = context.Input<Tensor>("Logits");
    const Tensor* softmax = context.Input<Tensor>("Softmax");

    const Tensor* loss_grad =
        context.Input<Tensor>(framework::GradVarName("Loss"));
    Tensor* logit_grad =
        context.Output<Tensor>(framework::GradVarName("Logits"));

    const int rid = context.Attr<int>("ring_id");
    const int nranks = context.Attr<int>("nranks");
    const int rank = context.Attr<int>("rank");

    const float margin1 = context.Attr<float>("margin1");
    const float margin2 = context.Attr<float>("margin2");
    const float margin3 = context.Attr<float>("margin3");
    const float scale = context.Attr<float>("scale");

    auto& dev_ctx =
        context.template device_context<platform::CUDADeviceContext>();

    if (logit_grad != softmax) {
      framework::TensorCopy(*softmax, context.GetPlace(),
                            context.device_context(), logit_grad);
    }
    const auto sofrmax_dims = softmax->dims();
    const int axis = sofrmax_dims.size() - 1;
    const int N = SizeToAxis(axis, sofrmax_dims);
    const int D = SizeFromAxis(axis, sofrmax_dims);

    Tensor logit_grad_2d;
    logit_grad_2d.ShareDataWith(*logit_grad).Resize({N, D});

    int blocks = NumBlocks(N * D);
    int threads = kNumCUDAThreads;
    const auto& label_type = labels->type();

    Tensor class_interval;
    GetClassInterval(dev_ctx.stream(), context.GetPlace(),
                     context.cuda_device_context(), rid, rank, nranks, D,
                     &class_interval);

    if (label_type == framework::proto::VarType::INT32) {
      CalculateGrad<T, int32_t><<<blocks, threads, 0, dev_ctx.stream()>>>(
          logit_grad_2d.data<T>(), loss_grad->data<T>(), logits->data<T>(),
          labels->data<int32_t>(), margin1, margin2, scale, rank, N, D,
          class_interval.data<int>());
    } else if (label_type == framework::proto::VarType::INT64) {
      CalculateGrad<T, int64_t><<<blocks, threads, 0, dev_ctx.stream()>>>(
          logit_grad_2d.data<T>(), loss_grad->data<T>(), logits->data<T>(),
          labels->data<int64_t>(), margin1, margin2, scale, rank, N, D,
          class_interval.data<int>());
    }
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;

REGISTER_OP_CUDA_KERNEL(
    margin_softmax_with_cross_entropy,
    ops::MarginSoftmaxWithCrossEntropyOpCUDAKernel<float>,
    ops::MarginSoftmaxWithCrossEntropyOpCUDAKernel<double>,
    ops::MarginSoftmaxWithCrossEntropyOpCUDAKernel<plat::float16>);

REGISTER_OP_CUDA_KERNEL(
    margin_softmax_with_cross_entropy_grad,
    ops::MarginSoftmaxWithCrossEntropyGradCUDAKernel<float>,
    ops::MarginSoftmaxWithCrossEntropyGradCUDAKernel<double>,
    ops::MarginSoftmaxWithCrossEntropyGradCUDAKernel<plat::float16>);
