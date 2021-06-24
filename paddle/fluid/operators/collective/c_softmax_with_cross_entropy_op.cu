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

#include "paddle/fluid/operators/collective/c_softmax_with_cross_entropy_op.h"
#include "paddle/fluid/operators/math/cross_entropy.h"
#include "paddle/fluid/operators/math/softmax_impl.h"
#include "paddle/fluid/platform/collective_helper.h"
#include "paddle/fluid/platform/nccl_helper.h"
#include "paddle/fluid/string/string_helper.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;

static constexpr int kNumCUDAThreads = 512;
static constexpr int kNumMaxinumNumBlocks = 4096;

static inline int NumBlocks(const int N) {
  return std::min((N + kNumCUDAThreads - 1) / kNumCUDAThreads,
                  kNumMaxinumNumBlocks);
}

template <typename T, typename IndexT>
__global__ void MaskLabelByIndex(T* predicted_logits, const T* logit,
                                 const IndexT* label, const int start_index,
                                 const int end_index, const int64_t N,
                                 const int64_t D, const int nranks) {
  CUDA_KERNEL_LOOP(i, N) {
    auto real_label = label[i];
    PADDLE_ENFORCE((real_label < D * nranks) && (real_label >= 0),
                   "The index is out of bounds, "
                   "please check whether the value of label and "
                   "input meet the class number. It should "
                   "be less than [%d], but received [%d]",
                   D * nranks, real_label);

    if (real_label >= start_index && real_label < end_index) {
      predicted_logits[i] = logit[i * D + real_label - start_index];
    }
  }
}

template <typename T, typename IndexT>
__global__ void MaskLabelByIndexGrad(T* logits_grad, const T* loss_grad,
                                     const IndexT* labels,
                                     const int start_index, const int end_index,
                                     const int64_t N, const int64_t D) {
  CUDA_KERNEL_LOOP(i, N * D) {
    auto row = i / D;
    auto col = i % D;
    if ((col + start_index) == labels[row]) {
      logits_grad[i] = (logits_grad[i] - static_cast<T>(1.0)) * loss_grad[row];
    } else {
      logits_grad[i] *= loss_grad[row];
    }
  }
}

template <typename T>
class CSoftmaxWithCrossEntropyOpCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    const Tensor* logits = ctx.Input<Tensor>("Logits");
    const Tensor* labels = ctx.Input<Tensor>("Label");
    Tensor* softmax = ctx.Output<Tensor>("Softmax");
    Tensor* loss = ctx.Output<Tensor>("Loss");

    const int rid = ctx.Attr<int>("ring_id");
    const int nranks = ctx.Attr<int>("nranks");
    const int rank = ctx.Attr<int>("rank");

    const auto& place = ctx.GetPlace();
    const auto& comm = platform::NCCLCommContext::Instance().Get(rid, place);
    auto& dev_ctx = ctx.template device_context<platform::CUDADeviceContext>();

    // use global calculate stream
    const auto stream = static_cast<platform::CUDADeviceContext*>(
                            platform::DeviceContextPool::Instance().Get(place))
                            ->stream();

    // allocate memory on device.
    softmax->mutable_data<T>(place);
    loss->mutable_data<T>(place);

    const auto& logits_dims = logits->dims();
    const auto& labels_dims = labels->dims();

    const int axis = logits_dims.size() - 1;
    const int N = SizeToAxis(axis, logits_dims);
    const int D = SizeFromAxis(axis, logits_dims);

    Tensor logits_2d, softmax_2d, loss_2d;
    logits_2d.ShareDataWith(*logits).Resize({N, D});
    softmax_2d.ShareDataWith(*softmax).Resize({N, D});
    loss_2d.ShareDataWith(*loss).Resize({N, 1});

    auto eigen_logits = math::EigenMatrix<T>::From(logits_2d);
    auto eigen_softmax = math::EigenMatrix<T>::From(softmax_2d);

    // step 1, obtain logit_max
    Tensor logits_max;
    logits_max =
        ctx.AllocateTmpTensor<T, platform::CUDADeviceContext>({N, 1}, dev_ctx);
    void* logits_max_buff = logits_max.mutable_data<T>(place);

    auto eigen_logits_max = math::EigenMatrix<T>::From(logits_max);
    Eigen::DSizes<int, 1> along_axis(1);
    eigen_logits_max.device(*dev_ctx.eigen_device()) =
        eigen_logits.maximum(along_axis);
    PADDLE_ENFORCE_CUDA_SUCCESS(platform::dynload::ncclAllReduce(
        logits_max_buff, logits_max_buff, logits_max.numel(),
        platform::ToNCCLDataType(logits_max.type()), ncclMax, comm->comm(),
        stream));

    // step 2, obtain logit - logit_max
    Eigen::DSizes<int, 2> batch_by_one(N, 1);
    Eigen::DSizes<int, 2> one_by_class(1, D);

    eigen_softmax.device(*dev_ctx.eigen_device()) =
        (eigen_logits -
         eigen_logits_max.reshape(batch_by_one).broadcast(one_by_class))
            .unaryExpr(math::ValueClip<T>());

    // step 3, obtain predict target
    Tensor predicted_logits;
    predicted_logits =
        ctx.AllocateTmpTensor<T, platform::CUDADeviceContext>({N, 1}, dev_ctx);
    predicted_logits.mutable_data<T>(place);

    auto t = framework::EigenVector<T>::Flatten(predicted_logits);
    t.device(*dev_ctx.eigen_device()) = t.constant(static_cast<T>(0));

    const int start_index = rank * D;
    const int end_index = start_index + D;

    int blocks = NumBlocks(N);
    int threads = kNumCUDAThreads;
    const auto& label_type = labels->type();

    if (label_type == framework::proto::VarType::INT32) {
      MaskLabelByIndex<T, int32_t><<<blocks, threads, 0, dev_ctx.stream()>>>(
          predicted_logits.data<T>(), softmax_2d.data<T>(),
          labels->data<int32_t>(), start_index, end_index, N, D, nranks);
    } else if (label_type == framework::proto::VarType::INT64) {
      MaskLabelByIndex<T, int64_t><<<blocks, threads, 0, dev_ctx.stream()>>>(
          predicted_logits.data<T>(), softmax_2d.data<T>(),
          labels->data<int64_t>(), start_index, end_index, N, D, nranks);
    }

    void* predict_logits_buff = predicted_logits.mutable_data<T>(place);
    PADDLE_ENFORCE_CUDA_SUCCESS(platform::dynload::ncclAllReduce(
        predict_logits_buff, predict_logits_buff, predicted_logits.numel(),
        platform::ToNCCLDataType(predicted_logits.type()), ncclSum,
        comm->comm(), stream));

    // step 4, obtain exp(logit)
    eigen_softmax.device(*dev_ctx.eigen_device()) = eigen_softmax.exp();

    // step 5, obtain sum_exp_logits
    Tensor sum_exp_logits;
    sum_exp_logits =
        ctx.AllocateTmpTensor<T, platform::CUDADeviceContext>({N, 1}, dev_ctx);
    void* sum_exp_logits_buff = sum_exp_logits.mutable_data<T>(place);

    auto eigen_sum_exp_logits = math::EigenMatrix<T>::From(sum_exp_logits);
    eigen_sum_exp_logits.device(*dev_ctx.eigen_device()) =
        eigen_softmax.sum(along_axis);

    PADDLE_ENFORCE_CUDA_SUCCESS(platform::dynload::ncclAllReduce(
        sum_exp_logits_buff, sum_exp_logits_buff, sum_exp_logits.numel(),
        platform::ToNCCLDataType(sum_exp_logits.type()), ncclSum, comm->comm(),
        stream));

    auto eigen_loss = math::EigenMatrix<T>::From(loss_2d);
    auto eigen_predicted_logits = math::EigenMatrix<T>::From(predicted_logits);

    eigen_loss.device(*dev_ctx.eigen_device()) =
        (eigen_sum_exp_logits.log().unaryExpr(math::TolerableValue<T>()) -
         eigen_predicted_logits)
            .unaryExpr(math::TolerableValue<T>());

    eigen_softmax.device(*dev_ctx.eigen_device()) =
        (eigen_softmax *
         eigen_sum_exp_logits.inverse().broadcast(one_by_class));
  }
};

template <typename T>
class CSoftmaxWithCrossEntropyGradCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    const Tensor* labels = context.Input<Tensor>("Label");
    const Tensor* loss_grad =
        context.Input<Tensor>(framework::GradVarName("Loss"));
    Tensor* logit_grad =
        context.Output<Tensor>(framework::GradVarName("Logits"));
    const Tensor* softmax = context.Input<Tensor>("Softmax");
    const int rank = context.Attr<int>("rank");
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
    const int start_index = rank * D;
    const int end_index = start_index + D;

    if (label_type == framework::proto::VarType::INT32) {
      MaskLabelByIndexGrad<T,
                           int32_t><<<blocks, threads, 0, dev_ctx.stream()>>>(
          logit_grad_2d.data<T>(), loss_grad->data<T>(),
          labels->data<int32_t>(), start_index, end_index, N, D);
    } else if (label_type == framework::proto::VarType::INT64) {
      MaskLabelByIndexGrad<T,
                           int64_t><<<blocks, threads, 0, dev_ctx.stream()>>>(
          logit_grad_2d.data<T>(), loss_grad->data<T>(),
          labels->data<int64_t>(), start_index, end_index, N, D);
    }
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;

REGISTER_OP_CUDA_KERNEL(
    c_softmax_with_cross_entropy,
    ops::CSoftmaxWithCrossEntropyOpCUDAKernel<float>,
    ops::CSoftmaxWithCrossEntropyOpCUDAKernel<double>,
    ops::CSoftmaxWithCrossEntropyOpCUDAKernel<plat::float16>);

REGISTER_OP_CUDA_KERNEL(
    c_softmax_with_cross_entropy_grad,
    ops::CSoftmaxWithCrossEntropyGradCUDAKernel<float>,
    ops::CSoftmaxWithCrossEntropyGradCUDAKernel<paddle::platform::float16>,
    ops::CSoftmaxWithCrossEntropyGradCUDAKernel<double>);
