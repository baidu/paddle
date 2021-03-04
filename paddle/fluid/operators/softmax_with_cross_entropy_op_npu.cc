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

#ifdef PADDLE_WITH_ASCEND_CL
#include "paddle/fluid/operators/math/softmax.h"
#include <memory>
#include <string>
#include "paddle/fluid/operators/math/cross_entropy.h"
#include "paddle/fluid/operators/npu_op_runner.h"
#include "paddle/fluid/operators/softmax_op.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;

template <typename DeviceContext, typename T>
class SoftmaxWithCrossEntropyNPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* logits = ctx.Input<Tensor>("Logits");
    auto* labels = ctx.Input<Tensor>("Label");
    auto* softmax = ctx.Output<Tensor>("Softmax");
    auto* loss = ctx.Output<Tensor>("Loss");

    int cls_num = logits->dims()[1];

    auto stream =
        ctx.template device_context<paddle::platform::NPUDeviceContext>()
            .stream();

    // softmax
    softmax->mutable_data<T>(ctx.GetPlace());
    auto runner_softmax = NpuOpRunner("SoftmaxV2", {*logits}, {*softmax}, {});
    runner_softmax.Run(stream);
    // cast label
    Tensor tmp_labels(labels->type());
    tmp_labels.Resize(labels->dims());
    tmp_labels.mutable_data<T>(ctx.GetPlace());
    auto runner_cast_label = NpuOpRunner("Cast", {*labels}, {tmp_labels},
                                         {{"dst_type", static_cast<int>(3)}});
    runner_cast_label.Run(stream);

    // on and off
    Tensor on_tensor(framework::proto::VarType::INT32);
    on_tensor.mutable_data<int>({1}, ctx.GetPlace());
    TensorFromVector(std::vector<int>{static_cast<int>(1)},
                     ctx.device_context(), &on_tensor);
    Tensor off_tensor(framework::proto::VarType::INT32);
    off_tensor.mutable_data<int>({1}, ctx.GetPlace());
    TensorFromVector(std::vector<int>{static_cast<int>(0)},
                     ctx.device_context(), &off_tensor);

    // one_hot
    Tensor tmp_onehot(on_tensor.type());
    tmp_onehot.Resize(logits->dims());
    tmp_onehot.mutable_data<int>(ctx.GetPlace());

    auto runner_onehot =
        NpuOpRunner("OneHotD", {tmp_labels, on_tensor, off_tensor},
                    {tmp_onehot}, {{"depth", cls_num}});
    runner_onehot.Run(stream);

    // to do squeeze, infer shape first
    PADDLE_ENFORCE_NE(
        logits->dims()[0], 1,
        platform::errors::InvalidArgument("logits first dim should not be 1"));
    PADDLE_ENFORCE_NE(
        logits->dims()[1], 1,
        platform::errors::InvalidArgument("logits secend dim should not be 1"));

    // SoftmaxCrossEntropyWithLogits
    Tensor tmp_scel(logits->type());
    std::vector<int> dim_vec;
    dim_vec.push_back(static_cast<int>(logits->dims()[0]));
    tmp_scel.Resize(framework::make_ddim(dim_vec));
    tmp_scel.mutable_data<T>(ctx.GetPlace());
    auto runner_scel = NpuOpRunner("SoftmaxCrossEntropyWithLogits",
                                   {*logits, tmp_onehot}, {tmp_scel}, {});
    runner_scel.Run(stream);

    // to do cast type

    // Tensor tmp_cast_loss();
    // tmp_scel.Resize(framework::make_ddim(logits->dims()[0]));
    // tmp_scel.mutable_data<T>(ctx.GetPlace());
    // auto runner_scel = NpuOpRunner("SoftmaxCrossEntropyWithLogits", {logits,
    // tmp_onehot}, {tmp_scel}, {});
    // runner_scel.Run(stream);
    std::vector<int> axes_vec;
    axes_vec.push_back(static_cast<int>(1));
    auto runner_unsqueeze =
        NpuOpRunner("Unsqueeze", {tmp_scel}, {*loss}, {{"axes", axes_vec}});
    runner_unsqueeze.Run(stream);
  }
};

template <typename DeviceContext, typename T>
class SoftmaxWithCrossEntropyGradNPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* labels = ctx.Input<Tensor>("Label");
    auto* softmax = ctx.Input<Tensor>("Softmax");
    auto* loss_grad = ctx.Input<Tensor>(framework::GradVarName("Loss"));
    auto* logits_grad = ctx.Output<Tensor>(framework::GradVarName("Logits"));

    int cls_num = softmax->dims()[1];

    auto stream =
        ctx.template device_context<paddle::platform::NPUDeviceContext>()
            .stream();

    // cast label
    Tensor tmp_labels(labels->type());
    tmp_labels.Resize(labels->dims());
    tmp_labels.mutable_data<T>(ctx.GetPlace());
    auto runner_cast_label = NpuOpRunner("Cast", {*labels}, {tmp_labels},
                                         {{"dst_type", static_cast<int>(3)}});
    runner_cast_label.Run(stream);

    // on and off
    Tensor on_tensor(framework::proto::VarType::INT32);
    on_tensor.mutable_data<int>({1}, ctx.GetPlace());
    TensorFromVector(std::vector<int>{static_cast<int>(1)},
                     ctx.device_context(), &on_tensor);
    Tensor off_tensor(framework::proto::VarType::INT32);
    off_tensor.mutable_data<int>({1}, ctx.GetPlace());
    TensorFromVector(std::vector<int>{static_cast<int>(0)},
                     ctx.device_context(), &off_tensor);

    // one_hot
    Tensor tmp_onehot(on_tensor.type());
    tmp_onehot.Resize(logits_grad->dims());
    tmp_onehot.mutable_data<int>(ctx.GetPlace());

    auto runner_onehot =
        NpuOpRunner("OneHotD", {tmp_labels, on_tensor, off_tensor},
                    {tmp_onehot}, {{"depth", cls_num}});
    runner_onehot.Run(stream);

    // sub
    Tensor tmp_sub(softmax->type());
    tmp_sub.Resize(softmax->dims());
    tmp_sub.mutable_data<T>(ctx.GetPlace());
    auto runner_sub = NpuOpRunner("Sub", {*softmax, tmp_onehot}, {tmp_sub}, {});

    runner_sub.Run(stream);
    // mul
    auto runner_mul =
        NpuOpRunner("Mul", {*loss_grad, tmp_sub}, {*logits_grad}, {});
    runner_mul.Run(stream);
  }
};
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OP_NPU_KERNEL(
    softmax_with_cross_entropy,
    ops::SoftmaxWithCrossEntropyNPUKernel<paddle::platform::NPUDeviceContext,
                                          float>,
    ops::SoftmaxWithCrossEntropyNPUKernel<paddle::platform::NPUDeviceContext,
                                          paddle::platform::float16>);
REGISTER_OP_NPU_KERNEL(
    softmax_with_cross_entropy_grad,
    ops::SoftmaxWithCrossEntropyGradNPUKernel<
        paddle::platform::NPUDeviceContext, float>,
    ops::SoftmaxWithCrossEntropyGradNPUKernel<
        paddle::platform::NPUDeviceContext, paddle::platform::float16>);
#endif
