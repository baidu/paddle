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
    auto* backprop = ctx.Output<Tensor>("Backprop");
    auto soft_label = ctx.Attr<bool>("soft_label");
    PADDLE_ENFORCE_EQ(soft_label, false,
                      platform::errors::Unimplemented(
                          "soft_label=True is not supported in "
                          "the npu kernel of softmax_with_cross_entropy."));

    const int rank = logits->dims().size();
    const int axis = CanonicalAxis(ctx.Attr<int>("axis"), rank);
    const int n = SizeToAxis(axis, logits->dims());
    const int d = SizeFromAxis(axis, logits->dims());

    PADDLE_ENFORCE_EQ(
        labels->numel(), n,
        platform::errors::Unimplemented(
            "The size of labels should be equal to SizeToAxis of logits,"
            "but got size of labels is %d and SizeToAxis is %d.",
            labels->numel(), n));

    loss->mutable_data<T>(ctx.GetPlace());
    backprop->mutable_data<T>(ctx.GetPlace());
    softmax->mutable_data<T>(ctx.GetPlace());

    Tensor logits_2d, labels_1d, loss_1d, backprop_2d, softmax_2d;
    logits_2d.ShareDataWith(*logits).Resize({n, d});
    labels_1d.ShareDataWith(*labels).Resize({n});
    loss_1d.ShareDataWith(*loss).Resize({n});
    backprop_2d.ShareDataWith(*backprop).Resize({n, d});
    softmax_2d.ShareDataWith(*softmax).Resize({n, d});

    auto stream =
        ctx.template device_context<paddle::platform::NPUDeviceContext>()
            .stream();

    std::vector<int> axes;
    for (auto i = axis; i < logits->dims().size(); ++i) {
      axes.push_back(i);
    }
    const auto& runner_softmax =
        NpuOpRunner("SoftmaxV2", {*logits}, {*softmax}, {{"axes", axes}});
    runner_softmax.Run(stream);

    // SparseSoftmaxCrossEntropyWithLogits
    const auto& runner_s =
        NpuOpRunner("SparseSoftmaxCrossEntropyWithLogits",
                    {logits_2d, labels_1d}, {loss_1d, backprop_2d}, {});
    runner_s.Run(stream);
  }
};

template <typename DeviceContext, typename T>
class SoftmaxWithCrossEntropyGradNPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* backprop = ctx.Input<Tensor>("Backprop");
    auto* loss_grad = ctx.Input<Tensor>(framework::GradVarName("Loss"));
    auto* logits_grad = ctx.Output<Tensor>(framework::GradVarName("Logits"));

    PADDLE_ENFORCE_NOT_NULL(backprop,
                            platform::errors::PreconditionNotMet(
                                "backprop should not be null in NPU kernel of "
                                "softmax_with_cross_entropy_grad."));
    logits_grad->mutable_data<T>(ctx.GetPlace());

    const int rank = logits_grad->dims().size();
    const int axis = CanonicalAxis(ctx.Attr<int>("axis"), rank);
    const int n = SizeToAxis(axis, logits_grad->dims());
    const int d = SizeFromAxis(axis, logits_grad->dims());

    Tensor logits_grad_2d, loss_grad_1d, backprop_2d;

    logits_grad_2d.ShareDataWith(*logits_grad).Resize({n, d});
    loss_grad_1d.ShareDataWith(*loss_grad).Resize({n});
    backprop_2d.ShareDataWith(*backprop).Resize({n, d});

    auto stream =
        ctx.template device_context<paddle::platform::NPUDeviceContext>()
            .stream();
    const auto& runner_mul =
        NpuOpRunner("Mul", {*loss_grad, *backprop}, {*logits_grad}, {});
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
