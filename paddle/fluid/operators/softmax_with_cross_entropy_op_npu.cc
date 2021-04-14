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

    int cls_num = logits->dims()[1];
    const int rank = logits->dims().size();
    const int axis = CanonicalAxis(ctx.Attr<int>("axis"), rank);
    std::vector<int> axes;
    for (auto i = axis; i < logits->dims().size(); ++i) {
      axes.push_back(i);
    }

    auto stream =
        ctx.template device_context<paddle::platform::NPUDeviceContext>()
            .stream();

    // softmax
    softmax->mutable_data<T>(ctx.GetPlace());
    auto runner_softmax =
        NpuOpRunner("SoftmaxV2", {*logits}, {*softmax}, {{"axes", axes}});
    runner_softmax.Run(stream);

    // cast label from int64/int32 to int32
    Tensor tmp_labels(framework::proto::VarType::INT32);
    if (labels->type() != framework::proto::VarType::INT32) {
      tmp_labels.Resize(labels->dims());
      tmp_labels.mutable_data(ctx.GetPlace(), framework::proto::VarType::INT32);
      auto dst_dtype = ConvertToNpuDtype(framework::proto::VarType::INT32);
      auto runner_cast_label =
          NpuOpRunner("Cast", {*labels}, {tmp_labels},
                      {{"dst_type", static_cast<int>(dst_dtype)}});
      runner_cast_label.Run(stream);
      labels = &tmp_labels;
    }

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
        NpuOpRunner("OneHotD", {*labels, on_tensor, off_tensor}, {tmp_onehot},
                    {{"axis", -1}, {"depth", cls_num}});
    runner_onehot.Run(stream);

    // cast one_hot from int32 to T
    Tensor cast_onehot(logits->type());
    cast_onehot.Resize(tmp_onehot.dims());
    cast_onehot.mutable_data<T>(ctx.GetPlace());
    auto dst_dtype = ConvertToNpuDtype(logits->type());
    auto runner_cast_onehot =
        NpuOpRunner("Cast", {tmp_onehot}, {cast_onehot},
                    {{"dst_type", static_cast<int>(dst_dtype)}});
    runner_cast_onehot.Run(stream);

    // SoftmaxCrossEntropyWithLogits
    Tensor backprop(logits->type());
    backprop.Resize(logits->dims());
    backprop.mutable_data<T>(ctx.GetPlace());

    loss->mutable_data<T>(ctx.GetPlace());

    // SoftmaxCrossEntropyWithLogits requires loss to be of shape [batch_size]
    auto loss_dims = loss->dims();
    loss->Resize({loss_dims[0]});
    auto runner_s = NpuOpRunner("SoftmaxCrossEntropyWithLogits",
                                {*logits, cast_onehot}, {*loss, backprop}, {});
    runner_s.Run(stream);
    loss->Resize(loss_dims);
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

    // cast label from int64/int32 to int32
    Tensor tmp_labels(framework::proto::VarType::INT32);
    if (labels->type() != framework::proto::VarType::INT32) {
      tmp_labels.Resize(labels->dims());
      tmp_labels.mutable_data(ctx.GetPlace(), framework::proto::VarType::INT32);
      auto dst_dtype = ConvertToNpuDtype(framework::proto::VarType::INT32);
      auto runner_cast_label =
          NpuOpRunner("Cast", {*labels}, {tmp_labels},
                      {{"dst_type", static_cast<int>(dst_dtype)}});
      runner_cast_label.Run(stream);
      labels = &tmp_labels;
    }

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
    tmp_onehot.Resize(softmax->dims());
    tmp_onehot.mutable_data<int>(ctx.GetPlace());

    auto runner_onehot =
        NpuOpRunner("OneHotD", {*labels, on_tensor, off_tensor}, {tmp_onehot},
                    {{"axis", -1}, {"depth", cls_num}});
    runner_onehot.Run(stream);

    // cast one_hot from int32 to T
    Tensor cast_onehot(softmax->type());
    cast_onehot.Resize(tmp_onehot.dims());
    cast_onehot.mutable_data<T>(ctx.GetPlace());
    auto dst_dtype = ConvertToNpuDtype(softmax->type());
    auto runner_cast_onehot =
        NpuOpRunner("Cast", {tmp_onehot}, {cast_onehot},
                    {{"dst_type", static_cast<int>(dst_dtype)}});
    runner_cast_onehot.Run(stream);

    // sub
    Tensor tmp_sub(softmax->type());
    tmp_sub.Resize(softmax->dims());
    tmp_sub.mutable_data<T>(ctx.GetPlace());
    auto runner_sub =
        NpuOpRunner("Sub", {*softmax, cast_onehot}, {tmp_sub}, {});

    runner_sub.Run(stream);
    // mul
    logits_grad->mutable_data<T>(ctx.GetPlace());
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
