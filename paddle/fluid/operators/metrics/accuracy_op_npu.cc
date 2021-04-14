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

#include <memory>
#include <string>

#include "paddle/fluid/operators/controlflow/compare_op.h"
#include "paddle/fluid/operators/metrics/accuracy_op.h"
#include "paddle/fluid/operators/npu_op_runner.h"

namespace paddle {
namespace operators {

template <typename DeviceContext, typename T>
class AccuracyNPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* pred = ctx.Input<Tensor>("Out");
    auto* label = ctx.Input<Tensor>("Label");
    // auto* logits = ctx.Input<Tensor>("Indices");

    auto* acc = ctx.Output<Tensor>("Accuracy");
    auto* correct = ctx.Output<Tensor>("Correct");
    auto* total = ctx.Output<Tensor>("Total");
    auto stream =
        ctx.template device_context<paddle::platform::NPUDeviceContext>()
            .stream();

    // cast pred
    Tensor tmp_pred(pred->type());
    tmp_pred.Resize(pred->dims());
    tmp_pred.mutable_data<int>(ctx.GetPlace());
    auto runner_cast_pred =
        NpuOpRunner("Cast", {*pred}, {tmp_pred},
                    {{"dst_type", static_cast<int>(ACL_INT32)}});
    runner_cast_pred.Run(stream);

    // cast label
    Tensor tmp_label(label->type());
    tmp_label.Resize(label->dims());
    tmp_label.mutable_data<int>(ctx.GetPlace());
    auto runner_cast_label =
        NpuOpRunner("Cast", {*label}, {tmp_label},
                    {{"dst_type", static_cast<int>(ACL_INT32)}});
    runner_cast_label.Run(stream);

    // equal
    Tensor tmp_equal(label->type());
    tmp_equal.Resize(label->dims());
    tmp_equal.mutable_data<bool>(ctx.GetPlace());
    auto runner_equal =
        NpuOpRunner("Equal", {tmp_pred, tmp_label}, {tmp_equal}, {});
    runner_equal.Run(stream);

    // cast equal
    Tensor tmp_equal_cast(label->type());
    tmp_equal_cast.Resize(label->dims());
    tmp_equal_cast.mutable_data<float>(ctx.GetPlace());
    auto runner_cast_equal =
        NpuOpRunner("Cast", {tmp_equal}, {tmp_equal_cast},
                    {{"dst_type", static_cast<float>(ACL_FLOAT)}});
    runner_cast_equal.Run(stream);

    // acc
    acc->mutable_data<float>(ctx.GetPlace());
    std::vector<int> axes_vec_1;
    auto runner_acc = NpuOpRunner("ReduceMeanD", {tmp_equal_cast}, {*acc},
                                  {{"keep_dims", false}, {"axes", axes_vec_1}});
    runner_acc.Run(stream);

    // correct
    correct->mutable_data<float>(ctx.GetPlace());
    std::vector<int> axes_vec_2;
    auto runner_correct =
        NpuOpRunner("ReduceSumD", {tmp_equal_cast}, {*correct},
                    {{"keep_dims", false}, {"axes", axes_vec_2}});
    runner_correct.Run(stream);

    // ones_tensor
    Tensor ones_tensor(label->type());
    ones_tensor.Resize(label->dims());
    ones_tensor.mutable_data<int>(ctx.GetPlace());
    auto runner_oneslike =
        NpuOpRunner("OnesLike", {tmp_label}, {ones_tensor}, {});
    runner_oneslike.Run(stream);

    // ones_tensor_cast
    Tensor ones_tensor_cast(label->type());
    ones_tensor_cast.Resize(label->dims());
    ones_tensor_cast.mutable_data<float>(ctx.GetPlace());
    auto runner_ones_cast =
        NpuOpRunner("Cast", {ones_tensor}, {ones_tensor_cast},
                    {{"dst_type", static_cast<float>(ACL_FLOAT)}});
    runner_ones_cast.Run(stream);

    // total
    total->mutable_data<float>(ctx.GetPlace());
    std::vector<int> axes_vec_3;
    auto runner_total =
        NpuOpRunner("ReduceSumD", {ones_tensor_cast}, {*total},
                    {{"keep_dims", false}, {"axes", axes_vec_3}});
    runner_total.Run(stream);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OP_NPU_KERNEL(
    accuracy, ops::AccuracyNPUKernel<paddle::platform::NPUDeviceContext, float>,
    ops::AccuracyNPUKernel<paddle::platform::NPUDeviceContext,
                           paddle::platform::float16>,
    ops::AccuracyNPUKernel<paddle::platform::NPUDeviceContext, int>,
    ops::AccuracyNPUKernel<paddle::platform::NPUDeviceContext, int64_t>);
