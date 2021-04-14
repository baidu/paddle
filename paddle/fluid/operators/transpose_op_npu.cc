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

#include <iostream>
#include <memory>
#include <string>

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/expand_op.h"
#include "paddle/fluid/operators/npu_op_runner.h"

namespace paddle {
namespace operators {

template <typename DeviceContext, typename T>
class TransposeNPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* x = ctx.Input<framework::LoDTensor>("X");
    auto* out = ctx.Output<framework::LoDTensor>("Out");
    std::vector<int> axis = ctx.Attr<std::vector<int>>("axis");
    framework::NPUAttributeMap attr_input = {{"perm", axis}};
    out->mutable_data<T>(ctx.device_context().GetPlace());
    auto runner = NpuOpRunner("TransposeD", {*x}, {*out}, attr_input);
    auto stream =
        ctx.template device_context<paddle::platform::NPUDeviceContext>()
            .stream();
    runner.Run(stream);
  }
};

template <typename T>
class TransposeGradNPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* out_grad =
        ctx.Input<framework::LoDTensor>(framework::GradVarName("Out"));
    auto* x_grad =
        ctx.Output<framework::LoDTensor>(framework::GradVarName("X"));
    std::vector<int> axis = ctx.Attr<std::vector<int>>("axis");
    std::vector<int> reversed_axis(axis);
    for (size_t i = 0; i < axis.size(); i++) {
      reversed_axis[axis[i]] = i;
    }
    x_grad->mutable_data<T>(ctx.GetPlace());
    framework::NPUAttributeMap attr_input = {{"perm", reversed_axis}};
    auto runner = NpuOpRunner("TransposeD", {*out_grad}, {*x_grad}, attr_input);
    auto stream =
        ctx.template device_context<paddle::platform::NPUDeviceContext>()
            .stream();
    runner.Run(stream);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OP_NPU_KERNEL(
    transpose2,
    ops::TransposeNPUKernel<paddle::platform::NPUDeviceContext, float>,
    ops::TransposeNPUKernel<paddle::platform::NPUDeviceContext,
                            paddle::platform::float16>,
    ops::TransposeNPUKernel<paddle::platform::NPUDeviceContext, int>,
    ops::TransposeNPUKernel<paddle::platform::NPUDeviceContext, uint8_t>,
    ops::TransposeNPUKernel<paddle::platform::NPUDeviceContext, int8_t>);

REGISTER_OP_NPU_KERNEL(transpose2_grad, ops::TransposeGradNPUKernel<float>,
                       ops::TransposeGradNPUKernel<paddle::platform::float16>,
                       ops::TransposeGradNPUKernel<int>,
                       ops::TransposeGradNPUKernel<uint8_t>,
                       ops::TransposeGradNPUKernel<int8_t>);
