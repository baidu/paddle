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
#include "paddle/fluid/operators/optimizers/adam_op.h"
#include <memory>
#include <string>
#include "paddle/fluid/operators/npu_op_runner.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;

template <typename DeviceContext, typename T>
class AdamNPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    // to do: select_rows
    auto* grad = ctx.Input<framework::Tensor>("Grad");
    auto* param = ctx.Input<framework::Tensor>("Param");
    auto* moment1 = ctx.Input<framework::Tensor>("Moment1");
    auto* moment2 = ctx.Input<framework::Tensor>("Moment2");
    auto* lr = ctx.Input<framework::Tensor>("LearningRate");
    auto* beta1_pow = ctx.Input<framework::Tensor>("Beta1Pow");
    auto* beta2_pow = ctx.Input<framework::Tensor>("Beta2Pow");
    T beta1 = static_cast<T>(ctx.Attr<float>("beta1"));
    T beta2 = static_cast<T>(ctx.Attr<float>("beta2"));
    T epsilon = static_cast<T>(ctx.Attr<float>("epsilon"));

    auto* param_out = ctx.Output<framework::Tensor>("ParamOut");
    param_out->mutable_data<T>(ctx.GetPlace());
    // reshape
    Tensor beta1_tensor(framework::proto::VarType::FP32);
    beta1_tensor.mutable_data<float>({1}, ctx.GetPlace());
    TensorFromVector(std::vector<T>{beta1}, ctx.device_context(),
                     &beta1_tensor);
    Tensor beta2_tensor(framework::proto::VarType::FP32);
    beta2_tensor.mutable_data<float>({1}, ctx.GetPlace());
    TensorFromVector(std::vector<T>{beta2}, ctx.device_context(),
                     &beta2_tensor);

    Tensor epsilon_tensor(framework::proto::VarType::FP32);
    epsilon_tensor.mutable_data<T>({1}, ctx.GetPlace());
    TensorFromVector(std::vector<T>{epsilon}, ctx.device_context(),
                     &epsilon_tensor);
    std::vector<framework::Tensor> inputs_vec = {
        *param, *moment1,     *moment2,     *beta1_pow,     *beta2_pow,
        *lr,    beta1_tensor, beta2_tensor, epsilon_tensor, *grad};
    // inputs_vec.push_back(*param, *moment1, *moment2, *beta1_pow, *beta2_pow);
    auto stream =
        ctx.template device_context<paddle::platform::NPUDeviceContext>()
            .stream();
    auto runner = NpuOpRunner("ApplyAdam", inputs_vec, {*param_out}, {});
    runner.Run(stream);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OP_NPU_KERNEL(
    adam, ops::AdamNPUKernel<paddle::platform::NPUDeviceContext, float>,
    ops::AdamNPUKernel<paddle::platform::NPUDeviceContext,
                       paddle::platform::float16>);
#endif
