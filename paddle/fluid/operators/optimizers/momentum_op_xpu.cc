/* Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */
#ifdef PADDLE_WITH_XPU
#include <string>
#include "paddle/fluid/operators/optimizers/sgd_op.h"
namespace paddle {
namespace operators {

template <typename DeviceContext, typename T>
class MomentumOpXPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    T mu = static_cast<T>(ctx.Attr<float>("mu"));
    bool use_nesterov = ctx.Attr<bool>("use_nesterov");

    auto learning_rate = ctx.Input<framework::Tensor>("LearningRate");
    auto param = ctx.Input<framework::Tensor>("Param");
    auto param_out = ctx.Output<framework::Tensor>("ParamOut");
    auto* velocity = ctx.Input<framework::Tensor>("Velocity");
    auto velocity_out = ctx.Output<framework::Tensor>("VelocityOut");
    param_out->mutable_data<T>(ctx.GetPlace());
    velocity_out->mutable_data<T>(ctx.GetPlace());
    auto* lr = learning_rate->data<T>();

    auto* grad_var = ctx.InputVar("Grad");
    PADDLE_ENFORCE_EQ(grad_var->IsType<framework::LoDTensor>(), true,
                      platform::errors::PermissionDenied(
                          "Unsupported Variable Type of Param & Grad in "
                          "MomentumOp-XPU. Excepted "
                          "LodTensor, But received [%s] and [%s]",
                          paddle::framework::ToTypeName(grad_var->Type())));

    auto grad = ctx.Input<framework::Tensor>("Grad");

    auto& dev_ctx = ctx.template device_context<DeviceContext>();
    int r = xpu::momentum(
        dev_ctx.x_context(), param->data<float>(), velocity->data<float>(),
        grad->data<float>(), lr, use_nesterov, mu, param_out->numel(),
        param_out->data<float>(), velocity_out->data<float>());
    PADDLE_ENFORCE_EQ(r, xpu::Error_t::SUCCESS,
                      platform::errors::PermissionDenied("XPU kernel error!"));
  }
};
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_XPU_KERNEL(
    momentum,
    ops::MomentumOpXPUKernel<paddle::platform::XPUDeviceContext, float>);
#endif
