/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/operators/optimizers/lamb_op.h"
#include "gflags/gflags.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;

#ifdef PADDLE_WITH_XPU
template <typename DeviceContext, typename T>
class LambOpXPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    using paddle::framework::LoDTensor;
    const auto* param_var = ctx.InputVar("Param");
    PADDLE_ENFORCE_EQ(param_var->IsType<framework::LoDTensor>(), true,
                      platform::errors::InvalidArgument(
                          "The Var(%s)'s type should be LoDTensor, "
                          "but the received is %s",
                          ctx.InputNames("Param").front(),
                          framework::ToTypeName(param_var->Type())));

    using paddle::framework::LoDTensor;

    // inputs
    T epsilon = static_cast<T>(ctx.Attr<float>("epsilon"));
    T weight_decay = static_cast<T>(ctx.Attr<float>("weight_decay"));
    T beta1 = static_cast<T>(ctx.Attr<float>("beta1"));
    T beta2 = static_cast<T>(ctx.Attr<float>("beta2"));
    auto& param = GET_DATA_SAFELY(ctx.Input<LoDTensor>("Param"), "Input",
                                  "Param", "Lamb");
    auto* grad_var = ctx.InputVar("Grad");
    auto& mom1 = GET_DATA_SAFELY(ctx.Input<LoDTensor>("Moment1"), "Input",
                                 "Moment1", "Lamb");
    auto& mom2 = GET_DATA_SAFELY(ctx.Input<LoDTensor>("Moment2"), "Input",
                                 "Moment2", "Lamb");
    auto& lr = GET_DATA_SAFELY(ctx.Input<LoDTensor>("LearningRate"), "Input",
                               "LearningRate", "Lamb");

    auto& beta1_pow = GET_DATA_SAFELY(ctx.Input<LoDTensor>("Beta1Pow"), "Input",
                                      "Beta1Pow", "Lamb");
    auto& beta2_pow = GET_DATA_SAFELY(ctx.Input<LoDTensor>("Beta2Pow"), "Input",
                                      "Beta2Pow", "Lamb");

    auto& param_out = GET_DATA_SAFELY(ctx.Output<LoDTensor>("ParamOut"),
                                      "Output", "ParamOut", "Lamb");
    auto& mom1_out = GET_DATA_SAFELY(ctx.Output<LoDTensor>("Moment1Out"),
                                     "Output", "Moment1Out", "Lamb");
    auto& mom2_out = GET_DATA_SAFELY(ctx.Output<LoDTensor>("Moment2Out"),
                                     "Output", "Moment2Out", "Lamb");
    auto& beta1_pow_out = GET_DATA_SAFELY(ctx.Output<LoDTensor>("Beta1PowOut"),
                                          "Output", "Beta1PowOut", "Lamb");
    auto& beta2_pow_out = GET_DATA_SAFELY(ctx.Output<LoDTensor>("Beta2PowOut"),
                                          "Output", "Beta2PowOut", "Lamb");
    auto& dev_ctx = ctx.template device_context<DeviceContext>();

    if (grad_var->IsType<framework::LoDTensor>()) {
      auto& grad = *ctx.Input<LoDTensor>("Grad");
      int r = xpu::lamb(dev_ctx.x_context(), grad.template data<T>(),
                        mom1.template data<T>(), mom2.template data<T>(),
                        param.template data<T>(), beta1_pow.template data<T>(),
                        beta2_pow.template data<T>(), beta1, beta2, epsilon,
                        weight_decay, lr.template data<T>(),
                        mom1_out.template mutable_data<T>(ctx.GetPlace()),
                        mom2_out.template mutable_data<T>(ctx.GetPlace()),
                        param_out.template mutable_data<T>(ctx.GetPlace()),
                        beta1_pow_out.template mutable_data<T>(ctx.GetPlace()),
                        beta2_pow_out.template mutable_data<T>(ctx.GetPlace()),
                        param.numel());

      if (r == xpu::Error_t::INVALID_PARAM) {
        PADDLE_ENFORCE_EQ(
            r, xpu::Error_t::SUCCESS,
            platform::errors::InvalidArgument(
                "XPU kernel error of LambOp, error message: INVALID_PARAM, "
                "please check your input & output."));
      } else if (r == xpu::Error_t::RUNTIME_ERROR) {
        PADDLE_ENFORCE_EQ(r, xpu::Error_t::SUCCESS,
                          platform::errors::Unavailable(
                              "XPU kernel error of LambOp, error message: "
                              "RUNTIME_ERROR, please check whether Baidu "
                              "Kunlun Card is properly installed."));
      } else if (r == xpu::Error_t::NO_ENOUGH_WORKSPACE) {
        PADDLE_ENFORCE_EQ(r, xpu::Error_t::SUCCESS,
                          platform::errors::ResourceExhausted(
                              "XPU kernel error of LambOp, error "
                              "message: NO_ENOUGH_WORKSPACE, XPU "
                              "has no enough memory."));
      } else {
        PADDLE_ENFORCE_EQ(r, xpu::Error_t::SUCCESS,
                          platform::errors::ResourceExhausted(
                              "XPU kernel error of LambOp, error "
                              "message: OTHER "
                              "XPU API returns error code: %d.",
                              r));
      }
    } else {
      PADDLE_THROW(platform::errors::InvalidArgument(
          "Variable type not supported by lamb_op. Expect LoDTensor, "
          "but got %s",
          framework::ToTypeName(param_var->Type())));
    }
  }
};
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_XPU_KERNEL(
    lamb, ops::LambOpXPUKernel<paddle::platform::XPUDeviceContext, float>);
#endif
