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

#include "paddle/fluid/operators/npu_op_runner.h"
#include "paddle/fluid/operators/optimizers/adam_op.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
using LoDTensor = framework::LoDTensor;

template <typename DeviceContext, typename T>
class AdamNPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    const auto* param_var = ctx.InputVar("Param");
    PADDLE_ENFORCE_EQ(param_var->IsType<framework::LoDTensor>(), true,
                      platform::errors::InvalidArgument(
                          "The Var(%s)'s type should be LoDTensor, "
                          "but the received is %s",
                          ctx.InputNames("Param").front(),
                          framework::ToTypeName(param_var->Type())));
    T epsilon = static_cast<T>(ctx.Attr<float>("epsilon"));
    auto* param = ctx.Input<LoDTensor>("Param");
    auto* grad_var = ctx.InputVar("Grad");
    PADDLE_ENFORCE_EQ(grad_var->IsType<framework::LoDTensor>(), true,
                      platform::errors::InvalidArgument(
                          "The Grad(%s)'s type should be LoDTensor, "
                          "but the received is %s",
                          ctx.InputNames("Grad").front(),
                          framework::ToTypeName(param_var->Type())));
    auto* grad = ctx.Input<LoDTensor>("Grad");
    auto* mom1 = ctx.Input<LoDTensor>("Moment1");
    auto* mom2 = ctx.Input<LoDTensor>("Moment2");
    auto* lr = ctx.Input<LoDTensor>("LearningRate");

    auto* beta1_pow = ctx.Input<LoDTensor>("Beta1Pow");
    auto* beta2_pow = ctx.Input<LoDTensor>("Beta2Pow");

    auto* param_out = ctx.Output<LoDTensor>("ParamOut");
    auto* mom1_out = ctx.Output<LoDTensor>("Moment1Out");
    auto* mom2_out = ctx.Output<LoDTensor>("Moment2Out");
    auto* beta1_pow_out = ctx.Output<LoDTensor>("Beta1PowOut");
    auto* beta2_pow_out = ctx.Output<LoDTensor>("Beta2PowOut");

    param_out->mutable_data<T>(ctx.GetPlace());
    mom1_out->mutable_data<T>(ctx.GetPlace());
    mom2_out->mutable_data<T>(ctx.GetPlace());
    beta1_pow_out->mutable_data<T>(ctx.GetPlace());
    beta2_pow_out->mutable_data<T>(ctx.GetPlace());

    T beta1 = static_cast<T>(ctx.Attr<float>("beta1"));
    if (ctx.HasInput("Beta1Tensor")) {
      auto* beta1_tensor = ctx.Input<framework::Tensor>("Beta1Tensor");
      PADDLE_ENFORCE_EQ(beta1_tensor->numel(), 1,
                        platform::errors::InvalidArgument(
                            "Input(Beta1Tensor) size must be 1, but get %d",
                            beta1_tensor->numel()));
      beta1 = static_cast<T>(GetAttrFromTensor(beta1_tensor));
    }
    T beta2 = static_cast<T>(ctx.Attr<float>("beta2"));
    if (ctx.HasInput("Beta2Tensor")) {
      auto* beta2_tensor = ctx.Input<framework::Tensor>("Beta2Tensor");
      PADDLE_ENFORCE_EQ(beta2_tensor->numel(), 1,
                        platform::errors::InvalidArgument(
                            "Input(Beta2Tensor) size must be 1, but get %d",
                            beta2_tensor->numel()));
      beta2 = static_cast<T>(GetAttrFromTensor(beta2_tensor));
    }
    VLOG(3) << "beta1_pow.numel() : " << beta1_pow->numel()
            << "beta2_pow.numel() : " << beta2_pow->numel();
    VLOG(3) << "param.numel(): " << param->numel();

    PADDLE_ENFORCE_EQ(beta1_pow_out->numel(), 1,
                      platform::errors::InvalidArgument(
                          "beta1 pow output size should be 1, but received "
                          "value is:%d.",
                          beta1_pow_out->numel()));

    PADDLE_ENFORCE_EQ(beta2_pow_out->numel(), 1,
                      platform::errors::InvalidArgument(
                          "beta2 pow output size should be 1, but received "
                          "value is:%d.",
                          beta2_pow_out->numel()));

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
    auto stream =
        ctx.template device_context<paddle::platform::NPUDeviceContext>()
            .stream();
    auto runner =
        NpuOpRunner("ApplyAdamD",
                    {
                        *param, *mom1, *mom2, *beta1_pow, *beta2_pow, *lr,
                        beta1_tensor, beta2_tensor, epsilon_tensor, *grad,
                    },
                    {
                        *param_out, *mom1_out, *mom2_out,
                    },
                    {});
    runner.Run(stream);

    // NOTE(zhiqiu): ApplyAdamD updates params inplace, so
    // if param and param_out is not same, we need to do copy.
    if (param_out->data<T>() != param->data<T>()) {
      ctx.template device_context<paddle::platform::NPUDeviceContext>().Wait();
      framework::TensorCopySync(*param, ctx.GetPlace(), param_out);
    }
    if (mom1_out->data<T>() != mom1->data<T>()) {
      ctx.template device_context<paddle::platform::NPUDeviceContext>().Wait();
      framework::TensorCopySync(*mom1, ctx.GetPlace(), mom1_out);
    }
    if (mom2_out->data<T>() != mom2->data<T>()) {
      ctx.template device_context<paddle::platform::NPUDeviceContext>().Wait();
      framework::TensorCopySync(*mom2, ctx.GetPlace(), mom2_out);
    }
    auto runner_m1 =
        NpuOpRunner("Mul", {*beta1_pow, beta1_tensor}, {*beta1_pow_out}, {});
    runner_m1.Run(stream);
    auto runner_m2 =
        NpuOpRunner("Mul", {*beta2_pow, beta2_tensor}, {*beta2_pow_out}, {});
    runner_m2.Run(stream);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OP_NPU_KERNEL(
    adam, ops::AdamNPUKernel<paddle::platform::NPUDeviceContext, float>,
    ops::AdamNPUKernel<paddle::platform::NPUDeviceContext,
                       paddle::platform::float16>);
