/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the Licnse. */

#ifdef PADDLE_WITH_ASCEND_CL
#include <memory>
#include <string>

#include "paddle/fluid/framework/ddim.h"
#include "paddle/fluid/framework/tensor_util.h"
#include "paddle/fluid/operators/activation_op.h"
#include "paddle/fluid/operators/npu_op_runner.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;

template <typename DeviceContext, typename T>
class PowNPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* x = ctx.Input<Tensor>("X");
    auto* out = ctx.Output<Tensor>("Out");
    auto factor = ctx.Attr<float>("factor");

    out->mutable_data<T>(ctx.GetPlace());

    auto runner = NpuOpRunner("Power", {*x}, {*out},
                              {{"power", factor},
                               {"scale", static_cast<float>(1.0)},
                               {"shift", static_cast<float>(0.0)}});

    auto stream =
        ctx.template device_context<paddle::platform::NPUDeviceContext>()
            .stream();
    runner.Run(stream);
  }
};

template <typename DeviceContext, typename T>
class PowGradNPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* x = ctx.Input<Tensor>("X");
    auto* dout = ctx.Input<Tensor>(framework::GradVarName("Out"));
    auto* dx = ctx.Output<Tensor>(framework::GradVarName("X"));
    auto factor = ctx.Attr<float>("factor");

    auto x_dims = x->dims();

    auto place = ctx.GetPlace();
    auto stream =
        ctx.template device_context<paddle::platform::NPUDeviceContext>()
            .stream();

    // NOTE(liym27): dx = dout * factor * x.pow(factor-1)

    // Step1: Compute x_pow = x.pow(factor-1)
    Tensor x_pow(x->type());
    x_pow.mutable_data<T>(x->dims(), place);
    auto runner_pow = NpuOpRunner("Power", {*x}, {x_pow},
                                  {{"power", factor - static_cast<float>(1)}});
    runner_pow.Run(stream);

    // Step 2: Construct a broadcast factor, which has the same shape with x.
    // 2.1 Get the shape of x
    Tensor x_shape(framework::proto::VarType::INT32);
    x_shape.mutable_data<int32_t>({x_dims.size()}, place);
    TensorFromVector(framework::vectorize<int32_t>(x_dims),
                     ctx.device_context(), &x_shape);

    // 2.2 Get a factor tensor with shape [1].
    Tensor factor_tensor(framework::proto::VarType::FP32);
    factor_tensor.mutable_data<float>({1}, place);
    TensorFromVector(std::vector<float>{factor}, ctx.device_context(),
                     &factor_tensor);

    // 2.3 Get the factor which has the shape with x and the same value with
    // factor.
    Tensor factor_bc_tensor(framework::proto::VarType::FP32);
    factor_bc_tensor.mutable_data<float>(x_dims, place);
    auto runner_bc = NpuOpRunner("BroadcastTo", {factor_tensor, x_shape},
                                 {factor_bc_tensor}, {});
    runner_bc.Run(stream);

    // Step 3: Compute x_power_mul_factor = factor * x.pow(factor-1)
    Tensor x_power_mul_factor(x->type());
    x_power_mul_factor.mutable_data<T>(x->dims(), place);
    auto runner_mul_1 =
        NpuOpRunner("Mul", {factor_bc_tensor, *x}, {x_power_mul_factor}, {});
    runner_mul_1.Run(stream);

    // Step 4: Compute dx = dout * factor * x.pow(factor-1)
    dx->mutable_data<T>(place);
    auto runner_mul_2 =
        NpuOpRunner("Mul", {*dout, x_power_mul_factor}, {*dx}, {});
    runner_mul_2.Run(stream);
  }
};

template <typename DeviceContext, typename T>
class ReluNPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* x = ctx.Input<Tensor>("X");
    auto* out = ctx.Output<Tensor>("Out");

    out->mutable_data<T>(ctx.GetPlace());

    auto runner = NpuOpRunner("Relu",
                              {
                                  *x,
                              },
                              {*out}, {});

    auto stream =
        ctx.template device_context<paddle::platform::NPUDeviceContext>()
            .stream();
    runner.Run(stream);
  }
};

template <typename DeviceContext, typename T>
class ReluGradNPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* x = ctx.Input<Tensor>("X");
    auto* dout = ctx.Input<Tensor>(framework::GradVarName("Out"));
    auto* dx = ctx.Output<Tensor>(framework::GradVarName("X"));

    dx->mutable_data<T>(ctx.GetPlace());

    auto runner = NpuOpRunner("ReluGrad", {*dout, *x}, {*dx}, {});

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
    pow, ops::PowNPUKernel<paddle::platform::NPUDeviceContext, float>,
    ops::PowNPUKernel<paddle::platform::NPUDeviceContext,
                      paddle::platform::float16>);

REGISTER_OP_NPU_KERNEL(
    pow_grad, ops::PowGradNPUKernel<paddle::platform::NPUDeviceContext, float>,
    ops::PowGradNPUKernel<paddle::platform::NPUDeviceContext,
                          paddle::platform::float16>);

REGISTER_OP_NPU_KERNEL(
    relu, ops::ReluNPUKernel<paddle::platform::NPUDeviceContext, float>,
    ops::ReluNPUKernel<paddle::platform::NPUDeviceContext,
                       paddle::platform::float16>);

REGISTER_OP_NPU_KERNEL(
    relu_grad,
    ops::ReluGradNPUKernel<paddle::platform::NPUDeviceContext, float>,
    ops::ReluNPUKernel<paddle::platform::NPUDeviceContext,
                       paddle::platform::float16>);
#endif