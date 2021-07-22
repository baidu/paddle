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
#include "paddle/fluid/operators/optimizers/momentum_op.h"

#include "paddle/fluid/operators/npu_op_runner.h"
#include "paddle/fluid/operators/optimizers/sgd_op.h"

namespace paddle {
namespace operators {

template <typename T>
class NPUMomentumOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto& dev_ctx = ctx.template device_context<platform::NPUDeviceContext>();

    std::string regularization_method =
        ctx.Attr<std::string>("regularization_method");
    auto regularization_coeff = ctx.Attr<float>("regularization_coeff");
    RegularizationType regularization_flag{
        RegularizationType::kNONE};  // disable regularization
    if (regularization_method == "l2_decay") {
      regularization_flag = RegularizationType::kL2DECAY;
    }

    T mu = static_cast<T>(ctx.Attr<float>("mu"));
    bool use_nesterov = ctx.Attr<bool>("use_nesterov");

    auto learning_rate = ctx.Input<framework::Tensor>("LearningRate");
    auto param = ctx.Input<framework::Tensor>("Param");
    auto velocity = ctx.Input<framework::Tensor>("Velocity");

    auto param_out = ctx.Output<framework::Tensor>("ParamOut");
    auto velocity_out = ctx.Output<framework::Tensor>("VelocityOut");

    param_out->mutable_data<T>(ctx.GetPlace());
    velocity_out->mutable_data<T>(ctx.GetPlace());

    auto* grad_var = ctx.InputVar("Grad");
    if (grad_var->IsType<framework::LoDTensor>()) {
      auto grad = ctx.Input<framework::Tensor>("Grad");
      Tensor mu_tensor;
      mu_tensor.mutable_data<T>(framework::make_ddim({1}), ctx.GetPlace());
      FillNpuTensorWithConstant<T>(&mu_tensor, mu);

      Tensor regularized_grad;
      if (regularization_flag == RegularizationType::kL2DECAY) {
        regularized_grad.mutable_data<T>(grad->dims(), ctx.GetPlace());
        const auto& runner1 = NpuOpRunner("Muls", {*param}, {regularized_grad},
                                          {{"value", regularization_coeff}});
        runner1.Run(dev_ctx.stream());
        const auto& runner2 = NpuOpRunner("Add", {regularized_grad, *grad},
                                          {regularized_grad}, {});
        runner2.Run(dev_ctx.stream());
      } else {
        regularized_grad.ShareDataWith(*grad);
      }
      framework::TensorCopy(*param, ctx.GetPlace(), dev_ctx, param_out);
      framework::TensorCopy(*velocity, ctx.GetPlace(), dev_ctx, velocity_out);
      // NOTE: ApplyMomentum will change the input
      const auto& runner = NpuOpRunner(
          "ApplyMomentum", {*param_out, *velocity_out, *learning_rate,
                            regularized_grad, mu_tensor},
          {*param_out}, {{"use_nesterov", use_nesterov}});
      runner.Run(dev_ctx.stream());
    } else if (grad_var->IsType<framework::SelectedRows>()) {
      PADDLE_ENFORCE_EQ(false, true, platform::errors::PermissionDenied(
                                         "Unsupport SparseMomentum"));
    } else {
      PADDLE_ENFORCE_EQ(false, true,
                        platform::errors::PermissionDenied(
                            "Unsupported Variable Type of Grad "
                            "in MomentumOp. Excepted LodTensor "
                            "or SelectedRows, But received [%s]",
                            paddle::framework::ToTypeName(grad_var->Type())));
    }
  }
};
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;
REGISTER_OP_NPU_KERNEL(momentum, ops::NPUMomentumOpKernel<float>,
                       ops::NPUMomentumOpKernel<plat::float16>);
