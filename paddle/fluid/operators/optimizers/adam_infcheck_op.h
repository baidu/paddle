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

#pragma once
#include <memory>
#include "paddle/fluid/operators/optimizers/adam_op.h"

namespace paddle {
namespace operators {
template <typename T>
inline T GetDataToCPU(const framework::Tensor& var) {
  if (platform::is_cpu_place(var.place())) {
    return var.data<T>()[0];
  }
  std::unique_ptr<framework::Tensor> cpu_var{new framework::Tensor()};
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  framework::TensorCopySync(var, platform::CPUPlace(), cpu_var.get());
#else
  PADDLE_THROW(platform::errors::PreconditionNotMet(
      "This version of PaddlePaddle does NOT support GPU but got GPU tensor "
      "Var in AdamInfCheckOp. Please compile WITH_GPU option."));
#endif
  return cpu_var->data<T>()[0];
}

template <typename T>
inline void CopyTensorSameContext(const framework::ExecutionContext& ctx,
                                  std::string src, std::string dst) {
  auto* param_in = ctx.Input<framework::Tensor>(src);
  auto* param_out = ctx.Output<framework::Tensor>(dst);
  param_out->mutable_data<T>(ctx.GetPlace());
  framework::TensorCopy(*param_in, ctx.GetPlace(), ctx.device_context(),
                        param_out);
}

class AdamInfCheckOp : public AdamOp {
 public:
  using AdamOp::AdamOp;
  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE_EQ(ctx->HasInput("InfCheck"), true,
                      platform::errors::NotFound(
                          "Input(InfCheck) of AdamOp should not be null."));
    auto dims = ctx->GetInputDim("InfCheck");
    PADDLE_ENFORCE_EQ(framework::product(dims), 1,
                      platform::errors::InvalidArgument(
                          "Infcheck should have 1 dimension, but received %d",
                          framework::product(dims)));
    return AdamOp::InferShape(ctx);
  }
};

template <typename DeviceContext, typename T>
class AdamInfCheckOpKernel : public framework::OpKernel<T> {
 public:
  AdamInfCheckOpKernel()
      : _adam_op_kernel(new AdamOpKernel<DeviceContext, T>()) {}
  void Compute(const framework::ExecutionContext& ctx) const override {
    const auto infcheck_tensor = ctx.Input<framework::Tensor>("InfCheck");
    const auto infcheck_flag = GetDataToCPU<bool>(*infcheck_tensor);
    if (infcheck_flag) {
      VLOG(3) << "The CPU input tensor exit inf or nan";
      ctx.device_context().Wait();
      CopyTensorSameContext<T>(ctx, "Param", "ParamOut");
      CopyTensorSameContext<T>(ctx, "Moment1", "Moment1Out");
      CopyTensorSameContext<T>(ctx, "Moment2", "Moment2Out");
      CopyTensorSameContext<T>(ctx, "Beta1Pow", "Beta1PowOut");
      CopyTensorSameContext<T>(ctx, "Beta2Pow", "Beta2PowOut");
      return;
    }
    VLOG(3) << "The CPU input tensor not exit inf or nan";
    return _adam_op_kernel->Compute(ctx);
  }

 private:
  std::unique_ptr<AdamOpKernel<DeviceContext, T>> _adam_op_kernel;
};

}  // namespace operators
}  // namespace paddle
