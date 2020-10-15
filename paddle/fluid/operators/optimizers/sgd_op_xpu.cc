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
#include "paddle/fluid/operators/optimizers/sgd_op.h"
#include <string>
namespace paddle {
namespace operators {

template <typename DeviceContext, typename T>
class SGDOpXPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    const auto *learning_rate = ctx.Input<framework::Tensor>("LearningRate");

    const auto *param_var = ctx.InputVar("Param");
    const auto *grad_var = ctx.InputVar("Grad");

    if (param_var->IsType<framework::LoDTensor>() &&
        grad_var->IsType<framework::LoDTensor>()) {
      const auto *param = ctx.Input<framework::Tensor>("Param");
      auto *param_out = ctx.Output<framework::Tensor>("ParamOut");
      // Actually, all tensors are LoDTensor except SelectedRows.
      const auto *grad = ctx.Input<framework::Tensor>("Grad");
      auto sz = param_out->numel();
      PADDLE_ENFORCE_EQ(param->numel(), sz,
                        platform::errors::InvalidArgument(
                            "The input tensor Param's numel of SgdOp "
                            "should be equal with ParamOut's numel. "
                            "But received Param's "
                            "numel = [%s], ParamOut's numel = [%s]",
                            param->numel(), sz));
      PADDLE_ENFORCE_EQ(grad->numel(), sz,
                        platform::errors::InvalidArgument(
                            "The input tensor Grad's numel of SgdOp "
                            "should be equal with ParamOut's numel. "
                            "But received Grad's "
                            "numel = [%s], ParamOut's numel = [%s]",
                            grad->numel(), sz));

      const T *lr = learning_rate->data<T>();
      const T *param_data = param->data<T>();
      const T *grad_data = grad->data<T>();
      T *out_data = param_out->mutable_data<T>(ctx.GetPlace());

      auto &dev_ctx = ctx.template device_context<DeviceContext>();
      int r = xpu::sgd(dev_ctx.x_context(), sz, grad_data, param_data, lr,
                       out_data);
      PADDLE_ENFORCE_EQ(
          r, xpu::Error_t::SUCCESS,
          platform::errors::PermissionDenied("XPU kernel error!"));
    } else {
      PADDLE_ENFORCE_EQ(false, true,
                        platform::errors::PermissionDenied(
                            "Unsupported Variable Type of Param & Grad in "
                            "SgdOp-XPU. Excepted "
                            "LodTensor, But received [%s] and [%s]",
                            paddle::framework::ToTypeName(param_var->Type())));
    }
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_XPU_KERNEL(
    sgd, ops::SGDOpXPUKernel<paddle::platform::XPUDeviceContext, float>);
#endif
