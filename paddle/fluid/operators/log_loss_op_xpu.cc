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

#include "paddle/fluid/operators/log_loss_op.h"
#include <memory>
namespace paddle {
namespace operators {

template <typename DeviceContext, typename T, typename AttrType = T>
class LogLossXPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* predict = ctx.Input<Tensor>("Predicted");
    auto* labels = ctx.Input<Tensor>("Labels");
    auto* loss = ctx.Output<Tensor>("Loss");
    auto epsilon = static_cast<T>(ctx.Attr<AttrType>("epsilon"));
    loss->mutable_data<T>(ctx.GetPlace());
    int n = predict->numel();
    auto& dev_ctx = ctx.template device_context<DeviceContext>();
    int r =
        xpu::log_loss_fwd(dev_ctx.x_context(), n, epsilon, predict->data<T>(),
                          labels->data<T>(), loss->data<T>());
    PADDLE_ENFORCE_EQ(r == xpu::Error_t::SUCCESS, true,
                      platform::errors::InvalidArgument("XPU kernel error!"));
  }
};
template <typename DeviceContext, typename T, typename AttrType = T>
class LogLossGradXPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* predict = ctx.Input<Tensor>("Predicted");
    auto* labels = ctx.Input<Tensor>("Labels");
    auto* dloss = ctx.Input<Tensor>(framework::GradVarName("Loss"));
    auto* dpred = ctx.Output<Tensor>(framework::GradVarName("Predicted"));
    if (!dpred) {
      return;
    }
    auto epsilon = static_cast<T>(ctx.Attr<AttrType>("epsilon"));
    dpred->mutable_data<T>(ctx.GetPlace());
    int n = predict->numel();
    auto& dev_ctx = ctx.template device_context<DeviceContext>();
    int r = xpu::log_loss_bwd(dev_ctx.x_context(), n, epsilon,
                              predict->data<T>(), labels->data<T>(),
                              dloss->data<T>(), dpred->data<T>());
    PADDLE_ENFORCE_EQ(r == xpu::Error_t::SUCCESS, true,
                      platform::errors::InvalidArgument("XPU kernel error!"));
  }
};

}  // namespace operators
}  // namespace paddle
namespace ops = paddle::operators;
REGISTER_OP_XPU_KERNEL(
    log_loss, ops::LogLossXPUKernel<paddle::platform::XPUDeviceContext, float>);
REGISTER_OP_XPU_KERNEL(
    log_loss_grad,
    ops::LogLossGradXPUKernel<paddle::platform::XPUDeviceContext, float>);

#endif
