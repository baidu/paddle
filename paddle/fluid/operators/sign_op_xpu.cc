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

#include "paddle/fluid/operators/sign_op.h"
#include "paddle/fluid/platform/xpu_header.h"
namespace paddle {
namespace operators {

template <typename DeviceContext, typename T>
class SignXPUKernel : public framework::OpKernel<T> {
 public:
  virtual void Compute(const framework::ExecutionContext& context) const {
    auto* out = context.Output<framework::Tensor>("Out");
    auto* in = context.Input<framework::Tensor>("X");
    out->mutable_data<T>(in->place());
    auto xpu_context = context.device_context<DeviceContext>().x_context();
    int r = xpu::activation_forward(xpu_context, xpu::Activation_t::SIGN,
                                    in->numel(), in->data<T>(), out->data<T>());
    PADDLE_ENFORCE_EQ(r, xpu::Error_t::SUCCESS,
                      platform::errors::Fatal("XPU sign kernel error!"));
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_XPU_KERNEL(
    sign, ops::SignXPUKernel<paddle::platform::XPUDeviceContext, float>);

#endif
