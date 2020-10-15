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

#include "paddle/fluid/operators/softmax_op.h"
#include "paddle/fluid/framework/op_registry.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
using DDim = framework::DDim;

template <typename DeviceContext, typename T>
class SoftmaxXPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* x = context.Input<Tensor>("X");
    auto* out = context.Output<Tensor>("Out");
    const int rank = x->dims().size();
    const int axis = CanonicalAxis(context.Attr<int>("axis"), rank);
    PADDLE_ENFORCE_EQ(axis == -1 || axis == rank - 1, true,
                      platform::errors::InvalidArgument(
                          "xpu softmax kernel only support last dimension of x "
                          "(axis==-1 or axis==x_dims-1), but received axis: "
                          "%d, x's shape: %s.",
                          axis, x->dims()));

    // allocate memory on device.
    out->mutable_data<T>(context.GetPlace());

    const int n = SizeToAxis(axis, x->dims());
    const int d = SizeFromAxis(axis, x->dims());

    auto& dev_ctx = context.template device_context<DeviceContext>();
    int r = xpu::softmax2d_forward(dev_ctx.x_context(), x->data<float>(),
                                   out->data<float>(), n, d, d <= 2048);
    PADDLE_ENFORCE_EQ(
        r, XPU_SUCCESS,
        platform::errors::External("XPU API(softmax2d_forward) return wrong "
                                   "value[%d], please check whether "
                                   "Baidu Kunlun Card is properly installed.",
                                   r));
  }
};

template <typename DeviceContext, typename T>
class SoftmaxGradXPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* out = context.Input<Tensor>("Out");
    auto* dout = context.Input<Tensor>(framework::GradVarName("Out"));
    auto* dx = context.Output<Tensor>(framework::GradVarName("X"));
    const int rank = dx->dims().size();
    const int axis = CanonicalAxis(context.Attr<int>("axis"), rank);

    // allocate memory on device.
    dx->mutable_data<T>(context.GetPlace());

    const int n = SizeToAxis(axis, dx->dims());
    const int d = SizeFromAxis(axis, dx->dims());

    auto& dev_ctx = context.template device_context<DeviceContext>();
    int r =
        xpu::softmax2d_backward(dev_ctx.x_context(), out->data<float>(),
                                dout->data<float>(), dx->data<float>(), n, d);
    PADDLE_ENFORCE_EQ(
        r, XPU_SUCCESS,
        platform::errors::External("XPU API(softmax2d_backward) return wrong "
                                   "value[%d], please check whether "
                                   "Baidu Kunlun Card is properly installed.",
                                   r));
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OP_XPU_KERNEL(
    softmax, ops::SoftmaxXPUKernel<paddle::platform::XPUDeviceContext, float>);
REGISTER_OP_XPU_KERNEL(
    softmax_grad,
    ops::SoftmaxGradXPUKernel<paddle::platform::XPUDeviceContext, float>);

#endif  // PADDLE_WITH_XPU
