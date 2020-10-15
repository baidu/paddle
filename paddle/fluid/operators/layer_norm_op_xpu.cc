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

#ifdef PADDLE_WITH_XPU

#include "paddle/fluid/operators/layer_norm_op.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
using DDim = framework::DDim;

template <typename DeviceContext, typename T>
class LayerNormXPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    const auto begin_norm_axis = ctx.Attr<int>("begin_norm_axis");
    const auto epsilon = ctx.Attr<float>("epsilon");
    const auto* x = ctx.Input<Tensor>("X");
    const auto& x_dims = x->dims();
    auto matrix_dim = framework::flatten_to_2d(x_dims, begin_norm_axis);
    int left = static_cast<int>(matrix_dim[0]);
    int right = static_cast<int>(matrix_dim[1]);
    const auto* scale = ctx.Input<Tensor>("Scale");
    const auto* bias = ctx.Input<Tensor>("Bias");
    auto* y = ctx.Output<Tensor>("Y");
    auto* mean = ctx.Output<Tensor>("Mean");
    auto* variance = ctx.Output<Tensor>("Variance");
    const auto* x_data = x->data<T>();
    const auto* scale_data = (scale == nullptr ? nullptr : scale->data<T>());
    const auto* bias_data = (bias == nullptr ? nullptr : bias->data<T>());
    auto* y_data = y->mutable_data<T>(ctx.GetPlace());
    auto* mean_data = mean->mutable_data<T>(ctx.GetPlace());
    auto* variance_data = variance->mutable_data<T>(ctx.GetPlace());
    auto& dev_ctx = ctx.template device_context<DeviceContext>();
    int r = xpu::layer_norm(dev_ctx.x_context(), left, right, x_data, y_data,
                            scale_data, bias_data, epsilon, mean_data,
                            variance_data, false);
    PADDLE_ENFORCE_EQ(
        r, XPU_SUCCESS,
        platform::errors::External("XPU API(layer_norm) return wrong "
                                   "value[%d], please check whether Baidu "
                                   "Kunlun Card is properly installed.",
                                   r));
  }
};

template <typename DeviceContext, typename T>
class LayerNormGradXPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    const auto begin_norm_axis = ctx.Attr<int>("begin_norm_axis");
    const auto epsilon = ctx.Attr<float>("epsilon");
    const auto* x = ctx.Input<Tensor>("X");
    const auto& x_dims = x->dims();
    auto matrix_dim = framework::flatten_to_2d(x_dims, begin_norm_axis);
    int left = static_cast<int>(matrix_dim[0]);
    int right = static_cast<int>(matrix_dim[1]);
    const auto* mean = ctx.Input<Tensor>("Mean");
    const auto* variance = ctx.Input<Tensor>("Variance");
    const auto* scale = ctx.Input<Tensor>("Scale");
    const auto* dy = ctx.Input<Tensor>(framework::GradVarName("Y"));
    auto* dx = ctx.Output<Tensor>(framework::GradVarName("X"));
    auto* dscale = ctx.Output<Tensor>(framework::GradVarName("Scale"));
    auto* dbias = ctx.Output<Tensor>(framework::GradVarName("Bias"));
    const auto* x_data = x->data<T>();
    const auto* dy_data = dy->data<T>();
    const auto* mean_data = mean->data<T>();
    const auto* variance_data = variance->data<T>();
    const auto* scale_data = (scale == nullptr ? nullptr : scale->data<T>());
    auto* dscale_data =
        (dscale == nullptr ? nullptr : dscale->mutable_data<T>(ctx.GetPlace()));
    auto* dbias_data =
        (dbias == nullptr ? nullptr : dbias->mutable_data<T>(ctx.GetPlace()));
    auto* dx_data =
        (dx == nullptr ? nullptr : dx->mutable_data<T>(ctx.GetPlace()));
    auto& dev_ctx = ctx.template device_context<DeviceContext>();
    int r = xpu::layer_norm_backward(
        dev_ctx.x_context(), left, right, x_data, scale_data, variance_data,
        mean_data, dy_data, dx_data, dscale_data, dbias_data, epsilon);
    PADDLE_ENFORCE_EQ(
        r, XPU_SUCCESS,
        platform::errors::External("XPU API(layer_norm_backward) return wrong "
                                   "value[%d], please check whether Baidu "
                                   "Kunlun Card is properly installed.",
                                   r));
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OP_XPU_KERNEL(
    layer_norm,
    ops::LayerNormXPUKernel<paddle::platform::XPUDeviceContext, float>);
REGISTER_OP_XPU_KERNEL(
    layer_norm_grad,
    ops::LayerNormGradXPUKernel<paddle::platform::XPUDeviceContext, float>);

#endif  // PADDLE_WITH_XPU
