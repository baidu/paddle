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

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/math/math_function.h"
#include "paddle/fluid/operators/math/maxouting.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;

template <typename DeviceContext, typename T>
class MaxOutKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    const Tensor* in_x = context.Input<Tensor>("X");
    Tensor* out = context.Output<Tensor>("Out");
    int groups = context.template Attr<int>("groups");
    int axis = context.template Attr<int>("axis");
    if (axis < 0) {
      axis += in_x->dims().size();
    }

    math::MaxOutFunctor<DeviceContext, T> maxout_forward;
    maxout_forward(context.template device_context<DeviceContext>(), *in_x, out,
                   groups, axis);
  }
};

template <typename DeviceContext, typename T>
class MaxOutGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    const Tensor* in_x = context.Input<Tensor>("X");
    const Tensor* out = context.Input<Tensor>("Out");
    const Tensor* out_grad =
        context.Input<Tensor>(framework::GradVarName("Out"));
    Tensor* in_x_grad = context.Output<Tensor>(framework::GradVarName("X"));
    int groups = context.template Attr<int>("groups");
    int axis = context.template Attr<int>("axis");
    if (axis < 0) {
      axis += in_x->dims().size();
    }

    auto& device_ctx = context.template device_context<DeviceContext>();
    math::SetConstant<DeviceContext, T> zero;
    if (in_x_grad) {
      in_x_grad->mutable_data<T>(context.GetPlace());
      zero(device_ctx, in_x_grad, static_cast<T>(0.0));
      math::MaxOutGradFunctor<DeviceContext, T> maxout_backward;
      maxout_backward(device_ctx, *in_x, in_x_grad, *out, *out_grad, groups,
                      axis);
    }
  }
};

}  // namespace operators
}  // namespace paddle
