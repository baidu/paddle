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

#pragma once

#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/framework/op_registry.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;

template <typename DeviceContext, typename T>
class ScalarMulKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* in_t = ctx.Input<Tensor>("X");
    auto* out_t = ctx.Output<Tensor>("Out");
    auto a = static_cast<T>(ctx.Attr<float>("a"));
    auto b = static_cast<T>(ctx.Attr<float>("b"));
    auto x = in_t->data<T>();
    auto out = out_t->mutable_data<T>(ctx.GetPlace());

    for (int i = 0; i < in_t->numel(); ++i) {
      out[i] = a * x[i] + b;
    }
  }
};

template <typename DeviceContext, typename T>
class ScalarMulGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* dout_t = ctx.Input<Tensor>(framework::GradVarName("Out"));
    auto* dx_t = ctx.Output<Tensor>(framework::GradVarName("X"));
    auto a = static_cast<T>(ctx.Attr<float>("a"));

    auto dout = dout_t->data<T>();
    auto dx = dx_t->mutable_data<T>(ctx.GetPlace());

    for (int i = 0; i < dout_t->numel(); ++i) {
      dx[i] = dout[i] * a;
    }
  }
};

}  // namespace operators
}  // namespace paddle
