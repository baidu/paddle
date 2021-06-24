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

#pragma once

#include <unsupported/Eigen/SpecialFunctions>
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/platform/for_range.h"

namespace paddle {
namespace operators {

template <typename T>
struct DigammaFunctor {
  DigammaFunctor(const T* input, T* output, int64_t numel)
      : input_(input), output_(output), numel_(numel) {}

  HOSTDEVICE void operator()(int64_t idx) const {
    output_[idx] = Eigen::numext::digamma(input_[idx]);
  }

 private:
  const T* input_;
  T* output_;
  int64_t numel_;
};

template <typename T>
struct DigammaGradFunctor {
  DigammaGradFunctor(const T* dout, const T* x, T* output, int64_t numel)
      : dout_(dout), x_(x), output_(output), numel_(numel) {}

  HOSTDEVICE void operator()(int64_t idx) const {
    output_[idx] = dout_[idx] * Eigen::numext::polygamma(T(1), x_[idx]);
  }

 private:
  const T* dout_;
  const T* x_;
  T* output_;
  int64_t numel_;
};

using Tensor = framework::Tensor;

template <typename DeviceContext, typename T>
class DigammaKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    const Tensor* x = context.Input<Tensor>("X");
    Tensor* out = context.Output<Tensor>("Out");

    auto numel = x->numel();
    auto* x_data = x->data<T>();
    auto* out_data = out->mutable_data<T>(context.GetPlace(),
                                          size_t(x->numel() * sizeof(T)));

    auto& dev_ctx = context.template device_context<DeviceContext>();
    platform::ForRange<DeviceContext> for_range(dev_ctx, numel);
    DigammaFunctor<T> functor(x_data, out_data, numel);
    for_range(functor);
  }
};

template <typename DeviceContext, typename T>
class DigammaGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    const Tensor* d_out = context.Input<Tensor>(framework::GradVarName("Out"));
    const Tensor* x = context.Input<Tensor>("X");
    auto* d_x = context.Output<Tensor>(framework::GradVarName("X"));

    auto numel = d_out->numel();
    auto* dout_data = d_out->data<T>();
    auto* x_data = x->data<T>();
    auto* dx_data = d_x->mutable_data<T>(
        context.GetPlace(), static_cast<size_t>(numel * sizeof(T)));

    auto& dev_ctx = context.template device_context<DeviceContext>();
    platform::ForRange<DeviceContext> for_range(dev_ctx, numel);
    DigammaGradFunctor<T> functor(dout_data, x_data, dx_data, numel);
    for_range(functor);
  }
};

}  // namespace operators
}  // namespace paddle
