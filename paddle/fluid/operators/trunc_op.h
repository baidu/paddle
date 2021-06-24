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

#include <math.h>
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/operator.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;

template <typename T>
class TruncKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    const Tensor* x = context.Input<Tensor>("X");
    Tensor* out = context.Output<Tensor>("Out");

    size_t numel = x->numel();
    const T* x_data = x->data<T>();
    T* out_data = out->mutable_data<T>(context.GetPlace());

    for (size_t i = 0; i < numel; i++) {
      out_data[i] = trunc(x_data[i]);
    }
  }
};

template <typename T>
class TruncGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* dx = context.Output<Tensor>(framework::GradVarName("X"));
    T* dx_data = dx->mutable_data<T>(context.GetPlace());

    int numel = dx->numel();
    memset(dx_data, 0.0, numel * sizeof(T));
  }
};

}  // namespace operators
}  // namespace paddle
