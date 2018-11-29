/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

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
#include <string>
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/math/blas.h"
#include "paddle/fluid/operators/math/detail/activation_functions.h"
#include "paddle/fluid/operators/math/lstm_compute.h"
#include "paddle/fluid/operators/math/sequence2batch.h"

namespace paddle {
namespace operators {

using LoDTensor = framework::LoDTensor;
using Tensor = framework::Tensor;

template <typename DeviceContext, typename T>
class CudnnLSTMKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    PADDLE_THROW(
        "CPU is not support for this kernel now. Will be add in the future");
  }
};

template <typename DeviceContext, typename T>
class CudnnLSTMGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {}
};

}  // namespace operators
}  // namespace paddle
