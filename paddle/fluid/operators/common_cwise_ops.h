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

#include <string>

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/operators/common_infer_kernel_type_functions.h"
#include "paddle/fluid/operators/common_infer_shape_functions.h"
#include "paddle/fluid/operators/elementwise/elementwise_op.h"
#include "paddle/fluid/operators/elementwise/elementwise_op_function.h"

// This file almostly contains all the coefficient-wise Operator class and
// OpKernel class

namespace paddle {
namespace operators {
using Tensor = framework::LoDTensor;

class BinaryOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext *ctx) const override {
    return BinaryOpBroadcastInferShape(ctx);
  }

  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override {
    return BinaryOpInferKernelType(ctx);
  }
};

template <typename DeviceContext, typename Functor, typename T>
class BinaryOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    auto *x = ctx.Input<Tensor>(ctx.GetInputNameByIdx(0));
    auto *y = ctx.Input<Tensor>(ctx.GetInputNameByIdx(1));
    auto *out = ctx.Output<Tensor>(ctx.GetOutputNameByIdx(0));
    out->mutable_data<T>(ctx.GetPlace());
    int axis = ctx.Attr<int>("axis");
    ElementwiseComputeEx<Functor, DeviceContext, T>(ctx, x, y, axis, Functor(),
                                                    out);
  }
};

class BinaryOpMaker : public ElementwiseOpMaker {
 protected:
  std::string GetName() const override { return "Mul"; }
  std::string GetEquation() const override { return "Out = X \\\\odot Y"; }

  void AddInputX() override {
    AddInput("X",
             "(Variable), Tensor or LoDTensor of any dimensions. Its dtype "
             "should be int32, int64, float32, float64.");
  }

  void AddInputY() override {
    AddInput("Y",
             "(Variable), Tensor or LoDTensor of any dimensions. Its dtype "
             "should be int32, int64, float32, float64.");
  }

  std::string GetOpFuntionality() const override {
    return "Multiply two tensors element-wise";
  }
};

}  // namespace operators
}  // namespace paddle

#define REGISTER_CPU_KERNEL_4(N, K, F, T0, T1, T2, T3)                        \
  REGISTER_OP_CPU_KERNEL(N, K<paddle::platform::CPUDeviceContext, F<T0>, T0>, \
                         K<paddle::platform::CPUDeviceContext, F<T1>, T1>,    \
                         K<paddle::platform::CPUDeviceContext, F<T2>, T2>,    \
                         K<paddle::platform::CPUDeviceContext, F<T3>, T3>)
