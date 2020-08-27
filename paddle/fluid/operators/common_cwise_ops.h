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

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/operators/common_infer_shape_functions.h"
#include "paddle/fluid/operators/elementwise/elementwise_op_function.h"

#ifdef PADDLE_WITH_MKLDNN
#include "paddle/fluid/platform/mkldnn_helper.h"
#endif

// This file almostly contains all the coefficient-wise Operator class and
// OpKernel class

namespace paddle {
namespace operators {
using Tensor = framework::LoDTensor;

class UnaryOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext *ctx) const override {
    return UnaryOpUnchangedInferShape(ctx);
  }
  // Use default GetExpectedKernelType
};

class UnaryGradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext *ctx) const override {
    return UnaryOpUnchangedInferShape(ctx);
  }
  // Use default GetExpectedKernelType
};

template <typename DeviceContext, typename Functor, typename T>
class UnaryOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    // NOTE(zhiqiu): only supports Tensor now
    auto *x = ctx.Input<Tensor>(ctx.GetInputNameByIdx(0));
    auto *out = ctx.Output<Tensor>(ctx.GetOutputNameByIdx(0));
    out->mutable_data<T>(ctx.GetPlace());

    auto eigen_in = framework::EigenVector<T>::Flatten(*x);
    auto eigen_out = framework::EigenVector<T>::Flatten(*out);
    auto &dev = *ctx.template device_context<DeviceContext>().eigen_device();

    eigen_out.device(dev) = eigen_in.unaryExpr(Functor());
  }
};

template <typename DeviceContext, typename Functor, typename T>
class UnaryGradOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    // NOTE(zhiqiu): only supports Tensor now
    auto *dout = ctx.Input<Tensor>(ctx.GetInputNameByIdx(0));
    auto *dx = ctx.Output<Tensor>(ctx.GetOutputNameByIdx(0));
    dx->mutable_data<T>(ctx.GetPlace());

    auto eigen_in = framework::EigenVector<T>::Flatten(*dout);
    auto eigen_out = framework::EigenVector<T>::Flatten(*dx);
    auto &dev = *ctx.template device_context<DeviceContext>().eigen_device();
    eigen_out.device(dev) = eigen_in.unaryExpr(Functor());
  }
};

#define REGISTER_OP_KERNEL_1(N, K, D, F, T0) \
  REGISTER_OP_CPU_KERNEL(N, K<paddle::platform::D##DeviceContext, F<T0>, T0>)

#define REGISTER_OP_KERNEL_2(N, K, D, F, T0, T1)                              \
  REGISTER_OP_CPU_KERNEL(N, K<paddle::platform::D##DeviceContext, F<T0>, T0>, \
                         K<paddle::platform::D##DeviceContext, F<T1>, T1>)

#define REGISTER_OP_KERNEL_3(N, K, D, F, T0, T1, T2)                          \
  REGISTER_OP_CPU_KERNEL(N, K<paddle::platform::D##DeviceContext, F<T0>, T0>, \
                         K<paddle::platform::D##DeviceContext, F<T1>, T1>, \
                         K<paddle::platform::D##DeviceContext, F<T1>, T1>))

#define REGISTER_OP_KERNEL_4(N, K, D, F, T0, T1, T2, T3)                      \
  REGISTER_OP_CPU_KERNEL(N, K<paddle::platform::D##DeviceContext, F<T0>, T0>, \
                         K<paddle::platform::D##DeviceContext, F<T1>, T1>,    \
                         K<paddle::platform::D##DeviceContext, F<T2>, T2>,    \
                         K<paddle::platform::D##DeviceContext, F<T3>, T3>)

#define REGISTER_OP_KERNEL_5(N, K, D, F, T0, T1, T2, T3, T4)                  \
  REGISTER_OP_CPU_KERNEL(N, K<paddle::platform::D##DeviceContext, F<T0>, T0>, \
                         K<paddle::platform::D##DeviceContext, F<T1>, T1>,    \
                         K<paddle::platform::D##DeviceContext, F<T2>, T2>,    \
                         K<paddle::platform::D##DeviceContext, F<T3>, T3>,    \
                         K<paddle::platform::D##DeviceContext, F<T4>, T4>)

}  // namespace operators
}  // namespace paddle
