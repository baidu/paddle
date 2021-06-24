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
#include <math.h>
#include <type_traits>
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/elementwise/elementwise_op_function.h"
#include "paddle/fluid/platform/transform.h"

namespace paddle {
namespace operators {

#define BITWISE_BINARY_FUNCTOR(func, expr, bool_expr)                          \
  template <typename T>                                                        \
  struct Bitwise##func##Functor {                                              \
    using ELEM_TYPE = T;                                                       \
    HOSTDEVICE T operator()(const T& a, const T& b) const { return a expr b; } \
  };                                                                           \
                                                                               \
  template <>                                                                  \
  struct Bitwise##func##Functor<bool> {                                        \
    using ELEM_TYPE = bool;                                                    \
    HOSTDEVICE bool operator()(const bool& a, const bool& b) const {           \
      return a bool_expr b;                                                    \
    }                                                                          \
  };

BITWISE_BINARY_FUNCTOR(And, &, &&)
BITWISE_BINARY_FUNCTOR(Or, |, ||)
BITWISE_BINARY_FUNCTOR(Xor, ^, !=)
#undef BITWISE_BINARY_FUNCTOR

template <typename T>
struct BitwiseNotFunctor {
  using ELEM_TYPE = T;
  HOSTDEVICE T operator()(const T& a) const { return ~a; }
};

template <>
struct BitwiseNotFunctor<bool> {
  using ELEM_TYPE = bool;
  HOSTDEVICE bool operator()(const bool& a) const { return !a; }
};

template <typename DeviceContext, typename Functor>
class BinaryBitwiseOpKernel
    : public framework::OpKernel<typename Functor::ELEM_TYPE> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    using T = typename Functor::ELEM_TYPE;
    auto func = Functor();
    auto* x = context.Input<framework::Tensor>("X");
    auto* y = context.Input<framework::Tensor>("Y");
    auto* out = context.Output<framework::Tensor>("Out");
    ElementwiseComputeEx<Functor, DeviceContext, T>(context, x, y, -1, func,
                                                    out);
  }
};

template <typename DeviceContext, typename Functor>
class UnaryBitwiseOpKernel
    : public framework::OpKernel<typename Functor::ELEM_TYPE> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    using T = typename Functor::ELEM_TYPE;
    auto func = Functor();
    auto* x = context.Input<framework::Tensor>("X");
    auto* out = context.Output<framework::Tensor>("Out");
    platform::Transform<DeviceContext> trans;
    trans(context.template device_context<DeviceContext>(), x->data<T>(),
          x->data<T>() + x->numel(), out->mutable_data<T>(context.GetPlace()),
          func);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = ::paddle::operators;
namespace plat = ::paddle::platform;

#define REGISTER_BINARY_BITWISE_KERNEL(op_type, dev, functor)                 \
  REGISTER_OP_##dev##_KERNEL(                                                 \
      op_type,                                                                \
      ops::BinaryBitwiseOpKernel<plat::dev##DeviceContext, functor<bool>>,    \
      ops::BinaryBitwiseOpKernel<plat::dev##DeviceContext, functor<uint8_t>>, \
      ops::BinaryBitwiseOpKernel<plat::dev##DeviceContext, functor<int8_t>>,  \
      ops::BinaryBitwiseOpKernel<plat::dev##DeviceContext, functor<int16_t>>, \
      ops::BinaryBitwiseOpKernel<plat::dev##DeviceContext, functor<int>>,     \
      ops::BinaryBitwiseOpKernel<plat::dev##DeviceContext, functor<int64_t>>);

#define REGISTER_UNARY_BITWISE_KERNEL(op_type, dev, functor)                 \
  REGISTER_OP_##dev##_KERNEL(                                                \
      op_type,                                                               \
      ops::UnaryBitwiseOpKernel<plat::dev##DeviceContext, functor<bool>>,    \
      ops::UnaryBitwiseOpKernel<plat::dev##DeviceContext, functor<uint8_t>>, \
      ops::UnaryBitwiseOpKernel<plat::dev##DeviceContext, functor<int8_t>>,  \
      ops::UnaryBitwiseOpKernel<plat::dev##DeviceContext, functor<int16_t>>, \
      ops::UnaryBitwiseOpKernel<plat::dev##DeviceContext, functor<int>>,     \
      ops::UnaryBitwiseOpKernel<plat::dev##DeviceContext, functor<int64_t>>);
