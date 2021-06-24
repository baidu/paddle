/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.

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
#include "paddle/fluid/operators/elementwise/elementwise_op.h"
#include "paddle/fluid/operators/math/blas.h"

namespace paddle {
namespace operators {

template <typename T>
struct ModFunctor {
  inline HOSTDEVICE T operator()(T a, T b) const {
    T res = a % b;
    if ((res != 0) && ((res < 0) != (b < 0))) res += b;
    return res;
  }
};

template <typename T>
struct InverseModFunctor {
  inline HOSTDEVICE T operator()(T a, T b) const {
    T res = b % a;
    if ((res != 0) && ((res < 0) != (a < 0))) res += a;
    return res;
  }
};

template <typename T>
struct ModFunctorFP {
  inline HOSTDEVICE T operator()(T a, T b) const {
    T res = fmod(a, b);
    if ((res != 0) && ((b < 0) != (res < 0))) res += b;
    return res;
  }
};

template <typename T>
struct InverseModFunctorFP {
  inline HOSTDEVICE T operator()(T a, T b) const {
    T res = fmod(b, a);
    if ((res != 0) && ((a < 0) != (res < 0))) res += a;
    return res;
  }
};

template <typename DeviceContext, typename T>
void elementwise_mod(const framework::ExecutionContext &ctx,
                     const framework::Tensor *x, const framework::Tensor *y,
                     framework::Tensor *z) {
  int axis = ctx.Attr<int>("axis");
  auto x_dims = x->dims();
  auto y_dims = y->dims();
  if (x_dims.size() >= y_dims.size()) {
    ElementwiseComputeEx<ModFunctor<T>, DeviceContext, T>(ctx, x, y, axis,
                                                          ModFunctor<T>(), z);
  } else {
    ElementwiseComputeEx<InverseModFunctor<T>, DeviceContext, T>(
        ctx, x, y, axis, InverseModFunctor<T>(), z);
  }
}

template <typename DeviceContext, typename T>
void elementwise_mod_fp(const framework::ExecutionContext &ctx,
                        const framework::Tensor *x, const framework::Tensor *y,
                        framework::Tensor *z) {
  int axis = ctx.Attr<int>("axis");
  auto x_dims = x->dims();
  auto y_dims = y->dims();
  if (x_dims.size() >= y_dims.size()) {
    ElementwiseComputeEx<ModFunctorFP<T>, DeviceContext, T>(
        ctx, x, y, axis, ModFunctorFP<T>(), z);
  } else {
    ElementwiseComputeEx<InverseModFunctorFP<T>, DeviceContext, T>(
        ctx, x, y, axis, InverseModFunctorFP<T>(), z);
  }
}

template <typename DeviceContext, typename T>
class ElementwiseModKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    auto *x = ctx.Input<framework::LoDTensor>("X");
    auto *y = ctx.Input<framework::LoDTensor>("Y");
    auto *z = ctx.Output<framework::LoDTensor>("Out");

    z->mutable_data<T>(ctx.GetPlace());

    // dtype of x and y is int64 or int32
    elementwise_mod<DeviceContext, T>(ctx, x, y, z);
  }
};

template <typename DeviceContext, typename T>
class ElementwiseModFPKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    auto *x = ctx.Input<framework::LoDTensor>("X");
    auto *y = ctx.Input<framework::LoDTensor>("Y");
    auto *z = ctx.Output<framework::LoDTensor>("Out");

    z->mutable_data<T>(ctx.GetPlace());

    // dtype of x and y is float or double
    elementwise_mod_fp<DeviceContext, T>(ctx, x, y, z);
  }
};

}  // namespace operators
}  // namespace paddle
