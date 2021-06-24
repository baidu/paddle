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
struct FloorDivFunctor {
  inline HOSTDEVICE T operator()(T a, T b) const {
#if defined(__HIPCC__) || defined(__CUDA_ARCH__)
    if (b == 0) {
      printf("Error: Divide by zero encounter in floor_divide\n");
#ifdef __HIPCC__
      abort();
#else
      asm("trap;");
#endif
    }
#else
    if (b == 0)
      PADDLE_THROW(platform::errors::InvalidArgument(
          "Divide by zero encounter in floor_divide"));
#endif
    return static_cast<T>(std::trunc(a / b));
  }
};

template <typename T>
struct InverseFloorDivFunctor {
  inline HOSTDEVICE T operator()(T a, T b) const {
#if defined(__HIPCC__) || defined(__CUDA_ARCH__)
    if (a == 0) {
      printf("Error: Divide by zero encounter in floor_divide\n");
#ifdef __HIPCC__
      abort();
#else
      asm("trap;");
#endif
    }
#else
    if (a == 0)
      PADDLE_THROW(platform::errors::InvalidArgument(
          "Divide by zero encounter in floor_divide"));
#endif
    return static_cast<T>(std::trunc(b / a));
  }
};

template <typename DeviceContext, typename T>
void elementwise_floor_div(const framework::ExecutionContext &ctx,
                           const framework::Tensor *x,
                           const framework::Tensor *y, framework::Tensor *z) {
  int axis = ctx.Attr<int>("axis");
  auto x_dims = x->dims();
  auto y_dims = y->dims();
  if (x_dims.size() >= y_dims.size()) {
    ElementwiseComputeEx<FloorDivFunctor<T>, DeviceContext, T>(
        ctx, x, y, axis, FloorDivFunctor<T>(), z);
  } else {
    ElementwiseComputeEx<InverseFloorDivFunctor<T>, DeviceContext, T>(
        ctx, x, y, axis, InverseFloorDivFunctor<T>(), z);
  }
}

template <typename DeviceContext, typename T>
class ElementwiseFloorDivKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    auto *x = ctx.Input<framework::LoDTensor>("X");
    auto *y = ctx.Input<framework::LoDTensor>("Y");
    auto *z = ctx.Output<framework::LoDTensor>("Out");

    z->mutable_data<T>(ctx.GetPlace());

    // dtype of x and y is int64 or int32
    elementwise_floor_div<DeviceContext, T>(ctx, x, y, z);
  }
};

}  // namespace operators
}  // namespace paddle
