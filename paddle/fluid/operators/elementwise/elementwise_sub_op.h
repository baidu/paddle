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

#include "paddle/fluid/operators/elementwise/elementwise_op.h"
#include "paddle/fluid/operators/elementwise/elementwise_op_function.cu.h"
#include "paddle/fluid/operators/elementwise/elementwise_op_function.h"
#include "paddle/fluid/operators/math/blas.h"

namespace paddle {
namespace operators {

template <typename DeviceContext, typename T>
void default_elementwise_sub(const framework::ExecutionContext& ctx,
                             const framework::Tensor* x,
                             const framework::Tensor* y, framework::Tensor* z) {
  int axis = ctx.Attr<int>("axis");
  auto x_dims = x->dims();
  auto y_dims = y->dims();
  if (x_dims.size() >= y_dims.size()) {
    ElementwiseComputeEx<SubFunctor<T>, DeviceContext, T>(ctx, x, y, axis,
                                                          SubFunctor<T>(), z);
  } else {
    ElementwiseComputeEx<InverseSubFunctor<T>, DeviceContext, T>(
        ctx, x, y, axis, InverseSubFunctor<T>(), z);
  }
}

template <typename DeviceContext, typename T, class Enable = void>
struct SameDimsElemwiseSub {
  void operator()(const framework::ExecutionContext& ctx,
                  const framework::Tensor* x, const framework::Tensor* y,
                  framework::Tensor* z);
};

template <typename DeviceContext, typename T>
class ElementwiseSubKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* x = ctx.Input<framework::LoDTensor>("X");
    auto* y = ctx.Input<framework::LoDTensor>("Y");
    auto* z = ctx.Output<framework::LoDTensor>("Out");
    z->mutable_data<T>(ctx.GetPlace());

    auto dims_equal = x->dims() == y->dims();
    if (dims_equal) {
      SameDimsElemwiseSub<DeviceContext, T> same_dims_sub;
      same_dims_sub(ctx, x, y, z);
    } else {
      default_elementwise_sub<DeviceContext, T>(ctx, x, y, z);
    }
  }
};

template <typename T>
struct SubGradDX {
  HOSTDEVICE T operator()(T x, T y, T out, T dout) const { return dout; }
};

template <typename T>
struct SubGradDY {
  HOSTDEVICE T operator()(T x, T y, T out, T dout) const { return -dout; }
};

template <typename DeviceContext, typename T>
typename std::enable_if<
    std::is_same<DeviceContext, platform::CPUDeviceContext>::value>::type
elementwise_sub_grad(const framework::ExecutionContext& ctx,
                     const framework::Tensor* x, const framework::Tensor* y,
                     const framework::Tensor* out,
                     const framework::Tensor* dout, framework::Tensor* dx,
                     framework::Tensor* dy) {
  int axis = ctx.Attr<int>("axis");
  ElemwiseExplicitGradCompute<DeviceContext, T, SubGradDX<T>, SubGradDY<T>>(
      ctx, *x, *y, *out, *dout, axis, dx, dy, SubGradDX<T>(), SubGradDY<T>());
}

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
// cuda definition
template <typename DeviceContext, typename T>
typename std::enable_if<
    std::is_same<DeviceContext, platform::CUDADeviceContext>::value>::type
elementwise_sub_grad(const framework::ExecutionContext& ctx,
                     const framework::Tensor* x, const framework::Tensor* y,
                     const framework::Tensor* out,
                     const framework::Tensor* dout, framework::Tensor* dx,
                     framework::Tensor* dy);
#endif

template <typename DeviceContext, typename T>
class ElementwiseSubGradKernel : public ElemwiseGradKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    ElemwiseGradKernel<T>::Compute(ctx);
    using Tensor = framework::Tensor;

    auto* x = ctx.Input<Tensor>("X");
    auto* y = ctx.Input<Tensor>("Y");
    auto* dout = ctx.Input<Tensor>(framework::GradVarName("Out"));
    auto* dx = ctx.Output<Tensor>(framework::GradVarName("X"));
    auto* dy = ctx.Output<Tensor>(framework::GradVarName("Y"));
    int axis = ctx.Attr<int>("axis");
    // skip out
    auto* out = dout;
    if (dx != nullptr && dy != nullptr && (dx->dims() == dy->dims())) {
      elementwise_sub_grad<DeviceContext, T>(ctx, x, y, out, dout, dx, dy);
    } else {
      ElemwiseExplicitGradCompute<DeviceContext, T, SubGradDX<T>, SubGradDY<T>>(
          ctx, *x, *y, *out, *dout, axis, dx, dy, SubGradDX<T>(),
          SubGradDY<T>());
    }
  }
};

template <typename DeviceContext, typename T>
class ElementwiseSubDoubleGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    using Tensor = framework::Tensor;

    auto* y = ctx.Input<Tensor>("Y");
    auto* dout = ctx.Input<Tensor>("DOut");
    auto* ddx = ctx.Input<Tensor>("DDX");
    auto* ddy = ctx.Input<Tensor>("DDY");

    auto* ddout = ctx.Output<Tensor>("DDOut");

    // DDOut = ddx - ddy
    if (ddout) {
      Tensor ddx_safe, ddy_safe;
      GetDoubleGradSafeTensor<DeviceContext, T>(ctx, dout, ddx, &ddx_safe);
      GetDoubleGradSafeTensor<DeviceContext, T>(ctx, y, ddy, &ddy_safe);

      ddout->mutable_data<T>(ctx.GetPlace());
      int axis = ctx.Attr<int>("axis");
      ElementwiseComputeEx<SubFunctor<T>, DeviceContext, T>(
          ctx, &ddx_safe, &ddy_safe, axis, SubFunctor<T>(), ddout);
    }
  }
};

}  // namespace operators
}  // namespace paddle
