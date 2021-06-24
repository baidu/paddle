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
#include <string>
#include "paddle/fluid/operators/elementwise/elementwise_op.h"
#include "paddle/fluid/operators/elementwise/elementwise_op_function.cu.h"
#include "paddle/fluid/operators/elementwise/elementwise_op_function.h"
#include "paddle/fluid/operators/math/blas.h"
#include "paddle/fluid/platform/cpu_info.h"

namespace paddle {
namespace operators {

class ElementwiseMulOp : public ElementwiseOp {
 public:
  using Tensor = framework::Tensor;
  using ElementwiseOp::ElementwiseOp;

  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    auto input_data_type =
        OperatorWithKernel::IndicateOrPromoteVarDataTypes(ctx, "X", "Y");

#ifdef PADDLE_WITH_MKLDNN
    if (this->CanMKLDNNBeUsed(ctx, input_data_type)) {
      return framework::OpKernelType(input_data_type, ctx.GetPlace(),
                                     framework::DataLayout::kMKLDNN,
                                     framework::LibraryType::kMKLDNN);
    }
#endif
    return framework::OpKernelType(input_data_type, ctx.GetPlace());
  }

  framework::OpKernelType GetKernelTypeForVar(
      const std::string& var_name, const framework::Tensor& tensor,
      const framework::OpKernelType& expected_kernel_type) const {
    if (framework::IsComplexType(expected_kernel_type.data_type_)) {
      // only promote inputs’s types when contains complex input
      return framework::OpKernelType(tensor.type(), tensor.place(),
                                     tensor.layout());
    } else {
      return framework::OpKernelType(expected_kernel_type.data_type_,
                                     tensor.place(), tensor.layout());
    }
  }
};

template <typename DeviceContext, typename T>
void default_elementwise_mul(const framework::ExecutionContext& ctx,
                             const framework::Tensor* x,
                             const framework::Tensor* y, framework::Tensor* z) {
  int axis = ctx.Attr<int>("axis");
  auto x_dims = x->dims();
  auto y_dims = y->dims();
  if (x_dims.size() >= y_dims.size()) {
    ElementwiseComputeEx<MulFunctor<T>, DeviceContext, T>(ctx, x, y, axis,
                                                          MulFunctor<T>(), z);
  } else {
    ElementwiseComputeEx<InverseMulFunctor<T>, DeviceContext, T>(
        ctx, x, y, axis, InverseMulFunctor<T>(), z);
  }
}

template <typename DeviceContext, typename T, class Enable = void>
struct SameDimsElemwiseMul {
  void operator()(const framework::ExecutionContext& ctx,
                  const framework::Tensor* x, const framework::Tensor* y,
                  framework::Tensor* z);
};

template <typename DeviceContext, typename T>
class ElementwiseMulKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto x_var = ctx.InputVar("X");
    PADDLE_ENFORCE_EQ(x_var != nullptr, true,
                      platform::errors::InvalidArgument(
                          "Cannot get input Variable X, Variable name = %s.",
                          ctx.InputName("X")));
    auto* y = ctx.Input<framework::LoDTensor>("Y");

    framework::Tensor x, *z;
    if (x_var->IsType<framework::SelectedRows>()) {
      PADDLE_ENFORCE_EQ(y->dims().size() == 1 && y->dims()[0] == 1, true,
                        platform::errors::InvalidArgument(
                            "For elementwise_op, if X is Sparse, Y must be "
                            "scalar. But reveived the size of Y = %s.",
                            y->dims().size()));
      auto& x_sele = x_var->Get<framework::SelectedRows>();
      auto out_sele = ctx.Output<framework::SelectedRows>("Out");
      x = x_sele.value();
      out_sele->set_rows(x_sele.rows());
      out_sele->set_height(x_sele.height());
      out_sele->mutable_value()->Resize(x_sele.value().dims());
      out_sele->mutable_value()->mutable_data(ctx.GetPlace(), x.type());
      z = ctx.Output<framework::SelectedRows>("Out")->mutable_value();
    } else if (x_var->IsType<framework::LoDTensor>()) {
      x = x_var->Get<framework::LoDTensor>();
      z = ctx.Output<framework::LoDTensor>("Out");
    } else {
      PADDLE_THROW(platform::errors::InvalidArgument(
          "X's type[%s] is not supported by elementwise_op. X's type should be "
          "LoDTensor or SelectedRows.",
          framework::ToTypeName(x_var->Type())));
    }

    z->mutable_data<T>(ctx.GetPlace());
    auto dims_equal = x.dims() == y->dims();
    if (dims_equal) {
      SameDimsElemwiseMul<DeviceContext, T> same_dims_mul;
      same_dims_mul(ctx, &x, y, z);
    } else {
      default_elementwise_mul<DeviceContext, T>(ctx, &x, y, z);
    }
  }
};
template <typename T>
struct MulGradDX {
  HOSTDEVICE T operator()(T x, T y, T out, T dout) const { return dout * y; }
};

template <typename T>
struct MulGradDX<paddle::platform::complex<T>> {
  HOSTDEVICE paddle::platform::complex<T> operator()(
      paddle::platform::complex<T> x, paddle::platform::complex<T> y,
      paddle::platform::complex<T> out,
      paddle::platform::complex<T> dout) const {
    paddle::platform::complex<T> y_conj(y.real, -y.imag);
    return dout * y_conj;
  }
};

template <typename T>
struct MulGradDY {
  HOSTDEVICE T operator()(T x, T y, T out, T dout) const { return dout * x; }
};

template <typename T>
struct MulGradDY<paddle::platform::complex<T>> {
  HOSTDEVICE paddle::platform::complex<T> operator()(
      paddle::platform::complex<T> x, paddle::platform::complex<T> y,
      paddle::platform::complex<T> out,
      paddle::platform::complex<T> dout) const {
    paddle::platform::complex<T> x_conj(x.real, -x.imag);
    return dout * x_conj;
  }
};

template <typename DeviceContext, typename T>
typename std::enable_if<
    std::is_same<DeviceContext, platform::CPUDeviceContext>::value>::type
elementwise_mul_grad(const framework::ExecutionContext& ctx,
                     const framework::Tensor* x, const framework::Tensor* y,
                     const framework::Tensor* out,
                     const framework::Tensor* dout, framework::Tensor* dx,
                     framework::Tensor* dy) {
  int axis = ctx.Attr<int>("axis");
  ElemwiseGradCompute<DeviceContext, T, MulGradDX<T>, MulGradDY<T>>(
      ctx, *x, *y, *out, *dout, axis, dx, dy, MulGradDX<T>(), MulGradDY<T>());
}

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
// cuda definition
template <typename DeviceContext, typename T>
typename std::enable_if<
    std::is_same<DeviceContext, platform::CUDADeviceContext>::value>::type
elementwise_mul_grad(const framework::ExecutionContext& ctx,
                     const framework::Tensor* x, const framework::Tensor* y,
                     const framework::Tensor* out,
                     const framework::Tensor* dout, framework::Tensor* dx,
                     framework::Tensor* dy);
#endif

template <typename DeviceContext, typename T>
class ElementwiseMulGradKernel : public ElemwiseGradKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    ElemwiseGradKernel<T>::Compute(ctx);
    using Tensor = framework::Tensor;

    auto* x = ctx.Input<Tensor>("X");
    auto* y = ctx.Input<Tensor>("Y");
    auto* dout = ctx.Input<Tensor>(framework::GradVarName("Out"));
    auto* out = dout;  // out is not necessary
    auto* dx = ctx.Output<Tensor>(framework::GradVarName("X"));
    auto* dy = ctx.Output<Tensor>(framework::GradVarName("Y"));
    int axis = ctx.Attr<int>("axis");
    if (dx != nullptr && dy != nullptr && (dx->dims() == dy->dims())) {
      elementwise_mul_grad<DeviceContext, T>(ctx, x, y, out, dout, dx, dy);
    } else {
      ElemwiseGradCompute<DeviceContext, T, MulGradDX<T>, MulGradDY<T>>(
          ctx, *x, *y, *out, *dout, axis, dx, dy, MulGradDX<T>(),
          MulGradDY<T>());
    }
  }
};

template <typename DeviceContext, typename T>
class ElementwiseMulDoubleGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    using Tensor = framework::Tensor;

    auto* x = ctx.Input<Tensor>("X");
    auto* y = ctx.Input<Tensor>("Y");
    auto* dout = ctx.Input<Tensor>("DOut");
    auto* ddx = ctx.Input<Tensor>("DDX");
    auto* ddy = ctx.Input<Tensor>("DDY");

    auto* dx = ctx.Output<Tensor>(framework::GradVarName("X"));
    auto* dy = ctx.Output<Tensor>(framework::GradVarName("Y"));
    auto* ddout = ctx.Output<Tensor>("DDOut");

    if (ddout) ddout->mutable_data<T>(ctx.GetPlace());

    Tensor ddx_safe, ddy_safe;
    GetDoubleGradSafeTensor<DeviceContext, T>(ctx, x, ddx, &ddx_safe);
    GetDoubleGradSafeTensor<DeviceContext, T>(ctx, y, ddy, &ddy_safe);

    // dx = dout * ddy
    // dy = dout * ddx
    // ddout = ddx * y + x * ddy
    // change computation sequence to save memory, so ddout can inplace ddx and
    // dx can be used as 'tmp' tensor
    // (1) dx = x * ddy
    // (2) dy = dout * ddx
    // (3) ddout = ddx * y
    // (4) ddout = ddout + dx
    // (5) dx = dout * ddy
    if (ddout) {
      int axis = ctx.Attr<int>("axis");
      auto& place =
          *ctx.template device_context<DeviceContext>().eigen_device();
      // size(ddout) > size(ddx), ddout can't use memory of ddx using inplace
      if (ddout->numel() > ddx->numel()) {
        ElemwiseGradCompute<DeviceContext, T, MulGradDX<T>, MulGradDY<T>>(
            ctx, ddx_safe, ddy_safe, *dout, *dout, axis, dx, dy, MulGradDX<T>(),
            MulGradDY<T>());

        Tensor ddout_tmp;
        ddout_tmp.mutable_data<T>(ddout->dims(), ctx.GetPlace());

        default_elementwise_mul<DeviceContext, T>(ctx, y, &ddx_safe, ddout);
        default_elementwise_mul<DeviceContext, T>(ctx, &ddy_safe, x,
                                                  &ddout_tmp);

        auto ddout_t = framework::EigenVector<T>::Flatten(*ddout);
        auto ddout_tmp_t = framework::EigenVector<T>::Flatten(ddout_tmp);
        ddout_t.device(place) = ddout_t + ddout_tmp_t;
      } else {
        // use dx to save memory, other than alloc tmp tensor
        Tensor* ddout_tmp = dx;

        default_elementwise_mul<DeviceContext, T>(ctx, x, &ddy_safe, ddout_tmp);
        // NOTE: in the following ElemwiseGradCompute, for the
        // first output tensor is nullptr, the branch to calculate first
        // output tensor will not be activated, DivGradDx function will not
        // be called and can be ignored, the first branch has little effect
        // on running speed.
        ElemwiseGradCompute<DeviceContext, T, MulGradDX<T>, MulGradDY<T>>(
            ctx, ddx_safe, ddy_safe, *dout, *dout, axis, nullptr, dy,
            MulGradDX<T>(), MulGradDY<T>());
        default_elementwise_mul<DeviceContext, T>(ctx, &ddx_safe, y, ddout);

        auto ddout_t = framework::EigenVector<T>::Flatten(*ddout);
        auto ddout_tmp_t = framework::EigenVector<T>::Flatten(*ddout_tmp);
        ddout_t.device(place) = ddout_t + ddout_tmp_t;
        default_elementwise_mul<DeviceContext, T>(ctx, dout, &ddy_safe, dx);
      }
    }
  }
};

}  // namespace operators
}  // namespace paddle
