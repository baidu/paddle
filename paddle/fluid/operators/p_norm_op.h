/* Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
Indicesou may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/math/math_function.h"

namespace paddle {
namespace operators {

inline void GetDims(const framework::DDim& dim, int axis, int* pre, int* n,
                    int* post, bool asvector) {
  *pre = 1;
  *post = 1;
  *n = dim[axis];
  if (asvector) {
    *n = product(dim);
  } else {
    for (int i = 0; i < axis; ++i) {
      (*pre) *= dim[i];
    }
    for (int i = axis + 1; i < dim.size(); ++i) {
      (*post) *= dim[i];
    }
  }
}

template <typename DeviceContext, typename T>
class PnormKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* in_x = ctx.Input<framework::Tensor>("X");
    auto* out_norm = ctx.Output<framework::Tensor>("Out");
    out_norm->mutable_data<T>(ctx.GetPlace());

    auto xdim = in_x->dims();
    float porder = ctx.Attr<float>("porder");
    int axis = ctx.Attr<int>("axis");
    bool asvector = ctx.Attr<bool>("asvector");
    if (axis < 0) axis = xdim.size() + axis;
    int pre, n, post;
    GetDims(xdim, axis, &pre, &n, &post, asvector);

    auto* place = ctx.template device_context<DeviceContext>().eigen_device();

    Eigen::DSizes<int, 3> shape(pre, n, post);
    Eigen::DSizes<int, 2> norm_shape(pre, post);

    auto x_e = framework::EigenVector<T>::Flatten(*in_x);
    auto norm_e = framework::EigenVector<T>::Flatten(*out_norm);

    auto x = x_e.reshape(shape);
    auto norm = norm_e.reshape(norm_shape);

    // p=0 means number of non-zero elements of (x)
    // p=inf means the maximum of |x|
    // p=-inf means the minimum of |x|
    // otherwise, Lp-norm = pow(sum(pow(|x|, p)), 1/p)
    Eigen::DSizes<int, 1> rdim(1);
    if (porder == 0) {
      norm.device(*place) = (x != x.constant(0)).template cast<T>().sum(rdim);
    } else if (porder == INFINITY) {
      norm.device(*place) = x.abs().maximum(rdim);
    } else if (porder == -INFINITY) {
      norm.device(*place) = x.abs().minimum(rdim);
    } else {
      norm.device(*place) = x.abs().pow(porder).sum(rdim).pow(1.0f / porder);
    }
  }
};

template <typename DeviceContext, typename T, typename AttrType = T>
class PnormGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* in_x = ctx.Input<framework::Tensor>("X");
    auto* in_norm = ctx.Input<framework::Tensor>("Out");
    auto* in_norm_dy =
        ctx.Input<framework::Tensor>(framework::GradVarName("Out"));
    auto* out_dx = ctx.Output<framework::Tensor>(framework::GradVarName("X"));
    out_dx->mutable_data<T>(ctx.GetPlace());

    T eps = static_cast<T>(ctx.Attr<float>("epsilon"));
    auto xdim = in_x->dims();
    float porder = ctx.Attr<float>("porder");

    int axis = ctx.Attr<int>("axis");
    bool asvector = ctx.Attr<bool>("asvector");
    if (axis < 0) axis = xdim.size() + axis;
    int pre, n, post;
    GetDims(xdim, axis, &pre, &n, &post, asvector);
    Eigen::DSizes<int, 3> shape(pre, n, post);
    Eigen::DSizes<int, 3> rshape(pre, 1, post);

    auto* place = ctx.template device_context<DeviceContext>().eigen_device();

    auto x_e = framework::EigenVector<T>::Flatten(*in_x);
    auto dx_e = framework::EigenVector<T>::Flatten(*out_dx);
    auto norm_e = framework::EigenVector<T>::Flatten(*in_norm);
    auto norm_dy_e = framework::EigenVector<T>::Flatten(*in_norm_dy);

    auto x = x_e.reshape(shape);
    auto dx = dx_e.reshape(shape);
    auto norm = norm_e.reshape(rshape);
    auto norm_dy = norm_dy_e.reshape(rshape);

    Eigen::DSizes<int, 1> rdim(1);
    Eigen::DSizes<int, 3> bcast(1, n, 1);

    if (porder == 0) {
      math::SetConstant<DeviceContext, T> set_zero;
      auto& dev_ctx = ctx.template device_context<DeviceContext>();
      set_zero(dev_ctx, out_dx, static_cast<T>(0));
    } else if (porder == INFINITY || porder == -INFINITY) {
      dx.device(*place) =
          (x.abs() == norm.broadcast(bcast)).template cast<T>() * x.sign() *
          norm_dy.broadcast(bcast);
    } else {
      dx.device(*place) =
          (x.abs()).pow(porder - 1.0f) /
          ((norm.broadcast(bcast)).pow(porder - 1.0f) + x.constant(eps));
      dx.device(*place) = dx * norm_dy.broadcast(bcast) * x.sign();
    }
  }
};
}  // namespace operators
}  // namespace paddle
