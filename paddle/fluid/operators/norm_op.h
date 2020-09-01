/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

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
template <typename T = int>
inline void GetDims(const framework::DDim& dim, int axis, T* pre, T* n,
                    T* post) {
  *pre = 1;
  *post = 1;
  *n = static_cast<T>(dim[axis]);
  for (int i = 0; i < axis; ++i) {
    (*pre) *= static_cast<T>(dim[i]);
  }
  for (int i = axis + 1; i < dim.size(); ++i) {
    (*post) *= static_cast<T>(dim[i]);
  }
}

template <typename DeviceContext, typename T>
class NormKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* in_x = ctx.Input<framework::Tensor>("X");
    auto* out_y = ctx.Output<framework::Tensor>("Out");
    auto* out_norm = ctx.Output<framework::Tensor>("Norm");
    out_y->mutable_data<T>(ctx.GetPlace());
    out_norm->mutable_data<T>(ctx.GetPlace());

    auto xdim = in_x->dims();
    T eps = static_cast<T>(ctx.Attr<float>("epsilon"));
    int axis = ctx.Attr<int>("axis");
    if (axis < 0) axis = xdim.size() + axis;
    int pre, n, post;
    GetDims(xdim, axis, &pre, &n, &post);

    auto* place = ctx.template device_context<DeviceContext>().eigen_device();

    Eigen::DSizes<int, 3> shape(pre, n, post);
    Eigen::DSizes<int, 2> norm_shape(pre, post);

    auto x_e = framework::EigenVector<T>::Flatten(*in_x);
    auto y_e = framework::EigenVector<T>::Flatten(*out_y);
    auto norm_e = framework::EigenVector<T>::Flatten(*out_norm);
    auto x = x_e.reshape(shape);
    auto y = y_e.reshape(shape);
    auto norm = norm_e.reshape(norm_shape);

    Eigen::DSizes<int, 1> rdim(1);
    // y = x / sqrt((sum(x * x) + epsilon))
    // norm = sqrt(sum(x * x) + epsilon)
    auto x2 = x * x;
    auto sum = x2.sum(rdim) + eps;
    norm.device(*place) = sum.sqrt();

    // y = x / norm
    Eigen::DSizes<int, 3> rshape(pre, 1, post);
    Eigen::DSizes<int, 3> bcast(1, n, 1);
    y.device(*place) = x / norm.reshape(rshape).broadcast(bcast);
  }
};

template <typename DeviceContext, typename T, typename AttrType = T>
class NormGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* in_x = ctx.Input<framework::Tensor>("X");
    auto* in_norm = ctx.Input<framework::Tensor>("Norm");
    auto* in_dy = ctx.Input<framework::Tensor>(framework::GradVarName("Out"));
    auto* out_dx = ctx.Output<framework::Tensor>(framework::GradVarName("X"));
    out_dx->mutable_data<T>(ctx.GetPlace());

    auto xdim = in_x->dims();
    int axis = ctx.Attr<int>("axis");
    if (axis < 0) axis = xdim.size() + axis;
    int pre, n, post;
    GetDims(xdim, axis, &pre, &n, &post);

    auto* place = ctx.template device_context<DeviceContext>().eigen_device();

    auto x_e = framework::EigenVector<T>::Flatten(*in_x);
    auto dy_e = framework::EigenVector<T>::Flatten(*in_dy);
    auto norm_e = framework::EigenVector<T>::Flatten(*in_norm);
    auto dx_e = framework::EigenVector<T>::Flatten(*out_dx);

    Eigen::DSizes<int, 3> shape(pre, n, post);
    Eigen::DSizes<int, 3> rshape(pre, 1, post);
    auto x = x_e.reshape(shape);
    auto dy = dy_e.reshape(shape);
    auto norm = norm_e.reshape(rshape);
    auto dx = dx_e.reshape(shape);

    framework::Tensor rsum;
    rsum.mutable_data<T>({pre, post}, ctx.GetPlace());
    auto sum = framework::EigenTensor<T, 2>::From(rsum);

    Eigen::DSizes<int, 1> rdim(1);
    Eigen::DSizes<int, 3> bcast(1, n, 1);

    // dx = ( dy/sqrt(sum(x*x)) ) * [1 - x*sum(x) / (sum(x*x) + e)]
    //    = [dy - dy * x * sum(x) / (sum(x*x) + e)] / sqrt(sum(x*x))
    //    = [dy - x * sum(x*dy) / (sum(x*x) + e)] / sqrt(sum(x*x))
    // 1. sum = sum(x*dy)
    sum.device(*place) = (x * dy).sum(rdim);
    // 2. dx = x * sum
    dx.device(*place) = sum.reshape(rshape).broadcast(bcast) * x;
    // 3. dx / (sum(x*x) + e)
    // where, norm.pow(2) = sum(x*x) + e, which is calculated in forward.
    dx.device(*place) = dx / norm.pow(2).broadcast(bcast);
    // 4. [dy - dx] / sqrt(sum(x*x))
    dx.device(*place) = (dy - dx) / norm.broadcast(bcast);
  }
};
}  // namespace operators
}  // namespace paddle
