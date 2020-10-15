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

#include <algorithm>
#include <utility>
#include <vector>
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/math/blas.h"
#ifdef PADDLE_WITH_MKLDNN
#include "paddle/fluid/platform/mkldnn_helper.h"
#endif

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;

template <typename DeviceContext, typename T>
class MVKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &context) const override {
    auto *x = context.Input<framework::Tensor>("X");
    auto *vec = context.Input<framework::Tensor>("Vec");

    auto *out = context.Output<framework::Tensor>("Out");

    auto dim_x = x->dims();

    // get data ptr
    const T *x_data = x->data<T>();
    const T *vec_data = vec->data<T>();
    T *out_data = out->mutable_data<T>(context.GetPlace());

    auto &dev_ctx = context.template device_context<DeviceContext>();
    auto blas = math::GetBlas<DeviceContext, T>(dev_ctx);

    blas.GEMV(false, dim_x[0], dim_x[1], static_cast<T>(1), x_data, vec_data,
              static_cast<T>(0), out_data);
  }
};

// Using dimensional constraints on matrix multiplication, it is
// straight-forward to check the following table for when X and Y
// are both matrices.
//
// dX = | dOut vec^T
// dVec = | X^T dOut
template <typename DeviceContext, typename T>
class MVGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &context) const override {
    auto *x = context.Input<framework::Tensor>("X");
    auto *vec = context.Input<framework::Tensor>("Vec");
    auto *dout =
        context.Input<framework::Tensor>(framework::GradVarName("Out"));
    auto *dx = context.Output<framework::Tensor>(framework::GradVarName("X"));
    auto *dvec =
        context.Output<framework::Tensor>(framework::GradVarName("Vec"));

    auto dim_x = x->dims();
    int m = dim_x[0];
    int n = dim_x[1];

    // get data ptr
    const T *x_data = x->data<T>();
    const T *vec_data = vec->data<T>();
    const T *dout_data = dout->data<T>();

    if (dx) {
      T *dx_data = dx->mutable_data<T>(context.GetPlace());

      for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
          dx_data[i * n + j] = dout_data[i] * vec_data[j];
        }
      }
    }

    if (dvec) {
      T *dvec_data = dvec->mutable_data<T>(context.GetPlace());

      auto &dev_ctx = context.template device_context<DeviceContext>();
      auto blas = math::GetBlas<DeviceContext, T>(dev_ctx);

      blas.GEMV(true, dim_x[0], dim_x[1], static_cast<T>(1), x_data, dout_data,
                static_cast<T>(0), dvec_data);
    }
  }
};

}  // namespace operators
}  // namespace paddle
