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

/* Acknowledgement: the following code is strongly inspired by
https://github.com/caffe2/caffe2/blob/master/caffe2/operators/lstm_unit_op.h
*/

#pragma once
#include "glog/logging.h"
#include "paddle/fluid/framework/op_registry.h"

namespace paddle {
namespace operators {

using framework::Tensor;

template <typename T>
inline T sigmoid(T x) {
  return 1. / (1. + exp(-x));
}

template <typename T>
inline T tanh(T x) {
  return 2. * sigmoid(2. * x) - 1.;
}

template <typename DeviceContext, typename T>
class LstmUnitKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    PADDLE_ENFORCE_EQ(
        platform::is_cpu_place(ctx.GetPlace()), true,
        paddle::platform::errors::PreconditionNotMet("It must use CPUPlace."));

    auto* x_tensor = ctx.Input<framework::Tensor>("X");
    auto* c_prev_tensor = ctx.Input<framework::Tensor>("C_prev");
    auto* c_tensor = ctx.Output<framework::Tensor>("C");
    auto* h_tensor = ctx.Output<framework::Tensor>("H");

    auto forget_bias = static_cast<T>(ctx.Attr<float>("forget_bias"));

    int b_size = c_tensor->dims()[0];
    int D = c_tensor->dims()[1];

    T* C = c_tensor->mutable_data<T>(ctx.GetPlace());
    T* H = h_tensor->mutable_data<T>(ctx.GetPlace());

    const T* X = x_tensor->data<T>();
    const T* C_prev = c_prev_tensor->data<T>();

    for (int n = 0; n < b_size; ++n) {
      for (int d = 0; d < D; ++d) {
        const T i = sigmoid(X[d]);
        const T f = sigmoid(X[1 * D + d] + forget_bias);
        const T o = sigmoid(X[2 * D + d]);
        const T g = tanh(X[3 * D + d]);
        const T c_prev = C_prev[d];
        const T c = f * c_prev + i * g;
        C[d] = c;
        const T tanh_c = tanh(c);
        H[d] = o * tanh_c;
      }
      C_prev += D;
      X += 4 * D;
      C += D;
      H += D;
    }
  }
};

template <typename DeviceContext, typename T>
class LstmUnitGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    PADDLE_ENFORCE_EQ(
        platform::is_cpu_place(ctx.GetPlace()), true,
        paddle::platform::errors::PreconditionNotMet("It must use CPUPlace."));

    auto x_tensor = ctx.Input<Tensor>("X");
    auto c_prev_tensor = ctx.Input<Tensor>("C_prev");
    auto c_tensor = ctx.Input<Tensor>("C");

    auto hdiff_tensor = ctx.Input<Tensor>(framework::GradVarName("H"));
    auto cdiff_tensor = ctx.Input<Tensor>(framework::GradVarName("C"));

    auto xdiff_tensor = ctx.Output<Tensor>(framework::GradVarName("X"));
    auto c_prev_diff_tensor =
        ctx.Output<Tensor>(framework::GradVarName("C_prev"));

    auto* X = x_tensor->data<T>();
    auto* C_prev = c_prev_tensor->data<T>();
    auto* C = c_tensor->data<T>();

    auto* H_diff = hdiff_tensor->data<T>();
    auto* C_diff = cdiff_tensor->data<T>();

    auto* C_prev_diff = c_prev_diff_tensor->mutable_data<T>(ctx.GetPlace());
    auto* X_diff = xdiff_tensor->mutable_data<T>(ctx.GetPlace());

    int N = c_tensor->dims()[0];
    int D = c_tensor->dims()[1];

    auto forget_bias = static_cast<T>(ctx.Attr<float>("forget_bias"));

    for (int n = 0; n < N; ++n) {
      for (int d = 0; d < D; ++d) {
        T* c_prev_diff = C_prev_diff + d;
        T* i_diff = X_diff + d;
        T* f_diff = X_diff + 1 * D + d;
        T* o_diff = X_diff + 2 * D + d;
        T* g_diff = X_diff + 3 * D + d;

        const T i = sigmoid(X[d]);
        const T f = sigmoid(X[1 * D + d] + forget_bias);
        const T o = sigmoid(X[2 * D + d]);
        const T g = tanh(X[3 * D + d]);
        const T c_prev = C_prev[d];
        const T c = C[d];
        const T tanh_c = tanh(c);
        const T c_term_diff = C_diff[d] + H_diff[d] * o * (1 - tanh_c * tanh_c);
        *c_prev_diff = c_term_diff * f;
        *i_diff = c_term_diff * g * i * (1 - i);
        *f_diff = c_term_diff * c_prev * f * (1 - f);
        *o_diff = H_diff[d] * tanh_c * o * (1 - o);
        *g_diff = c_term_diff * i * (1 - g * g);
      }
      C_prev += D;
      X += 4 * D;
      C += D;
      C_diff += D;
      H_diff += D;
      X_diff += 4 * D;
      C_prev_diff += D;
    }
  }
};

}  // namespace operators
}  // namespace paddle
