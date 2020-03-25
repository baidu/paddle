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

#include <vector>
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/operators/math/blas.h"
#include "paddle/fluid/operators/math/math_function.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
constexpr int kMULMKLDNNINT8 = 1;

template <typename DeviceContext>
struct QuantFp32ToInt8Functor {
  void operator()(const DeviceContext& ctx, const framework::Tensor& in,
                  const float scale, framework::Tensor* out) {}
};

template <typename DeviceContext>
struct GEMMINT8Functor {
  void operator()(const DeviceContext& ctx, bool transA, bool transB, int M,
                  int N, int K, float alpha, const int8_t* A, int lda,
                  const int8_t* B, int ldb, float beta, float* C, int ldc) {}
  void operator()(const DeviceContext& ctx, bool transA, bool transB, int M,
                  int N, int K, int32_t alpha, const int8_t* A, int lda,
                  const int8_t* B, int ldb, int32_t beta, int32_t* C, int ldc) {
  }
};

template <typename DeviceContext>
struct INT32ToFP32Functor {
  void operator()(const DeviceContext& ctx, const framework::Tensor& in,
                  framework::Tensor* out, float scale) {}
};

template <typename DeviceContext, typename T>
class MulKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    const Tensor* x = context.Input<Tensor>("X");
    const Tensor* y = context.Input<Tensor>("Y");
    Tensor* z = context.Output<Tensor>("Out");
    const Tensor x_matrix =
        x->dims().size() > 2
            ? framework::ReshapeToMatrix(
                  *x, context.template Attr<int>("x_num_col_dims"))
            : *x;
    const Tensor y_matrix =
        y->dims().size() > 2
            ? framework::ReshapeToMatrix(
                  *y, context.template Attr<int>("y_num_col_dims"))
            : *y;

    z->mutable_data<T>(context.GetPlace());
    auto z_dim = z->dims();
    if (z_dim.size() != 2) {
      z->Resize({x_matrix.dims()[0], y_matrix.dims()[1]});
    }
    int M, N, K;
    M = x_matrix.dims()[0];
    N = y_matrix.dims()[1];
    K = y_matrix.dims()[0];

    if (context.HasAttr("enable_int8") && context.Attr<bool>("enable_int8")) {
      float in_scale = context.Attr<float>("X_scale");
      std::vector<float> weight_scale =
          context.Attr<std::vector<float>>("weight_scale");

      PADDLE_ENFORCE_EQ(weight_scale.size(), 1,
                        "weight scale size shoud be equal to 1");
      auto& dev_ctx = context.template device_context<DeviceContext>();

      Tensor x_int8;
      x_int8.Resize(x->dims());
      x_int8.mutable_data<int8_t>(context.GetPlace());

      QuantFp32ToInt8Functor<DeviceContext> quant_func;
      quant_func(dev_ctx, *x, in_scale / 127., &x_int8);
      // the float here represents the output type
      const int8_t* x_int8_data = x_int8.data<int8_t>();
      const int8_t* y_int8_data = y->data<int8_t>();

      if (N % 4 == 0) {
        Tensor x_int8, out_int8;
        out_int8.Resize(z_dim);
        int32_t* out_int8_data =
            out_int8.mutable_data<int32_t>(context.GetPlace());
        int32_t alpha = 1;
        int32_t beta = 0;
        float scale =
            static_cast<float>(in_scale * weight_scale[0] / 127. / 127.);
        GEMMINT8Functor<DeviceContext> gemm_int8_func;
        gemm_int8_func(dev_ctx, false, false, M, N, K, alpha, x_int8_data, K,
                       y_int8_data, N, beta, out_int8_data, N);
        INT32ToFP32Functor<DeviceContext> int32_to_fp32_func;
        int32_to_fp32_func(dev_ctx, out_int8, z, scale);
      } else {
        float* z_data = z->mutable_data<float>(context.GetPlace());
        float alpha =
            static_cast<float>(in_scale * weight_scale[0] / 127. / 127.);
        float beta = 0.0f;
        GEMMINT8Functor<DeviceContext> gemm_int8_func;
        gemm_int8_func(dev_ctx, false, false, M, N, K, alpha, x_int8_data, K,
                       y_int8_data, N, beta, z_data, N);
      }

      if (z_dim.size() != 2) {
        z->Resize(z_dim);
      }
      return;
    }

    auto blas = math::GetBlas<DeviceContext, T>(context);

    blas.MatMul(x_matrix, y_matrix, z);
    if (z_dim.size() != 2) {
      z->Resize(z_dim);
    }
  }
};

template <typename DeviceContext, typename T>
class MulGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    int x_num_col_dims = ctx.template Attr<int>("x_num_col_dims");
    int y_num_col_dims = ctx.template Attr<int>("y_num_col_dims");
    auto* x = ctx.Input<framework::LoDTensor>("X");
    auto* y = ctx.Input<framework::LoDTensor>("Y");
    auto x_matrix = x->dims().size() > 2
                        ? framework::ReshapeToMatrix(*x, x_num_col_dims)
                        : static_cast<const Tensor&>(*x);
    auto y_matrix = y->dims().size() > 2
                        ? framework::ReshapeToMatrix(*y, y_num_col_dims)
                        : static_cast<const Tensor&>(*y);
    auto* dout = ctx.Input<framework::LoDTensor>(framework::GradVarName("Out"));

    Tensor dout_mat;
    dout_mat.ShareDataWith(*dout);
    dout_mat.Resize({framework::flatten_to_2d(x->dims(), x_num_col_dims)[0],
                     framework::flatten_to_2d(y->dims(), y_num_col_dims)[1]});

    auto* dx = ctx.Output<framework::LoDTensor>(framework::GradVarName("X"));
    auto* dy = ctx.Output<framework::LoDTensor>(framework::GradVarName("Y"));

    if (dx != nullptr) {
      dx->set_lod(x->lod());
    }
    if (dy != nullptr) {
      dy->set_lod(y->lod());
    }

    auto& dev_ctx = ctx.template device_context<DeviceContext>();
    auto blas = math::GetBlas<DeviceContext, T>(dev_ctx);
    if (dx) {
      dx->mutable_data<T>(ctx.GetPlace());
      Tensor dx_matrix = dx->dims().size() > 2
                             ? framework::ReshapeToMatrix(*dx, x_num_col_dims)
                             : *dx;

      // dx = dout * y'. dx: M x K, dout : M x N, y : K x N
      blas.MatMul(dout_mat, false, y_matrix, true, &dx_matrix);
    }
    if (dy) {
      dy->mutable_data<T>(ctx.GetPlace());
      Tensor dy_matrix = dy->dims().size() > 2
                             ? framework::ReshapeToMatrix(*dy, y_num_col_dims)
                             : *dy;
      // dy = x' * dout. dy K x N, dout : M x N, x : M x K
      blas.MatMul(x_matrix, true, dout_mat, false, &dy_matrix);
    }
  }
};

template <typename DeviceContext, typename T>
class MulDoubleGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    int x_num_col_dims = ctx.template Attr<int>("x_num_col_dims");
    int y_num_col_dims = ctx.template Attr<int>("y_num_col_dims");
    auto* x = ctx.Input<framework::LoDTensor>("X");
    auto* y = ctx.Input<framework::LoDTensor>("Y");
    auto x_mat = x->dims().size() > 2
                     ? framework::ReshapeToMatrix(*x, x_num_col_dims)
                     : static_cast<const Tensor&>(*x);
    auto y_mat = y->dims().size() > 2
                     ? framework::ReshapeToMatrix(*y, y_num_col_dims)
                     : static_cast<const Tensor&>(*y);

    const int m = framework::flatten_to_2d(x->dims(), x_num_col_dims)[0];
    const int n = framework::flatten_to_2d(y->dims(), y_num_col_dims)[1];

    auto* dout = ctx.Input<framework::LoDTensor>("DOut");
    Tensor dout_mat;
    dout_mat.ShareDataWith(*dout);
    dout_mat.Resize({m, n});

    auto* ddx = ctx.Input<framework::LoDTensor>("DDX");
    auto* ddy = ctx.Input<framework::LoDTensor>("DDY");

    auto* dx = ctx.Output<framework::LoDTensor>("DX");
    auto* dy = ctx.Output<framework::LoDTensor>("DY");
    auto* ddout = ctx.Output<framework::LoDTensor>("DDOut");

    Tensor ddout_mat;
    if (ddout) {
      ddout->set_lod(dout->lod());
      // allocate and reshape ddout
      ddout->mutable_data<T>(ctx.GetPlace());
      ddout_mat.ShareDataWith(*ddout);
      ddout_mat.Resize({m, n});
    }

    auto& dev_ctx = ctx.template device_context<DeviceContext>();
    auto blas = math::GetBlas<DeviceContext, T>(dev_ctx);
    // a flag to specify whether ddout value has been set, if flag
    // is false, MatMul beta should be 0 to set ddout, if flag is
    // true, MatMul beta should be 1 to add result to ddout.
    bool ddout_flag = false;
    if (ddx) {
      auto ddx_mat = ddx->dims().size() > 2
                         ? framework::ReshapeToMatrix(*ddx, x_num_col_dims)
                         : static_cast<const Tensor&>(*ddx);

      // dy = ddx' * dout. dy : K x M, ddx' : K x M, dout : M x N
      if (dy) {
        dy->set_lod(y->lod());
        // allocate and reshape dy
        dy->mutable_data<T>(ctx.GetPlace());
        Tensor dy_mat = dy->dims().size() > 2
                            ? framework::ReshapeToMatrix(*dy, y_num_col_dims)
                            : *dy;
        blas.MatMul(ddx_mat, true, dout_mat, false, &dy_mat);
      }
      // ddout1 = ddx * y. ddx : M x K, y : K x N, ddout1 : M x N
      if (ddout) {
        blas.MatMul(ddx_mat, false, y_mat, false, static_cast<T>(1.0),
                    &ddout_mat, static_cast<T>(ddout_flag));
        ddout_flag = true;
      }
    }
    if (ddy) {
      auto ddy_mat = ddy->dims().size() > 2
                         ? framework::ReshapeToMatrix(*ddy, y_num_col_dims)
                         : static_cast<const Tensor&>(*ddy);
      // dx = dout * ddy'. dout : M x N, ddy' : N x K, dx : M x K
      if (dx) {
        dx->set_lod(x->lod());
        // allocate and reshape dx
        dx->mutable_data<T>(ctx.GetPlace());
        Tensor dx_mat = dx->dims().size() > 2
                            ? framework::ReshapeToMatrix(*dx, x_num_col_dims)
                            : *dx;
        blas.MatMul(dout_mat, false, ddy_mat, true, &dx_mat);
      }
      // ddout2 = x * ddy. x : M x K, ddy : K x N, ddout2 : M x N
      if (ddout) {
        blas.MatMul(x_mat, false, ddy_mat, false, static_cast<T>(1.0),
                    &ddout_mat, static_cast<T>(ddout_flag));
      }
    }
  }
};

}  // namespace operators
}  // namespace paddle
