//   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifdef PADDLE_WITH_XPU

#include "paddle/fluid/operators/matmul_v2_op.h"
#include <string>
#include <vector>

namespace paddle {
namespace operators {

template <typename T>
void MatMulXPUFunction(const Tensor* X, const Tensor* Y,
                       const std::vector<std::int64_t>& x_dims,
                       const std::vector<std::int64_t>& y_dims, Tensor* Out,
                       bool trans_x, bool trans_y,
                       const paddle::framework::ExecutionContext& ctx) {
  const int x_ndim = x_dims.size();
  const int y_ndim = y_dims.size();

  auto& dev_ctx =
      ctx.template device_context<paddle::platform::XPUDeviceContext>();

  // currently only support x_ndim == y_dim and non-broadcast case
  PADDLE_ENFORCE_EQ(x_ndim, y_ndim, platform::errors::InvalidArgument(
                                        "Shape mistake in matmul_v2_op"));
  for (int i = 0; i < x_ndim - 2; i++) {
    PADDLE_ENFORCE_EQ(
        x_dims.data()[i], y_dims.data()[i],
        platform::errors::InvalidArgument("Shape mistake in matmul_v2_op"));
  }

  int ret = 0;
  if (x_ndim == 1 && y_ndim == 1) {
    PADDLE_ENFORCE_EQ(X->numel(), Y->numel(),
                      platform::errors::InvalidArgument(
                          "X's numbers is not equal to Y's numbers,"
                          "when X/Y's dims =1"));
    VLOG(3) << "MatMul's case 1";
    Out->Resize({1});
    Out->mutable_data<T>(ctx.GetPlace());
    ret = baidu::xpu::api::fc_int16(dev_ctx.x_context(), false, false, 1, 1,
                                    X->numel(), 1.0f, X->data<T>(),
                                    Y->data<T>(), 0.0f, Out->data<T>());
    PADDLE_ENFORCE_EQ(
        ret, XPU_SUCCESS,
        platform::errors::External(
            "XPU API return wrong value[%d] in matmul_v2, please check whether "
            "Baidu Kunlun Card is properly installed.",
            ret));
    return;
  }

  if (x_ndim == 1) {
    const int N = X->numel();
    if (trans_y) {
      PADDLE_ENFORCE_EQ(
          y_dims[y_ndim - 1], N,
          platform::errors::InvalidArgument("Input(Y) has error dim."));
    } else {
      PADDLE_ENFORCE_EQ(
          y_dims[y_ndim - 2], N,
          platform::errors::InvalidArgument("Input(Y) has error dim."));
    }
    std::vector<std::int64_t> out_dims(y_ndim - 1);
    if (trans_y) {
      std::copy_n(y_dims.cbegin(), y_ndim - 1, out_dims.begin());
    } else {
      std::copy_n(y_dims.cbegin(), y_ndim - 2, out_dims.begin());
      out_dims.back() = y_dims.back();
    }
    Out->Resize(framework::make_ddim(out_dims));
    Out->mutable_data<T>(ctx.GetPlace());
    if (trans_y) {
      const int M = Y->numel() / N;
      VLOG(3) << "MatMul's case 2";
      ret = baidu::xpu::api::fc_int16(dev_ctx.x_context(), false, true, 1, M, N,
                                      1.0f, X->data<T>(), Y->data<T>(), 0.0f,
                                      Out->data<T>());
      PADDLE_ENFORCE_EQ(
          ret, XPU_SUCCESS,
          platform::errors::External("XPU API return wrong value[%d] in "
                                     "matmul_v2, please check whether "
                                     "Baidu Kunlun Card is properly installed.",
                                     ret));
    } else {
      const int M = y_dims[y_ndim - 1];
      const int batch_size = Y->numel() / (M * N);
      for (int i = 0; i < batch_size; i++) {
        ret = baidu::xpu::api::fc_int16(
            dev_ctx.x_context(), false, false, 1, M, N, 1.0f, X->data<T>(),
            Y->data<T>() + i * M * N, 0.0f, Out->data<T>() + i * M);
        PADDLE_ENFORCE_EQ(ret, XPU_SUCCESS,
                          platform::errors::External(
                              "XPU API return wrong value[%d] in matmul_v2, "
                              "please check whether "
                              "Baidu Kunlun Card is properly installed.",
                              ret));
      }
    }
    return;
  }

  if (y_ndim == 1) {
    const int N = Y->numel();
    if (trans_x) {
      PADDLE_ENFORCE_EQ(
          x_dims[x_ndim - 2], N,
          platform::errors::InvalidArgument("Input(X) has error dim."));
    } else {
      PADDLE_ENFORCE_EQ(
          x_dims[x_ndim - 1], N,
          platform::errors::InvalidArgument("Input(X) has error dim."));
    }
    std::vector<std::int64_t> out_dims(x_ndim - 1);
    if (trans_x) {
      std::copy_n(x_dims.cbegin(), x_ndim - 2, out_dims.begin());
      out_dims.back() = x_dims.back();
    } else {
      std::copy_n(x_dims.cbegin(), x_ndim - 1, out_dims.begin());
    }
    Out->Resize(framework::make_ddim(out_dims));
    Out->mutable_data<T>(ctx.GetPlace());

    if (trans_x) {
      const int M = x_dims[x_ndim - 1];
      const int batch_size = X->numel() / (M * N);
      for (int i = 0; i < batch_size; i++) {
        ret = baidu::xpu::api::fc_int16(dev_ctx.x_context(), true, false, M, 1,
                                        N, 1.0f, X->data<T>() + i * M * N,
                                        Y->data<T>(), 0.0f,
                                        Out->data<T>() + i * M);
        PADDLE_ENFORCE_EQ(ret, XPU_SUCCESS,
                          platform::errors::External(
                              "XPU API return wrong value[%d] in matmul_v2, "
                              "please check whether "
                              "Baidu Kunlun Card is properly installed.",
                              ret));
      }
    } else {
      const int M = X->numel() / N;
      VLOG(3) << "MatMul's case 7";
      ret = baidu::xpu::api::fc_int16(dev_ctx.x_context(), false, false, M, 1,
                                      N, 1.0f, X->data<T>(), Y->data<T>(), 0.0f,
                                      Out->data<T>());
      PADDLE_ENFORCE_EQ(
          ret, XPU_SUCCESS,
          platform::errors::External("XPU API return wrong value[%d] in "
                                     "matmul_v2, please check whether "
                                     "Baidu Kunlun Card is properly installed.",
                                     ret));
    }
    return;
  }

  const int M = trans_x ? x_dims[x_ndim - 1] : x_dims[x_ndim - 2];
  const int K = trans_x ? x_dims[x_ndim - 2] : x_dims[x_ndim - 1];
  if (trans_y) {
    PADDLE_ENFORCE_EQ(y_dims[y_ndim - 1], K, platform::errors::InvalidArgument(
                                                 "Input(X) has error dim."));
  } else {
    PADDLE_ENFORCE_EQ(y_dims[y_ndim - 2], K, platform::errors::InvalidArgument(
                                                 "Input(X) has error dim."));
  }
  const int N = trans_y ? y_dims[y_ndim - 2] : y_dims[y_ndim - 1];
  const int ndim = (std::max)(x_ndim, y_ndim);
  std::vector<std::int64_t> out_broadcast_dims(ndim);
  int batch_size = 1;
  for (int i = 0; i < ndim - 2; i++) {
    PADDLE_ENFORCE_EQ(
        x_dims.data()[i], y_dims.data()[i],
        platform::errors::InvalidArgument("Shape mistake in matmul_v2_op"));
    out_broadcast_dims[i] = x_dims.data()[i];
    batch_size *= x_dims.data()[i];
  }

  out_broadcast_dims[ndim - 2] = M;
  out_broadcast_dims[ndim - 1] = N;

  Out->Resize(framework::make_ddim(out_broadcast_dims));
  Out->mutable_data<T>(ctx.GetPlace());
  ret = baidu::xpu::api::batched_gemm_int16(
      dev_ctx.x_context(), trans_x, trans_y, batch_size, M, N, K, 1.0f,
      X->data<T>(), Y->data<T>(), Out->data<T>(), nullptr, nullptr);
  PADDLE_ENFORCE_EQ(
      ret, XPU_SUCCESS,
      platform::errors::External(
          "XPU API return wrong value[%d] in matmul_v2, please check whether "
          "Baidu Kunlun Card is properly installed.",
          ret));
}

template <typename T>
class MatMulV2XPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const paddle::framework::ExecutionContext& ctx) const override {
    auto* X = ctx.Input<Tensor>("X");
    auto* Y = ctx.Input<Tensor>("Y");
    auto* Out = ctx.Output<Tensor>("Out");
    bool trans_x = ctx.Attr<bool>("trans_x");
    bool trans_y = ctx.Attr<bool>("trans_y");
    MatMulXPUFunction<T>(X, Y, vectorize(X->dims()), vectorize(Y->dims()), Out,
                         trans_x, trans_y, ctx);
  }
};

template <typename T>
class MatMulV2XPUGradKernel : public framework::OpKernel<T> {
 public:
  void MatMul(const framework::ExecutionContext& context,
              const framework::Tensor& a, bool trans_a,
              const framework::Tensor& b, bool trans_b,
              framework::Tensor* out) const {
    out->mutable_data<T>(context.GetPlace());
    MatMulXPUFunction<T>(&a, &b, vectorize(a.dims()), vectorize(b.dims()), out,
                         trans_a, trans_b, context);
  }

  void CalcInputGrad(const framework::ExecutionContext& context,
                     const framework::Tensor& a, bool trans_a,
                     bool is_fold_init_dims_a, const framework::Tensor& b,
                     bool trans_b, bool is_fold_init_dims_b,
                     framework::Tensor* out) const {
    if (out == nullptr) return;
    bool need_combine = (a.dims().size() == 3 || b.dims().size() == 3) &&
                        out->dims().size() == 2;
    if (!need_combine) {
      MatMul(context, a, trans_a, b, trans_b, out);
    } else {
      // currently not support this case
    }
  }

  void Compute(const framework::ExecutionContext& ctx) const override {
    bool transpose_x = ctx.Attr<bool>("trans_x");
    bool transpose_y = ctx.Attr<bool>("trans_y");

    auto x = *ctx.Input<framework::Tensor>("X");
    auto y = *ctx.Input<framework::Tensor>("Y");
    auto dout = *ctx.Input<framework::Tensor>(framework::GradVarName("Out"));

    // get dims
    std::vector<std::int64_t> x_dims = vectorize(x.dims());
    std::vector<std::int64_t> y_dims = vectorize(y.dims());
    std::vector<std::int64_t> dout_dims = vectorize(dout.dims());

    int x_ndim = x_dims.size();
    int y_ndim = y_dims.size();

    auto* dx = ctx.Output<Tensor>(framework::GradVarName("X"));
    auto* dy = ctx.Output<Tensor>(framework::GradVarName("Y"));

    auto& dev_ctx =
        ctx.template device_context<paddle::platform::XPUDeviceContext>();
    // Case1 : x's or y's dim = 1
    int ret = 0;
    if (x_ndim == 1 && y_ndim == 1) {
      if (dx) {
        dx->mutable_data<T>(ctx.GetPlace());
        ret = baidu::xpu::api::fc_int16(dev_ctx.x_context(), false, false,
                                        dx->numel(), 1, 1, 1.0f, y.data<T>(),
                                        dout.data<T>(), 0.0f, dx->data<T>());
        PADDLE_ENFORCE_EQ(ret, XPU_SUCCESS,
                          platform::errors::External(
                              "XPU API return wrong value[%d] in "
                              "matmul_v2_grad, please check whether "
                              "Baidu Kunlun Card is properly installed.",
                              ret));
      }
      if (dy) {
        dy->mutable_data<T>(ctx.GetPlace());
        ret = baidu::xpu::api::fc_int16(dev_ctx.x_context(), false, false,
                                        dy->numel(), 1, 1, 1.0f, x.data<T>(),
                                        dout.data<T>(), 0.0f, dy->data<T>());
        PADDLE_ENFORCE_EQ(ret, XPU_SUCCESS,
                          platform::errors::External(
                              "XPU API return wrong value[%d] in "
                              "matmul_v2_grad, please check whether "
                              "Baidu Kunlun Card is properly installed.",
                              ret));
      }
      return;
    }

    bool is_broadcast = true;
    if (x_ndim <= 2 || y_ndim <= 2) {
      is_broadcast = false;
    } else if (x_ndim != y_ndim) {
      is_broadcast = true;
    } else {
      is_broadcast = !std::equal(x_dims.cbegin(), x_dims.cbegin() + x_ndim - 2,
                                 y_dims.cbegin());
    }

    // currently only support non-broadcast case
    PADDLE_ENFORCE_EQ(
        is_broadcast, false,
        platform::errors::InvalidArgument("Shape mistake in matmul_v2_op"));

    // Case2: no broadcast or no batch size, it aims to speed and it is same as
    // matmul in old version.
    if (!is_broadcast) {
      ReshapeXYOutIntoMatrixSequence(&x, &y, &dout, transpose_x, transpose_y);
      framework::DDim dx_dims;
      if (dx) {
        dx_dims = dx->dims();
        if (dx_dims != x.dims()) {
          dx->Resize(x.dims());
        }
      }

      framework::DDim dy_dims;
      if (dy) {
        dy_dims = dy->dims();
        if (dy_dims != y.dims()) {
          dy->Resize(y.dims());
        }
      }
      if (transpose_x && transpose_y) {
        CalcInputGrad(ctx, y, true, true, dout, true, false, dx);
        CalcInputGrad(ctx, dout, true, true, x, true, false, dy);
      } else if (transpose_x) {
        CalcInputGrad(ctx, y, false, false, dout, true, false, dx);
        CalcInputGrad(ctx, x, false, false, dout, false, true, dy);
      } else if (transpose_y) {
        CalcInputGrad(ctx, dout, false, false, y, false, true, dx);
        CalcInputGrad(ctx, dout, true, true, x, false, true, dy);
      } else {
        CalcInputGrad(ctx, dout, false, false, y, true, false, dx);
        CalcInputGrad(ctx, x, true, true, dout, false, true, dy);
      }

      if (dx) {
        if (dx_dims != x.dims()) {
          dx->Resize(dx_dims);
        }
      }
      if (dy) {
        if (dy_dims != y.dims()) {
          dy->Resize(dy_dims);
        }
      }
    }
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OP_XPU_KERNEL(matmul_v2, ops::MatMulV2XPUKernel<float>);
REGISTER_OP_XPU_KERNEL(matmul_v2_grad, ops::MatMulV2XPUGradKernel<float>);

#endif
