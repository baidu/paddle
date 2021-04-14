/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <memory>
#include <string>

#include "paddle/fluid/operators/mul_op.h"
#include "paddle/fluid/operators/npu_op_runner.h"

namespace paddle {
namespace operators {

template <typename DeviceContext, typename T>
class MulNPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* x = ctx.Input<framework::Tensor>("X");
    auto* y = ctx.Input<framework::Tensor>("Y");
    auto* out = ctx.Output<framework::Tensor>("Out");
    int x_num_col_dims = ctx.Attr<int>("x_num_col_dims");
    int y_num_col_dims = ctx.Attr<int>("y_num_col_dims");
    auto stream =
        ctx.template device_context<paddle::platform::NPUDeviceContext>()
            .stream();
    if (x_num_col_dims == 1 && y_num_col_dims == 1) {
      if (x->dims().size() == 2 && y->dims().size() == 2) {
        out->mutable_data<T>(ctx.GetPlace());
        auto runner =
            NpuOpRunner("MatMul", {*x, *y}, {*out},
                        {{"transpose_x1", false}, {"transpose_x2", false}});

        runner.Run(stream);
      } else if (x->dims().size() == 3 && y->dims().size() == 2) {
        // reshape
        Tensor tmp_x(x->type());
        int64_t sec_dim = x->dims()[1] * x->dims()[2];
        int64_t first_dim = x->dims()[0];
        tmp_x.Resize(framework::make_ddim({first_dim, sec_dim}));
        tmp_x.mutable_data<T>(ctx.GetPlace());
        framework::TensorCopy(
            *x, ctx.GetPlace(),
            ctx.template device_context<platform::DeviceContext>(), &tmp_x);
        tmp_x.Resize(framework::make_ddim({first_dim, sec_dim}));
        out->mutable_data<T>(ctx.GetPlace());
        // matmul
        auto runner =
            NpuOpRunner("MatMul", {tmp_x, *y}, {*out},
                        {{"transpose_x1", false}, {"transpose_x2", false}});
        runner.Run(stream);
      } else {
        PADDLE_THROW(platform::errors::InvalidArgument("not suppert dims"));
      }
      // to do other
    } else if (x->dims().size() == 3 && y->dims().size() == 2) {
      // for example: x.shape=[2, 3, 4] y.shape=[4, 5], expect [2, 3, 5]
      PADDLE_ENFORCE_EQ(x_num_col_dims, 2,
                        platform::errors::InvalidArgument(
                            "now only support x_num_col_dims == 2: but got %d",
                            x_num_col_dims));
      // flatten => x.shape=[6, 4]
      Tensor tmp_x(x->type());
      int64_t first_dim = x->dims()[0] * x->dims()[1];
      int64_t sec_dim = x->dims()[2];
      tmp_x.Resize(framework::make_ddim({first_dim, sec_dim}));
      tmp_x.mutable_data<T>(ctx.GetPlace());
      framework::TensorCopy(
          *x, ctx.GetPlace(),
          ctx.template device_context<platform::DeviceContext>(), &tmp_x);
      tmp_x.Resize(framework::make_ddim({first_dim, sec_dim}));

      // matmul [6,4] , [4, 5] => [6, 5]
      Tensor tmp_matmul(x->type());
      tmp_matmul.Resize(framework::make_ddim({first_dim, y->dims()[1]}));
      tmp_matmul.mutable_data<T>(ctx.GetPlace());

      auto runner_matmul =
          NpuOpRunner("MatMul", {tmp_x, *y}, {tmp_matmul},
                      {{"transpose_x1", false}, {"transpose_x2", false}});

      runner_matmul.Run(stream);
      // reshape [6, 5] => [2, 3, 5]
      (*out).Resize(
          framework::make_ddim({x->dims()[0], x->dims()[1], y->dims()[1]}));
      out->mutable_data(ctx.GetPlace(), x->type());
      framework::TensorCopy(
          tmp_matmul, ctx.GetPlace(),
          ctx.template device_context<platform::DeviceContext>(), out);
      (*out).Resize(
          framework::make_ddim({x->dims()[0], x->dims()[1], y->dims()[1]}));
    }
  }
};

template <typename DeviceContext, typename T>
class MulGradNPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* x = ctx.Input<framework::Tensor>("X");
    auto* y = ctx.Input<framework::Tensor>("Y");
    auto* dout = ctx.Input<framework::Tensor>(framework::GradVarName("Out"));
    auto* dx = ctx.Output<framework::Tensor>(framework::GradVarName("X"));
    auto* dy = ctx.Output<framework::Tensor>(framework::GradVarName("Y"));
    int x_num_col_dims = ctx.Attr<int>("x_num_col_dims");
    int y_num_col_dims = ctx.Attr<int>("y_num_col_dims");
    auto stream =
        ctx.template device_context<paddle::platform::NPUDeviceContext>()
            .stream();
    if (x_num_col_dims == 1 && y_num_col_dims == 1) {
      if (x->dims().size() == 2 && y->dims().size() == 2) {
        if (dx) {
          dx->mutable_data<T>(ctx.GetPlace());
          auto runner_dx =
              NpuOpRunner("MatMul", {*dout, *y}, {*dx},
                          {{"transpose_x1", false}, {"transpose_x2", true}});

          runner_dx.Run(stream);
        }

        if (dy) {
          dy->mutable_data<T>(ctx.GetPlace());
          auto runner_dy =
              NpuOpRunner("MatMul", {*x, *dout}, {*dy},
                          {{"transpose_x1", true}, {"transpose_x2", false}});

          runner_dy.Run(stream);
        }
      } else if (x->dims().size() == 3 && y->dims().size() == 2) {
        // flatten => x.shape=[6, 4]
        // matmul
        if (dx) {
          // matmul [2, 5] * [12, 5] => [2, 12]
          dx->mutable_data<T>(ctx.GetPlace());
          auto dx_dims = dx->dims();
          dx->Resize(framework::make_ddim({dout->dims()[0], y->dims()[0]}));
          auto runner_matmul =
              NpuOpRunner("MatMul", {*dout, *y}, {*dx},
                          {{"transpose_x1", false}, {"transpose_x2", true}});
          runner_matmul.Run(stream);
          // reshape [2, 12] => [2, 3, 4]
          dx->Resize(dx_dims);
        }

        if (dy) {
          // flatten
          Tensor tmp_x(x->type());
          int64_t sec_dim = x->dims()[1] * x->dims()[2];
          int64_t first_dim = x->dims()[0];
          tmp_x.Resize(framework::make_ddim({first_dim, sec_dim}));
          tmp_x.mutable_data<T>(ctx.GetPlace());
          framework::TensorCopy(
              *x, ctx.GetPlace(),
              ctx.template device_context<platform::DeviceContext>(), &tmp_x);
          tmp_x.Resize(framework::make_ddim({first_dim, sec_dim}));
          dy->mutable_data<T>(ctx.GetPlace());
          auto runner_dy =
              NpuOpRunner("MatMul", {tmp_x, *dout}, {*dy},
                          {{"transpose_x1", true}, {"transpose_x2", false}});

          runner_dy.Run(stream);
        }
      }
    } else if (x->dims().size() == 3 && y->dims().size() == 2) {
      // for example: x.shape=[2, 3, 4] y.shape=[4, 5], expect [2, 3, 5]
      PADDLE_ENFORCE_EQ(x_num_col_dims, 2,
                        platform::errors::InvalidArgument(
                            "now only support x_num_col_dims == 2: but got %d",
                            x_num_col_dims));
      // tmp_dout both used by dx and dy
      Tensor tmp_dout(x->type());
      int64_t dout_first_dim = dout->dims()[0] * dout->dims()[1];
      int64_t dout_sec_dim = dout->dims()[2];
      tmp_dout.Resize(framework::make_ddim({dout_first_dim, dout_sec_dim}));
      tmp_dout.mutable_data<T>(ctx.GetPlace());
      framework::TensorCopy(
          *dout, ctx.GetPlace(),
          ctx.template device_context<platform::DeviceContext>(), &tmp_dout);
      tmp_dout.Resize(framework::make_ddim({dout_first_dim, dout_sec_dim}));

      if (dx) {
        // tmp_dout * y [6,5] * [4,5] => [6, 4]
        dx->mutable_data<T>(ctx.GetPlace());
        auto dx_dims = dx->dims();
        dx->Resize(framework::make_ddim({dout_first_dim, y->dims()[0]}));
        auto runner_matmul =
            NpuOpRunner("MatMul", {tmp_dout, *y}, {*dx},
                        {{"transpose_x1", false}, {"transpose_x2", true}});
        runner_matmul.Run(stream);
        // reshape [2, 12] => [2, 3, 4]
        dx->Resize(dx_dims);
      }
      if (dy) {
        // flatten x.shape [2,3,4] => [6, 4]
        Tensor tmp_x(x->type());
        int64_t first_dim = x->dims()[0] * x->dims()[1];
        int64_t sec_dim = x->dims()[2];
        tmp_x.Resize(framework::make_ddim({first_dim, sec_dim}));
        tmp_x.mutable_data<T>(ctx.GetPlace());
        framework::TensorCopy(
            *x, ctx.GetPlace(),
            ctx.template device_context<platform::DeviceContext>(), &tmp_x);
        tmp_x.Resize(framework::make_ddim({first_dim, sec_dim}));
        // mamtul [6,4] [6,5] =>[4,5]
        dy->mutable_data<T>(ctx.GetPlace());
        auto runner_dy =
            NpuOpRunner("MatMul", {tmp_x, tmp_dout}, {*dy},
                        {{"transpose_x1", true}, {"transpose_x2", false}});
        runner_dy.Run(stream);
      }
    }
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OP_NPU_KERNEL(
    mul, ops::MulNPUKernel<paddle::platform::NPUDeviceContext, float>,
    ops::MulNPUKernel<paddle::platform::NPUDeviceContext,
                      paddle::platform::float16>);
REGISTER_OP_NPU_KERNEL(
    mul_grad, ops::MulGradNPUKernel<paddle::platform::NPUDeviceContext, float>,
    ops::MulGradNPUKernel<paddle::platform::NPUDeviceContext,
                          paddle::platform::float16>);
