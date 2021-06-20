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
        // out->mutable_data<T>(ctx.GetPlace());
        // out->ResizeNPUDims(out->dims());
        out->ResizeNPUDims(framework::make_ddim(
            InferShapeNDToNZ(framework::vectorize(out->dims()))));
        out->set_npu_storage_layout(DataLayout::kFractalNZ);
        size_t npu_storage_size =
            out->npu_storage_numel() * framework::SizeOfType(x->type());
        out->mutable_data(ctx.GetPlace(), x->type(), npu_storage_size);

        Tensor x_nz_format = CastNPUFormat(*x, 29);
        Tensor y_nz_format = CastNPUFormat(*y, 29);
        // Tensor out_nz_format = GenerateNZTensor(*out);
        const auto& runner =
            NpuOpRunner("MatMul", {x_nz_format, y_nz_format}, {*out},
                        {{"transpose_x1", false}, {"transpose_x2", false}});

        runner.Run(stream);
        // RunTransDataNPUOP(out_nz_format, out, stream);
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
        // out->mutable_data<T>(ctx.GetPlace());
        // matmul
        // out->ResizeNPUDims(out->dims());
        out->ResizeNPUDims(framework::make_ddim(
            InferShapeNDToNZ(framework::vectorize(out->dims()))));
        out->set_npu_storage_layout(DataLayout::kFractalNZ);
        size_t npu_storage_size =
            out->npu_storage_numel() * framework::SizeOfType(x->type());
        out->mutable_data(ctx.GetPlace(), x->type(), npu_storage_size);

        Tensor x_nz_format = CastNPUFormat(tmp_x, 29);
        Tensor y_nz_format = CastNPUFormat(*y, 29);
        // Tensor out_nz_format = GenerateNZTensor(*out);
        const auto& runner =
            NpuOpRunner("MatMul", {x_nz_format, y_nz_format}, {*out},
                        {{"transpose_x1", false}, {"transpose_x2", false}});
        runner.Run(stream);
        // RunTransDataNPUOP(out_nz_format, out, stream);
      } else {
        PADDLE_THROW(
            platform::errors::InvalidArgument("npu error: not suppert dims"));
      }
      // to do other
    } else if (x->dims().size() == 3 && y->dims().size() == 2) {
      // for example: x.shape=[2, 3, 4] y.shape=[4, 5], expect [2, 3, 5]
      PADDLE_ENFORCE_EQ(x_num_col_dims, 2,
                        platform::errors::InvalidArgument(
                            "now only support x_num_col_dims == 2: but got %d",
                            x_num_col_dims));

      /*
      Tensor tmp_y(y->type());
      // tmp_y.ShareDataWith(*y);
      tmp_y.Resize(framework::make_ddim({x->dims()[0], y->dims()[0],
      y->dims()[1]}));
      // tmp_y.mutable_data<T>(ctx.GetPlace());
      std::vector<int64_t> broadcast_shape = {x->dims()[0], y->dims()[0],
      y->dims()[1]};

      if (y->npu_storage_layout() == DataLayout::kFractalNZ) {
        tmp_y.set_npu_storage_layout(y->npu_storage_layout());
        tmp_y.ResizeNPUDims(framework::make_ddim(
          InferShapeNDToNZ(framework::vectorize(tmp_y.dims()))));
        size_t npu_storage_size =
          tmp_y.npu_storage_numel() * framework::SizeOfType(x->type());
        tmp_y.mutable_data(ctx.GetPlace(), x->type(), npu_storage_size);
      } else {
        tmp_y.ResizeNPUDims(tmp_y.dims());
        tmp_y.mutable_data<T>(ctx.GetPlace());
      }

      const auto& broadcast_runner =
          NpuOpRunner("BroadcastToD", {*y}, {tmp_y},
                      {{"shape", broadcast_shape}});
      broadcast_runner.Run(stream);
      */

      Tensor x_nz_format = CastNPUFormat(*x, 29);
      Tensor y_nz_format = CastNPUFormat(*y, 29);

      // out->mutable_data<T>(ctx.GetPlace());
      // out->ResizeNPUDims(out->dims());
      // Tensor out_nz_format = GenerateNZTensor(*out);

      out->ResizeNPUDims(framework::make_ddim(
          InferShapeNDToNZ(framework::vectorize(out->dims()))));
      out->set_npu_storage_layout(DataLayout::kFractalNZ);
      size_t npu_storage_size =
          out->npu_storage_numel() * framework::SizeOfType(x->type());
      out->mutable_data(ctx.GetPlace(), x->type(), npu_storage_size);

      const auto& runner =
          NpuOpRunner("BatchMatMul", {x_nz_format, y_nz_format}, {*out},
                      {{"adj_x1", false}, {"adj_x2", false}});

      auto stream =
          ctx.template device_context<paddle::platform::NPUDeviceContext>()
              .stream();
      runner.Run(stream);

      // RunTransDataNPUOP(out_nz_format, out, stream);

      /*
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
      tmp_matmul.ResizeNPUDims(tmp_matmul.dims());

      Tensor x_nz_format = CastNPUFormat(tmp_x, 29);
      Tensor y_nz_format = CastNPUFormat(*y, 29);
      Tensor out_nz_format = GenerateNZTensor(tmp_matmul);

      // matmul [6,4] , [4, 5] => [6, 5]
      // Tensor out_nz_format(x->type());
      // out_nz_format.Resize(framework::make_ddim({first_dim, y->dims()[1]}));
      // out_nz_format.ResizeNPUDims(framework::make_ddim(
      //     InferShapeNDToNZ(framework::make_ddim({first_dim,
      //     y->dims()[1]}))));
      // out_nz_format.set_npu_storage_layout(DataLayout::kFractalNZ);
      // size_t npu_storage_size =
      //     out_nz_format.npu_storage_numel() *
      //     framework::SizeOfType(x->type());
      // out_nz_format.mutable_data(ctx.GetPlace(), x->type(),
      // npu_storage_size);

      const auto& runner_matmul =
          NpuOpRunner("MatMul", {x_nz_format, y_nz_format}, {out_nz_format},
                      {{"transpose_x1", false}, {"transpose_x2", false}});

      runner_matmul.Run(stream);
      RunTransDataNPUOP(out_nz_format, &tmp_matmul, stream);

      // reshape [6, 5] => [2, 3, 5]
      (*out).Resize(
          framework::make_ddim({x->dims()[0], x->dims()[1], y->dims()[1]}));
      out->mutable_data(ctx.GetPlace(), x->type());
      framework::TensorCopy(
          tmp_matmul, ctx.GetPlace(),
          ctx.template device_context<platform::DeviceContext>(), out);
      (*out).Resize(
          framework::make_ddim({x->dims()[0], x->dims()[1], y->dims()[1]}));
      */
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
      Tensor dout_nz_format = CastNPUFormat(*dout, 29);
      if (x->dims().size() == 2 && y->dims().size() == 2) {
        if (dx) {
          // dx->mutable_data<T>(ctx.GetPlace());
          // dx->ResizeNPUDims(dx->dims());

          dx->ResizeNPUDims(framework::make_ddim(
              InferShapeNDToNZ(framework::vectorize(dx->dims()))));
          dx->set_npu_storage_layout(DataLayout::kFractalNZ);
          size_t npu_storage_size =
              dx->npu_storage_numel() * framework::SizeOfType(x->type());
          dx->mutable_data(ctx.GetPlace(), y->type(), npu_storage_size);

          Tensor y_nz_format = CastNPUFormat(*y, 29);
          // Tensor dx_nz_format = GenerateNZTensor(*dx);

          const auto& runner_dx =
              NpuOpRunner("MatMul", {dout_nz_format, y_nz_format}, {*dx},
                          {{"transpose_x1", false}, {"transpose_x2", true}});

          runner_dx.Run(stream);
          // RunTransDataNPUOP(dx_nz_format, dx, stream);
        }

        if (dy) {
          // dy->mutable_data<T>(ctx.GetPlace());
          // dy->ResizeNPUDims(dy->dims());

          dy->ResizeNPUDims(framework::make_ddim(
              InferShapeNDToNZ(framework::vectorize(dy->dims()))));
          dy->set_npu_storage_layout(DataLayout::kFractalNZ);
          size_t npu_storage_size =
              dy->npu_storage_numel() * framework::SizeOfType(x->type());
          dy->mutable_data(ctx.GetPlace(), x->type(), npu_storage_size);

          Tensor x_nz_format = CastNPUFormat(*x, 29);
          // Tensor dy_nz_format = GenerateNZTensor(*dy);
          const auto& runner_dy =
              NpuOpRunner("MatMul", {x_nz_format, dout_nz_format}, {*dy},
                          {{"transpose_x1", true}, {"transpose_x2", false}});

          runner_dy.Run(stream);
          // RunTransDataNPUOP(dy_nz_format, dy, stream);
        }
      } else if (x->dims().size() == 3 && y->dims().size() == 2) {
        // flatten => x.shape=[6, 4]
        // matmul
        if (dx) {
          // matmul [2, 5] * [12, 5] => [2, 12]
          dx->mutable_data<T>(ctx.GetPlace());

          auto dx_dims = dx->dims();
          dx->Resize(framework::make_ddim({dout->dims()[0], y->dims()[0]}));
          dx->ResizeNPUDims(
              framework::make_ddim({dout->dims()[0], y->dims()[0]}));

          Tensor y_nz_format = CastNPUFormat(*y, 29);
          Tensor dx_nz_format = GenerateNZTensor(*dx);

          const auto& runner_matmul = NpuOpRunner(
              "MatMul", {dout_nz_format, y_nz_format}, {dx_nz_format},
              {{"transpose_x1", false}, {"transpose_x2", true}});
          runner_matmul.Run(stream);
          RunTransDataNPUOP(dx_nz_format, dx, stream);
          // reshape [2, 12] => [2, 3, 4]
          dx->Resize(dx_dims);
          dx->ResizeNPUDims(dx_dims);
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
          // dy->mutable_data<T>(ctx.GetPlace());
          // dy->ResizeNPUDims(dy->dims());

          dy->ResizeNPUDims(framework::make_ddim(
              InferShapeNDToNZ(framework::vectorize(dy->dims()))));
          dy->set_npu_storage_layout(DataLayout::kFractalNZ);
          size_t npu_storage_size =
              dy->npu_storage_numel() * framework::SizeOfType(x->type());
          dy->mutable_data(ctx.GetPlace(), x->type(), npu_storage_size);

          Tensor x_nz_format = CastNPUFormat(tmp_x, 29);
          // Tensor dy_nz_format = GenerateNZTensor(*dy);

          const auto& runner_dy =
              NpuOpRunner("MatMul", {x_nz_format, dout_nz_format}, {*dy},
                          {{"transpose_x1", true}, {"transpose_x2", false}});

          runner_dy.Run(stream);
          // RunTransDataNPUOP(dy_nz_format, dy, stream);
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
      Tensor dout_nz_format = CastNPUFormat(tmp_dout, 29);

      if (dx) {
        /*// tmp_dout * y [6,5] * [4,5] => [6, 4]
        dx->mutable_data<T>(ctx.GetPlace());
        auto dx_dims = dx->dims();
        dx->Resize(framework::make_ddim({dout_first_dim, y->dims()[0]}));
        dx->ResizeNPUDims(framework::make_ddim({dout_first_dim, y->dims()[0]}));

        Tensor y_nz_format = CastNPUFormat(*y, 29);
        Tensor dx_nz_format = GenerateNZTensor(*dx);

        const auto& runner_matmul =
            NpuOpRunner("MatMul", {dout_nz_format, y_nz_format}, {dx_nz_format},
                        {{"transpose_x1", false}, {"transpose_x2", true}});
        runner_matmul.Run(stream);
        RunTransDataNPUOP(dx_nz_format, dx, stream);
        // reshape [2, 12] => [2, 3, 4]
        dx->Resize(dx_dims);
        dx->ResizeNPUDims(dx_dims);*/

        Tensor y_nz_format = CastNPUFormat(*y, 29);

        dx->ResizeNPUDims(framework::make_ddim(
            InferShapeNDToNZ(framework::vectorize(dx->dims()))));
        dx->set_npu_storage_layout(DataLayout::kFractalNZ);
        size_t npu_storage_size =
            dx->npu_storage_numel() * framework::SizeOfType(x->type());
        dx->mutable_data(ctx.GetPlace(), x->type(), npu_storage_size);

        const auto& runner =
            NpuOpRunner("BatchMatMul", {dout_nz_format, y_nz_format}, {*dx},
                        {{"adj_x1", false}, {"adj_x2", true}});

        auto stream =
            ctx.template device_context<paddle::platform::NPUDeviceContext>()
                .stream();
        runner.Run(stream);
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
        // dy->mutable_data<T>(ctx.GetPlace());
        // dy->ResizeNPUDims(dy->dims());

        dy->ResizeNPUDims(framework::make_ddim(
            InferShapeNDToNZ(framework::vectorize(dy->dims()))));
        dy->set_npu_storage_layout(DataLayout::kFractalNZ);
        size_t npu_storage_size =
            dy->npu_storage_numel() * framework::SizeOfType(x->type());
        dy->mutable_data(ctx.GetPlace(), x->type(), npu_storage_size);

        Tensor x_nz_format = CastNPUFormat(tmp_x, 29);
        // Tensor dy_nz_format = GenerateNZTensor(*dy);
        const auto& runner_dy =
            NpuOpRunner("MatMul", {x_nz_format, dout_nz_format}, {*dy},
                        {{"transpose_x1", true}, {"transpose_x2", false}});
        runner_dy.Run(stream);
        // RunTransDataNPUOP(dy_nz_format, dy, stream);
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
