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
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/op_version_registry.h"
#include "paddle/fluid/operators/npu_op_runner.h"

namespace paddle {
namespace operators {

template <typename DeviceContext, typename T>
class MatMulNPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* x = ctx.Input<framework::Tensor>("X");
    auto* y = ctx.Input<framework::Tensor>("Y");
    auto* out = ctx.Output<framework::Tensor>("Out");
    bool transpose_x = ctx.Attr<bool>("transpose_X");
    bool transpose_y = ctx.Attr<bool>("transpose_Y");

    if (x->dims().size() == 2) {
      out->mutable_data<T>(ctx.GetPlace());

      const auto& runner = NpuOpRunner(
          "MatMul", {*x, *y}, {*out},
          {{"transpose_x1", transpose_x}, {"transpose_x2", transpose_y}});

      auto stream =
          ctx.template device_context<paddle::platform::NPUDeviceContext>()
              .stream();
      runner.Run(stream);

    } else if (x->dims().size() > 2) {
      out->mutable_data<T>(ctx.GetPlace());

      const auto& runner =
          NpuOpRunner("BatchMatMul", {*x, *y}, {*out},
                      {{"adj_x1", transpose_x}, {"adj_x2", transpose_y}});

      auto stream =
          ctx.template device_context<paddle::platform::NPUDeviceContext>()
              .stream();
      runner.Run(stream);
    }
  }
};

template <typename DeviceContext, typename T>
class MatMulGradNPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* x = ctx.Input<framework::Tensor>("X");
    auto* y = ctx.Input<framework::Tensor>("Y");
    auto* dout = ctx.Input<framework::Tensor>(framework::GradVarName("Out"));
    auto* dx = ctx.Output<framework::Tensor>(framework::GradVarName("X"));
    auto* dy = ctx.Output<framework::Tensor>(framework::GradVarName("Y"));
    bool transpose_y = ctx.Attr<bool>("transpose_Y");
    auto stream =
        ctx.template device_context<paddle::platform::NPUDeviceContext>()
            .stream();

    if (x->dims().size() == 2) {
      if (transpose_y) {
        if (dx) {
          dx->mutable_data<T>(ctx.GetPlace());
          const auto& runner_dx =
              NpuOpRunner("MatMul", {*dout, *y}, {*dx},
                          {{"transpose_x1", false}, {"transpose_x2", false}});

          runner_dx.Run(stream);
        }
        if (dy) {
          dy->mutable_data<T>(ctx.GetPlace());
          const auto& runner_dy =
              NpuOpRunner("MatMul", {*dout, *x}, {*dy},
                          {{"transpose_x1", true}, {"transpose_x2", false}});

          runner_dy.Run(stream);
        }

      } else {
        if (dx) {
          dx->mutable_data<T>(ctx.GetPlace());
          const auto& runner_dx =
              NpuOpRunner("MatMul", {*dout, *y}, {*dx},
                          {{"transpose_x1", false}, {"transpose_x2", true}});

          runner_dx.Run(stream);
        }
        if (dy) {
          dy->mutable_data<T>(ctx.GetPlace());
          const auto& runner_dy =
              NpuOpRunner("MatMul", {*x, *dout}, {*dy},
                          {{"transpose_x1", true}, {"transpose_x2", false}});

          runner_dy.Run(stream);
        }
      }
    } else if (x->dims().size() > 2) {
      if (transpose_y) {
        if (dx) {
          dx->mutable_data<T>(ctx.GetPlace());
          const auto& runner_dx =
              NpuOpRunner("BatchMatMul", {*dout, *y}, {*dx},
                          {{"adj_x1", false}, {"adj_x2", false}});

          runner_dx.Run(stream);
        }
        if (dy) {
          dy->mutable_data<T>(ctx.GetPlace());
          const auto& runner_dy =
              NpuOpRunner("BatchMatMul", {*dout, *x}, {*dy},
                          {{"adj_x1", true}, {"adj_x2", false}});

          runner_dy.Run(stream);
        }
      } else {
        if (dx) {
          dx->mutable_data<T>(ctx.GetPlace());
          const auto& runner_dx =
              NpuOpRunner("BatchMatMul", {*dout, *y}, {*dx},
                          {{"adj_x1", false}, {"adj_x2", true}});

          runner_dx.Run(stream);
        }
        if (dy) {
          dy->mutable_data<T>(ctx.GetPlace());
          if ((x->dims().size() == 3) && (dout->dims().size() == 3) &&
              (dy->dims().size() == 2)) {
            framework::Tensor dout_tmp;
            dout_tmp.ShareDataWith(*dout);
            std::vector<int> vec_dim =
                framework::vectorize<int>(dout_tmp.dims());
            std::vector<int> vec_dim_v{vec_dim[0] * vec_dim[1], vec_dim[2]};
            dout_tmp.Resize(framework::make_ddim(vec_dim_v));

            framework::Tensor x_tmp;
            x_tmp.ShareDataWith(*x);
            std::vector<int> vec_dim_x =
                framework::vectorize<int>(x_tmp.dims());
            std::vector<int> vec_dim_x_v{vec_dim_x[0] * vec_dim_x[1],
                                         vec_dim_x[2]};
            x_tmp.Resize(framework::make_ddim(vec_dim_x_v));
            const auto& runner_dy =
                NpuOpRunner("MatMul", {x_tmp, dout_tmp}, {*dy},
                            {{"transpose_x1", true}, {"transpose_x2", false}});
            runner_dy.Run(stream);
          } else {
            const auto& runner_dy =
                NpuOpRunner("BatchMatMul", {*x, *dout}, {*dy},
                            {{"adj_x1", true}, {"adj_x2", false}});
            runner_dy.Run(stream);
          }
        }
      }
    }
  }
};
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OP_NPU_KERNEL(
    matmul, ops::MatMulNPUKernel<paddle::platform::NPUDeviceContext, float>,
    ops::MatMulNPUKernel<paddle::platform::NPUDeviceContext,
                         paddle::platform::float16>);
REGISTER_OP_NPU_KERNEL(
    matmul_grad,
    ops::MatMulGradNPUKernel<paddle::platform::NPUDeviceContext, float>,
    ops::MatMulGradNPUKernel<paddle::platform::NPUDeviceContext,
                             paddle::platform::float16>);
