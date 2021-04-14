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

#include "paddle/fluid/operators/elementwise/elementwise_sub_op.h"
#include "paddle/fluid/operators/npu_op_runner.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;

template <typename T>
class ElementwiseSubNPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* x = ctx.Input<Tensor>("X");
    auto* y = ctx.Input<Tensor>("Y");
    auto* out = ctx.Output<Tensor>("Out");

    out->mutable_data<T>(ctx.GetPlace());

    auto runner = NpuOpRunner("Sub", {*x, *y}, {*out}, {});

    auto stream =
        ctx.template device_context<paddle::platform::NPUDeviceContext>()
            .stream();
    runner.Run(stream);
  }
};

template <typename T>
class ElementwiseSubGradNPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* dout = ctx.Input<Tensor>(framework::GradVarName("Out"));
    auto* dx = ctx.Output<Tensor>(framework::GradVarName("X"));
    auto* dy = ctx.Output<Tensor>(framework::GradVarName("Y"));

    auto stream =
        ctx.template device_context<paddle::platform::NPUDeviceContext>()
            .stream();

    // NOTE(zhiqiu): It seems Ascend Sub follow the broadcast sematics with
    // default axis=-1?
    // So, the sub_grad should do reduce if needed.
    // For example, the shape of each variable in elementwise_sub:
    // x, dx: [2, 3, 5]
    // y, dy: [1, 5]
    // out, dout: [2, 3, 5]
    // Then, out = x - y  =>  dx = dout, dy = -dout
    // And, the shape of dy can be computed by two stages reduce,
    // 1. [2, 3, 5] => [3, 5], ReduceSumD on axis = 0, keep_dims = false.
    // 2. [3, 5] => [1, 5], ReduceSumD on axis = 0, keep_dims = true.

    if (dx) {
      dx->mutable_data<T>(ctx.GetPlace());
      // For dx
      // stage 1
      auto reduce_ndim = dout->dims().size() - dx->dims().size();
      std::vector<int> axes;
      for (auto i = 0; i < reduce_ndim; ++i) {
        axes.push_back(i);
      }
      Tensor* tmp_dout = const_cast<Tensor*>(dout);
      Tensor reduced_dout(dx->type());
      if (axes.size() != 0) {
        std::vector<int64_t> reduced_dout_dims;
        for (auto i = reduce_ndim; i < dout->dims().size(); ++i) {
          reduced_dout_dims.push_back(dout->dims()[i]);
        }
        reduced_dout.Resize(framework::make_ddim(reduced_dout_dims));
        reduced_dout.mutable_data<T>(ctx.GetPlace());
        auto runner = NpuOpRunner("ReduceSumD", {*dout}, {reduced_dout},
                                  {{"axes", axes}, {"keep_dims", false}});
        runner.Run(stream);
        tmp_dout = &reduced_dout;
      }

      // stage 2
      axes.clear();
      for (auto i = 0; i < dx->dims().size(); ++i) {
        if (dx->dims()[i] == 1) {
          axes.push_back(i);
        }
      }
      if (axes.size() != 0) {
        auto runner = NpuOpRunner("ReduceSumD", {*tmp_dout}, {*dx},
                                  {{"axes", axes}, {"keep_dims", true}});
        runner.Run(stream);
      } else {
        framework::TensorCopySync(*tmp_dout, ctx.GetPlace(), dx);
      }
    }
    if (dy) {
      dy->mutable_data<T>(ctx.GetPlace());
      // For dy
      // stage 1
      auto reduce_ndim = dout->dims().size() - dy->dims().size();
      std::vector<int> axes;
      for (auto i = 0; i < reduce_ndim; ++i) {
        axes.push_back(i);
      }
      Tensor* tmp_dout = const_cast<Tensor*>(dout);
      Tensor reduced_dy(dy->type());
      Tensor reduced_dout(dy->type());

      if (axes.size() != 0) {
        std::vector<int64_t> reduced_dout_dims;
        for (auto i = reduce_ndim; i < dout->dims().size(); ++i) {
          reduced_dout_dims.push_back(dout->dims()[i]);
        }
        reduced_dout.Resize(framework::make_ddim(reduced_dout_dims));
        reduced_dout.mutable_data<T>(ctx.GetPlace());
        auto runner = NpuOpRunner("ReduceSumD", {*dout}, {reduced_dout},
                                  {{"axes", axes}, {"keep_dims", false}});
        runner.Run(stream);
        tmp_dout = &reduced_dout;
      }

      // stage 2
      axes.clear();
      Tensor* tmp_dy = tmp_dout;
      for (auto i = 0; i < dy->dims().size(); ++i) {
        if (dy->dims()[i] == 1) {
          axes.push_back(i);
        }
      }
      if (axes.size() != 0) {
        reduced_dy.Resize(dy->dims());
        reduced_dy.mutable_data<T>(ctx.GetPlace());
        auto runner = NpuOpRunner("ReduceSumD", {*tmp_dout}, {reduced_dy},
                                  {{"axes", axes}, {"keep_dims", true}});
        runner.Run(stream);
        tmp_dy = &reduced_dy;
      }

      // stage 3, negative
      auto runner = NpuOpRunner("Neg", {*tmp_dy}, {*dy}, {});
      runner.Run(stream);
    }
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;

REGISTER_OP_NPU_KERNEL(elementwise_sub, ops::ElementwiseSubNPUKernel<float>,
                       ops::ElementwiseSubNPUKernel<plat::float16>);

REGISTER_OP_NPU_KERNEL(elementwise_sub_grad,
                       ops::ElementwiseSubGradNPUKernel<float>,
                       ops::ElementwiseSubGradNPUKernel<plat::float16>);
