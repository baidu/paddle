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

#ifdef PADDLE_WITH_ASCEND_CL
#include <memory>
#include <string>

#include "paddle/fluid/operators/npu_op_runner.h"
#include "paddle/fluid/operators/reduce_ops/reduce_op.h"

namespace paddle {
namespace operators {

template <typename DeviceContext, typename T>
class ReduceSumNPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* x = ctx.Input<framework::Tensor>("X");
    auto* out = ctx.Output<framework::Tensor>("Out");
    bool reduce_all = ctx.Attr<bool>("reduce_all");
    bool keep_dims = ctx.Attr<bool>("keep_dim");
    auto dims = ctx.Attr<std::vector<int>>("dim");

    out->mutable_data<T>(ctx.GetPlace());

    auto stream =
        ctx.template device_context<paddle::platform::NPUDeviceContext>()
            .stream();
    if (reduce_all) {
      std::vector<int> dim_vec;
      for (int i = 0; i < x->dims().size(); i++) {
        dim_vec.push_back(i);
      }
      auto runner = NpuOpRunner("ReduceSumD", {*x}, {*out},
                                {{"axes", dim_vec}, {"keep_dims", keep_dims}});
      runner.Run(stream);

    } else {
      auto runner = NpuOpRunner("ReduceSumD", {*x}, {*out},
                                {{"axes", dims}, {"keep_dims", keep_dims}});
      runner.Run(stream);
    }
  }
};

template <typename DeviceContext, typename T>
class ReduceSumGradNPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* x = ctx.Input<framework::Tensor>("X");
    auto* out_grad =
        ctx.Input<framework::Tensor>(framework::GradVarName("Out"));
    auto* x_grad = ctx.Output<framework::Tensor>(framework::GradVarName("X"));
    Tensor shape_tensor(framework::proto::VarType::INT32);
    shape_tensor.mutable_data<int32_t>(x->dims(), ctx.GetPlace());
    TensorFromVector(framework::vectorize<int32_t>(x->dims()),
                     ctx.device_context(), &shape_tensor);

    x_grad->mutable_data<T>(ctx.GetPlace());

    auto stream =
        ctx.template device_context<paddle::platform::NPUDeviceContext>()
            .stream();

    auto runner =
        NpuOpRunner("BroadcastTo", {*out_grad, shape_tensor}, {*x_grad}, {});
    runner.Run(stream);
  }
};
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OP_NPU_KERNEL(
    reduce_sum,
    ops::ReduceSumNPUKernel<paddle::platform::NPUDeviceContext, float>,
    ops::ReduceSumNPUKernel<paddle::platform::NPUDeviceContext, int>,
    ops::ReduceSumNPUKernel<paddle::platform::NPUDeviceContext,
                            paddle::platform::float16>);
REGISTER_OP_NPU_KERNEL(
    reduce_sum_grad,
    ops::ReduceSumGradNPUKernel<paddle::platform::NPUDeviceContext, float>,
    ops::ReduceSumGradNPUKernel<paddle::platform::NPUDeviceContext, int>,
    ops::ReduceSumGradNPUKernel<paddle::platform::NPUDeviceContext,
                                paddle::platform::float16>);
#endif
