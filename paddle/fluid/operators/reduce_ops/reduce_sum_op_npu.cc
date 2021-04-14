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

#include "paddle/fluid/operators/npu_op_runner.h"
#include "paddle/fluid/operators/reduce_ops/reduce_op.h"
#include "paddle/fluid/operators/unsqueeze_op.h"

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

    // special case
    if (x->dims().size() == 1 && keep_dims == false) {
      keep_dims = true;
    }

    auto stream =
        ctx.template device_context<paddle::platform::NPUDeviceContext>()
            .stream();

    framework::Tensor cast_x;
    framework::Tensor cast_out;
    // NOTE: ReduceSumD only supports fp32 and fp16
    if (x->type() != framework::proto::VarType::FP32 &&
        x->type() != framework::proto::VarType::FP16) {
      cast_x.Resize(x->dims());
      cast_x.mutable_data<float>(ctx.GetPlace());
      auto dst_dtype = ConvertToNpuDtype(framework::proto::VarType::FP32);
      auto runner_cast = NpuOpRunner(
          "Cast", {*x}, {cast_x}, {{"dst_type", static_cast<int>(dst_dtype)}});
      runner_cast.Run(stream);

      cast_out.Resize(out->dims());
      cast_out.mutable_data<float>(ctx.GetPlace());
    } else {
      cast_x.ShareDataWith(*x);
      cast_out.ShareDataWith(*out);
    }

    if (reduce_all) {
      std::vector<int> dim_vec;
      for (int i = 0; i < x->dims().size(); i++) {
        dim_vec.push_back(i);
      }

      auto runner = NpuOpRunner("ReduceSumD", {cast_x}, {cast_out},
                                {{"axes", dim_vec}, {"keep_dims", keep_dims}});
      runner.Run(stream);

    } else {
      auto runner = NpuOpRunner("ReduceSumD", {cast_x}, {cast_out},
                                {{"axes", dims}, {"keep_dims", keep_dims}});
      runner.Run(stream);
    }

    if (x->type() != framework::proto::VarType::FP32 &&
        x->type() != framework::proto::VarType::FP16) {
      auto dst_dtype = ConvertToNpuDtype(out->type());
      auto runner_cast =
          NpuOpRunner("Cast", {cast_out}, {*out},
                      {{"dst_type", static_cast<int>(dst_dtype)}});
      runner_cast.Run(stream);
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
    bool reduce_all = ctx.Attr<bool>("reduce_all");
    bool keep_dims = ctx.Attr<bool>("keep_dim");
    auto dims = ctx.Attr<std::vector<int>>("dim");

    x_grad->mutable_data<T>(ctx.GetPlace());

    auto stream =
        ctx.template device_context<paddle::platform::NPUDeviceContext>()
            .stream();
    if (keep_dims || reduce_all) {
      auto runner = NpuOpRunner("BroadcastToD", {*out_grad}, {*x_grad},
                                {{"shape", framework::vectorize(x->dims())}});
      runner.Run(stream);
    } else {
      framework::DDim out_dims;
      out_dims = UnsqueezeKernel<DeviceContext, T>::GetOutputShape(
          dims, out_grad->dims());

      Tensor out_grad_tmp(out_grad->type());
      out_grad_tmp.Resize(out_dims);
      out_grad_tmp.mutable_data<T>(ctx.GetPlace());
      framework::TensorCopy(
          *out_grad, ctx.GetPlace(),
          ctx.template device_context<platform::DeviceContext>(),
          &out_grad_tmp);
      out_grad_tmp.Resize(out_dims);

      auto runner = NpuOpRunner("BroadcastToD", {out_grad_tmp}, {*x_grad},
                                {{"shape", framework::vectorize(x->dims())}});
      runner.Run(stream);
    }
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
