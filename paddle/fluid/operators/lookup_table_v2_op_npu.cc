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

#include <iostream>
#include <memory>
#include <string>
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/tensor_util.h"
#include "paddle/fluid/operators/npu_op_runner.h"

namespace paddle {
namespace operators {

template <typename DeviceContext, typename T>
class LookupTableV2NPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    auto *ids_t = ctx.Input<framework::LoDTensor>("Ids");      // int tensor
    auto *output_t = ctx.Output<framework::LoDTensor>("Out");  // float tensor
    auto *table_t = ctx.Input<framework::LoDTensor>("W");

    // It seems cann 20.1 accepts int64, but cann 20.2+ not.
    PADDLE_ENFORCE_EQ(ids_t->type(), framework::proto::VarType::INT32,
                     platform::errors::Unimplemented(
                          "The index of LookupTableV2 should be int32."));

    auto *table_var = ctx.InputVar("W");
    PADDLE_ENFORCE_EQ(
        table_var->IsType<framework::LoDTensor>(), true,
        platform::errors::InvalidArgument("npu only accept LoDTensor"));
    output_t->mutable_data<T>(ctx.GetPlace());

    NpuOpRunner runner;
    runner.SetType("GatherV2")
        .AddInput(*table_t)
        .AddInput(*ids_t)
        .AddInput(std::vector<int32_t>{0})
        .AddOutput(*output_t);
    runner.Run();
  }
};

template <typename T>
class LookupTableV2GradNPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    auto *ids_t = ctx.Input<framework::LoDTensor>("Ids");
    auto *output_grad_t =
        ctx.Input<framework::LoDTensor>(framework::GradVarName("Out"));
    auto *table_grad_t =
        ctx.Output<framework::LoDTensor>(framework::GradVarName("W"));
    table_grad_t->mutable_data<T>(ctx.GetPlace());

    auto stream =
        ctx.template device_context<paddle::platform::NPUDeviceContext>()
            .stream();

    const auto &runner_zeros =
        NpuOpRunner("ZerosLike", {*table_grad_t}, {*table_grad_t});
    runner_zeros.Run(stream);

    // NOTE(zhiqiu): It seems in cann 20.1, the first input and output
    // can be different tensor, but in cann 20.2+, it does inplace operation.
    // Thus, the first input and output should be same tensor.
    const auto &runner_scatter =
        NpuOpRunner("ScatterAdd", {*table_grad_t, *ids_t, *output_grad_t},
                    {*table_grad_t}, {{"use_locking", true}});
    runner_scatter.Run(stream);
  }
};
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OP_NPU_KERNEL(
    lookup_table_v2,
    ops::LookupTableV2NPUKernel<paddle::platform::NPUDeviceContext, float>,
    ops::LookupTableV2NPUKernel<paddle::platform::NPUDeviceContext, int>,
    ops::LookupTableV2NPUKernel<paddle::platform::NPUDeviceContext,
                                paddle::platform::float16>);

REGISTER_OP_NPU_KERNEL(
    lookup_table_v2_grad, ops::LookupTableV2GradNPUKernel<float>,
    ops::LookupTableV2GradNPUKernel<int>,
    ops::LookupTableV2GradNPUKernel<paddle::platform::float16>);
