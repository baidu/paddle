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

#include "paddle/fluid/operators/controlflow/logical_op.h"
#include "paddle/fluid/operators/npu_op_runner.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;

template <typename DeviceContext, typename T>
class LogicalNotNPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* x = ctx.Input<Tensor>("X");
    auto* out = ctx.Output<Tensor>("Out");

    out->mutable_data<T>(ctx.GetPlace());

    auto stream =
        ctx.template device_context<paddle::platform::NPUDeviceContext>()
            .stream();

    const auto& runner = NpuOpRunner("LogicalNot", {*x}, {*out}, {});
    runner.Run(stream);
  }
};

template <typename DeviceContext, typename T>
class LogicalOrNPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* x = ctx.Input<framework::Tensor>("X");
    auto* y = ctx.Input<framework::Tensor>("Y");
    auto* out = ctx.Output<framework::Tensor>("Out");

    out->mutable_data<T>(ctx.GetPlace());

    auto stream =
        ctx.template device_context<paddle::platform::NPUDeviceContext>()
            .stream();

    const auto& runner = NpuOpRunner("LogicalOr", {*x, *y}, {*out}, {});
    runner.Run(stream);
  }
};

template <typename DeviceContext, typename T>
class LogicalAndPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* x = ctx.Input<framework::Tensor>("X");
    auto* y = ctx.Input<framework::Tensor>("Y");
    auto* out = ctx.Output<framework::Tensor>("Out");

    out->mutable_data<T>(ctx.GetPlace());

    auto stream =
        ctx.template device_context<paddle::platform::NPUDeviceContext>()
            .stream();

    const auto& runner = NpuOpRunner("LogicalAnd", {*x, *y}, {*out}, {});
    runner.Run(stream);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;

REGISTER_OP_NPU_KERNEL(
    logical_not, ops::LogicalNotNPUKernel<plat::NPUDeviceContext, bool>,
    ops::LogicalNotNPUKernel<plat::NPUDeviceContext, int8_t>,
    ops::LogicalNotNPUKernel<plat::NPUDeviceContext, int16_t>,
    ops::LogicalNotNPUKernel<plat::NPUDeviceContext, int>,
    ops::LogicalNotNPUKernel<plat::NPUDeviceContext, int64_t>,
    ops::LogicalNotNPUKernel<plat::NPUDeviceContext, float>,
    ops::LogicalNotNPUKernel<plat::NPUDeviceContext, double>);

REGISTER_OP_NPU_KERNEL(logical_or,
                       ops::LogicalOrNPUKernel<plat::NPUDeviceContext, bool>,
                       ops::LogicalOrNPUKernel<plat::NPUDeviceContext, int8_t>,
                       ops::LogicalOrNPUKernel<plat::NPUDeviceContext, int16_t>,
                       ops::LogicalOrNPUKernel<plat::NPUDeviceContext, int>,
                       ops::LogicalOrNPUKernel<plat::NPUDeviceContext, int64_t>,
                       ops::LogicalOrNPUKernel<plat::NPUDeviceContext, float>,
                       ops::LogicalOrNPUKernel<plat::NPUDeviceContext, double>);

REGISTER_OP_NPU_KERNEL(logical_and,
                       ops::LogicalAndPUKernel<plat::NPUDeviceContext, bool>,
                       ops::LogicalAndPUKernel<plat::NPUDeviceContext, int8_t>,
                       ops::LogicalAndPUKernel<plat::NPUDeviceContext, int16_t>,
                       ops::LogicalAndPUKernel<plat::NPUDeviceContext, int>,
                       ops::LogicalAndPUKernel<plat::NPUDeviceContext, int64_t>,
                       ops::LogicalAndPUKernel<plat::NPUDeviceContext, float>,
                       ops::LogicalAndPUKernel<plat::NPUDeviceContext, double>);
