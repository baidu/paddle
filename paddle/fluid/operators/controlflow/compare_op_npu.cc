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

#include <algorithm>
#include <string>
#include <vector>

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/op_version_registry.h"
#include "paddle/fluid/operators/controlflow/compare_op.h"
#include "paddle/fluid/operators/elementwise/elementwise_op_function.h"
#include "paddle/fluid/operators/npu_op_runner.h"
#ifdef PADDLE_WITH_ASCEND_CL

namespace paddle {
namespace operators {

template <typename T>
class EqualNPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* x = ctx.Input<framework::LoDTensor>("X");
    auto* y = ctx.Input<framework::LoDTensor>("Y");
    auto* out = ctx.Output<framework::LoDTensor>("Out");
    out->mutable_data<bool>(ctx.GetPlace());

    auto runner = NpuOpRunner("Equal", {*x, *y}, {*out}, {});
    auto stream =
        ctx.template device_context<paddle::platform::NPUDeviceContext>()
            .stream();
    runner.Run(stream);
  }
};

template <typename DeviceContext, typename T>
class LessThanNPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* x = ctx.Input<framework::LoDTensor>("X");
    auto* y = ctx.Input<framework::LoDTensor>("Y");
    auto* z = ctx.Output<framework::LoDTensor>("Out");
    // int axis = context.Attr<int>("axis");
    z->mutable_data<bool>(ctx.GetPlace());  // allocate
    auto runner = NpuOpRunner("Less", {*x, *y}, {*z});
    auto stream =
        ctx.template device_context<paddle::platform::NPUDeviceContext>()
            .stream();
    runner.Run(stream);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;

REGISTER_OP_NPU_KERNEL(equal, ops::EqualNPUKernel<float>,
                       ops::EqualNPUKernel<plat::float16>,
                       ops::EqualNPUKernel<int>);

REGISTER_OP_NPU_KERNEL(
    less_than,
    ops::LessThanNPUKernel<paddle::platform::NPUDeviceContext, float>,
    ops::LessThanNPUKernel<paddle::platform::NPUDeviceContext,
                           paddle::platform::float16>);

#endif
