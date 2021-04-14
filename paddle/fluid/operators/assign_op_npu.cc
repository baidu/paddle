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

#include <string>

#include "paddle/fluid/operators/assign_op.h"
#include "paddle/fluid/operators/npu_op_runner.h"
#include "paddle/fluid/platform/float16.h"

namespace paddle {
namespace framework {
class OpDesc;
class Variable;
}  // namespace framework
namespace imperative {
class OpBase;
}  // namespace imperative
namespace platform {
struct CPUPlace;
struct CUDAPlace;
struct float16;
}  // namespace platform
}  // namespace paddle

namespace paddle {
namespace operators {
template <typename DeviceContext, typename T>
class AssignNPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* x = ctx.Input<framework::LoDTensor>("X");
    auto* out = ctx.Output<framework::LoDTensor>("Out");
    out->mutable_data<T>(ctx.GetPlace());

    auto runner = NpuOpRunner("Assign", {*out, *x}, {*out}, {});
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

REGISTER_OP_NPU_KERNEL(
    assign, ops::AssignNPUKernel<paddle::platform::NPUDeviceContext, int>,
    ops::AssignNPUKernel<paddle::platform::NPUDeviceContext, float>,
    ops::AssignNPUKernel<paddle::platform::NPUDeviceContext, double>)
