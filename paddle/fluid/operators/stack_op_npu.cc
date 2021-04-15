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
#include <vector>

#include "paddle/fluid/operators/activation_op.h"
#include "paddle/fluid/operators/npu_op_runner.h"
#include "paddle/fluid/operators/stack_op.h"
#include "paddle/fluid/operators/unsqueeze_op.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;

template <typename DeviceContext, typename T>
class StackNPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto x = ctx.MultiInput<Tensor>("X");
    size_t N = x.size();

    PADDLE_ENFORCE_GT(
        N, 0, platform::errors::InvalidArgument("number of input Tensor <= 0"));

    std::vector<paddle::framework::Tensor> x_list;
    std::vector<std::string> names;
    for (size_t i = 0; i < N; ++i) {
      x_list.push_back(*x[i]);
      names.push_back("x" + std::to_string(i));
    }

    int axis = ctx.Attr<int>("axis");

    if (axis < 0) {
      axis = axis + x_list[0].dims().size() + 1;
    }
    auto* out = ctx.Output<Tensor>("Y");

    auto place = ctx.GetPlace();

    auto stream =
        ctx.template device_context<paddle::platform::NPUDeviceContext>()
            .stream();

    out->mutable_data<T>(place);

    auto runner = NpuOpRunner("Pack", {x_list}, {*out},
                              {{"axis", axis}, {"N", static_cast<int>(N)}});

    runner.AddInputNames(names);
    runner.Run(stream);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OP_NPU_KERNEL(
    stack, ops::StackNPUKernel<paddle::platform::NPUDeviceContext, float>,
    ops::StackNPUKernel<paddle::platform::NPUDeviceContext,
                        paddle::platform::float16>);
