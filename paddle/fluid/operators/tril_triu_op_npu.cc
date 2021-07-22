/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the Licnse. */

#include "paddle/fluid/operators/tril_triu_op.h"
#include "paddle/fluid/operators/npu_op_runner.h"

namespace paddle {
namespace operators {

template <typename DeviceContext, typename T>
class TrilTriuNPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* x = ctx.Input<Tensor>("X");
    auto* out = ctx.Output<Tensor>("Out");
    int diagonal = ctx.Attr<int>("diagonal");
    bool lower = ctx.Attr<bool>("lower");

    out->mutable_data<T>(ctx.GetPlace());

    std::string op_type = "Tril";
    if (!lower) {
      op_type = "Triu";
    }

    framework::NPUAttributeMap attr_input = {{"diagonal", diagonal}};

    auto stream =
        ctx.template device_context<paddle::platform::NPUDeviceContext>()
            .stream();

    const auto& runner = NpuOpRunner(op_type, {*x}, {*out}, attr_input);
    runner.Run(stream);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;
REGISTER_OP_NPU_KERNEL(
    tril_triu, ops::TrilTriuNPUKernel<plat::NPUDeviceContext, float>,
    ops::TrilTriuNPUKernel<plat::NPUDeviceContext, plat::float16>);
