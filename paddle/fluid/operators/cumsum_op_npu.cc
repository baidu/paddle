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

#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/operators/cum_op.h"
#include "paddle/fluid/operators/npu_op_runner.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;

template <typename DeviceContext, typename T>
class CumSumNPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* x = ctx.Input<Tensor>("X");
    auto* out = ctx.Output<Tensor>("Out");
    int axis = ctx.Attr<int>("axis");
    bool exclusive = ctx.Attr<bool>("exclusive");
    bool reverse = ctx.Attr<bool>("reverse");

    out->mutable_data<T>(ctx.GetPlace());

    framework::NPUAttributeMap attr_input = {
        {"axis", axis}, {"exclusive", exclusive}, {"reverse", reverse}};

    auto stream =
        ctx.template device_context<paddle::platform::NPUDeviceContext>()
            .stream();

    bool flatten = ctx.Attr<bool>("flatten");
    if (flatten) {
      PADDLE_ENFORCE_EQ(
          axis, -1,
          platform::errors::InvalidArgument(
              "when flatten is true, attr axis must be default %d, but got %d",
              -1, axis));

      Tensor new_x(x->type());
      new_x.ShareDataWith(*x);

      new_x.Resize(framework::make_ddim({x->numel()}));

      const auto& runner = NpuOpRunner("CumsumD", {new_x}, {*out}, attr_input);
      runner.Run(stream);
    } else {
      const auto& runner = NpuOpRunner("CumsumD", {*x}, {*out}, attr_input);
      runner.Run(stream);
    }
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;
REGISTER_OP_NPU_KERNEL(
    cumsum, ops::CumSumNPUKernel<plat::NPUDeviceContext, int>,
    ops::CumSumNPUKernel<plat::NPUDeviceContext, float>,
    ops::CumSumNPUKernel<plat::NPUDeviceContext, plat::float16>);
