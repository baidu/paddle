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
#include "paddle/fluid/operators/scale_op.h"

namespace paddle {
namespace operators {

template <typename DeviceContext, typename T>
class ScaleNPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* x = ctx.Input<framework::LoDTensor>("X");
    auto* out = ctx.Output<framework::LoDTensor>("Out");
    auto scale = static_cast<T>(ctx.Attr<float>("scale"));
    auto bias = static_cast<T>(ctx.Attr<float>("bias"));
    auto bias_after_scale = ctx.Attr<bool>("bias_after_scale");
    float _power = 1.0;
    if (bias_after_scale) {
      out->mutable_data<T>(ctx.GetPlace());
      auto runner =
          NpuOpRunner("Power", {*x}, {*out},
                      {{"power", _power}, {"scale", scale}, {"shift", bias}});

      auto stream =
          ctx.template device_context<paddle::platform::NPUDeviceContext>()
              .stream();
      runner.Run(stream);
    }
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OP_NPU_KERNEL(
    scale, ops::ScaleNPUKernel<paddle::platform::NPUDeviceContext, float>);
#endif
