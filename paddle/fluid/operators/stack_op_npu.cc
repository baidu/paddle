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
#include <vector>

#include "paddle/fluid/operators/activation_op.h"
#include "paddle/fluid/operators/unsqueeze_op.h"
#include "paddle/fluid/operators/stack_op.h"
#include "paddle/fluid/operators/npu_op_runner.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;

template <typename DeviceContext, typename T>
class StackNPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
        //init X
        /*
    auto x = ctx.MultiInput<Tensor>("X");
    std::vector<paddle::framework::Tensor> x_2;
    for(uint32_t i = 0; i < x.size(); i++) {
      x_2[i] = *x[i];
    }
    */

        auto x = ctx.MultiInput<Tensor>("X");
    int n = static_cast<int>(x.size());
    //std::vector<const T*> x_list;
    std::vector<paddle::framework::Tensor> x_list;
    for (int i = 0; i < n; i++) {
      x_list.push_back(x[i]);
    }

    int axis = ctx.Attr<int>("axis");
        int32_t N = x.size();
    auto* out = ctx.Output<Tensor>("Out");

    auto place = ctx.GetPlace();

    auto stream =
        ctx.template device_context<paddle::platform::NPUDeviceContext>()
            .stream();

    out->mutable_data<T>(place);
        auto runner = NpuOpRunner("Pack", {x_list}, {*out}, {{"axis", axis}, {"N", N}});
    runner.Run(stream);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OP_NPU_KERNEL(
    stack,
    ops::StackNPUKernel<paddle::platform::NPUDeviceContext, float>,
    ops::StackNPUKernel<paddle::platform::NPUDeviceContext,
                       paddle::platform::float16>);

#endif
