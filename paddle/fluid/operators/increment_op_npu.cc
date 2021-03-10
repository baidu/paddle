//   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifdef PADDLE_WITH_ASCEND_CL

#include "paddle/fluid/operators/increment_op.h"
#include "paddle/fluid/platform/float16.h"
#include "paddle/fluid/operators/npu_op_runner.h"

namespace paddle {
namespace framework {
class OpDesc;
class Variable;
}  // namespace framework
namespace imperative {
class OpBase;
}  // namespace imperative
}  // namespace paddle

namespace paddle {
namespace operators {


template <typename DeviceContext, typename T>
class IncrementalNPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* x_tensor = context.Input<framework::Tensor>("X");
    auto* out_tensor = context.Output<framework::Tensor>("Out");
    float step = context.Attr<float>("step");
    out_tensor->mutable_data<T>(context.GetPlace());

    Tensor step_tensor(x_tensor->type());
    step_tensor.Resize({1});
    step_tensor.mutable_data<T>(context.GetPlace());
    std::vector<float> step_vec;
    step_vec.push_back(static_cast<float>(step));
    framework::TensorFromVector(
                     step_vec,
                     context.device_context(),
                     &step_tensor);

    auto runner = NpuOpRunner("Add",
                              {*x_tensor, step_tensor},
                              {*out_tensor},
                              {});
    auto stream =
      context.template device_context<paddle::platform::NPUDeviceContext>()
                .stream();
    runner.Run(stream);
  }
};

}  // namespace operators
}  // namespace paddle


namespace plat = paddle::platform;
namespace ops = paddle::operators;

REGISTER_OP_NPU_KERNEL(
    increment,
    ops::IncrementalNPUKernel<paddle::platform::NPUDeviceContext, float>,
    ops::IncrementalNPUKernel<paddle::platform::NPUDeviceContext, double>,
    ops::IncrementalNPUKernel<paddle::platform::NPUDeviceContext, int>,
    ops::IncrementalNPUKernel<paddle::platform::NPUDeviceContext, int64_t>);

#endif
