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

namespace paddle {
namespace operators {

template <typename DeviceContext, typename T>
class Reshape2NPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* x = ctx.Input<framework::Tensor>("X");
    auto* shape = ctx.Attr<std::vector<int>>> ("shape");
    auto* out = ctx.Output<framework::Tensor>("Out");
    auto org_shape = framework::vectorize(x->dims());
    // reshape
    int64_t shape_all = 1;
    int64_t org_shape_all = 1;
    int index = -1;
    for (int i = 0; i < shape.size(); i++) {
      if (shape[i] == 0) {
        shape[i] = org_shape[i];
      }
      if (shape[i] == -1) {
        index = i;
      } else {
        shape_all *= shape[i];
      }
      org_shape_all *= org_shape[i];
    }

    if (index >= 0) {
      shape[index] = org_shape_all / shape_all;
    }
    out.Resize(framework::make_ddim(shape));
    out->mutable_data(ctx.GetPlace(), x->type());
    framework::TensorCopy(
        *x, ctx.GetPlace(),
        ctx.template device_context<platform::DeviceContext>(), out);
  }
};

template <typename DeviceContext, typename T>
class Reshape2GradNPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* d_x = ctx.Output<framework::Tensor>(framework::GradVarName("X"));
    auto* d_out = ctx.Input<framework::Tensor>(framework::GradVarName("Out"));
    auto in_dims = d_x->dims();

    d_x->mutable_data(ctx.GetPlace(), d_out->type());
    framework::TensorCopy(
        *d_out, ctx.GetPlace(),
        ctx.template device_context<platform::DeviceContext>(), d_x);
    d_x->Resize(in_dims);
  }
};
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OP_NPU_KERNEL(
    reshpe2, ops::Reshape2NPUKernel<paddle::platform::NPUDeviceContext, float>,
    ops::Reshape2NPUKernel<paddle::platform::NPUDeviceContext,
                           paddle::platform::float16>);
REGISTER_OP_NPU_KERNEL(
    reshpe2_grad,
    ops::Reshape2GradNPUKernel<paddle::platform::NPUDeviceContext, float>,
    ops::Reshape2GradNPUKernel<paddle::platform::NPUDeviceContext,
                               paddle::platform::float16>);
#endif
