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

#include "paddle/fluid/operators/matmul_v2_op.h"
#include "paddle/fluid/operators/npu_op_runner.h"

namespace paddle {
namespace operators {

template <typename DeviceContext, typename T>
class MatMulV2NPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* x = ctx.Input<framework::LoDTensor>("X");
    auto* y = ctx.Input<framework::LoDTensor>("Y");
    auto* out = ctx.Output<framework::LoDTensor>("Out");
    bool transpose_x = ctx.Attr<bool>("transpose_X");  
    bool transpose_y = ctx.Attr<bool>("transpose_Y");  
    framework::AttributeMap attr_input= {{"transpose_x1", transpose_x}, {"transpose_x2", transpose_y}};

    out->mutable_data<T>(ctx.GetPlace());

    // TODO(zhiqiu): get the attr infomation of Ascend op and
    // convert paddle AttributeMap to Ascend attrs.
    // Ascend op add has no attribute ?
    // int axis = ctx.Attr<int>("axis");

    // NOTE(zhiqiu): the order of inputs and outputs is important
    auto runner = NpuOpRunner("MatMul", {*x, *y}, {*out}, attr_input);

    auto stream =
        ctx.template device_context<paddle::platform::NPUDeviceContext>()
            .stream();
    runner.Run(stream);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OP_NPU_KERNEL(
    matmul_v2,
    ops::MatMulV2NPUKernel<paddle::platform::NPUDeviceContext, float>);
#endif
