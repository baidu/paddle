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

#include "paddle/fluid/operators/mul_op.h"
#include "paddle/fluid/operators/npu_op_runner.h"

namespace paddle {
namespace operators {

template <typename DeviceContext, typename T>
class MulNPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* x = ctx.Input<framework::LoDTensor>("X");
    auto* y = ctx.Input<framework::LoDTensor>("Y");
    auto* out = ctx.Output<framework::LoDTensor>("Out");
    int x_num_col_dims = ctx.Attr<int>("x_num_col_dims");  
    int y_num_col_dims = ctx.Attr<int>("y_num_col_dims");  
    if (x_num_col_dims == 1 && y_num_col_dims == 1) {
      if (x->dims().size() == 2 && y->dims().size() == 2) {      
        framework::AttributeMap attr_input= {{"transpose_x1", false}, {"transpose_x2", false}};
        auto runner = NpuOpRunner("MatMul", {*x, *y}, {*out}, attr_input);
        out->mutable_data<T>(ctx.GetPlace());

        auto stream =
            ctx.template device_context<paddle::platform::NPUDeviceContext>()
                .stream();
        runner.Run(stream);
      }
    }
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OP_NPU_KERNEL(
    mul,
    ops::MulNPUKernel<paddle::platform::NPUDeviceContext, float>);
//    ops::MatMulV2NPUKernel<paddle::platform::NPUDeviceContext, paddle::platform::float16>);
#endif
