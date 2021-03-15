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

#include "paddle/fluid/operators/cast_op.h"
#include "paddle/fluid/operators/npu_op_runner.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;

template <typename DeviceContext, typename T>
class CastNPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* x = ctx.Input<Tensor>("X");
    int dtype = ctx.Attr<int>("out_dtype");
    int aclDtype = 1;

    auto* out = ctx.Output<Tensor>("Out");

    auto place = ctx.GetPlace();

    if (dtype == 0) {
	    aclDtype = 12;
    } else if (dtype == 1) {
	    aclDtype = 6;
    } else if (dtype == 2) {
	    aclDtype = 3;
    } else if (dtype == 3) {
	    aclDtype = 9;
    } else if (dtype == 4) {
	    aclDtype = 1;
    } else if (dtype == 5) {
	    aclDtype = 0;
    } else if (dtype == 6) {
	    aclDtype = 11;
    }

    if (aclDtype == 0) {
	    out->mutable_data<float>(place);
    } else if (aclDtype == 1) {
	    out->mutable_data<paddle::platform::float16>(place);
    } else if (aclDtype == 6) {
	    out->mutable_data<int16_t>(place);
    } else if (aclDtype == 3) {
	    out->mutable_data<int32_t>(place);
    } else if (aclDtype == 9) {
	    out->mutable_data<int64_t>(place);
    } else if (aclDtype == 11) {
	    out->mutable_data<double>(place);
    } else if (aclDtype == 12) {
	    out->mutable_data<bool>(place);
    }

    auto stream =
        ctx.template device_context<paddle::platform::NPUDeviceContext>()
            .stream();

    auto runner = NpuOpRunner("Cast", {*x}, {*out}, {{"dst_type", static_cast<int32_t>(aclDtype)}});
    runner.Run(stream);
  }
};
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OP_NPU_KERNEL(
    cast,
    ops::CastNPUKernel<paddle::platform::NPUDeviceContext, int16_t>,
    ops::CastNPUKernel<paddle::platform::NPUDeviceContext, int32_t>,
    ops::CastNPUKernel<paddle::platform::NPUDeviceContext, int64_t>,
    ops::CastNPUKernel<paddle::platform::NPUDeviceContext, bool>,
    ops::CastNPUKernel<paddle::platform::NPUDeviceContext, double>,
    ops::CastNPUKernel<paddle::platform::NPUDeviceContext, float>,
    ops::CastNPUKernel<paddle::platform::NPUDeviceContext,
    paddle::platform::float16>);
#endif
