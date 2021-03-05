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

#include "paddle/fluid/operators/softmax_op.h"
#include "paddle/fluid/operators/npu_op_runner.h"

#ifdef PADDLE_WITH_ASCEND_CL
namespace paddle {
namespace operators {

template <typename DeviceContext, typename T>
class SoftmaxNPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* in = ctx.Input<framework::LoDTensor>("X");
    auto axis = ctx.Attr<int>("axis");
    std::vector<int> axes;
    axes.push_back(axis);
    framework::AttributeMap attr_input = {{"axes", axes}};

    auto* out = ctx.Output<framework::LoDTensor>("Out");
    out->mutable_data<T>(ctx.GetPlace());

    auto runner = NpuOpRunner("SoftmaxV2", {*in}, {*out}, attr_input);

    auto stream =
        ctx.template device_context<paddle::platform::NPUDeviceContext>()
            .stream();
    runner.Run(stream);
  }
};


template <typename DeviceContext, typename T>
class SoftmaxGradNPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* out = ctx.Input<framework::LoDTensor>("Out");
    auto* dOut = ctx.Input<framework::LoDTensor>(framework::GradVarName("Out"));

    auto* dX = ctx.Output<framework::LoDTensor>(framework::GradVarName("X"));
    framework::LoDTensor dX_;
    dX = (dX == nullptr) ? &dX_ : dX;

    dX->Resize(out->dims());
    dX->mutable_data<T>(ctx.GetPlace());

    auto runner = NpuOpRunner("SoftmaxGrad", {*out, *dOut}, {*dX}, {});

    auto stream =
        ctx.template device_context<paddle::platform::NPUDeviceContext>()
            .stream();
    runner.Run(stream);
  }
};


}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;

REGISTER_OP_NPU_KERNEL(
    softmax,
    ops::SoftmaxNPUKernel<plat::NPUDeviceContext, float>,
    ops::SoftmaxNPUKernel<plat::NPUDeviceContext, double>,
    ops::SoftmaxNPUKernel<plat::NPUDeviceContext, plat::float16>);

REGISTER_OP_NPU_KERNEL(
    softmax_grad,
    ops::SoftmaxGradNPUKernel<plat::NPUDeviceContext, float>,
    ops::SoftmaxGradNPUKernel<plat::NPUDeviceContext, double>,
    ops::SoftmaxGradNPUKernel<plat::NPUDeviceContext, paddle::platform::float16>);

#endif
