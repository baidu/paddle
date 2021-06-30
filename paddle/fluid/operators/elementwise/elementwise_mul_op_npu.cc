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

#include "paddle/fluid/operators/elementwise/elementwise_mul_op.h"
#include "paddle/fluid/operators/npu_op_runner.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;

template <typename DeviceContext, typename T>
class ElementwiseMulNPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* x = ctx.Input<Tensor>("X");
    auto* y = ctx.Input<Tensor>("Y");

    auto* out = ctx.Output<Tensor>("Out");

    auto place = ctx.GetPlace();

    Tensor tmp_x(x->type());
    Tensor tmp_y(y->type());
    tmp_x.ShareDataWith(*x);
    tmp_y.ShareDataWith(*y);
    VLOG(4) << "yoki: elementwise x format"
            << framework::DataLayoutToString(x->npu_storage_layout());
    VLOG(4) << "yoki: elementwise y format"
            << framework::DataLayoutToString(y->npu_storage_layout());
    if (tmp_x.npu_storage_layout() != DataLayout::kFractalNZ) {
      VLOG(4) << "yoki: elementwise set format x";
      tmp_x.set_npu_storage_layout(x->layout());
      tmp_x.ResizeNPUDims(x->dims());
    }

    if (tmp_y.npu_storage_layout() != DataLayout::kFractalNZ) {
      VLOG(4) << "yoki: elementwise set format y";
      tmp_y.set_npu_storage_layout(y->layout());
      tmp_y.ResizeNPUDims(y->dims());
    }

    if (tmp_x.npu_storage_layout() == DataLayout::kFractalNZ ||
        tmp_y.npu_storage_layout() == DataLayout::kFractalNZ) {
      InferNPUStorageFormatAndDims(out, DataLayout::kFractalNZ);
    } else {
      InferNPUStorageFormatAndDims(out, DataLayout::kNCHW);
    }

    out->mutable_data<T>(place);

    auto stream =
        ctx.template device_context<paddle::platform::NPUDeviceContext>()
            .stream();

    const auto& runner = NpuOpRunner("Mul", {*x, *y}, {*out}, {});
    runner.Run(stream);
  }
};

template <typename DeviceContext, typename T>
class ElementwiseMulGradNPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* x = ctx.Input<Tensor>("X");
    auto* y = ctx.Input<Tensor>("Y");
    auto* dout = ctx.Input<Tensor>(framework::GradVarName("Out"));

    auto* dx = ctx.Output<Tensor>(framework::GradVarName("X"));
    auto* dy = ctx.Output<Tensor>(framework::GradVarName("Y"));

    auto place = ctx.GetPlace();

    auto stream =
        ctx.template device_context<paddle::platform::NPUDeviceContext>()
            .stream();

    if (dx) {
      InferNPUStorageFormatAndDims(dx, dout->npu_storage_layout());
      dx->mutable_data<T>(place);
      const auto& runner_dx = NpuOpRunner("Mul", {*dout, *y}, {*dx}, {});
      runner_dx.Run(stream);
    }

    if (dy) {
      InferNPUStorageFormatAndDims(dy, dout->npu_storage_layout());
      dy->mutable_data<T>(place);
      const auto& runner_dy = NpuOpRunner("Mul", {*x, *dout}, {*dy}, {});
      runner_dy.Run(stream);
    }
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OP_NPU_KERNEL(
    elementwise_mul,
    ops::ElementwiseMulNPUKernel<paddle::platform::NPUDeviceContext, float>,
    ops::ElementwiseMulNPUKernel<paddle::platform::NPUDeviceContext,
                                 paddle::platform::float16>);

REGISTER_OP_NPU_KERNEL(
    elementwise_mul_grad,
    ops::ElementwiseMulGradNPUKernel<paddle::platform::NPUDeviceContext, float>,
    ops::ElementwiseMulGradNPUKernel<paddle::platform::NPUDeviceContext,
                                     paddle::platform::float16>);
#endif
