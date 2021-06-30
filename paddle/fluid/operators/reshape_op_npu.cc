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

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/npu_op_runner.h"

namespace paddle {
namespace operators {

template <typename DeviceContext, typename T>
class Reshape2NPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* x = ctx.Input<framework::Tensor>("X");
    auto* out = ctx.Output<framework::Tensor>("Out");
    auto list_new_shape_tensor =
        ctx.MultiInput<framework::Tensor>("ShapeTensor");
    if (list_new_shape_tensor.size() > 0) {
      PADDLE_THROW(platform::errors::Unimplemented(
          "Input(ShapeTensor) is not supported on NPU."));
    }
    PADDLE_ENFORCE_EQ(ctx.Input<framework::LoDTensor>("Shape"), nullptr,
                      platform::errors::Unimplemented(
                          "Input(Shape) is not supported on NPU."));
    auto shape = out->dims();
    out->mutable_data(ctx.GetPlace(), x->type());

    auto stream =
        ctx.template device_context<paddle::platform::NPUDeviceContext>()
            .stream();
    Tensor trans_x(x->type());
    trans_x.Resize(x->dims());
    InferNPUStorageFormatAndDims(&trans_x, DataLayout::kNCHW);
    trans_x.mutable_data<T>(ctx.GetPlace());
    if (x->npu_storage_layout() == DataLayout::kFractalNZ) {
      RunTransDataNPUOP(*x, &trans_x, stream);
      VLOG(3) << "Transform data_format of reshape op.";
    } else {
      trans_x.ShareDataWith(*x);
    }

    framework::TensorCopy(
        trans_x, ctx.GetPlace(),
        ctx.template device_context<platform::DeviceContext>(), out);
    out->Resize(shape);
  }
};

template <typename DeviceContext, typename T>
class Reshape2GradNPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* d_x = ctx.Output<framework::Tensor>(framework::GradVarName("X"));
    auto* d_out = ctx.Input<framework::Tensor>(framework::GradVarName("Out"));
    auto in_dims = d_x->dims();

    auto stream =
        ctx.template device_context<paddle::platform::NPUDeviceContext>()
            .stream();
    Tensor trans_d_out(d_out->type());
    trans_d_out.Resize(d_out->dims());
    InferNPUStorageFormatAndDims(&trans_d_out, DataLayout::kNCHW);
    trans_d_out.mutable_data<T>(ctx.GetPlace());
    if (d_out->npu_storage_layout() == DataLayout::kFractalNZ) {
      RunTransDataNPUOP(*d_out, &trans_d_out, stream);
      VLOG(3) << "Transform data_format of reshape grad op.";
    } else {
      trans_d_out.ShareDataWith(*d_out);
    }

    d_x->mutable_data(ctx.GetPlace(), d_out->type());
    framework::TensorCopy(
        trans_d_out, ctx.GetPlace(),
        ctx.template device_context<platform::DeviceContext>(), d_x);
    d_x->Resize(in_dims);
  }
};
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OP_NPU_KERNEL(
    reshape2, ops::Reshape2NPUKernel<paddle::platform::NPUDeviceContext, float>,
    ops::Reshape2NPUKernel<paddle::platform::NPUDeviceContext, int>,
    ops::Reshape2NPUKernel<paddle::platform::NPUDeviceContext, int64_t>,
    ops::Reshape2NPUKernel<paddle::platform::NPUDeviceContext, bool>,
    ops::Reshape2NPUKernel<paddle::platform::NPUDeviceContext, double>,
    ops::Reshape2NPUKernel<paddle::platform::NPUDeviceContext, uint8_t>,
    ops::Reshape2NPUKernel<paddle::platform::NPUDeviceContext,
                           paddle::platform::float16>);
REGISTER_OP_NPU_KERNEL(
    reshape2_grad,
    ops::Reshape2GradNPUKernel<paddle::platform::NPUDeviceContext, float>,
    ops::Reshape2GradNPUKernel<paddle::platform::NPUDeviceContext, int>,
    ops::Reshape2GradNPUKernel<paddle::platform::NPUDeviceContext, int64_t>,
    ops::Reshape2GradNPUKernel<paddle::platform::NPUDeviceContext, bool>,
    ops::Reshape2GradNPUKernel<paddle::platform::NPUDeviceContext, double>,
    ops::Reshape2GradNPUKernel<paddle::platform::NPUDeviceContext, uint8_t>,
    ops::Reshape2GradNPUKernel<paddle::platform::NPUDeviceContext,
                               paddle::platform::float16>);
