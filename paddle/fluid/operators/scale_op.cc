/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/operators/scale_op.h"

#include <string>

namespace paddle {
namespace framework {
class InferShapeContext;
class OpDesc;
}  // namespace framework
namespace imperative {
class OpBase;
}  // namespace imperative
namespace platform {
class CPUDeviceContext;
}  // namespace platform
}  // namespace paddle

namespace paddle {
namespace operators {

class ScaleOp : public framework::OperatorWithKernel {
 public:
  ScaleOp(const std::string &type, const framework::VariableNameMap &inputs,
          const framework::VariableNameMap &outputs,
          const framework::AttributeMap &attrs)
      : OperatorWithKernel(type, inputs, outputs, attrs) {}

  void InferShape(framework::InferShapeContext *ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("X"), "Input", "X", "scale");
    OP_INOUT_CHECK(ctx->HasOutput("Out"), "Output", "Out", "scale");

    if (ctx->IsRuntime() && ctx->HasInput("ScaleTensor")) {
      auto scale = ctx->Inputs("ScaleTensor");
      PADDLE_ENFORCE_EQ(scale.size(), 1,
                        platform::errors::InvalidArgument(
                            "Input(ScaleTensor) size must be 1, "
                            "but received size is %d.",
                            scale.size()));
    }

    ctx->SetOutputDim("Out", ctx->GetInputDim("X"));
    ctx->ShareLoD("X", /*->*/ "Out");
  }
};

class ScaleOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "(Tensor) Input tensor of scale operator.");
    AddInput("ScaleTensor",
             "(Tensor) If provided, use this as "
             "scale factor, this has a higher priority than "
             "attr(scale), the shape of this tensor MUST BE 1.")
        .AsDispensable();
    AddOutput("Out", "(Tensor) Output tensor of scale operator.");
    AddComment(R"DOC(
**Scale operator**

Apply scaling and bias addition to the input tensor.

if bias_after_scale=True:

$$Out = scale*X + bias$$

else:

$$Out = scale*(X + bias)$$
)DOC");
    AddAttr<float>("scale", "The scaling factor of the scale operator.")
        .SetDefault(1.0);
    AddAttr<float>("bias", "The bias of the scale operator.").SetDefault(0.0);
    AddAttr<bool>(
        "bias_after_scale",
        "Apply bias addition after or before scaling. It is useful for "
        "numeric stability in some circumstances.")
        .SetDefault(true);
  }
};

class ScaleOpVarTypeInference : public framework::VarTypeInference {
 public:
  void operator()(framework::InferVarTypeContext *ctx) const override {
    ctx->SyncTypeAndDataType("X", "Out");
  }
};

template <typename T>
class ScaleGradMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

  void Apply(GradOpPtr<T> grad_op) const override {
    grad_op->SetType("scale");
    grad_op->SetInput("X", this->OutputGrad("Out"));
    if (this->HasInput("ScaleTensor") > 0) {
      grad_op->SetInput("ScaleTensor", this->Input("ScaleTensor"));
    }
    grad_op->SetOutput("Out", this->InputGrad("X"));
    grad_op->SetAttr("scale", this->GetAttr("scale"));
    grad_op->SetAttr("bias", 0.0f);
    grad_op->SetAttr("bias_after_scale", true);
  }
};

DECLARE_INPLACE_OP_INFERER(ScaleOpInplaceInferer, {"X", "Out"});
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OPERATOR(scale, ops::ScaleOp, ops::ScaleOpMaker,
                  ops::ScaleGradMaker<paddle::framework::OpDesc>,
                  ops::ScaleGradMaker<paddle::imperative::OpBase>,
                  ops::ScaleOpVarTypeInference, ops::ScaleOpInplaceInferer);
REGISTER_OP_CPU_KERNEL(
    scale, ops::ScaleKernel<paddle::platform::CPUDeviceContext, float>,
    ops::ScaleKernel<paddle::platform::CPUDeviceContext, double>,
    ops::ScaleKernel<paddle::platform::CPUDeviceContext, uint8_t>,
    ops::ScaleKernel<paddle::platform::CPUDeviceContext, int8_t>,
    ops::ScaleKernel<paddle::platform::CPUDeviceContext, int16_t>,
    ops::ScaleKernel<paddle::platform::CPUDeviceContext, int>,
    ops::ScaleKernel<paddle::platform::CPUDeviceContext, int64_t>,
    ops::ScaleKernel<paddle::platform::CPUDeviceContext,
                     paddle::platform::complex64>,
    ops::ScaleKernel<paddle::platform::CPUDeviceContext,
                     paddle::platform::complex128>);
