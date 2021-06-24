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

#include "paddle/fluid/operators/digamma_op.h"

namespace paddle {
namespace operators {

class DigammaOp : public framework::OperatorWithKernel {
 public:
  DigammaOp(const std::string &type, const framework::VariableNameMap &inputs,
            const framework::VariableNameMap &outputs,
            const framework::AttributeMap &attrs)
      : OperatorWithKernel(type, inputs, outputs, attrs) {}

  void InferShape(framework::InferShapeContext *ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("X"), "Input", "X", "Digamma");
    OP_INOUT_CHECK(ctx->HasOutput("Out"), "Output", "Out", "Digamma");

    auto in_dims = ctx->GetInputDim("X");

    ctx->SetOutputDim("Out", in_dims);
    ctx->ShareLoD("X", "Out");
  }
};

class DigammaOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "(Tensor), The input tensor of digamma operator.");
    AddOutput("Out", "(Tensor), The output tensor of digamma operator.");
    AddComment(R"DOC(
Digamma Operator.

This operator is used to perform elementwise digamma for input $X$.
$$out = \Psi(x) = \frac{ \Gamma^{'}(x) }{ \Gamma(x) }$$

)DOC");
  }
};

class DigammaGradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;
  void InferShape(framework::InferShapeContext *ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput(framework::GradVarName("Out")), "Input",
                   "Out@Grad", "DigammaGrad");
    OP_INOUT_CHECK(ctx->HasInput("X"), "Input", "X", "DigammaGrad");
    OP_INOUT_CHECK(ctx->HasOutput(framework::GradVarName("X")), "Output",
                   "X@Grad", "DigammaGrad");

    auto dout_dims = ctx->GetInputDim(framework::GradVarName("Out"));
    ctx->SetOutputDim(framework::GradVarName("X"), dout_dims);
    ctx->ShareLoD(framework::GradVarName("Out"), framework::GradVarName("X"));
  }
};

template <typename T>
class DigammaGradOpMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

  void Apply(GradOpPtr<T> retv) const override {
    retv->SetType("digamma_grad");
    retv->SetInput(framework::GradVarName("Out"), this->OutputGrad("Out"));
    retv->SetInput("X", this->Input("X"));
    retv->SetAttrMap(this->Attrs());
    retv->SetOutput(framework::GradVarName("X"), this->InputGrad("X"));
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OPERATOR(digamma, ops::DigammaOp, ops::DigammaOpMaker,
                  ops::DigammaGradOpMaker<paddle::framework::OpDesc>,
                  ops::DigammaGradOpMaker<paddle::imperative::OpBase>);
REGISTER_OPERATOR(digamma_grad, ops::DigammaGradOp);

REGISTER_OP_CPU_KERNEL(
    digamma, ops::DigammaKernel<paddle::platform::CPUDeviceContext, float>,
    ops::DigammaKernel<paddle::platform::CPUDeviceContext, double>);

REGISTER_OP_CPU_KERNEL(
    digamma_grad,
    ops::DigammaGradKernel<paddle::platform::CPUDeviceContext, float>,
    ops::DigammaGradKernel<paddle::platform::CPUDeviceContext, double>);
