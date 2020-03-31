/* Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/operators/where_op.h"

namespace paddle {
namespace operators {

class WhereOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE_EQ(ctx->HasInput("Condition"), true,
                      platform::errors::NotFound(
                          "Input(Condition) of whereOp should not be null."));
    PADDLE_ENFORCE_EQ(
        ctx->HasInput("X"), true,
        platform::errors::NotFound("Input(X) of whereOp should not be null."));
    PADDLE_ENFORCE_EQ(
        ctx->HasInput("Y"), true,
        platform::errors::NotFound("Input(Y) of whereOp should not be null."));
    PADDLE_ENFORCE_EQ(ctx->HasOutput("Out"), true,
                      platform::errors::NotFound(
                          "Output(Out) of whereOp should not be null."));

    auto x_size = ctx->GetInputDim("X").size();
    auto y_size = ctx->GetInputDim("Y").size();

    PADDLE_ENFORCE_GE(ctx->GetInputDim("Condition").size(), 1UL,
                      platform::errors::InvalidArgument(
                          "Input(Condition) of where should not be empty"));
    PADDLE_ENFORCE_GE(x_size, 1UL,
                      platform::errors::InvalidArgument(
                          "Inputs(X) of where should not be empty."));
    PADDLE_ENFORCE_GE(y_size, 1UL,
                      platform::errors::InvalidArgument(
                          "Inputs(Y) of where should not be empty."));
    PADDLE_ENFORCE_EQ(x_size, y_size,
                      platform::errors::InvalidArgument(
                          "The size of Inputs(Y) and Inputs(Y) should be same. "
                          "But received X size = %d, Y size = %d",
                          x_size, y_size));
    ctx->SetOutputDim("Out", ctx->GetInputDim("X"));
    ctx->ShareLoD("X", /*->*/ "Out");
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(
        OperatorWithKernel::IndicateVarDataType(ctx, "X"), ctx.GetPlace());
  }
};

class WhereGradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("X"), "Input(X) should not be null");
    PADDLE_ENFORCE(ctx->HasInput("Y"), "Input(Y) should not be null");
    PADDLE_ENFORCE(ctx->HasInput(framework::GradVarName("Out")),
                   "Input(Out@GRAD) should not be null");

    auto x_dims = ctx->GetInputDim("X");
    auto y_dims = ctx->GetInputDim("Y");

    auto x_grad_name = framework::GradVarName("X");
    auto y_grad_name = framework::GradVarName("Y");

    if (ctx->HasOutput(x_grad_name)) {
      ctx->SetOutputDim(x_grad_name, x_dims);
    }
    if (ctx->HasOutput(y_grad_name)) {
      ctx->SetOutputDim(y_grad_name, y_dims);
    }
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(OperatorWithKernel::IndicateVarDataType(
                                       ctx, framework::GradVarName("Out")),
                                   ctx.GetPlace());
  }
};

class WhereOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("Condition", "(Tensor) A bool tensor whose rank is at least 1");
    AddInput("X", "(Tensor), The first input tensor of where op");
    AddInput("Y", "(Tensor), The second input tensor of where op");
    AddOutput("Out", "(Tensor), The output tensor of mul op.");
    AddComment(R"DOC(
      Where Operator.
      Return a tensor of elements selected from either $X$ or $Y$, depending on condition.
)DOC");
  }
};

template <typename T>
class WhereOpGradMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> grad) const override {
    grad->SetType("where_grad");
    grad->SetInput("Condition", this->Input("Condition"));
    grad->SetInput("X", this->Input("X"));
    grad->SetInput("Y", this->Input("Y"));
    grad->SetInput(framework::GradVarName("Out"), this->OutputGrad("Out"));
    grad->SetOutput(framework::GradVarName("X"), this->InputGrad("X"));
    grad->SetOutput(framework::GradVarName("Y"), this->InputGrad("Y"));
  }
};
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(where, ops::WhereOp, ops::WhereOpMaker,
                  ops::WhereOpGradMaker<paddle::framework::OpDesc>,
                  ops::WhereOpGradMaker<paddle::imperative::OpBase>);

REGISTER_OPERATOR(where_grad, ops::WhereGradOp);
REGISTER_OP_CPU_KERNEL(
    where, ops::WhereKernel<paddle::platform::CPUDeviceContext, float>,
    ops::WhereKernel<paddle::platform::CPUDeviceContext, double>,
    ops::WhereKernel<paddle::platform::CPUDeviceContext, int>);
REGISTER_OP_CPU_KERNEL(
    where_grad, ops::WhereGradKernel<paddle::platform::CPUDeviceContext, float>,
    ops::WhereGradKernel<paddle::platform::CPUDeviceContext, double>,
    ops::WhereGradKernel<paddle::platform::CPUDeviceContext, int>);
