// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "paddle/fluid/operators/dist_op.h"
#include <string>
#include <vector>
#include "paddle/fluid/framework/op_registry.h"
namespace paddle {
namespace operators {

class DistOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext *ctx) const override {
    PADDLE_ENFORCE_EQ(ctx->HasInput("X"), true,
                      platform::errors::NotFound(
                          "The Input(X) of Op(dist) should not be null."));
    PADDLE_ENFORCE_EQ(ctx->HasInput("Y"), true,
                      platform::errors::NotFound(
                          "The Input(Y) of Op(dist) should not be null."));
    PADDLE_ENFORCE_EQ(ctx->HasOutput("Out"), true,
                      platform::errors::NotFound(
                          "The Output(Out) of Op(dist) should not be null."));
    auto out_dims = std::vector<int>(1);
    ctx->SetOutputDim("Out", {1});
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override {
    return framework::OpKernelType(
        OperatorWithKernel::IndicateVarDataType(ctx, "X"),
        ctx.device_context());
  }
};

class DistOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "The input Tensor of Dist Op.");
    AddInput("Y", "The Right-hand-side input Tensor of Dist Op.");
    AddOutput("Out",
              "The output of Dist Op, "
              "which is the p-norm of (X - Y)");
    AddAttr<float>("p", "the norm to be computed.").SetDefault(2.0f);
    AddComment(R"DOC(
Dist Operator.
Given two tensors X and Y, compute Lp-norm of (X-Y).
where, Z = X - Y
$$
\left \| Z \right \|_{p} = (\sum_{i=i}^{m} |z_i|^p)^{1/p}
$$

1. when p = 0, the 0-norm of z is simply the number of non-zero elements of z.
$$
\left \| Z \right \|_{\infty} = (z_{1}^{p} + z_{2}^{p} + ... + z_{n}^{p})^{1/p}
$$

2. when p = inf, the inf-norm of Z is the maximum element of Z.
$$
\left \| Z \right \|_{\infty} = \max_{i}\left | z_i \right |
$$

3. when p = -inf, the inf-norm of Z is the minimum element of Z.
$$
\left \| Z \right \|_{-\infty} = \min_{i}\left | z_i \right |
$$
    )DOC");
  }
};

class DistOpGrad : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext *ctx) const override {
    auto x_dims = ctx->GetInputDim("X");
    auto y_dims = ctx->GetInputDim("Y");
    if (ctx->HasOutput(framework::GradVarName("X"))) {
      ctx->SetOutputDim(framework::GradVarName("X"), x_dims);
    }
    if (ctx->HasOutput(framework::GradVarName("Y"))) {
      ctx->SetOutputDim(framework::GradVarName("Y"), y_dims);
    }
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override {
    return framework::OpKernelType(OperatorWithKernel::IndicateVarDataType(
                                       ctx, framework::GradVarName("Out")),
                                   ctx.device_context());
  }
};

template <typename T>
class DistGradOpMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> op) const override {
    op->SetType(this->ForwardOpType() + "_grad");
    op->SetInput("X", this->Input("X"));
    op->SetInput("Y", this->Input("Y"));
    op->SetInput("Out", this->Output("Out"));
    op->SetInput(framework::GradVarName("Out"), this->OutputGrad("Out"));

    op->SetOutput(framework::GradVarName("X"), this->InputGrad("X"));
    op->SetOutput(framework::GradVarName("Y"), this->InputGrad("Y"));
    op->SetAttrMap(this->Attrs());
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(dist, ops::DistOp, ops::DistOpMaker,
                  ops::DistGradOpMaker<paddle::framework::OpDesc>,
                  ops::DistGradOpMaker<paddle::imperative::OpBase>);
REGISTER_OPERATOR(dist_grad, ops::DistOpGrad);
REGISTER_OP_CPU_KERNEL(
    dist, ops::DistKernel<paddle::platform::CPUDeviceContext, float>,
    ops::DistKernel<paddle::platform::CPUDeviceContext, double>);
REGISTER_OP_CPU_KERNEL(
    dist_grad, ops::DistGradKernel<paddle::platform::CPUDeviceContext, float>,
    ops::DistGradKernel<paddle::platform::CPUDeviceContext, double>)
