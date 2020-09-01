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

#include "paddle/fluid/operators/scalar_mul_op.h"
#include "paddle/fluid/framework/op_registry.h"

namespace paddle {
namespace operators {

class ScalarMulOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("X"), "Input", "X", "ScalarMul");
    OP_INOUT_CHECK(ctx->HasOutput("Out"), "Output", "Out", "ScalarMul");

    auto in_dims = ctx->GetInputDim("X");
    ctx->SetOutputDim("Out", in_dims);
  }
};

class ScalarMulOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "(Tensor), The input tensor.");
    AddOutput("Out", "(Tensor), The output tensor of scalar mul op");
    AddAttr<float>("a", "(Scalar), The scaled factor of the input tensor");
    AddAttr<float>("b", "(Scalar), The bias of the scalr mul op");
    AddComment(R"DOC(
              Scalar Mul Operator.
              This operator is used to perform scalar multiply for input tensor X,
              Out = a * X + b
              )DOC");
  }
};

class ScalarMulGradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput(framework::GradVarName("Out")), "Input",
                   framework::GradVarName("Out"), "ScalarMulGrad");
    OP_INOUT_CHECK(ctx->HasOutput(framework::GradVarName("X")), "Output",
                   framework::GradVarName("X"), "ScalarMulGrad");

    auto in_dims = ctx->GetInputDim(framework::GradVarName("Out"));
    ctx->SetOutputDim(framework::GradVarName("X"), in_dims);
  }
};

template <typename T>
class ScalarMulOpGradMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

  void Apply(GradOpPtr<T> op) const override {
    op->SetType("scalar_mul_grad");
    // op->SetInput("X", this->Input("X"));
    op->SetInput(framework::GradVarName("Out"), this->OutputGrad("Out"));
    op->SetOutput(framework::GradVarName("X"), this->InputGrad("X"));
    op->SetAttrMap(this->Attrs());
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
using CPU = paddle::platform::CPUDeviceContext;
REGISTER_OPERATOR(scalar_mul, ops::ScalarMulOp, ops::ScalarMulOpMaker,
                  ops::ScalarMulOpGradMaker<paddle::framework::OpDesc>,
                  ops::ScalarMulOpGradMaker<paddle::imperative::OpBase>);
REGISTER_OPERATOR(scalar_mul_grad, ops::ScalarMulGradOp);
REGISTER_OP_CPU_KERNEL(scalar_mul, ops::ScalarMulKernel<CPU, float>,
                       ops::ScalarMulKernel<CPU, double>);
REGISTER_OP_CPU_KERNEL(scalar_mul_grad, ops::ScalarMulGradKernel<CPU, float>,
                       ops::ScalarMulGradKernel<CPU, double>);
