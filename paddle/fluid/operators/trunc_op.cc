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

#include "paddle/fluid/operators/trunc_op.h"

namespace paddle {
namespace operators {

class TruncOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext *ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("X"), "Input", "X", "trunc");
    OP_INOUT_CHECK(ctx->HasOutput("Out"), "Output", "Out", "trunc");
    auto input_dims = ctx->GetInputDim("X");
    ctx->SetOutputDim("Out", input_dims);
    ctx->ShareLoD("X", /*->*/ "Out");
  }
};

class TruncOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "(Tensor), The input tensor of trunc op.");
    AddOutput("Out", "(Tensor), The output tensor of trunc op.");
    AddComment(R"DOC(
Trunc Operator.
Returns a new tensor with the truncated integer values  of input.
$$out = trunc(x)$$
)DOC");
  }
};

class TruncGradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext *ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput(framework::GradVarName("Out")), "Input",
                   framework::GradVarName("Out"), "TruncGrad");
    OP_INOUT_CHECK(ctx->HasOutput(framework::GradVarName("X")), "Output",
                   framework::GradVarName("X"), "TruncGrad");

    auto dout_dims = ctx->GetInputDim(framework::GradVarName("Out"));
    ctx->SetOutputDim(framework::GradVarName("X"), dout_dims);
  }
};

template <typename T>
class TruncGradOpMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

  void Apply(GradOpPtr<T> retv) const override {
    retv->SetType("trunc_grad");
    retv->SetInput(framework::GradVarName("Out"), this->OutputGrad("Out"));
    retv->SetAttrMap(this->Attrs());
    retv->SetOutput(framework::GradVarName("X"), this->InputGrad("X"));
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(trunc, ops::TruncOp, ops::TruncOpMaker,
                  ops::TruncGradOpMaker<paddle::framework::OpDesc>,
                  ops::TruncGradOpMaker<paddle::imperative::OpBase>);

REGISTER_OPERATOR(trunc_grad, ops::TruncGradOp);

REGISTER_OP_CPU_KERNEL(trunc, ops::TruncKernel<float>, ops::TruncKernel<double>,
                       ops::TruncKernel<int>, ops::TruncKernel<int64_t>);

REGISTER_OP_CPU_KERNEL(trunc_grad, ops::TruncGradKernel<float>,
                       ops::TruncGradKernel<double>, ops::TruncGradKernel<int>,
                       ops::TruncGradKernel<int64_t>);
