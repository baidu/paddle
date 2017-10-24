/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. */

#include "paddle/operators/l2_loss_op.h"

namespace paddle {
namespace operators {

using framework::Tensor;

class L2LossOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("X"), "Input(X) should be not null.");
    PADDLE_ENFORCE(ctx->HasOutput("Out"), "Output(Out) should be not null.");

    ctx->SetOutputDim("Out", {1});
    ctx->ShareLoD("X", /*->*/ "Out");
  }
};

class L2LossGradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("X"), "Input(X) should be not null.");
    PADDLE_ENFORCE(ctx->HasInput(framework::GradVarName("Out")),
                   "Input(Out@GRAD) should be not null.");
    PADDLE_ENFORCE(ctx->HasOutput(framework::GradVarName("X")),
                   "Output(X@GRAD) should be not null.");

    ctx->SetOutputDim(framework::GradVarName("X"), ctx->GetInputDim("X"));
  }
};

class L2LossOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  L2LossOpMaker(framework::OpProto* proto, framework::OpAttrChecker* op_checker)
      : framework::OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput("X", "(Tensor) The input of l2_loss op.");
    AddOutput("Out", "(Tensor) The output of l2_loss op.");
    AddComment(R"DOC(
L2Loss Operator.

Computes half the squared L2 norm of a tensor.

Out = sum (X ** 2) / 2

)DOC");
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP(l2_loss, ops::L2LossOp, ops::L2LossOpMaker, l2_loss_grad,
            ops::L2LossGradOp);
REGISTER_OP_CPU_KERNEL(l2_loss,
                       ops::L2LossKernel<paddle::platform::CPUPlace, float>);
REGISTER_OP_CPU_KERNEL(
    l2_loss_grad, ops::L2LossGradKernel<paddle::platform::CPUPlace, float>);
