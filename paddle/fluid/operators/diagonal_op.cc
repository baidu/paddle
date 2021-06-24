// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/operators/diagonal_op.h"

namespace paddle {
namespace operators {

class DiagonalOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext *ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("Input"), "Input", "Input", "diagonal");
    OP_INOUT_CHECK(ctx->HasOutput("Out"), "Output", "Out", "diagonal");

    int offset_ = ctx->Attrs().Get<int>("offset");
    int axis1 = ctx->Attrs().Get<int>("axis1");
    int axis2 = ctx->Attrs().Get<int>("axis2");

    auto x_dims = ctx->GetInputDim("Input");
    int axis1_ = axis1 < 0 ? x_dims.size() + axis1 : axis1;
    int axis2_ = axis2 < 0 ? x_dims.size() + axis2 : axis2;

    PADDLE_ENFORCE_GE(
        x_dims.size(), 2,
        platform::errors::OutOfRange("Input's dim is out of range (expected at "
                                     "least 2 dimensions, but got %ld).",
                                     x_dims.size()));
    PADDLE_ENFORCE_LT(
        axis1_, x_dims.size(),
        platform::errors::OutOfRange(
            "Attr(axis1) is out of range (expected to be in range of [%ld, "
            "%ld], but got %ld).",
            -(x_dims.size()), (x_dims.size() - 1), axis1));
    PADDLE_ENFORCE_LT(
        axis2_, x_dims.size(),
        platform::errors::OutOfRange(
            "Attr(axis2) is out of range (expected to be in range of [%ld, "
            "%ld], but got %ld).",
            -(x_dims.size()), (x_dims.size() - 1), axis2));
    PADDLE_ENFORCE_NE(axis1_, axis2_,
                      platform::errors::InvalidArgument(
                          "The dimensions should not be identical "
                          "%d vs %d.",
                          axis1, axis2));

    auto out_dims = vectorize(x_dims);
    // from out_dims get the dim size of axis1_.
    auto axis1_size = out_dims[axis1_];
    auto axis2_size = out_dims[axis2_];
    // delete two dims by attr axis1 and axis2 from out_dims.
    /* example:
       out_dim = [2, 3, 4];
       axis1 = 0;
       axis2 = 1;
       according to the attr of axis1 and axis2, we get:
       out_dim = [4].
    */
    out_dims.erase(out_dims.begin() + std::max(axis1_, axis2_));
    out_dims.erase(out_dims.begin() + std::min(axis1_, axis2_));

    if (offset_ == 0) {
      out_dims.push_back(std::min(axis1_size, axis2_size));
    } else if (offset_ > 0) {
      if ((axis2_size - offset_) > 0) {
        out_dims.push_back(std::min(axis1_size, axis2_size - offset_));
      } else {
        out_dims.push_back(0);
      }
    } else {
      if ((axis1_size + offset_) > 0) {
        out_dims.push_back(std::min(axis1_size + offset_, axis2_size));
      } else {
        out_dims.push_back(0);
      }
    }
    ctx->SetOutputDim("Out", framework::make_ddim(out_dims));
  }
};

class DiagonalOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("Input",
             "(Tensor) The input tensor, from which the diagonals are taken.");
    AddOutput(
        "Out",
        "(Tensor) The partial view of input with the its diagonal elements.");
    AddAttr<int>(
        "offset",
        R"DOC((int, default 0), offset of the diagonal from the main diagonal. Can be both positive and negative. Default: 0.
        )DOC")
        .SetDefault(0);
    AddAttr<int>(
        "axis1",
        R"DOC((int, default 0), the first axis of the 2-D planes from which the diagonals should be taken. 
        Can be either positive or negative. Default: 0.
        )DOC")
        .SetDefault(0);
    AddAttr<int>(
        "axis2",
        R"DOC((int, default 1), the second axis of the 2-D planes from which the diagonals should be taken. 
        Can be either positive or negative. Default: 1.
        )DOC")
        .SetDefault(1);
    AddComment(R"DOC(
Diagonal Operator.
Return a partial view of input with the its diagonal elements of the input tensor.
The behavior of this operator is similar to how `numpy.diagonal` works.

)DOC");
  }
};

class DiagonalGradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext *ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("Input"), "Input", "Input", "DiagonalGrad");
    OP_INOUT_CHECK(ctx->HasOutput(framework::GradVarName("Input")), "Output",
                   framework::GradVarName("Input"), "DiagonalGrad");

    ctx->SetOutputDim(framework::GradVarName("Input"),
                      ctx->GetInputDim("Input"));
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override {
    return framework::OpKernelType(OperatorWithKernel::IndicateVarDataType(
                                       ctx, framework::GradVarName("Out")),
                                   ctx.GetPlace());
  }
};

template <typename T>
class DiagonalGradOpMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> grad_op) const override {
    grad_op->SetType("diagonal_grad");
    grad_op->SetInput("Input", this->Input("Input"));
    grad_op->SetInput(framework::GradVarName("Out"), this->OutputGrad("Out"));
    grad_op->SetOutput(framework::GradVarName("Input"),
                       this->InputGrad("Input"));
    grad_op->SetAttrMap(this->Attrs());
  }
};

DECLARE_NO_NEED_BUFFER_VARS_INFERER(DiagonalGradNoNeedBufferVarsInferer,
                                    "Input");

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(diagonal, ops::DiagonalOp, ops::DiagonalOpMaker,
                  ops::DiagonalGradOpMaker<paddle::framework::OpDesc>,
                  ops::DiagonalGradOpMaker<paddle::imperative::OpBase>);

REGISTER_OPERATOR(diagonal_grad, ops::DiagonalGradOp,
                  ops::DiagonalGradNoNeedBufferVarsInferer)

REGISTER_OP_CPU_KERNEL(diagonal, ops::DiagonalKernel<int>,
                       ops::DiagonalKernel<int64_t>, ops::DiagonalKernel<float>,
                       ops::DiagonalKernel<double>, ops::DiagonalKernel<bool>);

REGISTER_OP_CPU_KERNEL(diagonal_grad, ops::DiagonalGradKernel<int>,
                       ops::DiagonalGradKernel<int64_t>,
                       ops::DiagonalGradKernel<float>,
                       ops::DiagonalGradKernel<double>);
