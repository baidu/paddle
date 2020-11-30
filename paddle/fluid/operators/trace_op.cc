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

#include "paddle/fluid/operators/trace_op.h"

namespace paddle {
namespace operators {

class TraceOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext *ctx) const override {
    PADDLE_ENFORCE_EQ(
        ctx->HasInput("Input"), true,
        platform::errors::NotFound("Input of TraceOp is not found."));

    PADDLE_ENFORCE_EQ(
        ctx->HasOutput("Out"), true,
        platform::errors::NotFound("Output of TraceOp is not found."));

    int dim1 = ctx->Attrs().Get<int>("axis1");
    int dim2 = ctx->Attrs().Get<int>("axis2");

    auto x_dims = ctx->GetInputDim("Input");

    int dim1_ = dim1 < 0 ? x_dims.size() + dim1 : dim1;
    int dim2_ = dim2 < 0 ? x_dims.size() + dim2 : dim2;

    PADDLE_ENFORCE_GE(
        x_dims.size(), 2,
        platform::errors::OutOfRange(
            "trace requires an tensor of at least two dimensions"));
    PADDLE_ENFORCE_LT(
        dim1_, x_dims.size(),
        platform::errors::OutOfRange(
            "Attr(dim1) is out of range (expected to be in range of [%ld, "
            "%ld], but got %ld).",
            -(x_dims.size()), (x_dims.size() - 1), dim1));
    PADDLE_ENFORCE_LT(
        dim2_, x_dims.size(),
        platform::errors::OutOfRange(
            "Attr(dim2) is out of range (expected to be in range of [%ld, "
            "%ld], but got %ld).",
            -(x_dims.size()), (x_dims.size() - 1), dim2));
    PADDLE_ENFORCE_NE(dim1_, dim2_,
                      platform::errors::InvalidArgument(
                          "The dimensions should not be identical "
                          "%ld vs %ld.",
                          dim1, dim2));

    auto sizes = vectorize(x_dims);
    if (x_dims.size() == 2) {
      sizes.clear();
      sizes.push_back(1);
    } else {
      sizes.erase(sizes.begin() + std::max(dim1_, dim2_));
      sizes.erase(sizes.begin() + std::min(dim1_, dim2_));
    }
    ctx->SetOutputDim("Out", framework::make_ddim(sizes));
  }
};

class TraceOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("Input",
             "(Tensor) The input tensor, from which the diagonals are taken.");
    AddOutput("Out", "(Tensor) the sum along diagonals of the input tensor");
    AddAttr<int>(
        "offset",
        R"DOC((int, default 0), offset of the diagonal from the main diagonal. Can be both positive and negative. Defaults to 0.
        )DOC")
        .SetDefault(0);
    AddAttr<int>(
        "axis1",
        R"DOC((int, default 0), the first axis of the 2-D planes from which the diagonals should be taken. 
        Can be either positive or negative. Default: 0.
        )DOC")
        .SetDefault(-2);
    AddAttr<int>(
        "axis2",
        R"DOC((int, default 1), the second axis of the 2-D planes from which the diagonals should be taken. 
        Can be either positive or negative. Default: 1.
        )DOC")
        .SetDefault(-1);
    AddComment(R"DOC(
Trace Operator.
Return the sum along diagonals of the input tensor.
The behavior of this operator is similar to how `numpy.trace` works.

If Input is 2-D, returns the sum of diagonal. 
If Input has larger dimensions, then returns an tensor of diagonals sum, diagonals be taken from
the 2-D planes specified by dim1 and dim2.

)DOC");
  }
};
class TraceOpGrad : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext *ctx) const override {
    PADDLE_ENFORCE_EQ(
        ctx->HasInput("Input"), true,
        platform::errors::NotFound("Input(Input) of TraceOp is not found."));
    PADDLE_ENFORCE_EQ(ctx->HasOutput(framework::GradVarName("Input")), true,
                      platform::errors::NotFound(
                          "Output(Input@GRAD) of TraceGradOp is not found."));
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
class TraceGradOpMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> grad_op) const override {
    grad_op->SetType("trace_grad");
    grad_op->SetInput("Input", this->Input("Input"));
    grad_op->SetInput(framework::GradVarName("Out"), this->OutputGrad("Out"));
    grad_op->SetOutput(framework::GradVarName("Input"),
                       this->InputGrad("Input"));
    grad_op->SetAttrMap(this->Attrs());
  }
};

DECLARE_NO_NEED_BUFFER_VARS_INFERER(TraceGradNoNeedBufferVarsInferer, "Input");

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(trace, ops::TraceOp, ops::TraceOpMaker,
                  ops::TraceGradOpMaker<paddle::framework::OpDesc>,
                  ops::TraceGradOpMaker<paddle::imperative::OpBase>);

REGISTER_OPERATOR(trace_grad, ops::TraceOpGrad,
                  ops::TraceGradNoNeedBufferVarsInferer);
REGISTER_OP_CPU_KERNEL(
    trace, ops::TraceKernel<paddle::platform::CPUDeviceContext, int>,
    ops::TraceKernel<paddle::platform::CPUDeviceContext,
                     paddle::platform::complex64>,
    ops::TraceKernel<paddle::platform::CPUDeviceContext,
                     paddle::platform::complex128>,
    ops::TraceKernel<paddle::platform::CPUDeviceContext, float>,
    ops::TraceKernel<paddle::platform::CPUDeviceContext, double>,
    ops::TraceKernel<paddle::platform::CPUDeviceContext, int64_t>);
REGISTER_OP_CPU_KERNEL(
    trace_grad, ops::TraceGradKernel<paddle::platform::CPUDeviceContext, int>,
    ops::TraceGradKernel<paddle::platform::CPUDeviceContext,
                         paddle::platform::complex64>,
    ops::TraceGradKernel<paddle::platform::CPUDeviceContext,
                         paddle::platform::complex128>,
    ops::TraceGradKernel<paddle::platform::CPUDeviceContext, float>,
    ops::TraceGradKernel<paddle::platform::CPUDeviceContext, double>,
    ops::TraceGradKernel<paddle::platform::CPUDeviceContext, int64_t>);
