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

#include "paddle/fluid/operators/top_k_op.h"
#include <memory>

namespace paddle {
namespace operators {

class TopkOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE_EQ(ctx->HasInput("X"), true,
                      platform::errors::InvalidArgument(
                          "Input(X) of TopkOp should not be null."));
    PADDLE_ENFORCE_EQ(ctx->HasOutput("Out"), true,
                      platform::errors::InvalidArgument(
                          "Output(Out) of TopkOp should not be null."));
    PADDLE_ENFORCE_EQ(ctx->HasOutput("Indices"), true,
                      platform::errors::InvalidArgument(
                          "Output(Indices) of TopkOp should not be null."));

    auto input_dims = ctx->GetInputDim("X");
    const int k = static_cast<int>(ctx->Attrs().Get<int>("k"));

    PADDLE_ENFORCE_GE(k, 1,
                      platform::errors::InvalidArgument(
                          "Attribute k must be >= 1, but got k is %d.", k));
    PADDLE_ENFORCE_GE(input_dims.size(), 1, platform::errors::InvalidArgument(
                                                "input must have >= 1d shape"));

    if (ctx->IsRuntime()) {
      PADDLE_ENFORCE_GE(
          input_dims[input_dims.size() - 1], k,
          platform::errors::InvalidArgument("input must have >= k columns"));
    }

    framework::DDim dims = input_dims;
    dims[dims.size() - 1] = k;
    ctx->SetOutputDim("Out", dims);
    ctx->SetOutputDim("Indices", dims);
    ctx->ShareLoD("X", "Out");
    ctx->ShareLoD("X", "Indices");
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    framework::LibraryType library_{framework::LibraryType::kPlain};
    framework::DataLayout layout_ = framework::DataLayout::kAnyLayout;
    return framework::OpKernelType(
        OperatorWithKernel::IndicateVarDataType(ctx, "X"), ctx.device_context(),
        layout_, library_);
  }
};

class TopkOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "(Tensor) The input of Topk op");
    AddInput("K",
             "(Tensor)  Number of top elements to look for along "
             "the last dimension (along each row for matrices).")
        .AsDispensable();
    AddOutput("Out", "(Tensor) The output tensor of Topk op");
    AddOutput("Indices", "(Tensor) The indices of Topk elements of input");
    AddComment(R"DOC(
Top K operator

If the input is a vector (1d tensor), this operator finds the k largest 
entries in the vector and outputs their values and indices as vectors. 
Thus values[j] is the j-th largest entry in input, and its index is indices[j].

For matrices, this operator computes the top k entries in each row. )DOC");
    AddAttr<int>("k",
                 "(int, default 1) Number of top elements to look for along "
                 "the last dimension (along each row for matrices).")
        .SetDefault(1);
  }
};

class TopkOpGrad : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;
  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE_EQ(
        ctx->HasInput("X"), true,
        platform::errors::InvalidArgument("Input(X) should be not null"));
    PADDLE_ENFORCE_EQ(
        ctx->HasInput("Indices"), true,
        platform::errors::InvalidArgument("Input(Indices) should be not null"));
    PADDLE_ENFORCE_EQ(ctx->HasInput(framework::GradVarName("Out")), true,
                      platform::errors::InvalidArgument(
                          "Grad Input(Out) should be not null"));
    PADDLE_ENFORCE_EQ(
        ctx->HasOutput(framework::GradVarName("X")), true,
        platform::errors::InvalidArgument("Grad Output(X) should be not null"));

    auto x_dims = ctx->GetInputDim("X");
    ctx->SetOutputDim(framework::GradVarName("X"), x_dims);
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    auto data_type = OperatorWithKernel::IndicateVarDataType(
        ctx, framework::GradVarName("Out"));
    return framework::OpKernelType(data_type, ctx.device_context());
  }
};

template <typename T>
class TopkGradOpMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> op) const override {
    op->SetType("top_k_grad");
    op->SetInput(framework::GradVarName("Out"), this->OutputGrad("Out"));
    op->SetInput("X", this->Input("X"));
    op->SetInput("Indices", this->Output("Indices"));
    op->SetOutput(framework::GradVarName("X"), this->InputGrad("X"));
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(top_k, ops::TopkOp, ops::TopkOpMaker,
                  ops::TopkGradOpMaker<paddle::framework::OpDesc>,
                  ops::TopkGradOpMaker<paddle::imperative::OpBase>);

REGISTER_OPERATOR(top_k_grad, ops::TopkOpGrad);

REGISTER_OP_CPU_KERNEL(top_k,
                       ops::TopkKernel<paddle::platform::CPUPlace, float>,
                       ops::TopkKernel<paddle::platform::CPUPlace, double>);

REGISTER_OP_CPU_KERNEL(top_k_grad,
                       ops::TopkGradKernel<paddle::platform::CPUPlace, float>,
                       ops::TopkGradKernel<paddle::platform::CPUPlace, double>);
