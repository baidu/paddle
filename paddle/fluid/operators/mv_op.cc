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

#include "paddle/fluid/operators/mv_op.h"
namespace paddle {
namespace operators {

class MVOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "The matrix input of mv op");
    AddInput("Vec", "The vector input of mv op");
    AddOutput("Out", "The output of mv op");
    AddComment(R"DOC(
MV Operator.

This operator is used to perform matrix vector multiplication
of the input tensors `X` and `Vec`.
)DOC");
  }
};

class MVOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(framework::InferShapeContext *context) const override {
    OP_INOUT_CHECK(context->HasInput("X"), "Input", "X", "mv");
    OP_INOUT_CHECK(context->HasInput("Vec"), "Input", "Vec", "mv");
    OP_INOUT_CHECK(context->HasOutput("Out"), "Output", "Out", "mv");

    auto dim_x = context->GetInputDim("X");
    auto dim_vec = context->GetInputDim("Vec");
    PADDLE_ENFORCE_EQ(
        dim_x.size(), 2,
        platform::errors::InvalidArgument(
            "The rank of input X should be 2, but is %d", dim_x.size()));
    PADDLE_ENFORCE_EQ(
        dim_vec.size(), 1,
        platform::errors::InvalidArgument(
            "The rank of input Vec should be 1, but is %d", dim_vec.size()));
    PADDLE_ENFORCE_EQ(dim_x[1], dim_vec[0],
                      platform::errors::InvalidArgument(
                          "X's second dimension is expected to be equal to "
                          "Vec's first dimension"
                          "but recieved X'shape = [%s], Vec's shape = [%s]",
                          dim_x, dim_vec));

    framework::DDim dim_out = framework::make_ddim({dim_x[0]});

    context->SetOutputDim("Out", dim_out);
    context->ShareLoD("X", /*->*/ "Out");
  }
};

template <typename T>
class MVOpGradMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> retv) const override {
    retv->SetType("mv_grad");
    retv->SetInput("X", this->Input("X"));
    retv->SetInput("Vec", this->Input("Vec"));
    retv->SetInput(framework::GradVarName("Out"), this->OutputGrad("Out"));
    retv->SetOutput(framework::GradVarName("X"), this->InputGrad("X"));
    retv->SetOutput(framework::GradVarName("Vec"), this->InputGrad("Vec"));
  }
};

class MVOpGrad : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(framework::InferShapeContext *context) const override {
    OP_INOUT_CHECK(context->HasInput("X"), "Input", "X", "mv");
    OP_INOUT_CHECK(context->HasInput("Vec"), "Input", "Vec", "mv");
    OP_INOUT_CHECK(context->HasInput(framework::GradVarName("Out")), "Input",
                   "Out@GRAD", "mv");
    auto x_dims = context->GetInputDim("X");
    auto vec_dims = context->GetInputDim("Vec");

    auto x_grad_name = framework::GradVarName("X");
    auto vec_grad_name = framework::GradVarName("Vec");

    if (context->HasOutput(x_grad_name)) {
      context->SetOutputDim(x_grad_name, x_dims);
    }
    if (context->HasOutput(vec_grad_name)) {
      context->SetOutputDim(vec_grad_name, vec_dims);
    }
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;

REGISTER_OPERATOR(mv, ops::MVOp, ops::MVOpMaker,
                  ops::MVOpGradMaker<paddle::framework::OpDesc>,
                  ops::MVOpGradMaker<paddle::imperative::OpBase>);
REGISTER_OPERATOR(mv_grad, ops::MVOpGrad);

REGISTER_OP_CPU_KERNEL(
    mv, ops::MVKernel<paddle::platform::CPUDeviceContext, float>,
    ops::MVKernel<paddle::platform::CPUDeviceContext, double>);
REGISTER_OP_CPU_KERNEL(
    mv_grad, ops::MVGradKernel<paddle::platform::CPUDeviceContext, float>,
    ops::MVGradKernel<paddle::platform::CPUDeviceContext, double>);
