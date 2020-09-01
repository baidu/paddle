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

#include "paddle/fluid/operators/argsort_op.h"
#include <memory>

namespace paddle {
namespace operators {

class ArgsortOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("X"), "Input", "X", "argsort");
    OP_INOUT_CHECK(ctx->HasOutput("Out"), "Output", "Out", "argsort");
    OP_INOUT_CHECK(ctx->HasOutput("Indices"), "Output", "Indices", "argsort");

    auto in_dims = ctx->GetInputDim("X");
    int axis = ctx->Attrs().Get<int>("axis");

    auto num_dims = in_dims.size();
    PADDLE_ENFORCE_GE(axis, -num_dims,
                      platform::errors::InvalidArgument(
                          "'axis'(%d) must be greater than or equal to"
                          " -num_dims(%d).",
                          axis, -num_dims));
    PADDLE_ENFORCE_LT(
        axis, num_dims,
        platform::errors::InvalidArgument(
            "'axis'(%d) must be less than num_dims(%d).", axis, num_dims));

    // check the element num of sorted axis is not be larger than INT_MAX
    if (ctx->IsRuntime()) {
      const int& dtype = ctx->Attrs().Get<int>("dtype");
      if (dtype == framework::proto::VarType::INT32) {
        if (axis < 0) {
          axis += num_dims;
        }
        PADDLE_ENFORCE_LE(
            in_dims[axis], INT_MAX,
            "The element num at sorted axis in argsort/sort op is "
            "%d, is larger than int32 maximum value:%d, you must "
            "set the dtype of argsort/sort to 'int64'.",
            in_dims[axis], INT_MAX);
      }
    }

    ctx->ShareDim("X", "Out");
    ctx->ShareDim("X", "Indices");
    ctx->ShareLoD("X", "Out");
    ctx->ShareLoD("X", "Indices");
  }
};

class ArgsortGradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    ctx->SetOutputDim(framework::GradVarName("X"), ctx->GetInputDim("X"));
    ctx->ShareLoD("X", /*-->*/ framework::GradVarName("X"));
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(OperatorWithKernel::IndicateVarDataType(
                                       ctx, framework::GradVarName("Out")),
                                   ctx.device_context());
  }
};

class ArgsortOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "(Tensor) The input of Argsort op.");
    AddOutput("Out",
              "(Tensor) The sorted tensor of Argsort op, with the same "
              "shape as Input(X).");
    AddOutput("Indices",
              "(Tensor) The indices of a tensor giving the sorted order, with "
              "the same shape as Input(X).");
    AddComment(R"DOC(
Argsort operator

Performs sorting on the input tensor along the given axis and outputs two 
tensors, Output(Out) and Output(Indices). They reserve the same shape 
with Input(X), and Output(Out) represents the sorted tensor while 
Output(Indices) gives the sorted order along the given axis Attr(axis).

 )DOC");
    AddAttr<int>("axis",
                 "(int, default -1) The axis along which to sort the tensor. "
                 "When axis < 0, the actual axis will be the |axis|'th "
                 "counting backwards. Default -1, the last dimension.")
        .SetDefault(-1);
    AddAttr<bool>(
        "descending",
        "(bool, default false) The descending attribute is a flag to tell"
        "algorithm how to sort the input data."
        "If descending is true, will sort by descending order,"
        "else if false, sort by ascending order. Default value is false.")
        .SetDefault(false);

    AddAttr<int>("dtype",
                 "(int, 3), the dtype of indices, the indices dtype must be "
                 "int32, int64."
                 "default dtype is int64, and proto value is 3.")
        .SetDefault(3)
        .InEnum({2, 3});
  }
};

template <typename T>
class ArgsortGradOpMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> op) const override {
    op->SetType("argsort_grad");
    op->SetInput("Indices", this->Output("Indices"));
    op->SetInput("X", this->Input("X"));
    op->SetInput(framework::GradVarName("Out"), this->OutputGrad("Out"));
    op->SetOutput(framework::GradVarName("X"), this->InputGrad("X"));
    op->SetAttrMap(this->Attrs());
  }
};

DECLARE_NO_NEED_BUFFER_VARS_INFERER(ArgsortGradNoNeedBufferVarsInferer, "X");

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(argsort, ops::ArgsortOp, ops::ArgsortOpMaker,
                  ops::ArgsortGradOpMaker<paddle::framework::OpDesc>,
                  ops::ArgsortGradOpMaker<paddle::imperative::OpBase>);
REGISTER_OPERATOR(argsort_grad, ops::ArgsortGradOp,
                  ops::ArgsortGradNoNeedBufferVarsInferer);
REGISTER_OP_CPU_KERNEL(argsort,
                       ops::ArgsortKernel<paddle::platform::CPUPlace, float>,
                       ops::ArgsortKernel<paddle::platform::CPUPlace, double>,
                       ops::ArgsortKernel<paddle::platform::CPUPlace, int>,
                       ops::ArgsortKernel<paddle::platform::CPUPlace, int64_t>);
REGISTER_OP_CPU_KERNEL(
    argsort_grad, ops::ArgsortGradientKernel<paddle::platform::CPUPlace, float>,
    ops::ArgsortGradientKernel<paddle::platform::CPUPlace, double>,
    ops::ArgsortGradientKernel<paddle::platform::CPUPlace, int>,
    ops::ArgsortGradientKernel<paddle::platform::CPUPlace, int64_t>);
