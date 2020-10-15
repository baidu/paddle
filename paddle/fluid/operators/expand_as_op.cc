/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/operators/expand_as_op.h"
#include <memory>
#include <vector>

namespace paddle {
namespace operators {

using framework::Tensor;

class ExpandAsOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(framework::InferShapeContext* ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("X"), "Input", "X", "ExpandAs");
    OP_INOUT_CHECK(ctx->HasInput("target_tensor"), "Input", "target_tensor",
                   "ExpandAs");
    OP_INOUT_CHECK(ctx->HasOutput("Out"), "Output", "Out", "ExpandAs");
    auto x_dims = ctx->GetInputDim("X");
    auto target_tensor_dims = ctx->GetInputDim("target_tensor");
    PADDLE_ENFORCE_EQ(
        static_cast<size_t>(x_dims.size()), target_tensor_dims.size(),
        platform::errors::InvalidArgument(
            "The rank of Input(target_tensor) must be equal "
            "to the rank of Input(X). But received Input(X): input "
            "rank %u, input shape [%s]; received Input(target_tensor): "
            "input rank %u, input shape [%s].",
            x_dims.size(), x_dims, target_tensor_dims.size(),
            target_tensor_dims));
    PADDLE_ENFORCE_LE(
        x_dims.size(), 6,
        platform::errors::InvalidArgument(
            "The rank of Input(X) must not be greater than 6. But "
            "received: input rank %u, input shape [%s].",
            x_dims.size(), x_dims));
    std::vector<int64_t> out_shape(x_dims.size());
    ctx->SetOutputDim("Out", framework::make_ddim(out_shape));
  }
};

class ExpandAsOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X",
             "(Tensor, default Tensor<float>). A tensor with rank in [1, 6]."
             "X is the input to be expanded.");
    AddOutput("Out",
              "(Tensor, default Tensor<float>). A tensor with rank in [1, 6]."
              "The rank of Output(Out) have the same with Input(X). "
              "After expanding, size of each dimension of Output(Out) is equal "
              "to size of the corresponding dimension of Input(X) multiplying "
              "the corresponding value given by Attr(expand_times).");
    AddInput("target_tensor", "Expand tensor's shape for each dimension.");
    AddComment(R"DOC(
Expand as operator tiles the input by given times number. You should set times
number for each dimension by providing tensor 'expend_tensor'. The rank of X
should be in [1, 6]. Please note that size of 'expend_tensor' must be the same
with X's rank. Following is a using case:
Input(X) is a 3-D tensor with shape [2, 3, 1]:
        [
           [[1], [2], [3]],
           [[4], [5], [6]]
        ]
target_tensors'shape:  [2, 6, 2]
Output(Out) is a 3-D tensor with shape [2, 6, 2]:
        [
            [[1, 1], [2, 2], [3, 3], [1, 1], [2, 2], [3, 3]],
            [[4, 4], [5, 5], [6, 6], [4, 4], [5, 5], [6, 6]]
        ]
)DOC");
  }
};

class ExpandAsGradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(framework::InferShapeContext* ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("X"), "Input", "X", "ExpandAs");
    OP_INOUT_CHECK(ctx->HasInput(framework::GradVarName("Out")), "Input",
                   framework::GradVarName("Out"), "ExpandAs");

    auto x_dims = ctx->GetInputDim("X");
    auto x_grad_name = framework::GradVarName("X");
    if (ctx->HasOutput(x_grad_name)) {
      ctx->SetOutputDim(x_grad_name, x_dims);
    }
  }

  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(OperatorWithKernel::IndicateVarDataType(
                                       ctx, framework::GradVarName("Out")),
                                   ctx.device_context());
  }
};

template <typename T>
class ExpandAsGradOpMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> op) const override {
    op->SetType("expand_as_grad");
    op->SetInput("X", this->Input("X"));
    op->SetInput("target_tensor", this->Input("target_tensor"));
    op->SetInput(framework::GradVarName("Out"), this->OutputGrad("Out"));
    op->SetOutput(framework::GradVarName("X"), this->InputGrad("X"));
    op->SetAttrMap(this->Attrs());
  }
};

DECLARE_NO_NEED_BUFFER_VARS_INFERER(ExpandAsGradNoNeedBufVarsInferer, "X");

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(expand_as, ops::ExpandAsOp, ops::ExpandAsOpMaker,
                  ops::ExpandAsGradOpMaker<paddle::framework::OpDesc>,
                  ops::ExpandAsGradOpMaker<paddle::imperative::OpBase>);
REGISTER_OPERATOR(expand_as_grad, ops::ExpandAsGradOp,
                  ops::ExpandAsGradNoNeedBufVarsInferer);
REGISTER_OP_CPU_KERNEL(
    expand_as, ops::ExpandAsKernel<paddle::platform::CPUDeviceContext, float>,
    ops::ExpandAsKernel<paddle::platform::CPUDeviceContext, double>,
    ops::ExpandAsKernel<paddle::platform::CPUDeviceContext, int>,
    ops::ExpandAsKernel<paddle::platform::CPUDeviceContext, int64_t>,
    ops::ExpandAsKernel<paddle::platform::CPUDeviceContext, bool>);
REGISTER_OP_CPU_KERNEL(
    expand_as_grad,
    ops::ExpandAsGradKernel<paddle::platform::CPUDeviceContext, int>,
    ops::ExpandAsGradKernel<paddle::platform::CPUDeviceContext, int64_t>,
    ops::ExpandAsGradKernel<paddle::platform::CPUDeviceContext, float>,
    ops::ExpandAsGradKernel<paddle::platform::CPUDeviceContext, double>);
