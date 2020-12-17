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

#include "paddle/fluid/operators/mean_op.h"
#include <memory>
#include <string>
#include <unordered_map>

namespace paddle {
namespace operators {

class MeanOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("X"), "Input", "X", "mean");
    OP_INOUT_CHECK(ctx->HasOutput("Out"), "Output", "Out", "mean");
    ctx->SetOutputDim("Out", {1});
  }
};

class MeanOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "(Tensor) The input of mean op");
    AddOutput("Out", "(Tensor) The output of mean op");
    AddComment(R"DOC(
Mean Operator calculates the mean of all elements in X.

)DOC");
  }
};

class MeanOpInferVarType : public framework::PassInDtypeAndVarTypeToOutput {
 protected:
  std::unordered_map<std::string, std::string>& GetInputOutputWithSameType()
      const override {
    static std::unordered_map<std::string, std::string> m{{"X", /*->*/ "Out"}};
    return m;
  }
};

class MeanGradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    ctx->SetOutputDim(framework::GradVarName("X"), ctx->GetInputDim("X"));
    ctx->ShareLoD("X", framework::GradVarName("X"));
  }

  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    auto input_data_type = OperatorWithKernel::IndicateVarDataType(
        ctx, framework::GradVarName("Out"));
    return framework::OpKernelType(input_data_type, ctx.GetPlace());
  }
};

template <typename T>
class MeanGradMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> grad_op) const override {
    grad_op->SetType("mean_grad");
    grad_op->SetInput("X", this->Input("X"));
    grad_op->SetInput(framework::GradVarName("Out"), this->OutputGrad("Out"));
    grad_op->SetOutput(framework::GradVarName("X"), this->InputGrad("X"));
  }
};

DECLARE_NO_NEED_BUFFER_VARS_INFERER(MeanGradNoNeedBufferVarsInferer, "X");

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(mean, ops::MeanOp, ops::MeanOpMaker, ops::MeanOpInferVarType,
                  ops::MeanGradMaker<paddle::framework::OpDesc>,
                  ops::MeanGradMaker<paddle::imperative::OpBase>);
REGISTER_OPERATOR(mean_grad, ops::MeanGradOp,
                  ops::MeanGradNoNeedBufferVarsInferer);
REGISTER_OP_CPU_KERNEL(
    mean, ops::MeanKernel<paddle::platform::CPUDeviceContext, float>,
    ops::MeanKernel<paddle::platform::CPUDeviceContext, double>,
    ops::MeanKernel<paddle::platform::CPUDeviceContext,
                    paddle::platform::complex64>,
    ops::MeanKernel<paddle::platform::CPUDeviceContext,
                    paddle::platform::complex128>);
REGISTER_OP_CPU_KERNEL(
    mean_grad, ops::MeanGradKernel<paddle::platform::CPUDeviceContext, float>,
    ops::MeanGradKernel<paddle::platform::CPUDeviceContext, double>,
    ops::MeanGradKernel<paddle::platform::CPUDeviceContext,
                        paddle::platform::complex64>,
    ops::MeanGradKernel<paddle::platform::CPUDeviceContext,
                        paddle::platform::complex128>);
