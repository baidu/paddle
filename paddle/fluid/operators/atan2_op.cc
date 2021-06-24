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

#include "paddle/fluid/operators/atan2_op.h"

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace paddle {
namespace operators {

class Atan2Op : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("X1"), "Input", "X1", "atan2");
    OP_INOUT_CHECK(ctx->HasInput("X2"), "Input", "X2", "atan2");
    OP_INOUT_CHECK(ctx->HasOutput("Out"), "Output", "Out", "atan2");

    auto in_dims = ctx->GetInputDim("X1");

    ctx->SetOutputDim("Out", in_dims);
  }
};

class Atan2OpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X1", "(Tensor), The input tensor of atan2 op.");
    AddInput("X2", "(Tensor), The input tensor of atan2 op.");
    AddOutput("Out", "(Tensor), The output tensor of atan2 op.");
    AddComment(R"DOC(
Atan2 Operator.

This operator is used to perform elementwise atan2 for input $X1$, $X2$.
$$out = atan2(x1, x2)$$

)DOC");
  }
};

class Atan2GradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;
  void InferShape(framework::InferShapeContext* ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("X1"), "Input", "X1", "Atan2Grad");
    OP_INOUT_CHECK(ctx->HasInput("X2"), "Input", "X2", "Atan2Grad");
    OP_INOUT_CHECK(ctx->HasInput(framework::GradVarName("Out")), "Input",
                   "Out@Grad", "Atan2Grad");

    auto x1_grad_name = framework::GradVarName("X1");
    auto x2_grad_name = framework::GradVarName("X2");
    auto dout_dims = ctx->GetInputDim(framework::GradVarName("Out"));

    if (ctx->HasOutput(x1_grad_name)) {
      ctx->SetOutputDim(framework::GradVarName("X1"), dout_dims);
    }
    if (ctx->HasOutput(x2_grad_name)) {
      ctx->SetOutputDim(framework::GradVarName("X2"), dout_dims);
    }
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    auto dtype = OperatorWithKernel::IndicateVarDataType(ctx, "X1");
    return framework::OpKernelType(dtype, ctx.GetPlace());
  }
};

template <typename T>
class Atan2GradMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

  void Apply(GradOpPtr<T> retv) const override {
    retv->SetType("atan2_grad");
    retv->SetInput("X1", this->Input("X1"));
    retv->SetInput("X2", this->Input("X2"));
    retv->SetInput(framework::GradVarName("Out"), this->OutputGrad("Out"));
    retv->SetAttrMap(this->Attrs());
    retv->SetOutput(framework::GradVarName("X1"), this->InputGrad("X1"));
    retv->SetOutput(framework::GradVarName("X2"), this->InputGrad("X2"));
  }
};

class Atan2OpVarTypeInference : public framework::VarTypeInference {
 public:
  void operator()(framework::InferVarTypeContext* ctx) const override {
    auto type = ctx->GetInputDataType("X1");
    if (ctx->GetInputDataType("X1") == framework::proto::VarType::INT32 ||
        ctx->GetInputDataType("X1") == framework::proto::VarType::INT64 ||
        ctx->GetInputDataType("X2") == framework::proto::VarType::INT32 ||
        ctx->GetInputDataType("X2") == framework::proto::VarType::INT64) {
      type = framework::proto::VarType::FP64;
    }
    ctx->SetOutputDataType("Out", type);
  }
};
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OPERATOR(atan2, ops::Atan2Op, ops::Atan2OpMaker,
                  ops::Atan2GradMaker<paddle::framework::OpDesc>,
                  ops::Atan2GradMaker<paddle::imperative::OpBase>,
                  ops::Atan2OpVarTypeInference);

REGISTER_OPERATOR(atan2_grad, ops::Atan2GradOp);

REGISTER_OP_CPU_KERNEL(
    atan2, ops::Atan2Kernel<paddle::platform::CPUDeviceContext, int32_t>,
    ops::Atan2Kernel<paddle::platform::CPUDeviceContext, int64_t>,
    ops::Atan2Kernel<paddle::platform::CPUDeviceContext, float>,
    ops::Atan2Kernel<paddle::platform::CPUDeviceContext, double>,
    ops::Atan2Kernel<paddle::platform::CPUDeviceContext,
                     paddle::platform::float16>);

REGISTER_OP_CPU_KERNEL(
    atan2_grad, ops::Atan2GradKernel<paddle::platform::CPUDeviceContext, float>,
    ops::Atan2GradKernel<paddle::platform::CPUDeviceContext, double>,
    ops::Atan2GradKernel<paddle::platform::CPUDeviceContext,
                         paddle::platform::float16>);
