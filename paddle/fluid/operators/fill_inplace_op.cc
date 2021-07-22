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

#include "paddle/fluid/framework/op_registry.h"

namespace paddle {
namespace operators {

class FillInplaceOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddComment(R"DOC(Fill replace operator
                Fill an tensor inplace with `value` and `shape`. The type of the tensor is specify by
                `dtype`.
                )DOC");
    AddInput("X", "(Tensor) The input tensor.");
    AddOutput("Out",
              "Tensor, the clipped tensor, with the same shape and data type "
              "as input(x)");
    AddInput(
        "value",
        "The float values of tensor, whose dim is one, and no need of grad");
  }
};

class FillInplaceOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext *context) const override {
    OP_INOUT_CHECK(context->HasInput("X"), "Input", "X", "FillInplace");
    OP_INOUT_CHECK(context->HasOutput("Out"), "Output", "Out", "FillInplace");
    auto x_dims = context->GetInputDim("X");
    context->SetOutputDim("Out", x_dims);
    // context->ShareLoD("X", /*->*/ "Out");
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override {
    return framework::OpKernelType(
        OperatorWithKernel::IndicateVarDataType(ctx, "X"), ctx.GetPlace());
  }
};

class FillInplaceOpVarTypeInference : public framework::VarTypeInference {
 public:
  void operator()(framework::InferVarTypeContext *ctx) const override {}
};

template <typename T>
class FillInplaceKernel : public framework::OpKernel<T> {
 public:
  void Compute(const paddle::framework::ExecutionContext &ctx) const override {
    auto *in = ctx.Input<framework::Tensor>("X");
    auto *value = ctx.Input<framework::Tensor>("value");
    auto *out = ctx.Output<framework::Tensor>("Out");
    T *out_data = out->mutable_data<T>(ctx.GetPlace());
    auto fill_val = *(value->data<T>());
    std::fill(out_data, out_data + in->numel(), fill_val);
  }
};

class FillInplaceGradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext *ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput(framework::GradVarName("Out")), "Input",
                   "Out@GRAD", "mul");
    auto x_dims = ctx->GetInputDim(framework::GradVarName("Out"));
    auto x_grad_name = framework::GradVarName("X");
    if (ctx->HasOutput(x_grad_name)) {
      ctx->SetOutputDim(x_grad_name, x_dims);
    }
  }
};

template <typename T>
class FillInplaceGradOpMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> retv) const override {
    retv->SetType(this->ForwardOpType() + "_grad");
    retv->SetInput(framework::GradVarName("Out"), this->OutputGrad("Out"));
    retv->SetOutput(framework::GradVarName("X"), this->InputGrad("X"));
  }
};

template <typename T>
class FillInplaceGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const paddle::framework::ExecutionContext &ctx) const override {
    auto *dx = ctx.Output<framework::Tensor>(framework::GradVarName("X"));
    if (dx) {
      auto *data = dx->mutable_data<T>(ctx.GetPlace());
      std::fill(data, data + dx->numel(), T(0));
    }
  }
};

DECLARE_INPLACE_OP_INFERER(FillInplaceOpInplaceInferer, {"X", "Out"});
DECLARE_INPLACE_OP_INFERER(FillInplaceGradInplaceInferer,
                           {framework::GradVarName("Out"),
                            framework::GradVarName("X")});
}  // namespace operators
}  // namespace paddle
namespace ops = paddle::operators;

REGISTER_OPERATOR(fill_inplace, ops::FillInplaceOp, ops::FillInplaceOpMaker,
                  ops::FillInplaceOpVarTypeInference,
                  ops::FillInplaceGradOpMaker<paddle::framework::OpDesc>,
                  ops::FillInplaceGradOpMaker<paddle::imperative::OpBase>,
                  ops::FillInplaceOpInplaceInferer);

REGISTER_OPERATOR(fill_inplace_grad, ops::FillInplaceGradOp,
                  ops::FillInplaceGradInplaceInferer);

REGISTER_OP_CPU_KERNEL(fill_inplace, ops::FillInplaceKernel<float>,
                       ops::FillInplaceKernel<double>,
                       ops::FillInplaceKernel<int64_t>,
                       ops::FillInplaceKernel<int>,
                       ops::FillInplaceKernel<paddle::platform::float16>,
                       ops::FillInplaceKernel<bool>);

REGISTER_OP_CPU_KERNEL(fill_inplace_grad, ops::FillInplaceGradKernel<float>,
                       ops::FillInplaceGradKernel<double>,
                       ops::FillInplaceGradKernel<int64_t>,
                       ops::FillInplaceGradKernel<int>,
                       ops::FillInplaceGradKernel<paddle::platform::float16>,
                       ops::FillInplaceGradKernel<bool>);
