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

#include <string>

#include "paddle/fluid/operators/optimizers/sgd_op.h"

namespace paddle {
namespace operators {

class SGDOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext *ctx) const override {
    PADDLE_ENFORCE_EQ(ctx->HasInput("Param"), true,
                      platform::errors::NotFound(
                          "Input(Param) of SGDOp should not be null."));
    PADDLE_ENFORCE_EQ(
        ctx->HasInput("Grad"), true,
        platform::errors::NotFound("Input(Grad) of SGDOp should not be null."));
    PADDLE_ENFORCE_EQ(ctx->HasInput("LearningRate"), true,
                      platform::errors::NotFound(
                          "Input(LearningRate) of SGDOp should not be null."));
    PADDLE_ENFORCE_EQ(ctx->HasOutput("ParamOut"), true,
                      platform::errors::NotFound(
                          "Output(ParamOut) of SGDOp should not be null."));

    auto lr_dims = ctx->GetInputDim("LearningRate");
    PADDLE_ENFORCE_NE(framework::product(lr_dims), 0,
                      platform::errors::NotFound(
                          "Maybe the Input variable LearningRate has not "
                          "been initialized. You may need to confirm "
                          "if you put exe.run(startup_program) "
                          "after optimizer.minimize function."));
    PADDLE_ENFORCE_EQ(framework::product(lr_dims), 1,
                      platform::errors::InvalidArgument(
                          "Learning rate should have 1 element. But received "
                          "LearningRate dims [%s]",
                          framework::product(lr_dims)));
    auto param_dim = ctx->GetInputDim("Param");
    if (ctx->GetInputsVarType("Grad")[0] ==
        framework::proto::VarType::LOD_TENSOR) {
      PADDLE_ENFORCE_EQ(
          param_dim, ctx->GetInputDim("Grad"),
          platform::errors::InvalidArgument(
              "SGD Operator's input Param and Grad dimensions do not match. "
              "The Param %s shape is [%s], but the Grad %s shape is [%s].",
              ctx->Inputs("Param")[0], param_dim, ctx->Inputs("Grad")[0],
              ctx->GetInputDim("Grad")));
    }
    ctx->SetOutputDim("ParamOut", param_dim);
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override {
    auto data_type = OperatorWithKernel::IndicateVarDataType(ctx, "Param");
    return framework::OpKernelType(data_type, ctx.device_context());
  }

  framework::OpKernelType GetKernelTypeForVar(
      const std::string &var_name, const framework::Tensor &tensor,
      const framework::OpKernelType &expected_kernel_type) const {
    if (var_name == "LearningRate") {
      return framework::OpKernelType(tensor.type(), tensor.place(),
                                     tensor.layout());
    }
    return framework::OpKernelType(expected_kernel_type.data_type_,
                                   tensor.place(), tensor.layout());
  }
};

class SGDOpInferVarType : public framework::VarTypeInference {
 public:
  void operator()(framework::InferVarTypeContext *ctx) const override {
    auto in_var_type = ctx->GetInputType("Param");
    PADDLE_ENFORCE_EQ(in_var_type == framework::proto::VarType::SELECTED_ROWS ||
                          in_var_type == framework::proto::VarType::LOD_TENSOR,
                      true, platform::errors::InvalidArgument(
                                "The input Var's type should be LoDtensor or "
                                "SelectedRows, but the received type is %s",
                                in_var_type));

    ctx->SetOutputType("ParamOut", in_var_type, framework::ALL_ELEMENTS);
  }
};

class SGDOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("Param", "(Tensor or SelectedRows) Input parameter");
    AddInput("LearningRate", "(Tensor) Learning rate of SGD");
    AddInput("Grad", "(Tensor or SelectedRows) Input gradient");
    AddInput("MasterParam", "FP32 master weight for AMP.").AsDispensable();
    AddOutput("ParamOut",
              "(Tensor or SelectedRows, same with Param) "
              "Output parameter, should share the same memory with Param");
    AddOutput("MasterParamOut",
              "The updated FP32 master weight for AMP. "
              "It shared memory with Input(MasterParam).")
        .AsDispensable();
    AddAttr<bool>("multi_precision",
                  "(bool, default false) "
                  "Whether to use multi-precision during weight updating.")
        .SetDefault(false);
    AddAttr<float>(
        "rescale_grad",
        "(float, default 1.0) Multiply the gradient with `rescale_grad`"
        "before updating. Often choose to be `1.0/batch_size`.")
        .SetDefault(1.0f);
    AddComment(R"DOC(

SGD operator

This operator implements one step of the stochastic gradient descent algorithm.

$$param\_out = param - learning\_rate * grad$$

)DOC");
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(
    sgd, ops::SGDOp, ops::SGDOpMaker,
    paddle::framework::EmptyGradOpMaker<paddle::framework::OpDesc>,
    paddle::framework::EmptyGradOpMaker<paddle::imperative::OpBase>,
    ops::SGDOpInferVarType);
REGISTER_OP_CPU_KERNEL(
    sgd, ops::SGDOpKernel<paddle::platform::CPUDeviceContext, float>,
    ops::SGDOpKernel<paddle::platform::CPUDeviceContext,
                     paddle::platform::bfloat16>,
    ops::SGDOpKernel<paddle::platform::CPUDeviceContext, double>);
