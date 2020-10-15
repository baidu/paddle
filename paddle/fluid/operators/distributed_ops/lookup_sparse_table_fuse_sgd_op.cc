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

#include "paddle/fluid/operators/distributed_ops/lookup_sparse_table_fuse_sgd_op.h"

#include <string>
namespace paddle {
namespace operators {

class LargeScaleFuseSGDOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext *ctx) const override {
    PADDLE_ENFORCE(
        ctx->HasInput("Grad"),
        platform::errors::InvalidArgument(
            "Input(Grad) of LargeScaleFuseSGDOp should not be null."));
    PADDLE_ENFORCE(
        ctx->HasInput("LearningRate"),
        platform::errors::InvalidArgument(
            "Input(LearningRate) of LargeScaleFuseSGDOp should not be null."));

    auto lr_dims = ctx->GetInputDim("LearningRate");

    PADDLE_ENFORCE_NE(framework::product(lr_dims), 0,
                      platform::errors::InvalidArgument(
                          "Maybe the Input variable LearningRate has not "
                          "been initialized. You may need to confirm "
                          "if you put exe.run(startup_program) "
                          "after optimizer.minimize function."));

    PADDLE_ENFORCE_EQ(framework::product(lr_dims), 1,
                      platform::errors::InvalidArgument(
                          "Learning rate should have 1 element"));
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override {
    auto data_type = OperatorWithKernel::IndicateVarDataType(ctx, "Grad");
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

class LargeScaleFuseSGDOpInferVarType : public framework::VarTypeInference {
 public:
  void operator()(framework::InferVarTypeContext *ctx) const override {
    auto in_var_type = ctx->GetInputType("Grad");
    PADDLE_ENFORCE_EQ(in_var_type == framework::proto::VarType::SELECTED_ROWS ||
                          in_var_type == framework::proto::VarType::LOD_TENSOR,
                      true, platform::errors::InvalidArgument(
                                "The input Var's type should be LoDtensor or "
                                "SelectedRows, but the received type is %s",
                                in_var_type));
  }
};

class LargeScaleFuseSGDOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("Grad",
             "(SelectedRows) Ids's type should be SelectedRows"
             "THe ids to be looked up in W.");
    AddInput("LearningRate", "(Tensor) Learning rate of SGD");
    AddAttr<bool>("is_entry",
                  "(bool)"
                  "sparse table need entry");

    AddAttr<std::string>("tablename",
                         "(string)"
                         "sparse table name");

    AddAttr<std::vector<std::string>>("value_names",
                                      "(strings)"
                                      "sparse table name");

    AddComment(R"DOC(

LargeScaleFuseSGD operator

This operator implements one step of the stochastic gradient descent algorithm.

$$param\_out = param - learning\_rate * grad$$

)DOC");
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(
    lookup_sparse_table_fuse_sgd, ops::LargeScaleFuseSGDOp,
    ops::LargeScaleFuseSGDOpMaker,
    paddle::framework::EmptyGradOpMaker<paddle::framework::OpDesc>,
    paddle::framework::EmptyGradOpMaker<paddle::imperative::OpBase>,
    ops::LargeScaleFuseSGDOpInferVarType);

REGISTER_OP_CPU_KERNEL(
    lookup_sparse_table_fuse_sgd,
    ops::LargeScaleFuseSGDOpKernel<paddle::platform::CPUDeviceContext, float>);
