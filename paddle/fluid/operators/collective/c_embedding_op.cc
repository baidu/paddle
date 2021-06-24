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

#include "paddle/fluid/operators/collective/c_embedding_op.h"

namespace paddle {
namespace operators {

class CEmbeddingOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("W"), "Input", "W", "CEmbeddingOp");
    OP_INOUT_CHECK(ctx->HasInput("Ids"), "Input", "Ids", "CEmbeddingOp");
    OP_INOUT_CHECK(ctx->HasOutput("Out"), "Output", "Out", "CEmbeddingOp");

    auto table_dims = ctx->GetInputDim("W");
    auto ids_dims = ctx->GetInputDim("Ids");
    int ids_rank = ids_dims.size();

    VLOG(5) << "ids rank is " << ids_rank << std::endl;
    PADDLE_ENFORCE_EQ(table_dims.size(), 2,
                      platform::errors::InvalidArgument(
                          "The dimensions of the 'c_embedding' must be 2. "
                          "But received c_embedding's dimensions = %d, "
                          "c_embedding's shape = [%s].",
                          table_dims.size(), table_dims));

    auto output_dims = framework::vectorize(ids_dims);
    output_dims.push_back(table_dims[1]);
    ctx->SetOutputDim("Out", framework::make_ddim(output_dims));

    if (ctx->GetOutputsVarType("Out")[0] ==
        framework::proto::VarType::LOD_TENSOR) {
      ctx->ShareLoD("Ids", /*->*/ "Out");
    }
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    auto data_type = OperatorWithKernel::IndicateVarDataType(ctx, "W");
    return framework::OpKernelType(data_type, ctx.device_context());
  }
};

class CEmbeddingOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("W",
             "(Tensor) The input represents embedding tensors, "
             "which is a learnable parameter.");
    AddInput("Ids",
             "An input with type int64 "
             "contains the ids to be looked up in W.");
    AddOutput("Out", "The lookup results, which have the same type as W.");

    AddAttr<int64_t>("start_index",
                     "(int64, default 0), The starting index is indeed, "
                     "and the out-of-bounds will be set to 0 ")
        .SetDefault(0);
    AddComment(R"DOC(
c_embedding Operator.

This operator is used to perform lookups on the parameter W,
then concatenated into a dense tensor.

The input Ids can carry the LoD (Level of Details) information,
or not. And the output only shares the LoD information with input Ids.

)DOC");
  }
};

DECLARE_NO_NEED_BUFFER_VARS_INFERER(CEmbeddingGradOpNoBufferVarsInferer, "W");

template <typename T>
class CEmbeddingGradOpMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> op) const override {
    op->SetType("c_embedding_grad");

    op->SetInput("W", this->Input("W"));
    op->SetInput("Ids", this->Input("Ids"));
    op->SetInput(framework::GradVarName("Out"), this->OutputGrad("Out"));
    op->SetOutput(framework::GradVarName("W"), this->InputGrad("W"));

    op->SetAttrMap(this->Attrs());
  }
};

class CEmbeddingOpGrad : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    auto table_dims = ctx->GetInputDim("W");
    ctx->SetOutputDim(framework::GradVarName("W"), table_dims);
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    auto data_type = OperatorWithKernel::IndicateVarDataType(
        ctx, framework::GradVarName("Out"));
    return framework::OpKernelType(data_type, ctx.device_context());
  }
};

class CEmbeddingOpGradVarTypeInference : public framework::VarTypeInference {
 public:
  void operator()(framework::InferVarTypeContext* ctx) const override {
    auto out_var_name = framework::GradVarName("W");
    VLOG(3) << "c_embedding_grad op " << framework::GradVarName("W")
            << " is set to LoDTensor";
    ctx->SetOutputType(out_var_name, framework::proto::VarType::LOD_TENSOR);
    ctx->SetOutputDataType(out_var_name, ctx->GetInputDataType("W"));
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(c_embedding, ops::CEmbeddingOp, ops::CEmbeddingOpMaker,
                  ops::CEmbeddingGradOpMaker<paddle::framework::OpDesc>,
                  ops::CEmbeddingGradOpMaker<paddle::imperative::OpBase>);

REGISTER_OPERATOR(c_embedding_grad, ops::CEmbeddingOpGrad,
                  ops::CEmbeddingGradOpNoBufferVarsInferer,
                  ops::CEmbeddingOpGradVarTypeInference);

REGISTER_OP_CPU_KERNEL(c_embedding, ops::CEmbeddingOpCPUKernel<float>,
                       ops::CEmbeddingOpCPUKernel<double>);
