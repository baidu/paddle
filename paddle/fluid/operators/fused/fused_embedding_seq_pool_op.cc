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

#include "paddle/fluid/operators/fused/fused_embedding_seq_pool_op.h"

#include <memory>

#include "paddle/fluid/framework/var_type_inference.h"

namespace paddle {
namespace operators {

class FusedEmbeddingSeqPoolOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("W"), "Input", "W", "FusedEmbeddingSeqPool");
    OP_INOUT_CHECK(ctx->HasInput("Ids"), "Input", "Ids",
                   "FusedEmbeddingSeqPool");
    OP_INOUT_CHECK(ctx->HasOutput("Out"), "Output", "Out",
                   "FusedEmbeddingSeqPool");
    auto table_dims = ctx->GetInputDim("W");
    auto ids_dims = ctx->GetInputDim("Ids");
    const auto& combiner = ctx->Attrs().Get<std::string>("combiner");

    PADDLE_ENFORCE_EQ(table_dims.size(), 2,
                      platform::errors::InvalidArgument(
                          "The dim size of the input tensor 'W' should be 2. "
                          "But received W's size = %d.",
                          table_dims.size()));
    PADDLE_ENFORCE_EQ(
        ids_dims[ids_dims.size() - 1], 1,
        platform::errors::InvalidArgument(
            "The last dimension of the input tensor 'Ids' should be 1. "
            "But received Ids's size in the last dimension = %d.",
            ids_dims[ids_dims.size() - 1]));
    // we only support sum now
    PADDLE_ENFORCE_EQ(combiner, "sum",
                      platform::errors::Unimplemented(
                          "The pooling type of sequence_pool only support sum "
                          "now. So the 'combiner' must be 'sum'."));

    int64_t last_dim = FusedEmbeddingSeqPoolLastDim(table_dims, ids_dims);
    // in compile time, the lod level of ids must be 1
    framework::VarDesc* ids_desc =
        BOOST_GET(framework::VarDesc*, ctx->GetInputVarPtrs("Ids")[0]);
    PADDLE_ENFORCE_EQ(ids_desc->GetLoDLevel(), 1,
                      platform::errors::InvalidArgument(
                          "In compile time, the LoD Level of Ids should be 1. "
                          "But received the LoD Level of Ids = %d.",
                          ids_desc->GetLoDLevel()));

    // in compile time, the shape from Ids -> output
    // should be [-1, 1] -> [-1, embedding_size]
    ctx->SetOutputDim("Out", framework::make_ddim({-1, last_dim}));
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    auto data_type = OperatorWithKernel::IndicateVarDataType(ctx, "W");
    return framework::OpKernelType(data_type, ctx.device_context());
  }
};

class FusedEmbeddingSeqPoolOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("W",
             "(Tensor) The input represents embedding tensors, "
             "which is a learnable parameter.");
    AddInput("Ids",
             "An input with type int32 or int64 "
             "contains the ids to be looked up in W. "
             "The last dimension size must be 1.");
    AddOutput("Out", "The lookup results, which have the same type as W.");
    AddAttr<std::string>("combiner",
                         "(string, default sum) "
                         "A string specifying the reduction op. Currently sum "
                         "are supported, sum computes the weighted sum of the "
                         "embedding results for each row.")
        .SetDefault("sum");
    AddAttr<int64_t>("padding_idx",
                     "(int64, default -1) "
                     "If the value is -1, it makes no effect to lookup. "
                     "Otherwise the given value indicates padding the output "
                     "with zeros whenever lookup encounters it in Ids.")
        .SetDefault(kNoPadding);
    // NOTE(minqiyang): grad_inplace is an temporal attribute,
    // please do NOT set this attribute in python layer.
    AddAttr<bool>("grad_inplace",
                  "(boolean, default false) "
                  "If the grad op reuse the input's variable.")
        .SetDefault(false);
    AddAttr<bool>("is_sparse",
                  "(boolean, default false) "
                  "Sparse update.")
        .SetDefault(false);
    AddAttr<bool>(framework::kAllKernelsMustComputeRuntimeShape,
                  "Skip calling InferShape() function in the runtime.")
        .SetDefault(true);
    AddComment(R"DOC(
FusedEmbeddingSeqPool Operator.

Computes embeddings for the given ids and weights.

This operator is used to perform lookups on the parameter W,
then computes the weighted sum of the lookups results for each row
and concatenated into a dense tensor.

The input Ids should carry the LoD (Level of Details) information.
And the output will change the LoD information with input Ids.

)DOC");
  }
};

class FusedEmbeddingSeqPoolOpGrad : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    auto table_dims = ctx->GetInputDim("W");
    ctx->SetOutputDim(framework::GradVarName("W"), table_dims);
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    auto data_type = OperatorWithKernel::IndicateVarDataType(ctx, "W");
    return framework::OpKernelType(data_type, ctx.device_context());
  }
};

class FusedEmbeddingSeqPoolOpGradVarTypeInference
    : public framework::VarTypeInference {
 public:
  void operator()(framework::InferVarTypeContext* ctx) const override {
    auto out_var_name = framework::GradVarName("W");
    auto attr = ctx->GetAttr("is_sparse");
    bool is_sparse = BOOST_GET(bool, attr);
    if (is_sparse) {
      VLOG(3) << "fused_embedding_seq_pool_grad op "
              << framework::GradVarName("W") << " is set to SelectedRows";
      ctx->SetOutputType(out_var_name,
                         framework::proto::VarType::SELECTED_ROWS);
    } else {
      VLOG(3) << "fused_embedding_seq_pool_grad op "
              << framework::GradVarName("W") << " is set to LoDTensor";
      ctx->SetOutputType(out_var_name, framework::proto::VarType::LOD_TENSOR);
    }
    ctx->SetOutputDataType(out_var_name, ctx->GetInputDataType("W"));
  }
};

template <typename T>
class FusedEmbeddingSeqPoolGradOpMaker
    : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> op) const override {
    op->SetType("fused_embedding_seq_pool_grad");
    op->SetInput("Ids", this->Input("Ids"));
    op->SetInput("W", this->Input("W"));
    op->SetInput(framework::GradVarName("Out"), this->OutputGrad("Out"));
    op->SetOutput(framework::GradVarName("W"), this->InputGrad("W"));
    op->SetAttrMap(this->Attrs());
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OPERATOR(
    fused_embedding_seq_pool, ops::FusedEmbeddingSeqPoolOp,
    ops::FusedEmbeddingSeqPoolGradOpMaker<paddle::framework::OpDesc>,
    ops::FusedEmbeddingSeqPoolGradOpMaker<paddle::imperative::OpBase>,
    ops::FusedEmbeddingSeqPoolOpMaker);
REGISTER_OPERATOR(fused_embedding_seq_pool_grad,
                  ops::FusedEmbeddingSeqPoolOpGrad,
                  ops::FusedEmbeddingSeqPoolOpGradVarTypeInference);

REGISTER_OP_CPU_KERNEL(fused_embedding_seq_pool,
                       ops::FusedEmbeddingSeqPoolKernel<float>,
                       ops::FusedEmbeddingSeqPoolKernel<double>);
REGISTER_OP_CPU_KERNEL(fused_embedding_seq_pool_grad,
                       ops::FusedEmbeddingSeqPoolGradKernel<float>,
                       ops::FusedEmbeddingSeqPoolGradKernel<double>);
