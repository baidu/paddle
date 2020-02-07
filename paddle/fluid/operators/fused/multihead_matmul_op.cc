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

#include <vector>
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/detail/safe_ref.h"
#include "paddle/fluid/platform/errors.h"

namespace paddle {
namespace operators {

class MultiHeadMatMulV2Op : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(framework::InferShapeContext *context) const override {
    PADDLE_ENFORCE_EQ(
        context->HasInput("Input"), true,
        platform::errors::InvalidArgument(
            "Input(Input) of MultiHeadMatMul should not be null."));
    PADDLE_ENFORCE_EQ(context->HasInput("W"), true,
                      platform::errors::InvalidArgument(
                          "Input(W) of MultiHeadMatMul should not be null."));
    PADDLE_ENFORCE_EQ(
        context->HasInput("Bias"), true,
        platform::errors::InvalidArgument(
            "Input(Bias) of MultiHeadMatMul should not be null."));
    PADDLE_ENFORCE_EQ(
        context->HasInput("BiasQK"), true,
        platform::errors::InvalidArgument(
            "Input(BiasQK) of MultiHeadMatMul should not be null."));
    PADDLE_ENFORCE_EQ(
        context->HasOutput("Out"), true,
        platform::errors::InvalidArgument(
            "Output(Out) of MultiHeadMatMul should not be null."));

    auto dim_w = context->GetInputDim("W");
    PADDLE_ENFORCE_GT(
        dim_w.size(), 2,
        platform::errors::InvalidArgument(
            "Multihead input is expected at least a 3-D tensor, but "
            "it's %d-D tensor now.",
            dim_w.size()));

    auto dim_bias_q = context->GetInputDim("Bias");
    PADDLE_ENFORCE_GT(
        dim_bias_q.size(), 1,
        platform::errors::InvalidArgument(
            "Multihead input should be at least 2-D tensor, but it's "
            "%d-D tensor now.",
            dim_bias_q.size()));

    auto dim_bias_qk = context->GetInputDim("BiasQK");
    PADDLE_ENFORCE_GT(
        dim_bias_qk.size(), 3,
        platform::errors::InvalidArgument(
            "Multihead input bias qk should be at least 4-D tensor, "
            "but it's %d-D tensor now.",
            dim_bias_qk.size()));

    int b_indx = dim_bias_q.size() - 1;
    int indx = dim_q.size() - 1;

    PADDLE_ENFORCE_EQ(
        dim_bias_q[b_indx], dim_q[indx],
        platform::errors::InvalidArgument(
            "bias_q's last dim size should equal to"
            " q last dim size, but received bias_q's size is:%d q is:%d",
            dim_bias_q[b_indx], dim_q[indx]));
    PADDLE_ENFORCE_EQ(
        dim_bias_k[b_indx], dim_k[indx],
        platform::errors::InvalidArgument(
            "bias_k's last dim size should equal to"
            " k last dim size, but received bias_k's size is:%d k is:%d",
            dim_bias_k[b_indx], dim_k[indx]));
    PADDLE_ENFORCE_EQ(
        dim_bias_v[b_indx], dim_v[indx],
        platform::errors::InvalidArgument(
            "bias_v's last dim size should equal to"
            " v last dim size, but received bias_v's size is:%d v is:%d",
            dim_bias_v[b_indx], dim_v[indx]));

    PADDLE_ENFORCE_EQ(dim_q[0], dim_bias_qk[0],
                      platform::errors::InvalidArgument(
                          "q should have same batch size"
                          "with bias_qk, but received q's batch size is:%d "
                          "bias_qk's batch size is:%d",
                          dim_q[0], dim_bias_qk[0]));

    int head_number = context->Attrs().Get<int>("head_number");
    PADDLE_ENFORCE_GT(
        head_number, 1,
        platform::errors::InvalidArgument(
            "Multihead input head number should be at least 1, but it %d now.",
            head_number));
    // modify this
    auto dim_input = context->GetInputDim("Input");
    context->SetOutputDim("Out", dim_input);
    context->ShareLoD("Input", /*->*/ "Out");
  }
};

class MultiHeadMatMulV2OpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("Input", "The input of MultiHeadMatMul op");
    AddInput("W", "The weight input of MultiHeadMatMul op");
    AddInput("Bias", "The bias input of MultiHeadMatMul op");
    AddInput("BiasQK", "The QK bias input of MultiHeadMatMul op");
    AddOutput("Out", "The output of MultiHeadMatMul op");
    AddAttr<bool>("transpose_Q",
                  R"DOC(If true, use the transpose of `Q`.
        )DOC")
        .SetDefault(false);
    AddAttr<bool>("transpose_K",
                  R"DOC(If true, use the transpose of `K`.
        )DOC")
        .SetDefault(true);
    AddAttr<bool>("transpose_V",
                  R"DOC(If true, use the transpose of `V`.
        )DOC")
        .SetDefault(false);
    AddAttr<float>("alpha", "The scale of Out").SetDefault(1.0f);
    AddAttr<int>("head_number", "The number of heads of the matrix")
        .SetDefault(1);
    AddComment(R"DOC(
MultiHeadMatMul Operator.

This op is used for optimize multi head calculation in ernie model.
Not suggest to use in other case except has same structure as ernie.

Example of matrix multiplication with head_number of B
- X: [B, M, K], Y: [B, K, N] => Out: [B, M, N]

)DOC");
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_WITHOUT_GRADIENT(multihead_matmul, ops::MultiHeadMatMulV2Op,
                             ops::MultiHeadMatMulV2OpMaker);
