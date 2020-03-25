/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

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

class FusedFCElementwiseLayerNormOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext *ctx) const override {
    PADDLE_ENFORCE_EQ(
        ctx->HasInput("X"), true,
        "Input(X) of fused_fc_elementwise_layernorm should not be null.");
    PADDLE_ENFORCE_EQ(
        ctx->HasInput("W"), true,
        "Input(W) of fused_fc_elementwise_layernorm should not be null.");
    PADDLE_ENFORCE_EQ(
        ctx->HasInput("Y"), true,
        "Input(Y) of fused_fc_elementwise_layernorm should not be null.");
    PADDLE_ENFORCE_EQ(
        ctx->HasOutput("Out"), true,
        "Output(Out) of fused_fc_elementwise_layernorm should not be null.");

    auto w_dims = ctx->GetInputDim("W");
    PADDLE_ENFORCE_EQ(w_dims.size(), 2,
                      "Fully Connected input should be 2-D tensor.");

    if (ctx->HasInput("Bias0")) {
      auto bias0_dims = ctx->GetInputDim("Bias0");
      if (bias0_dims.size() == 2) {
        PADDLE_ENFORCE_EQ(bias0_dims[0], 1,
                          "The shape of Bias must be [1, dim].");
        PADDLE_ENFORCE_EQ(bias0_dims[1], w_dims[1],
                          "The shape of Bias must be [1, dim].");
      } else if (bias0_dims.size() == 1) {
        PADDLE_ENFORCE_EQ(bias0_dims[0], w_dims[1],
                          "The shape of Bias must be [1, dim].");
      }
    }

    auto x_dims = ctx->GetInputDim("X");
    int x_num_col_dims = ctx->Attrs().Get<int>("x_num_col_dims");
    PADDLE_ENFORCE_GT(
        x_dims.size(), x_num_col_dims,
        "The input tensor Input's rank of FCOp should be larger than "
        "in_num_col_dims.");

    auto x_mat_dims = framework::flatten_to_2d(x_dims, x_num_col_dims);
    PADDLE_ENFORCE_EQ(
        x_mat_dims[1], w_dims[0],
        "Fully Connected input and weigth size do not match. %s, %s");

    std::vector<int64_t> fc_out_dims;
    for (int i = 0; i < x_num_col_dims; ++i) {
      fc_out_dims.push_back(x_dims[i]);
    }
    fc_out_dims.push_back(w_dims[1]);

    auto y_dims = ctx->GetInputDim("Y");
    PADDLE_ENFORCE_EQ(framework::make_ddim(fc_out_dims), y_dims);

    auto begin_norm_axis = ctx->Attrs().Get<int>("begin_norm_axis");
    PADDLE_ENFORCE_LT(
        begin_norm_axis, y_dims.size(),
        "'begin_norm_axis' must be less than the rank of Input(Y).");

    auto y_mat_dim = framework::flatten_to_2d(y_dims, begin_norm_axis);
    int64_t dim_0 = y_mat_dim[0];
    int64_t dim_1 = y_mat_dim[1];
    if (ctx->HasInput("Scale")) {
      PADDLE_ENFORCE_EQ(ctx->GetInputDim("Scale").size(), 1);

      if (ctx->IsRuntime()) {
        PADDLE_ENFORCE_EQ(ctx->GetInputDim("Scale")[0], dim_1,
                          "scale should with right");
      }
    }
    if (ctx->HasInput("Bias1")) {
      PADDLE_ENFORCE_EQ(ctx->GetInputDim("Bias1").size(), 1);
      if (ctx->IsRuntime()) {
        PADDLE_ENFORCE_EQ(ctx->GetInputDim("Bias1")[0], dim_1,
                          "bias should with right");
      }
    }

    ctx->SetOutputDim("Out", y_dims);
    if (ctx->HasOutput("Mean")) {
      ctx->SetOutputDim("Mean", {dim_0});
    }
    if (ctx->HasOutput("Variance")) {
      ctx->SetOutputDim("Variance", {dim_0});
    }
    ctx->ShareLoD("X", "Out");
  }
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const {
    auto input_data_type = OperatorWithKernel::IndicateVarDataType(ctx, "X");
    return framework::OpKernelType(input_data_type, ctx.GetPlace());
  }
};

class FusedFCElementwiseLayerNormOpMaker
    : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "(Tensor), The input tensor of fully connected operation");
    AddInput("W",
             "(Tensor), The weight tensor of fully connected operation. It is "
             "a 2-D Tensor with shape (I, O)");
    AddInput("Bias0",
             "(Tensor, optional), The bias tensor of fully connecred "
             "operation. It is a 1-D Tensor with shape (O), or a 2-D Tensor "
             "with shape (1, O).")
        .AsDispensable();
    AddInput("Y",
             "(Tensor), The second input tensor of elementwise_add operation. "
             "Note that the shape should be the same as fully connect's result "
             "tensor.");
    AddInput(
        "Scale",
        "(Tensor, optional), It is a 1-D input Tensor of layer_norm operation.")
        .AsDispensable();
    AddInput(
        "Bias1",
        "(Tensor, optional), It is a 1-D input Tensor of layer_norm operation.")
        .AsDispensable();
    AddOutput("Out",
              "(Tensor), Output after normalization. The shape is the shame as "
              "layer_norm's input.");
    AddOutput("Mean", "(Tensor, optional), Mean of the current minibatch")
        .AsDispensable();
    AddOutput("Variance",
              "(Tensor, optional), Variance of the current minibatch")
        .AsDispensable();
    AddAttr<int>("x_num_col_dims",
                 "(int, default 1), This op can take tensors with more than "
                 "two dimensions as its inputs.")
        .SetDefault(1)
        .EqualGreaterThan(1);
    AddAttr<std::string>("activation_type",
                         "Activation type used in fully connected operator.")
        .SetDefault("");
    AddAttr<float>("epsilon",
                   "Constant for numerical stability [default 1e-5].")
        .SetDefault(1e-5)
        .AddCustomChecker([](const float &epsilon) {
          PADDLE_ENFORCE_GE(epsilon, 0.0f,
                            "'epsilon' should be between 0.0 and 0.001.");
          PADDLE_ENFORCE_LE(epsilon, 0.001f,
                            "'epsilon' should be between 0.0 and 0.001.");
        });
    AddAttr<int>("begin_norm_axis",
                 "the axis of `begin_norm_axis ... Rank(Y) - 1` will be "
                 "normalized. `begin_norm_axis` splits the tensor(`X`) to a "
                 "matrix [N,H]. [default 1].")
        .SetDefault(1)
        .AddCustomChecker([](const int &begin_norm_axis) {
          PADDLE_ENFORCE_GT(begin_norm_axis, 0,
                            "'begin_norm_axis' should be greater than zero.");
        });
    AddComment(R"DOC(
fc_out <= fc(X, W, Bias0)
add_out <= elementwise_add(fc_out, Y)
(out, mean, variance) <= layer_norm(add_out, Scale, Bias1)
)DOC");
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(
    fused_fc_elementwise_layernorm, ops::FusedFCElementwiseLayerNormOp,
    ops::FusedFCElementwiseLayerNormOpMaker,
    paddle::framework::EmptyGradOpMaker<paddle::framework::OpDesc>,
    paddle::framework::EmptyGradOpMaker<paddle::imperative::OpBase>);
