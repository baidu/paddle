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
    OP_INOUT_CHECK(ctx->HasInput("X"), "Input", "X",
                   "FusedFcElementwiseLayernorm");
    OP_INOUT_CHECK(ctx->HasInput("W"), "Input", "W",
                   "FusedFcElementwiseLayernorm");
    OP_INOUT_CHECK(ctx->HasInput("Y"), "Input", "Y",
                   "FusedFcElementwiseLayernorm");
    OP_INOUT_CHECK(ctx->HasOutput("Out"), "Output", "Out",
                   "FusedFcElementwiseLayernorm");

    auto w_dims = ctx->GetInputDim("W");
    PADDLE_ENFORCE_EQ(
        w_dims.size(), 2,
        platform::errors::InvalidArgument(
            "The input Weight of fc is expected to be a 2-D tensor. "
            "But received the number of Weight's dimensions is %d, ",
            "Weight's shape is %s.", w_dims.size(), w_dims));

    if (ctx->HasInput("Bias0")) {
      auto bias0_dims = ctx->GetInputDim("Bias0");

      PADDLE_ENFORCE_LE(bias0_dims.size(), 2,
                        platform::errors::InvalidArgument(
                            "The input Bias of fc is expected to be an 1-D or "
                            "2-D tensor. But received the number of Bias's "
                            "dimensions is %d, Bias's shape is %s.",
                            bias0_dims.size(), bias0_dims));

      PADDLE_ENFORCE_EQ(
          bias0_dims[bias0_dims.size() - 1], w_dims[1],
          platform::errors::InvalidArgument(
              "The last dimension of input Bias is expected be equal "
              "to the actual width of input Weight. But received the last "
              "dimension of Bias is %d, Bias's shape is %s; "
              "the actual width of Weight is %d, Weight's shape is %s.",
              bias0_dims[bias0_dims.size() - 1], bias0_dims, w_dims[1],
              w_dims));

      if (bias0_dims.size() == 2) {
        PADDLE_ENFORCE_EQ(
            bias0_dims[0], 1,
            platform::errors::InvalidArgument(
                "The first dimension of input Bias is expected to be 1, "
                "but received %d, Bias's shape is %s.",
                bias0_dims[0], bias0_dims));
      }
    }

    auto x_dims = ctx->GetInputDim("X");
    int x_num_col_dims = ctx->Attrs().Get<int>("x_num_col_dims");
    PADDLE_ENFORCE_LT(
        x_num_col_dims, x_dims.size(),
        platform::errors::InvalidArgument(
            "The attribute x_num_col_dims used to flatten input X to "
            "a 2-D tensor, is expected to be less than the number of "
            "input X's dimensions. But recieved x_num_col_dims is %d, "
            "the number of input X's dimensions is %d, input X's shape is %s.",
            x_num_col_dims, x_dims.size(), x_dims));

    auto x_mat_dims = framework::flatten_to_2d(x_dims, x_num_col_dims);
    PADDLE_ENFORCE_EQ(
        x_mat_dims[1], w_dims[0],
        platform::errors::InvalidArgument(
            "The input's second dimension and weight's first dimension is "
            "expected to be the same. But recieved input's second dimension is "
            "%d, input's shape is %s; weight's first dimension is %d, weight's "
            "shape is %s.",
            x_mat_dims[1], x_mat_dims, w_dims[0], w_dims));

    std::vector<int64_t> fc_out_dims;
    for (int i = 0; i < x_num_col_dims; ++i) {
      fc_out_dims.push_back(x_dims[i]);
    }
    fc_out_dims.push_back(w_dims[1]);

    auto y_dims = ctx->GetInputDim("Y");
    PADDLE_ENFORCE_EQ(framework::make_ddim(fc_out_dims), y_dims,
                      platform::errors::InvalidArgument(
                          "The output's shape of fc is expected to be equal to "
                          "that of input Y. But recieved output's shape of fc "
                          "is %s, input Y's shape is %s.",
                          framework::make_ddim(fc_out_dims), y_dims));

    auto begin_norm_axis = ctx->Attrs().Get<int>("begin_norm_axis");
    PADDLE_ENFORCE_LT(
        begin_norm_axis, y_dims.size(),
        platform::errors::InvalidArgument(
            "The attribute begin_norm_axis used to flatten input Y to a 2-D "
            "tensor, is expected to be less than the number of input Y's "
            "dimensions. But recieved begin_norm_axis is %d, the number of "
            "input Y's dimensions is %d, input Y's shape is %s.",
            begin_norm_axis, y_dims.size(), y_dims));

    auto y_mat_dim = framework::flatten_to_2d(y_dims, begin_norm_axis);
    int64_t dim_0 = y_mat_dim[0];
    int64_t dim_1 = y_mat_dim[1];
    if (ctx->HasInput("Scale")) {
      auto scale_dims = ctx->GetInputDim("Scale");
      PADDLE_ENFORCE_EQ(scale_dims.size(), 1,
                        platform::errors::InvalidArgument(
                            "The input Scale is expected to be an 1-D tensor. "
                            "But recieved the number of input Scale's "
                            "dimensions is %d, input Scale's shape is %s.",
                            scale_dims.size(), scale_dims));

      if (ctx->IsRuntime()) {
        PADDLE_ENFORCE_EQ(
            scale_dims[0], dim_1,
            platform::errors::InvalidArgument(
                "The first dimension of input Scale is expected to be equal to "
                "the second dimension of input Y after flattened. "
                "But recieved the first dimension of input Scale is %d, input "
                "Scale's shape is %s; the second dimension of flattened input "
                "Y is %d, input Y's shape is %s, flattened axis is %d.",
                scale_dims[0], scale_dims, dim_1, y_dims, begin_norm_axis));
      }
    }
    if (ctx->HasInput("Bias1")) {
      auto bias1_dims = ctx->GetInputDim("Bias1");
      PADDLE_ENFORCE_EQ(
          bias1_dims.size(), 1,
          platform::errors::InvalidArgument(
              "The input Bias1 is expected to be an 1-D tensor. "
              "But recieved the number of input Bias1's dimension is %d, "
              "input Bias1's shape is %s.",
              bias1_dims.size(), bias1_dims));

      if (ctx->IsRuntime()) {
        PADDLE_ENFORCE_EQ(
            bias1_dims[0], dim_1,
            platform::errors::InvalidArgument(
                "The first dimension of input Bias1 is expected to be equal to "
                "the second dimension of input Y after flattened. "
                "But recieved the first dimension of input Bias1 is %d, input "
                "Bias1's shape is %s; the second dimension of flatten input "
                "Y is %d, input Y's shape is %s, flattened axis is %d.",
                bias1_dims[0], bias1_dims, dim_1, y_dims, begin_norm_axis));
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
                            platform::errors::InvalidArgument(
                                "'epsilon' should be between 0.0 and 0.001."));
          PADDLE_ENFORCE_LE(epsilon, 0.001f,
                            platform::errors::InvalidArgument(
                                "'epsilon' should be between 0.0 and 0.001."));
        });
    AddAttr<int>("begin_norm_axis",
                 "the axis of `begin_norm_axis ... Rank(Y) - 1` will be "
                 "normalized. `begin_norm_axis` splits the tensor(`X`) to a "
                 "matrix [N,H]. [default 1].")
        .SetDefault(1)
        .AddCustomChecker([](const int &begin_norm_axis) {
          PADDLE_ENFORCE_GT(
              begin_norm_axis, 0,
              platform::errors::InvalidArgument(
                  "'begin_norm_axis' should be greater than zero."));
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
