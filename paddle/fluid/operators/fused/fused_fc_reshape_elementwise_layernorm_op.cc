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

class FusedFCReshapeElementwiseLayerNormOp
    : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext *ctx) const override {
    auto y_dims = ctx->GetInputDim("Y");
    ctx->SetOutputDim("Out", y_dims);
    ctx->ShareLoD("X", "Out");
  }
};

class FusedFCReshapeElementwiseLayerNormOpMaker
    : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "(Tensor), The input tensor of fully connected operation");
    AddInput(
        "W_1",
        "(Tensor), The weight tensor of fully connected_1 operation. It is "
        "a 2-D Tensor with shape (I, O)");
    AddInput("Bias0_1",
             "(Tensor, optional), The bias tensor of fully connected_1 "
             "operation. It is a 1-D Tensor with shape (O), or a 2-D Tensor "
             "with shape (1, O).")
        .AsDispensable();
    AddInput(
        "W_2",
        "(Tensor), The weight tensor of fully connected_2 operation. It is "
        "a 2-D Tensor with shape (I, O)");
    AddInput("Bias0_2",
             "(Tensor, optional), The bias tensor of fully connected_2 "
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
    AddAttr<int>("x_num_col_dims_1",
                 "(int, default 1), This op can take tensors with more than "
                 "two dimensions as its inputs.")
        .SetDefault(1)
        .EqualGreaterThan(1);
    AddAttr<int>("x_num_col_dims_2",
                 "(int, default 1), This op can take tensors with more than "
                 "two dimensions as its inputs.")
        .SetDefault(1)
        .EqualGreaterThan(1);
    AddAttr<std::string>("activation_type_1",
                         "Activation type used in fully connected operator.")
        .SetDefault("");
    AddAttr<std::string>("activation_type_2",
                         "Activation type used in fully connected operator.")
        .SetDefault("");
    AddAttr<std::vector<int>>(
        "shape1",
        "(std::vector<int>) Target shape of reshape_1 operator."
        "It has the lowest priority compare with Input(Shape) and "
        " Input(ShapeTensor).")
        .SetDefault({});
    AddAttr<std::vector<int>>(
        "shape2",
        "(std::vector<int>) Target shape of reshape_2 operator."
        "It has the lowest priority compare with Input(Shape) and "
        " Input(ShapeTensor).")
        .SetDefault({});

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
    fused_fc_reshape_elementwise_layernorm,
    ops::FusedFCReshapeElementwiseLayerNormOp,
    ops::FusedFCReshapeElementwiseLayerNormOpMaker,
    paddle::framework::EmptyGradOpMaker<paddle::framework::OpDesc>,
    paddle::framework::EmptyGradOpMaker<paddle::imperative::OpBase>);
