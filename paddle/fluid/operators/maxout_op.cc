/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 *     Unless required by applicable law or agreed to in writing, software
 *     distributed under the License is distributed on an "AS IS" BASIS,
 *     WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *     See the License for the specific language governing permissions and
 *     limitations under the License. */

#include "paddle/fluid/operators/maxout_op.h"
#include <vector>

namespace paddle {
namespace operators {

using framework::Tensor;

class MaxOutOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X",
             "A 4-D Tensor with data type of float32 or float64. "
             "The data format is NCHW or NHWC. Where N is "
             "batch size, C is the number of channels, "
             "H and W is the height and width of "
             "feature. ");
    AddOutput("Out",
              "A 4-D Tensor with same data type and data format "
              "with input Tensor. ");
    AddAttr<int>(
        "groups",
        "Specifies how many groups the input tensor will be split into "
        "at the channel dimension. And the number of output channel is "
        "the number of channels divided by groups. ");
    AddAttr<int>(
        "axis",
        "Specifies the index of channel dimension where maxout will "
        "be performed. It should be 1 when data format is NCHW, -1 or 3 "
        "when data format is NHWC. "
        "Default: 1. ")
        .SetDefault(1);
    AddComment(R"DOC(
MaxOut Operator.

Assumed the input shape is (N, Ci, H, W).
The output shape is (N, Co, H, W).
Then $Co = Ci / groups$ and the operator formula is as follows:

$$ y_{si+j} = \max_{k} x_{gsi + sk + j} $$
$$ g = groups $$
$$ s = \\frac{input.size}{num\\_channels} $$
$$ 0 \\le i < \\frac{num\\_channels}{groups} $$
$$ 0 \\le j < s $$
$$ 0 \\le k < groups $$

Please refer to Paper:
  - Maxout Networks: http://www.jmlr.org/proceedings/papers/v28/goodfellow13.pdf
  - Multi-digit Number Recognition from Street View \
    Imagery using Deep Convolutional Neural Networks: \
    https://arxiv.org/pdf/1312.6082v4.pdf

)DOC");
  }
};

class MaxOutOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;
  void InferShape(framework::InferShapeContext* ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("X"), "Input", "X", "maxout");
    OP_INOUT_CHECK(ctx->HasOutput("Out"), "Output", "Out", "maxout");

    auto in_x_dims = ctx->GetInputDim("X");
    int groups = ctx->Attrs().Get<int>("groups");
    int axis = ctx->Attrs().Get<int>("axis");
    // check groups > 1
    PADDLE_ENFORCE_GT(groups, 1, platform::errors::InvalidArgument(
                                     "Attr(groups) of Op(maxout) should be "
                                     "larger than 1. But received %d.",
                                     groups));
    PADDLE_ENFORCE_EQ(
        axis == 1 || axis == -1 || axis == 3, true,
        platform::errors::InvalidArgument(
            "axis only supported 1, -1 or 3, but recevied axis is: %d", axis));
    PADDLE_ENFORCE_EQ(in_x_dims.size(), 4,
                      platform::errors::InvalidArgument(
                          "x's dims should be 4, but received x's dims is: %d",
                          in_x_dims.size()));

    if (axis < 0) {
      axis += in_x_dims.size();
    }
    PADDLE_ENFORCE_EQ(
        in_x_dims[axis] % groups, 0,
        platform::errors::InvalidArgument(
            "The number of input channels for Op(maxout) "
            "should be divisible by Attr(groups). But received: the "
            "input's channels is [%d], the shape of input is [%s], "
            "the Attr(groups) is [%d], the Attr(axis) is [%d]. The "
            "error may come from wrong Attr(groups) or Attr(axis) setting.",
            in_x_dims[axis], in_x_dims, groups, axis));
    std::vector<int64_t> output_shape(
        {in_x_dims[0], in_x_dims[1], in_x_dims[2], in_x_dims[3]});
    output_shape[axis] = in_x_dims[axis] / groups;
    ctx->SetOutputDim("Out", framework::make_ddim(output_shape));
  }
};

class MaxOutOpGrad : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;
  void InferShape(framework::InferShapeContext* ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("X"), "Input", "X", "maxout_grad");
    OP_INOUT_CHECK(ctx->HasOutput(framework::GradVarName("X")), "Output",
                   "X@Grad", "maxout_grad");
    ctx->SetOutputDim(framework::GradVarName("X"), ctx->GetInputDim("X"));
  }
};
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(
    maxout, ops::MaxOutOp, ops::MaxOutOpMaker,
    paddle::framework::DefaultGradOpMaker<paddle::framework::OpDesc, true>,
    paddle::framework::DefaultGradOpMaker<paddle::imperative::OpBase, true>);
REGISTER_OPERATOR(maxout_grad, ops::MaxOutOpGrad);
REGISTER_OP_CPU_KERNEL(
    maxout, ops::MaxOutKernel<paddle::platform::CPUDeviceContext, float>,
    ops::MaxOutKernel<paddle::platform::CPUDeviceContext, double>);
REGISTER_OP_CPU_KERNEL(
    maxout_grad,
    ops::MaxOutGradKernel<paddle::platform::CPUDeviceContext, float>,
    ops::MaxOutGradKernel<paddle::platform::CPUDeviceContext, double>);
