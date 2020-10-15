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

#include "paddle/fluid/operators/sequence_ops/sequence_conv_op.h"

#include <algorithm>
#include <memory>
#include <string>
#include <unordered_set>

namespace paddle {
namespace operators {

class SequenceConvOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(framework::InferShapeContext *ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("X"), "Input", "X", "SequenceConv");
    OP_INOUT_CHECK(ctx->HasInput("Filter"), "Input", "Filter", "SequenceConv");
    OP_INOUT_CHECK(ctx->HasOutput("Out"), "Output", "Out", "SequenceConv");

    int context_length = ctx->Attrs().Get<int>("contextLength");
    int context_start = ctx->Attrs().Get<int>("contextStart");

    auto in_dims = ctx->GetInputDim("X");
    auto filter_dims = ctx->GetInputDim("Filter");
    PADDLE_ENFORCE_EQ(
        ctx->Attrs().Get<int>("contextStride"), 1,
        platform::errors::InvalidArgument(
            "Currently, SequenceConvOp only supports contextStride=1. But "
            "received contextStride = %u.",
            ctx->Attrs().Get<int>("contextStride")));
    PADDLE_ENFORCE_EQ(
        in_dims.size() == 2 && filter_dims.size() == 2, true,
        platform::errors::InvalidArgument(
            "Input(X, Filter) should be 2-D tensor. But received Input(X): "
            "input rank %u, input shape [%s]; received Input(Filter): "
            "input rank %u, input shape [%s].",
            in_dims.size(), in_dims, filter_dims.size(), filter_dims));
    PADDLE_ENFORCE_EQ(
        filter_dims[0], context_length * in_dims[1],
        platform::errors::InvalidArgument(
            "Filter's height should be context_length * "
            "input_hidden_size. But received: filter's height = %d, "
            "context_length * input_hidden_size = %d.",
            filter_dims[0], context_length * in_dims[1]));

    if (ctx->Attrs().Get<bool>("paddingTrainable")) {
      OP_INOUT_CHECK(ctx->HasInput("PaddingData"), "Input", "PaddingData",
                     "sequence_conv");
      framework::DDim padding_dim = ctx->GetInputDim("PaddingData");
      int up_pad = std::max(0, -context_start);
      int down_pad = std::max(0, context_start + context_length - 1);
      int total_pad = up_pad + down_pad;
      int input_width = static_cast<int>(in_dims[1]);
      bool start_equals_zero = context_start == 0;
      bool length_equals_one = context_length == 1;
      bool start_length = start_equals_zero && length_equals_one;

      PADDLE_ENFORCE_EQ(
          start_length, false,
          platform::errors::InvalidArgument(
              "If context_start is 0 and context_length is 1, paddingTrainable "
              "should be false."));
      PADDLE_ENFORCE_EQ(
          padding_dim.size(), 2,
          platform::errors::InvalidArgument(
              "Input(PaddingData) should be 2-D tensor. But received: "
              "input rank %u, input shape [%s].",
              padding_dim.size(), padding_dim));
      PADDLE_ENFORCE_EQ(
          padding_dim[0] == total_pad && padding_dim[1] == input_width, true,
          platform::errors::InvalidArgument("Input(PaddingData)'s shape is not "
                                            "consistent with 'context_start' "
                                            "and 'context_length'. Received "
                                            "Input(PaddingData): input rank "
                                            "%u, "
                                            "input shape [%s].",
                                            padding_dim.size(), padding_dim));
    }

    in_dims[1] = filter_dims[1];
    ctx->SetOutputDim("Out", in_dims);
    ctx->ShareLoD("X", "Out");
  }
};

class SequenceConvGradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(framework::InferShapeContext *ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput(framework::GradVarName("Out")), "Input",
                   framework::GradVarName("Out"), "SequenceConvGrad");
    OP_INOUT_CHECK(ctx->HasInput("X"), "Input", "X", "SequenceConvGrad");

    if (ctx->Attrs().Get<bool>("paddingTrainable") &&
        ctx->HasOutput(framework::GradVarName("PaddingData"))) {
      ctx->SetOutputDim(framework::GradVarName("PaddingData"),
                        ctx->GetInputDim("PaddingData"));
    }
    if (ctx->HasOutput(framework::GradVarName("X"))) {
      ctx->ShareDim("X", /*->*/ framework::GradVarName("X"));
      ctx->ShareLoD("X", /*->*/ framework::GradVarName("X"));
    }
    if (ctx->HasOutput(framework::GradVarName("Filter"))) {
      ctx->SetOutputDim(framework::GradVarName("Filter"),
                        ctx->GetInputDim("Filter"));
    }
  }
};

class SequenceConvOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput(
        "X",
        "(LoDTensor) the input(X) is a LodTensor, which supports "
        "variable-time length input sequence. The underlying tensor in "
        "this LoDTensor is a matrix with shape (T, N), where T is the "
        "total time steps in this mini-batch and N is the input_hidden_size.");
    AddInput("PaddingData",
             "(Tensor, optional) the input(PaddingData) is an optional "
             "parameter, and it is learnable. "
             "This is a tensor with shape (P, N), where P is the "
             "top_pad + bottom_pad, N is the input_hidden_size. In order to "
             "ensure the equal length of sequence before and after "
             "convolution, it is necessary to fill the top and bottom of each "
             "sequence according to context_length, context_stride and "
             "context_start")
        .AsDispensable();
    AddInput(
        "Filter",
        "(Tensor) the input(Filter) is an learnable parameter."
        "This is a tensor with shape (K, M), where K is the "
        "context_length * input_hidden_size, M is the output feature size.");
    AddOutput(
        "Out",
        "(LoDTensor) the output(Out) is a LodTensor, which support "
        "variable-time length output sequence. The underlying tensor in "
        "this LoDTensor is a matrix with shape (T, M), where, T is the "
        "total time steps in this mini-batch, M is the output feature size.");

    AddAttr<bool>("paddingTrainable",
                  "(bool, default:false) the padding data of SequenceConvOp "
                  "is trainable or not.")
        .SetDefault(false);
    AddAttr<int>("contextLength",
                 "(int) the contextLength of SequenceConvOp is the "
                 "height of the convolution kernel.")
        .GreaterThan(0);
    AddAttr<int>("contextStart",
                 "(int, default:0) the contextStart of SequenceConvOp "
                 "represents the beginning of the convolution of the number of "
                 "rows of sequence, which can be negative. The negative number "
                 "means to pad contextStart time-steps of zeros or learnable "
                 "parameters at the beginning of each instance. The positive "
                 "number means to skip contextStart time-steps of each "
                 "instance.")
        .SetDefault(0);
    AddAttr<int>("contextStride",
                 "(int, default:1) the contextStride of SequenceConvOp "
                 "represents the stride length of convolution kernel. "
                 "Currently, SequenceConvOp only supports"
                 "contextStride=1.")
        .SetDefault(1)
        .GreaterThan(0);

    AddComment(R"DOC(
Sequence Conv Operator.

SequenceConvOp performs convolution operation on features of contextLength
time-steps of each instance. The convolution operation calculates the output
based on the input, filter, strides and paddings parameters.
The size of each dimension of the parameters is checked during infer-shape.
In order to ensure the equal length of sequence before and after convolution,
it is necessary to fill the top and bottom of each sequence based on
context_length, context_stride and context_start.

    )DOC");
  }
};

template <typename T>
class SequenceConvGradOpMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> op) const override {
    op->SetType("sequence_conv_grad");
    op->SetAttrMap(this->Attrs());

    if (op->HasAttr("paddingTrainable") &&
        BOOST_GET_CONST(bool, op->GetAttr("paddingTrainable")) &&
        this->HasInput("PaddingData")) {
      op->SetInput("PaddingData", this->Input("PaddingData"));
      op->SetOutput(framework::GradVarName("PaddingData"),
                    this->InputGrad("PaddingData"));
    }

    op->SetInput("X", this->Input("X"));
    op->SetInput("Filter", this->Input("Filter"));
    op->SetInput(framework::GradVarName("Out"), this->OutputGrad("Out"));

    op->SetOutput(framework::GradVarName("X"), this->InputGrad("X"));
    op->SetOutput(framework::GradVarName("Filter"), this->InputGrad("Filter"));
  }
};

class SequenceConvGradNoNeedBufferVarsInference
    : public framework::NoNeedBufferVarsInference {
 public:
  using framework::NoNeedBufferVarsInference::NoNeedBufferVarsInference;

  const std::unordered_set<std::string> &operator()(
      const framework::InferNoNeedBufferVarsContext &ctx) const final {
    static const std::unordered_set<std::string> kPaddingData({"PaddingData"});
    if (!BOOST_GET_CONST(bool, ctx.GetAttr("paddingTrainable"))) {
      return kPaddingData;
    } else {
      return Empty();
    }
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(sequence_conv, ops::SequenceConvOp, ops::SequenceConvOpMaker,
                  ops::SequenceConvGradOpMaker<paddle::framework::OpDesc>,
                  ops::SequenceConvGradOpMaker<paddle::imperative::OpBase>);

REGISTER_OPERATOR(sequence_conv_grad, ops::SequenceConvGradOp,
                  ops::SequenceConvGradNoNeedBufferVarsInference);

REGISTER_OP_CPU_KERNEL(
    sequence_conv,
    ops::SequenceConvKernel<paddle::platform::CPUDeviceContext, float>,
    ops::SequenceConvKernel<paddle::platform::CPUDeviceContext, double>);
REGISTER_OP_CPU_KERNEL(
    sequence_conv_grad,
    ops::SequenceConvGradKernel<paddle::platform::CPUDeviceContext, float>,
    ops::SequenceConvGradKernel<paddle::platform::CPUDeviceContext, double>);
