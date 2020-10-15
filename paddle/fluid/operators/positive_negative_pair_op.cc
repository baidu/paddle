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

#include "paddle/fluid/operators/positive_negative_pair_op.h"
#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace operators {

class PositiveNegativePairOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext *ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("Score"), "Input", "Score",
                   "positive_negative_pair");
    OP_INOUT_CHECK(ctx->HasInput("Label"), "Input", "Label",
                   "positive_negative_pair");
    OP_INOUT_CHECK(ctx->HasInput("QueryID"), "Input", "QueryID",
                   "positive_negative_pair");
    OP_INOUT_CHECK(ctx->HasOutput("PositivePair"), "Output", "PositivePair",
                   "positive_negative_pair");
    OP_INOUT_CHECK(ctx->HasOutput("NegativePair"), "Output", "NegativePair",
                   "positive_negative_pair");
    OP_INOUT_CHECK(ctx->HasOutput("NeutralPair"), "Output", "NeutralPair",
                   "positive_negative_pair");

    auto scalar_dim = framework::make_ddim({1});
    if (ctx->HasInput("AccumulatePositivePair") ||
        ctx->HasInput("AccumulateNegativePair") ||
        ctx->HasInput("AccumulateNeutralPair")) {
      PADDLE_ENFORCE_EQ(
          ctx->HasInput("AccumulatePositivePair") &&
              ctx->HasInput("AccumulateNegativePair") &&
              ctx->HasInput("AccumulateNeutralPair"),
          true, platform::errors::InvalidArgument(
                    "All optional inputs(AccumulatePositivePair, "
                    "AccumulateNegativePair, AccumulateNeutralPair) of "
                    "PositiveNegativePairOp are required if one of them "
                    "is specified."));
      PADDLE_ENFORCE_EQ(
          ctx->GetInputDim("AccumulatePositivePair"), scalar_dim,
          platform::errors::InvalidArgument(
              "Shape of Input(AccumulatePositivePair) should be [1]. Received "
              "shape of Input(AccumulatePositivePair): [%s].",
              ctx->GetInputDim("AccumulatePositivePair")));
      PADDLE_ENFORCE_EQ(
          ctx->GetInputDim("AccumulateNegativePair"), scalar_dim,
          platform::errors::InvalidArgument(
              "Shape of Input(AccumulateNegativePair) should be [1]. Received "
              "shape of Input(AccumulateNegativePair): [%s].",
              ctx->GetInputDim("AccumulateNegativePair")));
      PADDLE_ENFORCE_EQ(
          ctx->GetInputDim("AccumulateNeutralPair"), scalar_dim,
          platform::errors::InvalidArgument(
              "Shape of Input(AccumulateNeutralPair) should be [1]. Received "
              "shape of Input(AccumulateNeutralPair): [%s].",
              ctx->GetInputDim("AccumulateNeutralPair")));
    }

    auto score_dim = ctx->GetInputDim("Score");
    auto label_dim = ctx->GetInputDim("Label");
    auto query_dim = ctx->GetInputDim("QueryID");
    PADDLE_ENFORCE_EQ(score_dim.size(), 2,
                      platform::errors::InvalidArgument(
                          "Score should be a 2-D tensor. Received shape of "
                          "Input(Score): [%s].",
                          score_dim));
    PADDLE_ENFORCE_EQ(label_dim.size(), 2,
                      platform::errors::InvalidArgument(
                          "Label should be a 2-D tensor. Received shape of "
                          "Input(Label): [%s].",
                          label_dim));

    if (ctx->IsRuntime() ||
        (score_dim[0] > 0 && label_dim[0] > 0 && query_dim[0] > 0)) {
      PADDLE_ENFORCE_EQ(
          label_dim[0], score_dim[0],
          platform::errors::InvalidArgument(
              "Input(Score) and Input(Label) should have the same "
              "height (batch size). Received: the shape of Input(Score) is "
              "[%s], while the shape of Input(Label) is [%s]. The first "
              "dimensions of them are different.",
              label_dim, score_dim));

      PADDLE_ENFORCE_EQ(
          label_dim[1], 1,
          platform::errors::InvalidArgument(
              "The width of Label should be 1, i.e. each item should "
              "have a scalar label. Received shape of Input(Label) is [%s]. "
              "The second dimension of it is %d, while the expected is %d.",
              label_dim, label_dim[1], 1));

      PADDLE_ENFORCE_EQ(
          query_dim, label_dim,
          platform::errors::InvalidArgument(
              "Input(QueryID) should have the same shape as Input(Label). "
              "Received: the shape of Input(QueryID) is [%s], "
              "while the shape of Input(Label) is [%s].",
              query_dim, label_dim));

      if (ctx->HasInput("Weight")) {
        PADDLE_ENFORCE_EQ(
            ctx->GetInputDim("Weight"), label_dim,
            platform::errors::InvalidArgument(
                "Input(Weight) should have the same shape as Input(Label). "
                "Received: the shape of Input(Weight) is [%s] while the shape "
                "of Input(Label) is [%s].",
                ctx->GetInputDim("Weight"), label_dim));
      }

      int column = ctx->Attrs().Get<int>("column");
      auto depth = score_dim[1];
      PADDLE_ENFORCE_LT(
          column, depth,
          platform::errors::OutOfRange(
              "Attr(column) should be less than depth(the second "
              "dimension of Input(Score)). Recieved Attr(column): %d, while "
              "depth is %d.",
              column, depth));
      PADDLE_ENFORCE_GE(
          column, -depth,
          platform::errors::OutOfRange(
              "Attr(column) should be greater than equal to negative "
              "depth, i.e. the second dimension of Input(Score). "
              "Recieved Attr(column): %d, while negative depth is %d.",
              column, -depth));
    }

    ctx->SetOutputDim("PositivePair", scalar_dim);
    ctx->SetOutputDim("NegativePair", scalar_dim);
    ctx->SetOutputDim("NeutralPair", scalar_dim);
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override {
    return framework::OpKernelType(
        OperatorWithKernel::IndicateVarDataType(ctx, "Score"),
        ctx.device_context());
  }
};

class PositiveNegativePairOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("Score",
             "(Tensor, float) Model Score on an item (with "
             "respect to QueryID). It's a 2-D tensor with shape [batch_size, "
             "depth], where the column specified by the attribute \"column\" "
             "is used as item score.");
    AddInput("Label",
             "(Tensor, float) Label of an item (with repsect to "
             "QueryId). It's a 2-D tensor with shape [batch_size, 1].");
    AddInput("QueryID",
             "(Tensor, int64) Query ID that indicates the context. Its shape "
             "should be the same as Label.");
    AddInput(
        "AccumulatePositivePair",
        "(float) Optional. The accumulated number of positive pairs over a "
        "stream of data. If provided, the output PositivePair will be "
        "initialized with this number rather than 0. it won't be modified "
        "in place.")
        .AsDispensable();
    AddInput(
        "AccumulateNegativePair",
        "(float) Optional. The accumulated number of negative pairs over a "
        "stream of data. If provided, the output NegativePair will be "
        "initialized with this number rather than 0. it won't be modified "
        "in place.")
        .AsDispensable();
    AddInput("AccumulateNeutralPair",
             "(float) Optional. The accumulated number of neutral pairs over a "
             "stream of data. If provided, the output NeutralPair will be "
             "initialized with this number rather than 0. it won't be modified "
             "in place.")
        .AsDispensable();
    AddInput("Weight",
             "(float) Optional. Weight of current item. If specified, its "
             "shape should be the same as Label, and the meaning of the output "
             "changes from numbers of pairs to the total sum of pairs' "
             "weights. Weight of a pair of items is the average of their "
             "weights.")
        .AsDispensable();
    AddOutput("PositivePair",
              "(float) Number of positive pairs, i.e. the pairs of "
              "items that are ranked correctly.");
    AddOutput("NegativePair",
              "(float) Number of negative pairs, i.e. the pairs of "
              "items that are ranked incorrectly.");
    AddOutput("NeutralPair",
              "(float) Number of neutral pairs, i.e. the pairs of items "
              "that have the same score.")
        .AsDispensable();
    AddAttr<int>(
        "column",
        "(int, default -1) The column position of Score used to rank items in "
        "descending order. It must be in the range of [-rank(Score), "
        "rank(Score)). "
        "If `dim < 0`, the dim to reduce is `rank + dim`. "
        "Noting that reducing on the first dim will make the LoD info lost.")
        .SetDefault(0);
    AddComment(R"DOC(
PositiveNegativePairOp can be used to evaluate Learning To Rank(LTR) model's
performance.

Within some context, e.g. the "query", a LTR model generates scores for a list
of items, which gives a partial order of the items. PositiveNegativePairOp
takes a list of reference rank order (Input("Label")) and the model generated
scores (Input(Score)) as inputs and counts the pairs that ranked correctly
and incorrectly.
)DOC");
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_WITHOUT_GRADIENT(positive_negative_pair,
                             ops::PositiveNegativePairOp,
                             ops::PositiveNegativePairOpMaker);
REGISTER_OP_CPU_KERNEL(
    positive_negative_pair,
    ops::PositiveNegativePairKernel<paddle::platform::CPUPlace, float>,
    ops::PositiveNegativePairKernel<paddle::platform::CPUPlace, double>);
