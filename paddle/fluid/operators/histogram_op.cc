/* Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/operators/histogram_op.h"

#include <string>
#include <unordered_map>
#include <vector>

namespace paddle {
namespace operators {

using framework::OpKernelType;
using framework::Tensor;

class HistogramOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext *ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("X"), "Input", "X", "histogram");
    OP_INOUT_CHECK(ctx->HasOutput("Out"), "Output", "Out", "histogram");
    const auto &nbins = ctx->Attrs().Get<int64_t>("bins");
    const auto &minval = ctx->Attrs().Get<int>("min");
    const auto &maxval = ctx->Attrs().Get<int>("max");

    PADDLE_ENFORCE_GE(nbins, 1,
                      platform::errors::InvalidArgument(
                          "The bins should be greater than or equal to 1."
                          "But received nbins is %d",
                          nbins));
    PADDLE_ENFORCE_GE(
        maxval, minval,
        platform::errors::InvalidArgument("max must be larger or equal to min."
                                          "But received max is %d, min is %d",
                                          maxval, minval));

    ctx->SetOutputDim("Out", framework::make_ddim({nbins}));
    ctx->ShareLoD("X", /*->*/ "Out");
  }

  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override {
    auto data_type = OperatorWithKernel::IndicateVarDataType(ctx, "X");
    return framework::OpKernelType(data_type, ctx.device_context());
  }
};

class HistogramOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "(Tensor) The input tensor of Histogram op,");
    AddOutput("Out", "(Tensor) The output tensor of Histogram op,");
    AddAttr<int64_t>("bins", "(int) number of histogram bins")
        .SetDefault(100)
        .EqualGreaterThan(1);
    AddAttr<int>("min", "(int) lower end of the range (inclusive)")
        .SetDefault(0);
    AddAttr<int>("max", "(int) upper end of the range (inclusive)")
        .SetDefault(0);
    AddComment(R"DOC(
          Histogram Operator.
          Computes the histogram of a tensor. The elements are sorted
          into equal width bins between min and max. If min and max are
          both zero, the minimum and maximum values of the data are used.
      )DOC");
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(
    histogram, ops::HistogramOp, ops::HistogramOpMaker,
    paddle::framework::EmptyGradOpMaker<paddle::framework::OpDesc>,
    paddle::framework::EmptyGradOpMaker<paddle::imperative::OpBase>);
REGISTER_OP_CPU_KERNEL(
    histogram, ops::HistogramKernel<paddle::platform::CPUDeviceContext, float>,
    ops::HistogramKernel<paddle::platform::CPUDeviceContext, double>,
    ops::HistogramKernel<paddle::platform::CPUDeviceContext, int>,
    ops::HistogramKernel<paddle::platform::CPUDeviceContext, int64_t>);
