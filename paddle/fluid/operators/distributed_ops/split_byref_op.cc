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

#include "paddle/fluid/operators/distributed_ops/split_byref_op.h"
#include "paddle/fluid/operators/split_op.h"

namespace paddle {
namespace operators {
using framework::Tensor;

class SplitByrefOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext *ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("X"), "Input", "Ids", "SplitByrefOp");
    OP_INOUT_CHECK(ctx->HasOutputs("Out"), "Output", "Out", "SplitByrefOp");

    auto in_dims = ctx->GetInputDim("X");
    auto outs_names = ctx->Outputs("Out");
    size_t num = static_cast<size_t>(ctx->Attrs().Get<int>("num"));
    auto sections = ctx->Attrs().Get<std::vector<int>>("sections");
    const size_t outs_number = outs_names.size();
    std::vector<framework::DDim> outs_dims;
    outs_dims.reserve(outs_number);

    if (num > 0) {
      int64_t in_axis_dim = 0;
      if (ctx->IsRuntime()) {
        in_axis_dim = in_dims[0];
      }
      PADDLE_ENFORCE_EQ(in_axis_dim % num, 0, platform::errors::InvalidArgument(
                                                  "tensor split does not result"
                                                  " in an equal division"));
      size_t out_axis_dim = in_axis_dim / num;
      for (size_t i = 0; i < outs_number; ++i) {
        auto dim = in_dims;
        dim[0] = out_axis_dim;
        outs_dims.push_back(dim);
      }
    } else if (sections.size() > 0) {
      PADDLE_ENFORCE_EQ(
          sections.size(), outs_number,
          platform::errors::InvalidArgument("tensor split sections size"
                                            "should be equal to output size"));
      for (size_t i = 0; i < outs_number; ++i) {
        auto dim = in_dims;
        dim[0] = sections[i];
        outs_dims.push_back(dim);
      }
    }
    ctx->SetOutputsDim("Out", outs_dims);
  }
};

class SplitByrefOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "(Tensor) Input tensor of the split operator.");
    AddOutput("Out", "(Tensor) Output tensors of the split operator.")
        .AsDuplicable();
    AddComment(R"DOC(
SplitByref operator

Split source tensor to sevaral tensors by axis 0. No copy in this operator
is performed, output tensor shares the same blocks of memory.
)DOC");
    AddAttr<std::vector<int>>("sections",
                              "(vector<int>) "
                              "the length of each output along the "
                              "specified axis.")
        .SetDefault(std::vector<int>{});
    AddAttr<int>("num",
                 "(int, default 0)"
                 "Number of sub-tensors. This must evenly divide "
                 "Input.dims()[axis]")
        .SetDefault(0);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
// NOTE: concat op default axis must be 0!
USE_CPU_ONLY_OP(concat);

REGISTER_OPERATOR(split_byref, ops::SplitByrefOp, ops::SplitByrefOpMaker,
                  ops::SplitGradMaker<paddle::framework::OpDesc>,
                  ops::SplitGradMaker<paddle::imperative::OpBase>);
REGISTER_OP_CPU_KERNEL(
    split_byref, ops::SplitByrefOpKernel<paddle::platform::CPUPlace, float>);
