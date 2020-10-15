// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "paddle/fluid/operators/reduce_ops/logsumexp_op.h"
#include <algorithm>
#include <string>
#include <vector>

namespace paddle {
namespace operators {

class LogsumexpOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("X"), "Input", "X", "logsumexp");
    OP_INOUT_CHECK(ctx->HasOutput("Out"), "Output", "Out", "logsumexp");
    auto x_dims = ctx->GetInputDim("X");
    auto x_rank = x_dims.size();
    PADDLE_ENFORCE_LE(x_rank, 4,
                      platform::errors::InvalidArgument(
                          "The input tensor X's dimensions of logsumexp "
                          "should be less or equal than 4. But received X's "
                          "dimensions = %d, X's shape = [%s].",
                          x_rank, x_dims));
    auto axis = ctx->Attrs().Get<std::vector<int>>("axis");
    PADDLE_ENFORCE_GT(
        axis.size(), 0,
        platform::errors::InvalidArgument(
            "The size of axis of logsumexp "
            "should be greater than 0. But received the size of axis "
            "of logsumexp is %d.",
            axis.size()));

    for (size_t i = 0; i < axis.size(); i++) {
      PADDLE_ENFORCE_LT(axis[i], x_rank,
                        platform::errors::InvalidArgument(
                            "axis[%d] should be in the "
                            "range [-D, D), where D is the dimensions of X and "
                            "D is %d. But received axis[%d] = %d.",
                            i, x_rank, i, axis[i]));
      PADDLE_ENFORCE_GE(axis[i], -x_rank,
                        platform::errors::InvalidArgument(
                            "axis[%d] should be in the "
                            "range [-D, D), where D is the dimensions of X and "
                            "D is %d. But received axis[%d] = %d.",
                            i, x_rank, i, axis[i]));
      if (axis[i] < 0) {
        axis[i] += x_rank;
      }
    }

    bool keepdim = ctx->Attrs().Get<bool>("keepdim");
    bool reduce_all = ctx->Attrs().Get<bool>("reduce_all");
    auto dims_vector = vectorize(x_dims);
    if (reduce_all) {
      if (keepdim)
        ctx->SetOutputDim(
            "Out", framework::make_ddim(std::vector<int64_t>(x_rank, 1)));
      else
        ctx->SetOutputDim("Out", {1});
    } else {
      auto dims_vector = vectorize(x_dims);
      if (keepdim) {
        for (size_t i = 0; i < axis.size(); ++i) {
          dims_vector[axis[i]] = 1;
        }
      } else {
        const int kDelFlag = -1;
        for (size_t i = 0; i < axis.size(); ++i) {
          dims_vector[axis[i]] = kDelFlag;
        }
        dims_vector.erase(
            std::remove(dims_vector.begin(), dims_vector.end(), kDelFlag),
            dims_vector.end());
      }
      if (!keepdim && dims_vector.size() == 0) {
        dims_vector.push_back(1);
      }
      auto out_dims = framework::make_ddim(dims_vector);
      ctx->SetOutputDim("Out", out_dims);
      if (axis.size() > 0 && axis[0] != 0) {
        // Only pass LoD when not reducing on the first dim.
        ctx->ShareLoD("X", /*->*/ "Out");
      }
    }
  }
};

class LogsumexpOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X",
             "(Tensor) The input tensor. Tensors with rank at most 4 are "
             "supported.");
    AddOutput("Out", "(Tensor) The result tensor.");
    AddAttr<std::vector<int>>(
        "axis",
        "(list<int>, default {0}) The dimensions to reduce. "
        "Must be in the range [-rank(input), rank(input)). "
        "If `axis[i] < 0`, the axis[i] to reduce is `rank + axis[i]`. "
        "Note that reducing on the first dim will make the LoD info lost.")
        .SetDefault({0});
    AddAttr<bool>("keepdim",
                  "(bool, default false) "
                  "If true, retain the reduced dimension with length 1.")
        .SetDefault(false);
    AddAttr<bool>("reduce_all",
                  "(bool, default false) "
                  "If true, output a scalar reduced along all dimensions.")
        .SetDefault(false);
    AddComment(string::Sprintf(R"DOC(
logsumexp Operator.

This operator computes the logsumexp of input tensor along the given axis.
The result tensor has 1 fewer dimension than the input unless keep_dim is true.
If reduce_all is true, just reduce along all dimensions and output a scalar.

)DOC"));
  }
};

class LogsumexpGrapOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("X"), "Input", "X", "logsumexp");
    OP_INOUT_CHECK(ctx->HasInput("Out"), "Input", "Out", "logsumexp");
    OP_INOUT_CHECK(ctx->HasInput(framework::GradVarName("Out")), "Input",
                   "Out@GRAD", "logsumexp");
    ctx->SetOutputDim(framework::GradVarName("X"), ctx->GetInputDim("X"));
  }
};

template <typename T>
class LogsumexpGradOpMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

  void Apply(GradOpPtr<T> op) const override {
    op->SetType("logsumexp_grad");
    op->SetInput("X", this->Input("X"));
    op->SetInput("Out", this->Output("Out"));
    op->SetInput(framework::GradVarName("Out"), this->OutputGrad("Out"));
    op->SetAttrMap(this->Attrs());
    op->SetOutput(framework::GradVarName("X"), this->InputGrad("X"));
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OPERATOR(logsumexp, ops::LogsumexpOp, ops::LogsumexpOpMaker,
                  ops::LogsumexpGradOpMaker<paddle::framework::OpDesc>,
                  ops::LogsumexpGradOpMaker<paddle::imperative::OpBase>);
REGISTER_OPERATOR(logsumexp_grad, ops::LogsumexpGrapOp);

REGISTER_OP_CPU_KERNEL(
    logsumexp, ops::LogsumexpKernel<paddle::platform::CPUDeviceContext, float>,
    ops::LogsumexpKernel<paddle::platform::CPUDeviceContext, double>);
REGISTER_OP_CPU_KERNEL(
    logsumexp_grad,
    ops::LogsumexpGradKernel<paddle::platform::CPUDeviceContext, float>,
    ops::LogsumexpGradKernel<paddle::platform::CPUDeviceContext, double>);
