/* Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
Indicesou may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/operators/p_norm_op.h"
#include <memory>
#include <string>
#include <vector>

namespace paddle {
namespace operators {

class PnormOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "(Tensor) A tensor of rank >= axis.");
    AddAttr<float>("porder",
                   "(float, default 2) The porder is the p order vector norm "
                   "to calculate. Available for porder=0, inf, -inf and any "
                   "real number.")
        .SetDefault(2.0f);
    AddAttr<int>("axis",
                 "The axis on which to apply norm operation. If axis < 0, "
                 "the dimension to pnorm is rank(X) + axis. -1 is "
                 "the last dimension.")
        .SetDefault(-1);
    AddAttr<float>("epsilon",
                   "(float, default 1e-12) The epsilon value is used "
                   "to avoid division by zero.")
        .SetDefault(1.0e-12f);
    AddAttr<bool>(
        "keepdim",
        "(bool, default false) Whether to keep the dimensions as the input.")
        .SetDefault(false);

    AddAttr<bool>("asvector",
                  "(bool, default false) as vector norm when axis is None and "
                  "input is matrix, ")
        .SetDefault(false);
    AddOutput("Out", "(Tensor) Output result tensor of p-norm");
    AddComment(R"DOC(
Pnorm Operator.
Given a tensor X, compute Lp-norm of X.

When p = 0, defining $0^0 = 0$, the zero-norm of X is simply the number of non-zero elements of X.
$$
||X||_{0} = \lim_{p \rightarrow 0} \sum_i |x_i|^p
$$

When p = inf, the inf-norm of X is the maximum element of X.
$$
||X||_\infty = \max_i |x_i|
$$

When p = -inf, the negative-inf-norm of X is the minimum element of X.
$$
||X||_{-\infty} = \min_i |x_i|
$$

Otherwise, the p-norm of X follows the formula,
$$
||X||_{p} = (\sum_i |x_i|^p)^{1/p}
$$
where, $\sum_i $ is calculated along the `axis` dimension.

)DOC");
  }
};

class PnormOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;
  void InferShape(framework::InferShapeContext* ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("X"), "Input", "X", "p_norm");
    OP_INOUT_CHECK(ctx->HasOutput("Out"), "Output", "Out", "p_norm");
    auto x_dim = ctx->GetInputDim("X");
    auto x_rank = x_dim.size();
    int axis = ctx->Attrs().Get<int>("axis");
    bool keepdim = ctx->Attrs().Get<bool>("keepdim");

    PADDLE_ENFORCE_GE(axis, -x_rank,
                      platform::errors::InvalidArgument(
                          "Attr(axis) value should be in range [-R, R-1], R is "
                          "the rank of Input(X). But received axis: %d, R: %d. "
                          "Current Input(X)'s shape is=[%s].",
                          axis, x_rank, x_dim));
    PADDLE_ENFORCE_LT(axis, x_rank,
                      platform::errors::InvalidArgument(
                          "Attr(axis) value should be in range [-R, R-1], R is "
                          "the rank of Input(X). But received axis: %d, R: %d. "
                          "Current Input(X)'s shape is=[%s].",
                          axis, x_rank, x_dim));

    std::vector<int> reduce_dims;
    bool asvector = ctx->Attrs().Get<bool>("asvector");
    if (asvector) {
      reduce_dims.emplace_back(1);
      if (keepdim) {
        for (int i = 1; i < x_dim.size(); ++i) {
          reduce_dims.emplace_back(1);
        }
        x_dim = framework::make_ddim(reduce_dims);
      }
    } else {
      if (axis < 0) axis = x_dim.size() + axis;
      for (int i = 0; i < x_dim.size(); ++i) {
        if (i != axis) reduce_dims.emplace_back(x_dim[i]);
      }
    }
    x_dim[axis] = 1;

    if (keepdim) {
      ctx->SetOutputDim("Out", x_dim);
    } else {
      ctx->SetOutputDim("Out", framework::make_ddim(reduce_dims));
    }
  }
};

class PnormOpGrad : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;
  void InferShape(framework::InferShapeContext* ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("X"), "Input", "X", "p_norm");
    OP_INOUT_CHECK(ctx->HasInput("Out"), "Input", "Out", "p_norm");
    OP_INOUT_CHECK(ctx->HasInput(framework::GradVarName("Out")), "Input",
                   "Out@GRAD", "p_norm");
    OP_INOUT_CHECK(ctx->HasOutput(framework::GradVarName("X")), "Output",
                   "X@GRAD", "p_norm");
    ctx->SetOutputDim(framework::GradVarName("X"), ctx->GetInputDim("X"));
  }
};

template <typename T>
class PnormOpGradOpMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> op) const override {
    op->SetType("p_norm_grad");
    op->SetAttrMap(this->Attrs());
    op->SetInput("X", this->Input("X"));
    op->SetInput("Out", this->Output("Out"));
    op->SetInput(framework::GradVarName("Out"), this->OutputGrad("Out"));
    op->SetOutput(framework::GradVarName("X"), this->InputGrad("X"));
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
using CPU = paddle::platform::CPUDeviceContext;

REGISTER_OPERATOR(p_norm, ops::PnormOp, ops::PnormOpMaker,
                  ops::PnormOpGradOpMaker<paddle::framework::OpDesc>,
                  ops::PnormOpGradOpMaker<paddle::imperative::OpBase>);
REGISTER_OPERATOR(p_norm_grad, ops::PnormOpGrad);
REGISTER_OP_CPU_KERNEL(p_norm, ops::PnormKernel<CPU, float>,
                       ops::PnormKernel<CPU, double>);
REGISTER_OP_CPU_KERNEL(p_norm_grad, ops::PnormGradKernel<CPU, float>,
                       ops::PnormGradKernel<CPU, double>);
