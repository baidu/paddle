// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/operators/reduce_ops/reduce_mean_op.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace paddle {
namespace operators {

// NOTE(dengkaipeng): Input(Out) is unnecessary in reduce_mean_grad
// calcualtion, but will incur a reduce_mean_grad op after
// reduce_mean_grad_grad, delete Input(Out) here.
// This change has no effect on reduce_mean_grad calculations.
template <typename T>
class ReduceMeanOpGradMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> op) const override {
    op->SetType("reduce_mean_grad");
    op->SetInput("X", this->Input("X"));
    op->SetInput(framework::GradVarName("Out"), this->OutputGrad("Out"));
    op->SetAttrMap(this->Attrs());
    op->SetOutput(framework::GradVarName("X"), this->InputGrad("X"));
  }
};

class ReduceMeanDoubleGradDescMaker : public framework::GradOpDescMakerBase {
 public:
  using framework::GradOpDescMakerBase::GradOpDescMakerBase;

  std::vector<std::unique_ptr<framework::OpDesc>> operator()() const override {
    std::vector<std::unique_ptr<framework::OpDesc>> ops;
    auto x_gg = OutputGrad(framework::GradVarName("X"));  // input ddx
    auto out_grads = InputGrad(framework::GradVarName("Out"));
    if (!out_grads.empty()) {
      auto* out_grad_op = new framework::OpDesc();
      out_grad_op->SetType("reduce_mean");
      out_grad_op->SetInput("X", x_gg);
      out_grad_op->SetAttrMap(Attrs());
      out_grad_op->SetOutput("Out", out_grads);
      ops.emplace_back(out_grad_op);
    }

    return ops;
  }
};
class ReduceMeanDoubleGradOpBaseMaker : public imperative::GradOpBaseMakerBase {
 public:
  using imperative::GradOpBaseMakerBase::GradOpBaseMakerBase;

  std::shared_ptr<imperative::GradOpNode> operator()() const override {
    auto out_grads = InputGrad(framework::GradVarName("Out"));
    if (!out_grads.empty()) {
      auto x_gg = OutputGrad(framework::GradVarName("X"));  // input ddx
      auto node = this->NewGradNode();
      {
        imperative::TracedGradOp op(node);
        op.SetType("reduce_mean");
        op.SetInput("X", x_gg);
        op.SetAttrMap(Attrs());
        op.SetOutput("Out", out_grads);
      }
      return node;
    } else {
      return nullptr;
    }
  }
};
DECLARE_NO_NEED_BUFFER_VARS_INFERER(ReduceMeanGradNoNeedBufferVarInferer, "X");
}  // namespace operators
}  // namespace paddle

class __reduce_meanMaker__ : public ops::ReduceOpMaker {
 protected:
  std::string GetName() const override { return "reduce_mean"; }
  std::string GetOpType() const override { return "Reduce reduce_mean"; }
};

REGISTER_OPERATOR(reduce_mean, ops::ReduceOp, __reduce_meanMaker__,
                  ops::ReduceMeanOpGradMaker<paddle::framework::OpDesc>,
                  ops::ReduceMeanOpGradMaker<paddle::imperative::OpBase>);
REGISTER_OPERATOR(reduce_mean_grad, ops::ReduceGradOp,
                  ops::ReduceMeanDoubleGradDescMaker,
                  ops::ReduceMeanDoubleGradOpBaseMaker,
                  ops::ReduceMeanGradNoNeedBufferVarInferer);
REGISTER_OP_CPU_KERNEL(reduce_mean,
                       ops::ReduceKernel<paddle::platform::CPUDeviceContext,
                                         bool, ops::MeanFunctor>,
                       ops::ReduceKernel<paddle::platform::CPUDeviceContext,
                                         float, ops::MeanFunctor>,
                       ops::ReduceKernel<paddle::platform::CPUDeviceContext,
                                         double, ops::MeanFunctor>);

template <typename T>
using CPUReduceMeanGradKernel =
    ops::ReduceGradKernel<paddle::platform::CPUDeviceContext, T,
                          ops::MeanGradFunctor, true>;

REGISTER_OP_CPU_KERNEL(reduce_mean_grad, CPUReduceMeanGradKernel<bool>,
                       CPUReduceMeanGradKernel<float>,
                       CPUReduceMeanGradKernel<double>);
