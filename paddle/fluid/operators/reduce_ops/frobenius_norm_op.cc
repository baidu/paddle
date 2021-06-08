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

#include "paddle/fluid/operators/reduce_ops/frobenius_norm_op.h"

#include <string>

namespace paddle {
namespace framework {
class OpDesc;
}  // namespace framework
namespace imperative {
class OpBase;
}  // namespace imperative
namespace platform {
class CPUDeviceContext;
struct CPUPlace;
}  // namespace platform
}  // namespace paddle

namespace paddle {
namespace operators {

template <typename T>
class FrobeniusNormOpGradMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> op) const override {
    op->SetType("frobenius_norm_grad");
    op->SetInput("X", this->Input("X"));
    op->SetInput("Out", this->Output("Out"));
    op->SetInput(framework::GradVarName("Out"), this->OutputGrad("Out"));
    op->SetOutput(framework::GradVarName("X"), this->InputGrad("X"));
    op->SetAttrMap(this->Attrs());
  }
};

}  // namespace operators
}  // namespace paddle

class FrobeniusNormOpMaker : public ops::ReduceOpMaker {
 protected:
  std::string GetName() const override { return "frobenius_norm"; }
  std::string GetOpType() const override { return "Reduce frobenius_norm"; }
};

REGISTER_OPERATOR(frobenius_norm, ops::ReduceOp, FrobeniusNormOpMaker,
                  ops::FrobeniusNormOpGradMaker<paddle::framework::OpDesc>,
                  ops::FrobeniusNormOpGradMaker<paddle::imperative::OpBase>);

REGISTER_OPERATOR(frobenius_norm_grad, ops::ReduceGradOp);

REGISTER_OP_CPU_KERNEL(frobenius_norm,
                       ops::ReduceKernel<paddle::platform::CPUDeviceContext,
                                         float, ops::FrobeniusNormFunctor>,
                       ops::ReduceKernel<paddle::platform::CPUDeviceContext,
                                         double, ops::FrobeniusNormFunctor>);

template <typename T>
using CPUFrobeniusNormGradKernel =
    ops::FrobeniusNormGradKernel<paddle::platform::CPUDeviceContext, T,
                                 ops::FrobeniusNormGradFunctor>;

REGISTER_OP_CPU_KERNEL(frobenius_norm_grad, CPUFrobeniusNormGradKernel<float>,
                       CPUFrobeniusNormGradKernel<double>);
