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

#include "paddle/fluid/operators/elementwise/elementwise_min_op.h"

#include <string>

#include "paddle/fluid/operators/elementwise/elementwise_op.h"

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

class ElementwiseMinOpMaker : public ElementwiseOpMaker {
 protected:
  std::string GetName() const override { return "Min"; }
  std::string GetEquation() const override { return "Out = min(X, Y)"; }

  void AddInputX() override {
    AddInput("X", "The first tensor holding the elements to be compared.");
  }

  void AddInputY() override {
    AddInput(
        "Y",
        "(Variable), The second tensor holding the elements to be compared.");
  }

  std::string GetOpFuntionality() const override {
    return "Compare two tensors and returns a new tensor containing the "
           "element-wise minima.";
  }
};

template <typename T>
class ElementwiseMinGradOpMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> op) const override {
    op->SetType("elementwise_min_grad");
    op->SetInput("X", this->Input("X"));
    op->SetInput("Y", this->Input("Y"));
    op->SetInput(framework::GradVarName("Out"), this->OutputGrad("Out"));
    op->SetOutput(framework::GradVarName("X"), this->InputGrad("X"));
    op->SetOutput(framework::GradVarName("Y"), this->InputGrad("Y"));
    op->SetAttrMap(this->Attrs());
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OPERATOR(elementwise_min, ops::ElementwiseOp,
                  ops::ElementwiseMinOpMaker, ops::ElementwiseOpInferVarType,
                  ops::ElementwiseMinGradOpMaker<paddle::framework::OpDesc>,
                  ops::ElementwiseMinGradOpMaker<paddle::imperative::OpBase>);

REGISTER_OPERATOR(elementwise_min_grad, ops::ElementwiseOpGrad);

REGISTER_OP_CPU_KERNEL(
    elementwise_min,
    ops::ElementwiseMinKernel<paddle::platform::CPUDeviceContext, float>,
    ops::ElementwiseMinKernel<paddle::platform::CPUDeviceContext, double>,
    ops::ElementwiseMinKernel<paddle::platform::CPUDeviceContext, int>,
    ops::ElementwiseMinKernel<paddle::platform::CPUDeviceContext, int64_t>);
REGISTER_OP_CPU_KERNEL(
    elementwise_min_grad,
    ops::ElementwiseMinGradKernel<paddle::platform::CPUDeviceContext, float>,
    ops::ElementwiseMinGradKernel<paddle::platform::CPUDeviceContext, double>,
    ops::ElementwiseMinGradKernel<paddle::platform::CPUDeviceContext, int>,
    ops::ElementwiseMinGradKernel<paddle::platform::CPUDeviceContext, int64_t>);
