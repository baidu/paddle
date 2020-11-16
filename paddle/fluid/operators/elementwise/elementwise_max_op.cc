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

#include "paddle/fluid/operators/elementwise/elementwise_max_op.h"

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

class ElementwiseMaxOpMaker : public ElementwiseOpMaker {
 protected:
  std::string GetName() const override { return "Max"; }
  std::string GetEquation() const override { return "Out = max(X, Y)"; }

  void AddInputX() override {
    AddInput("X", "The first tensor holding the elements to be compared.");
  }

  void AddInputY() override {
    AddInput("Y", "The second tensor holding the elements to be compared.");
  }

  std::string GetOpFuntionality() const override {
    return "Compare two tensors and returns a new tensor containing the "
           "element-wise maxima.";
  }
};

template <typename T>
class ElementwiseMaxGradOpMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> op) const override {
    op->SetType("elementwise_max_grad");
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

REGISTER_OPERATOR(elementwise_max, ops::ElementwiseOp,
                  ops::ElementwiseMaxOpMaker, ops::ElementwiseOpInferVarType,
                  ops::ElementwiseMaxGradOpMaker<paddle::framework::OpDesc>,
                  ops::ElementwiseMaxGradOpMaker<paddle::imperative::OpBase>);

REGISTER_OPERATOR(elementwise_max_grad, ops::ElementwiseOpGrad);

REGISTER_OP_CPU_KERNEL(
    elementwise_max,
    ops::ElementwiseMaxKernel<paddle::platform::CPUDeviceContext, float>,
    ops::ElementwiseMaxKernel<paddle::platform::CPUDeviceContext, double>,
    ops::ElementwiseMaxKernel<paddle::platform::CPUDeviceContext, int>,
    ops::ElementwiseMaxKernel<paddle::platform::CPUDeviceContext, int64_t>);
REGISTER_OP_CPU_KERNEL(
    elementwise_max_grad,
    ops::ElementwiseMaxGradKernel<paddle::platform::CPUDeviceContext, float>,
    ops::ElementwiseMaxGradKernel<paddle::platform::CPUDeviceContext, double>,
    ops::ElementwiseMaxGradKernel<paddle::platform::CPUDeviceContext, int>,
    ops::ElementwiseMaxGradKernel<paddle::platform::CPUDeviceContext, int64_t>);
