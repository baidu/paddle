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

#include "paddle/fluid/operators/elementwise/elementwise_div_op.h"
#include <memory>
#include <string>
#include "paddle/fluid/operators/elementwise/elementwise_op.h"
#include "paddle/fluid/platform/complex128.h"
#include "paddle/fluid/platform/complex64.h"

namespace paddle {
namespace operators {

template <typename T>
struct SameDimsElemwiseDiv<
    platform::CPUDeviceContext, T,
    typename std::enable_if<std::is_floating_point<T>::value>::type> {
  void operator()(const framework::ExecutionContext &ctx,
                  const framework::Tensor *x, const framework::Tensor *y,
                  framework::Tensor *z) {
    auto blas = math::GetBlas<platform::CPUDeviceContext, T>(ctx);
    blas.VDIV(x->numel(), x->data<T>(), y->data<T>(), z->data<T>());
  }
};

// use default div function for int32/int64 type because of divison zero
// checking.
template <typename T>
struct SameDimsElemwiseDiv<
    platform::CPUDeviceContext, T,
    typename std::enable_if<!std::is_floating_point<T>::value>::type> {
  void operator()(const framework::ExecutionContext &ctx,
                  const framework::Tensor *x, const framework::Tensor *y,
                  framework::Tensor *z) {
    default_elementwise_div<platform::CPUDeviceContext, T>(ctx, x, y, z);
  }
};

class ElementwiseDivOpMaker : public ElementwiseOpMaker {
 protected:
  std::string GetName() const override { return "Div"; }
  std::string GetEquation() const override { return "Out = X / Y"; }

  void AddInputX() override {
    AddInput("X",
             "(Variable), Tensor or LoDTensor of any dimensions. Its dtype "
             "should be int32, int64, float32, float64.");
  }

  void AddInputY() override {
    AddInput("Y",
             "(Variable), Tensor or LoDTensor of any dimensions. Its dtype "
             "should be int32, int64, float32, float64.");
  }

  std::string GetOpFuntionality() const override {
    return "Divide two tensors element-wise";
  }
};

template <typename T>
class ElementwiseDivGradOpMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> op) const override {
    op->SetType("elementwise_div_grad");
    op->SetInput("X", this->Input("X"));
    op->SetInput("Y", this->Input("Y"));
    op->SetInput("Out", this->Output("Out"));
    op->SetInput(framework::GradVarName("Out"), this->OutputGrad("Out"));
    op->SetOutput(framework::GradVarName("X"), this->InputGrad("X"));
    op->SetOutput(framework::GradVarName("Y"), this->InputGrad("Y"));
    op->SetAttrMap(this->Attrs());
  }
};

template <typename T>
class ElementwiseDivDoubleGradMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> op) const override {
    op->SetType("elementwise_div_grad_grad");
    op->SetInput("Y", this->Input("Y"));
    op->SetInput("Out", this->Input("Out"));
    op->SetInput("DDX", this->OutputGrad(framework::GradVarName("X")));
    op->SetInput("DDY", this->OutputGrad(framework::GradVarName("Y")));
    op->SetInput("DX", this->Output(framework::GradVarName("X")));

    op->SetAttrMap(this->Attrs());

    op->SetOutput(framework::GradVarName("Y"), this->InputGrad("Y"));
    op->SetOutput("DOut", this->InputGrad("Out"));
    op->SetOutput("DDOut", this->InputGrad(framework::GradVarName("Out")));
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OPERATOR(elementwise_div, ops::ElementwiseOp,
                  ops::ElementwiseDivOpMaker, ops::ElementwiseOpInferVarType,
                  ops::ElementwiseDivGradOpMaker<paddle::framework::OpDesc>,
                  ops::ElementwiseDivGradOpMaker<paddle::imperative::OpBase>);

REGISTER_OPERATOR(
    elementwise_div_grad, ops::ElementwiseOpGrad,
    ops::ElementwiseDivDoubleGradMaker<paddle::framework::OpDesc>,
    ops::ElementwiseDivDoubleGradMaker<paddle::imperative::OpBase>);

REGISTER_OPERATOR(elementwise_div_grad_grad, ops::ElementwiseDivOpDoubleGrad,
                  ops::ElementwiseDoubleGradOpInplaceInferer);

REGISTER_OP_CPU_KERNEL(
    elementwise_div,
    ops::ElementwiseDivKernel<paddle::platform::CPUDeviceContext,
                              paddle::platform::complex64>,
    ops::ElementwiseDivKernel<paddle::platform::CPUDeviceContext,
                              paddle::platform::complex128>,
    ops::ElementwiseDivKernel<paddle::platform::CPUDeviceContext, float>,
    ops::ElementwiseDivKernel<paddle::platform::CPUDeviceContext, double>,
    ops::ElementwiseDivKernel<paddle::platform::CPUDeviceContext, int>,
    ops::ElementwiseDivKernel<paddle::platform::CPUDeviceContext, int64_t>);
REGISTER_OP_CPU_KERNEL(
    elementwise_div_grad,
    ops::ElementwiseDivGradKernel<paddle::platform::CPUDeviceContext,
                                  paddle::platform::complex64>,
    ops::ElementwiseDivGradKernel<paddle::platform::CPUDeviceContext,
                                  paddle::platform::complex128>,
    ops::ElementwiseDivGradKernel<paddle::platform::CPUDeviceContext, float>,
    ops::ElementwiseDivGradKernel<paddle::platform::CPUDeviceContext, double>,
    ops::ElementwiseDivGradKernel<paddle::platform::CPUDeviceContext, int>,
    ops::ElementwiseDivGradKernel<paddle::platform::CPUDeviceContext, int64_t>);

REGISTER_OP_CPU_KERNEL(
    elementwise_div_grad_grad,
    ops::ElementwiseDivDoubleGradKernel<paddle::platform::CPUDeviceContext,
                                        paddle::platform::complex64>,
    ops::ElementwiseDivDoubleGradKernel<paddle::platform::CPUDeviceContext,
                                        paddle::platform::complex128>,
    ops::ElementwiseDivDoubleGradKernel<paddle::platform::CPUDeviceContext,
                                        float>,
    ops::ElementwiseDivDoubleGradKernel<paddle::platform::CPUDeviceContext,
                                        double>,
    ops::ElementwiseDivDoubleGradKernel<paddle::platform::CPUDeviceContext,
                                        int>,
    ops::ElementwiseDivDoubleGradKernel<paddle::platform::CPUDeviceContext,
                                        int64_t>);
