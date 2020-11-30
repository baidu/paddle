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

#include "paddle/fluid/operators/elementwise/elementwise_mul_op.h"
#include <memory>
#include <string>
#include "paddle/fluid/operators/elementwise/elementwise_op.h"
#include "paddle/fluid/platform/complex128.h"
#include "paddle/fluid/platform/complex64.h"

namespace paddle {
namespace operators {

template <typename T>
struct SameDimsElemwiseMul<
    platform::CPUDeviceContext, T,
    typename std::enable_if<std::is_floating_point<T>::value>::type> {
  void operator()(const framework::ExecutionContext &ctx,
                  const framework::Tensor *x, const framework::Tensor *y,
                  framework::Tensor *z) {
    auto blas = math::GetBlas<platform::CPUDeviceContext, T>(ctx);
    blas.VMUL(x->numel(), x->data<T>(), y->data<T>(), z->data<T>());
  }
};

template <typename T>
struct SameDimsElemwiseMul<
    platform::CPUDeviceContext, T,
    typename std::enable_if<!std::is_floating_point<T>::value>::type> {
  void operator()(const framework::ExecutionContext &ctx,
                  const framework::Tensor *x, const framework::Tensor *y,
                  framework::Tensor *z) {
    auto eigen_x = framework::EigenVector<T>::Flatten(*x);
    auto eigen_y = framework::EigenVector<T>::Flatten(*y);
    auto eigen_z = framework::EigenVector<T>::Flatten(*z);
    auto &place = *ctx.template device_context<platform::CPUDeviceContext>()
                       .eigen_device();
    eigen_z.device(place) = eigen_x * eigen_y;
  }
};

class ElementwiseMulOpMaker : public ElementwiseOpMaker {
 protected:
  std::string GetName() const override { return "Mul"; }
  std::string GetEquation() const override { return "Out = X \\\\odot Y"; }

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
    return "Multiply two tensors element-wise";
  }
};

template <typename T>
class ElementwiseMulOpGradMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> op) const override {
    op->SetType("elementwise_mul_grad");
    op->SetInput("X", this->Input("X"));
    op->SetInput("Y", this->Input("Y"));
    op->SetInput(framework::GradVarName("Out"), this->OutputGrad("Out"));
    op->SetAttrMap(this->Attrs());
    op->SetOutput(framework::GradVarName("X"), this->InputGrad("X"));
    op->SetOutput(framework::GradVarName("Y"), this->InputGrad("Y"));
  }
};

template <typename T>
class ElementwiseMulDoubleGradMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> op) const override {
    op->SetType("elementwise_mul_grad_grad");
    op->SetInput("X", this->Input("X"));
    op->SetInput("Y", this->Input("Y"));
    op->SetInput("DOut", this->Input(framework::GradVarName("Out")));
    op->SetInput("DDX", this->OutputGrad(framework::GradVarName("X")));
    op->SetInput("DDY", this->OutputGrad(framework::GradVarName("Y")));

    op->SetAttrMap(this->Attrs());

    op->SetOutput("DDOut", this->InputGrad(framework::GradVarName("Out")));
    op->SetOutput(framework::GradVarName("X"), this->InputGrad("X"));
    op->SetOutput(framework::GradVarName("Y"), this->InputGrad("Y"));
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(elementwise_mul, ops::ElementwiseMulOp,
                  ops::ElementwiseMulOpMaker, ops::ElementwiseOpInferVarType,
                  ops::ElementwiseMulOpGradMaker<paddle::framework::OpDesc>,
                  ops::ElementwiseMulOpGradMaker<paddle::imperative::OpBase>);
REGISTER_OPERATOR(
    elementwise_mul_grad, ops::ElementwiseOpGrad,
    ops::ElementwiseMulDoubleGradMaker<paddle::framework::OpDesc>,
    ops::ElementwiseMulDoubleGradMaker<paddle::imperative::OpBase>);

REGISTER_OPERATOR(elementwise_mul_grad_grad, ops::ElementwiseOpDoubleGrad,
                  ops::ElementwiseDoubleGradOpInplaceInferer);

REGISTER_OP_CPU_KERNEL(
    elementwise_mul,
    ops::ElementwiseMulKernel<paddle::platform::CPUDeviceContext,
                              paddle::platform::complex64>,
    ops::ElementwiseMulKernel<paddle::platform::CPUDeviceContext,
                              paddle::platform::complex128>,
    ops::ElementwiseMulKernel<paddle::platform::CPUDeviceContext, float>,
    ops::ElementwiseMulKernel<paddle::platform::CPUDeviceContext, double>,
    ops::ElementwiseMulKernel<paddle::platform::CPUDeviceContext, int>,
    ops::ElementwiseMulKernel<paddle::platform::CPUDeviceContext, int64_t>);
REGISTER_OP_CPU_KERNEL(
    elementwise_mul_grad,
    ops::ElementwiseMulGradKernel<paddle::platform::CPUDeviceContext,
                                  paddle::platform::complex64>,
    ops::ElementwiseMulGradKernel<paddle::platform::CPUDeviceContext,
                                  paddle::platform::complex128>,
    ops::ElementwiseMulGradKernel<paddle::platform::CPUDeviceContext, float>,
    ops::ElementwiseMulGradKernel<paddle::platform::CPUDeviceContext, double>,
    ops::ElementwiseMulGradKernel<paddle::platform::CPUDeviceContext, int>,
    ops::ElementwiseMulGradKernel<paddle::platform::CPUDeviceContext, int64_t>);
REGISTER_OP_CPU_KERNEL(
    elementwise_mul_grad_grad,
    ops::ElementwiseMulDoubleGradKernel<paddle::platform::CPUDeviceContext,
                                        paddle::platform::complex64>,
    ops::ElementwiseMulDoubleGradKernel<paddle::platform::CPUDeviceContext,
                                        paddle::platform::complex128>,
    ops::ElementwiseMulDoubleGradKernel<paddle::platform::CPUDeviceContext,
                                        float>,
    ops::ElementwiseMulDoubleGradKernel<paddle::platform::CPUDeviceContext,
                                        double>,
    ops::ElementwiseMulDoubleGradKernel<paddle::platform::CPUDeviceContext,
                                        int>,
    ops::ElementwiseMulDoubleGradKernel<paddle::platform::CPUDeviceContext,
                                        int64_t>);
