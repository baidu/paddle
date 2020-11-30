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

#include "paddle/fluid/operators/elementwise/elementwise_sub_op.h"

#include <string>

#include "paddle/fluid/operators/elementwise/elementwise_op.h"
#include "paddle/fluid/platform/complex128.h"
#include "paddle/fluid/platform/complex64.h"

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
struct SameDimsElemwiseSub<
    platform::CPUDeviceContext, T,
    typename std::enable_if<std::is_floating_point<T>::value>::type> {
  void operator()(const framework::ExecutionContext &ctx,
                  const framework::Tensor *x, const framework::Tensor *y,
                  framework::Tensor *z) {
    auto blas = math::GetBlas<platform::CPUDeviceContext, T>(ctx);
    blas.VSUB(x->numel(), x->data<T>(), y->data<T>(), z->data<T>());
  }
};

template <typename T>
struct SameDimsElemwiseSub<
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
    eigen_z.device(place) = eigen_x - eigen_y;
  }
};
class ElementwiseSubOpMaker : public ElementwiseOpMaker {
 protected:
  std::string GetName() const override { return "Sub"; }
  std::string GetEquation() const override { return "Out = X - Y"; }

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
    return "Substract two tensors element-wise";
  }
};

template <typename T>
class ElementwiseSubDoubleGradMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> op) const override {
    op->SetType("elementwise_sub_grad_grad");
    op->SetInput("Y", this->Input("Y"));
    op->SetInput("DOut", this->Input(framework::GradVarName("Out")));
    op->SetInput("DDX", this->OutputGrad(framework::GradVarName("X")));
    op->SetInput("DDY", this->OutputGrad(framework::GradVarName("Y")));

    op->SetAttrMap(this->Attrs());

    op->SetOutput("DDOut", this->InputGrad(framework::GradVarName("Out")));
  }
};

}  // namespace operators
}  // namespace paddle

REGISTER_ELEMWISE_GRAD_MAKER(elementwise_sub, Sub);
REGISTER_ELEMWISE_EXPLICIT_OP_WITHOUT_GRAD(elementwise_sub, Sub);

namespace ops = paddle::operators;

REGISTER_OPERATOR(
    elementwise_sub_grad, ops::ElementwiseOpGrad,
    ops::ElementwiseGradOpInplaceInferer, ops::ElementwiseGradNoBufVarsInferer,
    ops::ElementwiseSubDoubleGradMaker<paddle::framework::OpDesc>,
    ops::ElementwiseSubDoubleGradMaker<paddle::imperative::OpBase>);
REGISTER_OPERATOR(elementwise_sub_grad_grad,
                  ops::ElementwiseOpDoubleGradWithoutDXDY,
                  ops::ElementwiseDoubleGradOpInplaceInferer,
                  ops::ElementwiseDoubleGradNoBufVarsInferer);

REGISTER_OP_CPU_KERNEL(
    elementwise_sub,
    ops::ElementwiseSubKernel<paddle::platform::CPUDeviceContext,
                              paddle::platform::complex64>,
    ops::ElementwiseSubKernel<paddle::platform::CPUDeviceContext,
                              paddle::platform::complex128>,
    ops::ElementwiseSubKernel<paddle::platform::CPUDeviceContext, float>,
    ops::ElementwiseSubKernel<paddle::platform::CPUDeviceContext, double>,
    ops::ElementwiseSubKernel<paddle::platform::CPUDeviceContext, int>,
    ops::ElementwiseSubKernel<paddle::platform::CPUDeviceContext, int64_t>);
REGISTER_OP_CPU_KERNEL(
    elementwise_sub_grad,
    ops::ElementwiseSubGradKernel<paddle::platform::CPUDeviceContext,
                                  paddle::platform::complex64>,
    ops::ElementwiseSubGradKernel<paddle::platform::CPUDeviceContext,
                                  paddle::platform::complex128>,
    ops::ElementwiseSubGradKernel<paddle::platform::CPUDeviceContext, float>,
    ops::ElementwiseSubGradKernel<paddle::platform::CPUDeviceContext, double>,
    ops::ElementwiseSubGradKernel<paddle::platform::CPUDeviceContext, int>,
    ops::ElementwiseSubGradKernel<paddle::platform::CPUDeviceContext, int64_t>);
REGISTER_OP_CPU_KERNEL(
    elementwise_sub_grad_grad,
    ops::ElementwiseSubDoubleGradKernel<paddle::platform::CPUDeviceContext,
                                        paddle::platform::complex64>,
    ops::ElementwiseSubDoubleGradKernel<paddle::platform::CPUDeviceContext,
                                        paddle::platform::complex128>,
    ops::ElementwiseSubDoubleGradKernel<paddle::platform::CPUDeviceContext,
                                        float>,
    ops::ElementwiseSubDoubleGradKernel<paddle::platform::CPUDeviceContext,
                                        double>,
    ops::ElementwiseSubDoubleGradKernel<paddle::platform::CPUDeviceContext,
                                        int>,
    ops::ElementwiseSubDoubleGradKernel<paddle::platform::CPUDeviceContext,
                                        int64_t>);
