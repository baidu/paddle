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

#include "paddle/fluid/operators/reduce_ops/reduce_sum_op.h"

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

// NOTE: Input(Out) is unnecessary in reduce_sum_grad, and Input(X) needs no
// buffer

template <typename T>
class ReduceSumOpGradMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> op) const override {
    op->SetType("reduce_sum_grad");
    op->SetInput("X", this->Input("X"));
    op->SetInput(framework::GradVarName("Out"), this->OutputGrad("Out"));
    op->SetAttrMap(this->Attrs());
    op->SetOutput(framework::GradVarName("X"), this->InputGrad("X"));
  }

  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const {
    int in_dtype = ctx.Attr<int>("in_dtype");
    if (in_dtype >= 0) {
      return framework::OpKernelType(
          static_cast<framework::proto::VarType::Type>(in_dtype),
          ctx.GetPlace());
    }
    return framework::OpKernelType(
        framework::OperatorWithKernel::IndicateVarDataType(
            ctx, framework::GradVarName("Out")),
        ctx.GetPlace());
  }
};

template <typename T>
class ReduceSumDoubleOpGradMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> op) const override {
    op->SetInput("X", this->OutputGrad(framework::GradVarName("X")));
    op->SetOutput("Out", this->InputGrad(framework::GradVarName("Out")));
    op->SetAttrMap(this->Attrs());
    op->SetType("reduce_sum");
  }
};

DECLARE_NO_NEED_BUFFER_VARS_INFERER(ReduceSumGradNoNeedBufferVarInferer, "X");
class ReduceSumVarTypeInference : public paddle::framework::VarTypeInference {
 public:
  void operator()(paddle::framework::InferVarTypeContext* ctx) const override {
    auto data_type = static_cast<paddle::framework::proto::VarType::Type>(
        BOOST_GET_CONST(int, ctx->GetAttr("out_dtype")));
    if (data_type >= 0) {
      ctx->SetOutputDataType("Out", data_type);
    }
  }
};

}  // namespace operators
}  // namespace paddle

class ReduceSumOpMaker : public ops::ReduceOpMaker {
 protected:
  std::string GetName() const override { return "reduce_sum"; }
  std::string GetOpType() const override { return "Reduce reduce_sum"; }
};

REGISTER_OPERATOR(reduce_sum, ops::ReduceOp, ReduceSumOpMaker,
                  ops::ReduceSumVarTypeInference,
                  ops::ReduceSumOpGradMaker<paddle::framework::OpDesc>,
                  ops::ReduceSumOpGradMaker<paddle::imperative::OpBase>);
REGISTER_OPERATOR(reduce_sum_grad, ops::ReduceGradOp,
                  ops::ReduceSumDoubleOpGradMaker<paddle::framework::OpDesc>,
                  ops::ReduceSumDoubleOpGradMaker<paddle::imperative::OpBase>,
                  ops::ReduceSumGradNoNeedBufferVarInferer);

REGISTER_OP_CPU_KERNEL(
    reduce_sum,
    ops::ReduceKernel<paddle::platform::CPUDeviceContext, bool,
                      ops::SumFunctor>,
    ops::ReduceKernel<paddle::platform::CPUDeviceContext, float,
                      ops::SumFunctor>,
    ops::ReduceKernel<paddle::platform::CPUDeviceContext, double,
                      ops::SumFunctor>,
    ops::ReduceKernel<paddle::platform::CPUDeviceContext, int, ops::SumFunctor>,
    ops::ReduceKernel<paddle::platform::CPUDeviceContext, int64_t,
                      ops::SumFunctor>,
    ops::ReduceKernel<paddle::platform::CPUDeviceContext,
                      paddle::platform::complex<float>, ops::SumFunctor>,
    ops::ReduceKernel<paddle::platform::CPUDeviceContext,
                      paddle::platform::complex<double>,

                      ops::SumFunctor>);

template <typename T>
using CPUReduceSumGradKernel =
    ops::ReduceSumGradKernel<paddle::platform::CPUDeviceContext, T,
                             ops::SumGradFunctor, true>;

REGISTER_OP_CPU_KERNEL(
    reduce_sum_grad, CPUReduceSumGradKernel<bool>,
    CPUReduceSumGradKernel<float>, CPUReduceSumGradKernel<double>,
    CPUReduceSumGradKernel<int>, CPUReduceSumGradKernel<int64_t>,
    CPUReduceSumGradKernel<paddle::platform::complex<float>>,
    CPUReduceSumGradKernel<paddle::platform::complex<double>>);
