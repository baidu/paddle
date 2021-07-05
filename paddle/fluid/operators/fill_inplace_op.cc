/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/framework/op_registry.h"

namespace paddle {
namespace operators {

class Fill_OpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddComment(R"DOC(Fill replace operator
                Fill an tensor inplace with `value` and `shape`. The type of the tensor is specify by
                `dtype`.
                )DOC");
    AddInput("X", "(LoDTensor) The input tensor.");
    AddInput("value", "The float values of tensor, whose dim is one");
  }
};

class Fill_Op : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext *context) const override {
    OP_INOUT_CHECK(context->HasInput("X"), "Input", "X", "Fill_");
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override {
    return framework::OpKernelType(
        OperatorWithKernel::IndicateVarDataType(ctx, "X"), ctx.GetPlace());
  }
};

class Fill_OpVarTypeInference : public framework::VarTypeInference {
 public:
  void operator()(framework::InferVarTypeContext *ctx) const override {}
};

template <typename T>
class Fill_Kernel : public framework::OpKernel<T> {
 public:
  void Compute(const paddle::framework::ExecutionContext &ctx) const override {
    auto *in = ctx.Input<framework::LoDTensor>("X");
    auto *value = ctx.Input<framework::LoDTensor>("value");

    platform::CPUPlace cpu;
    auto *data = const_cast<framework::LoDTensor *>(in)->mutable_data<T>(cpu);
    auto fill_val = *(value->data<T>());
    std::fill(data, data + in->numel(), fill_val);
  }
};

}  // namespace operators
}  // namespace paddle
namespace ops = paddle::operators;
REGISTER_OPERATOR(
    fill_inplace, ops::Fill_Op, ops::Fill_OpMaker, ops::Fill_OpVarTypeInference,
    paddle::framework::EmptyGradOpMaker<paddle::framework::OpDesc>,
    paddle::framework::EmptyGradOpMaker<paddle::imperative::OpBase>);
REGISTER_OP_CPU_KERNEL(fill_inplace, ops::Fill_Kernel<float>,
                       ops::Fill_Kernel<double>, ops::Fill_Kernel<int64_t>,
                       ops::Fill_Kernel<int>,
                       ops::Fill_Kernel<paddle::platform::float16>);
