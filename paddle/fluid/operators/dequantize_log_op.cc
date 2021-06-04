/* Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/operators/dequantize_log_op.h"

#include <string>

namespace paddle {
namespace framework {
class InferShapeContext;
class OpDesc;
template <typename T>
class EmptyGradOpMaker;
}  // namespace framework
namespace imperative {
class OpBase;
}  // namespace imperative
namespace platform {
struct CPUPlace;
}  // namespace platform
}  // namespace paddle

namespace paddle {
namespace operators {

template <typename T>
struct DequantizeFunctor<platform::CPUDeviceContext, T> {
  void operator()(const platform::CPUDeviceContext& dev_ctx,
                  const framework::Tensor* in, const framework::Tensor* dict,
                  framework::Tensor* out) {
    const auto* dict_data = dict->data<float>();
    const T* input_data = in->data<T>();
    auto* output_data = out->mutable_data<float>(dev_ctx.GetPlace());
    int ind = in->numel();
    for (size_t i = 0; i < (unsigned)ind; i++) {
      if (input_data[i] < 0) {
        output_data[i] = -dict_data[input_data[i] + 128];
      } else {
        output_data[i] = dict_data[input_data[i]];
      }
    }
  }
};

template struct DequantizeFunctor<platform::CPUDeviceContext, int8_t>;

class DequantizeLogOp : public framework::OperatorWithKernel {
 public:
  DequantizeLogOp(const std::string& type,
                  const framework::VariableNameMap& inputs,
                  const framework::VariableNameMap& outputs,
                  const framework::AttributeMap& attrs)
      : OperatorWithKernel(type, inputs, outputs, attrs) {}

  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE_EQ(ctx->HasInput("X"), true,
                      platform::errors::NotFound(
                          "Input(X) of DequantizeLogOp is not found."));
    PADDLE_ENFORCE_EQ(ctx->HasOutput("Out"), true,
                      platform::errors::NotFound(
                          "Output(Out) of DequantizeLogOp is not found."));

    ctx->ShareDim("X", /*->*/ "Out");
    ctx->ShareLoD("X", /*->*/ "Out");
  }

  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    auto data_type = OperatorWithKernel::IndicateVarDataType(ctx, "X");
    auto type = framework::OpKernelType(data_type, ctx.device_context());
    return type;
  }
};

class DequantizeLogOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X",
             "(int8 Tensor) The input with int8 type is the "
             "low precision tensor.");
    AddInput("Dict", "(float) The Dict in quantization stage.");
    AddOutput("Out",
              "(float32 Tensor) The output is the dequantized high "
              "precision tensor.");
    AddComment(R"DOC(
DequantizeLogOp operator.

This calculation is an opposite operation of QuantizeLogOp:



)DOC");
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
using CPU = paddle::platform::CPUDeviceContext;

REGISTER_OPERATOR(
    dequantize_log, ops::DequantizeLogOp, ops::DequantizeLogOpMaker,
    paddle::framework::EmptyGradOpMaker<paddle::framework::OpDesc>,
    paddle::framework::EmptyGradOpMaker<paddle::imperative::OpBase>);
REGISTER_OP_CPU_KERNEL(dequantize_log, ops::DequantizeLogKernel<CPU, int8_t>);
