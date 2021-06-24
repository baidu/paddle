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

#include "paddle/fluid/operators/concat_op.h"

#include <memory>
#include <string>
#include <vector>

#ifdef PADDLE_WITH_MKLDNN
#include <paddle/fluid/platform/mkldnn_helper.h>
#endif

namespace paddle {
namespace operators {
using Tensor = framework::Tensor;

class ConcatOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext *ctx) const override {
    OP_INOUT_CHECK(ctx->HasInputs("X"), "Input", "X", "Concat");
    OP_INOUT_CHECK(ctx->HasOutput("Out"), "Output", "Out", "Concat");

    auto inputs_dims = ctx->GetInputsDim("X");

    const size_t inputs_num = inputs_dims.size();
    PADDLE_ENFORCE_GT(
        inputs_num, static_cast<size_t>(0),
        platform::errors::InvalidArgument(
            "The number of input tensors in concat op should > 0. But "
            "received inputs' length is 0."));
    if (inputs_num == 1) {
      VLOG(3) << "Warning: concat op have only one input, may waste memory";
    }

    if (ctx->HasInput("AxisTensor")) {
      auto out_dims =
          framework::make_ddim(std::vector<int>(inputs_dims[0].size(), -1));
      ctx->SetOutputDim("Out", out_dims);
      ctx->ShareLoD("X", /*->*/ "Out");
    } else {
      size_t axis =
          ComputeAxis(static_cast<int64_t>(ctx->Attrs().Get<int>("axis")),
                      static_cast<int64_t>(inputs_dims[0].size()));
      framework::DDim out_dims =
          ComputeAndCheckShape(ctx->IsRuntime(), inputs_dims, axis);
      if (out_dims[axis] < 0) {
        out_dims[axis] = -1;
      }
      ctx->SetOutputDim("Out", out_dims);
      ctx->ShareLoD("X", /*->*/ "Out");
    }
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override {
    auto inputs = ctx.MultiInput<Tensor>("X");
    auto input_data_type = framework::proto::VarType::Type(0);
    bool flag = 0;
    for (auto *input : inputs) {
      if (input->IsInitialized() && input->numel() > 0) {
        input_data_type = input->type();
        flag = 1;
        break;
      }
    }
    if (flag == 0) {
      PADDLE_THROW(platform::errors::InvalidArgument(
          "All Inputs of Concat OP are Empty!"));
    }
#ifdef PADDLE_WITH_MKLDNN
    if (this->CanMKLDNNBeUsed(ctx, input_data_type)) {
      return framework::OpKernelType(input_data_type, ctx.GetPlace(),
                                     framework::DataLayout::kMKLDNN,
                                     framework::LibraryType::kMKLDNN);
    }
#endif
    return framework::OpKernelType(input_data_type, ctx.GetPlace());
  }

  framework::OpKernelType GetKernelTypeForVar(
      const std::string &var_name, const Tensor &tensor,
      const framework::OpKernelType &expected_kernel_type) const override {
    if (var_name == "AxisTensor") {
      return expected_kernel_type;
    }
    return framework::OpKernelType(expected_kernel_type.data_type_,
                                   tensor.place(), tensor.layout());
  }
};

class ConcatOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "Input tensors of concat operator.").AsDuplicable();
    AddOutput("Out", "Output tensor of concat operator.");
    AddAttr<bool>(
        "use_mkldnn",
        "(bool, default false) Indicates if MKL-DNN kernel will be used")
        .SetDefault(false);
    AddAttr<int>("axis",
                 "The axis along which the input tensors will be concatenated."
                 "The axis could also be negative numbers. Negative axis is "
                 "interpreted as counting from the end of the rank."
                 "i.e., axis + rank(X) th dimension.")
        .SetDefault(0);
    AddInput("AxisTensor",
             "(Tensor) The axis along which the input tensors will be "
             "concatenated.  "
             "It has higher priority than Attr(axis). "
             "The shape of AxisTensor must be [1].")
        .AsDispensable();
    AddAttr<bool>(
        "use_quantizer",
        "(bool, default false) "
        "This parameter is no longer used. Use 'mkldnn_data_type' instead.")
        .SetDefault(false);
    AddAttr<std::string>(
        "mkldnn_data_type",
        "(string, default \"float32\"). Data type of mkldnn kernel")
        .SetDefault("float32")
        .InEnum({"float32", "int8", "bfloat16"});
    AddComment(R"DOC(
Concat Operator.

Concatenate the input tensors along dimension axis.
Examples:
  Input[0] = [[1,2],[3,4]]
  Input[1] = [[5,6]]
  axis = 0
  Output = [[1,2],
            [3,4],
            [5,6]]

)DOC");
  }
};

class ConcatOpGrad : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext *ctx) const override {
    auto in_x = "X";
    auto out_x_g_n = framework::GradVarName(in_x);
    ctx->SetOutputsDim(out_x_g_n, ctx->GetInputsDim(in_x));

    ctx->ShareAllLoD(in_x, out_x_g_n);
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override {
    return framework::OpKernelType(OperatorWithKernel::IndicateVarDataType(
                                       ctx, framework::GradVarName("Out")),
                                   ctx.GetPlace());
  }

  framework::OpKernelType GetKernelTypeForVar(
      const std::string &var_name, const Tensor &tensor,
      const framework::OpKernelType &expected_kernel_type) const override {
    if (var_name == "AxisTensor") {
      return expected_kernel_type;
    }
    return framework::OpKernelType(expected_kernel_type.data_type_,
                                   tensor.place(), tensor.layout());
  }
};

DECLARE_NO_NEED_BUFFER_VARS_INFERER(ConcatOpGradNoNeedBufferVarInferer, "X");

template <typename T>
class ConcatGradOpMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> op) const override {
    op->SetType("concat_grad");
    op->SetInput("X", this->Input("X"));
    if (this->HasInput("AxisTensor")) {
      op->SetInput("AxisTensor", this->Input("AxisTensor"));
    }
    op->SetInput(framework::GradVarName("Out"), this->OutputGrad("Out"));
    op->SetOutput(framework::GradVarName("X"), this->InputGrad("X", false));
    op->SetAttrMap(this->Attrs());
  }
};

template <typename T>
class ConcatDoubleGradOpMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> grad_op) const override {
    grad_op->SetType("concat");
    grad_op->SetInput("X", this->OutputGrad(framework::GradVarName("X")));
    grad_op->SetOutput("Out", this->InputGrad(framework::GradVarName("Out")));
    grad_op->SetAttrMap(this->Attrs());
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(concat, ops::ConcatOp, ops::ConcatOpMaker,
                  ops::ConcatGradOpMaker<paddle::framework::OpDesc>,
                  ops::ConcatGradOpMaker<paddle::imperative::OpBase>);
REGISTER_OPERATOR(concat_grad, ops::ConcatOpGrad,
                  ops::ConcatDoubleGradOpMaker<paddle::framework::OpDesc>,
                  ops::ConcatDoubleGradOpMaker<paddle::imperative::OpBase>,
                  ops::ConcatOpGradNoNeedBufferVarInferer);
REGISTER_OP_CPU_KERNEL(
    concat, ops::ConcatKernel<paddle::platform::CPUDeviceContext, double>,
    ops::ConcatKernel<paddle::platform::CPUDeviceContext, float>,
    ops::ConcatKernel<paddle::platform::CPUDeviceContext, bool>,
    ops::ConcatKernel<paddle::platform::CPUDeviceContext, int64_t>,
    ops::ConcatKernel<paddle::platform::CPUDeviceContext,
                      paddle::platform::float16>,
    ops::ConcatKernel<paddle::platform::CPUDeviceContext, int>,
    ops::ConcatKernel<paddle::platform::CPUDeviceContext, uint8_t>);
REGISTER_OP_CPU_KERNEL(
    concat_grad,
    ops::ConcatGradKernel<paddle::platform::CPUDeviceContext, double>,
    ops::ConcatGradKernel<paddle::platform::CPUDeviceContext, float>,
    ops::ConcatGradKernel<paddle::platform::CPUDeviceContext, bool>,
    ops::ConcatGradKernel<paddle::platform::CPUDeviceContext, int64_t>,
    ops::ConcatGradKernel<paddle::platform::CPUDeviceContext,
                          paddle::platform::float16>,
    ops::ConcatGradKernel<paddle::platform::CPUDeviceContext, int>,
    ops::ConcatGradKernel<paddle::platform::CPUDeviceContext, uint8_t>);
