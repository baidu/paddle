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

#pragma once

#include <algorithm>  // for max
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "paddle/fluid/framework/data_layout.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/op_version_registry.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/operators/common_infer_shape_functions.h"
#include "paddle/fluid/operators/elementwise/elementwise_op_function.h"

#ifdef PADDLE_WITH_MKLDNN
#include "paddle/fluid/platform/mkldnn_helper.h"
#endif

namespace paddle {
namespace operators {

class ElementwiseOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  using Tensor = framework::Tensor;

  void InferShape(framework::InferShapeContext *ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("X"), "Input", "X", "ElementwiseOp");
    OP_INOUT_CHECK(ctx->HasInput("Y"), "Input", "Y", "ElementwiseOp");
    OP_INOUT_CHECK(ctx->HasOutput("Out"), "Output", "Out", "ElementwiseOp");

    PADDLE_ENFORCE_EQ(
        ctx->GetInputsVarType("Y").front(),
        framework::proto::VarType::LOD_TENSOR,
        platform::errors::InvalidArgument(
            "The input var's type should be LoDTensor, but the "
            "received is %s [%s].",
            ctx->GetInputsVarType("Y").front(), ctx->Inputs("Y").front()));

    if (ctx->GetInputsVarType("X").front() ==
        framework::proto::VarType::SELECTED_ROWS) {
      PADDLE_ENFORCE_EQ(
          ctx->GetInputDim("Y").size(), 1u,
          platform::errors::InvalidArgument(
              "For elementwise_op, if X is Sparse(VarType.SELECTED_ROWS"
              "), Y must be scalar, the size of Y should be 1. "
              "But reveived the size of Y = %s.",
              ctx->GetInputDim("Y").size()));
      PADDLE_ENFORCE_EQ(
          ctx->GetInputDim("Y")[0], 1,
          platform::errors::InvalidArgument(
              "For elementwise_op, if X is Sparse(VarType.SELECTED_ROWS"
              "), Y must be scalar, the first dimension of Y should be 1. "
              "But reveived the first dimension of Y = %s.",
              ctx->GetInputDim("Y")[0]));
    } else if (ctx->GetInputsVarType("X").front() !=
               framework::proto::VarType::LOD_TENSOR) {
      PADDLE_THROW(platform::errors::InvalidArgument(
          "Input X's type[%s] is not supported by elementwise_op. Please set "
          "its type to LOD_TENSOR.",
          ctx->GetInputsVarType("X").front()));
    }

    if (ctx->GetInputDim("X") == ctx->GetInputDim("Y")) {
      ctx->ShareDim("X", /*->*/ "Out");
      ctx->ShareLoD("X", /*->*/ "Out");
    } else {
      auto x_dims = ctx->GetInputDim("X");
      auto y_dims = ctx->GetInputDim("Y");
      int max_dim = std::max(x_dims.size(), y_dims.size());
      int axis = ctx->Attrs().Get<int>("axis");
      PADDLE_ENFORCE_EQ((axis >= (-1 * max_dim)) && (axis < max_dim), true,
                        platform::errors::InvalidArgument(
                            "The axis range must be [%s, %s), but axis is %s. "
                            "Please set the axis again.",
                            -1 * max_dim, max_dim, axis));
      axis = (axis < 0 ? (std::abs(x_dims.size() - y_dims.size()) + axis + 1)
                       : axis);
      std::vector<int> x_dims_array(max_dim);
      std::vector<int> y_dims_array(max_dim);
      std::vector<int> out_dims_array(max_dim);
      GetBroadcastDimsArrays(x_dims, y_dims, x_dims_array.data(),
                             y_dims_array.data(), out_dims_array.data(),
                             max_dim, axis);
      ctx->SetOutputDim("Out", framework::make_ddim(out_dims_array));
      // to do
      ctx->ShareLoD("X", /*->*/ "Out");
    }
  }

  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override {
    auto input_data_type =
        OperatorWithKernel::IndicateOrPromoteVarDataTypes(ctx, "X", "Y");

/*
#ifdef PADDLE_WITH_MKLDNN
    if (this->CanMKLDNNBeUsed(ctx, input_data_type)) {
      return framework::OpKernelType(input_data_type, ctx.GetPlace(),
                                     framework::DataLayout::kMKLDNN,
                                     framework::LibraryType::kMKLDNN);
    }
#endif
*/
    return framework::OpKernelType(input_data_type, ctx.GetPlace());
  }

  framework::OpKernelType GetKernelTypeForVar(
      const std::string &var_name, const framework::Tensor &tensor,
      const framework::OpKernelType &expected_kernel_type) const {
    if (framework::IsComplexType(expected_kernel_type.data_type_)) {
      // only promote inputs’s types when contains complex input
      return framework::OpKernelType(tensor.type(), tensor.place(),
                                     tensor.layout());
    } else {
      return framework::OpKernelType(expected_kernel_type.data_type_,
                                     tensor.place(), tensor.layout());
    }
  }
};

class ElementwiseOpInferVarType
    : public framework::PassInDtypeAndVarTypeToOutput {
 protected:
  std::unordered_map<std::string, std::string> &GetInputOutputWithSameType()
      const override {
    static std::unordered_map<std::string, std::string> m{{"X", /*->*/ "Out"}};
    return m;
  }
};

class ElementwiseOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() final {
    AddInputX();
    AddInputY();
    AddOpOutput();

    AddAttr<int>("axis",
                 "(int, default -1). If X.dimension != Y.dimension,"
                 "Y.dimension must be a subsequence of x.dimension. And axis "
                 "is the start dimension index "
                 "for broadcasting Y onto X. ")
        .SetDefault(-1);
    AddAttr<bool>("use_mkldnn", "(bool, default false). Used by MKLDNN.")
        .SetDefault(false);
    AddAttr<std::string>("x_data_format", "This parameter is no longer used.")
        .SetDefault("");
    AddAttr<std::string>("y_data_format", "This parameter is no longer used.")
        .SetDefault("");
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
    /* int8 parameters */
    AddAttr<float>("Scale_x",
                   "(float, default 1.0f), The quantize scale of X tensor")
        .SetDefault(1.0f);
    AddAttr<float>("Scale_y",
                   "(float, default 1.0f), The quantize scale of Y tensor")
        .SetDefault(1.0f);
    AddAttr<float>("Scale_out",
                   "(float, default 1.0f), The quantize scale of output data")
        .SetDefault(1.0f);
    AddOpComment();
  }

 protected:
  virtual void AddInputX() {
    AddInput("X", "(Tensor), The first input tensor of elementwise op.");
  }
  virtual void AddInputY() {
    AddInput("Y", "(Tensor), The second input tensor of elementwise op.");
  }
  virtual void AddOpOutput() {
    AddOutput("Out",
              "N-dimension tensor. A location into which the result is stored. "
              "It's dimension "
              "equals with x");
  }
  virtual void AddOpComment() { AddComment(GetCommentExamples()); }

  virtual std::string GetOpFuntionality() const { return ""; }

  virtual std::string GetName() const = 0;
  virtual std::string GetEquation() const = 0;

  std::string GetCommentExamples() const {
    return string::Sprintf(R"DOC(
Elementwise %s Operator.

%s

The equation is:

$$%s$$

- $X$: a tensor of any dimension.
- $Y$: a tensor whose dimensions must be less than or equal to the dimensions of $X$.

There are two cases for this operator:

1. The shape of $Y$ is the same with $X$.
2. The shape of $Y$ is a continuous subsequence of $X$.

For case 2:

1. Broadcast $Y$ to match the shape of $X$, where $axis$ is the start dimension index
   for broadcasting $Y$ onto $X$.
2. If $axis$ is -1 (default), $axis = rank(X) - rank(Y)$.
3. The trailing dimensions of size 1 for $Y$ will be ignored for the consideration of
   subsequence, such as shape(Y) = (2, 1) => (2).

For example:

  .. code-block:: text

    shape(X) = (2, 3, 4, 5), shape(Y) = (,)
    shape(X) = (2, 3, 4, 5), shape(Y) = (5,)
    shape(X) = (2, 3, 4, 5), shape(Y) = (4, 5), with axis=-1(default) or axis=2
    shape(X) = (2, 3, 4, 5), shape(Y) = (3, 4), with axis=1
    shape(X) = (2, 3, 4, 5), shape(Y) = (2), with axis=0
    shape(X) = (2, 3, 4, 5), shape(Y) = (2, 1), with axis=0

)DOC",
                           GetName(), GetOpFuntionality(), GetEquation());
  }
};

class ElementwiseOpGrad : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;
  using Tensor = framework::Tensor;

  void InferShape(framework::InferShapeContext *ctx) const override {
    auto out_grad_name = framework::GradVarName("Out");
    OP_INOUT_CHECK(ctx->HasInput("Y"), "Input", "Y", "ElementwiseOpGrad");
    OP_INOUT_CHECK(ctx->HasInput(out_grad_name), "Input", out_grad_name,
                   "ElementwiseOpGrad");
    auto x_grad_name = framework::GradVarName("X");
    auto y_grad_name = framework::GradVarName("Y");
    if (ctx->HasOutput(x_grad_name)) {
      ctx->ShareDim("X", /*->*/ x_grad_name);
      ctx->ShareLoD("X", /*->*/ x_grad_name);
    }
    if (ctx->HasOutput(y_grad_name)) {
      ctx->ShareDim("Y", /*->*/ y_grad_name);
      ctx->ShareLoD("Y", /*->*/ y_grad_name);
    }
  }

  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override {
    auto input_data_type = OperatorWithKernel::IndicateVarDataType(
        ctx, framework::GradVarName("Out"));

#ifdef PADDLE_WITH_MKLDNN
    // If broadcasting is needed, use native implementation
    auto CanMKLDNNElementwiseGradBeUsed = [&]() {
      auto dx_dims = ctx.Input<Tensor>("X")->dims();
      auto dy_dims = ctx.Input<Tensor>("Y")->dims();
      // No broadcast or broadcasting of data on inner dims is supported
      return (dx_dims[dx_dims.size() - 1] == dy_dims[dy_dims.size() - 1]);
    };

    if (this->CanMKLDNNBeUsed(ctx, input_data_type) &&
        CanMKLDNNElementwiseGradBeUsed()) {
      return framework::OpKernelType(input_data_type, ctx.GetPlace(),
                                     framework::DataLayout::kMKLDNN,
                                     framework::LibraryType::kMKLDNN);
    }
#endif
    return framework::OpKernelType(input_data_type, ctx.GetPlace());
  }

  framework::OpKernelType GetKernelTypeForVar(
      const std::string &var_name, const framework::Tensor &tensor,
      const framework::OpKernelType &expected_kernel_type) const override {
    if (framework::IsComplexType(expected_kernel_type.data_type_)) {
      // only promote inputs’s types when contains complex input
      return framework::OpKernelType(tensor.type(), tensor.place(),
                                     tensor.layout());
    } else {
      return framework::OpKernelType(expected_kernel_type.data_type_,
                                     tensor.place(), tensor.layout());
    }
  }
};

class ElementwiseOpDoubleGrad : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;
  using Tensor = framework::Tensor;

  void InferShape(framework::InferShapeContext *ctx) const override {
    auto x_grad_name = framework::GradVarName("X");
    auto y_grad_name = framework::GradVarName("Y");
    if (ctx->HasOutput(x_grad_name)) {
      ctx->ShareDim("X", x_grad_name);
      ctx->ShareLoD("X", x_grad_name);
    }
    if (ctx->HasOutput(y_grad_name)) {
      ctx->ShareDim("Y", y_grad_name);
      ctx->ShareLoD("Y", y_grad_name);
    }
    if (ctx->HasOutput("DDOut")) {
      ctx->ShareDim("DOut", "DDOut");
      ctx->ShareLoD("DOut", "DDOut");
    }
  }

  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override {
    auto input_data_type = OperatorWithKernel::IndicateVarDataType(ctx, "DOut");

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
      const std::string &var_name, const framework::Tensor &tensor,
      const framework::OpKernelType &expected_kernel_type) const {
    if (framework::IsComplexType(expected_kernel_type.data_type_)) {
      // only promote inputs’s types when contains complex input
      return framework::OpKernelType(tensor.type(), tensor.place(),
                                     tensor.layout());
    } else {
      return framework::OpKernelType(expected_kernel_type.data_type_,
                                     tensor.place(), tensor.layout());
    }
  }
};

class ElementwiseOpDoubleGradWithoutDXDY
    : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;
  using Tensor = framework::Tensor;

  void InferShape(framework::InferShapeContext *ctx) const override {
    if (ctx->HasOutput("DDOut")) {
      ctx->ShareDim("DOut", "DDOut");
      ctx->ShareLoD("DOut", "DDOut");
    }
  }

  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override {
    framework::proto::VarType::Type input_data_type;
    if (ctx.HasInput("DDX") == false) {
      OP_INOUT_CHECK(ctx.HasInput("DDY"), "Input", "DDY",
                     "ElementwiseOpDoubleGradWithoutDXDY");
      input_data_type = OperatorWithKernel::IndicateVarDataType(ctx, "DDY");
    } else if (ctx.HasInput("DDY") == false) {
      OP_INOUT_CHECK(ctx.HasInput("DDX"), "Input", "DDX",
                     "ElementwiseOpDoubleGradWithoutDXDY");
      input_data_type = OperatorWithKernel::IndicateVarDataType(ctx, "DDX");
    } else {
      input_data_type =
          OperatorWithKernel::IndicateOrPromoteVarDataTypes(ctx, "DDX", "DDY");
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
      const std::string &var_name, const framework::Tensor &tensor,
      const framework::OpKernelType &expected_kernel_type) const {
    if (framework::IsComplexType(expected_kernel_type.data_type_)) {
      // only promote inputs’s types when contains complex input
      return framework::OpKernelType(tensor.type(), tensor.place(),
                                     tensor.layout());
    } else {
      return framework::OpKernelType(expected_kernel_type.data_type_,
                                     tensor.place(), tensor.layout());
    }
  }
};

template <typename T>
class ElemwiseGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &context) const override {
    auto *dx =
        context.Output<framework::LoDTensor>(framework::GradVarName("X"));
    if (dx != nullptr) {
      auto &dout =
          *context.Input<framework::LoDTensor>(framework::GradVarName("Out"));
      dx->set_lod(dout.lod());
    }
  }
};

DECLARE_INPLACE_OP_INFERER(ElementwiseOpInplaceInferer, {"X", "Out"});
DECLARE_INPLACE_OP_INFERER(ElementwiseGradOpInplaceInferer,
                           {framework::GradVarName("Out"),
                            framework::GradVarName("X")});
DECLARE_INPLACE_OP_INFERER(ElementwiseDoubleGradOpInplaceInferer,
                           {"DDX", "DDOut"});

DECLARE_NO_NEED_BUFFER_VARS_INFERER(ElementwiseGradNoBufVarsInferer, "X", "Y");
DECLARE_NO_NEED_BUFFER_VARS_INFERER(ElementwiseDoubleGradNoBufVarsInferer, "Y",
                                    "DOut");

}  // namespace operators
}  // namespace paddle
#define REGISTER_ELEMWISE_GRAD_MAKER(kernel_type, op_name)              \
  template <typename T>                                                 \
  class kernel_type##GradMaker                                          \
      : public paddle::framework::SingleGradOpMaker<T> {                \
   public:                                                              \
    using ::paddle::framework::SingleGradOpMaker<T>::SingleGradOpMaker; \
                                                                        \
   protected:                                                           \
    void Apply(::paddle::framework::GradOpPtr<T> op) const override {   \
      op->SetType(#kernel_type "_grad");                                \
      op->SetInput("X", this->Input("X"));                              \
      op->SetInput("Y", this->Input("Y"));                              \
      op->SetInput(::paddle::framework::GradVarName("Out"),             \
                   this->OutputGrad("Out"));                            \
      op->SetAttrMap(this->Attrs());                                    \
      op->SetOutput(::paddle::framework::GradVarName("X"),              \
                    this->InputGrad("X"));                              \
      op->SetOutput(::paddle::framework::GradVarName("Y"),              \
                    this->InputGrad("Y"));                              \
    }                                                                   \
  }

#define REGISTER_ELEMWISE_EXPLICIT_OP_WITHOUT_GRAD(op_type, op_name)    \
  REGISTER_OPERATOR(op_type, ::paddle::operators::ElementwiseOp,        \
                    ::paddle::operators::Elementwise##op_name##OpMaker, \
                    ::paddle::operators::ElementwiseOpInferVarType,     \
                    op_type##GradMaker<::paddle::framework::OpDesc>,    \
                    op_type##GradMaker<::paddle::imperative::OpBase>,   \
                    ::paddle::operators::ElementwiseOpInplaceInferer);
