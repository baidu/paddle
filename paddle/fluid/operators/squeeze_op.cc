/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/operators/squeeze_op.h"

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "paddle/fluid/framework/op_registry.h"

namespace paddle {
namespace operators {

framework::DDim GetOutputShape(const std::vector<int> squeeze_dims,
                               const framework::DDim &in_dims,
                               bool is_runtime) {
  size_t num_squeeze_dims = squeeze_dims.size();
  std::vector<bool> should_squeeze(in_dims.size(), false);

  // Mark dimensions need to be squeezed.
  if (num_squeeze_dims == 0) {
    for (int i = 0; i < in_dims.size(); ++i) {
      if (in_dims[i] == 1) {
        should_squeeze[i] = true;
      }
    }
  } else {
    for (size_t i = 0; i < num_squeeze_dims; ++i) {
      int current = squeeze_dims[i] < 0 ? squeeze_dims[i] + in_dims.size()
                                        : squeeze_dims[i];

      PADDLE_ENFORCE_GE(
          current, 0,
          platform::errors::InvalidArgument(
              "Each axis in Attr(axes) should be in the range of [%d, %d]"
              "But current axis is:%d, input tensor's shape = [%s].",
              -in_dims.size(), in_dims.size() - 1, current, in_dims));
      PADDLE_ENFORCE_LT(
          current, in_dims.size(),
          platform::errors::InvalidArgument(
              "Each axis in Attr(axes) should be in the range of [%d, %d]"
              "But current axis is:%d, input tensor's shape = [%s].",
              -in_dims.size(), in_dims.size() - 1, current, in_dims));

      if (!should_squeeze[current]) {
        if (is_runtime) {
          // At run time, dim of 1 is allowed to squeeze
          if (in_dims[current] == 1) {
            should_squeeze[current] = true;
          }
        } else {
          // At compile time, dim of -1 or 1 is allowed to squeeze
          if (in_dims[current] == 1 || in_dims[current] == -1) {
            should_squeeze[current] = true;
          }
        }
      }
    }
  }
  // Make output dimensions
  std::vector<int64_t> output_shape;
  for (int i = 0; i < in_dims.size(); ++i) {
    if (!should_squeeze[i]) {
      output_shape.push_back(in_dims[i]);
    }
  }
  return framework::make_ddim(output_shape);
}

class SqueezeOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext *ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("X"), "Input", "X", "Squeeze");
    OP_INOUT_CHECK(ctx->HasOutput("Out"), "Output", "Out", "Squeeze");

    const auto &x_dims = ctx->GetInputDim("X");
    // Check input tensor dims (<6) Eigen limit.
    PADDLE_ENFORCE_LE(x_dims.size(), 6,
                      platform::errors::InvalidArgument(
                          "The dimensions of Input(X) "
                          "should be in the range of [1, 6] (Eigen limit)."
                          "But received X's dimensions = %d, X's shape=[%s].",
                          x_dims.size(), x_dims));

    const auto &axes = ctx->Attrs().Get<std::vector<int>>("axes");
    auto out_dims = GetOutputShape(axes, x_dims, false);
    ctx->SetOutputDim("Out", out_dims);
    if (x_dims[0] == out_dims[0]) {
      // Only pass LoD when the first dimension of output and Input(X)
      // are the same.
      ctx->ShareLoD("X", "Out");
    }
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override {
    auto input_data_type =
        framework::OperatorWithKernel::IndicateVarDataType(ctx, "X");

#ifdef PADDLE_WITH_MKLDNN
    if (this->CanMKLDNNBeUsed(ctx, input_data_type)) {
      return framework::OpKernelType(input_data_type, ctx.GetPlace(),
                                     framework::DataLayout::kMKLDNN,
                                     framework::LibraryType::kMKLDNN);
    }
#endif
    return framework::OpKernelType(input_data_type, ctx.GetPlace());
  }
};

class SqueezeGradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext *context) const override {
    context->SetOutputDim(framework::GradVarName("X"),
                          context->GetInputDim("X"));
    context->ShareLoD("X", framework::GradVarName("X"));
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override {
    auto input_data_type = framework::OperatorWithKernel::IndicateVarDataType(
        ctx, framework::GradVarName("Out"));

#ifdef PADDLE_WITH_MKLDNN
    if (this->CanMKLDNNBeUsed(ctx, input_data_type)) {
      return framework::OpKernelType(input_data_type, ctx.GetPlace(),
                                     framework::DataLayout::kMKLDNN,
                                     framework::LibraryType::kMKLDNN);
    }
#endif
    return framework::OpKernelType(input_data_type, ctx.GetPlace());
  }
};

class SqueezeOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "(Tensor). The input tensor of squeeze operator.");
    AddOutput("Out", "(Tensor). The output tensor of squeeze operator.");
    AddAttr<std::vector<int>>("axes",
                              "(std::vector<int>). List of integers,"
                              " indicating the dimensions to squeeze.")
        .SetDefault({});
    AddAttr<bool>("use_mkldnn",
                  "(bool, default false) Only used in mkldnn kernel")
        .SetDefault(false);
    AddAttr<std::string>(
        "mkldnn_data_type",
        "(string, default \"float32\"). Data type of mkldnn kernel")
        .SetDefault("float32")
        .InEnum({"float32", "bfloat16"});
    AddComment(R"DOC(
        Squeeze Operator.

        Remove single-dimensional entries from the shape of a tensor.
        Takes a parameter axes with a list of axes to squeeze.
        If axes is not provided, all the single dimensions will be removed from the shape.
        If an axis is selected with shape entry not equal to one, an error is raised.

        Examples:
        Case 1:
          Given
            X.shape = (1, 3, 1, 5)
          and
            axes = [0]
          we get:
            Out.shape = (3, 1, 5)

        Case 2:
          Given
            X.shape = (1, 3, 1, 5)
          and
            axes = []
          we get:
            Out.shape = (3, 5)
    )DOC");
  }
};

class Squeeze2Op : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext *ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("X"), "Input", "X", "Squeeze2");
    OP_INOUT_CHECK(ctx->HasOutput("Out"), "Output", "Out", "Squeeze2");

    const auto &x_dims = ctx->GetInputDim("X");
    // Check input tensor dims (<6) Eigen limit.
    PADDLE_ENFORCE_LE(x_dims.size(), 6,
                      platform::errors::InvalidArgument(
                          "The dimensions of Input(X) "
                          "should be in the range of [1, 6] (Eigen limit)."
                          "But received X's dimensions = %d, X's shape = [%s].",
                          x_dims.size(), x_dims));

    const auto &axes = ctx->Attrs().Get<std::vector<int>>("axes");

    auto out_dims = GetOutputShape(axes, x_dims, false);
    ctx->SetOutputDim("Out", out_dims);
    if (x_dims[0] == out_dims[0]) {
      // Only pass LoD when the first dimension of output and Input(X)
      // are the same.
      ctx->ShareLoD("X", "Out");
    }

    OP_INOUT_CHECK(ctx->HasOutput("XShape"), "Output", "XShape", "Squeeze2");

    std::vector<int64_t> xshape_dims(x_dims.size() + 1);
    xshape_dims[0] = 0;
    for (int i = 0; i < x_dims.size(); ++i) {
      xshape_dims[i + 1] = x_dims[i];
    }
    ctx->SetOutputDim("XShape", framework::make_ddim(xshape_dims));
    ctx->ShareLoD("X", /*->*/ "XShape");
  }

  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override {
    auto input_data_type =
        framework::OperatorWithKernel::IndicateVarDataType(ctx, "X");

#ifdef PADDLE_WITH_MKLDNN
    if (this->CanMKLDNNBeUsed(ctx, input_data_type)) {
      return framework::OpKernelType(input_data_type, ctx.GetPlace(),
                                     framework::DataLayout::kMKLDNN,
                                     framework::LibraryType::kMKLDNN);
    }
#endif
    return framework::OpKernelType(input_data_type, ctx.GetPlace());
  }
};

template <typename T>
class SqueezeGradOpMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

  void Apply(GradOpPtr<T> grad_op) const override {
    grad_op->SetType("squeeze_grad");
    grad_op->SetInput("X", this->Input("X"));
    grad_op->SetInput(framework::GradVarName("Out"), this->OutputGrad("Out"));
    grad_op->SetOutput(framework::GradVarName("X"), this->InputGrad("X"));
    grad_op->SetAttrMap(this->Attrs());
  }
};

class Squeeze2GradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext *context) const override {
    OP_INOUT_CHECK(context->HasInput("XShape"), "Input", "XShape",
                   "Squeeze2Grad");
    OP_INOUT_CHECK(context->HasInput(framework::GradVarName("Out")), "Input",
                   framework::GradVarName("Out"), "Squeeze2Grad");
    auto xshape_dims = context->GetInputDim("XShape");
    auto x_dims = framework::slice_ddim(xshape_dims, 1, xshape_dims.size());
    context->SetOutputDim(framework::GradVarName("X"), x_dims);
    context->ShareLoD("XShape", framework::GradVarName("X"));
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override {
    auto input_data_type = framework::OperatorWithKernel::IndicateVarDataType(
        ctx, framework::GradVarName("Out"));

#ifdef PADDLE_WITH_MKLDNN
    if (this->CanMKLDNNBeUsed(ctx, input_data_type)) {
      return framework::OpKernelType(input_data_type, ctx.GetPlace(),
                                     framework::DataLayout::kMKLDNN,
                                     framework::LibraryType::kMKLDNN);
    }
#endif
    return framework::OpKernelType(input_data_type, ctx.GetPlace());
  }
};

template <typename T>
class SqueezeDoubleGradOpMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

  void Apply(GradOpPtr<T> grad_op) const override {
    grad_op->SetType("squeeze");
    grad_op->SetInput("X", this->OutputGrad(framework::GradVarName("X")));
    grad_op->SetOutput("Out", this->InputGrad(framework::GradVarName("Out")));
    grad_op->SetAttrMap(this->Attrs());
  }
};

// FIXME(zcd): squeeze2 adds an intermediate output(XShape) based on squeeze,
// the XShape is used to carry the shape and lod of X which will be used in
// squeeze_grad, in this way, the framework can reuse the memory of X
// immediately the squeeze2_op is finished.
// Considering compatibility issues, we could not fix squeeze2_op
class Squeeze2OpMaker : public SqueezeOpMaker {
 public:
  void Make() override {
    SqueezeOpMaker::Make();
    AddOutput("XShape",
              "XShape is just used to store the shape and lod of X, which will "
              "be used in SqueezeGradOp.")
        .AsIntermediate();
  }
};

template <typename T>
class Squeeze2GradOpMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

  void Apply(GradOpPtr<T> grad_op) const override {
    grad_op->SetType("squeeze2_grad");
    grad_op->SetInput("XShape", this->Output("XShape"));
    grad_op->SetInput(framework::GradVarName("Out"), this->OutputGrad("Out"));
    grad_op->SetOutput(framework::GradVarName("X"), this->InputGrad("X"));
    grad_op->SetAttrMap(this->Attrs());
  }
};

template <typename T>
class Squeeze2DoubleGradOpMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

  void Apply(GradOpPtr<T> grad_op) const override {
    grad_op->SetType("squeeze2");
    grad_op->SetInput("X", this->OutputGrad(framework::GradVarName("X")));
    grad_op->SetOutput("Out", this->InputGrad(framework::GradVarName("Out")));
    grad_op->SetOutput("XShape", this->Input("XShape"));
    grad_op->SetAttrMap(this->Attrs());
  }
};

DECLARE_INPLACE_OP_INFERER(SqueezeInplaceInferer, {"X", "Out"});
DECLARE_INPLACE_OP_INFERER(SqueezeGradInplaceInferer,
                           {framework::GradVarName("Out"),
                            framework::GradVarName("X")});
DECLARE_NO_NEED_BUFFER_VARS_INFERER(SqueezeGradNoNeedBufferVarsInferer, "X");
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(squeeze, ops::SqueezeOp, ops::SqueezeOpMaker,
                  ops::SqueezeGradOpMaker<paddle::framework::OpDesc>,
                  ops::SqueezeGradOpMaker<paddle::imperative::OpBase>);
REGISTER_OPERATOR(squeeze_grad, ops::SqueezeGradOp,
                  ops::SqueezeDoubleGradOpMaker<paddle::framework::OpDesc>,
                  ops::SqueezeDoubleGradOpMaker<paddle::imperative::OpBase>,
                  ops::SqueezeGradNoNeedBufferVarsInferer);

REGISTER_OPERATOR(squeeze2, ops::Squeeze2Op, ops::Squeeze2OpMaker,
                  ops::Squeeze2GradOpMaker<paddle::framework::OpDesc>,
                  ops::Squeeze2GradOpMaker<paddle::imperative::OpBase>,
                  ops::SqueezeInplaceInferer);
REGISTER_OPERATOR(squeeze2_grad, ops::Squeeze2GradOp,
                  ops::Squeeze2DoubleGradOpMaker<paddle::framework::OpDesc>,
                  ops::Squeeze2DoubleGradOpMaker<paddle::imperative::OpBase>,
                  ops::SqueezeGradInplaceInferer);

REGISTER_OP_CPU_KERNEL(
    squeeze, ops::SqueezeKernel<paddle::platform::CPUDeviceContext, float>,
    ops::SqueezeKernel<paddle::platform::CPUDeviceContext, double>,
    ops::SqueezeKernel<paddle::platform::CPUDeviceContext, bool>,
    ops::SqueezeKernel<paddle::platform::CPUDeviceContext, int>,
    ops::SqueezeKernel<paddle::platform::CPUDeviceContext, uint8_t>,
    ops::SqueezeKernel<paddle::platform::CPUDeviceContext, int8_t>,
    ops::SqueezeKernel<paddle::platform::CPUDeviceContext, int64_t>);
REGISTER_OP_CPU_KERNEL(
    squeeze_grad,
    ops::SqueezeGradKernel<paddle::platform::CPUDeviceContext, float>,
    ops::SqueezeGradKernel<paddle::platform::CPUDeviceContext, double>,
    ops::SqueezeGradKernel<paddle::platform::CPUDeviceContext, bool>,
    ops::SqueezeGradKernel<paddle::platform::CPUDeviceContext, int>,
    ops::SqueezeGradKernel<paddle::platform::CPUDeviceContext, uint8_t>,
    ops::SqueezeGradKernel<paddle::platform::CPUDeviceContext, int8_t>,
    ops::SqueezeGradKernel<paddle::platform::CPUDeviceContext, int64_t>);
REGISTER_OP_CPU_KERNEL(
    squeeze2, ops::Squeeze2Kernel<paddle::platform::CPUDeviceContext, float>,
    ops::Squeeze2Kernel<paddle::platform::CPUDeviceContext, double>,
    ops::Squeeze2Kernel<paddle::platform::CPUDeviceContext, bool>,
    ops::Squeeze2Kernel<paddle::platform::CPUDeviceContext, int>,
    ops::Squeeze2Kernel<paddle::platform::CPUDeviceContext, uint8_t>,
    ops::Squeeze2Kernel<paddle::platform::CPUDeviceContext, int8_t>,
    ops::Squeeze2Kernel<paddle::platform::CPUDeviceContext, int64_t>);
REGISTER_OP_CPU_KERNEL(
    squeeze2_grad,
    ops::Squeeze2GradKernel<paddle::platform::CPUDeviceContext, float>,
    ops::Squeeze2GradKernel<paddle::platform::CPUDeviceContext, double>,
    ops::Squeeze2GradKernel<paddle::platform::CPUDeviceContext, bool>,
    ops::Squeeze2GradKernel<paddle::platform::CPUDeviceContext, int>,
    ops::Squeeze2GradKernel<paddle::platform::CPUDeviceContext, uint8_t>,
    ops::Squeeze2GradKernel<paddle::platform::CPUDeviceContext, int8_t>,
    ops::Squeeze2GradKernel<paddle::platform::CPUDeviceContext, int64_t>);
