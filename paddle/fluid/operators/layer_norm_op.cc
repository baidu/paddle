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

#include "paddle/fluid/operators/layer_norm_op.h"
#include <memory>
#include <string>

#ifdef PADDLE_WITH_MKLDNN
#include "paddle/fluid/platform/mkldnn_helper.h"
#endif

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
using LoDTensor = framework::LoDTensor;
using DataLayout = framework::DataLayout;

class LayerNormOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext *ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("X"), "Input", "X", "LayerNorm");
    OP_INOUT_CHECK(ctx->HasOutput("Y"), "Output", "Y", "LayerNorm");
    OP_INOUT_CHECK(ctx->HasOutput("Mean"), "Output", "Mean", "LayerNorm");
    OP_INOUT_CHECK(ctx->HasOutput("Variance"), "Output", "Variance",
                   "LayerNorm");

    auto x_dim = ctx->GetInputDim("X");
    auto begin_norm_axis = ctx->Attrs().Get<int>("begin_norm_axis");
    PADDLE_ENFORCE_LT(
        begin_norm_axis, x_dim.size(),
        platform::errors::InvalidArgument(
            "'begin_norm_axis' must be less than the dimensions of X,"
            "But received 'begin_norm_axis' is [%d],"
            "received the dimensions of X is [%d].",
            begin_norm_axis, x_dim.size()));

    auto matrix_dim = framework::flatten_to_2d(x_dim, begin_norm_axis);
    int left = static_cast<int>(matrix_dim[0]);
    int right = static_cast<int>(matrix_dim[1]);
    if (ctx->HasInput("Scale")) {
      PADDLE_ENFORCE_EQ(ctx->GetInputDim("Scale").size(), 1,
                        platform::errors::InvalidArgument(
                            "The dimensions of Input(Scale) must be 1, but "
                            "received dimensions of"
                            "Input(Scale) is [%d]",
                            ctx->GetInputDim("Scale").size()));

      if (ctx->IsRuntime()) {
        PADDLE_ENFORCE_EQ(
            ctx->GetInputDim("Scale")[0], right,
            platform::errors::InvalidArgument(
                "The first dimension value of Input(Scale) must equal to be the"
                "second dimension value of the flattened 2D matrix of Input(X),"
                "But received the first dimension value of Input(Scale) is"
                "[%d], the second dimension value of the flattened 2D matrix of"
                " Input(Scale) is [%d].",
                ctx->GetInputDim("Scale")[0], right));
      }
    }
    if (ctx->HasInput("Bias")) {
      PADDLE_ENFORCE_EQ(ctx->GetInputDim("Bias").size(), 1,
                        platform::errors::InvalidArgument(
                            "The dimensions of Input(Bias) must be 1, but "
                            "received dimensions of"
                            "Input(Bias) is [%d]",
                            ctx->GetInputDim("Bias").size()));
      if (ctx->IsRuntime()) {
        PADDLE_ENFORCE_EQ(
            ctx->GetInputDim("Bias")[0], right,
            platform::errors::InvalidArgument(
                "The first dimension value of Input(Bias) must equal to be the"
                "second dimension value of the flattened 2D matrix of Input(X),"
                "But received the first dimension value of Input(Bias) is"
                "[%d], the second dimension value of the flattened 2D matrix of"
                " Input(Bias) is [%d].",
                ctx->GetInputDim("Scale")[0], right));
      }
    }

    ctx->SetOutputDim("Y", ctx->GetInputDim("X"));
    ctx->SetOutputDim("Mean", {left});
    ctx->SetOutputDim("Variance", {left});
    ctx->ShareLoD("X", "Y");
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override {
    auto input_data_type = OperatorWithKernel::IndicateVarDataType(ctx, "X");
    // By default, the type of the scale, bias, mean,
    // and var tensors should both be float. (For float or float16 input tensor)
    // or double (For double input tensor).
    auto ln_param_type = framework::proto::VarType::FP32;
    if (input_data_type == framework::proto::VarType::FP64) {
      ln_param_type = framework::proto::VarType::FP64;
    }
    if (ctx.HasInput("Scale")) {
      PADDLE_ENFORCE_EQ(ln_param_type, ctx.Input<Tensor>("Scale")->type(),
                        platform::errors::InvalidArgument(
                            "Scale input should be of float type"));
    }
    if (ctx.HasInput("Bias")) {
      PADDLE_ENFORCE_EQ(ln_param_type, ctx.Input<Tensor>("Bias")->type(),
                        platform::errors::InvalidArgument(
                            "Bias input should be of float type"));
    }

    framework::LibraryType library = framework::LibraryType::kPlain;
    framework::DataLayout layout = framework::DataLayout::kAnyLayout;

#ifdef PADDLE_WITH_MKLDNN
    if (library == framework::LibraryType::kPlain &&
        this->CanMKLDNNBeUsed(ctx)) {
      library = framework::LibraryType::kMKLDNN;
      layout = framework::DataLayout::kMKLDNN;
    }
#endif

    return framework::OpKernelType(input_data_type, ctx.GetPlace(), layout,
                                   library);
  }
};

class LayerNormOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "The input tensor.");
    AddInput("Scale",
             "(optional) Scale is a 1-dimensional tensor of size "
             "H(`begin_norm_axis` splits the tensor(`X`) to a matrix [N,H])."
             "It is applied to the output.")
        .AsDispensable();
    AddInput("Bias",
             "(optional) Bias is a 1-dimensional tensor of size "
             "H(`begin_norm_axis` splits the tensor(`X`) to a matrix [N,H])."
             "It is applied to the output.")
        .AsDispensable();
    AddOutput("Y", "Result after normalization.");
    AddOutput("Mean", "Mean of the current mini batch.").AsIntermediate();
    AddOutput("Variance", "Variance of the current mini batch.")
        .AsIntermediate();

    AddAttr<float>("epsilon",
                   "Constant for numerical stability [default 1e-5].")
        .SetDefault(1e-5)
        .AddCustomChecker([](const float &epsilon) {
          PADDLE_ENFORCE_EQ(epsilon >= 0.0f && epsilon <= 0.001f, true,
                            platform::errors::InvalidArgument(
                                "'epsilon' in Op(LayerNorm) should be between"
                                "0.0 and 0.001, But received [%s].",
                                epsilon));
        });
    AddAttr<int>("begin_norm_axis",
                 "the axis of `begin_norm_axis ... Rank(X) - 1` will be "
                 "normalized. `begin_norm_axis` splits the tensor(`X`) to a "
                 "matrix [N,H]. [default 1].")
        .SetDefault(1)
        .AddCustomChecker([](const int &begin_norm_axis) {
          PADDLE_ENFORCE_GT(begin_norm_axis, 0,
                            platform::errors::InvalidArgument(
                                "'begin_norm_axis' in Op(LayerNorm) should be"
                                "greater than zero. But received [%d].",
                                begin_norm_axis));
        });
    AddAttr<bool>("use_mkldnn",
                  "(bool, default false) Only used in mkldnn kernel")
        .SetDefault(false);
    AddAttr<std::string>(
        "mkldnn_data_type",
        "(string, default \"float32\"). Data type of mkldnn kernel")
        .SetDefault("float32")
        .InEnum({"float32", "bfloat16"});
    AddAttr<bool>("is_test",
                  "(bool, default false) Set to true for inference only, false "
                  "for training. Some layers may run faster when this is true.")
        .SetDefault(false);

    AddComment(R"DOC(
Assume feature vectors exist on dimensions
:attr:`begin_norm_axis ... rank(input)` and calculate the moment statistics
along these dimensions for each feature vector :math:`a` with size
:math:`H`, then normalize each feature vector using the corresponding
statistics. After that, apply learnable gain and bias on the normalized
tensor to scale and shift if :attr:`scale` and :attr:`shift` are set.

Refer to `Layer Normalization <https://arxiv.org/pdf/1607.06450v1.pdf>`_
)DOC");
  }
};

class LayerNormGradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext *ctx) const override {
    // check input
    OP_INOUT_CHECK(ctx->HasInput("X"), "Input", "X", "LayerNormGrad");
    OP_INOUT_CHECK(ctx->HasInput("Mean"), "Input", "Mean", "LayerNormGrad");
    OP_INOUT_CHECK(ctx->HasInput("Variance"), "Input", "Variance",
                   "LayerNormGrad");
    OP_INOUT_CHECK(ctx->HasInput(framework::GradVarName("Y")), "Input",
                   framework::GradVarName("Y"), "LayerNormGrad");

    // check output
    if (ctx->HasOutput(framework::GradVarName("X"))) {
      ctx->SetOutputDim(framework::GradVarName("X"), ctx->GetInputDim("X"));
    }
    if (ctx->HasOutput(framework::GradVarName("Scale"))) {
      ctx->SetOutputDim(framework::GradVarName("Scale"),
                        ctx->GetInputDim("Scale"));
    }
    if (ctx->HasOutput(framework::GradVarName("Bias"))) {
      ctx->SetOutputDim(framework::GradVarName("Bias"),
                        ctx->GetInputDim("Bias"));
    }
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override {
    const auto *var = ctx.InputVar(framework::GradVarName("Y"));
    PADDLE_ENFORCE_NOT_NULL(var, platform::errors::NotFound(
                                     "Y@GRAD of LayerNorm Op is not found."));
    const Tensor *t = nullptr;
    if (var->IsType<Tensor>()) {
      t = &var->Get<Tensor>();
    } else if (var->IsType<LoDTensor>()) {
      t = &var->Get<LoDTensor>();
    }
    PADDLE_ENFORCE_NOT_NULL(
        t, platform::errors::NotFound("Y@GRAD of LayerNorm Op is not found."));
    return framework::OpKernelType(t->type(), ctx.GetPlace());
  }
};

template <typename T>
class LayerNormGradOpMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> op) const override {
    op->SetType("layer_norm_grad");
    op->SetInput("X", this->Input("X"));
    op->SetInput("Mean", this->Output("Mean"));
    op->SetInput("Variance", this->Output("Variance"));
    if (this->HasInput("Scale")) {
      op->SetInput("Scale", this->Input("Scale"));
      op->SetOutput(framework::GradVarName("Scale"), this->InputGrad("Scale"));
    }

    if (this->HasInput("Bias")) {
      op->SetInput("Bias", this->Input("Bias"));
      op->SetOutput(framework::GradVarName("Bias"), this->InputGrad("Bias"));
    }

    op->SetInput(framework::GradVarName("Y"), this->OutputGrad("Y"));
    op->SetOutput(framework::GradVarName("X"), this->InputGrad("X"));
    op->SetAttrMap(this->Attrs());
  }
};

DECLARE_NO_NEED_BUFFER_VARS_INFERER(LayerNormGradNoNeedBufferVarInferer,
                                    "Bias");

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(layer_norm, ops::LayerNormOp, ops::LayerNormOpMaker,
                  ops::LayerNormGradOpMaker<paddle::framework::OpDesc>,
                  ops::LayerNormGradOpMaker<paddle::imperative::OpBase>);
REGISTER_OPERATOR(layer_norm_grad, ops::LayerNormGradOp,
                  ops::LayerNormGradNoNeedBufferVarInferer);
REGISTER_OP_CPU_KERNEL(
    layer_norm, ops::LayerNormKernel<paddle::platform::CPUDeviceContext, float>,
    ops::LayerNormKernel<paddle::platform::CPUDeviceContext, double>);
REGISTER_OP_CPU_KERNEL(
    layer_norm_grad,
    ops::LayerNormGradKernel<paddle::platform::CPUDeviceContext, float>,
    ops::LayerNormGradKernel<paddle::platform::CPUDeviceContext, double>);
