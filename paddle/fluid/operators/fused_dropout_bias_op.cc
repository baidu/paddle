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

#include "paddle/fluid/operators/fused_dropout_bias_op.h"
#include <memory>
#include <string>

namespace paddle {
namespace operators {

using framework::Tensor;

class DropoutBiasFuseOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("X"), "Input", "X", "DropoutBiasFuse");
    OP_INOUT_CHECK(ctx->HasInput("Bias"), "Input", "Bias", "DropoutBiasFuse");
    OP_INOUT_CHECK(ctx->HasOutput("Out"), "Output", "Out", "DropoutBiasFuse");
    auto x_dims = ctx->GetInputDim("X");
    ctx->SetOutputDim("Out", x_dims);
    auto bias_dims = ctx->GetInputDim("Bias");
    PADDLE_ENFORCE_LE(
        bias_dims.size(), 2,
        platform::errors::InvalidArgument(
            "The shape of Bias is expected have 1 or 2 dimensions "
            "but got %d.",
            bias_dims.size()));
    if (bias_dims.size() == 2) {
      PADDLE_ENFORCE_EQ(
          bias_dims[0], 1,
          platform::errors::InvalidArgument(
              "The first dimension of input Bias is expected be 1 "
              "but got %d.",
              bias_dims[0]));
      PADDLE_ENFORCE_EQ(
          bias_dims[bias_dims.size() - 1], x_dims[x_dims.size() - 1],
          platform::errors::InvalidArgument(
              "The last dimension of input Bias is expected be equal "
              "to the last dimension of input X. But received the "
              "last dimension of Bias is %d, Bias's shape is %s; "
              "the actual last dimension of X is %d, X's shape is %s.",
              bias_dims[bias_dims.size() - 1], bias_dims,
              x_dims[x_dims.size() - 1], x_dims));
    }
    if (ctx->Attrs().Get<bool>("is_test") == false) {
      ctx->SetOutputDim("Mask", x_dims);
    }
    ctx->ShareLoD("X", "Out");
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(
        OperatorWithKernel::IndicateVarDataType(ctx, "X"), ctx.GetPlace());
  }
};

class DropoutBiasFuseOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "The input of dropout_bias_fuse op.");
    AddInput("Seed", "The seed of dropout op").AsDispensable();
    AddInput("Bias", "The bias for the (X + b)");
    AddOutput("Out", "The output of dropout_bias_fuse op.");
    AddOutput("Mask", "The random sampled dropout mask.").AsIntermediate();
    AddAttr<float>("dropout_prob", "Probability of setting units to zero.")
        .SetDefault(.5f)
        .AddCustomChecker([](const float& drop_p) {
          PADDLE_ENFORCE_EQ(drop_p >= 0.0f && drop_p <= 1.0f, true,
                            platform::errors::InvalidArgument(
                                "'dropout_prob' must be between 0.0 and 1.0."));
        });
    AddAttr<bool>("is_test", "Set to true for inference only")
        .SetDefault(false);
    AddAttr<bool>("fix_seed", "Fix seed, only for test or debug!")
        .SetDefault(false);
    AddAttr<int>("seed", "Dropout random seed.").SetDefault(0);
    AddAttr<std::string>("dropout_implementation",
                         "There are two kinds of ways to implement dropout"
                         "downgrade_in_infer(default) or upscale_in_train")
        .SetDefault("downgrade_in_infer")
        .AddCustomChecker([](const std::string& type) {
          PADDLE_ENFORCE_EQ(
              type == "downgrade_in_infer" || type == "upscale_in_train", true,
              platform::errors::InvalidArgument(
                  "can only be downgrade_in_infer or upscale_in_train"));
        });
    AddComment(R"DOC(
Dropout Bias Fuse Operator.
Fused elementwise add with dropout.
)DOC");
  }
};

class DropoutBiasFuseOpGrad : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE_EQ(ctx->Attrs().Get<bool>("is_test"), false,
                      platform::errors::InvalidArgument(
                          "GradOp is only callable when is_test is false"));
    OP_INOUT_CHECK(ctx->HasInput("Mask"), "Input", "Mask", "DropoutGrad");
    OP_INOUT_CHECK(ctx->HasInput(framework::GradVarName("Out")), "Input",
                   framework::GradVarName("Out"), "DropoutGrad");
    auto out_dims = ctx->GetInputDim(framework::GradVarName("Out"));
    ctx->SetOutputDim(framework::GradVarName("X"), out_dims);
    ctx->SetOutputDim(framework::GradVarName("Bias"),
                      {1, out_dims[out_dims.size() - 1]});
    ctx->ShareLoD(framework::GradVarName("Out"), framework::GradVarName("X"));
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(OperatorWithKernel::IndicateVarDataType(
                                       ctx, framework::GradVarName("Out")),
                                   ctx.GetPlace());
  }
};

template <typename T>
class DropoutBiasFuseGradOpMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> op) const override {
    op->SetType("fused_dropout_bias_grad");
    op->SetInput(framework::GradVarName("Out"), this->OutputGrad("Out"));
    op->SetInput("Mask", this->Output("Mask"));
    op->SetOutput(framework::GradVarName("X"), this->InputGrad("X"));
    op->SetOutput(framework::GradVarName("Bias"), this->InputGrad("Bias"));
    op->SetAttrMap(this->Attrs());
  }
};
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(fused_dropout_bias, ops::DropoutBiasFuseOp,
                  ops::DropoutBiasFuseOpMaker,
                  ops::DropoutBiasFuseGradOpMaker<paddle::framework::OpDesc>,
                  ops::DropoutBiasFuseGradOpMaker<paddle::imperative::OpBase>);
REGISTER_OPERATOR(fused_dropout_bias_grad, ops::DropoutBiasFuseOpGrad);
REGISTER_OP_CPU_KERNEL(
    fused_dropout_bias,
    ops::CPUDropoutBiasFuseKernel<paddle::platform::CPUDeviceContext, float>,
    ops::CPUDropoutBiasFuseKernel<paddle::platform::CPUDeviceContext, double>);
REGISTER_OP_CPU_KERNEL(
    fused_dropout_bias_grad,
    ops::DropoutBiasFuseGradKernel<paddle::platform::CPUDeviceContext, float>,
    ops::DropoutBiasFuseGradKernel<paddle::platform::CPUDeviceContext, double>);
