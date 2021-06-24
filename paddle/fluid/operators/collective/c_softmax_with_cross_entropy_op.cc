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

#include "paddle/fluid/operators/collective/c_softmax_with_cross_entropy_op.h"

namespace paddle {
namespace operators {

class CSoftmaxWithCrossEntropyOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("Logits"), "Input", "Logits",
                   "CSoftmaxWithCrossEntropyOp");
    OP_INOUT_CHECK(ctx->HasInput("Label"), "Input", "Label",
                   "CSoftmaxWithCrossEntropyOp");

    OP_INOUT_CHECK(ctx->HasOutput("Softmax"), "Output", "Softmax",
                   "CSoftmaxWithCrossEntropyOp");
    OP_INOUT_CHECK(ctx->HasOutput("Loss"), "Output", "Loss",
                   "CSoftmaxWithCrossEntropyOp");

    auto logits_dims = ctx->GetInputDim("Logits");
    auto labels_dims = ctx->GetInputDim("Label");

    auto logits_rank = logits_dims.size();
    auto axis = logits_rank - 1;
    for (int i = 0; i < logits_rank; i++) {
      if (i != axis) {
        if (ctx->IsRuntime() || (logits_dims[i] > 0 && labels_dims[i] > 0)) {
          PADDLE_ENFORCE_EQ(logits_dims[i], labels_dims[i],
                            platform::errors::InvalidArgument(
                                "Input(Logits) and Input(Label) should in "
                                "same shape in dimensions except axis."));
        }
      }
    }

    PADDLE_ENFORCE_EQ(
        labels_dims[logits_rank - 1], 1UL,
        platform::errors::InvalidArgument(
            "the last dimension of Input(Label) should be 1."
            "But received: the last dimension of Input(Label) is [%d],"
            "the last dimension is [%d]",
            labels_dims[logits_rank - 1], logits_rank - 1));

    ctx->SetOutputDim("Softmax", logits_dims);

    logits_dims[axis] = 1;
    ctx->SetOutputDim("Loss", logits_dims);

    ctx->ShareLoD("Logits", /*->*/ "Softmax");
    ctx->ShareLoD("Logits", /*->*/ "Loss");
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(
        OperatorWithKernel::IndicateVarDataType(ctx, "Logits"),
        ctx.device_context());
  }
};

class CSoftmaxWithCrossEntropyOpMaker
    : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() {
    AddInput("Logits",
             "(Tensor, default: Tensor<float>), The input tensor of unscaled "
             "log probabilities, whose dimension :attr:`axis` should be scaled "
             "by softmax.");
    AddInput(
        "Label",
        "(Tensor) The input tensor of groud truth label. If :attr:`soft_label` "
        "is set to false, Label is a Tensor<int64> in same shape with "
        "Input(Logits) except the shape in dimension :attr:`axis` as 1. If "
        "soft_label is set to true, Label is a Tensor<float/double> in same "
        "shape with Input(Logits).");
    AddOutput(
        "Softmax",
        "(Tensor, default: Tensor<float>), A tensor in same shape with "
        "Input(Logits). "
        "The outputs value of softmax activation by given the input batch, "
        "which will be used in backward calculation.");
    AddOutput("Loss",
              "(Tensor, default: Tensor<float>), A tensor in same shape with "
              "Input(Logits) "
              "except the shape in dimension :attr:`axis` as 1. The cross "
              "entropy loss.");
    AddAttr<int>("ring_id", "(int default 0) nccl communication ring id.")
        .SetDefault(0);
    AddAttr<int>("rank",
                 "(int default 0) rank id for CSoftmaxWithCrossEntropy.")
        .SetDefault(0);
    AddAttr<int>("nranks",
                 "(int default 1) nranks id for CSoftmaxWithCrossEntropy.")
        .SetDefault(0);
    AddComment(R"DOC(
CSoftmaxWithCrossEntropy Operator

)DOC");
  }
};

class CSoftmaxWithCrossEntropyOpGrad : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE_EQ(ctx->HasInput(framework::GradVarName("Loss")), true,
                      platform::errors::InvalidArgument(
                          "Input(Loss@Grad) should not be null."));
    PADDLE_ENFORCE_EQ(ctx->HasInput("Softmax"), true,
                      platform::errors::InvalidArgument(
                          "Input(Softmax) should be not null."));
    PADDLE_ENFORCE_EQ(
        ctx->HasInput("Label"), true,
        platform::errors::InvalidArgument("Input(Label) should be not null."));

    PADDLE_ENFORCE_EQ(ctx->HasOutput(framework::GradVarName("Logits")), true,
                      platform::errors::InvalidArgument(
                          "Output(Logits@Grad) should be not null."));

    ctx->SetOutputDim(framework::GradVarName("Logits"),
                      ctx->GetInputDim("Softmax"));
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(OperatorWithKernel::IndicateVarDataType(
                                       ctx, framework::GradVarName("Loss")),
                                   ctx.device_context());
  }
};

template <typename T>
class CSoftmaxWithCrossEntropyOpGradMaker
    : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> op) const override {
    op->SetType("c_softmax_with_cross_entropy_grad");

    op->SetInput("Softmax", this->Output("Softmax"));
    op->SetInput("Label", this->Input("Label"));
    op->SetInput(framework::GradVarName("Loss"), this->OutputGrad("Loss"));
    op->SetAttrMap(this->Attrs());
    op->SetOutput(framework::GradVarName("Logits"), this->InputGrad("Logits"));
  }
};

DECLARE_INPLACE_OP_INFERER(CSoftmaxWithCrossEntropyInplaceInferer,
                           {"Logits", "Softmax"});

DECLARE_INPLACE_OP_INFERER(CSoftmaxWithCrossEntropyGradInplaceInferer,
                           {"Softmax", framework::GradVarName("Logits")});

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;

REGISTER_OPERATOR(
    c_softmax_with_cross_entropy, ops::CSoftmaxWithCrossEntropyOp,
    ops::CSoftmaxWithCrossEntropyOpMaker,
    ops::CSoftmaxWithCrossEntropyOpGradMaker<paddle::framework::OpDesc>,
    ops::CSoftmaxWithCrossEntropyOpGradMaker<paddle::imperative::OpBase>,
    ops::CSoftmaxWithCrossEntropyInplaceInferer);

REGISTER_OPERATOR(c_softmax_with_cross_entropy_grad,
                  ops::CSoftmaxWithCrossEntropyOpGrad,
                  ops::CSoftmaxWithCrossEntropyGradInplaceInferer);

REGISTER_OP_CPU_KERNEL(c_softmax_with_cross_entropy,
                       ops::CSoftmaxWithCrossEntropyOpCPUKernel<float>,
                       ops::CSoftmaxWithCrossEntropyOpCPUKernel<double>,
                       ops::CSoftmaxWithCrossEntropyOpCPUKernel<plat::float16>);
