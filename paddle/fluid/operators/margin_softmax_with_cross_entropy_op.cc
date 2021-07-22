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

#include "paddle/fluid/operators/margin_softmax_with_cross_entropy_op.h"

namespace paddle {
namespace operators {

class MarginSoftmaxWithCrossEntropyOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("Logits"), "Input", "Logits",
                   "MarginSoftmaxWithCrossEntropyOp");
    OP_INOUT_CHECK(ctx->HasInput("Label"), "Input", "Label",
                   "MarginSoftmaxWithCrossEntropyOp");

    OP_INOUT_CHECK(ctx->HasOutput("Softmax"), "Output", "Softmax",
                   "MarginSoftmaxWithCrossEntropyOp");
    OP_INOUT_CHECK(ctx->HasOutput("Loss"), "Output", "Loss",
                   "MarginSoftmaxWithCrossEntropyOp");

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

    if (labels_dims.size() > 1) {
      PADDLE_ENFORCE_EQ(
          labels_dims[logits_rank - 1], 1UL,
          platform::errors::InvalidArgument(
              "the last dimension of Input(Label) should be 1."
              "But received: the last dimension of Input(Label) is [%d],"
              "the last dimension is [%d]",
              labels_dims[logits_rank - 1], logits_rank - 1));
    }

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

class MarginSoftmaxWithCrossEntropyOpMaker
    : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() {
    AddInput("Logits",
             "(Tensor, default: Tensor<float>), The input tensor of unscaled "
             "log probabilities, whose dimension :attr:`axis` should be scaled "
             "by softmax.");
    AddInput(
        "Label",
        "(Tensor) The input tensor of groud truth label. Label is a "
        "Tensor<int64> in same shape with Input(Logits) except the shape in "
        "dimension :attr:`axis` as 1.");
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
                 "(int default 0) rank id for MarginSoftmaxWithCrossEntropy.")
        .SetDefault(0);
    AddAttr<int>("nranks",
                 "(int default 1) nranks id for MarginSoftmaxWithCrossEntropy.")
        .SetDefault(1);
    AddAttr<float>("margin1", "(float default 1.0) margin1 for MarginLoss.")
        .SetDefault(1.0);
    AddAttr<float>("margin2", "(float default 0.5) margin2 for MarginLoss.")
        .SetDefault(0.5);
    AddAttr<float>("margin3", "(float default 0.0) margin3 for MarginLoss.")
        .SetDefault(0.0);
    AddAttr<float>("scale", "(float default 64.0) scale for MarginLoss.")
        .SetDefault(64.0);
    AddComment(R"DOC(
MarginSoftmaxWithCrossEntropy Operator
.. math::

    L=-\frac{1}{N}\sum^N_{i=1}\log\frac{e^{s(cos(m_{1}\theta_{y_i}+m_{2})-m_{3})}}{e^{s(cos(m_{1}\theta_{y_i}+m_{2})-m_{3})}+\sum^n_{j=1,j\neq y_i} e^{scos\theta_{y_i}}}

where the :math: `\theta_{y_i}` is the angle between the feature :math: `x` and
the representation of class :math: `i`. The details of ArcFace loss
could be referred to https://arxiv.org/abs/1801.07698.

Note that the Op supports model parallel and single GPU. And Logits.shape[-1] can be different each rank.

)DOC");
  }
};

class MarginSoftmaxWithCrossEntropyOpGrad
    : public framework::OperatorWithKernel {
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
class MarginSoftmaxWithCrossEntropyOpGradMaker
    : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> op) const override {
    op->SetType("margin_softmax_with_cross_entropy_grad");

    op->SetInput("Softmax", this->Output("Softmax"));
    op->SetInput("Logits", this->Input("Logits"));
    op->SetInput("Label", this->Input("Label"));
    op->SetInput(framework::GradVarName("Loss"), this->OutputGrad("Loss"));
    op->SetAttrMap(this->Attrs());
    op->SetOutput(framework::GradVarName("Logits"), this->InputGrad("Logits"));
  }
};

DECLARE_INPLACE_OP_INFERER(MarginSoftmaxWithCrossEntropyGradInplaceInferer,
                           {"Softmax", framework::GradVarName("Logits")});

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;

REGISTER_OPERATOR(
    margin_softmax_with_cross_entropy, ops::MarginSoftmaxWithCrossEntropyOp,
    ops::MarginSoftmaxWithCrossEntropyOpMaker,
    ops::MarginSoftmaxWithCrossEntropyOpGradMaker<paddle::framework::OpDesc>,
    ops::MarginSoftmaxWithCrossEntropyOpGradMaker<paddle::imperative::OpBase>);

REGISTER_OPERATOR(margin_softmax_with_cross_entropy_grad,
                  ops::MarginSoftmaxWithCrossEntropyOpGrad,
                  ops::MarginSoftmaxWithCrossEntropyGradInplaceInferer);

REGISTER_OP_CPU_KERNEL(
    margin_softmax_with_cross_entropy,
    ops::MarginSoftmaxWithCrossEntropyOpCPUKernel<float>,
    ops::MarginSoftmaxWithCrossEntropyOpCPUKernel<double>,
    ops::MarginSoftmaxWithCrossEntropyOpCPUKernel<plat::float16>);
