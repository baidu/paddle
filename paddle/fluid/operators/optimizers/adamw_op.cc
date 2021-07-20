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

#include "paddle/fluid/operators/optimizers/adamw_op.h"
#include "paddle/fluid/framework/op_version_registry.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;

void AdamWOp::InferShape(framework::InferShapeContext *ctx) const {
  PADDLE_ENFORCE_EQ(ctx->HasInput("Param"), true,
                    platform::errors::NotFound(
                        "Input(Param) of AdamWOp should not be null."));
  PADDLE_ENFORCE_EQ(
      ctx->HasInput("Grad"), true,
      platform::errors::NotFound("Input(Grad) of AdamWOp should not be null."));
  PADDLE_ENFORCE_EQ(ctx->HasInput("Moment1"), true,
                    platform::errors::NotFound(
                        "Input(Moment1) of AdamWOp should not be null."));
  PADDLE_ENFORCE_EQ(ctx->HasInput("Moment2"), true,
                    platform::errors::NotFound(
                        "Input(Moment2) of AdamWOp should not be null."));
  PADDLE_ENFORCE_EQ(ctx->HasInput("LearningRate"), true,
                    platform::errors::NotFound(
                        "Input(LearningRate) of AdamWOp should not be null."));
  PADDLE_ENFORCE_EQ(ctx->HasInput("Beta1Pow"), true,
                    platform::errors::NotFound(
                        "Input(Beta1Pow) of AdamWOp should not be null."));
  PADDLE_ENFORCE_EQ(ctx->HasInput("Beta2Pow"), true,
                    platform::errors::NotFound(
                        "Input(Beta2Pow) of AdamWOp should not be null."));

  PADDLE_ENFORCE_EQ(ctx->HasOutput("ParamOut"), true,
                    platform::errors::NotFound(
                        "Output(ParamOut) of AdamWOp should not be null."));
  PADDLE_ENFORCE_EQ(ctx->HasOutput("Moment1Out"), true,
                    platform::errors::NotFound(
                        "Output(Moment1Out) of AdamWOp should not be null."));
  PADDLE_ENFORCE_EQ(ctx->HasOutput("Moment2Out"), true,
                    platform::errors::NotFound(
                        "Output(Moment2Out) of AdamWOp should not be null."));

  auto lr_dims = ctx->GetInputDim("LearningRate");
  PADDLE_ENFORCE_NE(
      framework::product(lr_dims), 0,
      platform::errors::InvalidArgument(
          "The number of LearningRate shall not be 0, but received %d. Maybe "
          "the Input variable LearningRate has not "
          "been initialized. You may need to confirm "
          "if you put exe.run(startup_program) "
          "after optimizer.minimize function.",
          framework::product(lr_dims)));
  PADDLE_ENFORCE_EQ(
      framework::product(lr_dims), 1,
      platform::errors::InvalidArgument(
          "Learning rate should have 1 dimension, but received %d",
          framework::product(lr_dims)));
  auto beta1_pow_dims = ctx->GetInputDim("Beta1Pow");
  VLOG(3) << "dims of Beta1Pow : [" << beta1_pow_dims << "]";
  PADDLE_ENFORCE_GE(framework::product(beta1_pow_dims), 1,
                    platform::errors::InvalidArgument(
                        "The size of Beta1 power accumulator should be greater "
                        "than 0, but received %d.",
                        framework::product(beta1_pow_dims)));
  auto beta2_pow_dims = ctx->GetInputDim("Beta2Pow");
  VLOG(3) << "dims of Beta2Pow : [" << beta2_pow_dims << "]";
  PADDLE_ENFORCE_GE(framework::product(beta2_pow_dims), 1,
                    platform::errors::InvalidArgument(
                        "The size of Beta2 power accumulator should be greater "
                        "than 0, but received %d.",
                        framework::product(beta2_pow_dims)));

  auto param_dims = ctx->GetInputDim("Param");
  if (ctx->GetInputsVarType("Grad")[0] ==
      framework::proto::VarType::LOD_TENSOR) {
    PADDLE_ENFORCE_EQ(
        param_dims, ctx->GetInputDim("Grad"),
        platform::errors::InvalidArgument(
            "Param and Grad input of AdamWOp should have same dimension. But "
            "received Param dims: [%s], Grad dims: [%s].",
            param_dims, ctx->GetInputDim("Grad")));
  }
  PADDLE_ENFORCE_EQ(
      param_dims, ctx->GetInputDim("Moment1"),
      platform::errors::InvalidArgument(
          "Param and Moment1 input of AdamWOp should have same dimension. But "
          "received Param dims: [%s], Moment1 dims: [%s].",
          param_dims, ctx->GetInputDim("Moment1")));
  PADDLE_ENFORCE_EQ(
      param_dims, ctx->GetInputDim("Moment2"),
      platform::errors::InvalidArgument(
          "Param and Moment2 input of AdamWOp should have same dimension. But "
          "received Param dims: [%s], Moment2 dims: [%s].",
          param_dims, ctx->GetInputDim("Moment2")));

  ctx->SetOutputDim("ParamOut", param_dims);
  ctx->SetOutputDim("Moment1Out", param_dims);
  ctx->SetOutputDim("Moment2Out", param_dims);
  ctx->SetOutputDim("Beta1PowOut", beta1_pow_dims);
  ctx->SetOutputDim("Beta2PowOut", beta2_pow_dims);
}

framework::OpKernelType AdamWOp::GetExpectedKernelType(
    const framework::ExecutionContext &ctx) const {
  auto input_data_type = OperatorWithKernel::IndicateVarDataType(ctx, "Param");
  return framework::OpKernelType(input_data_type, ctx.GetPlace());
}

framework::OpKernelType AdamWOp::GetKernelTypeForVar(
    const std::string &var_name, const framework::Tensor &tensor,
    const framework::OpKernelType &expected_kernel_type) const {
  if (var_name == "Beta1Pow" || var_name == "Beta2Pow") {
    return expected_kernel_type;
  } else {
    return framework::OpKernelType(expected_kernel_type.data_type_,
                                   tensor.place(), tensor.layout());
  }
}

class AdamWOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("Param", "(Tensor) Input parameter");
    AddInput("Grad", "(Tensor) Input gradient");
    AddInput("LearningRate", "(Tensor) Learning rate");
    AddInput("Moment1", "(Tensor) Input first moment");
    AddInput("Moment2", "(Tensor) Input second moment");
    AddInput("Beta1Pow", "(Tensor) Input beta1 power accumulator");
    AddInput("Beta2Pow", "(Tensor) Input beta2 power accumulator");

    AddInput("Beta1Tensor",
             "(Tensor<float32>, optional) If provided, AdamW will use this "
             "as beta1, this has a higher priority than attr(beta1), the "
             "shape of this tensor MUST BE [1].")
        .AsDispensable();
    AddInput("Beta2Tensor",
             "(Tensor<float32>, optional) If provided, AdamW will use this "
             "as beta2, this has a higher priority than attr(beta2), the "
             "shape of this tensor MUST BE [1].")
        .AsDispensable();
    AddInput("EpsilonTensor",
             "(Tensor<float32>, optional) If provided, AdamW will use this "
             "as epsilon, this has a higher priority than attr(epsilon), the "
             "shape of this tensor MUST BE [1].")
        .AsDispensable();
    AddInput("WeightDecayTensor",
             "(Tensor<float32>, optional) If provided, AdamW will use this "
             "as weightdecay, this has a higher priority than "
             "attr(weightdecay), the "
             "shape of this tensor MUST BE [1].")
        .AsDispensable();
    AddInput("MasterParam", "FP32 master weight for AMP.").AsDispensable();

    AddOutput("ParamOut", "(Tensor) Output parameter");
    AddOutput("Moment1Out", "(Tensor) Output first moment");
    AddOutput("Moment2Out", "(Tensor) Output second moment");
    AddOutput("Beta1PowOut", "(Tensor) Output beta1 power accumulator");
    AddOutput("Beta2PowOut", "(Tensor) Output beta2 power accumulator");
    AddOutput("MasterParamOut",
              "The updated FP32 master weight for AMP. "
              "It shared memory with Input(MasterParam).")
        .AsDispensable();

    AddAttr<float>("beta1",
                   "(float, default 0.9) "
                   "Exponential decay rate for the "
                   "first moment estimates.")
        .SetDefault(0.9f);
    AddAttr<float>("beta2",
                   "(float, default 0.999) "
                   "exponential decay rate for the "
                   "second moment estimates.")
        .SetDefault(0.999f);
    AddAttr<float>("epsilon",
                   "(float, default 1.0e-8) "
                   "Constant for numerical stability")
        .SetDefault(1.0e-8f);
    AddAttr<float>("weightdecay",
                   "(float, default 0.01) "
                   "Constant for numerical stability")
        .SetDefault(0.01);
    AddAttr<bool>(
        "lazy_mode",
        "(bool, default false) "
        "only update the parameter that has gradient in sparse update")
        .SetDefault(false);
    AddAttr<int64_t>("min_row_size_to_use_multithread",
                     "(int64_t, default 0) "
                     "when not zero, if param row size is larger then "
                     "min_row_size_to_use_multithread and "
                     "inner_op_parallelism is larger then 0, sparse update "
                     "will run in multithread mode")
        .SetDefault(1000);
    AddAttr<bool>("multi_precision",
                  "(bool, default false) "
                  "Whether to use multi-precision during weight updating.")
        .SetDefault(false);
    // TODO(zhiqiu): We could set Beta1PowOut and Beta2PowOut
    // as dispensable since they are not used when use_global_beta_pow is true.
    AddAttr<bool>("use_global_beta_pow",
                  "(bool, default false) "
                  "Whether to use global beta_pow for whole model instead of "
                  "creating beta_pow for each parameter.")
        .SetDefault(false);

    AddComment(R"DOC(
AdamW Optimizer.

This implements the AdamW optimizer from Section 2 of the AdamW
paper : https://arxiv.org/pdf/1711.05101.pdf.
AdamW is a first-order gradient-based optimization method based on
adaptive estimates of lower-order moments.

AdamW updates:

$$
moment\_1\_out = \beta_1 * moment\_1 + (1 - \beta_1) * grad \\
moment\_2_\out = \beta_2 * moment\_2 + (1 - \beta_2) * grad * grad \\
learning\_rate\_1 & = 1 - learning\_rate * \lambda
learning\_rate\_2 & = learning\_rate * \\
                    \\frac{\sqrt{1 - {\\beta}_2^t}}{1 - {beta}_1^t}
param\_out & = learning\_rate\_1 * param - learning\_rate\_2 * \\
                    (\\frac{moment\_1}{\sqrt{moment\_2} + \epsilon * \sqrt{1 - {\\beta}_2^t}})
$$

)DOC");
  }
};
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_WITHOUT_GRADIENT(adamw, ops::AdamWOp, ops::AdamWOpMaker);
REGISTER_OP_CPU_KERNEL(
    adamw, ops::AdamWOpKernel<paddle::platform::CPUDeviceContext, float>,
    ops::AdamWOpKernel<paddle::platform::CPUDeviceContext, double>);

REGISTER_OP_VERSION(adamw)
    .AddCheckpoint(
        R"ROC(
      Upgrade adamw add 1 attribute [multi_precision].
    )ROC",
        paddle::framework::compatible::OpVersionDesc().NewAttr(
            "multi_precision",
            "(bool) Whether to use multi-precision during weight updating.",
            false))
    .AddCheckpoint(
        R"ROC(
      Upgrade adamw, add 1 dispensable input [EpsilonTensor].
    )ROC",
        paddle::framework::compatible::OpVersionDesc().NewInput(
            "EpsilonTensor",
            "If provided, Adamw will use this as epsilon, "
            "this has a higher priority than attr(epsilon). "
            "For better performance in npu kernel. "))
    .AddCheckpoint(
        R"ROC(
      Upgrade adamw, add 1 dispensable input [WeightDecayTensor].
    )ROC",
        paddle::framework::compatible::OpVersionDesc().NewInput(
            "WeightDecayTensor",
            "If provided, Adamw will use this as weightdecay, "
            "this has a higher priority than attr(weightdecay). "
            "For better performance in npu kernel. "))
    .AddCheckpoint(
        R"ROC(
      Upgrade adamw, add 1 attribute [use_global_beta_pow].
    )ROC",
        paddle::framework::compatible::OpVersionDesc().NewAttr(
            "use_global_beta_pow",
            "If true, Adamw will use global beta_pow for whole model "
            "instead of creating beta_pow for each parameter."
            "In that case, the outputs(Beta1PowOut, Beta2PowOut) will not be "
            "used in adamw op, "
            "and beta_pow will be updated after all adamw op in the model.",
            false));
