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

#include "paddle/fluid/operators/optimizers/lamb_op.h"

#include <string>

#include "paddle/fluid/framework/op_version_registry.h"

namespace paddle {
namespace operators {

class LambOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext *ctx) const override {
    PADDLE_ENFORCE_EQ(ctx->HasInput("Param"), true,
                      platform::errors::NotFound(
                          "Input(Param) of LambOp should not be null."));
    PADDLE_ENFORCE_EQ(ctx->HasInput("Grad"), true,
                      platform::errors::NotFound(
                          "Input(Grad) of LambOp should not be null."));
    PADDLE_ENFORCE_EQ(ctx->HasInput("Moment1"), true,
                      platform::errors::NotFound(
                          "Input(Moment1) of LambOp should not be null."));
    PADDLE_ENFORCE_EQ(ctx->HasInput("Moment2"), true,
                      platform::errors::NotFound(
                          "Input(Moment2) of LambOp should not be null."));
    PADDLE_ENFORCE_EQ(ctx->HasInput("LearningRate"), true,
                      platform::errors::NotFound(
                          "Input(LearningRate) of LambOp should not be null."));
    PADDLE_ENFORCE_EQ(ctx->HasInput("Beta1Pow"), true,
                      platform::errors::NotFound(
                          "Input(Beta1Pow) of LambOp should not be null."));
    PADDLE_ENFORCE_EQ(ctx->HasInput("Beta2Pow"), true,
                      platform::errors::NotFound(
                          "Input(Beta2Pow) of LambOp should not be null."));

    PADDLE_ENFORCE_EQ(ctx->HasOutput("ParamOut"), true,
                      platform::errors::NotFound(
                          "Output(ParamOut) of LambOp should not be null."));
    PADDLE_ENFORCE_EQ(ctx->HasOutput("Moment1Out"), true,
                      platform::errors::NotFound(
                          "Output(Moment1Out) of LambOp should not be null."));
    PADDLE_ENFORCE_EQ(ctx->HasOutput("Moment2Out"), true,
                      platform::errors::NotFound(
                          "Output(Moment2Out) of LambOp should not be null."));
    PADDLE_ENFORCE_EQ(ctx->HasOutput("Beta1PowOut"), true,
                      platform::errors::NotFound(
                          "Output(Beta1PowOut) of LambOp should not be null."));
    PADDLE_ENFORCE_EQ(ctx->HasOutput("Beta2PowOut"), true,
                      platform::errors::NotFound(
                          "Output(Beta2PowOut) of LambOp should not be null."));

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
            "Learning rate should have 1 dimension, but received %d.",
            framework::product(lr_dims)));
    auto beta1_pow_dims = ctx->GetInputDim("Beta1Pow");
    PADDLE_ENFORCE_GE(framework::product(beta1_pow_dims), 1,
                      platform::errors::InvalidArgument(
                          "The size of Beta1 power accumulator should be "
                          "greater than 0, but received %d.",
                          framework::product(beta1_pow_dims)));
    auto beta2_pow_dims = ctx->GetInputDim("Beta2Pow");
    PADDLE_ENFORCE_GE(framework::product(beta2_pow_dims), 1,
                      platform::errors::InvalidArgument(
                          "The size of Beta2 power accumulator should be "
                          "greater than 0, but received %d.",
                          framework::product(beta2_pow_dims)));

    auto param_dims = ctx->GetInputDim("Param");
    if (ctx->GetInputsVarType("Grad")[0] ==
        framework::proto::VarType::LOD_TENSOR) {
      PADDLE_ENFORCE_EQ(
          param_dims, ctx->GetInputDim("Grad"),
          platform::errors::InvalidArgument(
              "Param and Grad input of LambOp should have same dimension. But "
              "received Param dims: [%s], Grad dims: [%s].",
              param_dims, ctx->GetInputDim("Grad")));
    }
    PADDLE_ENFORCE_EQ(
        param_dims, ctx->GetInputDim("Moment1"),
        platform::errors::InvalidArgument(
            "Param and Moment1 input of LambOp should have same dimension. But "
            "received Param dims: [%s], Moment1 dims: [%s].",
            param_dims, ctx->GetInputDim("Moment1")));
    PADDLE_ENFORCE_EQ(
        param_dims, ctx->GetInputDim("Moment2"),
        platform::errors::InvalidArgument(
            "Param and Moment2 input of LambOp should have same dimension. But "
            "received Param dims: [%s], Moment2 dims: [%s].",
            param_dims, ctx->GetInputDim("Moment2")));

    ctx->SetOutputDim("ParamOut", param_dims);
    ctx->SetOutputDim("Moment1Out", param_dims);
    ctx->SetOutputDim("Moment2Out", param_dims);
    ctx->SetOutputDim("Beta1PowOut", beta1_pow_dims);
    ctx->SetOutputDim("Beta2PowOut", beta2_pow_dims);
  }

  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override {
    auto input_data_type =
        OperatorWithKernel::IndicateVarDataType(ctx, "Param");
    return framework::OpKernelType(input_data_type, ctx.GetPlace());
  }
  framework::OpKernelType GetKernelTypeForVar(
      const std::string &var_name, const framework::Tensor &tensor,
      const framework::OpKernelType &expected_kernel_type) const override {
    if (var_name == "Beta1Pow" || var_name == "Beta2Pow") {
      return expected_kernel_type;
    } else {
      return framework::OpKernelType(expected_kernel_type.data_type_,
                                     tensor.place(), tensor.layout());
    }
  }
};

class LambOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("Param",
             "(LoDTensor, default LoDTensor<float>) "
             "Input parameter that has to be updated.");
    AddInput("Grad",
             "(LoDTensor, default LoDTensor<float>) "
             "Input gradient of the parameter.");
    AddInput("LearningRate", "(Tensor) Learning rate.");
    AddInput("Moment1", "(Tensor) Input first moment.");
    AddInput("Moment2", "(Tensor) Input second moment.");
    AddInput("Beta1Pow", "(Tensor) Input beta1 power accumulator.");
    AddInput("Beta2Pow", "(Tensor) Input beta2 power accumulator.");

    AddOutput("ParamOut", "(Tensor) Output parameter.");
    AddOutput("Moment1Out", "(Tensor) Output first moment.");
    AddOutput("Moment2Out", "(Tensor) Output second moment.");
    AddOutput("Beta1PowOut", "(Tensor) Output beta1 power accumulator")
        .AsDispensable();
    AddOutput("Beta2PowOut", "(Tensor) Output beta2 power accumulator")
        .AsDispensable();
    AddAttr<float>("weight_decay", "(float) Weight decay rate.");
    AddAttr<float>("beta1",
                   "(float, default 0.9) The exponential decay rate for the "
                   "1st moment estimates.")
        .SetDefault(0.9);
    AddAttr<float>("beta2",
                   "(float, default 0.999) The exponential decay rate for the "
                   "2nd moment estimates.")
        .SetDefault(0.999);
    AddAttr<float>("epsilon",
                   "(float, default 1.0e-6) "
                   "Constant for numerical stability.")
        .SetDefault(1.0e-6f);

    AddComment(R"DOC(
LAMB (Layer-wise Adaptive Moments optimizer for Batching training) Optimizer.

LAMB Optimizer is designed to scale up the batch size of training without losing 
accuracy, which supports adaptive element-wise updating and accurate layer-wise 
correction. For more information, please refer to https://arxiv.org/abs/1904.00962.

The updating of parameters follows:

$$
m_t &= \beta_1 m_{t - 1}+ (1 - \beta_1)g_t \\

v_t &= \beta_2 v_{t - 1}  + (1 - \beta_2)g_t^2 \\

m_t &= \frac{m_t}{\beta_1^t} \\

v_t &= \frac{v_t}{\beta_2^t} \\

r_t &= \frac{m_t}{\sqrt{v_t}+\epsilon} \\

w_t &= w_{t-1} -\eta_t \frac{\left \| w_{t-1}\right \|}{\left \| r_t + \lambda w_{t-1}\right \|} (r_t + \lambda w_{t-1})
$$

where $m$ is the 1st moment, and $v$ the 2nd moment, $\eta$ the 
learning rate, $\lambda$ the weight decay rate.
)DOC");
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_WITHOUT_GRADIENT(lamb, ops::LambOp, ops::LambOpMaker);
REGISTER_OP_CPU_KERNEL(
    lamb, ops::LambOpKernel<paddle::platform::CPUDeviceContext, float>,
    ops::LambOpKernel<paddle::platform::CPUDeviceContext, double>);

/* ==========================  register checkpoint ===========================*/
REGISTER_OP_VERSION(lamb).AddCheckpoint(
    R"ROC(Upgrade lamb, add two new outputs [Beta1PowOut] and [Beta2PowOut].)ROC",
    paddle::framework::compatible::OpVersionDesc()
        .NewInput("Beta1PowOut",
                  "The Output beta1 power accumulator. 'Beta1PowOut' is "
                  "dispensable.")
        .NewInput("Beta2PowOut",
                  "The Output beta2 power accumulator. 'Beta2PowOut' is "
                  "dispensable."));
