/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/operators/optimizers/lars_momentum_op.h"
#include "paddle/fluid/operators/optimizers/momentum_op.h"

namespace paddle {
namespace operators {

class LarsMomentumOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("Param",
             "(LoDTensor, default LoDTensor<float>) "
             "Input parameter that has to be updated");
    AddInput("Grad",
             "(LoDTensor, default LoDTensor<float>) "
             "Input gradient of the parameter");
    AddInput("Velocity",
             "(LoDTensor, default LoDTensor<float>) "
             "Input velocity (corresponding to the parameter) "
             "that has to be updated");
    AddInput("LearningRate",
             "(LoDTensor, default LoDTensor<float>) "
             "Input learning rate");
    AddInput("MasterParam", "FP32 master weight for AMP.").AsDispensable();

    AddOutput("ParamOut",
              "(LoDTensor) This output is updated parameter. "
              "It shared memory with Input(Param).");
    AddOutput("VelocityOut",
              "(LoDTensor) This output is updated velocity. "
              "It shared memory with Input(Velocity).");
    AddOutput("MasterParamOut",
              "The updated FP32 master weight for AMP. "
              "It shared memory with Input(MasterParam).")
        .AsDispensable();

    AddAttr<float>("mu", "(float) Momentum coefficient");
    AddAttr<float>("lars_coeff", "(float, default 0.001) LARS coefficient.")
        .SetDefault(0.001);
    AddAttr<float>("lars_weight_decay",
                   "(float, default 0.0005) LARS weight decay")
        .SetDefault(0.0005);
    AddAttr<float>("epsilon",
                   "(float, default 0.0) epsilon to avoid Division by Zero.")
        .SetDefault(0.0);
    AddAttr<bool>("multi_precision",
                  "(bool, default false) "
                  "Whether to use multi-precision during weight updating.")
        .SetDefault(false);
    AddAttr<float>(
        "rescale_grad",
        "(float, default 1.0) Multiply the gradient with `rescale_grad`"
        "before updating. Often choose to be `1.0/batch_size`.")
        .SetDefault(1.0f);

    AddComment(R"DOC(
Lars Momentum Optimizer.

This optimizer use LARS (https://arxiv.org/abs/1708.03888) to optimize each
weight using a local learning rate:

$$
local\_lr = \eta  *
    \frac{\left \| param \right \|}{\left \| grad \right \| + \beta *\left \| param \right \|} \\
velocity = mu * velocity +
    local\_lr * (grad + \beta * param) \\
param = param - velocity. \\
$$

Note that we use lars_weight_decay here to decay weights, you may need not to
use L2 regularizers in case of using LARS.

)DOC");
  }
};

class LarsMomentumOpVarTypeInference : public framework::VarTypeInference {
 public:
  void operator()(framework::InferVarTypeContext* ctx) const override {}
};
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(
    lars_momentum, ops::MomentumOp, ops::LarsMomentumOpMaker,
    paddle::framework::EmptyGradOpMaker<paddle::framework::OpDesc>,
    paddle::framework::EmptyGradOpMaker<paddle::imperative::OpBase>,
    ops::LarsMomentumOpVarTypeInference);
REGISTER_OP_CPU_KERNEL(lars_momentum, ops::LarsMomentumOpKernel<float>,
                       ops::LarsMomentumOpKernel<double>);
