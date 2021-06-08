/* Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/operators/partial_sum_op.h"

#include <memory>
#include <string>
#include <vector>

namespace paddle {
namespace operators {
using Tensor = framework::Tensor;

class PartialSumOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext *ctx) const override {
    PADDLE_ENFORCE_GE(ctx->Inputs("X").size(), 1UL,
                      platform::errors::InvalidArgument(
                          "Inputs(X) of PartialSumOp should not be empty."));

    PADDLE_ENFORCE_EQ(ctx->HasOutput("Out"), true,
                      platform::errors::InvalidArgument(
                          "Output(Out) of PartialSumOp should not be null."));

    auto inputs_dims = ctx->GetInputsDim("X");

    const size_t inputs_num = inputs_dims.size();
    PADDLE_ENFORCE_GT(inputs_num, 0,
                      platform::errors::InvalidArgument(
                          "ShapeError: Input tensors count should > 0. But "
                          "recevied inputs' length is 0."));
    if (inputs_num == 1) {
      VLOG(3) << "Warning: partial_sum op have only one input, may be useless";
    }

    int start_index = ctx->Attrs().Get<int>("start_index");
    int length = ctx->Attrs().Get<int>("length");

    // Only suppert two dimensions now, should be extended later
    // when length is -1, need make sure all dimensions to be added are the same
    int64_t batch_size = -1;
    int64_t input_len = -1;
    for (size_t i = 0; i < inputs_num; ++i) {
      PADDLE_ENFORCE_EQ(inputs_dims[i].size(), 2,
                        platform::errors::InvalidArgument(
                            "Only suppert two dimensions input now."));
      if (i == 0) {
        batch_size = inputs_dims[0][0];
        input_len = inputs_dims[0][1];
      } else {
        PADDLE_ENFORCE_EQ(inputs_dims[i][0], batch_size,
                          platform::errors::InvalidArgument(
                              "The batch size of all inputs must be same"));
        PADDLE_ENFORCE_EQ(inputs_dims[i][1], input_len,
                          platform::errors::InvalidArgument(
                              "The input len of all inputs must be same"));
      }
    }
    PADDLE_ENFORCE_GT(input_len, start_index,
                      platform::errors::OutOfRange(
                          "start_index must be less than input len"));
    if (length > 0) {
      PADDLE_ENFORCE_GE(
          input_len, start_index + length,
          platform::errors::OutOfRange(
              "start_index + length is larger than input length"));
    }

    std::vector<int64_t> out_dims(2);
    out_dims[0] = batch_size;
    out_dims[1] = (length == -1) ? input_len - start_index : length;
    ctx->SetOutputDim("Out", framework::make_ddim(out_dims));
    ctx->ShareLoD("X", /*->*/ "Out");
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override {
    auto inputs = ctx.MultiInput<Tensor>("X");
    auto input_data_type = framework::proto::VarType::Type(0);
    bool flag = false;
    for (auto *input : inputs) {
      if (input->IsInitialized() && input->numel() > 0) {
        input_data_type = input->type();
        flag = true;
        break;
      }
    }

    PADDLE_ENFORCE_EQ(flag, 1,
                      platform::errors::InvalidArgument(
                          "All Inputs of PartialSum OP are Empty!"));
    return framework::OpKernelType(input_data_type, platform::CPUPlace());
  }
};

class PartialSumGradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext *ctx) const override {
    auto in_x = "X";
    auto out_x_g_n = framework::GradVarName(in_x);
    ctx->SetOutputsDim(out_x_g_n, ctx->GetInputsDim(in_x));

    auto in_names = ctx->Inputs(in_x);
    auto out_names = ctx->Outputs(out_x_g_n);

    PADDLE_ENFORCE_EQ(
        in_names.size(), out_names.size(),
        platform::errors::InvalidArgument(
            "The number of arguments in %s[%d] and %s[%d] is not equal.", in_x,
            in_names.size(), out_x_g_n, out_names.size()));
    for (size_t i = 0; i < in_names.size(); ++i) {
      if (out_names[i] != framework::kEmptyVarName) {
        ctx->ShareLoD(in_x, out_x_g_n, i, i);
      }
    }
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override {
    return framework::OpKernelType(OperatorWithKernel::IndicateVarDataType(
                                       ctx, framework::GradVarName("Out")),
                                   ctx.device_context());
  }
};

class PartialSumOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "Input tensors of partial_sum operator.").AsDuplicable();
    AddOutput("Out", "Output tensor of partial_sum operator.");
    AddAttr<bool>(
        "use_mkldnn",
        "(bool, default false) Indicates if MKL-DNN kernel will be used")
        .SetDefault(false);
    AddAttr<int>("start_index", "The start index of tensor wanted to be added.")
        .SetDefault(0);
    AddAttr<int>("length", "The length of tensor wanted to be added.")
        .SetDefault(-1);
    AddComment(R"DOC(
PartialSum Operator.
This Op can sum the vars by specifying the initial position(start_index) and length(length). 
This OP exists in contrib, which means that it is not shown to the public.
Only 2-D Tensor or LodTensor input is supported. Slice and concat can only be 
performed along the second dimension.

Examples:
  Input[0] = [[1,2,3],[3,4,5]]
  Input[1] = [[5,6,7],[7,8,9]]
  start_index = 0
  length = 2
  Output = [[6,8],
            [10,12]]
)DOC");
  }
};

template <typename T>
class PartialSumGradMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> op) const override {
    op->SetType("partial_sum_grad");
    op->SetInput("X", this->Input("X"));
    op->SetInput(framework::GradVarName("Out"), this->OutputGrad("Out"));
    op->SetOutput(framework::GradVarName("X"), this->InputGrad("X", false));
    op->SetAttr("start_index", this->GetAttr("start_index"));
    op->SetAttr("length", this->GetAttr("length"));
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(partial_sum, ops::PartialSumOp, ops::PartialSumOpMaker,
                  ops::PartialSumGradMaker<paddle::framework::OpDesc>,
                  ops::PartialSumGradMaker<paddle::imperative::OpBase>);

REGISTER_OPERATOR(partial_sum_grad, ops::PartialSumGradOp);

REGISTER_OP_CPU_KERNEL(
    partial_sum,
    ops::PartialSumKernel<paddle::platform::CPUDeviceContext, float>,
    ops::PartialSumKernel<paddle::platform::CPUDeviceContext, int>,
    ops::PartialSumKernel<paddle::platform::CPUDeviceContext, double>,
    ops::PartialSumKernel<paddle::platform::CPUDeviceContext, int64_t>);

REGISTER_OP_CPU_KERNEL(partial_sum_grad, ops::PartialSumGradientOpKernel<float>,
                       ops::PartialSumGradientOpKernel<int>,
                       ops::PartialSumGradientOpKernel<double>,
                       ops::PartialSumGradientOpKernel<int64_t>);
