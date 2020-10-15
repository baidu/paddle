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

#include "paddle/fluid/operators/top_k_v2_op.h"
#include <memory>

namespace paddle {
namespace operators {

class TopkV2Op : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("X"), "Input", "X", "topk_v2");
    OP_INOUT_CHECK(ctx->HasOutput("Out"), "Output", "Out", "topk_v2");
    OP_INOUT_CHECK(ctx->HasOutput("Indices"), "Output", "Indices", "topk_v2");

    auto input_dims = ctx->GetInputDim("X");
    const int& dim_size = input_dims.size();
    int axis = static_cast<int>(ctx->Attrs().Get<int>("axis"));
    PADDLE_ENFORCE_EQ(
        (axis < dim_size) && (axis >= (-1 * dim_size)), true,
        paddle::platform::errors::InvalidArgument(
            "the axis of topk must be [-%d, %d), but you set axis is %d",
            dim_size, dim_size, axis));

    if (axis < 0) axis += dim_size;

    int k;
    auto k_is_tensor = ctx->HasInput("K");
    if (k_is_tensor) {
      k = -1;
    } else {
      k = static_cast<int>(ctx->Attrs().Get<int>("k"));
      PADDLE_ENFORCE_EQ(k >= 1, true,
                        paddle::platform::errors::InvalidArgument(
                            "the attribute of k in the topk must >= 1 or be a "
                            "Tensor, but received %d .",
                            k));
    }

    PADDLE_ENFORCE_GE(input_dims.size(), 1,
                      paddle::platform::errors::InvalidArgument(
                          "input of topk must have >= 1d shape"));

    if (ctx->IsRuntime()) {
      PADDLE_ENFORCE_GE(
          input_dims[axis], k,
          paddle::platform::errors::InvalidArgument(
              "input of topk op must have >= %d columns in axis of %d", k,
              axis));
    }

    framework::DDim dims = input_dims;

    dims[axis] = k;
    ctx->SetOutputDim("Out", dims);
    ctx->SetOutputDim("Indices", dims);
    ctx->ShareLoD("X", "Out");
    ctx->ShareLoD("X", "Indices");
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    framework::LibraryType library_{framework::LibraryType::kPlain};
    framework::DataLayout layout_ = framework::DataLayout::kAnyLayout;
    return framework::OpKernelType(
        OperatorWithKernel::IndicateVarDataType(ctx, "X"), ctx.device_context(),
        layout_, library_);
  }
};

class TopkV2OpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "(Tensor) The input of Topk op");
    AddInput("K",
             "(Tensor)  Number of top elements to look for along "
             "the last dimension (along each row for matrices).")
        .AsDispensable();
    AddOutput("Out", "(Tensor) The output tensor of Topk op");
    AddOutput("Indices", "(Tensor) The indices of Topk elements of input");
    AddComment(R"DOC(
Top K operator

If the input is a vector (1d tensor), this operator finds the k largest 
entries in the vector and outputs their values and indices as vectors. 
Thus values[j] is the j-th largest entry in input, and its index is indices[j].

For matrices, this operator computes the top k entries in each row. )DOC");
    AddAttr<int>("k",
                 "(int, default 1) Number of top elements to look for along "
                 "the tensor).")
        .SetDefault(1);
    AddAttr<int>("axis",
                 "the axis to sort and get the k indices, value."
                 "if not set, will get k value in last axis.")
        .SetDefault(-1);
    AddAttr<bool>("largest",
                  "control flag whether to return largest or smallest")
        .SetDefault(true);
    AddAttr<bool>("sorted",
                  "control flag whether to return elements in sorted order")
        .SetDefault(true);
  }
};

class TopkV2OpGrad : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;
  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE_EQ(
        ctx->HasInput("X"), true,
        platform::errors::InvalidArgument("Input(X) should be not null"));
    PADDLE_ENFORCE_EQ(
        ctx->HasInput("Indices"), true,
        platform::errors::InvalidArgument("Input(Indices) should be not null"));
    PADDLE_ENFORCE_EQ(ctx->HasInput(framework::GradVarName("Out")), true,
                      platform::errors::InvalidArgument(
                          "Grad Input(Out) should be not null"));
    PADDLE_ENFORCE_EQ(
        ctx->HasOutput(framework::GradVarName("X")), true,
        platform::errors::InvalidArgument("Grad Output(X) should be not null"));

    auto x_dims = ctx->GetInputDim("X");
    ctx->SetOutputDim(framework::GradVarName("X"), x_dims);
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    auto data_type = OperatorWithKernel::IndicateVarDataType(
        ctx, framework::GradVarName("Out"));
    return framework::OpKernelType(data_type, ctx.device_context());
  }
};

template <typename T>
class TopkV2GradOpMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> op) const override {
    op->SetType("top_k_v2_grad");
    op->SetInput(framework::GradVarName("Out"), this->OutputGrad("Out"));
    op->SetInput("X", this->Input("X"));
    op->SetInput("Indices", this->Output("Indices"));
    op->SetOutput(framework::GradVarName("X"), this->InputGrad("X"));
    op->SetAttrMap(this->Attrs());
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(top_k_v2, ops::TopkV2Op, ops::TopkV2OpMaker,
                  ops::TopkV2GradOpMaker<paddle::framework::OpDesc>,
                  ops::TopkV2GradOpMaker<paddle::imperative::OpBase>);

REGISTER_OPERATOR(top_k_v2_grad, ops::TopkV2OpGrad);

REGISTER_OP_CPU_KERNEL(top_k_v2,
                       ops::TopkV2Kernel<paddle::platform::CPUPlace, float>,
                       ops::TopkV2Kernel<paddle::platform::CPUPlace, double>,
                       ops::TopkV2Kernel<paddle::platform::CPUPlace, int32_t>,
                       ops::TopkV2Kernel<paddle::platform::CPUPlace, int64_t>)

REGISTER_OP_CPU_KERNEL(
    top_k_v2_grad, ops::TopkV2GradKernel<paddle::platform::CPUPlace, float>,
    ops::TopkV2GradKernel<paddle::platform::CPUPlace, double>,
    ops::TopkV2GradKernel<paddle::platform::CPUPlace, int32_t>,
    ops::TopkV2GradKernel<paddle::platform::CPUPlace, int64_t>)
