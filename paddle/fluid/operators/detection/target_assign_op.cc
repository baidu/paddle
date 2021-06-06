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

#include "paddle/fluid/operators/detection/target_assign_op.h"

namespace paddle {
namespace operators {

class TargetAssignOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE_EQ(ctx->HasInput("X"), true,
                      platform::errors::InvalidArgument(
                          "Input(X) of TargetAssignOp should not be null"));
    PADDLE_ENFORCE_EQ(
        ctx->HasInput("MatchIndices"), true,
        platform::errors::InvalidArgument(
            "Input(MatchIndices) of TargetAssignOp should not be null"));

    PADDLE_ENFORCE_EQ(ctx->HasOutput("Out"), true,
                      platform::errors::InvalidArgument(
                          "Output(Out) of TargetAssignOp should not be null."));
    PADDLE_ENFORCE_EQ(
        ctx->HasOutput("OutWeight"), true,
        platform::errors::InvalidArgument(
            "Output(OutWeight) of TargetAssignOp should not be null."));

    auto in_dims = ctx->GetInputDim("X");
    auto mi_dims = ctx->GetInputDim("MatchIndices");

    PADDLE_ENFORCE_EQ(
        in_dims.size() == 3UL || in_dims.size() == 4UL, true,
        platform::errors::InvalidArgument(
            "Expected the rank of Input(X) is 3 or 4. But received %d.",
            in_dims.size()));
    PADDLE_ENFORCE_EQ(mi_dims.size(), 2,
                      platform::errors::InvalidArgument(
                          "The rank of Input(MatchIndices) must be 2."));

    if (ctx->HasInput("NegIndices")) {
      auto neg_dims = ctx->GetInputDim("NegIndices");
      PADDLE_ENFORCE_EQ(
          neg_dims.size() == 2UL || neg_dims.size() == 3UL, true,
          platform::errors::InvalidArgument(
              "Expected the rank of Input(NegIndices) must be 2 or 3. "
              "But received %d.",
              neg_dims.size()));
      PADDLE_ENFORCE_EQ(
          neg_dims[neg_dims.size() - 1], 1,
          platform::errors::InvalidArgument(
              "The last dimension of Input(NegIndices) must be 1."));
    }

    auto n = mi_dims[0];
    auto m = mi_dims[1];
    auto k = in_dims[in_dims.size() - 1];
    ctx->SetOutputDim("Out", {n, m, k});
    ctx->SetOutputDim("OutWeight", {n, m, 1});
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(
        OperatorWithKernel::IndicateVarDataType(ctx, "X"),
        ctx.device_context());
  }
};

class TargetAssignOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X",
             "(LoDTensor or Tensor), When the input is LoDTensor, its shape "
             "must be [M, P, K]. When the input is Tensor, its shape can be "
             "[B, F, K] or [B, F, A, K], where M = B * F. "
             "Some elements in X will be assigned to Out based on the "
             "MatchIndices and NegIndices.");
    AddInput("MatchIndices",
             "(Tensor, default Tensor<int>), The input matched indices "
             "with shape [N, P], If MatchIndices[i][j] is -1, the j-th entity "
             "of column is not matched to any entity of row in i-th instance.");
    AddInput("NegIndices",
             "(LoDTensor or Tensor, default LoDTensor<int>), The input "
             "negative example indices are an optional input with shape "
             "[Neg, 1], where Neg is the total number of negative example "
             "indices and input is LoDTensor. When the input is Tensor, "
             "its shape must be [B, M, 1], where Neg = B * M and it becomes "
             "negative example mask. ")
        .AsDispensable();
    AddAttr<int>("mismatch_value",
                 "(int, default 0), Fill this value to the "
                 "mismatched location.")
        .SetDefault(0);
    AddOutput("Out",
              "(Tensor), The output is a 3D Tensor with shape [N, P, K], "
              "N and P is the same as they are in NegIndices, K is the "
              "same as it in input of X. If MatchIndices[i][j] "
              "is -1, the Out[i][j][0 : K] is the mismatch_value.");
    AddOutput("OutWeight",
              "(Tensor), The weight for output with the shape of [N, P, 1]");
    AddComment(R"DOC(
This operator can be, for given the target bounding boxes or labels,
to assign classification and regression targets to each prediction as well as
weights to prediction. The weights is used to specify which prediction would
not contribute to training loss.

For each instance, the output `Out` and`OutWeight` are assigned based on
`MatchIndices` and `NegIndices`.
Assumed that the row offset for each instance in `X` is called lod,
this operator assigns classification/regression targets by performing the
following steps:

1. Assigning all outpts based on `MatchIndices`:

If id = MatchIndices[i][j] > 0,

    Out[i][j][0 : K] = X[lod[i] + id][j % P][0 : K]
    OutWeight[i][j] = 1.

Otherwise,

    Out[j][j][0 : K] = {mismatch_value, mismatch_value, ...}
    OutWeight[i][j] = 0.

2. Assigning OutWeight based on `NegIndices` if `NegIndices` is provided:

Assumed that the row offset for each instance in `NegIndices` is called neg_lod,
for i-th instance and each `id` of NegIndices in this instance:

    Out[i][id][0 : K] = {mismatch_value, mismatch_value, ...}
    OutWeight[i][id] = 1.0

    )DOC");
  }
};

template <typename T, typename WT>
struct NegTargetAssignFunctor<platform::CPUDeviceContext, T, WT> {
  void operator()(const platform::CPUDeviceContext& ctx, const int* neg_indices,
                  const size_t* lod, const int N, const int M, const int K,
                  const int mismatch_value, T* out, WT* out_wt) {
    for (int i = 0; i < N; ++i) {
      for (size_t j = lod[i]; j < lod[i + 1]; ++j) {
        int id = neg_indices[j];
        int off = (i * M + id) * K;
        for (int k = 0; k < K; ++k) {
          out[off + k] = mismatch_value;
          out_wt[off + k] = static_cast<WT>(1.0);
        }
      }
    }
  }
};

template struct NegTargetAssignFunctor<platform::CPUDeviceContext, int, float>;
template struct NegTargetAssignFunctor<platform::CPUDeviceContext, float,
                                       float>;

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(
    target_assign, ops::TargetAssignOp, ops::TargetAssignOpMaker,
    paddle::framework::EmptyGradOpMaker<paddle::framework::OpDesc>,
    paddle::framework::EmptyGradOpMaker<paddle::imperative::OpBase>);
REGISTER_OP_CPU_KERNEL(
    target_assign,
    ops::TargetAssignKernel<paddle::platform::CPUDeviceContext, int, float>,
    ops::TargetAssignKernel<paddle::platform::CPUDeviceContext, float, float>);
