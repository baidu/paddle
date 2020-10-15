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

#include "paddle/fluid/operators/distributed_ops/split_ids_op.h"

#include <memory>

namespace paddle {
namespace operators {

class SplitIdsOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("Ids", "(LoDTensor) the input ids with shape{batch_num, 1}")
        .AsDuplicable();

    AddOutput("Out", "(LoDTensors) The outputs of the input Ids.")
        .AsDuplicable();

    AddComment(R"DOC(
Split a LoDTensor of Ids into multi LoDTensors, the number is pserver's number
Example:
  Input:
    X = [[1,2,3,4,5,6],[2,3]]

  Out(3 output):
    if compress is True:
        out0 = [3, 3, 6]
        out1 = [1, 4]
        out2 = [2, 2, 5]
    else:
        out0 = [3, 6]
        out1 = [1, 4]
        out2 = [2, 5]
)DOC");
  }
};

class SplitIdsOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext *ctx) const override {
    OP_INOUT_CHECK(ctx->HasInputs("Ids"), "Input", "Ids", "SplitIdsOp");
    OP_INOUT_CHECK(ctx->HasOutputs("Out"), "Output", "Out", "SplitIdsOp");

    auto ids_var_type = ctx->GetInputsVarType("Ids").front();
    auto ids_dims = ctx->GetInputsDim("Ids");
    if (ids_var_type == framework::proto::VarType::LOD_TENSOR) {
      PADDLE_ENFORCE_EQ(
          ids_dims[0].size(), 2,
          platform::errors::InvalidArgument(
              "ShapeError: The dimensions of the 'split_ids' must be 2. "
              "But received split_ids's dimensions = %d, "
              "split_ids's shape = [%s].",
              ids_dims[0].size(), ids_dims[0]));
    }
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override {
    return framework::OpKernelType(
        OperatorWithKernel::IndicateVarDataType(ctx, "Ids"), ctx.GetPlace());
  }
};

class SplitIdsOpInferVarType : public framework::VarTypeInference {
 public:
  void operator()(framework::InferVarTypeContext *ctx) const override {
    auto input_type = ctx->GetInputType("Ids");
    ctx->SetOutputType("Out", input_type, framework::ALL_ELEMENTS);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(split_ids, ops::SplitIdsOp, ops::SplitIdsOpMaker,
                  ops::SplitIdsOpInferVarType);

REGISTER_OP_CPU_KERNEL(
    split_ids, ops::SplitIdsOpKernel<paddle::platform::CPUPlace, int64_t>,
    ops::SplitIdsOpKernel<paddle::platform::CPUPlace, float>);
