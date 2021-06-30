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

#include "paddle/fluid/operators/trans_data_op.h"

namespace paddle {
namespace operators {
class TransDataOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext *ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("X"), "Input", "X", "trans_data");
    OP_INOUT_CHECK(ctx->HasOutput("Out"), "Output", "Out", "trans_data");

    auto input_dims = ctx->GetInputDim("X");
    ctx->SetOutputDim("Out", input_dims);
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override {
    auto data_type = OperatorWithKernel::IndicateVarDataType(ctx, "X");
    return framework::OpKernelType(data_type, ctx.device_context());
  }
};

class TransDataOpProtoMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "Input Tensor that needs to transform data_format");
    AddComment(R"DOC(
TransData operator

This operator only used in NPU device. It will transform data_format of NPU Tensor.
)DOC");
    AddAttr<int>("acl_format", "NPU data_format to be transformed.")
        .SetDefault(true);
    AddOutput("Out", "Tensor that has been transformed.");
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OPERATOR(trans_data, ops::TransDataOp, ops::TransDataOpProtoMaker);
