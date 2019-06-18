//   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "paddle/fluid/operators/shard_index_op.h"
#include <string>
#include <vector>
#include "paddle/fluid/framework/framework.pb.h"

namespace paddle {
namespace operators {

class ShardIndexOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;
  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("X"),
                   "Input(X) of ShardIndexOp should not be null.");
    PADDLE_ENFORCE(ctx->HasOutput("Out"),
                   "Output(Out) of ShardIndexOp should not be null.");

    auto x_dims = ctx->GetInputDim("X");
    PADDLE_ENFORCE_GE(x_dims.size(), 2,
                      "Rank of Input(X) should be at least 2.");
    if (ctx->IsRuntime() || x_dims[x_dims.size() - 1] > 0) {
      PADDLE_ENFORCE_GE(x_dims[x_dims.size() - 1], 1U,
                        "Last dimension of Input(X) should be 1.");
    }

    ctx->SetOutputDim("Out", x_dims);
    ctx->ShareLoD("X", /* --> */ "Out");
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(ctx.Input<framework::Tensor>("X")->type(),
                                   ctx.device_context());
  }
};

class ShardIndexOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X",
             "(LoDTensor, LoDTensor<int>) Input variable with rank at least 2. "
             "The last dimension of X should be 1. Each value of X is an index "
             "to indicate the position.");
    AddOutput(
        "Out",
        "(Tensor, Tensor<float>) Output tensor with same shape as X. "
        "The tensor consists of sharding representations of values in X.");

    AddAttr<int>("range",
                 "A positive integer to specify the range of the label.");
    AddAttr<int>("nshards", "The number of shards for dividing label range");
    AddAttr<int>("shard_id", "The shard index");
    AddComment(R"DOC(
Shard Index Operator. This operator creates the sharding representations for input
index values. The following example will help to explain the function of this
operator:

X is a LoDTensor:
  X.lod = [[0, 1, 4]]
  X.shape = [4, 1]
  X.data = [[1], [2], [3], [0]]

set range = 4, nshards = 2

Out is a LoDTensor:
  Out.lod = [[0, 1, 4]]
  Out.shape = [4, 1]
  Out.data = [[1], [0], [1], [0]]

)DOC");
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_WITHOUT_GRADIENT(shard_index, ops::ShardIndexOp,
                             ops::ShardIndexOpMaker);
REGISTER_OP_CPU_KERNEL(
    shard_index, ops::ShardIndexKernel<paddle::platform::CPUDeviceContext, int>,
    ops::ShardIndexKernel<paddle::platform::CPUDeviceContext, int64_t>);
