//   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/operators/ascend_trigger_op.h"

namespace paddle {
namespace operators {

class AscendTriggerOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;
  void InferShape(framework::InferShapeContext* ctx) const override {}

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(framework::proto::VarType::FP32,
                                   ctx.device_context());
  }
};

class AscendTriggerOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("FeedList", "FeedList of Ascend SubGraph").AsDuplicable();
    AddOutput("FetchList", "FetchList of Ascend SubGraph").AsDuplicable();
    AddAttr<int>("graph_idx", "(int, the graph index").SetDefault(-1);
    AddComment(R"DOC(
Trigger Ascend SubGraph

)DOC");
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(ascend_trigger, ops::AscendTriggerOp,
                  ops::AscendTriggerOpMaker);
REGISTER_OP_CPU_KERNEL(ascend_trigger, ops::AscendTriggerCPUKernel<float>)
