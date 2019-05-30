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

#include <future>  // NOLINT
#include <ostream>

#include "paddle/fluid/operators/collective/c_broadcast_op.h"

namespace paddle {
namespace operators {

class CBroadcastOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    ctx->SetOutputDim("Out", ctx->GetInputDim("X"));
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(ctx.Input<framework::Tensor>("X")->type(),
                                   ctx.GetPlace());
  }
};

class CBroadcastOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() {
    AddInput("X", "(Tensor) tensor to be broadcasted.");
    AddOutput("Out", "(Tensor) the result of broadcast.");
    AddAttr<int>("ring_id", "(int default 0) nccl communication ring id.")
        .SetDefault(0);
    AddAttr<int>("root", "(int default 0) root id for broadcasting.")
        .SetDefault(0);
    AddComment(R"DOC(
***CBroadcast Operator***

Call ncclBcast internally.
)DOC");
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;

REGISTER_OP_WITHOUT_GRADIENT(c_broadcast, ops::CBroadcastOp,
                             ops::CBroadcastOpMaker);

REGISTER_OP_CPU_KERNEL(
    c_broadcast, ops::CBroadcastOpKernel<plat::CPUDeviceContext, float>,
    ops::CBroadcastOpKernel<plat::CPUDeviceContext, double>,
    ops::CBroadcastOpKernel<plat::CPUDeviceContext, int>,
    ops::CBroadcastOpKernel<plat::CPUDeviceContext, int64_t>,
    ops::CBroadcastOpKernel<plat::CPUDeviceContext, plat::float16>);
