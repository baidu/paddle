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

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/platform/profiler.h"

namespace paddle {
namespace operators {

class MarkerOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;
  void InferShape(framework::InferShapeContext* ctx) const override {
    std::string marker_role = ctx->Attrs().Get<std::string>("marker_role");
    std::string marker_pos = ctx->Attrs().Get<std::string>("marker_pos");

    VLOG(3) << "The role is:" << marker_role << ";"
            << "The position is:" << marker_pos << ".";
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(framework::proto::VarType::FP32,
                                   ctx.GetPlace());
  }
};

class MarkerOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddAttr<std::string>("marker_role",
                         "(string, default forward)forward or backward,"
                         " mark different stages of porcess.")
        .SetDefault("forward");
    AddAttr<std::string>(
        "marker_pos",
        "(string, default B)the posititon where the marker is placed, "
        "B stands for begin of duration,"
        " E stands for end of duration.")
        .SetDefault("B");
    AddComment(
        R"DOC(Marker Operator - Add marker at the beginning/end of a forward/backward process.)DOC");
  }
};

template <typename T>
class MarkerOpCPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto marker_role = ctx.Attr<std::string>("marker_role");
    auto marker_pos = ctx.Attr<std::string>("marker_pos");

    platform::RecordEvent record_event(
        "MarkerCPU", platform::EventRole::kInnerOp,
        "marker_" + marker_role + "_" + marker_pos);
  }
};
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OP_WITHOUT_GRADIENT(marker, ops::MarkerOp, ops::MarkerOpMaker);
REGISTER_OP_CPU_KERNEL(marker, ops::MarkerOpCPUKernel<float>);
