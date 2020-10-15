/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */
#if defined(PADDLE_WITH_NCCL)
#include <nccl.h>
#endif
#include <stdint.h>
#include <ostream>
#include <string>

#include "paddle/fluid/framework/executor.h"
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/op_info.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/threadpool.h"
#include "paddle/fluid/operators/distributed/distributed.h"
#include "paddle/fluid/operators/distributed/request_handler_impl.h"
#if defined(PADDLE_WITH_NCCL)
#include "paddle/fluid/platform/collective_helper.h"
#include "paddle/fluid/platform/nccl_helper.h"
#endif

namespace paddle {
namespace operators {

class CCommInitAllInferShape : public framework::InferShapeBase {
 public:
  ~CCommInitAllInferShape() {}
  void operator()(framework::InferShapeContext* ctx) const override{};
};

class CCommInitAllOp : public framework::OperatorBase {
 public:
  CCommInitAllOp(const std::string& type,
                 const framework::VariableNameMap& inputs,
                 const framework::VariableNameMap& outputs,
                 const framework::AttributeMap& attrs)
      : OperatorBase(type, inputs, outputs, attrs) {}

  void RunImpl(const framework::Scope& scope,
               const platform::Place& place) const override {
    PADDLE_ENFORCE_EQ(is_gpu_place(place), true,
                      platform::errors::PreconditionNotMet(
                          "CCommInitAllOp can run on gpu place only"));

#if defined(PADDLE_WITH_NCCL)
    std::vector<int> devices = Attr<std::vector<int>>("devices");
    if (devices.empty()) {
      devices = platform::GetSelectedDevices();
    }

    int rid = Attr<int>("ring_id");

    platform::NCCLCommContext::Instance().CreateAllNCCLComms(devices, rid);
#else
    PADDLE_THROW(platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU."));
#endif
  }
};

class CCommInitAllOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddComment(R"DOC(
CCommInitAll operator

Initialize all collective communicatoin context
)DOC");
    AddAttr<std::vector<int>>(
        "devices",
        "(std::vector<int>) which devices does the nccl comm initialized on")
        .SetDefault({});
    AddAttr<int>("ring_id", "(int default 0) user specified ring id")
        .SetDefault(0);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OPERATOR(c_comm_init_all, ops::CCommInitAllOp,
                  ops::CCommInitAllInferShape, ops::CCommInitAllOpMaker);
