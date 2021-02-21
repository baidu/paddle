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

#if defined(PADDLE_WITH_ASCEND_CL)
#include "paddle/fluid/platform/collective_helper.h"
#include "paddle/fluid/platform/hccl_helper.h"

#include <string>

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/npu_op_runner.h"

namespace paddle {
namespace framework {
class Scope;
}  // namespace framework
}  // namespace paddle

namespace paddle {
namespace operators {

class CCommInitOpNPU : public framework::OperatorBase {
 public:
  CCommInitOpNPU(const std::string& type,
              const framework::VariableNameMap& inputs,
              const framework::VariableNameMap& outputs,
              const framework::AttributeMap& attrs)
      : OperatorBase(type, inputs, outputs, attrs) {}

  void RunImpl(const framework::Scope& scope,
               const platform::Place& place) const override {
    std::string rank_table_file = Attr<std::string>("rank_table_file");
    int rank_id = Attr<int>("rank_id");
    int device_id = Attr<int>("device_id");
    platform::HCCLCommContext::Instance().CreateHCCLComm(rank_table_file,
      rank_id, device_id);
  }
};

class CCommInitOpNPUMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddComment(R"DOC(
CCommInit operator on NPU

Initialize collective communication context within this trainer
)DOC");
    AddAttr<std::string>("rank_table_file",
        "(string) path to rank_table_file");
    AddAttr<int>("rank_id", "(int) world rank id of the process");
    AddAttr<int>("device_id", "(int) device id of the process/thread");
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OPERATOR(c_comm_init, ops::CCommInitOpNPU,
   ops::CCommInitOpNPUMaker);

#endif
