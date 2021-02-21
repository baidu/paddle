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

#ifdef PADDLE_WITH_ASCEND_CL
#include <hccl/hccl.h>
#include <hccl/hccl_types.h>

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

class CCreateGroupOpNPU : public framework::OperatorBase {
 public:
  CCreateGroupOpNPU(const std::string& type,
              const framework::VariableNameMap& inputs,
              const framework::VariableNameMap& outputs,
              const framework::AttributeMap& attrs)
      : OperatorBase(type, inputs, outputs, attrs) {}

  void RunImpl(const framework::Scope& scope,
               const platform::Place& place) const override {
    string group_name = Attr<string>("group_name");
    int nranks = Attr<int>("nranks");
    vector<int> rank_ids = Attr<vector<int>>("rank_ids");
    platform::HCCLCommContext::Instance().CreateHCCLGroup(
        group_name, nranks, rank_ids);
  }
};

class CCreateGroupOpNPUMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddComment(R"DOC(
CCreateGroup operator on NPU

Create collective communication group on NPU
)DOC");
    AddAttr<string>("group_name",
        "(string) name of the collective communication group");
    AddAttr<int>("nranks", "(int) number of the group");
    AddAttr<int>("rank_ids",
                 "(list of int) The world rank id of the group members");
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OPERATOR(c_create_group, ops::CCreateGroupOpNPU,
    ops::CCreateGroupNPUMaker);

#endif
