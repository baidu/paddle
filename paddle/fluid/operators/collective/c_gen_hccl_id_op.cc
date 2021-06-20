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
#include <string>

#include "glog/logging.h"
#include "paddle/fluid/framework/op_proto_maker.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/framework/var_type_traits.h"
#include "paddle/fluid/platform/device_context.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/fluid/platform/place.h"
#include "paddle/fluid/platform/dynload/hccl.h"

namespace paddle {
namespace operators {

#ifdef PADDLE_WITH_ASCEND_CL

class CGenHCCLIdOp : public framework::OperatorBase {
 public:
  CGenHCCLIdOp(const std::string& type,
               const framework::VariableNameMap& inputs,
               const framework::VariableNameMap& outputs,
               const framework::AttributeMap& attrs)
      : OperatorBase(type, inputs, outputs, attrs) {}

  void RunImpl(const framework::Scope& scope,
               const platform::Place& dev_place) const override {
    int rank = Attr<int>("rank");
    int rank_count = Attr<int>("rank_count");
    std::string endpoint = Attr<std::string>("endpoint");
    std::string group_name = Attr<std::string>("group_name");
    int split_index = Attr<int>("split_index");

    VLOG(2) << "rank = " << rank
            << ", endpoint = " << endpoint
            << ", rank_count = " << rank_count
            << ", group_name = " << group_name
            << ", split_index = " << split_index;

    PADDLE_ENFORCE_NPU_SUCCESS(platform::dynload::eccl_gen_unique_id(rank,
                            endpoint.c_str(), rank_count,
                            split_index, group_name.c_str()));
  }
};

#else

class CGenHCCLIdOp : public framework::OperatorBase {
 public:
  CGenHCCLIdOp(const std::string& type,
               const framework::VariableNameMap& inputs,
               const framework::VariableNameMap& outputs,
               const framework::AttributeMap& attrs)
      : OperatorBase(type, inputs, outputs, attrs) {}

  void RunImpl(const framework::Scope& scope,
               const platform::Place& dev_place) const override {}
};

#endif

class CGenHCCLIdOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    VLOG(3) << "ele";
    // AddOutput("Out", "Raw variable contains a HCCL UniqueId instaces.");
    AddComment(R"DOC(
CGenHCCLId operator

For trainer 0: generate a new UniqueId and send it to all the other trainers.
For trainer 1~n: start a gRPC server to get the UniqueId, once got, stop the server.
)DOC");
    AddAttr<std::string>("endpoint",
                         "(string), e.g. 127.0.0.1:6175 "
                         "Common store for booststrap, usually the ip of rank0 in each group");

    AddAttr<std::string>("group_name",
                         "(string), e.g. world_group  "
                         "The group id used for ECCL, which may map to ringid!");

    AddAttr<int>("rank",
                 "(int default 0) "
                 "The rank of the trainer in distributed training.")
        .SetDefault(0);

    AddAttr<int>("rank_count",
                 "(int default 0) "
                 "The rank number in distributed training.")
        .SetDefault(0);

    AddAttr<int>("split_index",
                 "(int default 0) "
                 "The position that split the pod.")
        .SetDefault(0);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OPERATOR(c_gen_hccl_id, ops::CGenHCCLIdOp, ops::CGenHCCLIdOpMaker);
