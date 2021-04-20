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

#ifdef PADDLE_WITH_ASCEND_CL
#include "paddle/fluid/operators/collective/gen_hccl_id_op_helper.h"
#endif

namespace paddle {
namespace operators {

#ifdef PADDLE_WITH_ASCEND_CL

class CGenHCCLIdOp : public framework::OperatorBase {
 public:
  CGenHCCLIdOp(const std::string& type,
               const framework::VariableNameMap& inputs,
               const framework::VariableNameMap& outputs,
               const framework::AttributeMap& attrs)
      : OperatorBase(type, inputs, outputs, attrs) {
  }

  void RunImpl(const framework::Scope& scope,
               const platform::Place& dev_place) const override {
    int rank = Attr<int>("rank");
    framework::Scope& local_scope = scope.NewScope();

    std::function<std::string(size_t)> func = [&](size_t i) -> std::string {
      return Output("Out");
    };

    if (rank == 0) {
      std::vector<std::string> endpoint_list =
          Attr<std::vector<std::string>>("other_endpoints");
      SendBroadCastHCCLID(endpoint_list, 1, func, local_scope);
    } else {
      std::string endpoint = Attr<std::string>("endpoint");
      RecvBroadCastHCCLID(endpoint, 1, func, local_scope);
    }
    scope.DeleteScope(&local_scope);
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
               const platform::Place& dev_place) const override {
  }
};

#endif

class CGenHCCLIdOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    VLOG(3) << "ele";
    AddOutput("Out", "Raw variable contains a HCCL UniqueId instaces.");
    AddComment(R"DOC(
CGenHCCLId operator

For trainer 0: generate a new UniqueId and send it to all the other trainers.
For trainer 1~n: start a gRPC server to get the UniqueId, once got, stop the server.
)DOC");
    AddAttr<std::string>("endpoint",
                         "(string), e.g. 127.0.0.1:6175 "
                         "current listen endpoint");
    AddAttr<std::vector<std::string>>(
        "other_endpoints",
        "['trainer1_ip:port', 'trainer2_ip:port', ...] "
        "list of other trainer endpoints")
        .SetDefault({});
    AddAttr<int>("rank",
                 "(int default 0) "
                 "The rank of the trainer in distributed training.")
        .SetDefault(0);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OPERATOR(c_gen_hccl_id, ops::CGenHCCLIdOp, ops::CGenHCCLIdOpMaker);
