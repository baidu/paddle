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
#include "paddle/fluid/framework/executor.h"
#include "paddle/fluid/framework/op_proto_maker.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/framework/program_desc.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/framework/var_type_traits.h"
#include "paddle/fluid/operators/distributed/distributed.h"
#include "paddle/fluid/operators/distributed/request_handler.h"
#include "paddle/fluid/operators/distributed/request_handler_impl.h"
#include "paddle/fluid/operators/distributed/rpc_client.h"
#include "paddle/fluid/platform/device_context.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/fluid/platform/place.h"

namespace paddle {
namespace operators {

class CGenNCCLIdOp : public framework::OperatorBase {
 public:
  CGenNCCLIdOp(const std::string& type,
               const framework::VariableNameMap& inputs,
               const framework::VariableNameMap& outputs,
               const framework::AttributeMap& attrs)
      : OperatorBase(type, inputs, outputs, attrs) {}

  void RunImpl(const framework::Scope& scope,
               const platform::Place& dev_place) const override {
    platform::DeviceContextPool& pool = platform::DeviceContextPool::Instance();
    // put nccl id in CPUPlace
    auto& dev_ctx = *pool.Get(platform::CPUPlace());
    int rank = Attr<int>("rank");
    framework::Scope& local_scope = scope.NewScope();

    if (rank == 0) {
      GenerateAndSend(&local_scope, dev_ctx);
    } else {
      GetIdByServer(&local_scope, dev_ctx);
    }
    scope.DeleteScope(&local_scope);
  }

 private:
  void GenerateAndSend(framework::Scope* scope,
                       const platform::DeviceContext& dev_ctx) const {
    std::string var_name = Output("Out");
    auto var = scope->FindVar(var_name);
    PADDLE_ENFORCE_NOT_NULL(
        var, platform::errors::InvalidArgument("Output can not be Null"));
    auto id = var->GetMutable<ncclUniqueId>();
    PADDLE_ENFORCE_EQ(platform::dynload::ncclGetUniqueId(id), 0,
                      platform::errors::InvalidArgument(
                          "ncclGetUniqueId failed with id %s", id));

    std::vector<std::string> endpoint_list =
        Attr<std::vector<std::string>>("other_endpoints");
    distributed::RPCClient* client =
        distributed::RPCClient::GetInstance<RPCCLIENT_T>(0);

    for (auto& ep : endpoint_list) {
      VLOG(3) << "sending nccl id to " << ep;
      client->AsyncSendVar(ep, dev_ctx, *scope, var_name);
    }
    client->Wait();
    for (auto& ep : endpoint_list) {
      client->AsyncSendBatchBarrier(ep);
    }
    client->Wait();
    VLOG(3) << "sending completed...";
  }

  void GetIdByServer(framework::Scope* scope,
                     const platform::DeviceContext& dev_ctx) const {
    std::string endpoint = Attr<std::string>("endpoint");
    // NOTE: Can not use unique_ptr here because the default
    // deleter will call GRPC Server's base class's dtor and
    // that will cause a wired crash.
    distributed::RequestSendHandler rpc_h(distributed::DistributedMode::kSync);
    std::unique_ptr<distributed::RPCServer> rpc_service(
        new RPCSERVER_T(endpoint, 1));

    rpc_service->RegisterRPC(distributed::kRequestSend, &rpc_h);
    rpc_h.SetRPCServer(rpc_service.get());

    framework::ProgramDesc empty_program;
    framework::Executor executor(dev_ctx.GetPlace());
    rpc_h.SetScope(scope);
    rpc_h.SetDevCtx(&dev_ctx);
    rpc_h.SetProgram(&empty_program);
    rpc_h.SetExecutor(&executor);

    std::thread server_thread(
        std::bind(&distributed::RPCServer::StartServer, rpc_service.get()));

    rpc_service->SetCond(distributed::kRequestSend);
    VLOG(3) << "start getting nccl id from trainer 0...";
    rpc_service->WaitBarrier(distributed::kRequestSend);
    VLOG(3) << "got nccl id and stop server...";
    rpc_service->ShutDown();
    VLOG(3) << "rpc server stopped";
    server_thread.join();
  }
};

class CGenNCCLIdOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddOutput("Out", "Raw variable contains a NCCL UniqueId instaces.");
    AddComment(R"DOC(
CGenNCCLId operator

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

REGISTER_OPERATOR(c_gen_nccl_id, ops::CGenNCCLIdOp, ops::CGenNCCLIdOpMaker);
