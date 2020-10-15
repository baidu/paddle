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

#include <stdlib.h>
#include <unistd.h>
#include <chrono>  // NOLINT
#include <memory>
#include <string>
#include <thread>  // NOLINT
#include <unordered_map>

#include "gtest/gtest.h"
#include "paddle/fluid/framework/block_desc.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/operator.h"

#include "paddle/fluid/operators/distributed/distributed.h"
#include "paddle/fluid/operators/distributed/heart_beat_monitor.h"
#include "paddle/fluid/operators/distributed/large_scale_kv.h"
#include "paddle/fluid/operators/distributed/request_handler_impl.h"
#include "paddle/fluid/operators/distributed/rpc_client.h"
#include "paddle/fluid/operators/distributed/rpc_server.h"

namespace framework = paddle::framework;
namespace platform = paddle::platform;
namespace distributed = paddle::operators::distributed;

USE_NO_KERNEL_OP(lookup_sparse_table_read);
USE_NO_KERNEL_OP(checkpoint_notify);
USE_OP(scale);

std::unique_ptr<distributed::RPCServer> g_rpc_service;
std::unique_ptr<distributed::RequestHandler> g_req_handler;

framework::BlockDesc* AppendSendAndRecvBlock(framework::ProgramDesc* program) {
  auto root_block = program->MutableBlock(0);
  auto* block = program->AppendBlock(*root_block);

  framework::OpDesc* op = block->AppendOp();
  op->SetType("scale");
  op->SetInput("X", {"x"});
  op->SetOutput("Out", {"res"});
  op->SetAttr("scale", 0.5f);

  auto& out = *root_block->Var("res");
  out.SetType(framework::proto::VarType::LOD_TENSOR);
  out.SetShape({1, 10});

  return block;
}

void CreateVarsOnScope(framework::Scope* scope, platform::CPUPlace* place) {
  auto w_var = scope->Var("w");
  w_var->GetMutable<framework::SelectedRows>();

  auto out_var = scope->Var("out");
  out_var->GetMutable<framework::LoDTensor>();

  auto ids_var = scope->Var("ids");
  ids_var->GetMutable<framework::LoDTensor>();

  auto x_var = scope->Var("x");
  x_var->GetMutable<framework::LoDTensor>();

  auto res_var = scope->Var("res");
  res_var->GetMutable<framework::LoDTensor>();
}

void InitTensorsOnClient(framework::Scope* scope, platform::CPUPlace* place,
                         int64_t rows_numel) {
  CreateVarsOnScope(scope, place);
  auto ids_var = scope->Var("ids")->GetMutable<framework::LoDTensor>();
  int64_t* ids_ptr =
      ids_var->mutable_data<int64_t>(framework::DDim({rows_numel, 1}), *place);
  for (int64_t i = 0; i < rows_numel; ++i) ids_ptr[i] = i * 2;

  auto x_var = scope->Var("x")->GetMutable<framework::LoDTensor>();
  float* x_ptr =
      x_var->mutable_data<float>(framework::DDim({1, rows_numel}), *place);
  for (int64_t i = 0; i < rows_numel; ++i) x_ptr[i] = 1.0;
}

void InitTensorsOnServer(framework::Scope* scope, platform::CPUPlace* place,
                         int64_t rows_numel) {
  CreateVarsOnScope(scope, place);
  auto w = scope->Var("w")->GetMutable<framework::SelectedRows>();
  auto w_value = w->mutable_value();
  w_value->Resize({rows_numel, 10});
  for (int64_t i = 0; i < rows_numel; ++i) w->AutoGrownIndex(i, true);

  auto ptr = w_value->mutable_data<float>(*place);

  for (int64_t i = 0; i < w_value->numel(); ++i) {
    ptr[i] = static_cast<float>(i / 10);
  }
}

void StartServer(const std::string& rpc_name) {
  framework::ProgramDesc program;
  framework::Scope scope;
  platform::CPUPlace place;
  framework::Executor exe(place);
  platform::CPUDeviceContext ctx(place);

  std::unordered_map<std::string,
                     std::shared_ptr<framework::ExecutorPrepareContext>>
      prefetch_var_name_to_prepared;

  g_req_handler->SetProgram(&program);
  g_req_handler->SetPrefetchPreparedCtx(&prefetch_var_name_to_prepared);
  g_req_handler->SetDevCtx(&ctx);
  g_req_handler->SetScope(&scope);
  g_req_handler->SetExecutor(&exe);

  g_rpc_service->RegisterRPC(rpc_name, g_req_handler.get());

  //  distributed::HeartBeatMonitor::Init(1, true, "w@grad");

  g_req_handler->SetRPCServer(g_rpc_service.get());

  std::thread server_thread(
      std::bind(&distributed::RPCServer::StartServer, g_rpc_service.get()));

  server_thread.join();
}

void StartSendAndRecvServer(const std::string& rpc_name) {
  framework::ProgramDesc program;
  framework::Scope scope;
  platform::CPUPlace place;
  framework::Executor exe(place);
  platform::CPUDeviceContext ctx(place);
  auto block = AppendSendAndRecvBlock(&program);
  std::string in_var_name("x");
  std::vector<int> prefetch_block_ids{block->ID()};
  auto prepared = exe.Prepare(program, prefetch_block_ids);
  InitTensorsOnServer(&scope, &place, 10);

  std::unordered_map<std::string,
                     std::shared_ptr<framework::ExecutorPrepareContext>>
      grad_to_prepared_ctx;
  grad_to_prepared_ctx[in_var_name] = prepared[0];

  g_req_handler->SetProgram(&program);
  g_req_handler->SetGradToPreparedCtx(&grad_to_prepared_ctx);
  g_req_handler->SetDevCtx(&ctx);
  g_req_handler->SetScope(&scope);
  g_req_handler->SetExecutor(&exe);

  g_rpc_service->RegisterRPC(rpc_name, g_req_handler.get());
  g_req_handler->SetRPCServer(g_rpc_service.get());

  std::thread server_thread(
      std::bind(&distributed::RPCServer::StartServer, g_rpc_service.get()));

  server_thread.join();
}

TEST(COMPLETE, CPU) {
  setenv("http_proxy", "", 1);
  setenv("https_proxy", "", 1);
  g_req_handler.reset(
      new distributed::RequestSendHandler(distributed::DistributedMode::kSync));
  g_rpc_service.reset(new RPCSERVER_T("127.0.0.1:0", 2));
  distributed::RPCClient* client =
      distributed::RPCClient::GetInstance<RPCCLIENT_T>(0);
  PADDLE_ENFORCE_NE(client, nullptr,
                    platform::errors::InvalidArgument(
                        "Client Start Fail, Check Your Code & Env"));
  std::thread server_thread(StartServer, distributed::kRequestSend);
  g_rpc_service->WaitServerReady();
  int port = g_rpc_service->GetSelectedPort();
  std::string ep = paddle::string::Sprintf("127.0.0.1:%d", port);
  client->AsyncSendComplete(ep);
  client->Wait();

  EXPECT_EQ(g_rpc_service->GetClientNum(), 1);

  g_rpc_service->ShutDown();
  server_thread.join();
  g_rpc_service.reset(nullptr);
  g_req_handler.reset(nullptr);
}

TEST(SENDANDRECV, CPU) {
  setenv("http_proxy", "", 1);
  setenv("https_proxy", "", 1);
  g_req_handler.reset(new distributed::RequestSendAndRecvHandler(
      distributed::DistributedMode::kAsync));
  g_rpc_service.reset(new RPCSERVER_T("127.0.0.1:0", 1));
  distributed::RPCClient* client =
      distributed::RPCClient::GetInstance<RPCCLIENT_T>(0);
  PADDLE_ENFORCE_NE(client, nullptr,
                    platform::errors::InvalidArgument(
                        "Client Start Fail, Check Your Code & Env"));
  std::thread server_thread(StartSendAndRecvServer,
                            distributed::kRequestSendAndRecv);
  g_rpc_service->WaitServerReady();
  int port = g_rpc_service->GetSelectedPort();
  std::string ep = paddle::string::Sprintf("127.0.0.1:%d", port);

  framework::Scope scope;
  platform::CPUPlace place;
  platform::CPUDeviceContext ctx(place);

  // create var on local scope
  int64_t rows_numel = 10;
  InitTensorsOnClient(&scope, &place, rows_numel);
  std::string in_var_name("x");
  std::string out_var_name("res");

  client->AsyncSendAndRecv(ep, ctx, scope, in_var_name, out_var_name);
  client->Wait();
  auto var = scope.Var(out_var_name);
  auto value = var->GetMutable<framework::LoDTensor>();
  auto ptr = value->mutable_data<float>(place);

  for (int64_t i = 0; i < rows_numel; ++i) {
    EXPECT_EQ(ptr[i], 0.5);
  }
  g_rpc_service->ShutDown();
  server_thread.join();
  LOG(INFO) << "begin reset";
  g_rpc_service.reset(nullptr);
  g_req_handler.reset(nullptr);
}

void StartCheckpointServer(const std::string& rpc_name) {
  framework::ProgramDesc program;
  framework::Scope scope;
  platform::CPUPlace place;
  framework::Executor exe(place);
  platform::CPUDeviceContext ctx(place);

  std::vector<distributed::SparseMeta> metas;

  auto meta = distributed::SparseMeta();
  meta.name = "embedding.block0";
  meta.value_names = {"Param"};
  meta.value_dims = {64};
  meta.mode = distributed::Mode::training;
  meta.grad_name = "embedding@Grad";
  meta.cached_varnames = {"kSparseIds"};
  meta.initializer_attrs = {"fill_constant&1.0"};
  meta.entry = "none";

  metas.push_back(meta);
  distributed::LargeScaleKV::Init(metas);

  auto* ins = distributed::LargeScaleKV::GetInstance();
  ins->Get("embedding.block0")->Init({0, 1, 2, 3, 4, 5, 6, 7, 8, 9});

  std::unordered_map<std::string,
                     std::shared_ptr<framework::ExecutorPrepareContext>>
      prefetch_var_name_to_prepared;

  g_req_handler->SetProgram(&program);
  g_req_handler->SetPrefetchPreparedCtx(&prefetch_var_name_to_prepared);
  g_req_handler->SetDevCtx(&ctx);
  g_req_handler->SetScope(&scope);
  g_req_handler->SetExecutor(&exe);

  g_rpc_service->RegisterRPC(rpc_name, g_req_handler.get());

  g_req_handler->SetRPCServer(g_rpc_service.get());

  std::thread server_thread(
      std::bind(&distributed::RPCServer::StartServer, g_rpc_service.get()));

  server_thread.join();
}

TEST(LARGE_SCALE_CHECKPOINT, CPU) {
  setenv("http_proxy", "", 1);
  setenv("https_proxy", "", 1);

  paddle::framework::Scope scope;
  paddle::platform::CPUPlace place;

  g_req_handler.reset(new distributed::RequestCheckpointHandler(
      distributed::DistributedMode::kAsync));
  g_rpc_service.reset(new RPCSERVER_T("127.0.0.1:0", 1));

  distributed::RPCClient* client =
      distributed::RPCClient::GetInstance<RPCCLIENT_T>(0);

  PADDLE_ENFORCE_NE(client, nullptr,
                    platform::errors::InvalidArgument(
                        "Client Start Fail, Check Your Code & Env"));

  std::thread server_thread(StartCheckpointServer,
                            distributed::kRequestCheckpoint);
  g_rpc_service->WaitServerReady();

  int port = g_rpc_service->GetSelectedPort();
  std::string ep = paddle::string::Sprintf("127.0.0.1:%d", port);

  auto save_path =
      paddle::string::Sprintf("%s/%s/%s", "/tmp/large_scale_table/base",
                              "embedding", "embedding.block0");
  int mode = 0;
  client->AsyncCheckpointNotify(ep, save_path, "embedding.block0", mode);
  client->Wait();

  save_path =
      paddle::string::Sprintf("%s/%s/%s", "/tmp/large_scale_table/delta",
                              "embedding", "embedding.block0");
  mode = 1;
  client->AsyncCheckpointNotify(ep, save_path, "embedding.block0", mode);
  client->Wait();

  paddle::framework::AttributeMap attrs;

  std::vector<std::string> eps = {ep};
  attrs["endpoints"] = eps;
  attrs["dirname"] = std::string("/tmp/large_scale_table/delta1");
  attrs["varname"] = std::string("embedding");
  attrs["mode"] = 2;
  std::vector<std::string> slices = {"embedding.block0"};
  attrs["slice_varnames"] = slices;
  std::vector<std::string> remotes = {"embedding.block0"};
  attrs["remote_varnames"] = remotes;

  auto ops =
      framework::OpRegistry::CreateOp("checkpoint_notify", {}, {}, attrs, true);
  ops->Run(scope, place);

  g_rpc_service->ShutDown();
  server_thread.join();
  LOG(INFO) << "begin reset";
  g_rpc_service.reset(nullptr);
  g_req_handler.reset(nullptr);
}
