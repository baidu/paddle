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

#include <stdio.h>  // for removing the port file
#include <csignal>
#include <cstdlib>
#include <fstream>
#include <thread>  // NOLINT
#include <vector>

#include "gflags/gflags.h"

#include "paddle/fluid/operators/distributed/distributed.h"
#include "paddle/fluid/operators/math/math_function.h"

#include "paddle/fluid/operators/distributed/async_sparse_param_update_recorder.h"
#include "paddle/fluid/operators/distributed/request_handler_impl.h"
#include "paddle/fluid/operators/distributed_ops/fl_listen_and_serv_op.h"

#include "paddle/fluid/platform/profiler.h"

DEFINE_int32(flrpc_send_thread_num, 12, "number of threads for rpc send");
DEFINE_int32(flrpc_get_thread_num, 12, "number of threads for rpc get");

namespace paddle {
namespace operators {

void FlRunServer(std::shared_ptr<distributed::RPCServer> service) {
  service->StartServer();
}
static void flsplit(const std::string &str, char sep,
                    std::vector<std::string> *pieces) {
  pieces->clear();
  if (str.empty()) {
    return;
  }
  size_t pos = 0;
  size_t next = str.find(sep, pos);
  while (next != std::string::npos) {
    pieces->push_back(str.substr(pos, next - pos));
    pos = next + 1;
    next = str.find(sep, pos);
  }
  if (!str.substr(pos).empty()) {
    pieces->push_back(str.substr(pos));
  }
}

static void FlParallelExecuteBlocks(
    const std::vector<size_t> &parallel_blkids, framework::Executor *executor,
    const std::vector<std::shared_ptr<framework::ExecutorPrepareContext>>
        &prepared,
    framework::ProgramDesc *program, framework::Scope *scope) {
  std::vector<std::future<void>> fs;
  for (size_t idx : parallel_blkids) {
    fs.push_back(framework::Async([&executor, &prepared, &scope, idx]() {
      int run_block = idx;  // thread local
      try {
        VLOG(3) << "running server block: " << run_block
                << "pointer: " << prepared[run_block].get();
        executor->RunPreparedContext(prepared[run_block].get(), scope);
      } catch (const std::exception &e) {
        PADDLE_THROW(platform::errors::Fatal(
            "Run %d-th sub program failed. The exception is:\n%s.", idx,
            e.what()));
      }
    }));
  }
  for (size_t i = 0; i < fs.size(); ++i) fs[i].wait();
}

FlListenAndServOp::FlListenAndServOp(const std::string &type,
                                     const framework::VariableNameMap &inputs,
                                     const framework::VariableNameMap &outputs,
                                     const framework::AttributeMap &attrs)
    : OperatorBase(type, inputs, outputs, attrs) {}

FlListenAndServOp::~FlListenAndServOp() {}

void FlListenAndServOp::SavePort() const {
  // NOTE: default write file to /tmp/paddle.selected_port
  rpc_service_->SavePort();
}

static int64_t GetTimestamp() {
  struct timeval tp;
  gettimeofday(&tp, NULL);
  return tp.tv_sec * 1000 + tp.tv_usec / 1000;
}

void FlListenAndServOp::RunSyncLoop(framework::Executor *executor,
                                    framework::ProgramDesc *program,
                                    framework::Scope *recv_scope,
                                    platform::DeviceContext *dev_ctx) const {
  VLOG(2) << "RunSyncLoop";
  size_t num_blocks = program->Size();
  auto optimize_blocks =
      Attr<std::vector<framework::BlockDesc *>>(kOptimizeBlocks);
  PADDLE_ENFORCE_GE(num_blocks, 2,
                    platform::errors::InvalidArgument(
                        "server program should have at least 2 blocks"));

  // Prepare all the server block
  std::vector<int> optimize_blocks_list;
  for (size_t i = 1; i < program->Size(); ++i) {
    optimize_blocks_list.push_back(i);
  }
  auto optimize_prepared = executor->Prepare(*program, optimize_blocks_list);
  // Insert placeholder for block0 which holds current op itself,
  // NOTE the first block in `optimize_prepared` should never be ran.
  optimize_prepared.insert(
      optimize_prepared.begin(),
      std::shared_ptr<framework::ExecutorPrepareContext>(nullptr));

  while (true) {
    // Get from multiple trainers, we don't care about the order in which
    // the gradients arrives, just add suffix 0~n and merge the gradient.
    VLOG(3) << "wait all clients to get pserver parameters back";
    rpc_service_->SetCond(distributed::kRequestGet);
    VLOG(3) << "wait all clients to send fetch_barrier";
    rpc_service_->WaitBarrier(distributed::kRequestGet);

    if (rpc_service_->IsExit()) {
      rpc_service_->SetCond(distributed::kRequestGet);
      break;
    }

    VLOG(3) << "wait all clients to send after_optimizer parameters";
    rpc_service_->SetCond(distributed::kRequestSend);
    VLOG(3) << "wait all clients to send send_barrier";
    rpc_service_->WaitBarrier(distributed::kRequestSend);
    VLOG(3) << "ResetBarrierCounter";
    rpc_service_->ResetBarrierCounter();
    // NOTE: if is_gpu_place, CUDA kernels are launched by multiple threads
    // and this will still work.
    // The optimize blocks which have the same parent ID would run parallel
    // TODO(Yancey1989): need to use ParallelExecutor for future
    int32_t last_parent_blkid = optimize_blocks[0]->Parent();
    std::vector<size_t> parallel_blkids;
    parallel_blkids.push_back(optimize_blocks[0]->ID());
    double ts = GetTimestamp();
    for (size_t i = 1; i < optimize_blocks.size(); ++i) {
      // skip the first optimize block because it is already in the
      // parallel_blkids.
      int blkid = optimize_blocks[i]->ID();
      if (program->Block(blkid).Parent() != last_parent_blkid) {
        FlParallelExecuteBlocks(parallel_blkids, executor, optimize_prepared,
                                program, recv_scope);
        parallel_blkids.clear();
        last_parent_blkid = program->Block(blkid).Parent();
      }
      parallel_blkids.push_back(blkid);
    }
    FlParallelExecuteBlocks(parallel_blkids, executor, optimize_prepared,
                            program, recv_scope);
    VLOG(3) << "run all blocks spent " << GetTimestamp() - ts << "(ms)";
  }  // while(true)
}

static void FillRequestCtx(distributed::RequestHandler *h,
                           framework::Scope *scope,
                           platform::DeviceContext *dev_ctx,
                           framework::Executor *executor,
                           framework::ProgramDesc *program,
                           distributed::RPCServer *rpc_server) {
  h->SetScope(scope);
  h->SetDevCtx(dev_ctx);
  h->SetExecutor(executor);
  h->SetProgram(program);
  h->SetRPCServer(rpc_server);
}

void FlListenAndServOp::RunImpl(const framework::Scope &scope,
                                const platform::Place &dev_place) const {
  // Mark this as PS that it should decide profiling by listening from trainer.
  platform::SetProfileListener();
  platform::DeviceContextPool &pool = platform::DeviceContextPool::Instance();
  auto &dev_ctx = *pool.Get(dev_place);
  framework::Scope &recv_scope = scope.NewScope();

  bool sync_mode = Attr<bool>("sync_mode");
  auto fan_in = Attr<int>("Fanin");
  auto inputs = Inputs("X");

  PADDLE_ENFORCE_EQ(!rpc_service_, true, platform::errors::InvalidArgument(
                                             "rpc_service_ must null"));
  std::string endpoint = Attr<std::string>("endpoint");

  VLOG(4) << "sync_mode:" << sync_mode << ", fan_in:" << fan_in
          << ", end_point:" << endpoint;

  rpc_service_.reset(new RPCSERVER_T(endpoint, fan_in));

  request_send_handler_.reset(
      new distributed::RequestSendHandler(!sync_mode, false));
  request_get_handler_.reset(
      new distributed::RequestGetHandler(!sync_mode, false));

  rpc_service_->RegisterRPC(distributed::kRequestSend,
                            request_send_handler_.get(),
                            FLAGS_flrpc_send_thread_num);
  rpc_service_->RegisterRPC(distributed::kRequestGet,
                            request_get_handler_.get(),
                            FLAGS_flrpc_get_thread_num);
  auto optimize_blocks =
      Attr<std::vector<framework::BlockDesc *>>(kOptimizeBlocks);
  PADDLE_ENFORCE_GE(
      optimize_blocks.size(), 1,
      platform::errors::InvalidArgument(
          "optimize blocks should be 1 at least on the pserver side."));
  auto *program = optimize_blocks[0]->Program();
  framework::Executor executor(dev_place);

  auto f = std::bind(FillRequestCtx, std::placeholders::_1, &recv_scope,
                     &dev_ctx, &executor, program, rpc_service_.get());

  f(request_send_handler_.get());
  f(request_get_handler_.get());

  // start the server listening after all member initialized.
  server_thread_.reset(new std::thread(FlRunServer, rpc_service_));
  VLOG(3) << "wait server thread to become ready...";
  rpc_service_->WaitServerReady();

  // register SIGINT(from ctrl+C) and SIGTERM(from kill) signal handlers
  signal(SIGINT, FlSignalHandler::StopAndExit);
  signal(SIGTERM, FlSignalHandler::StopAndExit);

  // Cache the type of the received vars as `sparse_vars_` and `dense_vars_`
  // so that we can reset them at the end of each iteration.
  // NOTE: only used in sync update

  // Write to a file of server selected port for python use.
  SavePort();
  RunSyncLoop(&executor, program, &recv_scope, &dev_ctx);
}

class FlListenAndServOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() {
    AddInput("X", "(Tensor) Variables that server recv.").AsDuplicable();
    AddComment(R"DOC(" + "ListenAndServ operator" + "\n" + "This operator" +
" will start a RPC server which can receive variables from send_op and send" +
"back variables to recv_op.)DOC");
    AddAttr<std::string>("endpoint",
                         "(string, default 127.0.0.1:6164)"
                         "IP address to listen on.")
        .SetDefault("127.0.0.1:6164")
        .AddCustomChecker([](const std::string &ip) { return !ip.empty(); });
    AddAttr<bool>("sync_mode", "if works at sync_mode or not").SetDefault(true);
    AddAttr<int>("Fanin", "How many clients send to this server.")
        .SetDefault(1);
    AddAttr<std::vector<framework::BlockDesc *>>(
        kOptimizeBlocks, "Optimize blocks to run on server side.")
        .SetDefault({});
  }
};

void FlSignalHandler::StopAndExit(int signal_num) {
  // Do not use VLOG here for the device for printing maybe already released.
  // exit will release interal allocated resoureces.
  auto file_path = string::Sprintf("/tmp/paddle.%d.port", ::getpid());
  remove(file_path.c_str());
  exit(0);
}

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OP_WITHOUT_GRADIENT(fl_listen_and_serv, ops::FlListenAndServOp,
                             ops::FlListenAndServOpMaker);
