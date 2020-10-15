// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "paddle/fluid/operators/distributed/brpc/brpc_server.h"
#include <memory>
#include <unordered_map>
#include "paddle/fluid/framework/threadpool.h"
#include "paddle/fluid/operators/distributed/brpc/brpc_sendrecvop_utils.h"
#include "paddle/fluid/operators/distributed/brpc/brpc_variable_response.h"
#include "paddle/fluid/operators/distributed/request_handler.h"

namespace sendrecv {

namespace distributed = paddle::operators::distributed;

typedef std::unordered_map<std::string, distributed::RequestHandler*>
    HandlerMap;

class BRPCServiceImpl : public SendRecvService {
 public:
  explicit BRPCServiceImpl(const HandlerMap& rpc_call_map,
                           distributed::RPCServer* rpc_server)
      : rpc_server_(rpc_server) {
    VLOG(3) << "BRPCServiceImpl size: " << rpc_call_map.size();
    auto it = rpc_call_map.find(distributed::kRequestSend);
    if (it != rpc_call_map.end()) {
      request_send_h_ = it->second;
      send_threads_.reset(new paddle::framework::ThreadPool(
          rpc_server_->GetThreadNum(distributed::kRequestSend)));
    }

    it = rpc_call_map.find(distributed::kRequestGet);
    if (it != rpc_call_map.end()) {
      request_get_h_ = it->second;
      get_threads_.reset(new paddle::framework::ThreadPool(
          rpc_server_->GetThreadNum(distributed::kRequestGet)));
    }

    it = rpc_call_map.find(distributed::kRequestGetNoBarrier);
    if (it != rpc_call_map.end()) {
      request_getnobarrier_h_ = it->second;
      getnobarrier_threads_.reset(new paddle::framework::ThreadPool(
          rpc_server_->GetThreadNum(distributed::kRequestGetNoBarrier)));
    }

    it = rpc_call_map.find(distributed::kRequestPrefetch);
    if (it != rpc_call_map.end()) {
      request_prefetch_h_ = it->second;
      prefetch_threads_.reset(new paddle::framework::ThreadPool(
          rpc_server_->GetThreadNum(distributed::kRequestPrefetch)));
    }

    it = rpc_call_map.find(distributed::kRequestCheckpoint);
    if (it != rpc_call_map.end()) {
      request_checkpoint_h_ = it->second;
      checkpoint_notify_threads_.reset(new paddle::framework::ThreadPool(
          rpc_server_->GetThreadNum(distributed::kRequestPrefetch)));
    }

    it = rpc_call_map.find(distributed::kRequestGetMonomerVariable);
    if (it != rpc_call_map.end()) {
      request_get_monomer_handler_h_ = it->second;
    }

    it = rpc_call_map.find(distributed::kRequestGetMonomerBarrier);
    if (it != rpc_call_map.end()) {
      request_get_monomer_barrier_handler_h_ = it->second;
    }
  }

  virtual ~BRPCServiceImpl() {}
  void SendVariable(google::protobuf::RpcController* cntl_butil,
                    const VariableMessage* request, VoidMessage* response,
                    google::protobuf::Closure* done) override {
    send_threads_->Run(
        [=] { _SendVariable(cntl_butil, request, response, done); });
  }

  void _SendVariable(google::protobuf::RpcController* cntl_butil,
                     const VariableMessage* request, VoidMessage* response,
                     google::protobuf::Closure* done) {
    PADDLE_ENFORCE_NOT_NULL(
        request_send_h_, platform::errors::PreconditionNotMet(
                             "RequestSend handler should be registed first!"));
    brpc::ClosureGuard done_guard(done);
    brpc::Controller* cntl = static_cast<brpc::Controller*>(cntl_butil);

    std::string varname = request->varname();
    VLOG(3) << "RequestSend var_name:" << varname
            << ", trainer_id:" << request->trainer_id()
            << ", from:" << cntl->remote_side();

    distributed::BRPCVariableResponse resp(request_send_h_->scope(),
                                           request_send_h_->dev_ctx(),
                                           request_send_h_->distributed_mode());
    PADDLE_ENFORCE_EQ(
        resp.Parse(cntl->request_attachment(), *request), 0,
        platform::errors::InvalidArgument("parse iobuf to tensor error!"));

    auto scope = resp.GetMutableLocalScope();
    auto invar = resp.GetVar();
    int trainer_id = request->trainer_id();
    paddle::framework::Variable* outvar = nullptr;

    request_send_h_->Handle(varname, scope, invar, &outvar, trainer_id);
  }

  void GetVariable(google::protobuf::RpcController* cntl_butil,
                   const VariableMessage* request, VariableMessage* response,
                   google::protobuf::Closure* done) override {
    get_threads_->Run(
        [=] { _GetVariable(cntl_butil, request, response, done); });
  }

  void GetVariableNoBarrier(google::protobuf::RpcController* cntl_butil,
                            const VariableMessage* request,
                            VariableMessage* response,
                            google::protobuf::Closure* done) override {
    getnobarrier_threads_->Run(
        [=] { _GetVariableNoBarrier(cntl_butil, request, response, done); });
  }

  void _GetVariable(google::protobuf::RpcController* cntl_butil,
                    const VariableMessage* request, VariableMessage* response,
                    google::protobuf::Closure* done) {
    PADDLE_ENFORCE_NOT_NULL(
        request_get_h_, platform::errors::PreconditionNotMet(
                            "RequestGet handler should be registed first!"));

    brpc::ClosureGuard done_guard(done);
    brpc::Controller* cntl = static_cast<brpc::Controller*>(cntl_butil);

    std::string varname = request->varname();
    std::string out_varname = request->out_varname();
    VLOG(3) << "RequestGet varname:" << varname
            << ", out_varname:" << out_varname
            << ", trainer_id:" << request->trainer_id()
            << ", from:" << cntl->remote_side();

    auto scope = request_get_h_->scope();
    paddle::framework::Variable* invar = nullptr;
    int trainer_id = request->trainer_id();
    paddle::framework::Variable* outvar = nullptr;

    request_get_h_->Handle(varname, scope, invar, &outvar, trainer_id,
                           out_varname);

    if (outvar) {
      distributed::SerializeToIOBuf(out_varname, outvar,
                                    *request_get_h_->dev_ctx(), response,
                                    &cntl->response_attachment(), "", false);
    }
  }

  void _GetVariableNoBarrier(google::protobuf::RpcController* cntl_butil,
                             const VariableMessage* request,
                             VariableMessage* response,
                             google::protobuf::Closure* done) {
    PADDLE_ENFORCE_NOT_NULL(
        request_getnobarrier_h_,
        platform::errors::PreconditionNotMet(
            "RequestGetNoBarrier handler should be registed first!"));

    brpc::ClosureGuard done_guard(done);
    brpc::Controller* cntl = static_cast<brpc::Controller*>(cntl_butil);

    std::string varname = request->varname();
    std::string out_varname = request->out_varname();
    int trainer_id = request->trainer_id();

    VLOG(3) << "RequestGetNoBarrier varname:" << varname
            << ", out_varname:" << out_varname << ", trainer_id:" << trainer_id
            << ", from:" << cntl->remote_side();

    auto scope = request_getnobarrier_h_->scope();
    paddle::framework::Variable* invar = nullptr;
    paddle::framework::Variable* outvar = nullptr;

    request_getnobarrier_h_->Handle(varname, scope, invar, &outvar, trainer_id,
                                    out_varname);

    if (outvar) {
      distributed::SerializeToIOBuf(
          out_varname, outvar, *request_getnobarrier_h_->dev_ctx(), response,
          &cntl->response_attachment(), "", false);
    }
  }

  void PrefetchVariable(google::protobuf::RpcController* cntl_butil,
                        const VariableMessage* request,
                        VariableMessage* response,
                        google::protobuf::Closure* done) override {
    prefetch_threads_->Run(
        [=] { _PrefetchVariable(cntl_butil, request, response, done); });
  }

  void _PrefetchVariable(google::protobuf::RpcController* cntl_butil,
                         const VariableMessage* request,
                         VariableMessage* response,
                         google::protobuf::Closure* done) {
    PADDLE_ENFORCE_NOT_NULL(request_prefetch_h_,
                   platform::errors::PreconditionNotMet(
                       "kRequestPrefetch handler should be registed first!");

    brpc::ClosureGuard done_guard(done);
    brpc::Controller* cntl = static_cast<brpc::Controller*>(cntl_butil);

    // prefetch process...
    std::string in_var_name = request->varname();
    std::string out_var_name = request->out_varname();
    VLOG(3) << "RequestPrefetch, in_var_name: " << in_var_name
            << ", out_var_name: " << out_var_name
            << ", trainer_id:" << request->trainer_id()
            << ", from:" << cntl->remote_side();

    distributed::BRPCVariableResponse resp(
        request_prefetch_h_->scope(), request_prefetch_h_->dev_ctx(), true);

    PADDLE_ENFORCE_EQ(resp.Parse(cntl->request_attachment(), *request), 0,
                   platform::errors::InvalidArgument(
                       "parse iobuf to tensor error!"));

    auto scope = resp.GetMutableLocalScope();
    auto invar = scope->FindVar(in_var_name);
    std::string table_name = request->table_name();
    int trainer_id = request->trainer_id();
    paddle::framework::Variable* outvar = scope->Var(out_var_name);

    request_prefetch_h_->Handle(in_var_name, scope, invar, &outvar, trainer_id,
                                out_var_name, table_name);

    distributed::SerializeToIOBuf(out_var_name, outvar,
                                  *request_prefetch_h_->dev_ctx(), response,
                                  &cntl->response_attachment(), "", true);
  }

  void CheckpointNotify(google::protobuf::RpcController* cntl_butil,
                        const VariableMessage* request, VoidMessage* response,
                        google::protobuf::Closure* done) override {
    checkpoint_notify_threads_->Run(
        [=] { _CheckpointNotify(cntl_butil, request, response, done); });
  }

  void _CheckpointNotify(google::protobuf::RpcController* cntl_butil,
                         const VariableMessage* request, VoidMessage* response,
                         google::protobuf::Closure* done) {
    PADDLE_ENFORCE_NOT_NULL(
        request_checkpoint_h_,
        platform::errors::PreconditionNotMet(
            "kRequestCheckpointNotify handler should be registed first!"));

    brpc::ClosureGuard done_guard(done);
    brpc::Controller* cntl = static_cast<brpc::Controller*>(cntl_butil);

    distributed::BRPCVariableResponse resp(request_checkpoint_h_->scope(),
                                           request_checkpoint_h_->dev_ctx());

    auto scope = resp.GetMutableLocalScope();

    std::string checkpoint_notify = request->varname();
    std::string checkpoint_dir = request->out_varname();
    int trainer_id = request->trainer_id();

    VLOG(4) << "RequestCheckpointNotify notify: " << checkpoint_notify
            << ", dir: " << checkpoint_dir
            << ", trainer_id:" << request->trainer_id()
            << ", from:" << cntl->remote_side();

    request_checkpoint_h_->Handle(checkpoint_notify, scope, nullptr, nullptr,
                                  trainer_id, checkpoint_dir);
  }

  void GetMonomerVariable(google::protobuf::RpcController* cntl_butil,
                          const VariableMessage* request,
                          VariableMessage* response,
                          google::protobuf::Closure* done) override {
    PADDLE_ENFORCE_NOT_NULL(
        request_get_monomer_handler_h_,
        platform::errors::PreconditionNotMet(
            "kRequestGetMonomerVariable handler should be registed first!"));

    brpc::ClosureGuard done_guard(done);
    brpc::Controller* cntl = static_cast<brpc::Controller*>(cntl_butil);

    // proc request.
    std::string varname = request->varname();
    VLOG(3) << "GetMonomerVariable " << varname
            << ", trainer_id:" << request->trainer_id()
            << ", from:" << cntl->remote_side();

    rpc_server_->WaitVarCond(varname);
    distributed::MonomerHandle h = rpc_server_->GetMonomer(varname);

    auto scope = h.scope_;
    auto invar = scope->FindVar(varname);
    paddle::framework::Variable* outvar = nullptr;

    request_get_monomer_handler_h_->Handle(varname, scope, invar, &outvar,
                                           request->trainer_id());

    if (outvar) {
      distributed::SerializeToIOBuf(varname, outvar, *h.dev_ctx_, response,
                                    &cntl->response_attachment(), "", false);
    }
  }

  void GetMonomerBarrier(google::protobuf::RpcController* cntl_butil,
                         const VariableMessage* request, VoidMessage* response,
                         google::protobuf::Closure* done) override {
    PADDLE_ENFORCE_NOT_NULL(
        request_get_monomer_barrier_handler_h_,
        platform::errors::PreconditionNotMet(
            "RequestGetMonomerBarrier handler should be registed first!"));

    brpc::ClosureGuard done_guard(done);
    brpc::Controller* cntl = static_cast<brpc::Controller*>(cntl_butil);

    std::string varname = request->varname();
    VLOG(3) << "RequestGetMonomerBarrier var_name:" << varname
            << ", trainer_id:" << request->trainer_id()
            << ", from:" << cntl->remote_side();

    rpc_server_->WaitVarCond(varname);
    distributed::MonomerHandle h = rpc_server_->GetMonomer(varname);

    paddle::framework::Scope* scope = nullptr;
    paddle::framework::Variable* invar = nullptr;
    paddle::framework::Variable* outvar = nullptr;

    request_get_monomer_barrier_handler_h_->Handle(
        varname, scope, invar, &outvar, request->trainer_id());
  }

 private:
  distributed::RequestHandler* request_send_h_{nullptr};
  distributed::RequestHandler* request_get_h_{nullptr};
  distributed::RequestHandler* request_getnobarrier_h_{nullptr};
  distributed::RequestHandler* request_prefetch_h_{nullptr};
  distributed::RequestHandler* request_checkpoint_h_{nullptr};
  distributed::RequestHandler* request_get_monomer_handler_h_{nullptr};
  distributed::RequestHandler* request_get_monomer_barrier_handler_h_{nullptr};

  distributed::RPCServer* rpc_server_{nullptr};

  // FIXME(gongwb): brpc should support process one rpc use one threadpool.
  std::unique_ptr<paddle::framework::ThreadPool> send_threads_;
  std::unique_ptr<paddle::framework::ThreadPool> get_threads_;
  std::unique_ptr<paddle::framework::ThreadPool> getnobarrier_threads_;
  std::unique_ptr<paddle::framework::ThreadPool> prefetch_threads_;
  std::unique_ptr<paddle::framework::ThreadPool> checkpoint_notify_threads_;
};
}  // namespace sendrecv

namespace paddle {
namespace operators {
namespace distributed {

void AsyncBRPCServer::StartServer() {
  // Instance of your service.
  sendrecv::BRPCServiceImpl service_impl(rpc_call_map_, this);

  // Add the service into server. Notice the second parameter, because the
  // service is put on stack, we don't want server to delete it, otherwise
  // use brpc::SERVER_OWNS_SERVICE.
  if (server_.AddService(&service_impl, brpc::SERVER_DOESNT_OWN_SERVICE) != 0) {
    PADDDLE_THROW(platform::errors::Unavailable(
        "Failed to add service into BRPC server."));
    return;
  }

  brpc::ServerOptions options;
#ifdef PADDLE_WITH_BRPC_RDMA
  options.use_rdma = true;
#endif
  options.idle_timeout_sec = idle_timeout_s_;
  options.max_concurrency = max_concurrency_;
  if (server_.Start(bind_address_.c_str(), &options) != 0) {
    PADDDLE_THROW(platform::errors::Unavailable(
        "Failed to start EchoServer %s.", bind_address_));
    return;
  }

  butil::EndPoint ep = server_.listen_address();
  selected_port_ = ep.port;

  {
    std::lock_guard<std::mutex> lock(this->mutex_ready_);
    ready_ = 1;
  }
  condition_ready_.notify_all();

  server_.Join();
}

void AsyncBRPCServer::ShutDownImpl() { server_.Stop(1000); }

void AsyncBRPCServer::WaitServerReady() {
  VLOG(3) << "AsyncGRPCServer is wait server ready";
  std::unique_lock<std::mutex> lock(this->mutex_ready_);
  condition_ready_.wait(lock, [=] { return this->ready_ == 1; });
  VLOG(3) << "AsyncGRPCServer WaitSeverReady";
}

};  // namespace distributed
};  // namespace operators
};  // namespace paddle
