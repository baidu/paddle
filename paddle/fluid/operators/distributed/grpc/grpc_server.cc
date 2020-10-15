/*Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <unistd.h>
#include <limits>
#include <memory>
#include <string>

#include "paddle/fluid/operators/distributed/grpc/grpc_serde.h"
#include "paddle/fluid/operators/distributed/grpc/grpc_server.h"

namespace grpc {
class ChannelArguments;
}  // namespace grpc
namespace paddle {
namespace framework {
class Variable;
}  // namespace framework
namespace operators {
namespace distributed {
class GRPCVariableResponse;
}  // namespace distributed
}  // namespace operators
}  // namespace paddle

using ::grpc::ServerAsyncResponseWriter;

DECLARE_bool(rpc_disable_reuse_port);
DECLARE_int32(rpc_retry_bind_port);

namespace paddle {
namespace operators {
namespace distributed {

enum CallStatus { PROCESS = 0, FINISH };

// reference:
// https://stackoverflow.com/questions/41732884/grpc-multiple-services-in-cpp-async-server
class RequestBase {
 public:
  explicit RequestBase(GrpcService::AsyncService* service,
                       ::grpc::ServerCompletionQueue* cq,
                       RequestHandler* request_handler, int req_id)
      : service_(service),
        cq_(cq),
        status_(PROCESS),
        request_handler_(request_handler),
        req_id_(req_id) {
    PADDLE_ENFORCE_NOT_NULL(cq_, platform::errors::InvalidArgument(
                                     "ServerCompletionQueue cq are empty"));
  }
  virtual ~RequestBase() {}
  virtual void Process() = 0;

  std::string Status2String(const std::string& method) {
    std::string status = "Process";
    if (status_ == FINISH) {
      status = "Finish";
    }

    std::ostringstream s;
    s << method << " name:[" << GetReqName() << "]"
      << ", ep:[" << ctx_.peer() << "]"
      << " " << status << " using req_id:" << req_id_;
    return s.str();
  }

  CallStatus Status() const {
    std::lock_guard<std::mutex> l(status_mu_);
    return status_;
  }

  template <typename T>
  void Finish(const T& reply, ServerAsyncResponseWriter<T>* responder) {
    std::lock_guard<std::mutex> l(status_mu_);
    status_ = FINISH;
    responder->Finish(reply, ::grpc::Status::OK,
                      reinterpret_cast<void*>(static_cast<intptr_t>(req_id_)));
  }
  virtual std::string GetReqName() = 0;

 protected:
  mutable std::mutex status_mu_;
  ::grpc::ServerContext ctx_;
  GrpcService::AsyncService* service_;
  ::grpc::ServerCompletionQueue* cq_;
  CallStatus status_;
  RequestHandler* request_handler_;
  int req_id_;
};

class RequestSend final : public RequestBase {
 public:
  explicit RequestSend(GrpcService::AsyncService* service,
                       ::grpc::ServerCompletionQueue* cq,
                       RequestHandler* request_handler, int req_id)
      : RequestBase(service, cq, request_handler, req_id), responder_(&ctx_) {
    request_.reset(new GRPCVariableResponse(request_handler->scope(),
                                            request_handler->dev_ctx(), true));
    int method_id = static_cast<int>(distributed::GrpcMethod::kSendVariable);
    service_->RequestAsyncUnary(
        method_id, &ctx_, request_.get(), &responder_, cq_, cq_,
        reinterpret_cast<void*>(static_cast<intptr_t>(req_id)));
  }
  virtual ~RequestSend() {}
  std::string GetReqName() override { return request_->Varname(); }

  void Process() override {
    std::string varname = GetReqName();

    auto scope = request_->GetMutableLocalScope();
    auto invar = request_->GetVar();
    int trainer_id = request_->GetTrainerId();

    VLOG(4) << "RequestSend var_name:" << varname << " trainer: " << trainer_id;

    framework::Variable* outvar = nullptr;
    request_handler_->Handle(varname, scope, invar, &outvar, trainer_id);
    Finish(reply_, &responder_);
  }

 protected:
  sendrecv::VoidMessage reply_;
  std::shared_ptr<GRPCVariableResponse> request_;
  ServerAsyncResponseWriter<sendrecv::VoidMessage> responder_;
};

class RequestGet final : public RequestBase {
 public:
  explicit RequestGet(GrpcService::AsyncService* service,
                      ::grpc::ServerCompletionQueue* cq,
                      RequestHandler* request_handler, int req_id)
      : RequestBase(service, cq, request_handler, req_id), responder_(&ctx_) {
    auto method_id = static_cast<int>(distributed::GrpcMethod::kGetVariable);
    service_->RequestAsyncUnary(
        method_id, &ctx_, &request_, &responder_, cq_, cq_,
        reinterpret_cast<void*>(static_cast<intptr_t>(req_id)));
  }

  virtual ~RequestGet() {}

  std::string GetReqName() override { return request_.varname(); }

  void Process() override {
    // proc request.
    std::string varname = request_.varname();
    std::string out_varname = request_.out_varname();
    std::string table_name = request_.table_name();
    int trainer_id = request_.trainer_id();

    VLOG(4) << "RequestGet " << out_varname << " from " << varname;

    auto scope = request_handler_->scope();
    framework::Variable* invar = nullptr;
    framework::Variable* outvar = nullptr;

    tmp_scope_ = std::move(scope->NewTmpScope());
    request_handler_->Handle(varname, tmp_scope_.get(), invar, &outvar,
                             trainer_id, out_varname, table_name);

    VLOG(1) << "before SerializeToByteBuffer";
    if (outvar) {
      SerializeToByteBuffer(out_varname, outvar, *request_handler_->dev_ctx(),
                            &reply_);
    }
    VLOG(1) << "after SerializeToByteBuffer";
    Finish(reply_, &responder_);
  }

 protected:
  sendrecv::VariableMessage request_;
  ::grpc::ByteBuffer reply_;
  std::unique_ptr<framework::Scope> tmp_scope_;
  ServerAsyncResponseWriter<::grpc::ByteBuffer> responder_;
};

class RequestGetNoBarrier final : public RequestBase {
 public:
  explicit RequestGetNoBarrier(GrpcService::AsyncService* service,
                               ::grpc::ServerCompletionQueue* cq,
                               RequestHandler* request_handler, int req_id)
      : RequestBase(service, cq, request_handler, req_id), responder_(&ctx_) {
    auto method_id =
        static_cast<int>(distributed::GrpcMethod::kGetVariableNoBarrier);
    service_->RequestAsyncUnary(
        method_id, &ctx_, &request_, &responder_, cq_, cq_,
        reinterpret_cast<void*>(static_cast<intptr_t>(req_id)));
  }

  virtual ~RequestGetNoBarrier() {}

  std::string GetReqName() override { return request_.varname(); }

  void Process() override {
    // proc request.
    std::string varname = request_.varname();
    std::string out_varname = request_.out_varname();
    int trainer_id = request_.trainer_id();

    VLOG(4) << "RequestGetNoBarrier " << out_varname << " from " << varname;

    auto scope = request_handler_->scope();
    framework::Variable* invar = nullptr;
    framework::Variable* outvar = nullptr;

    request_handler_->Handle(varname, scope, invar, &outvar, trainer_id,
                             out_varname);

    if (outvar) {
      SerializeToByteBuffer(out_varname, outvar, *request_handler_->dev_ctx(),
                            &reply_);
    }
    Finish(reply_, &responder_);
  }

 protected:
  sendrecv::VariableMessage request_;
  ::grpc::ByteBuffer reply_;
  ServerAsyncResponseWriter<::grpc::ByteBuffer> responder_;
};

class RequestGetMonomerVariable final : public RequestBase {
 public:
  explicit RequestGetMonomerVariable(GrpcService::AsyncService* service,
                                     ::grpc::ServerCompletionQueue* cq,
                                     RequestHandler* request_handler,
                                     int req_id, RPCServer* rpc_server)
      : RequestBase(service, cq, request_handler, req_id),
        responder_(&ctx_),
        rpc_server_(rpc_server) {
    auto method_id =
        static_cast<int>(distributed::GrpcMethod::kGetMonomerVariable);
    service_->RequestAsyncUnary(
        method_id, &ctx_, &request_, &responder_, cq_, cq_,
        reinterpret_cast<void*>(static_cast<intptr_t>(req_id)));
  }

  virtual ~RequestGetMonomerVariable() {}

  std::string GetReqName() override { return request_.varname(); }

  void Process() override {
    // proc request.
    std::string varname = request_.varname();

    rpc_server_->WaitVarCond(varname);
    MonomerHandle h = rpc_server_->GetMonomer(varname);

    auto scope = h.scope_;
    auto invar = scope->FindVar(varname);
    framework::Variable* outvar = nullptr;

    request_handler_->Handle(varname, scope, invar, &outvar,
                             request_.trainer_id());

    if (outvar) {
      SerializeToByteBuffer(varname, outvar, *h.dev_ctx_, &reply_);
    }
    Finish(reply_, &responder_);
  }

 protected:
  sendrecv::VariableMessage request_;
  ::grpc::ByteBuffer reply_;
  ServerAsyncResponseWriter<::grpc::ByteBuffer> responder_;
  RPCServer* rpc_server_{nullptr};
};

class RequestGetMonomerBarrier final : public RequestBase {
 public:
  explicit RequestGetMonomerBarrier(GrpcService::AsyncService* service,
                                    ::grpc::ServerCompletionQueue* cq,
                                    RequestHandler* request_handler, int req_id,
                                    RPCServer* rpc_server)
      : RequestBase(service, cq, request_handler, req_id),
        responder_(&ctx_),
        rpc_server_(rpc_server) {
    auto method_id =
        static_cast<int>(distributed::GrpcMethod::kGetMonomerBarrier);
    service_->RequestAsyncUnary(
        method_id, &ctx_, &request_, &responder_, cq_, cq_,
        reinterpret_cast<void*>(static_cast<intptr_t>(req_id)));
  }

  virtual ~RequestGetMonomerBarrier() {}

  std::string GetReqName() override { return request_.varname(); }

  void Process() override {
    // proc request.
    std::string varname = request_.varname();
    VLOG(4) << "RequestGetMonomerBarrier " << varname;

    rpc_server_->WaitVarCond(varname);
    MonomerHandle h = rpc_server_->GetMonomer(varname);

    framework::Scope* scope = nullptr;
    framework::Variable* invar = nullptr;
    framework::Variable* outvar = nullptr;

    request_handler_->Handle(varname, scope, invar, &outvar,
                             request_.trainer_id());

    Finish(reply_, &responder_);
  }

 protected:
  sendrecv::VariableMessage request_;
  sendrecv::VoidMessage reply_;
  ServerAsyncResponseWriter<sendrecv::VoidMessage> responder_;
  RPCServer* rpc_server_{nullptr};
};

class RequestPrefetch final : public RequestBase {
 public:
  explicit RequestPrefetch(GrpcService::AsyncService* service,
                           ::grpc::ServerCompletionQueue* cq,
                           RequestHandler* request_handler, int req_id)
      : RequestBase(service, cq, request_handler, req_id),
        responder_(&ctx_),
        local_scope_(nullptr) {
    request_.reset(new GRPCVariableResponse(request_handler->scope(),
                                            request_handler->dev_ctx(), true));
    int method_id =
        static_cast<int>(distributed::GrpcMethod::kPrefetchVariable);
    service_->RequestAsyncUnary(
        method_id, &ctx_, request_.get(), &responder_, cq_, cq_,
        reinterpret_cast<void*>(static_cast<intptr_t>(req_id)));
  }

  virtual ~RequestPrefetch() {}

  std::string GetReqName() override { return request_->Varname(); }

  void Process() override {
    // prefetch process...
    std::string in_var_name = request_->Varname();
    std::string out_var_name = request_->OutVarname();
    std::string table_name = request_->TableName();
    int trainer_id = request_->GetTrainerId();

    VLOG(4) << "RequestPrefetch, in_var_name: " << in_var_name
            << " out_var_name: " << out_var_name << " trainer: " << trainer_id;

    auto scope = request_->GetMutableLocalScope();
    auto invar = scope->FindVar(in_var_name);
    // out var must be created in local scope!
    framework::Variable* outvar = scope->Var(out_var_name);

    request_handler_->Handle(in_var_name, scope, invar, &outvar, trainer_id,
                             out_var_name, table_name);

    SerializeToByteBuffer(out_var_name, outvar, *request_handler_->dev_ctx(),
                          &reply_);
    Finish(reply_, &responder_);
  }

 protected:
  std::shared_ptr<GRPCVariableResponse> request_;
  ::grpc::ByteBuffer reply_;
  ServerAsyncResponseWriter<::grpc::ByteBuffer> responder_;
  framework::Scope* local_scope_;
};

class RequestCheckpointNotify final : public RequestBase {
 public:
  explicit RequestCheckpointNotify(GrpcService::AsyncService* service,
                                   ::grpc::ServerCompletionQueue* cq,
                                   RequestHandler* request_handler, int req_id)
      : RequestBase(service, cq, request_handler, req_id), responder_(&ctx_) {
    request_.reset(new GRPCVariableResponse(request_handler->scope(),
                                            request_handler->dev_ctx()));
    int method_id =
        static_cast<int>(distributed::GrpcMethod::kCheckpointNotify);
    service_->RequestAsyncUnary(
        method_id, &ctx_, request_.get(), &responder_, cq_, cq_,
        reinterpret_cast<void*>(static_cast<intptr_t>(req_id)));
  }

  virtual ~RequestCheckpointNotify() {}

  std::string GetReqName() override { return request_->Varname(); }

  void Process() override {
    auto scope = request_->GetMutableLocalScope();

    std::string checkpoint_notify = request_->Varname();
    std::string checkpoint_dir = request_->OutVarname();
    int trainer_id = request_->GetTrainerId();
    std::string table_name = request_->TableName();

    VLOG(4) << "RequestCheckpointNotify notify: " << checkpoint_notify
            << ", dir: " << checkpoint_dir;

    request_handler_->Handle(checkpoint_notify, scope, nullptr, nullptr,
                             trainer_id, checkpoint_dir, table_name);
    Finish(reply_, &responder_);
  }

 protected:
  std::shared_ptr<GRPCVariableResponse> request_;
  sendrecv::VoidMessage reply_;
  ServerAsyncResponseWriter<sendrecv::VoidMessage> responder_;
};

class RequestNotify final : public RequestBase {
 public:
  explicit RequestNotify(GrpcService::AsyncService* service,
                         ::grpc::ServerCompletionQueue* cq,
                         RequestHandler* request_handler, int req_id)
      : RequestBase(service, cq, request_handler, req_id), responder_(&ctx_) {
    request_.reset(new GRPCVariableResponse(request_handler->scope(),
                                            request_handler->dev_ctx(), true));
    int method_id = static_cast<int>(distributed::GrpcMethod::kRequestNotify);
    service_->RequestAsyncUnary(
        method_id, &ctx_, request_.get(), &responder_, cq_, cq_,
        reinterpret_cast<void*>(static_cast<intptr_t>(req_id)));
  }
  virtual ~RequestNotify() {}
  std::string GetReqName() override { return request_->Varname(); }

  void Process() override {
    std::string varname = GetReqName();
    VLOG(4) << "RequestNotify var_name:" << varname;

    auto scope = request_->GetMutableLocalScope();
    auto invar = request_->GetVar();
    int trainer_id = request_->GetTrainerId();
    framework::Variable* outvar = nullptr;
    request_handler_->Handle(varname, scope, invar, &outvar, trainer_id);
    Finish(reply_, &responder_);
  }

 protected:
  sendrecv::VoidMessage reply_;
  std::shared_ptr<GRPCVariableResponse> request_;
  ServerAsyncResponseWriter<sendrecv::VoidMessage> responder_;
};

class RequestSendAndRecv final : public RequestBase {
 public:
  explicit RequestSendAndRecv(GrpcService::AsyncService* service,
                              ::grpc::ServerCompletionQueue* cq,
                              RequestHandler* request_handler, int req_id)
      : RequestBase(service, cq, request_handler, req_id), responder_(&ctx_) {
    request_.reset(new GRPCVariableResponse(request_handler->scope(),
                                            request_handler->dev_ctx(), true));

    int method_id =
        static_cast<int>(distributed::GrpcMethod::kRequestSendAndRecv);

    service_->RequestAsyncUnary(
        method_id, &ctx_, request_.get(), &responder_, cq_, cq_,
        reinterpret_cast<void*>(static_cast<intptr_t>(req_id)));
  }

  virtual ~RequestSendAndRecv() {}
  std::string GetReqName() override { return request_->Varname(); }

  void Process() override {
    std::string in_var_name = request_->Varname();
    std::string out_var_name = request_->OutVarname();
    std::string table_name = request_->TableName();
    int trainer_id = request_->GetTrainerId();

    VLOG(4) << "RequestSendAndRecv, in_var_name: " << in_var_name
            << " out_var_name: " << out_var_name << " trainer: " << trainer_id;
    auto scope = request_->GetMutableLocalScope();
    auto invar = scope->FindVar(in_var_name);
    framework::Variable* outvar = nullptr;
    request_handler_->Handle(in_var_name, scope, invar, &outvar, trainer_id,
                             out_var_name, table_name);
    SerializeToByteBuffer(out_var_name, outvar, *request_handler_->dev_ctx(),
                          &reply_);
    Finish(reply_, &responder_);
  }

 protected:
  std::shared_ptr<GRPCVariableResponse> request_;
  ::grpc::ByteBuffer reply_;
  ServerAsyncResponseWriter<::grpc::ByteBuffer> responder_;
};

void AsyncGRPCServer::WaitServerReady() {
  VLOG(4) << "AsyncGRPCServer is waiting server ready";
  std::unique_lock<std::mutex> lock(this->mutex_ready_);
  condition_ready_.wait(lock, [=] { return this->ready_ == 1; });
  VLOG(4) << "AsyncGRPCServer WaitSeverReady";
}

// Define an option subclass in order to disable SO_REUSEPORT for the
// server socket.
// Come from:
// https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/distributed_runtime/rpc/grpc_server_lib.cc
class NoReusePortOption : public ::grpc::ServerBuilderOption {
 public:
  void UpdateArguments(::grpc::ChannelArguments* args) override {
    args->SetInt(GRPC_ARG_ALLOW_REUSEPORT, 0);
  }

  void UpdatePlugins(std::vector<std::unique_ptr<::grpc::ServerBuilderPlugin>>*
                         plugins) override {}
};

void AsyncGRPCServer::StartServer() {
  for (int i = 0; i < FLAGS_rpc_retry_bind_port; i++) {
    ::grpc::ServerBuilder builder;
    std::unique_ptr<GrpcService::AsyncService> service(
        new GrpcService::AsyncService());
    builder.AddListeningPort(bind_address_, ::grpc::InsecureServerCredentials(),
                             &selected_port_);

    builder.SetMaxSendMessageSize(std::numeric_limits<int>::max());
    builder.SetMaxReceiveMessageSize(std::numeric_limits<int>::max());
    if (FLAGS_rpc_disable_reuse_port) {
      builder.SetOption(
          std::unique_ptr<::grpc::ServerBuilderOption>(new NoReusePortOption));
      LOG(INFO) << "set FLAGS_rpc_disable_reuse_port";
    }
    builder.RegisterService(service.get());

    for (auto t : rpc_call_map_) {
      rpc_cq_[t.first].reset(builder.AddCompletionQueue().release());
    }

    server_ = builder.BuildAndStart();
    if (selected_port_ != 0) {
      LOG(INFO) << "Server listening on " << bind_address_
                << " successful, selected port: " << selected_port_;
      service_.reset(service.release());
      break;
    }

    LOG(WARNING) << "Server listening on " << bind_address_
                 << " failed, selected port: " << selected_port_
                 << ", retry after 3 seconds!";

    sleep(3);
  }

  PADDLE_ENFORCE_NE(
      selected_port_, 0,
      platform::errors::Unavailable("can't bind to address:%s", bind_address_));

  std::function<void(const std::string&, int)> f =
      std::bind(&AsyncGRPCServer::TryToRegisterNewOne, this,
                std::placeholders::_1, std::placeholders::_2);

  for (auto& t : rpc_call_map_) {
    auto& rpc_name = t.first;
    auto& cq = rpc_cq_[rpc_name];
    auto threadnum = rpc_thread_num_[rpc_name];
    auto& reqs = rpc_reqs_[rpc_name];

    reqs.reserve(kRequestBufSize);

    for (int i = 0; i < kRequestBufSize; i++) {
      VLOG(6) << "TryToRegisterNewOne on RPC NAME: " << rpc_name << " I: " << i;
      TryToRegisterNewOne(rpc_name, i);
    }

    for (int i = 0; i < threadnum; i++) {
      rpc_threads_[rpc_name].emplace_back(new std::thread(std::bind(
          &AsyncGRPCServer::HandleRequest, this, cq.get(), rpc_name, f)));
      VLOG(4) << t.first << " creates threads!";
    }
  }

  {
    std::lock_guard<std::mutex> lock(this->mutex_ready_);
    ready_ = 1;
  }
  condition_ready_.notify_all();

  // wait server
  server_->Wait();

  for (auto& t : rpc_threads_) {
    auto& threads = t.second;
    for (size_t i = 0; i < threads.size(); ++i) {
      threads[i]->join();
      VLOG(4) << t.first << " threads ends!";
    }
  }
}

void AsyncGRPCServer::ShutdownQueue() {
  for (auto& t : rpc_cq_) {
    t.second->Shutdown();
    VLOG(4) << t.first << " queue shutdown!";
  }
}

void AsyncGRPCServer::ShutDownImpl() {
  std::unique_lock<std::mutex> lock(cq_mutex_);
  is_shut_down_ = true;
  ShutdownQueue();

  VLOG(4) << "server_ shutdown!";
  server_->Shutdown();
}

void AsyncGRPCServer::TryToRegisterNewOne(const std::string& rpc_name,
                                          int req_id) {
  std::unique_lock<std::mutex> lock(cq_mutex_);
  if (is_shut_down_) {
    VLOG(4) << "shutdown, do not TryToRegisterNewSendOne";
    return;
  }

  VLOG(4) << "TryToRegisterNewOne on RPC NAME: " << rpc_name
          << " REQ ID: " << req_id;

  auto& reqs = rpc_reqs_[rpc_name];
  auto& handler = rpc_call_map_[rpc_name];
  auto& cq = rpc_cq_[rpc_name];

  RequestBase* b = nullptr;
  if (rpc_name == kRequestSend) {
    b = new RequestSend(service_.get(), cq.get(), handler, req_id);
  } else if (rpc_name == kRequestGet) {
    b = new RequestGet(service_.get(), cq.get(), handler, req_id);

  } else if (rpc_name == kRequestGetNoBarrier) {
    b = new RequestGetNoBarrier(service_.get(), cq.get(), handler, req_id);
  } else if (rpc_name == kRequestGetMonomerVariable) {
    b = new RequestGetMonomerVariable(service_.get(), cq.get(), handler, req_id,
                                      this);
  } else if (rpc_name == kRequestGetMonomerBarrier) {
    b = new RequestGetMonomerBarrier(service_.get(), cq.get(), handler, req_id,
                                     this);
  } else if (rpc_name == kRequestPrefetch) {
    b = new RequestPrefetch(service_.get(), cq.get(), handler, req_id);
  } else if (rpc_name == kRequestCheckpoint) {
    b = new RequestCheckpointNotify(service_.get(), cq.get(), handler, req_id);
  } else if (rpc_name == kRequestNotify) {
    b = new RequestNotify(service_.get(), cq.get(), handler, req_id);
  } else if (rpc_name == kRequestSendAndRecv) {
    b = new RequestSendAndRecv(service_.get(), cq.get(), handler, req_id);
  } else {
    PADDLE_THROW(
        platform::errors::InvalidArgument("not supported rpc: %s", rpc_name));
  }

  reqs[req_id] = b;

  VLOG(4) << "TryToRegisterNewOne status:" << b->Status();
}

void AsyncGRPCServer::HandleRequest(
    ::grpc::ServerCompletionQueue* cq, const std::string& rpc_name,
    std::function<void(const std::string&, int)> TryToRegisterNewOne) {
  void* tag = NULL;
  bool ok = false;

  while (true) {
    VLOG(4) << "HandleRequest " << rpc_name << " wait next";
    if (!cq->Next(&tag, &ok)) {
      VLOG(4) << "CompletionQueue " << rpc_name << " shutdown!";
      break;
    }

    int req_id = static_cast<int>(reinterpret_cast<intptr_t>(tag));
    VLOG(4) << "HandleRequest " << rpc_name << ", req_id:" << req_id
            << " get next";

    auto& reqs = rpc_reqs_[rpc_name];
    RequestBase* base = nullptr;
    {
      PADDLE_ENFORCE_EQ(
          (req_id >= 0 && req_id < kRequestBufSize), true,
          platform::errors::OutOfRange("request id: %s out of bounds: [0, %s)",
                                       req_id, kRequestBufSize));
      std::unique_lock<std::mutex> lock(cq_mutex_);
      base = reqs[req_id];
    }

    VLOG(3) << base->Status2String(rpc_name);

    // reference:
    // https://github.com/tensorflow/tensorflow/issues/5596
    // https://groups.google.com/forum/#!topic/grpc-io/xftlRy-IQwM
    // https://groups.google.com/forum/#!topic/grpc-io/ywATt88Ef_I
    if (!ok) {
      VLOG(4) << "completion queue:" << rpc_name << " recv no regular event"
              << " context:" << base->Status2String(rpc_name);
      TryToRegisterNewOne(rpc_name, req_id);
      delete base;
      continue;
    }

    switch (base->Status()) {
      case PROCESS: {
        base->Process();
        break;
      }
      case FINISH: {
        TryToRegisterNewOne(rpc_name, req_id);
        delete base;
        break;
      }
      default: { assert(false); }
    }
  }
}

}  // namespace distributed
}  // namespace operators
}  // namespace paddle
