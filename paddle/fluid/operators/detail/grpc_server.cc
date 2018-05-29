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

#include <limits>
#include <string>

#include "paddle/fluid/operators/detail/grpc_server.h"

using ::grpc::ServerAsyncResponseWriter;

DEFINE_int32(rpc_server_handle_send_threads, 5,
             "Number of threads used to handle send at rpc server.");
DEFINE_int32(rpc_server_handle_get_threads, 5,
             "Number of threads used to handle get at rpc server.");
DEFINE_int32(rpc_server_handle_prefetch_threads, 1,
             "Number of threads used to handle prefetch at rpc server.");

namespace paddle {
namespace operators {
namespace detail {
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
    PADDLE_ENFORCE(cq_);
  }
  virtual ~RequestBase() {}
  virtual void Process() = 0;

  CallStatus Status() { return status_; }
  void SetStatus(CallStatus status) { status_ = status; }
  virtual std::string GetReqName() = 0;

 protected:
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
    request_.reset(new VariableResponse(request_handler->scope(),
                                        request_handler->dev_ctx(),
                                        !request_handler->sync_mode()));
    int method_id = static_cast<int>(detail::GrpcMethod::kSendVariable);
    service_->RequestAsyncUnary(
        method_id, &ctx_, request_.get(), &responder_, cq_, cq_,
        reinterpret_cast<void*>(static_cast<intptr_t>(req_id)));
  }

  virtual ~RequestSend() {}

  std::string GetReqName() override { return request_->Varname(); }

  void Process() override {
    std::string var_name = GetReqName();
    VLOG(3) << "RequestSend var_name:" << var_name;

    request_handler_->Handle(static_cast<void*>(request_.get()),
                             static_cast<void*>(&reply_));

    status_ = FINISH;
    responder_.Finish(reply_, ::grpc::Status::OK,
                      reinterpret_cast<void*>(static_cast<intptr_t>(req_id_)));
  }

 protected:
  sendrecv::VoidMessage reply_;
  std::shared_ptr<VariableResponse> request_;
  ServerAsyncResponseWriter<sendrecv::VoidMessage> responder_;
};

class RequestGet final : public RequestBase {
 public:
  explicit RequestGet(GrpcService::AsyncService* service,
                      ::grpc::ServerCompletionQueue* cq,
                      RequestHandler* request_handler, int req_id)
      : RequestBase(service, cq, request_handler, req_id), responder_(&ctx_) {
    auto method_id = static_cast<int>(detail::GrpcMethod::kGetVariable);
    service_->RequestAsyncUnary(
        method_id, &ctx_, &request_, &responder_, cq_, cq_,
        reinterpret_cast<void*>(static_cast<intptr_t>(req_id)));
  }

  virtual ~RequestGet() {}

  std::string GetReqName() override { return request_.varname(); }

  void Process() override {
    // proc request.
    std::string var_name = request_.varname();
    VLOG(3) << "RequestGet " << var_name;

    request_handler_->Handle(static_cast<void*>(&request_),
                             static_cast<void*>(&reply_));

    status_ = FINISH;
    responder_.Finish(reply_, ::grpc::Status::OK,
                      reinterpret_cast<void*>(static_cast<intptr_t>(req_id_)));
  }

 protected:
  sendrecv::VariableMessage request_;
  ::grpc::ByteBuffer reply_;
  ServerAsyncResponseWriter<::grpc::ByteBuffer> responder_;
};

class RequestPrefetch final : public RequestBase {
 public:
  explicit RequestPrefetch(GrpcService::AsyncService* service,
                           ::grpc::ServerCompletionQueue* cq,
                           RequestHandler* request_handler, int req_id)
      : RequestBase(service, cq, request_handler, req_id),
        responder_(&ctx_),
        local_scope_(nullptr) {
    request_.reset(new VariableResponse(request_handler->scope(),
                                        request_handler->dev_ctx(),
                                        !request_handler->sync_mode()));

    int method_id = static_cast<int>(detail::GrpcMethod::kPrefetchVariable);
    service_->RequestAsyncUnary(
        method_id, &ctx_, request_.get(), &responder_, cq_, cq_,
        reinterpret_cast<void*>(static_cast<intptr_t>(req_id)));
  }

  virtual ~RequestPrefetch() {}

  std::string GetReqName() override { return request_->Varname(); }

  void Process() override {
    // prefetch process...
    std::string var_name = request_->OutVarname();
    VLOG(3) << "RequestPrefetch " << var_name;

    request_handler_->Handle(static_cast<void*>(request_.get()),
                             static_cast<void*>(&reply_));

    status_ = FINISH;
    responder_.Finish(reply_, ::grpc::Status::OK,
                      reinterpret_cast<void*>(static_cast<intptr_t>(req_id_)));
  }

 protected:
  std::shared_ptr<VariableResponse> request_;
  ::grpc::ByteBuffer reply_;
  ServerAsyncResponseWriter<::grpc::ByteBuffer> responder_;
  framework::Scope* local_scope_;
};

void AsyncGRPCServer::WaitServerReady() {
  VLOG(3) << "AsyncGRPCServer is wait server ready";
  std::unique_lock<std::mutex> lock(this->mutex_ready_);
  condition_ready_.wait(lock, [=] { return this->ready_ == 1; });
  VLOG(3) << "AsyncGRPCServer WaitSeverReady";
}

void AsyncGRPCServer::StartServer() {
  ::grpc::ServerBuilder builder;
  builder.AddListeningPort(bind_address_, ::grpc::InsecureServerCredentials(),
                           &selected_port_);

  builder.SetMaxSendMessageSize(std::numeric_limits<int>::max());
  builder.SetMaxReceiveMessageSize(std::numeric_limits<int>::max());
  builder.RegisterService(&service_);

  cq_send_ = builder.AddCompletionQueue();
  cq_get_ = builder.AddCompletionQueue();
  cq_prefetch_ = builder.AddCompletionQueue();

  server_ = builder.BuildAndStart();
  LOG(INFO) << "Server listening on " << bind_address_
            << " selected port: " << selected_port_;

  std::function<void(int)> send_register = std::bind(
      &AsyncGRPCServer::TryToRegisterNewSendOne, this, std::placeholders::_1);
  std::function<void(int)> get_register = std::bind(
      &AsyncGRPCServer::TryToRegisterNewGetOne, this, std::placeholders::_1);
  std::function<void(int)> prefetch_register =
      std::bind(&AsyncGRPCServer::TryToRegisterNewPrefetchOne, this,
                std::placeholders::_1);

  if (rpc_call_map_.find(kRequestSend) != rpc_call_map_.end()) {
    for (int i = 0; i < kSendReqsBufSize; ++i) {
      TryToRegisterNewSendOne(i);
    }
    for (int i = 0; i < FLAGS_rpc_server_handle_send_threads; ++i) {
      t_sends_.emplace_back(new std::thread(std::bind(
          &AsyncGRPCServer::HandleRequest, this, cq_send_.get(),
          static_cast<int>(GrpcMethod::kSendVariable), send_register)));
    }
  }

  if (rpc_call_map_.find(kRequestGet) != rpc_call_map_.end()) {
    for (int i = 0; i < kGetReqsBufSize; ++i) {
      TryToRegisterNewGetOne(i);
    }
    for (int i = 0; i < FLAGS_rpc_server_handle_get_threads; ++i) {
      t_gets_.emplace_back(new std::thread(
          std::bind(&AsyncGRPCServer::HandleRequest, this, cq_get_.get(),
                    static_cast<int>(GrpcMethod::kGetVariable), get_register)));
    }
  }

  if (rpc_call_map_.find(kRequestPrefetch) != rpc_call_map_.end()) {
    for (int i = 0; i < kPrefetchReqsBufSize; ++i) {
      TryToRegisterNewPrefetchOne(i);
    }
    for (int i = 0; i < FLAGS_rpc_server_handle_prefetch_threads; ++i) {
      t_prefetchs_.emplace_back(new std::thread(std::bind(
          &AsyncGRPCServer::HandleRequest, this, cq_prefetch_.get(),
          static_cast<int>(GrpcMethod::kPrefetchVariable), prefetch_register)));
    }
  }

  {
    std::lock_guard<std::mutex> lock(this->mutex_ready_);
    ready_ = 1;
  }
  condition_ready_.notify_all();
  // wait server
  server_->Wait();
  for (size_t i = 0; i < t_sends_.size(); ++i) {
    t_sends_[i]->join();
  }
  VLOG(3) << "t_sends_ ends";

  for (size_t i = 0; i < t_gets_.size(); ++i) {
    t_gets_[i]->join();
  }
  VLOG(3) << "t_gets_ ends";

  for (size_t i = 0; i < t_prefetchs_.size(); ++i) {
    t_prefetchs_[i]->join();
  }
  VLOG(3) << "t_prefetchs_ ends";
}

void AsyncGRPCServer::ShutdownQueue() {
  VLOG(3) << "cq_send_ shutdown!";
  cq_send_->Shutdown();

  VLOG(3) << "cq_get_ shutdown!";
  cq_get_->Shutdown();

  VLOG(3) << "cq_prefetch_ shutdown!";
  cq_prefetch_->Shutdown();
}

void AsyncGRPCServer::ShutDownImpl() {
  std::unique_lock<std::mutex> lock(cq_mutex_);
  is_shut_down_ = true;
  ShutdownQueue();

  VLOG(3) << "server_ shutdown!";
  server_->Shutdown();
}

void AsyncGRPCServer::TryToRegisterNewSendOne(int i) {
  std::unique_lock<std::mutex> lock(cq_mutex_);
  if (is_shut_down_) {
    VLOG(3) << "shutdown, do not TryToRegisterNewSendOne";
    return;
  }

  VLOG(4) << "register send req_id:" << i
          << ", handler:" << rpc_call_map_[kRequestSend];
  RequestSend* send = new RequestSend(&service_, cq_send_.get(),
                                      rpc_call_map_[kRequestSend], i);
  send_reqs_[i] = send;
  VLOG(4) << "Create RequestSend status:" << send->Status();
}

void AsyncGRPCServer::TryToRegisterNewGetOne(int req_id) {
  std::unique_lock<std::mutex> lock(cq_mutex_);
  if (is_shut_down_) {
    VLOG(3) << "shutdown, do not TryToRegisterNewGetOne";
    return;
  }

  VLOG(4) << "register get req_id:" << req_id;
  RequestGet* get = new RequestGet(&service_, cq_get_.get(),
                                   rpc_call_map_[kRequestGet], req_id);
  get_reqs_[req_id] = get;
  VLOG(4) << "Create RequestGet status:" << get->Status();
}

void AsyncGRPCServer::TryToRegisterNewPrefetchOne(int req_id) {
  std::unique_lock<std::mutex> lock(cq_mutex_);
  if (is_shut_down_) {
    VLOG(3) << "shutdown, do not TryToRegisterNewPrefetchOne";
    return;
  }

  VLOG(4) << "register prefetch req_id:" << req_id;
  RequestPrefetch* prefetch = new RequestPrefetch(
      &service_, cq_prefetch_.get(), rpc_call_map_[kRequestPrefetch], req_id);

  prefetch_reqs_[req_id] = prefetch;

  VLOG(4) << "Create RequestPrefetch status:" << prefetch->Status();
}

void AsyncGRPCServer::HandleRequest(
    ::grpc::ServerCompletionQueue* cq, int rpc_id,
    std::function<void(int)> TryToRegisterNewOne) {
  void* tag = NULL;
  bool ok = false;

  while (true) {
    VLOG(3) << "HandleRequest queue_id:" << rpc_id << " wait next";
    if (!cq->Next(&tag, &ok)) {
      LOG(INFO) << "CompletionQueue " << rpc_id << " shutdown!";
      break;
    }

    int req_id = static_cast<int>(reinterpret_cast<intptr_t>(tag));
    VLOG(3) << "HandleRequest queue_id:" << rpc_id << ", req_id:" << req_id
            << " get next";

    RequestBase* base = nullptr;
    {
      std::lock_guard<std::mutex> lock(cq_mutex_);
      if (rpc_id == static_cast<int>(GrpcMethod::kGetVariable)) {
        PADDLE_ENFORCE(req_id >= 0 && req_id < kGetReqsBufSize);
        base = get_reqs_[req_id];
      } else if (rpc_id == static_cast<int>(GrpcMethod::kSendVariable)) {
        PADDLE_ENFORCE(req_id >= 0 && req_id < kSendReqsBufSize);
        base = send_reqs_[req_id];
      } else if (rpc_id == static_cast<int>(GrpcMethod::kPrefetchVariable)) {
        PADDLE_ENFORCE(req_id >= 0 && req_id < kPrefetchReqsBufSize);
        base = prefetch_reqs_[req_id];
      } else {
        PADDLE_ENFORCE(false, "not surpported rpc_id");
      }
    }

    // reference:
    // https://github.com/tensorflow/tensorflow/issues/5596
    // https://groups.google.com/forum/#!topic/grpc-io/xftlRy-IQwM
    // https://groups.google.com/forum/#!topic/grpc-io/ywATt88Ef_I
    if (!ok) {
      LOG(WARNING) << "completion queue:" << rpc_id
                   << " recv no regular event:argument name["
                   << base->GetReqName() << "]";
      TryToRegisterNewOne(req_id);
      delete base;
      continue;
    }

    VLOG(3) << "queue id:" << rpc_id << ", req_id:" << req_id
            << ", status:" << base->Status();

    switch (base->Status()) {
      case PROCESS: {
        std::unique_lock<std::mutex> lock(cq_mutex_);
        base->Process();
        break;
      }
      case FINISH: {
        TryToRegisterNewOne(req_id);
        delete base;
        break;
      }
      default: { assert(false); }
    }
  }
}

}  // namespace detail
}  // namespace operators
}  // namespace paddle
