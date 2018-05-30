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

  for (auto t : rpc_call_map_) {
    rpc_cq_[t.first].reset(builder.AddCompletionQueue().release());
  }

  server_ = builder.BuildAndStart();
  LOG(INFO) << "Server listening on " << bind_address_
            << " selected port: " << selected_port_;

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
      TryToRegisterNewOne(rpc_name, i);
    }

    for (int i = 0; i < threadnum; i++) {
      rpc_threads_[rpc_name].emplace_back(new std::thread(std::bind(
          &AsyncGRPCServer::HandleRequest, this, cq.get(), rpc_name, f)));
      VLOG(3) << t.first << " creates threads!";
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
      VLOG(3) << t.first << " threads ends!";
    }
  }
}

void AsyncGRPCServer::ShutdownQueue() {
  for (auto& t : rpc_cq_) {
    t.second->Shutdown();
    VLOG(3) << t.first << " shutdown!";
  }
}

void AsyncGRPCServer::ShutDownImpl() {
  std::unique_lock<std::mutex> lock(cq_mutex_);
  is_shut_down_ = true;
  ShutdownQueue();

  VLOG(3) << "server_ shutdown!";
  server_->Shutdown();
}

void AsyncGRPCServer::TryToRegisterNewOne(const std::string& rpc_name,
                                          int req_id) {
  std::unique_lock<std::mutex> lock(cq_mutex_);
  if (is_shut_down_) {
    VLOG(3) << "shutdown, do not TryToRegisterNewSendOne";
    return;
  }

  VLOG(4) << "register send rpc_name:" << rpc_name
          << ", handler:" << rpc_call_map_[kRequestSend];

  auto& reqs = rpc_reqs_[rpc_name];
  auto& handler = rpc_call_map_[rpc_name];
  auto& cq = rpc_cq_[rpc_name];

  RequestBase* b = nullptr;
  if (rpc_name == kRequestSend) {
    b = new RequestSend(&service_, cq.get(), handler, req_id);
  } else if (rpc_name == kRequestGet) {
    b = new RequestGet(&service_, cq.get(), handler, req_id);
  } else if (rpc_name == kRequestPrefetch) {
    b = new RequestPrefetch(&service_, cq.get(), handler, req_id);
  } else {
    PADDLE_ENFORCE(false, "not surpported rpc");
  }

  reqs[req_id] = b;

  VLOG(4) << "Create RequestSend status:" << b->Status();
}

void AsyncGRPCServer::HandleRequest(
    ::grpc::ServerCompletionQueue* cq, const std::string& rpc_name,
    std::function<void(const std::string&, int)> TryToRegisterNewOne) {
  void* tag = NULL;
  bool ok = false;

  while (true) {
    VLOG(3) << "HandleRequest " << rpc_name << " wait next";
    if (!cq->Next(&tag, &ok)) {
      LOG(INFO) << "CompletionQueue " << rpc_name << " shutdown!";
      break;
    }

    int req_id = static_cast<int>(reinterpret_cast<intptr_t>(tag));
    VLOG(3) << "HandleRequest " << rpc_name << ", req_id:" << req_id
            << " get next";

    auto& reqs = rpc_reqs_[rpc_name];
    RequestBase* base = nullptr;
    {
      std::lock_guard<std::mutex> lock(cq_mutex_);
      base = reqs[req_id];
      PADDLE_ENFORCE(req_id >= 0 && req_id < kRequestBufSize);
    }

    // reference:
    // https://github.com/tensorflow/tensorflow/issues/5596
    // https://groups.google.com/forum/#!topic/grpc-io/xftlRy-IQwM
    // https://groups.google.com/forum/#!topic/grpc-io/ywATt88Ef_I
    if (!ok) {
      LOG(WARNING) << "completion queue:" << rpc_name
                   << " recv no regular event:argument name["
                   << base->GetReqName() << "]";
      TryToRegisterNewOne(rpc_name, req_id);
      delete base;
      continue;
    }

    VLOG(3) << "queue id:" << rpc_name << ", req_id:" << req_id
            << ", status:" << base->Status();

    switch (base->Status()) {
      case PROCESS: {
        std::unique_lock<std::mutex> lock(cq_mutex_);
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

}  // namespace detail
}  // namespace operators
}  // namespace paddle
