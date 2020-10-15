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

#include "paddle/fluid/operators/distributed/brpc/brpc_client.h"
#include "paddle/fluid/framework/threadpool.h"
#include "paddle/fluid/operators/distributed/brpc/brpc_sendrecvop_utils.h"
#include "paddle/fluid/platform/profiler.h"

namespace paddle {
namespace operators {
namespace distributed {

DEFINE_int32(timeout_ms, 30000, "RPC timeout in milliseconds");
DEFINE_int32(max_retry, 3, "Max retries(not including the first RPC)");

BRPCClient::~BRPCClient() { Wait(); }

void HandleSendResponse(brpc::Controller* cntl, sendrecv::VoidMessage* response,
                        VarHandlePtr var_h, ChannelQueuePtr ch_ptr,
                        ChannelContextPtr ch_ctx, BRPCClient* cls) {
  // std::unique_ptr makes sure cntl/response will be deleted before returning.
  std::unique_ptr<brpc::Controller> cntl_guard(cntl);
  std::unique_ptr<sendrecv::VoidMessage> response_guard(response);

  // this channel can be used by other now.
  ch_ptr->Push(ch_ctx);

  if (cntl->Failed()) {
    PADDLE_THROW(platform::errors::Unavailable(
        "Failed to send variable %s, error text is %s.", var_h->name(),
        cntl->ErrorText()));
    var_h->Finish(false);
    cls->DecreaseReqCount();
    return;
  }
  var_h->Finish(true);
  cls->DecreaseReqCount();

  VLOG(4) << "HandleSendResponse from: " << cntl->remote_side()
          << ", varname: " << var_h->name()
          << ", latency: " << cntl->latency_us() << "us";
  VLOG(4) << "Finish HandleSendResponse";
}

VarHandlePtr BRPCClient::AsyncSendVar(const std::string& ep,
                                      const platform::DeviceContext& ctx,
                                      const framework::Scope& scope,
                                      const std::string& var_name,
                                      int64_t time_out) {
  const platform::DeviceContext* p_ctx = &ctx;
  const std::string ep_val = ep;
  const std::string var_name_val = var_name;
  const framework::Scope* p_scope = &scope;
  const auto ch_ptr = GetChannel(ep_val);
  const std::string method = kSendRPC;
  VarHandlePtr var_h(new VarHandle(ep, method, var_name_val, p_ctx, p_scope));

  framework::AsyncIO([=] {
    auto ch_ctx = ch_ptr->Pop();
    brpc::Controller* cntl = new brpc::Controller();
    sendrecv::VoidMessage* response = new sendrecv::VoidMessage();
    cntl->set_timeout_ms(time_out);

    auto* var = p_scope->FindVar(var_name_val);
    sendrecv::VariableMessage request;
    distributed::SerializeToIOBuf(var_name_val, var, *p_ctx, &request,
                                  &cntl->request_attachment(), "", false,
                                  trainer_id_);

    google::protobuf::Closure* done = brpc::NewCallback(
        &HandleSendResponse, cntl, response, var_h, ch_ptr, ch_ctx, this);

    platform::RecordRPCEvent record_event(method);

    ch_ctx->stub->SendVariable(cntl, &request, response, done);

    if (UNLIKELY(platform::IsProfileEnabled())) {
      var_h->Wait();
    }
  });
  req_count_++;

  return var_h;
}
void HandleFetchBarrierResponse(brpc::Controller* cntl,
                                sendrecv::VariableMessage* response,
                                VarHandlePtr var_h, ChannelQueuePtr ch_ptr,
                                ChannelContextPtr ch_ctx, BRPCClient* cls) {
  // std::unique_ptr makes sure cntl/response will be deleted before returning.
  std::unique_ptr<brpc::Controller> cntl_guard(cntl);
  std::unique_ptr<sendrecv::VariableMessage> response_guard(response);

  // this channel can be used other now.
  ch_ptr->Push(ch_ctx);

  if (cntl->Failed()) {
    PADDLE_THROW(platform::errors::Unavailable(
        "Failed to get HandleFetchBarrierResponse %s, error text is %s.",
        var_h->name(), cntl->ErrorText()));
    var_h->Finish(false);
    cls->DecreaseReqCount();
    return;
  }

  var_h->Finish(true);
  cls->DecreaseReqCount();

  VLOG(4) << "HandleFetchBarrierResponse from: " << cntl->remote_side()
          << ", varname: " << var_h->name()
          << ", latency: " << cntl->latency_us() << "us";
  VLOG(4) << "Finish HandleFetchBarrierResponse";
}
void HandleGetResponse(brpc::Controller* cntl,
                       sendrecv::VariableMessage* response, VarHandlePtr var_h,
                       ChannelQueuePtr ch_ptr, ChannelContextPtr ch_ctx,
                       BRPCClient* cls) {
  // std::unique_ptr makes sure cntl/response will be deleted before returning.
  std::unique_ptr<brpc::Controller> cntl_guard(cntl);
  std::unique_ptr<sendrecv::VariableMessage> response_guard(response);

  // this channel can be used other now.
  ch_ptr->Push(ch_ctx);

  if (cntl->Failed()) {
    PADDLE_THROW(platform::errors::Unavailable(
        "Failed to get variable %s, error text is %s.", var_h->name(),
        cntl->ErrorText()));
    cls->DecreaseReqCount();
    var_h->Finish(false);
    return;
  }

  VLOG(4) << "HandleGetResponse from: " << cntl->remote_side()
          << ", varname: " << var_h->name()
          << ", latency: " << cntl->latency_us() << "us";

  framework::Variable* outvar = nullptr;
  int trainer_id;
  distributed::DeserializeFromIOBuf(*response, cntl->response_attachment(),
                                    *var_h->ctx(), var_h->scope(), &outvar,
                                    &trainer_id);
  VLOG(4) << "Finish HandleGetResponse";
  cls->DecreaseReqCount();
  var_h->Finish(true);
}

VarHandlePtr BRPCClient::_AsyncGetVar(const std::string& ep,
                                      const platform::DeviceContext& ctx,
                                      const framework::Scope& scope,
                                      const std::string& var_name,
                                      const std::string& out_var_name,
                                      const std::string& method_name,
                                      int64_t time_out) {
  const platform::DeviceContext* p_ctx = &ctx;
  const std::string ep_val = ep;
  const std::string var_name_val = var_name;
  const std::string out_varname_val = out_var_name;
  const framework::Scope* p_scope = &scope;
  const auto ch_ptr = GetChannel(ep_val);
  const std::string method = kGetRPC;
  VarHandlePtr var_h(
      new VarHandle(ep, method, out_varname_val, p_ctx, p_scope));

  framework::AsyncIO([=] {
    auto ch_ctx = ch_ptr->Pop();

    brpc::Controller* cntl = new brpc::Controller();
    sendrecv::VariableMessage* response = new sendrecv::VariableMessage();
    cntl->set_timeout_ms(time_out);

    sendrecv::VariableMessage req;
    req.set_varname(var_name_val);
    req.set_out_varname(out_varname_val);
    req.set_trainer_id(trainer_id_);

    google::protobuf::Closure* done = brpc::NewCallback(
        &HandleGetResponse, cntl, response, var_h, ch_ptr, ch_ctx, this);

    platform::RecordRPCEvent record_event(method);

    if (method_name == kGetMonomerRPC) {
      ch_ctx->stub->GetMonomerVariable(cntl, &req, response, done);
    } else if (method_name == kGetNoBarrierRPC) {
      ch_ctx->stub->GetVariableNoBarrier(cntl, &req, response, done);
    } else {
      ch_ctx->stub->GetVariable(cntl, &req, response, done);
    }

    if (UNLIKELY(platform::IsProfileEnabled())) {
      var_h->Wait();
    }
  });

  req_count_++;

  return var_h;
}

VarHandlePtr BRPCClient::AsyncGetVarNoBarrier(
    const std::string& ep, const platform::DeviceContext& ctx,
    const framework::Scope& scope, const std::string& var_name,
    const std::string& out_var_name, int64_t time_out) {
  std::string var_name_no_barrier =
      string::Sprintf("%s%s", var_name, WITHOUT_BARRIER_MESSAGE);

  return _AsyncGetVar(ep, ctx, scope, var_name_no_barrier, out_var_name,
                      kGetNoBarrierRPC, time_out);
}

VarHandlePtr BRPCClient::AsyncGetMonomerVariable(
    const std::string& ep, const platform::DeviceContext& ctx,
    const framework::Scope& scope, const std::string& var_name,
    int64_t time_out) {
  return _AsyncGetVar(ep, ctx, scope, var_name, var_name, kGetMonomerRPC,
                      time_out);
}

VarHandlePtr BRPCClient::AsyncGetMonomerBarrier(const std::string& ep,
                                                const std::string& var_name,
                                                int64_t time_out) {
  return AsyncSendMessage(ep, kSendMonomerFetchBarrierRPC, var_name, time_out);
}

VarHandlePtr BRPCClient::AsyncGetVar(const std::string& ep,
                                     const platform::DeviceContext& ctx,
                                     const framework::Scope& scope,
                                     const std::string& var_name,
                                     const std::string& out_var_name,
                                     const std::string& table_name,
                                     int64_t time_out) {
  return _AsyncGetVar(ep, ctx, scope, var_name, out_var_name, kGetRPC,
                      time_out);
}

VarHandlePtr BRPCClient::AsyncPrefetchVar(const std::string& ep,
                                          const platform::DeviceContext& ctx,
                                          const framework::Scope& scope,
                                          const std::string& in_var_name,
                                          const std::string& out_var_name,
                                          const std::string& table_name,
                                          int64_t time_out) {
  const platform::DeviceContext* p_ctx = &ctx;
  const std::string ep_val = ep;
  const std::string in_var_name_val = in_var_name;
  const std::string out_var_name_val = out_var_name;
  const std::string table_name_val = table_name;
  const framework::Scope* p_scope = &scope;
  const auto ch_ptr = GetChannel(ep_val);

  const std::string method = kPrefetchRPC;

  VarHandlePtr var_h(
      new VarHandle(ep, method, out_var_name_val, p_ctx, p_scope));

  framework::AsyncIO([=] {
    auto ch_ctx = ch_ptr->Pop();

    brpc::Controller* cntl = new brpc::Controller();
    sendrecv::VariableMessage* response = new sendrecv::VariableMessage();
    cntl->set_timeout_ms(time_out);

    auto* var = p_scope->FindVar(in_var_name_val);
    sendrecv::VariableMessage req;
    distributed::SerializeToIOBuf(in_var_name_val, var, *p_ctx, &req,
                                  &cntl->request_attachment(), out_var_name_val,
                                  false, 0, table_name_val);

    platform::RecordRPCEvent record_event(method);

    google::protobuf::Closure* done = brpc::NewCallback(
        &HandleGetResponse, cntl, response, var_h, ch_ptr, ch_ctx, this);

    ch_ctx->stub->PrefetchVariable(cntl, &req, response, done);

    if (UNLIKELY(platform::IsProfileEnabled())) {
      var_h->Wait();
    }
  });

  req_count_++;
  return var_h;
}

VarHandlePtr BRPCClient::AsyncSendBatchBarrier(const std::string& ep,
                                               int64_t time_out) {
  return AsyncSendMessage(ep, kBatchBarrierRPC, BATCH_BARRIER_MESSAGE,
                          time_out);
}

VarHandlePtr BRPCClient::AsyncSendFetchBarrier(const std::string& ep,
                                               int64_t time_out) {
  auto ch_ptr = GetChannel(ep);
  auto ch_ctx = ch_ptr->Pop();

  brpc::Controller* cntl = new brpc::Controller();
  sendrecv::VariableMessage* response = new sendrecv::VariableMessage();
  cntl->set_timeout_ms(time_out);

  sendrecv::VariableMessage req;
  req.set_varname(FETCH_BARRIER_MESSAGE);

  const std::string method = kFetchBarrierRPC;
  // var handle
  VarHandlePtr var_h(
      new VarHandle(ep, method, FETCH_BARRIER_MESSAGE, nullptr, nullptr));

  platform::RecordRPCEvent record_event(method);

  google::protobuf::Closure* done = brpc::NewCallback(
      &HandleFetchBarrierResponse, cntl, response, var_h, ch_ptr, ch_ctx, this);

  ch_ctx->stub->GetVariable(cntl, &req, response, done);

  req_count_++;

  if (UNLIKELY(platform::IsProfileEnabled())) {
    var_h->Wait();
  }

  return var_h;
}

bool BRPCClient::Wait() {
  VLOG(9) << "begin to brpcclient wait";
  {
    std::unique_lock<std::mutex> lk(sync_mutex_);
    sync_cond_.wait(lk, [this] { return req_count_ == 0; });
  }
  VLOG(9) << "end to brpcclient wait";
  return true;
}

ChannelQueuePtr BRPCClient::GetChannel(const std::string& ep) {
  VLOG(4) << "begin to GetChannel:" << ep;
  {
    std::lock_guard<std::mutex> guard(chan_mutex_);
    auto it = channels_.find(ep);
    if (it != channels_.end()) {
      VLOG(4) << "end to GetChannel:" << ep;
      return it->second;
    }
  }

  ChannelQueuePtr q(new framework::BlockingQueue<ChannelContextPtr>());

  brpc::ChannelOptions options;
#ifdef PADDLE_WITH_BRPC_RDMA
  options.use_rdma = true;
#endif
  options.protocol = "baidu_std";
  // don't use pooled type. the server can't afford that.
  options.connection_type = "single";
  options.connect_timeout_ms = 1000;
  options.timeout_ms = FLAGS_timeout_ms /*milliseconds*/;
  options.max_retry = FLAGS_max_retry;

  VLOG(1) << "create " << brpc_channel_num_per_server_
          << " brpc channels to pserver:" << ep;

  for (int i = 0; i < brpc_channel_num_per_server_; ++i) {
    std::shared_ptr<ChannelContext> c(new ChannelContext());
    if (c->channel.Init(ep.c_str(), &options) != 0) {
      PADDLE_THROW(
          platform::errors::Unavailable("Failed to initialize channel."));
      return nullptr;
    }

    c->stub.reset(new sendrecv::SendRecvService_Stub(
        static_cast<google::protobuf::RpcChannel*>(&c->channel)));
    q->Push(c);
  }

  {
    std::lock_guard<std::mutex> guard(chan_mutex_);
    channels_[ep] = q;
  }

  VLOG(4) << "end to GetChannel:" << ep;
  return q;
}

VarHandlePtr BRPCClient::AsyncSendComplete(const std::string& ep,
                                           int64_t time_out) {
  return AsyncSendMessage(ep, kSendCompleteRPC, COMPLETE_MESSAGE, time_out);
}

void BRPCClient::SendComplete() {
  for (auto& kv : channels_) {
    AsyncSendComplete(kv.first);
  }
}

VarHandlePtr BRPCClient::AsyncSendVarMessage(
    const std::string& ep, const std::string& method_name,
    const sendrecv::VariableMessage& req, int64_t time_out) {
  auto ch_ptr = GetChannel(ep);
  auto ch_ctx = ch_ptr->Pop();

  brpc::Controller* cntl = new brpc::Controller();
  sendrecv::VoidMessage* response = new sendrecv::VoidMessage();
  cntl->set_timeout_ms(time_out);

  platform::RecordRPCEvent record_event(method_name);

  VarHandlePtr var_h(
      new VarHandle(ep, method_name, req.varname(), nullptr, nullptr));

  google::protobuf::Closure* done = brpc::NewCallback(
      &HandleSendResponse, cntl, response, var_h, ch_ptr, ch_ctx, this);

  if (method_name == kCheckPointNotifyRPC) {
    ch_ctx->stub->CheckpointNotify(cntl, &req, response, done);
  } else if (method_name == kSendMonomerFetchBarrierRPC) {
    ch_ctx->stub->GetMonomerBarrier(cntl, &req, response, done);
  } else {
    ch_ctx->stub->SendVariable(cntl, &req, response, done);
  }
  req_count_++;

  if (UNLIKELY(platform::IsProfileEnabled())) {
    var_h->Wait();
  }

  return var_h;
}

VarHandlePtr BRPCClient::AsyncSendMessage(const std::string& ep,
                                          const std::string& method_name,
                                          const std::string& message,
                                          int64_t time_out) {
  sendrecv::VariableMessage req;
  req.set_varname(message);

  return AsyncSendVarMessage(ep, method_name, req, time_out);
}

VarHandlePtr BRPCClient::AsyncCheckpointNotify(const std::string& ep,
                                               const std::string& dirname,
                                               const std::string& varname,
                                               const int mode,
                                               int64_t time_out) {
  sendrecv::VariableMessage req;
  req.set_varname(varname);
  req.set_out_varname(dirname);

  return AsyncSendVarMessage(ep, "CheckPointNotifyRPC", req, time_out);
}

}  // namespace distributed
}  // namespace operators
}  // namespace paddle
