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
#include <limits>

#include "glog/logging.h"  // For VLOG
#include "paddle/fluid/framework/threadpool.h"
#include "paddle/fluid/operators/distributed/grpc/grpc_client.h"
#include "paddle/fluid/operators/distributed/grpc/grpc_serde.h"
#include "paddle/fluid/operators/distributed/request_handler.h"
#include "paddle/fluid/platform/port.h"
#include "paddle/fluid/platform/profiler.h"

DECLARE_bool(rpc_disable_reuse_port);

namespace paddle {
namespace operators {
namespace distributed {

void GRPCClient::InitImpl() {
  // start the client process thread
  // TODO(wuyi): can make this in a threadpool
  PADDLE_ENFORCE_EQ(client_thread_ == nullptr, true,
                    platform::errors::PreconditionNotMet(
                        "please not re init proceed thread"));
  client_thread_.reset(new std::thread(std::bind(&GRPCClient::Proceed, this)));
}

void GRPCClient::SendComplete() {
  std::unique_lock<std::mutex> lk(completed_mutex_);
  if (!completed_) {
    for (auto& it : channels_) {
      VLOG(3) << "send complete message to " << it.first;
      this->AsyncSendComplete(it.first);
    }
    PADDLE_ENFORCE_EQ(this->Wait(), true, platform::errors::PreconditionNotMet(
                                              "internal grpc service error."));
    completed_ = true;
  }
}

GRPCClient::~GRPCClient() {
  stopped_ = true;
  Wait();
  cq_.Shutdown();
  {
    std::lock_guard<std::mutex> guard(chan_mutex_);
    for (auto& it : channels_) {
      it.second.reset();
    }
    channels_.clear();
  }
  client_thread_->join();
}

VarHandlePtr GRPCClient::AsyncSendVar(const std::string& ep,
                                      const platform::DeviceContext& ctx,
                                      const framework::Scope& scope,
                                      const std::string& var_name,
                                      int64_t time_out) {
  const platform::DeviceContext* p_ctx = &ctx;
  const std::string ep_val = ep;
  const std::string var_name_val = var_name;
  const framework::Scope* p_scope = &scope;
  const auto ch = GetChannel(ep_val);
  const std::string method = kSendRPC;

  int retry_times_ = 0;

  while (true) {
    SendProcessor* s = new SendProcessor(ch);
    VarHandlePtr h(new VarHandle(ep, method, var_name_val, p_ctx, p_scope));
    s->Prepare(h, time_out);

    framework::AsyncIO([var_name_val, p_scope, p_ctx, s, method, h, this] {
      auto* var = p_scope->FindVar(var_name_val);

      ::grpc::ByteBuffer req;
      SerializeToByteBuffer(var_name_val, var, *p_ctx, &req, "", trainer_id_);

      VLOG(3) << s->GetVarHandlePtr()->String() << " begin";

      // stub context
      s->response_call_back_ = nullptr;

      platform::RecordRPCEvent record_event(method);

      auto call = s->stub_g_.PrepareUnaryCall(
          s->context_.get(), "/sendrecv.SendRecvService/SendVariable", req,
          &cq_);
      call->StartCall();
      call->Finish(&s->reply_, &s->status_, reinterpret_cast<void*>(s));

      if (UNLIKELY(platform::IsProfileEnabled())) {
        h->Wait();
      }
    });
    req_count_++;

    if (FLAGS_rpc_retry_times > 0 && retry_times_ < FLAGS_rpc_retry_times) {
      h->Wait();
      if (h->should_retry) {
        VLOG(3) << "rpc call failed, retry times " << retry_times_;
        retry_times_++;
        std::random_device rd;
        std::this_thread::sleep_for(std::chrono::milliseconds(rd() % 5));
        continue;
      }
    }

    return h;
  }
}

void ProcGetResponse(const VarHandle& var_h,
                     const ::grpc::ByteBuffer& ret_msg) {
  VLOG(4) << "ProcGetResponse";
  framework::Variable* outvar = nullptr;
  // get response's trainer_id is not used
  int trainer_id;
  DeserializeFromByteBuffer(ret_msg, *var_h.ctx(), var_h.scope(), &outvar,
                            &trainer_id);
}

void ProcGetRecvResponse(const VarHandle& var_h,
                         const ::grpc::ByteBuffer& ret_msg) {
  VLOG(4) << "ProcGetRecvResponse";
  framework::Variable* outvar = nullptr;
  int trainer_id;
  DeserializeRecvFromByteBuffer(ret_msg, *var_h.ctx(), var_h.scope(), &outvar,
                                &trainer_id);
}

template <typename T>
void RequestToByteBuffer(const T& proto, ::grpc::ByteBuffer* result) {
  ::grpc::Slice slice(proto.ByteSizeLong());
  proto.SerializeWithCachedSizesToArray(const_cast<uint8_t*>(slice.begin()));
  ::grpc::ByteBuffer tmp(&slice, 1);
  result->Swap(&tmp);
}

VarHandlePtr GRPCClient::AsyncGetVar(const std::string& ep,
                                     const platform::DeviceContext& ctx,
                                     const framework::Scope& scope,
                                     const std::string& var_name,
                                     const std::string& out_varname,
                                     const std::string& table_name,
                                     int64_t time_out) {
  return _AsyncGetVar(ep, ctx, scope, kGetRPC, var_name, out_varname,
                      "/sendrecv.SendRecvService/GetVariable", table_name,
                      time_out);
}

VarHandlePtr GRPCClient::AsyncGetVarNoBarrier(
    const std::string& ep, const platform::DeviceContext& ctx,
    const framework::Scope& scope, const std::string& var_name,
    const std::string& out_varname, int64_t time_out) {
  std::string var_name_no_barrier =
      string::Sprintf("%s%s", var_name, WITHOUT_BARRIER_MESSAGE);

  return _AsyncGetVar(
      ep, ctx, scope, kGetNoBarrierRPC, var_name_no_barrier, out_varname,
      "/sendrecv.SendRecvService/GetVariableNoBarrier", "", time_out);
}

VarHandlePtr GRPCClient::AsyncGetMonomerVariable(
    const std::string& ep, const platform::DeviceContext& ctx,
    const framework::Scope& scope, const std::string& var_name,
    int64_t time_out) {
  return _AsyncGetVar(ep, ctx, scope, kGetMonomerRPC, var_name, var_name,
                      "/sendrecv.SendRecvService/GetMonomerVariable", "",
                      time_out);
}

VarHandlePtr GRPCClient::_AsyncGetVar(
    const std::string& ep, const platform::DeviceContext& ctx,
    const framework::Scope& scope, const std::string& method,
    const std::string& var_name, const std::string& out_varname,
    const std::string& rpc_path, const std::string& table_name,
    int64_t time_out) {
  const platform::DeviceContext* p_ctx = &ctx;
  const std::string ep_val = ep;
  const std::string var_name_val = var_name;
  const std::string out_varname_val = out_varname;
  const std::string table_name_val = table_name;
  const framework::Scope* p_scope = &scope;
  const auto ch = GetChannel(ep_val);

  int retry_times_ = 0;

  while (true) {
    GetProcessor* s = new GetProcessor(ch);

    VarHandlePtr h(new VarHandle(ep, method, out_varname_val, p_ctx, p_scope));
    s->Prepare(h, time_out);

    framework::AsyncIO([var_name_val, out_varname_val, table_name_val, s,
                        method, p_ctx, h, rpc_path, this] {
      // prepare input
      sendrecv::VariableMessage req;
      req.set_varname(var_name_val);
      req.set_out_varname(out_varname_val);
      req.set_trainer_id(trainer_id_);
      req.set_table_name(table_name_val);
      ::grpc::ByteBuffer buf;
      RequestToByteBuffer<sendrecv::VariableMessage>(req, &buf);

      VLOG(3) << s->GetVarHandlePtr()->String() << " begin";

      // stub context
      s->response_call_back_ = ProcGetResponse;

      platform::RecordRPCEvent record_event(method);

      auto call =
          s->stub_g_.PrepareUnaryCall(s->context_.get(), rpc_path, buf, &cq_);
      call->StartCall();
      call->Finish(&s->reply_, &s->status_, reinterpret_cast<void*>(s));

      if (UNLIKELY(platform::IsProfileEnabled())) {
        h->Wait();
      }
    });
    req_count_++;

    if (FLAGS_rpc_retry_times > 0 && retry_times_ < FLAGS_rpc_retry_times) {
      h->Wait();
      if (h->should_retry) {
        VLOG(3) << "rpc call failed, retry times " << retry_times_;
        retry_times_++;
        std::random_device rd;
        std::this_thread::sleep_for(std::chrono::milliseconds(rd() % 5));
        continue;
      }
    }

    return h;
  }
}

VarHandlePtr GRPCClient::AsyncPrefetchVar(const std::string& ep,
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
  const auto ch = GetChannel(ep_val);

  const std::string method = kPrefetchRPC;
  int retry_times_ = 0;

  while (true) {
    GetProcessor* s = new GetProcessor(ch);
    VarHandlePtr h(new VarHandle(ep, method, out_var_name_val, p_ctx, p_scope));
    s->Prepare(h, kPrefetchTimeout);

    framework::AsyncIO([in_var_name_val, out_var_name_val, ep_val, p_scope,
                        p_ctx, s, method, h, table_name_val, this] {
      auto* var = p_scope->FindVar(in_var_name_val);

      ::grpc::ByteBuffer req;
      SerializeToByteBuffer(in_var_name_val, var, *p_ctx, &req,
                            out_var_name_val, 0, table_name_val);

      VLOG(3) << s->GetVarHandlePtr()->String() << " begin";

      // stub context
      s->response_call_back_ = ProcGetResponse;

      platform::RecordRPCEvent record_event(method);

      auto call = s->stub_g_.PrepareUnaryCall(
          s->context_.get(), "/sendrecv.SendRecvService/PrefetchVariable", req,
          &cq_);
      call->StartCall();
      call->Finish(&s->reply_, &s->status_, static_cast<void*>(s));

      if (UNLIKELY(platform::IsProfileEnabled())) {
        h->Wait();
      }
    });
    req_count_++;

    if (FLAGS_rpc_retry_times > 0 && retry_times_ < FLAGS_rpc_retry_times) {
      h->Wait();
      if (h->should_retry) {
        VLOG(3) << "rpc call failed, retry times " << retry_times_;
        retry_times_++;
        std::random_device rd;
        std::this_thread::sleep_for(std::chrono::milliseconds(rd() % 5));
        continue;
      }
    }

    return h;
  }
}

VarHandlePtr GRPCClient::AsyncSendBatchBarrier(const std::string& ep,
                                               int64_t time_out) {
  const auto ch = GetChannel(ep);

  BatchBarrierProcessor* s = new BatchBarrierProcessor(ch);
  const std::string method = kBatchBarrierRPC;
  VarHandlePtr h(
      new VarHandle(ep, method, BATCH_BARRIER_MESSAGE, nullptr, nullptr));
  s->Prepare(h, time_out);

  sendrecv::VariableMessage req;
  req.set_varname(BATCH_BARRIER_MESSAGE);

  platform::RecordRPCEvent record_event(method);

  auto rpc = s->stub_->AsyncSendVariable(s->context_.get(), req, &cq_);
  rpc->Finish(&s->reply_, &s->status_, reinterpret_cast<void*>(s));
  req_count_++;

  if (UNLIKELY(platform::IsProfileEnabled())) {
    h->Wait();
  }

  return h;
}

VarHandlePtr GRPCClient::AsyncSendFetchBarrier(const std::string& ep,
                                               int64_t time_out) {
  const auto ch = GetChannel(ep);
  FetchBarrierProcessor* s = new FetchBarrierProcessor(ch);
  const std::string method = kFetchBarrierRPC;
  VarHandlePtr h(
      new VarHandle(ep, method, FETCH_BARRIER_MESSAGE, nullptr, nullptr));
  s->Prepare(h, time_out);

  sendrecv::VariableMessage req;
  req.set_varname(FETCH_BARRIER_MESSAGE);

  platform::RecordRPCEvent record_event(method);

  auto rpc = s->stub_->AsyncGetVariable(s->context_.get(), req, &cq_);
  rpc->Finish(&s->reply_, &s->status_, reinterpret_cast<void*>(s));
  req_count_++;

  if (UNLIKELY(platform::IsProfileEnabled())) {
    h->Wait();
  }

  return h;
}

VarHandlePtr GRPCClient::AsyncGetMonomerBarrier(const std::string& ep,
                                                const std::string& var_name,
                                                int64_t time_out) {
  const auto ch = GetChannel(ep);
  BatchBarrierProcessor* s = new BatchBarrierProcessor(ch);
  const std::string method = kSendMonomerFetchBarrierRPC;
  VarHandlePtr h(new VarHandle(ep, method, var_name, nullptr, nullptr));
  s->Prepare(h, time_out);

  VLOG(30) << s->GetVarHandlePtr()->String() << " begin";

  sendrecv::VariableMessage req;
  req.set_varname(var_name);

  platform::RecordRPCEvent record_event(method);

  auto rpc = s->stub_->AsyncGetMonomerBarrier(s->context_.get(), req, &cq_);
  rpc->Finish(&s->reply_, &s->status_, reinterpret_cast<void*>(s));
  req_count_++;

  if (UNLIKELY(platform::IsProfileEnabled())) {
    h->Wait();
  }

  return h;
}

VarHandlePtr GRPCClient::AsyncSendComplete(const std::string& ep,
                                           int64_t time_out) {
  const auto ch = GetChannel(ep);

  BatchBarrierProcessor* s = new BatchBarrierProcessor(ch);
  const std::string method = kSendCompleteRPC;
  VarHandlePtr h(new VarHandle(ep, method, COMPLETE_MESSAGE, nullptr, nullptr));
  s->Prepare(h, time_out);

  sendrecv::VariableMessage req;
  req.set_trainer_id(trainer_id_);
  req.set_varname(COMPLETE_MESSAGE);

  platform::RecordRPCEvent record_event(method);

  auto rpc = s->stub_->AsyncSendVariable(s->context_.get(), req, &cq_);
  rpc->Finish(&s->reply_, &s->status_, reinterpret_cast<void*>(s));
  req_count_++;

  if (UNLIKELY(platform::IsProfileEnabled())) {
    h->Wait();
  }

  return h;
}

VarHandlePtr GRPCClient::AsyncCheckpointNotify(const std::string& ep,
                                               const std::string& dirname,
                                               const std::string& varname,
                                               const int mode,
                                               int64_t time_out) {
  const auto ch = GetChannel(ep);

  CheckpointNotifyProcessor* s = new CheckpointNotifyProcessor(ch);

  const std::string method = kCheckPointNotifyRPC;

  VarHandlePtr h(
      new VarHandle(ep, method, CHECKPOINT_SAVE_MESSAGE, nullptr, nullptr));
  s->Prepare(h, time_out);

  sendrecv::VariableMessage req;
  req.set_varname(varname);
  req.set_table_name(std::to_string(mode));
  req.set_out_varname(dirname);

  platform::RecordRPCEvent record_event(method);

  auto rpc = s->stub_->AsyncCheckpointNotify(s->context_.get(), req, &cq_);
  rpc->Finish(&s->reply_, &s->status_, reinterpret_cast<void*>(s));
  req_count_++;

  if (UNLIKELY(platform::IsProfileEnabled())) {
    h->Wait();
  }

  return h;
}

VarHandlePtr GRPCClient::AsyncDistributeNotify(
    const std::string& ep, const platform::DeviceContext& ctx,
    const framework::Scope& scope, const std::string& var_name,
    int64_t time_out) {
  const platform::DeviceContext* p_ctx = &ctx;
  const std::string ep_val = ep;
  const std::string var_name_val = var_name;
  const framework::Scope* p_scope = &scope;
  const auto ch = GetChannel(ep_val);
  const std::string method = kRequestNotify;

  SendProcessor* s = new SendProcessor(ch);
  VarHandlePtr h(new VarHandle(ep, method, var_name_val, p_ctx, p_scope));
  s->Prepare(h, time_out);

  framework::AsyncIO([var_name_val, p_scope, p_ctx, s, method, h, this] {
    auto* var = p_scope->FindVar(var_name_val);

    ::grpc::ByteBuffer req;
    SerializeToByteBuffer(var_name_val, var, *p_ctx, &req, "", trainer_id_);

    VLOG(3) << s->GetVarHandlePtr()->String() << " begin";

    // stub context
    s->response_call_back_ = nullptr;

    platform::RecordRPCEvent record_event(method);

    auto call = s->stub_g_.PrepareUnaryCall(
        s->context_.get(), "/sendrecv.SendRecvService/DistributeNotify", req,
        &cq_);
    call->StartCall();
    call->Finish(&s->reply_, &s->status_, reinterpret_cast<void*>(s));
  });
  req_count_++;

  if (UNLIKELY(platform::IsProfileEnabled())) {
    h->Wait();
  }

  return h;
}

VarHandlePtr GRPCClient::AsyncSendAndRecv(const std::string& ep,
                                          const platform::DeviceContext& ctx,
                                          const framework::Scope& scope,
                                          const std::string& send_var_name,
                                          const std::string& recv_var_name,
                                          const std::string& table_name,
                                          int64_t time_out) {
  const platform::DeviceContext* p_ctx = &ctx;
  const std::string ep_val = ep;
  const std::string send_var_name_val = send_var_name;
  const std::string recv_var_name_val = recv_var_name;
  const std::string table_name_val = table_name;
  const framework::Scope* p_scope = &scope;
  const auto ch = GetChannel(ep_val);
  const std::string method = kSendAndRecvRPC;
  VLOG(4) << "GRPCClient::SendAndRecv Begin ,Send_var_name: "
          << send_var_name_val << " Recv_var_name: " << recv_var_name_val;
  int retry_times_ = 0;

  while (true) {
    SendAndRecvProcessor* s = new SendAndRecvProcessor(ch);
    VarHandlePtr h(
        new VarHandle(ep, method, send_var_name_val, p_ctx, p_scope));
    VarHandlePtr h_recv(
        new VarHandle(ep, method, recv_var_name_val, p_ctx, p_scope));
    s->Prepare(h, time_out);
    s->RecvPrepare(h_recv);

    framework::AsyncIO([send_var_name_val, recv_var_name_val, table_name_val,
                        p_scope, p_ctx, s, method, h, this] {
      auto* send_var = p_scope->FindVar(send_var_name_val);
      send_var->GetMutable<framework::LoDTensor>()->set_lod({});
      ::grpc::ByteBuffer buf;
      VLOG(4) << "SerializeToByteBuffer: send_var_name_val: "
              << send_var_name_val
              << " recv_var_name_val: " << recv_var_name_val;
      SerializeToByteBuffer(send_var_name_val, send_var, *p_ctx, &buf,
                            recv_var_name_val, trainer_id_, table_name_val);

      VLOG(3) << s->GetVarHandlePtr()->String() << " begin";

      // stub context
      s->response_call_back_ = ProcGetRecvResponse;

      platform::RecordRPCEvent record_event(method);

      auto call = s->stub_g_.PrepareUnaryCall(
          s->context_.get(), "/sendrecv.SendRecvService/SendAndRecvVariable",
          buf, &cq_);
      call->StartCall();
      call->Finish(&s->reply_, &s->status_, reinterpret_cast<void*>(s));

      if (UNLIKELY(platform::IsProfileEnabled())) {
        h->Wait();
      }
    });
    req_count_++;

    if (FLAGS_rpc_retry_times > 0 && retry_times_ < FLAGS_rpc_retry_times) {
      h->Wait();
      if (h->should_retry) {
        VLOG(3) << "rpc call failed, retry times " << retry_times_;
        retry_times_++;
        std::random_device rd;
        std::this_thread::sleep_for(std::chrono::milliseconds(rd() % 5));
        continue;
      }
    }

    return h;
  }
}

bool GRPCClient::Wait() {
  std::unique_lock<std::mutex> lk(sync_mutex_);
  sync_cond_.wait(lk, [this] { return (req_count_ == 0 || ok_ == false); });
  return ok_;
}

inline bool ShouldRetry(const std::string& method, int error_code) {
  if (method == kPrefetchRPC) {
    return true;
  }

  if (error_code == grpc::StatusCode::DEADLINE_EXCEEDED) {
    return true;
  }

  return false;
}

void GRPCClient::Proceed() {
  void* tag = nullptr;
  bool ok = false;

  VLOG(3) << "GRPCClient Proceed begin";
  while (!stopped_ && cq_.Next(&tag, &ok)) {
    BaseProcessor* c = static_cast<BaseProcessor*>(tag);
    GPR_ASSERT(ok);
    PADDLE_ENFORCE_NOT_NULL(
        c, platform::errors::PreconditionNotMet("Make BaseProcessor failed."));

    if (c->status_.ok()) {
      VLOG(3) << c->GetVarHandlePtr()->String() << " process";
      c->Process();
    } else if (ShouldRetry(c->GetVarHandlePtr()->method(),
                           c->status_.error_code())) {
      VLOG(0) << c->GetVarHandlePtr()->String()
              << " meets grpc error, error_code:" << c->status_.error_code()
              << " error_message:" << c->status_.error_message()
              << " error_details:" << c->status_.error_details()
              << " should retry!";
      c->GetVarHandlePtr()->should_retry = true;
      c->Finish(false);
    } else {
      PADDLE_THROW(platform::errors::External(
          "%s meets grpc error, error_code is %d, error message is %s, error "
          "details is %s.",
          c->GetVarHandlePtr()->String(), c->status_.error_code(),
          c->status_.error_message(), c->status_.error_details()));
      c->Finish(false);
    }

    bool notify = false;
    {
      std::lock_guard<std::mutex> lk(sync_mutex_);
      req_count_--;
      notify = (req_count_ <= 0 || !c->status_.ok());
    }

    delete c;

    if (notify) {
      sync_cond_.notify_all();
    }
  }

  // Last log message
  // Avoid using VLOG() and LOG(): in the destructor of google::LogMessage() a
  // static Mutex log_mutex is used for synchronization, which might have been
  // destructed at this moment.
  if (FLAGS_v >= 3) {
    std::string msg("GRPCClient Proceed end");
    fwrite(msg.c_str(), msg.length(), 1, stderr);
  }
}

std::shared_ptr<grpc::Channel> GRPCClient::GetChannel(const std::string& ep) {
  std::lock_guard<std::mutex> guard(chan_mutex_);
  auto it = channels_.find(ep);
  if (it != channels_.end()) {
    return it->second;
  }

  // Channel configurations:
  grpc::ChannelArguments args;
  args.SetInt(GRPC_ARG_MAX_RECONNECT_BACKOFF_MS, 2000);
  if (FLAGS_rpc_disable_reuse_port) {
    args.SetInt(GRPC_ARG_ALLOW_REUSEPORT, 0);
  }
  args.SetCompressionAlgorithm(GRPC_COMPRESS_NONE);
  args.SetMaxSendMessageSize(std::numeric_limits<int>::max());
  args.SetMaxReceiveMessageSize(std::numeric_limits<int>::max());

  auto ch =
      grpc::CreateCustomChannel(ep, grpc::InsecureChannelCredentials(), args);
  channels_[ep] = ch;
  return ch;
}

}  // namespace distributed
}  // namespace operators
}  // namespace paddle
