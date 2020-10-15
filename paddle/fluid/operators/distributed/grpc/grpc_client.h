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

#pragma once

#include <time.h>
#include <atomic>
#include <chrono>              // NOLINT
#include <condition_variable>  // NOLINT
#include <ctime>
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>  // NOLINT
#include <string>
#include <thread>  // NOLINT
#include <unordered_map>
#include <vector>

#include "grpc++/channel.h"
#include "grpc++/generic/generic_stub.h"
#include "grpc++/grpc++.h"
#include "grpc++/support/byte_buffer.h"
#include "grpc++/support/slice.h"
#include "grpc/support/log.h"
#include "paddle/fluid/framework/blocking_queue.h"
#include "paddle/fluid/framework/data_type.h"
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/framework/selected_rows.h"
#include "paddle/fluid/operators/distributed/distributed_pb.h"
#include "paddle/fluid/operators/distributed/request_handler.h"
#include "paddle/fluid/operators/distributed/rpc_client.h"
#include "paddle/fluid/operators/distributed/sendrecvop_utils.h"
#include "paddle/fluid/platform/macros.h"  // for DISABLE_COPY_AND_ASSIGN

namespace grpc {
class Channel;
}  // namespace grpc
namespace paddle {
namespace framework {
class Scope;
}  // namespace framework
namespace platform {
class DeviceContext;
}  // namespace platform
}  // namespace paddle

namespace paddle {
namespace operators {
namespace distributed {

void ProcGetResponse(const VarHandle& var_h, const grpc::ByteBuffer& msg);

void ProcGetRecvResponse(const VarHandle& var_h, const grpc::ByteBuffer& msg);

class BaseProcessor {
 public:
  BaseProcessor() { context_ = nullptr; }

  virtual ~BaseProcessor() {}

  virtual void Prepare(VarHandlePtr h, int64_t time_out) {
    var_h_ = h;

    context_.reset(new grpc::ClientContext());
    context_->set_wait_for_ready(true);
    if (time_out) {
      std::chrono::system_clock::time_point deadline =
          std::chrono::system_clock::now() +
          std::chrono::milliseconds(time_out);
      context_->set_deadline(deadline);
    }
  }

  void Process() {
    ProcessImpl();
    var_h_->Finish(true);
  }

  VarHandlePtr GetVarHandlePtr() { return var_h_; }
  bool Wait() { return var_h_->Wait(); }
  void Finish(bool ok) { return var_h_->Finish(ok); }
  virtual void ProcessImpl() = 0;

  std::unique_ptr<grpc::ClientContext> context_;
  grpc::Status status_;

 protected:
  VarHandlePtr var_h_;
};

typedef std::function<void(const VarHandle&, const ::grpc::ByteBuffer&)>
    RequestSendCallBack;

class SendProcessor : public BaseProcessor {
 public:
  explicit SendProcessor(std::shared_ptr<grpc::Channel> ch)
      : BaseProcessor(), stub_g_(ch) {}

  virtual ~SendProcessor() {}

  void ProcessImpl() override {
    if (response_call_back_) {
      response_call_back_(*var_h_.get(), reply_);
    }
  }

  ::grpc::GenericStub stub_g_;
  ::grpc::ByteBuffer reply_;
  RequestSendCallBack response_call_back_ = nullptr;
};

typedef std::function<void(const VarHandle&, const ::grpc::ByteBuffer&)>
    RequestGetCallBack;

class GetProcessor : public BaseProcessor {
 public:
  explicit GetProcessor(std::shared_ptr<grpc::Channel> ch)
      : BaseProcessor(), stub_g_(ch) {}

  virtual ~GetProcessor() {}

  void ProcessImpl() override {
    if (response_call_back_) {
      response_call_back_(*var_h_.get(), reply_);
    }
  }

  ::grpc::ByteBuffer reply_;
  ::grpc::GenericStub stub_g_;
  RequestGetCallBack response_call_back_ = ProcGetResponse;
};

class SendAndRecvProcessor : public BaseProcessor {
 public:
  explicit SendAndRecvProcessor(std::shared_ptr<grpc::Channel> ch)
      : BaseProcessor(), stub_g_(ch) {}

  virtual ~SendAndRecvProcessor() {}

  void ProcessImpl() override {
    if (response_call_back_) {
      response_call_back_(*var_h_recv_.get(), reply_);
      var_h_recv_->Finish(true);
    }
  }

  void RecvPrepare(VarHandlePtr h_recv) { var_h_recv_ = h_recv; }

  ::grpc::ByteBuffer reply_;
  ::grpc::GenericStub stub_g_;
  RequestGetCallBack response_call_back_ = ProcGetResponse;
  VarHandlePtr var_h_recv_;
};

class BatchBarrierProcessor : public BaseProcessor {
 public:
  explicit BatchBarrierProcessor(std::shared_ptr<grpc::Channel> ch)
      : BaseProcessor() {
    stub_ = sendrecv::SendRecvService::NewStub(ch);
  }

  virtual ~BatchBarrierProcessor() {}

  void ProcessImpl() override {}
  sendrecv::VoidMessage reply_;
  std::unique_ptr<sendrecv::SendRecvService::Stub> stub_;
};

class FetchBarrierProcessor : public BaseProcessor {
 public:
  explicit FetchBarrierProcessor(std::shared_ptr<grpc::Channel> ch)
      : BaseProcessor() {
    stub_ = sendrecv::SendRecvService::NewStub(ch);
  }

  virtual ~FetchBarrierProcessor() {}

  void ProcessImpl() override {}
  sendrecv::VariableMessage reply_;
  std::unique_ptr<sendrecv::SendRecvService::Stub> stub_;
};

class CheckpointNotifyProcessor : public BaseProcessor {
 public:
  explicit CheckpointNotifyProcessor(std::shared_ptr<grpc::Channel> ch)
      : BaseProcessor() {
    stub_ = sendrecv::SendRecvService::NewStub(ch);
  }

  virtual ~CheckpointNotifyProcessor() {}

  void ProcessImpl() override {}
  sendrecv::VoidMessage reply_;
  std::unique_ptr<sendrecv::SendRecvService::Stub> stub_;
};

class GRPCClient : public RPCClient {
 public:
  GRPCClient() : ok_(true), completed_(false), stopped_(false) {}
  virtual ~GRPCClient();

  VarHandlePtr AsyncSendVar(const std::string& ep,
                            const platform::DeviceContext& ctx,
                            const framework::Scope& scope,
                            const std::string& var_name,
                            int64_t time_out = FLAGS_rpc_deadline) override;

  VarHandlePtr AsyncGetVar(const std::string& ep,
                           const platform::DeviceContext& ctx,
                           const framework::Scope& scope,
                           const std::string& var_name,
                           const std::string& out_varname,
                           const std::string& table_name = "",
                           int64_t time_out = FLAGS_rpc_deadline) override;

  VarHandlePtr AsyncGetVarNoBarrier(
      const std::string& ep, const platform::DeviceContext& ctx,
      const framework::Scope& scope, const std::string& var_name,
      const std::string& out_varname,
      int64_t time_out = FLAGS_rpc_deadline) override;

  VarHandlePtr AsyncGetMonomerVariable(
      const std::string& ep, const platform::DeviceContext& ctx,
      const framework::Scope& scope, const std::string& var_name,
      int64_t time_out = FLAGS_rpc_deadline) override;

  VarHandlePtr AsyncPrefetchVar(const std::string& ep,
                                const platform::DeviceContext& ctx,
                                const framework::Scope& scope,
                                const std::string& in_var_name,
                                const std::string& out_var_name,
                                const std::string& table_name = "",
                                int64_t time_out = FLAGS_rpc_deadline) override;

  VarHandlePtr AsyncSendBatchBarrier(
      const std::string& ep, int64_t time_out = FLAGS_rpc_deadline) override;

  VarHandlePtr AsyncSendFetchBarrier(const std::string& ep,
                                     int64_t time_out) override;

  VarHandlePtr AsyncGetMonomerBarrier(
      const std::string& ep, const std::string& var_name,
      int64_t time_out = FLAGS_rpc_deadline) override;

  VarHandlePtr AsyncCheckpointNotify(
      const std::string& ep, const std::string& dirname,
      const std::string& varname, const int mode,
      int64_t time_out = FLAGS_rpc_deadline) override;

  VarHandlePtr AsyncDistributeNotify(
      const std::string& ep, const platform::DeviceContext& ctx,
      const framework::Scope& scope, const std::string& var_name,
      int64_t time_out = FLAGS_rpc_deadline) override;

  VarHandlePtr AsyncSendAndRecv(const std::string& ep,
                                const platform::DeviceContext& ctx,
                                const framework::Scope& scope,
                                const std::string& send_var_name,
                                const std::string& recv_var_name,
                                const std::string& table_name = "",
                                int64_t time_out = FLAGS_rpc_deadline) override;

  VarHandlePtr AsyncSendComplete(
      const std::string& ep, int64_t time_out = FLAGS_rpc_deadline) override;

  bool Wait() override;

  void SendComplete() override;

  void InitImpl() override;

 private:
  void Proceed();

  std::shared_ptr<grpc::Channel> GetChannel(const std::string& ep);
  VarHandlePtr _AsyncGetVar(
      const std::string& ep, const platform::DeviceContext& ctx,
      const framework::Scope& scope, const std::string& method,
      const std::string& var_name, const std::string& out_varname,
      const std::string& rpc_path, const std::string& table_name = "",
      int64_t time_out = FLAGS_rpc_deadline);

 private:
  grpc::CompletionQueue cq_;
  std::unordered_map<std::string, std::shared_ptr<grpc::Channel>> channels_;
  std::unique_ptr<std::thread> client_thread_{nullptr};

  // mutex for Wait client sync
  std::mutex sync_mutex_;
  std::condition_variable sync_cond_;
  std::atomic<int64_t> req_count_{0};
  bool ok_;

  // mutex for GetChannel thread safety
  std::mutex chan_mutex_;
  DISABLE_COPY_AND_ASSIGN(GRPCClient);

  // mutex for sending complete message only once
  std::mutex completed_mutex_;
  bool completed_;

  volatile bool stopped_;
};

}  // namespace distributed
}  // namespace operators
}  // namespace paddle
