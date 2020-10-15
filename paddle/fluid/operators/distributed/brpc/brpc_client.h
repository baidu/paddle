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

#include <chrono>  // NOLINT
#include <ctime>
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>  // NOLINT
#include <string>
#include <unordered_map>
#include <vector>

#include "brpc/channel.h"
#include "paddle/fluid/framework/blocking_queue.h"
#include "paddle/fluid/framework/data_type.h"
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/framework/selected_rows.h"
#include "paddle/fluid/operators/distributed/brpc/brpc_sendrecvop_utils.h"
#include "paddle/fluid/operators/distributed/distributed_pb.h"
#include "paddle/fluid/operators/distributed/request_handler.h"
#include "paddle/fluid/operators/distributed/rpc_client.h"
#include "paddle/fluid/platform/macros.h"  // for DISABLE_COPY_AND_ASSIGN

namespace paddle {
namespace operators {
namespace distributed {

struct ChannelContext {
  brpc::Channel channel;
  std::shared_ptr<sendrecv::SendRecvService_Stub> stub;
};

typedef std::shared_ptr<ChannelContext> ChannelContextPtr;
typedef std::shared_ptr<framework::BlockingQueue<ChannelContextPtr>>
    ChannelQueuePtr;

class BRPCClient : public RPCClient {
 public:
  BRPCClient() {}
  virtual ~BRPCClient();

  VarHandlePtr AsyncSendVar(const std::string& ep,
                            const platform::DeviceContext& ctx,
                            const framework::Scope& scope,
                            const std::string& var_name,
                            int64_t time_out = FLAGS_rpc_deadline) override;

  VarHandlePtr AsyncGetVar(const std::string& ep,
                           const platform::DeviceContext& ctx,
                           const framework::Scope& scope,
                           const std::string& var_name,
                           const std::string& out_var_name,
                           const std::string& table_name = "",
                           int64_t time_out = FLAGS_rpc_deadline) override;

  VarHandlePtr AsyncGetMonomerBarrier(
      const std::string& ep, const std::string& var_name,
      int64_t time_out = FLAGS_rpc_deadline) override;

  VarHandlePtr AsyncGetMonomerVariable(
      const std::string& ep, const platform::DeviceContext& ctx,
      const framework::Scope& scope, const std::string& var_name,
      int64_t time_out = FLAGS_rpc_deadline) override;

  VarHandlePtr AsyncGetVarNoBarrier(const std::string& ep,
                                    const platform::DeviceContext& ctx,
                                    const framework::Scope& scope,
                                    const std::string& var_name,
                                    const std::string& out_varname,
                                    int64_t time_out = FLAGS_rpc_deadline);

  VarHandlePtr AsyncPrefetchVar(const std::string& ep,
                                const platform::DeviceContext& ctx,
                                const framework::Scope& scope,
                                const std::string& in_var_name,
                                const std::string& out_var_name,
                                const std::string& table_name = "",
                                int64_t time_out = FLAGS_rpc_deadline) override;

  VarHandlePtr AsyncSendBatchBarrier(
      const std::string& ep, int64_t time_out = FLAGS_rpc_deadline) override;

  VarHandlePtr AsyncSendFetchBarrier(
      const std::string& ep, int64_t time_out = FLAGS_rpc_deadline) override;

  VarHandlePtr AsyncCheckpointNotify(
      const std::string& ep, const std::string& dirname,
      const std::string& varname, const int mode,
      int64_t time_out = FLAGS_rpc_deadline) override;

  bool Wait() override;

  void SendComplete() override;

 private:
  VarHandlePtr _AsyncGetVar(
      const std::string& ep, const platform::DeviceContext& ctx,
      const framework::Scope& scope, const std::string& var_name,
      const std::string& out_var_name, const std::string& method_name,
      const std::string& table_name, int64_t time_out = FLAGS_rpc_deadline);

  void Proceed();
  ChannelQueuePtr GetChannel(const std::string& ep);

  VarHandlePtr AsyncSendComplete(const std::string& ep,
                                 int64_t time_out = FLAGS_rpc_deadline);

  VarHandlePtr AsyncSendMessage(const std::string& ep,
                                const std::string& method_name,
                                const std::string& message, int64_t time_out);

  VarHandlePtr AsyncSendVarMessage(const std::string& ep,
                                   const std::string& method_name,
                                   const sendrecv::VariableMessage& req,
                                   int64_t time_out);

  friend void HandleSendResponse(brpc::Controller* cntl,
                                 sendrecv::VoidMessage* response,
                                 VarHandlePtr var_h, ChannelQueuePtr ch_ptr,
                                 ChannelContextPtr ch_ctx, BRPCClient* cls);

  friend void HandleGetResponse(brpc::Controller* cntl,
                                sendrecv::VariableMessage* response,
                                VarHandlePtr var_h, ChannelQueuePtr ch_ptr,
                                ChannelContextPtr ch_ctx, BRPCClient* cls);

  friend void HandleFetchBarrierResponse(brpc::Controller* cntl,
                                         sendrecv::VariableMessage* response,
                                         VarHandlePtr var_h,
                                         ChannelQueuePtr ch_ptr,
                                         ChannelContextPtr ch_ctx,
                                         BRPCClient* cls);
  void DecreaseReqCount() {
    if (--req_count_ <= 0) {
      sync_cond_.notify_all();
    }
  }

 private:
  std::unordered_map<std::string, ChannelQueuePtr> channels_;

  // mutex for Wait client sync
  std::mutex sync_mutex_;
  std::condition_variable sync_cond_;
  std::atomic<int64_t> req_count_{0};

  static constexpr int brpc_channel_num_per_server_ = 4;

  // mutex for GetChannel thread safety
  std::mutex chan_mutex_;
  DISABLE_COPY_AND_ASSIGN(BRPCClient);
};

}  // namespace distributed
}  // namespace operators
}  // namespace paddle
