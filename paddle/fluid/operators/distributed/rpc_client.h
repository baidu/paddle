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

#pragma once

#include <condition_variable>  // NOLINT
#include <memory>
#include <string>

#include "gflags/gflags.h"
#include "paddle/fluid/framework/data_type.h"
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/operators/distributed/request_handler.h"

namespace paddle {
namespace framework {
class Scope;
}  // namespace framework
namespace platform {
class DeviceContext;
}  // namespace platform
}  // namespace paddle

DECLARE_int32(rpc_deadline);
DECLARE_int32(rpc_retry_times);

namespace paddle {
namespace operators {
namespace distributed {

class RPCClient {
 public:
  RPCClient() {}
  virtual ~RPCClient() {}
  virtual VarHandlePtr AsyncSendVar(const std::string& ep,
                                    const platform::DeviceContext& ctx,
                                    const framework::Scope& scope,
                                    const std::string& var_name,
                                    int64_t time_out = FLAGS_rpc_deadline) = 0;

  virtual VarHandlePtr AsyncGetVar(const std::string& ep,
                                   const platform::DeviceContext& ctx,
                                   const framework::Scope& scope,
                                   const std::string& var_name,
                                   const std::string& out_varname,
                                   const std::string& table_name = "",
                                   int64_t time_out = FLAGS_rpc_deadline) = 0;

  virtual VarHandlePtr AsyncGetVarNoBarrier(
      const std::string& ep, const platform::DeviceContext& ctx,
      const framework::Scope& scope, const std::string& var_name,
      const std::string& out_varname,
      int64_t time_out = FLAGS_rpc_deadline) = 0;

  virtual VarHandlePtr AsyncGetMonomerVariable(
      const std::string& ep, const platform::DeviceContext& ctx,
      const framework::Scope& scope, const std::string& var_name,
      int64_t time_out = FLAGS_rpc_deadline) = 0;

  virtual VarHandlePtr AsyncPrefetchVar(
      const std::string& ep, const platform::DeviceContext& ctx,
      const framework::Scope& scope, const std::string& in_var_name,
      const std::string& out_var_name, const std::string& table_name = "",
      int64_t time_out = FLAGS_rpc_deadline) = 0;

  virtual VarHandlePtr AsyncSendBatchBarrier(
      const std::string& ep, int64_t time_out = FLAGS_rpc_deadline) = 0;

  virtual VarHandlePtr AsyncSendFetchBarrier(
      const std::string& ep, int64_t time_out = FLAGS_rpc_deadline) = 0;

  virtual VarHandlePtr AsyncGetMonomerBarrier(
      const std::string& ep, const std::string& var_name,
      int64_t time_out = FLAGS_rpc_deadline) = 0;

  virtual VarHandlePtr AsyncCheckpointNotify(
      const std::string& ep, const std::string& dirname,
      const std::string& varname, const int mode,
      int64_t time_out = FLAGS_rpc_deadline) = 0;

  virtual VarHandlePtr AsyncDistributeNotify(
      const std::string& ep, const platform::DeviceContext& ctx,
      const framework::Scope& scope, const std::string& var_name,
      int64_t time_out = FLAGS_rpc_deadline) = 0;

  virtual VarHandlePtr AsyncSendAndRecv(
      const std::string& ep, const platform::DeviceContext& ctx,
      const framework::Scope& scope, const std::string& send_var_name,
      const std::string& recv_var_name, const std::string& table_name = "",
      int64_t time_out = FLAGS_rpc_deadline) = 0;

  virtual VarHandlePtr AsyncSendComplete(
      const std::string& ep, int64_t time_out = FLAGS_rpc_deadline) = 0;

  // Complete tells all the pserver instances that finishe the training,
  // the pserver can reduce it's barrier count, and continue to train
  // with other trainers.
  virtual void SendComplete() = 0;

  virtual bool Wait() = 0;

  template <typename T>
  static RPCClient* GetInstance(int trainer_id) {
    std::call_once(init_flag_, &RPCClient::Init<T>, trainer_id);
    return rpc_client_.get();
  }

  // Init is called by GetInstance.
  template <typename T>
  static void Init(int trainer_id) {
    VLOG(1) << "init rpc client with trainer_id " << trainer_id;
    trainer_id_ = trainer_id;
    if (rpc_client_.get() == nullptr) {
      rpc_client_.reset(new T());
      rpc_client_->InitImpl();
    }
  }

  virtual void InitImpl() {}

 protected:
  // each trainer have exact one trainer id, it should be static
  static int trainer_id_;

 private:
  static std::once_flag init_flag_;
  static std::unique_ptr<RPCClient> rpc_client_;
};
}  // namespace distributed
}  // namespace operators
}  // namespace paddle
