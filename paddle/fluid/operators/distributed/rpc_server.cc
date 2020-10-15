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

#include "paddle/fluid/operators/distributed/rpc_server.h"

#include <fstream>
#include <string>

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

class RequestHandler;

void RPCServer::ShutDown() {
  VLOG(3) << "RPCServer ShutDown ";
  ShutDownImpl();

  exit_flag_ = true;
  barrier_cond_.notify_all();
  rpc_cond_.notify_all();
}

void RPCServer::SavePort() const {
  auto file_path = string::Sprintf("/tmp/paddle.%d.port", ::getpid());
  std::ofstream port_file;
  port_file.open(file_path);
  port_file << selected_port_;
  port_file.close();
  VLOG(3) << "selected port written to " << file_path;
}

void RPCServer::WaitBarrier(const std::string& rpc_name) {
  VLOG(3) << "WaitBarrier in: " << rpc_name;
  std::unique_lock<std::mutex> lock(this->mutex_);
  barrier_cond_.wait(lock, [this, &rpc_name] {
    return ((barrier_counter_[rpc_name] == client_num_ && client_num_ != 0) ||
            exit_flag_.load());
  });

  VLOG(3) << "WaitBarrier out: " << rpc_name
          << " counter: " << barrier_counter_[rpc_name];
}

void RPCServer::IncreaseBatchBarrier(const std::string rpc_name) {
  VLOG(3) << "RPCServer begin IncreaseBatchBarrier " << rpc_name;
  // barrier msg should make sure that it's in the right cond(send|recv)
  WaitCond(rpc_name);
  int b = 0;
  std::unique_lock<std::mutex> lock(mutex_);
  b = ++barrier_counter_[rpc_name];
  VLOG(3) << rpc_name << " barrier_counter: " << b;
  if (b >= client_num_) {
    lock.unlock();
    VLOG(3) << "BatchBarrier counter reach " << client_num_ << " for "
            << rpc_name;
    barrier_cond_.notify_all();
    lock.lock();
  }
}

void RPCServer::Complete() {
  {
    std::unique_lock<std::mutex> lock(mutex_);
    client_num_--;
    need_reset_all_vars_ = true;

    VLOG(3) << "decrease client_num to: " << client_num_;
    if (cur_cond_.load() == rpc_cond_map_[kRequestGet]) {
      barrier_counter_[kRequestGet]--;
    }
  }
  barrier_cond_.notify_all();
}

bool RPCServer::NeedResetAllVars() {
  std::unique_lock<std::mutex> lock(mutex_);
  return need_reset_all_vars_;
}

int RPCServer::GetClientNum() {
  std::unique_lock<std::mutex> lock(mutex_);
  return client_num_;
}

void RPCServer::ResetBarrierCounter() {
  VLOG(3) << "RPCServer ResetBarrierCounter ";
  std::unique_lock<std::mutex> lock(mutex_);
  for (auto& t : barrier_counter_) {
    t.second = 0;
  }
  need_reset_all_vars_ = false;
}

void RPCServer::RegisterRPC(const std::string& rpc_name,
                            RequestHandler* handler, int thread_num) {
  rpc_call_map_[rpc_name] = handler;
  rpc_thread_num_[rpc_name] = thread_num;

  static int cond = -1;
  rpc_cond_map_[rpc_name] = ++cond;
  VLOG(3) << "RegisterRPC rpc_name: " << rpc_name << ", handler: " << handler
          << ", cond: " << rpc_cond_map_[rpc_name];
}

void RPCServer::SetCond(const std::string& rpc_name) {
  VLOG(3) << "RPCServer SetCond " << rpc_name;
  {
    std::unique_lock<std::mutex> lock(mutex_);
    cur_cond_ = rpc_cond_map_[rpc_name];
  }

  rpc_cond_.notify_all();
}

void RPCServer::WaitCond(const std::string& rpc_name) {
  VLOG(3) << "RPCServer WaitCond in " << rpc_name;
  int cond = 0;
  {
    std::unique_lock<std::mutex> lock(mutex_);
    cond = rpc_cond_map_[rpc_name];
  }

  std::unique_lock<std::mutex> lock(mutex_);
  rpc_cond_.wait(
      lock, [=] { return (cur_cond_.load() == cond || exit_flag_.load()); });
  VLOG(3) << "RPCServer WaitCond out " << rpc_name;
}

void RPCServer::RegisterVar(const std::string& var_name,
                            const std::string& rpc_name,
                            framework::Scope* scope,
                            platform::DeviceContext* dev_ctx) {
  MonomerHandle h;
  h.var_name_ = var_name;
  h.rpc_name_ = rpc_name;
  h.scope_ = scope;
  h.dev_ctx_ = dev_ctx;

  {
    std::unique_lock<std::mutex> lock(mutex_);
    PADDLE_ENFORCE_EQ(
        var_map_.find(var_name), var_map_.end(),
        platform::errors::AlreadyExists("%s already in var_map.", var_name));
    var_map_[var_name] = h;
  }

  rpc_cond_.notify_all();
  VLOG(3) << "RegisterVar context:" << h.String();
}

void RPCServer::IncreaseVarBarrier(const std::string& var_name) {
  int b = 0;
  MonomerHandle h;
  {
    std::unique_lock<std::mutex> lock(mutex_);
    b = ++var_map_[var_name].barrier_;
    h = var_map_[var_name];
  }

  if (b >= client_num_) {
    barrier_cond_.notify_all();
  }

  VLOG(3) << "IncreaseVarBarrier context:" << h.String();
}

void RPCServer::WaitVarBarrier(const std::string& var_name) {
  VLOG(3) << "WaitVarBarrier var_name:" << var_name;

  std::unique_lock<std::mutex> lock(mutex_);
  barrier_cond_.wait(lock, [&]() {
    return ((var_map_[var_name].barrier_ >= client_num_ && client_num_ != 0) ||
            exit_flag_.load());
  });

  VLOG(3) << "WaitVarBarrier context: " << var_map_[var_name].String();
}

void RPCServer::SetVarCond(const std::string& var_name) {
  VLOG(3) << "SetVarCond var_name:" << var_name;
  {
    std::unique_lock<std::mutex> lock(mutex_);
    if (var_map_.find(var_name) != var_map_.end()) {
      rpc_cond_.notify_all();
    }
  }
}

void RPCServer::WaitVarCond(const std::string& var_name) {
  VLOG(3) << "WaitVarCond var_name:" << var_name;

  std::unique_lock<std::mutex> lock(mutex_);
  rpc_cond_.wait(lock, [=] {
    return (var_map_.find(var_name) != var_map_.end() || exit_flag_.load());
  });

  VLOG(3) << "WaitVarCond var_name:" << var_name << " end";
}

MonomerHandle RPCServer::GetMonomer(const std::string& var_name) {
  MonomerHandle h;
  {
    std::unique_lock<std::mutex> lock(mutex_);
    h = var_map_[var_name];
  }

  return h;
}

void RPCServer::ClearRegisteredVars() {
  std::unique_lock<std::mutex> lock(mutex_);
  var_map_.clear();
}

void RPCServer::ClearVar(const std::string& var_name) {
  std::unique_lock<std::mutex> lock(mutex_);
  var_map_.erase(var_name);
}
}  // namespace distributed
}  // namespace operators
}  // namespace paddle
