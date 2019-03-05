/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.

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

#include <deque>
#include <memory>
#include <string>
#include <vector>

#include <ThreadPool.h>

#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/framework/variable.h"
#include "paddle/fluid/operators/distributed/rpc_common.h"
#include "paddle/fluid/platform/device_context.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/fluid/platform/place.h"

namespace paddle {
namespace operators {
namespace distributed {

using Scope = framework::Scope;
using Variable = framework::Variable;

template <typename T>
class BlockingQueue {
 public:
  explicit BlockingQueue(size_t capacity) : capacity_(capacity) {
    PADDLE_ENFORCE_GT(capacity_, 0, "The capacity must be greater than 0.");
  }

  bool Push(const T& elem) {
    std::unique_lock<std::mutex> lock(mutex_);
    send_cv_.wait(lock, [&] { return queue_.size() < capacity_; });
    PADDLE_ENFORCE_LT(queue_.size(), capacity_);
    queue_.push_back(elem);
    recv_cv_.notify_one();
    return true;
  }

  bool Push(T&& elem) {
    std::unique_lock<std::mutex> lock(mutex_);
    send_cv_.wait(lock, [&] { return queue_.size() < capacity_; });
    PADDLE_ENFORCE_LT(queue_.size(), capacity_);
    queue_.emplace_back(std::move(elem));
    recv_cv_.notify_one();
    return true;
  }

  T Pop() {
    std::unique_lock<std::mutex> lock(mutex_);
    recv_cv_.wait(lock, [=] { return !queue_.empty(); });
    T rc(std::move(queue_.front()));
    queue_.pop_front();
    return rc;
  }

  size_t Cap() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return capacity_;
  }

  size_t Size() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return queue_.size();
  }

 private:
  const size_t capacity_;
  std::deque<T> queue_;

  mutable std::mutex mutex_;
  std::condition_variable recv_cv_;
  std::condition_variable send_cv_;
};

using RpcCtxMap = std::unordered_map<std::string, RpcContext>;

class Communicator {
 public:
  Communicator(const RpcCtxMap& send_varname_to_ctx,
               const RpcCtxMap& recv_varname_to_ctx, Scope* recv_scope)
      : send_varname_to_ctx_(send_varname_to_ctx),
        recv_varname_to_ctx_(recv_varname_to_ctx),
        recv_scope_(recv_scope) {
    // get all send information from graph, build vars_to_send
    send_scope_.reset(new Scope());
    for (auto& iter : send_varname_to_ctx_) {
      send_varname_to_queue_[iter.first] =
          std::make_shared<BlockingQueue<std::shared_ptr<Variable>>>(10);
    }
    // TODO(qiao): default 5, need to config
    send_threadpool_.reset(new ::ThreadPool(5));
    recv_threadpool_.reset(new ::ThreadPool(5));
  }

  ~Communicator() {
    VLOG(3) << "~Communicator";
    running_ = false;
    send_thread_->join();
    recv_thread_->join();
    VLOG(3) << "~Communicator done";
  }

  void Start();

  // send grad
  void Send(const std::string& var_name, const framework::Scope& scope);

 private:
  void SendThread();
  void RecvThread();

  bool running_ = false;
  std::unordered_map<std::string,
                     std::shared_ptr<BlockingQueue<std::shared_ptr<Variable>>>>
      send_varname_to_queue_;
  RpcCtxMap send_varname_to_ctx_;
  RpcCtxMap recv_varname_to_ctx_;
  std::unique_ptr<std::thread> send_thread_;
  std::unique_ptr<std::thread> recv_thread_;
  Scope* recv_scope_;                  // should be global scope
  std::unique_ptr<Scope> send_scope_;  // an independent scope
  std::unique_ptr<::ThreadPool> send_threadpool_{nullptr};
  std::unique_ptr<::ThreadPool> recv_threadpool_{nullptr};

  // the following code is for initialize the commnunicator
 public:
  static void Init(const RpcCtxMap& send_varname_to_ctx,
                   const RpcCtxMap& recv_varname_to_ctx, Scope* recv_scope) {
    InitImpl(send_varname_to_ctx, recv_varname_to_ctx, recv_scope);
  }

  static Communicator* GetInstance() { return communicator_.get(); }

 private:
  // Init is called by GetInstance.
  static void InitImpl(const RpcCtxMap& send_varname_to_ctx,
                       const RpcCtxMap& recv_varname_to_ctx,
                       Scope* recv_scope) {
    if (communicator_ == nullptr) {
      communicator_.reset(new Communicator(send_varname_to_ctx,
                                           recv_varname_to_ctx, recv_scope));
    }
  }

 private:
  static std::once_flag init_flag_;
  static std::unique_ptr<Communicator> communicator_;
};

}  // namespace distributed
}  // namespace operators
}  // namespace paddle
