// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
#include <mutex>  //NOLINT
#include <utility>

#include "paddle/fluid/framework/variable.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/fluid/platform/macros.h"

namespace paddle {
namespace framework {

/*
 * AsyncVariable is a future type containding a Variable value.
 *
 * Note: Should we design a Base class, and use derive mechanism?
 *
 */
class AsyncVariable {
 public:
  enum EnumState {
    kUnknown = 0,
    kNotAvailable = 1,
    kAvailable = 2,
  };

  AsyncVariable();

  ~AsyncVariable() {}

  bool isAvailable() const { return state_ == EnumState::kAvailable; }

  template <typename T>
  const T& Get() const {
    if (state_ != EnumState::kAvailable) {
      std::unique_lock<std::mutex> lock(mutex_);
      cv_.wait(lock, [this] { return state_ == EnumState::kAvailable; });
    }
    PADDLE_ENFORCE_NOT_NULL(
        inner_var_,
        platform::errors::Unavailable(
            "inner_var_ should not be null while calling Get<T>()."));
    return inner_var_->Get<T>();
  }

  template <typename T>
  T* GetMutable() {
    if (state_ != EnumState::kAvailable) {
      std::unique_lock<std::mutex> lock(mutex_);
      cv_.wait(lock, [this] { return state_ == EnumState::kAvailable; });
    }
    PADDLE_ENFORCE_NOT_NULL(
        inner_var_,
        platform::errors::Unavailable(
            "inner_var_ should not be null while calling GetMutable<T>()."));
    return inner_var_->GetMutable<T>();
  }

  template <typename T, typename... Args>
  void Emplace(Args&&... args) {
    PADDLE_ENFORCE_NE(
        state_, EnumState::kAvailable,
        platform::errors::Unimplemented("Emplace AsyncVariable for multiple "
                                        "times is not implemented now."));

    if (state_ == EnumState::kNotAvailable) {
      std::lock_guard<std::mutex> lock(mutex_);
      inner_var_.reset(new framework::Variable());
      // TODO(Aurelius84): T is truly available only after calling
      // `tensor->mutable_data<T>(args)`.
      state_ = EnumState::kAvailable;
      this->GetMutable<T>();
    }
    // Once inner_var_ is available, we notify all thread.
    cv_.notify_all();
  }

  template <typename WaiterT>
  void AndThen(WaiterT&& waiter);

 private:
  DISABLE_COPY_AND_ASSIGN(AsyncVariable);

  std::shared_ptr<framework::Variable> inner_var_;
  EnumState state_;
  mutable std::condition_variable cv_;
  mutable std::mutex mutex_;
};
}  // namespace framework
}  // namespace paddle
