//  Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <condition_variable>  // NOLINT
#include <memory>
#include <mutex>   // NOLINT
#include <thread>  // NOLINT
#include <utility>

#include "paddle/fluid/framework/var_type_traits.h"
#include "paddle/fluid/platform/enforce.h"

#pragma once

namespace paddle {
namespace framework {

/**
 * Variable class for asynchronized ops.
 **/
class AsyncVariable {
 public:
  AsyncVariable();
  AsyncVariable(const AsyncVariable&) = delete;
  AsyncVariable& operator=(const AsyncVariable&) = delete;

  bool IsAvailable() const { return state_ == EnumState::kAvailable; }

  template <typename T>
  const T& Get() const {
    if (state_ != EnumState::kAvailable) {
      std::unique_lock<std::mutex> lock(mutex_);
      cv_.wait(lock, [this] { return state_ == EnumState::kAvailable; });
    }
    return *static_cast<const T*>(holder_->Ptr());
  }

  template <typename T>
  T* GetMutable() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (state_ != EnumState::kAvailable) {
      state_ = EnumState::kAvailable;
      holder_.reset(new PlaceholderImpl<T>());
    } else {
      PADDLE_ENFORCE_EQ(
          holder_->Type(), VarTypeTrait<T>::kId,
          platform::errors::InvalidArgument(
              "The Variable type must be %s, but the type it holds is %s.",
              ToTypeName(VarTypeTrait<T>::kId), ToTypeName(holder_->Type())));
    }
    cv_.notify_all();
    return static_cast<T*>(holder_->Ptr());
  }

  template <typename T>
  void Emplace(T&& arg) {
    PADDLE_ENFORCE_EQ(
        state_, EnumState::kNotAvailable,
        platform::errors::Unimplemented("Emplace AsyncVariable for multiple "
                                        "times is not implemented now."));
    {
      std::lock_guard<std::mutex> lock(mutex_);
      state_ = EnumState::kAvailable;
      holder_.reset(new PlaceholderImpl<T>(arg));
    }
    cv_.notify_all();
    return;
  }

  template <typename WaiterT>
  void AndThen(WaiterT&& waiter) {
    // TODO(zhhsplendid): Implement it.
  }

  ~AsyncVariable();

  // Enum representing the states for AsyncVaraible
  enum EnumState {
    // Reserve
    kUknown = 0,
    // Data is not available
    kNotAvailable = 1,
    // Data is available
    kAvailable = 2,
  };

 private:
  // class to hold data member for AsyncVariable
  class Placeholder {
   public:
    virtual ~Placeholder() PADDLE_MAY_THROW {}

    inline int Type() const { return type_; }
    inline const void* Ptr() const { return ptr_; }
    inline void* Ptr() { return ptr_; }

   protected:
    inline void Init(void* p, int type) {
      ptr_ = p;
      type_ = type;
    }

    void* ptr_;
    int type_;
  };

  // Placeholder hides type T, so it doesn't appear as a template
  // parameter of Variable.
  template <typename T>
  class PlaceholderImpl : public Placeholder {
   public:
    static_assert(
        IsRegisteredVarType<T>(),
        "Not registered type. Please register T inside var_type_traits.h");
    PlaceholderImpl() { this->Init(&obj_, VarTypeTrait<T>::kId); }
    explicit PlaceholderImpl(const T& obj) : obj_(obj) {
      this->Init(&obj_, VarTypeTrait<T>::kId);
    }

   private:
    T obj_;
  };

  // Holder for data member
  std::shared_ptr<Placeholder> holder_;
  // Enum state of the current AsyncVariable
  EnumState state_;
  // Condition variable used for wait-notify
  mutable std::condition_variable cv_;
  // Mutex used for wait-notify
  mutable std::mutex mutex_;
};

}  // namespace framework
}  // namespace paddle
