//   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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
#include <deque>
#include <utility>

#include "paddle/fluid/platform/enforce.h"

#include "paddle/fluid/platform/timer.h"

namespace paddle {
namespace operators {
namespace reader {

template <typename T>
class BlockingQueue {
  // BlockingQueue is for buffered reading and is supposed to use only the
  // reader package. It is true that we could and we should have been using
  // framework::Channel, but which has currently a deadlock bug. BlockingQueue
  // is a workaround and a simplified version of framework::Channel as it
  // doesn't support GPU and it implements on buffered blocking queue.
 public:
  explicit BlockingQueue(size_t capacity, bool speed_test_mode = false)
      : capacity_(capacity), speed_test_mode_(speed_test_mode) {
    PADDLE_ENFORCE_GT(capacity_, static_cast<size_t>(0),
                      platform::errors::InvalidArgument(
                          "The capacity of a reader::BlockingQueue must be "
                          "greater than 0, but received capacity is %d.",
                          capacity_));
  }

  bool Send(const T& elem) {
    platform::Timer timer;
    timer.Start();
    std::unique_lock<std::mutex> lock(mutex_);
    send_cv_.wait(
        lock, [&] { return queue_.size() < capacity_ || closed_ || killed_; });
    timer.Pause();
    VLOG(0) << "BlockingQueue: Send: lock time: " << timer.ElapsedSec() << " s";
    EnforceNotKilled();
    if (closed_) {
      VLOG(5)
          << "WARNING: Sending an element to a closed reader::BlokcingQueue.";
      return false;
    }
    PADDLE_ENFORCE_LT(
        queue_.size(), capacity_,
        platform::errors::PermissionDenied(
            "The queue size cannot exceed the set queue capacity."));
    VLOG(0) << "BlockingQueue: Send an elem";
    queue_.push_back(elem);
    receive_cv_.notify_one();
    return true;
  }

  bool Send(T&& elem) {
    std::unique_lock<std::mutex> lock(mutex_);
    send_cv_.wait(
        lock, [&] { return queue_.size() < capacity_ || closed_ || killed_; });
    EnforceNotKilled();
    if (closed_) {
      VLOG(5)
          << "WARNING: Sending an element to a closed reader::BlokcingQueue.";
      return false;
    }
    PADDLE_ENFORCE_LT(
        queue_.size(), capacity_,
        platform::errors::PermissionDenied(
            "The queue size cannot exceed the set queue capacity."));
    VLOG(0) << "BlockingQueue: Send an elem by move";
    queue_.emplace_back(std::move(elem));
    receive_cv_.notify_one();
    return true;
  }

  bool Receive(T* elem) {
    platform::Timer timer;
    timer.Start();
    std::unique_lock<std::mutex> lock(mutex_);
    receive_cv_.wait(lock,
                     [&] { return !queue_.empty() || closed_ || killed_; });
    EnforceNotKilled();
    timer.Pause();
    VLOG(0) << "BlockingQueue: Receive: lock time: " << timer.ElapsedSec()
            << " s";
    if (!queue_.empty()) {
      timer.Start();
      PADDLE_ENFORCE_NOT_NULL(
          elem, platform::errors::InvalidArgument(
                    "The holder to receive queue data is null pointer."));
      *elem = queue_.front();
      timer.Pause();
      VLOG(0) << "BlockingQueue: Receive: read time: " << timer.ElapsedSec()
              << " s";
      VLOG(0) << "BlockingQueue: queue_.front data ptr: "
              << reinterpret_cast<uintptr_t>(&(queue_.front()));
      VLOG(0) << "BlockingQueue: elem data ptr: "
              << reinterpret_cast<uintptr_t>(elem);
      timer.Start();
      if (LIKELY(!speed_test_mode_)) {
        queue_.pop_front();
      }
      timer.Pause();
      VLOG(0) << "BlockingQueue: Receive: pop time: " << timer.ElapsedSec()
              << " s";
      timer.Start();
      send_cv_.notify_one();
      timer.Pause();
      VLOG(0) << "BlockingQueue: Receive: notify time: " << timer.ElapsedSec()
              << " s";
      return true;
    } else {
      PADDLE_ENFORCE_EQ(closed_, true,
                        platform::errors::PermissionDenied(
                            "Blocking queue status error, if queue is empty "
                            "when pop data, it should be closed."));
      VLOG(3) << "queue is closed! return nothing.";
      return false;
    }
  }

  T Receive() {
    platform::Timer timer;
    timer.Start();
    std::unique_lock<std::mutex> lock(mutex_);
    receive_cv_.wait(lock,
                     [&] { return !queue_.empty() || closed_ || killed_; });
    EnforceNotKilled();
    timer.Pause();
    VLOG(0) << "BlockingQueue: Receive: lock time: " << timer.ElapsedSec()
            << " s";

    if (!queue_.empty()) {
      T elem;
      timer.Start();
      elem = queue_.front();
      timer.Pause();
      VLOG(0) << "BlockingQueue: Receive: read time: " << timer.ElapsedSec()
              << " s";
      VLOG(0) << "BlockingQueue: queue_.front data ptr: "
              << reinterpret_cast<uintptr_t>(&(queue_.front()));
      VLOG(0) << "BlockingQueue: elem data ptr: "
              << reinterpret_cast<uintptr_t>(&elem);
      timer.Start();
      if (LIKELY(!speed_test_mode_)) {
        queue_.pop_front();
      }
      timer.Pause();
      VLOG(0) << "BlockingQueue: Receive: pop time: " << timer.ElapsedSec()
              << " s";
      timer.Start();
      send_cv_.notify_one();
      timer.Pause();
      VLOG(0) << "BlockingQueue: Receive: notify time: " << timer.ElapsedSec()
              << " s";
      return elem;
    } else {
      PADDLE_ENFORCE_EQ(closed_, true,
                        platform::errors::PermissionDenied(
                            "Blocking queue status error, if queue is empty "
                            "when pop data, it should be closed."));
      VLOG(3) << "queue is closed! return nothing.";
      return nullptr;
    }
  }

  void ReOpen() {
    std::lock_guard<std::mutex> lock(mutex_);
    EnforceNotKilled();
    VLOG(1) << "reopen queue";
    closed_ = false;
    std::deque<T> new_deque;
    queue_.swap(new_deque);
    send_cv_.notify_all();
    receive_cv_.notify_all();
  }

  void Close() {
    std::lock_guard<std::mutex> lock(mutex_);
    VLOG(1) << "close queue";
    closed_ = true;
    send_cv_.notify_all();
    receive_cv_.notify_all();
  }

  bool IsClosed() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return closed_;
  }

  size_t Cap() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return capacity_;
  }

  size_t Size() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return queue_.size();
  }

  void Kill() {
    std::lock_guard<std::mutex> lock(mutex_);
    VLOG(1) << "kill queue";
    closed_ = true;
    killed_ = true;
    send_cv_.notify_all();
    receive_cv_.notify_all();
  }

 private:
  inline void EnforceNotKilled() {
    PADDLE_ENFORCE_NE(killed_, true, platform::errors::Fatal(
                                         "Blocking queue is killed because the "
                                         "data reader raises an exception."));
  }

 private:
  size_t capacity_;
  bool speed_test_mode_;
  bool closed_{false};
  bool killed_{false};  // the queue is broken since exception raises
  std::deque<T> queue_;

  mutable std::mutex mutex_;
  mutable std::condition_variable receive_cv_;
  mutable std::condition_variable send_cv_;
};
}  // namespace reader
}  // namespace operators
}  // namespace paddle
