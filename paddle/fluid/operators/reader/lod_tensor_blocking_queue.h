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

#include <memory>
#include <utility>
#include <vector>

#include "paddle/fluid/framework/ddim.h"
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/operators/reader/blocking_queue.h"
#include "paddle/fluid/platform/place.h"

namespace paddle {
namespace operators {
namespace reader {

class LoDTensorBlockingQueue {
 public:
  explicit LoDTensorBlockingQueue(size_t capacity, bool speed_test_mode = false)
      : queue_(capacity, speed_test_mode) {}

  ~LoDTensorBlockingQueue() { VLOG(10) << "Destruct LoDTensorBlockingQueue"; }

  bool Push(const std::vector<framework::LoDTensor>& lod_tensor_vec) {
    return queue_.Send(lod_tensor_vec);
  }

  bool Push(std::vector<framework::LoDTensor>&& lod_tensor_vec) {
    return queue_.Send(std::move(lod_tensor_vec));
  }

  std::vector<framework::LoDTensor> Pop(bool* ok = nullptr) {
    std::vector<framework::LoDTensor> lod_tensor_vec;
    bool success = queue_.Receive(&lod_tensor_vec);
    VLOG(0) << "LodTensorBlockingQueue: lod_tensor_vec data ptr: "
            << reinterpret_cast<uintptr_t>(&lod_tensor_vec);
    if (ok != nullptr) *ok = success;
    return lod_tensor_vec;
  }

  void Pop(std::vector<framework::LoDTensor>* out, bool* ok = nullptr) {
    bool success = queue_.Receive(out);
    VLOG(0) << "LodTensorBlockingQueue: new pop data ptr: "
            << reinterpret_cast<uintptr_t>(out);
    if (ok != nullptr) *ok = success;
  }

  inline size_t Cap() const { return queue_.Cap(); }

  inline size_t Size() const { return queue_.Size(); }

  inline void ReOpen() { queue_.ReOpen(); }

  inline void Close() {
    VLOG(1) << "LoDTensorBlockingQueue close";
    queue_.Close();
  }

  inline bool IsClosed() const { return queue_.IsClosed(); }

  inline void Kill() { queue_.Kill(); }

  inline bool WaitForInited(size_t) { return true; }

 private:
  BlockingQueue<std::vector<framework::LoDTensor>> queue_;
};

class SharedLoDTensorBlockingQueue {
 public:
  explicit SharedLoDTensorBlockingQueue(size_t capacity,
                                        bool speed_test_mode = false)
      : queue_(capacity, speed_test_mode) {}

  ~SharedLoDTensorBlockingQueue() {
    VLOG(10) << "Destruct SharedLoDTensorBlockingQueue";
  }

  bool Push(const std::shared_ptr<std::vector<framework::LoDTensor>>&
                lod_tensor_vec_ptr) {
    VLOG(0) << "SharedLoDTensorBlockingQueue: push a sample.";
    PADDLE_ENFORCE_NOT_NULL(
        lod_tensor_vec_ptr,
        platform::errors::InvalidArgument("SharedLoDTensorBlockingQueue: Push: "
                                          "lod_tensor_vec_ptr is nullptr."));
    return queue_.Send(lod_tensor_vec_ptr);
  }

  std::shared_ptr<std::vector<framework::LoDTensor>> Pop() {
    auto out = queue_.Receive();
    VLOG(0) << "SharedLoDTensorBlockingQueue: new pop data ptr: "
            << reinterpret_cast<uintptr_t>(&out);
    return out;
  }

  inline size_t Cap() const { return queue_.Cap(); }

  inline size_t Size() const { return queue_.Size(); }

  inline void ReOpen() { queue_.ReOpen(); }

  inline void Close() {
    VLOG(1) << "SharedLoDTensorBlockingQueue close";
    queue_.Close();
  }

  inline bool IsClosed() const { return queue_.IsClosed(); }

  inline void Kill() { queue_.Kill(); }

  inline bool WaitForInited(size_t) { return true; }

 private:
  BlockingQueue<std::shared_ptr<std::vector<framework::LoDTensor>>> queue_;
};

class MultiLoDTensorBlockingQueue {
 public:
  explicit MultiLoDTensorBlockingQueue(size_t capacity,
                                       bool speed_test_mode = false)
      : capacity_(capacity), speed_test_mode_(speed_test_mode) {}

  ~MultiLoDTensorBlockingQueue() {
    VLOG(10) << "Destruct LoDTensorBlockingQueue";
  }

  const std::shared_ptr<LoDTensorBlockingQueue>& GetQueue(size_t idx) const {
    EnforceIsInited();
    PADDLE_ENFORCE_LT(
        idx, queues_.size(),
        platform::errors::OutOfRange("The queue index is out of range"));
    return queues_[idx];
  }

  bool Push(const std::vector<framework::LoDTensor>& lod_tensor_vec) {
    VLOG(0) << "MultiLoDTensorBlockingQueue: Push: queue index: "
            << push_index_ % queues_.size();
    return CurPushQueue()->Push(lod_tensor_vec);
  }

  void Pop(std::vector<framework::LoDTensor>* out, bool* ok = nullptr) {
    VLOG(0) << "MultiLoDTensorBlockingQueue: Pop: queue index: "
            << pop_index_ % queues_.size();
    CurPopQueue()->Pop(out, ok);
  }

  inline size_t Cap() const { return capacity_; }

  inline size_t Size() const {
    size_t size = 0;
    for (auto& item : queues_) {
      size += item->Size();
    }
    return size;
  }

  inline void ReOpen() {
    for (auto& item : queues_) {
      item->ReOpen();
    }
  }

  inline void Close() {
    for (auto& item : queues_) {
      item->Close();
    }
  }

  inline void Kill() {
    for (auto& item : queues_) {
      item->Kill();
    }
  }

  bool WaitForInited(size_t milliseconds) {
    std::unique_lock<std::mutex> lock(init_mutex_);
    return cv_.wait_for(lock, std::chrono::milliseconds(milliseconds),
                        [this] { return !queues_.empty(); });
  }

  void SetQueueSize(size_t q_size) {
    {
      std::lock_guard<std::mutex> lock(init_mutex_);
      if (!queues_.empty()) {
        PADDLE_ENFORCE_EQ(queues_.size(), q_size,
                          platform::errors::InvalidArgument(
                              "queues should be only inited once"));
        return;
      }

      VLOG(1) << "Init queue with size " << q_size;
      queues_.resize(q_size);
      for (auto& item : queues_) {
        auto cap = (capacity_ + q_size - 1) / q_size;
        item.reset(new LoDTensorBlockingQueue(cap, speed_test_mode_));
      }
    }
    cv_.notify_all();
  }

  inline void Reset() {
    auto queue_cnt = queues_.size();
    for (auto& item : queues_) {
      auto cap = (capacity_ + queue_cnt - 1) / queue_cnt;
      item.reset(new LoDTensorBlockingQueue(cap, speed_test_mode_));
    }
    push_index_ = 0;
    pop_index_ = 0;
  }

 private:
  const std::shared_ptr<LoDTensorBlockingQueue>& CurPushQueue() {
    return queues_[(push_index_++) % queues_.size()];
  }

  const std::shared_ptr<LoDTensorBlockingQueue>& CurPopQueue() {
    return queues_[(pop_index_++) % queues_.size()];
  }

  void EnforceIsInited() const {
    PADDLE_ENFORCE_EQ(queues_.empty(), false,
                      platform::errors::NotFound("queue has not been inited"));
  }

 private:
  std::vector<std::shared_ptr<LoDTensorBlockingQueue>> queues_;
  mutable uint64_t push_index_{0};
  mutable uint64_t pop_index_{0};

  const size_t capacity_;
  const bool speed_test_mode_;

  mutable std::mutex init_mutex_;
  mutable std::condition_variable cv_;
};

class OrderedMultiDeviceLoDTensorBlockingQueue {
 public:
  OrderedMultiDeviceLoDTensorBlockingQueue(size_t capacity,
                                           bool speed_test_mode = false)
      : capacity_(capacity), speed_test_mode_(speed_test_mode) {}

  ~OrderedMultiDeviceLoDTensorBlockingQueue() {
    VLOG(10) << "Destruct OrderedMultiDeviceLoDTensorBlockingQueue";
  }

  bool WaitForInited(size_t milliseconds) {
    std::unique_lock<std::mutex> lock(init_mutex_);
    return cv_.wait_for(lock, std::chrono::milliseconds(milliseconds),
                        [this] { return !queues_.empty(); });
  }

  void SetDeviceCount(size_t dev_cnt) {
    {
      std::lock_guard<std::mutex> lock(init_mutex_);
      PADDLE_ENFORCE_GE(dev_cnt, 1,
                        platform::errors::InvalidArgument(
                            "Device count to init "
                            "OrderedMultiDeviceLoDTensorBlockingQueue"
                            " must be larger than 1"));
      if (!queues_.empty()) {
        PADDLE_ENFORCE_EQ(queues_.size(), dev_cnt,
                          platform::errors::InvalidArgument(
                              "queues should be only inited once"));
        return;
      }

      VLOG(1) << "Init queue with size " << dev_cnt;
      queues_.resize(dev_cnt);
      for (auto& item : queues_) {
        auto cap = (capacity_ + dev_cnt - 1) / dev_cnt;
        item.reset(new LoDTensorBlockingQueue(cap, speed_test_mode_));
      }
    }
    cv_.notify_all();
  }

  const std::shared_ptr<LoDTensorBlockingQueue>& GetQueue(size_t idx) const {
    EnforceIsInited();
    PADDLE_ENFORCE_LT(
        idx, queues_.size(),
        platform::errors::OutOfRange("The queue index is out of range"));
    return queues_[idx];
  }

  bool Push(const std::vector<framework::LoDTensor>& lod_tensor_vec) {
    return CurQueue()->Push(lod_tensor_vec);
  }

  inline size_t Size() const {
    size_t size = 0;
    for (auto& item : queues_) {
      size += item->Size();
    }
    return size;
  }

  inline void Close() {
    for (auto& item : queues_) {
      item->Close();
    }
  }

  inline void Kill() {
    for (auto& item : queues_) {
      item->Kill();
    }
  }

  inline void Reset() {
    {
      std::lock_guard<std::mutex> reset_lock(reset_mutex_);
      for (auto& method : reset_methods_) {
        if (method) method();
      }
    }

    auto dev_cnt = queues_.size();
    for (auto& item : queues_) {
      auto cap = (capacity_ + dev_cnt - 1) / dev_cnt;
      item.reset(new LoDTensorBlockingQueue(cap, speed_test_mode_));
    }
    data_index_ = 0;
  }

  inline void SetResetMethod(size_t idx,
                             const std::function<void()>& reset_method) {
    std::lock_guard<std::mutex> reset_lock(reset_mutex_);
    EnforceIsInited();
    if (reset_methods_.size() <= idx) {
      reset_methods_.resize(idx + 1);
    }
    reset_methods_[idx] = reset_method;
  }

  inline size_t Cap() const { return capacity_; }

 private:
  const std::shared_ptr<LoDTensorBlockingQueue>& CurQueue() {
    return queues_[(data_index_++) % queues_.size()];
  }

 private:
  void EnforceIsInited() const {
    PADDLE_ENFORCE_EQ(queues_.empty(), false,
                      platform::errors::NotFound("queue has not been inited"));
  }

 private:
  std::vector<std::shared_ptr<LoDTensorBlockingQueue>> queues_;
  mutable uint64_t data_index_{0};

  size_t dev_cnt_{0};
  const size_t capacity_;
  const bool speed_test_mode_;
  bool is_closed_{false};

  std::vector<std::function<void()>> reset_methods_;
  mutable std::mutex reset_mutex_;

  mutable std::mutex init_mutex_;
  mutable std::condition_variable cv_;
};

class LoDTensorBlockingQueueHolder {
 public:
  void InitOnce(size_t capacity, bool speed_test_mode = false) {
    PADDLE_ENFORCE_EQ(
        queue_, nullptr,
        platform::errors::AlreadyExists("LoDTensorBlockingQueueHolder::"
                                        "InitOnce() can only be called once"));
    queue_.reset(new LoDTensorBlockingQueue(capacity, speed_test_mode));
  }

  inline const std::shared_ptr<LoDTensorBlockingQueue>& GetQueue() const {
    return queue_;
  }

 private:
  std::shared_ptr<LoDTensorBlockingQueue> queue_;
};

class SharedLoDTensorBlockingQueueHolder {
 public:
  void InitOnce(size_t capacity, bool speed_test_mode = false) {
    PADDLE_ENFORCE_EQ(
        queue_, nullptr,
        platform::errors::AlreadyExists("SharedLoDTensorBlockingQueueHolder::"
                                        "InitOnce() can only be called once"));
    queue_.reset(new SharedLoDTensorBlockingQueue(capacity, speed_test_mode));
  }

  inline const std::shared_ptr<SharedLoDTensorBlockingQueue>& GetQueue() const {
    return queue_;
  }

 private:
  std::shared_ptr<SharedLoDTensorBlockingQueue> queue_;
};

class MultiLoDTensorBlockingQueueHolder {
 public:
  void InitOnce(size_t capacity, bool speed_test_mode = false) {
    PADDLE_ENFORCE_EQ(
        queue_, nullptr,
        platform::errors::AlreadyExists("LoDTensorBlockingQueueHolder::"
                                        "InitOnce() can only be called once"));
    queue_.reset(new MultiLoDTensorBlockingQueue(capacity, speed_test_mode));
  }

  inline const std::shared_ptr<MultiLoDTensorBlockingQueue>& GetQueue() const {
    return queue_;
  }

 private:
  std::shared_ptr<MultiLoDTensorBlockingQueue> queue_;
};

class OrderedMultiDeviceLoDTensorBlockingQueueHolder {
 public:
  void InitOnce(size_t capacity, bool speed_test_mode = false) {
    PADDLE_ENFORCE_EQ(queue_, nullptr,
                      platform::errors::AlreadyExists(
                          "OrderedMultiDeviceLoDTensorBlockingQueueHolder::"
                          "InitOnce() can only be called once"));
    queue_.reset(new OrderedMultiDeviceLoDTensorBlockingQueue(capacity,
                                                              speed_test_mode));
  }

  inline const std::shared_ptr<OrderedMultiDeviceLoDTensorBlockingQueue>&
  GetQueue() const {
    return queue_;
  }

 private:
  std::shared_ptr<OrderedMultiDeviceLoDTensorBlockingQueue> queue_;
};

}  // namespace reader
}  // namespace operators
}  // namespace paddle
