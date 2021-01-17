/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.

licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once

#include <string>
#ifdef PADDLE_WITH_CUDA
#include <cuda_runtime.h>
#endif
#ifdef PADDLE_WITH_HIP
#include <hip/hip_runtime.h>
#endif
#include "paddle/fluid/platform/place.h"

namespace paddle {
namespace platform {

enum class EventType { kMark, kPushRange, kPopRange };

enum class EventRole {
  kOrdinary,  // only record op time with op type key
  kInnerOp,   // record op detail time with op type key
  kUniqueOp,  // record op detail time with op unique name key
  kSpecial,   // record event such as PE which is outer of thread local
};

class Event {
 public:
  // The DeviceContext is used to get the cuda stream.
  // If CPU profiling mode, can pass nullptr.
  Event(EventType type, std::string name, uint32_t thread_id,
        EventRole role = EventRole::kOrdinary);

  const EventType& type() const;
  Event* parent() const { return parent_; }
  void set_parent(Event* parent) { parent_ = parent; }
  std::string name() const { return name_; }
  EventRole role() const { return role_; }
  uint32_t thread_id() const { return thread_id_; }
  void set_name(std::string name) { name_ = name; }
  void set_role(EventRole role) { role_ = role; }

#ifdef PADDLE_WITH_CUDA
#ifndef PADDLE_WITH_CUPTI
  cudaEvent_t event() const { return event_; }
  int device() const { return device_; }
#endif
#endif
#ifdef PADDLE_WITH_HIP
  hipEvent_t event() const { return event_; }
  int device() const { return device_; }
#endif
  double CpuElapsedMs(const Event& e) const;
  double CudaElapsedMs(const Event& e) const;

 private:
  EventType type_;
  std::string name_{};
  Event* parent_{nullptr};
  uint32_t thread_id_;
  EventRole role_{};
  int64_t cpu_ns_;
  bool visited_status_{false};
#ifdef PADDLE_WITH_CUDA
#ifdef PADDLE_WITH_CUPTI
  int64_t gpu_ns_ = 0;

 public:
  void AddCudaElapsedTime(int64_t start_ns, int64_t end_ns) {
    gpu_ns_ += end_ns - start_ns;
  }

 private:
#else
  cudaEvent_t event_ = nullptr;
  int device_ = -1;
#endif
#endif
#ifdef PADDLE_WITH_HIP
  hipEvent_t event_ = nullptr;
  int device_ = -1;
#endif
};

class MemEvent {
 public:
  MemEvent(EventType type, uint64_t start_ns, uint64_t end_ns, size_t bytes,
           Place place, int64_t thread_id, const std::string& annotation)
      : type_(type),
        start_ns_(start_ns),
        end_ns_(end_ns),
        bytes_(bytes),
        place_(place),
        thread_id_(thread_id),
        annotation_(annotation) {}

  const EventType& type() const { return type_; }
  uint64_t start_ns() const { return start_ns_; }
  uint64_t end_ns() const { return end_ns_; }
  size_t bytes() const { return bytes_; }
  Place place() const { return place_; }
  int64_t thread_id() const { return thread_id_; }
  const std::string& annotation() const { return annotation_; }

 private:
  EventType type_;
  uint64_t start_ns_ = 0;
  uint64_t end_ns_ = 0;
  size_t bytes_;
  Place place_;
  int64_t thread_id_;
  std::string annotation_;
};

}  // namespace platform
}  // namespace paddle
