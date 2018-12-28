/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

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
#include <forward_list>
#include <list>
#include <string>
#include <vector>
#include "paddle/fluid/platform/device_context.h"

namespace paddle {
namespace platform {

enum EventType { kMark, kPushRange, kPopRange };

class Event {
 public:
  // The DeviceContext is used to get the cuda stream.
  // If CPU profiling mode, can pass nullptr.
  Event(EventType type, std::string name, uint32_t thread_id,
        const DeviceContext* dev_ctx, size_t flops = 0);

  const EventType& type() const;
  std::string name() const { return name_; }
  uint32_t thread_id() const { return thread_id_; }
  bool has_cuda() const { return has_cuda_; }
  size_t flops() const { return flops_; }

#ifdef PADDLE_WITH_CUDA
  cudaEvent_t event() const { return event_; }
  int device() const { return device_; }
#endif

  double CpuElapsedMs(const Event& e) const;
  double CudaElapsedMs(const Event& e) const;

 private:
  EventType type_;
  std::string name_;
  uint32_t thread_id_;
  int64_t cpu_ns_;
  size_t flops_;
  bool has_cuda_;
#ifdef PADDLE_WITH_CUDA
  cudaEvent_t event_ = nullptr;
  int device_ = -1;
#endif
};

enum ProfilerState {
  kDisabled,  // disabled state
  kCPU,       // CPU profiling state
  kCUDA,      // GPU profiling state
  kAll,       // Profile both CPU and GPU. (Currently experimental).
};

void Mark(const std::string& name, const DeviceContext* dev_ctx);

void PushEvent(const std::string& name, const DeviceContext* dev_ctx,
               size_t flops = 0);

void PopEvent(const std::string& name, const DeviceContext* dev_ctx);

struct RecordEvent {
  // dev_ctx can be set to nullptr if device is cpu.
  RecordEvent(const std::string& name, const DeviceContext* dev_ctx,
              size_t flops = 0);

  ~RecordEvent();

  bool is_enabled_;
  uint64_t start_ns_;
  // The device context is used by Event to get the current cuda stream.
  const DeviceContext* dev_ctx_;
  // Event name
  std::string name_;
  // Need to distinguish name by op type, block_id, program_id and perhaps
  // different kernel invocations within an op.
  std::string full_name_;
};

class RecordRPCEvent {
 public:
  // dev_ctx can be set to nullptr if device is cpu.
  RecordRPCEvent(const std::string& name, const DeviceContext* dev_ctx);
  ~RecordRPCEvent() {}

 private:
  std::unique_ptr<RecordEvent> event_;
};

struct RecordBlock {
  explicit RecordBlock(int block_id);
  ~RecordBlock();

 private:
  bool is_enabled_;
  std::string name_;
  uint64_t start_ns_;
};

// Return the event list of all threads. Assumed the returned value calls
// event_lists, event_lists[i][j] represents the j-th Event of i-th thread.
std::vector<std::vector<Event>> GetAllEvents();

// Candidate keys to sort the profiling report
enum EventSortingKey { kDefault, kCalls, kTotal, kMin, kMax, kAve };

// Enable the profiling function.
void EnableProfiler(ProfilerState state);

// Clear the g_all_event_lists, which is total event lists of all threads.
void ResetProfiler();

void DisableProfiler(EventSortingKey sorted_key,
                     const std::string& profile_path);

const int kEnableProfiler = 1;
const int kDisableProfiler = 2;
// Test if the profiler is currently enabled.
bool IsProfileEnabled();
// Whether the trainer should send profiling state to PS.
bool ShouldSendProfileState();
// Mark current process as PS by assigning a lister id.
void SetProfileListener();
int64_t ListenerId();

}  // namespace platform
}  // namespace paddle
