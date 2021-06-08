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
#include <map>
#include <memory>
#include <mutex>  // NOLINT
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "paddle/fluid/framework/type_defs.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/fluid/platform/event.h"
#include "paddle/fluid/platform/place.h"
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
#include "paddle/fluid/platform/gpu_info.h"
#endif
namespace paddle {
namespace platform {

const int kEnableProfiler = 1;
const int kDisableProfiler = 2;

enum class ProfilerState {
  kDisabled,  // disabled state
  kCPU,       // CPU profiling state
  kCUDA,      // GPU profiling state
  kAll,       // Profile both CPU and GPU. (Currently experimental).
};

// it is the flag to control to print the profiling result
enum class TracerOption {
  kDefault,      // print the different op type profiling result
  kOpDetail,     // print the detail profiling result of different op type
  kAllOpDetail,  // print the detail profiling result of different op name
};

// Candidate keys to sort the profiling report
enum class EventSortingKey {
  kDefault,
  kCalls,
  kTotal,
  kMin,
  kMax,
  kAve,
  kCPUTime,
  kGPUTime
};

struct MemoryProfierReport {
  size_t alloc_times{0};
  size_t alloc_size{0};
  size_t free_times{0};
  size_t free_size{0};
};

// The information of each event given in the profiling report
struct EventItem {
  std::string name;
  int calls;
  double total_time;
  double max_time;
  double ave_time;
  double min_time;
  double cpu_time;
  double gpu_time;
  float ratio;
  EventRole role;
};

struct OverHead {
  bool print_overhead = false;
  bool print_explanation = false;
  double elapsed_time = 0.;      // the elapsed time of all events
  double accumulated_time = 0.;  // the accumulated time of all events
  double compute_time = 0.0;
  double framework_time = 0.0;
  EventItem memcpy_item;
  std::vector<EventItem> sub_memcpy_items;
};

struct MemEvenRecorder {
 public:
  void PushMemRecord(const void* ptr, const Place& place, size_t size);
  void PopMemRecord(const void* ptr, const Place& place);
  void Flush();
  static MemEvenRecorder& Instance() { return recorder; }

 private:
  struct RecordMemEvent {
    RecordMemEvent(Place place, size_t bytes);
    ~RecordMemEvent();

    Place place_;
    size_t bytes_;
    uint64_t start_ns_;
    uint64_t end_ns_;
    std::string alloc_in_;
    std::string free_in_;
  };

  static MemEvenRecorder recorder;
  std::map<Place,
           std::unordered_map<const void*, std::unique_ptr<RecordMemEvent>>>
      address_memevent_;
  std::mutex mtx_;
  MemEvenRecorder() {}
  DISABLE_COPY_AND_ASSIGN(MemEvenRecorder);
};

struct RecordEvent {
  RecordEvent(const std::string& name,
              const EventRole role = EventRole::kOrdinary,
              const std::string attr = "none");

  ~RecordEvent();

  bool is_enabled_{false};
  bool is_pushed_{false};
  uint64_t start_ns_;
  // Event name
  std::string name_;
  // Need to distinguish name by op type, block_id, program_id and perhaps
  // different kernel invocations within an op.
  std::string full_name_;
  EventRole role_{EventRole::kOrdinary};
};

class RecordRPCEvent {
 public:
  explicit RecordRPCEvent(const std::string& name);
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

template <typename T>
struct EventList {
  constexpr static size_t kMB = 1024 * 1024;
  constexpr static size_t kEventBlockSize = 16 * kMB;
  constexpr static size_t kEventSize = sizeof(T);
  constexpr static size_t kEventAlign = alignof(T);
  constexpr static size_t kNumBlock =
      kEventBlockSize /
      ((kEventSize + kEventAlign - 1) / kEventAlign * kEventAlign);

  template <typename... Args>
  T* Record(Args&&... args) {
    if (event_blocks.empty() || event_blocks.front().size() == kNumBlock) {
      event_blocks.emplace_front();
      event_blocks.front().reserve(kNumBlock);
    }
    event_blocks.front().emplace_back(std::forward<Args>(args)...);
    return &event_blocks.front().back();
  }

  std::vector<T> Reduce() {
    std::vector<T> result;
    for (auto& block : event_blocks) {
      result.insert(result.begin(), std::make_move_iterator(block.begin()),
                    std::make_move_iterator(block.end()));
    }
    event_blocks.clear();
    return result;
  }

  void Clear() { event_blocks.clear(); }

  std::forward_list<std::vector<T>> event_blocks;
};

void Mark(const std::string& name);
void PushMemEvent(uint64_t start_ns, uint64_t end_ns, size_t bytes,
                  const Place& place, const std::string& annotation);
void PopMemEvent(uint64_t start_ns, uint64_t end_ns, size_t bytes,
                 const Place& place, const std::string& annotation);
Event* PushEvent(const std::string& name, const EventRole role,
                 const std::string attr = "none");
void PopEvent(const std::string& name, const EventRole role,
              const std::string attr = "none");
// Return the event list of all threads. Assumed the returned value calls
// event_lists, event_lists[i][j] represents the j-th Event of i-th thread.
std::vector<std::vector<Event>> GetAllEvents();

// Enable the profiling function.
void EnableProfiler(ProfilerState state);
// Clear the g_all_event_lists, which is total event lists of all threads.
void ResetProfiler();
void DisableProfiler(EventSortingKey sorted_key,
                     const std::string& profile_path);
// Test if the profiler is currently enabled.
bool IsProfileEnabled();
// Whether the trainer should send profiling state to PS.
bool ShouldSendProfileState();
std::string OpName(const framework::VariableNameMap& name_map,
                   const std::string& type_name);
void SetTracerOption(TracerOption option);
platform::TracerOption GetTracerOption();
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
void DummyKernelAndEvent();
#endif

// Mark current process as PS by assigning a lister id.
void SetProfileListener();
int64_t ListenerId();

void NvprofEnableRecordEvent();
void NvprofDisableRecordEvent();

}  // namespace platform
}  // namespace paddle
