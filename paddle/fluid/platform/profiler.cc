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

#include "paddle/fluid/platform/profiler.h"

#include <memory>
#include <mutex>  // NOLINT
#include <random>
#include <string>
#include <utility>

#include "paddle/fluid/platform/device_tracer.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/fluid/platform/profiler_helper.h"
#ifdef PADDLE_WITH_CUDA
#include "paddle/fluid/platform/dynload/nvtx.h"
#endif

DEFINE_bool(enable_rpc_profiler, false, "Enable rpc profiler or not.");

namespace paddle {
namespace platform {

MemEvenRecorder MemEvenRecorder::recorder;

Event::Event(EventType type, std::string name, uint32_t thread_id,
             EventRole role, std::string attr)
    : type_(type),
      name_(std::move(name)),
      thread_id_(thread_id),
      role_(role),
      attr_(std::move(attr)) {
  cpu_ns_ = GetTimeInNsec();
}

const EventType &Event::type() const { return type_; }

double Event::CpuElapsedMs(const Event &e) const {
  return (e.cpu_ns_ - cpu_ns_) / (1000000.0);
}

double Event::CudaElapsedMs(const Event &e) const {
#ifdef PADDLE_WITH_CUPTI
  return gpu_ns_ / 1000000.0;
#else
  LOG_FIRST_N(WARNING, 1) << "CUDA CUPTI is not enabled";
  return 0;
#endif
}

RecordEvent::RecordEvent(const std::string &name, const EventRole role,
                         const std::string attr) {
#ifndef _WIN32
#ifdef PADDLE_WITH_CUDA
  if (g_enable_nvprof_hook) {
    dynload::nvtxRangePushA(name.c_str());
    is_pushed_ = true;
  }
#endif
#endif
  if (g_state == ProfilerState::kDisabled || name.empty()) return;

  // do some initialization
  start_ns_ = PosixInNsec();
  role_ = role;
  is_enabled_ = true;
  // lock is not needed, the code below is thread-safe
  // Maybe need the same push/pop behavior.
  Event *e = PushEvent(name, role, attr);
  SetCurAnnotation(e);
  name_ = e->name();
}

RecordEvent::~RecordEvent() {
#ifndef _WIN32
#ifdef PADDLE_WITH_CUDA
  if (g_enable_nvprof_hook && is_pushed_) {
    dynload::nvtxRangePop();
  }
#endif
#endif
  if (g_state == ProfilerState::kDisabled || !is_enabled_) return;
  // lock is not needed, the code below is thread-safe
  DeviceTracer *tracer = GetDeviceTracer();
  if (tracer) {
    tracer->AddCPURecords(CurAnnotationName(), start_ns_, PosixInNsec(),
                          BlockDepth(), g_thread_id);
  }
  ClearCurAnnotation();
  PopEvent(name_, role_);
}

void MemEvenRecorder::PushMemRecord(const void *ptr, const Place &place,
                                    size_t size) {
  if (g_state == ProfilerState::kDisabled) return;
  std::lock_guard<std::mutex> guard(mtx_);
  auto &events = address_memevent_[place];
  PADDLE_ENFORCE_EQ(events.count(ptr), 0,
                    platform::errors::InvalidArgument(
                        "The Place can't exist in the stage of PushMemRecord"));
  events.emplace(ptr, std::make_unique<RecordMemEvent>(place, size));
}

void MemEvenRecorder::PopMemRecord(const void *ptr, const Place &place) {
  if (g_state == ProfilerState::kDisabled) return;
  std::lock_guard<std::mutex> guard(mtx_);
  auto &events = address_memevent_[place];
  auto iter = events.find(ptr);
  // The ptr maybe not in address_memevent
  if (iter != events.end()) {
    events.erase(iter);
  }
}

void MemEvenRecorder::Flush() {
  std::lock_guard<std::mutex> guard(mtx_);
  address_memevent_.clear();
}

MemEvenRecorder::RecordMemEvent::RecordMemEvent(Place place, size_t bytes)
    : place_(std::move(place)),
      bytes_(bytes),
      start_ns_(PosixInNsec()),
      alloc_in_(CurAnnotationName()) {
  PushMemEvent(start_ns_, end_ns_, bytes_, place_, alloc_in_);
}

MemEvenRecorder::RecordMemEvent::~RecordMemEvent() {
  DeviceTracer *tracer = GetDeviceTracer();
  end_ns_ = PosixInNsec();

  auto annotation_free = CurAnnotationName();
  if (tracer) {
    tracer->AddMemInfoRecord(start_ns_, end_ns_, bytes_, place_, alloc_in_,
                             annotation_free, g_mem_thread_id);
  }
  PopMemEvent(start_ns_, end_ns_, bytes_, place_, annotation_free);
}

RecordRPCEvent::RecordRPCEvent(const std::string &name) {
  if (FLAGS_enable_rpc_profiler) {
    event_ = std::make_unique<platform::RecordEvent>(name);
  }
}

RecordBlock::RecordBlock(int block_id)
    : is_enabled_(false), start_ns_(PosixInNsec()) {
  // lock is not needed, the code below is thread-safe
  if (g_state == ProfilerState::kDisabled) return;
  is_enabled_ = true;
  SetCurBlock(block_id);
  name_ = string::Sprintf("block_%d", block_id);
}

RecordBlock::~RecordBlock() {
  // lock is not needed, the code below is thread-safe
  if (g_state == ProfilerState::kDisabled || !is_enabled_) return;
  DeviceTracer *tracer = GetDeviceTracer();
  if (tracer) {
    // We try to put all blocks at the same nested depth in the
    // same timeline lane. and distinguish the using thread_id.
    tracer->AddCPURecords(name_, start_ns_, PosixInNsec(), BlockDepth(),
                          g_thread_id);
  }
  ClearCurBlock();
}

void PushMemEvent(uint64_t start_ns, uint64_t end_ns, size_t bytes,
                  const Place &place, const std::string &annotation) {
  GetMemEventList().Record(EventType::kPushRange, start_ns, end_ns, bytes,
                           place, g_mem_thread_id, annotation);
}

void PopMemEvent(uint64_t start_ns, uint64_t end_ns, size_t bytes,
                 const Place &place, const std::string &annotation) {
  GetMemEventList().Record(EventType::kPopRange, start_ns, end_ns, bytes, place,
                           g_mem_thread_id, annotation);
}

void Mark(const std::string &name) {
  GetEventList().Record(EventType::kMark, name, g_thread_id);
}

Event *PushEvent(const std::string &name, const EventRole role,
                 std::string attr) {
  return GetEventList().Record(EventType::kPushRange, name, g_thread_id, role,
                               attr);
}

void PopEvent(const std::string &name, const EventRole role, std::string attr) {
  GetEventList().Record(EventType::kPopRange, name, g_thread_id, role, attr);
}
void EnableProfiler(ProfilerState state) {
  PADDLE_ENFORCE_NE(state, ProfilerState::kDisabled,
                    platform::errors::InvalidArgument(
                        "Can't enable profiling, since the input state is"
                        "ProfilerState::kDisabled"));
  SynchronizeAllDevice();
  std::lock_guard<std::mutex> l(profiler_mu);
  if (state == g_state) {
    return;
  }
  g_state = state;
  should_send_profile_state = true;
  GetDeviceTracer()->Enable();
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  if (g_state == ProfilerState::kCUDA || g_state == ProfilerState::kAll ||
      g_state == ProfilerState::kCPU) {
    // Generate some dummy events first to reduce the startup overhead.
    DummyKernelAndEvent();
    GetDeviceTracer()->Reset();
  }
#endif
  // Mark the profiling start.
  Mark("_start_profiler_");
}

void ResetProfiler() {
  SynchronizeAllDevice();
  GetDeviceTracer()->Reset();
  MemEvenRecorder::Instance().Flush();
  std::lock_guard<std::mutex> guard(g_all_event_lists_mutex);
  for (auto &g_all_event_list : g_all_event_lists) {
    g_all_event_list->Clear();
  }
  for (auto &g_all_mem_event_list : g_all_mem_event_lists) {
    g_all_mem_event_list->Clear();
  }
}

void DisableProfiler(EventSortingKey sorted_key,
                     const std::string &profile_path) {
  SynchronizeAllDevice();
  MemEvenRecorder::Instance().Flush();

  std::lock_guard<std::mutex> l(profiler_mu);
  if (g_state == ProfilerState::kDisabled) return;
  // Mark the profiling stop.
  Mark("_stop_profiler_");
  DealWithShowName();

  DeviceTracer *tracer = GetDeviceTracer();
  if (tracer->IsEnabled()) {
    tracer->Disable();
    tracer->GenEventKernelCudaElapsedTime();
    tracer->GenProfile(profile_path);
  }

  std::vector<std::vector<Event>> all_events = GetAllEvents();

  ParseEvents(all_events, true, sorted_key);
  ParseEvents(all_events, false, sorted_key);
  if (VLOG_IS_ON(5)) {
    std::vector<std::vector<MemEvent>> all_mem_events = GetMemEvents();
    ParseMemEvents(all_mem_events);
  }

  ResetProfiler();
  g_state = ProfilerState::kDisabled;
  g_tracer_option = TracerOption::kDefault;
  should_send_profile_state = true;
}

std::vector<std::vector<Event>> GetAllEvents() {
  std::lock_guard<std::mutex> guard(g_all_event_lists_mutex);
  std::vector<std::vector<Event>> result;
  for (auto &g_all_event_list : g_all_event_lists) {
    result.emplace_back(g_all_event_list->Reduce());
  }
  return result;
}

bool IsProfileEnabled() { return g_state != ProfilerState::kDisabled; }

bool ShouldSendProfileState() { return should_send_profile_state; }

std::string OpName(const framework::VariableNameMap &name_map,
                   const std::string &type_name) {
  if (platform::GetTracerOption() != platform::TracerOption::kAllOpDetail ||
      !IsProfileEnabled())
    return "";

  std::string ret = type_name + "%";
  for (const auto &it : name_map) {
    auto name_outputs = it.second;
    if (!name_outputs.empty()) {
      ret = ret + name_outputs[0];
      break;
    }
  }
  ret = ret + "%";

  return ret;
}

void SetTracerOption(TracerOption option) {
  std::lock_guard<std::mutex> l(profiler_mu);
  g_tracer_option = option;
}

platform::TracerOption GetTracerOption() { return g_tracer_option; }

void SetProfileListener() {
  std::mt19937 rng;
  rng.seed(std::random_device()());
  std::uniform_int_distribution<std::mt19937::result_type> dist6(
      1, std::numeric_limits<int>::max());
  profiler_lister_id = dist6(rng);
}

int64_t ListenerId() { return profiler_lister_id; }

void NvprofEnableRecordEvent() {
  SynchronizeAllDevice();
  g_enable_nvprof_hook = true;
}

void NvprofDisableRecordEvent() { g_enable_nvprof_hook = false; }

}  // namespace platform
}  // namespace paddle
