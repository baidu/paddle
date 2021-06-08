/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/platform/device_tracer.h"

#include <deque>
#include <forward_list>
#include <fstream>
#include <mutex>  // NOLINT
#include <string>
#include <thread>  // NOLINT

#include "glog/logging.h"

namespace paddle {
namespace platform {
namespace {
// Tracking the nested block stacks of each thread.
#ifdef PADDLE_WITH_SW
// sw not supported thread_local
std::deque<int> block_id_stack;
std::deque<Event *> annotation_stack;
#else
// Tracking the nested event stacks.
thread_local std::deque<int> block_id_stack;
// Tracking the nested event stacks.
thread_local std::deque<Event *> annotation_stack;
#endif
// stack to strore event sunch as pe and so on
static std::deque<Event *> main_thread_annotation_stack{};
static std::deque<std::string> main_thread_annotation_stack_name{};

std::map<uint32_t, int32_t> system_thread_id_map;

std::once_flag tracer_once_flag;
DeviceTracer *tracer = nullptr;

void PrintCuptiHint() {
  static bool showed = false;
  if (showed) return;
  showed = true;
  LOG(WARNING) << "Invalid timestamp occurred. Please try increasing the "
                  "FLAGS_multiple_of_cupti_buffer_size.";
}

}  // namespace
#ifdef PADDLE_WITH_CUPTI

namespace {
// The experimental best performance is
// the same size with CUPTI device buffer size(8M)
uint64_t kBufSize = 1024 * 1024 * 8;
uint64_t kAlignSize = 8;
std::unordered_map<CUpti_CallbackId, std::string> runtime_cbid_str,
    driver_cbid_str;

#define ALIGN_BUFFER(buffer, align)                                 \
  (((uintptr_t)(buffer) & ((align)-1))                              \
       ? ((buffer) + (align) - ((uintptr_t)(buffer) & ((align)-1))) \
       : (buffer))

#define CUPTI_CALL(call)                                                   \
  do {                                                                     \
    CUptiResult _status = call;                                            \
    if (_status != CUPTI_SUCCESS) {                                        \
      const char *errstr;                                                  \
      dynload::cuptiGetResultString(_status, &errstr);                     \
      fprintf(stderr, "%s:%d: error: function %s failed with error %s.\n", \
              __FILE__, __LINE__, #call, errstr);                          \
      exit(-1);                                                            \
    }                                                                      \
  } while (0)

std::string MemcpyKind(CUpti_ActivityMemcpyKind kind) {
  switch (kind) {
    case CUPTI_ACTIVITY_MEMCPY_KIND_HTOD:
      return "MEMCPY_HtoD";
    case CUPTI_ACTIVITY_MEMCPY_KIND_DTOH:
      return "MEMCPY_DtoH";
    case CUPTI_ACTIVITY_MEMCPY_KIND_HTOA:
      return "MEMCPY_HtoA";
    case CUPTI_ACTIVITY_MEMCPY_KIND_ATOH:
      return "MEMCPY_AtoH";
    case CUPTI_ACTIVITY_MEMCPY_KIND_ATOA:
      return "MEMCPY_AtoA";
    case CUPTI_ACTIVITY_MEMCPY_KIND_ATOD:
      return "MEMCPY_AtoD";
    case CUPTI_ACTIVITY_MEMCPY_KIND_DTOA:
      return "MEMCPY_DtoA";
    case CUPTI_ACTIVITY_MEMCPY_KIND_DTOD:
      return "MEMCPY_DtoD";
    case CUPTI_ACTIVITY_MEMCPY_KIND_HTOH:
      return "MEMCPY_HtoH";
    case CUPTI_ACTIVITY_MEMCPY_KIND_PTOP:
      return "MEMCPY_PtoP";
    case CUPTI_ACTIVITY_MEMCPY_KIND_FORCE_INT:
      return "MEMCPY_FORCE_INT";
    default:
      break;
  }
  return "MEMCPY";
}

std::string DriverKind(CUpti_CallbackId cbid) {
  auto iter = driver_cbid_str.find(cbid);
  if (iter == driver_cbid_str.end())
    return "Driver API " + std::to_string(cbid);
  return iter->second;
}

std::string RuntimeKind(CUpti_CallbackId cbid) {
  auto iter = runtime_cbid_str.find(cbid);
  if (iter == runtime_cbid_str.end())
    return "Runtime API " + std::to_string(cbid);
  return iter->second;
}

void EnableActivity() {
  // Device activity record is created when CUDA initializes, so we
  // want to enable it before cuInit() or any CUDA runtime call.
  CUPTI_CALL(dynload::cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MEMCPY));
  CUPTI_CALL(
      dynload::cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL));
  // CUPTI_CALL(dynload::cuptiActivityEnable(CUPTI_ACTIVITY_KIND_KERNEL));
  CUPTI_CALL(dynload::cuptiActivityEnable(CUPTI_ACTIVITY_KIND_DRIVER));
  CUPTI_CALL(dynload::cuptiActivityEnable(CUPTI_ACTIVITY_KIND_RUNTIME));
  // We don't track these activities for now.
  CUPTI_CALL(dynload::cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MEMSET));
  // CUPTI_CALL(dynload::cuptiActivityEnable(CUPTI_ACTIVITY_KIND_OVERHEAD));
  // CUPTI_CALL(dynload::cuptiActivityEnable(CUPTI_ACTIVITY_KIND_DEVICE));
  // CUPTI_CALL(dynload::cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONTEXT));
  // CUPTI_CALL(dynload::cuptiActivityEnable(CUPTI_ACTIVITY_KIND_DRIVER));
  // CUPTI_CALL(dynload::cuptiActivityEnable(CUPTI_ACTIVITY_KIND_RUNTIME));
  // CUPTI_CALL(dynload::cuptiActivityEnable(CUPTI_ACTIVITY_KIND_NAME));
  // CUPTI_CALL(dynload::cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MARKER));
}

void DisableActivity() {
  CUPTI_CALL(dynload::cuptiActivityDisable(CUPTI_ACTIVITY_KIND_MEMCPY));
  CUPTI_CALL(
      dynload::cuptiActivityDisable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL));
  // CUPTI_CALL(dynload::cuptiActivityDisable(CUPTI_ACTIVITY_KIND_DEVICE));
  // Disable all other activity record kinds.
  // CUPTI_CALL(dynload::cuptiActivityDisable(CUPTI_ACTIVITY_KIND_CONTEXT));
  CUPTI_CALL(dynload::cuptiActivityDisable(CUPTI_ACTIVITY_KIND_DRIVER));
  CUPTI_CALL(dynload::cuptiActivityDisable(CUPTI_ACTIVITY_KIND_RUNTIME));
  CUPTI_CALL(dynload::cuptiActivityDisable(CUPTI_ACTIVITY_KIND_MEMSET));
  // CUPTI_CALL(dynload::cuptiActivityDisable(CUPTI_ACTIVITY_KIND_NAME));
  // CUPTI_CALL(dynload::cuptiActivityDisable(CUPTI_ACTIVITY_KIND_MARKER));
  // CUPTI_CALL(dynload::cuptiActivityDisable(CUPTI_ACTIVITY_KIND_OVERHEAD));
}

void CUPTIAPI bufferRequested(uint8_t **buffer, size_t *size,
                              size_t *maxNumRecords) {
  uint8_t *buf = reinterpret_cast<uint8_t *>(malloc(kBufSize + kAlignSize));
  *size = kBufSize;
  *buffer = ALIGN_BUFFER(buf, kAlignSize);
  *maxNumRecords = 0;
}

void CUPTIAPI bufferCompleted(CUcontext ctx, uint32_t streamId, uint8_t *buffer,
                              size_t size, size_t validSize) {
  static std::thread::id cupti_thread_id(0);
  if (cupti_thread_id == std::thread::id(0))
    cupti_thread_id = std::this_thread::get_id();
  PADDLE_ENFORCE_EQ(
      std::this_thread::get_id(), cupti_thread_id,
      platform::errors::PermissionDenied(
          "Only one thread is allowed to call bufferCompleted()."));
  CUptiResult status;
  CUpti_Activity *record = NULL;
  if (validSize > 0) {
    do {
      status = dynload::cuptiActivityGetNextRecord(buffer, validSize, &record);
      if (status == CUPTI_SUCCESS) {
        switch (record->kind) {
          case CUPTI_ACTIVITY_KIND_KERNEL:
          case CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL: {
#if CUDA_VERSION >= 9000
            auto *kernel =
                reinterpret_cast<const CUpti_ActivityKernel4 *>(record);
#else
            auto *kernel =
                reinterpret_cast<const CUpti_ActivityKernel3 *>(record);
#endif
            tracer->AddKernelRecords(kernel->name, kernel->start, kernel->end,
                                     kernel->deviceId, kernel->streamId,
                                     kernel->correlationId);
            break;
          }
          case CUPTI_ACTIVITY_KIND_MEMCPY: {
            auto *memcpy =
                reinterpret_cast<const CUpti_ActivityMemcpy *>(record);
            tracer->AddMemRecords(
                MemcpyKind(
                    static_cast<CUpti_ActivityMemcpyKind>(memcpy->copyKind)),
                memcpy->start, memcpy->end, memcpy->deviceId, memcpy->streamId,
                memcpy->correlationId, memcpy->bytes);
            break;
          }
          case CUPTI_ACTIVITY_KIND_MEMCPY2: {
            auto *memcpy =
                reinterpret_cast<const CUpti_ActivityMemcpy2 *>(record);
            tracer->AddMemRecords(
                MemcpyKind(
                    static_cast<CUpti_ActivityMemcpyKind>(memcpy->copyKind)),
                memcpy->start, memcpy->end, memcpy->deviceId, memcpy->streamId,
                memcpy->correlationId, memcpy->bytes);
            break;
          }
          case CUPTI_ACTIVITY_KIND_MEMSET: {
            auto *memset =
                reinterpret_cast<const CUpti_ActivityMemset *>(record);
            tracer->AddKernelRecords("MEMSET", memset->start, memset->end,
                                     memset->deviceId, memset->streamId,
                                     memset->correlationId);
            break;
          }
          case CUPTI_ACTIVITY_KIND_DRIVER: {
            auto *api = reinterpret_cast<const CUpti_ActivityAPI *>(record);
            if (api->start != 0 && api->end != 0) {
              // -1 device id represents ActiveKind api call
              tracer->AddActiveKindRecords(
                  DriverKind(api->cbid), api->start, api->end, -1,
                  GetThreadIdFromSystemThreadId(api->threadId),
                  api->correlationId);
            }
            break;
          }
          case CUPTI_ACTIVITY_KIND_RUNTIME: {
            auto *api = reinterpret_cast<const CUpti_ActivityAPI *>(record);
            if (api->start != 0 && api->end != 0) {
              // -1 device id represents ActiveKind api call
              tracer->AddActiveKindRecords(
                  RuntimeKind(api->cbid), api->start, api->end, -1,
                  GetThreadIdFromSystemThreadId(api->threadId),
                  api->correlationId);
            }
            break;
          }
          default: {
            break;
          }
        }
      } else if (status == CUPTI_ERROR_MAX_LIMIT_REACHED) {
        // Seems not an error in this case.
        break;
      } else {
        CUPTI_CALL(status);
      }
    } while (1);

    size_t dropped;
    CUPTI_CALL(
        dynload::cuptiActivityGetNumDroppedRecords(ctx, streamId, &dropped));
    if (dropped != 0) {
      fprintf(stderr, "Dropped %u activity records\n", (unsigned int)dropped);
      PrintCuptiHint();
    }
  }
  free(buffer);
}

void initCuptiCbidStr();

}  // namespace

#endif  // PADDLE_WITH_CUPTI

class DeviceTracerImpl : public DeviceTracer {
 public:
  DeviceTracerImpl() {
#ifdef PADDLE_WITH_CUPTI
    initCuptiCbidStr();
#endif
  }

  void AddAnnotation(uint32_t id, Event *event) override {
#ifdef PADDLE_WITH_SW
    std::forward_list<std::pair<uint32_t, Event *>> *local_correlations_pairs =
        nullptr;
#else
    thread_local std::forward_list<std::pair<uint32_t, Event *>>
        *local_correlations_pairs = nullptr;
#endif
    if (local_correlations_pairs == nullptr) {
      std::lock_guard<std::mutex> l(trace_mu_);
      correlations_pairs.emplace_front();
      local_correlations_pairs = &correlations_pairs.front();
    }
    local_correlations_pairs->push_front(std::make_pair(id, event));
  }

  void AddCPURecords(const std::string &anno, uint64_t start_ns,
                     uint64_t end_ns, int64_t device_id,
                     int64_t thread_id) override {
    if (anno.empty()) {
      VLOG(1) << "Empty timeline annotation.";
      return;
    }
#ifdef PADDLE_WITH_SW
    std::forward_list<CPURecord> *local_cpu_records_ = nullptr;
#else
    thread_local std::forward_list<CPURecord> *local_cpu_records_ = nullptr;
#endif
    if (local_cpu_records_ == nullptr) {
      std::lock_guard<std::mutex> l(trace_mu_);
      cpu_records_.emplace_front();
      local_cpu_records_ = &cpu_records_.front();
    }
    local_cpu_records_->push_front(
        CPURecord{anno, start_ns, end_ns, device_id, thread_id});
  }

  void AddMemRecords(const std::string &name, uint64_t start_ns,
                     uint64_t end_ns, int64_t device_id, int64_t stream_id,
                     uint32_t correlation_id, uint64_t bytes) override {
    // 0 means timestamp information could not be collected for the kernel.
    if (start_ns == 0 || end_ns == 0 || start_ns == end_ns) {
      VLOG(3) << name << " cannot be traced";
      PrintCuptiHint();
      return;
    }
    // NOTE(liangdun): lock is not needed, only one thread call this function.
    mem_records_.push_front(MemRecord{name, start_ns, end_ns, device_id,
                                      stream_id, correlation_id, bytes});
  }

  void AddMemInfoRecord(uint64_t start_ns, uint64_t end_ns, size_t bytes,
                        const Place &place, const std::string &alloc_in,
                        const std::string &free_in,
                        int64_t thread_id) override {
    if (0 == start_ns || 0 == end_ns) {
      VLOG(3) << alloc_in << ", " << free_in << " Cannot be traced.";
      return;
    }
#ifdef PADDLE_WITH_SW
    std::forward_list<MemInfoRecord> *local_mem_info_record = nullptr;
#else
    thread_local std::forward_list<MemInfoRecord> *local_mem_info_record =
        nullptr;
#endif
    if (local_mem_info_record == nullptr) {
      std::lock_guard<std::mutex> l(trace_mu_);
      mem_info_record_.emplace_front();
      local_mem_info_record = &mem_info_record_.front();
    }
    local_mem_info_record->emplace_front(MemInfoRecord{
        start_ns, end_ns, bytes, place, thread_id, alloc_in, free_in});
  }

  void AddActiveKindRecords(const std::string &anno, uint64_t start_ns,
                            uint64_t end_ns, int64_t device_id,
                            int64_t thread_id,
                            uint32_t correlation_id) override {
    if (anno.empty()) {
      VLOG(1) << "Empty timeline annotation.";
      return;
    }
#ifdef PADDLE_WITH_SW
    std::forward_list<ActiveKindRecord> *local_active_kind_records = nullptr;
#else
    thread_local std::forward_list<ActiveKindRecord>
        *local_active_kind_records = nullptr;
#endif
    if (local_active_kind_records == nullptr) {
      std::lock_guard<std::mutex> l(trace_mu_);
      active_kind_records_.emplace_front();
      local_active_kind_records = &active_kind_records_.front();
    }
    //  lock is not needed, only one thread call this function.
    local_active_kind_records->push_front(ActiveKindRecord{
        anno, start_ns, end_ns, device_id, thread_id, correlation_id});
  }

  void AddKernelRecords(std::string name, uint64_t start, uint64_t end,
                        int64_t device_id, int64_t stream_id,
                        uint32_t correlation_id) override {
    // 0 means timestamp information could not be collected for the kernel.
    if (start == 0 || end == 0 || start == end) {
      VLOG(3) << correlation_id << " cannot be traced";
      PrintCuptiHint();
      return;
    }
    // NOTE(liangdun): lock is not needed, only one thread call this function.
    kernel_records_.push_front(
        KernelRecord{name, start, end, device_id, stream_id, correlation_id});
  }

  bool IsEnabled() override {
    std::lock_guard<std::mutex> l(trace_mu_);
    return enabled_;
  }

  void Enable() override {
    std::lock_guard<std::mutex> l(trace_mu_);
    if (enabled_) {
      return;
    }

#ifdef PADDLE_WITH_CUPTI
    EnableActivity();

    // Register callbacks for buffer requests and completed by CUPTI.
    CUPTI_CALL(dynload::cuptiActivityRegisterCallbacks(bufferRequested,
                                                       bufferCompleted));

    CUptiResult ret;
    ret = dynload::cuptiSubscribe(
        &subscriber_, static_cast<CUpti_CallbackFunc>(ApiCallback), this);
    if (ret == CUPTI_ERROR_MAX_LIMIT_REACHED) {
      fprintf(stderr, "CUPTI subcriber limit reached.\n");
    } else if (ret != CUPTI_SUCCESS) {
      fprintf(stderr, "Failed to create CUPTI subscriber.\n");
    }
    const std::vector<int> runtime_cbids {
      CUPTI_RUNTIME_TRACE_CBID_cudaMemcpy_v3020,
          CUPTI_RUNTIME_TRACE_CBID_cudaSetupArgument_v3020,
          CUPTI_RUNTIME_TRACE_CBID_cudaMemcpyAsync_v3020,
          CUPTI_RUNTIME_TRACE_CBID_cudaMemset_v3020,
          CUPTI_RUNTIME_TRACE_CBID_cudaMemsetAsync_v3020,
          CUPTI_RUNTIME_TRACE_CBID_cudaLaunch_v3020,
          CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_v7000
#if CUDA_VERSION >= 9000
          ,
          CUPTI_RUNTIME_TRACE_CBID_cudaLaunchCooperativeKernel_v9000,
          CUPTI_RUNTIME_TRACE_CBID_cudaLaunchCooperativeKernelMultiDevice_v9000
#endif
    };
    const std::vector<int> driver_cbids{CUPTI_DRIVER_TRACE_CBID_cuLaunch,
                                        CUPTI_DRIVER_TRACE_CBID_cuLaunchGrid,
                                        CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel};
    for (auto cbid : runtime_cbids)
      CUPTI_CALL(dynload::cuptiEnableCallback(
          1, subscriber_, CUPTI_CB_DOMAIN_RUNTIME_API, cbid));
    for (auto cbid : driver_cbids)
      CUPTI_CALL(dynload::cuptiEnableCallback(
          1, subscriber_, CUPTI_CB_DOMAIN_DRIVER_API, cbid));
    CUPTI_CALL(dynload::cuptiGetTimestamp(&start_ns_));
#endif  // PADDLE_WITH_CUPTI
    enabled_ = true;
  }

  void Reset() override {
#ifdef PADDLE_WITH_CUPTI
    CUPTI_CALL(
        dynload::cuptiActivityFlushAll(CUPTI_ACTIVITY_FLAG_FLUSH_FORCED));
#endif
    std::lock_guard<std::mutex> l(trace_mu_);
    kernel_records_.clear();
    mem_records_.clear();
    correlations_.clear();
    for (auto &tmp : correlations_pairs) tmp.clear();
    for (auto &tmp : cpu_records_) tmp.clear();
    for (auto &tmp : mem_info_record_) tmp.clear();
    for (auto &tmp : active_kind_records_) tmp.clear();
  }

  void GenEventKernelCudaElapsedTime() override {
#ifdef PADDLE_WITH_CUPTI
    if (correlations_.empty())
      for (auto &tmp : correlations_pairs)
        for (auto &pair : tmp) correlations_[pair.first] = pair.second;
    for (const KernelRecord &r : kernel_records_) {
      auto c = correlations_.find(r.correlation_id);
      if (c != correlations_.end() && c->second != nullptr) {
        Event *e = c->second;
        Event *parent = e->parent();
        while (parent) {
          parent->AddCudaElapsedTime(r.start_ns, r.end_ns);
          parent = parent->parent();
        }
        e->AddCudaElapsedTime(r.start_ns, r.end_ns);
      }
    }
    for (const auto &r : mem_records_) {
      auto c = correlations_.find(r.correlation_id);
      if (c != correlations_.end() && c->second != nullptr) {
        Event *e = c->second;
        Event *parent = e->parent();
        while (parent) {
          parent->AddCudaElapsedTime(r.start_ns, r.end_ns);
          parent = parent->parent();
        }
        e->AddCudaElapsedTime(r.start_ns, r.end_ns);
      }
    }
#endif
  }

  proto::Profile GenProfile(const std::string &profile_path) override {
    int miss = 0, find = 0;
    std::lock_guard<std::mutex> l(trace_mu_);
    proto::Profile profile_pb;
    profile_pb.set_start_ns(start_ns_);
    profile_pb.set_end_ns(end_ns_);
    if (correlations_.empty()) {
      for (auto &tmp : correlations_pairs) {
        for (auto &pair : tmp) correlations_[pair.first] = pair.second;
      }
    }

    for (const KernelRecord &r : kernel_records_) {
      auto *event = profile_pb.add_events();
      event->set_type(proto::Event::GPUKernel);
      auto c = correlations_.find(r.correlation_id);
      if (c != correlations_.end() && c->second != nullptr) {
        event->set_name(c->second->name());
        event->set_detail_info(c->second->attr());
        find++;
      } else {
        VLOG(10) << "Missing Kernel Event: " + r.name;
        miss++;
        event->set_name(r.name);
      }
      event->set_start_ns(r.start_ns);
      event->set_end_ns(r.end_ns);
      event->set_sub_device_id(r.stream_id);
      event->set_device_id(r.device_id);
    }
    VLOG(1) << "KernelRecord event miss: " << miss << " find: " << find;

    for (auto &tmp : cpu_records_) {
      for (const CPURecord &r : tmp) {
        auto *event = profile_pb.add_events();
        event->set_type(proto::Event::CPU);
        event->set_name(r.name);
        event->set_start_ns(r.start_ns);
        event->set_end_ns(r.end_ns);
        event->set_sub_device_id(r.thread_id);
        event->set_device_id(r.device_id);
      }
    }

    for (auto &tmp : active_kind_records_) {
      for (const ActiveKindRecord &r : tmp) {
        auto *event = profile_pb.add_events();
        event->set_type(proto::Event::CPU);
        auto c = correlations_.find(r.correlation_id);
        if (c != correlations_.end() && c->second != nullptr) {
          event->set_name(c->second->name());
          event->set_detail_info(r.name);
        } else {
          event->set_name(r.name);
        }
        event->set_start_ns(r.start_ns);
        event->set_end_ns(r.end_ns);
        event->set_sub_device_id(r.thread_id);
        event->set_device_id(r.device_id);
      }
    }
    miss = find = 0;
    for (const MemRecord &r : mem_records_) {
      auto *event = profile_pb.add_events();
      event->set_type(proto::Event::GPUKernel);
      auto c = correlations_.find(r.correlation_id);
      if (c != correlations_.end() && c->second != nullptr) {
        event->set_name(c->second->name());
        event->set_detail_info(r.name);
        find++;
      } else {
        miss++;
        event->set_name(r.name);
      }
      event->set_start_ns(r.start_ns);
      event->set_end_ns(r.end_ns);
      event->set_sub_device_id(r.stream_id);
      event->set_device_id(r.device_id);
      event->mutable_memcopy()->set_bytes(r.bytes);
    }
    VLOG(1) << "MemRecord event miss: " << miss << " find: " << find;

    for (auto &tmp : mem_info_record_) {
      for (const auto &r : tmp) {
        auto *event = profile_pb.add_mem_events();
        event->set_device_id(0);
        if (platform::is_cpu_place(r.place)) {
          event->set_place(proto::MemEvent::CPUPlace);
        } else if (platform::is_gpu_place(r.place)) {
          event->set_place(proto::MemEvent::CUDAPlace);
          event->set_device_id(
              BOOST_GET_CONST(platform::CUDAPlace, r.place).GetDeviceId());
        } else if (platform::is_cuda_pinned_place(r.place)) {
          event->set_place(proto::MemEvent::CUDAPinnedPlace);
        } else if (platform::is_npu_place(r.place)) {
          event->set_place(proto::MemEvent::NPUPlace);
        } else {
          PADDLE_THROW(platform::errors::Unimplemented(
              "The current place is not supported."));
        }
        event->set_alloc_in(r.alloc_in);
        event->set_free_in(r.free_in);
        event->set_start_ns(r.start_ns);
        event->set_end_ns(r.end_ns);
        event->set_bytes(r.bytes);
        event->set_thread_id(r.thread_id);
      }
    }

    std::ofstream profile_f;
    profile_f.open(profile_path,
                   std::ios::out | std::ios::trunc | std::ios::binary);
    profile_pb.SerializeToOstream(&profile_f);
    profile_f.close();
    return profile_pb;
  }

  void Disable() override {
#ifdef PADDLE_WITH_CUPTI
    // flush might cause additional calls to DeviceTracker.
    CUPTI_CALL(
        dynload::cuptiActivityFlushAll(CUPTI_ACTIVITY_FLAG_FLUSH_FORCED));
#endif  // PADDLE_WITH_CUPTI
    std::lock_guard<std::mutex> l(trace_mu_);
#ifdef PADDLE_WITH_CUPTI
    DisableActivity();
    CUPTI_CALL(dynload::cuptiUnsubscribe(subscriber_));
    CUPTI_CALL(dynload::cuptiGetTimestamp(&end_ns_));
#endif  // PADDLE_WITH_CUPTI
    enabled_ = false;
  }

 private:
#ifdef PADDLE_WITH_CUPTI
  static void CUPTIAPI ApiCallback(void *userdata, CUpti_CallbackDomain domain,
                                   CUpti_CallbackId cbid, const void *cbdata) {
    auto *cbInfo = reinterpret_cast<const CUpti_CallbackData *>(cbdata);
    DeviceTracerImpl *tracer = reinterpret_cast<DeviceTracerImpl *>(userdata);
    if (cbInfo->callbackSite == CUPTI_API_ENTER) {
      Event *event = CurAnnotation();
      tracer->AddAnnotation(cbInfo->correlationId, event);
    }
  }
  CUpti_SubscriberHandle subscriber_;
#endif  // PADDLE_WITH_CUPTI
  std::mutex trace_mu_;
  bool enabled_{false};
  uint64_t start_ns_;
  uint64_t end_ns_;
  std::forward_list<KernelRecord> kernel_records_;
  std::forward_list<MemRecord> mem_records_;
  std::forward_list<std::forward_list<CPURecord>> cpu_records_;
  std::forward_list<std::forward_list<MemInfoRecord>> mem_info_record_;
  std::forward_list<std::forward_list<ActiveKindRecord>> active_kind_records_;
  std::forward_list<std::forward_list<std::pair<uint32_t, Event *>>>
      correlations_pairs;
  std::unordered_map<uint32_t, Event *> correlations_;
};

void CreateTracer(DeviceTracer **t) { *t = new DeviceTracerImpl(); }

DeviceTracer *GetDeviceTracer() {
  std::call_once(tracer_once_flag, CreateTracer, &tracer);
  return tracer;
}

// In order to record PE time, we add main_thread_annotation_stack
// for all event between PE run, we treat it as PE's child Event,
// so when event is not in same thread of PE event, we need add
// father event(PE::run event) for this event
void SetCurAnnotation(Event *event) {
  if (!annotation_stack.empty()) {
    event->set_parent(annotation_stack.back());
    event->set_name(annotation_stack.back()->name() + "/" + event->name());
  }
  if (annotation_stack.empty() && !main_thread_annotation_stack.empty() &&
      main_thread_annotation_stack.back()->thread_id() != event->thread_id()) {
    event->set_parent(main_thread_annotation_stack.back());
    event->set_name(main_thread_annotation_stack.back()->name() + "/" +
                    event->name());
  }
  annotation_stack.push_back(event);

  if (event->role() == EventRole::kSpecial) {
    std::string name = event->name();
    if (!main_thread_annotation_stack_name.empty()) {
      name = main_thread_annotation_stack_name.back() + "/" + event->name();
    }
    main_thread_annotation_stack_name.push_back(name);
    main_thread_annotation_stack.push_back(event);
  }
}

void ClearCurAnnotation() {
  if (!main_thread_annotation_stack.empty()) {
    std::string name = annotation_stack.back()->name();
    std::string main_name = main_thread_annotation_stack.back()->name();
    int main_name_len = main_name.length();
    int name_len = name.length();
    int prefix_len = main_name_len - name_len;

    if ((prefix_len > 0 && main_name.at(prefix_len - 1) == '/' &&
         name == main_name.substr(prefix_len, name_len)) ||
        (name == main_name)) {
      main_thread_annotation_stack_name.pop_back();
      main_thread_annotation_stack.pop_back();
    }
  }
  annotation_stack.pop_back();
}

Event *CurAnnotation() {
  if (annotation_stack.empty()) return nullptr;
  return annotation_stack.back();
}
std::string CurAnnotationName() {
  if (annotation_stack.empty()) return "Unknown";
  return annotation_stack.back()->name();
}

void SetCurBlock(int block_id) { block_id_stack.push_back(block_id); }

void ClearCurBlock() { block_id_stack.pop_back(); }

int BlockDepth() { return block_id_stack.size(); }

uint32_t GetCurSystemThreadId() {
  std::stringstream ss;
  ss << std::this_thread::get_id();
  uint32_t id = static_cast<uint32_t>(std::stoull(ss.str()));
  return id;
}

void RecoreCurThreadId(int32_t id) {
  auto gid = GetCurSystemThreadId();
  VLOG(1) << "RecoreCurThreadId: " << gid << " -> " << id;
  system_thread_id_map[gid] = id;
}

int32_t GetThreadIdFromSystemThreadId(uint32_t id) {
  auto it = system_thread_id_map.find(id);
  if (it != system_thread_id_map.end()) return it->second;
  // return origin id if no event is recorded in this thread.
  return static_cast<int32_t>(id);
}

#ifdef PADDLE_WITH_CUPTI
namespace {

void initCuptiCbidStr() {
  static bool called = false;
  if (called) return;
  called = true;
#define REGISTER_RUNTIME_CBID_STR(cbid) \
  runtime_cbid_str[CUPTI_RUNTIME_TRACE_CBID_##cbid] = #cbid

  REGISTER_RUNTIME_CBID_STR(cudaBindTexture_v3020);
  REGISTER_RUNTIME_CBID_STR(cudaConfigureCall_v3020);
  REGISTER_RUNTIME_CBID_STR(cudaDeviceGetAttribute_v5000);
  REGISTER_RUNTIME_CBID_STR(cudaDeviceGetStreamPriorityRange_v5050);
  REGISTER_RUNTIME_CBID_STR(cudaDeviceSynchronize_v3020);
  REGISTER_RUNTIME_CBID_STR(cudaDriverGetVersion_v3020);
  REGISTER_RUNTIME_CBID_STR(cudaEventCreateWithFlags_v3020);
  REGISTER_RUNTIME_CBID_STR(cudaEventDestroy_v3020);
  REGISTER_RUNTIME_CBID_STR(cudaEventDestroy_v3020);
  REGISTER_RUNTIME_CBID_STR(cudaEventQuery_v3020);
  REGISTER_RUNTIME_CBID_STR(cudaEventRecord_v3020);
  REGISTER_RUNTIME_CBID_STR(cudaFreeHost_v3020);
  REGISTER_RUNTIME_CBID_STR(cudaFree_v3020);
  REGISTER_RUNTIME_CBID_STR(cudaFuncGetAttributes_v3020);
  REGISTER_RUNTIME_CBID_STR(cudaGetDeviceCount_v3020);
  REGISTER_RUNTIME_CBID_STR(cudaGetDeviceProperties_v3020);
  REGISTER_RUNTIME_CBID_STR(cudaGetDevice_v3020);
  REGISTER_RUNTIME_CBID_STR(cudaGetErrorString_v3020);
  REGISTER_RUNTIME_CBID_STR(cudaGetLastError_v3020);
  REGISTER_RUNTIME_CBID_STR(cudaHostAlloc_v3020);
  REGISTER_RUNTIME_CBID_STR(cudaHostGetDevicePointer_v3020);
  REGISTER_RUNTIME_CBID_STR(cudaLaunchKernel_v7000);
  REGISTER_RUNTIME_CBID_STR(cudaMallocHost_v3020);
  REGISTER_RUNTIME_CBID_STR(cudaMalloc_v3020);
  REGISTER_RUNTIME_CBID_STR(cudaMemcpyAsync_v3020);
  REGISTER_RUNTIME_CBID_STR(cudaMemcpy_v3020);
  REGISTER_RUNTIME_CBID_STR(cudaMemsetAsync_v3020);
  REGISTER_RUNTIME_CBID_STR(cudaMemset_v3020);
  REGISTER_RUNTIME_CBID_STR(
      cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags_v7000);
  REGISTER_RUNTIME_CBID_STR(cudaPeekAtLastError_v3020);
  REGISTER_RUNTIME_CBID_STR(cudaRuntimeGetVersion_v3020);
  REGISTER_RUNTIME_CBID_STR(cudaSetDevice_v3020);
  REGISTER_RUNTIME_CBID_STR(cudaStreamCreate_v3020);
  REGISTER_RUNTIME_CBID_STR(cudaStreamCreateWithFlags_v5000);
  REGISTER_RUNTIME_CBID_STR(cudaStreamCreateWithPriority_v5050);
  REGISTER_RUNTIME_CBID_STR(cudaStreamDestroy_v5050);
  REGISTER_RUNTIME_CBID_STR(cudaStreamSynchronize_v3020);
  REGISTER_RUNTIME_CBID_STR(cudaStreamWaitEvent_v3020);
  REGISTER_RUNTIME_CBID_STR(cudaUnbindTexture_v3020);
  REGISTER_RUNTIME_CBID_STR(cudaSetupArgument_v3020);
  REGISTER_RUNTIME_CBID_STR(cudaLaunch_v3020);
  REGISTER_RUNTIME_CBID_STR(cudaDeviceGetPCIBusId_v4010);
#if CUDA_VERSION >= 9000
  REGISTER_RUNTIME_CBID_STR(cudaLaunchCooperativeKernel_v9000);
  REGISTER_RUNTIME_CBID_STR(cudaLaunchCooperativeKernelMultiDevice_v9000);
#endif

#undef REGISTER_RUNTIME_CBID_STR
}
}  // namespace
#endif  // PADDLE_WITH_CUPTI

}  // namespace platform
}  // namespace paddle
