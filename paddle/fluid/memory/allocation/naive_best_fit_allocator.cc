// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/memory/allocation/naive_best_fit_allocator.h"

#include <mutex>

#include "gflags/gflags.h"
#include "glog/logging.h"
#include "paddle/fluid/memory/detail/buddy_allocator.h"
#include "paddle/fluid/memory/detail/system_allocator.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/fluid/platform/gpu_info.h"
#include "paddle/fluid/platform/npu_info.h"
#include "paddle/fluid/platform/profiler.h"

#include "paddle/fluid/string/printf.h"
#include "paddle/fluid/string/split.h"
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
#include "paddle/fluid/platform/cuda_device_guard.h"
#endif
#ifdef PADDLE_WITH_XPU
#include "paddle/fluid/platform/xpu_header.h"
#endif

DEFINE_bool(init_allocated_mem, false,
            "It is a mistake that the values of the memory allocated by "
            "BuddyAllocator are always zeroed in some op's implementation. "
            "To find this error in time, we use init_allocated_mem to indicate "
            "that initializing the allocated memory with a small value "
            "during unit testing.");
DECLARE_double(fraction_of_gpu_memory_to_use);
DECLARE_uint64(initial_gpu_memory_in_mb);
DECLARE_uint64(reallocate_gpu_memory_in_mb);
DECLARE_bool(benchmark);

namespace paddle {
namespace memory {
namespace legacy {
template <typename Place>
void *Alloc(const Place &place, size_t size);

template <typename Place>
void Free(const Place &place, void *p, size_t size);

template <typename Place>
uint64_t Release(const Place &place);

template <typename Place>
size_t Used(const Place &place);

struct Usage : public boost::static_visitor<size_t> {
  size_t operator()(const platform::CPUPlace &cpu) const;
  size_t operator()(const platform::CUDAPlace &gpu) const;
  size_t operator()(const platform::CUDAPinnedPlace &cuda_pinned) const;
};

size_t memory_usage(const platform::Place &p);

using BuddyAllocator = detail::BuddyAllocator;

BuddyAllocator *GetCPUBuddyAllocator() {
  // We tried thread_local for inference::RNN1 model, but that not works much
  // for multi-thread test.
  static std::once_flag init_flag;
  static detail::BuddyAllocator *a = nullptr;

  std::call_once(init_flag, []() {
    a = new detail::BuddyAllocator(
        std::unique_ptr<detail::SystemAllocator>(new detail::CPUAllocator),
        platform::CpuMinChunkSize(), platform::CpuMaxChunkSize());
  });

  return a;
}

template <>
void *Alloc<platform::CPUPlace>(const platform::CPUPlace &place, size_t size) {
  VLOG(10) << "Allocate " << size << " bytes on " << platform::Place(place);
  void *p = GetCPUBuddyAllocator()->Alloc(size);
  if (FLAGS_init_allocated_mem) {
    memset(p, 0xEF, size);
  }
  VLOG(10) << "  pointer=" << p;
  return p;
}

template <>
void Free<platform::CPUPlace>(const platform::CPUPlace &place, void *p,
                              size_t size) {
  VLOG(10) << "Free pointer=" << p << " on " << platform::Place(place);
  GetCPUBuddyAllocator()->Free(p);
}

template <>
uint64_t Release<platform::CPUPlace>(const platform::CPUPlace &place) {
  return GetCPUBuddyAllocator()->Release();
}

template <>
size_t Used<platform::CPUPlace>(const platform::CPUPlace &place) {
  return GetCPUBuddyAllocator()->Used();
}

// For kunlun XPU
#ifdef PADDLE_WITH_XPU
class XPUBuddyAllocatorList {
 private:
  XPUBuddyAllocatorList() : devices_(platform::GetXPUSelectedDevices()) {
    auto xpu_num = devices_.size();
    allocators_.resize(xpu_num);
    init_flags_.reserve(xpu_num);
    for (size_t i = 0; i < xpu_num; ++i) {
      init_flags_.emplace_back(new std::once_flag());
    }
  }

  static XPUBuddyAllocatorList *CreateNewInstance() {
    return new XPUBuddyAllocatorList();
  }

 public:
  static XPUBuddyAllocatorList *Instance() {
    static auto *instance = CreateNewInstance();
    return instance;
  }

  BuddyAllocator *Get(int xpu_id) {
    auto pos = std::distance(
        devices_.begin(), std::find(devices_.begin(), devices_.end(), xpu_id));
    PADDLE_ENFORCE_LT(pos, devices_.size(),
                      platform::errors::OutOfRange(
                          "The index exceeds the size of devices, the size of "
                          "devices is %d, the index is %d",
                          devices_.size(), pos));

    std::call_once(*init_flags_[pos], [this, pos, xpu_id] {
      platform::XPUDeviceGuard guard(xpu_id);
      allocators_[pos].reset(new BuddyAllocator(
          std::unique_ptr<detail::SystemAllocator>(
              new detail::XPUAllocator(devices_[pos])),
          platform::XPUMinChunkSize(), platform::XPUMaxChunkSize()));
    });

    return allocators_[pos].get();
  }

 private:
  std::vector<int> devices_;
  std::vector<std::unique_ptr<std::once_flag>> init_flags_;
  std::vector<std::unique_ptr<BuddyAllocator>> allocators_;
};

BuddyAllocator *GetXPUBuddyAllocator(int xpu_id) {
  return XPUBuddyAllocatorList::Instance()->Get(xpu_id);
}
#endif

template <>
void *Alloc<platform::XPUPlace>(const platform::XPUPlace &place, size_t size) {
#ifdef PADDLE_WITH_XPU
  auto *buddy_allocator = GetXPUBuddyAllocator(place.device);
  auto *ptr = buddy_allocator->Alloc(size);
  if (ptr == nullptr) {
    PADDLE_THROW(platform::errors::ResourceExhausted(
        "Cannot allocate %s in XPU %d.", string::HumanReadableSize(size),
        place.device));
  } else {
    if (FLAGS_init_allocated_mem) {
      PADDLE_THROW(platform::errors::Unimplemented(
          "xpu memory FLAGS_init_allocated_mem is not implemented."));
    }
  }
  VLOG(10) << "xpu alloc pointer=" << ptr;
  return ptr;
#else
  PADDLE_THROW(
      platform::errors::PermissionDenied("'XPUPlace' is not supported."));
  return nullptr;
#endif
}

template <>
void Free<platform::XPUPlace>(const platform::XPUPlace &place, void *p,
                              size_t size) {
#ifdef PADDLE_WITH_XPU
  platform::XPUDeviceGuard(place.device);
  xpu_wait();
  GetXPUBuddyAllocator(place.device)->Free(p);
#else
  PADDLE_THROW(
      platform::errors::PermissionDenied("'XPUPlace' is not supported."));
#endif
}

template <>
uint64_t Release<platform::XPUPlace>(const platform::XPUPlace &place) {
#ifdef PADDLE_WITH_XPU
  return GetXPUBuddyAllocator(place.device)->Release();
#else
  PADDLE_THROW(
      platform::errors::PermissionDenied("'XPUPlace' is not supported."));
#endif
}

template <>
size_t Used<platform::XPUPlace>(const platform::XPUPlace &place) {
#ifdef PADDLE_WITH_XPU
  return GetXPUBuddyAllocator(place.device)->Used();
#else
  PADDLE_THROW(platform::errors::PermissionDenied(
      "'XPUPlace' is not supported in CPU only device."));
#endif
}

// For Ascend NPU
#ifdef PADDLE_WITH_ASCEND_CL
class NPUBuddyAllocatorList {
 private:
  NPUBuddyAllocatorList() : devices_(platform::GetSelectedNPUDevices()) {
    auto npu_num = devices_.size();
    allocators_.resize(npu_num);
    init_flags_.reserve(npu_num);
    for (size_t i = 0; i < npu_num; ++i) {
      init_flags_.emplace_back(new std::once_flag());
    }
  }

  static NPUBuddyAllocatorList *CreateNewInstance() {
    return new NPUBuddyAllocatorList();
  }

 public:
  static NPUBuddyAllocatorList *Instance() {
    static auto *instance = CreateNewInstance();
    return instance;
  }

  BuddyAllocator *Get(int npu_id) {
    auto pos = std::distance(
        devices_.begin(), std::find(devices_.begin(), devices_.end(), npu_id));
    PADDLE_ENFORCE_LT(pos, devices_.size(),
                      platform::errors::OutOfRange(
                          "The index exceeds the size of devices, the size of "
                          "devices is %d, the index is %d",
                          devices_.size(), pos));

    std::call_once(*init_flags_[pos], [this, pos] {
      platform::SetNPUDeviceId(devices_[pos]);
      allocators_[pos].reset(new BuddyAllocator(
          std::unique_ptr<detail::SystemAllocator>(
              new detail::NPUAllocator(devices_[pos])),
          platform::NPUMinChunkSize(), platform::NPUMaxChunkSize()));
      VLOG(10) << "\n\nNOTE:\n"
               << "You can set GFlags environment variable "
               << "'FLAGS_fraction_of_gpu_memory_to_use' "
               << "or 'FLAGS_initial_gpu_memory_in_mb' "
               << "or 'FLAGS_reallocate_gpu_memory_in_mb' "
               << "to change the memory size for GPU usage.\n"
               << "Current 'FLAGS_fraction_of_gpu_memory_to_use' value is "
               << FLAGS_fraction_of_gpu_memory_to_use
               << ". Current 'FLAGS_initial_gpu_memory_in_mb' value is "
               << FLAGS_initial_gpu_memory_in_mb
               << ". Current 'FLAGS_reallocate_gpu_memory_in_mb' value is "
               << FLAGS_reallocate_gpu_memory_in_mb << "\n\n";
    });

    return allocators_[pos].get();
  }

 private:
  std::vector<int> devices_;
  std::vector<std::unique_ptr<std::once_flag>> init_flags_;
  std::vector<std::unique_ptr<BuddyAllocator>> allocators_;
};

BuddyAllocator *GetNPUBuddyAllocator(int npu_id) {
  return NPUBuddyAllocatorList::Instance()->Get(npu_id);
}

BuddyAllocator *GetNPUPinnedBuddyAllocator() {
  static std::once_flag init_flag;
  static BuddyAllocator *ba = nullptr;

  std::call_once(init_flag, []() {
    ba = new BuddyAllocator(std::unique_ptr<detail::SystemAllocator>(
                                new detail::NPUPinnedAllocator),
                            platform::NPUPinnedMinChunkSize(),
                            platform::NPUPinnedMaxChunkSize());
  });

  return ba;
}

#endif

template <>
size_t Used<platform::NPUPlace>(const platform::NPUPlace &place) {
#ifdef PADDLE_WITH_ASCEND_CL
  return GetNPUBuddyAllocator(place.device)->Used();
#else
  PADDLE_THROW(platform::errors::PermissionDenied(
      "'NPUPlace' is not supported in CPU only device."));
#endif
}

template <>
void *Alloc<platform::NPUPlace>(const platform::NPUPlace &place, size_t size) {
#ifdef PADDLE_WITH_ASCEND_CL
  auto *buddy_allocator = GetNPUBuddyAllocator(place.device);
  auto *ptr = buddy_allocator->Alloc(size);
  if (ptr == nullptr) {
    platform::NPUDeviceGuard(place.device);
    size_t avail, total;
    platform::NPUMemoryUsage(&avail, &total);
    PADDLE_THROW(platform::errors::ResourceExhausted(
        "Cannot allocate %s in GPU %d, avaliable %s, total %s, GpuMinChunkSize "
        "%s, GpuMaxChunkSize %s, GPU memory used: %s.",
        string::HumanReadableSize(size), place.device,
        string::HumanReadableSize(avail), string::HumanReadableSize(total),
        string::HumanReadableSize(buddy_allocator->GetMinChunkSize()),
        string::HumanReadableSize(buddy_allocator->GetMaxChunkSize()),
        string::HumanReadableSize(Used<platform::NPUPlace>(place))));
  } else {
    if (FLAGS_init_allocated_mem) {
      aclrtMemset(ptr, size, 0xEF, size);
    }
  }
  VLOG(10) << "Allocate " << size << " bytes on " << platform::Place(place);
  return ptr;
#else
  PADDLE_THROW(platform::errors::PermissionDenied(
      "'NPUPlace' is not supported in CPU only device."));
#endif
}

template <>
void Free<platform::NPUPlace>(const platform::NPUPlace &place, void *p,
                              size_t size) {
#ifdef PADDLE_WITH_ASCEND_CL
  VLOG(10) << "Free pointer=" << p << " on " << platform::Place(place);
  GetNPUBuddyAllocator(place.device)->Free(p);
#else
  PADDLE_THROW(platform::errors::PermissionDenied(
      "'NPUPlace' is not supported in CPU only device."));
#endif
}

template <>
uint64_t Release<platform::NPUPlace>(const platform::NPUPlace &place) {
#ifdef PADDLE_WITH_ASCEND_CL
  return GetNPUBuddyAllocator(place.device)->Release();
#else
  PADDLE_THROW(platform::errors::PermissionDenied(
      "'NPUPlace' is not supported in CPU only device."));
#endif
}

template <>
size_t Used<platform::NPUPinnedPlace>(const platform::NPUPinnedPlace &place) {
#ifdef PADDLE_WITH_ASCEND_CL
  return GetNPUPinnedBuddyAllocator()->Used();
#else
  PADDLE_THROW(platform::errors::PermissionDenied(
      "'NPUPinnedPlace' is not supported in CPU only device."));
#endif
}

template <>
void *Alloc<platform::NPUPinnedPlace>(const platform::NPUPinnedPlace &place,
                                      size_t size) {
#ifdef PADDLE_WITH_ASCEND_CL
  auto *buddy_allocator = GetNPUPinnedBuddyAllocator();
  void *ptr = buddy_allocator->Alloc(size);

  if (ptr == nullptr) {
    LOG(WARNING) << "aclrtMallocHost Cannot allocate " << size
                 << " bytes in NPUPinnedPlace";
  }
  if (FLAGS_init_allocated_mem) {
    memset(ptr, 0xEF, size);
  }
  return ptr;
#else
  PADDLE_THROW(platform::errors::PermissionDenied(
      "'NPUPinnedPlace' is not supported in CPU only device."));
#endif
}

template <>
void Free<platform::NPUPinnedPlace>(const platform::NPUPinnedPlace &place,
                                    void *p, size_t size) {
#ifdef PADDLE_WITH_ASCEND_CL
  GetNPUPinnedBuddyAllocator()->Free(p);
#else
  PADDLE_THROW(platform::errors::PermissionDenied(
      "'NPUPinnedPlace' is not supported in CPU only device."));
#endif
}

template <>
uint64_t Release<platform::NPUPinnedPlace>(
    const platform::NPUPinnedPlace &place) {
#ifdef PADDLE_WITH_ASCEND_CL
  return GetNPUPinnedBuddyAllocator()->Release();
#else
  PADDLE_THROW(platform::errors::PermissionDenied(
      "'NPUPinnedPlace' is not supported in CPU only device."));
#endif
}

// For CUDA
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
class GPUBuddyAllocatorList {
 private:
  GPUBuddyAllocatorList() : devices_(platform::GetSelectedDevices()) {
    auto gpu_num = devices_.size();
    allocators_.resize(gpu_num);
    init_flags_.reserve(gpu_num);
    for (size_t i = 0; i < gpu_num; ++i) {
      init_flags_.emplace_back(new std::once_flag());
    }
  }

  static GPUBuddyAllocatorList *CreateNewInstance() {
    return new GPUBuddyAllocatorList();
  }

 public:
  static GPUBuddyAllocatorList *Instance() {
    static auto *instance = CreateNewInstance();
    return instance;
  }

  BuddyAllocator *Get(int gpu_id) {
    auto pos = std::distance(
        devices_.begin(), std::find(devices_.begin(), devices_.end(), gpu_id));
    PADDLE_ENFORCE_LT(pos, devices_.size(),
                      platform::errors::OutOfRange(
                          "The index exceeds the size of devices, the size of "
                          "devices is %d, the index is %d",
                          devices_.size(), pos));

    std::call_once(*init_flags_[pos], [this, pos] {
      platform::SetDeviceId(devices_[pos]);
      allocators_[pos].reset(new BuddyAllocator(
          std::unique_ptr<detail::SystemAllocator>(
              new detail::GPUAllocator(devices_[pos])),
          platform::GpuMinChunkSize(), platform::GpuMaxChunkSize()));
      VLOG(10) << "\n\nNOTE:\n"
               << "You can set GFlags environment variable "
               << "'FLAGS_fraction_of_gpu_memory_to_use' "
               << "or 'FLAGS_initial_gpu_memory_in_mb' "
               << "or 'FLAGS_reallocate_gpu_memory_in_mb' "
               << "to change the memory size for GPU usage.\n"
               << "Current 'FLAGS_fraction_of_gpu_memory_to_use' value is "
               << FLAGS_fraction_of_gpu_memory_to_use
               << ". Current 'FLAGS_initial_gpu_memory_in_mb' value is "
               << FLAGS_initial_gpu_memory_in_mb
               << ". Current 'FLAGS_reallocate_gpu_memory_in_mb' value is "
               << FLAGS_reallocate_gpu_memory_in_mb << "\n\n";
    });

    return allocators_[pos].get();
  }

 private:
  std::vector<int> devices_;
  std::vector<std::unique_ptr<std::once_flag>> init_flags_;
  std::vector<std::unique_ptr<BuddyAllocator>> allocators_;
};

BuddyAllocator *GetGPUBuddyAllocator(int gpu_id) {
  return GPUBuddyAllocatorList::Instance()->Get(gpu_id);
}
#endif

template <>
size_t Used<platform::CUDAPlace>(const platform::CUDAPlace &place) {
#if (defined PADDLE_WITH_CUDA || defined PADDLE_WITH_HIP)
  return GetGPUBuddyAllocator(place.device)->Used();
#else
  PADDLE_THROW(platform::errors::PermissionDenied(
      "'CUDAPlace' is not supported in CPU only device."));
#endif
}

template <>
void *Alloc<platform::CUDAPlace>(const platform::CUDAPlace &place,
                                 size_t size) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  auto *buddy_allocator = GetGPUBuddyAllocator(place.device);
  auto *ptr = buddy_allocator->Alloc(size);
  if (ptr == nullptr) {
    platform::CUDADeviceGuard(place.device);
    size_t avail, total;
    platform::GpuMemoryUsage(&avail, &total);
    PADDLE_THROW(platform::errors::ResourceExhausted(
        "Cannot allocate %s in GPU %d, avaliable %s, total %s, GpuMinChunkSize "
        "%s, GpuMaxChunkSize %s, GPU memory used: %s.",
        string::HumanReadableSize(size), place.device,
        string::HumanReadableSize(avail), string::HumanReadableSize(total),
        string::HumanReadableSize(buddy_allocator->GetMinChunkSize()),
        string::HumanReadableSize(buddy_allocator->GetMaxChunkSize()),
        string::HumanReadableSize(Used<platform::CUDAPlace>(place))));
  } else {
    if (FLAGS_init_allocated_mem) {
#ifdef PADDLE_WITH_HIP
      hipMemset(ptr, 0xEF, size);
#else
      cudaMemset(ptr, 0xEF, size);
#endif
    }
  }
  return ptr;
#else
  PADDLE_THROW(platform::errors::PermissionDenied(
      "'CUDAPlace' is not supported in CPU only device."));
#endif
}

template <>
void Free<platform::CUDAPlace>(const platform::CUDAPlace &place, void *p,
                               size_t size) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  GetGPUBuddyAllocator(place.device)->Free(p);
#else
  PADDLE_THROW(platform::errors::PermissionDenied(
      "'CUDAPlace' is not supported in CPU only device."));
#endif
}

template <>
uint64_t Release<platform::CUDAPlace>(const platform::CUDAPlace &place) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  return GetGPUBuddyAllocator(place.device)->Release();
#else
  PADDLE_THROW(platform::errors::PermissionDenied(
      "'CUDAPlace' is not supported in CPU only device."));
#endif
}

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
BuddyAllocator *GetCUDAPinnedBuddyAllocator() {
  static std::once_flag init_flag;
  static BuddyAllocator *ba = nullptr;

  std::call_once(init_flag, []() {
    ba = new BuddyAllocator(std::unique_ptr<detail::SystemAllocator>(
                                new detail::CUDAPinnedAllocator),
                            platform::CUDAPinnedMinChunkSize(),
                            platform::CUDAPinnedMaxChunkSize());
  });

  return ba;
}
#endif

template <>
size_t Used<platform::CUDAPinnedPlace>(const platform::CUDAPinnedPlace &place) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  return GetCUDAPinnedBuddyAllocator()->Used();
#else
  PADDLE_THROW(platform::errors::PermissionDenied(
      "'CUDAPinnedPlace' is not supported in CPU only device."));
#endif
}

template <>
void *Alloc<platform::CUDAPinnedPlace>(const platform::CUDAPinnedPlace &place,
                                       size_t size) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  auto *buddy_allocator = GetCUDAPinnedBuddyAllocator();
  void *ptr = buddy_allocator->Alloc(size);

  if (ptr == nullptr) {
    LOG(WARNING) << "cudaHostAlloc Cannot allocate " << size
                 << " bytes in CUDAPinnedPlace";
  }
  if (FLAGS_init_allocated_mem) {
    memset(ptr, 0xEF, size);
  }
  return ptr;
#else
  PADDLE_THROW(platform::errors::PermissionDenied(
      "'CUDAPinnedPlace' is not supported in CPU only device."));
#endif
}

template <>
void Free<platform::CUDAPinnedPlace>(const platform::CUDAPinnedPlace &place,
                                     void *p, size_t size) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  GetCUDAPinnedBuddyAllocator()->Free(p);
#else
  PADDLE_THROW(platform::errors::PermissionDenied(
      "'CUDAPinnedPlace' is not supported in CPU only device."));
#endif
}

template <>
uint64_t Release<platform::CUDAPinnedPlace>(
    const platform::CUDAPinnedPlace &place) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  return GetCUDAPinnedBuddyAllocator()->Release();
#else
  PADDLE_THROW(platform::errors::PermissionDenied(
      "'CUDAPinnedPlace' is not supported in CPU only device."));
#endif
}

struct AllocVisitor : public boost::static_visitor<void *> {
  inline explicit AllocVisitor(size_t size) : size_(size) {}

  template <typename Place>
  inline void *operator()(const Place &place) const {
    return Alloc<Place>(place, size_);
  }

 private:
  size_t size_;
};

struct FreeVisitor : public boost::static_visitor<void> {
  inline explicit FreeVisitor(void *ptr, size_t size)
      : ptr_(ptr), size_(size) {}

  template <typename Place>
  inline void operator()(const Place &place) const {
    Free<Place>(place, ptr_, size_);
  }

 private:
  void *ptr_;
  size_t size_;
};

struct ReleaseVisitor : public boost::static_visitor<uint64_t> {
  template <typename Place>
  inline uint64_t operator()(const Place &place) const {
    return Release<Place>(place);
  }
};

size_t Usage::operator()(const platform::CPUPlace &cpu) const {
  return Used(cpu);
}

size_t Usage::operator()(const platform::CUDAPlace &gpu) const {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  return Used(gpu);
#else
  PADDLE_THROW(platform::errors::PermissionDenied(
      "'CUDAPlace' is not supported in CPU only device."));
#endif
}

size_t Usage::operator()(const platform::CUDAPinnedPlace &cuda_pinned) const {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  return Used(cuda_pinned);
#else
  PADDLE_THROW(platform::errors::PermissionDenied(
      "'CUDAPinnedPlace' is not supported in CPU only device."));
#endif
}
}  // namespace legacy

namespace allocation {

Allocation *NaiveBestFitAllocator::AllocateImpl(size_t size) {
  void *ptr = boost::apply_visitor(legacy::AllocVisitor(size), place_);
  auto *tmp_alloc = new Allocation(ptr, size, place_);
  platform::MemEvenRecorder::Instance().PushMemRecord(
      static_cast<void *>(tmp_alloc), place_, size);
  return tmp_alloc;
}

void NaiveBestFitAllocator::FreeImpl(Allocation *allocation) {
  boost::apply_visitor(
      legacy::FreeVisitor(allocation->ptr(), allocation->size()),
      allocation->place());
  platform::MemEvenRecorder::Instance().PopMemRecord(
      static_cast<void *>(allocation), place_);
  delete allocation;
}

uint64_t NaiveBestFitAllocator::ReleaseImpl(const platform::Place &place) {
  return boost::apply_visitor(legacy::ReleaseVisitor(), place);
}

}  // namespace allocation
}  // namespace memory
}  // namespace paddle
