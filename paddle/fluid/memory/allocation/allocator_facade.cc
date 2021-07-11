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

#include "paddle/fluid/memory/allocation/allocator_facade.h"

#include "gflags/gflags.h"
#include "paddle/fluid/memory/allocation/allocator.h"
#include "paddle/fluid/memory/allocation/allocator_strategy.h"
#include "paddle/fluid/memory/allocation/auto_growth_best_fit_allocator.h"
#include "paddle/fluid/memory/allocation/cpu_allocator.h"
#include "paddle/fluid/memory/allocation/naive_best_fit_allocator.h"
#ifdef PADDLE_WITH_ASCEND_CL
#include "paddle/fluid/memory/allocation/npu_pinned_allocator.h"
#endif
#include "paddle/fluid/memory/allocation/retry_allocator.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/fluid/platform/place.h"
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
#include "paddle/fluid/memory/allocation/cuda_allocator.h"
#include "paddle/fluid/memory/allocation/pinned_allocator.h"
#include "paddle/fluid/memory/allocation/thread_local_allocator.h"
#include "paddle/fluid/platform/gpu_info.h"
#endif
#ifdef PADDLE_WITH_XPU
#include "paddle/fluid/platform/xpu_info.h"
#endif
#include "paddle/fluid/platform/npu_info.h"

DEFINE_int64(
    gpu_allocator_retry_time, 10000,
    "The retry time (milliseconds) when allocator fails "
    "to allocate memory. No retry if this value is not greater than 0");

DEFINE_bool(use_system_allocator, false,
            "Whether to use system allocator to allocate CPU and GPU memory. "
            "Only used for unittests.");

namespace paddle {
namespace memory {
namespace allocation {

class AllocatorFacadePrivate {
 public:
  using AllocatorMap = std::map<platform::Place, std::shared_ptr<Allocator>>;

  AllocatorFacadePrivate() {
    auto strategy = GetAllocatorStrategy();
    VLOG(3) << "allocate strategy:" << static_cast<int>(strategy);
    switch (strategy) {
      case AllocatorStrategy::kNaiveBestFit: {
        InitNaiveBestFitCPUAllocator();
#ifdef PADDLE_WITH_XPU
        for (int dev_id = 0; dev_id < platform::GetXPUDeviceCount(); ++dev_id) {
          InitNaiveBestFitXPUAllocator(platform::XPUPlace(dev_id));
        }
#endif
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
        for (int dev_id = 0; dev_id < platform::GetCUDADeviceCount();
             ++dev_id) {
          InitNaiveBestFitCUDAAllocator(platform::CUDAPlace(dev_id));
        }
        InitNaiveBestFitCUDAPinnedAllocator();
#endif
#ifdef PADDLE_WITH_ASCEND_CL
        VLOG(3) << "device_count:" << platform::GetNPUDeviceCount();
        for (int dev_id = 0; dev_id < platform::GetNPUDeviceCount(); ++dev_id) {
          InitNaiveBestFitNPUAllocator(platform::NPUPlace(dev_id));
        }
        InitNaiveBestFitNPUPinnedAllocator();
#endif
        break;
      }

      case AllocatorStrategy::kAutoGrowth: {
        InitNaiveBestFitCPUAllocator();
#ifdef PADDLE_WITH_XPU
        for (int dev_id = 0; dev_id < platform::GetXPUDeviceCount(); ++dev_id) {
          InitNaiveBestFitXPUAllocator(platform::XPUPlace(dev_id));
        }
#endif
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
        for (int dev_id = 0; dev_id < platform::GetCUDADeviceCount();
             ++dev_id) {
          InitAutoGrowthCUDAAllocator(platform::CUDAPlace(dev_id));
        }
        InitNaiveBestFitCUDAPinnedAllocator();
#endif
        break;
      }

      case AllocatorStrategy::kThreadLocal: {
        InitNaiveBestFitCPUAllocator();
#ifdef PADDLE_WITH_XPU
        for (int dev_id = 0; dev_id < platform::GetXPUDeviceCount(); ++dev_id) {
          InitNaiveBestFitXPUAllocator(platform::XPUPlace(dev_id));
        }
#endif
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
        for (int dev_id = 0; dev_id < platform::GetCUDADeviceCount();
             ++dev_id) {
          InitThreadLocalCUDAAllocator(platform::CUDAPlace(dev_id));
        }
        InitNaiveBestFitCUDAPinnedAllocator();
#endif
        break;
      }

      default: {
        PADDLE_THROW(platform::errors::InvalidArgument(
            "Unsupported allocator strategy: %d", static_cast<int>(strategy)));
      }
    }
    InitZeroSizeAllocators();
    InitSystemAllocators();

    if (FLAGS_gpu_allocator_retry_time > 0) {
      WrapCUDARetryAllocator(FLAGS_gpu_allocator_retry_time);
    }

    CheckAllocThreadSafe();
  }

  inline const std::shared_ptr<Allocator>& GetAllocator(
      const platform::Place& place, size_t size) {
    const auto& allocators =
        (size > 0 ? (UNLIKELY(FLAGS_use_system_allocator) ? system_allocators_
                                                          : allocators_)
                  : zero_size_allocators_);

    VLOG(10) << "allocators_ size:" << allocators_.size() << ", find place:" << place << ", size:" << size;
    for (auto it=allocators_.begin(); it!=allocators_.end(); ++it){
        VLOG(10) << "allocators_ key:" << it->first;
    }

    auto iter = allocators.find(place);
    PADDLE_ENFORCE_NE(iter, allocators.end(),
                      platform::errors::NotFound(
                          "No allocator found for the place, %s", place));
    return iter->second;
  }

 private:
  void InitSystemAllocators() {
    system_allocators_[platform::CPUPlace()] = std::make_shared<CPUAllocator>();
#ifdef PADDLE_WITH_XPU
    int device_count = platform::GetXPUDeviceCount();
    for (int i = 0; i < device_count; ++i) {
      platform::XPUPlace p(i);
      system_allocators_[p] = std::make_shared<NaiveBestFitAllocator>(p);
    }
#endif
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
    system_allocators_[platform::CUDAPinnedPlace()] =
        std::make_shared<CPUPinnedAllocator>();
    int device_count = platform::GetCUDADeviceCount();
    for (int i = 0; i < device_count; ++i) {
      platform::CUDAPlace p(i);
      system_allocators_[p] = std::make_shared<CUDAAllocator>(p);
    }
#endif
  }

  void InitNaiveBestFitCPUAllocator() {
    allocators_[platform::CPUPlace()] =
        std::make_shared<NaiveBestFitAllocator>(platform::CPUPlace());
  }

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  void InitNaiveBestFitCUDAPinnedAllocator() {
    allocators_[platform::CUDAPinnedPlace()] =
        std::make_shared<NaiveBestFitAllocator>(platform::CUDAPinnedPlace());
  }

  void InitNaiveBestFitCUDAAllocator(platform::CUDAPlace p) {
    allocators_[p] = std::make_shared<NaiveBestFitAllocator>(p);
  }

  void InitThreadLocalCUDAAllocator(platform::CUDAPlace p) {
    allocators_[p] = std::make_shared<ThreadLocalCUDAAllocator>(p);
  }

  void InitAutoGrowthCUDAAllocator(platform::CUDAPlace p) {
    auto cuda_allocator = std::make_shared<CUDAAllocator>(p);
    allocators_[p] = std::make_shared<AutoGrowthBestFitAllocator>(
        cuda_allocator, platform::GpuMinChunkSize());
  }
#endif

#ifdef PADDLE_WITH_XPU
  void InitNaiveBestFitXPUAllocator(platform::XPUPlace p) {
    allocators_[p] = std::make_shared<NaiveBestFitAllocator>(p);
  }
#endif

#ifdef PADDLE_WITH_ASCEND_CL
  void InitNaiveBestFitNPUAllocator(platform::NPUPlace p) {
    VLOG(4) << "allocators_[" << p << "]";
    allocators_[p] = std::make_shared<NaiveBestFitAllocator>(p);
  }

  void InitNaiveBestFitNPUPinnedAllocator() {
    allocators_[platform::NPUPinnedPlace()] =
        std::make_shared<paddle::memory::allocation::NPUPinnedAllocator>();
  }

#endif

  class ZeroSizeAllocator : public Allocator {
   public:
    explicit ZeroSizeAllocator(platform::Place place) : place_(place) {}

    bool IsAllocThreadSafe() const override { return true; }

   protected:
    Allocation* AllocateImpl(size_t size) override {
      return new Allocation(nullptr, 0, place_);
    }

    void FreeImpl(Allocation* allocation) override { delete allocation; }

   private:
    platform::Place place_;
  };

  void InitZeroSizeAllocators() {
    std::vector<platform::Place> places;
    places.emplace_back(platform::CPUPlace());
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
    int device_count = platform::GetCUDADeviceCount();
    for (int dev_id = 0; dev_id < device_count; ++dev_id) {
      places.emplace_back(platform::CUDAPlace(dev_id));
    }
    places.emplace_back(platform::CUDAPinnedPlace());
#endif
#ifdef PADDLE_WITH_XPU
    int device_count = platform::GetXPUDeviceCount();
    for (int dev_id = 0; dev_id < device_count; ++dev_id) {
      places.emplace_back(platform::XPUPlace(dev_id));
    }
#endif

    for (auto& p : places) {
      zero_size_allocators_[p] = std::make_shared<ZeroSizeAllocator>(p);
    }
  }

  static void CheckAllocThreadSafe(const AllocatorMap& allocators) {
    for (auto& pair : allocators) {
      PADDLE_ENFORCE_EQ(pair.second->IsAllocThreadSafe(), true,
                        platform::errors::InvalidArgument(
                            "Public allocators must be thread safe"));
    }
  }

  void CheckAllocThreadSafe() const {
    CheckAllocThreadSafe(allocators_);
    CheckAllocThreadSafe(zero_size_allocators_);
    CheckAllocThreadSafe(system_allocators_);
  }

  void WrapCUDARetryAllocator(size_t retry_time) {
    PADDLE_ENFORCE_GT(
        retry_time, 0,
        platform::errors::InvalidArgument(
            "Retry time should be larger than 0, but got %d", retry_time));
    for (auto& pair : allocators_) {
      if (platform::is_gpu_place(pair.first)) {
        pair.second = std::make_shared<RetryAllocator>(pair.second, retry_time);
      }
    }
  }

 private:
  AllocatorMap allocators_;
  AllocatorMap zero_size_allocators_;
  AllocatorMap system_allocators_;
};

// Pimpl. Make interface clean.
AllocatorFacade::AllocatorFacade() : m_(new AllocatorFacadePrivate()) {}
// delete m_ may cause core dump when the destructor of python in conflict with
// cpp.
AllocatorFacade::~AllocatorFacade() {}

AllocatorFacade& AllocatorFacade::Instance() {
  static AllocatorFacade instance;
  return instance;
}

std::shared_ptr<Allocation> AllocatorFacade::AllocShared(
    const platform::Place& place, size_t size) {
  return std::shared_ptr<Allocation>(Alloc(place, size));
}

AllocationPtr AllocatorFacade::Alloc(const platform::Place& place,
                                     size_t size) {
  return m_->GetAllocator(place, size)->Allocate(size);
}

uint64_t AllocatorFacade::Release(const platform::Place& place) {
  return m_->GetAllocator(place, /* A non-zero num to choose allocator_ */ 1)
      ->Release(place);
}

const std::shared_ptr<Allocator>& AllocatorFacade::GetAllocator(
    const platform::Place& place) {
  return m_->GetAllocator(place, /* A non-zero num to choose allocator_ */ 1);
}

}  // namespace allocation
}  // namespace memory
}  // namespace paddle
