/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */
#define GLOG_NO_ABBREVIATED_SEVERITIES

#include "paddle/fluid/memory/detail/system_allocator.h"

#ifdef _WIN32
#include <malloc.h>
#ifndef NOMINMAX
#define NOMINMAX  // msvc max/min macro conflict with std::min/max
#endif
#include <windows.h>  // VirtualLock/VirtualUnlock
#else
#include <sys/mman.h>  // for mlock and munlock
#endif
#include "gflags/gflags.h"
#include "paddle/fluid/memory/allocation/allocator.h"
#include "paddle/fluid/platform/cpu_info.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/fluid/platform/gpu_info.h"
#include "paddle/fluid/platform/npu_info.h"

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
#include "paddle/fluid/platform/cuda_device_guard.h"
#endif
#ifdef PADDLE_WITH_XPU
#include "paddle/fluid/platform/xpu_header.h"
#include "paddle/fluid/platform/xpu_info.h"
#endif

DECLARE_bool(use_pinned_memory);
DECLARE_double(fraction_of_gpu_memory_to_use);
DECLARE_uint64(initial_gpu_memory_in_mb);
DECLARE_uint64(reallocate_gpu_memory_in_mb);

namespace paddle {
namespace memory {
namespace detail {

void* AlignedMalloc(size_t size) {
  void* p = nullptr;
  size_t alignment = 32ul;
#ifdef PADDLE_WITH_MKLDNN
  // refer to https://github.com/01org/mkl-dnn/blob/master/include/mkldnn.hpp
  // memory alignment
  alignment = 4096ul;
#endif
#ifdef _WIN32
  p = _aligned_malloc(size, alignment);
#else
  int error = posix_memalign(&p, alignment, size);
  PADDLE_ENFORCE_EQ(
      error, 0,
      platform::errors::ResourceExhausted(
          "Fail to alloc memory of %ld size, error code is %d.", size, error));
#endif
  PADDLE_ENFORCE_NOT_NULL(p, platform::errors::ResourceExhausted(
                                 "Fail to alloc memory of %ld size.", size));
  return p;
}

void* CPUAllocator::Alloc(size_t* index, size_t size) {
  // According to http://www.cplusplus.com/reference/cstdlib/malloc/,
  // malloc might not return nullptr if size is zero, but the returned
  // pointer shall not be dereferenced -- so we make it nullptr.
  if (size <= 0) return nullptr;

  *index = 0;  // unlock memory

  void* p = AlignedMalloc(size);

  if (p != nullptr) {
    if (FLAGS_use_pinned_memory) {
      *index = 1;
#ifdef _WIN32
      VirtualLock(p, size);
#else
      mlock(p, size);  // lock memory
#endif
    }
  }

  return p;
}

void CPUAllocator::Free(void* p, size_t size, size_t index) {
  if (p != nullptr && index == 1) {
#ifdef _WIN32
    VirtualUnlock(p, size);
#else
    munlock(p, size);
#endif
  }
#ifdef _WIN32
  _aligned_free(p);
#else
  free(p);
#endif
}

bool CPUAllocator::UseGpu() const { return false; }

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)

void* GPUAllocator::Alloc(size_t* index, size_t size) {
  // CUDA documentation doesn't explain if cudaMalloc returns nullptr
  // if size is 0.  We just make sure it does.
  if (size <= 0) return nullptr;

  void* p;
  auto result = platform::RecordedCudaMalloc(&p, size, gpu_id_);

  if (result == gpuSuccess) {
    *index = 0;
    gpu_alloc_size_ += size;
    return p;
  } else {
    size_t avail, total, actual_avail, actual_total;
    bool is_limited = platform::RecordedCudaMemGetInfo(
        &avail, &total, &actual_avail, &actual_total, gpu_id_);
    size_t allocated = total - avail;

    std::string err_msg;
    if (is_limited) {
      auto limit_size = (total >> 20);
      err_msg = string::Sprintf(
          "\n   3) Set environment variable `FLAGS_gpu_memory_limit_mb` to a "
          "larger value. Currently `FLAGS_gpu_memory_limit_mb` is %d, so the "
          "maximum GPU memory usage is limited to %d MB.\n"
          "      The command is `export FLAGS_gpu_memory_limit_mb=xxx`.",
          limit_size, limit_size);
    }

    PADDLE_THROW_BAD_ALLOC(platform::errors::ResourceExhausted(
        "\n\nOut of memory error on GPU %d. "
        "Cannot allocate %s memory on GPU %d, %s memory has been allocated and "
        "available memory is only %s.\n\n"
        "Please check whether there is any other process using GPU %d.\n"
        "1. If yes, please stop them, or start PaddlePaddle on another GPU.\n"
        "2. If no, please try one of the following suggestions:\n"
        "   1) Decrease the batch size of your model.\n"
        "   2) FLAGS_fraction_of_gpu_memory_to_use is %.2lf now, "
        "please set it to a higher value but less than 1.0.\n"
        "      The command is "
        "`export FLAGS_fraction_of_gpu_memory_to_use=xxx`.%s\n\n",
        gpu_id_, string::HumanReadableSize(size), gpu_id_,
        string::HumanReadableSize(allocated), string::HumanReadableSize(avail),
        gpu_id_, FLAGS_fraction_of_gpu_memory_to_use, err_msg));
  }
}

void GPUAllocator::Free(void* p, size_t size, size_t index) {
  PADDLE_ENFORCE_EQ(index, 0, platform::errors::InvalidArgument(
                                  "The index should be 0, index is %d", index));
  PADDLE_ENFORCE_GE(gpu_alloc_size_, size,
                    platform::errors::InvalidArgument(
                        "The size of memory (%d) to free exceeds the size of "
                        "allocated gpu memory (%d)",
                        size, gpu_alloc_size_));
  gpu_alloc_size_ -= size;

  platform::RecordedCudaFree(p, size, gpu_id_);
}

bool GPUAllocator::UseGpu() const { return true; }

// PINNED memory allows direct DMA transfers by the GPU to and from system
// memory. It’s locked to a physical address.
void* CUDAPinnedAllocator::Alloc(size_t* index, size_t size) {
  if (size <= 0) return nullptr;

  // NOTE: here, we use CUDAPinnedMaxAllocSize as the maximum memory size
  // of host pinned allocation. Allocates too much would reduce
  // the amount of memory available to the underlying system for paging.
  size_t usable =
      paddle::platform::CUDAPinnedMaxAllocSize() - cuda_pinnd_alloc_size_;

  if (size > usable) {
    LOG(WARNING) << "Cannot malloc " << size / 1024.0 / 1024.0
                 << " MB pinned memory."
                 << ", available " << usable / 1024.0 / 1024.0 << " MB";
    return nullptr;
  }

  void* p;
// PINNED memory is visible to all CUDA contexts.
#ifdef PADDLE_WITH_HIP
  hipError_t result = hipHostMalloc(&p, size, hipHostMallocPortable);
#else
  cudaError_t result = cudaHostAlloc(&p, size, cudaHostAllocPortable);
#endif

  if (result == gpuSuccess) {
    *index = 1;  // PINNED memory
    cuda_pinnd_alloc_size_ += size;
    return p;
  } else {
    LOG(WARNING) << "cudaHostAlloc failed.";
    return nullptr;
  }

  return nullptr;
}

void CUDAPinnedAllocator::Free(void* p, size_t size, size_t index) {
  gpuError_t err;
  PADDLE_ENFORCE_EQ(index, 1, platform::errors::InvalidArgument(
                                  "The index should be 1, but got %d", index));

  PADDLE_ENFORCE_GE(cuda_pinnd_alloc_size_, size,
                    platform::errors::InvalidArgument(
                        "The size of memory (%d) to free exceeds the size of "
                        "allocated cuda pinned memory (%d)",
                        size, cuda_pinnd_alloc_size_));
  cuda_pinnd_alloc_size_ -= size;
#ifdef PADDLE_WITH_HIP
  err = hipHostFree(p);
  if (err != hipErrorDeinitialized) {
    PADDLE_ENFORCE_EQ(
        err, hipSuccess,
        platform::errors::Fatal(
            "hipFreeHost failed in GPUPinnedAllocator, error code is %d", err));
  }
#else
  err = cudaFreeHost(p);

  // Purposefully allow cudaErrorCudartUnloading, because
  // that is returned if you ever call cudaFreeHost after the
  // driver has already shutdown. This happens only if the
  // process is terminating, in which case we don't care if
  // cudaFreeHost succeeds.
  if (err != cudaErrorCudartUnloading) {
    PADDLE_ENFORCE_EQ(
        err, 0,
        platform::errors::Fatal(
            "cudaFreeHost failed in GPUPinnedAllocator, error code is %d",
            err));
  }
#endif
}

bool CUDAPinnedAllocator::UseGpu() const { return false; }

#endif

#ifdef PADDLE_WITH_ASCEND_CL
void* NPUAllocator::Alloc(size_t* index, size_t size) {
  if (size <= 0) return nullptr;

  void* p;
  auto result = platform::RecordedNPUMalloc(&p, size, npu_id_);

  if (result == ACL_ERROR_NONE) {
    *index = 0;
    npu_alloc_size_ += size;
    return p;
  } else {
    size_t avail, total, actual_avail, actual_total;
    bool is_limited = platform::RecordedNPUMemGetInfo(
        &avail, &total, &actual_avail, &actual_total, npu_id_);

    std::string err_msg;
    if (is_limited) {
      auto limit_size = (total >> 20);
      err_msg = string::Sprintf(
          "\n   3) Set environment variable `FLAGS_gpu_memory_limit_mb` to a "
          "larger value. Currently `FLAGS_gpu_memory_limit_mb` is %d, so the "
          "maximum GPU memory usage is limited to %d MB.\n"
          "      The command is `export FLAGS_gpu_memory_limit_mb=xxx`.",
          limit_size, limit_size);
    }

    PADDLE_THROW_BAD_ALLOC(platform::errors::ResourceExhausted(
        "\n\nOut of memory error on NPU %d. "
        "Cannot allocate %s memory on NPU %d, "
        "available memory is only %s.\n\n"
        "Please check whether there is any other process using NPU %d.\n"
        "1. If yes, please stop them, or start PaddlePaddle on another NPU.\n"
        "2. If no, please try one of the following suggestions:\n"
        "   1) Decrease the batch size of your model.\n"
        "   2) FLAGS_fraction_of_gpu_memory_to_use is %.2lf now, "
        "please set it to a higher value but less than 1.0.\n"
        "      The command is "
        "`export FLAGS_fraction_of_gpu_memory_to_use=xxx`.%s\n\n",
        npu_id_, string::HumanReadableSize(size), npu_id_,
        string::HumanReadableSize(avail), npu_id_,
        FLAGS_fraction_of_gpu_memory_to_use, err_msg));
  }
}

void NPUAllocator::Free(void* p, size_t size, size_t index) {
  VLOG(4) << "Free " << p << " size " << size;
  PADDLE_ENFORCE_EQ(index, 0, platform::errors::InvalidArgument(
                                  "The index should be 0, index is %d", index));
  PADDLE_ENFORCE_GE(npu_alloc_size_, size,
                    platform::errors::InvalidArgument(
                        "The size of memory (%d) to free exceeds the size of "
                        "allocated gpu memory (%d)",
                        size, npu_alloc_size_));
  npu_alloc_size_ -= size;

  platform::RecordedNPUFree(p, size, npu_id_);
}

bool NPUAllocator::UseGpu() const { return true; }

void* NPUPinnedAllocator::Alloc(size_t* index, size_t size) {
  if (size <= 0) return nullptr;

  size_t usable =
      paddle::platform::NPUPinnedMaxAllocSize() - npu_pinnd_alloc_size_;

  if (size > usable) {
    LOG(WARNING) << "Cannot malloc " << size / 1024.0 / 1024.0
                 << " MB pinned memory."
                 << ", available " << usable / 1024.0 / 1024.0 << " MB";
    return nullptr;
  }

  void* p;
  // PINNED memory is visible to all NPU contexts.
  auto result = aclrtMallocHost(&p, size);

  if (result == ACL_ERROR_NONE) {
    *index = 1;  // PINNED memory
    npu_pinnd_alloc_size_ += size;
    return p;
  } else {
    LOG(WARNING) << "aclrtMallocHost failed.";
    return nullptr;
  }

  return nullptr;
}

void NPUPinnedAllocator::Free(void* p, size_t size, size_t index) {
  aclError err;
  PADDLE_ENFORCE_EQ(index, 1, platform::errors::InvalidArgument(
                                  "The index should be 1, but got %d", index));

  PADDLE_ENFORCE_GE(npu_pinnd_alloc_size_, size,
                    platform::errors::InvalidArgument(
                        "The size of memory (%d) to free exceeds the size of "
                        "allocated npu pinned memory (%d)",
                        size, npu_pinnd_alloc_size_));
  npu_pinnd_alloc_size_ -= size;
  err = aclrtFreeHost(p);

  if (err != ACL_ERROR_NONE) {
    PADDLE_ENFORCE_EQ(
        err, 0,
        platform::errors::Fatal(
            "aclrtFreeHost failed in NPUPinnedAllocator, error code is %d",
            err));
  }
}

bool NPUPinnedAllocator::UseGpu() const { return false; }

#endif

#ifdef PADDLE_WITH_XPU
void* XPUAllocator::Alloc(size_t* index, size_t size) {
  if (size <= 0) return nullptr;

  void* p = nullptr;
  platform::XPUDeviceGuard guard(xpu_id_);

  int ret = xpu_malloc(reinterpret_cast<void**>(&p), size);
  if (ret != XPU_SUCCESS) {
    std::cout << "xpu memory malloc(" << size << ") failed, try again\n";
    xpu_wait();
    ret = xpu_malloc(reinterpret_cast<void**>(&p), size);
  }
  if (ret != XPU_SUCCESS) {
    PADDLE_THROW_BAD_ALLOC(platform::errors::ResourceExhausted(
        "\n\nOut of memory error on XPU %d. ", xpu_id_));
  } else {
    *index = 0;
    xpu_alloc_size_ += size;
    return p;
  }
}

void XPUAllocator::Free(void* p, size_t size, size_t index) {
  PADDLE_ENFORCE_EQ(index, 0, platform::errors::InvalidArgument(
                                  "The index should be 0, index is %d", index));
  PADDLE_ENFORCE_GE(xpu_alloc_size_, size,
                    platform::errors::InvalidArgument(
                        "The size of memory (%d) to free exceeds the size of "
                        "allocated xpu memory (%d)",
                        size, xpu_alloc_size_));
  xpu_alloc_size_ -= size;

  platform::XPUDeviceGuard guard(xpu_id_);
  xpu_free(p);
}

bool XPUAllocator::UseGpu() const { return false; }

bool XPUAllocator::UseXPU() { return true; }

#endif

}  // namespace detail
}  // namespace memory
}  // namespace paddle
