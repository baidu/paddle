/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/platform/npu_info.h"
#include <algorithm>
#include <cstdlib>
#include <memory>

#include "gflags/gflags.h"

#include "paddle/fluid/platform/lock_guard_ptr.h"
#include "paddle/fluid/platform/macros.h"
#include "paddle/fluid/platform/monitor.h"
#include "paddle/fluid/string/split.h"

DECLARE_double(fraction_of_gpu_memory_to_use);
DECLARE_uint64(initial_gpu_memory_in_mb);
DECLARE_uint64(reallocate_gpu_memory_in_mb);
DECLARE_bool(enable_cublas_tensor_op_math);
DECLARE_uint64(gpu_memory_limit_mb);
DECLARE_string(selected_npus);

constexpr static float fraction_reserve_gpu_memory = 0.05f;

USE_NPU_MEM_STAT;

namespace paddle {
namespace platform {

static int GetNPUDeviceCountImpl() {
  uint32_t count;
  PADDLE_ENFORCE_NPU_SUCCESS(aclrtGetDeviceCount(&count));
  return count;
}

int GetNPUDeviceCount() {
  static auto dev_cnt = GetNPUDeviceCountImpl();
  return dev_cnt;
}

int NPUCanAccessPeer(int src, int dst) {
  int can = 0;
  PADDLE_ENFORCE_NPU_SUCCESS(aclrtDeviceCanAccessPeer(&can, src, dst));
  return can;
}

// For example, "1.0.1"
std::string GetNPURuntimeVersion(int id) {
  PADDLE_ENFORCE_LT(id, GetNPUDeviceCount(),
                    platform::errors::InvalidArgument(
                        "Device id must be less than NPU count, "
                        "but received id is: %d. NPU count is: %d.",
                        id, GetNPUDeviceCount()));
  int major = 0, minor = 0, patch = 0;
  PADDLE_ENFORCE_NPU_SUCCESS(aclrtGetVersion(&major, &minor, &patch));
  return string::Sprintf("%d.%d.%d", major, minor, patch);
}

int GetCurrentNPUDeviceId() {
  int device_id;
  PADDLE_ENFORCE_NPU_SUCCESS(aclrtGetDevice(&device_id));
  return device_id;
}

//! Get a list of device ids from environment variable or use all.
std::vector<int> GetSelectedNPUDevices() {
  // use user specified NPUs in single-node multi-process mode.
  std::vector<int> devices;
  if (!FLAGS_selected_npus.empty()) {
    auto devices_str = paddle::string::Split(FLAGS_selected_npus, ',');
    for (auto id : devices_str) {
      devices.push_back(atoi(id.c_str()));
    }
  } else {
    int count = GetNPUDeviceCount();
    for (int i = 0; i < count; ++i) {
      devices.push_back(i);
    }
  }
  return devices;
}

void SetNPUDeviceId(int id) {
  PADDLE_ENFORCE_LT(id, GetNPUDeviceCount(),
                    platform::errors::InvalidArgument(
                        "Device id must be less than NPU count, "
                        "but received id is: %d. NPU count is: %d.",
                        id, GetNPUDeviceCount()));
  // NOTE(zihqiu): It is recommended to call aclrtSetDevice and aclrtResetDevice
  // pairly.
  PADDLE_ENFORCE_NPU_SUCCESS(aclrtSetDevice(id));
}

void ResetNPUDeviceId(int id) {
  PADDLE_ENFORCE_LT(id, GetNPUDeviceCount(),
                    platform::errors::InvalidArgument(
                        "Device id must be less than NPU count, "
                        "but received id is: %d. NPU count is: %d.",
                        id, GetNPUDeviceCount()));
  PADDLE_ENFORCE_NPU_SUCCESS(aclrtResetDevice(id));
}

void NPUMemoryUsage(size_t *available, size_t *total) {
  size_t actual_available, actual_total;
  RecordedNPUMemGetInfo(available, total, &actual_available, &actual_total,
                        platform::GetCurrentNPUDeviceId());
}

size_t NPUAvailableMemToAlloc() {
  size_t total = 0;
  size_t available = 0;
  NPUMemoryUsage(&available, &total);
  size_t reserving =
      static_cast<size_t>(fraction_reserve_gpu_memory * available);
  // If available size is less than minimum chunk size, no usable memory exists
  size_t available_to_alloc = available - reserving;
  size_t min_chunk_size = NPUMinChunkSize();
  if (available_to_alloc < min_chunk_size) {
    available_to_alloc = 0;
  }
  VLOG(10) << "NPU usage " << (available >> 20) << "M/" << (total >> 20)
           << "M, " << (available_to_alloc >> 20) << "M available to allocate";
  return available_to_alloc;
}

size_t NPUMaxAllocSize() {
  return std::max(NPUInitAllocSize(), NPUReallocSize());
}

static size_t NPUAllocSize(bool realloc) {
  size_t available_to_alloc = NPUAvailableMemToAlloc();
  PADDLE_ENFORCE_GT(
      available_to_alloc, 0,
      platform::errors::ResourceExhausted("Not enough available NPU memory."));
  // If FLAGS_initial_gpu_memory_in_mb is 0, then initial memory will be
  // allocated by fraction
  size_t flag_mb = realloc ? FLAGS_reallocate_gpu_memory_in_mb
                           : FLAGS_initial_gpu_memory_in_mb;
  size_t alloc_bytes =
      (flag_mb > 0ul ? flag_mb << 20 : available_to_alloc *
                                           FLAGS_fraction_of_gpu_memory_to_use);
  PADDLE_ENFORCE_GE(
      available_to_alloc, alloc_bytes,
      platform::errors::ResourceExhausted("Not enough available NPU memory."));
  VLOG(10) << "Alloc size is " << (alloc_bytes >> 20)
           << " MiB, is it Re-alloc: " << realloc;
  return alloc_bytes;
}

size_t NPUInitAllocSize() { return NPUAllocSize(/* realloc = */ false); }

size_t NPUReallocSize() { return NPUAllocSize(/* realloc = */ true); }

size_t NPUMinChunkSize() {
  // NOTE(zhiqiu): It seems the min chunk size should be 512 on NPU,
  // though no document specify that explicitly.
  // See https://gitee.com/zhiqiuchen/Ascend/tree/master/test_reduce_sum_d for
  // details.
  return 1 << 9;
}

size_t NPUMaxChunkSize() {
  size_t max_chunk_size = NPUMaxAllocSize();
  VLOG(10) << "Max chunk size " << (max_chunk_size >> 20) << "M";
  return max_chunk_size;
}

void NPUMemcpyAsync(void *dst, const void *src, size_t count,
                    enum aclrtMemcpyKind kind, aclrtStream stream,
                    size_t dst_max_count) {
  dst_max_count = dst_max_count ? dst_max_count : count;
  VLOG(4) << dst << " " << dst_max_count << " " << src << " " << count << " "
          << kind << " " << stream;
  PADDLE_ENFORCE_NPU_SUCCESS(
      aclrtMemcpyAsync(dst, dst_max_count, src, count, kind, stream));
}

void NPUMemcpySync(void *dst, const void *src, size_t count,
                   enum aclrtMemcpyKind kind, size_t dst_max_count) {
  // NOTE(zhiqiu):  The default max_count is count
  dst_max_count = dst_max_count ? dst_max_count : count;
  PADDLE_ENFORCE_NPU_SUCCESS(aclrtMemcpy(dst, dst_max_count, src, count, kind));
}

void NPUMemcpyPeerASync(void *dst, int dst_device, const void *src,
                        size_t count, enum aclrtMemcpyKind kind,
                        aclrtStream stream, size_t dst_max_count) {
  dst_max_count = dst_max_count ? dst_max_count : count;
  PADDLE_ENFORCE_NPU_SUCCESS(
      aclrtMemcpyAsync(dst, dst_max_count, src, count, kind, stream));
}

void NPUMemcpyPeerSync(void *dst, int dst_device, const void *src, size_t count,
                       enum aclrtMemcpyKind kind, size_t dst_max_count) {
  // NOTE(zhiqiu):  The default max_count is count
  dst_max_count = dst_max_count ? dst_max_count : count;
  PADDLE_ENFORCE_NPU_SUCCESS(aclrtMemcpy(dst, dst_max_count, src, count, kind));
}

void NPUMemsetAsync(void *dst, int value, size_t count, aclrtStream stream,
                    size_t max_count) {
  max_count = max_count ? max_count : count;
  PADDLE_ENFORCE_NPU_SUCCESS(
      aclrtMemsetAsync(dst, max_count, value, count, stream));
}

void NPUStreamSync(aclrtStream stream) {
  PADDLE_ENFORCE_NPU_SUCCESS(aclrtSynchronizeStream(stream));
}

static void RaiseNonOutOfMemoryError(aclError *status) {
  if (*status == ACL_ERROR_BAD_ALLOC) {
    *status = ACL_ERROR_NONE;
  }
  PADDLE_ENFORCE_NPU_SUCCESS(*status);
}

class RecordedNPUMallocHelper {
 private:
  explicit RecordedNPUMallocHelper(int dev_id, uint64_t limit_size = 0)
      : dev_id_(dev_id), limit_size_(limit_size) {
    if (NeedRecord()) {
      mtx_.reset(new std::mutex());
    }
  }

  DISABLE_COPY_AND_ASSIGN(RecordedNPUMallocHelper);

 public:
  static RecordedNPUMallocHelper *Instance(int dev_id) {
    std::call_once(once_flag_, [] {
      int dev_cnt = GetNPUDeviceCount();
      instances_.reserve(dev_cnt);
      for (int i = 0; i < dev_cnt; ++i) {
        // NOTE(zhiqiu): share the flags with gpu, avoid more flags.
        instances_.emplace_back(
            new RecordedNPUMallocHelper(i, FLAGS_gpu_memory_limit_mb << 20));
      }
    });

    PADDLE_ENFORCE_GE(
        dev_id, 0,
        platform::errors::OutOfRange(
            "Device id must be not less than 0, but got %d.", dev_id));
    PADDLE_ENFORCE_LT(
        dev_id, instances_.size(),
        platform::errors::OutOfRange("Device id %d exceeds npu card number %d.",
                                     dev_id, instances_.size()));
    return instances_[dev_id].get();
  }

  /**
   * Try to allocate `size` npu memory. Only ACL_ERROR_BAD_ALLOC
   * or ACL_ERROR_NONE would be returned.
   */
  aclError Malloc(void **ptr, size_t size) {
    LockGuardPtr<std::mutex> lock(mtx_);
    if (UNLIKELY(NeedRecord() && cur_size_ + size > limit_size_)) {
      return ACL_ERROR_BAD_ALLOC;
    }

    NPUDeviceGuard guard(dev_id_);
    auto result = aclrtMalloc(ptr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    if (result == ACL_ERROR_NONE) {
      if (NeedRecord()) {
        cur_size_ += size;
      }
      STAT_INT_ADD("STAT_npu" + std::to_string(dev_id_) + "_mem_size", size);
      return result;
    } else {
      RaiseNonOutOfMemoryError(&result);
      // Non out of memory error would be raised inside
      // RaiseNonOutOfMemoryError. Therefore, we can
      // return cudaErrorMemoryAllocation directly here.
      return ACL_ERROR_BAD_ALLOC;
    }
  }

  /**
   * Free gpu memory. Usually, free is not allowed to raise error.
   * If it does raise error, the process should be crashed.
   */
  void Free(void *ptr, size_t size) {
    NPUDeviceGuard guard(dev_id_);
    auto result = aclrtFree(ptr);
    PADDLE_ENFORCE_NPU_SUCCESS(result);
    if (NeedRecord()) {
      std::lock_guard<std::mutex> guard(*mtx_);
      cur_size_ -= size;
    }
    STAT_INT_SUB("STAT_npu" + std::to_string(dev_id_) + "_mem_size", size);
  }

  bool GetMemInfo(size_t *avail, size_t *total, size_t *actual_avail,
                  size_t *actual_total) {
    {
      NPUDeviceGuard guard(dev_id_);
      auto result = aclrtGetMemInfo(ACL_HBM_MEM, actual_avail, actual_total);
      if (result != ACL_ERROR_NONE) {
        *actual_avail = 0;
      }
      RaiseNonOutOfMemoryError(&result);
    }

    if (NeedRecord()) {
      std::lock_guard<std::mutex> guard(*mtx_);
      *avail = std::min(*actual_avail, limit_size_ - cur_size_);
      *total = std::min(*actual_total, limit_size_);
      return *total < *actual_total;
    } else {
      *avail = *actual_avail;
      *total = *actual_total;
      return false;
    }
  }

  inline bool NeedRecord() const { return limit_size_ != 0; }

  uint64_t RecordedSize() const {
    LockGuardPtr<std::mutex> lock(mtx_);
    return NeedRecord() ? cur_size_ : 0;
  }

  uint64_t LimitSize() const { return limit_size_; }

 private:
  const int dev_id_;
  const uint64_t limit_size_;
  uint64_t cur_size_{0};

  mutable std::unique_ptr<std::mutex> mtx_;

  static std::once_flag once_flag_;
  static std::vector<std::unique_ptr<RecordedNPUMallocHelper>> instances_;
};

std::once_flag RecordedNPUMallocHelper::once_flag_;
std::vector<std::unique_ptr<RecordedNPUMallocHelper>>
    RecordedNPUMallocHelper::instances_;

aclError RecordedNPUMalloc(void **ptr, size_t size, int dev_id) {
  return RecordedNPUMallocHelper::Instance(dev_id)->Malloc(ptr, size);
}

void RecordedNPUFree(void *p, size_t size, int dev_id) {
  return RecordedNPUMallocHelper::Instance(dev_id)->Free(p, size);
}

bool RecordedNPUMemGetInfo(size_t *avail, size_t *total, size_t *actual_avail,
                           size_t *actual_total, int dev_id) {
  return RecordedNPUMallocHelper::Instance(dev_id)->GetMemInfo(
      avail, total, actual_avail, actual_total);
}

uint64_t RecordedNPUMallocSize(int dev_id) {
  return RecordedNPUMallocHelper::Instance(dev_id)->RecordedSize();
}

bool IsNPUMallocRecorded(int dev_id) {
  return RecordedNPUMallocHelper::Instance(dev_id)->NeedRecord();
}

AclInstance::~AclInstance() {}

AclInstance &AclInstance::Instance() {
  static AclInstance instance;
  return instance;
}

AclInstance::AclInstance() {
  PADDLE_ENFORCE_NPU_SUCCESS(aclInit(nullptr));
  VLOG(4) << "Call aclrtSetDevice ";
  // NOTE(zhiqiu): why set devices here?
  // Because ACL creates a default context which contains 2 streams
  // when calling aclrtSetDeviceId, so usually we do not need to
  // create contexts explicitly. And, for each device, aclrtSetDeviceId
  // need to call parily with aclrtResetDeviceId to destory the default
  // context. Here, we use this singleton and static instance to manage
  // the devices to make sure they will be resetted before program exit.
  devices_ = platform::GetSelectedNPUDevices();
  for (auto it = devices_.rbegin(); it != devices_.rend(); ++it) {
    SetNPUDeviceId(*it);
    VLOG(4) << "Call aclrtSetDevice " << *it;
  }
}

void AclInstance::Finalize() {
  // NOTE(zhiqiu): DO NOT perform finalize in destructor
  // to avoid problems caused by destructor order of static
  // object.
  for (size_t i = 0; i < devices_.size(); ++i) {
    auto status = aclrtResetDevice(devices_[i]);
    VLOG(4) << "Call aclrtResetDevice " << devices_[i]
            << " status = " << status;
  }
  auto status = aclFinalize();
  VLOG(4) << "Call aclFinalize, status = " << status;
}

}  // namespace platform
}  // namespace paddle
