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

#pragma once

#ifdef PADDLE_WITH_ASCEND_CL
#include <stddef.h>

#include <string>
#include <vector>

#include "acl/acl.h"
#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace platform {

//! Get the total number of NPU devices in system.
int GetNPUDeviceCount();

//! Get the runtime version of the ith NPU
std::string GetNPURuntimeVersion(int id);
//! Check if this device can access peer or not.
int NPUCanAccessPeer(int src, int dst);

//! Get the current NPU device id in system.
int GetCurrentNPUDeviceId();

//! Get the current NPU stream.
int GetCurrentStream();

//! Get a list of device ids from environment variable or use all.
std::vector<int> GetSelectedNPUDevices();

//! Set the NPU device id for next execution.
void SetNPUDeviceId(int device_id);

//! Reset the NPU device id for next execution.
void ResetNPUDeviceId(int device_id);

//! Get the memory usage of current NPU device.
void NPUMemoryUsage(size_t *available, size_t *total);

//! Get the available memory to allocate, which is the size of available npu
//! minus reserving.
size_t NPUAvailableMemToAlloc();

//! Get the maximum allocation size of current NPU device.
size_t NPUMaxAllocSize();

//! Get the initial allocation size of current NPU device.
size_t NPUInitAllocSize();

//! Get the re-allocation size of current NPU device.
size_t NPUReallocSize();

//! Get the minimum chunk size for NPU buddy allocator.
size_t NPUMinChunkSize();

//! Get the maximum chunk size for NPU buddy allocator.
size_t NPUMaxChunkSize();

//! Copy memory from address src to dst asynchronously.
void NPUMemcpyAsync(void *dst, const void *src, size_t count,
                    enum aclrtMemcpyKind kind, aclrtStream stream,
                    size_t dst_max_count = 0);

//! Copy memory from address src to dst synchronously.
void NPUMemcpySync(void *dst, const void *src, size_t count,
                   enum aclrtMemcpyKind kind, size_t dst_max_count = 0);

//! Set memory dst with value count size asynchronously
void NPUMemsetAsync(void *dst, int value, size_t count, aclrtStream stream,
                    size_t max_count = 0);

//! Copy memory from one device to another device asynchronously.
void NPUMemcpyPeerAsync(void *dst, int dst_device, const void *src,
                        int src_device, size_t count, aclrtStream stream,
                        size_t max_count = 0);

//! Copy memory from one device to another device synchronously.
void NPUMemcpyPeerSync(void *dst, int dst_device, const void *src,
                       int src_device, size_t count, size_t max_count = 0);

//! Blocks until stream has completed all operations.
void NPUStreamSync(aclrtStream stream);

//! aclrtMalloc with recorded info
aclError RecordedNPUMalloc(void **ptr, size_t size, int dev_id);

//! aclrtFree with recorded info
void RecordedNPUFree(void *p, size_t size, int dev_id);

//! Get available and total gpu memory with considering limitation
bool RecordedNPUMemGetInfo(size_t *avail, size_t *total, size_t *actual_avail,
                           size_t *actual_total, int dev_id);

//! Get recorded actrtMalloc size. If record is disabled, return 0.
uint64_t RecordedNPUMallocSize(int dev_id);

bool IsNPUMallocRecorded(int dev_id);

class NPUDeviceGuard {
 public:
  explicit inline NPUDeviceGuard(int dev_id) {
    int prev_id = platform::GetCurrentNPUDeviceId();
    if (prev_id != dev_id) {
      prev_id_ = prev_id;
      platform::SetNPUDeviceId(dev_id);
    }
  }

  inline ~NPUDeviceGuard() {
    if (prev_id_ != -1) {
      platform::SetNPUDeviceId(prev_id_);
    }
  }

  NPUDeviceGuard(const NPUDeviceGuard &o) = delete;
  NPUDeviceGuard &operator=(const NPUDeviceGuard &o) = delete;

 private:
  int prev_id_{-1};
};

}  // namespace platform
}  // namespace paddle

#endif
