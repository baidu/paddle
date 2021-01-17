// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

#include "gflags/gflags.h"
#include "gtest/gtest.h"
#include "paddle/fluid/platform/cuda_device_guard.h"
#include "paddle/fluid/platform/gpu_info.h"

DECLARE_uint64(gpu_memory_limit_mb);

namespace paddle {
namespace platform {

static constexpr uint64_t GPU_MEMORY_LIMIT_MB = 500;
static constexpr int DEVICE_ID = 0;

TEST(test_record_malloc, test_limit_gpu_memory) {
  FLAGS_gpu_memory_limit_mb = GPU_MEMORY_LIMIT_MB;
  size_t limit = FLAGS_gpu_memory_limit_mb << 20;

  {
    ASSERT_TRUE(IsCudaMallocRecorded(DEVICE_ID));
    ASSERT_EQ(RecordedCudaMallocSize(DEVICE_ID), 0UL);
  }

  size_t avail, total;
  {
    size_t actual_avail, actual_total;
    RecordedCudaMemGetInfo(&avail, &total, &actual_avail, &actual_total,
                           DEVICE_ID);
    ASSERT_EQ(total, limit);
    ASSERT_EQ(hipGetLastError(), hipSuccess);
  }

  {
    CUDADeviceGuard guard(DEVICE_ID);
    GpuMemoryUsage(&avail, &total);
    ASSERT_EQ(total, limit);
    ASSERT_EQ(hipGetLastError(), hipSuccess);
  }

  hipError_t err = hipSuccess;

  void *p1 = nullptr;
  size_t size1 = limit / 4 * 3;
  {
    err = platform::RecordedCudaMalloc(&p1, size1, DEVICE_ID);
    ASSERT_EQ(err, hipSuccess);
    ASSERT_EQ(hipGetLastError(), hipSuccess);
    ASSERT_NE(p1, nullptr);

    ASSERT_EQ(RecordedCudaMallocSize(DEVICE_ID), size1);
  }

  void *p2 = nullptr;
  size_t size2 = limit / 2;
  {
    err = platform::RecordedCudaMalloc(&p2, size2, DEVICE_ID);
    ASSERT_EQ(err, hipErrorOutOfMemory);
    ASSERT_EQ(hipGetLastError(), hipSuccess);
    ASSERT_EQ(p2, nullptr);

    ASSERT_EQ(RecordedCudaMallocSize(DEVICE_ID), size1);
  }

  {
    platform::RecordedCudaFree(p1, size1, DEVICE_ID);
    ASSERT_EQ(RecordedCudaMallocSize(DEVICE_ID), 0UL);
  }

  {
    err = platform::RecordedCudaMalloc(&p2, size2, DEVICE_ID);
    ASSERT_EQ(err, hipSuccess);
    ASSERT_EQ(hipGetLastError(), hipSuccess);
    ASSERT_NE(p2, nullptr);
    ASSERT_EQ(RecordedCudaMallocSize(DEVICE_ID), size2);
  }

  {
    platform::RecordedCudaFree(p2, size2, DEVICE_ID);
    ASSERT_EQ(RecordedCudaMallocSize(DEVICE_ID), 0UL);
  }
}

}  // namespace platform
}  // namespace paddle
