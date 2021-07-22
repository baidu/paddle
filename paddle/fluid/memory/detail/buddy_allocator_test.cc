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

#include "paddle/fluid/memory/detail/buddy_allocator.h"

#include <memory>

#ifdef WITH_GPERFTOOLS
#include "gperftools/profiler.h"
#endif
#include <fstream>
#include <string>

#include "gflags/gflags.h"
#include "gtest/gtest.h"
#include "paddle/fluid/platform/gpu_info.h"
#include "paddle/fluid/platform/npu_info.h"

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP) || \
    defined(PADDLE_WITH_ASCEND_CL)
DECLARE_double(fraction_of_gpu_memory_to_use);
DECLARE_uint64(initial_gpu_memory_in_mb);
DECLARE_uint64(reallocate_gpu_memory_in_mb);
#endif

#ifdef PADDLE_WITH_XPU
DECLARE_uint64(initial_xpu_memory_in_mb);
DECLARE_uint64(reallocate_xpu_memory_in_mb);
#endif

namespace paddle {
namespace memory {
namespace detail {

constexpr static int TEST_GPU_ID = 0;

int* TestBuddyAllocator(BuddyAllocator* allocator, size_t size_bytes,
                        bool use_system_allocator = false,
                        bool free_ptr = true) {
  bool freed = false;
  size_t used_bytes = allocator->Used();

  if (size_bytes > 0) {
    void* p = allocator->Alloc(size_bytes);

    EXPECT_NE(p, nullptr);

    if (size_bytes < allocator->GetMaxChunkSize()) {
      // Not allocate from SystemAllocator
      EXPECT_FALSE(use_system_allocator);
      EXPECT_GE(allocator->Used(), used_bytes + size_bytes);
    } else {
      // Allocate from SystemAllocator doesn't count in Used()
      EXPECT_TRUE(use_system_allocator);
      EXPECT_EQ(allocator->Used(), used_bytes);
    }

    int* intp = static_cast<int*>(p);
    if (!free_ptr) {
      return intp;
    }
    std::shared_ptr<int> ptr(intp, [&](void* p) {
      allocator->Free(intp);
      freed = true;
    });
  } else {
    freed = true;
  }

  EXPECT_EQ(used_bytes, allocator->Used());
  EXPECT_TRUE(freed);
  return nullptr;
}

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
TEST(BuddyAllocator, GpuFraction) {
  // In a 16 GB machine, the pool size will be about 160 MB
  FLAGS_fraction_of_gpu_memory_to_use = 0.01;
  FLAGS_initial_gpu_memory_in_mb = 0;
  FLAGS_reallocate_gpu_memory_in_mb = 0;

  BuddyAllocator buddy_allocator(
      std::unique_ptr<SystemAllocator>(new GPUAllocator(TEST_GPU_ID)),
      platform::GpuMinChunkSize(), platform::GpuMaxChunkSize());

  // Less than pool size
  TestBuddyAllocator(&buddy_allocator, 10);
  TestBuddyAllocator(&buddy_allocator, 10 << 10);
  TestBuddyAllocator(&buddy_allocator, 10 << 20);

  // Greater than max chunk size
  TestBuddyAllocator(&buddy_allocator, 300 << 20,
                     /* use_system_allocator = */ true);
  TestBuddyAllocator(&buddy_allocator, 1 * static_cast<size_t>(1 << 30),
                     /* use_system_allocator = */ true);
}

TEST(BuddyAllocator, InitRealloc) {
  FLAGS_initial_gpu_memory_in_mb = 100;
  FLAGS_reallocate_gpu_memory_in_mb = 50;

  EXPECT_EQ(platform::GpuMaxChunkSize(), static_cast<size_t>(100 << 20));

  BuddyAllocator buddy_allocator(
      std::unique_ptr<SystemAllocator>(new GPUAllocator(TEST_GPU_ID)),
      platform::GpuMinChunkSize(), platform::GpuMaxChunkSize());

  // Less then initial size and reallocate size
  TestBuddyAllocator(&buddy_allocator, 10 << 20);
  // Between initial size and reallocate size and not exceed pool
  TestBuddyAllocator(&buddy_allocator, 80 << 20);
  TestBuddyAllocator(&buddy_allocator, 99 << 20);
  // Greater than max chunk size
  TestBuddyAllocator(&buddy_allocator, 101 << 20,
                     /* use_system_allocator = */ true);
  TestBuddyAllocator(&buddy_allocator, 1 * static_cast<size_t>(1 << 30),
                     /* use_system_allocator = */ true);
}

TEST(BuddyAllocator, ReallocSizeGreaterThanInit) {
  FLAGS_initial_gpu_memory_in_mb = 5;
  FLAGS_reallocate_gpu_memory_in_mb = 10;

  EXPECT_EQ(platform::GpuMaxChunkSize(), static_cast<size_t>(10 << 20));

  BuddyAllocator buddy_allocator(
      std::unique_ptr<SystemAllocator>(new GPUAllocator(TEST_GPU_ID)),
      platform::GpuMinChunkSize(), platform::GpuMaxChunkSize());

  // Less than initial size and reallocate size
  TestBuddyAllocator(&buddy_allocator, 1 << 20);
  // Between initial size and reallocate size and exceed pool
  TestBuddyAllocator(&buddy_allocator, 6 << 20);
  TestBuddyAllocator(&buddy_allocator, 8 << 20);
  TestBuddyAllocator(&buddy_allocator, 9 << 20);
  // Greater than max trunk size
  TestBuddyAllocator(&buddy_allocator, 11 << 20,
                     /* use_system_allocator = */ true);
  TestBuddyAllocator(&buddy_allocator, 1 * static_cast<size_t>(1 << 30),
                     /* use_system_allocator = */ true);
}

TEST(BuddyAllocator, FractionRefillPool) {
  FLAGS_fraction_of_gpu_memory_to_use = 0.6;
  FLAGS_initial_gpu_memory_in_mb = 0;
  FLAGS_reallocate_gpu_memory_in_mb = 0;

  size_t max_chunk_size = platform::GpuMaxChunkSize();
  BuddyAllocator buddy_allocator(
      std::unique_ptr<SystemAllocator>(new GPUAllocator(TEST_GPU_ID)),
      platform::GpuMinChunkSize(), max_chunk_size);

  // Less than pool size
  int* p0 = TestBuddyAllocator(&buddy_allocator, max_chunk_size - 1000,
                               /* use_system_allocator = */ false,
                               /* free_ptr = */ false);
  // Max chunk size should be same during allocation
  EXPECT_EQ(max_chunk_size, buddy_allocator.GetMaxChunkSize());

  size_t alloc =
      platform::GpuAvailableMemToAlloc() * FLAGS_fraction_of_gpu_memory_to_use;
  // Exceed pool trigger refilling size of fraction of avaiable gpu, and should
  // be able to alloc 60% of the remaining GPU
  int* p1 = TestBuddyAllocator(&buddy_allocator, alloc,
                               /* use_system_allocator = */ false,
                               /* free_ptr = */ false);
  // Max chunk size should be same during allocation
  EXPECT_EQ(max_chunk_size, buddy_allocator.GetMaxChunkSize());

  alloc =
      platform::GpuAvailableMemToAlloc() * FLAGS_fraction_of_gpu_memory_to_use;
  // Exceed pool trigger refilling size of fraction of avaiable gpu, and should
  // be able to alloc 60% of the remaining GPU
  TestBuddyAllocator(&buddy_allocator, alloc,
                     /* use_system_allocator = */ false);
  // Max chunk size should be same during allocation
  EXPECT_EQ(max_chunk_size, buddy_allocator.GetMaxChunkSize());

  buddy_allocator.Free(p0);
  buddy_allocator.Free(p1);
}

TEST(BuddyAllocator, AllocFromAvailable) {
  FLAGS_fraction_of_gpu_memory_to_use = 0.7;
  FLAGS_initial_gpu_memory_in_mb = 0;
  FLAGS_reallocate_gpu_memory_in_mb = 0;

  size_t total = 0, available = 0;
  platform::SetDeviceId(TEST_GPU_ID);
  platform::GpuMemoryUsage(&available, &total);

  // Take half of available GPU
  void* p;
#ifdef PADDLE_WITH_HIP
  hipError_t result = hipMalloc(&p, available >> 1);
  EXPECT_TRUE(result == hipSuccess);
#else
  cudaError_t result = cudaMalloc(&p, available >> 1);
  EXPECT_TRUE(result == cudaSuccess);
#endif

  // BuddyAllocator should be able to alloc the remaining GPU
  BuddyAllocator buddy_allocator(
      std::unique_ptr<SystemAllocator>(new GPUAllocator(TEST_GPU_ID)),
      platform::GpuMinChunkSize(), platform::GpuMaxChunkSize());

  TestBuddyAllocator(&buddy_allocator, 10);
  TestBuddyAllocator(&buddy_allocator, 10 << 10);
  TestBuddyAllocator(&buddy_allocator, 10 << 20);
  TestBuddyAllocator(&buddy_allocator, static_cast<size_t>(1 << 30));

  if (p) {
#ifdef PADDLE_WITH_HIP
    EXPECT_TRUE(hipFree(p) == hipSuccess);
#else
    EXPECT_TRUE(cudaFree(p) == cudaSuccess);
#endif
  }
}

TEST(BuddyAllocator, AllocFromAvailableWhenFractionIsOne) {
  FLAGS_fraction_of_gpu_memory_to_use = 1.0;
  FLAGS_initial_gpu_memory_in_mb = 0;
  FLAGS_reallocate_gpu_memory_in_mb = 0;

  void* p = nullptr;

#ifdef PADDLE_WITH_HIP
  EXPECT_TRUE(hipMalloc(&p, static_cast<size_t>(1) << 30) == hipSuccess);
#else
  EXPECT_TRUE(cudaMalloc(&p, static_cast<size_t>(1) << 30) == cudaSuccess);
#endif

  // BuddyAllocator should be able to alloc the remaining GPU
  BuddyAllocator buddy_allocator(
      std::unique_ptr<SystemAllocator>(new GPUAllocator(TEST_GPU_ID)),
      platform::GpuMinChunkSize(), platform::GpuMaxChunkSize());

  TestBuddyAllocator(&buddy_allocator, static_cast<size_t>(1) << 30);
  TestBuddyAllocator(&buddy_allocator, static_cast<size_t>(1) << 30);

  if (p) {
#ifdef PADDLE_WITH_HIP
    EXPECT_TRUE(hipFree(p) == hipSuccess);
#else
    EXPECT_TRUE(cudaFree(p) == cudaSuccess);
#endif
  }
}

TEST(BuddyAllocator, SpeedAna) {
  // In a 16 GB machine, the pool size will be about 160 MB
  FLAGS_fraction_of_gpu_memory_to_use = 0.5;
  FLAGS_initial_gpu_memory_in_mb = 0;
  FLAGS_reallocate_gpu_memory_in_mb = 0;

  BuddyAllocator buddy_allocator(
      std::unique_ptr<SystemAllocator>(new GPUAllocator(TEST_GPU_ID)),
      platform::GpuMinChunkSize(), platform::GpuMaxChunkSize());

  // Less than pool size
  TestBuddyAllocator(&buddy_allocator, 10);
  TestBuddyAllocator(&buddy_allocator, 10 << 10);
  TestBuddyAllocator(&buddy_allocator, 10 << 20);

  std::fstream in_file;
  in_file.open("buddy_allocator_test_data", std::ios::in);

  std::vector<void*> vec_ptr;
  std::vector<int> vec_size;
  std::vector<int> vec_pos;
  std::vector<bool> vec_free_flag;

  std::string line;
  int size, id;
  while (in_file >> size >> id) {
    vec_size.push_back(size);
    vec_pos.push_back(id);
  }

  vec_ptr.reserve(vec_size.size());

  auto start = std::chrono::steady_clock::now();

#ifdef WITH_GPERFTOOLS
  ProfilerStart("test.prof");
#endif
  for (size_t loop = 0; loop < 5000; ++loop) {
    vec_ptr.clear();
    for (size_t i = 0; i < vec_size.size(); ++i) {
      if (vec_pos[i] == -1) {
        auto res = buddy_allocator.Alloc(vec_size[i]);

        vec_ptr.push_back(res);
      } else {
        vec_ptr.push_back(nullptr);

        auto free_ptr = vec_ptr[vec_pos[i]];
        EXPECT_NE(free_ptr, nullptr);

        vec_ptr[vec_pos[i]] = nullptr;

        buddy_allocator.Free(free_ptr);
      }
    }

    for (size_t i = 0; i < vec_size.size(); ++i) {
      if (vec_ptr[i] != nullptr) {
        buddy_allocator.Free(vec_ptr[i]);
      }
    }
  }

#ifdef WITH_GPERFTOOLS
  ProfilerStop();
#endif
  auto end = std::chrono::steady_clock::now();
  std::chrono::duration<double> diff = end - start;
  std::cerr << "time cost " << diff.count() << std::endl;
}

TEST(BuddyAllocator, Release) {
  // In a 8 GB machine, the pool size will be about 800 MB
  FLAGS_fraction_of_gpu_memory_to_use = 0.1;
  FLAGS_initial_gpu_memory_in_mb = 0;
  FLAGS_reallocate_gpu_memory_in_mb = 0;

  BuddyAllocator buddy_allocator(
      std::unique_ptr<SystemAllocator>(new GPUAllocator(TEST_GPU_ID)),
      platform::GpuMinChunkSize(), platform::GpuMaxChunkSize());

  // Less than pool size
  TestBuddyAllocator(&buddy_allocator, 10);
  TestBuddyAllocator(&buddy_allocator, 10 << 10);
  TestBuddyAllocator(&buddy_allocator, 50 << 20);

  buddy_allocator.Release();
}
#endif

#ifdef PADDLE_WITH_ASCEND_CL
TEST(BuddyAllocator, NpuFraction) {
  // In a 16 GB machine, the pool size will be about 160 MB
  FLAGS_fraction_of_gpu_memory_to_use = 0.005;
  FLAGS_fraction_of_gpu_memory_to_use = 0.92;
  FLAGS_initial_gpu_memory_in_mb = 0;
  FLAGS_reallocate_gpu_memory_in_mb = 0;

  BuddyAllocator buddy_allocator(
      std::unique_ptr<SystemAllocator>(new NPUAllocator(0)),
      platform::NPUMinChunkSize(), platform::NPUMaxChunkSize());

  // Less than pool size
  TestBuddyAllocator(&buddy_allocator, 10);
  TestBuddyAllocator(&buddy_allocator, 10 << 10);
  TestBuddyAllocator(&buddy_allocator, 10 << 20);
  buddy_allocator.Release();

  // Greater than max chunk size
  TestBuddyAllocator(&buddy_allocator, 300 << 20,
                     /* use_system_allocator = */ true);
  TestBuddyAllocator(&buddy_allocator, 1 * static_cast<size_t>(1 << 30),
                     /* use_system_allocator = */ true);
}
#endif

#ifdef PADDLE_WITH_XPU
TEST(BuddyAllocator, NpuFraction) {
  // In a 16 GB machine, the pool size will be about 160 MB
  FLAGS_initial_xpu_memory_in_mb = 0;
  FLAGS_reallocate_xpu_memory_in_mb = 0;

  BuddyAllocator buddy_allocator(
      std::unique_ptr<SystemAllocator>(new XPUAllocator(0)),
      platform::XPUMinChunkSize(), platform::XPUMaxChunkSize());

  // Less than pool size
  TestBuddyAllocator(&buddy_allocator, 5);
  TestBuddyAllocator(&buddy_allocator, 5 << 10);
  TestBuddyAllocator(&buddy_allocator, 5 << 20);
  buddy_allocator.Release();

  // Greater than max chunk size
  TestBuddyAllocator(&buddy_allocator, 300 << 20,
                     /* use_system_allocator = */ true);
  TestBuddyAllocator(&buddy_allocator, 1 * static_cast<size_t>(1 << 30),
                     /* use_system_allocator = */ true);
}
#endif

}  // namespace detail
}  // namespace memory
}  // namespace paddle
