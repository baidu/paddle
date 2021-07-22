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

#include "paddle/fluid/memory/detail/system_allocator.h"

#include <memory>

#include "gflags/gflags.h"
#include "gtest/gtest.h"
#include "paddle/fluid/memory/allocation/allocator.h"

DECLARE_bool(use_pinned_memory);

void TestAllocator(paddle::memory::detail::SystemAllocator* a, size_t size) {
  bool freed = false;
  {
    size_t index;
    void* p = a->Alloc(&index, size);
    if (size > 0) {
      EXPECT_NE(p, nullptr);
    } else {
      EXPECT_EQ(p, nullptr);
    }

    int* i = static_cast<int*>(p);
    std::shared_ptr<int> ptr(i, [&](void* p) {
      freed = true;
      a->Free(p, size, index);
    });
  }
  EXPECT_TRUE(freed);
}

TEST(CPUAllocator, NoLockMem) {
  FLAGS_use_pinned_memory = false;
  paddle::memory::detail::CPUAllocator a;
  TestAllocator(&a, 2048);
  TestAllocator(&a, 0);
}

TEST(CPUAllocator, LockMem) {
  FLAGS_use_pinned_memory = true;
  paddle::memory::detail::CPUAllocator a;
  TestAllocator(&a, 2048);
  TestAllocator(&a, 0);
}

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
TEST(GPUAllocator, Alloc) {
  paddle::memory::detail::GPUAllocator a(0);
  TestAllocator(&a, 2048);
  TestAllocator(&a, 0);
}

TEST(CUDAPinnedAllocator, Alloc) {
  paddle::memory::detail::CUDAPinnedAllocator a;
  TestAllocator(&a, 2048);
  TestAllocator(&a, 0);
}

TEST(GPUAllocator, AllocFailure) {
  paddle::memory::detail::GPUAllocator allocator(0);
  size_t index;
  size_t alloc_size = (static_cast<size_t>(1) << 40);  // Very large number
  try {
    allocator.Alloc(&index, alloc_size);
    ASSERT_TRUE(false);
  } catch (paddle::memory::allocation::BadAlloc&) {
#ifdef PADDLE_WITH_HIP
    PADDLE_ENFORCE_CUDA_SUCCESS(hipGetLastError());
#else
    PADDLE_ENFORCE_CUDA_SUCCESS(cudaGetLastError());
#endif
  }
}
#endif

#ifdef PADDLE_WITH_ASCEND_CL
TEST(NPUAllocator, Alloc) {
  paddle::memory::detail::NPUAllocator a(0);
  TestAllocator(&a, 1 << 20);
  TestAllocator(&a, 1);
}
#endif

#ifdef PADDLE_WITH_XPU
TEST(XPUAllocator, Alloc) {
  paddle::memory::detail::XPUAllocator a(0);
  TestAllocator(&a, 1 << 20);
  TestAllocator(&a, 1);
}
#endif
