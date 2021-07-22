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

#include "paddle/fluid/memory/allocation/naive_best_fit_allocator.h"

#include "gtest/gtest.h"

namespace paddle {
namespace memory {
namespace allocation {

TEST(NaiveBestFitAllocatorTest, CpuAlloc) {
  NaiveBestFitAllocator alloc{platform::CPUPlace()};
  {
    size_t size = (1 << 20);
    auto allocation = alloc.Allocate(size);
  }
  alloc.Release(platform::CPUPlace());

  size_t size = (1 << 20);
  auto allocation = alloc.Allocate(size);
  alloc.Release(platform::CPUPlace());
}

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
TEST(NaiveBestFitAllocatorTest, GpuAlloc) {
  NaiveBestFitAllocator alloc{platform::CUDAPlace(0)};
  {
    size_t size = (1 << 20);
    auto allocation = alloc.Allocate(size);
  }
  alloc.Release(platform::CUDAPlace(0));

  size_t size = (1 << 20);
  auto allocation = alloc.Allocate(size);
  alloc.Release(platform::CUDAPlace(0));
}

TEST(NaiveBestFitAllocatorTest, CudaPinnedAlloc) {
  NaiveBestFitAllocator alloc{platform::CUDAPinnedPlace()};
  {
    size_t size = (1 << 20);
    auto allocation = alloc.Allocate(size);
  }
  alloc.Release(platform::CUDAPinnedPlace());

  size_t size = (1 << 20);
  auto allocation = alloc.Allocate(size);
  alloc.Release(platform::CUDAPinnedPlace());
}
#endif

#ifdef PADDLE_WITH_ASCEND_CL
TEST(NaiveBestFitAllocatorTest, NpuAlloc) {
  NaiveBestFitAllocator alloc{platform::NPUPlace(0)};
  {
    size_t size = (1 << 20);
    auto allocation = alloc.Allocate(size);
  }
  sleep(10);
  alloc.Release(platform::NPUPlace(0));

  size_t size = (1 << 20);
  auto allocation = alloc.Allocate(size);
  alloc.Release(platform::NPUPlace(0));
}
#endif

#ifdef PADDLE_WITH_XPU
TEST(NaiveBestFitAllocatorTest, XpuAlloc) {
  NaiveBestFitAllocator alloc{platform::XPUPlace(0)};
  {
    size_t size = (1 << 20);
    auto allocation = alloc.Allocate(size);
  }
  sleep(10);
  alloc.Release(platform::XPUPlace(0));

  size_t size = (1 << 20);
  auto allocation = alloc.Allocate(size);
  alloc.Release(platform::XPUPlace(0));
}
#endif

}  // namespace allocation
}  // namespace memory
}  // namespace paddle
