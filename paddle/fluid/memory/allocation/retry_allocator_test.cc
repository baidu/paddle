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

#include "paddle/fluid/memory/allocation/retry_allocator.h"

#include <algorithm>
#include <chrono>              // NOLINT
#include <condition_variable>  // NOLINT
#include <mutex>               // NOLINT
#include <string>
#include <thread>  // NOLINT
#include <vector>

#include "gtest/gtest.h"
#include "paddle/fluid/memory/allocation/best_fit_allocator.h"
#include "paddle/fluid/memory/allocation/cpu_allocator.h"
#include "paddle/fluid/memory/allocation/locked_allocator.h"
#ifdef PADDLE_WITH_CUDA
#include "paddle/fluid/memory/allocation/cuda_allocator.h"
#endif

namespace paddle {
namespace memory {
namespace allocation {

TEST(RetryAllocator, RetryAllocator) {
  CPUAllocator cpu_allocator;

  size_t size = (1 << 20);
  auto cpu_allocation = cpu_allocator.Allocate(size);

  std::unique_ptr<BestFitAllocator> best_fit_allocator(
      new BestFitAllocator(cpu_allocation.get()));
  std::unique_ptr<LockedAllocator> locked_allocator(
      new LockedAllocator(std::move(best_fit_allocator)));

  size_t thread_num = 4;
  size_t sleep_time = 40;
  size_t extra_time = 20;

  // Reserve to perform more tests in the future
  std::vector<std::shared_ptr<Allocator>> allocators;
  {
    std::unique_ptr<BestFitAllocator> best_fit_allocator(
        new BestFitAllocator(cpu_allocation.get()));
    std::unique_ptr<LockedAllocator> locked_allocator(
        new LockedAllocator(std::move(best_fit_allocator)));
    allocators.push_back(std::make_shared<RetryAllocator>(
        std::move(locked_allocator),
        (thread_num - 1) * (sleep_time + extra_time)));
  }

  for (auto &allocator : allocators) {
    std::vector<std::thread> threads(thread_num);
    std::vector<void *> addresses(threads.size(), nullptr);

    std::mutex mutex;
    std::condition_variable cv;
    bool flag = false;

    for (size_t i = 0; i < threads.size(); ++i) {
      threads[i] = std::thread([&, i]() {
        {
          std::unique_lock<std::mutex> lock(mutex);
          cv.wait(lock, [&] { return flag; });
        }

        auto ret = allocator->Allocate(size - 1);
        addresses[i] = ret->ptr();
        std::this_thread::sleep_for(std::chrono::milliseconds(sleep_time));
      });
    }

    {
      std::lock_guard<std::mutex> lock(mutex);
      flag = true;
      cv.notify_all();
    }

    for (auto &th : threads) {
      th.join();
    }

    void *val = cpu_allocation->ptr();
    bool is_all_equal = std::all_of(addresses.begin(), addresses.end(),
                                    [val](void *p) { return p == val; });
    ASSERT_TRUE(is_all_equal);
  }
}

class DummyAllocator : public Allocator {
 public:
  bool IsAllocThreadSafe() const override { return true; }

 protected:
  Allocation *AllocateImpl(size_t size) override {
    PADDLE_THROW_BAD_ALLOC(platform::errors::ResourceExhausted(
        "Here is a test exception, always BadAlloc."));
  }

  void FreeImpl(Allocation *) override {}
};

TEST(RetryAllocator, RetryAllocatorLastAllocFailure) {
  size_t retry_ms = 10;
  {
    RetryAllocator allocator(std::make_shared<DummyAllocator>(), retry_ms);
    try {
      auto allocation = allocator.Allocate(100);
      ASSERT_TRUE(false);
      allocation.reset();
    } catch (BadAlloc &ex) {
      ASSERT_TRUE(std::string(ex.what()).find("always BadAlloc") !=
                  std::string::npos);
    }
  }

#ifdef PADDLE_WITH_CUDA
  {
    platform::CUDAPlace p(0);
    RetryAllocator allocator(std::make_shared<CUDAAllocator>(p), retry_ms);
    size_t allocate_size = (static_cast<size_t>(1) << 40);  // Very large number
    try {
      auto allocation = allocator.Allocate(allocate_size);
      ASSERT_TRUE(false);
      allocation.reset();
    } catch (BadAlloc &ex) {
      ASSERT_TRUE(std::string(ex.what()).find("Cannot allocate") !=
                  std::string::npos);
    }
  }
#endif
}

}  // namespace allocation
}  // namespace memory
}  // namespace paddle
