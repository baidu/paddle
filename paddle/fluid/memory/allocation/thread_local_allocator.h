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

#pragma once

#include <memory>
#include "paddle/fluid/memory/allocation/allocator.h"
#include "paddle/fluid/memory/detail/buddy_allocator.h"
#include "paddle/fluid/memory/detail/system_allocator.h"
#include "paddle/fluid/platform/gpu_info.h"

namespace paddle {
namespace memory {
namespace allocation {

class ThreadLocalAllocator
    : public Allocator,
      public std::enable_shared_from_this<ThreadLocalAllocator> {
 public:
  explicit ThreadLocalAllocator(const platform::Place& p) : place_(p) {
    if (platform::is_gpu_place(place_)) {
      buddy_allocator_.reset(new memory::detail::BuddyAllocator(
          std::unique_ptr<memory::detail::SystemAllocator>(
              new memory::detail::GPUAllocator(
                  boost::get<platform::CUDAPlace>(place_).device)),
          platform::GpuMinChunkSize(), platform::GpuMaxChunkSize()));
    } else {
      LOG(FATAL) << "Thread local allocator only supports CUDAPlace now.";
    }
  }

 protected:
  Allocation* AllocateImpl(size_t size) override;
  void FreeImpl(Allocation* allocation) override;

 private:
  std::unique_ptr<memory::detail::BuddyAllocator> buddy_allocator_;
  platform::Place place_;
};

const std::shared_ptr<Allocator>& GetThreadLocalAllocator();

}  // namespace allocation
}  // namespace memory
}  // namespace paddle
