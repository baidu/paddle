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

#include "paddle/fluid/memory/allocation/buffered_allocator.h"

#include <memory>

namespace paddle {
namespace memory {
namespace allocation {

BufferedAllocator::BufferedAllocator(std::shared_ptr<Allocator> allocator)
    : underlying_allocator_(std::move(allocator)) {
  PADDLE_ENFORCE_NOT_NULL(
      underlying_allocator_,
      platform::errors::InvalidArgument(
          "Underlying allocator of BufferedAllocator is NULL"));
  if (underlying_allocator_->IsAllocThreadSafe()) {
    mtx_ = std::make_unique<std::mutex>();
  }
}

BufferedAllocator::~BufferedAllocator() { FreeCache(-1UL); }

void BufferedAllocator::FreeCache(size_t size) {
  platform::LockGuardPtr<std::mutex> guard(mtx_);
  if (UNLIKELY(size == 0)) return;
  size_t cur = 0;
  while (!allocations_.empty()) {  // free the largest
    auto it = --allocations_.end();
    cur += it->second->size();
    underlying_allocator_->Free(it->second.release());
    allocations_.erase(it);
    if (cur >= size) return;
  }
}

bool BufferedAllocator::IsAllocThreadSafe() const { return mtx_ != nullptr; }

void BufferedAllocator::FreeImpl(Allocation *allocation) {
  platform::LockGuardPtr<std::mutex> guard(mtx_);
  allocations_.emplace(allocation->size(), AllocationPtr(allocation));
}

Allocation *BufferedAllocator::AllocateImpl(size_t size) {
  {
    platform::LockGuardPtr<std::mutex> guard(mtx_);
    auto it = allocations_.lower_bound(size);
    if (it != allocations_.end() && it->first < size * 2) {
      AllocationPtr result(std::move(it->second));
      allocations_.erase(it);
      return result.release();
    }
  }

  try {
    return underlying_allocator_->Allocate(size).release();
  } catch (BadAlloc &) {
    FreeCache(size);
    return underlying_allocator_->Allocate(size).release();
  }
}

}  // namespace allocation
}  // namespace memory
}  // namespace paddle
