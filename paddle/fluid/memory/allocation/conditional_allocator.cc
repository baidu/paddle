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

#include "paddle/fluid/memory/allocation/conditional_allocator.h"

namespace paddle {
namespace memory {
namespace allocation {

ConditionalAllocator& ConditionalAllocator::AddAllocator(
    std::function<bool(size_t, Allocator::Attr)> func,
    std::shared_ptr<ManagedAllocator> allocator) {
  underlying_allocators_.emplace_back(std::move(func), std::move(allocator));
  return *this;
}
std::unique_ptr<Allocation> ConditionalAllocator::Allocate(
    size_t size, Allocator::Attr attr) {
  return SelectAndInvoke(size, attr, [&](ManagedAllocator& allocator) {
    return allocator.Allocate(size, attr);
  });
}
std::shared_ptr<Allocation> ConditionalAllocator::AllocateShared(
    size_t size, Allocator::Attr attr) {
  return SelectAndInvoke(size, attr, [&](ManagedAllocator& allocator) {
    return allocator.AllocateShared(size, attr);
  });
}
bool ConditionalAllocator::IsAllocThreadSafe() const { return true; }

}  // namespace allocation
}  // namespace memory
}  // namespace paddle
