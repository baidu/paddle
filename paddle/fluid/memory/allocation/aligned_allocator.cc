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

#include "paddle/fluid/memory/allocation/aligned_allocator.h"

namespace paddle {
namespace memory {
namespace allocation {

ThinAlignedAllocator::ThinAlignedAllocator(
    std::shared_ptr<ManagedAllocator> underlyning_allocator)
    : underlying_allocator_(std::move(underlyning_allocator)) {}

std::shared_ptr<Allocation> ThinAlignedAllocator::AllocateShared(
    size_t size, Allocator::Attr attr) {
  return std::shared_ptr<Allocation>(Allocate(size, attr).release());
}
}  // namespace allocation
}  // namespace memory
}  // namespace paddle
