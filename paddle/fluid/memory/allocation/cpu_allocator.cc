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

#include "paddle/fluid/memory/allocation/cpu_allocator.h"

#include <cstdlib>

#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace memory {
namespace allocation {

bool CPUAllocator::IsAllocThreadSafe() const { return true; }

void CPUAllocator::FreeImpl(Allocation *allocation) {
  void *p = allocation->ptr();
#ifdef _WIN32
  _aligned_free(p);
#else
  free(p);
#endif
  delete allocation;
}

Allocation *CPUAllocator::AllocateImpl(size_t size) {
  void *p;
#ifdef _WIN32
  p = _aligned_malloc(size, kAlignment);
#else
  int error = posix_memalign(&p, kAlignment, size);
  PADDLE_ENFORCE_EQ(
      error, 0,
      platform::errors::ResourceExhausted(
          "Fail to alloc memory of %ld size, error code is %d.", size, error));
#endif
  return new Allocation(p, size, platform::CPUPlace());
}
}  // namespace allocation
}  // namespace memory
}  // namespace paddle
