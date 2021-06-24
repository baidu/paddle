/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/platform/device_memory_aligment.h"

namespace paddle {
namespace platform {
size_t Alignment(size_t size, const platform::Place &place, int align_size) {
  size_t alignment = 0;
  if (align_size > 0) {
    alignment = align_size;
  } else {
    alignment = 1024;
    if (platform::is_cpu_place(place)) {
      alignment = CpuMinChunkSize();
    } else {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      alignment = GpuMinChunkSize();
#elif defined(PADDLE_WITH_XPU)
      alignment = alignment;
#elif defined(PADDLE_WITH_ASCEND_CL)
      alignment = NPUMinChunkSize();
#else
      PADDLE_THROW(platform::errors::PreconditionNotMet(
          "Fluid is not compiled with CUDA/XPU/NPU."));
#endif
    }
  }
  size_t remaining = size % alignment;
  return remaining == 0 ? size : size + (alignment - remaining);
}
}  // namespace platform
}  // namespace paddle
