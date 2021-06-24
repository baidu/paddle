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

#pragma once
#include <stddef.h>

#include "paddle/fluid/platform/cpu_info.h"
#include "paddle/fluid/platform/place.h"
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
#include "paddle/fluid/platform/gpu_info.h"
#elif defined(PADDLE_WITH_ASCEND_CL)
#include "paddle/fluid/platform/npu_info.h"
#endif
#if defined(PADDLE_WITH_ASCEND_CL)
#include "paddle/fluid/platform/npu_info.h"
#endif

namespace paddle {
namespace platform {
size_t Alignment(size_t size, const platform::Place &place,
                 int align_size = -1);
}  // namespace platform
}  // namespace paddle
