/* Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.

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

#ifdef PADDLE_WITH_HIP
#include <hip/hip_runtime.h>
#elif defined PADDLE_WITH_ASCEND_CL
#include "acl/acl.h"
#else
#include <cuda_runtime.h>
#endif

namespace paddle {

#ifdef PADDLE_WITH_HIP
#define gpuSuccess hipSuccess
using gpuStream_t = hipStream_t;
using gpuError_t = hipError_t;
using gpuEvent_t = hipEvent_t;
#elif defined PADDLE_WITH_ASCEND_CL
using npuEvent_t = aclrtEvent;
#else
#define gpuSuccess cudaSuccess
using gpuStream_t = cudaStream_t;
using gpuError_t = cudaError_t;
using gpuEvent_t = cudaEvent_t;
#endif

}  // namespace paddle
