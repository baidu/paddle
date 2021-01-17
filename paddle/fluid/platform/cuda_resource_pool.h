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

#if defined PADDLE_WITH_CUDA || defined PADDLE_WITH_HIP
#if defined PADDLE_WITH_CUDA
#include <cuda.h>
#include <cuda_runtime.h>
#elif defined PADDLE_WITH_HIP
#include <hip/hip_runtime.h>
#endif
#include <memory>
#include <type_traits>
#include <vector>

#include "paddle/fluid/platform/resource_pool.h"

namespace paddle {
namespace platform {

#if defined PADDLE_WITH_CUDA
using CudaStreamObject = std::remove_pointer<cudaStream_t>::type;
using CudaEventObject = std::remove_pointer<cudaEvent_t>::type;
#elif defined PADDLE_WITH_HIP
using CudaStreamObject = std::remove_pointer<hipStream_t>::type;
using CudaEventObject = std::remove_pointer<hipEvent_t>::type;
#endif

class CudaStreamResourcePool {
 public:
  std::shared_ptr<CudaStreamObject> New(int dev_idx);

  static CudaStreamResourcePool &Instance();

 private:
  CudaStreamResourcePool();

  DISABLE_COPY_AND_ASSIGN(CudaStreamResourcePool);

 private:
  std::vector<std::shared_ptr<ResourcePool<CudaStreamObject>>> pool_;
};

class CudaEventResourcePool {
 public:
  std::shared_ptr<CudaEventObject> New(int dev_idx);

  static CudaEventResourcePool &Instance();

 private:
  CudaEventResourcePool();

  DISABLE_COPY_AND_ASSIGN(CudaEventResourcePool);

 private:
  std::vector<std::shared_ptr<ResourcePool<CudaEventObject>>> pool_;
};

}  // namespace platform
}  // namespace paddle

#endif
