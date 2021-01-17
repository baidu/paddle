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

#include <hip/hip_runtime.h>
#include <mutex>  // NOLINT

#include "paddle/fluid/platform/dynload/dynamic_loader.h"
#include "paddle/fluid/platform/port.h"

namespace paddle {
namespace platform {
namespace dynload {

extern std::once_flag rocm_dso_flag;
extern void* rocm_dso_handle;
extern bool HasCUDADriver();

#define DECLARE_DYNAMIC_LOAD_ROCM_WRAP(__name)                           \
  struct DynLoad__##__name {                                             \
    template <typename... Args>                                          \
    auto operator()(Args... args) -> DECLARE_TYPE(__name, args...) {     \
      using rocm_func = decltype(&::__name);                             \
      std::call_once(rocm_dso_flag, []() {                               \
        rocm_dso_handle = paddle::platform::dynload::GetCUDADsoHandle(); \
      });                                                                \
      static void* p_##__name = dlsym(rocm_dso_handle, #__name);         \
      return reinterpret_cast<rocm_func>(p_##__name)(args...);           \
    }                                                                    \
  };                                                                     \
  extern struct DynLoad__##__name __name

/**
 * include all needed cuda driver functions
 **/
#define ROCM_ROUTINE_EACH(__macro)                            \
  __macro(hipDriverGetVersion);                               \
  __macro(hipGetDeviceCount);                                 \
  __macro(hipGetErrorString);                                 \
  __macro(hipModuleLoadData);                                 \
  __macro(hipModuleGetFunction);                              \
  __macro(hipModuleUnload);                                   \
  /*rocm3.5 not support the function*/                        \
  /* __macro(hipOccupancyMaxActiveBlocksPerMultiprocessor);*/ \
  __macro(hipModuleLaunchKernel);                             \
  __macro(hipLaunchKernel);                                   \
  __macro(hipGetDevice);                                      \
  __macro(hipDevicePrimaryCtxGetState)

ROCM_ROUTINE_EACH(DECLARE_DYNAMIC_LOAD_ROCM_WRAP);

#undef DECLARE_DYNAMIC_LOAD_ROCM_WRAP

}  // namespace dynload
}  // namespace platform
}  // namespace paddle
