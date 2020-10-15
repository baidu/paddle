/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

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

#include <NvInfer.h>
#if !defined(_WIN32)
#include <dlfcn.h>
#endif

#include <mutex>  // NOLINT

#include "paddle/fluid/platform/dynload/dynamic_loader.h"
#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace platform {
namespace dynload {

void* GetTensorRtHandle();

extern std::once_flag tensorrt_dso_flag;
extern void* tensorrt_dso_handle;

#define DECLARE_DYNAMIC_LOAD_TENSORRT_WRAP(__name)                            \
  struct DynLoad__##__name {                                                  \
    template <typename... Args>                                               \
    auto operator()(Args... args) -> DECLARE_TYPE(__name, args...) {          \
      std::call_once(tensorrt_dso_flag, []() {                                \
        tensorrt_dso_handle = paddle::platform::dynload::GetTensorRtHandle(); \
      });                                                                     \
      static void* p_##__name = dlsym(tensorrt_dso_handle, #__name);          \
      if (p_##__name == nullptr) {                                            \
        return nullptr;                                                       \
      }                                                                       \
      using tensorrt_func = decltype(&::__name);                              \
      return reinterpret_cast<tensorrt_func>(p_##__name)(args...);            \
    }                                                                         \
  };                                                                          \
  extern DynLoad__##__name __name

#if (NV_TENSORRT_MAJOR >= 6)
#define TENSORRT_RAND_ROUTINE_EACH(__macro) \
  __macro(createInferBuilder_INTERNAL);     \
  __macro(createInferRuntime_INTERNAL);     \
  __macro(getPluginRegistry);
#else
#define TENSORRT_RAND_ROUTINE_EACH(__macro) \
  __macro(createInferBuilder_INTERNAL);     \
  __macro(createInferRuntime_INTERNAL);
#endif

TENSORRT_RAND_ROUTINE_EACH(DECLARE_DYNAMIC_LOAD_TENSORRT_WRAP)

}  // namespace dynload
}  // namespace platform
}  // namespace paddle
