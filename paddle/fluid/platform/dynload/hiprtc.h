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

#include <hip/hiprtc.h>
#include <mutex>  // NOLINT
#include "paddle/fluid/platform/dynload/dynamic_loader.h"
#include "paddle/fluid/platform/port.h"

namespace paddle {
namespace platform {
namespace dynload {

extern std::once_flag hiprtc_dso_flag;
extern void* hiprtc_dso_handle;
extern bool HasNVRTC();

#define DECLARE_DYNAMIC_LOAD_HIPRTC_WRAP(__name)                            \
  struct DynLoad__##__name {                                                \
    template <typename... Args>                                             \
    auto operator()(Args... args) -> DECLARE_TYPE(__name, args...) {        \
      using hiprtc_func = decltype(&::__name);                              \
      std::call_once(hiprtc_dso_flag, []() {                                \
        hiprtc_dso_handle = paddle::platform::dynload::GetNVRTCDsoHandle(); \
      });                                                                   \
      static void* p_##__name = dlsym(hiprtc_dso_handle, #__name);          \
      return reinterpret_cast<hiprtc_func>(p_##__name)(args...);            \
    }                                                                       \
  };                                                                        \
  extern struct DynLoad__##__name __name

/**
 * include all needed hiprtc functions
 **/
#define HIPRTC_ROUTINE_EACH(__macro) \
  __macro(hiprtcVersion);            \
  __macro(hiprtcGetErrorString);     \
  __macro(hiprtcCompileProgram);     \
  __macro(hiprtcCreateProgram);      \
  __macro(hiprtcDestroyProgram);     \
  __macro(hiprtcGetCode);            \
  __macro(hiprtcGetCodeSize);        \
  __macro(hiprtcGetProgramLog);      \
  __macro(hiprtcGetProgramLogSize)

HIPRTC_ROUTINE_EACH(DECLARE_DYNAMIC_LOAD_HIPRTC_WRAP);

#undef DECLARE_DYNAMIC_LOAD_HIPRTC_WRAP

}  // namespace dynload
}  // namespace platform
}  // namespace paddle
