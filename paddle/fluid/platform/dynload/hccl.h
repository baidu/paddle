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

#ifdef PADDLE_WITH_ASCEND_CL

#include <mutex>  // NOLINT

#include "paddle/fluid/platform/dynload/eccl.h"
#include "paddle/fluid/platform/dynload/eccl_types.h"

#include "paddle/fluid/platform/port.h"
#include "paddle/fluid/platform/dynload/dynamic_loader.h"

#define HCOM_GROUP_PREFIX "HCOM_GROUP_"

namespace paddle {
namespace platform {
namespace dynload {

extern std::once_flag hccl_dso_flag;
extern void* hccl_dso_handle;

#define DECLARE_DYNAMIC_LOAD_HCCL_WRAP(__name)                           \
  struct DynLoad__##__name {                                             \
    template <typename... Args>                                          \
    auto operator()(Args... args) -> decltype(__name(args...)) {         \
      using HCCL_func = decltype(&::__name);                             \
      std::call_once(hccl_dso_flag, []() {                               \
        hccl_dso_handle = paddle::platform::dynload::GetHCCLDsoHandle(); \
      });                                                                \
      static void* p_##__name = dlsym(hccl_dso_handle, #__name);         \
      return reinterpret_cast<HCCL_func>(p_##__name)(args...);           \
    }                                                                    \
  };                                                                     \
  extern DynLoad__##__name __name

#define HCCL_RAND_ROUTINE_EACH(__macro)       \
  __macro(eccl_gen_unique_id);                \
  __macro(eccl_init_comm_global);             \
  __macro(eccl_destroy_comm_global);          \
  __macro(eccl_reduce_scatter);               \
  __macro(eccl_all_reduce);                   \
  __macro(eccl_broadcast);                    \
  __macro(eccl_all_gather);                   \
  __macro(eccl_reduce);                       \
  __macro(eccl_sync_stream);

HCCL_RAND_ROUTINE_EACH(DECLARE_DYNAMIC_LOAD_HCCL_WRAP)

#if HCCL_VERSION_CODE >= 2212
#define HCCL_RAND_ROUTINE_EACH_AFTER_2212(__macro) __macro(HCCLBroadcast);
HCCL_RAND_ROUTINE_EACH_AFTER_2212(DECLARE_DYNAMIC_LOAD_HCCL_WRAP)
#endif

#if HCCL_VERSION_CODE >= 2703
#define HCCL_RAND_ROUTINE_EACH_AFTER_2703(__macro) \
  __macro(HCCLSend);                               \
  __macro(HCCLRecv);
HCCL_RAND_ROUTINE_EACH_AFTER_2703(DECLARE_DYNAMIC_LOAD_HCCL_WRAP)
#endif

}  // namespace dynload
}  // namespace platform
}  // namespace paddle
#endif
