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

// #include <hccl/hccl.h>
// #include <hccl/hccl_types.h>
#include <mutex>  // NOLINT

#include "paddle/fluid/platform/dynload/dynamic_loader.h"
#include "paddle/fluid/platform/dynload/hcom.h"
#include "paddle/fluid/platform/port.h"

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

#define HCCL_RAND_ROUTINE_EACH(__macro)         \
  __macro(hcom_init);                           \
  __macro(hcom_destroy);                        \
  __macro(hcom_bind_model);                     \
  __macro(hcom_unbind_model);                   \
  __macro(hcom_send);                           \
  __macro(hcom_receive);                        \
  __macro(hcom_broadcast);                      \
  __macro(hcom_all_gather);                     \
  __macro(hcom_all_reduce);                     \
  __macro(hcom_reduce_scatter);                 \
  __macro(hcom_create_group);                   \
  __macro(hcom_destroy_group);                  \
  __macro(hcom_get_rank_id);                    \
  __macro(hcom_get_local_rank_id);              \
  __macro(hcom_get_local_rank_size);            \
  __macro(hcom_get_split_strategy);             \
  __macro(hcom_set_split_strategy_by_size);     \
  __macro(hcom_set_split_strategy_by_index);    \
  __macro(hcom_get_group_rank_from_world_rank); \
  __macro(hcom_get_world_rank_from_group_rank);

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
