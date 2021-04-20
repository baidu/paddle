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

#include "paddle/fluid/platform/dynload/hccl.h"

namespace paddle {
namespace platform {
namespace dynload {

std::once_flag hccl_dso_flag;
void *hccl_dso_handle;

#define DEFINE_WRAP(__name) DynLoad__##__name __name

HCCL_RAND_ROUTINE_EACH(DEFINE_WRAP);

#if HCCL_VERSION_CODE >= 2212
HCCL_RAND_ROUTINE_EACH_AFTER_2212(DEFINE_WRAP)
#endif

#if HCCL_VERSION_CODE >= 2703
HCCL_RAND_ROUTINE_EACH_AFTER_2703(DEFINE_WRAP)
#endif

}  // namespace dynload
}  // namespace platform
}  // namespace paddle
