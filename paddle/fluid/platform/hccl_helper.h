//   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

#if defined(PADDLE_WITH_HCCL) || defined(PADDLE_WITH_RCCL) || \
    defined(PADDLE_WITH_ASCEND_CL)

#include <stdio.h>
#include <memory>
#include <string>
#include <thread>  // NOLINT
#include <typeindex>
#include <unordered_map>
#include <vector>

#ifdef PADDLE_WITH_ASCEND_CL
#include "paddle/fluid/platform/dynload/hccl.h"
#endif

#include "paddle/fluid/framework/data_type.h"
#include "paddle/fluid/platform/collective_helper.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/fluid/platform/float16.h"

#define HCCL_ID_VARNAME "HCCLID"

namespace paddle {
namespace platform {

inline EcclDataType ToHCCLDataType(framework::proto::VarType::Type type) {
  if (type == framework::proto::VarType::FP32) {
    return DT_FLOAT32;
  } else if (type == framework::proto::VarType::FP16) {
    return DT_FLOAT16;
  } else if (type == framework::proto::VarType::INT32) {
    return DT_INT32;
  } else if (type == framework::proto::VarType::INT8) {
    return DT_INT8;
  } else {
    PADDLE_THROW(platform::errors::Unimplemented(
        "This datatype in hccl is not supported."));
  }
}

}  // namespace platform
}  // namespace paddle
#endif
