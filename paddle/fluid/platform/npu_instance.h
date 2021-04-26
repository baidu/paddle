/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

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
#include <stddef.h>

#include <string>
#include <vector>

#include "acl/acl.h"
#include "paddle/fluid/platform/npu_info.h"

namespace paddle {
namespace platform {

class AclInstance {
 public:
  // NOTE(zhiiu): Commonly, exception in destructor is not recommended, so
  // no PADDLE_ENFORCE here, call acl API directly.
  ~AclInstance();
  AclInstance(const AclInstance &o) = delete;
  const AclInstance &operator=(const AclInstance &o) = delete;
  static AclInstance &Instance();
  void Finalize();

 private:
  // forbid calling default constructor
  AclInstance();
  std::vector<int> devices_;
};

}  // namespace platform
}  // namespace paddle

#endif
