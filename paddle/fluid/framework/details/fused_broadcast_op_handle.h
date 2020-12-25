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

#include <map>
#include <string>
#include <vector>

#include "paddle/fluid/framework/details/broadcast_op_handle.h"
#include "paddle/fluid/framework/details/multi_devices_helper.h"
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/framework/selected_rows.h"
#include "paddle/fluid/platform/device_context.h"

namespace paddle {
namespace framework {
namespace ir {
class Node;
}  // namespace ir
}  // namespace framework
namespace platform {
struct NCCLContextMap;
}  // namespace platform
}  // namespace paddle

#if defined(PADDLE_WITH_NCCL)
#include "paddle/fluid/platform/nccl_helper.h"
#endif

namespace paddle {
namespace framework {
namespace details {

struct FusedBroadcastOpHandle : public BroadcastOpHandle {
 public:
#if defined(PADDLE_WITH_NCCL)
  FusedBroadcastOpHandle(ir::Node *node,
                         const std::vector<Scope *> local_scopes,
                         const std::vector<platform::Place> &places,
                         const platform::NCCLContextMap *nccl_ctx)
      : BroadcastOpHandle(node, local_scopes, places, nccl_ctx) {}
#endif
#if defined(PADDLE_WITH_XPU_BKCL)
  FusedBroadcastOpHandle(ir::Node *node,
                         const std::vector<Scope *> local_scopes,
                         const std::vector<platform::Place> &places,
                         const platform::BKCLContextMap *bkcl_ctx)
      : BroadcastOpHandle(node, local_scopes, places, bkcl_ctx) {}
#endif
  FusedBroadcastOpHandle(ir::Node *node,
                         const std::vector<Scope *> local_scopes,
                         const std::vector<platform::Place> &places)
      : BroadcastOpHandle(node, local_scopes, places) {}
  std::string Name() const override;

 protected:
  void RunImpl() override;
};

}  // namespace details
}  // namespace framework
}  // namespace paddle
