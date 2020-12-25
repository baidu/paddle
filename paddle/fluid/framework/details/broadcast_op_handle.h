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

#include "paddle/fluid/framework/details/op_handle_base.h"
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/framework/selected_rows.h"
#include "paddle/fluid/platform/device_context.h"

namespace paddle {
namespace framework {
namespace details {
struct VarHandle;
}  // namespace details
namespace ir {
class Node;
}  // namespace ir
}  // namespace framework
namespace platform {
#if defined(PADDLE_WITH_NCCL)
struct NCCLContextMap;
#endif
#if defined(PADDLE_WITH_XPU_BKCL)
struct BKCLContextMap;
#endif
}  // namespace platform
}  // namespace paddle

#if defined(PADDLE_WITH_NCCL)
#include "paddle/fluid/platform/nccl_helper.h"
#elif defined(PADDLE_WITH_XPU_BKCL)
#include "paddle/fluid/platform/bkcl_helper.h"
#endif

namespace paddle {
namespace framework {
namespace details {

struct BroadcastOpHandle : public OpHandleBase {
 public:
#if defined(PADDLE_WITH_NCCL)
  BroadcastOpHandle(ir::Node *node, const std::vector<Scope *> &local_scopes,
                    const std::vector<platform::Place> &places,
                    const platform::NCCLContextMap *nccl_ctxs)
      : OpHandleBase(node),
        local_scopes_(local_scopes),
        places_(places),
        nccl_ctxs_(nccl_ctxs) {
    if (nccl_ctxs_) {
      for (auto &p_ctx : nccl_ctxs_->contexts_) {
        this->SetDeviceContext(platform::CUDAPlace(p_ctx.first),
                               p_ctx.second.ctx_.get());
      }
    }
  }
#endif
#if defined(PADDLE_WITH_XPU_BKCL)
  BroadcastOpHandle(ir::Node *node, const std::vector<Scope *> &local_scopes,
                    const std::vector<platform::Place> &places,
                    const platform::BKCLContextMap *bkcl_ctxs)
      : OpHandleBase(node),
        local_scopes_(local_scopes),
        places_(places),
        bkcl_ctxs_(bkcl_ctxs) {
    if (bkcl_ctxs_) {
      for (auto &p_ctx : bkcl_ctxs_->contexts_) {
        this->SetDeviceContext(platform::XPUPlace(p_ctx.first),
                               p_ctx.second.ctx_.get());
      }
    }
  }
#endif
  BroadcastOpHandle(ir::Node *node, const std::vector<Scope *> &local_scopes,
                    const std::vector<platform::Place> &places)
      : OpHandleBase(node), local_scopes_(local_scopes), places_(places) {}

  std::string Name() const override;

  bool IsMultiDeviceTransfer() override { return true; };

 protected:
  void RunImpl() override;

  std::vector<Scope *> GetLocalScopes() override { return local_scopes_; }

  void BroadcastOneVar(const VarHandle &in_var_handle,
                       const std::vector<VarHandle *> &out_var_handles,
                       const std::vector<Scope *> &var_scopes);

  std::vector<Scope *> local_scopes_;
  std::vector<platform::Place> places_;
#if defined(PADDLE_WITH_NCCL)
  const platform::NCCLContextMap *nccl_ctxs_;
#elif defined(PADDLE_WITH_XPU_BKCL)
  const platform::BKCLContextMap *bkcl_ctxs_;
#endif

  void InitOutputValue(const VarHandle &in_var_handle,
                       const std::vector<VarHandle *> &out_var_handles) const;
};
}  // namespace details
}  // namespace framework
}  // namespace paddle
