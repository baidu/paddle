//   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

#include <string>
#include <utility>
#include <vector>

#include "paddle/fluid/framework/details/all_reduce_op_handle.h"
#include "paddle/fluid/framework/details/op_handle_base.h"
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/scope.h"

namespace paddle {
namespace framework {
namespace ir {
class Node;
}  // namespace ir
}  // namespace framework
namespace platform {
class NCCLCommunicator;
}  // namespace platform
}  // namespace paddle
#if defined(PADDLE_WITH_NCCL)
#include "paddle/fluid/framework/details/nccl_op_handle.h"
#include "paddle/fluid/platform/nccl_helper.h"
#elif defined(PADDLE_WITH_XPU_BKCL)
#include "paddle/fluid/platform/bkcl_helper.h"
#endif

namespace paddle {
namespace framework {
namespace details {

#if defined(PADDLE_WITH_NCCL)
struct FusedAllReduceOpHandle : public AllReduceOpHandle {
  FusedAllReduceOpHandle(ir::Node *node,
                         const std::vector<Scope *> &local_scopes,
                         const std::vector<platform::Place> &places,
                         const size_t num_of_all_reduce,
                         const platform::NCCLCommunicator *ctxs);
#elif defined(PADDLE_WITH_XPU_BKCL)
struct FusedAllReduceOpHandle : public AllReduceOpHandle {
  FusedAllReduceOpHandle(ir::Node *node,
                         const std::vector<Scope *> &local_scopes,
                         const std::vector<platform::Place> &places,
                         const size_t num_of_all_reduce,
                         const platform::BKCLCommunicator *ctxs);
#else
struct FusedAllReduceOpHandle : public AllReduceOpHandle {
  FusedAllReduceOpHandle(ir::Node *node,
                         const std::vector<Scope *> &local_scopes,
                         const std::vector<platform::Place> &places,
                         const size_t num_of_all_reduce);
#endif
  std::string Name() const override;

 protected:
  void RunImpl() override;

 private:
  size_t num_of_all_reduce_;

  // Check the dtype of the input
  void GetDTypeAndNumel(
      const std::vector<std::pair<std::string, const LoDTensor *>> &g_tensor,
      proto::VarType::Type *dtype, int64_t *total_num) const;

  // Get gradient's name and LoDTensor
  void GetGradLoDTensor(const size_t &scope_idx,
                        const std::vector<VarHandle *> &in_var_handles,
                        const std::vector<VarHandle *> &out_var_handles,
                        std::vector<std::pair<std::string, const LoDTensor *>>
                            *grad_tensor) const;

  bool InputIsInDifferentPlace(
      const std::vector<VarHandle *> &in_var_handles) const;

  void FusedAllReduceFunc(const std::vector<VarHandle *> &in_var_handles,
                          const std::vector<VarHandle *> &out_var_handles);
};

}  // namespace details
}  // namespace framework
}  // namespace paddle
