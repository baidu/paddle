// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include "paddle/fluid/framework/details/computation_op_handle.h"
#include "paddle/fluid/framework/details/multi_devices_helper.h"
#include "paddle/fluid/framework/details/share_tensor_buffer_op_handle.h"
#include "paddle/fluid/framework/ir/memory_optimize_pass/memory_optimization_var_info.h"
#include "paddle/fluid/framework/ir/memory_optimize_pass/memory_reuse_pass.h"
#include "paddle/fluid/framework/ir/memory_optimize_pass/reference_count_pass_helper.h"
#include "paddle/fluid/framework/ir/pass.h"

namespace paddle {
namespace framework {
namespace ir {

class BufferSharedInplaceOpPass : public MemoryReusePass {
 protected:
  std::string ReuseType() const override { return "inplace"; }

  void Run(Graph *graph) const override;
};

void BufferSharedInplaceOpPass::Run(Graph *graph) const {
  const auto &last_live_ops =
      Get<std::vector<LastLiveOpsOfVars>>(kLastLiveOpsOfVars);

  bool use_cuda = Get<bool>(kUseCuda);

  // Step 1: Build a reverse map of last_live_ops
  // i.e.: op -> vars
  std::unordered_map<details::ComputationOpHandle *,
                     std::unordered_map<std::string, ir::Node *>>
      candidate_ops;
  for (auto &each_scope_ops : last_live_ops) {
    for (auto &pair : each_scope_ops) {
      // If variable has more than 1 last lived ops, this variable cannot
      // be inplaced.
      if (pair.second.ops().size() != 1) {
        continue;
      }

      auto *op = *(pair.second.ops().begin());
      const std::string &op_type = op->GetOp()->Type();
      const framework::OpDesc *op_desc = op->Node()->Op();
      PADDLE_ENFORCE_NOT_NULL(
          op_desc, platform::errors::NotFound("Op(%s) can not find opdesc.",
                                              op->Name()));

      auto &infer_inplace = OpInfoMap::Instance().Get(op_type).infer_inplace_;
      if (!infer_inplace) {
        continue;
      }

      const std::string &var_name = pair.first;
      auto in_nodes = this->FindNodesByName(var_name, op->Node()->inputs);
      if (in_nodes.size() == 1) {
        candidate_ops[op][var_name] = *in_nodes.begin();
      }
    }
  }

  // Step 2: Check which vars can be inplaced indeed
  for (auto &op_vars_pair : candidate_ops) {
    auto *op = op_vars_pair.first;
    auto &vars = op_vars_pair.second;

    const std::string &op_type = op->GetOp()->Type();
    auto *op_desc = op->Node()->Op();

    auto in_to_outs =
        OpInfoMap::Instance().Get(op_type).infer_inplace_(use_cuda);

    if (IsIdentityOp(op_type)) {
      auto identity_in_to_outs = InOutPairOfIdentityOp(op_type);
      for (auto &in_out_pair : identity_in_to_outs) {
        VLOG(4) << "WARN: identity inplace op " << op_type << " "
                << in_out_pair.first << " -> " << in_out_pair.second
                << " would be considered inside "
                   "buffer_shared_identity_inplace_op_pass";
        in_to_outs.erase(in_out_pair.first);
      }
    }

    for (auto &pair : in_to_outs) {
      auto &in_param = pair.first;
      auto &in_args = op_desc->Input(in_param);
      if (in_args.empty()) {
        VLOG(4) << "Cannot inplace because Input(" << in_param
                << ") is empty in " << op_type;
        continue;
      }

      auto &in_arg = in_args[0];
      auto iter = vars.find(in_arg);
      if (iter == vars.end()) {
        VLOG(4) << "Cannot inplace maybe because Input(" << in_param
                << ")=" << in_arg << " is not lastly used in op " << op_type
                << ", or it occurs multiple times in input or occurs in output";
        continue;
      }

      ir::Node *in_node = iter->second;

      auto &out_param = pair.second;
      auto &out_args = op_desc->Output(out_param);

      if (out_args.empty()) {
        VLOG(4) << "Cannot inplace because Output(" << out_param
                << ") is empty in " << op_type;
        continue;
      }

      auto &out_arg = out_args[0];
      auto out_nodes = this->FindNodesByName(out_arg, op->Node()->outputs);
      if (out_nodes.size() != 1) {
        VLOG(4) << "Cannot inplace because Output(" << out_param
                << ")=" << out_arg << " occurs " << out_nodes.size()
                << " time(s) in output of op " << op_type;
        continue;
      }

      auto *out_node = *out_nodes.begin();

      auto &in_var_handle = in_node->Wrapper<details::VarHandleBase>();
      auto &out_var_handle = out_node->Wrapper<details::VarHandleBase>();

      auto *in_var_handle_ptr =
          dynamic_cast<details::VarHandle *>(&in_var_handle);
      auto *out_var_handle_ptr =
          dynamic_cast<details::VarHandle *>(&out_var_handle);

      if (in_var_handle_ptr == nullptr || out_var_handle_ptr == nullptr) {
        continue;
      }

      bool success = this->TryReuseVar(in_var_handle_ptr, out_var_handle_ptr);
      if (success) {
        VLOG(4) << "Inplace performed in op " << op_type << ": "
                << in_var_handle_ptr->Name() << " -> "
                << out_var_handle_ptr->Name()
                << ". Debug String is: " << op->GetOp()->DebugString();
      } else {
        VLOG(3) << "Inplace failed in op " << op_type << ": "
                << in_var_handle_ptr->Name() << " -> "
                << out_var_handle_ptr->Name();
      }
    }
  }
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(buffer_shared_inplace_pass,
              paddle::framework::ir::BufferSharedInplaceOpPass)
    .RequirePassAttr(paddle::framework::ir::kMemOptVarInfoMapList)
    .RequirePassAttr(paddle::framework::ir::kLastLiveOpsOfVars)
    .RequirePassAttr(paddle::framework::ir::kUseCuda)
    .RequirePassAttr(paddle::framework::ir::kSkipReuseVars);
