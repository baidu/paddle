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

#pragma once

#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include "paddle/fluid/framework/details/computation_op_handle.h"
#include "paddle/fluid/framework/details/multi_devices_helper.h"
#include "paddle/fluid/framework/details/share_tensor_buffer_op_handle.h"
#include "paddle/fluid/framework/ir/graph.h"
#include "paddle/fluid/framework/ir/graph_helper.h"
#include "paddle/fluid/framework/ir/memory_optimize_pass/memory_optimization_var_info.h"
#include "paddle/fluid/framework/ir/memory_optimize_pass/reference_count_pass_helper.h"
#include "paddle/fluid/framework/ir/pass.h"

namespace paddle {
namespace framework {
namespace ir {

/*
 * MemoryReusePass is the base class of InplacePass and MemoryOptimizePass.
 *
 * Unlike the legacy Python API fluid.memory_optimize() which changes
 * variable names in the program/graph, MemoryReusePass inserts
 * ShareTensorBufferOpHandle into the graph. It is because if we use the
 * way of changing variable names:
 *
 * 1. There are so many corner cases we should skip. For example, (1) variables
 *    that relates to send/recv ops cannot be renamed (otherwise, pserver
 *    and trainer cannot find the matching variables), (2) ins/outs of ops
 *    containing sub-blocks cannot be optimized, (3) variables inside
 *    op_role_vars cannot be renamed.
 *
 * 2. It is very difficult to avoid reusing variables that users want to fetch.
 *    This is because the memory-optimize passes/transpiler runs before users
 *    fetch, i.e., exe.run(...). We cannot know what users want to fetch in the
 *    future. As a result, we have to set var.persistable = True before
 *    applying memory-optimize passes/transpiler, which is rather ugly and not
 *    friendly to users.
 *
 * 3. Dim and LoD of the reused variable would be changed, which may result
 *    in potential errors in InferShape stage of the following ops. What's
 *    more, it makes that we cannot use the information from
 *    NoNeedBufferVarsInference.
 *
 * Considering the drawbacks of the former renaming strategy, we design a
 * novel memory-optimize pass to fix these issues. Whether in-place is
 * performed can be decided during run-time. ShareTensorBufferOpHandle
 * would only share tensor buffers between in/out, never rename variable,
 * and not change dim and LoD of variable. If users want to fetch a certain
 * variable, we can skip in-place during run-time.
 *
 * The only concern on speed performance may be: there are too many
 * ShareTensorBufferOpHandles in the graph. This can be avoided by moving
 * tensor buffer sharing in each ComputationOpHandle::Run() method. We need
 * a pass to clean all ShareTensorBufferOpHandles and move sharing to
 * ComputationOpHandle::Run() in the future.
 */
class MemoryReusePass : public Pass {
 protected:
  void ApplyImpl(Graph *graph) const final;

  virtual void Run(Graph *graph) const = 0;

  virtual std::string ReuseType() const = 0;

  bool TryReuseVar(details::VarHandle *in_var,
                   details::VarHandle *out_var) const;

  bool IsInVarReusable(const details::VarHandle &in_var) const;

  bool IsOutVarReusable(const details::VarHandle &out_var) const;

  bool IsLastVersionVar(const details::VarHandle &var) const;

  static std::unordered_set<Node *> FindNodesByName(
      const std::string &name, const std::vector<Node *> &nodes);

  size_t ScopeNum() const { return all_vars_->size(); }

  int64_t GetMemorySize(const details::VarHandle &var) const;

  void AddReuseVar(details::ComputationOpHandle *op, details::VarHandle *in_var,
                   details::VarHandle *out_var) const;

  void MarkAsNotReusableInVar(const details::VarHandle &var) const;

 private:
  VarDesc *GetVarDesc(const details::VarHandle &var) const;

  bool IsVarPairReusable(const details::VarHandle &in_var,
                         const details::VarHandle &out_var) const;

  bool IsInVarAlreadyReused(const details::VarHandle &in_var) const;

  bool IsOutVarAlreadyReused(const details::VarHandle &out_var) const;

  details::ShareTensorBufferOpHandle *InsertShareTensorBufferOpHandleToGraph(
      details::ComputationOpHandle *op) const;

  void CollectShareTensorBufferOpHandles() const;

  void CollectReusedVars() const;

  void UpdateLastLiveOpOfVar(details::ComputationOpHandle *op,
                             details::VarHandle *in_var,
                             details::VarHandle *out_var) const;

 private:
  bool IsPinnedVar(const VarDesc &out_var_desc) const;

 private:
  mutable Graph *graph_;
  mutable bool use_cuda_;

  mutable details::GraphVars *all_vars_;
  mutable MemOptVarInfoMapList *var_infos_;

  mutable std::vector<LastLiveOpsOfVars> *last_live_ops_of_vars_;

  mutable std::unordered_map<details::ComputationOpHandle *,
                             details::ShareTensorBufferOpHandle *>
      ops_;

  mutable std::vector<std::unordered_set<std::string>> reused_in_var_names_;
  mutable std::vector<std::unordered_set<std::string>> reused_out_var_names_;

  mutable std::vector<std::unordered_map<std::string, VarDesc *>> var_descs_;
  mutable details::PinnedVars *pinned_var_set_;

  mutable std::vector<SkipReuseVars> *skip_reuse_vars_;
};

using InplaceInToOutMap = std::unordered_map<std::string, std::string>;

bool IsIdentityOp(const std::string &type);

const InplaceInToOutMap &InOutPairOfIdentityOp(const std::string &type);

}  // namespace ir
}  // namespace framework
}  // namespace paddle
