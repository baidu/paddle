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

#include <queue>

#include "paddle/fluid/framework/ir/memory_optimize_pass/memory_reuse_pass.h"

namespace paddle {
namespace framework {
namespace ir {

using OpHandleBase = details::OpHandleBase;
using ComputationOpHandle = details::ComputationOpHandle;
using VarHandle = details::VarHandle;
using VarHandleBase = details::VarHandleBase;

/**
 * The vars in `IdentityInplaceVarInfo` build a `FOREST`.
 *
 * This means that:
 *  1. there is no circle. Otherwise, it must be a bug of Paddle framework.
 *  2. input edge number of any var must be 0 (for root) or 1.
 *  3. output edge number of any var must be 0 (for leaf) or larger than 0.
 *
 * We would use the characteristics of `FOREST` in the following codes,
 * so we prove these 3 conditions here.
 *
 * Condition 1 and 3 is obvious. Let us prove condition 2:
 *
 *  In order to perform in-place reuse, `output_var` must be the 1st version
 *  in SSA graph (it is guaranteed in `MemoryReusePass::IsOutVarReusable`
 *  method). Therefore, if the input edge number of `output_var` is more than 1,
 *  it means `output_var` must be outputs of at least 2 different ops (one op
 *  with the same output is not allowed in Paddle framework). However, if
 *  `output_var` is the output of 2 different ops, `output_var`s in these 2 ops
 *  must have different version, which is guaranteed by SSA graph. Therefore,
 *  `output_var` must be the unique output of unique op. Condition 2 is proved.
 */
using IdentityInplaceVarInfo = std::unordered_map<
    VarHandle * /*input_var*/,
    std::vector<std::pair<VarHandle * /*output_var*/,
                          ComputationOpHandle * /*identity op*/>>>;

class VarHandleForest {
 public:
  IdentityInplaceVarInfo ConvertToIdentityInplaceVarInfo() const {
    IdentityInplaceVarInfo result;
    for (auto &pair : ops_) {
      auto *in_var = pair.first.first;
      auto *out_var = pair.first.second;
      auto *op = pair.second;
      result[in_var].emplace_back(out_var, op);
    }
    return result;
  }

  void Insert(VarHandle *in_var, VarHandle *out_var, ComputationOpHandle *op) {
    PADDLE_ENFORCE_NOT_NULL(in_var,
                            platform::errors::InvalidArgument(
                                "in_var cannot be nullptr, this may be a bug"));

    PADDLE_ENFORCE_NOT_NULL(
        out_var, platform::errors::InvalidArgument(
                     "out_var cannot be nullptr, this may be a bug"));

    PADDLE_ENFORCE_NOT_NULL(op, platform::errors::InvalidArgument(
                                    "op cannot be nullptr, this may be a bug"));

    PADDLE_ENFORCE_EQ(
        ops_.count(std::make_pair(in_var, out_var)), 0,
        platform::errors::AlreadyExists(
            "Insert duplicate nodes in forest, this may be a bug"));

    auto &parent_of_out_var = parent_[out_var];
    PADDLE_ENFORCE_EQ(
        parent_of_out_var, nullptr,
        platform::errors::AlreadyExists(
            "Insert duplicate nodes in forest, this may be a bug"));

    parent_of_out_var = in_var;
    parent_[in_var];  // insert nullptr to parent_ if in_var not exist

    auto &in_var_children = children_[in_var];
    PADDLE_ENFORCE_EQ(
        in_var_children.count(out_var), 0,
        platform::errors::AlreadyExists(
            "Insert duplicate nodes in forest, this may be a bug"));
    in_var_children.insert(out_var);

    children_[out_var];  // insert empty set to children_ if out_var not exist

    ops_[std::make_pair(in_var, out_var)] = op;
  }

  VarHandle *Parent(VarHandle *var) const {
    auto iter = parent_.find(var);
    PADDLE_ENFORCE_EQ(
        iter != parent_.end(), true,
        platform::errors::NotFound("Variable not found, this may be a bug"));
    return iter->second;
  }

  VarHandle *Root(VarHandle *var) const {
    while (auto *parent = Parent(var)) {
      var = parent;
    }
    return var;
  }

  const std::unordered_set<VarHandle *> &Children(VarHandle *var) const {
    auto iter = children_.find(var);
    PADDLE_ENFORCE_EQ(
        iter != children_.end(), true,
        platform::errors::NotFound("Variable not found, this may be a bug"));
    return iter->second;
  }

  bool IsRoot(VarHandle *var) const { return Parent(var) == nullptr; }

  bool IsLeaf(VarHandle *var) const { return Children(var).empty(); }

  void PruneLeaf(VarHandle *var) {
    PADDLE_ENFORCE_EQ(IsLeaf(var), true, platform::errors::InvalidArgument(
                                             "Only support prune leaf vars"));
    auto *parent = Parent(var);
    parent_.erase(var);
    children_.erase(var);
    ops_.erase(std::make_pair(parent, var));
    if (parent) {
      MutableChildren(parent)->erase(var);
      // Make sure that there is no tree with only one node
      if (IsRoot(parent) && IsLeaf(parent)) {
        PruneLeaf(parent);
      }
    }
  }

  std::unordered_set<VarHandle *> Leaves() const {
    std::unordered_set<VarHandle *> result;
    for (auto &pair : children_) {
      if (pair.second.empty()) {
        result.insert(pair.first);
      }
    }
    return result;
  }

  bool HasVar(VarHandle *var) const { return parent_.count(var) > 0; }

  template <typename Callback>
  bool BreadthFirstVisit(VarHandle *var, Callback &&callback) const {
    auto *root = Root(var);
    PADDLE_ENFORCE_NOT_NULL(root,
                            platform::errors::InvalidArgument(
                                "Root cannot be nullptr, this may be a bug"));
    std::queue<VarHandle *> q;
    q.push(root);
    while (!q.empty()) {
      var = q.front();
      q.pop();

      if (!callback(var)) {
        return false;
      }

      for (auto *out_var : Children(var)) {
        q.push(out_var);
      }
    }
    return true;
  }

 private:
  std::unordered_set<VarHandle *> *MutableChildren(VarHandle *var) {
    auto iter = children_.find(var);
    PADDLE_ENFORCE_EQ(
        iter != children_.end(), true,
        platform::errors::NotFound("Variable not found, this may be a bug"));
    return &(iter->second);
  }

 private:
  std::unordered_map<VarHandle * /*out_var*/, VarHandle * /*in_var*/> parent_;
  std::unordered_map<VarHandle * /*in_var*/,
                     std::unordered_set<VarHandle * /*out_var*/>>
      children_;
  std::map<std::pair<VarHandle * /*in_var*/, VarHandle * /*out_var*/>,
           ComputationOpHandle *>
      ops_;
};

static std::unordered_set<VarHandle *> PruneAndMarkNotReusableLeaves(
    VarHandleForest *forest,
    const std::unordered_set<VarHandle *> &is_read_by_other_ops,
    const std::unordered_set<VarHandle *> &non_last_version_vars,
    const std::unordered_set<VarHandle *> &reused_out_vars) {
  auto is_linear_reuse = [forest](VarHandle *var) {
    return forest->BreadthFirstVisit(var, [forest](VarHandle *v) {
      return forest->Children(v).size() <= 1;
    });
  };

  auto has_non_identity_last_lived_ops = [&](VarHandle *var) {
    return !forest->BreadthFirstVisit(var, [&](VarHandle *v) {
      if (!forest->IsLeaf(v) && is_read_by_other_ops.count(v) > 0) {
        return false;
      } else {
        return true;
      }
    });
  };

  bool has_prune_leaf = true;

  // Prune until the forest is stable
  while (has_prune_leaf) {
    has_prune_leaf = false;
    for (auto *leaf : forest->Leaves()) {
      if (!is_linear_reuse(leaf) && reused_out_vars.count(leaf) > 0) {
        forest->PruneLeaf(leaf);
        has_prune_leaf = true;
      }
    }

    for (auto *leaf : forest->Leaves()) {
      if (is_linear_reuse(leaf) && has_non_identity_last_lived_ops(leaf) &&
          non_last_version_vars.count(leaf) > 0) {
        forest->PruneLeaf(leaf);
        has_prune_leaf = true;
      }
    }
  }

  std::unordered_set<VarHandle *> non_reusable_vars;
  for (auto *leaf : forest->Leaves()) {
    if (is_linear_reuse(leaf)) {
      if (has_non_identity_last_lived_ops(leaf)) {
        non_reusable_vars.insert(leaf);
      }
    } else {
      non_reusable_vars.insert(leaf);
    }
  }
  return non_reusable_vars;
}

class BufferSharedIdentityInplaceOpPass : public MemoryReusePass {
 protected:
  std::string ReuseType() const override { return "identity_inplace"; }

  void Run(Graph *graph) const override;

 private:
  std::pair<IdentityInplaceVarInfo, std::unordered_set<VarHandle *>>
  RunOnEachScope(const std::unordered_map<std::string, LastLiveOpOfVarInfo>
                     &var_infos) const;

  static VarHandle *TryFindMatchedIdentityInplacedOutput(
      const VarHandle &in_var, const ComputationOpHandle &op);
};

std::pair<IdentityInplaceVarInfo, std::unordered_set<VarHandle *>>
BufferSharedIdentityInplaceOpPass::RunOnEachScope(
    const std::unordered_map<std::string, LastLiveOpOfVarInfo> &var_infos)
    const {
  VarHandleForest var_forest;
  std::unordered_set<VarHandle *> is_read_by_other_ops;
  std::unordered_set<VarHandle *> non_last_version_vars;

  /**
   * The memories of some out vars may be reused by non-identity ops,
   * this vars must be forced to be leaf when identity inplaced is performed.
   * If branched identity inplace is performed (many vars may reuse the memory
   * of the same var), this kind of leaf nodes must be pruned (otherwise,
   * caculation result may be wrong).
   *
   * If non-branched identity inplace is performed, this kind of leaf nodes
   * should not be pruned.
   */
  std::unordered_set<VarHandle *> reused_out_vars;
  for (auto &pair : var_infos) {
    auto *in_var = pair.second.var();
    if (!IsInVarReusable(*in_var)) continue;

    // force leaf vars
    if (reused_out_vars.count(in_var) > 0) continue;

    auto in_arg = in_var->Name();

    auto last_live_ops = pair.second.ops();
    /**
     * If all `ops` only read `in_var` and do not write `in_var`,
     * we should scan all output nodes of `in_var`. If any output
     * node of `in_var` is an identity op node, in-place can be
     * performed.
     */
    bool failure = false;
    for (auto *op : last_live_ops) {
      failure = FindNodesByName(in_arg, op->Node()->inputs).empty() ||
                !FindNodesByName(in_arg, op->Node()->outputs).empty();
      if (failure) break;
    }

    if (failure) continue;

    std::unordered_set<ComputationOpHandle *> candidate_ops;
    for (auto *op_node : in_var->Node()->outputs) {
      auto *op = dynamic_cast<ComputationOpHandle *>(
          &(op_node->Wrapper<OpHandleBase>()));
      if (op == nullptr) {
        candidate_ops.clear();
        break;
      }
      if (IsIdentityOp(op->GetOp()->Type())) {
        VLOG(5) << "Found identity op " << op->GetOp()->Type();
        candidate_ops.insert(op);
      }
    }

    if (candidate_ops.empty()) continue;

    for (auto *op : candidate_ops) {
      auto *out_var = TryFindMatchedIdentityInplacedOutput(*in_var, *op);
      if (out_var == nullptr) continue;
      if (!IsOutVarReusable(*out_var)) {
        VLOG(5) << "Cannot inplace because " << out_var->Name()
                << " is not reusable";
        continue;
      }

      if (!IsInVarReusable(*out_var)) {
        reused_out_vars.insert(out_var);
      }

      if (!IsLastVersionVar(*out_var)) {
        non_last_version_vars.insert(out_var);
      }

      var_forest.Insert(in_var, out_var, op);
      last_live_ops.erase(op);
    }

    // If last live ops of `in_var` is not a subset of identity in-placed
    // ops, the leaf vars of such ops must be non-reusable. Otherwise,
    // the calculation results may be wrong after CrossOpMemoryReusePass.
    // Those vars must be marked as "non-reusable"!
    if (!last_live_ops.empty()) {
      is_read_by_other_ops.insert(in_var);
    }
  }

  auto non_reusable_vars =
      PruneAndMarkNotReusableLeaves(&var_forest, is_read_by_other_ops,
                                    non_last_version_vars, reused_out_vars);

  return std::make_pair(var_forest.ConvertToIdentityInplaceVarInfo(),
                        std::move(non_reusable_vars));
}

VarHandle *
BufferSharedIdentityInplaceOpPass::TryFindMatchedIdentityInplacedOutput(
    const VarHandle &in_var, const ComputationOpHandle &op) {
  auto &in_out_pairs = InOutPairOfIdentityOp(op.GetOp()->Type());
  std::unique_ptr<std::string> out_slot;
  for (auto &in_to_out : in_out_pairs) {
    auto &in_slot = in_to_out.first;
    auto &inputs = op.GetOp()->Inputs();
    auto in_iter = inputs.find(in_slot);
    if (in_iter == inputs.end()) continue;
    if (in_iter->second.size() != 1) continue;
    if (in_iter->second[0] == in_var.Name()) {
      out_slot.reset(new std::string(in_to_out.second));
      break;
    }
  }

  if (!out_slot) return nullptr;
  auto &outputs = op.GetOp()->Outputs();
  auto out_iter = outputs.find(*out_slot);
  if (out_iter == outputs.end()) return nullptr;
  if (out_iter->second.size() != 1) return nullptr;

  const auto &out_arg = out_iter->second[0];
  auto out_nodes = FindNodesByName(out_arg, op.Node()->outputs);
  if (out_nodes.size() != 1) return nullptr;
  auto *out_node = *(out_nodes.begin());

  PADDLE_ENFORCE_EQ(out_node->IsWrappedBy<VarHandleBase>(), true,
                    platform::errors::InvalidArgument(
                        "out node must be wrapped by VarHandleBase"));
  return dynamic_cast<VarHandle *>(&(out_node->Wrapper<VarHandleBase>()));
}

void BufferSharedIdentityInplaceOpPass::Run(Graph *graph) const {
  const auto &last_live_ops =
      Get<std::vector<LastLiveOpsOfVars>>(kLastLiveOpsOfVars);

  std::vector<IdentityInplaceVarInfo> inplace_infos;
  inplace_infos.reserve(last_live_ops.size());

  std::vector<std::unordered_set<VarHandle *>> non_reusable_leaf_vars;
  non_reusable_leaf_vars.reserve(last_live_ops.size());

  for (auto &each_scope_var_infos : last_live_ops) {
    auto each_scope_ret = RunOnEachScope(each_scope_var_infos);
    inplace_infos.emplace_back(std::move(each_scope_ret.first));
    non_reusable_leaf_vars.emplace_back(std::move(each_scope_ret.second));
  }

  for (auto &inplace_info : inplace_infos) {
    for (auto &in_to_out : inplace_info) {
      auto *in_var = in_to_out.first;
      for (auto &op_out_var_pair : in_to_out.second) {
        auto *out_var = op_out_var_pair.first;
        auto *op = op_out_var_pair.second;
        VLOG(5) << "Identity inplace in " << op->GetOp()->Type() << ": "
                << in_var->Name() << " -> " << out_var->Name() << " in scope "
                << in_var->scope_idx();
        AddReuseVar(op, in_var, out_var);
      }
    }
  }

  for (auto &vars_each_scope : non_reusable_leaf_vars) {
    for (auto *each_var : vars_each_scope) {
      VLOG(1) << "Mark " << each_var->Name() << " on scope "
              << each_var->scope_idx() << " as not reusable";
      MarkAsNotReusableInVar(*each_var);
    }
  }
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(buffer_shared_identity_inplace_pass,
              paddle::framework::ir::BufferSharedIdentityInplaceOpPass)
    .RequirePassAttr(paddle::framework::ir::kMemOptVarInfoMapList)
    .RequirePassAttr(paddle::framework::ir::kLastLiveOpsOfVars)
    .RequirePassAttr(paddle::framework::ir::kUseCuda)
    .RequirePassAttr(paddle::framework::ir::kSkipReuseVars);
