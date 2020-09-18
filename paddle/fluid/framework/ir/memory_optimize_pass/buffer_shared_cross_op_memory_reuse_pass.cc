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

#include <algorithm>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include "paddle/fluid/framework/details/computation_op_handle.h"
#include "paddle/fluid/framework/details/multi_devices_helper.h"
#include "paddle/fluid/framework/details/share_tensor_buffer_op_handle.h"
#include "paddle/fluid/framework/ir/memory_optimize_pass/memory_optimization_var_info.h"
#include "paddle/fluid/framework/ir/memory_optimize_pass/memory_reuse_pass.h"
#include "paddle/fluid/framework/ir/memory_optimize_pass/op_graph_view.h"
#include "paddle/fluid/framework/ir/memory_optimize_pass/reference_count_pass_helper.h"
#include "paddle/fluid/framework/ir/pass.h"

namespace paddle {
namespace framework {
namespace ir {

using OpHandleBase = details::OpHandleBase;
using ComputationOpHandle = details::ComputationOpHandle;
using VarHandle = details::VarHandle;
using VarHandleBase = details::VarHandleBase;
using DummyVarHandle = details::DummyVarHandle;

enum NodeDependency { kSame = 0, kNoDep = 1, kBefore = 2, kAfter = 3 };

static NodeDependency ReverseNodeDependency(NodeDependency dep) {
  return dep == NodeDependency::kBefore
             ? NodeDependency::kAfter
             : (dep == NodeDependency::kAfter ? NodeDependency::kBefore : dep);
}

class BufferSharedCrossOpMemoryReusePass : public MemoryReusePass {
 protected:
  std::string ReuseType() const override { return "cross_op_memory_reuse"; }

  void Run(Graph *graph) const override;

 private:
  void RunOnScopeIdx(size_t idx) const;

  // Toposort ops. Different strategies can be used in the future.
  std::vector<OpHandleBase *> SortOp(const OpGraphView &graph_view) const;

  // Build the initial dependency matrix, and initializing all fields,
  // including `ops_`, `op_to_idx_`, `deps_`
  void BuildOpDependencyMap() const;

  // Get op index inside `ops_`, used to find dependency inside `deps_`
  size_t OpIndex(const ComputationOpHandle *op) const;

  size_t ResolveDependencyBetween(
      ComputationOpHandle *op,
      const std::unordered_set<ComputationOpHandle *> &prev_ops) const;

  // Get dependency relationship between op1 and op2
  // Notice: GetOpDep(op1, op2) == ReverseNodeDependency(GetOpDep(op2, op1))
  NodeDependency GetOpDep(const ComputationOpHandle *op1,
                          const ComputationOpHandle *op2) const;

  void SetOpDep(const ComputationOpHandle *op1, const ComputationOpHandle *op2,
                NodeDependency dep) const;

 private:
  mutable Graph *graph_;

  // All ops in the graph, grouped by scope index
  mutable std::vector<std::vector<ComputationOpHandle *>> ops_;

  // Index of each op in `ops_`, grouped by scope index.
  // Index of each op is the index inside `deps_`.
  mutable std::vector<std::unordered_map<const ComputationOpHandle *, size_t>>
      op_to_idx_;

  // Dependency matrix of between any 2 ops
  // If deps_[scope_idx][i][j] is equal to:
  //  1. kSame, Op(i) and Op(j) are the same ops, only when i == j.
  //  2. kNoDep, Op(i) and Op(j) have no dependency between each other.
  //  3. kBefore, Op(i) is the preceding op of Op(j).
  //  4. kAfter, Op(i) is the pending op of Op(j).
  mutable std::vector<std::vector<std::vector<NodeDependency>>> deps_;
};

void BufferSharedCrossOpMemoryReusePass::Run(Graph *graph) const {
  graph_ = graph;
  BuildOpDependencyMap();
  for (size_t i = 0; i < ScopeNum(); ++i) {
    RunOnScopeIdx(i);
  }
}

// Note(zjl): The reason why I separate SortOp from BuildOpDependencyMap()
// is that we can use different sorting strategies in the future to
// evaluate the effects of different sorting strategies.
// Currently, I use BFS, but we can use other kinds of sorting strategy
// in the future, as long as the new strategy reaches higher memory reuse
// ratio.
std::vector<OpHandleBase *> BufferSharedCrossOpMemoryReusePass::SortOp(
    const OpGraphView &graph_view) const {
  std::vector<OpHandleBase *> sorted_ops;
  sorted_ops.reserve(graph_view.OpNumber());
  graph_view.BreadthFirstVisit(
      [&](OpHandleBase *cur_op) { sorted_ops.emplace_back(cur_op); });
  PADDLE_ENFORCE_EQ(sorted_ops.size(), graph_view.OpNumber(),
                    platform::errors::InvalidArgument(
                        "Sorted ops size(%d) not equal to graph op size(%d). "
                        "There are unvisited ops.",
                        sorted_ops.size(), graph_view.OpNumber()));
  return sorted_ops;
}

/**
 * Try to reuse unlived vars.
 *
 * What we do is: transverse all outputs of each op, and find a suitable
 * unused var, and then reuse its memory as output.
 *
 * How to determine unused vars?
 *
 * Case 1: unlived vars after all preceding ops run. In this case, no extra
 *   edge would be added to the graph.
 *
 * Case 2: unlived vars after all preceding ops and all no-dep ops run. In
 *   this case, the reused var is from no-dep ops, so that we have to add
 *   extra edge to resolve data hazard.
 *
 *
 * If Case 2 occurs, what we should do to resolve data hazard?
 *
 *  - Step 1: add a dep var between reused_op and share_tensor_buffer_op,
 *            that is: reused_op -> dep_var -> share_tensor_buffer_op.
 *
 *  - Step 2: Update deps_, all preceding ops of reused_op should be
 *            preceding ops of op.
 */
void BufferSharedCrossOpMemoryReusePass::RunOnScopeIdx(size_t idx) const {
  auto &ops = ops_[idx];

  auto &last_live_ops_of_vars =
      Get<std::vector<LastLiveOpsOfVars>>(kLastLiveOpsOfVars)[idx];

  // Build a reverse map of `last_live_ops_of_vars`,
  // i.e., VarHandle -> last lived ops of VarHandle
  std::unordered_map<VarHandle *, std::unordered_set<ComputationOpHandle *>>
      var_to_ops;
  for (auto &pair : last_live_ops_of_vars) {
    for (auto *op : pair.second.ops()) {
      var_to_ops[pair.second.var()].insert(op);
    }
  }

  // Deep copy of `var_to_ops`, used to get last lived ops of each unlived var
  auto original_var_to_ops = var_to_ops;

  // Memory size of VarHandle -> list<VarHandle>
  std::map<int64_t, std::list<VarHandle *>> unlived_var_pool;
  size_t reuse_num = 0;

  for (auto *op : ops) {
    // Transverse all output args of op, find whether there is unlived var
    // can be reused.
    auto out_args = op->Node()->Op()->OutputArgumentNames();
    for (auto &out_arg : out_args) {
      auto out_nodes = this->FindNodesByName(out_arg, op->Node()->outputs);
      // If out_arg is kEmptyVarName, it may not be found in output nodes.
      if (out_nodes.size() != 1) {
        continue;
      }

      auto *out_node = *(out_nodes.begin());
      auto *out_var =
          dynamic_cast<VarHandle *>(&(out_node->Wrapper<VarHandleBase>()));
      PADDLE_ENFORCE_NOT_NULL(
          out_var, platform::errors::NotFound(
                       "Can not find a valid Var Node for Var %s.", out_arg));

      // If out_arg is not reusable, skip it
      if (!IsOutVarReusable(*out_var)) {
        continue;
      }

      auto mem_size = GetMemorySize(*out_var);
      // Special case: if memory size of out_var is 0, skip it
      if (mem_size == 0) {
        continue;
      }

      // Find a suitable unlived var from `unlived_var_pool`
      // Here, we use `find`, but we can perform `lower_bound` if
      // it is better in the future.
      auto iter = unlived_var_pool.find(std::abs(mem_size));
      if (iter == unlived_var_pool.end()) {
        continue;
      }

      // Obtain candidate_vars that can be reused.
      auto &candidate_vars = iter->second;
      for (auto var_iter = candidate_vars.begin();
           var_iter != candidate_vars.end(); ++var_iter) {
        bool success = this->TryReuseVar(*var_iter, out_var);
        if (!success) continue;

        // If memory reuse is successful, we should do some post-processing.
        ++reuse_num;
        auto &prev_ops = original_var_to_ops.at(*var_iter);

        // Add extra dependencies between `op` and last lived ops of reused var
        // (i.e. prev_ops) if needed.
        // All `prev_ops` must be preceding ops of op to avoid data hazard.
        size_t new_added_dep_num = ResolveDependencyBetween(op, prev_ops);
        VLOG(3) << "Variable can be reused between: " << (*var_iter)->Name()
                << " -> " << out_var->Name() << " when running op "
                << op->Name() << ", add extra dependency " << new_added_dep_num
                << "/" << prev_ops.size();

        // erase reused var from ``original_var_to_ops`
        original_var_to_ops.erase(*var_iter);

        // erase reused var from `candidate_vars`
        candidate_vars.erase(var_iter);
        if (candidate_vars.empty()) {
          // erase reused var from `unlived_var_pool` if there is no other vars
          // which has same size with reused var.
          unlived_var_pool.erase(iter);
        }
        break;
      }
    }

    // After all output args have been transversed, we should check whether
    // there is new unlived var after `op` runs.
    for (auto op_iter = var_to_ops.begin(); op_iter != var_to_ops.end();) {
      // erase op from `var_to_ops` first
      op_iter->second.erase(op);
      if (op_iter->second.empty()) {
        // there is a unlived var, since all lived ops have run
        VarHandle *unlived_var = op_iter->first;
        var_to_ops.erase(op_iter++);
        if (IsInVarReusable(*unlived_var)) {
          auto mem_size = GetMemorySize(*unlived_var);
          if (mem_size != 0) {
            unlived_var_pool[std::abs(mem_size)].push_front(unlived_var);
          }
        }
      } else {
        ++op_iter;
      }
    }
  }
  VLOG(4) << "Reuse " << reuse_num << " variable(s) in Scope " << idx;
}

size_t BufferSharedCrossOpMemoryReusePass::ResolveDependencyBetween(
    ComputationOpHandle *op,
    const std::unordered_set<ComputationOpHandle *> &prev_ops) const {
  size_t new_added_dep_num = 0;
  size_t op_idx = OpIndex(op);
  auto &deps = deps_[op->GetScopeIdx()];
  for (auto *prev_op : prev_ops) {
    auto op_dep = GetOpDep(prev_op, op);
    if (op_dep == NodeDependency::kBefore) continue;
    PADDLE_ENFORCE_EQ(op_dep, NodeDependency::kNoDep,
                      platform::errors::InvalidArgument(
                          "The graph has circle, this may be a bug."));

    auto iter =
        std::find_if(prev_op->Outputs().begin(), prev_op->Outputs().end(),
                     [](VarHandleBase *var) {
                       return dynamic_cast<DummyVarHandle *>(var) != nullptr;
                     });

    if (iter != prev_op->Outputs().end()) {
      op->AddInput(*iter);
    } else {
      auto *dep_var = new DummyVarHandle(graph_->CreateControlDepVar());
      graph_->Get<details::GraphDepVars>(details::kGraphDepVars)
          .emplace(dep_var);
      prev_op->AddOutput(dep_var);
      op->AddInput(dep_var);
    }

    // All preceding ops of `prev_op` should be preceding ops of `op`
    size_t prev_op_idx = OpIndex(prev_op);
    for (size_t i = 0; i < deps[prev_op_idx].size(); ++i) {
      if (deps[prev_op_idx][i] != NodeDependency::kAfter) {
        continue;
      }

      deps[i][op_idx] = NodeDependency::kBefore;
      deps[op_idx][i] = NodeDependency::kAfter;
    }

    // All pending ops of `op` should be pending ops of `prev_op`.
    for (size_t i = 0; i < deps[op_idx].size(); ++i) {
      if (deps[op_idx][i] != NodeDependency::kBefore) {
        continue;
      }

      deps[i][prev_op_idx] = NodeDependency::kAfter;
      deps[prev_op_idx][i] = NodeDependency::kBefore;
    }

    // `prev_op` is one of preceding op of `op`
    SetOpDep(prev_op, op, NodeDependency::kBefore);
    ++new_added_dep_num;
  }
  return new_added_dep_num;
}

void BufferSharedCrossOpMemoryReusePass::BuildOpDependencyMap() const {
  PADDLE_ENFORCE_EQ(ops_.empty(), true, platform::errors::InvalidArgument(
                                            "Ops must be initialized here."));
  PADDLE_ENFORCE_EQ(
      op_to_idx_.empty(), true,
      platform::errors::InvalidArgument("Op to idx must be initialized here."));
  PADDLE_ENFORCE_EQ(deps_.empty(), true, platform::errors::InvalidArgument(
                                             "Deps must be initialized here."));

  // Toposort ops
  OpGraphView graph_view(ir::FilterByNodeWrapper<OpHandleBase>(*graph_));
  auto ops = SortOp(graph_view);

  size_t scope_num = this->ScopeNum();
  size_t op_num = ops.size();

  // A map to record all preceding ops of each op
  std::unordered_map<OpHandleBase *, std::unordered_set<OpHandleBase *>>
      preceding_ops;

  // BFS to fill `preceding_ops`
  graph_view.BreadthFirstVisit([&](OpHandleBase *cur_op) {
    // All preceding ops of cur_op should be:
    //  - preceding ops of cur_op, that is connected to cur_op directely
    //  - all preceding ops of `direct preceding ops of cur_op`
    auto &all_preceding_ops_of_cur_op = preceding_ops[cur_op];
    for (auto &preceding_op : graph_view.PrecedingOps(cur_op)) {
      all_preceding_ops_of_cur_op.insert(preceding_op);
      auto &prev_preceding_ops = preceding_ops[preceding_op];
      all_preceding_ops_of_cur_op.insert(prev_preceding_ops.begin(),
                                         prev_preceding_ops.end());
    }
  });
  PADDLE_ENFORCE_EQ(preceding_ops.size(), op_num,
                    platform::errors::InvalidArgument(
                        "Preceding ops size(%d) must equal to op num(%d).",
                        preceding_ops.size(), op_num));

  // Find out ComputationOpHandles only
  ops_.resize(scope_num);
  op_to_idx_.resize(scope_num);
  for (auto *op : ops) {
    auto *compute_op = dynamic_cast<ComputationOpHandle *>(op);
    if (compute_op == nullptr) continue;
    size_t scope_idx = compute_op->GetScopeIdx();
    ops_[scope_idx].emplace_back(compute_op);
    op_to_idx_[scope_idx].emplace(compute_op, op_to_idx_[scope_idx].size());
  }

  // Fill deps_ according to `preceding_ops`
  deps_.resize(scope_num);
  for (size_t i = 0; i < deps_.size(); ++i) {
    deps_[i].resize(ops_[i].size());
    for (auto &item : deps_[i]) {
      item.assign(ops_[i].size(), NodeDependency::kNoDep);
    }
  }

  for (auto &ops_on_each_device : ops_) {
    for (auto *op : ops_on_each_device) {
      SetOpDep(op, op, NodeDependency::kSame);
      for (auto *preceding_op : preceding_ops[op]) {
        auto *compute_preceding_op =
            dynamic_cast<ComputationOpHandle *>(preceding_op);
        if (compute_preceding_op != nullptr &&
            compute_preceding_op->GetScopeIdx() == op->GetScopeIdx()) {
          SetOpDep(compute_preceding_op, op, NodeDependency::kBefore);
        }
      }
    }
  }
}

size_t BufferSharedCrossOpMemoryReusePass::OpIndex(
    const ComputationOpHandle *op) const {
  auto iter = op_to_idx_[op->GetScopeIdx()].find(op);
  PADDLE_ENFORCE_NE(iter, op_to_idx_[op->GetScopeIdx()].end(),
                    platform::errors::NotFound(
                        "Can not find op(%s) in op_to_idx_.", op->Name()));
  return iter->second;
}

NodeDependency BufferSharedCrossOpMemoryReusePass::GetOpDep(
    const ComputationOpHandle *op1, const ComputationOpHandle *op2) const {
  PADDLE_ENFORCE_EQ(op1->GetScopeIdx(), op2->GetScopeIdx(),
                    platform::errors::InvalidArgument(
                        "Op(%s) and op(%s) must in the same scope.",
                        op1->Name(), op2->Name()));
  return deps_[op1->GetScopeIdx()][OpIndex(op1)][OpIndex(op2)];
}

void BufferSharedCrossOpMemoryReusePass::SetOpDep(
    const ComputationOpHandle *op1, const ComputationOpHandle *op2,
    NodeDependency dep) const {
  PADDLE_ENFORCE_EQ(op1->GetScopeIdx(), op2->GetScopeIdx(),
                    platform::errors::InvalidArgument(
                        "Op(%s) and op(%s) must in the same scope.",
                        op1->Name(), op2->Name()));
  if (op1 == op2) {
    PADDLE_ENFORCE_EQ(
        dep, NodeDependency::kSame,
        platform::errors::InvalidArgument(
            "Set Same Op(%s) Dep, dep must be kSame type.", op1->Name()));
    auto idx = OpIndex(op1);
    deps_[op1->GetScopeIdx()][idx][idx] = NodeDependency::kSame;
  } else {
    auto idx1 = OpIndex(op1);
    auto idx2 = OpIndex(op2);
    PADDLE_ENFORCE_EQ((dep != NodeDependency::kSame && idx1 != idx2), true,
                      platform::errors::InvalidArgument(
                          "Op(%s) and Op(%s) should not have same "
                          "index(%d), and dep should not kSame type.",
                          op1->Name(), op2->Name(), idx1));
    deps_[op1->GetScopeIdx()][idx1][idx2] = dep;
    deps_[op1->GetScopeIdx()][idx2][idx1] = ReverseNodeDependency(dep);
  }
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(buffer_shared_cross_op_memory_reuse_pass,
              paddle::framework::ir::BufferSharedCrossOpMemoryReusePass)
    .RequirePassAttr(paddle::framework::ir::kMemOptVarInfoMapList)
    .RequirePassAttr(paddle::framework::ir::kLastLiveOpsOfVars)
    .RequirePassAttr(paddle::framework::ir::kUseCuda)
    .RequirePassAttr(paddle::framework::ir::kSkipReuseVars);
