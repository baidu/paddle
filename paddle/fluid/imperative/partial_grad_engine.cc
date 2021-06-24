// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/imperative/partial_grad_engine.h"

#include <algorithm>
#include <map>
#include <memory>
#include <queue>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>
#include "paddle/fluid/imperative/gradient_accumulator.h"
#include "paddle/fluid/imperative/layer.h"
#include "paddle/fluid/imperative/op_base.h"
#include "paddle/fluid/imperative/tracer.h"
#include "paddle/fluid/operators/math/math_function.h"
#include "paddle/fluid/platform/device_context.h"
#include "paddle/fluid/platform/profiler.h"
#include "paddle/fluid/string/string_helper.h"

DECLARE_bool(sort_sum_gradient);

namespace paddle {
namespace imperative {

struct HashPair {
  template <class T1, class T2>
  size_t operator()(const std::pair<T1, T2> &p) const noexcept {
    auto hash1 = std::hash<T1>{}(p.first);
    auto hash2 = std::hash<T2>{}(p.second);
    return hash1 ^ hash2;
  }
};

/**
 * This function prunes the graph to get the ops between `output_targets`
 * and `input_target_grads`.
 *
 *
 * The inputs are:
 *
 *  - input_target_grads: the input target grads. It may be changed.
 *  - output_targets: the output target vars. It may be changed.
 *
 *
 * The outputs are:
 *
 *  - startup_op_ptr: startup ops of the pruned graph.
 *  - pending_ops_ptr: contains all the pending ops of each op in the graph.
 *  - op_deps_ptr: the preceding op number of each op in the graph.
 *  - related_grad_vars_ptr: all grad vars in the pruned graph.
 */
static void GetGraphInfoBetweenTargets(
    std::unordered_set<VariableWrapper *> *input_target_grads,
    std::unordered_set<VarBase *> *output_targets,
    std::unordered_set<OpBase *> *startup_ops_ptr,
    std::unordered_map<OpBase *, std::unordered_set<OpBase *>> *pending_ops_ptr,
    std::unordered_map<OpBase *, size_t> *op_deps_ptr,
    std::unordered_set<VariableWrapper *> *related_grad_vars_ptr,
    const std::unordered_set<VariableWrapper *> &no_grad_var_grad) {
  /**
   * Step 1. Find the candidate startup grad ops, prepared for following BFS.
   */
  std::queue<std::pair<OpBase *, GradOpNode *>> q;
  std::unordered_set<GradOpNode *> visited;
  for (auto iter = output_targets->begin(); iter != output_targets->end();) {
    auto *output_target = *iter;
    PADDLE_ENFORCE_NOT_NULL(
        output_target,
        platform::errors::NotFound("output_target must not be nullptr"));
    if (output_target->OverridedStopGradient() ||
        output_target->GradVarBase() == nullptr ||
        output_target->GradVarBase()->GradNode() == nullptr) {
      VLOG(10) << output_target->Name()
               << " is pruned because it stops gradient or has no grad var";
      iter = output_targets->erase(iter);
      continue;
    }

    auto &grad_node = output_target->GradVarBase()->GradNode();
    if (visited.count(grad_node.get()) == 0) {
      for (auto &op : *grad_node) {
        q.emplace(&op, grad_node.get());
      }
    }
    ++iter;
  }

  /**
   * Step 2. BFS the graph and find all grad ops which generate the
   * input_target_grads. Notice that not all candidate startup ops
   * would be connected with input_target_grads, that is to say,
   * not all input_target_grads would be found.
   */
  std::unordered_set<VariableWrapper *> found_input_target_grads;
  std::unordered_set<OpBase *> endpoint_ops;
  std::unordered_map<OpBase *, std::unordered_set<OpBase *>> preceding_ops;
  while (!q.empty()) {
    auto op_node_pair = q.front();
    q.pop();

    auto *op = op_node_pair.first;
    auto *node = op_node_pair.second;

    for (auto &output_pair : op->GetOutsMap()) {
      if (!output_pair.second.IsGrad()) {
        VLOG(10) << "WARNING: " << op->Type() << " outputs a forward var";
        continue;
      }

      for (auto &out_var : output_pair.second) {
        if (out_var && input_target_grads->count(out_var.get()) > 0) {
          VLOG(10) << "Found endpoint op " << op->Type() << " which generates "
                   << out_var->Name();
          found_input_target_grads.insert(out_var.get());
          endpoint_ops.emplace(op);
        }
      }
    }

    for (auto &pending_node : node->GradPendingNodes()) {
      if (visited.count(pending_node.get()) == 0) {
        for (auto &pending_op : *pending_node) {
          preceding_ops[&pending_op].insert(op);
          q.emplace(&pending_op, pending_node.get());
        }
      }
    }
  }

  /**
   * Step 3. Based on the found input_target_grads, BFS the graph in reverse
   * order. `target_vars` would record all grad vars in the graph, and
   * `startup_ops` would be the final startup ops of the graph.
   */
  *input_target_grads = found_input_target_grads;

  auto &pending_ops = *pending_ops_ptr;
  pending_ops.clear();

  auto &startup_ops = *startup_ops_ptr;
  startup_ops.clear();

  auto &op_deps = *op_deps_ptr;
  op_deps.clear();

  auto &target_vars = *related_grad_vars_ptr;
  target_vars = *input_target_grads;

  std::queue<std::pair<OpBase * /*op*/, OpBase * /*pending op*/>> op_queue;
  std::unordered_set<std::pair<OpBase *, OpBase *>, HashPair> op_base_visited;
  for (auto &endpoint_op : endpoint_ops) {
    op_queue.emplace(endpoint_op, nullptr);
    op_base_visited.emplace(endpoint_op, nullptr);
  }

  while (!op_queue.empty()) {
    auto op_pair = op_queue.front();
    auto *op = op_pair.first;
    auto *pending_op = op_pair.second;

    op_queue.pop();

    bool is_valid = false;
    for (auto &output_pair : op->GetOutsMap()) {
      if (!output_pair.second.IsGrad()) {
        continue;
      }

      for (auto &out_var : output_pair.second) {
        if (out_var && target_vars.count(out_var.get()) > 0) {
          is_valid = true;
          break;
        }
      }

      if (is_valid) {
        break;
      }
    }

    if (!is_valid) {
      continue;
    }

    is_valid = false;
    for (auto &input_pair : op->GetInsMap()) {
      if (!input_pair.second.IsGrad()) {
        continue;
      }

      for (auto &in_var : input_pair.second) {
        if (in_var && no_grad_var_grad.count(in_var.get()) == 0) {
          target_vars.insert(in_var.get());
          is_valid = true;
        }
      }
    }

    if (!is_valid) {
      continue;
    }

    op_deps[op];
    if (pending_op) {
      VLOG(10) << "Pending op of " << op->Type() << " is "
               << pending_op->Type();

      pending_ops[op].insert(pending_op);
      ++op_deps[pending_op];
    } else {
      pending_ops[op];
    }

    auto iter = preceding_ops.find(op);
    if (iter != preceding_ops.end()) {
      for (auto &preceding_op : iter->second) {
        if (op_base_visited.count(std::make_pair(preceding_op, op)) == 0) {
          op_queue.emplace(preceding_op, op);
          op_base_visited.emplace(preceding_op, op);
        }
      }
    }
  }

  for (auto &pair : op_deps) {
    if (pair.second == 0) {
      auto *op = pair.first;
      VLOG(10) << "Found startup op " << op->Type();
      startup_ops.insert(op);
    }
  }

  /**
   * Step 4. Prune output_targets which is not the input of startup_ops
   */
  for (auto iter = output_targets->begin(); iter != output_targets->end();) {
    auto &grad_node = (*iter)->GradVarBase()->GradNode();
    bool is_valid = std::find_if(grad_node->begin(), grad_node->end(),
                                 [&](OpBase &op) {  // NOLINT
                                   return startup_ops.count(&op) > 0;
                                 }) != grad_node->end();
    if (is_valid) {
      ++iter;
    } else {
      iter = output_targets->erase(iter);
    }
  }
}

// Get debug string of op types contained in `node`
static std::string GradOpTypes(const GradOpNode &node) {
  std::vector<std::string> node_types;
  for (auto &op : node) {
    node_types.emplace_back(op.Type());
  }
  return string::join_strings(node_types, ',');
}

// Get debug string of grad node of `var`'s gradient
static std::string GradOpTypes(const VarBase &var) {
  if (!var.GradVarBase() || !var.GradVarBase()->GradNode()) {
    return "";
  } else {
    return GradOpTypes(*(var.GradVarBase()->GradNode()));
  }
}

// Get pending op types of `node`
static std::string GradPendingOpTypes(const GradOpNode &node) {
  std::vector<std::string> node_types;
  for (auto &n : node.GradPendingNodes()) {
    node_types.emplace_back(GradOpTypes(*n));
  }
  return string::join_strings(node_types, ',');
}

static void FillConstantLike(const VariableWrapper &ref_var,
                             VariableWrapper *dst_var,
                             const platform::Place &place, float value) {
  auto &ref_tensor = ref_var.Var().Get<framework::LoDTensor>();
  auto *dst_tensor = dst_var->MutableVar()->GetMutable<framework::LoDTensor>();
  auto *dev_ctx = platform::DeviceContextPool::Instance().Get(place);
  dst_tensor->Resize(ref_tensor.dims());
  dst_tensor->mutable_data(place, ref_var.DataType());
  operators::math::set_constant(*dev_ctx, dst_tensor, value);
}

/**
 * A data structure for gradient accumulation
 */
class GradientAccumulationInfo {
 private:
  using PartialGradGradTraceIdPair =
      std::pair<std::weak_ptr<VariableWrapper> /*partial grad grad var*/,
                size_t /*trace_id*/>;

 public:
  explicit GradientAccumulationInfo(const std::shared_ptr<VariableWrapper> &var,
                                    bool sort_gradient, bool create_graph)
      : mapped_grad_var_(var.get()),
        sort_gradient_(sort_gradient),
        create_graph_(create_graph) {}

  void IncreaseTotalRefCnt() {
    ++total_ref_cnt_;

    // The gradient accumulator is needed only when total_ref_cnt_ > 1.
    // grad_var_ would be created only when total_ref_cnt_ > 1.
    if (total_ref_cnt_ > 1) {
      if (!grad_var_) {
        grad_var_ = std::make_shared<VarBase>(true, mapped_grad_var_->Name());
        grad_var_->SetOverridedStopGradient(false);
        if (sort_gradient_) {
          accumulator_.reset(
              new SortedGradientAccumulator(grad_var_->SharedVar().get()));
        } else {
          accumulator_.reset(
              new EagerGradientAccumulator(grad_var_->SharedVar().get()));
        }
        accumulator_->IncreaseRefCnt();
      }
      accumulator_->IncreaseRefCnt();
    }
  }

  size_t TotalRefCnt() { return total_ref_cnt_; }

  const std::shared_ptr<VarBase> &GradVarBase() const { return grad_var_; }

  std::shared_ptr<VariableWrapper> GradVar() const {
    return grad_var_ == nullptr ? nullptr : grad_var_->SharedVar();
  }

  VariableWrapper *MappedGradVar() { return mapped_grad_var_; }

  std::vector<std::shared_ptr<VariableWrapper>> SumGradient(
      std::shared_ptr<VariableWrapper> grad_var_partial, size_t trace_id,
      bool *is_finished, bool unchange_input = false) {
    PADDLE_ENFORCE_NOT_NULL(grad_var_partial,
                            platform::errors::PermissionDenied(
                                "Partial grad of %s would not be nullptr",
                                mapped_grad_var_->Name()));
    PADDLE_ENFORCE_GT(total_ref_cnt_, 1,
                      platform::errors::PermissionDenied(
                          "Gradient accumulation should not be called when "
                          "reference count is 1 or 0"));

    ++cur_ref_cnt_;
    PADDLE_ENFORCE_LE(cur_ref_cnt_, total_ref_cnt_,
                      platform::errors::PermissionDenied(
                          "Reference count overflows, this may be a bug"));

    *is_finished = (cur_ref_cnt_ == total_ref_cnt_);
    accumulator_->SumGrad(grad_var_partial, trace_id, unchange_input);

    if (*is_finished && accumulator_->HasInnerVar()) {
      accumulator_->AccumulateGrad();
    }

    if (create_graph_) {
      VLOG(10) << "Store partial grad grad for double grad "
               << mapped_grad_var_->Name();
      partial_grad_grads_.emplace_back(grad_var_partial->GetWeakGradVar(),
                                       trace_id);
    }

    if (!(*is_finished) || !create_graph_) {
      return {};
    }

    if (sort_gradient_) {
      std::sort(partial_grad_grads_.begin(), partial_grad_grads_.end(),
                [](const PartialGradGradTraceIdPair &p1,
                   const PartialGradGradTraceIdPair &p2) {
                  return p1.second > p2.second;
                });
    }

    // Only when create_graph_ = True, the return value would be not empty
    std::vector<std::shared_ptr<VariableWrapper>> result;
    result.reserve(partial_grad_grads_.size());
    for (auto &pair : partial_grad_grads_) {
      if (auto var = pair.first.lock()) {
        result.emplace_back(var);
      }
    }
    return result;
  }

 private:
  std::shared_ptr<VarBase> grad_var_;
  VariableWrapper *mapped_grad_var_;
  std::unique_ptr<GradientAccumulator> accumulator_;
  std::vector<PartialGradGradTraceIdPair> partial_grad_grads_;
  size_t total_ref_cnt_{0};
  size_t cur_ref_cnt_{0};
  bool sort_gradient_;
  bool create_graph_;
};

class ReadyGradVarInfoMap {
 private:
  struct ReadyVarInfo {
    std::shared_ptr<VarBase> var;
    size_t cur_ref_cnt{0};
    size_t total_ref_cnt{0};
  };

 public:
  void IncreaseRefCnt(const VariableWrapper *var) {
    ++(vars_[var].total_ref_cnt);
  }

  std::shared_ptr<VarBase> Get(const VariableWrapper *var,
                               const platform::Place &place, bool *is_last) {
    auto iter = vars_.find(var);
    PADDLE_ENFORCE_EQ(
        iter != vars_.end(), true,
        platform::errors::NotFound("Variable %s not found, this may be a bug",
                                   var->Name()));
    auto &ready_var = iter->second;
    PADDLE_ENFORCE_LT(ready_var.cur_ref_cnt, ready_var.total_ref_cnt,
                      platform::errors::PermissionDenied(
                          "Reference count overflows for %s", var->Name()));

    if (ready_var.var == nullptr && ready_var.cur_ref_cnt == 0) {
      ready_var.var = std::make_shared<VarBase>(var->Name());
      VLOG(10) << "Fill zero for " << var->Name() << " because it is not ready";
      FillConstantLike(*var, ready_var.var->SharedVar().get(), place, 0.0f);
    } else {
      PADDLE_ENFORCE_NOT_NULL(
          ready_var.var,
          platform::errors::NotFound(
              "%s is not found when reference count does not decreases to 0"));
    }

    if (++ready_var.cur_ref_cnt == ready_var.total_ref_cnt) {
      *is_last = true;
      return std::move(ready_var.var);  // move to set ready_var.var to nullptr
    } else {
      *is_last = false;
      return ready_var.var;
    }
  }

  // Set a var as a ready var.
  // If the var is one of target vars, store it inside `target_vars_` as well.
  bool Set(const VariableWrapper *mapped_var,
           const std::shared_ptr<VarBase> &var) {
    PADDLE_ENFORCE_NOT_NULL(
        var,
        platform::errors::PermissionDenied(
            "Cannot set nullptr as ready grad var for %s", mapped_var->Name()));
    {
      auto target_iter = target_vars_.find(mapped_var);
      if (target_iter != target_vars_.end()) {
        PADDLE_ENFORCE_EQ(
            target_iter->second, nullptr,
            platform::errors::PermissionDenied("Cannot set target var %s twice",
                                               mapped_var->Name()));
        target_iter->second = var;
      }
    }

    auto iter = vars_.find(mapped_var);
    if (iter != vars_.end()) {  // This var is ready for next op's input
      auto &ready_var = iter->second;
      PADDLE_ENFORCE_EQ(
          ready_var.var, nullptr,
          platform::errors::PermissionDenied("Cannot set target var %s twice",
                                             mapped_var->Name()));
      PADDLE_ENFORCE_EQ(
          ready_var.cur_ref_cnt, 0,
          platform::errors::PermissionDenied(
              "Reference count must be 0 when ready var %s is set",
              mapped_var->Name()));
      ready_var.var = var;
      return true;
    } else {
      VLOG(10) << "Do not record " << mapped_var->Name()
               << " because it is not input of any following ops";
      return false;
    }
  }

  void Clear() {
    vars_.clear();
    target_vars_.clear();
  }

  // Mark a var as target var
  void SetTarget(const VariableWrapper *var) {
    PADDLE_ENFORCE_EQ(target_vars_[var], nullptr,
                      platform::errors::PermissionDenied(
                          "Target var would not be generated when marking"));
  }

  // Get target var
  const std::shared_ptr<VarBase> &GetTarget(const VariableWrapper *var) const {
    auto iter = target_vars_.find(var);
    PADDLE_ENFORCE_EQ(iter != target_vars_.end(), true,
                      platform::errors::NotFound("Target var %s does not exist",
                                                 var->Name()));
    PADDLE_ENFORCE_NOT_NULL(
        iter->second, platform::errors::PermissionDenied(
                          "Target var %s should not be nullptr", var->Name()));
    return iter->second;
  }

 private:
  std::unordered_map<const VariableWrapper *, ReadyVarInfo> vars_;
  std::unordered_map<const VariableWrapper *, std::shared_ptr<VarBase>>
      target_vars_;
};

class PartialGradTask {
 public:
  PartialGradTask(const std::vector<std::shared_ptr<VarBase>> &input_targets,
                  const std::vector<std::shared_ptr<VarBase>> &output_targets,
                  const std::vector<std::shared_ptr<VarBase>> &output_grads,
                  const std::vector<std::shared_ptr<VarBase>> &no_grad_vars,
                  const platform::Place &place, bool create_graph,
                  bool retain_graph, bool allow_unused, bool only_inputs);

  std::vector<std::shared_ptr<VarBase>> Run();

 private:
  void RunEachOp(OpBase *op);

  void PrepareInitialReadyVarsMap(const OpBase *op);

  void PrepareInitialGradientAccumulators(const OpBase *op);

  std::vector<std::shared_ptr<VarBase>> CreateResult();

  bool IsValidGradVar(const std::shared_ptr<VariableWrapper> &var) const {
    return var && no_grad_var_grad_.count(var.get()) == 0;
  }

 private:
  std::unordered_set<OpBase *> startup_ops_;
  std::unordered_map<OpBase *, std::unordered_set<OpBase *>> pending_ops_;
  std::unordered_map<OpBase *, size_t> op_deps_;

  ReadyGradVarInfoMap ready_grad_vars_;

  std::unordered_map<VariableWrapper *,
                     std::unique_ptr<GradientAccumulationInfo>>
      grad_accumulators_;

  std::vector<std::shared_ptr<GradOpNode>> double_grad_nodes_;

  std::vector<
      std::pair<GradientAccumulationInfo *, std::shared_ptr<VariableWrapper>>>
      grads_to_accumulate_;

  // Input targets that are reachable
  std::vector<std::shared_ptr<VarBase>> input_targets_;
  std::unordered_set<VariableWrapper *> input_target_grads_;

  std::unordered_set<VariableWrapper *> no_grad_var_grad_;
  std::vector<std::weak_ptr<VariableWrapper>> reset_stop_gradient_vars_;

  platform::Place place_;
  bool create_graph_;
  bool retain_graph_;
  bool allow_unused_;
  bool only_inputs_;
};

PartialGradTask::PartialGradTask(
    const std::vector<std::shared_ptr<VarBase>> &input_targets,
    const std::vector<std::shared_ptr<VarBase>> &output_targets,
    const std::vector<std::shared_ptr<VarBase>> &output_grads,
    const std::vector<std::shared_ptr<VarBase>> &no_grad_vars,
    const platform::Place &place, bool create_graph, bool retain_graph,
    bool allow_unused, bool only_inputs) {
  input_targets_ = input_targets;
  place_ = place;
  create_graph_ = create_graph;
  retain_graph_ = retain_graph;
  allow_unused_ = allow_unused;
  only_inputs_ = only_inputs;

  PADDLE_ENFORCE_EQ(only_inputs_, true,
                    platform::errors::Unimplemented(
                        "only_inputs=False is not supported yet"));

  for (auto &var : no_grad_vars) {
    if (var && var->GradVarBase()) {
      no_grad_var_grad_.insert(var->GradVarBase()->SharedVar().get());
    }
  }

  PADDLE_ENFORCE_EQ(
      input_targets.empty(), false,
      platform::errors::PermissionDenied("inputs can not be empty"));
  PADDLE_ENFORCE_EQ(
      output_targets.empty(), false,
      platform::errors::PermissionDenied("outputs can not be empty"));

  std::unordered_set<VarBase *> out_set;
  for (auto &output : output_targets) {
    PADDLE_ENFORCE_NOT_NULL(output,
                            platform::errors::PermissionDenied(
                                "Variable inside outputs should not be null"));
    PADDLE_ENFORCE_EQ(
        output->GradVarBase() && !output->OverridedStopGradient(), true,
        platform::errors::PermissionDenied(
            "Variable %s inside outputs has no gradient", output->Name()));
    PADDLE_ENFORCE_EQ(
        out_set.count(output.get()), 0,
        platform::errors::AlreadyExists("outputs contain duplicate variable %s",
                                        output->Name()));
    PADDLE_ENFORCE_EQ(IsValidGradVar(output->GradVarBase()->SharedVar()), true,
                      platform::errors::PermissionDenied(
                          "outputs contain var that is inside no_grad_set"));

    out_set.insert(output.get());
  }

  std::unordered_set<VarBase *> in_set;
  std::unordered_set<VariableWrapper *> one_grad_vars;
  for (auto &input : input_targets) {
    PADDLE_ENFORCE_NOT_NULL(input,
                            platform::errors::PermissionDenied(
                                "Variable inside inputs should not be null"));
    PADDLE_ENFORCE_EQ(
        input->GradVarBase() && !input->OverridedStopGradient(), true,
        platform::errors::PermissionDenied(
            "Variable %s inside inputs has no gradient", input->Name()));
    PADDLE_ENFORCE_EQ(
        in_set.count(input.get()), 0,
        platform::errors::AlreadyExists("inputs contain duplicate variable %s",
                                        input->Name()));
    in_set.insert(input.get());
    input_target_grads_.insert(input->GradVarBase()->SharedVar().get());

    PADDLE_ENFORCE_EQ(IsValidGradVar(input->GradVarBase()->SharedVar()), true,
                      platform::errors::PermissionDenied(
                          "inputs contain var that is inside no_grad_set"));

    // Record same vars between inputs and outputs
    if (out_set.count(input.get()) > 0) {
      one_grad_vars.insert(input->GradVarBase()->SharedVar().get());
    }
  }

  std::unordered_set<VariableWrapper *> related_grad_vars;
  GetGraphInfoBetweenTargets(&input_target_grads_, &out_set, &startup_ops_,
                             &pending_ops_, &op_deps_, &related_grad_vars,
                             no_grad_var_grad_);

  for (auto &op_pair : pending_ops_) {
    auto *op = op_pair.first;
    PrepareInitialReadyVarsMap(op);
    PrepareInitialGradientAccumulators(op);
  }

  for (auto &input_grad : input_target_grads_) {
    ready_grad_vars_.SetTarget(input_grad);
  }

  for (auto &one_grad : one_grad_vars) {
    VLOG(10) << "Add same in/out target " << one_grad->Name();
    input_target_grads_.insert(one_grad);
    ready_grad_vars_.SetTarget(one_grad);
  }

  VLOG(10) << "Valid op number " << pending_ops_.size();

  if (!output_grads.empty()) {
    PADDLE_ENFORCE_EQ(output_targets.size(), output_grads.size(),
                      platform::errors::InvalidArgument(
                          "grad_outputs number should be equal to outputs"));
  }

  for (size_t i = 0; i < output_targets.size(); ++i) {
    auto *mapped_out_grad_var =
        output_targets[i]->GradVarBase()->SharedVar().get();

    if (related_grad_vars.count(mapped_out_grad_var) == 0 &&
        one_grad_vars.count(mapped_out_grad_var) == 0) {
      VLOG(10) << mapped_out_grad_var->Name() << " should be None";
      continue;
    }

    std::shared_ptr<VariableWrapper> out_grad_var;
    bool unchange_input = false;
    if (output_grads.empty() || output_grads[i] == nullptr) {
      VLOG(10) << "Fill 1.0f for " << output_targets[i]->Name();
      out_grad_var = std::make_shared<VariableWrapper>(
          framework::GradVarName(output_targets[i]->Name()));
      FillConstantLike(*(output_targets[i]->SharedVar()), out_grad_var.get(),
                       place_, 1.0f);
    } else {
      VLOG(10) << "Use user provided grad var for "
               << output_targets[i]->Name();
      const auto &out_tensor =
          output_targets[i]->Var().Get<framework::LoDTensor>();
      const auto &grad_tensor =
          output_grads[i]->Var().Get<framework::LoDTensor>();
      PADDLE_ENFORCE_EQ(
          grad_tensor.dims(), out_tensor.dims(),
          platform::errors::InvalidArgument(
              "The %d-th grad_output's shape does not match the %d-th output",
              i, i));
      PADDLE_ENFORCE_EQ(grad_tensor.type(), out_tensor.type(),
                        platform::errors::InvalidArgument(
                            "The %d-th grad_output's data type does not "
                            "match the %d-th output",
                            i, i));
      out_grad_var = output_grads[i]->SharedVar();
      PADDLE_ENFORCE_EQ(IsValidGradVar(out_grad_var), true,
                        platform::errors::PermissionDenied(
                            "grad_outputs contain var inside no_grad_set"));

      if (out_grad_var->OverridedStopGradient()) {
        VLOG(10) << "Grad var " << out_grad_var->Name()
                 << " should reset stop gradient";
        reset_stop_gradient_vars_.emplace_back(out_grad_var);
      }

      unchange_input = true;
    }

    out_grad_var->SetOverridedStopGradient(false);
    auto grad_accumulator_iter = grad_accumulators_.find(mapped_out_grad_var);
    if (grad_accumulator_iter == grad_accumulators_.end()) {
      ready_grad_vars_.Set(mapped_out_grad_var,
                           std::make_shared<VarBase>(out_grad_var));
      VLOG(10) << "Fill 1.0f or user-provided gradient as ready var "
               << out_grad_var->Name();
    } else {
      auto &accumulator = grad_accumulator_iter->second;
      accumulator->IncreaseTotalRefCnt();
      bool is_finished = false;
      accumulator->SumGradient(out_grad_var, 0, &is_finished, unchange_input);
      PADDLE_ENFORCE_EQ(
          is_finished, false,
          platform::errors::Fatal("gradient accumulator should not finish"));
      VLOG(10) << "Add 1.0f or user-provided gradient to gradient accumulator"
               << out_grad_var->Name();
    }
  }
}

std::vector<std::shared_ptr<VarBase>> PartialGradTask::Run() {
  VLOG(10) << "Startup op number " << startup_ops_.size();
  std::queue<OpBase *> q;
  for (auto *op : startup_ops_) {
    q.push(op);
  }

  while (!q.empty()) {
    auto *op = q.front();
    q.pop();

    VLOG(10) << "Start to run " << op->Type();
    op->EnforceHasInOut();
    RunEachOp(op);
    if (!retain_graph_) {
      op->ClearBackwardTrace();
    }
    VLOG(10) << "End to run " << op->Type();

    auto iter = pending_ops_.find(op);
    if (iter == pending_ops_.end()) {
      VLOG(10) << "Finish running because " << op->Type()
               << " has no pending ops";
      continue;
    }

    for (auto &pending_op : iter->second) {
      auto dep_iter = op_deps_.find(pending_op);
      PADDLE_ENFORCE_EQ(
          dep_iter != op_deps_.end(), true,
          platform::errors::Fatal("Dependency number of %s does not exist",
                                  pending_op->Type()));
      if (--(dep_iter->second) == 0) {
        q.push(pending_op);
      }
    }
  }

  VLOG(10) << "Created " << double_grad_nodes_.size() << " double grad ops";
  return CreateResult();
}

void PartialGradTask::RunEachOp(OpBase *op) {
  // Prepare new inputs
  NameVarMap<VarBase> tmp_ins;
  for (auto &input_pair : op->GetInsMap()) {
    auto &new_inputs = tmp_ins[input_pair.first];
    new_inputs.reserve(input_pair.second.size());

    if (!input_pair.second.IsGrad()) {
      for (auto &fwd_var : input_pair.second) {
        if (fwd_var) {
          new_inputs.emplace_back(new VarBase(fwd_var));
          VLOG(10) << "Unpacked forward var " << fwd_var->Name()
                   << ", grad ops: " << GradOpTypes(*new_inputs.back());
        } else {
          new_inputs.emplace_back();
        }
      }
    } else {
      for (auto &grad_var : input_pair.second) {
        if (grad_var) {
          bool is_last;
          new_inputs.emplace_back(
              ready_grad_vars_.Get(grad_var.get(), op->place(), &is_last));
          VLOG(10) << "Got ready grad var " << grad_var->Name() << " "
                   << new_inputs.back().get();
        } else {
          new_inputs.emplace_back();
        }
      }
    }
  }

  // Prepare new outputs
  NameVarMap<VarBase> tmp_outs;
  for (auto &output_pair : op->GetOutsMap()) {
    auto &new_outputs = tmp_outs[output_pair.first];
    if (!output_pair.second.IsGrad()) {
      for (auto &fwd_var : output_pair.second) {
        // unpack forward var
        if (fwd_var) {
          new_outputs.emplace_back(new VarBase(fwd_var));
          VLOG(10) << "Unpacked forward var " << fwd_var->Name();
        } else {
          new_outputs.emplace_back();
        }
      }
    } else {
      for (auto &grad_var : output_pair.second) {
        if (IsValidGradVar(grad_var)) {
          VLOG(10) << "Creating output grad var " << grad_var->Name();
          auto new_grad_var_iter = grad_accumulators_.find(grad_var.get());
          PADDLE_ENFORCE_EQ(new_grad_var_iter != grad_accumulators_.end(), true,
                            platform::errors::Fatal(
                                "Cannot find gradient accumulator of %s %p",
                                grad_var->Name(), grad_var.get()));

          auto new_grad_var = std::make_shared<VarBase>(true, grad_var->Name());
          new_grad_var->SetOverridedStopGradient(false);
          new_grad_var->SetForwardDataType(grad_var->ForwardDataType());
          if (new_grad_var_iter->second->TotalRefCnt() > 1) {
            grads_to_accumulate_.emplace_back(new_grad_var_iter->second.get(),
                                              new_grad_var->SharedVar());
          } else {
            PADDLE_ENFORCE_EQ(
                new_grad_var_iter->second->GradVar(), nullptr,
                platform::errors::AlreadyExists(
                    "When reference count is 1, the grad var should not be "
                    "created in gradient accumulator"));
            grad_accumulators_.erase(new_grad_var_iter);
            ready_grad_vars_.Set(grad_var.get(), new_grad_var);
          }
          VLOG(10) << "Created output grad var " << grad_var->Name();
          new_outputs.emplace_back(std::move(new_grad_var));
        } else {
          new_outputs.emplace_back();
        }
      }
    }
  }

  // Run op
  OpBase::Run(op->InnerOp(), tmp_ins, tmp_outs, op->Attrs(),
              op->DefaultAttrsMap(), op->place());

  if (create_graph_) {
    auto double_grad_node =
        CreateGradOpNode(op->InnerOp(), tmp_ins, tmp_outs, op->Attrs(),
                         op->DefaultAttrsMap(), op->place(), {});
    PADDLE_ENFORCE_NOT_NULL(
        double_grad_node,
        platform::errors::NotFound("The Op %s doesn't have any grad op. If you "
                                   "don't intend calculating higher order "
                                   "derivatives, please set `create_graph` to "
                                   "False.",
                                   op->Type()));
    VLOG(10) << "Create " << double_grad_node->size()
             << " double grad op(s) for " << op->Type()
             << ", pending ops: " << GradPendingOpTypes(*double_grad_node);
    double_grad_nodes_.emplace_back(std::move(double_grad_node));
  }

  VLOG(10) << "There are " << grads_to_accumulate_.size() << " to sum gradient";

  // Gradient accumulation and add assign op
  for (auto &pair : grads_to_accumulate_) {
    auto *accumulator_info = pair.first;
    auto &grad_var = pair.second;

    bool is_finished = false;
    VLOG(10) << "Start to sum " << accumulator_info->MappedGradVar()->Name();
    auto partial_grad_grads = accumulator_info->SumGradient(
        std::move(grad_var), op->id(), &is_finished);

    if (!partial_grad_grads.empty()) {
      auto sum_grad_var_grad =
          accumulator_info->GradVarBase()->MutableGradVarBase();
      sum_grad_var_grad->SetOverridedStopGradient(false);

      auto assign_node = std::make_shared<GradOpNode>();
      sum_grad_var_grad->SetGradNode(assign_node);

      VLOG(10) << "Add " << partial_grad_grads.size() << " assign op for "
               << sum_grad_var_grad->Name();

      for (auto &grad_grad : partial_grad_grads) {
        auto *assign_op = &(assign_node->emplace_back());
        assign_op->SetType("assign");  // Can use "scale" as static graph mode
        assign_op->SetInput("X", {sum_grad_var_grad->SharedVar()}, true);
        assign_op->SetOutput("Out", {grad_grad}, true);
        assign_op->CheckAttrs();
        assign_op->SetId(OpBase::GenerateUniqueId());
        assign_op->SetPlace(op->place());

        if (auto grad_pending_node = grad_grad->GetGradNode()) {
          assign_node->InsertGradPendingNode(std::move(grad_pending_node));
        }
      }
      VLOG(10) << "Pending ops of assign is "
               << GradPendingOpTypes(*assign_node);
      double_grad_nodes_.emplace_back(assign_node);
    }

    if (is_finished) {
      VLOG(10) << "Sum has finished for "
               << accumulator_info->MappedGradVar()->Name() << " "
               << accumulator_info->GradVarBase();
      ready_grad_vars_.Set(accumulator_info->MappedGradVar(),
                           accumulator_info->GradVarBase());
      grad_accumulators_.erase(accumulator_info->MappedGradVar());
    }
  }

  grads_to_accumulate_.clear();
}

void PartialGradTask::PrepareInitialReadyVarsMap(const OpBase *op) {
  for (auto &in_var_pair : op->GetInsMap()) {
    if (!in_var_pair.second.IsGrad()) {
      continue;
    }

    for (auto &var : in_var_pair.second) {
      if (var) {
        ready_grad_vars_.IncreaseRefCnt(var.get());
      }
    }
  }
}

void PartialGradTask::PrepareInitialGradientAccumulators(const OpBase *op) {
  for (auto &out_var_pair : op->GetOutsMap()) {
    if (!out_var_pair.second.IsGrad()) {
      continue;
    }

    for (auto &var : out_var_pair.second) {
      if (var == nullptr) {
        continue;
      }

      auto &accumulator = grad_accumulators_[var.get()];

      if (!accumulator) {
        accumulator.reset(new GradientAccumulationInfo(
            var, FLAGS_sort_sum_gradient, create_graph_));
      }

      accumulator->IncreaseTotalRefCnt();
    }
  }
}

std::vector<std::shared_ptr<VarBase>> PartialGradTask::CreateResult() {
  std::vector<std::shared_ptr<VarBase>> result;
  result.reserve(input_targets_.size());
  for (size_t i = 0; i < input_targets_.size(); ++i) {
    auto &input_target = input_targets_[i];
    PADDLE_ENFORCE_NOT_NULL(
        input_target->GradVarBase(),
        platform::errors::InvalidArgument("input should have gradient"));
    auto *original_grad_var = input_target->GradVarBase()->SharedVar().get();
    auto iter = input_target_grads_.find(original_grad_var);
    if (iter != input_target_grads_.end()) {
      auto ready_var = ready_grad_vars_.GetTarget(original_grad_var);
      ready_var->SetOverridedStopGradient(!create_graph_);
      result.emplace_back(std::move(ready_var));
    } else {  // return None if it does not appear in the graph
      PADDLE_ENFORCE_EQ(allow_unused_, true,
                        platform::errors::InvalidArgument(
                            "The %d-th input does not appear in the backward "
                            "graph. Please check the input variable or set "
                            "allow_unused=True to get None result.",
                            i));
      result.emplace_back();
    }
  }

  for (auto &weak_var : reset_stop_gradient_vars_) {
    if (auto var = weak_var.lock()) {
      VLOG(10) << "Reset " << var->Name() << " stop gradient";
      var->SetOverridedStopGradient(!var->OverridedStopGradient());
    }
  }

  ready_grad_vars_.Clear();
  grad_accumulators_.clear();
  double_grad_nodes_.clear();
  reset_stop_gradient_vars_.clear();
  return result;
}

PartialGradEngine::PartialGradEngine(
    const std::vector<std::shared_ptr<VarBase>> &input_targets,
    const std::vector<std::shared_ptr<VarBase>> &output_targets,
    const std::vector<std::shared_ptr<VarBase>> &output_grads,
    const std::vector<std::shared_ptr<VarBase>> &no_grad_vars,
    const platform::Place &place, bool create_graph, bool retain_graph,
    bool allow_unused, bool only_inputs)
    : task_(new PartialGradTask(input_targets, output_targets, output_grads,
                                no_grad_vars, place, create_graph, retain_graph,
                                allow_unused, only_inputs)) {}

PartialGradEngine::~PartialGradEngine() { Clear(); }

std::vector<std::shared_ptr<VarBase>> PartialGradEngine::GetResult() const {
  return results_;
}

void PartialGradEngine::Clear() {
  if (task_) {
    delete task_;
    task_ = nullptr;
  }
}

void PartialGradEngine::Execute() {
  PADDLE_ENFORCE_NOT_NULL(task_, platform::errors::PermissionDenied(
                                     "PartialGradEngine has been destructed"));
  VLOG(10) << "Starts to execute PartialGradEngine";
  results_ = task_->Run();
  Clear();
}

}  // namespace imperative
}  // namespace paddle
