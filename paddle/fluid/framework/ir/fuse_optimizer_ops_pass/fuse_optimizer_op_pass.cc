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

#include "paddle/fluid/framework/ir/fuse_optimizer_ops_pass/fuse_optimizer_op_pass.h"
#include <algorithm>
#include <set>
#include <unordered_set>
#include "paddle/fluid/framework/ir/graph_helper.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/operator.h"

namespace paddle {
namespace framework {
namespace ir {

void FuseOptimizerOpPass::ApplyImpl(ir::Graph *graph) const {
  ir::Graph &result = *graph;

  const std::string fuse_op_type = GetOpType();
  std::vector<std::string> aux_var_names = GetAuxiliaryVarNames();
  aux_var_names.emplace_back(kParam);
  aux_var_names.emplace_back(kGrad);

  // Step 1: Get the specified op and auxiliary variables.
  std::vector<ir::Node *> topo_nodes = ir::TopologySortOperations(result);
  auto vars_info = GetVarInfo(result);
  std::vector<ir::Node *> opt_nodes;
  size_t opt_ops_num = 0;
  // Note: Only take care about the dense gradients.
  for (auto &node : topo_nodes) {
    if (node->Op()->Type() == fuse_op_type) {
      auto grad_name = node->Op()->Input(kGrad);
      PADDLE_ENFORCE_EQ(grad_name.size(), static_cast<size_t>(1),
                        "The %s operator has multiple gradient input. Expected "
                        "it to only have one gradient input.",
                        fuse_op_type);
      if (IsLoDTensorType(GetTypeOfVar(vars_info, grad_name[0]))) {
        opt_nodes.emplace_back(node);
      }
      ++opt_ops_num;
    }
  }

  VLOG(6) << "Find " << fuse_op_type << " operators : " << opt_ops_num
          << ", and " << opt_nodes.size() << " for dense gradients.";

  if (opt_nodes.size() == 0) return;
  if (result.Has(details::kFusedOptType)) {
    auto &opt_type = result.Get<details::FusedOptType>(details::kFusedOptType);
    VLOG(6) << "Currently only support fusing one type optimizer op. "
            << opt_type << " has been fused.";
  }

  // There should not have no-ctr-var between the opt_nodes that link the
  // op_node
  // of opt_nodes.
  if (HasVarDepsBetweenOps(topo_nodes, opt_nodes)) {
    VLOG(6) << "There are interdependent variables among these optimization "
               "operators, which can not be handled well at present.";
    return;
  }

  result.Set(details::kFusedOptType, new details::FusedOptType);
  result.Get<details::FusedOptType>(details::kFusedOptType) = fuse_op_type;
  if (!result.Has(details::kProgramDescs)) {
    result.Set(details::kProgramDescs, new details::ProgramDescs);
  }

  // Step 2: Insert fused_var_name to FusedVars, and the FusedVars need be
  // initialized in scopes before execution.
  if (!result.Has(details::kFusedVars)) {
    result.Set(details::kFusedVars, new details::FusedVars);
  }
  std::unordered_map<std::string, std::vector<std::string>> aux_var_map;
  GetFusingVarNamesMap(aux_var_names, opt_nodes, &aux_var_map);
  std::unordered_map<std::string, std::string> fused_vars_name;
  fused_vars_name.reserve(aux_var_names.size());
  auto &fused_var_set = result.Get<details::FusedVars>(details::kFusedVars);
  const std::string prefix(details::kFusedVarNamePrefix);
  for (auto &var_name : aux_var_names) {
    // NOTE: the fused_var_name should be unique.
    auto fused_var_name = prefix + "_" + fuse_op_type + "_" + var_name + "_" +
                          aux_var_map[var_name][0];
    VLOG(6) << var_name << ": " << fused_var_name;
    PADDLE_ENFORCE_EQ(fused_var_set.count(fused_var_name), 0);
    fused_var_set.insert(fused_var_name);
    fused_vars_name.emplace(var_name, fused_var_name);
  }

  // Step 3: Get the fused Gradient's name
  bool grad_fused = false;
  if (result.Has(details::kParamsAndDenseGrads)) {
    // NOTE: kParamsAndDenseGrads is generated by
    // alloc_continue_space_for_grad_pass
    auto &params_and_dense_grads =
        result.Get<details::ParamsAndGrads>(details::kParamsAndDenseGrads);
    PADDLE_ENFORCE_LE(
        params_and_dense_grads.size(), aux_var_map.at(kGrad).size(),
        "The number of dense gradients should be little than optimizer ops.");

    std::unordered_set<std::string> opt_grad_set(aux_var_map.at(kGrad).size());
    for (auto &p_g : params_and_dense_grads) {
      opt_grad_set.insert(p_g.second);
    }
    std::vector<size_t> new_grad_idx;
    for (size_t idx = 0; idx < aux_var_map.at(kGrad).size(); ++idx) {
      auto &grad = aux_var_map.at(kGrad).at(idx);
      if (!opt_grad_set.count(grad)) {
        new_grad_idx.emplace_back(idx);
      }
    }

    // NOTE(zcd): the gradient of kParamsAndDenseGrads may be different
    // with the kGrad. The gradients of kParamsAndDenseGrads is
    // collected during backward stage, but in optimization state, the
    // some gradient's name maybe changed.
    if (new_grad_idx.size() == 0) {
      if (!result.Has(details::kFusedGrads)) {
        PADDLE_THROW(
            "The coalesce_grad_tensor_pass should "
            "be called before this pass.");
      }
      auto &fused_grad = result.Get<details::FusedGrads>(details::kFusedGrads);
      PADDLE_ENFORCE_NE(fused_grad.size(), 0,
                        "The fused gradient should not be empty.");
      if (fused_grad.size() > 1) {
        VLOG(6) << "Because the dtype of those gradients "
                   "is not unified, so the number of fused gradients is "
                   "more than one, but it is not supported currently.";
        return;
      }
      auto &fused_vars = result.Get<details::FusedVars>(details::kFusedVars);
      auto iter =
          std::find(fused_vars.begin(), fused_vars.end(), fused_grad.front());
      PADDLE_ENFORCE_EQ(iter != fused_vars.end(), true,
                        "Not found the fused gradient variable.");
      fused_vars_name[kGrad] = fused_grad.front();

      // Sort the parameters and auxiliary variables according
      // to parameters' name to make variables' name correspond correctly.
      SortParametersAndAuxVars(params_and_dense_grads, &aux_var_map,
                               &opt_nodes);
      grad_fused = true;
    } else {
      VLOG(6) << "The number of new gradients is " << new_grad_idx.size();
      if (new_grad_idx.size() == 1) return;
      // NOTE(zcd): If the gradients of backward stage and optimization stage
      // have diff, Only take care of the the gradient of optimization stage.
      GradientsFilter(new_grad_idx, &opt_nodes, &aux_var_map);
    }
  }

  // Pass pre-condition check: check dtype of fusing vars
  auto fusing_var_dtype =
      GetDtypeOfVar(vars_info, aux_var_map.at(kParam).front());
  for (auto vars : aux_var_map) {
    for (auto &var_name : vars.second) {
      if (fusing_var_dtype != GetDtypeOfVar(vars_info, var_name)) {
        VLOG(6) << "Currently the fuse_optimizer_ops strategy in mixed "
                   "precision scenarios is not yet supported.";
        return;
      }
    }
  }

  // Pass pre-condition check: gradients generated op kernel
  auto fusing_grad_var_names = aux_var_map.at(kGrad);
  for (auto grad_var_name : fusing_grad_var_names) {
    if (!GradGeneratedOpSupportGPU(vars_info, grad_var_name)) {
      VLOG(6) << "Currently the fuse_optimizer_ops strategy is risky when "
                 "gradient generated operator doesn't support GPU device.";
      return;
    }
  }

  LOG(WARNING) << "Find " << fuse_op_type << " operators : " << opt_ops_num
               << ", and " << opt_nodes.size() << " for dense gradients. "
               << "To make the speed faster, those optimization are fused "
                  "during training.";

  // Step 4: Alloc continuous space for Parameters and AuxiliaryVar(e.g.
  // Moment1, Moment2, Beta1Pow, Beta2Pow) of all the optimizer ops
  // separately.
  if (!grad_fused) {
    InitFusedGradsAndAllocSpaceForGrads(
        aux_var_map.at(kParam), aux_var_map.at(kGrad),
        fused_vars_name.at(kGrad), fusing_var_dtype, &result);
  }
  aux_var_names.pop_back();
  InitFusedVarsAndAllocSpaceForVars(aux_var_names, aux_var_map, fused_vars_name,
                                    fusing_var_dtype, &result);

  // Step 5: Fuse optimizer Ops and Scale Ops
  auto *fused_opt_node =
      FuseOptimizerOps(aux_var_map, fused_vars_name, opt_nodes, &result);

  InsertInputAndOutputForFusedOpNode(opt_nodes, graph, fused_opt_node);
  // Step 6: Remove optimizer Ops
  for (auto &opt_op : opt_nodes) {
    graph->RemoveNode(opt_op);
  }
}

bool FuseOptimizerOpPass::HasVarDepsBetweenOps(
    const std::vector<Node *> &topo_nodes,
    const std::vector<Node *> &opt_nodes) const {
  std::unordered_map<Node *, std::unordered_set<Node *>> preceding_ops;
  std::unordered_map<Node *, std::unordered_set<Node *>> pending_ops;
  for (auto &op : topo_nodes) {
    preceding_ops[op];
    pending_ops[op];
    for (auto &var : op->outputs) {
      if (var->IsCtrlVar()) continue;
      for (auto &pending_op : var->outputs) {
        preceding_ops[pending_op].insert(op);
        pending_ops[op].insert(pending_op);
      }
    }
  }

  std::unordered_set<Node *> opt_node_set(opt_nodes.begin(), opt_nodes.end());
  auto has_var_deps = [](const std::unordered_set<Node *> &op_set1,
                         const std::unordered_set<Node *> &op_set2) -> bool {
    std::set<Node *> intersect_ops;
    set_intersection(op_set1.begin(), op_set1.end(), op_set2.begin(),
                     op_set2.end(),
                     inserter(intersect_ops, intersect_ops.begin()));
    return !intersect_ops.empty();
  };

  for (auto opt_node : opt_node_set) {
    if (has_var_deps(preceding_ops.at(opt_node), opt_node_set)) {
      return true;
    }
    if (has_var_deps(pending_ops.at(opt_node), opt_node_set)) {
      return true;
    }
  }
  return false;
}

bool FuseOptimizerOpPass::GradGeneratedOpSupportGPU(
    const std::unordered_map<std::string, std::vector<ir::Node *>> &vars_info,
    const std::string &grad_var_name) const {
  auto grad_var_nodes = vars_info.at(grad_var_name);
  PADDLE_ENFORCE_EQ(
      grad_var_nodes.size(), 1,
      "The gradient variable %s has multiple generated operators.",
      grad_var_name);
  auto grad_var_node = grad_var_nodes.front();
  for (auto in_node : grad_var_node->inputs) {
    if (in_node->IsOp() && in_node->Op()) {
      VLOG(6) << "Op kernel suport GPU check: " << in_node->Op()->Type();
      if (!framework::OpSupportGPU(in_node->Op()->Type())) {
        return false;
      }
    }
  }
  return true;
}

void FuseOptimizerOpPass::GradientsFilter(
    const std::vector<size_t> &new_grad_idx, std::vector<Node *> *opt_nodes,
    std::unordered_map<std::string, std::vector<std::string>> *aux_var_map)
    const {
  for (auto &aux_vars : *aux_var_map) {
    std::vector<std::string> sorted_vars;
    sorted_vars.reserve(aux_vars.second.size());
    for (size_t i : new_grad_idx) {
      sorted_vars.emplace_back(aux_vars.second.at(i));
    }
    std::swap(aux_vars.second, sorted_vars);
    if (VLOG_IS_ON(6)) {
      std::stringstream out;
      for (auto &var_name : aux_vars.second) {
        out << var_name << " ";
      }
      VLOG(6) << aux_vars.first << ": " << out.str();
    }
  }
  std::vector<Node *> sorted_ops;
  for (size_t i : new_grad_idx) {
    sorted_ops.emplace_back(opt_nodes->at(i));
  }
  std::swap(*opt_nodes, sorted_ops);
}

void FuseOptimizerOpPass::InitFusedGradsAndAllocSpaceForGrads(
    const std::vector<std::string> &params,
    const std::vector<std::string> &grads, const std::string &fused_grad_name,
    const proto::VarType::Type &dtype, ir::Graph *result) const {
  auto &pinned_var_set =
      result->GetOrInit<details::PinnedVars>(details::kPinnedVars);

  auto vars_info = GetVarInfo(*result);
  // The Gradients should not be reused during memory optimization.
  for (auto &grad_var_name : grads) {
    auto iter = vars_info.find(grad_var_name);
    PADDLE_ENFORCE_EQ(iter != vars_info.end(), true, "%s is not found.",
                      grad_var_name);
    PADDLE_ENFORCE_EQ(!iter->second.empty(), true, "%s is not found.",
                      grad_var_name);
    PADDLE_ENFORCE_NOT_NULL(iter->second.front()->Var());
    PADDLE_ENFORCE_EQ(
        IsLoDTensorType(iter->second.front()->Var()->GetType()), true,
        "Currently the gradient type only should be LoDTensor when "
        "fusing optimizer ops.");
    for (auto var : iter->second) {
      pinned_var_set.insert(var->Var()->Name());
    }
  }

  // Define Ops
  result->Get<details::ProgramDescs>(details::kProgramDescs).emplace_back();
  ProgramDesc &program_desc =
      result->Get<details::ProgramDescs>(details::kProgramDescs).back();
  auto *global_block = program_desc.MutableBlock(0);
  AppendAllocContinuousSpace(params, grads, fused_grad_name, dtype,
                             global_block, false, false);
}

std::unordered_map<std::string, std::vector<Node *>>
FuseOptimizerOpPass::GetVarInfo(const Graph &result) const {
  std::unordered_map<std::string, std::vector<Node *>> vars;
  for (Node *node : result.Nodes()) {
    if (node->IsVar() && node->Var()) {
      // Note: The graph may have the same name node. For example, parameter
      // is the input of optimizer and it also is the output of optimizer;
      vars[node->Var()->Name()].emplace_back(node);
    }
  }
  return vars;
}

bool FuseOptimizerOpPass::IsLoDTensorType(
    const proto::VarType::Type &type) const {
  // Current only support LOD_TENSOR.
  return type == proto::VarType::LOD_TENSOR;
}

const VarDesc *FuseOptimizerOpPass::GetVarDescFromVarsInfo(
    const std::unordered_map<std::string, std::vector<Node *>> &vars_info,
    const std::string &var_name) const {
  auto grad_iter = vars_info.find(var_name);
  PADDLE_ENFORCE_EQ(grad_iter != vars_info.end(), true, "%s is not found.",
                    var_name);
  PADDLE_ENFORCE_EQ(!grad_iter->second.empty(), true, "%s is not found.",
                    var_name);
  PADDLE_ENFORCE_NOT_NULL(grad_iter->second.front()->Var());
  return grad_iter->second.front()->Var();
}

proto::VarType::Type FuseOptimizerOpPass::GetDtypeOfVar(
    const std::unordered_map<std::string, std::vector<ir::Node *>> &vars_info,
    const std::string &name) const {
  auto var_desc = GetVarDescFromVarsInfo(vars_info, name);
  return var_desc->GetDataType();
}

proto::VarType::Type FuseOptimizerOpPass::GetTypeOfVar(
    const std::unordered_map<std::string, std::vector<Node *>> &vars_info,
    const std::string &name) const {
  auto var_desc = GetVarDescFromVarsInfo(vars_info, name);
  return var_desc->GetType();
}

void FuseOptimizerOpPass::InitFusedVarsAndAllocSpaceForVars(
    const std::vector<std::string> &aux_var_names,
    const std::unordered_map<std::string, std::vector<std::string>>
        &aux_var_map,
    const std::unordered_map<std::string, std::string> &fused_vars_name,
    const proto::VarType::Type &dtype, ir::Graph *result) const {
  // Define Ops
  result->Get<details::ProgramDescs>(details::kProgramDescs).emplace_back();
  ProgramDesc &program_desc =
      result->Get<details::ProgramDescs>(details::kProgramDescs).back();
  auto *global_block = program_desc.MutableBlock(0);
  for (auto &var_name : aux_var_names) {
    AppendAllocContinuousSpace(
        aux_var_map.at(var_name), aux_var_map.at(var_name),
        fused_vars_name.at(var_name), dtype, global_block, true);
  }
}

void FuseOptimizerOpPass::SortParametersAndAuxVars(
    const std::vector<std::pair<std::string, std::string>> &params_grads,
    std::unordered_map<std::string, std::vector<std::string>> *aux_vars_set,
    std::vector<ir::Node *> *ops) const {
  PADDLE_ENFORCE_NE(aux_vars_set->count(kGrad), static_cast<size_t>(0));
  auto &grad_vec = aux_vars_set->at(kGrad);

  std::vector<size_t> grad_sort_idx;
  grad_sort_idx.reserve(grad_vec.size());

  for (auto &p_g : params_grads) {
    auto iter = std::find(grad_vec.begin(), grad_vec.end(), p_g.second);
    PADDLE_ENFORCE_EQ(iter != grad_vec.end(), true,
                      "%s is not found in grad_vec", p_g.second);
    auto idx = std::distance(grad_vec.begin(), iter);
    grad_sort_idx.emplace_back(idx);
  }

  for (auto &aux_vars : *aux_vars_set) {
    std::vector<std::string> sorted_vars;
    sorted_vars.reserve(aux_vars.second.size());
    for (size_t i = 0; i < aux_vars.second.size(); ++i) {
      sorted_vars.emplace_back(aux_vars.second.at(grad_sort_idx[i]));
    }
    std::swap(aux_vars.second, sorted_vars);

    if (VLOG_IS_ON(6)) {
      std::stringstream out;
      for (auto &var_name : aux_vars.second) {
        out << var_name << " ";
      }
      VLOG(6) << aux_vars.first << ": " << out.str();
    }
  }

  std::vector<ir::Node *> sorted_ops;
  sorted_ops.reserve(ops->size());
  for (size_t i = 0; i < ops->size(); ++i) {
    sorted_ops.emplace_back(ops->at(grad_sort_idx[i]));
  }
  std::swap(*ops, sorted_ops);
}

void FuseOptimizerOpPass::GetFusingVarNamesMap(
    const std::vector<std::string> &aux_vars_name,
    const std::vector<ir::Node *> &opt_nodes,
    std::unordered_map<std::string, std::vector<std::string>> *aux_args_name)
    const {
  for (auto &node : opt_nodes) {
    for (auto &var_n : aux_vars_name) {
      auto arg_names = node->Op()->Input(var_n);
      PADDLE_ENFORCE_EQ(arg_names.size(), static_cast<size_t>(1),
                        "The input variable of optimizer to be fused is "
                        "invalid. Excepted %s only has one %s input.",
                        node->Op()->Type(), var_n);
      (*aux_args_name)[var_n].emplace_back(arg_names[0]);
    }
  }
}

void FuseOptimizerOpPass::AppendAllocContinuousSpace(
    const std::vector<std::string> &in_args,
    const std::vector<std::string> &out_args, const std::string &fused_out_arg,
    const proto::VarType::Type &dtype, BlockDesc *global_block, bool copy_data,
    bool check_name) const {
  auto op_desc = global_block->AppendOp();
  op_desc->SetType("coalesce_tensor");
  op_desc->SetInput("Input", in_args);
  op_desc->SetOutput("Output", out_args);
  op_desc->SetOutput("FusedOutput", {fused_out_arg});
  op_desc->SetAttr("copy_data", copy_data);
  op_desc->SetAttr("check_name", check_name);
  op_desc->SetAttr("dtype", static_cast<int>(dtype));
}

void FuseOptimizerOpPass::InsertInputAndOutputForFusedOpNode(
    const std::vector<ir::Node *> &op_nodes, ir::Graph *graph,
    ir::Node *fused_opt_node) const {
  std::unordered_set<ir::Node *> inputs;
  std::unordered_set<ir::Node *> outputs;
  for (auto opt_op : op_nodes) {
    inputs.insert(opt_op->inputs.begin(), opt_op->inputs.end());
    for (auto &input : opt_op->inputs) {
      replace(input->outputs.begin(), input->outputs.end(), opt_op,
              fused_opt_node);
    }
    outputs.insert(opt_op->outputs.begin(), opt_op->outputs.end());
    for (auto &output : opt_op->outputs) {
      replace(output->inputs.begin(), output->inputs.end(), opt_op,
              fused_opt_node);
    }
  }

  // Remove the dependence vars between op_nodes.
  std::unordered_set<ir::Node *> out_dep_vars;
  std::unordered_set<ir::Node *> not_useful_vars;

  auto deal_with_ctrl_vars = [&out_dep_vars, &not_useful_vars,
                              &fused_opt_node](ir::Node *ctr_var_node) {
    PADDLE_ENFORCE_EQ(ctr_var_node->inputs.size(), 1);
    if (ctr_var_node->inputs.front() == fused_opt_node) {
      PADDLE_ENFORCE_GT(ctr_var_node->outputs.size(), 0);
      auto output_ops = ctr_var_node->outputs;
      output_ops.erase(std::remove_if(output_ops.begin(), output_ops.end(),
                                      [&fused_opt_node](const ir::Node *node) {
                                        return node == fused_opt_node;
                                      }),
                       output_ops.end());
      if (!output_ops.empty()) {
        out_dep_vars.insert(ctr_var_node);
      }
      not_useful_vars.insert(ctr_var_node);
    }
  };

  for (auto *in_node : inputs) {
    if (in_node->IsCtrlVar()) {
      deal_with_ctrl_vars(in_node);
    }
  }

  for (auto *out_node : outputs) {
    if (out_node->IsCtrlVar()) {
      deal_with_ctrl_vars(out_node);
    }
  }

  for (auto &node : not_useful_vars) {
    if (inputs.count(node)) {
      inputs.erase(node);
    }
    if (outputs.count(node)) {
      outputs.erase(node);
    }
  }

  for (auto &dep_var : out_dep_vars) {
    if (not_useful_vars.count(dep_var)) {
      not_useful_vars.erase(dep_var);
    }
    dep_var->inputs.clear();
    dep_var->inputs.emplace_back(fused_opt_node);
  }

  outputs.insert(out_dep_vars.begin(), out_dep_vars.end());
  fused_opt_node->inputs.insert(fused_opt_node->inputs.begin(), inputs.begin(),
                                inputs.end());
  fused_opt_node->outputs.insert(fused_opt_node->outputs.begin(),
                                 outputs.begin(), outputs.end());

  for (auto &ctrl_var_node : not_useful_vars) {
    graph->RemoveNode(ctrl_var_node);
  }
}
}  // namespace ir
}  // namespace framework
}  // namespace paddle
