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
#include "paddle/fluid/framework/ir/multi_devices_graph_pass/multi_devices_graph_pass.h"
#include <algorithm>
#include <fstream>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>
#include "paddle/fluid/framework/details/all_reduce_op_handle.h"
#include "paddle/fluid/framework/details/broadcast_op_handle.h"
#include "paddle/fluid/framework/details/computation_op_handle.h"
#include "paddle/fluid/framework/details/fetch_barrier_op_handle.h"
#include "paddle/fluid/framework/details/fused_broadcast_op_handle.h"
#include "paddle/fluid/framework/details/reduce_op_handle.h"
#include "paddle/fluid/framework/details/rpc_op_handle.h"
#include "paddle/fluid/framework/details/scale_loss_grad_op_handle.h"
#include "paddle/fluid/framework/ir/graph_helper.h"
#include "paddle/fluid/framework/ir/node.h"
#include "paddle/fluid/framework/op_info.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/operators/math/math_function.h"

#if defined(PADDLE_WITH_DGC)
#include "paddle/fluid/framework/details/sparse_all_reduce_op_handle.h"
#endif

namespace paddle {
namespace framework {
namespace ir {

namespace {
// TODO(panyx0718): Clean this up as well.
// all operators. NOTE that even we use a vector here, the operators is
// unordered.
typedef std::vector<details::OpHandleBase *> GraphOps;
const char kGraphOps[] = "ops";

bool OpHaveRole(const ir::Node &node, const framework::OpRole &role) {
  return BOOST_GET_CONST(int, node.Op()->GetAttr(
                                  OpProtoAndCheckerMaker::OpRoleAttrName())) ==
         static_cast<int>(role);
}

void PolishGraphToSupportDataHazards(ir::Graph *graph) {
  for (auto &var_map : graph->Get<details::GraphVars>(details::kGraphVars)) {
    for (auto &name_pair : var_map) {
      if (name_pair.second.size() <= 1) {
        continue;
      }
      auto it_new = name_pair.second.rbegin();
      auto it_old = name_pair.second.rbegin();
      ++it_old;
      for (; it_old != name_pair.second.rend(); it_new = it_old, ++it_old) {
        details::OpHandleBase *write_op = (*it_new)->GeneratedOp();
        const auto &read_ops = (*it_old)->PendingOps();

        for (auto *read_op : read_ops) {
          // Manually add a dependency var from read_op to write_op;
          if (read_op == write_op) {
            // Read Write is the same op.
            continue;
          }
          bool has_dep = false;
          for (auto *r_out : read_op->Outputs()) {
            for (auto *w_in : write_op->Inputs()) {
              if (r_out->Node() == w_in->Node()) {
                has_dep = true;
                break;
              }
            }
          }
          if (has_dep) continue;

          auto *dep_var =
              new details::DummyVarHandle(graph->CreateControlDepVar());
          read_op->AddOutput(dep_var);
          write_op->AddInput(dep_var);
          graph->Get<details::GraphDepVars>(details::kGraphDepVars)
              .emplace(dep_var);
        }
      }
    }
  }
}

details::VarHandle *CreateOrGetLatestVarHandle(ir::Graph *graph, ir::Node *node,
                                               const platform::Place &place,
                                               size_t place_offset) {
  auto &var_holders =
      graph->Get<details::GraphVars>(details::kGraphVars)[place_offset];
  auto &var_holder = var_holders[node->Name()];
  details::VarHandle *var = nullptr;
  if (var_holder.empty()) {
    if (node->Var()) {
      var = new details::VarHandle(graph->CreateVarNode(node->Var()), 0,
                                   place_offset, node->Name(), place);
    } else {
      var = new details::VarHandle(
          graph->CreateEmptyNode(node->Name(), ir::Node::Type::kVariable), 0,
          place_offset, node->Name(), place);
    }
    var_holder.emplace_back(var);
  } else {
    var = *var_holder.rbegin();
  }
  return var;
}

void CreateOpOutput(ir::Graph *graph, details::OpHandleBase *op_handle,
                    ir::Node *new_node, const platform::Place &place,
                    size_t place_offset) {
  auto &vars = graph->Get<details::GraphVars>(
      details::kGraphVars)[place_offset][new_node->Name()];
  size_t version = vars.size();
  auto var = new details::VarHandle(new_node, version, place_offset,
                                    new_node->Name(), place);
  vars.emplace_back(var);
  op_handle->AddOutput(var);
}

void AddOutputToLeafOps(ir::Graph *graph) {
  for (auto &op : graph->Get<GraphOps>(kGraphOps)) {
    if (!op->Outputs().empty()) {
      continue;
    }
    auto *dummy_leaf =
        new details::DummyVarHandle(graph->CreateControlDepVar());
    graph->Get<details::GraphDepVars>(details::kGraphDepVars)
        .emplace(dummy_leaf);
    op->AddOutput(dummy_leaf);
  }
}
}  // namespace

void MultiDevSSAGraphBuilderBase::CheckGraph(const ir::Graph &graph) const {}

void MultiDevSSAGraphBuilderBase::Init() const {
  all_vars_.clear();

  loss_var_name_ = Get<const std::string>(kLossVarName);
  VLOG(10) << "Init MultiDevSSAGraphBuilder, loss name: " << loss_var_name_;
  places_ = Get<const std::vector<platform::Place>>(details::kPlaces);
  local_scopes_ = Get<const std::vector<Scope *>>(details::kLocalScopes);
  strategy_ = Get<const details::BuildStrategy>(kStrategy);
#if defined(PADDLE_WITH_NCCL)
  multi_nccl_ctxs_ = &Get<platform::NCCLCommunicator>(details::kNCCLCtxs);
  nccl_ctxs_ = nullptr;
  if (multi_nccl_ctxs_) {
    nccl_ctxs_ = multi_nccl_ctxs_->DefaultFlatCtx();
  }
#elif defined(PADDLE_WITH_XPU_BKCL)
  multi_bkcl_ctxs_ = &Get<platform::BKCLCommunicator>(details::kBKCLCtxs);
  bkcl_ctxs_ = nullptr;
  if (multi_bkcl_ctxs_) {
    bkcl_ctxs_ = multi_bkcl_ctxs_->DefaultFlatCtx();
  }
#endif
  PADDLE_ENFORCE_EQ(
      places_.size(), local_scopes_.size(),
      platform::errors::InvalidArgument(
          "Places size and LocalScopes not equal "
          "Places size(%d), LocalScopes size(%d) "
          "If use multi devices， Places size must equas to LocalScopes size.",
          places_.size(), local_scopes_.size()));
}

void MultiDevSSAGraphBuilderBase::ApplyImpl(ir::Graph *graph) const {
  Init();
  CheckGraph(*graph);
  std::vector<ir::Node *> sorted_ops = SortOperations(*graph);

  auto nodes = graph->ReleaseNodes();
  ir::Graph &result = *graph;

  std::vector<ir::Node *> isolated_vars;

  for (auto &node : nodes) {
    if (node->IsVar() && node->Var()) {
      all_vars_.emplace(node->Name(), node->Var());

      if (node->inputs.empty() && node->outputs.empty()) {
        isolated_vars.emplace_back(node.get());
      }
    }
  }

  // We cannot invoke resize. It is a bug of GCC 4.8
  result.Set(details::kGraphVars, new details::GraphVars(places_.size()));
  result.Set(details::kGraphDepVars, new details::GraphDepVars);
  result.Set(kGraphOps, new GraphOps);

  for (auto *var_node : isolated_vars) {
    CreateIsolatedVarNode(&result, var_node);
  }

  bool is_forwarding = true;

  for (ir::Node *node : sorted_ops) {
    if (DealWithSpecialOp(&result, node)) {
      continue;
    } else {
      // This op runs on all devices
      if (IsScaleLossOp(node)) {
        // user can customize loss@grad if not use_default_grad_scale_
        InsertScaleLossGradOp(&result, node);
        // This assumes the backward generating code will ensure IsScaleLossOp
        // is true only for the op that scale the final scalar loss.
        // It also assumes backward op will always follow the forward op in
        // the block.
        is_forwarding = false;
      } else {
        CreateComputationalOps(&result, node, places_.size());
      }

      // Insert collective ops if nranks > 1
      if (!is_forwarding && Get<size_t>(details::kNRanks) > 1) {
        auto &op_desc = *(node->Op());
        bool is_bk_op = details::IsOpRole(op_desc, OpRole::kBackward);
        // optimize op is already processed in DealWithSpecialOp,
        // here we only consider backward op
        if (!is_bk_op) continue;

        /*
         * the op that will generate the gradient of on parameter will have
         one attr op_role_var
         * to record the parameter and gradient, like:
          attrs {
            name: "op_role_var"
            type: STRINGS
            strings: "fc_1.b_0"
            strings: "fc_1.b_0@GRAD"
          }
         */

        // Currently, we assume that once gradient is generated, it can be
        // broadcast, and each gradient is only broadcast once.
        auto backward_vars = details::GetOpRoleVarsOrEmpty(op_desc);
        for (size_t i = 0; i < backward_vars.size(); i += 2) {
          auto &p_name = backward_vars[i];
          auto &g_name = backward_vars[i + 1];
          VLOG(10) << "Bcast " << g_name << " for parameter " << p_name
                   << " op_type " << node->Op()->Type();
          if (NeedCollectiveForGrad(g_name, sorted_ops)) {
            InsertCollectiveOp(&result, p_name, g_name);
          }
        }
      }
    }
  }

  InsertPostprocessOps(&result);

  /*
  Dependency graph has been constructed. However, there are still data
  hazards need to be handled.
  */
  PolishGraphToSupportDataHazards(&result);

  /*
   * Only variables should be the leaves of graph.
   */
  AddOutputToLeafOps(&result);

  result.Erase(kGraphOps);
}

void MultiDevSSAGraphBuilderBase::InsertScaleLossGradOp(
    ir::Graph *result, const ir::Node *node) const {
  // user can customize loss@grad if not use_default_grad_scale_
  size_t loss_scale = 0;
  switch (this->strategy_.gradient_scale_) {
    case details::BuildStrategy::GradientScaleStrategy::kOne:
      loss_scale = 1;
      break;
    case details::BuildStrategy::GradientScaleStrategy::kCoeffNumDevice:
      loss_scale = Get<size_t>(details::kNRanks);
      break;
    case details::BuildStrategy::GradientScaleStrategy::kCustomized:
      loss_scale = 0;
      break;
    default:
      PADDLE_THROW(platform::errors::Unimplemented(
          "Unknown gradient scale strategy. Now only supports One, "
          "CoeffNumDevice and Customized strategies."));
      break;
  }

  VLOG(3) << "loss_scale: " << loss_scale;

  if (loss_scale) {
    // TODO(paddle-dev): Why is there no input for this op_handle?
    auto loss_grad_name = node->Op()->OutputArgumentNames()[0];
    auto out_dtype = this->all_vars_.at(loss_grad_name)->GetDataType();
    this->CreateScaleLossGradOp(result, loss_grad_name, node->outputs[0],
                                loss_scale, out_dtype);
  }
}

bool MultiDevSSAGraphBuilderBase::DealWithSpecialOp(ir::Graph *result,
                                                    ir::Node *node) const {
  return false;
}

std::vector<ir::Node *> MultiDevSSAGraphBuilderBase::SortOperations(
    const ir::Graph &graph) const {
  return ir::TopologySortOperations(graph);
}

bool MultiDevSSAGraphBuilderBase::UseGPU() const {
  bool use_gpu = false;
#if defined(PADDLE_WITH_NCCL)
  use_gpu = nccl_ctxs_ != nullptr;
#endif
  return use_gpu;
}

bool MultiDevSSAGraphBuilderBase::NeedCollectiveForGrad(
    const std::string &grad_name, std::vector<ir::Node *> ops) const {
  // if we have allreduce_op for current gradient variable in the graph,
  // then we don't need to add allreduce_op_handle for this gradient
  // NOTE: This is for the case that all gradients should add collective ops
  for (auto *node : ops) {
    if (node->Op()->Type() != "allreduce") continue;
    for (auto in_name : node->Op()->InputArgumentNames()) {
      if (in_name == grad_name) {
        return false;
      }
    }
  }
  return true;
}

void MultiDevSSAGraphBuilderBase::CreateOpHandleIOs(ir::Graph *result,
                                                    ir::Node *node,
                                                    size_t place_id) const {
  auto p = places_[place_id];
  auto *op_handle = result->Get<GraphOps>(kGraphOps).back();
  op_handle->SetDeviceContext(p,
                              platform::DeviceContextPool::Instance().Get(p));

  for (ir::Node *input : node->inputs) {
    details::VarHandle *var =
        CreateOrGetLatestVarHandle(result, input, p, place_id);
    op_handle->AddInput(var);
  }

  for (ir::Node *output : node->outputs) {
    ir::Node *new_node = nullptr;
    if (output->Var()) {
      new_node = result->CreateVarNode(output->Var());
    } else {
      new_node =
          result->CreateEmptyNode(output->Name(), ir::Node::Type::kVariable);
    }
    CreateOpOutput(result, op_handle, new_node, p, place_id);
  }
}

void MultiDevSSAGraphBuilderBase::SetCommunicationContext(
    details::OpHandleBase *op_handle, const platform::Place &p) const {
#if defined(PADDLE_WITH_NCCL)
  if (nccl_ctxs_ == nullptr) {
    op_handle->SetDeviceContext(p,
                                platform::DeviceContextPool::Instance().Get(p));
  }
#elif defined(PADDLE_WITH_XPU_BKCL)
  if (bkcl_ctxs_ == nullptr) {
    op_handle->SetDeviceContext(p,
                                platform::DeviceContextPool::Instance().Get(p));
  }
#else
  op_handle->SetDeviceContext(p,
                              platform::DeviceContextPool::Instance().Get(p));
#endif
}

void MultiDevSSAGraphBuilderBase::CreateBroadcastOp(ir::Graph *result,
                                                    const std::string &p_name,
                                                    size_t src_dev_id) const {
#if defined(PADDLE_WITH_NCCL)
  auto *op_handle = new details::BroadcastOpHandle(
      result->CreateEmptyNode("broadcast", ir::Node::Type::kOperation),
      local_scopes_, places_, nccl_ctxs_);
#elif defined(PADDLE_WITH_XPU_BKCL)
  auto *op_handle = new details::BroadcastOpHandle(
      result->CreateEmptyNode("broadcast", ir::Node::Type::kOperation),
      local_scopes_, places_, bkcl_ctxs_);
#else
  auto *op_handle = new details::BroadcastOpHandle(
      result->CreateEmptyNode("broadcast", ir::Node::Type::kOperation),
      local_scopes_, places_);
#endif
  result->Get<GraphOps>(kGraphOps).emplace_back(op_handle);

  auto *in = result->Get<details::GraphVars>(details::kGraphVars)
                 .at(src_dev_id)
                 .at(p_name)
                 .back();
  op_handle->AddInput(in);

  for (size_t i = 0; i < places_.size(); ++i) {
    auto &p = places_[i];
    SetCommunicationContext(op_handle, p);
    auto &vars =
        result->Get<details::GraphVars>(details::kGraphVars).at(i).at(p_name);
    auto *out_var = new details::VarHandle(
        result->CreateEmptyNode(p_name, ir::Node::Type::kVariable), vars.size(),
        i, p_name, p);
    vars.emplace_back(out_var);
    op_handle->AddOutput(out_var);
  }
}

void MultiDevSSAGraphBuilderBase::CreateFusedBroadcastOp(
    ir::Graph *result,
    const std::vector<std::unordered_set<std::string>> &bcast_varnames) const {
#if defined(PADDLE_WITH_NCCL)
  auto *op_handle = new details::FusedBroadcastOpHandle(
      result->CreateEmptyNode("fused_broadcast", ir::Node::Type::kOperation),
      local_scopes_, places_, nccl_ctxs_);
#elif defined(PADDLE_WITH_XPU_BKCL)
  auto *op_handle = new details::FusedBroadcastOpHandle(
      result->CreateEmptyNode("fused_broadcast", ir::Node::Type::kOperation),
      local_scopes_, places_, bkcl_ctxs_);
#else
  auto *op_handle = new details::FusedBroadcastOpHandle(
      result->CreateEmptyNode("fused_broadcast", ir::Node::Type::kOperation),
      local_scopes_, places_);
#endif
  result->Get<GraphOps>(kGraphOps).emplace_back(op_handle);

  for (size_t i = 0; i < places_.size(); ++i) {
    auto &p = places_[i];
    SetCommunicationContext(op_handle, p);
  }

  for (size_t dev_id = 0; dev_id < bcast_varnames.size(); ++dev_id) {
    for (auto &p_name : bcast_varnames[dev_id]) {
      auto *in = result->Get<details::GraphVars>(details::kGraphVars)
                     .at(dev_id)
                     .at(p_name)
                     .back();
      op_handle->AddInput(in);
      for (size_t out_dev_id = 0; out_dev_id < places_.size(); ++out_dev_id) {
        auto &p = places_[out_dev_id];
        auto &vars = result->Get<details::GraphVars>(details::kGraphVars)
                         .at(out_dev_id)
                         .at(p_name);
        auto *out_var = new details::VarHandle(
            result->CreateEmptyNode(p_name, ir::Node::Type::kVariable),
            vars.size(), out_dev_id, p_name, p);
        vars.emplace_back(out_var);
        op_handle->AddOutput(out_var);
      }
    }
  }
}

void MultiDevSSAGraphBuilderBase::CreateComputationalOp(ir::Graph *result,
                                                        ir::Node *node,
                                                        size_t dev_id) const {
  result->Get<GraphOps>(kGraphOps).emplace_back(
      new details::ComputationOpHandle(result->CreateOpNode(node->Op()),
                                       local_scopes_[dev_id], places_[dev_id],
                                       dev_id));
  CreateOpHandleIOs(result, node, dev_id);
}

void MultiDevSSAGraphBuilderBase::CreateAllReduceOp(ir::Graph *result,
                                                    const std::string &og,
                                                    bool is_encoded) const {
  details::OpHandleBase *op_handle = nullptr;

  auto append_allreduce_op = [&](
      const std::vector<Scope *> &scopes,
      const std::vector<platform::Place> &places) -> details::OpHandleBase * {
#if defined(PADDLE_WITH_DGC) && defined(PADDLE_WITH_NCCL)
    if (is_encoded) {
      result->Get<GraphOps>(kGraphOps).emplace_back(
          new details::SparseAllReduceOpHandle(
              result->CreateEmptyNode("allreduce", ir::Node::Type::kOperation),
              scopes, places, multi_nccl_ctxs_, is_encoded,
              strategy_.num_trainers_ * places_.size()));
    } else {
      result->Get<GraphOps>(kGraphOps).emplace_back(
          new details::AllReduceOpHandle(
              result->CreateEmptyNode("allreduce", ir::Node::Type::kOperation),
              scopes, places, multi_nccl_ctxs_));
    }
#elif defined(PADDLE_WITH_NCCL)
    result->Get<GraphOps>(kGraphOps).emplace_back(
        new details::AllReduceOpHandle(
            result->CreateEmptyNode("allreduce", ir::Node::Type::kOperation),
            scopes, places, multi_nccl_ctxs_));
#elif defined(PADDLE_WITH_XPU_BKCL)
    result->Get<GraphOps>(kGraphOps).emplace_back(
        new details::AllReduceOpHandle(
            result->CreateEmptyNode("allreduce", ir::Node::Type::kOperation),
            scopes, places, multi_bkcl_ctxs_));
#else
    result->Get<GraphOps>(kGraphOps).emplace_back(
        new details::AllReduceOpHandle(
            result->CreateEmptyNode("allreduce", ir::Node::Type::kOperation),
            scopes, places));
#endif
    return result->Get<GraphOps>(kGraphOps).back();
  };

  if (!strategy_.enable_parallel_graph_)
    op_handle = append_allreduce_op(local_scopes_, places_);

  for (size_t i = 0; i < places_.size(); ++i) {
    if (strategy_.enable_parallel_graph_) {
      op_handle = append_allreduce_op({local_scopes_[i]}, {places_[i]});
    }

    SetCommunicationContext(op_handle, places_[i]);
    auto &vars = result->Get<details::GraphVars>(details::kGraphVars)[i][og];
    PADDLE_ENFORCE_EQ(vars.empty(), false,
                      platform::errors::InvalidArgument(
                          "Can not find Var(%s) in Place[%d] "
                          "Paddle Can not add AllReduce OP for Var(%s).",
                          og, i, og));
    auto &prev_grad = vars.back();
    op_handle->AddInput(prev_grad);
    VLOG(10) << "all_reduce_op_handle add input " << prev_grad->DebugString();

    auto var = new details::VarHandle(
        result->CreateEmptyNode(og, ir::Node::Type::kVariable), vars.size(), i,
        og, places_[i]);
    vars.emplace_back(var);
    op_handle->AddOutput(var);
    VLOG(10) << "all_reduce_op_handle add output " << og
             << ", handle:" << var->DebugString();
  }
}

void MultiDevSSAGraphBuilderBase::CreateScaleLossGradOp(
    ir::Graph *result, const std::string &loss_grad_name,
    ir::Node *out_var_node, size_t loss_scale,
    proto::VarType::Type dtype) const {
  for (size_t i = 0; i < places_.size(); ++i) {
    auto *dev_ctx = platform::DeviceContextPool::Instance().Get(places_[i]);
    auto *op_handle = new details::ScaleLossGradOpHandle(
        result->CreateEmptyNode("scale_loss_grad", ir::Node::Type::kOperation),
        loss_scale, local_scopes_[i], places_[i], dev_ctx, dtype);
    result->Get<GraphOps>(kGraphOps).emplace_back(op_handle);

    // FIXME: Currently ScaleLossGradOp only use device_count as scale
    // factor. So it does not depend on any other operators.
    // VarHandle *loss = GetVarHandle(loss_var_name, place);
    // loss->pending_ops_.emplace_back(op_handle);
    // op_handle->inputs_.emplace_back(loss);

    CreateOpOutput(result, op_handle,
                   result->CreateVarNode(out_var_node->Var()), places_[i], i);
  }
}

void MultiDevSSAGraphBuilderBase::CreateComputationalOps(
    ir::Graph *result, ir::Node *node, size_t num_places) const {
  for (size_t scope_idx = 0; scope_idx < num_places; ++scope_idx) {
    auto p = places_[scope_idx];
    auto s = local_scopes_[scope_idx];
    result->Get<GraphOps>(kGraphOps).emplace_back(
        new details::ComputationOpHandle(result->CreateOpNode(node->Op()), s, p,
                                         scope_idx));
    CreateOpHandleIOs(result, node, scope_idx);
  }
}

details::VarHandle *MultiDevSSAGraphBuilderBase::CreateReduceOp(
    ir::Graph *result, const std::string &og, size_t dst_dev_id) const {
#if defined(PADDLE_WITH_NCCL)
  result->Get<GraphOps>(kGraphOps).emplace_back(new details::ReduceOpHandle(
      result->CreateEmptyNode("reduce", ir::Node::Type::kOperation),
      local_scopes_, places_, nccl_ctxs_));
#elif defined(PADDLE_WITH_XPU_BKCL)
  result->Get<GraphOps>(kGraphOps).emplace_back(new details::ReduceOpHandle(
      result->CreateEmptyNode("reduce", ir::Node::Type::kOperation),
      local_scopes_, places_, bkcl_ctxs_));
#else
  result->Get<GraphOps>(kGraphOps).emplace_back(new details::ReduceOpHandle(
      result->CreateEmptyNode("reduce", ir::Node::Type::kOperation),
      local_scopes_, places_));
#endif
  auto *op_handle = result->Get<GraphOps>(kGraphOps).back();

  for (size_t i = 0; i < places_.size(); ++i) {
    auto &p = places_[i];
    SetCommunicationContext(op_handle, p);
    auto &vars = result->Get<details::GraphVars>(details::kGraphVars)[i][og];
    PADDLE_ENFORCE_EQ(vars.empty(), false,
                      platform::errors::InvalidArgument(
                          "Can not find Var(%s) in Place[%d] "
                          "Paddle Can not add Reduce OP for Var(%s).",
                          og, i, og));
    auto &prev_grad = vars.back();
    op_handle->AddInput(prev_grad);
  }
  auto &vars =
      result->Get<details::GraphVars>(details::kGraphVars)[dst_dev_id][og];
  auto var = new details::VarHandle(
      result->CreateEmptyNode(og, ir::Node::Type::kVariable), vars.size(),
      dst_dev_id, og, places_[dst_dev_id]);
  vars.emplace_back(var);
  op_handle->AddOutput(var);
  return var;
}

bool MultiDevSSAGraphBuilderBase::IsScaleLossOp(ir::Node *node) const {
  return !loss_var_name_.empty() && node->Op() &&
         BOOST_GET_CONST(int, node->Op()->GetAttr(
                                  OpProtoAndCheckerMaker::OpRoleAttrName())) ==
             (static_cast<int>(OpRole::kBackward) |
              static_cast<int>(OpRole::kLoss));
}

bool MultiDevSSAGraphBuilderBase::IsSparseGradient(
    const std::string &og) const {
  PADDLE_ENFORCE_NE(all_vars_.count(og), 0,
                    platform::errors::InvalidArgument(
                        "Can not find Var(%s) in VarDescs "
                        "Paddle Can not add Collective OP for Var(%s).",
                        og, og));
  return all_vars_.at(og)->GetType() == proto::VarType::SELECTED_ROWS;
}

void MultiDevSSAGraphBuilderBase::CreateIsolatedVarNode(
    ir::Graph *graph, ir::Node *var_node) const {
  for (size_t i = 0; i < places_.size(); ++i) {
    VLOG(10) << "Create isolated var node " << var_node->Name() << " at device "
             << i;
    CreateOrGetLatestVarHandle(graph, var_node, places_[i], i);
  }
}

void AllReduceSSAGraphBuilder::InsertCollectiveOp(
    ir::Graph *result, const std::string &p_name,
    const std::string &g_name) const {
  if (IsSparseGradient(g_name)) {
    CreateReduceOp(result, g_name, 0);
    CreateBroadcastOp(result, g_name, 0);
  } else {
#if defined(PADDLE_WITH_DGC)
    CreateAllReduceOp(result, g_name, IsEncoded(p_name));
#else
    CreateAllReduceOp(result, g_name);
#endif
  }
}

int BalanceVarSSAGraphBuilder::GetVarDeviceID(
    const std::string &varname) const {
  auto got = sharded_var_device_.find(varname);
  if (got == sharded_var_device_.end()) {
    auto pos = varname.find(framework::kNewGradSuffix);
    if (pos != std::string::npos) {
      got = sharded_var_device_.find(varname.substr(0, pos));
    }
  }
  return got == sharded_var_device_.end() ? -1 : got->second;
}

int BalanceVarSSAGraphBuilder::GetOpDeviceID(ir::Node *node) const {
  if (strategy_.reduce_ != details::BuildStrategy::ReduceStrategy::kReduce) {
    return -1;
  }
  if (!OpHaveRole(*node, framework::OpRole::kOptimize)) {
    return -1;
  }
  auto param_grad = BOOST_GET_CONST(
      std::vector<std::string>,
      node->Op()->GetAttr(OpProtoAndCheckerMaker::OpRoleVarAttrName()));

  PADDLE_ENFORCE_EQ(
      param_grad.size(), 2U,
      platform::errors::InvalidArgument(
          "In Node %s, the size of attribute %s must be 2, include Parameter "
          "and Parameter@Grad.",
          node->Name(), OpProtoAndCheckerMaker::OpRoleVarAttrName()));
  int dev_id = GetVarDeviceID(param_grad[1]);
  PADDLE_ENFORCE_NE(dev_id, -1, platform::errors::NotFound(
                                    "Can not find Device ID, for NodeName:%s, "
                                    "NodeType:%s, Param:%s, Param@Grad:%s"
                                    "For this fault, you can consult the "
                                    "Paddle technical personnel for answer ",
                                    node->Name(), node->Op()->Type(),
                                    param_grad[0], param_grad[1]));
  return dev_id;
}

size_t BalanceVarSSAGraphBuilder::GetAppropriateDeviceID(
    const std::vector<std::string> &var_names) const {
  int64_t numel_sum = 0;
  for (auto var_name : var_names) {
    if (all_vars_.find(var_name) == all_vars_.end()) continue;
    auto var_desc = all_vars_.at(var_name);
    PADDLE_ENFORCE_NOT_NULL(var_desc,
                            platform::errors::NotFound(
                                "Can not find Var(%s) in Var Desc.", var_name));
    auto dim = framework::make_ddim(var_desc->GetShape());
    int64_t numel = framework::product(dim);
    PADDLE_ENFORCE_GT(numel, 0,
                      platform::errors::InvalidArgument(
                          "The numel of Var(%s) must greater than 0"
                          "Please check your code，about Var(%s) Shape.",
                          var_name, var_name));
    numel_sum += numel;
  }

  auto smallest =
      std::min_element(std::begin(balance_vars_), std::end(balance_vars_));
  size_t dev_id =
      static_cast<size_t>(std::distance(std::begin(balance_vars_), smallest));
  balance_vars_[dev_id] += numel_sum;
  return dev_id;
}

void BalanceVarSSAGraphBuilder::ResetState() const {
  balance_vars_.clear();
  sharded_var_device_.clear();

  balance_vars_.resize(places_.size(), 0);
}

void ReduceSSAGraphBuilder::Init() const {
  MultiDevSSAGraphBuilderBase::Init();
  ResetState();
}

void ReduceSSAGraphBuilder::ResetState() const {
  BalanceVarSSAGraphBuilder::ResetState();
  bcast_var_name_set_.clear();
  bcast_var_name_set_.resize(places_.size());
}

void ReduceSSAGraphBuilder::InsertCollectiveOp(
    ir::Graph *result, const std::string &p_name,
    const std::string &g_name) const {
  size_t cur_device_id = GetAppropriateDeviceID({g_name});
  CreateReduceOp(result, g_name, cur_device_id);
  sharded_var_device_.emplace(g_name, cur_device_id);
  bcast_var_name_set_[cur_device_id].emplace(p_name);
}

bool ReduceSSAGraphBuilder::DealWithSpecialOp(ir::Graph *result,
                                              ir::Node *node) const {
  int op_dev_id = BalanceVarSSAGraphBuilder::GetOpDeviceID(node);
  if (op_dev_id != -1) {
    // This op only runs on one specific device.
    CreateComputationalOp(result, node, op_dev_id);
    for (ir::Node *n : node->outputs) {
      sharded_var_device_.emplace(n->Name(), op_dev_id);
    }
    return true;
  }
  return false;
}

void ReduceSSAGraphBuilder::InsertPostprocessOps(ir::Graph *result) const {
  if (UseGPU()) {
    if (strategy_.fuse_broadcast_ops_ == true) {
      CreateFusedBroadcastOp(result, bcast_var_name_set_);
    } else {
      for (size_t dev_id = 0; dev_id < bcast_var_name_set_.size(); ++dev_id) {
        auto &to_bcast_set = bcast_var_name_set_[dev_id];
        for (auto &bcast_name : to_bcast_set) {
          CreateBroadcastOp(result, bcast_name, dev_id);
        }
      }
    }
  }
}

int ReduceSSAGraphBuilder::GetOpDeviceID(
    ir::Node *node,
    std::unordered_map<std::string, std::vector<ir::Node *>> *delay_ops) const {
  if (!OpHaveRole(*node, framework::OpRole::kOptimize)) {
    return -1;
  }

  auto param_grad = BOOST_GET_CONST(
      std::vector<std::string>,
      node->Op()->GetAttr(OpProtoAndCheckerMaker::OpRoleVarAttrName()));

  PADDLE_ENFORCE_EQ(
      param_grad.size(), 2U,
      platform::errors::InvalidArgument(
          "In Node %s, The size of attribute %s must be 2, include Parameter "
          "and Parameter@Grad.",
          node->Name(), OpProtoAndCheckerMaker::OpRoleVarAttrName()));
  int dev_id = GetVarDeviceID(param_grad[1]);

  if (dev_id == -1) {
    (*delay_ops)[param_grad[1]].push_back(node);
    return -2;
  }
  return dev_id;
}

std::vector<ir::Node *> ReduceSSAGraphBuilder::SortOperations(
    const ir::Graph &graph) const {
  std::vector<ir::Node *> sorted_ops = ir::TopologySortOperations(graph);
  return SortForReduceMode(sorted_ops);
}

std::vector<ir::Node *> ReduceSSAGraphBuilder::SortForReduceMode(
    const std::vector<ir::Node *> &topo_ops) const {
  std::vector<ir::Node *> sorted_ops;
  std::unordered_map<std::string, std::vector<ir::Node *>> delayed_op;
  sorted_ops.reserve(topo_ops.size());
  ResetState();

  auto insert_delayed_op = [&](const std::string &var_name, int dev_id) {
    sharded_var_device_.emplace(var_name, dev_id);
    if (delayed_op.count(var_name)) {
      auto &ops = delayed_op.at(var_name);
      sorted_ops.insert(sorted_ops.end(), ops.begin(), ops.end());
      delayed_op.at(var_name).clear();
    }
  };

  for (ir::Node *node : topo_ops) {
    int op_dev_id = GetOpDeviceID(node, &delayed_op);
    if (op_dev_id > -1) {
      // This op only runs on one specific device.
      sorted_ops.emplace_back(node);
      for (ir::Node *n : node->outputs) {
        insert_delayed_op(n->Name(), op_dev_id);
      }
    } else if (op_dev_id == -1) {
      // This op runs on all devices, and its output may have parameter's
      // gradients.
      sorted_ops.emplace_back(node);
      bool is_bk_op = static_cast<bool>(
          BOOST_GET_CONST(int, node->Op()->GetAttr(
                                   OpProtoAndCheckerMaker::OpRoleAttrName())) &
          static_cast<int>(OpRole::kBackward));
      if (!is_bk_op) continue;
      // Currently, we assume that once gradient is generated, it can be
      // broadcast, and each gradient is only broadcast once.
      auto backward_vars = details::GetOpRoleVarsOrEmpty(*(node->Op()));
      for (size_t i = 0; i < backward_vars.size(); i += 2) {
        auto &g_name = backward_vars[i + 1];
        size_t cur_device_id = GetAppropriateDeviceID({g_name});
        insert_delayed_op(g_name, static_cast<int>(cur_device_id));
      }
    } else if (op_dev_id == -2) {
      // The Op on which the Op depends has not yet been generated.
    }
  }

  PADDLE_ENFORCE_EQ(sorted_ops.size(), topo_ops.size(),
                    platform::errors::InvalidArgument(
                        "Sorted ops calc error!"
                        "The result for sorted ops size(%d) must be "
                        "equal to topo ops size(%d).",
                        sorted_ops.size(), topo_ops.size()));

  ResetState();
  return sorted_ops;
}

void DistSSAGraphBuilder::Init() const {
  MultiDevSSAGraphBuilderBase::Init();
  ResetState();
}

void DistSSAGraphBuilder::ResetState() const {
  BalanceVarSSAGraphBuilder::ResetState();
  bcast_var_name_set_.clear();
  bcast_var_name_set_.resize(places_.size());
}

bool DistSSAGraphBuilder::DealWithSpecialOp(ir::Graph *result,
                                            ir::Node *node) const {
  bool insert_op = false;
  if (OpHaveRole(*node, OpRole::kRPC)) {
    int op_dev_id = CreateRPCOp(result, node);
    PADDLE_ENFORCE_NE(op_dev_id, -1, platform::errors::InvalidArgument(
                                         "Can not schedule the RPC operator to "
                                         "the right place. NodeName:%s.",
                                         node->Name()));
    if (node->Op()->Type() == "recv") {
      auto recv_vars_attr =
          BOOST_GET_CONST(std::vector<std::string>,
                          node->Op()->GetNullableAttr(
                              OpProtoAndCheckerMaker::OpRoleVarAttrName()));
      PADDLE_ENFORCE_EQ(
          recv_vars_attr.size(), 2UL,
          platform::errors::InvalidArgument(
              "In Node %s, the size of attribute %s must be 2, include "
              "Parameter and Parameter@Grad.",
              node->Name(),
              OpProtoAndCheckerMaker::OpRoleVarAttrName()));  // [parameter,
                                                              // gradient]
      if (recv_vars_attr[0].find(".block") == std::string::npos) {
        bcast_var_name_set_[op_dev_id].emplace(recv_vars_attr[0]);
      }
    }
    insert_op = true;
    need_broadcast_var_ = true;
  } else if (OpHaveRole(*node, OpRole::kDist)) {
    int op_dev_id = CreateDistTrainOp(result, node);
    if (node->Op()->Type() == "concat") {
      // the input(block of parameter) of concat is on different device,
      // the output(parameter) will on one device.
      auto origin_param_name = node->Op()->OutputArgumentNames()[0];
      bcast_var_name_set_[op_dev_id].emplace(origin_param_name);
    }
    insert_op = true;
  } else {
    int op_dev_id = GetOpDeviceID(node);
    if (op_dev_id != -1) {  // This op only runs on one specific device.
      // optimize op will be processed here.
      CreateComputationalOp(result, node, op_dev_id);
      for (ir::Node *n : node->outputs) {
        sharded_var_device_.emplace(n->Name(), op_dev_id);
      }
      insert_op = true;
    }
  }
  return insert_op;
}

void SetOpInputsAllPlaces(ir::Graph *result, ir::Node *node, int num_places) {
  auto *op_handle = result->Get<GraphOps>(kGraphOps).back();
  for (ir::Node *input : node->inputs) {
    details::VarHandle *var = nullptr;
    for (int place_offset = 0; place_offset < num_places; ++place_offset) {
      auto &var_holders =
          result->Get<details::GraphVars>(details::kGraphVars)[place_offset];
      auto &var_holder = var_holders[input->Name()];
      if (!var_holder.empty()) {
        var = *var_holder.rbegin();
        op_handle->AddInput(var);
      }
    }
  }
}

// Create RPC related op handles that connects its in ops and out ops.
int DistSSAGraphBuilder::CreateRPCOp(ir::Graph *result, ir::Node *node) const {
  int op_dev_id = -1;
  if (node->Op()->Type() == "send") {
    // TODO(paddle-dev): getting the first var is not safe.
    op_dev_id = GetVarDeviceID(node->inputs[0]->Name());
    PADDLE_ENFORCE_EQ(ir::IsControlDepVar(*node->inputs[0]), false,
                      platform::errors::InvalidArgument(
                          "This hack no longer holds, please fix."));
    // the variable name which contains .block means it was split by
    // split_byref op
    if (strategy_.reduce_ ==
            details::BuildStrategy::ReduceStrategy::kAllReduce &&
        node->inputs[0]->Name().find(".block") == std::string::npos) {
      std::vector<std::string> input_var_names;
      for (ir::Node *n : node->inputs) {
        input_var_names.push_back(n->Name());
      }
      auto send_param_grad = BOOST_GET_CONST(
          std::vector<std::string>,
          node->Op()->GetAttr(OpProtoAndCheckerMaker::OpRoleVarAttrName()));
      PADDLE_ENFORCE_EQ(
          send_param_grad.size(), 2U,
          platform::errors::InvalidArgument(
              "In Node %s, the size of attribute %s must be 2, include "
              "Parameter and Parameter@Grad.",
              node->Name(), OpProtoAndCheckerMaker::OpRoleVarAttrName()));
      op_dev_id = GetAppropriateDeviceID({send_param_grad[1]});
      VLOG(10) << "send grad " << input_var_names[0] << " origin "
               << send_param_grad[1] << " place: " << op_dev_id;
      for (auto &varname : input_var_names) {
        sharded_var_device_.emplace(varname, op_dev_id);
      }
      sharded_var_device_.emplace(send_param_grad[1], op_dev_id);
    }
  } else if (node->Op()->Type() == "recv") {
    std::vector<std::string> output_var_names;
    for (ir::Node *n : node->outputs) {
      output_var_names.push_back(n->Name());
    }
    auto recv_param_grad = BOOST_GET_CONST(
        std::vector<std::string>,
        node->Op()->GetAttr(OpProtoAndCheckerMaker::OpRoleVarAttrName()));
    if (recv_param_grad.size() == 2U) {
      op_dev_id = GetVarDeviceID(recv_param_grad[1]);
      VLOG(10) << "recv param " << recv_param_grad[0]
               << " get grad place: " << recv_param_grad[1]
               << " place: " << op_dev_id;
    } else {
      op_dev_id = GetAppropriateDeviceID(output_var_names);
    }
    for (auto &varname : output_var_names) {
      sharded_var_device_.emplace(varname, op_dev_id);
    }
  } else {
    // send_barrier, fetch_barrier will run on place 0;
    op_dev_id = 0;
  }

  PADDLE_ENFORCE_NE(
      op_dev_id, -1,
      platform::errors::NotFound("Can not find the right place for rpc op: %s.",
                                 node->Op()->Type()));
  // Create fetch_barrier op handle to enable output on all devices.
  // **NOTE** fetch_barrier should output variables list same as recv op does.
  if (node->Op()->Type() == "fetch_barrier") {
    result->Get<GraphOps>(kGraphOps).emplace_back(
        new details::FetchBarrierOpHandle(result->CreateOpNode(node->Op()),
                                          local_scopes_, places_));
  } else {
    result->Get<GraphOps>(kGraphOps).emplace_back(new details::RPCOpHandle(
        result->CreateOpNode(node->Op()), *node->Op(), local_scopes_[op_dev_id],
        node->Op()->Type(), places_[op_dev_id]));
  }

  if (node->Op()->Type() == "send") {
    CreateOpHandleIOs(result, node, op_dev_id);
  } else {
    // send_barrier, recv, fetch_barrier's inputs are deps var, get them from
    // all places
    auto p = places_[op_dev_id];
    auto *op_handle = result->Get<GraphOps>(kGraphOps).back();
    op_handle->SetDeviceContext(p,
                                platform::DeviceContextPool::Instance().Get(p));

    SetOpInputsAllPlaces(result, node, places_.size());
    for (ir::Node *output : node->outputs) {
      int outvar_dev_id = op_dev_id;
      if (node->Op()->Type() == "fetch_barrier") {
        outvar_dev_id = GetVarDeviceID(output->Name());
        PADDLE_ENFORCE_NE(outvar_dev_id, -1,
                          platform::errors::NotFound(
                              "Can not find the right place for the var: %s.",
                              output->Name()));
      }
      p = places_[outvar_dev_id];
      ir::Node *new_node = nullptr;
      if (output->Var()) {
        new_node = result->CreateVarNode(output->Var());
      } else {
        new_node =
            result->CreateEmptyNode(output->Name(), ir::Node::Type::kVariable);
      }
      CreateOpOutput(result, op_handle, new_node, p, outvar_dev_id);
    }
  }
  return op_dev_id;
}

int DistSSAGraphBuilder::CreateDistTrainOp(ir::Graph *result,
                                           ir::Node *node) const {
  int op_dev_id = -1;
  std::vector<std::string> input_var_names;
  std::vector<std::string> output_var_names;
  for (ir::Node *input : node->inputs) {
    input_var_names.push_back(input->Name());
  }
  for (ir::Node *output : node->outputs) {
    output_var_names.push_back(output->Name());
  }

  if (node->Op()->Type() == "split_byref" ||
      node->Op()->Type() == "split_selected_rows" ||
      node->Op()->Type() == "split_ids") {
    // TODO(paddle-dev): getting the first var is not safe.
    op_dev_id = GetVarDeviceID(input_var_names[0]);
    if (strategy_.reduce_ ==
        details::BuildStrategy::ReduceStrategy::kAllReduce) {
      op_dev_id = GetAppropriateDeviceID(input_var_names);
      for (auto &varname : input_var_names) {
        sharded_var_device_.emplace(varname, op_dev_id);
      }
    }
    for (auto &varname : output_var_names) {
      sharded_var_device_.emplace(varname, op_dev_id);
    }
  } else if (node->Op()->Type() == "concat") {
    op_dev_id = GetVarDeviceID(input_var_names[0]);
    for (auto &varname : output_var_names) {
      sharded_var_device_.emplace(varname, op_dev_id);
    }
  } else {
    LOG(ERROR) << "got unexpected dist op: " << node->Op()->Type();
    PADDLE_THROW(
        platform::errors::Unimplemented("The distribute training related op "
                                        "should be in [split_byref, concat]."));
  }

  PADDLE_ENFORCE_NE(op_dev_id, -1,
                    platform::errors::NotFound(
                        "Can not find right place for distributed op: %s.",
                        node->Op()->Type()));

  CreateComputationalOp(result, node, op_dev_id);
  return op_dev_id;
}

#if defined(PADDLE_WITH_DGC)
bool AllReduceSSAGraphBuilder::IsEncoded(const std::string &p_name) const {
  auto k_name = p_name + details::g_dgc_k;
  auto it = all_vars_.find(k_name);
  if (it == all_vars_.end()) {
    VLOG(10) << "can't find k_name, so it's not encoded:" << k_name;
    return false;
  }

  return true;
}
#else
bool AllReduceSSAGraphBuilder::IsEncoded(const std::string &p_name) const {
  return false;
}
#endif

void DistSSAGraphBuilder::InsertCollectiveOp(ir::Graph *result,
                                             const std::string &p_name,
                                             const std::string &g_name) const {
  // collective gradient to each device
  size_t cur_device_id = 0;
  switch (strategy_.reduce_) {
    case details::BuildStrategy::ReduceStrategy::kReduce:
      cur_device_id = GetAppropriateDeviceID({g_name});
      CreateReduceOp(result, g_name, cur_device_id);
      sharded_var_device_.emplace(g_name, cur_device_id);
      break;
    case details::BuildStrategy::ReduceStrategy::kAllReduce:
      if (IsSparseGradient(g_name)) {
        CreateReduceOp(result, g_name, 0);
        CreateBroadcastOp(result, g_name, 0);
      } else {
        CreateAllReduceOp(result, g_name);
      }
      break;
    default:
      PADDLE_THROW(platform::errors::Unimplemented(
          "Unknown reduce strategy. Now only supports Reduce and AllReduce "
          "strategies."));
      break;
  }
}

void DistSSAGraphBuilder::InsertPostprocessOps(ir::Graph *result) const {
  // broad cast received parameters when training in parameter server mode.
  if (need_broadcast_var_) {
    // There are 4 conditions:
    // 1. GPU && Reduce: Reduce gradient then broadcast gradient to other GPUS.
    // Need to broadcast received parameters to other GPU.
    // 2. GPU && AllReduce: AllReduce all gradient to each GPU. Need to
    // broadcast received parameters to other GPU.
    // 3. CPU && AllReduce: AllReduce all gradient to each thread. Need to
    // broadcast received parameters to other scope.
    // 4. CPU && Reduce: because all parameters share the same memory, did not
    // broadcast received parameters.
    if (!UseGPU() &&
        strategy_.reduce_ == details::BuildStrategy::ReduceStrategy::kReduce) {
      return;
    }
    if (strategy_.fuse_broadcast_ops_ == true) {
      CreateFusedBroadcastOp(result, bcast_var_name_set_);
    } else {
      for (size_t dev_id = 0; dev_id < bcast_var_name_set_.size(); ++dev_id) {
        auto &to_bcast_set = bcast_var_name_set_[dev_id];
        for (auto &bcast_name : to_bcast_set) {
          CreateBroadcastOp(result, bcast_name, dev_id);
        }
      }
    }
  }
}

std::unordered_set<std::string> &MultiDevSSAGraphBuilder() {
  static std::unordered_set<std::string> regs;
  return regs;
}

static int MultiDevSSAGraphBuilderRegister(const std::string &builder_mode) {
  MultiDevSSAGraphBuilder().insert(builder_mode);
  return 0;
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

#define REGISTER_MULTI_DEVICES_PASS(pass_name, pass_class)                \
  STATIC_ASSERT_GLOBAL_NAMESPACE(                                         \
      _reg_ssa_graph_builder_##pass_name,                                 \
      "REGISTER_MULTI_DEVICES_PASS must be called in global namespace."); \
  int _reg_ssa_graph_builder_entry_##pass_name =                          \
      paddle::framework::ir::MultiDevSSAGraphBuilderRegister(#pass_name); \
  REGISTER_PASS(pass_name, pass_class)                                    \
      .RequirePassAttr(paddle::framework::ir::kLossVarName)               \
      .RequirePassAttr(paddle::framework::details::kPlaces)               \
      .RequirePassAttr(paddle::framework::details::kLocalScopes)          \
      .RequirePassAttr(paddle::framework::ir::kStrategy)                  \
      .RequirePassAttr(paddle::framework::details::kNRanks)

REGISTER_MULTI_DEVICES_PASS(reduce_mode_multi_devices_pass,
                            paddle::framework::ir::ReduceSSAGraphBuilder);
REGISTER_MULTI_DEVICES_PASS(all_reduce_mode_multi_devices_pass,
                            paddle::framework::ir::AllReduceSSAGraphBuilder);
REGISTER_MULTI_DEVICES_PASS(dist_multi_devices_pass,
                            paddle::framework::ir::DistSSAGraphBuilder);
REGISTER_MULTI_DEVICES_PASS(async_multi_devices_pass,
                            paddle::framework::ir::AsyncSSAGraphBuilder);
