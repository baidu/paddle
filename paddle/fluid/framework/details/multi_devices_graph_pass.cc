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
#include <algorithm>
#include <fstream>
#include <string>
#include <utility>
#include <vector>

#include "paddle/fluid/framework/details/all_reduce_op_handle.h"
#include "paddle/fluid/framework/details/broadcast_op_handle.h"
#include "paddle/fluid/framework/details/computation_op_handle.h"
#include "paddle/fluid/framework/details/data_balance_op_handle.h"
#include "paddle/fluid/framework/details/multi_devices_graph_pass.h"
#include "paddle/fluid/framework/details/reduce_op_handle.h"
#include "paddle/fluid/framework/details/rpc_op_handle.h"
#include "paddle/fluid/framework/details/scale_loss_grad_op_handle.h"
#include "paddle/fluid/framework/ir/graph_helper.h"
#include "paddle/fluid/framework/ir/node.h"
#include "paddle/fluid/framework/op_info.h"
#include "paddle/fluid/framework/scope.h"

namespace paddle {
namespace framework {
namespace details {
namespace {
void PolishGraphToSupportDataHazards(ir::Graph *graph) {
  for (auto &var_map : graph->Get<GraphVars>(kGraphVars)) {
    for (auto &name_pair : var_map) {
      if (name_pair.second.size() <= 1) {
        continue;
      }
      auto it_new = name_pair.second.rbegin();
      auto it_old = name_pair.second.rbegin();
      ++it_old;
      for (; it_old != name_pair.second.rend(); it_new = it_old, ++it_old) {
        OpHandleBase *write_op = (*it_new)->GeneratedOp();
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

          auto *dep_var = new DummyVarHandle(graph->CreateControlDepVar());
          read_op->AddOutput(dep_var);
          write_op->AddInput(dep_var);
          graph->Get<GraphDepVars>(kGraphDepVars).emplace(dep_var);
        }
      }
    }
  }
}

VarHandle *CreateOrGetLatestVarHandle(ir::Graph *graph, ir::Node *node,
                                      const platform::Place &place,
                                      size_t place_offset) {
  auto &var_holders = graph->Get<GraphVars>(kGraphVars)[place_offset];
  auto &var_holder = var_holders[node->Name()];
  VarHandle *var = nullptr;
  if (var_holder.empty()) {
    if (node->Var()) {
      var = new VarHandle(graph->CreateVarNode(node->Var()), 0, place_offset,
                          node->Name(), place);
    } else {
      var = new VarHandle(
          graph->CreateEmptyNode(node->Name(), ir::Node::Type::kVariable), 0,
          place_offset, node->Name(), place);
    }
    var_holder.emplace_back(var);
  } else {
    var = var_holder.rbegin()->get();
  }
  return var;
}

void CreateOpOutput(ir::Graph *graph, OpHandleBase *op_handle,
                    ir::Node *new_node, const platform::Place &place,
                    size_t place_offset) {
  auto &vars =
      graph->Get<GraphVars>(kGraphVars)[place_offset][new_node->Name()];
  size_t version = vars.size();
  auto var =
      new VarHandle(new_node, version, place_offset, new_node->Name(), place);
  vars.emplace_back(var);
  op_handle->AddOutput(var);
}

void AddOutputToLeafOps(ir::Graph *graph) {
  for (auto &op : graph->Get<GraphOps>(kGraphOps)) {
    if (!op->Outputs().empty()) {
      continue;
    }
    auto *dummy_leaf = new DummyVarHandle(graph->CreateControlDepVar());
    graph->Get<GraphDepVars>(kGraphDepVars).emplace(dummy_leaf);
    op->AddOutput(dummy_leaf);
  }
}
}  // namespace

static const char kLossVarName[] = "loss_var_name";
static const char kPlaces[] = "places";
static const char kParams[] = "params";
static const char kLocalScopes[] = "local_scopes";
static const char kStrategy[] = "strategy";

void MultiDevSSAGraphBuilder::Init() const {
  all_vars_.clear();
  balance_vars_.clear();

  loss_var_name_ = Get<const std::string>(kLossVarName);
  places_ = Get<const std::vector<platform::Place>>(kPlaces);
  local_scopes_ = Get<const std::vector<Scope *>>(kLocalScopes);
  strategy_ = Get<const BuildStrategy>(kStrategy);
#ifdef PADDLE_WITH_CUDA
  nccl_ctxs_ = &Get<platform::NCCLContextMap>("nccl_ctxs");
#endif

  for (auto &p : Get<const std::unordered_set<std::string>>(kParams)) {
    grad_names_.insert(GradVarName(p));
  }
  balance_vars_.resize(places_.size(), 0);
  if (strategy_.enable_data_balance_ && places_.size() == 1) {
    LOG(WARNING) << "It is no need to enable data balance when there is only "
                    "one place. enable_data_balance is set to False.";
    strategy_.enable_data_balance_ = false;
  }
}

void MultiDevSSAGraphBuilder::CreateOpHandleIOs(ir::Graph *result,
                                                ir::Node *node,
                                                size_t place_id) const {
  auto p = places_[place_id];
  auto *op_handle = result->Get<GraphOps>(kGraphOps).back().get();
  op_handle->SetDeviceContext(p,
                              platform::DeviceContextPool::Instance().Get(p));

  for (ir::Node *input : node->inputs) {
    VarHandle *var = CreateOrGetLatestVarHandle(result, input, p, place_id);
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

std::vector<std::string> MultiDevSSAGraphBuilder::FindDistTrainSendVars(
    const std::vector<ir::Node *> &nodes) const {
  std::vector<std::string> send_vars;
  // since parameters are all in block 0,
  // it's enough to only scan send ops in block 0
  for (auto &node : nodes) {
    OpDesc *op = node->Op();
    // TODO(Yancey1989): use a graceful method to find send op,
    // instead of the the hard code string
    if (op->Type() == "send") {
      auto op_vars = op->InputArgumentNames();
      send_vars.reserve(send_vars.size() +
                        std::distance(op_vars.begin(), op_vars.end()));
      send_vars.insert(send_vars.end(), op_vars.begin(), op_vars.end());
    }
  }
  return send_vars;
}

std::vector<std::string> MultiDevSSAGraphBuilder::FindDistTrainRecvVars(
    const std::vector<ir::Node *> &nodes) const {
  std::vector<std::string> recv_vars;
  for (auto &node : nodes) {
    OpDesc *op = node->Op();
    // TODO(Yancey1989): use a graceful method to find recv op,
    // instead of the hard code string
    if (op->Type() == "recv") {
      auto op_vars = op->OutputArgumentNames();
      recv_vars.reserve(recv_vars.size() +
                        std::distance(op_vars.begin(), op_vars.end()));
      recv_vars.insert(recv_vars.end(), op_vars.begin(), op_vars.end());
    }
  }
  return recv_vars;
}

size_t MultiDevSSAGraphBuilder::GetAppropriateDeviceID(
    const std::vector<std::string> &var_names) const {
  int64_t numel_sum = 0;
  for (auto var_name : var_names) {
    if (all_vars_.find(var_name) == all_vars_.end()) continue;
    auto var_desc = all_vars_.at(var_name);
    PADDLE_ENFORCE_NOT_NULL(var_desc);
    auto dim = framework::make_ddim(var_desc->GetShape());
    int64_t numel = framework::product(dim);
    PADDLE_ENFORCE_GT(numel, 0);
    numel_sum += numel;
  }

  auto smallest =
      std::min_element(std::begin(balance_vars_), std::end(balance_vars_));
  size_t dev_id =
      static_cast<size_t>(std::distance(std::begin(balance_vars_), smallest));
  balance_vars_[dev_id] += numel_sum;
  return dev_id;
}

// Topology sort the graph nodes from inputs to outputs.
// Since SSAGraphBuilder depends on forward/backward nodes to assign devices
// to parameter/gradients before optimizer ops, topo sort is insufficient. (
// some optimizer ops might not depend on any nodes), we manually move all
// optimizer nodes after last backward nodes.
// However, the assumption by SSAGraphBuilder should be relaxed in the future.
std::vector<ir::Node *> SortOpsAndDelayOptimizeOp(const ir::Graph &graph) {
  std::vector<ir::Node *> ret = ir::TopologySortOperations(graph);
  size_t last_backward = 0;
  for (size_t i = 0; i < ret.size(); ++i) {
    if (boost::get<int>(
            ret[i]->Op()->GetAttr(OpProtoAndCheckerMaker::OpRoleAttrName())) ==
        static_cast<int>(OpRole::kBackward)) {
      last_backward = i;
    }
  }

  std::vector<ir::Node *> optimize_ops;
  std::vector<ir::Node *> sorted_ret;
  for (size_t i = 0; i < ret.size(); ++i) {
    if (i < last_backward) {
      if (static_cast<bool>(boost::get<int>(ret[i]->Op()->GetAttr(
                                OpProtoAndCheckerMaker::OpRoleAttrName())) &
                            static_cast<int>(OpRole::kOptimize))) {
        optimize_ops.push_back(ret[i]);
      } else {
        sorted_ret.push_back(ret[i]);
      }
    } else if (i == last_backward) {
      sorted_ret.push_back(ret[i]);
      // Verify that no operations before optimize ops depends on optimize ops.
      std::unordered_set<ir::Node *> optimize_set(optimize_ops.begin(),
                                                  optimize_ops.end());
      for (ir::Node *n : sorted_ret) {
        for (ir::Node *in : n->inputs) {
          for (ir::Node *pre_n : in->inputs) {
            PADDLE_ENFORCE(optimize_set.find(pre_n) == optimize_set.end(),
                           "optimize operations cannot be depended by forward "
                           "or backward node %s -> %s",
                           pre_n->Name(), n->Name());
          }
        }
      }
      sorted_ret.insert(sorted_ret.end(), optimize_ops.begin(),
                        optimize_ops.end());
    } else {
      sorted_ret.push_back(ret[i]);
    }
  }
  return sorted_ret;
}

std::unique_ptr<ir::Graph> MultiDevSSAGraphBuilder::ApplyImpl(
    std::unique_ptr<ir::Graph> graph) const {
  Init();
  // Give the topology sort order and rebuild the graph structure.
  std::vector<ir::Node *> sorted_ops = SortOpsAndDelayOptimizeOp(*graph);
  auto nodes = graph->ReleaseNodes();
  ir::Graph &result = *graph;

  for (auto &node : nodes) {
    if (node->IsVar() && node->Var()) {
      all_vars_.emplace(node->Name(), node->Var());
    }
  }
  std::unordered_set<std::string> og_has_been_broadcast;

  // We cannot invoke resize. It is a bug of GCC 4.8
  result.Set(kGraphVars, new GraphVars(places_.size()));
  result.Set(kGraphDepVars, new GraphDepVars);
  result.Set(kGraphOps, new GraphOps);
  result.Set(kShardedVarDevice, new ShardedVarDevice);

  // find send/recv vars so that we can place the distributed training
  // related op in the place 0
  auto send_vars = FindDistTrainSendVars(sorted_ops);
  auto recv_vars = FindDistTrainRecvVars(sorted_ops);

  std::vector<std::unordered_set<std::string>> bcast_var_name_set;
  bcast_var_name_set.resize(places_.size());

  size_t cur_device_id = 0;
  bool is_forwarding = true;
  bool is_dist_train = false;

  for (ir::Node *node : sorted_ops) {
    if (boost::get<int>(
            node->Op()->GetAttr(OpProtoAndCheckerMaker::OpRoleAttrName())) ==
        static_cast<int>(OpRole::kRPC)) {
      int op_dev_id = CreateRPCOp(&result, node);
      PADDLE_ENFORCE(op_dev_id != -1,
                     "Can not schedule the RPC operator to the right place.");
      if (node->Op()->Type() == "recv") {
        auto recv_vars_attr =
            boost::get<std::vector<std::string>>(node->Op()->GetNullableAttr(
                OpProtoAndCheckerMaker::OpRoleVarAttrName()));
        PADDLE_ENFORCE(recv_vars_attr.size() == 2UL);  // [parameter, gradient]
        if (recv_vars_attr[0].find(".block") == std::string::npos) {
          bcast_var_name_set[op_dev_id].emplace(recv_vars_attr[0]);
        }
      }
      is_dist_train = true;
    } else if (boost::get<int>(node->Op()->GetAttr(
                   OpProtoAndCheckerMaker::OpRoleAttrName())) ==
               static_cast<int>(OpRole::kDist)) {
      int op_dev_id = CreateDistTrainOp(&result, node);
      if (node->Op()->Type() == "concat") {
        auto origin_param_name = node->Op()->OutputArgumentNames()[0];
        bcast_var_name_set[op_dev_id].emplace(origin_param_name);
      }
    } else if (IsScaleLossOp(node)) {
      // user can customize loss@grad if not use_default_grad_scale_
      if (strategy_.gradient_scale_ !=
          BuildStrategy::GradientScaleStrategy::kCustomized) {
        // TODO(paddle-dev): Why is there no input for this op_handle?
        auto loss_grad_name = node->Op()->OutputArgumentNames()[0];
        CreateScaleLossGradOp(&result, loss_grad_name);
      }
      // This assumes the backward generating code will ensure IsScaleLossOp
      // is true only for the op that scale the final scalar loss.
      // It also assumes backward op will always follow the forward op in
      // the block.
      is_forwarding = false;
    } else {
      int op_dev_id = GetOpDeviceID(result, node);
      if (op_dev_id != -1) {  // This op only runs on one specific device.
        CreateComputationalOp(&result, node, op_dev_id);
        for (ir::Node *n : node->outputs) {
          graph->Get<ShardedVarDevice>(kShardedVarDevice)
              .emplace(n->Name(), op_dev_id);
        }
      } else {
        // This op runs on all devices, and its output may have parameter's
        // gradients.
        // TODO(paddle-dev): Why is so special about "read" op?
        if (node->Op()->Type() == "read" && strategy_.enable_data_balance_) {
          node->Op()->SetAttr("throw_eof_exp", false);
          CreateComputationalOps(&result, node, places_.size());
          const auto &data_var_names = node->Op()->Output("Out");
          InsertDataBalanceOp(&result, data_var_names);
        } else {
          CreateComputationalOps(&result, node, places_.size());
        }

        if (!is_forwarding && places_.size() > 1) {
          // Currently, we assume that once gradient is generated, it can be
          // broadcast, and each gradient is only broadcast once.
          if (static_cast<bool>(boost::get<int>(node->Op()->GetAttr(
                                    OpProtoAndCheckerMaker::OpRoleAttrName())) &
                                static_cast<int>(OpRole::kBackward))) {
            try {
              auto backward_vars = boost::get<std::vector<std::string>>(
                  node->Op()->GetNullableAttr(
                      OpProtoAndCheckerMaker::OpRoleVarAttrName()));

              PADDLE_ENFORCE_EQ(backward_vars.size() % 2, 0);

              for (size_t i = 0; i < backward_vars.size(); i += 2) {
                auto &p_name = backward_vars[i];
                auto &g_name = backward_vars[i + 1];
                VLOG(10) << "Bcast " << g_name << " for parameter " << p_name;

                switch (strategy_.reduce_) {
                  case BuildStrategy::ReduceStrategy::kReduce:
                    cur_device_id = GetAppropriateDeviceID({g_name});
                    CreateReduceOp(&result, g_name, cur_device_id);
                    graph->Get<ShardedVarDevice>(kShardedVarDevice)
                        .emplace(g_name, cur_device_id);
                    if (!is_dist_train) {
                      bcast_var_name_set[cur_device_id].emplace(p_name);
                    }
                    break;
                  case BuildStrategy::ReduceStrategy::kAllReduce:
                    if (IsSparseGradient(g_name)) {
                      CreateReduceOp(&result, g_name, 0);
                      CreateBroadcastOp(&result, g_name, 0);
                    } else {
                      InsertAllReduceOp(&result, g_name);
                    }
                    break;
                  default:
                    LOG(FATAL) << "Unknown reduce strategy ";
                    break;
                }
              }
            } catch (boost::bad_get e) {
            }
          }
        }
      }
    }
  }
  bool use_gpu = false;
#ifdef PADDLE_WITH_CUDA
  use_gpu = nccl_ctxs_ != nullptr;
#endif

  // Insert broadcast operators principle:
  // 1. Broadcast optimized parameters in Reduce strategy;
  // 2. No need broadcast optimized parameters in AllReduce strategy because of
  //    the optimization sub-graph would be run on every GPU;
  // 3. Allways broadcast received parameters in Distribute Training.
  if ((use_gpu &&
       strategy_.reduce_ == BuildStrategy::ReduceStrategy::kReduce) ||
      is_dist_train) {
    for (size_t dev_id = 0; dev_id < bcast_var_name_set.size(); ++dev_id) {
      auto &to_bcast_set = bcast_var_name_set[dev_id];
      for (auto &bcast_name : to_bcast_set) {
        CreateBroadcastOp(&result, bcast_name, dev_id);
      }
    }
  }
  /*
  Dependency graph has been constructed. However, there are still data
  hazards need to be handled.
 */
  PolishGraphToSupportDataHazards(&result);

  /*
   * Only variables should be the leaves of graph.
   */
  AddOutputToLeafOps(&result);
  PADDLE_ENFORCE(!ir::HasCircle(result));
  return graph;
}

bool MultiDevSSAGraphBuilder::IsSparseGradient(const std::string &og) const {
  PADDLE_ENFORCE(all_vars_.count(og) != 0);
  if (all_vars_.at(og)->GetType() == proto::VarType::SELECTED_ROWS) {
    return true;
  }
  return false;
}

void MultiDevSSAGraphBuilder::SetCommunicationContext(
    OpHandleBase *op_handle, const platform::Place &p) const {
#ifdef PADDLE_WITH_CUDA
  if (nccl_ctxs_ == nullptr) {
    op_handle->SetDeviceContext(p,
                                platform::DeviceContextPool::Instance().Get(p));
  }
#else
  op_handle->SetDeviceContext(p,
                              platform::DeviceContextPool::Instance().Get(p));
#endif
}

void MultiDevSSAGraphBuilder::CreateBroadcastOp(ir::Graph *result,
                                                const std::string &p_name,
                                                size_t src_dev_id) const {
#ifdef PADDLE_WITH_CUDA
  auto *op_handle = new BroadcastOpHandle(
      result->CreateEmptyNode("broadcast", ir::Node::Type::kOperation),
      local_scopes_, places_, nccl_ctxs_);
#else
  auto *op_handle = new BroadcastOpHandle(
      result->CreateEmptyNode("broadcast", ir::Node::Type::kOperation),
      local_scopes_, places_);
#endif
  result->Get<GraphOps>(kGraphOps).emplace_back(op_handle);

  auto *in =
      result->Get<GraphVars>(kGraphVars).at(src_dev_id).at(p_name).back().get();
  op_handle->AddInput(in);

  for (size_t i = 0; i < places_.size(); ++i) {
    auto &p = places_[i];
    SetCommunicationContext(op_handle, p);
    auto &vars = result->Get<GraphVars>(kGraphVars).at(i).at(p_name);
    auto *out_var = new VarHandle(
        result->CreateEmptyNode(p_name, ir::Node::Type::kVariable), vars.size(),
        i, p_name, p);
    vars.emplace_back(out_var);
    op_handle->AddOutput(out_var);
  }
}

void MultiDevSSAGraphBuilder::CreateComputationalOp(ir::Graph *result,
                                                    ir::Node *node,
                                                    int dev_id) const {
  result->Get<GraphOps>(kGraphOps).emplace_back(
      new ComputationOpHandle(result->CreateOpNode(node->Op()),
                              local_scopes_[dev_id], places_[dev_id]));
  CreateOpHandleIOs(result, node, dev_id);
}

void MultiDevSSAGraphBuilder::InsertAllReduceOp(ir::Graph *result,
                                                const std::string &og) const {
#ifdef PADDLE_WITH_CUDA
  result->Get<GraphOps>(kGraphOps).emplace_back(new AllReduceOpHandle(
      result->CreateEmptyNode("allreduce", ir::Node::Type::kOperation),
      local_scopes_, places_, nccl_ctxs_));
#else
  result->Get<GraphOps>(kGraphOps).emplace_back(new AllReduceOpHandle(
      result->CreateEmptyNode("allreduce", ir::Node::Type::kOperation),
      local_scopes_, places_));
#endif
  auto *op_handle = result->Get<GraphOps>(kGraphOps).back().get();

  for (size_t i = 0; i < places_.size(); ++i) {
    auto &p = places_[i];
    SetCommunicationContext(op_handle, p);
    auto &vars = result->Get<GraphVars>(kGraphVars)[i][og];
    PADDLE_ENFORCE(!vars.empty());
    auto &prev_grad = vars.back();
    op_handle->AddInput(prev_grad.get());

    auto var =
        new VarHandle(result->CreateEmptyNode(og, ir::Node::Type::kVariable),
                      vars.size(), i, og, p);
    vars.emplace_back(var);
    op_handle->AddOutput(var);
  }
}

void MultiDevSSAGraphBuilder::InsertDataBalanceOp(
    ir::Graph *result, const std::vector<std::string> &datas) const {
#ifdef PADDLE_WITH_CUDA
  result->Get<GraphOps>(kGraphOps).emplace_back(new DataBalanceOpHandle(
      result->CreateEmptyNode("data_balance", ir::Node::Type::kOperation),
      local_scopes_, places_, nccl_ctxs_));
#else
  result->Get<GraphOps>(kGraphOps).emplace_back(new DataBalanceOpHandle(
      result->CreateEmptyNode("data_balance", ir::Node::Type::kOperation),
      local_scopes_, places_));
#endif
  auto *op_handle = result->Get<GraphOps>(kGraphOps).back().get();
  for (size_t i = 0; i < places_.size(); ++i) {
    auto &p = places_[i];
    SetCommunicationContext(op_handle, p);
    for (const std::string &d_name : datas) {
      auto &vars = result->Get<GraphVars>(kGraphVars)[i][d_name];
      PADDLE_ENFORCE(!vars.empty());
      op_handle->AddInput(vars.back().get());
      auto var = new VarHandle(
          result->CreateEmptyNode(d_name, ir::Node::Type::kVariable),
          vars.size(), i, d_name, p);
      vars.emplace_back(var);
      op_handle->AddOutput(var);
    }
  }
}

int MultiDevSSAGraphBuilder::GetOpDeviceID(const ir::Graph &graph,
                                           ir::Node *node) const {
  if (strategy_.reduce_ != BuildStrategy::ReduceStrategy::kReduce) {
    return -1;
  }
  int op_role = boost::get<int>(
      node->Op()->GetAttr(framework::OpProtoAndCheckerMaker::OpRoleAttrName()));
  if (op_role != static_cast<int>(framework::OpRole::kOptimize)) {
    return -1;
  }
  auto param_grad = boost::get<std::vector<std::string>>(
      node->Op()->GetAttr(OpProtoAndCheckerMaker::OpRoleVarAttrName()));

  PADDLE_ENFORCE_EQ(param_grad.size(), 2U);
  int dev_id = GetVarDeviceID(graph, param_grad[1]);
  PADDLE_ENFORCE_NE(dev_id, -1, "dev_id should not be -1.[%s, %s, %s]",
                    node->Op()->Type(), param_grad[0], param_grad[1]);
  return dev_id;
}

int MultiDevSSAGraphBuilder::GetVarDeviceID(const ir::Graph &graph,
                                            const std::string &varname) const {
  auto &sharded_var_device = graph.Get<ShardedVarDevice>(kShardedVarDevice);
  auto got = sharded_var_device.find(varname);
  return got == sharded_var_device.end() ? -1 : got->second;
}

void MultiDevSSAGraphBuilder::CreateScaleLossGradOp(
    ir::Graph *result, const std::string &loss_grad_name) const {
  for (size_t i = 0; i < places_.size(); ++i) {
    // Insert ScaleCost OpHandle
    auto *dev_ctx = platform::DeviceContextPool::Instance().Get(places_[i]);
    auto *op_handle = new ScaleLossGradOpHandle(
        result->CreateEmptyNode("scale_loss_grad", ir::Node::Type::kOperation),
        local_scopes_.size(), local_scopes_[i], places_[i], dev_ctx);
    result->Get<GraphOps>(kGraphOps).emplace_back(op_handle);

    // FIXME: Currently ScaleLossGradOp only use device_count as scale
    // factor. So it does not depend on any other operators.
    // VarHandle *loss = GetVarHandle(loss_var_name, place);
    // loss->pending_ops_.emplace_back(op_handle);
    // op_handle->inputs_.emplace_back(loss);

    CreateOpOutput(
        result, op_handle,
        result->CreateEmptyNode(loss_grad_name, ir::Node::Type::kVariable),
        places_[i], i);
  }
}

void MultiDevSSAGraphBuilder::CreateComputationalOps(ir::Graph *result,
                                                     ir::Node *node,
                                                     size_t num_places) const {
  for (size_t scope_idx = 0; scope_idx < num_places; ++scope_idx) {
    auto p = places_[scope_idx];
    auto s = local_scopes_[scope_idx];
    result->Get<GraphOps>(kGraphOps).emplace_back(
        new ComputationOpHandle(result->CreateOpNode(node->Op()), s, p));
    CreateOpHandleIOs(result, node, scope_idx);
  }
}

VarHandle *MultiDevSSAGraphBuilder::CreateReduceOp(ir::Graph *result,
                                                   const std::string &og,
                                                   int dst_dev_id) const {
#ifdef PADDLE_WITH_CUDA
  result->Get<GraphOps>(kGraphOps).emplace_back(new ReduceOpHandle(
      result->CreateEmptyNode("reduce", ir::Node::Type::kOperation),
      local_scopes_, places_, nccl_ctxs_));
#else
  result->Get<GraphOps>(kGraphOps).emplace_back(new ReduceOpHandle(
      result->CreateEmptyNode("reduce", ir::Node::Type::kOperation),
      local_scopes_, places_));
#endif
  auto *op_handle = result->Get<GraphOps>(kGraphOps).back().get();

  for (size_t i = 0; i < places_.size(); ++i) {
    auto &p = places_[i];
    SetCommunicationContext(op_handle, p);
    auto &vars = result->Get<GraphVars>(kGraphVars)[i][og];
    PADDLE_ENFORCE(!vars.empty());
    auto &prev_grad = vars.back();
    op_handle->AddInput(prev_grad.get());
  }
  auto &vars = result->Get<GraphVars>(kGraphVars)[dst_dev_id][og];
  auto var =
      new VarHandle(result->CreateEmptyNode(og, ir::Node::Type::kVariable),
                    vars.size(), dst_dev_id, og, places_[dst_dev_id]);
  vars.emplace_back(var);
  op_handle->AddOutput(var);
  return var;
}

int MultiDevSSAGraphBuilder::CreateDistTrainOp(ir::Graph *result,
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
    op_dev_id = GetVarDeviceID(*result, input_var_names[0]);
    if (strategy_.reduce_ == BuildStrategy::ReduceStrategy::kAllReduce) {
      op_dev_id = GetAppropriateDeviceID(input_var_names);
      for (auto &varname : input_var_names) {
        result->Get<ShardedVarDevice>(kShardedVarDevice)
            .emplace(varname, op_dev_id);
      }
    }
    for (auto &varname : output_var_names) {
      result->Get<ShardedVarDevice>(kShardedVarDevice)
          .emplace(varname, op_dev_id);
    }
  } else if (node->Op()->Type() == "concat") {
    op_dev_id = GetVarDeviceID(*result, input_var_names[0]);
    for (auto &varname : output_var_names) {
      result->Get<ShardedVarDevice>(kShardedVarDevice)
          .emplace(varname, op_dev_id);
    }
  } else {
    LOG(ERROR) << "got unexpected dist op: " << node->Op()->Type();
    PADDLE_THROW(
        "the distribute training related op should be in [split_byref, "
        "concat].");
  }

  PADDLE_ENFORCE(op_dev_id != -1,
                 "can not find right place for distributed op: %s",
                 node->Op()->Type());

  CreateComputationalOp(result, node, op_dev_id);
  return op_dev_id;
}

void SetOpInputsAllPlaces(ir::Graph *result, ir::Node *node, int num_places) {
  auto *op_handle = result->Get<GraphOps>(kGraphOps).back().get();
  for (ir::Node *input : node->inputs) {
    VarHandle *var = nullptr;
    for (int place_offset = 0; place_offset < num_places; ++place_offset) {
      auto &var_holders = result->Get<GraphVars>(kGraphVars)[place_offset];
      auto &var_holder = var_holders[input->Name()];
      if (!var_holder.empty()) {
        var = var_holder.rbegin()->get();
        op_handle->AddInput(var);
      }
    }
  }
}

// Create RPC related op handles that connects its in ops and out ops.
int MultiDevSSAGraphBuilder::CreateRPCOp(ir::Graph *result,
                                         ir::Node *node) const {
  int op_dev_id = -1;
  if (node->Op()->Type() == "send") {
    // TODO(paddle-dev): getting the first var is not safe.
    op_dev_id = GetVarDeviceID(*result, node->inputs[0]->Name());
    PADDLE_ENFORCE(!ir::IsControlDepVar(*node->inputs[0]),
                   "This hack no longer holds, please fix.");
    // the variable name which contains .block means it was splited by
    // split_byref op
    if (strategy_.reduce_ == BuildStrategy::ReduceStrategy::kAllReduce &&
        node->inputs[0]->Name().find(".block") == std::string::npos) {
      std::vector<std::string> input_var_names;
      for (ir::Node *n : node->inputs) {
        input_var_names.push_back(n->Name());
      }
      auto send_param_grad = boost::get<std::vector<std::string>>(
          node->Op()->GetAttr(OpProtoAndCheckerMaker::OpRoleVarAttrName()));
      PADDLE_ENFORCE_EQ(send_param_grad.size(), 2U);
      op_dev_id = GetAppropriateDeviceID({send_param_grad[1]});
      VLOG(10) << "send grad " << input_var_names[0] << " origin "
               << send_param_grad[1] << " place: " << op_dev_id;
      for (auto &varname : input_var_names) {
        result->Get<ShardedVarDevice>(kShardedVarDevice)
            .emplace(varname, op_dev_id);
      }
      result->Get<ShardedVarDevice>(kShardedVarDevice)
          .emplace(send_param_grad[1], op_dev_id);
    }
  } else if (node->Op()->Type() == "recv") {
    std::vector<std::string> output_var_names;
    for (ir::Node *n : node->outputs) {
      output_var_names.push_back(n->Name());
    }
    auto recv_param_grad = boost::get<std::vector<std::string>>(
        node->Op()->GetAttr(OpProtoAndCheckerMaker::OpRoleVarAttrName()));
    if (recv_param_grad.size() == 2U) {
      op_dev_id = GetVarDeviceID(*result, recv_param_grad[1]);
      VLOG(10) << "recv param " << recv_param_grad[0]
               << " get grad place: " << recv_param_grad[1]
               << " place: " << op_dev_id;
    } else {
      op_dev_id = GetAppropriateDeviceID(output_var_names);
    }
    for (auto &varname : output_var_names) {
      result->Get<ShardedVarDevice>(kShardedVarDevice)
          .emplace(varname, op_dev_id);
    }
  } else {
    // send_barrier, fetch_barrier will run on place 0;
    op_dev_id = 0;
  }

  PADDLE_ENFORCE(op_dev_id != -1, "can not find the right place for rpc op: %s",
                 node->Op()->Type());
  result->Get<GraphOps>(kGraphOps).emplace_back(new RPCOpHandle(
      result->CreateOpNode(node->Op()), *node->Op(), local_scopes_[op_dev_id],
      node->Op()->Type(), places_[op_dev_id]));

  if (node->Op()->Type() == "send") {
    CreateOpHandleIOs(result, node, op_dev_id);
  } else {
    // send_barrier, recv, fetch_barrier's inputs are deps var, get them from
    // all places
    auto p = places_[op_dev_id];
    auto *op_handle = result->Get<GraphOps>(kGraphOps).back().get();
    op_handle->SetDeviceContext(p,
                                platform::DeviceContextPool::Instance().Get(p));

    SetOpInputsAllPlaces(result, node, places_.size());
    for (ir::Node *output : node->outputs) {
      int outvar_dev_id = op_dev_id;
      if (node->Op()->Type() == "fetch_barrier") {
        outvar_dev_id = GetVarDeviceID(*result, output->Name());
        PADDLE_ENFORCE_NE(outvar_dev_id, -1);
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

bool MultiDevSSAGraphBuilder::IsScaleLossOp(ir::Node *node) const {
  return boost::get<int>(
             node->Op()->GetAttr(OpProtoAndCheckerMaker::OpRoleAttrName())) ==
             (static_cast<int>(OpRole::kBackward) |
              static_cast<int>(OpRole::kLoss)) &&
         !loss_var_name_.empty();  // If loss_var is empty. This is test mode
}
}  // namespace details
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(multi_devices_pass,
              paddle::framework::details::MultiDevSSAGraphBuilder)
    .RequirePassAttr(paddle::framework::details::kLossVarName)
    .RequirePassAttr(paddle::framework::details::kPlaces)
    .RequirePassAttr(paddle::framework::details::kParams)
    .RequirePassAttr(paddle::framework::details::kLocalScopes)
    .RequirePassAttr(paddle::framework::details::kStrategy);
