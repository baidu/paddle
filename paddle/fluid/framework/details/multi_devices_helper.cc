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
#include "paddle/fluid/framework/details/multi_devices_helper.h"

#include "paddle/fluid/framework/details/computation_op_handle.h"
#include "paddle/fluid/framework/details/eager_deletion_op_handle.h"
#include "paddle/fluid/framework/details/share_tensor_buffer_op_handle.h"
#include "paddle/fluid/framework/ir/graph_helper.h"

namespace paddle {
namespace framework {
namespace details {

static constexpr size_t kUndefinedDevIdx = -1UL;

// NOTE(paddle-dev): the following ops are related to multi-device
// communication. If the graph contains any of the following ops,
// it cannot separate into multiple graphs on each device.
static std::unordered_set<std::string> kMultiDeviceOps{
    "sync_batch_norm",
    "sync_batch_norm_grad",
    "allreduce",
    "c_allreduce_sum",
    "c_allreduce_prod",
    "c_allreduce_min",
    "c_allreduce_max",
    "c_allgather",
    "c_reducescatter",
    "c_broadcast",
    "c_comm_init",
    "c_comm_init_all",
    "c_comm_init_multitrainer",
    "c_gen_nccl_id",
    "c_sync_comm_stream",
    "send",
    "recv",
    "send_barrier",
    "fetch_barrier",
};

static size_t GetScopeIdxFromOp(const details::OpHandleBase &op) {
  if (auto *compute_op =
          dynamic_cast<const details::ComputationOpHandle *>(&op)) {
    return kMultiDeviceOps.count(compute_op->GetOp()->Type()) == 0
               ? compute_op->GetScopeIdx()
               : kUndefinedDevIdx;
  } else if (auto *gc_op =
                 dynamic_cast<const details::EagerDeletionOpHandle *>(&op)) {
    return gc_op->GetScopeIdx();
  } else if (auto *share_op =
                 dynamic_cast<const details::ShareTensorBufferOpHandle *>(
                     &op)) {
    return share_op->GetScopeIdx();
  } else {
    return kUndefinedDevIdx;
  }
}

static bool ContainMultiDeviceOp(const ProgramDesc &program,
                                 size_t begin_block_idx) {
  for (size_t block_idx = begin_block_idx; block_idx < program.Size();
       ++block_idx) {
    for (auto *op_desc : program.Block(block_idx).AllOps()) {
      if (kMultiDeviceOps.count(op_desc->Type()) > 0) {
        return true;
      }
    }
  }
  return false;
}

static size_t GetUniqueDeviceIdOfOp(const details::OpHandleBase &op) {
  size_t dev_idx = GetScopeIdxFromOp(op);
  if (dev_idx == kUndefinedDevIdx) {
    return kUndefinedDevIdx;
  }

  const auto &ins = op.Inputs();
  const auto &outs = op.Outputs();
  auto in_outs = ins;
  in_outs.insert(in_outs.end(), outs.begin(), outs.end());

  for (auto *var : in_outs) {
    auto *var_handle = dynamic_cast<details::VarHandle *>(var);
    if (var_handle == nullptr) {
      continue;
    }

    if (dev_idx != var_handle->scope_idx()) {
      return kUndefinedDevIdx;
    }
  }

  return dev_idx;
}

static bool IsDataParallelInferenceGraphImpl(
    const ir::Graph &graph,
    std::unordered_map<details::OpHandleBase *, size_t> *p_op_to_dev_idx,
    size_t *p_place_num) {
  auto &place_num = *p_place_num;
  auto &op_to_dev_idx = *p_op_to_dev_idx;

  auto clear_result = [&] {
    place_num = 0;
    op_to_dev_idx.clear();
    return false;
  };

  clear_result();

  // If sub-block contains multi-devices ops, we cannot separate
  if (ContainMultiDeviceOp(graph.OriginProgram(), 1)) {
    return clear_result();
  }

  auto op_handles = ir::FilterByNodeWrapper<OpHandleBase>(graph);
  if (op_handles.empty()) {
    return clear_result();
  }

  for (auto &op : op_handles) {
    auto dev_idx = GetUniqueDeviceIdOfOp(*op);
    if (dev_idx == kUndefinedDevIdx) {
      VLOG(10) << "Op " << op->Name() << " is not determined";
      return clear_result();
    }
    place_num = std::max(place_num, dev_idx + 1);
    op_to_dev_idx[op] = dev_idx;
  }

  for (auto &op : op_handles) {
    auto dev_idx = op_to_dev_idx.at(op);
    for (auto &in_var : op->Inputs()) {
      if (in_var->GeneratedOp()) {
        auto iter = op_to_dev_idx.find(in_var->GeneratedOp());
        if (iter == op_to_dev_idx.end() || iter->second != dev_idx) {
          return clear_result();
        }
      }
    }

    for (auto &out_var : op->Outputs()) {
      for (auto &pending_op : out_var->PendingOps()) {
        auto iter = op_to_dev_idx.find(pending_op);
        if (iter == op_to_dev_idx.end() || iter->second != dev_idx) {
          return clear_result();
        }
      }
    }
  }

  PADDLE_ENFORCE_GE(
      place_num, 1,
      platform::errors::NotFound(
          "No place found, this may be a bug.\nIt would be helpful if you "
          "could inform us of how this conversion went by opening a github "
          "issue at https://github.com/PaddlePaddle/Paddle/issues/new. And "
          "we will resolve it with high priority."));

  return true;
}

bool IsDataParallelInferenceGraph(const ir::Graph &graph) {
  size_t place_num;
  std::unordered_map<details::OpHandleBase *, size_t> op_to_dev_idx;
  return IsDataParallelInferenceGraphImpl(graph, &op_to_dev_idx, &place_num);
}

/**
 * This function tries to separate the original graph into multiple graphs, in
 * which each graph would only run on single device. This is usually used to
 * separate a data-parallel inference graph to multiple graphs on each device.
 *
 * The graph can be separated into multiple single device graphs if and only if:
 *
 *  - the graph does not contain any ops related to multi-devices communication,
 *    such as allreduce, send, recv, sync_batch_norm, etc.
 *
 *  - ops on different devices do not depend on each other. That is to say, the
 *    graph has several disconnected sub-graphs.
 */
std::vector<std::unique_ptr<ir::Graph>> TrySeparateToMultipleSingleDeviceGraphs(
    ir::Graph *graph) {
  size_t place_num;
  std::unordered_map<details::OpHandleBase *, size_t> op_to_dev_idx;
  if (!IsDataParallelInferenceGraphImpl(*graph, &op_to_dev_idx, &place_num)) {
    return {};
  }

  if (place_num == 1) {
    return {};
  }

  std::vector<std::unique_ptr<ir::Graph>> graphs(place_num);
  for (auto &g : graphs) {
    g.reset(new ir::Graph(ProgramDesc()));
    g->Set(kGraphVars, new GraphVars(1UL));
    g->Set(kGraphDepVars, new GraphDepVars());
  }

  std::vector<VarHandle *> isolated_var_handles;
  for (auto *node : graph->Nodes()) {
    if (!node->IsWrappedBy<VarHandleBase>()) {
      continue;
    }

    auto &var_handle_base = node->Wrapper<VarHandleBase>();
    auto *var_handle = dynamic_cast<VarHandle *>(&var_handle_base);
    if (var_handle && var_handle->PendingOps().empty() &&
        var_handle->GeneratedOp() == nullptr) {
      isolated_var_handles.emplace_back(var_handle);
    }
  }

  for (auto *var_handle : isolated_var_handles) {
    auto dev_idx = var_handle->scope_idx();
    auto &src_vars = graph->Get<GraphVars>(kGraphVars)[dev_idx];
    auto *dst_graph = graphs[dev_idx].get();
    auto &dst_vars = dst_graph->Get<GraphVars>(kGraphVars)[0];
    VLOG(10) << "Move isolated var " << var_handle->Name() << " at device "
             << dev_idx;
    dst_graph->AddNode(graph->RemoveNode(var_handle->Node()).release());
    dst_vars[var_handle->Name()].emplace_back(var_handle);
    src_vars.erase(var_handle->Name());
  }

  for (auto &pair : op_to_dev_idx) {
    auto *op = pair.first;
    auto dev_idx = pair.second;

    auto *ret_graph = graphs[dev_idx].get();
    auto &ret_vars = ret_graph->Get<GraphVars>(kGraphVars)[0];
    auto &ret_dummy_vars = ret_graph->Get<GraphDepVars>(kGraphDepVars);
    auto &origin_vars = graph->Get<GraphVars>(kGraphVars)[dev_idx];

    ret_graph->AddNode(graph->RemoveNode(op->Node()).release());

    auto handler = [&](const std::vector<VarHandleBase *> &vars) {
      for (auto *var : vars) {
        if (graph->Nodes().count(var->Node()) > 0) {
          ret_graph->AddNode(graph->RemoveNode(var->Node()).release());
          auto *dummy_var = dynamic_cast<DummyVarHandle *>(var);
          if (dummy_var == nullptr) {
            ret_vars.emplace(var->Name(), origin_vars.at(var->Name()));
          } else {
            ret_dummy_vars.emplace(dummy_var);
          }
        }
      }
    };

    handler(op->Inputs());
    handler(op->Outputs());
  }

  graph->Erase(kGraphVars);
  graph->Erase(kGraphDepVars);

  for (auto &g : graphs) {
    CopyGraphAttrIfExists<ProgramDescs>(*graph, g.get(), kProgramDescs);
    CopyGraphAttrIfExists<FusedVars>(*graph, g.get(), kFusedVars);
  }
  return graphs;
}

static bool HasDropLastReadOpImpl(const ir::Graph &graph, bool drop_last) {
  auto ops = ir::FilterByNodeWrapper<OpHandleBase>(graph);
  for (auto *op : ops) {
    auto *compute_op = dynamic_cast<ComputationOpHandle *>(op);
    if (compute_op && compute_op->GetOp()->Type() == "read" &&
        compute_op->GetOp()->Attr<bool>("drop_last") == drop_last) {
      VLOG(10) << "The graph has drop_last=" << drop_last << " read op";
      return true;
    }
  }
  VLOG(10) << "The graph does not have drop_last=" << drop_last << " read op";
  return false;
}

bool HasDropLastReadOp(const ir::Graph &graph) {
  return HasDropLastReadOpImpl(graph, true);
}

bool HasKeepLastReadOp(const ir::Graph &graph) {
  return HasDropLastReadOpImpl(graph, false);
}

}  // namespace details
}  // namespace framework
}  // namespace paddle
