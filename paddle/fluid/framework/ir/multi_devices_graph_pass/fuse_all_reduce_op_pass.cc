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

#include <algorithm>
#include <string>
#include <vector>

#include "paddle/fluid/framework/details/all_reduce_op_handle.h"
#include "paddle/fluid/framework/details/container_cast.h"
#include "paddle/fluid/framework/details/fused_all_reduce_op_handle.h"
#include "paddle/fluid/framework/details/multi_devices_helper.h"
#include "paddle/fluid/framework/ir/graph_helper.h"

namespace paddle {
namespace framework {
namespace ir {

class FuseAllReduceOpPass : public ir::Pass {
 protected:
  void ApplyImpl(ir::Graph *graph) const override {
    if (Get<size_t>(details::kNRanks) <= 1) {
      VLOG(6) << "The number of place is" << Get<size_t>(details::kNRanks)
              << ", there doesn't need apply FuseAllReduceOpPass.";
      return;
    }

    auto &places = Get<const std::vector<platform::Place>>(details::kPlaces);
    auto &local_scopes = Get<const std::vector<Scope *>>(details::kLocalScopes);

#if defined(PADDLE_WITH_NCCL)
    auto *multi_nccl_ctxs =
        &Get<platform::NCCLCommunicator>(details::kNCCLCtxs);
#elif defined(PADDLE_WITH_XPU_BKCL)
    auto *multi_bkcl_ctxs =
        &Get<platform::BKCLCommunicator>(details::kBKCLCtxs);
#endif

    ir::Graph &result = *graph;
    auto &params_grads =
        result.Get<details::ParamsAndGrads>(details::kParamsAndDenseGrads);
    size_t num_of_all_reduce = params_grads.size();
    std::unordered_set<std::string> grads;
    grads.reserve(num_of_all_reduce);
    for (auto p_g : params_grads) {
      grads.insert(p_g.second);
    }

    std::unordered_map<std::string, Node *> all_reduce_ops =
        GetAllReduceOps(result, places, grads);

    VLOG(6) << "Find all_reduce_ops: " << all_reduce_ops.size();
    if (all_reduce_ops.size() == 0) {
      return;
    }

    PADDLE_ENFORCE_EQ(
        all_reduce_ops.size(), grads.size(),
        platform::errors::Unimplemented(
            "The number of all_reduce OpHandle(%d) is not equal to the "
            "number of grads(%d). Maybe some gradients are sparse type, "
            "it is not supported currently.",
            all_reduce_ops.size(), grads.size()));

    auto &group_params_grads = graph->Get<details::GroupParamsAndGrads>(
        details::kGroupParamsAndDenseGrads);

    LOG(WARNING) << string::Sprintf(
        "Find all_reduce operators: %d. To make the speed faster, some "
        "all_reduce ops are fused during training, after fusion, "
        "the number of all_reduce ops is %d.",
        all_reduce_ops.size(), group_params_grads.size());

    for (auto &group_p_g : group_params_grads) {
      size_t group_size = group_p_g.size();
      PADDLE_ENFORCE_GT(
          group_size, static_cast<size_t>(0),
          platform::errors::InvalidArgument(
              "Parameter and Parameter@grad in one group, must not be empty."));
      std::vector<ir::Node *> group_all_reduce_ops;
      group_all_reduce_ops.reserve(group_size);
      for (auto &p_g : group_p_g) {
        group_all_reduce_ops.emplace_back(all_reduce_ops.at(p_g.second));
      }
#if defined(PADDLE_WITH_NCCL)
      InsertFusedAllReduce(places, local_scopes, group_size,
                           group_all_reduce_ops, multi_nccl_ctxs, &result);
#elif defined(PADDLE_WITH_XPU_BKCL)
      InsertFusedAllReduce(places, local_scopes, group_size,
                           group_all_reduce_ops, multi_bkcl_ctxs, &result);
#else
      InsertFusedAllReduce(places, local_scopes, group_size,
                           group_all_reduce_ops, &result);
#endif
    }
  }

  std::unordered_map<std::string, Node *> GetAllReduceOps(
      const Graph &result, const std::vector<platform::Place> &places,
      const std::unordered_set<std::string> &grads) const {
    size_t num_place = places.size();
    std::unordered_map<std::string, Node *> all_reduce_ops;
    all_reduce_ops.reserve(grads.size());
    for (auto &node : result.Nodes()) {
      if (node->IsOp()) {
        PADDLE_ENFORCE_EQ(
            node->IsWrappedBy<details::OpHandleBase>(), true,
            platform::errors::InvalidArgument(
                "Op Node(%s) should Wrapped by OpHandleBase.", node->Name()));
        auto *all_reduce_op_handle = dynamic_cast<details::AllReduceOpHandle *>(
            &node->Wrapper<details::OpHandleBase>());
        if (all_reduce_op_handle) {
#if defined(PADDLE_WITH_DGC)
          PADDLE_ENFORCE_NE(
              all_reduce_op_handle->Name(), "sparse_all_reduce",
              platform::errors::InvalidArgument(
                  "DGC doesn't support fuse for now, if you want to use DGC "
                  "you need set strategy.fuse_all_reduce_ops = False."));
#endif
          auto inputs = details::DynamicCast<details::VarHandle>(
              all_reduce_op_handle->Inputs());
          PADDLE_ENFORCE_EQ(inputs.size(), num_place,
                            platform::errors::InvalidArgument(
                                "The input size(%d) of all reduce op must "
                                "equal to place cnt(%d)!",
                                inputs.size(), num_place));
          // The inputs' name should be the same.
          auto &grad_name = inputs[0]->name();
          for (size_t i = 1; i < inputs.size(); ++i) {
            PADDLE_ENFORCE_EQ(
                inputs[i]->name(), grad_name,
                platform::errors::InvalidArgument(
                    "The input name should be the same.diff name: %s %s.",
                    inputs[i]->name(), grad_name));
          }
          PADDLE_ENFORCE_NE(
              grads.count(grad_name), static_cast<size_t>(0),
              platform::errors::InvalidArgument(
                  "Parameter@grad(%s) must in grad set.", grad_name));
          all_reduce_ops.emplace(grad_name, node);
        }
      }
    }
    return all_reduce_ops;
  }

  void InsertFusedAllReduce(const std::vector<platform::Place> &places,
                            const std::vector<Scope *> &local_scopes,
                            const size_t num_of_all_reduce,
                            const std::vector<ir::Node *> &all_reduce_ops,
#if defined(PADDLE_WITH_NCCL)
                            const platform::NCCLCommunicator *multi_nccl_ctxs,
#elif defined(PADDLE_WITH_XPU_BKCL)
                            const platform::BKCLCommunicator *multi_bkcl_ctxs,
#endif
                            ir::Graph *result) const {
    std::vector<details::VarHandleBase *> inputs;
    std::vector<details::VarHandleBase *> outputs;
    for (auto &op : all_reduce_ops) {
      auto &op_handle = op->Wrapper<details::OpHandleBase>();
      inputs.insert(inputs.end(), op_handle.Inputs().begin(),
                    op_handle.Inputs().end());
      // Remove output
      for_each(op_handle.Inputs().begin(), op_handle.Inputs().end(),
               [&op_handle](details::VarHandleBase *var_handle) {
                 var_handle->RemoveOutput(&op_handle, op_handle.Node());
               });

      outputs.insert(outputs.end(), op_handle.Outputs().begin(),
                     op_handle.Outputs().end());
      // Remove Input
      for_each(op_handle.Outputs().begin(), op_handle.Outputs().end(),
               [](details::VarHandleBase *var_handle) {
                 var_handle->ClearGeneratedOp();
               });

      result->RemoveNode(op_handle.Node());
    }

#if defined(PADDLE_WITH_NCCL)
    CreateFusedAllReduceOp(inputs, outputs, num_of_all_reduce, places,
                           local_scopes, multi_nccl_ctxs, result);
#elif defined(PADDLE_WITH_XPU_BKCL)
    CreateFusedAllReduceOp(inputs, outputs, num_of_all_reduce, places,
                           local_scopes, multi_bkcl_ctxs, result);
#else
    CreateFusedAllReduceOp(inputs, outputs, num_of_all_reduce, places,
                           local_scopes, result);
#endif
  }

 private:
  void CreateFusedAllReduceOp(
      const std::vector<details::VarHandleBase *> &inputs,
      const std::vector<details::VarHandleBase *> &outputs,
      const size_t num_of_all_reduce,
      const std::vector<platform::Place> &places,
      const std::vector<Scope *> &local_scopes,
#if defined(PADDLE_WITH_NCCL)
      const platform::NCCLCommunicator *multi_nccl_ctxs,
#elif defined(PADDLE_WITH_XPU_BKCL)
      const platform::BKCLCommunicator *multi_bkcl_ctxs,
#endif
      ir::Graph *result) const {
#if defined(PADDLE_WITH_NCCL)
    auto *op_handle = new details::FusedAllReduceOpHandle(
        result->CreateEmptyNode("fused_all_reduce", ir::Node::Type::kOperation),
        local_scopes, places, num_of_all_reduce, multi_nccl_ctxs);
#elif defined(PADDLE_WITH_XPU_BKCL)
    auto *op_handle = new details::FusedAllReduceOpHandle(
        result->CreateEmptyNode("fused_all_reduce", ir::Node::Type::kOperation),
        local_scopes, places, num_of_all_reduce, multi_bkcl_ctxs);
#else
    auto *op_handle = new details::FusedAllReduceOpHandle(
        result->CreateEmptyNode("fused_all_reduce", ir::Node::Type::kOperation),
        local_scopes, places, num_of_all_reduce);
#endif

    for (auto in : inputs) {
      op_handle->AddInput(in);
    }

    for (auto out : outputs) {
      op_handle->AddOutput(out);
    }

#if defined(PADDLE_WITH_NCCL)
    if (!multi_nccl_ctxs) {
      SetCommunicationContext(places, op_handle);
    }
#elif defined(PADDLE_WITH_XPU_BKCL)
    if (!multi_bkcl_ctxs) {
      SetCommunicationContext(places, op_handle);
    }
#else
    SetCommunicationContext(places, op_handle);
#endif
  }

  void SetCommunicationContext(
      const std::vector<platform::Place> &places,
      details::FusedAllReduceOpHandle *op_handle) const {
    for (size_t i = 0; i < places.size(); ++i) {
      op_handle->SetDeviceContext(
          places[i], platform::DeviceContextPool::Instance().Get(places[i]));
    }
  }
};

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(fuse_all_reduce_op_pass,
              paddle::framework::ir::FuseAllReduceOpPass)
    .RequirePassAttr(paddle::framework::details::kNRanks);
