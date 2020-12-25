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

#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "paddle/fluid/framework/details/build_strategy.h"
#include "paddle/fluid/framework/details/multi_devices_helper.h"
#include "paddle/fluid/framework/ir/graph.h"

namespace paddle {
namespace framework {
namespace details {
class OpHandleBase;
struct VarHandle;
}  // namespace details
namespace ir {
class Graph;
}  // namespace ir
}  // namespace framework
}  // namespace paddle

namespace paddle {
namespace platform {
class NCCLContextMap;
class NCCLCommunicator;
class BKCLContextMap;
class BKCLCommunicator;
}

namespace framework {
class Scope;

namespace ir {

constexpr char kLossVarName[] = "loss_var_name";
constexpr char kStrategy[] = "strategy";

class MultiDevSSAGraphBuilderBase : public ir::Pass {
 protected:
  void ApplyImpl(ir::Graph *graph) const override;

  virtual void Init() const;

  virtual void CheckGraph(const ir::Graph &graph) const;

  virtual std::vector<ir::Node *> SortOperations(const ir::Graph &graph) const;

  virtual void InsertCollectiveOp(ir::Graph *result, const std::string &p_name,
                                  const std::string &g_name) const = 0;

  virtual bool DealWithSpecialOp(ir::Graph *result, ir::Node *node) const;

  virtual void InsertPostprocessOps(ir::Graph *result) const = 0;

  bool UseGPU() const;

  virtual bool NeedCollectiveForGrad(const std::string &grad_name,
                                     std::vector<ir::Node *> ops) const;

  bool IsScaleLossOp(ir::Node *node) const;

  void CreateComputationalOps(ir::Graph *result, ir::Node *node,
                              size_t num_places) const;

  void CreateScaleLossGradOp(ir::Graph *result,
                             const std::string &loss_grad_name,
                             ir::Node *out_var_node, size_t loss_scale,
                             proto::VarType::Type dtype) const;

  details::VarHandle *CreateReduceOp(ir::Graph *result, const std::string &og,
                                     size_t dst_dev_id) const;

  void CreateComputationalOp(ir::Graph *result, ir::Node *node,
                             size_t dev_id) const;

  bool IsSparseGradient(const std::string &og) const;

  void CreateAllReduceOp(ir::Graph *result, const std::string &og,
                         bool is_encoded = false) const;

  void CreateBroadcastOp(ir::Graph *result, const std::string &p_name,
                         size_t src_dev_id) const;

  void InsertScaleLossGradOp(ir::Graph *result, const ir::Node *node) const;

  void CreateFusedBroadcastOp(
      ir::Graph *result,
      const std::vector<std::unordered_set<std::string>> &bcast_varnames) const;

  void SetCommunicationContext(details::OpHandleBase *op_handle,
                               const platform::Place &p) const;

  void CreateOpHandleIOs(ir::Graph *result, ir::Node *node,
                         size_t device_id) const;

  void CreateIsolatedVarNode(ir::Graph *result, ir::Node *var_node) const;

#if defined(PADDLE_WITH_NCCL)
  mutable platform::NCCLContextMap *nccl_ctxs_{nullptr};
  mutable platform::NCCLCommunicator *multi_nccl_ctxs_{nullptr};
#elif defined(PADDLE_WITH_XPU_BKCL)
  mutable platform::BKCLContextMap *bkcl_ctxs_{nullptr};
  mutable platform::BKCLCommunicator *multi_bkcl_ctxs_{nullptr};
#endif

  mutable std::string loss_var_name_;
  mutable std::vector<platform::Place> places_;
  mutable std::vector<Scope *> local_scopes_;

  mutable details::BuildStrategy strategy_;
  mutable std::unordered_map<std::string, VarDesc *> all_vars_;
};

class AllReduceSSAGraphBuilder : public MultiDevSSAGraphBuilderBase {
 protected:
  virtual void InsertCollectiveOp(ir::Graph *result, const std::string &p_name,
                                  const std::string &g_name) const;

  virtual void InsertPostprocessOps(ir::Graph *result) const {}

  bool IsEncoded(const std::string &p_name) const;
};

class AsyncSSAGraphBuilder : public MultiDevSSAGraphBuilderBase {
 protected:
  void InsertCollectiveOp(ir::Graph *result, const std::string &p_name,
                          const std::string &g_name) const override {}

  bool NeedCollectiveForGrad(const std::string &grad_name,
                             std::vector<ir::Node *> ops) const override {
    return false;
  }

  bool DealWithSpecialOp(ir::Graph *result, ir::Node *node) const override {
    if (node->Op()->Type() == "recv") {
      VLOG(1) << "set recv op do_not_run to true";
      node->Op()->SetAttr("do_not_run", 1);
      node->Op()->Flush();
    }
    return false;
  }

  void InsertPostprocessOps(ir::Graph *result) const override {}
};

class BalanceVarSSAGraphBuilder : public MultiDevSSAGraphBuilderBase {
 protected:
  int GetVarDeviceID(const std::string &varname) const;

  int GetOpDeviceID(ir::Node *node) const;

  size_t GetAppropriateDeviceID(
      const std::vector<std::string> &var_names) const;

  virtual void ResetState() const;

  mutable std::unordered_map<std::string, int> sharded_var_device_;
  mutable std::vector<int64_t> balance_vars_;
};

class ReduceSSAGraphBuilder : public BalanceVarSSAGraphBuilder {
 protected:
  virtual void Init() const;

  virtual void InsertCollectiveOp(ir::Graph *result, const std::string &p_name,
                                  const std::string &g_name) const;

  virtual bool DealWithSpecialOp(ir::Graph *result, ir::Node *node) const;

  virtual void InsertPostprocessOps(ir::Graph *result) const;

  virtual std::vector<ir::Node *> SortOperations(const ir::Graph &graph) const;

  virtual void ResetState() const;

  int GetOpDeviceID(ir::Node *node,
                    std::unordered_map<std::string, std::vector<ir::Node *>>
                        *delay_ops) const;

  std::vector<ir::Node *> SortForReduceMode(
      const std::vector<ir::Node *> &topo_ops) const;

  mutable std::vector<std::unordered_set<std::string>> bcast_var_name_set_;
};

class DistSSAGraphBuilder : public BalanceVarSSAGraphBuilder {
 protected:
  virtual void Init() const;

  virtual bool DealWithSpecialOp(ir::Graph *result, ir::Node *node) const;

  virtual void InsertPostprocessOps(ir::Graph *result) const;

  virtual void InsertCollectiveOp(ir::Graph *result, const std::string &p_name,
                                  const std::string &g_name) const;

  virtual void ResetState() const;

  int CreateRPCOp(ir::Graph *result, ir::Node *node) const;

  int CreateDistTrainOp(ir::Graph *result, ir::Node *node) const;

  mutable std::vector<std::unordered_set<std::string>> bcast_var_name_set_;
  mutable bool need_broadcast_var_{false};
};

std::unordered_set<std::string> &MultiDevSSAGraphBuilder();

}  // namespace ir
}  // namespace framework
}  // namespace paddle
