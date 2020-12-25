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
#include <map>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "paddle/fluid/framework/details/execution_strategy.h"
#include "paddle/fluid/framework/details/var_handle.h"
#include "paddle/fluid/framework/ir/node.h"
#include "paddle/fluid/platform/device_context.h"
#include "paddle/fluid/platform/macros.h"

namespace paddle {
namespace platform {
class DeviceContext;
}  // namespace platform
}  // namespace paddle

namespace paddle {
namespace framework {

class Scope;
namespace details {
struct VarHandleBase;
}  // namespace details
namespace ir {
class Node;
}  // namespace ir

namespace details {
using DeviceType = paddle::platform::DeviceType;
namespace p = paddle::platform;
// Wraps ir::Node and provide helper utilities.
// It's responsible for populating necessary fields of ir::Node.
class OpHandleBase {
 public:
  /**
   * NOTE(zjl): Some op should have higher priority than others.
   * The higher priority op would run first without switching
   * threads in Executor.
   */
  enum Priority { kHighest = 0, kNormal = 1 };

  // Owned by `node`. No need to be deleted explicitly.
  explicit OpHandleBase(ir::Node *node) : node_(node) {
    node_->WrappedBy(this);
  }

  virtual ~OpHandleBase() PADDLE_MAY_THROW;

  std::string DebugString() const;

  virtual Priority GetPriority() const { return kNormal; }

  virtual bool GetSkipRunning() const { return skip_running_; }

  virtual void SetSkipRunning(bool skip_runing) { skip_running_ = skip_runing; }

  virtual std::string Name() const = 0;

  void Run(DeviceType use_device);

  virtual void RecordWaitEventOnCtx(platform::DeviceContext *waited_ctx);

  void AddInput(VarHandleBase *in);

  void AddOutput(VarHandleBase *out);

  // This method adds the wait events of all the input on all the device
  // context.
  // NOTE: This Wait is asynchronous operation.
  // NOTE: wait_for_feed is added to wait for feed var, since it has
  // generated op, no event and cannot perform event wait. It is only
  // used in fetch_async_op_handle currently.
  virtual void WaitInputVarGenerated(bool wait_for_feed = false);

  // This method adds the wait events of all the input on the specified device
  // context.
  // NOTE: This Wait is asynchronous operation.
  virtual void WaitInputVarGenerated(const platform::Place &place);

  virtual bool NeedWait(VarHandleBase *in_var);

  // If the Op involves data transfer of multiple devices that
  // will likely block other computations.
  virtual bool IsMultiDeviceTransfer() { return false; }

  const platform::DeviceContext *DeviceContext(platform::Place place) {
    auto it = dev_ctxes_.find(place);
    return it != dev_ctxes_.end() ? it->second : nullptr;
  }
  const std::map<platform::Place, platform::DeviceContext *> &DeviceContext() {
    return dev_ctxes_;
  }

  void SetDeviceContext(platform::Place place, platform::DeviceContext *ctx_) {
    dev_ctxes_[place] = ctx_;
  }

  const std::vector<VarHandleBase *> &Inputs() const { return inputs_; }

  size_t NoDupInputSize() const {
    std::unordered_set<VarHandleBase *> res;
    for (auto *var : inputs_) {
      res.emplace(var);
    }
    return res.size();
  }

  size_t NotReadyInputSize() const;

  const std::vector<VarHandleBase *> &Outputs() const { return outputs_; }

  size_t NoDummyInputSize() const;

  ir::Node *Node() { return node_; }

  const ir::Node *Node() const { return node_; }

  void SetLocalExecScopes(
      const std::unordered_map<Scope *, Scope *> &scope_map);

 protected:
  virtual std::vector<Scope *> GetLocalScopes() = 0;

  void RunAndRecordEvent(const std::function<void()> &callback);

  void RunAndRecordEvent(platform::Place p,
                         const std::function<void()> &callback);

  virtual void RunImpl() = 0;

  virtual void InitCUDA();
  virtual void InitXPU();

  ir::Node *node_;
  std::vector<VarHandleBase *> inputs_;
  std::vector<VarHandleBase *> outputs_;
  std::map<platform::Place, platform::DeviceContext *> dev_ctxes_;

  std::vector<Scope *> local_exec_scopes_;
  bool skip_running_ = false;

#ifdef PADDLE_WITH_CUDA
  std::unordered_map<int, cudaEvent_t> events_;
#endif

  DISABLE_COPY_AND_ASSIGN(OpHandleBase);
};

}  // namespace details
}  // namespace framework
}  // namespace paddle
