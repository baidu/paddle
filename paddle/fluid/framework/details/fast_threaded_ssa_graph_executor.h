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

#pragma once
#include <ThreadPool.h>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
#include "paddle/fluid/framework/async_variable.h"
#include "paddle/fluid/framework/blocking_queue.h"
#include "paddle/fluid/framework/details/exception_holder.h"
#include "paddle/fluid/framework/details/execution_strategy.h"
#include "paddle/fluid/framework/details/ssa_graph_executor.h"

namespace paddle {
namespace framework {
class Scope;
namespace details {

class OpHandleBase;
class FastThreadedSSAGraphExecutor : public SSAGraphExecutor {
 public:
  FastThreadedSSAGraphExecutor(const ExecutionStrategy &strategy,
                               const std::vector<Scope *> &local_scopes,
                               const std::vector<Scope *> &local_exec_scopes,
                               const std::vector<platform::Place> &places,
                               ir::Graph *graph);
  FetchResultType Run(const std::vector<std::string> &fetch_tensors,
                      bool return_merged) override;
  const ir::Graph &Graph() const override;

 private:
  // Note(zcd): the ThreadPool should be placed last so that ThreadPool should
  // be destroyed first.
  ExecutionStrategy strategy_;
  std::vector<Scope *> local_scopes_;
  std::vector<Scope *> local_exec_scopes_;
  std::vector<platform::Place> places_;
  ir::Graph *graph_;

  std::unordered_map<OpHandleBase *, int> op_deps_;
  std::vector<OpHandleBase *> bootstrap_ops_;

  platform::DeviceContextPool fetch_ctxs_;
  std::atomic<int> remaining_;

  std::future<
      std::unique_ptr<std::unordered_map<OpHandleBase *, std::atomic<int>>>>
      atomic_op_deps_;
  ExceptionHolder exception_;

  ::ThreadPool pool_;
  ::ThreadPool prepare_pool_;

  std::vector<OpHandleBase *> traced_ops_;

  // TODO(Aurelius84): determine return and params
  void RunOpWithAsyncVariable(
      std::unordered_map<OpHandleBase *, std::atomic<int>> *op_deps,
      const std::vector<OpHandleBase *> &ready_op,
      const std::shared_ptr<BlockingQueue<size_t>> &complete_q);

  void UpdateAsyncVariableState(const std::vector<VarHandleBase *>);

  bool RunOp(OpHandleBase *op,
             const std::shared_ptr<BlockingQueue<size_t>> &complete_q,
             size_t *complete);

  void RunOpAsync(std::unordered_map<OpHandleBase *, std::atomic<int>> *op_deps,
                  OpHandleBase *op,
                  const std::shared_ptr<BlockingQueue<size_t>> &complete_q);

  void PrepareAtomicOpDeps();

  inline void RecordOps(OpHandleBase *op);

  inline void ExecutionFinal(std::vector<OpHandleBase *> *fetch_ops);

  inline bool RunOpSync(OpHandleBase *op);

  bool RunTracedOps(const std::vector<OpHandleBase *> &traced_ops);

  void InsertFetchOps(
      const std::vector<std::string> &fetch_tensors, FetchResultType *fetches,
      std::unordered_map<std::string, std::vector<VarHandleBase *>>
          *fetched_vars,
      std::unordered_map<OpHandleBase *, std::atomic<int>> *op_deps,
      std::vector<OpHandleBase *> *fetch_ops,
      std::vector<OpHandleBase *> *ready_fetch_ops, bool return_merged);
};
}  // namespace details
}  // namespace framework
}  // namespace paddle
