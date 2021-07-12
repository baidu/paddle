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
#include "paddle/fluid/framework/details/fast_threaded_ssa_graph_executor.h"

#include <cstdlib>
#include <deque>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "paddle/fluid/framework/async_variable.h"
#include "paddle/fluid/framework/details/computation_op_handle.h"
#include "paddle/fluid/framework/details/fetch_async_op_handle.h"
#include "paddle/fluid/framework/details/multi_devices_helper.h"
#include "paddle/fluid/framework/ir/graph_helper.h"
#include "paddle/fluid/platform/profiler.h"

namespace paddle {
namespace framework {
namespace details {

FastThreadedSSAGraphExecutor::FastThreadedSSAGraphExecutor(
    const ExecutionStrategy &strategy, const std::vector<Scope *> &local_scopes,
    const std::vector<Scope *> &local_exec_scopes,
    const std::vector<platform::Place> &places, ir::Graph *graph)
    : strategy_(strategy),
      local_scopes_(local_scopes),
      local_exec_scopes_(local_exec_scopes),
      places_(places),
      graph_(graph),
      fetch_ctxs_(places),
      pool_(strategy.num_threads_),
      // add one more thread for generate op_deps
      prepare_pool_(1) {
  VLOG(3) << "threads number is " << strategy.num_threads_;
  for (auto &op : ir::FilterByNodeWrapper<OpHandleBase>(*graph_)) {
    int dep = static_cast<int>(op->NotReadyInputSize());
    op_deps_.emplace(op, dep);
    if (dep == 0) {
      bootstrap_ops_.emplace_back(op);
    }
  }
  PADDLE_ENFORCE_GT(op_deps_.size(), 0,
                    platform::errors::PreconditionNotMet(
                        "The graph doesn't have operators."));
  PrepareAtomicOpDeps();
}

FetchResultType FastThreadedSSAGraphExecutor::Run(
    const std::vector<std::string> &fetch_tensors, bool return_merged) {
  VLOG(3) << "enter FastThreadedSSAGraphExecutor Run";
  std::unique_ptr<platform::RecordEvent> event(
      new platform::RecordEvent("FastThreadedSSAGraphExecutorPrepare"));
  std::unique_ptr<std::unordered_map<OpHandleBase *, std::atomic<int>>>
      op_deps = atomic_op_deps_.get();
  PrepareAtomicOpDeps();
  size_t num_ops = op_deps->size();

  FetchResultType fetches;
  if (return_merged) {
    fetches = FetchList(fetch_tensors.size());
  } else {
    fetches = FetchUnmergedList(fetch_tensors.size());
  }
  std::unordered_map<std::string, std::vector<VarHandleBase *>> fetched_vars;
  std::vector<OpHandleBase *> fetch_ops;
  std::vector<OpHandleBase *> ready_fetch_ops;
  exception_.Clear();

  InsertFetchOps(fetch_tensors, &fetches, &fetched_vars, op_deps.get(),
                 &fetch_ops, &ready_fetch_ops, return_merged);
  event.reset(nullptr);
  if (strategy_.num_threads_ == 1 && traced_ops_.size() == num_ops) {
    // If the num_threads is 1, we can record the order of operator's
    // execution in the first iteration, and in subsequent iterations,
    // run the recorded operators directly. This strategy could make the
    // execution faster.
    VLOG(3) << "Run the traced ops.";
    bool is_exception_free =
        RunTracedOps(traced_ops_) && RunTracedOps(fetch_ops);
    if (!is_exception_free) {
      ExecutionFinal(&fetch_ops);
    }
  } else {
    traced_ops_.clear();
    remaining_ = 0;
    auto complete_q = std::make_shared<BlockingQueue<size_t>>();
    // If run op with AsyncVariable
    if (IsEnableAsync()) {
      RunOpWithAsyncVariable(op_deps.get(), bootstrap_ops_, complete_q);
      RunOpWithAsyncVariable(op_deps.get(), ready_fetch_ops, complete_q);
    } else {  // original logic
      for (auto op : bootstrap_ops_) {
        RunOpAsync(op_deps.get(), op, complete_q);
      }
      for (auto op : ready_fetch_ops) {
        RunOpAsync(op_deps.get(), op, complete_q);
      }
    }

    size_t num_complete = 0;
    while (num_complete != op_deps->size()) {
      size_t num_comp = complete_q->Pop();
      if (num_comp == -1UL) {
        int remaining = 0;
        while (true) {
          remaining = remaining_;
          if (remaining == 0) {
            break;
          }
          for (int i = 0; i < remaining; ++i) {
            complete_q->Pop();
          }
        }
        if (exception_.IsCaught()) {
          ExecutionFinal(&fetch_ops);
        }
      }
      num_complete += num_comp;
    }
  }
  // Wait FetchOps.
  ClearFetchOp(graph_, &fetch_ops);

  for (auto &place : places_) {
    fetch_ctxs_.Get(place)->Wait();
  }

  return fetches;
}

void FastThreadedSSAGraphExecutor::InsertFetchOps(
    const std::vector<std::string> &fetch_tensors, FetchResultType *fetches,
    std::unordered_map<std::string, std::vector<VarHandleBase *>> *fetched_vars,
    std::unordered_map<OpHandleBase *, std::atomic<int>> *op_deps,
    std::vector<OpHandleBase *> *fetch_ops,
    std::vector<OpHandleBase *> *ready_fetch_ops, bool return_merged) {
  std::unordered_set<std::string> fetch_tensor_set(fetch_tensors.begin(),
                                                   fetch_tensors.end());
  for (auto &fetch_var_name : fetch_tensor_set) {
    for (auto &var_map : graph_->Get<GraphVars>(kGraphVars)) {
      auto it = var_map.find(fetch_var_name);
      if (it != var_map.end()) {
        (*fetched_vars)[fetch_var_name].push_back(*it->second.rbegin());
      }
    }
  }

  for (size_t i = 0; i < fetch_tensors.size(); ++i) {
    auto &var_name = fetch_tensors.at(i);
    auto fetched_var_it = fetched_vars->find(var_name);
    PADDLE_ENFORCE_NE(
        fetched_var_it, fetched_vars->end(),
        platform::errors::PreconditionNotMet(
            "Cannot find fetched variable(%s) in current computation graph. "
            "Possible reasons are:\n"
            "  1. The variable to be fetched is not defined in main program.\n"
            "  2. The variable to be fetched is not an input or output of any "
            "operator.\n"
            "  3. Confirm that you have used the fetch `Variable` format "
            "instead of the string literal('%s') in `fetch_list` parameter "
            "when using `executor.run` method. In other words, the format of "
            "`executor.run(fetch_list=[fetch_var])`(fetch_var is a Variable) "
            "is recommended.",
            var_name, var_name));

    auto &vars = fetched_var_it->second;

    ir::Node *fetch_node =
        graph_->CreateEmptyNode("fetch", ir::Node::Type::kOperation);
    auto *op = new FetchAsyncOpHandle(fetch_node, fetches, i, &local_scopes_,
                                      &local_exec_scopes_, return_merged);
    fetch_ops->emplace_back(op);

    for (auto &p : places_) {
      op->SetDeviceContext(p, fetch_ctxs_.Get(p));
    }

    for (auto *var : vars) {
      op->AddInput(var);
    }

    for (auto *var : vars) {
      auto *op = var->GeneratedOp();
      auto *compute_op = dynamic_cast<details::ComputationOpHandle *>(op);
      if (compute_op) {
        compute_op->SetLockAndRecordEventFree(false);
      }
    }

    int dep = static_cast<int>(op->NotReadyInputSize());
    (*op_deps)[op] = dep;
    if (dep == 0) {
      ready_fetch_ops->emplace_back(op);
    }
  }
}

bool FastThreadedSSAGraphExecutor::RunOp(
    OpHandleBase *op, const std::shared_ptr<BlockingQueue<size_t>> &complete_q,
    size_t *complete) {
  RunOpSync(op);
  if (LIKELY(!exception_.IsCaught())) {
    if (LIKELY(!strategy_.dry_run_)) {
      RecordOps(op);
    }
    ++(*complete);
    return true;
  } else {
    --remaining_;
    complete_q->Push(-1UL);
    return false;
  }
}

void FastThreadedSSAGraphExecutor::RunOpWithAsyncVariable(
    std::unordered_map<OpHandleBase *, std::atomic<int>> *op_deps,
    const std::vector<OpHandleBase *> &ready_ops,
    const std::shared_ptr<BlockingQueue<size_t>> &complete_q) {
  std::deque<OpHandleBase *> op_queue;
  for (auto *ready_op : ready_ops) {
    VLOG(3) << "Set op : " << ready_op->Name() << " all Inputs async state";
    UpdateAsyncVariableState(ready_op->Inputs());
    op_queue.push_back(ready_op);
  }

  while (!op_queue.empty()) {
    OpHandleBase *op_to_run = op_queue.front();
    op_queue.pop_front();

    // TODO(Aurelius84): Consider special case in MultiDeviceTransfer
    if (op_to_run->IsMultiDeviceTransfer()) {
      VLOG(3) << "Found MultiDeviceTranser op: " << op_to_run->Name()
              << ", skip it now.";
    }

    ++remaining_;  // indicate the current running thread nums

    // Note(Aurelius84): Run op async.
    this->pool_.enqueue([=] {
      size_t complete = 0;

      if (!RunOp(op_to_run, complete_q, &complete)) {
        VLOG(3) << "Run op " << op_to_run->Name() << " failed.";
        return;
      }
      // TODO(Aurelius84): mark all outputs as available and notify all other
      // thread.
      VLOG(3) << "Set op : " << op_to_run->Name() << " all Outputs async state";
      UpdateAsyncVariableState(op_to_run->Outputs());

      --remaining_;
      complete_q->Push(complete);
    });

    // step 4: deal with its downstream op
    VLOG(3) << "start to deal with downstream op from " << op_to_run->Name();
    auto &outputs = op_to_run->Outputs();
    for (auto &output : outputs) {
      for (auto &pending_op : output->PendingOps()) {
        std::atomic<int> &op_dep = op_deps->at(pending_op);
        VLOG(3) << pending_op->Name()
                << " dep value(before -1): " << op_dep.load();
        if (op_dep.fetch_sub(1) != 1) continue;

        // TODO(Aurelius84): Consider to avoid thread switch by launching one op
        // in current thread.
        VLOG(3) << "put downstream op into queue." << pending_op->Name();
        op_queue.push_back(pending_op);
      }
    }
  }
}

void FastThreadedSSAGraphExecutor::UpdateAsyncVariableState(
    const std::vector<VarHandleBase *> vars) {
  // TODO(Aurelius84): we should only update Available state for ready_op's
  // input vars once, and use callback
  // and hook to deal with other op's outputs.
  for (auto &var_handle_base : vars) {
    PADDLE_ENFORCE_NOT_NULL(var_handle_base,
                            platform::errors::PreconditionNotMet(
                                "var_handle_base should not be nullptr."));
    auto *var_handle = static_cast<VarHandle *>(var_handle_base);
    auto &scope = local_exec_scopes_.at(var_handle->scope_idx());
    auto *var = scope->FindVar(var_handle->Name());
    if (var == nullptr) {
      // control_var
      VLOG(3) << var_handle->Name() << " is not found in scope.";
    } else {
      VLOG(3) << "Set var: " << var_handle->Name()
              << " with AsyncState::kAvailable";
      var->NotifyAvailable();
    }
  }
}

void FastThreadedSSAGraphExecutor::RunOpAsync(
    std::unordered_map<OpHandleBase *, std::atomic<int>> *op_deps,
    OpHandleBase *op,
    const std::shared_ptr<BlockingQueue<size_t>> &complete_q) {
  ++remaining_;
  this->pool_.enqueue([=] {
    std::deque<OpHandleBase *> op_queue;
    op_queue.push_front(op);

    size_t complete = 0;
    while (!op_queue.empty()) {
      OpHandleBase *op_to_run = op_queue.back();
      op_queue.pop_back();

      // The Op involves data transfer of multiple devices may block other
      // computations emit. For example:
      // 1 step, queue=[Share, Allreduce], which Share is high priority
      // 2 step, Share exec, pending_op=Grad, queue=[Allreduce, Grad]
      // 3 step, Allreduce run with sync. Although Allreduce and Grad do not
      // have topo dependency, but Grad must wait for Allreduce to complete
      // before scheduling.
      // In this scenario, calculation and communication may not overlap.
      // Therefore, emit the op in the queue before running multi device op.
      if (op_to_run->IsMultiDeviceTransfer()) {
        while (!op_queue.empty()) {
          OpHandleBase *post_op = op_queue.back();
          op_queue.pop_back();
          RunOpAsync(op_deps, post_op, complete_q);
        }
      }

      if (!RunOp(op_to_run, complete_q, &complete)) {
        return;
      }

      auto &outputs = op_to_run->Outputs();
      op_to_run = nullptr;
      for (auto &output : outputs) {
        for (auto &pending_op : output->PendingOps()) {
          std::atomic<int> &deps = op_deps->at(pending_op);
          if (deps.fetch_sub(1) != 1) continue;

          // NOTE(zjl): op with highest priority should run
          // first without switching to another thread.
          if (pending_op->GetPriority() == OpHandleBase::Priority::kHighest) {
            op_queue.push_back(pending_op);
          } else if (pending_op->IsMultiDeviceTransfer()) {
            // multi device ops should be scheduled prior to computing ops
            op_queue.push_front(pending_op);
          } else {
            if (op_to_run == nullptr) {
              op_to_run = pending_op;
            } else {
              RunOpAsync(op_deps, pending_op, complete_q);
            }
          }
        }
      }

      if (op_to_run != nullptr) {
        op_queue.push_front(op_to_run);
      }
    }
    --remaining_;
    complete_q->Push(complete);
  });
}

void FastThreadedSSAGraphExecutor::PrepareAtomicOpDeps() {
  atomic_op_deps_ = prepare_pool_.enqueue([&] {
    auto *op_deps = new std::unordered_map<OpHandleBase *, std::atomic<int>>;
    for (auto &pair : op_deps_) {
      (*op_deps)[pair.first] = pair.second;
    }
    return std::unique_ptr<
        std::unordered_map<OpHandleBase *, std::atomic<int>>>(op_deps);
  });
}

const ir::Graph &FastThreadedSSAGraphExecutor::Graph() const { return *graph_; }

void FastThreadedSSAGraphExecutor::RecordOps(OpHandleBase *op) {
  if (strategy_.num_threads_ == 1 && !dynamic_cast<FetchAsyncOpHandle *>(op)) {
    traced_ops_.emplace_back(op);
  }
}

void FastThreadedSSAGraphExecutor::ExecutionFinal(
    std::vector<OpHandleBase *> *fetch_ops) {
  VLOG(3) << "caught exception " << exception_.Type() << ", rethrow it";
  // NOTE: If a new exception occurs in this ClearFetchOp operation, it will
  // cause the loss of exception triggered firstly not thrown.
  // Instead, the cleanup operation should only be performed when an EOF
  // exception is caught. If other exceptions are triggered, the ClearFetchOp
  // should not be continued.
  if (exception_.Type() == "EOF") {
    ClearFetchOp(graph_, fetch_ops);
  }
  exception_.ReThrow();
}

bool FastThreadedSSAGraphExecutor::RunTracedOps(
    const std::vector<OpHandleBase *> &traced_ops) {
  for (auto &op : traced_ops) {
    if (!RunOpSync(op)) return false;
  }
  return true;
}

bool FastThreadedSSAGraphExecutor::RunOpSync(OpHandleBase *op) {
  try {
    VLOG(10) << op << " " << op->Name() << " : " << op->DebugString();
    if (LIKELY(!strategy_.dry_run_)) {
      op->Run(strategy_.use_device_);
    }
    VLOG(10) << op << " " << op->Name() << " Done ";
    return true;
  } catch (std::exception &e) {
    // FIXME(Aurelius84): FIXME just for debug log.
    VLOG(2) << op->Name() << " " << e.what();

    exception_.Catch(std::current_exception());
    return false;
  }
}

}  // namespace details
}  // namespace framework
}  // namespace paddle
