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

#include "paddle/fluid/framework/details/scope_buffered_ssa_graph_executor.h"

#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "paddle/fluid/framework/details/multi_devices_helper.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/variable_helper.h"
#include "paddle/fluid/platform/profiler.h"
namespace paddle {
namespace framework {
namespace details {

ScopeBufferedSSAGraphExecutor::ScopeBufferedSSAGraphExecutor(
    ExecutionStrategy strategy, std::vector<Scope *> local_scopes,
    std::vector<Scope *> local_exec_scopes, std::vector<VariableInfo> var_infos,
    std::vector<platform::Place> places,
    std::unique_ptr<SSAGraphExecutor> &&underlying_executor)
    : strategy_(std::move(strategy)),
      underlying_executor_(std::move(underlying_executor)),
      local_scopes_(std::move(local_scopes)),
      local_exec_scopes_(std::move(local_exec_scopes)),
      var_infos_(std::move(var_infos)),
      places_(std::move(places)),
      scope_monitor_(places_, local_exec_scopes_) {
  PADDLE_ENFORCE_EQ(
      local_scopes_.size(), local_exec_scopes_.size(),
      platform::errors::InvalidArgument(
          "The number of local scopes and the number of local execution scopes "
          "should be equal, but got number of local scopes is %d and "
          "number of local execution scopes is %d.",
          local_scopes_.size(), local_exec_scopes_.size()));
  PrepareLocalExeScopes();
}

FetchResultType ScopeBufferedSSAGraphExecutor::Run(
    const std::vector<std::string> &fetch_tensors, bool return_merged) {
  if (drop_scope_counter_ == 0) {
    platform::RecordEvent e("InitLocalVars");
    InitVariables();
  }

  FetchResultType fetch_data;
  std::exception_ptr eptr = nullptr;

  auto exe_run_func = [&]() {
    try {
      fetch_data = underlying_executor_->Run(fetch_tensors, return_merged);
    } catch (...) {
      eptr = std::current_exception();
    }
  };

  if (strategy_.skip_drop_scope_ ||
      strategy_.num_iteration_per_drop_scope_ == 1) {
    exe_run_func();
  } else {
    scope_monitor_.Apply(exe_run_func, fetch_tensors.size() > 0);
  }

  if (VLOG_IS_ON(5)) {
    for (auto *scope : local_exec_scopes_) {
      VLOG(5) << "Left "
              << string::HumanReadableSize(GetScopeVarMemorySize(scope))
              << " on scope " << scope << " before deleting";
    }
  }

  ++drop_scope_counter_;
  if (!strategy_.skip_drop_scope_ &&
      (drop_scope_counter_ == strategy_.num_iteration_per_drop_scope_ ||
       DropScopeOrNot())) {
    DropLocalExeScopes();
  }

  if (VLOG_IS_ON(5)) {
    for (auto *scope : local_exec_scopes_) {
      VLOG(5) << "Left "
              << string::HumanReadableSize(GetScopeVarMemorySize(scope))
              << " on scope " << scope << " after deleting";
    }
  }

  if (eptr) {
    std::rethrow_exception(eptr);
  } else {
    return fetch_data;
  }
}

bool ScopeBufferedSSAGraphExecutor::DropScopeOrNot() const {
  for (auto &var : tensor_array_vars_) {
    auto tensor_array = var->GetMutable<LoDTensorArray>();
    for (LoDTensor &tensor : *tensor_array) {
      if (tensor.IsInitialized()) {
        return true;
      }
    }
    tensor_array->clear();
  }
  return false;
}

void ScopeBufferedSSAGraphExecutor::InitVariables() {
  for (auto &info : tmp_var_infos_) {
    for (auto &pair : info) {
      InitializeVariable(pair.first, pair.second);
    }
  }

  const ir::Graph &graph = Graph();
  if (!is_initialized_) {
    // startup_program_descs only need to be executed once
    if (graph.Has(details::kStartupProgramDescs)) {
      auto &program_descs =
          graph.Get<details::ProgramDescs>(details::kStartupProgramDescs);

      for (auto &program_desc : program_descs) {
        for (auto &op_desc : program_desc.Block(0).AllOps()) {
          for (size_t i = 0; i < local_exec_scopes_.size(); ++i) {
            auto op = OpRegistry::CreateOp(*op_desc);
            op->Run(*local_exec_scopes_[i], places_[i]);
          }
        }
      }
    }
    is_initialized_ = true;
  }

  if (graph.Has(details::kProgramDescs)) {
    auto &program_descs =
        graph.Get<details::ProgramDescs>(details::kProgramDescs);

    for (auto &program_desc : program_descs) {
      for (auto &op_desc : program_desc.Block(0).AllOps()) {
        for (size_t i = 0; i < local_exec_scopes_.size(); ++i) {
          auto op = OpRegistry::CreateOp(*op_desc);
          op->Run(*local_exec_scopes_[i], places_[i]);
        }
      }
    }
  }
}

void ScopeBufferedSSAGraphExecutor::DropLocalExeScopes() {
  platform::RecordEvent drop_scope_event("DropLocalExeScopes");
  drop_scope_counter_ = 0;
  for (auto &p : places_) {
    platform::DeviceContextPool::Instance().Get(p)->Wait();
  }
  scope_monitor_.ClearHistoryLocalExecScopes();
  for (size_t i = 0; i < local_exec_scopes_.size(); ++i) {
    local_exec_scopes_[i]->EraseVarsExcept(preserve_vars_[i]);
    local_exec_scopes_[i]->DropKids();
    for (auto &preserve_var : preserve_vars_[i]) {
      preserve_var->Clear();
    }
    VLOG(3) << "Drop local execution scope: " << local_scopes_[i];
  }
}

void ScopeBufferedSSAGraphExecutor::PrepareLocalExeScopes() {
  // Create local scopes.
  preserve_vars_.resize(local_scopes_.size());
  tmp_var_infos_.resize(local_scopes_.size());

  for (auto it = local_scopes_.rbegin(); it != local_scopes_.rend(); ++it) {
    size_t idx = local_scopes_.size() - 1 - (it - local_scopes_.rbegin());
    auto *scope = local_scopes_[idx];
    auto *local_scope = local_exec_scopes_[idx];

    for (auto &info : var_infos_) {
      if (info.persistable_) {  // Persistable
        auto var = scope->FindVar(info.name_);
        if (var != nullptr) {
          VLOG(2)
              << info.name_
              << " has been initialized beforehand in global scope, skipped";
          continue;
        }
        InitializeVariable(scope->Var(info.name_), info.type_);
      } else {
        Variable *tmp_var = local_scope->Var(info.name_);
        preserve_vars_[idx].emplace(tmp_var);
        tmp_var_infos_[idx].emplace_back(tmp_var, info.type_);
        if (info.type_ == proto::VarType::LOD_TENSOR_ARRAY) {
          tensor_array_vars_.emplace_back(tmp_var);
        }
      }
    }
  }
}

bool ScopeBufferedSSAGraphExecutor::NeedCreateLocalExeScope() {
  return drop_scope_counter_ == 0;
}

}  // namespace details
}  // namespace framework
}  // namespace paddle
