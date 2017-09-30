/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. */

#include "paddle/operators/dynamic_recurrent_op.h"

namespace paddle {
namespace operators {
void DynamicRecurrentOp::Run(const Scope& scope,
                             const platform::DeviceContext& dev_ctx) const {
  arg_cache_.Init(kArgName, *this, scope, &arg_);
  SplitInputs(scope);
  CreateScopes(scope);
  WriteStepInputs(scope);

  // call stepnet
}

void DynamicRecurrentOp::SplitInputs(const Scope& scope) const {
  // TODO(superjom) make level a config
  // TODO(superjom) check all the inputs has the same LoD
  int level = 0;
  const auto& inlinks = arg_cache_.inlinks;
  for (auto& item : inlinks) {
    const auto& var = item.second;
    const auto& tensor = var->Get<LoDTensor>();
    TensorArray& ta = step_inputs_[item.first];
    dy_seq_metas_[item.first] =
        ta.Unpack(tensor, level, true /*length_descend*/);
  }
}

void DynamicRecurrentOp::WriteStepInputs(const Scope& scope) const {
  const auto& inlinks = arg_cache_.inlinks;
  for (auto& item : inlinks) {
    TensorArray& ta = step_inputs_[item.first];
    for (size_t step = 0; step < ta.size(); step++) {
      auto tensor = ta.Read(step);
      auto& step_scope = arg_cache_.GetScope(step);
      step_scope.FindVar(item.first)
          ->GetMutable<LoDTensor>()
          ->ShareDataWith<value_type>(tensor);
    }
  }
}

void DynamicRecurrentOp::WriteStepOutputs(const Scope& scope) const {
  for (size_t step = 0; step < arg_cache_.scopes->size(); step++) {
    auto& scope = arg_cache_.GetScope(step);
    for (auto& item : step_outputs_) {
      const auto& step_output = scope.FindVar(item.first)->Get<LoDTensor>();
      item.second.WriteShared(step, step_output);
    }
  }
}

void DynamicRecurrentOp::CreateScopes(const Scope& scope) const {
  for (size_t step = arg_cache_.scopes->size(); step < step_inputs_.size();
       step++) {
    CreateTempInputsInScope(arg_cache_.GetScope(step));
    CreateTempOutputsInScope(arg_cache_.GetScope(step));
  }
}

void DynamicRecurrentOp::ConcatOutputs(const Scope& scope) const {
  // TODO(superjom) transform this to a config
  int level = 0;
  // TODO(superjom) pass in some lod
  // just a placeholder
  framework::LoD lod;
  for (auto& item : step_outputs_) {
    auto tensor = item.second.Pack(level, dy_seq_metas_[item.first], lod);
    auto& output = arg_cache_.outlinks[item.first]->Get<LoDTensor>();
    const_cast<LoDTensor*>(&output)->ShareDataWith<value_type>(tensor);
  }
}

void DynamicRecurrentOp::InitStates(Scope* step_scopes) const {}

void DynamicRecurrentOp::ArgCache::Init(
    const rnn::ArgumentName& name, const paddle::framework::OperatorBase& op,
    const paddle::framework::Scope& scope, const rnn::Argument* arg) {
  InitArgument(name, op, arg);
  CacheScopes(scope, *arg);
  CacheInlinks(scope, arg->inlinks);
  CacheOutlinks(scope, arg->outlinks);
}

// NOTE(superjom) should be called after SplitInputs
void DynamicRecurrentOp::CreateTempInputsInScope(Scope& scope) const {
  for (auto& input : stepnet_->Inputs()) {
    for (const std::string& var : input.second) {
      if (!scope.FindVar(var)) {
        scope.NewVar(var)->GetMutable<LoDTensor>();
      }
    }
  }
}

void DynamicRecurrentOp::CreateTempOutputsInScope(Scope& scope) const {
  for (auto& input : stepnet_->Outputs()) {
    for (const std::string& var : input.second) {
      if (!scope.FindVar(var)) {
        scope.NewVar(var)->GetMutable<LoDTensor>();
      }
    }
  }
}

void DynamicRecurrentOp::ArgCache::InitArgument(const rnn::ArgumentName& name,
                                                const OperatorBase& op,
                                                const rnn::Argument* arg) {
  rnn::InitArgument(name, arg, op, false /*is_grad*/);
}

void DynamicRecurrentOp::ArgCache::CacheScopes(const Scope& scope,
                                               const rnn::Argument& arg) {
  auto scopes_var = scope.FindVar(arg.step_scopes);
  PADDLE_ENFORCE(scopes_var != nullptr,
                 "the step_scopes output argument [%s] should be created first "
                 "by framework.",
                 arg.step_scopes);
  scopes = scopes_var->GetMutable<std::vector<Scope*>>();
}

void DynamicRecurrentOp::ArgCache::CacheInlinks(
    const Scope& scope, const std::vector<std::string>& names) {
  for (auto name : names) {
    auto* var = GetVariable(scope, name);
    inlinks[name] = var;
  }
}

void DynamicRecurrentOp::ArgCache::CacheOutlinks(
    const Scope& scope, const std::vector<std::string>& names) {
  for (auto name : names) {
    auto* var = GetVariable(scope, name);
    inlinks[name] = var;
  }
}

Variable* DynamicRecurrentOp::ArgCache::GetVariable(const Scope& scope,
                                                    const std::string& name) {
  auto* var = scope.FindVar(name);
  PADDLE_ENFORCE_NOT_NULL(var, "variable [%s] not exist in scope", name);
  return var;
}

const rnn::ArgumentName DynamicRecurrentOp::kArgName{
    "step_net", "step_scopes",  "inlinks",      "outlinks",
    "memories", "pre_memories", "boot_memories"};

}  // namespace operators
}  // namespace paddle
