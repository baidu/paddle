// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
#include <algorithm>
#include <cstdint>
#include <list>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/framework/type_defs.h"
#include "paddle/fluid/framework/var_type.h"
#include "paddle/fluid/framework/variable.h"
#include "paddle/fluid/imperative/flags.h"
#include "paddle/fluid/imperative/hooks.h"
#include "paddle/fluid/imperative/saved_variable_wrapper_list.h"
#include "paddle/fluid/imperative/type_defs.h"
#include "paddle/fluid/imperative/variable_wrapper.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/fluid/platform/macros.h"

namespace paddle {
namespace framework {
class Variable;
}  // namespace framework
}  // namespace paddle

namespace paddle {
namespace imperative {

class GradOpNode;
class OpBase;
class VariableWrapper;

class ThreadSafeNameSet {
 public:
  void Insert(const std::string& name);

  void Remove(const std::string& name);

  std::vector<std::string> Names() const;

 private:
  std::multiset<std::string> set_;
  mutable std::mutex mtx_;
};

class VarBase {
  DISABLE_COPY_AND_ASSIGN(VarBase);

 public:
  static std::vector<std::string> AliveVarNames();

 public:
  explicit VarBase(bool has_grad, const std::string& name)
      : var_(std::make_shared<VariableWrapper>(name)),
        grad_var_(has_grad ? new VarBase(false, GradVarName()) : nullptr) {
    if (has_grad) {
      var_->SetGradVar(grad_var_->var_);
    }

    if (IsDebugEnabled()) {
      VLOG(10) << "Construct VarBase: " << Name();
      name_set_.Insert(Name());
    }
  }

  explicit VarBase(const std::string& name) : VarBase(true, name) {}

  // NOTE(zengjinle): be careful when you use this constructor!!!
  // Unpack VarBase from VariableWrapper.
  explicit VarBase(const std::shared_ptr<VariableWrapper>& var);

  ~VarBase() {
    VLOG(10) << "Destruct VarBase: " << Name();
    if (IsDebugEnabled()) {
      name_set_.Remove(Name());
    }
  }

  const std::shared_ptr<VariableWrapper>& SharedVar() const { return var_; }

  const framework::Variable& Var() const { return var_->Var(); }

  framework::Variable* MutableVar() { return var_->MutableVar(); }

  bool HasGradVar() const { return grad_var_ != nullptr; }

  const std::shared_ptr<VarBase>& GradVarBase() const { return grad_var_; }

  void ClearGradVarBase() { grad_var_ = nullptr; }

  void SetGradVarBase(const VarBase& grad_var) {
    MutableGradVarBase()->CopyFrom(grad_var, true);
  }

  const std::shared_ptr<VarBase>& MutableGradVarBase() {
    if (grad_var_ == nullptr) {
      if (auto grad_var_wrapper = var_->GetGradVar()) {
        grad_var_ = std::make_shared<VarBase>(grad_var_wrapper);
      } else {
        grad_var_ = std::make_shared<VarBase>(false, GradVarName());
        var_->SetGradVar(grad_var_->var_);
        grad_var_->var_->SetGradNode(grad_var_->grad_node_);
      }
      // NOTE(zhiqiu): we should keep grad_var_'s stop_gradient property
      // same as fwd varbase
      grad_var_->SetOverridedStopGradient(var_->InnerOverridedStopGradient());
    }
    return grad_var_;
  }

  const framework::Variable& GradVar() const {
    PADDLE_ENFORCE_NOT_NULL(
        grad_var_,
        platform::errors::NotFound("Gradient of %s does not exist", Name()));
    return grad_var_->Var();
  }

  framework::Variable* MutableGradVar() {
    PADDLE_ENFORCE_NOT_NULL(
        grad_var_,
        platform::errors::NotFound("Gradient of %s does not exist", Name()));
    return grad_var_->MutableVar();
  }

  void SetOverridedStopGradient(bool stop_gradient) {
    var_->SetOverridedStopGradient(stop_gradient);
    if (grad_var_) {
      grad_var_->SetOverridedStopGradient(stop_gradient);
    }
  }

  bool OverridedStopGradient() const { return var_->OverridedStopGradient(); }

  bool IsLeaf() const { return var_->IsLeaf(); }

  void InnerSetOverridedStopGradient(bool stop_gradient) {
    if (var_->InnerOverridedStopGradient() == -1) {
      var_->InnerSetOverridedStopGradient(stop_gradient);
      if (grad_var_) {
        grad_var_->InnerSetOverridedStopGradient(stop_gradient);
      }
    }
  }

  void SetPersistable(bool persistable) { var_->SetPersistable(persistable); }

  bool Persistable() const { return var_->Persistable(); }

  // Only grad var is allowed to call these 2 methods
  void SetGradNode(const std::shared_ptr<GradOpNode>& node) {
    grad_node_ = node;
    var_->SetGradNode(node);
  }

  size_t GradOpNum() const;

  const std::shared_ptr<GradOpNode>& GradNode() const { return grad_node_; }

  void ClearGradNode() { SetGradNode(nullptr); }

  const std::string& Name() const { return var_->Name(); }

  void SetName(const std::string& name) {
    var_->SetName(name);
    if (grad_var_) {
      grad_var_->SetName(GradVarName());
    }
  }

  std::string GradVarName() { return framework::GradVarName(Name()); }

  void SetGraphIsFreed(bool free) { graph_is_free_ = free; }

  const bool& GraphIsFreed() const { return graph_is_free_; }

  void SetType(framework::proto::VarType::Type type) { var_->SetType(type); }

  framework::proto::VarType::Type Type() const { return var_->Type(); }

  void SetDataType(framework::proto::VarType::Type data_type) {
    var_->SetDataType(data_type);
    if (grad_var_) {
      grad_var_->SetDataType(data_type);
    }
  }

  framework::proto::VarType::Type DataType() const { return var_->DataType(); }

  void SetForwardDataType(framework::proto::VarType::Type data_type) {
    var_->SetForwardDataType(data_type);
  }

  framework::proto::VarType::Type ForwardDataType() const {
    return var_->ForwardDataType();
  }

  const platform::Place Place() const { return var_->Place(); }

  void ClearGradient();

  std::shared_ptr<VarBase> NewVarBase(const platform::Place& dst_place,
                                      const bool blocking) const;

  void CopyFrom(const imperative::VarBase& src, bool blocking);

  void BumpInplaceVersion();

  /* Hook related method: now only used for GradVarBase */
  bool HasVariableWrapperHook() const { return var_->HasVariableWrapperHook(); }

  int64_t AddVariableWrapperHook(std::shared_ptr<VariableWrapperHook>&& hook) {
    return var_->AddVariableWrapperHook(
        std::forward<std::shared_ptr<VariableWrapperHook>>(hook));
  }

  bool RemoveVariableWrapperHook(const int64_t& hook_id) {
    return var_->RemoveVariableWrapperHook(hook_id);
  }

  const std::map<int64_t, std::shared_ptr<VariableWrapperHook>>&
  GetVariableWrapperHooks() const {
    return var_->GetVariableWrapperHooks();
  }

  void AddVoidHook(std::shared_ptr<std::function<void()>>&& hook) {
    var_->AddVoidHook(
        std::forward<std::shared_ptr<std::function<void()>>>(hook));
  }

 private:
  /**
   * NOTE(zengjinle): never remove the const qualifier of `var_` if you are
   * not very familiar with the autograd idea (including the higher order
   * derivative).
   */
  const std::shared_ptr<VariableWrapper> var_;

  std::shared_ptr<VarBase> grad_var_;

  /**
   * NOTE(zengjinle): should consider whether to implement an inlined vector
   * or other things like that.
   */
  std::shared_ptr<GradOpNode> grad_node_;

  bool graph_is_free_ = false;

  mutable size_t copied_counter_ = 0;

  static ThreadSafeNameSet name_set_;
};

class Layer {
 public:
  virtual ~Layer() {}

  virtual std::vector<std::shared_ptr<VarBase>> Forward(
      const std::vector<std::shared_ptr<VarBase>>& inputs) {
    return {};
  }
};

std::shared_ptr<GradOpNode> CreateGradOpNode(
    const framework::OperatorBase& op, const NameVarBaseMap& ins,
    const NameVarBaseMap& outs, const framework::AttributeMap& attrs,
    const framework::AttributeMap& default_attrs, const platform::Place& place,
    const std::map<std::string, std::string>& inplace_map);

void ClearNoNeedBufferInputs(OpBase* op);

}  // namespace imperative
}  // namespace paddle
