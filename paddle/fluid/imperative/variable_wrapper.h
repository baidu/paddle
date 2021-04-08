// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
#include <memory>
#include <string>
#include <utility>

#include "paddle/fluid/framework/op_kernel_type.h"
#include "paddle/fluid/framework/variable.h"
#include "paddle/fluid/imperative/hooks.h"
#include "paddle/fluid/imperative/op_base.h"

namespace paddle {
namespace imperative {

class VariableWrapperHook;
class InplaceVariableWrapperHook;
class VarBase;
class GradOpNode;

class VariableWrapper {
 public:
  friend class VarBase;

  explicit VariableWrapper(const std::string& name) : name_(name) {}
  VariableWrapper(const std::string& name, const framework::Variable& variable)
      : var_(variable), name_(name) {}

  ~VariableWrapper() { VLOG(10) << "Destruct VariableWrapper: " << Name(); }

  const framework::Variable& Var() const { return var_; }

  framework::Variable* MutableVar() { return &var_; }

  // This is used for python api
  void SetOverridedStopGradient(bool stop_gradient) {
    overrided_stop_gradient_ = static_cast<int>(stop_gradient);

    if (auto grad_var = grad_var_.lock()) {
      grad_var->SetOverridedStopGradient(stop_gradient);
    }
  }

  // This is used for python api
  bool OverridedStopGradient() const { return overrided_stop_gradient_ != 0; }

  // This is used inside C++
  int InnerOverridedStopGradient() const { return overrided_stop_gradient_; }

  // This is used inside C++
  void InnerSetOverridedStopGradient(bool stop_gradient) {
    if (overrided_stop_gradient_ == -1) {
      overrided_stop_gradient_ = static_cast<int>(stop_gradient);
    } else {
      VLOG(6) << "Ignore Stop gradient conversion for Var: " << Name()
              << "Set value is: " << overrided_stop_gradient_;
    }

    if (auto grad_var = grad_var_.lock()) {
      grad_var->InnerSetOverridedStopGradient(stop_gradient);
    }
  }

  bool IsLeaf() const {
    if (OverridedStopGradient()) {
      return true;
    }
    if (HasGradVar() && !GetGradVar()->HasGradNode()) {
      return true;
    }
    return false;
  }

  bool IsLeafGrad() const {
    if (!HasGradNode() && !OverridedStopGradient()) {
      return true;
    }
    return false;
  }

  void SetPersistable(bool persistable) { persistable_ = persistable; }

  bool Persistable() const { return persistable_; }

  bool IsEmpty() const {
    bool is_empty = true;
    if (var_.IsInitialized()) {
      const framework::Tensor* tensor = nullptr;
      if (var_.IsType<framework::LoDTensor>()) {
        tensor = &(var_.Get<framework::LoDTensor>());
      } else if (var_.IsType<framework::SelectedRows>()) {
        tensor = &(var_.Get<framework::SelectedRows>().value());
      } else {
        PADDLE_THROW(platform::errors::PermissionDenied(
            "Only support LoDTensor and SelectedRows for gradient var"));
      }
      if (tensor && tensor->IsInitialized()) {
        is_empty = false;
      }
    }
    return is_empty || is_empty_;
  }

  // TODO(zhouwei): fix Tensor.clear_gradient() bug, function SetIsEmpty() isn't
  // need
  void SetIsEmpty(bool is_empty) { is_empty_ = is_empty; }

  const std::string& Name() const { return name_; }

  void SetName(const std::string& name) { name_ = name; }

  void SetType(framework::proto::VarType::Type type) { type_ = type; }

  framework::proto::VarType::Type Type() const { return type_; }

  std::shared_ptr<VariableWrapper> GetGradVar() const {
    return grad_var_.lock();
  }

  const std::weak_ptr<VariableWrapper>& GetWeakGradVar() const {
    return grad_var_;
  }

  std::shared_ptr<GradOpNode> GetGradNode() const { return grad_node_.lock(); }

  bool HasGradNode() const { return !grad_node_.expired(); }

  bool HasGradVar() const { return !grad_var_.expired(); }

  void SetDataType(framework::proto::VarType::Type data_type) {
    data_type_ = data_type;
  }

  framework::proto::VarType::Type DataType() const {
    const framework::Tensor* tensor = nullptr;
    if (var_.IsInitialized()) {
      if (type_ == framework::proto::VarType::LOD_TENSOR) {
        tensor = &(var_.Get<framework::LoDTensor>());
      } else if (type_ == framework::proto::VarType::SELECTED_ROWS) {
        tensor = &(var_.Get<framework::SelectedRows>().value());
      } else {
        VLOG(6) << "Variable " << name_ << " is not initialized";
        return data_type_;
      }
    }
    if (tensor && tensor->IsInitialized()) {
      return tensor->type();
    } else {
      VLOG(6) << "The tensor of variable " << name_ << " is not initialized";
      return data_type_;
    }
  }

  void SetForwardDataType(framework::proto::VarType::Type data_type) {
    fwd_data_type_ = data_type;
  }

  framework::proto::VarType::Type ForwardDataType() const {
    return fwd_data_type_;
  }

  const platform::Place Place() const {
    const framework::Tensor* tensor = nullptr;
    auto place =
        platform::CPUPlace();  // Default place for var not initialized.
    if (var_.IsInitialized()) {
      if (type_ == framework::proto::VarType::LOD_TENSOR) {
        tensor = &(var_.Get<framework::LoDTensor>());
      } else if (type_ == framework::proto::VarType::SELECTED_ROWS) {
        tensor = &(var_.Get<framework::SelectedRows>().value());
      } else {
        VLOG(6) << "Variable " << name_ << " is not initialized";
        return place;
      }
    }
    if (tensor && tensor->IsInitialized()) {
      return tensor->place();
    } else {
      VLOG(6) << "The tensor of variable " << name_ << " is not initialized";
      return place;
    }
  }

  uint32_t InplaceVersionSnapshot() const { return inplace_version_snapshot_; }

  void ResetInplaceVersion() {
    auto new_version = var_.CurrentInplaceVersion();

    VLOG(6) << "The wrapper version of VariableWrapper '" << name_
            << "' will be updated from " << inplace_version_snapshot_ << "to "
            << new_version;
    inplace_version_snapshot_ = new_version;
  }

  bool hasCacheKey(const paddle::framework::OpKernelType& key) {
    return var_cache.find(key) != var_cache.end();
  }

  std::shared_ptr<VariableWrapper> getCacheValue(
      const paddle::framework::OpKernelType& key) {
    return var_cache[key];
  }

  void setCacheValue(const paddle::framework::OpKernelType& key,
                     std::shared_ptr<VariableWrapper> val) {
    var_cache[key] = val;
    return;
  }

  /* Hook related methods */
  bool HasHook() const { return !hooks_.empty(); }

  bool HasMutableHook() const { return !mutable_hooks_.empty(); }

  int64_t AddHook(std::shared_ptr<VariableWrapperHook>&& hook) {
    hooks_.emplace(next_hook_id_, std::move(hook));
    return next_hook_id_++;
  }

  bool RemoveHook(const int64_t& hook_id) {
    auto remove_cnt = hooks_.erase(hook_id);
    if (remove_cnt == 0) {
      return false;
    }
    return true;
  }

  const std::map<int64_t, std::shared_ptr<VariableWrapperHook>>& GetHooks()
      const {
    return hooks_;
  }

  void AddMutableHook(std::shared_ptr<InplaceVariableWrapperHook>&& hook) {
    mutable_hooks_.emplace_back(std::move(hook));
  }

  const std::vector<std::shared_ptr<InplaceVariableWrapperHook>>&
  GetMutableHooks() const {
    return mutable_hooks_;
  }

 private:
  void SetGradVar(const std::shared_ptr<VariableWrapper>& var) {
    auto shared_var = grad_var_.lock();
    if (shared_var != var) {
      PADDLE_ENFORCE_EQ(
          shared_var, nullptr,
          platform::errors::PermissionDenied(
              "Cannot set gradient variable wrapper twice for %s", name_));
      grad_var_ = var;
    }
  }

  void SetGradNode(const std::shared_ptr<GradOpNode>& grad_node) {
    if (!grad_node) {
      grad_node_.reset();
      return;
    }

    auto shared_node = grad_node_.lock();
    if (shared_node != grad_node) {
      if (grad_node->InplaceGradNameMap().empty()) {
        // grad_node doesn't have Inplace message
        PADDLE_ENFORCE_EQ(
            shared_node, nullptr,
            platform::errors::PermissionDenied(
                "Cannot set gradient op twice unless using Inplace Strategy."));
      } else if (shared_node) {
        VLOG(3) << "The gradient op of Var (" << Name()
                << ") has been set twice. Because Inplace Strategy is used.";
      }
      grad_node_ = grad_node;
    }
  }

 private:
  framework::Variable var_;
  std::string name_;

  // Used for cache the dtype promotioned variableWrapper in real and complex
  // compute of Paddle Quantum
  std::map<paddle::framework::OpKernelType, std::shared_ptr<VariableWrapper>>
      var_cache;
  // add this property for users may set stop_gradient themselves and this
  // should override the frameworks setting (-1) unset, (1) true, (0) false
  int overrided_stop_gradient_{-1};
  bool persistable_{false};

  // Used for checking whether there is any inplace operation affecting gradient
  // calculation.
  uint32_t inplace_version_snapshot_{0};

  framework::proto::VarType::Type type_{framework::proto::VarType::LOD_TENSOR};
  framework::proto::VarType::Type data_type_{framework::proto::VarType::FP32};

  // See [ Why need handle complex gradient to real gradient? ]
  // Used for grad var to get the data type of its corresponding forward var,
  // if inconsistent, the data type of grad var needs to be casted to be
  // consistent with forward var
  framework::proto::VarType::Type fwd_data_type_{
      static_cast<framework::proto::VarType::Type>(-1)};

  std::weak_ptr<VariableWrapper> grad_var_;
  std::weak_ptr<GradOpNode> grad_node_;

  // TODO(zhouwei): fix bug of Tensor.clear_gradient(), function SetIsEmpty()
  // isn't need
  bool is_empty_{false};

  // NOTE(chenweihang): only grad var can hold hooks now
  int64_t next_hook_id_{0};
  // Hooks used to register hook for grad var, support adding and removing,
  // key is the accumulated int64_t value
  std::map<int64_t, std::shared_ptr<VariableWrapperHook>> hooks_;
  // Hooks executed after the execution of the entire backward process is over,
  // currently only supported for reducing in distributed training
  std::vector<std::shared_ptr<InplaceVariableWrapperHook>> mutable_hooks_;
};

}  // namespace imperative
}  // namespace paddle
