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

#include <atomic>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include "paddle/fluid/framework/type_defs.h"
#include "paddle/fluid/imperative/saved_variable_wrapper_list.h"
#include "paddle/fluid/imperative/type_defs.h"
#include "paddle/fluid/imperative/variable_wrapper.h"
#include "paddle/fluid/platform/place.h"

namespace paddle {
namespace imperative {

// TODO(zjl): to support py_func layer
class OpBase {
 public:
  OpBase() = default;

  OpBase(const OpBase&) = delete;

  OpBase(OpBase&&) = default;

  OpBase& operator=(const OpBase&) = delete;

  OpBase& operator=(OpBase&&) = default;

  ~OpBase() { VLOG(3) << "Destruct Op: " << Type(); }

  const std::string& Type() const {
    return op_ ? op_->Type() : UnknownOpType();
  }

  const framework::AttributeMap& Attrs() const { return attrs_; }

  const framework::AttributeMap& DefaultAttrsMap() const {
    return *default_attrs_;
  }

  const framework::OpInfo& Info() const {
    PADDLE_ENFORCE_NOT_NULL(op_, platform::errors::PreconditionNotMet(
                                     "OpBase::Info() should be called after "
                                     "OpBase::SetType() is called"));
    return op_->Info();
  }

  const framework::OperatorBase& InnerOp() const {
    PADDLE_ENFORCE_NOT_NULL(op_, platform::errors::PreconditionNotMet(
                                     "OpBase::InnerOp() should be called after "
                                     "OpBase::SetType() is called"));
    return *op_;
  }

  void ClearBackwardTrace();

  NameVarMap<VariableWrapper>* GetMutableOutsMap() { return &outs_; }

  NameVarMap<VariableWrapper>* GetMutableInsMap() { return &ins_; }

  const NameVarMap<VariableWrapper>& GetInsMap() const { return ins_; }

  const NameVarMap<VariableWrapper>& GetOutsMap() const { return outs_; }

  void SetType(const std::string& type);

  void CheckAttrs() {
    auto& info = Info();
    if (info.Checker() != nullptr) {
      info.Checker()->Check(&attrs_, true);
    }
  }

  void SetInput(const std::string& name, VariableWrapperList vars,
                bool is_grad) {
    auto& in_vars = ins_[name];
    *(in_vars.MutableVarList()) = std::move(vars);
    in_vars.SetIsGrad(is_grad);
  }

  void SetOutput(const std::string& name, VariableWrapperList vars,
                 bool is_grad) {
    auto& out_vars = outs_[name];
    *(out_vars.MutableVarList()) = std::move(vars);
    out_vars.SetIsGrad(is_grad);
  }

  void SetAttrMap(const framework::AttributeMap& attrs) { attrs_ = attrs; }

  void SetDefaultAttrsMap(const framework::AttributeMap& default_attrs) {
    default_attrs_ = &default_attrs;
  }

  void SetAttr(const std::string& name, const framework::Attribute& v) {
    attrs_[name] = v;
  }

  void SetBlockAttr(const std::string& name, framework::BlockDesc* block) {
    PADDLE_THROW(platform::errors::PermissionDenied(
        "SetBlockAttr is not support in dygraph OpBase"));
  }

  const framework::AttributeMap& Attrs() { return attrs_; }

  const framework::AttributeMap& DefaultAttrsMap() { return *default_attrs_; }

  bool HasAttr(const std::string& name) const {
    return attrs_.count(name) > 0 || default_attrs_->count(name) > 0;
  }

  const framework::Attribute& GetAttr(const std::string& name) const {
    auto it = attrs_.find(name);
    if (it != attrs_.end()) {
      return it->second;
    } else {
      auto it_default = default_attrs_->find(name);
      PADDLE_ENFORCE_NE(
          it_default, default_attrs_->end(),
          platform::errors::NotFound("can not find attribute [%s]", name));
      return it_default->second;
    }
  }

  template <typename T>
  inline const T& Attr(const std::string& name) const {
    return BOOST_GET_CONST(T, GetAttr(name));
  }

  size_t id() const { return id_; }

  void SetId(size_t id) { id_ = id; }

  const platform::Place& place() const { return place_; }

  void SetPlace(const platform::Place& place) { place_ = place; }

  void EnforceHasInOut() const {
    PADDLE_ENFORCE_NE(
        ins_.empty() && outs_.empty(), true,
        platform::errors::NotFound(
            "Inputs and outputs of %s do not exist. This may be because:\n"
            "1. You use some output variables of the previous batch as the "
            "inputs of the current batch. Please try to call \"stop_gradient "
            "= True\" or \"detach()\" for these variables.\n"
            "2. You calculate backward twice for the same subgraph without "
            "setting retain_graph=True. Please set retain_graph=True in the "
            "first backward call.\n\n",
            Type()));
  }

  static size_t GenerateUniqueId() {
    static std::atomic<size_t> unique_id{0};
    return unique_id.fetch_add(1);
  }

  static void Run(const framework::OperatorBase& op,
                  const NameVarMap<VarBase>& ins,
                  const NameVarMap<VarBase>& outs,
                  const framework::AttributeMap& attrs,
                  const framework::AttributeMap& default_attrs,
                  const platform::Place& place);

  static void Run(const framework::OperatorBase& op,
                  const NameVarMap<VariableWrapper>& ins,
                  const NameVarMap<VariableWrapper>& outs,
                  const framework::AttributeMap& attrs,
                  const framework::AttributeMap& default_attrs,
                  const platform::Place& place);

 private:
  static const std::string& UnknownOpType() {
    static std::string kUnknownOpType{"unknown"};
    return kUnknownOpType;
  }

 private:
  NameVarMap<VariableWrapper> ins_;
  NameVarMap<VariableWrapper> outs_;
  framework::AttributeMap attrs_;
  const framework::AttributeMap* default_attrs_;
  std::unique_ptr<framework::OperatorBase> op_;
  platform::Place place_;
  size_t id_{-1UL};
};

class GradOpNode {
 public:
  GradOpNode() = default;

  void reserve(size_t size) { ops_.reserve(size); }

  size_t size() const { return ops_.size(); }

  bool empty() const { return ops_.empty(); }

  void clear() { ops_.clear(); }

  void pop_back() { ops_.pop_back(); }

  template <typename... ARGS>
  OpBase& emplace_back(ARGS&&... args) {  // NOLINT
    ops_.emplace_back(std::forward<ARGS>(args)...);
    return ops_.back();
  }

  const OpBase& back() const { return ops_.back(); }

  OpBase& back() { return ops_.back(); }

  OpBase& operator[](size_t idx) { return ops_[idx]; }

  const OpBase& operator[](size_t idx) const { return ops_[idx]; }

  /* Iterator related */
  using Iterator = std::vector<OpBase>::iterator;
  using ConstIterator = std::vector<OpBase>::const_iterator;

  Iterator begin() { return ops_.begin(); }

  Iterator end() { return ops_.end(); }

  ConstIterator begin() const { return ops_.begin(); }

  ConstIterator end() const { return ops_.end(); }

  void InsertGradPendingNode(const std::shared_ptr<GradOpNode>& node) {
    if (node &&
        std::find(grad_pending_nodes_.begin(), grad_pending_nodes_.end(),
                  node) == grad_pending_nodes_.end()) {
      grad_pending_nodes_.emplace_back(node);
    }
  }

  void SetInplaceGradNameMap(
      const std::map<std::string, std::string>& inplace_input_map) {
    for (auto& pair : inplace_input_map) {
      VLOG(10) << "Set mapping relationship ("
               << framework::GradVarName(pair.first) << ", "
               << framework::GradVarName(pair.second)
               << ") for Inplace grad node.";
      inplace_grad_name_map_[framework::GradVarName(pair.first)] =
          framework::GradVarName(pair.second);
    }
  }

  const std::map<std::string, std::string>& InplaceGradNameMap() const {
    return inplace_grad_name_map_;
  }

  const std::vector<std::shared_ptr<GradOpNode>>& GradPendingNodes() const {
    return grad_pending_nodes_;
  }

 private:
  DISABLE_COPY_AND_ASSIGN(GradOpNode);

 private:
  std::vector<OpBase> ops_;
  std::vector<std::shared_ptr<GradOpNode>> grad_pending_nodes_;
  // Mapping relationship between grad output and grad input of the grad node of
  // Inplace op.
  std::map<std::string, std::string> inplace_grad_name_map_;
};

}  // namespace imperative
}  // namespace paddle
