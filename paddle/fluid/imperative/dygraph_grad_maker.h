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

#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "paddle/fluid/imperative/layer.h"
#include "paddle/fluid/imperative/op_base.h"
#include "paddle/fluid/imperative/type_defs.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/fluid/platform/macros.h"

extern uint64_t time_TracedGradOp_attr;
extern uint64_t time_TracedGradOp_in;
extern uint64_t time_TracedGradOp_out;
extern uint64_t time_find_in;
extern uint64_t time_find_out;
extern uint64_t time_find_in_grad;
extern uint64_t time_find_out_grad;
extern uint64_t lala_start_time;

namespace paddle {
namespace imperative {

enum TracedVarRole { kForward = 0, kBackward = 1 };

template <typename T, TracedVarRole kRole>
class TracedVarList : public std::vector<std::shared_ptr<T>> {
 private:
  using BaseClass = std::vector<std::shared_ptr<T>>;

 public:
  using BaseClass::BaseClass;
};

inline uint64_t GetPosixInUsec3() {
  struct timeval tv;
  gettimeofday(&tv, nullptr);
  return (static_cast<uint64_t>(tv.tv_sec) * 1000000 + tv.tv_usec);
}
class tmpclass3;
extern tmpclass3 aaaa;

class GradOpBaseMakerBase {
 public:
  explicit GradOpBaseMakerBase(
      const std::string& type, const NameVarBaseMap& var_base_map_in,
      const NameVarBaseMap& var_base_map_out,
      const framework::AttributeMap& attrs,
      const std::map<std::string, std::string>& inplace_map)
      : type_(type),
        var_base_map_in_(var_base_map_in),
        var_base_map_out_(var_base_map_out),
        attrs_(attrs),
        inplace_map_(inplace_map) {}

  virtual ~GradOpBaseMakerBase() = default;

  virtual std::shared_ptr<GradOpNode> operator()() const = 0;

  TracedVarList<VarBase, TracedVarRole::kBackward> InputGrad(
      const std::string& name, bool drop_empty_grad = true) const {
    lala_start_time = GetPosixInUsec3();
    auto ret =
        GetVarBaseList<TracedVarRole::kBackward>(name, /*is_input=*/true);
    time_find_in_grad = GetPosixInUsec3() - lala_start_time + time_find_in_grad;
    return ret;
  }

  TracedVarList<VarBase, TracedVarRole::kBackward> OutputGrad(
      const std::string& name) const {
    lala_start_time = GetPosixInUsec3();
    auto ret =
        GetVarBaseList<TracedVarRole::kBackward>(name, /*is_input=*/false);
    time_find_out_grad =
        GetPosixInUsec3() - lala_start_time + time_find_out_grad;
    return ret;
  }

  TracedVarList<VarBase, TracedVarRole::kForward> Input(
      const std::string& name) const {
    lala_start_time = GetPosixInUsec3();
    auto ret = GetVarBaseList<TracedVarRole::kForward>(name, /*is_input=*/true);
    time_find_in = GetPosixInUsec3() - lala_start_time + time_find_in;
    return ret;
  }

  TracedVarList<VarBase, TracedVarRole::kForward> Output(
      const std::string& name) const {
    lala_start_time = GetPosixInUsec3();
    auto ret =
        GetVarBaseList<TracedVarRole::kForward>(name, /*is_input=*/false);
    time_find_out = GetPosixInUsec3() - lala_start_time + time_find_out;
    return ret;
  }

  static TracedVarList<VarBase, TracedVarRole::kForward> EmptyInput() {
    return {};
  }

  static TracedVarList<VarBase, TracedVarRole::kForward> EmptyOutput() {
    return {};
  }

  static TracedVarList<VarBase, TracedVarRole::kBackward> EmptyOutputGrad() {
    return {};
  }

  static TracedVarList<VarBase, TracedVarRole::kBackward> EmptyInputGrad() {
    return {};
  }

  std::vector<std::string> InputNames() const {
    std::vector<std::string> vec_temp;
    vec_temp.reserve(var_base_map_in_.size());
    for (auto& it : var_base_map_in_) {
      vec_temp.emplace_back(it.first);
    }
    return vec_temp;
  }

  std::vector<std::string> OutputNames() const {
    std::vector<std::string> vec_temp;
    vec_temp.reserve(var_base_map_out_.size());
    for (auto& it : var_base_map_out_) {
      vec_temp.emplace_back(it.first);
    }
    return vec_temp;
  }

  const framework::AttributeMap& Attrs() const { return attrs_; }

  const framework::Attribute& GetAttr(const std::string& name) const {
    auto it = attrs_.find(name);
    PADDLE_ENFORCE_EQ(
        it != attrs_.end(), true,
        platform::errors::NotFound(
            "Cannot find attribute [%s] in operator [%s]", name, type_));
    return it->second;
  }

  template <typename T>
  inline const T& Attr(const std::string& name) const {
    return BOOST_GET_CONST(T, GetAttr(name));
  }

  const std::string& ForwardOpType() const { return type_; }

 protected:
  bool HasInput(const std::string& name) const {
    return var_base_map_in_.count(name) > 0;
  }

  bool HasOutput(const std::string& name) const {
    return var_base_map_out_.count(name) > 0;
  }

  static std::shared_ptr<GradOpNode> NewGradNode() {
    return std::make_shared<GradOpNode>();
  }

  const std::map<std::string, std::string>& GetInplaceMap() const {
    return inplace_map_;
  }

 private:
  template <TracedVarRole kRole>
  TracedVarList<VarBase, kRole> GetVarBaseList(const std::string& name,
                                               bool is_input) const {
    const auto& data_map = is_input ? var_base_map_in_ : var_base_map_out_;
    auto iterator = data_map.find(name);
    TracedVarList<VarBase, kRole> vec_temp;
    if (iterator != data_map.end()) {
      vec_temp.reserve(iterator->second.size());

      bool is_valid = false;
      for (auto& var_base_temp : iterator->second) {
        if (!var_base_temp) {
          vec_temp.emplace_back();
          continue;
        }

        if (kRole == TracedVarRole::kBackward) {
          if (!var_base_temp->HasGradVar()) {
            VLOG(6) << "GradVarBase of var " << var_base_temp->Name()
                    << " in OP " << type_ << " is null";
            var_base_temp->MutableGradVarBase();
          }
          auto grad_var_base_tmp = var_base_temp->GradVarBase();

          if (!is_input) {
            auto* tensor = grad_var_base_tmp->MutableVar()
                               ->GetMutable<framework::LoDTensor>();
            tensor->Resize(
                var_base_temp->Var().Get<framework::LoDTensor>().dims());
          }
          vec_temp.emplace_back(grad_var_base_tmp);
        } else {
          vec_temp.emplace_back(var_base_temp);
        }
        is_valid = true;
      }

      if (!is_valid) {
        vec_temp.clear();
      }
    }

    return vec_temp;
  }

 private:
  const std::string& type_;
  const NameVarBaseMap& var_base_map_in_;
  const NameVarBaseMap& var_base_map_out_;
  const framework::AttributeMap& attrs_;
  const std::map<std::string, std::string>& inplace_map_;
};

class TracedGradOp {
  DISABLE_COPY_AND_ASSIGN(TracedGradOp);

 public:
  explicit TracedGradOp(const std::shared_ptr<GradOpNode>& node)
      : node_(node), op_(&(node->emplace_back())) {}

  ~TracedGradOp() {
    if (UNLIKELY(op_->GetOutsMap().empty())) {
      node_->pop_back();
    } else {
      op_->CheckAttrs();
    }
  }

  template <TracedVarRole kRole>
  void SetInput(const std::string& name,
                const TracedVarList<VarBase, kRole>& vars) {
    if (vars.empty()) {
      return;
    }
    lala_start_time = GetPosixInUsec3();
    if (kRole == TracedVarRole::kBackward) {
      for (auto& var : vars) {
        if (var && !var->OverridedStopGradient()) {
          var->SetGraphIsFreed(false);
          auto dirty_grad_node = var->GradNode();
          if (dirty_grad_node) {
            map_dirty_grad_node_[var] = dirty_grad_node;
          }
          var->SetGradNode(node_);
        }
      }
    }

    auto var_wrappers = ToVarWrapperList<kRole>(vars);

    if (!var_wrappers.empty()) {
      op_->SetInput(name, std::move(var_wrappers),
                    kRole == TracedVarRole::kBackward);
    }
    time_TracedGradOp_in =
        GetPosixInUsec3() - lala_start_time + time_TracedGradOp_in;
  }

  template <TracedVarRole kRole>
  void SetOutput(const std::string& name,
                 const TracedVarList<VarBase, kRole>& vars) {
    if (vars.empty()) {
      return;
    }
    lala_start_time = GetPosixInUsec3();
    if (kRole == TracedVarRole::kBackward) {
      if (vars.size() == 1 && vars.front()->OverridedStopGradient()) {
        return;
      } else {
        for (auto& var : vars) {
          if (var && !var->OverridedStopGradient() && var->GradNode()) {
            if (map_dirty_grad_node_.find(var) != map_dirty_grad_node_.end()) {
              node_->InsertGradPendingNode(map_dirty_grad_node_[var]);
            } else {
              node_->InsertGradPendingNode(var->GradNode());
            }
          }
        }
      }
    }

    auto var_wrappers = ToVarWrapperList<kRole>(vars);
    if (!var_wrappers.empty()) {
      op_->SetOutput(name, std::move(var_wrappers),
                     kRole == TracedVarRole::kBackward);
    }
    time_TracedGradOp_out =
        GetPosixInUsec3() - lala_start_time + time_TracedGradOp_out;
  }

  std::string Type() const { return op_->Type(); }

  void SetType(const std::string& type) { op_->SetType(type); }

  const framework::OperatorBase& InnerOp() const { return op_->InnerOp(); }

  void SetAttrMap(const framework::AttributeMap& attrs) {
    lala_start_time = GetPosixInUsec3();
    op_->SetAttrMap(attrs);
    time_TracedGradOp_attr =
        GetPosixInUsec3() - lala_start_time + time_TracedGradOp_attr;
  }

  void SetAttr(const std::string& name, const framework::Attribute& v) {
    op_->SetAttr(name, v);
  }

  bool HasAttr(const std::string& name) const { return op_->HasAttr(name); }

  const framework::Attribute& GetAttr(const std::string& name) const {
    return op_->GetAttr(name);
  }

  template <typename T>
  inline const T& Attr(const std::string& name) const {
    return op_->Attr<T>(name);
  }

 private:
  template <TracedVarRole kRole>
  static std::vector<std::shared_ptr<VariableWrapper>> ToVarWrapperList(
      const std::vector<std::shared_ptr<VarBase>>& vars) {
    std::vector<std::shared_ptr<VariableWrapper>> result;
    result.reserve(vars.size());
    bool has_valid = false;
    for (auto& var : vars) {
      if (UNLIKELY(!var || (kRole == TracedVarRole::kBackward &&
                            var->OverridedStopGradient()))) {
        result.emplace_back();
      } else {
        auto var_wrapper = SnapshotVarWrapper(var->SharedVar());
        result.emplace_back(var_wrapper);
        has_valid = true;
      }
    }

    if (!has_valid) {
      result.clear();
    }
    return result;
  }

  // Get a snapshot of VariableWrapper at a certain inplace version.
  // The inplace version number of VariableWrapper is used for inplace
  // detection in gradient compution.
  static const std::shared_ptr<VariableWrapper> SnapshotVarWrapper(
      const std::shared_ptr<VariableWrapper>& var_wrapper) {
    // NOTE(liym27):
    //  Use original var_wrapper if its inplace_version is not
    //  changed. Otherwise, it will affect the accuracy of the model
    //  results and affect double grad.
    if (!var_wrapper->MutableVar()->IsInitialized() ||
        var_wrapper->InplaceVersionSnapshot() ==
            var_wrapper->MutableVar()->CurrentInplaceVersion()) {
      return var_wrapper;
    } else {
      VariableWrapper new_var_wrapper = *var_wrapper.get();
      new_var_wrapper.ResetInplaceVersion();
      return std::make_shared<VariableWrapper>(new_var_wrapper);
    }
  }

 private:
  const std::shared_ptr<GradOpNode>& node_;
  OpBase* op_;
  // Inplace op has recursion problems when performing grad calculation.
  // Because the input and output of inplace op are the same, the grad
  // node of inplace var will be overwritten.
  // This map is used to store the grad node of inplace var in temporary.
  std::unordered_map<std::shared_ptr<VarBase>, std::shared_ptr<GradOpNode>>
      map_dirty_grad_node_;
};

}  // namespace imperative
}  // namespace paddle
