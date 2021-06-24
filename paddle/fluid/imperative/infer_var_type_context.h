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

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
#include "paddle/fluid/framework/type_defs.h"
#include "paddle/fluid/framework/var_type_inference.h"
#include "paddle/fluid/imperative/type_defs.h"
#include "paddle/fluid/imperative/variable_wrapper.h"

namespace paddle {
namespace imperative {

// infer var type context for imperative mode
template <typename VarType>
class RuntimeInferVarTypeContext : public framework::InferVarTypeContext {
 public:
  RuntimeInferVarTypeContext(const NameVarMap<VarType>& inputs,
                             const NameVarMap<VarType>& outputs,
                             const framework::AttributeMap& attrs_map,
                             const framework::AttributeMap& default_attrs_map)
      : InferVarTypeContext(nullptr, nullptr),
        inputs_(inputs),
        outputs_(outputs),
        attrs_(attrs_map),
        default_attrs_(default_attrs_map) {}

  virtual ~RuntimeInferVarTypeContext() {}

  framework::Attribute GetAttr(const std::string& name) const override {
    auto it = attrs_.find(name);

    if (it == attrs_.end()) {
      it = default_attrs_.find(name);
      if (it == default_attrs_.end()) {
        PADDLE_THROW(platform::errors::NotFound(
            "Can not find [%s] in attributes.", name));
      }
    }

    return it->second;
  }

  bool HasInput(const std::string& name) const override {
    auto it = inputs_.find(name);
    return (it != inputs_.end() && it->second.size() > 0);
  }

  bool HasOutput(const std::string& name) const override {
    auto it = outputs_.find(name);
    return (it != outputs_.end() && it->second.size() > 0);
  }

  size_t InputSize(const std::string& name) const {
    return inputs_.at(name).size();
  }

  const std::string& InputVarName(const std::string& name,
                                  const int index = 0) const {
    return inputs_.at(name)[index]->Name();
  }

  bool InputTypeAnyOf(const std::string& name,
                      framework::proto::VarType::Type type) const override {
    auto& inputs = inputs_.at(name);
    return std::any_of(inputs.begin(), inputs.end(),
                       [&type](const std::shared_ptr<VarType>& var) {
                         return var->Type() == type;
                       });
  }

  bool InputTypeAllOf(const std::string& name,
                      framework::proto::VarType::Type type) const override {
    auto& inputs = inputs_.at(name);
    return std::all_of(inputs.begin(), inputs.end(),
                       [&type](const std::shared_ptr<VarType>& var) {
                         return var->Type() == type;
                       });
  }

  void SyncTypeAndDataType(const std::string& input_name,
                           const std::string& output_name,
                           int index = 0) override {
    auto in_var = inputs_.at(input_name)[index];
    auto out_var = outputs_.at(output_name)[index];
    if (in_var != out_var) {
      this->SetVarBaseType(out_var, in_var->Type());
      this->SetVarBaseDataType(out_var, in_var->DataType());
    }
  }

  void SetOutputType(const std::string& name,
                     framework::proto::VarType::Type type,
                     int index = 0) override {
    if (index == framework::ALL_ELEMENTS) {
      for (auto& item : outputs_.at(name)) {
        this->SetVarBaseType(item, type);
      }
    } else {
      auto& var = outputs_.at(name)[index];
      this->SetVarBaseType(var, type);
    }
  }

  void SetVarBaseType(std::shared_ptr<VarType> out,
                      framework::proto::VarType::Type type) {
    out->SetType(type);
    if ((out->MutableVar()->IsInitialized() == true) &&
        (out->MutableVar()->Type() != type)) {
      out->MutableVar()->Clear();
    }
  }

  void SetVarBaseDataType(std::shared_ptr<VarType> out,
                          framework::proto::VarType::Type type) {
    out->SetDataType(type);
  }

  framework::proto::VarType::Type GetInputType(
      const std::string& name, const int& index = 0) const override {
    return inputs_.at(name)[index]->Type();
  }

  framework::proto::VarType::Type GetOutputType(
      const std::string& name, const int& index = 0) const override {
    return outputs_.at(name)[index]->Type();
  }

  framework::proto::VarType::Type GetInputDataType(
      const std::string& name, const int& index = 0) const override {
    return inputs_.at(name)[index]->DataType();
  }

  void SetOutputDataType(const std::string& name,
                         framework::proto::VarType::Type type,
                         int index = 0) override {
    if (framework::ALL_ELEMENTS == index) {
      for (auto& item : outputs_.at(name)) {
        this->SetVarBaseDataType(item, type);
      }
    } else {
      auto& var = outputs_.at(name)[index];
      this->SetVarBaseDataType(var, type);
    }
  }

  bool IsDygraph() const override { return true; }

 protected:
  bool HasVar(const std::string& name) const override {
    PADDLE_THROW(platform::errors::PermissionDenied(
        "HasVar is not supported in runtime InferVarType"));
  }

  const std::vector<std::string>& InputVars(
      const std::string& name) const override {
    PADDLE_THROW(platform::errors::PermissionDenied(
        "InputVars is not supported in runtime InferVarType"));
  }

  const std::vector<std::string>& OutputVars(
      const std::string& name) const override {
    PADDLE_THROW(platform::errors::PermissionDenied(
        "OutputVars is not supported in runtime InferVarType"));
  }

  framework::proto::VarType::Type GetVarType(
      const std::string& name) const override {
    PADDLE_THROW(platform::errors::PermissionDenied(
        "Do not manipulate var in runtime InferVarType"));
  }

  void SetVarType(const std::string& name,
                  framework::proto::VarType::Type type) override {
    PADDLE_THROW(platform::errors::PermissionDenied(
        "Do not manipulate var in runtime InferVarType"));
  }

  framework::proto::VarType::Type GetVarDataType(
      const std::string& name) const override {
    PADDLE_THROW(platform::errors::PermissionDenied(
        "Do not manipulate var in runtime InferVarType"));
  }

  void SetVarDataType(const std::string& name,
                      framework::proto::VarType::Type type) override {
    PADDLE_THROW(platform::errors::PermissionDenied(
        "Do not manipulate var in runtime InferVarType"));
  }

  std::vector<framework::proto::VarType::Type> GetVarDataTypes(
      const std::string& name) const override {
    PADDLE_THROW(platform::errors::PermissionDenied(
        "GetVarDataTypes is not supported in runtime InferVarType"));
  }

  void SetVarDataTypes(const std::string& name,
                       const std::vector<framework::proto::VarType::Type>&
                           multiple_data_type) override {
    PADDLE_THROW(platform::errors::PermissionDenied(
        "SetVarDataTypes is not supported in runtime InferVarType"));
  }

  std::vector<int64_t> GetVarShape(const std::string& name) const override {
    PADDLE_THROW(platform::errors::PermissionDenied(
        "Do not handle Shape in runtime InferVarType"));
  }

  void SetVarShape(const std::string& name,
                   const std::vector<int64_t>& dims) override {
    PADDLE_THROW(platform::errors::PermissionDenied(
        "Do not handle Shape in runtime InferVarType"));
  }

  int32_t GetVarLoDLevel(const std::string& name) const override {
    PADDLE_THROW(platform::errors::PermissionDenied(
        "Do not handle LoDLevel in runtime InferVarType"));
  }

  void SetVarLoDLevel(const std::string& name, int32_t lod_level) override {
    PADDLE_THROW(platform::errors::PermissionDenied(
        "Do not handle LoDLevel in runtime InferVarType"));
  }

 private:
  const NameVarMap<VarType>& inputs_;
  const NameVarMap<VarType>& outputs_;
  const framework::AttributeMap& attrs_;
  const framework::AttributeMap& default_attrs_;
};

}  // namespace imperative
}  // namespace paddle
