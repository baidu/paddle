/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once

#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "paddle/fluid/framework/attribute.h"
#include "paddle/fluid/framework/op_info.h"
#include "paddle/fluid/framework/type_defs.h"
#include "paddle/fluid/framework/var_desc.h"

namespace paddle {
namespace framework {

class BlockDesc;
class ProgramDesc;

// If an op's OpInfo has proto, it is a forward op; otherwise, it is a grad op
static inline bool IsOpInfoHasProto(std::string type) {
  auto &op_proto = paddle::framework::OpInfoMap::Instance().Get(type).proto_;
  return op_proto != nullptr;
}

class OpDesc {
 public:
  OpDesc() {}

  OpDesc(const std::string &type, const VariableNameMap &inputs,
         const VariableNameMap &outputs, const AttributeMap &attrs);

  OpDesc(const proto::OpDesc &desc, BlockDesc *block);

  explicit OpDesc(BlockDesc *block) : block_(block) {}

  OpDesc(const OpDesc &other, BlockDesc *block);

  void CopyFrom(const OpDesc &op_desc);

  proto::OpDesc *Proto();

  std::string Type() const { return desc_.type(); }

  void SetType(const std::string &type) {
    desc_.set_type(type);
    InitOrderedInputOutputNamesForForwardOp();
  }

  const std::vector<std::string> &Input(const std::string &name) const;

  std::vector<std::string> InputArgumentNames() const;

  void SetInput(const std::string &param_name,
                const std::vector<std::string> &args);

  const std::vector<std::string> &Output(const std::string &name) const;

  bool HasOutput(const std::string &name) const;

  std::vector<std::string> OutputArgumentNames() const;

  void SetOutput(const std::string &param_name,
                 const std::vector<std::string> &args);

  bool HasAttr(const std::string &name) const {
    return attrs_.find(name) != attrs_.end();
  }

  bool HasProtoAttr(const std::string &name) const;

  proto::AttrType GetAttrType(const std::string &name) const;

  std::vector<std::string> AttrNames() const;

  void SetAttr(const std::string &name, const Attribute &v);
  void RemoveAttr(const std::string &name);

  void SetBlockAttr(const std::string &name, BlockDesc *block);

  void SetBlocksAttr(const std::string &name, std::vector<BlockDesc *> blocks);

  Attribute GetAttr(const std::string &name) const;

  template <typename T>
  T GetAttrIfExists(const std::string &name) const {
    T result{};
    if (HasAttr(name)) {
      result = BOOST_GET_CONST(T, GetAttr(name));
    }
    return result;
  }

  const proto::OpProto::Attr &GetProtoAttr(const std::string &name) const;

  Attribute GetNullableAttr(const std::string &name) const;

  int GetBlockAttrId(const std::string &name) const;

  std::vector<int> GetBlocksAttrIds(const std::string &name) const;

  void Rename(const std::string &old_name, const std::string &new_name);

  void RenameOutput(const std::string &old_name, const std::string &new_name);

  void RenameInput(const std::string &old_name, const std::string &new_name);

  // Only be used in C++
  const AttributeMap &GetAttrMap() const;

  // Only be used in C++
  void SetAttrMap(const AttributeMap &attr_map);

  std::vector<std::string> InputNames() const { return MapKeys(inputs_); }
  std::vector<std::string> OutputNames() const { return MapKeys(outputs_); }

  const VariableNameMap &Inputs() const { return inputs_; }

  const VariableNameMap &Outputs() const { return outputs_; }

  AttributeMap *MutableAttrMap() {
    this->need_update_ = true;
    return &this->attrs_;
  }

  void CheckAttrs();

  void InferShape(const BlockDesc &block) const;

  void InferVarType(BlockDesc *block) const;

  void SetIsTarget(bool is_target) { desc_.set_is_target(is_target); }

  void Flush();

  BlockDesc *Block() { return this->block_; }

  const BlockDesc *Block() const { return this->block_; }

  const std::vector<std::string> &OrderedInputNames() const {
    return input_names_;
  }

  const std::vector<std::string> &OrderedOutputNames() const {
    return output_names_;
  }

  void InitOrderedInputOutputNamesForForwardOp() {
    if (IsOpInfoHasProto(Type())) {
      auto &op_proto =
          paddle::framework::OpInfoMap::Instance().Get(Type()).proto_;
      if (input_names_.empty()) {
        for (auto &in : op_proto->inputs()) {
          input_names_.emplace_back(in.name());
        }
      }
      if (output_names_.empty()) {
        for (auto &out : op_proto->outputs()) {
          output_names_.emplace_back(out.name());
        }
      }
    }
  }

  std::string GetInputNameByIdx(size_t idx) const {
    PADDLE_ENFORCE_LT(idx, input_names_.size(),
                      platform::errors::OutOfRange(
                          "The index should be less than the size of inputs of "
                          "operator %s, but got index is %d and size is %d",
                          Type(), idx, input_names_.size()));
    return input_names_[idx];
  }

  std::string GetOutputNameByIdx(size_t idx) const {
    PADDLE_ENFORCE_LT(
        idx, output_names_.size(),
        platform::errors::OutOfRange(
            "The index should be less than the size of outputs of "
            "operator %s, but got index is %d and size is %d",
            Type(), idx, output_names_.size()));
    return output_names_[idx];
  }

 private:
  template <typename MapType>
  static std::vector<typename MapType::key_type> MapKeys(const MapType &map) {
    std::vector<typename MapType::key_type> ret_val;
    ret_val.reserve(map.size());
    std::transform(
        map.begin(), map.end(), std::back_inserter(ret_val),
        [](const typename MapType::value_type &pair) { return pair.first; });
    return ret_val;
  }

  proto::OpDesc desc_;
  BlockDesc *block_;  // not_own
  // input arg name => input variable names
  VariableNameMap inputs_;
  // output arg name => output variable names
  VariableNameMap outputs_;
  // keep input/output order, so we can get name by idx
  std::vector<std::string> input_names_;
  std::vector<std::string> output_names_;
  AttributeMap attrs_;

  // need_update_ indicate there some local changes not be synchronized. If
  // local changes should be synchronized, need_update_ should be set to true.
  bool need_update_{false};
};
}  // namespace framework
}  // namespace paddle
