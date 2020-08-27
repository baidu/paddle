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

#include "paddle/fluid/framework/op_registry.h"

#include <vector>

namespace paddle {
namespace framework {

std::unique_ptr<OperatorBase> OpRegistry::CreateOp(
    const std::string& type, const VariableNameMap& inputs,
    const VariableNameMap& outputs, AttributeMap attrs,
    const std::vector<std::string>& input_names,
    const std::vector<std::string>& output_names, bool attr_check) {
  auto& info = OpInfoMap::Instance().Get(type);
  if (attr_check && info.Checker() != nullptr) {
    info.Checker()->Check(&attrs);
  }
  auto op = info.Creator()(type, inputs, outputs, attrs);
  op->SetOrderedInputNames(input_names);
  op->SetOrderedOutputNames(output_names);
  return std::unique_ptr<OperatorBase>(op);
}

static VariableNameMap ConvertOpDescVarsToVarNameMap(
    const google::protobuf::RepeatedPtrField<proto::OpDesc::Var>&
        op_desc_vars) {
  VariableNameMap ret_val;
  for (auto& var : op_desc_vars) {
    auto& var_names = ret_val[var.parameter()];
    auto& var_names_in_proto = var.arguments();
    var_names.reserve(static_cast<size_t>(var_names_in_proto.size()));
    std::copy(var_names_in_proto.begin(), var_names_in_proto.end(),
              std::back_inserter(var_names));
  }
  return ret_val;
}

std::unique_ptr<OperatorBase> OpRegistry::CreateOp(
    const proto::OpDesc& op_desc, const std::vector<std::string>& input_names,
    const std::vector<std::string>& output_names) {
  VLOG(1) << "CreateOp directly from OpDesc is deprecated. It should only be"
             "used in unit tests. Use CreateOp(const OpDesc& op_desc) "
             "instead.";
  VariableNameMap inputs = ConvertOpDescVarsToVarNameMap(op_desc.inputs());
  VariableNameMap outputs = ConvertOpDescVarsToVarNameMap(op_desc.outputs());
  AttributeMap attrs;
  for (auto& attr : op_desc.attrs()) {
    attrs[attr.name()] = GetAttrValue(attr);
  }

  return CreateOp(op_desc.type(), inputs, outputs, attrs, input_names,
                  output_names);
}

std::unique_ptr<OperatorBase> OpRegistry::CreateOp(const OpDesc& op_desc) {
  return CreateOp(op_desc.Type(), op_desc.Inputs(), op_desc.Outputs(),
                  op_desc.GetAttrMap(), op_desc.OrderedInputNames(),
                  op_desc.OrderedOutputNames());
}

std::unique_ptr<OperatorBase> OpRegistry::CreateOp(const std::string& type) {
  return CreateOp(type, {}, {}, {}, {}, {}, false);
}

}  // namespace framework
}  // namespace paddle
