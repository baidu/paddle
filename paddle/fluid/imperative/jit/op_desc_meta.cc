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

#include "paddle/fluid/imperative/jit/op_desc_meta.h"

#include <utility>

namespace paddle {
namespace imperative {
namespace jit {

OpDescMeta::OpDescMeta(std::string type, const NameVarBaseMap &inputs,
                       const NameVarBaseMap &outputs,
                       framework::AttributeMap attrs)
    : type_(std::move(type)), attrs_(std::move(attrs)) {
  auto *proto = framework::OpInfoMap::Instance().GetNullable(type_);
  if (proto && proto->Checker()) {
    proto->Checker()->Check(&attrs_);
  }

  for (auto &pair : inputs) {
    inputs_[pair.first].assign(pair.second.begin(), pair.second.end());
  }

  for (auto &pair : outputs) {
    outputs_[pair.first].assign(pair.second.begin(), pair.second.end());
  }
}

const std::string &OpDescMeta::Type() const { return type_; }

const WeakNameVarBaseMap &OpDescMeta::Inputs() const { return inputs_; }

const WeakNameVarBaseMap &OpDescMeta::Outputs() const { return outputs_; }

const framework::AttributeMap &OpDescMeta::Attrs() const { return attrs_; }

}  // namespace jit
}  // namespace imperative
}  // namespace paddle
