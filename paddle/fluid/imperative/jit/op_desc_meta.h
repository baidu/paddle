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

#include <string>

#include "paddle/fluid/imperative/layer.h"
#include "paddle/fluid/imperative/type_defs.h"

namespace paddle {
namespace imperative {
namespace jit {

class OpDescMeta {
 public:
  OpDescMeta(std::string type, const NameVarBaseMap &inputs,
             const NameVarBaseMap &outputs, framework::AttributeMap attrs);

  const std::string &Type() const;

  const WeakNameVarBaseMap &Inputs() const;

  const WeakNameVarBaseMap &Outputs() const;

  const framework::AttributeMap &Attrs() const;

 private:
  std::string type_;
  WeakNameVarBaseMap inputs_;
  WeakNameVarBaseMap outputs_;
  framework::AttributeMap attrs_;
};

}  // namespace jit
}  // namespace imperative
}  // namespace paddle
