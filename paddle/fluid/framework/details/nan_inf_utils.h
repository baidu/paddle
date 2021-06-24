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
#include <string>
#include <vector>

#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/imperative/type_defs.h"
#include "paddle/fluid/platform/place.h"

namespace paddle {
namespace framework {
namespace details {
// assert false when meets NAN or inf
void CheckVarHasNanOrInf(const std::string& op_type,
                         const framework::Scope& scope,
                         const std::string& var_name,
                         const platform::Place& place);

void CheckVarHasNanOrInf(const std::string& op_type,
                         const std::string& var_name,
                         const framework::Variable* var,
                         const platform::Place& place);

void CheckOpHasNanOrInf(const framework::OperatorBase& op,
                        const framework::Scope& scope,
                        const platform::Place& place);

template <typename VarType>
void CheckOpHasNanOrInfInDygraph(const std::string& op_type,
                                 const imperative::NameVarMap<VarType>& op_outs,
                                 platform::Place place) {
  for (const auto& pair : op_outs) {
    for (const auto& ivar : pair.second) {
      auto* var = ivar->MutableVar();
      if (var == nullptr) continue;
      CheckVarHasNanOrInf(op_type, ivar->Name(), var, place);
    }
  }
}

}  // namespace details
}  // namespace framework
}  // namespace paddle
