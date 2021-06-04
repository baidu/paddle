/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/platform/lodtensor_printer.h"

#include "paddle/fluid/framework/scope.h"

namespace paddle {
namespace framework {
class LoDTensor;
class Variable;
}  // namespace framework
}  // namespace paddle

namespace paddle {
namespace platform {

void PrintVar(framework::Scope* scope, const std::string& var_name,
              const std::string& print_info, std::stringstream* sstream) {
  framework::Variable* var = scope->FindVar(var_name);
  if (var == nullptr) {
    VLOG(0) << "Variable Name " << var_name << " does not exist in your scope";
    return;
  }
  auto* tensor = var->GetMutable<framework::LoDTensor>();
  if (tensor == nullptr) {
    VLOG(0) << "tensor of variable " << var_name
            << " does not exist in your scope";
    return;
  }

  *sstream << print_info << ": ";

#define PrintTensorCallback(cpp_type, proto_type) \
  do {                                            \
    if (tensor->type() == proto_type) {           \
      *sstream << "[";                            \
      auto* data = tensor->data<cpp_type>();      \
      auto element_num = tensor->numel();         \
      if (element_num > 0) {                      \
        *sstream << data[0];                      \
        for (int j = 1; j < element_num; ++j) {   \
          *sstream << " " << data[j];             \
        }                                         \
      }                                           \
      *sstream << "]";                            \
    }                                             \
  } while (0)

  _ForEachDataType_(PrintTensorCallback);
}

}  // end namespace platform
}  // end namespace paddle
