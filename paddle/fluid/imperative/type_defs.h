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

#include <map>
#include <memory>
#include <string>
#include <vector>

namespace paddle {
namespace imperative {

class VariableWrapper;
class VarBase;
class OpBase;
class Tracer;

template <typename T>
using NameVarMap = std::map<std::string, std::vector<std::shared_ptr<T>>>;

using NameVarBaseMap = NameVarMap<VarBase>;
using NameVariableWrapperMap = NameVarMap<VariableWrapper>;

using WeakNameVarBaseMap =
    std::map<std::string, std::vector<std::weak_ptr<VarBase>>>;

}  // namespace imperative
}  // namespace paddle
