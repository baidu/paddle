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
class SavedVariableWrapperList;
class VarBase;
class OpBase;
class GradOpNode;
class Tracer;

// test for prec
using WeakNameVarBaseMap =
    std::map<std::string, std::vector<std::weak_ptr<VarBase>>>;

namespace details {
template <typename T>
struct NameVarMapTrait {};

template <>
struct NameVarMapTrait<VarBase> {
  using Type = std::map<std::string, std::vector<std::shared_ptr<VarBase>>>;
};

template <>
struct NameVarMapTrait<VariableWrapper> {
  using Type = std::map<std::string, SavedVariableWrapperList>;
};
}  // namespace details

template <typename T>
using NameVarMap = typename details::NameVarMapTrait<T>::Type;

using NameVarBaseMap = NameVarMap<VarBase>;
using NameVariableWrapperMap = NameVarMap<VariableWrapper>;

using VariableWrapperList = std::vector<std::shared_ptr<VariableWrapper>>;

}  // namespace imperative
}  // namespace paddle
