/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

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
#include <paddle/utils/Any.h>
#include <string>
#include <unordered_map>

namespace paddle {
namespace topology {

class Attribute : public std::unordered_map<std::string, any> {};

class WithAttribute {
public:
  Attribute attributes;

  template <typename T>
  const T& getAttr(const std::string& name) const {
    auto attrPtr = &attributes.find(name)->second;
    return *any_cast<T>(attrPtr);
  }

  template <typename T>
  T& getAttr(const std::string& name) {
    auto attrPtr = &attributes.find(name)->second;
    return *any_cast<T>(attrPtr);
  }
};
}  // namespace topology
}  // namespace paddle
