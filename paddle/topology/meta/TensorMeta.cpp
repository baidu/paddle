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
#include "TensorMeta.h"

namespace paddle {
namespace topology {
namespace meta {

TensorMeta &TensorMeta::addShape(size_t dims) {
  addAttribute<std::vector<int>>("shape", "The shape of tensor")
      .mustSet()
      .dimsEq(dims);
  return *this;
}

TensorMeta &TensorMeta::addSequenceType(
    const std::unordered_set<SequenceType, std::hash<int>> &supportedTypes) {
  addAttribute<SequenceType>("sequence_type", "The sequence types of tensor")
      .mustSet()
      .in(supportedTypes);
  return *this;
}

TensorMeta &TensorMeta::addDataType(
    const std::unordered_set<DataType, std::hash<int>> &supportedTypes) {
  addAttribute<DataType>("data_type", "The data types of tensor")
      .mustSet()
      .in(supportedTypes);
  return *this;
}

}  // namespace meta
}  // namespace topology
}  // namespace paddle
