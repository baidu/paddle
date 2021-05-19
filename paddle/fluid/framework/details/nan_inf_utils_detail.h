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

#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/platform/place.h"

namespace paddle {
namespace framework {
class Tensor;
}  // namespace framework
}  // namespace paddle

namespace paddle {
namespace framework {
namespace details {

template <typename DeviceContext>
struct TensorCheckerVisitor {
  TensorCheckerVisitor(const std::string& op_type, const std::string& var_name,
                       const framework::Tensor& tensor,
                       const platform::Place& place)
      : op_type_(op_type),
        var_name_(var_name),
        tensor_(tensor),
        place_(place) {}

  template <typename T>
  void apply(
      typename std::enable_if<std::is_integral<T>::value>::type* = 0) const {
    VLOG(10) << var_name_ << " need not to check, it's type is not float point";
  }

  template <typename T>
  void apply(
      typename std::enable_if<
          std::is_floating_point<T>::value ||
          std::is_same<T, ::paddle::platform::complex<float>>::value ||
          std::is_same<T, ::paddle::platform::complex<double>>::value>::type* =
          0) const;

  std::string op_type_;
  std::string var_name_;
  const framework::Tensor& tensor_;
  const platform::Place& place_;
};

template <typename DeviceContext>
void tensor_check(const std::string& op_type, const std::string& var_name,
                  const framework::Tensor& tensor,
                  const platform::Place& place);

}  // namespace details
}  // namespace framework
}  // namespace paddle
