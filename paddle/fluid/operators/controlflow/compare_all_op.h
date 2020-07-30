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
#include <math.h>
#include <algorithm>
#include <type_traits>
#include <vector>
#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/elementwise/elementwise_op_function.h"
#include "paddle/fluid/platform/transform.h"

namespace paddle {
namespace operators {

template <typename T>
struct EqualReduceFunctor {
  using ELEM_TYPE = T;
  HOSTDEVICE bool operator()(const T& a, const T& b) const {
    if (std::is_floating_point<T>::value) {
      // This branch will be optimized while compiling if T is integer. It is
      // safe to cast a and b to double.
      return fabs(static_cast<double>(a - b)) < 1e-8;
    } else {
      return (a == b);
    }
  }
};

}  // namespace operators
}  // namespace paddle
