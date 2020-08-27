/* Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.

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

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/operator.h"

// This file almostly contains all the coefficient-wise Operator class and
// OpKernel class

namespace paddle {
namespace operators {
namespace functors {

// binary functor
template <typename T>
struct Pow {
  inline HOSTDEVICE T operator()(T a, T b) const {
#ifdef __CUDA_ARCH__
    // On CUDAPlace, std::pow(3, 1) calls pow(float, float), and
    // it will return a float number like 2.99... , which floor to 2
    // when cast to int by default and it is wrong.
    // Use llrint to cast it to the nearest integer, which is 3.
    if (std::is_integral<T>::value) {
      return std::llrint(std::pow(a, b));
    }
#endif
    return std::pow(a, b);
  }
};

template <typename T>
struct Add {
  inline HOSTDEVICE T operator()(T a, T b) const { return a + b; }
};

template <typename T>
struct Sub {
  inline HOSTDEVICE T operator()(T a, T b) const { return a - b; }
};

template <typename T>
struct Mul {
  inline HOSTDEVICE T operator()(T a, T b) const { return a * b; }
};

template <typename T>
struct Div {
  inline HOSTDEVICE T operator()(T a, T b) const { return a / b; }
};

// unary functor
template <typename T>
struct IsNan {
  inline HOSTDEVICE T operator()(T a) const { return std::isnan(a); }
};

template <typename T>
struct Neg {
  inline HOSTDEVICE T operator()(T a) const { return -a; }
};

}  // namespace functors
}  // namespace operators
}  // namespace paddle
