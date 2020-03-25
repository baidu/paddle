/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.

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

#include <string>
#include <vector>
#include "paddle/fluid/operators/math/math_function.h"
#include "paddle/fluid/platform/device_context.h"

namespace paddle {
namespace operators {
namespace math {

template <typename DeviceContext, typename T>
class FCFunctor {
 public:
  void operator()(const DeviceContext& context, const int M, const int N,
                  const int K, const T* X, const T* W, T* Y,
                  const T* B = nullptr, bool relu = false,
                  bool weight_pass = false);
};

template <typename DeviceContext>
class FCInt8Functor;

template <>
class FCInt8Functor<platform::CPUDeviceContext> {
 public:
  void operator()(const platform::CPUDeviceContext& context, int M, int N,
                  int K, const framework::Tensor& in,
                  const framework::Tensor& W, framework::Tensor* Y, float scale,
                  std::vector<float> weight_scale, const framework::Tensor& B,
                  bool relu = false, bool weight_pass = false) {}
};

#ifdef PADDLE_WITH_CUDA
template <>
class FCInt8Functor<platform::CUDADeviceContext> {
 public:
  void operator()(const platform::CUDADeviceContext& context, int M, int N,
                  int K, const framework::Tensor& in,
                  const framework::Tensor& W, framework::Tensor* Y,
                  float in_scale, std::vector<float> weight_scale,
                  const framework::Tensor& B, bool relu = false,
                  bool weight_pass = false);
};
#endif

}  // namespace math
}  // namespace operators
}  // namespace paddle
