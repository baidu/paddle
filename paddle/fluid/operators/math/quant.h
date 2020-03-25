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
#include <vector>
#include "paddle/fluid/operators/math/math_function.h"
#include "paddle/fluid/platform/cudnn_helper.h"
#include "paddle/fluid/platform/transform.h"

namespace paddle {
namespace operators {
namespace math {

template <typename DeviceContext>
struct QuantFp32ToInt8Functor;

template <typename DeviceContext>
struct GEMMINT8Functor;

template <typename DeviceContext>
struct INT32ToFP32Functor;

template <>
struct QuantFp32ToInt8Functor<platform::CPUDeviceContext> {
  void operator()(const platform::CPUDeviceContext& ctx,
                  const framework::Tensor& in, const float scale,
                  framework::Tensor* out) {}
};

template <>
struct GEMMINT8Functor<platform::CPUDeviceContext> {
  void operator()(const platform::CPUDeviceContext& ctx, bool transA,
                  bool transB, int M, int N, int K, float alpha,
                  const int8_t* A, int lda, const int8_t* B, int ldb,
                  float beta, float* C, int ldc) {}
  void operator()(const platform::CPUDeviceContext& ctx, bool transA,
                  bool transB, int M, int N, int K, int32_t alpha,
                  const int8_t* A, int lda, const int8_t* B, int ldb,
                  int32_t beta, int32_t* C, int ldc) {}
};

template <>
struct INT32ToFP32Functor<platform::CPUDeviceContext> {
  void operator()(const platform::CPUDeviceContext& ctx,
                  const framework::Tensor& in, framework::Tensor* out,
                  float scale) {}
};

#ifdef PADDLE_WITH_CUDA

template <>
struct QuantFp32ToInt8Functor<platform::CUDADeviceContext> {
  void operator()(const platform::CUDADeviceContext& ctx,
                  const framework::Tensor& in, const float scale,
                  framework::Tensor* out);
};

template <>
struct GEMMINT8Functor<platform::CUDADeviceContext> {
  void operator()(const platform::CUDADeviceContext& ctx, bool transA,
                  bool transB, int M, int N, int K, float alpha,
                  const int8_t* A, int lda, const int8_t* B, int ldb,
                  float beta, float* C, int ldc);
  void operator()(const platform::CUDADeviceContext& ctx, bool transA,
                  bool transB, int M, int N, int K, int32_t alpha,
                  const int8_t* A, int lda, const int8_t* B, int ldb,
                  int32_t beta, int32_t* C, int ldc);
};

template <>
struct INT32ToFP32Functor<platform::CUDADeviceContext> {
  void operator()(const platform::CUDADeviceContext& ctx,
                  const framework::Tensor& in, framework::Tensor* out,
                  float scale);
};

#endif

}  // namespace math
}  // namespace operators
}  // namespace paddle
