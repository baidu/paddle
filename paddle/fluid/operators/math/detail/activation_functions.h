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
#include <stdexcept>
#include <string>
#include "paddle/fluid/platform/cpu_info.h"
#include "paddle/fluid/platform/hostdevice.h"

namespace paddle {
namespace operators {
namespace math {
namespace detail {

#define SIGMOID_THRESHOLD_MIN -40.0
#define SIGMOID_THRESHOLD_MAX 13.0
#define EXP_MAX_INPUT 40.0

enum ActivationType {
  kSigmoid,
  KSigmoidV2,
  kReLU,
  kTanh,
  kTanhV2,
  kIdentity,
};

inline ActivationType GetActivationType(const std::string &type) {
  if (type == "sigmoid") {
    return ActivationType::kSigmoid;
  } else if (type == "sigmoid_v2") {
    return ActivationType::KSigmoidV2;
  } else if (type == "relu") {
    return ActivationType::kReLU;
  } else if (type == "tanh") {
    return ActivationType::kTanh;
  } else if (type == "tanh_v2") {
    return ActivationType::kTanhV2;
  } else if (type == "identity" || type == "") {
    return ActivationType::kIdentity;
  }
  throw std::invalid_argument("The input type is not supported");
}

namespace forward {

template <typename T>
DEVICE T Identity(const T a) {
  return a;
}

template <typename T>
DEVICE T Relu(const T a) {
  return a > static_cast<T>(0.0) ? a : static_cast<T>(0.0);
}

template <typename T>
DEVICE T Sigmoid(const T a) {
  const T min = SIGMOID_THRESHOLD_MIN;
  const T max = SIGMOID_THRESHOLD_MAX;
  T tmp = (a < min) ? min : ((a > max) ? max : a);
  return static_cast<T>(1.0) / (static_cast<T>(1.0) + exp(-tmp));
}

template <typename T>
DEVICE T SigmoidV2(const T a) {
  return static_cast<T>(1.0) / (static_cast<T>(1.0) + exp(-a));
}

template <typename T>
DEVICE T Tanh(const T a) {
  T tmp = -2.0 * a;
  tmp = (tmp > EXP_MAX_INPUT) ? EXP_MAX_INPUT : tmp;
  return (2.0 / (1.0 + exp(tmp))) - 1.0;
}

template <typename T>
DEVICE T TanhV2(const T a) {
  T tmp = -2.0 * a;
  return (2.0 / (1.0 + exp(tmp))) - 1.0;
}

}  // namespace forward

namespace backward {

template <typename T>
DEVICE T Identity(const T a, const T b) {
  return a;
}

template <typename T>
DEVICE T Relu(const T a, const T b) {
  return a * (b > 0.0 ? 1.0 : 0.0);
}

template <typename T>
DEVICE T Sigmoid(const T a, const T b) {
  return a * b * (1.0 - b);
}

template <typename T>
DEVICE T Tanh(const T a, const T b) {
  return a * (1.0 - b * b);
}

}  // namespace backward

template <typename T>
struct Active {
  typedef T (*Act)(T);
  typedef T (*ActGrad)(T, T);
};

static DEVICE Active<float>::Act kActFloat[] = {
    &forward::Sigmoid<float>, &forward::SigmoidV2<float>,
    &forward::Relu<float>,    &forward::Tanh<float>,
    &forward::TanhV2<float>,  &forward::Identity<float>};

static DEVICE Active<float>::ActGrad kActGradFloat[] = {
    &backward::Sigmoid<float>, &backward::Sigmoid<float>,
    &backward::Relu<float>,    &backward::Tanh<float>,
    &backward::Tanh<float>,    &backward::Identity<float>};

static DEVICE Active<double>::Act kActDouble[] = {
    &forward::Sigmoid<double>, &forward::SigmoidV2<double>,
    &forward::Relu<double>,    &forward::Tanh<double>,
    &forward::TanhV2<double>,  &forward::Identity<double>};

static DEVICE Active<double>::ActGrad kActGradDouble[] = {
    &backward::Sigmoid<double>, &backward::Sigmoid<double>,
    &backward::Relu<double>,    &backward::Tanh<double>,
    &backward::Tanh<double>,    &backward::Identity<double>};

namespace forward {
inline DEVICE float activation(float a, int index) {
  return kActFloat[index](a);
}

inline DEVICE double activation(double a, int index) {
  return kActDouble[index](a);
}

}  // namespace forward

namespace backward {
inline DEVICE float activation(float a, float b, int index) {
  return kActGradFloat[index](a, b);
}

inline DEVICE double activation(double a, double b, int index) {
  return kActGradDouble[index](a, b);
}
}  // namespace backward

#ifdef __AVX__
namespace forward {
namespace avx {
__m256 Relu(const __m256 a);
__m256 Sigmoid(const __m256 a);
__m256 SigmoidV2(const __m256 a);
__m256 Tanh(const __m256 a);
__m256 TanhV2(const __m256 a);
__m256 Identity(const __m256 a);
}  // namespace avx
}  // namespace forward

namespace backward {
namespace avx {
__m256 Relu(const __m256 a, const __m256 b);
__m256 Sigmoid(const __m256 a, const __m256 b);
__m256 Tanh(const __m256 a, const __m256 b);
__m256 Identity(const __m256 a, const __m256 b);
}  // namespace avx
}  // namespace backward

static Active<__m256>::Act kActAvx[] = {
    &forward::avx::Sigmoid, &forward::avx::SigmoidV2, &forward::avx::Relu,
    &forward::avx::Tanh,    &forward::avx::TanhV2,    &forward::avx::Identity};

static Active<__m256>::ActGrad kActGradAvx[] = {
    &backward::avx::Sigmoid, &backward::avx::Sigmoid, &backward::avx::Relu,
    &backward::avx::Tanh,    &backward::avx::Tanh,    &backward::avx::Identity};

namespace forward {
inline __m256 activation(__m256 a, int index) { return kActAvx[index](a); }
}  // namespace forward

namespace backward {
inline __m256 activation(__m256 a, __m256 b, int index) {
  return kActGradAvx[index](a, b);
}
}  // namespace backward

#endif

}  // namespace detail
}  // namespace math
}  // namespace operators
}  // namespace paddle
