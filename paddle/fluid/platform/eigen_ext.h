// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/platform/bfloat16.h"
#include "paddle/fluid/platform/complex.h"
#include "paddle/fluid/platform/float16.h"
#include "paddle/fluid/platform/hostdevice.h"

#include "unsupported/Eigen/CXX11/Tensor"

namespace Eigen {

using float16 = paddle::platform::float16;
template <typename T>
using complex = paddle::platform::complex<T>;

template <typename T>
struct NumTraits;

template <>
struct NumTraits<paddle::platform::bfloat16>
    : GenericNumTraits<paddle::platform::bfloat16> {
  enum {
    IsSigned = true,
    IsInteger = false,
    IsComplex = false,
    RequireInitialization = false
  };

  HOSTDEVICE static inline paddle::platform::bfloat16 epsilon() {
    return paddle::platform::raw_uint16_to_bfloat16(0x3400);
  }
  HOSTDEVICE static inline paddle::platform::bfloat16 dummy_precision() {
    return paddle::platform::bfloat16(1e-5f);
  }
  HOSTDEVICE static inline paddle::platform::bfloat16 highest() {
    return paddle::platform::raw_uint16_to_bfloat16(0x7f7f);
  }
  HOSTDEVICE static inline paddle::platform::bfloat16 lowest() {
    return paddle::platform::raw_uint16_to_bfloat16(0xff7f);
  }
  HOSTDEVICE static inline paddle::platform::bfloat16 infinity() {
    return paddle::platform::raw_uint16_to_bfloat16(0x7f80);
  }
  HOSTDEVICE static inline paddle::platform::bfloat16 quiet_NaN() {
    return paddle::platform::raw_uint16_to_bfloat16(0xffc1);
  }
};

template <>
struct NumTraits<complex<float>> : GenericNumTraits<std::complex<float>> {
  typedef float Real;
  typedef typename NumTraits<float>::Literal Literal;
  enum {
    IsComplex = 1,
    RequireInitialization = NumTraits<float>::RequireInitialization,
    ReadCost = 2 * NumTraits<float>::ReadCost,
    AddCost = 2 * NumTraits<Real>::AddCost,
    MulCost = 4 * NumTraits<Real>::MulCost + 2 * NumTraits<Real>::AddCost
  };

  EIGEN_DEVICE_FUNC
  static inline Real epsilon() { return NumTraits<Real>::epsilon(); }
  EIGEN_DEVICE_FUNC
  static inline Real dummy_precision() {
    return NumTraits<Real>::dummy_precision();
  }
  EIGEN_DEVICE_FUNC
  static inline int digits10() { return NumTraits<Real>::digits10(); }
};

template <>
struct NumTraits<complex<double>> : GenericNumTraits<std::complex<double>> {
  typedef double Real;
  typedef typename NumTraits<double>::Literal Literal;
  enum {
    IsComplex = 1,
    RequireInitialization = NumTraits<double>::RequireInitialization,
    ReadCost = 2 * NumTraits<double>::ReadCost,
    AddCost = 2 * NumTraits<Real>::AddCost,
    MulCost = 4 * NumTraits<Real>::MulCost + 2 * NumTraits<Real>::AddCost
  };

  EIGEN_DEVICE_FUNC
  static inline Real epsilon() { return NumTraits<Real>::epsilon(); }
  EIGEN_DEVICE_FUNC
  static inline Real dummy_precision() {
    return NumTraits<Real>::dummy_precision();
  }
  EIGEN_DEVICE_FUNC
  static inline int digits10() { return NumTraits<Real>::digits10(); }
};

template <>
struct NumTraits<float16> : GenericNumTraits<float16> {
  enum {
    IsSigned = true,
    IsInteger = false,
    IsComplex = false,
    RequireInitialization = false
  };

  HOSTDEVICE static inline float16 epsilon() {
    return paddle::platform::raw_uint16_to_float16(0x0800);
  }
  HOSTDEVICE static inline float16 dummy_precision() { return float16(1e-2f); }
  HOSTDEVICE static inline float16 highest() {
    return paddle::platform::raw_uint16_to_float16(0x7bff);
  }
  HOSTDEVICE static inline float16 lowest() {
    return paddle::platform::raw_uint16_to_float16(0xfbff);
  }
  HOSTDEVICE static inline float16 infinity() {
    return paddle::platform::raw_uint16_to_float16(0x7c00);
  }
  HOSTDEVICE static inline float16 quiet_NaN() {
    return paddle::platform::raw_uint16_to_float16(0x7c01);
  }
};

namespace numext {

//////////// bfloat methods /////////////

template <>
HOSTDEVICE inline bool(isnan)(const paddle::platform::bfloat16& a) {
  return (paddle::platform::isnan)(a);
}

template <>
HOSTDEVICE inline bool(isinf)(const paddle::platform::bfloat16& a) {
  return (paddle::platform::isinf)(a);
}

template <>
HOSTDEVICE inline bool(isfinite)(const paddle::platform::bfloat16& a) {
  return (paddle::platform::isfinite)(a);
}

template <>
HOSTDEVICE inline paddle::platform::bfloat16 exp(
    const paddle::platform::bfloat16& a) {
  return paddle::platform::bfloat16(::expf(static_cast<float>(a)));
}

template <>
HOSTDEVICE inline paddle::platform::bfloat16 expm1(
    const paddle::platform::bfloat16& a) {
  return paddle::platform::bfloat16(::expm1f(static_cast<float>(a)));
}

template <>
HOSTDEVICE inline paddle::platform::bfloat16 erf(
    const paddle::platform::bfloat16& a) {
  return paddle::platform::bfloat16(::erff(static_cast<float>(a)));
}

template <>
HOSTDEVICE inline paddle::platform::bfloat16 log(
    const paddle::platform::bfloat16& a) {
  return paddle::platform::bfloat16(::logf(static_cast<float>(a)));
}

template <>
HOSTDEVICE inline paddle::platform::bfloat16 tanh(
    const paddle::platform::bfloat16& a) {
  return paddle::platform::bfloat16(::tanhf(static_cast<float>(a)));
}

template <>
HOSTDEVICE inline paddle::platform::bfloat16 sqrt(
    const paddle::platform::bfloat16& a) {
  return paddle::platform::bfloat16(::sqrtf(static_cast<float>(a)));
}

template <>
HOSTDEVICE inline paddle::platform::bfloat16 ceil(
    const paddle::platform::bfloat16& a) {
  return paddle::platform::bfloat16(::ceilf(static_cast<float>(a)));
}

template <>
HOSTDEVICE inline paddle::platform::bfloat16 floor(
    const paddle::platform::bfloat16& a) {
  return paddle::platform::bfloat16(::floorf(static_cast<float>(a)));
}

template <>
HOSTDEVICE inline paddle::platform::bfloat16 round(
    const paddle::platform::bfloat16& a) {
  return paddle::platform::bfloat16(::roundf(static_cast<float>(a)));
}

template <>
HOSTDEVICE inline paddle::platform::bfloat16 pow(
    const paddle::platform::bfloat16& a, const paddle::platform::bfloat16& b) {
  return paddle::platform::bfloat16(
      ::powf(static_cast<float>(a), static_cast<float>(b)));
}

template <>
HOSTDEVICE inline paddle::platform::bfloat16 abs(
    const paddle::platform::bfloat16& a) {
  return paddle::platform::bfloat16(::fabs(static_cast<float>(a)));
}

template <>
HOSTDEVICE inline paddle::platform::bfloat16 mini(
    const paddle::platform::bfloat16& a, const paddle::platform::bfloat16& b) {
  return b < a ? b : a;
}

template <>
HOSTDEVICE inline paddle::platform::bfloat16 maxi(
    const paddle::platform::bfloat16& a, const paddle::platform::bfloat16& b) {
  return a < b ? b : a;
}

//////////// complex<float> methods /////////////

template <>
HOSTDEVICE inline bool(isnan)(const complex<float>& a) {
  return (paddle::platform::isnan)(a);
}

template <>
HOSTDEVICE inline bool(isinf)(const complex<float>& a) {
  return (paddle::platform::isinf)(a);
}

template <>
HOSTDEVICE inline bool(isfinite)(const complex<float>& a) {
  return (paddle::platform::isfinite)(a);
}

template <>
HOSTDEVICE inline complex<float> exp(const complex<float>& a) {
  float com = ::expf(a.real);
  float res_real = com * ::cosf(a.imag);
  float res_imag = com * ::sinf(a.imag);
  return complex<float>(res_real, res_imag);
}

template <>
HOSTDEVICE inline complex<float> log(const complex<float>& a) {
  return paddle::platform::log(a);
}

template <>
HOSTDEVICE inline complex<float> tanh(const complex<float>& a) {
  return paddle::platform::tanh(a);
}

template <>
HOSTDEVICE inline complex<float> sqrt(const complex<float>& a) {
  return paddle::platform::sqrt(a);
}

template <>
HOSTDEVICE inline complex<float> ceil(const complex<float>& a) {
  return complex<float>(::ceilf(a.real), ::ceilf(a.imag));
}

template <>
HOSTDEVICE inline complex<float> floor(const complex<float>& a) {
  return complex<float>(::floorf(a.real), ::floor(a.imag));
}

template <>
HOSTDEVICE inline complex<float> round(const complex<float>& a) {
  return complex<float>(::roundf(a.real), ::roundf(a.imag));
}

template <>
HOSTDEVICE inline complex<float> pow(const complex<float>& a,
                                     const complex<float>& b) {
  return paddle::platform::pow(a, b);
}

template <>
HOSTDEVICE inline float abs(const complex<float>& a) {
  return paddle::platform::abs(a);
}

//////////// complex<double> methods /////////////

template <>
HOSTDEVICE inline bool(isnan)(const complex<double>& a) {
  return (paddle::platform::isnan)(a);
}

template <>
HOSTDEVICE inline bool(isinf)(const complex<double>& a) {
  return (paddle::platform::isinf)(a);
}

template <>
HOSTDEVICE inline bool(isfinite)(const complex<double>& a) {
  return (paddle::platform::isfinite)(a);
}

template <>
HOSTDEVICE inline complex<double> exp(const complex<double>& a) {
  double com = ::expf(a.real);
  double res_real = com * ::cosf(a.imag);
  double res_imag = com * ::sinf(a.imag);
  return complex<double>(res_real, res_imag);
}

template <>
HOSTDEVICE inline complex<double> log(const complex<double>& a) {
  return paddle::platform::log(a);
}

template <>
HOSTDEVICE inline complex<double> tanh(const complex<double>& a) {
  return paddle::platform::tanh(a);
}

template <>
HOSTDEVICE inline complex<double> sqrt(const complex<double>& a) {
  return paddle::platform::sqrt(a);
}

template <>
HOSTDEVICE inline complex<double> ceil(const complex<double>& a) {
  return complex<double>(::ceilf(a.real), ::ceilf(a.imag));
}

template <>
HOSTDEVICE inline complex<double> floor(const complex<double>& a) {
  return complex<double>(::floorf(a.real), ::floor(a.imag));
}

template <>
HOSTDEVICE inline complex<double> round(const complex<double>& a) {
  return complex<double>(::roundf(a.real), ::roundf(a.imag));
}

template <>
HOSTDEVICE inline complex<double> pow(const complex<double>& a,
                                      const complex<double>& b) {
  return paddle::platform::pow(a, b);
}

template <>
HOSTDEVICE inline double abs(const complex<double>& a) {
  return paddle::platform::abs(a);
}

//////////// float16 methods /////////////

template <>
HOSTDEVICE inline bool(isnan)(const float16& a) {
  return (paddle::platform::isnan)(a);
}

template <>
HOSTDEVICE inline bool(isinf)(const float16& a) {
  return (paddle::platform::isinf)(a);
}

template <>
HOSTDEVICE inline bool(isfinite)(const float16& a) {
  return (paddle::platform::isfinite)(a);
}

template <>
HOSTDEVICE inline float16 exp(const float16& a) {
  return float16(::expf(static_cast<float>(a)));
}

template <>
HOSTDEVICE inline float16 expm1(const float16& a) {
  return float16(::expm1f(static_cast<float>(a)));
}

template <>
HOSTDEVICE inline float16 erf(const float16& a) {
  return float16(::erff(static_cast<float>(a)));
}

template <>
HOSTDEVICE inline float16 log(const float16& a) {
  return float16(::logf(static_cast<float>(a)));
}

template <>
HOSTDEVICE inline float16 tanh(const float16& a) {
  return float16(::tanhf(static_cast<float>(a)));
}

template <>
HOSTDEVICE inline float16 sqrt(const float16& a) {
  return float16(::sqrtf(static_cast<float>(a)));
}

template <>
HOSTDEVICE inline float16 ceil(const float16& a) {
  return float16(::ceilf(static_cast<float>(a)));
}

template <>
HOSTDEVICE inline float16 floor(const float16& a) {
  return float16(::floorf(static_cast<float>(a)));
}

template <>
HOSTDEVICE inline float16 round(const float16& a) {
  return float16(::roundf(static_cast<float>(a)));
}

template <>
HOSTDEVICE inline float16 pow(const float16& a, const float16& b) {
  return float16(::powf(static_cast<float>(a), static_cast<float>(b)));
}

template <>
HOSTDEVICE inline float16 abs(const float16& a) {
  return float16(::fabs(static_cast<float>(a)));
}

template <>
HOSTDEVICE inline float16 mini(const float16& a, const float16& b) {
  return b < a ? b : a;
}

template <>
HOSTDEVICE inline float16 maxi(const float16& a, const float16& b) {
  return a < b ? b : a;
}

}  // namespace numext
}  // namespace Eigen
