// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

#include <stdint.h>
#include <limits>
#if !defined(_WIN32)
#define PADDLE_ALIGN(x) __attribute__((aligned(x)))
#else
#define PADDLE_ALIGN(x) __declspec(align(x))
#endif

#ifdef PADDLE_WITH_CUDA
#include <cuComplex.h>
#include <thrust/complex.h>
#endif  // PADDLE_WITH_CUDA

#ifdef PADDLE_WITH_HIP
#include <hip/hip_complex.h>
#include <thrust/complex.h>  // NOLINT
#endif

#include <cstring>

#include "paddle/fluid/platform/complex128.h"
#include "paddle/fluid/platform/hostdevice.h"
#include "unsupported/Eigen/CXX11/Tensor"

namespace Eigen {
template <typename T>
struct NumTraits;
}  // namespace Eigen

namespace paddle {
namespace platform {

struct PADDLE_ALIGN(8) complex64 {
 public:
  float real;
  float imag;

  complex64() = default;
  complex64(const complex64& o) = default;
  complex64& operator=(const complex64& o) = default;
  complex64(complex64&& o) = default;
  complex64& operator=(complex64&& o) = default;
  ~complex64() = default;

  HOSTDEVICE complex64(float real, float imag) : real(real), imag(imag) {}
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)

  HOSTDEVICE inline explicit complex64(const thrust::complex<float>& c) {
    real = c.real();
    imag = c.imag();
  }

  HOSTDEVICE inline explicit operator thrust::complex<float>() const {
    return thrust::complex<float>(real, imag);
  }

#ifdef PADDLE_WITH_HIP
  HOSTDEVICE inline explicit operator hipFloatComplex() const {
    return make_hipFloatComplex(real, imag);
  }
#else
  HOSTDEVICE inline explicit operator cuFloatComplex() const {
    return make_cuFloatComplex(real, imag);
  }
#endif
#endif

  HOSTDEVICE complex64(const float& val) : real(val), imag(0) {}
  HOSTDEVICE complex64(const double& val)
      : real(static_cast<float>(val)), imag(0) {}
  HOSTDEVICE complex64(const int& val)
      : real(static_cast<float>(val)), imag(0) {}
  HOSTDEVICE complex64(const int64_t& val)
      : real(static_cast<float>(val)), imag(0) {}
  HOSTDEVICE complex64(const complex128& val)
      : real(static_cast<float>(val.real)),
        imag(static_cast<float>(val.imag)) {}

  HOSTDEVICE inline explicit operator std::complex<float>() {
    return static_cast<std::complex<float>>(std::complex<float>(real, imag));
  }

  template <class T>
  HOSTDEVICE inline explicit complex64(const T& val)
      : real(complex64(static_cast<float>(val)).real) {}

  HOSTDEVICE complex64(const std::complex<float> val)
      : real(val.real()), imag(val.imag()) {}

  HOSTDEVICE inline complex64& operator=(bool b) {
    real = b ? 1 : 0;
    imag = 0;
    return *this;
  }

  HOSTDEVICE inline complex64& operator=(int8_t val) {
    real = static_cast<float>(val);
    imag = 0;
    return *this;
  }

  HOSTDEVICE inline complex64& operator=(uint8_t val) {
    real = static_cast<float>(val);
    imag = 0;
    return *this;
  }

  HOSTDEVICE inline complex64& operator=(int16_t val) {
    real = static_cast<float>(val);
    imag = 0;
    return *this;
  }

  HOSTDEVICE inline complex64& operator=(uint16_t val) {
    real = static_cast<float>(val);
    imag = 0;
    return *this;
  }

  HOSTDEVICE inline complex64& operator=(int32_t val) {
    real = static_cast<float>(val);
    imag = 0;
    return *this;
  }

  HOSTDEVICE inline complex64& operator=(uint32_t val) {
    real = static_cast<float>(val);
    imag = 0;
    return *this;
  }

  HOSTDEVICE inline complex64& operator=(int64_t val) {
    real = static_cast<float>(val);
    imag = 0;
    return *this;
  }

  HOSTDEVICE inline complex64& operator=(uint64_t val) {
    real = static_cast<float>(val);
    imag = 0;
    return *this;
  }

  HOSTDEVICE inline complex64& operator=(float val) {
    real = val;
    imag = 0;
    return *this;
  }

  HOSTDEVICE inline complex64& operator=(double val) {
    real = static_cast<float>(val);
    imag = 0;
    return *this;
  }

  HOSTDEVICE inline operator float() const { return this->real; }

  HOSTDEVICE inline explicit operator bool() const {
    return static_cast<bool>(this->real) || static_cast<bool>(this->imag);
  }

  HOSTDEVICE inline explicit operator int8_t() const {
    return static_cast<int8_t>(this->real);
  }

  HOSTDEVICE inline explicit operator uint8_t() const {
    return static_cast<uint8_t>(this->real);
  }

  HOSTDEVICE inline explicit operator int16_t() const {
    return static_cast<int16_t>(this->real);
  }

  HOSTDEVICE inline explicit operator uint16_t() const {
    return static_cast<uint16_t>(this->real);
  }

  HOSTDEVICE inline explicit operator int32_t() const {
    return static_cast<int32_t>(this->real);
  }

  HOSTDEVICE inline explicit operator uint32_t() const {
    return static_cast<uint32_t>(this->real);
  }

  HOSTDEVICE inline explicit operator int64_t() const {
    return static_cast<int64_t>(this->real);
  }

  HOSTDEVICE inline explicit operator uint64_t() const {
    return static_cast<uint64_t>(this->real);
  }

  HOSTDEVICE inline explicit operator double() const {
    return static_cast<double>(this->real);
  }

  HOSTDEVICE inline operator complex128() const {
    return complex128(static_cast<double>(this->real),
                      static_cast<double>(this->imag));
  }
};

HOSTDEVICE inline complex64 operator+(const complex64& a, const complex64& b) {
#if defined(__CUDA_ARCH__) || defined(__HIPCC__)
  return complex64(thrust::complex<float>(a.real, a.imag) +
                   thrust::complex<float>(b.real, b.imag));
#else
  return complex64(a.real + b.real, a.imag + b.imag);
#endif
}

HOSTDEVICE inline complex64 operator-(const complex64& a, const complex64& b) {
#if defined(__CUDA_ARCH__) || defined(__HIPCC__)
  return complex64(thrust::complex<float>(a.real, a.imag) -
                   thrust::complex<float>(b.real, b.imag));
#else
  return complex64(a.real - b.real, a.imag - b.imag);
#endif
}

HOSTDEVICE inline complex64 operator*(const complex64& a, const complex64& b) {
#if defined(__CUDA_ARCH__) || defined(__HIPCC__)
  return complex64(thrust::complex<float>(a.real, a.imag) *
                   thrust::complex<float>(b.real, b.imag));
#else
  return complex64(a.real * b.real - a.imag * b.imag,
                   a.imag * b.real + b.imag * a.real);
#endif
}

HOSTDEVICE inline complex64 operator/(const complex64& a, const complex64& b) {
#if defined(__CUDA_ARCH__) || defined(__HIPCC__)
  return complex64(thrust::complex<float>(a.real, a.imag) /
                   thrust::complex<float>(b.real, b.imag));
#else
  float denominator = b.real * b.real + b.imag * b.imag;
  return complex64((a.real * b.real + a.imag * b.imag) / denominator,
                   (a.imag * b.real - a.real * b.imag) / denominator);
#endif
}

HOSTDEVICE inline complex64 operator-(const complex64& a) {
#if defined(__CUDA_ARCH__) || defined(__HIPCC__)
  return complex64(-thrust::complex<float>(a.real, a.imag));
#else
  complex64 res;
  res.real = -a.real;
  res.imag = -a.imag;
  return res;
#endif
}

HOSTDEVICE inline complex64& operator+=(complex64& a,  // NOLINT
                                        const complex64& b) {
#if defined(__CUDA_ARCH__) || defined(__HIPCC__)
  a = complex64(thrust::complex<float>(a.real, a.imag) +=
                thrust::complex<float>(b.real, b.imag));
  return a;
#else
  a.real += b.real;
  a.imag += b.imag;
  return a;
#endif
}

HOSTDEVICE inline complex64& operator-=(complex64& a,  // NOLINT
                                        const complex64& b) {
#if defined(__CUDA_ARCH__) || defined(__HIPCC__)
  a = complex64(thrust::complex<float>(a.real, a.imag) -=
                thrust::complex<float>(b.real, b.imag));
  return a;
#else
  a.real -= b.real;
  a.imag -= b.imag;
  return a;
#endif
}

HOSTDEVICE inline complex64& operator*=(complex64& a,  // NOLINT
                                        const complex64& b) {
#if defined(__CUDA_ARCH__) || defined(__HIPCC__)
  a = complex64(thrust::complex<float>(a.real, a.imag) *=
                thrust::complex<float>(b.real, b.imag));
  return a;
#else
  a.real = a.real * b.real - a.imag * b.imag;
  a.imag = a.imag * b.real + b.imag * a.real;
  return a;
#endif
}

HOSTDEVICE inline complex64& operator/=(complex64& a,  // NOLINT
                                        const complex64& b) {
#if defined(__CUDA_ARCH__) || defined(__HIPCC__)
  a = complex64(thrust::complex<float>(a.real, a.imag) /=
                thrust::complex<float>(b.real, b.imag));
  return a;
#else
  float denominator = b.real * b.real + b.imag * b.imag;
  a.real = (a.real * b.real + a.imag * b.imag) / denominator;
  a.imag = (a.imag * b.real - a.real * b.imag) / denominator;
  return a;
#endif
}

HOSTDEVICE inline complex64 raw_uint16_to_complex64(uint16_t a) {
  complex64 res;
  res.real = a;
  return res;
}

HOSTDEVICE inline bool operator==(const complex64& a, const complex64& b) {
  return a.real == b.real && a.imag == b.imag;
}

HOSTDEVICE inline bool operator!=(const complex64& a, const complex64& b) {
  return a.real != b.real || a.imag != b.imag;
}

HOSTDEVICE inline bool operator<(const complex64& a, const complex64& b) {
  return static_cast<float>(a.real) < static_cast<float>(b.real);
}

HOSTDEVICE inline bool operator<=(const complex64& a, const complex64& b) {
  return static_cast<float>(a.real) <= static_cast<float>(b.real);
}

HOSTDEVICE inline bool operator>(const complex64& a, const complex64& b) {
  return static_cast<float>(a.real) > static_cast<float>(b.real);
}

HOSTDEVICE inline bool operator>=(const complex64& a, const complex64& b) {
  return static_cast<float>(a.real) >= static_cast<float>(b.real);
}

HOSTDEVICE inline bool(isnan)(const complex64& a) {
#if defined(__CUDA_ARCH__)
  // __isnanf not supported on HIP platform
  return __isnanf(a.real) || __isnanf(a.imag);
#else
  return std::isnan(a.real) || std::isnan(a.imag);
#endif
}

HOSTDEVICE inline bool(isinf)(const complex64& a) {
#if defined(__CUDA_ARCH__)
  // __isinff not supported on HIP platform
  return __isinff(a.real) || __isinff(a.imag);
#else
  return std::isinf(a.real) || std::isinf(a.imag);
#endif
}

HOSTDEVICE inline bool(isfinite)(const complex64& a) {
  return !((isnan)(a)) && !((isinf)(a));
}

HOSTDEVICE inline float(abs)(const complex64& a) {
#if defined(__CUDA_ARCH__) || defined(__HIPCC__)
  return complex64(thrust::abs(thrust::complex<float>(a.real, a.imag)));
#else
  return std::abs(std::complex<float>(a.real, a.imag));
#endif
}

HOSTDEVICE inline complex64(pow)(const complex64& a, const complex64& b) {
#if defined(__CUDA_ARCH__) || defined(__HIPCC__)
  return complex64(thrust::pow(thrust::complex<float>(a.real, a.imag),
                               thrust::complex<float>(b.real, b.imag)));
#else
  return std::pow(std::complex<float>(a), std::complex<float>(b));
#endif
}

HOSTDEVICE inline complex64(sqrt)(const complex64& a) {
#if defined(__CUDA_ARCH__) || defined(__HIPCC__)
  return complex64(thrust::sqrt(thrust::complex<float>(a.real, a.imag)));
#else
  return std::sqrt(std::complex<float>(a));
#endif
}

HOSTDEVICE inline complex64(tanh)(const complex64& a) {
#if defined(__CUDA_ARCH__) || defined(__HIPCC__)
  return complex64(thrust::tanh(thrust::complex<float>(a.real, a.imag)));
#else
  return std::tanh(std::complex<float>(a));
#endif
}

HOSTDEVICE inline complex64(log)(const complex64& a) {
#if defined(__CUDA_ARCH__) || defined(__HIPCC__)
  return complex64(thrust::log(thrust::complex<float>(a.real, a.imag)));
#else
  return std::log(std::complex<float>(a));
#endif
}

inline std::ostream& operator<<(std::ostream& os, const complex64& a) {
  os << "real:" << a.real << " imag:" << a.imag;
  return os;
}

}  // namespace platform
}  // namespace paddle

namespace std {

template <>
struct is_pod<paddle::platform::complex64> {
  static const bool value =
      is_trivial<paddle::platform::complex64>::value &&
      is_standard_layout<paddle::platform::complex64>::value;
};

template <>
struct is_floating_point<paddle::platform::complex64>
    : std::integral_constant<
          bool, std::is_same<paddle::platform::complex64,
                             typename std::remove_cv<
                                 paddle::platform::complex64>::type>::value> {};
template <>
struct is_signed<paddle::platform::complex64> {
  static const bool value = false;
};

template <>
struct is_unsigned<paddle::platform::complex64> {
  static const bool value = false;
};

inline bool isnan(const paddle::platform::complex64& a) {
  return paddle::platform::isnan(a);
}

inline bool isinf(const paddle::platform::complex64& a) {
  return paddle::platform::isinf(a);
}

template <>
struct numeric_limits<paddle::platform::complex64> {
  static const bool is_specialized = false;
  static const bool is_signed = false;
  static const bool is_integer = false;
  static const bool is_exact = false;
  static const bool has_infinity = false;
  static const bool has_quiet_NaN = false;
  static const bool has_signaling_NaN = false;
  static const float_denorm_style has_denorm = denorm_absent;
  static const bool has_denorm_loss = false;
  static const std::float_round_style round_style = std::round_toward_zero;
  static const bool is_iec559 = false;
  static const bool is_bounded = false;
  static const bool is_modulo = false;
  static const int digits = 0;
  static const int digits10 = 0;
  static const int max_digits10 = 0;
  static const int radix = 0;
  static const int min_exponent = 0;
  static const int min_exponent10 = 0;
  static const int max_exponent = 0;
  static const int max_exponent10 = 0;
  static const bool traps = false;
  static const bool tinyness_before = false;

  static paddle::platform::complex64(min)() {
    return paddle::platform::complex64(0.0, 0.0);
  }
  static paddle::platform::complex64 lowest() {
    return paddle::platform::complex64(0.0, 0.0);
  }
  static paddle::platform::complex64(max)() {
    return paddle::platform::complex64(0.0, 0.0);
  }
  static paddle::platform::complex64 epsilon() {
    return paddle::platform::complex64(0.0, 0.0);
  }
  static paddle::platform::complex64 round_error() {
    return paddle::platform::complex64(0.0, 0.0);
  }
  static paddle::platform::complex64 infinity() {
    return paddle::platform::complex64(0.0, 0.0);
  }
  static paddle::platform::complex64 quiet_NaN() {
    return paddle::platform::complex64(0.0, 0.0);
  }
  static paddle::platform::complex64 signaling_NaN() {
    return paddle::platform::complex64(0.0, 0.0);
  }
  static paddle::platform::complex64 denorm_min() {
    return paddle::platform::complex64(0.0, 0.0);
  }
};

}  // namespace std

#define MKL_Complex8 paddle::platform::complex64
