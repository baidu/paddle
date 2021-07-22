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

#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>

#ifdef PADDLE_WITH_CUDA
#include <cuda.h>
#endif

#if defined(__CUDACC__) && CUDA_VERSION >= 11000
#define PADDLE_CUDA_BF16
#include <cuda_bf16.h>
#endif

#if !defined(_WIN32)
#define PADDLE_ALIGN(x) __attribute__((aligned(x)))
#else
#define PADDLE_ALIGN(x) __declspec(align(x))
#endif

#if (defined(__CUDACC__) || defined(__HIPCC__))
#define HOSTDEVICE __host__ __device__
#define DEVICE __device__
#define HOST __host__
#else
#define HOSTDEVICE
#define DEVICE
#define HOST
#endif

namespace paddle {
namespace platform {

struct PADDLE_ALIGN(2) bfloat16 {
 public:
  uint16_t x;

  // Constructors
  bfloat16() = default;
  bfloat16(const bfloat16& o) = default;
  bfloat16& operator=(const bfloat16& o) = default;
  bfloat16(bfloat16&& o) = default;
  bfloat16& operator=(bfloat16&& o) = default;
  ~bfloat16() = default;

  HOSTDEVICE inline explicit bfloat16(float val) {
#ifdef PADDLE_WITH_HIP
    uint32_t res = 0;
    uint32_t* tempRes;
    // We should be using memcpy in order to respect the strict aliasing rule
    // but it fails in the HIP environment.
    tempRes = reinterpret_cast<uint32_t*>(&val);
    res = *tempRes;
    x = res >> 16;
#else
#if defined(PADDLE_CUDA_BF16)
    __nv_bfloat16 tmp = __float2bfloat16(val);
    x = *reinterpret_cast<uint16_t*>(&tmp);
#else
    std::memcpy(&x, reinterpret_cast<char*>(&val) + 2, 2);
#endif
#endif
  }

#if defined(PADDLE_CUDA_BF16)
  HOSTDEVICE inline explicit bfloat16(const __nv_bfloat16& val) {
    x = *reinterpret_cast<const uint16_t*>(&val);
  }
#endif

  template <class T>
  HOSTDEVICE inline explicit bfloat16(const T& val)
      : x(bfloat16(static_cast<float>(val)).x) {}

// Assignment operators
#if defined(PADDLE_CUDA_BF16)
  HOSTDEVICE inline bfloat16& operator=(const __nv_bfloat16& val) {
    x = *reinterpret_cast<const uint16_t*>(&val);
    return *this;
  }
#endif

  HOSTDEVICE inline bfloat16& operator=(bool b) {
    x = b ? 0x3f80 : 0;
    return *this;
  }

  HOSTDEVICE inline bfloat16& operator=(int8_t val) {
    x = bfloat16(val).x;
    return *this;
  }

  HOSTDEVICE inline bfloat16& operator=(uint8_t val) {
    x = bfloat16(val).x;
    return *this;
  }

  HOSTDEVICE inline bfloat16& operator=(int16_t val) {
    x = bfloat16(val).x;
    return *this;
  }

  HOSTDEVICE inline bfloat16& operator=(uint16_t val) {
    x = bfloat16(val).x;
    return *this;
  }

  HOSTDEVICE inline bfloat16& operator=(int32_t val) {
    x = bfloat16(val).x;
    return *this;
  }

  HOSTDEVICE inline bfloat16& operator=(uint32_t val) {
    x = bfloat16(val).x;
    return *this;
  }

  HOSTDEVICE inline bfloat16& operator=(int64_t val) {
    x = bfloat16(val).x;
    return *this;
  }

  HOSTDEVICE inline bfloat16& operator=(uint64_t val) {
    x = bfloat16(val).x;
    return *this;
  }

  HOSTDEVICE inline bfloat16& operator=(float val) {
    x = bfloat16(val).x;
    return *this;
  }

  HOSTDEVICE inline bfloat16& operator=(double val) {
    x = bfloat16(val).x;
    return *this;
  }

  // Conversion opertors
  HOSTDEVICE inline explicit operator float() const {
#ifdef PADDLE_CUDA_BF16
    return __bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(&x));
#else
    float val = 0.f;
    uint16_t temp = x;
    memcpy(reinterpret_cast<char*>(&val) + 2, reinterpret_cast<char*>(&temp),
           2);
    return val;
#endif
  }

#ifdef PADDLE_CUDA_BF16
  HOSTDEVICE inline explicit operator __nv_bfloat16() const {
    return *reinterpret_cast<const __nv_bfloat16*>(&x);
  }
#endif

  HOSTDEVICE inline explicit operator bool() const { return (x & 0x7fff) != 0; }

  HOSTDEVICE inline explicit operator int8_t() const {
    return static_cast<int8_t>(static_cast<float>(*this));
  }

  HOSTDEVICE inline explicit operator uint8_t() const {
    return static_cast<uint8_t>(static_cast<float>(*this));
  }

  HOSTDEVICE inline explicit operator int16_t() const {
    return static_cast<int16_t>(static_cast<float>(*this));
  }

  HOSTDEVICE inline explicit operator uint16_t() const {
    return static_cast<uint16_t>(static_cast<float>(*this));
  }

  HOSTDEVICE inline explicit operator int32_t() const {
    return static_cast<int32_t>(static_cast<float>(*this));
  }

  HOSTDEVICE inline explicit operator uint32_t() const {
    return static_cast<uint32_t>(static_cast<float>(*this));
  }

  HOSTDEVICE inline explicit operator int64_t() const {
    return static_cast<int64_t>(static_cast<float>(*this));
  }

  HOSTDEVICE inline explicit operator uint64_t() const {
    return static_cast<uint64_t>(static_cast<float>(*this));
  }

  HOSTDEVICE inline explicit operator double() const {
    return static_cast<double>(static_cast<float>(*this));
  }
};

HOSTDEVICE inline bfloat16 operator+(const bfloat16& a, const bfloat16& b) {
  return bfloat16(static_cast<float>(a) + static_cast<float>(b));
}

HOSTDEVICE inline bfloat16 operator-(const bfloat16& a, const bfloat16& b) {
  return bfloat16(static_cast<float>(a) - static_cast<float>(b));
}

HOSTDEVICE inline bfloat16 operator*(const bfloat16& a, const bfloat16& b) {
  return bfloat16(static_cast<float>(a) * static_cast<float>(b));
}

HOSTDEVICE inline bfloat16 operator/(const bfloat16& a, const bfloat16& b) {
  return bfloat16(static_cast<float>(a) / static_cast<float>(b));
}

template <typename T>
HOSTDEVICE inline typename std::enable_if<std::is_pod<T>::value, bfloat16>::type
operator+(const T& a, const bfloat16& b) {
  return bfloat16(static_cast<float>(a) + static_cast<float>(b));
}

template <typename T>
HOSTDEVICE inline typename std::enable_if<std::is_pod<T>::value, bfloat16>::type
operator-(const T& a, const bfloat16& b) {
  return bfloat16(static_cast<float>(a) - static_cast<float>(b));
}

template <typename T>
HOSTDEVICE inline typename std::enable_if<std::is_pod<T>::value, bfloat16>::type
operator*(const T& a, const bfloat16& b) {
  return bfloat16(static_cast<float>(a) * static_cast<float>(b));
}

template <typename T>
HOSTDEVICE inline typename std::enable_if<std::is_pod<T>::value, bfloat16>::type
operator/(const T& a, const bfloat16& b) {
  return bfloat16(static_cast<float>(a) / static_cast<float>(b));
}

template <typename T>
HOSTDEVICE inline typename std::enable_if<std::is_pod<T>::value, bfloat16>::type
operator+(const bfloat16& a, const T& b) {
  return bfloat16(static_cast<float>(a) + static_cast<float>(b));
}

template <typename T>
HOSTDEVICE inline typename std::enable_if<std::is_pod<T>::value, bfloat16>::type
operator-(const bfloat16& a, const T& b) {
  return bfloat16(static_cast<float>(a) - static_cast<float>(b));
}

template <typename T>
HOSTDEVICE inline typename std::enable_if<std::is_pod<T>::value, bfloat16>::type
operator*(const bfloat16& a, const T& b) {
  return bfloat16(static_cast<float>(a) * static_cast<float>(b));
}

template <typename T>
HOSTDEVICE inline typename std::enable_if<std::is_pod<T>::value, bfloat16>::type
operator/(const bfloat16& a, const T& b) {
  return bfloat16(static_cast<float>(a) / static_cast<float>(b));
}

HOSTDEVICE inline bfloat16 operator-(const bfloat16& a) {
  bfloat16 res;
  res.x = a.x ^ 0x8000;
  return res;
}

HOSTDEVICE inline bfloat16& operator+=(bfloat16& a,  // NOLINT
                                       const bfloat16& b) {
  a = bfloat16(static_cast<float>(a) + static_cast<float>(b));
  return a;
}

HOSTDEVICE inline bfloat16& operator-=(bfloat16& a,  // NOLINT
                                       const bfloat16& b) {
  a = bfloat16(static_cast<float>(a) - static_cast<float>(b));
  return a;
}

HOSTDEVICE inline bfloat16& operator*=(bfloat16& a,  // NOLINT
                                       const bfloat16& b) {
  a = bfloat16(static_cast<float>(a) * static_cast<float>(b));
  return a;
}

HOSTDEVICE inline bfloat16& operator/=(bfloat16& a,  // NOLINT
                                       const bfloat16& b) {
  a = bfloat16(static_cast<float>(a) / static_cast<float>(b));
  return a;
}

HOSTDEVICE inline bfloat16 raw_uint16_to_bfloat16(uint16_t a) {
  bfloat16 res;
  res.x = a;
  return res;
}

// Comparison operators
HOSTDEVICE inline bool operator==(const bfloat16& a, const bfloat16& b) {
  return static_cast<float>(a) == static_cast<float>(b);
}

HOSTDEVICE inline bool operator!=(const bfloat16& a, const bfloat16& b) {
  return static_cast<float>(a) != static_cast<float>(b);
}

HOSTDEVICE inline bool operator<(const bfloat16& a, const bfloat16& b) {
  return static_cast<float>(a) < static_cast<float>(b);
}

HOSTDEVICE inline bool operator<=(const bfloat16& a, const bfloat16& b) {
  return static_cast<float>(a) <= static_cast<float>(b);
}

HOSTDEVICE inline bool operator>(const bfloat16& a, const bfloat16& b) {
  return static_cast<float>(a) > static_cast<float>(b);
}

HOSTDEVICE inline bool operator>=(const bfloat16& a, const bfloat16& b) {
  return static_cast<float>(a) >= static_cast<float>(b);
}

HOSTDEVICE inline bool(isnan)(const bfloat16& a) {
  return (a.x & 0x7FFF) > 0x7F80;
}

HOSTDEVICE inline bool(isinf)(const bfloat16& a) {
  return (a.x & 0x7F80) == 0x7F80;
}

HOSTDEVICE inline bool(isfinite)(const bfloat16& a) {
  return !((isnan)(a)) && !((isinf)(a));
}

inline std::ostream& operator<<(std::ostream& os, const bfloat16& a) {
  os << a.x;
  return os;
}

}  // namespace platform
}  // namespace paddle

namespace std {

template <>
struct is_pod<paddle::platform::bfloat16> {
  static const bool value =
      is_trivial<paddle::platform::bfloat16>::value &&
      is_standard_layout<paddle::platform::bfloat16>::value;
};

template <>
struct is_floating_point<paddle::platform::bfloat16>
    : std::integral_constant<
          bool, std::is_same<paddle::platform::bfloat16,
                             typename std::remove_cv<
                                 paddle::platform::bfloat16>::type>::value> {};
template <>
struct is_signed<paddle::platform::bfloat16> {
  static const bool value = true;
};

template <>
struct is_unsigned<paddle::platform::bfloat16> {
  static const bool value = false;
};

inline bool isnan(const paddle::platform::bfloat16& a) {
  return paddle::platform::isnan(a);
}

inline bool isinf(const paddle::platform::bfloat16& a) {
  return paddle::platform::isinf(a);
}

template <>
struct numeric_limits<paddle::platform::bfloat16> {
  static const bool is_specialized = true;
  static const bool is_signed = true;
  static const bool is_integer = false;
  static const bool is_exact = false;
  static const bool has_infinity = true;
  static const bool has_quiet_NaN = true;
  static const bool has_signaling_NaN = true;
  static const float_denorm_style has_denorm = denorm_present;
  static const bool has_denorm_loss = false;
  static const std::float_round_style round_style = std::round_to_nearest;
  static const bool is_iec559 = false;
  static const bool is_bounded = false;
  static const bool is_modulo = false;
  static const int digits = 8;
  static const int digits10 = 2;
  static const int max_digits10 = 9;
  static const int radix = 2;
  static const int min_exponent = -125;
  static const int min_exponent10 = -37;
  static const int max_exponent = 128;
  static const int max_exponent10 = 38;
  static const bool traps = true;
  static const bool tinyness_before = false;

  static paddle::platform::bfloat16(min)() {
    return paddle::platform::raw_uint16_to_bfloat16(0x007f);
  }
  static paddle::platform::bfloat16 lowest() {
    return paddle::platform::raw_uint16_to_bfloat16(0xff7f);
  }
  static paddle::platform::bfloat16(max)() {
    return paddle::platform::raw_uint16_to_bfloat16(0x7f7f);
  }
  static paddle::platform::bfloat16 epsilon() {
    return paddle::platform::raw_uint16_to_bfloat16(0x3400);
  }
  static paddle::platform::bfloat16 round_error() {
    return paddle::platform::bfloat16(0.5);
  }
  static paddle::platform::bfloat16 infinity() {
    return paddle::platform::raw_uint16_to_bfloat16(0x7f80);
  }
  static paddle::platform::bfloat16 quiet_NaN() {
    return paddle::platform::raw_uint16_to_bfloat16(0xffc1);
  }
  static paddle::platform::bfloat16 signaling_NaN() {
    return paddle::platform::raw_uint16_to_bfloat16(0xff81);
  }
  static paddle::platform::bfloat16 denorm_min() {
    return paddle::platform::raw_uint16_to_bfloat16(0x0001);
  }
};

}  // namespace std
