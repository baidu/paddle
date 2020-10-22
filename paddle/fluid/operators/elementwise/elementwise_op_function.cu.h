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

#include <glog/logging.h>
#include "paddle/fluid/platform/enforce.h"
#include "paddle/fluid/platform/float16.h"
#include "paddle/fluid/platform/hostdevice.h"
#define PADDLE_CUDA_THREAD_SIZE 512

#ifdef PADDLE_WITH_CUDA
#include <cuda.h>
#endif  // PADDLE_WITH_CUDA

#ifdef PADDLE_CUDA_FP16
#include <cuda_fp16.h>
#endif

#if CUDA_VERSION < 9000
#define __h2div h2div
#endif

#define DIV_ERROR_INFO                               \
  "Integer division by zero encountered in divide. " \
  "Please check.\n"
namespace paddle {
namespace operators {

#define DEFINE_SIMPLE_BINARY_FUNCTOR(Func, expr)                   \
  template <typename T, class Enable = void>                       \
  struct Func##Functor {                                           \
    inline HOSTDEVICE T operator()(const T& a, const T& b) const { \
      return a expr b;                                             \
    }                                                              \
  };                                                               \
  template <typename T, class Enable = void>                       \
  struct Inverse##Func##Functor {                                  \
    inline HOSTDEVICE T operator()(const T& a, const T& b) const { \
      return b expr a;                                             \
    }                                                              \
  };

DEFINE_SIMPLE_BINARY_FUNCTOR(Add, +)
DEFINE_SIMPLE_BINARY_FUNCTOR(Sub, -)
DEFINE_SIMPLE_BINARY_FUNCTOR(Mul, *)
DEFINE_SIMPLE_BINARY_FUNCTOR(Div, /)
#undef DEFINE_SIMPLE_BINARY_FUNCTOR

// special div functor for int32/int64. check divison has a zero
template <typename T>
struct DivFunctor<T,
                  typename std::enable_if<std::is_integral<T>::value>::type> {
  inline HOSTDEVICE T operator()(const T& a, const T& b) const {
#if defined(__CUDA_ARCH__)
    PADDLE_ENFORCE(b != 0, DIV_ERROR_INFO);
#else
    PADDLE_ENFORCE(b != 0, platform::errors::InvalidArgument(DIV_ERROR_INFO));
#endif
    return a / b;
  }
};

#define DEFINE_SIMPLE_CUDA_BINARY_FUNCTOR(Func, expr)                         \
  template <typename T, class Enable = void>                                  \
  struct Func##RangeFunctor {                                                 \
    Func##RangeFunctor(const T* x, const T* y, T* z) : x_(x), y_(y), z_(z) {} \
    inline HOSTDEVICE void operator()(size_t id) const {                      \
      z_[id] = x_[id] expr y_[id];                                            \
    }                                                                         \
    const T* x_;                                                              \
    const T* y_;                                                              \
    T* z_;                                                                    \
  };
DEFINE_SIMPLE_CUDA_BINARY_FUNCTOR(Add, +)
DEFINE_SIMPLE_CUDA_BINARY_FUNCTOR(Sub, -)
DEFINE_SIMPLE_CUDA_BINARY_FUNCTOR(Mul, *)
DEFINE_SIMPLE_CUDA_BINARY_FUNCTOR(Div, /)
#undef DEFINE_SIMPLE_CUDA_BINARY_FUNCTOR

// special div functor for int32/int64. check divison has a zero
template <typename T>
struct DivRangeFunctor<
    T, typename std::enable_if<std::is_integral<T>::value>::type> {
  DivRangeFunctor(const T* x, const T* y, T* z) : x_(x), y_(y), z_(z) {}
  inline HOSTDEVICE void operator()(size_t id) const {
#if defined(__CUDA_ARCH__)
    PADDLE_ENFORCE(y_[id] != 0, DIV_ERROR_INFO);
#else
    PADDLE_ENFORCE(y_[id] != 0,
                   platform::errors::InvalidArgument(DIV_ERROR_INFO));
#endif
    z_[id] = x_[id] / y_[id];
  }
  const T* x_;
  const T* y_;
  T* z_;
};

#ifdef PADDLE_CUDA_FP16
inline DEVICE half2 half2_add(const half2& a, const half2& b) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
  return __hadd2(a, b);
#else
  float a1 = __low2float(a);
  float a2 = __high2float(a);
  float b1 = __low2float(b);
  float b2 = __high2float(b);
  float r1 = a1 + b1;
  float r2 = a2 + b2;
  return __floats2half2_rn(r1, r2);
#endif
}

inline DEVICE half2 half2_sub(const half2& a, const half2& b) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
  return __hsub2(a, b);
#else
  float a1 = __low2float(a);
  float a2 = __high2float(a);
  float b1 = __low2float(b);
  float b2 = __high2float(b);
  float r1 = a1 - b1;
  float r2 = a2 - b2;
  return __floats2half2_rn(r1, r2);
#endif
}

inline DEVICE half2 half2_mul(const half2& a, const half2& b) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
  return __hmul2(a, b);
#else
  float a1 = __low2float(a);
  float a2 = __high2float(a);
  float b1 = __low2float(b);
  float b2 = __high2float(b);
  float r1 = a1 * b1;
  float r2 = a2 * b2;
  return __floats2half2_rn(r1, r2);
#endif
}

inline DEVICE half2 half2_div(const half2& a, const half2& b) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
  return __h2div(a, b);
#else
  float a1 = __low2float(a);
  float a2 = __high2float(a);
  float b1 = __low2float(b);
  float b2 = __high2float(b);
  float r1 = a1 / b1;
  float r2 = a2 / b2;
  return __floats2half2_rn(r1, r2);
#endif
}

#define DEFINE_SIMPLE_CUDA_BINARY_KERNEL(Func, expr, FP16Function)           \
  template <typename T>                                                      \
  __global__ void SameDimsElemwise##Func##CUDAKernel(const T* x, const T* y, \
                                                     T* z, int64_t size) {   \
    int col = blockIdx.x * blockDim.x + threadIdx.x;                         \
    while (col < size) {                                                     \
      z[col] = x[col] expr y[col];                                           \
      col += blockDim.x * gridDim.x;                                         \
    }                                                                        \
  }                                                                          \
  template <>                                                                \
  inline __global__ void SameDimsElemwise##Func##CUDAKernel<half>(           \
      const half* x, const half* y, half* z, int64_t size) {                 \
    int start = threadIdx.x + blockDim.x * blockIdx.x;                       \
    int stride = blockDim.x * gridDim.x;                                     \
    int n2 = size / 2;                                                       \
    const half2* x2 = reinterpret_cast<const half2*>(x);                     \
    const half2* y2 = reinterpret_cast<const half2*>(y);                     \
    half2* z2 = reinterpret_cast<half2*>(z);                                 \
    for (int i = start; i < n2; i += stride) {                               \
      z2[i] = FP16Function(x2[i], y2[i]);                                    \
    }                                                                        \
    if (start == 0 && (size % 2)) {                                          \
      z[size - 1] = __float2half(__half2float(x[size - 1])                   \
                                     expr __half2float(y[size - 1]));        \
    }                                                                        \
  }
DEFINE_SIMPLE_CUDA_BINARY_KERNEL(Add, +, half2_add)
DEFINE_SIMPLE_CUDA_BINARY_KERNEL(Sub, -, half2_sub)
DEFINE_SIMPLE_CUDA_BINARY_KERNEL(Mul, *, half2_mul)
DEFINE_SIMPLE_CUDA_BINARY_KERNEL(Div, /, half2_div)
#undef DEFINE_SIMPLE_CUDA_BINARY_KERNEL

#endif  // PADDLE_CUDA_FP16

}  // namespace operators
}  // namespace paddle
