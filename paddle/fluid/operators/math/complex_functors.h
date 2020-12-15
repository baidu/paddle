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

#include <type_traits>

#include "paddle/fluid/platform/complex128.h"
#include "paddle/fluid/platform/complex64.h"
#include "paddle/fluid/platform/hostdevice.h"

namespace paddle {
namespace operators {
namespace math {

template <bool B, typename T>
struct cond {
  static constexpr bool value = B;
  using type = T;
};

template <bool B, typename TrueF, typename FalseF>
struct eval_if {
  using type = typename TrueF::type;
};

template <typename TrueF, typename FalseF>
struct eval_if<false, TrueF, FalseF> {
  using type = typename FalseF::type;
};

template <bool B, typename T, typename F>
using eval_if_t = typename eval_if<B, T, F>::type;

template <typename Head, typename... Tail>
struct select {
  using type = eval_if_t<Head::value, Head, select<Tail...>>;
};

template <typename Head, typename... Tail>
using select_t = typename select<Head, Tail...>::type;

template <typename T>
struct RealFunctor {
  using RealT =
      select_t<cond<std::is_same<T, platform::complex64>::value, float>,
               cond<std::is_same<T, platform::complex128>::value, double>, T>;

  RealFunctor(const T* input, RealT* output, int64_t numel)
      : input_(input), output_(output), numel_(numel) {}

  HOSTDEVICE void operator()(int64_t idx) const {
    output_[idx] = input_[idx].real;
  }

  const T* input_;
  RealT* output_;
  int64_t numel_;
};

template <typename T>
struct ImagFunctor {
  using ImagT =
      select_t<cond<std::is_same<T, platform::complex64>::value, float>,
               cond<std::is_same<T, platform::complex128>::value, double>, T>;

  ImagFunctor(const T* input, ImagT* output, int64_t numel)
      : input_(input), output_(output), numel_(numel) {}

  HOSTDEVICE void operator()(int64_t idx) const {
    output_[idx] = input_[idx].imag;
  }

  const T* input_;
  ImagT* output_;
  int64_t numel_;
};

}  // namespace math
}  // namespace operators
}  // namespace paddle
