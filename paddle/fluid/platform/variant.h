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

// Boost 1.41.0 requires __CUDACC_VER__, but in CUDA 9 __CUDACC_VER__
// is removed, so we have to manually define __CUDACC_VER__ instead.
// For details, please refer to
// https://github.com/PaddlePaddle/Paddle/issues/6626
#if defined(__CUDACC__) && defined(__CUDACC_VER_MAJOR__)
#undef __CUDACC_VER__
#define __CUDACC_VER__                                  \
  __CUDACC_VER_BUILD__ + __CUDACC_VER_MAJOR__ * 10000 + \
      __CUDACC_VER_MINOR__ * 100
#endif

#include "boost/config.hpp"

// Because Boost 1.41.0's variadic templates has bug on nvcc, boost
// will disable variadic template support in NVCC mode.  Define
// BOOST_NO_CXX11_VARIADIC_TEMPLATES on gcc/clang to generate same
// function symbols.  For details,
// https://github.com/PaddlePaddle/Paddle/issues/3386
#if (defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP))
#ifndef BOOST_NO_CXX11_VARIADIC_TEMPLATES
#define BOOST_NO_CXX11_VARIADIC_TEMPLATES
#endif
#endif

#include <boost/any.hpp>
#include <boost/mpl/comparison.hpp>
#include <boost/mpl/less_equal.hpp>
#include <boost/optional.hpp>
#include <boost/variant.hpp>

// some platform-independent defintion
#if defined(_WIN32)
#define UNUSED
#define __builtin_expect(EXP, C) (EXP)
#else
#define UNUSED __attribute__((unused))
#endif
