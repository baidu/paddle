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

#include <algorithm>
#include <type_traits>

#include "paddle/fluid/platform/device_context.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/fluid/platform/hostdevice.h"
#include "paddle/fluid/platform/place.h"

#if (defined(__NVCC__) || defined(__HIPCC__))
#include <thrust/execution_policy.h>
#include <thrust/transform.h>
#include "paddle/fluid/platform/details/cuda_transform_iterator_cast.h"
#endif

namespace paddle {
namespace platform {

// Transform applys a unary or a binary functor on each element in a
// range defined by a pair of iterators.
//
// - The specialization for CPU calls std::transform.
// - The specialization for CUDA calls thrust::tranform.
//
// NOTE: We need to define InputIter and OutputIter defined as
//       different types, because the InputIter points op's inputs and
//       OutputIter pints to op's outputs.
//
// NOTE: We don't assume that InputIter to be const InputType* and
//       OutputIter to be OutputType*, because we might use a iterator
//       class, paddle::fluid::operators::RowwiseTRansformIterator.
template <typename DeviceContext>
struct Transform {
  // The unary version.
  template <typename InputIter, typename OutputIter, typename UnaryOperation>
  void operator()(const DeviceContext& context, InputIter first, InputIter last,
                  OutputIter result, UnaryOperation op);

  // The binary version.
  template <typename InputIter1, typename InputIter2, typename OutputIter,
            typename BinaryOperation>
  void operator()(const DeviceContext& context, InputIter1 first1,
                  InputIter1 last1, InputIter2 first2, OutputIter result,
                  BinaryOperation op);
};

template <>
struct Transform<platform::CPUDeviceContext> {
  template <typename InputIter, typename OutputIter, typename UnaryOperation>
  void operator()(const platform::CPUDeviceContext& context, InputIter first,
                  InputIter last, OutputIter result, UnaryOperation op) {
    std::transform(first, last, result, op);
  }

  template <typename InputIter1, typename InputIter2, typename OutputIter,
            typename BinaryOperation>
  void operator()(const platform::CPUDeviceContext& context, InputIter1 first1,
                  InputIter1 last1, InputIter2 first2, OutputIter result,
                  BinaryOperation op) {
    std::transform(first1, last1, first2, result, op);
  }
};

#if (defined(__NVCC__) || defined(__HIPCC__))
template <>
struct Transform<platform::CUDADeviceContext> {
  template <typename InputIter, typename OutputIter, typename UnaryOperation>
  void operator()(const platform::CUDADeviceContext& context, InputIter first,
                  InputIter last, OutputIter result, UnaryOperation op) {
    auto place = context.GetPlace();
    PADDLE_ENFORCE_EQ(is_gpu_place(place), true,
                      platform::errors::PreconditionNotMet(
                          "The CUDA Transform must be used in GPU place."));
#ifdef __NVCC__
    thrust::transform(thrust::cuda::par.on(context.stream()),
#else
    thrust::transform(thrust::hip::par.on(context.stream()),
#endif
                      details::CastToCUDATransformIterator(first),
                      details::CastToCUDATransformIterator(last),
                      details::CastToCUDATransformIterator(result), op);
  }

  template <typename InputIter1, typename InputIter2, typename OutputIter,
            typename BinaryOperation>
  void operator()(const platform::CUDADeviceContext& context, InputIter1 first1,
                  InputIter1 last1, InputIter2 first2, OutputIter result,
                  BinaryOperation op) {
    auto place = context.GetPlace();
    PADDLE_ENFORCE_EQ(is_gpu_place(place), true,
                      platform::errors::PreconditionNotMet(
                          "The CUDA Transform must be used in GPU place."));
#ifdef __NVCC__
    thrust::transform(thrust::cuda::par.on(context.stream()),
#else
    thrust::transform(thrust::hip::par.on(context.stream()),
#endif
                      details::CastToCUDATransformIterator(first1),
                      details::CastToCUDATransformIterator(last1),
                      details::CastToCUDATransformIterator(first2),
                      details::CastToCUDATransformIterator(result), op);
  }
};
#endif

}  // namespace platform
}  // namespace paddle
