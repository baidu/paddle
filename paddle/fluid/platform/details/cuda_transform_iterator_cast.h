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

#if !defined(__NVCC__) && !defined(__HIPCC__)
#error device_ptr_cast must be include by .cu file
#endif

#include <type_traits>  // For std::remove_pointer and std::is_pointer.

#include "thrust/device_ptr.h"

namespace paddle {
namespace platform {
namespace details {

// PointerToThrustDevicePtr has two speicalizations, one casts a (CUDA
// device) pointer into thrust::device_ptr, the other keeps rest types
// un-casted.
template <typename T, bool is_ptr>
struct PointerToThrustDevicePtr;

template <typename T>
struct PointerToThrustDevicePtr<T, true> {
  using ELEM = typename std::remove_pointer<T>::type;
  using RTYPE = thrust::device_ptr<ELEM>;

  inline thrust::device_ptr<ELEM> operator()(ELEM* ele) const {
    return thrust::device_pointer_cast(ele);
  }
};

template <typename T>
struct PointerToThrustDevicePtr<T, false> {
  using RTYPE = T;
  inline RTYPE operator()(RTYPE it) const { return it; }
};

// CastToCUDATransformIterator casts a pointer to thrust::device_ptr
// so it could be used as the iterator of thrust::transform.  It
// doesn't cast other types.
//
// We need CastToCUDATransformIterator because it is often that we
// want to use device memory pointers as transform iterators, e.g., to
// transform a block of float32 to float16.  In this case, we want
// CastToCUDATransformIterator to cast float16/32 pointers to
// thrust::device_ptr, otherwise they cannot work as the iterator
// required by thrust::transform.  At the same time, we don't want to
// cast thrust::device_ptr to thrust::device_ptr repeatedly.
template <typename T>
auto CastToCUDATransformIterator(T t) ->
    typename PointerToThrustDevicePtr<T, std::is_pointer<T>::value>::RTYPE {
  PointerToThrustDevicePtr<T, std::is_pointer<T>::value> cast;
  return cast(t);
}

}  // namespace details
}  // namespace platform
}  // namespace paddle
