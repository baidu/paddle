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
#include "paddle/fluid/framework/ddim.h"
#include "paddle/fluid/memory/memcpy.h"
#include "paddle/fluid/platform/device_context.h"

namespace paddle {
namespace operators {
namespace detail {

template <typename T, int Rank>
struct StridedMemcpyFunctor;

template <typename T>
struct StridedMemcpyFunctor<T, 0> {
  void operator()(const platform::DeviceContext& dev_ctx, const T* src,
                  const int64_t* src_stride, const int64_t* dst_dim,
                  const int64_t* dst_stride, T* dst) const {
    auto place = dev_ctx.GetPlace();
    if (platform::is_cpu_place(place)) {
      auto& cpu_place = BOOST_GET_CONST(platform::CPUPlace, place);
      memory::Copy(cpu_place, dst, cpu_place, src, sizeof(T));
    } else {
#ifdef PADDLE_WITH_CUDA
      auto& gpu_place = BOOST_GET_CONST(platform::CUDAPlace, place);
      auto& cuda_ctx =
          reinterpret_cast<const platform::CUDADeviceContext&>(dev_ctx);
      memory::Copy(gpu_place, dst, gpu_place, src, sizeof(T),
                   cuda_ctx.stream());
#else
      PADDLE_THROW(
          platform::errors::Unavailable("Paddle is not compiled with GPU."));
#endif
    }
  }
};

template <typename T>
struct StridedMemcpyFunctor<T, 1> {
  void operator()(const platform::DeviceContext& dev_ctx, const T* src,
                  const int64_t* src_stride, const int64_t* dst_dim,
                  const int64_t* dst_stride, T* dst) const {
    auto place = dev_ctx.GetPlace();
    if (platform::is_cpu_place(place)) {
      auto& cpu_place = BOOST_GET_CONST(platform::CPUPlace, place);
      memory::Copy(cpu_place, dst, cpu_place, src, sizeof(T) * dst_dim[0]);
    } else {
#ifdef PADDLE_WITH_CUDA
      auto& gpu_place = BOOST_GET_CONST(platform::CUDAPlace, place);
      auto& cuda_ctx =
          reinterpret_cast<const platform::CUDADeviceContext&>(dev_ctx);
      memory::Copy(gpu_place, dst, gpu_place, src, sizeof(T) * dst_dim[0],
                   cuda_ctx.stream());
#else
      PADDLE_THROW(
          platform::errors::Unavailable("Paddle is not compiled with GPU."));
#endif
    }
  }
};

template <typename T, int Rank>
struct StridedMemcpyFunctor {
  void operator()(const platform::DeviceContext& dev_ctx, const T* src,
                  const int64_t* src_stride, const int64_t* dst_dim,
                  const int64_t* dst_stride, T* dst) const {
    for (int64_t i = 0; i < dst_dim[0]; ++i) {
      StridedMemcpyFunctor<T, Rank - 1> func;
      func(dev_ctx, src, src_stride + 1, dst_dim + 1, dst_stride + 1, dst);
      src += src_stride[0];
      dst += dst_stride[0];
    }
  }
};

template <typename T>
struct StridedCopyDimVisitor {
  StridedCopyDimVisitor(const platform::DeviceContext& dev_ctx, const T* src,
                        const framework::DDim& src_stride,
                        const framework::DDim& dst_stride, T* dst)
      : dev_ctx_(dev_ctx),
        src_(src),
        src_stride_(src_stride),
        dst_stride_(dst_stride),
        dst_(dst) {}

  template <int D>
  void operator()(const framework::Dim<D>& dst_dim) const {
    StridedMemcpyFunctor<T, D> functor;
    functor(dev_ctx_, src_, src_stride_.Get(), dst_dim.Get(), dst_stride_.Get(),
            dst_);
  }

  const platform::DeviceContext& dev_ctx_;
  const T* src_;
  const framework::DDim& src_stride_;
  const framework::DDim& dst_stride_;
  T* dst_;
};

}  // namespace detail
}  // namespace operators
}  // namespace paddle
