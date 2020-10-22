// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

#include <vector>
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/platform/device_context.h"
#include "paddle/fluid/platform/for_range.h"
#ifdef PADDLE_WITH_CUDA
#include <thrust/random.h>
#endif

namespace paddle {
namespace operators {

template <typename DeviceContext>
struct Random;

template <>
struct Random<platform::CPUDeviceContext> {
  using Engine = std::minstd_rand;

  template <typename T>
  using UniformIntDist = std::uniform_int_distribution<T>;
};

#ifdef PADDLE_WITH_CUDA
template <>
struct Random<platform::CUDADeviceContext> {
  using Engine = thrust::minstd_rand;

  template <typename T>
  using UniformIntDist = thrust::uniform_int_distribution<T>;
};
#endif

template <typename T>
HOSTDEVICE inline void StridedMemcpy(const T* x, const size_t* x_dims, T* out,
                                     const size_t* out_dims, int i, int rank,
                                     size_t prod_x_remain,
                                     size_t prod_out_remain,
                                     const size_t* offsets) {
  size_t x_dim_i = x_dims[i];
  size_t out_dim_i = out_dims[i];
  size_t x_stride = prod_x_remain / x_dim_i;
  size_t out_stride = prod_out_remain / out_dim_i;
  size_t offset_i = offsets[i];

  if (i == rank - 1) {
#if defined(__CUDA_ARCH__)
    PADDLE_ENFORCE(x_stride == 1,
                   "When i:%d == rank:%d - 1, x_stride of random_crop_op "
                   "expected to be 1, but got %ld. Please check input "
                   "value.",
                   i, rank, x_stride);
    PADDLE_ENFORCE(out_stride == 1,
                   "When i:%d == rank:%d - 1, out_stride of random_crop_op "
                   "expected to be 1, but got %ld. Please check input "
                   "value.",
                   i, rank, out_stride);
#else
    PADDLE_ENFORCE_EQ(
        x_stride, 1, platform::errors::InvalidArgument(
                         "When i:%d == rank:%d - 1, x_stride of random_crop_op "
                         "expected to be 1, but got %ld. Please check input "
                         "value.",
                         i, rank, x_stride));
    PADDLE_ENFORCE_EQ(
        out_stride, 1,
        platform::errors::InvalidArgument(
            "When i:%d == rank:%d - 1, out_stride of random_crop_op "
            "expected to be 1, but got %ld. Please check input "
            "value.",
            i, rank, out_stride));
#endif
    x += offset_i;
    for (size_t j = 0; j < out_dim_i; ++j) {
      *out++ = *x++;
    }
  } else {
    x += offset_i * x_stride;
    for (size_t j = 0; j < out_dim_i; ++j) {
      StridedMemcpy<T>(x, x_dims, out, out_dims, i + 1, rank, x_stride,
                       out_stride, offsets);
      x += x_stride;
      out += out_stride;
    }
  }
}

template <typename DeviceContext, typename T>
struct RandomCropFunctor {
  const T* x_;
  T* out_;
  size_t x_dims_[9];
  size_t out_dims_[9];
  int num_batchsize_dims_;
  int rank_;
  int64_t seed_;

  size_t prod_batchsize_dims_;
  size_t prod_x_ins_dims_;
  size_t prod_out_ins_dims_;

  RandomCropFunctor(const T* x, T* out, const framework::DDim& x_dims,
                    const framework::DDim& out_dims, int num_batchsize_dims,
                    int64_t seed)
      : x_(x),
        out_(out),
        num_batchsize_dims_(num_batchsize_dims),
        rank_(x_dims.size()),
        seed_(seed) {
    PADDLE_ENFORCE_EQ(
        x_dims.size(), out_dims.size(),
        platform::errors::InvalidArgument(
            "The dimensions of Input(X) must equal to be the dimensions"
            "of Output(Out), but received dimensions of Input(X) is [%d],"
            "received dimensions of Output(Out) is [%d].",
            x_dims.size(), out_dims.size()));
    PADDLE_ENFORCE_GT(
        rank_, num_batchsize_dims_,
        platform::errors::InvalidArgument(
            "The dimensions of Input(X) must be greater than the diff"
            "value of Input(X)'s dimensions minus Atrr(shape)'s dimensions,"
            "But received Input(X)'s dimensions is [%d], received value of"
            "Input(X)'s dimensions minus Attr(shape)'s dimensions is [%d].",
            rank_, num_batchsize_dims_));
    prod_batchsize_dims_ = 1;
    prod_x_ins_dims_ = 1;
    prod_out_ins_dims_ = 1;
    for (size_t i = 0; i < static_cast<size_t>(rank_); ++i) {
      size_t x_dim_i = x_dims[i];
      size_t out_dim_i = out_dims[i];
      x_dims_[i] = x_dim_i;
      out_dims_[i] = out_dim_i;
      if (i < static_cast<size_t>(num_batchsize_dims_)) {
        PADDLE_ENFORCE_EQ(
            x_dim_i, out_dim_i,
            platform::errors::InvalidArgument(
                "The first [%d] dimension value of Input(X) and Output(Out)"
                "must be equal, but received the [%d] dimension value of"
                "Input(X) and Output(Out) respectively are [%d] and [%d].",
                num_batchsize_dims_, i, x_dim_i, out_dim_i));
        prod_batchsize_dims_ *= x_dim_i;
      } else {
        prod_x_ins_dims_ *= x_dim_i;
        prod_out_ins_dims_ *= out_dim_i;
      }
    }
  }

  HOSTDEVICE void operator()(size_t ins_idx) {
    typename Random<DeviceContext>::Engine engine(seed_);
    engine.discard(ins_idx * (rank_ - num_batchsize_dims_));
    size_t offsets[9] = {};
    for (int i = num_batchsize_dims_; i < rank_; ++i) {
      typename Random<DeviceContext>::template UniformIntDist<size_t> dist(
          0, x_dims_[i] - out_dims_[i]);
      offsets[i - num_batchsize_dims_] = dist(engine);
    }

    const T* x = x_ + ins_idx * prod_x_ins_dims_;
    T* out = out_ + ins_idx * prod_out_ins_dims_;

    StridedMemcpy<T>(x, x_dims_ + num_batchsize_dims_, out,
                     out_dims_ + num_batchsize_dims_, 0,
                     rank_ - num_batchsize_dims_, prod_x_ins_dims_,
                     prod_out_ins_dims_, offsets);
  }
};

template <typename DeviceContext, typename T>
class RandomCropKernel : public framework::OpKernel<T> {
 public:
  virtual void Compute(const framework::ExecutionContext& ctx) const {
    int64_t seed = 0;
    auto& seed_tensor = GET_DATA_SAFELY(ctx.Input<framework::LoDTensor>("Seed"),
                                        "Input", "Seed", "RandomCrop");
    if (seed_tensor.IsInitialized()) {
      if (platform::is_cpu_place(seed_tensor.place())) {
        seed = *seed_tensor.template data<int64_t>();
      } else {
        LOG(WARNING) << "It is slow to place seed in GPU memory. Please verify "
                        "your program";
        framework::LoDTensor cpu_seed;
        framework::TensorCopySync(seed_tensor, platform::CPUPlace(), &cpu_seed);
        seed = *cpu_seed.data<int64_t>();
      }
    } else {
      VLOG(5) << "WARNING: The input 'Seed' is not initialized, use attribute "
                 "'startup_seed' instead.";
      seed = ctx.Attr<int>("startup_seed");
    }
    auto shape = ctx.Attr<std::vector<int>>("shape");
    auto& x = GET_DATA_SAFELY(ctx.Input<framework::LoDTensor>("X"), "Input",
                              "X", "RandomCrop");
    auto& out = GET_DATA_SAFELY(ctx.Output<framework::LoDTensor>("Out"),
                                "Output", "Out", "RandomCrop");

    int num_batchsize_dims = x.dims().size() - shape.size();
    RandomCropFunctor<DeviceContext, T> functor(
        x.template data<T>(), out.template mutable_data<T>(ctx.GetPlace()),
        x.dims(), out.dims(), num_batchsize_dims, seed);
    platform::ForRange<DeviceContext> for_range(
        ctx.template device_context<DeviceContext>(),
        functor.prod_batchsize_dims_);

    for_range(functor);

    Random<platform::CPUDeviceContext>::Engine engine(seed);
    engine.discard(functor.prod_batchsize_dims_ *
                   (functor.rank_ - functor.num_batchsize_dims_));
    *ctx.Output<framework::LoDTensor>("SeedOut")->mutable_data<int64_t>(
        framework::make_ddim({1}), platform::CPUPlace()) = engine();
  }
};

// TODO(fengjiayi): Backward of random crop op

}  // namespace operators
}  // namespace paddle
