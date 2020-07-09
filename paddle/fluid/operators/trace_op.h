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
#include <algorithm>
#include <vector>
#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/math/math_function.h"
#include "paddle/fluid/platform/for_range.h"

namespace paddle {
namespace operators {

template <typename T>
struct DiagonalFunctor {
  DiagonalFunctor(const T* input, const int64_t* diag_stride,
                  const int64_t* ret_strides, int64_t pos, int64_t dim_size,
                  T* diag)
      : input_(input),
        diag_stride_(diag_stride),
        ret_strides_(ret_strides),
        pos_(pos),
        dim_size_(dim_size),
        diag_(diag) {}

  HOSTDEVICE void operator()(size_t idx) const {
    int64_t position = pos_;
    int64_t num = idx;
    for (int64_t i = 0; i < dim_size_; i++) {
      position += num / diag_stride_[i] * ret_strides_[i];
      num = num % diag_stride_[i];
    }
    diag_[idx] = input_[position];
  }

  const T* input_;
  const int64_t* diag_stride_;
  const int64_t* ret_strides_;
  int64_t pos_;
  int64_t dim_size_;
  T* diag_;
};

template <typename T>
struct TraceGradFunctor {
  TraceGradFunctor(const T* d_out, const int64_t* out_stride,
                   const int64_t* x_strides, int64_t pos, int64_t dim_size,
                   int64_t dim1, int64_t dim2, int64_t diag_size, T* d_x)
      : d_out_(d_out),
        out_stride_(out_stride),
        x_strides_(x_strides),
        pos_(pos),
        dim_size_(dim_size),
        dim1_(dim1),
        dim2_(dim2),
        diag_size_(diag_size),
        d_x_(d_x) {}

  HOSTDEVICE void operator()(size_t idx) const {
    int64_t num = idx - pos_;
    int64_t position = 0;
    if (num >= 0) {
      int64_t dim1 = 0;
      int64_t dim2 = 0;
      int64_t out_idx = 0;
      for (int64_t i = 0; i < dim_size_; i++) {
        if (i != dim1_ && i != dim2_) {
          position += num / x_strides_[i] * out_stride_[out_idx++];
        } else if (i == dim1_) {
          dim1 = num / x_strides_[i];
        } else {
          dim2 = num / x_strides_[i];
        }
        num = num % x_strides_[i];
      }
      if (dim1 == dim2 && dim1 < diag_size_) {
        d_x_[idx] = d_out_[position];
      }
    }
  }
  const T* d_out_;
  const int64_t* out_stride_;
  const int64_t* x_strides_;
  int64_t pos_;
  int64_t dim_size_;
  int64_t dim1_;
  int64_t dim2_;
  int64_t diag_size_;
  T* d_x_;
};

template <typename DeviceContext, typename T>
framework::Tensor Diagonal(const framework::ExecutionContext& context,
                           const framework::Tensor* input, int64_t offset,
                           int64_t dim1, int64_t dim2) {
  auto* input_data = input->data<T>();
  auto input_dims = input->dims();
  auto input_stride = framework::stride(input_dims);
  auto dim1_ = dim1 < 0 ? input_dims.size() + dim1 : dim1;
  auto dim2_ = dim2 < 0 ? input_dims.size() + dim2 : dim2;
  auto len1 = input_dims[std::min(dim1_, dim2_)];
  auto len2 = input_dims[std::max(dim1_, dim2_)];
  auto stride1 = input_stride[std::min(dim1_, dim2_)];
  auto stride2 = input_stride[std::max(dim1_, dim2_)];

  int offset_stride = 0;
  if (offset >= 0) {
    offset_stride = stride2;
    len2 -= offset;
  } else {
    offset_stride = stride1;
    len1 += offset;
  }
  int diag_size = len2 < len1 ? len2 : len1;

  if (diag_size > 0) {
    auto ret_strides = vectorize(input_stride);
    auto ret_dims = vectorize(input_dims);
    ret_strides.erase(ret_strides.begin() + std::max(dim1_, dim2_));
    ret_strides.erase(ret_strides.begin() + std::min(dim1_, dim2_));
    ret_dims.erase(ret_dims.begin() + std::max(dim1_, dim2_));
    ret_dims.erase(ret_dims.begin() + std::min(dim1_, dim2_));
    if (ret_strides.empty()) {
      ret_strides.push_back(1);
      ret_dims.push_back(1);
    }
    ret_strides.push_back(stride1 + stride2);
    ret_dims.push_back(diag_size);
    framework::Tensor diag;
    framework::DDim diag_dims = framework::make_ddim(ret_dims);
    auto dig_stride = framework::stride(diag_dims);
    auto diag_data = diag.mutable_data<T>(diag_dims, context.GetPlace());

    int64_t pos = std::abs(offset) * offset_stride;
    int64_t dim_size = ret_strides.size();
#ifdef __NVCC__
    thrust::device_vector<int64_t> diag_vec(vectorize(dig_stride));
    const int64_t* diag_arr = thrust::raw_pointer_cast(diag_vec.data());
    thrust::device_vector<int64_t> ret_vec(ret_strides);
    const int64_t* ret_arr = thrust::raw_pointer_cast(ret_vec.data());
#else
    auto* diag_arr = dig_stride.Get();
    const auto* ret_arr = ret_strides.data();
#endif

    auto& dev_ctx = context.template device_context<DeviceContext>();
    platform::ForRange<DeviceContext> for_range(dev_ctx, diag.numel());
    DiagonalFunctor<T> functor(input_data, diag_arr, ret_arr, pos, dim_size,
                               diag_data);
    for_range(functor);
    return diag;
  } else {
    return {};
  }
}

template <typename DeviceContext, typename T>
class TraceKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* input = context.Input<framework::Tensor>("Input");
    auto* out = context.Output<framework::Tensor>("Out");

    const int64_t offset = context.Attr<int>("diagonal");
    const int64_t dim1 = context.Attr<int>("start_axis");
    const int64_t dim2 = context.Attr<int>("stop_axis");

    auto output_dims = out->dims();

    out->mutable_data<T>(context.GetPlace());

    const framework::Tensor diag =
        Diagonal<DeviceContext, T>(context, input, offset, dim1, dim2);
    if (diag.numel() > 0) {
      auto x = framework::EigenMatrix<T>::Reshape(diag, diag.dims().size() - 1);
      auto output = framework::EigenVector<T>::Flatten(*out);
      auto& place =
          *context.template device_context<DeviceContext>().eigen_device();
      auto reduce_dim = Eigen::array<int, 1>({1});
      output.device(place) = x.sum(reduce_dim);
      out->Resize(output_dims);
    }
  }
};

template <typename DeviceContext, typename T>
class TraceGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    const auto* d_out =
        context.Input<framework::Tensor>(framework::GradVarName("Out"));
    auto* d_x =
        context.Output<framework::Tensor>(framework::GradVarName("Input"));

    int64_t offset = context.Attr<int>("diagonal");
    int64_t dim1 = context.Attr<int>("start_axis");
    int64_t dim2 = context.Attr<int>("stop_axis");

    auto input_dims = d_x->dims();
    auto input_stride = framework::stride(input_dims);
    auto output_dims = d_out->dims();
    auto output_stride = framework::stride(output_dims);

    auto* out_data = d_out->data<T>();
    T* x_data = d_x->mutable_data<T>(context.GetPlace());

    math::SetConstant<DeviceContext, T> set_zero;
    auto& dev_ctx = context.template device_context<DeviceContext>();
    set_zero(dev_ctx, d_x, static_cast<T>(0.0));

    auto dim1_ = dim1 < 0 ? input_dims.size() + dim1 : dim1;
    auto dim2_ = dim2 < 0 ? input_dims.size() + dim2 : dim2;
    auto len1 = input_dims[std::min(dim1_, dim2_)];
    auto len2 = input_dims[std::max(dim1_, dim2_)];
    auto stride1 = input_stride[std::min(dim1_, dim2_)];
    auto stride2 = input_stride[std::max(dim1_, dim2_)];

    int offset_stride = 0;
    if (offset >= 0) {
      offset_stride = stride2;
      len2 -= offset;
    } else {
      offset_stride = stride1;
      len1 += offset;
    }
    int64_t diag_size = len2 < len1 ? len2 : len1;
    int64_t pos = std::abs(offset) * offset_stride;
    if (diag_size > 0) {
#ifdef __NVCC__
      thrust::device_vector<int64_t> output_vec(vectorize(output_stride));
      const int64_t* output_arr = thrust::raw_pointer_cast(output_vec.data());
      thrust::device_vector<int64_t> input_vec(vectorize(input_stride));
      const int64_t* input_arr = thrust::raw_pointer_cast(input_vec.data());

#else
      const auto* output_arr = output_stride.Get();
      const auto* input_arr = input_stride.Get();
#endif

      platform::ForRange<DeviceContext> for_range(dev_ctx, d_x->numel());
      TraceGradFunctor<T> functor(out_data, output_arr, input_arr, pos,
                                  input_dims.size(), dim1_, dim2_, diag_size,
                                  x_data);
      for_range(functor);
    }
  }
};

}  // namespace operators
}  // namespace paddle
