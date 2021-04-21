/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

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

#ifdef PADDLE_WITH_HIP
#include "paddle/fluid/platform/miopen_helper.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
using DataLayout = framework::DataLayout;
template <typename T>
using CudnnDataType = platform::CudnnDataType<T>;
template <typename T>
using BatchNormParamType = typename CudnnDataType<T>::BatchNormParamType;
using BatchNormMode = miopenBatchNormMode_t;

inline BatchNormMode GetBatchNormMode(bool test_mode) {
  return miopenBNSpatial;
}

template <typename T>
inline DataLayout GetComputeFormat(DataLayout data_layout, bool test_mode,
                                   bool condition = true) {
  // HIP do not support compute format of NHWC
  return DataLayout::kNCHW;
}

template <typename T>
class BatchNormWrapper {
 public:
  BatchNormWrapper(int dim_size, const std::vector<int> &dims,
                   const std::vector<int> &strides, BatchNormMode mode,
                   double epsilon, bool fuse_with_add_relu = false,
                   bool fuse_with_relu = false) {
    PADDLE_ENFORCE_CUDA_SUCCESS(
        platform::dynload::miopenCreateTensorDescriptor(&x_desc_));
    PADDLE_ENFORCE_CUDA_SUCCESS(
        platform::dynload::miopenCreateTensorDescriptor(&bn_param_desc_));

    PADDLE_ENFORCE_CUDA_SUCCESS(platform::dynload::miopenSetTensorDescriptor(
        x_desc_, CudnnDataType<T>::type, dim_size,
        const_cast<int *>(dims.data()), const_cast<int *>(strides.data())));
    PADDLE_ENFORCE_CUDA_SUCCESS(
        platform::dynload::miopenDeriveBNTensorDescriptor(bn_param_desc_,
                                                          x_desc_, mode));

    mode_ = mode;
    epsilon_ = epsilon;
    fuse_with_add_relu_ = false;
    fuse_with_relu_ = false;
  }

  ~BatchNormWrapper() {
    PADDLE_ENFORCE_CUDA_SUCCESS(
        platform::dynload::miopenDestroyTensorDescriptor(x_desc_));
    PADDLE_ENFORCE_CUDA_SUCCESS(
        platform::dynload::miopenDestroyTensorDescriptor(bn_param_desc_));
  }

  void Infer(const platform::CUDADeviceContext &ctx, const Tensor &x,
             const Tensor &scale, const Tensor &bias, const Tensor &est_mean,
             const Tensor &est_var, Tensor *y) {
    PADDLE_ENFORCE_CUDA_SUCCESS(
        platform::dynload::miopenBatchNormalizationForwardInference(
            ctx.cudnn_handle(), miopenBNSpatial,
            const_cast<void *>(
                static_cast<const void *>(CudnnDataType<T>::kOne())),
            const_cast<void *>(
                static_cast<const void *>(CudnnDataType<T>::kZero())),
            x_desc_, static_cast<const void *>(x.data<T>()), x_desc_,
            static_cast<void *>(y->mutable_data<T>(ctx.GetPlace())),
            bn_param_desc_, const_cast<void *>(static_cast<const void *>(
                                scale.data<BatchNormParamType<T>>())),
            const_cast<void *>(
                static_cast<const void *>(bias.data<BatchNormParamType<T>>())),
            const_cast<void *>(static_cast<const void *>(
                est_mean.data<BatchNormParamType<T>>())),
            const_cast<void *>(static_cast<const void *>(
                est_var.data<BatchNormParamType<T>>())),
            epsilon_));
  }

  void TrainForward(const platform::CUDADeviceContext &ctx, const Tensor &x,
                    const Tensor &scale, const Tensor &bias, Tensor *y,
                    Tensor *mean_out, Tensor *variance_out, Tensor *saved_mean,
                    Tensor *saved_variance, Tensor *reserve_space,
                    double this_factor) {
    PADDLE_ENFORCE_CUDA_SUCCESS(
        platform::dynload::miopenBatchNormalizationForwardTraining(
            ctx.cudnn_handle(), mode_,
            const_cast<void *>(
                static_cast<const void *>(CudnnDataType<T>::kOne())),
            const_cast<void *>(
                static_cast<const void *>(CudnnDataType<T>::kZero())),
            x_desc_, static_cast<const void *>(x.data<T>()), x_desc_,
            static_cast<void *>(y->mutable_data<T>(ctx.GetPlace())),
            bn_param_desc_, const_cast<void *>(static_cast<const void *>(
                                scale.data<BatchNormParamType<T>>())),
            const_cast<void *>(
                static_cast<const void *>(bias.data<BatchNormParamType<T>>())),
            this_factor,
            static_cast<void *>(
                mean_out->mutable_data<BatchNormParamType<T>>(ctx.GetPlace())),
            static_cast<void *>(
                variance_out->mutable_data<BatchNormParamType<T>>(
                    ctx.GetPlace())),
            epsilon_,
            static_cast<void *>(saved_mean->mutable_data<BatchNormParamType<T>>(
                ctx.GetPlace())),
            static_cast<void *>(
                saved_variance->mutable_data<BatchNormParamType<T>>(
                    ctx.GetPlace()))));
  }

  void TrainBackward(const platform::CUDADeviceContext &ctx, const Tensor &x,
                     const Tensor &d_y, const Tensor &scale,
                     const Tensor &saved_mean, const Tensor &saved_variance,
                     const Tensor &reserve_space, Tensor *d_x, Tensor *d_scale,
                     Tensor *d_bias) {
    const auto *saved_mean_ptr = saved_mean.data<BatchNormParamType<T>>();
    const auto *saved_var_ptr = saved_variance.data<BatchNormParamType<T>>();

    PADDLE_ENFORCE_CUDA_SUCCESS(
        platform::dynload::miopenBatchNormalizationBackward(
            ctx.cudnn_handle(), mode_, CudnnDataType<T>::kOne(),
            CudnnDataType<T>::kZero(), CudnnDataType<T>::kOne(),
            CudnnDataType<T>::kZero(), x_desc_, x.data<T>(), x_desc_,
            d_y.data<T>(), x_desc_, d_x->mutable_data<T>(ctx.GetPlace()),
            bn_param_desc_, scale.data<BatchNormParamType<T>>(),
            d_scale->mutable_data<BatchNormParamType<T>>(ctx.GetPlace()),
            d_bias->mutable_data<BatchNormParamType<T>>(ctx.GetPlace()),
            epsilon_, saved_mean_ptr, saved_var_ptr));
  }

 private:
  miopenTensorDescriptor_t x_desc_;
  miopenTensorDescriptor_t bn_param_desc_;
  BatchNormMode mode_;
  double epsilon_;
  bool fuse_with_add_relu_;
  bool fuse_with_relu_;
};

}  // namespace operators
}  // namespace paddle
#endif  // PADDLE_WITH_HIP
