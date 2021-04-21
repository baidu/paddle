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

#ifdef PADDLE_WITH_CUDA
#include "paddle/fluid/platform/cudnn_helper.h"

DECLARE_bool(cudnn_batchnorm_spatial_persistent);

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
using DataLayout = framework::DataLayout;
template <typename T>
using CudnnDataType = platform::CudnnDataType<T>;
template <typename T>
using BatchNormParamType = typename CudnnDataType<T>::BatchNormParamType;
using BatchNormMode = cudnnBatchNormMode_t;

inline BatchNormMode GetBatchNormMode(bool test_mode) {
#if CUDNN_VERSION_MIN(7, 0, 1)
  if (!test_mode && FLAGS_cudnn_batchnorm_spatial_persistent) {
    // Note: PERSISTENT not implemented for inference.
    return CUDNN_BATCHNORM_SPATIAL_PERSISTENT;
  }
#endif  // CUDNN_VERSION_MIN(7, 0, 1)
  return CUDNN_BATCHNORM_SPATIAL;
}

template <typename T>
inline DataLayout GetComputeFormat(DataLayout data_layout, bool test_mode,
                                   bool condition = true) {
  auto dtype = platform::CudnnDataType<T>::type;
  const bool fast_nhwc_batch_norm =
      test_mode || (dtype == CUDNN_DATA_HALF &&
                    FLAGS_cudnn_batchnorm_spatial_persistent && condition);
  if (fast_nhwc_batch_norm && data_layout == DataLayout::kNHWC) {
    return DataLayout::kNHWC;
  } else {
    return DataLayout::kNCHW;
  }
}

template <typename T>
class BatchNormWrapper {
 public:
  BatchNormWrapper(int dim_size, const std::vector<int> &dims,
                   const std::vector<int> &strides, BatchNormMode mode,
                   double epsilon, bool fuse_with_add_relu = false,
                   bool fuse_with_relu = false) {
    PADDLE_ENFORCE_CUDA_SUCCESS(
        platform::dynload::cudnnCreateTensorDescriptor(&x_desc_));
    PADDLE_ENFORCE_CUDA_SUCCESS(
        platform::dynload::cudnnCreateTensorDescriptor(&bn_param_desc_));

    PADDLE_ENFORCE_CUDA_SUCCESS(platform::dynload::cudnnSetTensorNdDescriptor(
        x_desc_, CudnnDataType<T>::type, dim_size, dims.data(),
        strides.data()));
    PADDLE_ENFORCE_CUDA_SUCCESS(
        platform::dynload::cudnnDeriveBNTensorDescriptor(bn_param_desc_,
                                                         x_desc_, mode));

    mode_ = mode;
    epsilon_ = epsilon;
    fuse_with_add_relu_ = fuse_with_add_relu;
    fuse_with_relu_ = fuse_with_relu;
  }

  ~BatchNormWrapper() {
    PADDLE_ENFORCE_CUDA_SUCCESS(
        platform::dynload::cudnnDestroyTensorDescriptor(x_desc_));
    PADDLE_ENFORCE_CUDA_SUCCESS(
        platform::dynload::cudnnDestroyTensorDescriptor(bn_param_desc_));
  }

  void Infer(const platform::CUDADeviceContext &ctx, const Tensor &x,
             const Tensor &scale, const Tensor &bias, const Tensor &est_mean,
             const Tensor &est_var, Tensor *y) {
    // Note: PERSISTENT not implemented for inference.
    PADDLE_ENFORCE_CUDA_SUCCESS(
        platform::dynload::cudnnBatchNormalizationForwardInference(
            /* handle= */ ctx.cudnn_handle(),
            /* mode= */ CUDNN_BATCHNORM_SPATIAL,
            /* alpha= */ CudnnDataType<T>::kOne(),
            /* beta= */ CudnnDataType<T>::kZero(),
            /* xDesc= */ x_desc_,
            /* x= */ x.data<T>(),
            /* yDesc= */ x_desc_,
            /* y= */ y->mutable_data<T>(ctx.GetPlace()),
            /* bnScaleBiasMeanVarDesc= */ bn_param_desc_,
            /* bnScale= */ scale.data<BatchNormParamType<T>>(),
            /* bnBias= */ bias.data<BatchNormParamType<T>>(),
            /* estimatedMean= */ est_mean.data<BatchNormParamType<T>>(),
            /* estimatedVariance= */ est_var.data<BatchNormParamType<T>>(),
            /* epsilon= */ epsilon_));
  }

  void TrainForward(const platform::CUDADeviceContext &ctx, const Tensor &x,
                    const Tensor &scale, const Tensor &bias, Tensor *y,
                    Tensor *mean_out, Tensor *variance_out, Tensor *saved_mean,
                    Tensor *saved_variance, Tensor *reserve_space,
                    double this_factor) {
#if CUDNN_VERSION_MIN(7, 4, 1)
    TrainForwardEx(ctx, x, scale, bias, y, mean_out, variance_out, saved_mean,
                   saved_variance, reserve_space, this_factor);
#else
    auto *mean_out_ptr =
        mean_out->mutable_data<BatchNormParamType<T>>(ctx.GetPlace());
    auto *variance_out_ptr =
        variance_out->mutable_data<BatchNormParamType<T>>(ctx.GetPlace());
    auto *saved_mean_ptr =
        saved_mean->mutable_data<BatchNormParamType<T>>(ctx.GetPlace());
    auto *saved_variance_ptr =
        saved_variance->mutable_data<BatchNormParamType<T>>(ctx.GetPlace());

    PADDLE_ENFORCE_CUDA_SUCCESS(
        platform::dynload::cudnnBatchNormalizationForwardTraining(
            /* handle= */ ctx.cudnn_handle(),
            /* mode */ mode_,
            /* alpha= */ CudnnDataType<T>::kOne(),
            /* beta= */ CudnnDataType<T>::kZero(),
            /* xDesc= */ x_desc_,
            /* x= */ x.data<T>(),
            /* yDesc= */ x_desc_,
            /* y= */ y->mutable_data<T>(ctx.GetPlace()),
            /* bnScaleBiasMeanVarDesc= */ bn_param_desc_,
            /* bnScale= */ scale.data<BatchNormParamType<T>>(),
            /* bnBias= */ bias.data<BatchNormParamType<T>>(),
            /* exponentialAverageFactor */ this_factor,
            /* resultRunningMean= */ mean_out_ptr,
            /* resultRunningVariance= */ variance_out_ptr,
            /* epsilon= */ epsilon_,
            /* resultSaveMean= */ saved_mean_ptr,
            /* resultSaveInvVariance */ saved_variance_ptr));
#endif
  }

  void TrainBackward(const platform::CUDADeviceContext &ctx, const Tensor &x,
                     const Tensor &d_y, const Tensor &scale,
                     const Tensor &saved_mean, const Tensor &saved_variance,
                     const Tensor &reserve_space, Tensor *d_x, Tensor *d_scale,
                     Tensor *d_bias) {
#if CUDNN_VERSION_MIN(7, 4, 1)
    TrainBackwardEx(ctx, x, d_y, scale, saved_mean, saved_variance,
                    reserve_space, d_x, d_scale, d_bias);
#else
    PADDLE_ENFORCE_CUDA_SUCCESS(
        platform::dynload::cudnnBatchNormalizationBackward(
            /* handle= */ ctx.cudnn_handle(),
            /* mode= */ mode_,
            /* alphaDataDiff= */ CudnnDataType<T>::kOne(),
            /* betaDataDiff= */ CudnnDataType<T>::kZero(),
            /* alphaParamDiff= */ CudnnDataType<T>::kOne(),
            /* betaParamDiff= */ CudnnDataType<T>::kZero(),
            /* xDesc= */ x_desc_,
            /* x= */ x.data<T>(),
            /* dyDesc= */ x_desc_,
            /* dy= */ d_y.data<T>(),
            /* dxDesc= */ x_desc_,
            /* dx= */ d_x->mutable_data<T>(ctx.GetPlace()),
            /* bnScaleBiasDiffDesc= */ bn_param_desc_,
            /* bnScale=  */ scale.data<BatchNormParamType<T>>(),
            /* resultBnScaleDiff= */ d_scale
                ->mutable_data<BatchNormParamType<T>>(ctx.GetPlace()),
            /* resultBnBiasDiff= */ d_bias->mutable_data<BatchNormParamType<T>>(
                ctx.GetPlace()),
            /* epsilon= */ epsilon_,
            /* savedMean= */ saved_mean.data<BatchNormParamType<T>>(),
            /* savedInvVariance= */ saved_variance
                .data<BatchNormParamType<T>>()));
#endif
  }

 private:
#if CUDNN_VERSION_MIN(7, 4, 1)
  void TrainForwardEx(const platform::CUDADeviceContext &ctx, const Tensor &x,
                      const Tensor &scale, const Tensor &bias, Tensor *y,
                      Tensor *mean_out, Tensor *variance_out,
                      Tensor *saved_mean, Tensor *saved_variance,
                      Tensor *reserve_space, double this_factor) {
    // Create reserve space and workspace for batch norm.
    // Reserve space will be used in the backward.
    size_t workspace_size = GetForwardWorkSpaceSize(ctx);
    Tensor workspace_tensor;
    void *workspace_ptr =
        workspace_tensor.mutable_data(ctx.GetPlace(), x.type(), workspace_size);

    PADDLE_ENFORCE_NOT_NULL(
        reserve_space,
        platform::errors::NotFound(
            "The argument ReserveSpace of batch_norm op is not found."));
    size_t reserve_space_size = GetReserveSpaceSize(ctx);
    void *reserve_space_ptr = reserve_space->mutable_data(
        ctx.GetPlace(), x.type(), reserve_space_size);

    auto *mean_out_ptr =
        mean_out->mutable_data<BatchNormParamType<T>>(ctx.GetPlace());
    auto *variance_out_ptr =
        variance_out->mutable_data<BatchNormParamType<T>>(ctx.GetPlace());
    auto *saved_mean_ptr =
        saved_mean->mutable_data<BatchNormParamType<T>>(ctx.GetPlace());
    auto *saved_variance_ptr =
        saved_variance->mutable_data<BatchNormParamType<T>>(ctx.GetPlace());

    PADDLE_ENFORCE_CUDA_SUCCESS(
        platform::dynload::cudnnBatchNormalizationForwardTrainingEx(
            /* handle= */ ctx.cudnn_handle(),
            /* mode= */ mode_,
            /* bnOps= */ GetBatchNormOps(),
            /* alpha= */ CudnnDataType<T>::kOne(),
            /* beta= */ CudnnDataType<T>::kZero(),
            /* xDesc= */ x_desc_,
            /* xData= */ x.data<T>(),
            /* zDesc= */ nullptr,
            /* zData= */ nullptr,
            /* yDesc= */ x_desc_,
            /* yData= */ y->mutable_data<T>(ctx.GetPlace()),
            /* bnScaleBiasMeanVarDesc= */ bn_param_desc_,
            /* bnScaleData= */ scale.data<BatchNormParamType<T>>(),
            /* bnBiasData= */ bias.data<BatchNormParamType<T>>(),
            /* exponentialAverageFactor= */ this_factor,
            /* resultRunningMeanData= */ mean_out_ptr,
            /* resultRunningVarianceData= */ variance_out_ptr,
            /* epsilon= */ epsilon_,
            /* saveMean= */ saved_mean_ptr,
            /* saveInvVariance= */ saved_variance_ptr,
            /* activationDesc= */ nullptr,
            /* workspace= */ workspace_ptr,
            /* workSpaceSizeInBytes= */ workspace_size,
            /* reserveSpace= */ reserve_space_ptr,
            /* reserveSpaceSizeInBytes= */ reserve_space_size));
  }

  void TrainBackwardEx(const platform::CUDADeviceContext &ctx, const Tensor &x,
                       const Tensor &d_y, const Tensor &scale,
                       const Tensor &saved_mean, const Tensor &saved_variance,
                       const Tensor &reserve_space, Tensor *d_x,
                       Tensor *d_scale, Tensor *d_bias) {
    size_t workspace_size = GetBackwardWorkSpaceSize(ctx);
    Tensor workspace_tensor;
    void *workspace_ptr =
        workspace_tensor.mutable_data(ctx.GetPlace(), x.type(), workspace_size);

    auto reserve_space_size = reserve_space.memory_size();

    auto *d_scale_ptr =
        d_scale->mutable_data<BatchNormParamType<T>>(ctx.GetPlace());
    auto *d_bias_ptr =
        d_bias->mutable_data<BatchNormParamType<T>>(ctx.GetPlace());

    PADDLE_ENFORCE_CUDA_SUCCESS(
        platform::dynload::cudnnBatchNormalizationBackwardEx(
            /* handle= */ ctx.cudnn_handle(),
            /* mode= */ mode_,
            /* bnOps= */ GetBatchNormOps(),
            /* alphaDataDiff= */ CudnnDataType<T>::kOne(),
            /* betaDataDiff= */ CudnnDataType<T>::kZero(),
            /* alphaParamDiff= */ CudnnDataType<T>::kOne(),
            /* betaParamDiff= */ CudnnDataType<T>::kZero(),
            /* xDesc= */ x_desc_,
            /* xData= */ x.data<T>(),
            /* yDesc= */ nullptr,
            /* yData= */ nullptr,
            /* dyDesc= */ x_desc_,
            /* dyData= */ d_y.data<T>(),
            /* dzDesc= */ nullptr,
            /* dzData= */ nullptr,
            /* dxDesc= */ x_desc_,
            /* dxData= */ d_x->mutable_data<T>(ctx.GetPlace()),
            /* dBnScaleBiasDesc= */ bn_param_desc_,
            /* bnScaleData= */ scale.data<BatchNormParamType<T>>(),
            /* bnBiasData= */ nullptr,
            /* dBnScaleData= */ d_scale_ptr,
            /* dBnBiasData= */ d_bias_ptr,
            /* epsilon= */ epsilon_,
            /* savedMean= */ saved_mean.data<BatchNormParamType<T>>(),
            /* savedInvVariance= */ saved_variance
                .data<BatchNormParamType<T>>(),
            /* activationDesc= */ nullptr,
            /* workspace= */ workspace_ptr,
            /* workSpaceSizeInBytes= */ workspace_size,
            /* reserveSpace= */ const_cast<T *>(reserve_space.data<T>()),
            /*reserveSpaceSizeInBytes=*/reserve_space_size));
  }

  cudnnBatchNormOps_t GetBatchNormOps() {
    if (fuse_with_add_relu_) {
      return CUDNN_BATCHNORM_OPS_BN_ADD_ACTIVATION;
    } else if (fuse_with_relu_) {
      return CUDNN_BATCHNORM_OPS_BN_ACTIVATION;
    } else {
      return CUDNN_BATCHNORM_OPS_BN;
    }
  }

  size_t GetForwardWorkSpaceSize(const platform::CUDADeviceContext &ctx) {
    size_t workspace_size = 0;

    // cudnn batchnorm workspace
    PADDLE_ENFORCE_CUDA_SUCCESS(
        platform::dynload::
            cudnnGetBatchNormalizationForwardTrainingExWorkspaceSize(
                /* handle= */ ctx.cudnn_handle(),
                /* mode= */ mode_,
                /* bnOps= */ GetBatchNormOps(),
                /* xDesc= */ x_desc_,
                /* zDesc= */ nullptr,
                /* yDesc= */ x_desc_,
                /* bnScaleBiasMeanVarDesc= */ bn_param_desc_,
                /* activationDesc= */ nullptr,
                /* sizeInBytes= */ &workspace_size));
    return workspace_size;
  }

  size_t GetBackwardWorkSpaceSize(const platform::CUDADeviceContext &ctx) {
    size_t workspace_size = 0;

    PADDLE_ENFORCE_CUDA_SUCCESS(
        platform::dynload::cudnnGetBatchNormalizationBackwardExWorkspaceSize(
            /* handle= */ ctx.cudnn_handle(),
            /* mode= */ mode_,
            /* bnOps= */ GetBatchNormOps(),
            /* xDesc= */ x_desc_,
            /* yDesc= */ x_desc_,
            /* dyDesc= */ x_desc_,
            /* dzDesc= */ nullptr,
            /* dxDesc= */ x_desc_,
            /* bnScaleBiasMeanVarDesc= */ bn_param_desc_,
            /* activationDesc= */ nullptr,
            /* sizeInBytes= */ &workspace_size));
    return workspace_size;
  }

  size_t GetReserveSpaceSize(const platform::CUDADeviceContext &ctx) {
    size_t reserve_space_size = 0;

    // cudnn batchnorm reserve space
    PADDLE_ENFORCE_CUDA_SUCCESS(
        platform::dynload::cudnnGetBatchNormalizationTrainingExReserveSpaceSize(
            /* handle= */ ctx.cudnn_handle(),
            /* mode= */ mode_,
            /* bnOps= */ GetBatchNormOps(),
            /* activationDesc= */ nullptr,
            /* xDesc= */ x_desc_,
            /* sizeInBytes= */ &reserve_space_size));
    return reserve_space_size;
  }
#endif

 private:
  cudnnTensorDescriptor_t x_desc_;
  cudnnTensorDescriptor_t bn_param_desc_;
  BatchNormMode mode_;
  double epsilon_;
  bool fuse_with_add_relu_;
  bool fuse_with_relu_;
};

}  // namespace operators
}  // namespace paddle
#endif  // PADDLE_WITH_CUDA
