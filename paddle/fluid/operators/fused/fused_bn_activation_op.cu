// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

#include <algorithm>
#include <cfloat>
#include <string>
#include <vector>
#include "cub/cub.cuh"
#include "paddle/fluid/framework/data_layout.h"
#include "paddle/fluid/operators/activation_op.h"
#include "paddle/fluid/operators/fused/fused_bn_activation_op.h"
#include "paddle/fluid/operators/math/math_function.h"
#include "paddle/fluid/operators/norm_utils.h"
#include "paddle/fluid/platform/bfloat16.h"
#include "paddle/fluid/platform/cudnn_helper.h"
#include "paddle/fluid/platform/float16.h"

DECLARE_bool(cudnn_batchnorm_spatial_persistent);

namespace paddle {
namespace operators {
using Tensor = framework::Tensor;
template <typename T>
using CudnnDataType = platform::CudnnDataType<T>;
template <typename T>
using BatchNormParamType = typename CudnnDataType<T>::BatchNormParamType;

template <typename T>
class FusedBatchNormActKernel<platform::CUDADeviceContext, T>
    : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    PADDLE_ENFORCE_EQ(
        platform::is_gpu_place(ctx.GetPlace()), true,
        platform::errors::PreconditionNotMet("It must use CUDAPlace."));
    double epsilon = static_cast<double>(ctx.Attr<float>("epsilon"));
    float momentum = ctx.Attr<float>("momentum");
    std::string act_type = ctx.Attr<std::string>("act_type");

    if (epsilon <= CUDNN_BN_MIN_EPSILON - FLT_EPSILON) {
      LOG(ERROR) << "Provided epsilon is smaller than "
                 << "CUDNN_BN_MIN_EPSILON. Setting it to "
                 << "CUDNN_BN_MIN_EPSILON instead.";
    }
    epsilon = std::max(epsilon, CUDNN_BN_MIN_EPSILON);

    // Get the size for each dimension.
    // NHWC [batch_size, in_height, in_width, in_channels]
    const auto *x = ctx.Input<Tensor>("X");
    const auto &x_dims = x->dims();
    PADDLE_ENFORCE_EQ(x_dims.size() >= 2 && x_dims.size() <= 5, true,
                      platform::errors::PreconditionNotMet(
                          "The Input dim size should be between 2 and 5"));

    const auto *scale = ctx.Input<Tensor>("Scale");
    const auto *bias = ctx.Input<Tensor>("Bias");

    // Run training mode.
    // obtain running mean and running inv var, and see if we need to
    // initialize them.
    auto *mean_out = ctx.Output<Tensor>("MeanOut");
    auto *variance_out = ctx.Output<Tensor>("VarianceOut");
    mean_out->mutable_data<BatchNormParamType<T>>(ctx.GetPlace());
    variance_out->mutable_data<BatchNormParamType<T>>(ctx.GetPlace());

    auto *saved_mean = ctx.Output<Tensor>("SavedMean");
    auto *saved_variance = ctx.Output<Tensor>("SavedVariance");
    saved_mean->mutable_data<BatchNormParamType<T>>(ctx.GetPlace());
    saved_variance->mutable_data<BatchNormParamType<T>>(ctx.GetPlace());

    auto *y = ctx.Output<Tensor>("Y");
    y->mutable_data<T>(ctx.GetPlace());

    int N, C, H, W, D;
    const DataLayout data_layout = DataLayout::kNHWC;
    ExtractNCWHD(x_dims, data_layout, &N, &C, &H, &W, &D);

    auto &dev_ctx = ctx.template device_context<platform::CUDADeviceContext>();
    if ((N * H * W * D) == 1) {
      // Only 1 element in normalization dimension,
      // skip the batch norm calculation, let y = act(x).
      auto x_v = framework::EigenVector<T>::Flatten(*x);
      auto y_v = framework::EigenVector<T>::Flatten(*y);
      auto &dev = *dev_ctx.eigen_device();
      if (act_type == "relu") {
        ReluCUDAFunctor<T>()(dev, x_v, y_v);
      } else {
        PADDLE_THROW(
            platform::errors::Unimplemented("Unsupported activation type"));
      }
      return;
    }

    // ------------------- cudnn descriptors ---------------------
    auto handle = dev_ctx.cudnn_handle();
    cudnnTensorDescriptor_t data_desc_;
    cudnnTensorDescriptor_t bn_param_desc_;
    cudnnBatchNormMode_t mode_ = CUDNN_BATCHNORM_SPATIAL_PERSISTENT;

    PADDLE_ENFORCE_CUDA_SUCCESS(
        platform::dynload::cudnnCreateTensorDescriptor(&data_desc_));
    PADDLE_ENFORCE_CUDA_SUCCESS(
        platform::dynload::cudnnCreateTensorDescriptor(&bn_param_desc_));

    VLOG(3) << "Setting descriptors.";
    std::vector<int> dims = {N, C, H, W, D};
    std::vector<int> strides = {H * W * D * C, 1, W * D * C, D * C, C};

    PADDLE_ENFORCE_CUDA_SUCCESS(platform::dynload::cudnnSetTensorNdDescriptor(
        data_desc_, CudnnDataType<T>::type,
        x_dims.size() > 3 ? x_dims.size() : 4, dims.data(), strides.data()));

    PADDLE_ENFORCE_CUDA_SUCCESS(
        platform::dynload::cudnnDeriveBNTensorDescriptor(bn_param_desc_,
                                                         data_desc_, mode_));

    double this_factor = 1. - momentum;
    cudnnBatchNormOps_t bnOps_ = CUDNN_BATCHNORM_OPS_BN_ACTIVATION;
    platform::ScopedActivationDescriptor scope_act_desc;
    cudnnActivationDescriptor_t activation_desc_ =
        scope_act_desc.descriptor<T>(act_type);
    size_t workspace_size = 0;
    size_t reserve_space_size = 0;
    void *reserve_space_ptr = nullptr;
    void *workspace_ptr = nullptr;
    Tensor workspace_tensor;
    // Create reserve space and workspace for batch norm.
    // Create tensor for each batchnorm op, it will be used in the
    // backward. Thus this tensor shouldn't be temp.
    auto *reserve_space = ctx.Output<Tensor>("ReserveSpace");
    PADDLE_ENFORCE_NOT_NULL(
        reserve_space,
        platform::errors::NotFound(
            "The argument ReserveSpace of batch_norm op is not found."));

    // --------------- cudnn batchnorm workspace ---------------
    PADDLE_ENFORCE_CUDA_SUCCESS(
        platform::dynload::
            cudnnGetBatchNormalizationForwardTrainingExWorkspaceSize(
                /*handle=*/handle,
                /*mode=*/mode_,
                /*bnOps=*/bnOps_,
                /*xDesc=*/data_desc_,
                /*zDesc=*/nullptr,
                /*yDesc=*/data_desc_,
                /*bnScaleBiasMeanVarDesc=*/bn_param_desc_,
                /*activationDesc=*/activation_desc_,
                /*sizeInBytes=*/&workspace_size));

    // -------------- cudnn batchnorm reserve space --------------
    PADDLE_ENFORCE_CUDA_SUCCESS(
        platform::dynload::cudnnGetBatchNormalizationTrainingExReserveSpaceSize(
            /*handle=*/handle,
            /*mode=*/mode_,
            /*bnOps=*/bnOps_,
            /*activationDesc=*/activation_desc_,
            /*xDesc=*/data_desc_,
            /*sizeInBytes=*/&reserve_space_size));

    reserve_space_ptr = reserve_space->mutable_data(ctx.GetPlace(), x->type(),
                                                    reserve_space_size);
    workspace_ptr = workspace_tensor.mutable_data(ctx.GetPlace(), x->type(),
                                                  workspace_size);
    PADDLE_ENFORCE_CUDA_SUCCESS(
        platform::dynload::cudnnBatchNormalizationForwardTrainingEx(
            handle, mode_, bnOps_, CudnnDataType<T>::kOne(),
            CudnnDataType<T>::kZero(), data_desc_, x->template data<T>(),
            nullptr, nullptr, data_desc_, y->template data<T>(), bn_param_desc_,
            scale->template data<BatchNormParamType<T>>(),
            bias->template data<BatchNormParamType<T>>(), this_factor,
            mean_out->template mutable_data<BatchNormParamType<T>>(
                ctx.GetPlace()),
            variance_out->template mutable_data<BatchNormParamType<T>>(
                ctx.GetPlace()),
            epsilon, saved_mean->template mutable_data<BatchNormParamType<T>>(
                         ctx.GetPlace()),
            saved_variance->template mutable_data<BatchNormParamType<T>>(
                ctx.GetPlace()),
            activation_desc_, workspace_ptr, workspace_size, reserve_space_ptr,
            reserve_space_size));

    // clean when exit.
    PADDLE_ENFORCE_CUDA_SUCCESS(
        platform::dynload::cudnnDestroyTensorDescriptor(data_desc_));
    PADDLE_ENFORCE_CUDA_SUCCESS(
        platform::dynload::cudnnDestroyTensorDescriptor(bn_param_desc_));
  }
};

template <typename T>
class FusedBatchNormActGradKernel<platform::CUDADeviceContext, T>
    : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    PADDLE_ENFORCE_EQ(
        platform::is_gpu_place(ctx.GetPlace()), true,
        platform::errors::PreconditionNotMet("It must use CUDAPlace."));
    double epsilon = static_cast<double>(ctx.Attr<float>("epsilon"));
    std::string act_type = ctx.Attr<std::string>("act_type");

    const auto *x = ctx.Input<Tensor>("X");
    const auto *y = ctx.Input<Tensor>("Y");
    const auto *d_y = ctx.Input<Tensor>(framework::GradVarName("Y"));
    const auto *scale = ctx.Input<Tensor>("Scale");
    const auto *bias = ctx.Input<Tensor>("Bias");
    const auto *reserve_space = ctx.Input<Tensor>("ReserveSpace");

    const auto &x_dims = x->dims();

    PADDLE_ENFORCE_EQ(x_dims.size() >= 2 && x_dims.size() <= 5, true,
                      platform::errors::PreconditionNotMet(
                          "The Input dim size should be between 2 and 5"));
    int N, C, H, W, D;
    const DataLayout data_layout = DataLayout::kNHWC;
    ExtractNCWHD(x_dims, data_layout, &N, &C, &H, &W, &D);

    // init output
    auto *d_x = ctx.Output<Tensor>(framework::GradVarName("X"));
    auto *d_scale = ctx.Output<Tensor>(framework::GradVarName("Scale"));
    auto *d_bias = ctx.Output<Tensor>(framework::GradVarName("Bias"));

    d_x->mutable_data<T>(ctx.GetPlace());
    PADDLE_ENFORCE_EQ(
        d_scale && d_bias, true,
        platform::errors::PreconditionNotMet(
            "Both the scale grad and the bias grad must not be null."));
    d_scale->mutable_data<BatchNormParamType<T>>(ctx.GetPlace());
    d_bias->mutable_data<BatchNormParamType<T>>(ctx.GetPlace());
    PADDLE_ENFORCE_EQ(scale->dims().size(), 1UL,
                      platform::errors::PreconditionNotMet(
                          "The scale only has one dimension."));
    PADDLE_ENFORCE_EQ(
        scale->dims()[0], C,
        platform::errors::PreconditionNotMet(
            "The size of scale is equal to the channel of Input(X)."));

    auto &dev_ctx = ctx.template device_context<platform::CUDADeviceContext>();
    if ((N * H * W * D) == 1) {
      if (act_type == "relu") {
        auto x_v = framework::EigenVector<T>::Flatten(*x);
        auto y_v = framework::EigenVector<T>::Flatten(*y);
        auto dx_v = framework::EigenVector<T>::Flatten(*d_x);
        auto dy_v = framework::EigenVector<T>::Flatten(*d_y);
        auto &dev = *dev_ctx.eigen_device();
        ReluGradFunctor<T>()(dev, x_v, y_v, dy_v, dx_v);
      } else {
        PADDLE_THROW(
            platform::errors::Unimplemented("Unsupported activation type"));
      }
      math::SetConstant<platform::CUDADeviceContext, BatchNormParamType<T>>
          functor;
      functor(dev_ctx, d_scale, static_cast<BatchNormParamType<T>>(0));
      functor(dev_ctx, d_bias, static_cast<BatchNormParamType<T>>(0));
      return;
    }

    std::vector<int> dims = {N, C, H, W, D};
    std::vector<int> strides = {H * W * C * D, 1, W * D * C, D * C, C};
    // ------------------- cudnn descriptors ---------------------
    cudnnTensorDescriptor_t data_desc_;
    cudnnTensorDescriptor_t bn_param_desc_;
    cudnnBatchNormMode_t mode_ = CUDNN_BATCHNORM_SPATIAL_PERSISTENT;

    PADDLE_ENFORCE_CUDA_SUCCESS(
        platform::dynload::cudnnCreateTensorDescriptor(&data_desc_));
    PADDLE_ENFORCE_CUDA_SUCCESS(
        platform::dynload::cudnnCreateTensorDescriptor(&bn_param_desc_));
    if (epsilon <= CUDNN_BN_MIN_EPSILON - FLT_EPSILON) {
      LOG(ERROR) << "Provided epsilon is smaller than "
                 << "CUDNN_BN_MIN_EPSILON. Setting it to "
                 << "CUDNN_BN_MIN_EPSILON instead.";
    }
    epsilon = std::max(epsilon, CUDNN_BN_MIN_EPSILON);

    PADDLE_ENFORCE_CUDA_SUCCESS(platform::dynload::cudnnSetTensorNdDescriptor(
        data_desc_, CudnnDataType<T>::type,
        x_dims.size() > 3 ? x_dims.size() : 4, dims.data(), strides.data()));
    PADDLE_ENFORCE_CUDA_SUCCESS(
        platform::dynload::cudnnDeriveBNTensorDescriptor(bn_param_desc_,
                                                         data_desc_, mode_));

    const auto *saved_mean = ctx.Input<Tensor>("SavedMean");
    const auto *saved_var = ctx.Input<Tensor>("SavedVariance");
    const auto *saved_mean_data =
        saved_mean->template data<BatchNormParamType<T>>();
    const auto *saved_var_data =
        saved_var->template data<BatchNormParamType<T>>();

    size_t workspace_size = 0;
    void *workspace_ptr = nullptr;
    Tensor workspace_tensor;
    auto reserve_space_size = reserve_space->memory_size();
    cudnnBatchNormOps_t bnOps_ = CUDNN_BATCHNORM_OPS_BN_ACTIVATION;
    platform::ScopedActivationDescriptor scope_act_desc;
    cudnnActivationDescriptor_t activation_desc_ =
        scope_act_desc.descriptor<T>(act_type);
    // --------------- cudnn batchnorm workspace ---------------
    PADDLE_ENFORCE_CUDA_SUCCESS(
        platform::dynload::cudnnGetBatchNormalizationBackwardExWorkspaceSize(
            /*handle=*/dev_ctx.cudnn_handle(),
            /*mode=*/mode_,
            /*bnOps=*/bnOps_,
            /*xDesc=*/data_desc_,
            /*yDesc=*/data_desc_,
            /*dyDesc=*/data_desc_,
            /*dzDesc=*/nullptr,
            /*dxDesc=*/data_desc_,
            /*bnScaleBiasMeanVarDesc=*/bn_param_desc_,
            /*activationDesc=*/activation_desc_,
            /*sizeInBytes=*/&workspace_size));

    workspace_ptr = workspace_tensor.mutable_data(ctx.GetPlace(), x->type(),
                                                  workspace_size);

    PADDLE_ENFORCE_CUDA_SUCCESS(
        platform::dynload::cudnnBatchNormalizationBackwardEx(
            /*handle=*/dev_ctx.cudnn_handle(),
            /*mode=*/mode_,
            /*bnOps=*/bnOps_,
            /*alphaDataDiff=*/CudnnDataType<T>::kOne(),
            /*betaDataDiff=*/CudnnDataType<T>::kZero(),
            /*alphaParamDiff=*/CudnnDataType<T>::kOne(),
            /*betaParamDiff=*/CudnnDataType<T>::kZero(),
            /*xDesc=*/data_desc_,
            /*xData=*/x->template data<T>(),
            /*yDesc=*/data_desc_,
            /*yData=*/y->template data<T>(),
            /*dyDesc=*/data_desc_,
            /*dyData=*/d_y->template data<T>(),
            /*dzDesc=*/nullptr,
            /*dzData=*/nullptr,
            /*dxDesc=*/data_desc_,
            /*dxData=*/d_x->template mutable_data<T>(ctx.GetPlace()),
            /*dBnScaleBiasDesc=*/bn_param_desc_,
            /*bnScaleData=*/scale->template data<BatchNormParamType<T>>(),
            /*bnBiasData=*/bias->template data<BatchNormParamType<T>>(),
            /*dBnScaleData=*/d_scale
                ->template mutable_data<BatchNormParamType<T>>(ctx.GetPlace()),
            /*dBnBiasData=*/d_bias
                ->template mutable_data<BatchNormParamType<T>>(ctx.GetPlace()),
            /*epsilon=*/epsilon,
            /*savedMean=*/saved_mean_data,
            /*savedInvVariance=*/saved_var_data,
            /*activationDesc=*/activation_desc_,
            /*workspace=*/workspace_ptr,
            /*workSpaceSizeInBytes=*/workspace_size,
            /*reserveSpace=*/const_cast<T *>(reserve_space->template data<T>()),
            /*reserveSpaceSizeInBytes=*/reserve_space_size));

    // clean when exit.
    PADDLE_ENFORCE_CUDA_SUCCESS(
        platform::dynload::cudnnDestroyTensorDescriptor(data_desc_));
    PADDLE_ENFORCE_CUDA_SUCCESS(
        platform::dynload::cudnnDestroyTensorDescriptor(bn_param_desc_));
  }
};

}  // namespace operators
}  // namespace paddle

#if CUDNN_VERSION >= 7401
namespace ops = paddle::operators;
namespace plat = paddle::platform;
#if CUDNN_VERSION >= 8100
REGISTER_OP_CUDA_KERNEL(
    fused_batch_norm_act,
    ops::FusedBatchNormActKernel<plat::CUDADeviceContext, float>,
    ops::FusedBatchNormActKernel<plat::CUDADeviceContext, double>,
    ops::FusedBatchNormActKernel<plat::CUDADeviceContext, plat::float16>,
    ops::FusedBatchNormActKernel<plat::CUDADeviceContext, plat::bfloat16>);
REGISTER_OP_CUDA_KERNEL(
    fused_batch_norm_act_grad,
    ops::FusedBatchNormActGradKernel<plat::CUDADeviceContext, float>,
    ops::FusedBatchNormActGradKernel<plat::CUDADeviceContext, double>,
    ops::FusedBatchNormActGradKernel<plat::CUDADeviceContext, plat::float16>,
    ops::FusedBatchNormActGradKernel<plat::CUDADeviceContext, plat::bfloat16>);
#else
REGISTER_OP_CUDA_KERNEL(
    fused_batch_norm_act,
    ops::FusedBatchNormActKernel<plat::CUDADeviceContext, float>,
    ops::FusedBatchNormActKernel<plat::CUDADeviceContext, double>,
    ops::FusedBatchNormActKernel<plat::CUDADeviceContext, plat::float16>);
REGISTER_OP_CUDA_KERNEL(
    fused_batch_norm_act_grad,
    ops::FusedBatchNormActGradKernel<plat::CUDADeviceContext, float>,
    ops::FusedBatchNormActGradKernel<plat::CUDADeviceContext, double>,
    ops::FusedBatchNormActGradKernel<plat::CUDADeviceContext, plat::float16>);
#endif
#endif
