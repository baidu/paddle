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

#include <algorithm>
#include <cfloat>
#include <string>
#include <vector>
#ifdef __NVCC__
#include "cub/cub.cuh"
#endif
#ifdef __HIPCC__
#include <hipcub/hipcub.hpp>
namespace cub = hipcub;
#endif
#include "paddle/fluid/framework/data_layout.h"
#include "paddle/fluid/operators/batch_norm_op.h"
#include "paddle/fluid/operators/math/math_function.h"
#include "paddle/fluid/operators/norm_utils.cu.h"
#include "paddle/fluid/platform/float16.h"

DECLARE_bool(cudnn_batchnorm_spatial_persistent);

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
using DataLayout = framework::DataLayout;
template <typename T>
using CudnnDataType = platform::CudnnDataType<T>;
template <typename T>
using BatchNormParamType = typename CudnnDataType<T>::BatchNormParamType;

template <typename T, framework::DataLayout layout>
static __global__ void BNForwardInference(
    const T *x, const BatchNormParamType<T> *mean,
    const BatchNormParamType<T> *variance, const BatchNormParamType<T> *scale,
    const BatchNormParamType<T> *bias, const int C, const int N, const int HxW,
    const double epsilon, T *y) {
  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  int num = N * C * HxW;
  for (int i = gid; i < num; i += stride) {
    const int c = layout == framework::DataLayout::kNCHW ? i / HxW % C : i % C;
    BatchNormParamType<T> x_sub_mean =
        static_cast<BatchNormParamType<T>>(x[i]) - mean[c];
    BatchNormParamType<T> inv_var = 1 / sqrt(variance[c] + epsilon);
    y[i] = static_cast<T>(scale[c] * x_sub_mean * inv_var + bias[c]);
  }
}

template <typename T, int BlockDim, framework::DataLayout layout>
static __global__ LAUNCH_BOUNDS(BlockDim) void BNForwardTraining(
    const T *x, const BatchNormParamType<T> *scale,
    const BatchNormParamType<T> *bias, const int C, const int N, const int HxW,
    const double epsilon, double exponentialAverageFactor, T *y,
    BatchNormParamType<T> *mean, BatchNormParamType<T> *variance,
    BatchNormParamType<T> *save_mean,
    BatchNormParamType<T> *save_inv_variance) {
  int outer_size = C;
  int inner_size = N * HxW;
  typedef cub::BlockReduce<BatchNormParamType<T>, BlockDim> BlockReduce;
  __shared__ typename BlockReduce::TempStorage mean_storage;
  __shared__ typename BlockReduce::TempStorage variance_storeage;
  __shared__ BatchNormParamType<T> mean_val;
  __shared__ BatchNormParamType<T> variance_val;
  __shared__ BatchNormParamType<T> inv_var_val;

  for (int i = blockIdx.x; i < outer_size; i += gridDim.x) {
    BatchNormParamType<T> x_sum = static_cast<BatchNormParamType<T>>(0);
    BatchNormParamType<T> x_square_sum = static_cast<BatchNormParamType<T>>(0);

    for (int j = threadIdx.x; j < inner_size; j += blockDim.x) {
      const int index = layout == framework::DataLayout::kNCHW
                            ? (j / HxW * C + i) * HxW + j % HxW
                            : j * outer_size + i;
      BatchNormParamType<T> x_i = static_cast<BatchNormParamType<T>>(x[index]);
      x_sum += x_i;
      x_square_sum += x_i * x_i;
    }
    x_sum = BlockReduce(mean_storage).Reduce(x_sum, cub::Sum());
    x_square_sum =
        BlockReduce(variance_storeage).Reduce(x_square_sum, cub::Sum());
    if (threadIdx.x == 0) {
      mean_val = x_sum / inner_size;
      variance_val = x_square_sum / inner_size - mean_val * mean_val;
      inv_var_val = 1 / sqrt(variance_val + epsilon);

      if (save_mean && save_inv_variance) {
        save_mean[i] = mean_val;
        save_inv_variance[i] = inv_var_val;
      }
      mean[i] = (1 - exponentialAverageFactor) * mean_val +
                exponentialAverageFactor * mean[i];
      variance[i] = (1 - exponentialAverageFactor) * variance_val +
                    exponentialAverageFactor * variance[i];
    }
    __syncthreads();

    for (int j = threadIdx.x; j < inner_size; j += blockDim.x) {
      const int index = layout == framework::DataLayout::kNCHW
                            ? (j / HxW * C + i) * HxW + j % HxW
                            : j * outer_size + i;
      BatchNormParamType<T> x_sub_mean =
          static_cast<BatchNormParamType<T>>(x[index]) - mean_val;
      y[index] = scale[i] * x_sub_mean * inv_var_val + bias[i];
    }
  }
}

template <typename T>
class BatchNormKernel<platform::CUDADeviceContext, T>
    : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    PADDLE_ENFORCE_EQ(
        platform::is_gpu_place(ctx.GetPlace()), true,
        platform::errors::InvalidArgument("It must use CUDAPlace."));
    double epsilon = static_cast<double>(ctx.Attr<float>("epsilon"));
    float momentum = ctx.Attr<float>("momentum");
    const bool is_test = ctx.Attr<bool>("is_test");
    const bool use_global_stats = ctx.Attr<bool>("use_global_stats");
    const bool trainable_stats = ctx.Attr<bool>("trainable_statistics");
    const std::string data_layout_str = ctx.Attr<std::string>("data_layout");
    const DataLayout data_layout =
        framework::StringToDataLayout(data_layout_str);

    bool test_mode = is_test && (!trainable_stats);

    // Get the size for each dimension.
    // NCHW [batch_size, in_channels, in_height, in_width]
    const auto *x = ctx.Input<Tensor>("X");
    const auto &x_dims = x->dims();
    PADDLE_ENFORCE_EQ(
        x_dims.size() >= 2 && x_dims.size() <= 5, true,
        platform::errors::InvalidArgument(
            "The size of input's dimensions should be between 2 and 5"
            "But received: the size of input's dimensions is [%d]",
            x_dims.size()));

    auto *y = ctx.Output<Tensor>("Y");
    y->mutable_data<T>(ctx.GetPlace());

    int N, C, H, W, D;
    ExtractNCWHD(x_dims, data_layout, &N, &C, &H, &W, &D);

    auto dtype = platform::CudnnDataType<T>::type;

#ifdef PADDLE_WITH_HIP
    auto compute_format = data_layout == DataLayout::kNHWC ? DataLayout::kNHWC
                                                           : DataLayout::kNCHW;

// TODO(wangran16): wait for MIOpen to improve the performance of BN
// HIP do not support compute format of NHWC
// auto compute_format = DataLayout::kNCHW;
#else
    const bool fast_nhwc_batch_norm =
        test_mode ||
        (dtype == CUDNN_DATA_HALF && FLAGS_cudnn_batchnorm_spatial_persistent);

    auto compute_format =
        fast_nhwc_batch_norm && data_layout == DataLayout::kNHWC
            ? DataLayout::kNHWC
            : DataLayout::kNCHW;
#endif

    Tensor transformed_x(x->type());
    Tensor transformed_y(y->type());
    if (data_layout == DataLayout::kNHWC &&
        compute_format == DataLayout::kNCHW && x_dims.size() > 2) {
      VLOG(3) << "Transform input tensor from NHWC to NCHW.";
      ResizeToChannelFirst<platform::CUDADeviceContext, T>(ctx, x,
                                                           &transformed_x);
      TransToChannelFirst<platform::CUDADeviceContext, T>(ctx, x,
                                                          &transformed_x);
      ResizeToChannelFirst<platform::CUDADeviceContext, T>(ctx, y,
                                                           &transformed_y);
    } else {
      transformed_x.ShareDataWith(*x);
      transformed_y.ShareDataWith(*y);
    }

// ------------------- cudnn descriptors ---------------------
#ifdef PADDLE_WITH_HIP
// TODO(wangran16): wait for MIOpen to improve the performance of BN
// miopenTensorDescriptor_t data_desc_;
// miopenTensorDescriptor_t bn_param_desc_;
// miopenBatchNormMode_t mode_;

// PADDLE_ENFORCE_CUDA_SUCCESS(
//     platform::dynload::miopenCreateTensorDescriptor(&data_desc_));
// PADDLE_ENFORCE_CUDA_SUCCESS(
//     platform::dynload::miopenCreateTensorDescriptor(&bn_param_desc_));
#else
    cudnnTensorDescriptor_t data_desc;
    cudnnTensorDescriptor_t bn_param_desc;
    cudnnBatchNormMode_t mode;

    PADDLE_ENFORCE_CUDA_SUCCESS(
        platform::dynload::cudnnCreateTensorDescriptor(&data_desc));
    PADDLE_ENFORCE_CUDA_SUCCESS(
        platform::dynload::cudnnCreateTensorDescriptor(&bn_param_desc));
#endif

    PADDLE_ENFORCE_GT(epsilon, CUDNN_BN_MIN_EPSILON - FLT_EPSILON,
                      platform::errors::InvalidArgument(
                          "Provided epsilon is expected to be greater than "
                          "CUDNN_BN_MIN_EPSILON. But recieved epsilon is %E, "
                          "CUDNN_BN_MIN_EPSILON is %E.",
                          epsilon, CUDNN_BN_MIN_EPSILON));
    epsilon = std::max(epsilon, CUDNN_BN_MIN_EPSILON);

#ifdef PADDLE_WITH_HIP
// TODO(wangran16): wait for MIOpen to improve the performance of BN
// mode_ = miopenBNSpatial;
#elif CUDNN_VERSION_MIN(7, 0, 1)
    if (FLAGS_cudnn_batchnorm_spatial_persistent) {
      mode = CUDNN_BATCHNORM_SPATIAL_PERSISTENT;
    } else {
      mode = CUDNN_BATCHNORM_SPATIAL;
    }
#else
    mode = CUDNN_BATCHNORM_SPATIAL;
#endif  // CUDNN_VERSION_MIN(7, 0, 1)

    VLOG(3) << "Setting descriptors.";
    std::vector<int> dims;
    std::vector<int> strides;
    if (compute_format == DataLayout::kNCHW) {
      dims = {N, C, H, W, D};
      strides = {C * H * W * D, H * W * D, W * D, D, 1};
    } else {
      dims = {N, C, H, W, D};
      strides = {H * W * D * C, 1, W * D * C, D * C, C};
    }

#ifdef PADDLE_WITH_HIP
// TODO(wangran16): wait for MIOpen to improve the performance of BN
// PADDLE_ENFORCE_CUDA_SUCCESS(platform::dynload::miopenSetTensorDescriptor(
//     data_desc_, CudnnDataType<T>::type,
//     x_dims.size() > 3 ? x_dims.size() : 4, const_cast<int *>(dims.data()),
//     const_cast<int *>(strides.data())));
// Note: PERSISTENT not implemented for inference
// PADDLE_ENFORCE_CUDA_SUCCESS(
//     platform::dynload::miopenDeriveBNTensorDescriptor(
//         bn_param_desc_, data_desc_, test_mode ? miopenBNSpatial : mode_));
#else
    PADDLE_ENFORCE_CUDA_SUCCESS(platform::dynload::cudnnSetTensorNdDescriptor(
        data_desc, CudnnDataType<T>::type,
        x_dims.size() > 3 ? x_dims.size() : 4, dims.data(), strides.data()));
    // Note: PERSISTENT not implemented for inference
    PADDLE_ENFORCE_CUDA_SUCCESS(
        platform::dynload::cudnnDeriveBNTensorDescriptor(
            bn_param_desc, data_desc,
            test_mode ? CUDNN_BATCHNORM_SPATIAL : mode));
#endif

    const auto *scale = ctx.Input<Tensor>("Scale");
    const auto *bias = ctx.Input<Tensor>("Bias");

    auto &dev_ctx = ctx.template device_context<platform::CUDADeviceContext>();

    auto handle = dev_ctx.cudnn_handle();

    // Depending on whether we are running test or not, we have two paths.
    // It is training mode when it's not reference AND not using pre-trained
    // model.
    bool training = !(test_mode || use_global_stats);
    if (!training) {
      // only when test we use input to do computation.
      const auto *est_mean = ctx.Input<Tensor>("Mean");
      const auto *est_var = ctx.Input<Tensor>("Variance");
      // Run inference mode.
      PADDLE_ENFORCE_EQ(
          est_mean->dims().size(), 1UL,
          platform::errors::InvalidArgument(
              "The size of mean's dimensions must equal to 1."
              "But received: the size of mean's dimensions mean is [%d],"
              "the dimensions of mean is [%s].",
              est_mean->dims().size(), est_mean->dims()));
      PADDLE_ENFORCE_EQ(
          est_var->dims().size(), 1UL,
          platform::errors::InvalidArgument(
              "The size of variance's dimensions must equal to 1."
              "But received: the size of variance's dimensions is [%d],"
              "the dimensions of variance is [%s].",
              est_var->dims().size(), est_var->dims()));
      PADDLE_ENFORCE_EQ(
          est_mean->dims()[0], C,
          platform::errors::InvalidArgument(
              "The first dimension of mean must equal to the number of "
              "Channels, which is [%d]. But received: the first dimension"
              "of mean is [%d], the dimensions of mean is [%s].",
              C, est_mean->dims()[0], est_mean->dims()));
      PADDLE_ENFORCE_EQ(
          est_var->dims()[0], C,
          platform::errors::InvalidArgument(
              "The first dimension of variance must equal to the number"
              "of Channels, which is [%d]. But received: the first dimension of"
              "variance is [%d], the dimensions of variance is [%s].",
              C, est_var->dims()[0], est_var->dims()));

#ifdef PADDLE_WITH_HIP
      const int block_size = 256;
      const int grid_size = (N * C * H * W * D + block_size - 1) / block_size;
      if (compute_format == DataLayout::kNCHW) {
        BNForwardInference<
            T,
            DataLayout::kNCHW><<<grid_size, block_size, 0, dev_ctx.stream()>>>(
            transformed_x.template data<T>(),
            est_mean->template data<BatchNormParamType<T>>(),
            est_var->template data<BatchNormParamType<T>>(),
            scale->template data<BatchNormParamType<T>>(),
            bias->template data<BatchNormParamType<T>>(), C, N, H * W * D,
            epsilon, transformed_y.template data<T>());
      } else {
        BNForwardInference<
            T,
            DataLayout::kNHWC><<<grid_size, block_size, 0, dev_ctx.stream()>>>(
            transformed_x.template data<T>(),
            est_mean->template data<BatchNormParamType<T>>(),
            est_var->template data<BatchNormParamType<T>>(),
            scale->template data<BatchNormParamType<T>>(),
            bias->template data<BatchNormParamType<T>>(), C, N, H * W * D,
            epsilon, transformed_y.template data<T>());
      }

// TODO(wangran16): wait for MIOpen to improve the performance of BN
// PADDLE_ENFORCE_CUDA_SUCCESS(
//     platform::dynload::miopenBatchNormalizationForwardInference(
//         handle, miopenBNSpatial,
//         const_cast<void *>(
//             static_cast<const void *>(CudnnDataType<T>::kOne())),
//         const_cast<void *>(
//             static_cast<const void *>(CudnnDataType<T>::kZero())),
//         data_desc_,
//         static_cast<const void *>(transformed_x.template data<T>()),
//         data_desc_,
//         static_cast<void *>(
//             transformed_y.template mutable_data<T>(ctx.GetPlace())),
//         bn_param_desc_,
//         const_cast<void *>(static_cast<const void *>(
//             scale->template data<BatchNormParamType<T>>())),
//         const_cast<void *>(static_cast<const void *>(
//             bias->template data<BatchNormParamType<T>>())),
//         const_cast<void *>(static_cast<const void *>(
//             est_mean->template data<BatchNormParamType<T>>())),
//         const_cast<void *>(static_cast<const void *>(
//             est_var->template data<BatchNormParamType<T>>())),
//         epsilon));
#else
      PADDLE_ENFORCE_CUDA_SUCCESS(
          platform::dynload::cudnnBatchNormalizationForwardInference(
              handle,
              // Note: PERSISTENT not implemented for inference
              CUDNN_BATCHNORM_SPATIAL, CudnnDataType<T>::kOne(),
              CudnnDataType<T>::kZero(), data_desc,
              transformed_x.template data<T>(), data_desc,
              transformed_y.template mutable_data<T>(ctx.GetPlace()),
              bn_param_desc, scale->template data<BatchNormParamType<T>>(),
              bias->template data<BatchNormParamType<T>>(),
              est_mean->template data<BatchNormParamType<T>>(),
              est_var->template data<BatchNormParamType<T>>(), epsilon));
#endif
    } else {
      // if MomentumTensor is set, use MomentumTensor value, momentum
      // is only used in this training branch
      if (ctx.HasInput("MomentumTensor")) {
        const auto *mom_tensor = ctx.Input<Tensor>("MomentumTensor");
        Tensor mom_cpu;
        TensorCopySync(*mom_tensor, platform::CPUPlace(), &mom_cpu);
        momentum = mom_cpu.data<float>()[0];
      }

      // Run training mode.
      // obtain running mean and running inv var, and there is no need
      // to initialize them.

      auto *mean_out = ctx.Output<Tensor>("MeanOut");
      auto *variance_out = ctx.Output<Tensor>("VarianceOut");
      auto *mean_out_ptr =
          mean_out->mutable_data<BatchNormParamType<T>>(ctx.GetPlace());
      auto *variance_out_ptr =
          variance_out->mutable_data<BatchNormParamType<T>>(ctx.GetPlace());

      auto *saved_mean = ctx.Output<Tensor>("SavedMean");
      auto *saved_variance = ctx.Output<Tensor>("SavedVariance");
      auto *saved_mean_ptr =
          saved_mean->mutable_data<BatchNormParamType<T>>(ctx.GetPlace());
      auto *saved_variance_ptr =
          saved_variance->mutable_data<BatchNormParamType<T>>(ctx.GetPlace());

      if ((N * H * W * D) == 1) {
        // Only 1 element in normalization dimension,
        // skip the batch norm calculation, let y = x.
        framework::TensorCopy(*x, ctx.GetPlace(), y);
      } else {
        double this_factor = 1. - momentum;

        bool called = false;
#if CUDNN_VERSION_MIN(7, 4, 1)
        called = true;
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
                    /*mode=*/mode,
                    /*bnIps=*/CUDNN_BATCHNORM_OPS_BN,
                    /*xDesc=*/data_desc,
                    /*zDesc=*/nullptr,
                    /*yDesc=*/data_desc,
                    /*bnScaleBiasMeanVarDesc=*/bn_param_desc,
                    /*activationDesc=*/nullptr,
                    /*sizeInBytes=*/&workspace_size));

        // -------------- cudnn batchnorm reserve space --------------
        PADDLE_ENFORCE_CUDA_SUCCESS(
            platform::dynload::
                cudnnGetBatchNormalizationTrainingExReserveSpaceSize(
                    /*handle=*/handle,
                    /*mode=*/mode,
                    /*bnOps=*/CUDNN_BATCHNORM_OPS_BN,
                    /*activationDesc=*/nullptr,
                    /*xDesc=*/data_desc,
                    /*sizeInBytes=*/&reserve_space_size));

        reserve_space_ptr = reserve_space->mutable_data(
            ctx.GetPlace(), transformed_x.type(), reserve_space_size);
        workspace_ptr = workspace_tensor.mutable_data(
            ctx.GetPlace(), transformed_x.type(), workspace_size);
        PADDLE_ENFORCE_CUDA_SUCCESS(
            platform::dynload::cudnnBatchNormalizationForwardTrainingEx(
                /* handle= */ handle,
                /* mode= */ mode,
                /* bnOps= */ CUDNN_BATCHNORM_OPS_BN,
                /* alpha= */ CudnnDataType<T>::kOne(),
                /* beta= */ CudnnDataType<T>::kZero(),
                /* xDesc= */ data_desc,
                /* xData= */ transformed_x.template data<T>(),
                /* zDesc= */ nullptr,
                /* zData= */ nullptr,
                /* yDesc= */ data_desc,
                /* yData= */ transformed_y.template data<T>(),
                /* bnScaleBiasMeanVarDesc= */ bn_param_desc,
                /* bnScaleData= */ scale
                    ->template data<BatchNormParamType<T>>(),
                /* bnBiasData= */ bias->template data<BatchNormParamType<T>>(),
                /* exponentialAverageFactor= */ this_factor,
                /* resultRunningMeanData= */ mean_out_ptr,
                /* resultRunningVarianceData= */ variance_out_ptr,
                /* epsilon= */ epsilon,
                /* saveMean= */ saved_mean_ptr,
                /* saveInvVariance= */ saved_variance_ptr,
                /* activationDesc= */ nullptr,
                /* workspace= */ workspace_ptr,
                /* workSpaceSizeInBytes= */ workspace_size,
                /* reserveSpace= */ reserve_space_ptr,
                /* reserveSpaceSizeInBytes= */ reserve_space_size));
#endif  // CUDNN_VERSION_MIN(7, 4, 1)
        if (!called) {
#ifdef PADDLE_WITH_HIP
          const int num = transformed_x.numel();
          const int block = 256;
          const int max_threads = dev_ctx.GetMaxPhysicalThreadCount();
          const int max_blocks = std::max(max_threads / block, 1);
          const int grid = std::min(C, max_blocks);
          if (compute_format == DataLayout::kNCHW) {
            BNForwardTraining<
                T, block,
                DataLayout::kNCHW><<<grid, block, 0, dev_ctx.stream()>>>(
                transformed_x.template data<T>(),
                scale->template data<BatchNormParamType<T>>(),
                bias->template data<BatchNormParamType<T>>(), C, N, H * W * D,
                epsilon, this_factor, transformed_y.template data<T>(),
                mean_out->template data<BatchNormParamType<T>>(),
                variance_out->template data<BatchNormParamType<T>>(),
                saved_mean->template data<BatchNormParamType<T>>(),
                saved_variance->template data<BatchNormParamType<T>>());
          } else {
            BNForwardTraining<
                T, block,
                DataLayout::kNHWC><<<grid, block, 0, dev_ctx.stream()>>>(
                transformed_x.template data<T>(),
                scale->template data<BatchNormParamType<T>>(),
                bias->template data<BatchNormParamType<T>>(), C, N, H * W * D,
                epsilon, this_factor, transformed_y.template data<T>(),
                mean_out->template data<BatchNormParamType<T>>(),
                variance_out->template data<BatchNormParamType<T>>(),
                saved_mean->template data<BatchNormParamType<T>>(),
                saved_variance->template data<BatchNormParamType<T>>());
          }

// TODO(wangran16): wait for MIOpen to improve the performance of BN
// PADDLE_ENFORCE_CUDA_SUCCESS(
//     platform::dynload::miopenBatchNormalizationForwardTraining(
//         handle, mode_, const_cast<void *>(static_cast<const void *>(
//                            CudnnDataType<T>::kOne())),
//         const_cast<void *>(
//             static_cast<const void *>(CudnnDataType<T>::kZero())),
//         data_desc_,
//         static_cast<const void *>(transformed_x.template data<T>()),
//         data_desc_,
//         static_cast<void *>(
//             transformed_y.template mutable_data<T>(ctx.GetPlace())),
//         bn_param_desc_,
//         const_cast<void *>(static_cast<const void *>(
//             scale->template data<BatchNormParamType<T>>())),
//         const_cast<void *>(static_cast<const void *>(
//             bias->template data<BatchNormParamType<T>>())),
//         this_factor,
//         static_cast<void *>(
//             mean_out->template mutable_data<BatchNormParamType<T>>(
//                 ctx.GetPlace())),
//         static_cast<void *>(variance_out->template mutable_data<
//                             BatchNormParamType<T>>(ctx.GetPlace())),
//         epsilon,
//         static_cast<void *>(
//             saved_mean->template mutable_data<BatchNormParamType<T>>(
//                 ctx.GetPlace())),
//         static_cast<void *>(saved_variance->template mutable_data<
//                             BatchNormParamType<T>>(ctx.GetPlace()))));
#else
          PADDLE_ENFORCE_CUDA_SUCCESS(
              platform::dynload::cudnnBatchNormalizationForwardTraining(
                  handle, mode, CudnnDataType<T>::kOne(),
                  CudnnDataType<T>::kZero(), data_desc,
                  transformed_x.template data<T>(), data_desc,
                  transformed_y.template mutable_data<T>(ctx.GetPlace()),
                  bn_param_desc, scale->template data<BatchNormParamType<T>>(),
                  bias->template data<BatchNormParamType<T>>(), this_factor,
                  mean_out->template mutable_data<BatchNormParamType<T>>(
                      ctx.GetPlace()),
                  variance_out->template mutable_data<BatchNormParamType<T>>(
                      ctx.GetPlace()),
                  epsilon,
                  saved_mean->template mutable_data<BatchNormParamType<T>>(
                      ctx.GetPlace()),
                  saved_variance->template mutable_data<BatchNormParamType<T>>(
                      ctx.GetPlace())));
#endif
        }
      }
    }

    if (data_layout == DataLayout::kNHWC &&
        compute_format == DataLayout::kNCHW && x_dims.size() > 2) {
      VLOG(3) << "Transform batchnorm output from NCHW to NHWC";
      TransToChannelLast<paddle::platform::CUDADeviceContext, T>(
          ctx, &transformed_y, y);
    }
#ifdef PADDLE_WITH_HIP
// TODO(wangran16): wait for MIOpen to improve the performance of BN
// clean when exit.
// PADDLE_ENFORCE_CUDA_SUCCESS(
//     platform::dynload::miopenDestroyTensorDescriptor(data_desc_));
// PADDLE_ENFORCE_CUDA_SUCCESS(
//     platform::dynload::miopenDestroyTensorDescriptor(bn_param_desc_));
#else
    // clean when exit.
    PADDLE_ENFORCE_CUDA_SUCCESS(
        platform::dynload::cudnnDestroyTensorDescriptor(data_desc));
    PADDLE_ENFORCE_CUDA_SUCCESS(
        platform::dynload::cudnnDestroyTensorDescriptor(bn_param_desc));
#endif
  }
};

template <typename T, int BlockDim, framework::DataLayout layout>
static __global__ LAUNCH_BOUNDS(BlockDim) void KeBNBackwardScaleBias(
    const T *dy, const T *x, const BatchNormParamType<T> *mean,
    const BatchNormParamType<T> *variance, const double epsilon, const int N,
    const int C, const int HxW, BatchNormParamType<T> *dscale,
    BatchNormParamType<T> *dbias) {
  const int outer_size = C;
  const int inner_size = N * HxW;
  typedef cub::BlockReduce<BatchNormParamType<T>, BlockDim> BlockReduce;
  __shared__ typename BlockReduce::TempStorage ds_storage;
  __shared__ typename BlockReduce::TempStorage db_storage;

  for (int i = blockIdx.x; i < outer_size; i += gridDim.x) {
    BatchNormParamType<T> ds_sum = static_cast<BatchNormParamType<T>>(0);
    BatchNormParamType<T> db_sum = static_cast<BatchNormParamType<T>>(0);

    BatchNormParamType<T> inv_var_i = 1.0 / sqrt(variance[i] + epsilon);
    BatchNormParamType<T> mean_i = mean[i];
    for (int j = threadIdx.x; j < inner_size; j += blockDim.x) {
      const int index = layout == framework::DataLayout::kNCHW
                            ? (j / HxW * C + i) * HxW + j % HxW
                            : j * outer_size + i;
      ds_sum += static_cast<BatchNormParamType<T>>(dy[index]) *
                (static_cast<BatchNormParamType<T>>(x[index]) - mean_i);
      db_sum += static_cast<BatchNormParamType<T>>(dy[index]);
    }
    ds_sum = BlockReduce(ds_storage).Reduce(ds_sum, cub::Sum());
    db_sum = BlockReduce(db_storage).Reduce(db_sum, cub::Sum());
    if (threadIdx.x == 0) {
      dscale[i] = ds_sum * inv_var_i;
      dbias[i] = db_sum;
    }
    __syncthreads();
  }
}

template <typename T, framework::DataLayout layout>
static __global__ void KeBNBackwardData(const T *dy,
                                        const BatchNormParamType<T> *scale,
                                        const BatchNormParamType<T> *variance,
                                        const double epsilon, const int C,
                                        const int HxW, const int num, T *dx) {
  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = gid; i < num; i += stride) {
    const int c = layout == framework::DataLayout::kNCHW ? i / HxW % C : i % C;
    BatchNormParamType<T> inv_var = 1.0 / sqrt(variance[c] + epsilon);
    dx[i] = static_cast<T>(static_cast<BatchNormParamType<T>>(dy[i]) *
                           scale[c] * inv_var);
  }
}

template <typename T>
static __global__ void KeBNRestoreData(const framework::DataLayout layout, T *x,
                                       const BatchNormParamType<T> *scale,
                                       const BatchNormParamType<T> *bias,
                                       const BatchNormParamType<T> *mean,
                                       const BatchNormParamType<T> *variance,
                                       double epsilon, int C, int M,
                                       const int num, const T *y) {
  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = gid; i < num; i += stride) {
    const int c = layout == framework::DataLayout::kNCHW ? (i / M) % C : i % C;
    auto y_i = static_cast<BatchNormParamType<T>>(y[i]);
    auto x_i = (y_i - bias[c]) / scale[c] / variance[c] + mean[c];
    x[i] = static_cast<T>(x_i);
  }
}

template <typename T>
class InplaceHelper {
 public:
  void operator()(const framework::DataLayout layout, T *x,
                  const BatchNormParamType<T> *scale,
                  const BatchNormParamType<T> *bias,
                  const BatchNormParamType<T> *mean,
                  const BatchNormParamType<T> *variance, double epsilon, int C,
                  int M, const int num, const T *y, int grid2, const int block,
                  const gpuStream_t &stream) {
    PADDLE_ENFORCE_EQ(x, y, platform::errors::InvalidArgument(
                                "X and Y should be inplaced in inplace mode"));
    KeBNRestoreData<<<grid2, block, 0, stream>>>(
        layout, x, scale, bias, mean, variance, epsilon, C, M, num, y);
  }
};

template <typename T, int BlockDim, framework::DataLayout layout>
static __global__ LAUNCH_BOUNDS(BlockDim) void BNBackward(
    const T *dy, const T *x, const BatchNormParamType<T> *scale,
    const BatchNormParamType<T> *saved_mean,
    const BatchNormParamType<T> *saved_inv_variance, const int C, const int N,
    const int HxW, const double epsilon, T *dx, BatchNormParamType<T> *dscale,
    BatchNormParamType<T> *dbias) {
  const int outer_size = C;
  const int inner_size = N * HxW;
  typedef cub::BlockReduce<BatchNormParamType<T>, BlockDim> BlockReduce;
  __shared__ typename BlockReduce::TempStorage ds_storage;
  __shared__ typename BlockReduce::TempStorage db_storage;
  __shared__ typename BlockReduce::TempStorage mean_storage;
  __shared__ typename BlockReduce::TempStorage variance_storeage;
  __shared__ BatchNormParamType<T> inv_var_val;
  __shared__ BatchNormParamType<T> mean_val;
  __shared__ BatchNormParamType<T> dscale_val;
  __shared__ BatchNormParamType<T> dbias_val;

  for (int i = blockIdx.x; i < outer_size; i += gridDim.x) {
    BatchNormParamType<T> ds_sum = static_cast<BatchNormParamType<T>>(0);
    BatchNormParamType<T> db_sum = static_cast<BatchNormParamType<T>>(0);

    if (saved_mean && saved_inv_variance) {
      if (threadIdx.x == 0) {
        inv_var_val = saved_inv_variance[i];
        mean_val = saved_mean[i];
      }
    } else {
      BatchNormParamType<T> x_sum = static_cast<BatchNormParamType<T>>(0);
      BatchNormParamType<T> x_square_sum =
          static_cast<BatchNormParamType<T>>(0);

      for (int j = threadIdx.x; j < inner_size; j += blockDim.x) {
        const int index = layout == framework::DataLayout::kNCHW
                              ? (j / HxW * C + i) * HxW + j % HxW
                              : j * outer_size + i;
        BatchNormParamType<T> x_i =
            static_cast<BatchNormParamType<T>>(x[index]);
        x_sum += x_i;
        x_square_sum += x_i * x_i;
      }
      x_sum = BlockReduce(mean_storage).Reduce(x_sum, cub::Sum());
      x_square_sum =
          BlockReduce(variance_storeage).Reduce(x_square_sum, cub::Sum());
      if (threadIdx.x == 0) {
        mean_val = x_sum / inner_size;
        inv_var_val =
            1 / sqrt(x_square_sum / inner_size - mean_val * mean_val + epsilon);
      }
    }
    __syncthreads();

    for (int j = threadIdx.x; j < inner_size; j += blockDim.x) {
      const int index = layout == framework::DataLayout::kNCHW
                            ? (j / HxW * C + i) * HxW + j % HxW
                            : j * outer_size + i;
      BatchNormParamType<T> dy_i =
          static_cast<BatchNormParamType<T>>(dy[index]);
      ds_sum +=
          dy_i * (static_cast<BatchNormParamType<T>>(x[index]) - mean_val);
      db_sum += dy_i;
    }
    ds_sum = BlockReduce(ds_storage).Reduce(ds_sum, cub::Sum());
    db_sum = BlockReduce(db_storage).Reduce(db_sum, cub::Sum());
    if (threadIdx.x == 0) {
      dscale_val = ds_sum * inv_var_val;
      dbias_val = db_sum;
      dscale[i] = dscale_val;
      dbias[i] = dbias_val;
    }
    __syncthreads();

    for (int j = threadIdx.x; j < inner_size; j += blockDim.x) {
      const int index = layout == framework::DataLayout::kNCHW
                            ? (j / HxW * C + i) * HxW + j % HxW
                            : j * outer_size + i;
      dx[index] = scale[i] * inv_var_val *
                  (static_cast<BatchNormParamType<T>>(dy[index]) -
                   dbias_val / static_cast<BatchNormParamType<T>>(inner_size) -
                   (static_cast<BatchNormParamType<T>>(x[index]) - mean_val) *
                       inv_var_val * dscale_val / inner_size);
    }
  }
}

template <typename T, int BlockDim, framework::DataLayout layout>
static __global__ LAUNCH_BOUNDS(BlockDim) void BNBackwardData(
    const T *dy, const BatchNormParamType<T> *scale,
    const BatchNormParamType<T> *mean, const T *x,
    const BatchNormParamType<T> *variance, const int C, const int N,
    const int HxW, T *dx) {
  const int outer_size = C;
  const int inner_size = N * HxW;
  typedef cub::BlockReduce<BatchNormParamType<T>, BlockDim> BlockReduce;
  __shared__ typename BlockReduce::TempStorage dy_storage;
  __shared__ typename BlockReduce::TempStorage dy_x_sub_mean_storage;
  __shared__ BatchNormParamType<T> dy_sum_val;
  __shared__ BatchNormParamType<T> dy_x_sub_mean_sum_val;

  for (int i = blockIdx.x; i < outer_size; i += gridDim.x) {
    BatchNormParamType<T> inv_var_i = variance[i];
    BatchNormParamType<T> mean_i = mean[i];
    BatchNormParamType<T> dy_sum = static_cast<BatchNormParamType<T>>(0);
    BatchNormParamType<T> dy_x_sub_mean_sum =
        static_cast<BatchNormParamType<T>>(0);
    for (int j = threadIdx.x; j < inner_size; j += blockDim.x) {
      const int index = layout == framework::DataLayout::kNCHW
                            ? (j / HxW * C + i) * HxW + j % HxW
                            : j * outer_size + i;
      BatchNormParamType<T> dy_i =
          static_cast<BatchNormParamType<T>>(dy[index]);
      dy_sum += dy_i;
      dy_x_sub_mean_sum +=
          dy_i * (static_cast<BatchNormParamType<T>>(x[index]) - mean_i);
    }

    dy_sum = BlockReduce(dy_storage).Reduce(dy_sum, cub::Sum());
    dy_x_sub_mean_sum = BlockReduce(dy_x_sub_mean_storage)
                            .Reduce(dy_x_sub_mean_sum, cub::Sum());

    if (threadIdx.x == 0) {
      dy_sum_val = dy_sum;
      dy_x_sub_mean_sum_val = dy_x_sub_mean_sum;
    }
    __syncthreads();
    for (int j = threadIdx.x; j < inner_size; j += blockDim.x) {
      const int index = layout == framework::DataLayout::kNCHW
                            ? (j / HxW * C + i) * HxW + j % HxW
                            : j * outer_size + i;
      dx[index] =
          (static_cast<BatchNormParamType<T>>(dy[index]) -
           dy_sum_val / static_cast<BatchNormParamType<T>>(inner_size) -
           (static_cast<BatchNormParamType<T>>(x[index]) - mean_i) *
               dy_x_sub_mean_sum_val * inv_var_i * inv_var_i / inner_size) *
          scale[i] * inv_var_i;
    }
  }
}

template <typename T>
class BatchNormGradKernel<platform::CUDADeviceContext, T>
    : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    PADDLE_ENFORCE_EQ(
        platform::is_gpu_place(ctx.GetPlace()), true,
        platform::errors::InvalidArgument("It must use CUDAPlace."));
    double epsilon = static_cast<double>(ctx.Attr<float>("epsilon"));
    const std::string data_layout_str = ctx.Attr<std::string>("data_layout");
    bool use_global_stats = ctx.Attr<bool>("use_global_stats");

    const DataLayout data_layout =
        framework::StringToDataLayout(data_layout_str);
    const auto *d_y = ctx.Input<Tensor>(framework::GradVarName("Y"));
    const auto *scale = ctx.Input<Tensor>("Scale");
    const auto *bias = ctx.Input<Tensor>("Bias");

    auto *d_x = ctx.Output<Tensor>(framework::GradVarName("X"));
    auto *d_scale = ctx.Output<Tensor>(framework::GradVarName("Scale"));
    auto *d_bias = ctx.Output<Tensor>(framework::GradVarName("Bias"));

    // batch_norm with inplace as false will take X as grad input, which
    // is same as cuDNN batch_norm backward calculation, batch_norm
    // with inplace as true only take Y as input and X should be calculate
    // by inverse operation of batch_norm on Y
    const Tensor *x;
    bool is_inplace;
    if (ctx.HasInput("Y")) {
      x = ctx.Input<Tensor>("Y");
      is_inplace = true;
      PADDLE_ENFORCE_EQ(d_x, d_y,
                        platform::errors::InvalidArgument(
                            "X@GRAD and Y@GRAD not inplace in inplace mode"));
    } else {
      x = ctx.Input<Tensor>("X");
      is_inplace = false;
      PADDLE_ENFORCE_NE(d_x, d_y,
                        platform::errors::InvalidArgument(
                            "X@GRAD and Y@GRAD inplaced in non-inplace mode"));
    }

    const bool is_test = ctx.Attr<bool>("is_test");
    use_global_stats = is_test || use_global_stats;

    const auto &x_dims = x->dims();

    PADDLE_ENFORCE_EQ(
        x_dims.size() >= 2 && x_dims.size() <= 5, true,
        platform::errors::InvalidArgument(
            "The size of input's dimensions should be between 2 and 5."
            "But received: the size of input's dimensions is [%d],"
            "the dimensions of input is [%s]",
            x_dims.size(), x_dims));
    int N, C, H, W, D;
    ExtractNCWHD(x_dims, data_layout, &N, &C, &H, &W, &D);

    // init output
    d_x->mutable_data<T>(ctx.GetPlace());

    if (d_scale && d_bias) {
      d_scale->mutable_data<BatchNormParamType<T>>(ctx.GetPlace());
      d_bias->mutable_data<BatchNormParamType<T>>(ctx.GetPlace());
    }
    PADDLE_ENFORCE_EQ(
        scale->dims().size(), 1UL,
        platform::errors::InvalidArgument(
            "The size of scale's dimensions must equal to 1. But received: "
            "the size of scale's dimensions is [%d], the dimensions of scale "
            "is [%s].",
            scale->dims().size(), scale->dims()));
    PADDLE_ENFORCE_EQ(
        scale->dims()[0], C,
        platform::errors::InvalidArgument(
            "The first dimension of scale must equal to Channels[%d]. But "
            "received: the first dimension of scale is [%d]",
            C, scale->dims()[0]));

    auto dtype = platform::CudnnDataType<T>::type;
    const auto *reserve_space = ctx.Input<Tensor>("ReserveSpace");
#ifdef PADDLE_WITH_HIP
    auto compute_format = data_layout == DataLayout::kNHWC ? DataLayout::kNHWC
                                                           : DataLayout::kNCHW;

// TODO(wangran16): wait for MIOpen to improve the performance of BN
// HIP do not support compute format of NHWC
// auto compute_format = DataLayout::kNCHW;
#else
    const bool fast_nhwc_batch_norm =
        dtype == CUDNN_DATA_HALF && FLAGS_cudnn_batchnorm_spatial_persistent &&
        reserve_space != nullptr;
    auto compute_format =
        fast_nhwc_batch_norm && data_layout == DataLayout::kNHWC
            ? DataLayout::kNHWC
            : DataLayout::kNCHW;
#endif

    Tensor transformed_x(x->type());
    Tensor transformed_d_y(d_y->type());
    Tensor transformed_d_x(d_x->type());
    if (data_layout == DataLayout::kNHWC &&
        compute_format == DataLayout::kNCHW) {
      VLOG(3) << "Transform input tensor from NHWC to NCHW.";
      ResizeToChannelFirst<platform::CUDADeviceContext, T>(ctx, x,
                                                           &transformed_x);
      TransToChannelFirst<platform::CUDADeviceContext, T>(ctx, x,
                                                          &transformed_x);
      ResizeToChannelFirst<platform::CUDADeviceContext, T>(ctx, d_y,
                                                           &transformed_d_y);
      TransToChannelFirst<platform::CUDADeviceContext, T>(ctx, d_y,
                                                          &transformed_d_y);
      ResizeToChannelFirst<platform::CUDADeviceContext, T>(ctx, d_x,
                                                           &transformed_d_x);
    } else {
      transformed_x.ShareDataWith(*x);
      transformed_d_y.ShareDataWith(*d_y);
      transformed_d_x.ShareDataWith(*d_x);
    }

    std::vector<int> dims;
    std::vector<int> strides;
    if (compute_format == DataLayout::kNCHW) {
      dims = {N, C, H, W, D};
      strides = {C * H * W * D, H * W * D, W * D, D, 1};
    } else {
      dims = {N, C, H, W, D};
      strides = {H * W * C * D, 1, W * D * C, D * C, C};
    }

    auto &dev_ctx = ctx.template device_context<platform::CUDADeviceContext>();
    const int num = transformed_x.numel();
#ifdef HIPCC
    const int block = 256;
#else
    const int block = 512;
#endif
    int max_threads = dev_ctx.GetMaxPhysicalThreadCount();
    const int max_blocks = std::max(max_threads / block, 1);
    int grid1 = (num + block - 1) / block;
    int grid2 = std::min(C, max_blocks);
    auto stream = dev_ctx.stream();
    InplaceHelper<T> inplace_functor;

    if (!use_global_stats) {
      if ((N * H * W * D) == 1) {
        framework::TensorCopy(*d_y, ctx.GetPlace(), d_x);
        math::SetConstant<platform::CUDADeviceContext, BatchNormParamType<T>>
            functor;
        functor(dev_ctx, d_scale, static_cast<BatchNormParamType<T>>(0));
        functor(dev_ctx, d_bias, static_cast<BatchNormParamType<T>>(0));
        return;
      }

// ------------------- cudnn descriptors ---------------------
#ifdef PADDLE_WITH_HIP
// TODO(wangran16): wait for MIOpen to improve the performance of BN
// miopenTensorDescriptor_t data_desc_;
// miopenTensorDescriptor_t bn_param_desc_;
// miopenBatchNormMode_t mode_;

// PADDLE_ENFORCE_CUDA_SUCCESS(
//     platform::dynload::miopenCreateTensorDescriptor(&data_desc_));
// PADDLE_ENFORCE_CUDA_SUCCESS(
//     platform::dynload::miopenCreateTensorDescriptor(&bn_param_desc_));
#else
      cudnnTensorDescriptor_t data_desc;
      cudnnTensorDescriptor_t bn_param_desc;
      cudnnBatchNormMode_t mode;

      PADDLE_ENFORCE_CUDA_SUCCESS(
          platform::dynload::cudnnCreateTensorDescriptor(&data_desc));
      PADDLE_ENFORCE_CUDA_SUCCESS(
          platform::dynload::cudnnCreateTensorDescriptor(&bn_param_desc));
#endif

      PADDLE_ENFORCE_GT(epsilon, CUDNN_BN_MIN_EPSILON - FLT_EPSILON,
                        platform::errors::InvalidArgument(
                            "Provided epsilon is expected to be greater than "
                            "CUDNN_BN_MIN_EPSILON. But recieved epsilon is %E, "
                            "CUDNN_BN_MIN_EPSILON is %E.",
                            epsilon, CUDNN_BN_MIN_EPSILON));
      epsilon = std::max(epsilon, CUDNN_BN_MIN_EPSILON);

#ifdef PADDLE_WITH_HIP
// TODO(wangran16): wait for MIOpen to improve the performance of BN
// mode_ = miopenBNSpatial;
#elif CUDNN_VERSION_MIN(7, 0, 1)
      if (FLAGS_cudnn_batchnorm_spatial_persistent) {
        mode = CUDNN_BATCHNORM_SPATIAL_PERSISTENT;
      } else {
        mode = CUDNN_BATCHNORM_SPATIAL;
      }
#else
      mode = CUDNN_BATCHNORM_SPATIAL;
#endif  // CUDNN_VERSION_MIN(7, 0, 1)

#ifdef PADDLE_WITH_HIP
// TODO(wangran16): wait for MIOpen to improve the performance of BN
// PADDLE_ENFORCE_CUDA_SUCCESS(platform::dynload::miopenSetTensorDescriptor(
//     data_desc_, CudnnDataType<T>::type,
//     x_dims.size() > 3 ? x_dims.size() : 4, const_cast<int *>(dims.data()),
//     const_cast<int *>(strides.data())));
// PADDLE_ENFORCE_CUDA_SUCCESS(
//     platform::dynload::miopenDeriveBNTensorDescriptor(bn_param_desc_,
//                                                       data_desc_, mode_));
#else
      PADDLE_ENFORCE_CUDA_SUCCESS(platform::dynload::cudnnSetTensorNdDescriptor(
          data_desc, CudnnDataType<T>::type,
          x_dims.size() > 3 ? x_dims.size() : 4, dims.data(), strides.data()));
      PADDLE_ENFORCE_CUDA_SUCCESS(
          platform::dynload::cudnnDeriveBNTensorDescriptor(bn_param_desc,
                                                           data_desc, mode));
#endif

      const auto *saved_mean = ctx.Input<Tensor>("SavedMean");
      const auto *saved_var = ctx.Input<Tensor>("SavedVariance");
      const auto *saved_mean_data =
          saved_mean->template data<BatchNormParamType<T>>();
      const auto *saved_var_data =
          saved_var->template data<BatchNormParamType<T>>();

      if (is_inplace) {
        inplace_functor(compute_format, transformed_x.data<T>(),
                        scale->template data<BatchNormParamType<T>>(),
                        bias->template data<BatchNormParamType<T>>(),
                        saved_mean_data, saved_var_data, epsilon, C, H * W * D,
                        num, transformed_x.data<T>(), grid2, block, stream);
      }

      // This branch calls CUDNN APIs
      if (d_scale && d_bias) {
        bool called = false;
#if CUDNN_VERSION_MIN(7, 4, 1)
        called = true;
        size_t workspace_size = 0;
        void *workspace_ptr = nullptr;
        Tensor workspace_tensor;
        auto reserve_space_size = reserve_space->memory_size();
        // --------------- cudnn batchnorm workspace ---------------
        PADDLE_ENFORCE_CUDA_SUCCESS(
            platform::dynload::
                cudnnGetBatchNormalizationBackwardExWorkspaceSize(
                    /*handle=*/dev_ctx.cudnn_handle(),
                    /*mode=*/mode,
                    /*bnIps=*/CUDNN_BATCHNORM_OPS_BN,
                    /*xDesc=*/data_desc,
                    /*yDesc=*/data_desc,
                    /*dyDesc=*/data_desc,
                    /*dzDesc=*/nullptr,
                    /*dxDesc=*/data_desc,
                    /*bnScaleBiasMeanVarDesc=*/bn_param_desc,
                    /*activationDesc=*/nullptr,
                    /*sizeInBytes=*/&workspace_size));

        workspace_ptr = workspace_tensor.mutable_data(
            ctx.GetPlace(), transformed_x.type(), workspace_size);

        PADDLE_ENFORCE_CUDA_SUCCESS(
            platform::dynload::cudnnBatchNormalizationBackwardEx(
                /*handle=*/dev_ctx.cudnn_handle(),
                /*mode=*/mode,
                /*bnOps=*/CUDNN_BATCHNORM_OPS_BN,
                /*alphaDataDiff=*/CudnnDataType<T>::kOne(),
                /*betaDataDiff=*/CudnnDataType<T>::kZero(),
                /*alphaParamDiff=*/CudnnDataType<T>::kOne(),
                /*betaParamDiff=*/CudnnDataType<T>::kZero(),
                /*xDesc=*/data_desc,
                /*xData=*/transformed_x.template data<T>(),
                /*yDesc=*/nullptr,
                /*yData=*/nullptr,
                /*dyDesc=*/data_desc,
                /*dyData=*/transformed_d_y.template data<T>(),
                /*dzDesc=*/nullptr,
                /*dzData=*/nullptr,
                /*dxDesc=*/data_desc,
                /*dxData=*/transformed_d_x.template mutable_data<T>(
                    ctx.GetPlace()),
                /*dBnScaleBiasDesc=*/bn_param_desc,
                /*bnScaleData=*/scale->template data<BatchNormParamType<T>>(),
                /*bnBiasData=*/nullptr,
                /*dBnScaleData=*/d_scale
                    ->template mutable_data<BatchNormParamType<T>>(
                        ctx.GetPlace()),
                /*dBnBiasData=*/d_bias
                    ->template mutable_data<BatchNormParamType<T>>(
                        ctx.GetPlace()),
                /*epsilon=*/epsilon,
                /*savedMean=*/saved_mean_data,
                /*savedInvVariance=*/saved_var_data,
                /*activationDesc=*/nullptr,
                /*workspace=*/workspace_ptr,
                /*workSpaceSizeInBytes=*/workspace_size,
                /*reserveSpace=*/const_cast<T *>(
                    reserve_space->template data<T>()),
                /*reserveSpaceSizeInBytes=*/reserve_space_size));
#endif  // CUDNN_VERSION_MIN(7, 4, 1)
        if (!called) {
#ifdef PADDLE_WITH_HIP
          if (compute_format == DataLayout::kNCHW) {
            BNBackward<
                T, block,
                DataLayout::kNCHW><<<grid2, block, 0, dev_ctx.stream()>>>(
                transformed_d_y.template data<T>(),
                transformed_x.template data<T>(),
                scale->template data<BatchNormParamType<T>>(), saved_mean_data,
                saved_var_data, C, N, H * W * D, epsilon,
                transformed_d_x.template data<T>(),
                d_scale->template mutable_data<BatchNormParamType<T>>(
                    ctx.GetPlace()),
                d_bias->template mutable_data<BatchNormParamType<T>>(
                    ctx.GetPlace()));
          } else {
            BNBackward<
                T, block,
                DataLayout::kNHWC><<<grid2, block, 0, dev_ctx.stream()>>>(
                transformed_d_y.template data<T>(),
                transformed_x.template data<T>(),
                scale->template data<BatchNormParamType<T>>(), saved_mean_data,
                saved_var_data, C, N, H * W * D, epsilon,
                transformed_d_x.template data<T>(),
                d_scale->template mutable_data<BatchNormParamType<T>>(
                    ctx.GetPlace()),
                d_bias->template mutable_data<BatchNormParamType<T>>(
                    ctx.GetPlace()));
          }

// TODO(wangran16): wait for MIOpen to improve the performance of BN
// PADDLE_ENFORCE_CUDA_SUCCESS(
//     platform::dynload::miopenBatchNormalizationBackward(
//         dev_ctx.cudnn_handle(), mode_, CudnnDataType<T>::kOne(),
//         CudnnDataType<T>::kZero(), CudnnDataType<T>::kOne(),
//         CudnnDataType<T>::kZero(), data_desc_,
//         transformed_x.template data<T>(), data_desc_,
//         transformed_d_y.template data<T>(), data_desc_,
//         transformed_d_x.template mutable_data<T>(ctx.GetPlace()),
//         bn_param_desc_, scale->template data<BatchNormParamType<T>>(),
//         d_scale->template mutable_data<BatchNormParamType<T>>(
//             ctx.GetPlace()),
//         d_bias->template mutable_data<BatchNormParamType<T>>(
//             ctx.GetPlace()),
//         epsilon, saved_mean_data, saved_var_data));
#else
          PADDLE_ENFORCE_CUDA_SUCCESS(
              platform::dynload::cudnnBatchNormalizationBackward(
                  dev_ctx.cudnn_handle(), mode, CudnnDataType<T>::kOne(),
                  CudnnDataType<T>::kZero(), CudnnDataType<T>::kOne(),
                  CudnnDataType<T>::kZero(), data_desc,
                  transformed_x.template data<T>(), data_desc,
                  transformed_d_y.template data<T>(), data_desc,
                  transformed_d_x.template mutable_data<T>(ctx.GetPlace()),
                  bn_param_desc, scale->template data<BatchNormParamType<T>>(),
                  d_scale->template mutable_data<BatchNormParamType<T>>(
                      ctx.GetPlace()),
                  d_bias->template mutable_data<BatchNormParamType<T>>(
                      ctx.GetPlace()),
                  epsilon, saved_mean_data, saved_var_data));
#endif
        }

        if (data_layout == DataLayout::kNHWC &&
            compute_format == DataLayout::kNCHW) {
          VLOG(3) << "Transform batchnorm output from NCHW to NHWC";
          TransToChannelLast<paddle::platform::CUDADeviceContext, T>(
              ctx, &transformed_d_x, d_x);
        }
      } else {
        // This branch call CUDA kernels
        if (compute_format == DataLayout::kNCHW) {
          if (d_x) {
            BNBackwardData<T, block, framework::DataLayout::kNCHW><<<
                grid2, block, 0, dev_ctx.stream()>>>(
                d_y->data<T>(), scale->data<BatchNormParamType<T>>(),
                saved_mean_data, x->data<T>(), saved_var_data, C, N, H * W * D,
                d_x->data<T>());
          }
        } else {
          if (d_x) {
            BNBackwardData<T, block, framework::DataLayout::kNHWC><<<
                grid2, block, 0, dev_ctx.stream()>>>(
                d_y->data<T>(), scale->data<BatchNormParamType<T>>(),
                saved_mean_data, x->data<T>(), saved_var_data, C, N, H * W * D,
                d_x->data<T>());
          }
        }
      }

#ifdef PADDLE_WITH_HIP
// TODO(wangran16): wait for MIOpen to improve the performance of BN
// clean when exit.
// PADDLE_ENFORCE_CUDA_SUCCESS(
//     platform::dynload::miopenDestroyTensorDescriptor(data_desc_));
// PADDLE_ENFORCE_CUDA_SUCCESS(
//     platform::dynload::miopenDestroyTensorDescriptor(bn_param_desc_));
#else
      // clean when exit.
      PADDLE_ENFORCE_CUDA_SUCCESS(
          platform::dynload::cudnnDestroyTensorDescriptor(data_desc));
      PADDLE_ENFORCE_CUDA_SUCCESS(
          platform::dynload::cudnnDestroyTensorDescriptor(bn_param_desc));
#endif
    } else {
      const auto *running_mean = ctx.Input<Tensor>("Mean");
      const auto *running_var = ctx.Input<Tensor>("Variance");

      const auto *running_mean_data =
          running_mean->template data<BatchNormParamType<T>>();
      const auto *running_var_data =
          running_var->template data<BatchNormParamType<T>>();

      if (is_inplace) {
        auto px = *x;
        inplace_functor(data_layout, px.mutable_data<T>(ctx.GetPlace()),
                        scale->template data<BatchNormParamType<T>>(),
                        bias->template data<BatchNormParamType<T>>(),
                        running_mean_data, running_var_data, epsilon, C,
                        H * W * D, num, x->data<T>(), grid2, block, stream);
      }

      if (compute_format == DataLayout::kNCHW) {
        if (d_x) {
          KeBNBackwardData<
              T, framework::DataLayout::kNCHW><<<grid1, block, 0, stream>>>(
              d_y->data<T>(), scale->data<BatchNormParamType<T>>(),
              running_var_data, epsilon, C, H * W, num, d_x->data<T>());
        }
        if (d_scale && d_bias) {
          KeBNBackwardScaleBias<
              T, block,
              framework::DataLayout::kNCHW><<<grid2, block, 0, stream>>>(
              d_y->data<T>(), x->data<T>(), running_mean_data, running_var_data,
              epsilon, N, C, H * W * D, d_scale->data<BatchNormParamType<T>>(),
              d_bias->data<BatchNormParamType<T>>());
        }
      } else {
        if (d_x) {
          KeBNBackwardData<
              T, framework::DataLayout::kNHWC><<<grid1, block, 0, stream>>>(
              d_y->data<T>(), scale->data<BatchNormParamType<T>>(),
              running_var_data, epsilon, C, H * W, num, d_x->data<T>());
        }
        if (d_scale && d_bias) {
          KeBNBackwardScaleBias<
              T, block,
              framework::DataLayout::kNHWC><<<grid2, block, 0, stream>>>(
              d_y->data<T>(), x->data<T>(), running_mean_data, running_var_data,
              epsilon, N, C, H * W * D, d_scale->data<BatchNormParamType<T>>(),
              d_bias->data<BatchNormParamType<T>>());
        }
      }
    }
  }
};

template <typename T>
class BatchNormDoubleGradKernel<platform::CUDADeviceContext, T>
    : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    const auto *X = ctx.Input<Tensor>("X");
    const auto *Scale = ctx.Input<Tensor>("Scale");
    const auto *dY = ctx.Input<Tensor>("DY");
    const auto *Saved_mean = ctx.Input<Tensor>("SavedMean");
    const auto *Saved_variance = ctx.Input<Tensor>("SavedVariance");
    const double epsilon = static_cast<double>(ctx.Attr<float>("epsilon"));
    const bool use_global_stats = ctx.Attr<bool>("use_global_stats");
    const bool is_test = ctx.Attr<bool>("is_test");

    PADDLE_ENFORCE_EQ(
        is_test, false,
        platform::errors::InvalidArgument(
            "`is_test = True` CANNOT be used in train program. If "
            "you want to use global status in pre_train model, "
            "please set `use_global_stats = True`"));

    const std::string data_layout_str = ctx.Attr<std::string>("data_layout");
    const DataLayout data_layout =
        framework::StringToDataLayout(data_layout_str);

    const auto *ddX = ctx.Input<Tensor>("DDX");
    const auto *ddScale = ctx.Input<Tensor>("DDScale");
    const auto *ddBias = ctx.Input<Tensor>("DDBias");

    auto *dX = ctx.Output<Tensor>("DX");
    auto *dScale = ctx.Output<Tensor>("DScale");
    auto *ddY = ctx.Output<Tensor>("DDY");

    NormDoubleGradFunctor<platform::CUDADeviceContext, T>(
        ctx, data_layout, X, Scale, dY, Saved_mean, Saved_variance, epsilon,
        use_global_stats, ddX, ddScale, ddBias, dX, dScale, ddY);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;
#ifdef PADDLE_WITH_HIP
// MIOPEN do not support double
REGISTER_OP_CUDA_KERNEL(
    batch_norm, ops::BatchNormKernel<plat::CUDADeviceContext, float>,
    ops::BatchNormKernel<plat::CUDADeviceContext, plat::float16>);
REGISTER_OP_CUDA_KERNEL(
    batch_norm_grad, ops::BatchNormGradKernel<plat::CUDADeviceContext, float>,
    ops::BatchNormGradKernel<plat::CUDADeviceContext, plat::float16>);
REGISTER_OP_CUDA_KERNEL(
    batch_norm_grad_grad,
    ops::BatchNormDoubleGradKernel<plat::CUDADeviceContext, float>);
#else
REGISTER_OP_CUDA_KERNEL(
    batch_norm, ops::BatchNormKernel<plat::CUDADeviceContext, float>,
    ops::BatchNormKernel<plat::CUDADeviceContext, double>,
    ops::BatchNormKernel<plat::CUDADeviceContext, plat::float16>);
REGISTER_OP_CUDA_KERNEL(
    batch_norm_grad, ops::BatchNormGradKernel<plat::CUDADeviceContext, float>,
    ops::BatchNormGradKernel<plat::CUDADeviceContext, double>,
    ops::BatchNormGradKernel<plat::CUDADeviceContext, plat::float16>);
REGISTER_OP_CUDA_KERNEL(
    batch_norm_grad_grad,
    ops::BatchNormDoubleGradKernel<plat::CUDADeviceContext, float>,
    ops::BatchNormDoubleGradKernel<plat::CUDADeviceContext, double>);
#endif
