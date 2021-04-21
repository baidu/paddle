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
#ifdef PADDLE_WITH_HIP
#include "paddle/fluid/operators/batch_norm_miopen_helper.h"
#else
#include "paddle/fluid/operators/batch_norm_cudnn_helper.h"
#endif
#include "paddle/fluid/operators/batch_norm_op.h"
#include "paddle/fluid/operators/math/math_function.h"
#include "paddle/fluid/operators/norm_utils.cu.h"
#include "paddle/fluid/platform/float16.h"

namespace paddle {
namespace operators {

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

    bool test_mode = is_test && (!trainable_stats);
    auto compute_format = GetComputeFormat<T>(data_layout, test_mode);

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

    PADDLE_ENFORCE_GT(epsilon, CUDNN_BN_MIN_EPSILON - FLT_EPSILON,
                      platform::errors::InvalidArgument(
                          "Provided epsilon is expected to be greater than "
                          "CUDNN_BN_MIN_EPSILON. But recieved epsilon is %E, "
                          "CUDNN_BN_MIN_EPSILON is %E.",
                          epsilon, CUDNN_BN_MIN_EPSILON));
    epsilon = std::max(epsilon, CUDNN_BN_MIN_EPSILON);

    std::vector<int> dims;
    std::vector<int> strides;
    if (compute_format == DataLayout::kNCHW) {
      dims = {N, C, H, W, D};
      strides = {C * H * W * D, H * W * D, W * D, D, 1};
    } else {
      dims = {N, C, H, W, D};
      strides = {H * W * D * C, 1, W * D * C, D * C, C};
    }

    // cudnn descriptors
    VLOG(3) << "Setting cudnn/miopen descriptors.";
    BatchNormMode mode = GetBatchNormMode(test_mode);
    BatchNormWrapper<T> bn_wrapper(x_dims.size() > 3 ? x_dims.size() : 4, dims,
                                   strides, mode, epsilon, false, false);

    const auto *scale = ctx.Input<Tensor>("Scale");
    const auto *bias = ctx.Input<Tensor>("Bias");

    auto &dev_ctx = ctx.template device_context<platform::CUDADeviceContext>();

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

      bn_wrapper.Infer(dev_ctx, transformed_x, *scale, *bias, *est_mean,
                       *est_var, &transformed_y);
    } else {
      // If MomentumTensor is set, use MomentumTensor value, momentum
      // is only used in this training branch
      if (ctx.HasInput("MomentumTensor")) {
        const auto *mom_tensor = ctx.Input<Tensor>("MomentumTensor");
        Tensor mom_cpu;
        TensorCopySync(*mom_tensor, platform::CPUPlace(), &mom_cpu);
        momentum = mom_cpu.data<float>()[0];
      }

      if ((N * H * W * D) == 1) {
        // Only 1 element in normalization dimension,
        // skip the batch norm calculation, let y = x.
        framework::TensorCopy(*x, ctx.GetPlace(), y);
      } else {
        double this_factor = 1. - momentum;

        auto *mean_out = ctx.Output<Tensor>("MeanOut");
        auto *variance_out = ctx.Output<Tensor>("VarianceOut");
        auto *saved_mean = ctx.Output<Tensor>("SavedMean");
        auto *saved_variance = ctx.Output<Tensor>("SavedVariance");
        auto *reserve_space = ctx.Output<Tensor>("ReserveSpace");

        saved_mean->mutable_data<BatchNormParamType<T>>(ctx.GetPlace());
        saved_variance->mutable_data<BatchNormParamType<T>>(ctx.GetPlace());
        math::SetConstant<platform::CUDADeviceContext, BatchNormParamType<T>>
            set_zero;
        set_zero(dev_ctx, saved_mean, static_cast<BatchNormParamType<T>>(0));
        set_zero(dev_ctx, saved_variance,
                 static_cast<BatchNormParamType<T>>(0));

        bn_wrapper.TrainForward(dev_ctx, transformed_x, *scale, *bias,
                                &transformed_y, mean_out, variance_out,
                                saved_mean, saved_variance, reserve_space,
                                this_factor);
      }
    }

    if (data_layout == DataLayout::kNHWC &&
        compute_format == DataLayout::kNCHW && x_dims.size() > 2) {
      VLOG(3) << "Transform batchnorm output from NCHW to NHWC";
      TransToChannelLast<paddle::platform::CUDADeviceContext, T>(
          ctx, &transformed_y, y);
    }
  }
};

template <typename T, int BlockDim, framework::DataLayout layout>
static __global__ void KeBNBackwardScaleBias(
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
static __global__ void BNBackwardData(const T *dy,
                                      const BatchNormParamType<T> *scale,
                                      const BatchNormParamType<T> *mean,
                                      const T *x,
                                      const BatchNormParamType<T> *variance,
                                      const int C, const int N, const int HxW,
                                      T *dx) {
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
    const bool use_global_stats = ctx.Attr<bool>("use_global_stats");

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
    PADDLE_ENFORCE_EQ(
        is_test, false,
        platform::errors::InvalidArgument(
            "`is_test = True` CANNOT be used in train program. If "
            "you want to use global status in pre_train model, "
            "please set `use_global_stats = True`"));

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

    const auto *reserve_space = ctx.Input<Tensor>("ReserveSpace");
    auto compute_format =
        GetComputeFormat<T>(data_layout, false, reserve_space != nullptr);

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
    const int block = 512;
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

      PADDLE_ENFORCE_GT(epsilon, CUDNN_BN_MIN_EPSILON - FLT_EPSILON,
                        platform::errors::InvalidArgument(
                            "Provided epsilon is expected to be greater than "
                            "CUDNN_BN_MIN_EPSILON. But recieved epsilon is %E, "
                            "CUDNN_BN_MIN_EPSILON is %E.",
                            epsilon, CUDNN_BN_MIN_EPSILON));
      epsilon = std::max(epsilon, CUDNN_BN_MIN_EPSILON);

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

      if (d_scale && d_bias) {
        // Call CUDNN APIs
        BatchNormMode mode = GetBatchNormMode(false);
        BatchNormWrapper<T> bn_wrapper(x_dims.size() > 3 ? x_dims.size() : 4,
                                       dims, strides, mode, epsilon, false,
                                       false);

        bn_wrapper.TrainBackward(
            dev_ctx, transformed_x, transformed_d_y, *scale, *saved_mean,
            *saved_var, *reserve_space, &transformed_d_x, d_scale, d_bias);

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
