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

#include <string>
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/math/math_function.h"
#include "paddle/fluid/operators/pool_op.h"
#include "paddle/fluid/platform/cudnn_helper.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
using ScopedTensorDescriptor = platform::ScopedTensorDescriptor;
using ScopedPoolingDescriptor = platform::ScopedPoolingDescriptor;
using DataLayout = platform::DataLayout;
using PoolingMode = platform::PoolingMode;
template <typename T>
using ScalingParamType = typename platform::CudnnDataType<T>::ScalingParamType;

DataLayout getLayoutFromStr(std::string data_format) {
  if (data_format == "NHWC") {
    return DataLayout::kNHWC;
  } else if (data_format == "NCHW") {
    return DataLayout::kNCHW;
  } else if (data_format == "NCDHW") {
    return DataLayout::kNCDHW;
  } else {
    return DataLayout::kNCDHW;
  }
}

template <typename T>
class PoolCUDNNOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    PADDLE_ENFORCE_EQ(
        platform::is_gpu_place(ctx.GetPlace()), true,
        platform::errors::InvalidArgument("Pool operator CUDA kernel must use "
                                          "CUDAPlace rather than CPUPlace."));

    const Tensor *input = ctx.Input<Tensor>("X");
    Tensor *output = ctx.Output<Tensor>("Out");
    output->mutable_data<T>(ctx.GetPlace());
    std::string pooling_type = ctx.Attr<std::string>("pooling_type");
    bool exclusive = ctx.Attr<bool>("exclusive");
    bool adaptive = ctx.Attr<bool>("adaptive");
    std::vector<int> ksize = ctx.Attr<std::vector<int>>("ksize");
    std::vector<int> strides = ctx.Attr<std::vector<int>>("strides");
    std::vector<int> paddings = ctx.Attr<std::vector<int>>("paddings");
    std::string data_format = ctx.Attr<std::string>("data_format");
    bool global_pooling = ctx.Attr<bool>("global_pooling");
    std::string padding_algorithm = ctx.Attr<std::string>("padding_algorithm");
    const bool channel_last = (data_format == "NHWC" || data_format == "NDHWC");

    // update paddings
    auto in_x_dims = input->dims();
    framework::DDim data_dims;
    if (channel_last) {
      data_dims = framework::slice_ddim(in_x_dims, 1, in_x_dims.size() - 1);
    } else {
      data_dims = framework::slice_ddim(in_x_dims, 2, in_x_dims.size());
    }
    UpdatePadding(&paddings, global_pooling, adaptive, padding_algorithm,
                  data_dims, strides, ksize);
    if (data_dims.size() * 2 == static_cast<int>(paddings.size())) {
      for (int i = 0; i < data_dims.size(); ++i) {
        paddings.erase(paddings.begin() + i + 1);
      }
    }

    if (global_pooling) {
      UpdateKsize(&ksize, data_dims);
    }

    const std::string str_NCHW = "NCHW", str_NHWC = "NHWC";
    const std::string str_NCDHW = "NCDHW", str_NDHWC = "NDHWC";

    // -----------------transformed tensor ------------------------

    Tensor transformed_input(input->type());
    Tensor transformed_output(output->type());
    DataLayout layout;

    if (data_format == str_NDHWC) {
      layout = DataLayout::kNCDHW;
      auto &dev_ctx =
          ctx.template device_context<paddle::platform::CUDADeviceContext>();
      std::vector<int> axis{0, 4, 1, 2, 3};

      // input
      transformed_input.Resize(input->dims());

      auto in_dims_vec = framework::vectorize(input->dims());
      in_dims_vec[1] = input->dims()[4];
      in_dims_vec[2] = input->dims()[1];
      in_dims_vec[3] = input->dims()[2];
      in_dims_vec[4] = input->dims()[3];
      transformed_input.Resize(framework::make_ddim(in_dims_vec));
      transformed_input.mutable_data(ctx.GetPlace(), input->type());

      math::Transpose<paddle::platform::CUDADeviceContext, T, 5> trans5;
      trans5(dev_ctx, *input, &transformed_input, axis);

      // output
      transformed_output.Resize(output->dims());

      auto out_dims_vec = framework::vectorize(output->dims());
      out_dims_vec[1] = output->dims()[4];
      out_dims_vec[2] = output->dims()[1];
      out_dims_vec[3] = output->dims()[2];
      out_dims_vec[4] = output->dims()[3];
      transformed_output.Resize(framework::make_ddim(out_dims_vec));

    } else {
      layout = getLayoutFromStr(data_format);
      transformed_input = *input;
      transformed_output = *output;
    }

    const T *tranformed_input_data = transformed_input.data<T>();
    T *tranformed_output_data = transformed_output.mutable_data<T>(
        transformed_output.dims(), ctx.GetPlace());

    // ------------------- cudnn descriptors ---------------------
    ScopedTensorDescriptor input_desc;
    ScopedTensorDescriptor output_desc;
    ScopedPoolingDescriptor pool_desc;

    cudnnTensorDescriptor_t cudnn_input_desc = input_desc.descriptor<T>(
        layout, framework::vectorize<int>(transformed_input.dims()));
    cudnnTensorDescriptor_t cudnn_output_desc = output_desc.descriptor<T>(
        layout, framework::vectorize<int>(transformed_output.dims()));

    PoolingMode pooling_mode;
    if (pooling_type == "max") {
      pooling_mode = PoolingMode::kMaximum;
    } else {
      pooling_mode = exclusive ? PoolingMode::kAverageExclusive
                               : PoolingMode::kAverageInclusive;
    }

    cudnnPoolingDescriptor_t cudnn_pool_desc =
        pool_desc.descriptor(pooling_mode, ksize, paddings, strides);

    // ------------------- cudnn pool algorithm ---------------------
    auto handle = ctx.cuda_device_context().cudnn_handle();
    ScalingParamType<T> alpha = 1.0f, beta = 0.0f;

    PADDLE_ENFORCE_CUDA_SUCCESS(platform::dynload::cudnnPoolingForward(
        handle, cudnn_pool_desc, &alpha, cudnn_input_desc,
        tranformed_input_data, &beta, cudnn_output_desc,
        tranformed_output_data));
    // add
    if (data_format == str_NDHWC) {
      auto &dev_ctx =
          ctx.template device_context<paddle::platform::CUDADeviceContext>();
      std::vector<int> axis{0, 2, 3, 4, 1};
      math::Transpose<paddle::platform::CUDADeviceContext, T, 5> trans5_v2;
      trans5_v2(dev_ctx, transformed_output, output, axis);
    }
  }
};

template <typename T>
class PoolCUDNNGradOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    PADDLE_ENFORCE_EQ(
        platform::is_gpu_place(ctx.GetPlace()), true,
        platform::errors::InvalidArgument("Pool operator CUDA kernel must use "
                                          "CUDAPlace rather than CPUPlace."));

    const Tensor *input = ctx.Input<Tensor>("X");
    const Tensor *output = ctx.Input<Tensor>("Out");
    const Tensor *output_grad =
        ctx.Input<Tensor>(framework::GradVarName("Out"));
    Tensor *input_grad = ctx.Output<Tensor>(framework::GradVarName("X"));

    std::string pooling_type = ctx.Attr<std::string>("pooling_type");
    bool exclusive = ctx.Attr<bool>("exclusive");
    bool adaptive = ctx.Attr<bool>("adaptive");
    std::vector<int> ksize = ctx.Attr<std::vector<int>>("ksize");
    std::vector<int> strides = ctx.Attr<std::vector<int>>("strides");
    std::vector<int> paddings = ctx.Attr<std::vector<int>>("paddings");
    std::string data_format = ctx.Attr<std::string>("data_format");
    bool global_pooling = ctx.Attr<bool>("global_pooling");
    std::string padding_algorithm = ctx.Attr<std::string>("padding_algorithm");
    const bool channel_last = (data_format == "NHWC" || data_format == "NDHWC");

    // update paddings
    auto in_x_dims = input->dims();
    framework::DDim data_dims;
    if (channel_last) {
      data_dims = framework::slice_ddim(in_x_dims, 1, in_x_dims.size() - 1);
    } else {
      data_dims = framework::slice_ddim(in_x_dims, 2, in_x_dims.size());
    }
    UpdatePadding(&paddings, global_pooling, adaptive, padding_algorithm,
                  data_dims, strides, ksize);
    if (data_dims.size() * 2 == static_cast<int>(paddings.size())) {
      for (int i = 0; i < data_dims.size(); ++i) {
        paddings.erase(paddings.begin() + i + 1);
      }
    }

    if (global_pooling) {
      UpdateKsize(&ksize, data_dims);
    }

    // ------- tensor grad --------------
    Tensor transformed_input(input->type());
    Tensor transformed_output(output->type());
    Tensor transformed_output_grad(output_grad->type());

    input_grad->mutable_data<T>(ctx.GetPlace());
    Tensor transformed_input_grad(input_grad->type());
    DataLayout layout;
    const std::string str_NCHW = "NCHW", str_NHWC = "NHWC";
    const std::string str_NCDHW = "NCDHW", str_NDHWC = "NDHWC";
    if (data_format == str_NDHWC) {
      layout = DataLayout::kNCDHW;
      auto &dev_ctx =
          ctx.template device_context<paddle::platform::CUDADeviceContext>();
      std::vector<int> axis{0, 4, 1, 2, 3};

      // input
      transformed_input.Resize(input->dims());
      auto in_dims_vec = framework::vectorize(input->dims());
      in_dims_vec[1] = input->dims()[4];
      in_dims_vec[2] = input->dims()[1];
      in_dims_vec[3] = input->dims()[2];
      in_dims_vec[4] = input->dims()[3];
      transformed_input.Resize(framework::make_ddim(in_dims_vec));
      transformed_input.mutable_data(ctx.GetPlace(), input->type());

      math::Transpose<paddle::platform::CUDADeviceContext, T, 5> trans5;
      trans5(dev_ctx, *input, &transformed_input, axis);

      // output
      transformed_output.Resize(output->dims());
      auto out_dims_vec = framework::vectorize(output->dims());
      out_dims_vec[1] = output->dims()[4];
      out_dims_vec[2] = output->dims()[1];
      out_dims_vec[3] = output->dims()[2];
      out_dims_vec[4] = output->dims()[3];
      transformed_output.Resize(framework::make_ddim(out_dims_vec));

      transformed_output.mutable_data(ctx.GetPlace(), output->type());

      math::Transpose<paddle::platform::CUDADeviceContext, T, 5> trans5_v2;
      trans5_v2(dev_ctx, *output, &transformed_output, axis);

      // output grad
      transformed_output_grad.Resize(framework::make_ddim(out_dims_vec));
      transformed_output_grad.mutable_data(ctx.GetPlace(), output_grad->type());

      math::Transpose<paddle::platform::CUDADeviceContext, T, 5> trans5_v3;
      trans5_v3(dev_ctx, *output_grad, &transformed_output_grad, axis);

      // input grad
      transformed_input_grad.Resize(framework::make_ddim(in_dims_vec));

    } else {
      layout = getLayoutFromStr(data_format);
      transformed_input = *input;
      transformed_output = *output;
      transformed_output_grad = *output_grad;
      transformed_input_grad = *input_grad;
    }

    const T *input_data = transformed_input.data<T>();
    const T *output_data = transformed_output.data<T>();
    const T *output_grad_data = transformed_output_grad.data<T>();

    // ------------------- cudnn descriptors ---------------------
    ScopedTensorDescriptor input_desc;
    ScopedTensorDescriptor output_desc;
    ScopedPoolingDescriptor pool_desc;

    cudnnTensorDescriptor_t cudnn_input_desc = input_desc.descriptor<T>(
        layout, framework::vectorize<int>(transformed_input.dims()));
    cudnnTensorDescriptor_t cudnn_output_desc = output_desc.descriptor<T>(
        layout, framework::vectorize<int>(transformed_output.dims()));

    PoolingMode pooling_mode;
    if (pooling_type == "max") {
      if (FLAGS_cudnn_deterministic) {
        pooling_mode = PoolingMode::kMaximumDeterministic;
      } else {
        pooling_mode = PoolingMode::kMaximum;
      }
    } else {
      pooling_mode = exclusive ? PoolingMode::kAverageExclusive
                               : PoolingMode::kAverageInclusive;
    }

    cudnnPoolingDescriptor_t cudnn_pool_desc =
        pool_desc.descriptor(pooling_mode, ksize, paddings, strides);

    // ------------------- cudnn pool algorithm ---------------------
    auto handle = ctx.cuda_device_context().cudnn_handle();
    ScalingParamType<T> alpha = 1.0f, beta = 0.0f;
    if (input_grad) {
      T *input_grad_data = transformed_input_grad.mutable_data<T>(
          transformed_input_grad.dims(), ctx.GetPlace());
      // Because beta is zero, it is unnecessary to reset input_grad.
      PADDLE_ENFORCE_CUDA_SUCCESS(platform::dynload::cudnnPoolingBackward(
          handle, cudnn_pool_desc, &alpha, cudnn_output_desc, output_data,
          cudnn_output_desc, output_grad_data, cudnn_input_desc, input_data,
          &beta, cudnn_input_desc, input_grad_data));

      if (data_format == str_NDHWC) {
        auto &dev_ctx =
            ctx.template device_context<paddle::platform::CUDADeviceContext>();
        std::vector<int> axis{0, 2, 3, 4, 1};
        math::Transpose<paddle::platform::CUDADeviceContext, T, 5> trans5_v4;
        trans5_v4(dev_ctx, transformed_input_grad, input_grad, axis);
      }
    }
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;

REGISTER_OP_KERNEL(pool2d, CUDNN, plat::CUDAPlace,
                   ops::PoolCUDNNOpKernel<float>,
                   ops::PoolCUDNNOpKernel<double>,
                   ops::PoolCUDNNOpKernel<plat::float16>);
REGISTER_OP_KERNEL(pool2d_grad, CUDNN, plat::CUDAPlace,
                   ops::PoolCUDNNGradOpKernel<float>,
                   ops::PoolCUDNNGradOpKernel<double>,
                   ops::PoolCUDNNGradOpKernel<plat::float16>);

REGISTER_OP_KERNEL(pool3d, CUDNN, plat::CUDAPlace,
                   ops::PoolCUDNNOpKernel<float>,
                   ops::PoolCUDNNOpKernel<double>,
                   ops::PoolCUDNNOpKernel<plat::float16>);
REGISTER_OP_KERNEL(pool3d_grad, CUDNN, plat::CUDAPlace,
                   ops::PoolCUDNNGradOpKernel<float>,
                   ops::PoolCUDNNGradOpKernel<double>);
