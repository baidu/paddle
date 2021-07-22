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

#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/memory/memory.h"
#ifdef PADDLE_WITH_HIP
#include "paddle/fluid/operators/conv_miopen_helper.h"
#else
#include "paddle/fluid/operators/conv_cudnn_helper.h"
#endif
#include "paddle/fluid/operators/conv_transpose_op.h"
#include "paddle/fluid/operators/math/math_function.h"
#include "paddle/fluid/operators/math/padding.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;

template <typename T, int D>
static void DataTranspose(const framework::ExecutionContext& ctx,
                          const Tensor* input, Tensor* output,
                          const std::vector<int>& axis, int flag = 0) {
  auto& dev_ctx = ctx.template device_context<platform::CUDADeviceContext>();
  math::Transpose<platform::CUDADeviceContext, T, D> transpose;
  auto in_dims = input->dims();
  std::vector<int64_t> input_transpose_vec;
  for (size_t i = 0; i < axis.size(); ++i) {
    if (flag == 0)
      input_transpose_vec.push_back(in_dims[axis[i]]);
    else
      input_transpose_vec.push_back(in_dims[i]);
  }
  framework::DDim input_transpose_dims(
      framework::make_ddim(input_transpose_vec));
  output->mutable_data<T>(input_transpose_dims, ctx.GetPlace());
  transpose(dev_ctx, *input, output, axis);
}

template <typename T>
class CUDNNConvTransposeOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    PADDLE_ENFORCE_EQ(
        platform::is_gpu_place(ctx.GetPlace()), true,
        paddle::platform::errors::PreconditionNotMet("It must use CUDAPlace."));
    auto* input = ctx.Input<Tensor>("Input");
    auto* filter = ctx.Input<Tensor>("Filter");
    auto* output = ctx.Output<Tensor>("Output");

    std::vector<int> strides = ctx.Attr<std::vector<int>>("strides");
    std::vector<int> paddings = ctx.Attr<std::vector<int>>("paddings");
    std::string padding_algorithm = ctx.Attr<std::string>("padding_algorithm");

    // cudnn v5 does not support dilations
    std::vector<int> dilations = ctx.Attr<std::vector<int>>("dilations");
    int groups = ctx.Attr<int>("groups");
    const T* filter_data = filter->data<T>();
    const std::string data_layout_str = ctx.Attr<std::string>("data_format");
    const paddle::platform::CudnnDataLayout data_layout =
        (data_layout_str != "NHWC" ? platform::CudnnDataLayout::kNCHW
                                   : platform::CudnnDataLayout::kNHWC);

    // if channel_last, transpose to channel_first
    Tensor input_transpose;
    std::vector<int> input_vec = framework::vectorize<int>(input->dims());
    std::vector<int> output_vec = framework::vectorize<int>(output->dims());
    if (data_layout == platform::CudnnDataLayout::kNHWC) {
      if (strides.size() == 2U) {
        std::vector<int> axis = {0, 3, 1, 2};
        for (size_t i = 0; i < axis.size(); ++i) {
          input_vec[i] = input->dims()[axis[i]];
          output_vec[i] = output->dims()[axis[i]];
        }
        DataTranspose<T, 4>(ctx, input, &input_transpose, axis);
      } else if (strides.size() == 3U) {
        std::vector<int> axis = {0, 4, 1, 2, 3};
        for (size_t i = 0; i < axis.size(); ++i) {
          input_vec[i] = input->dims()[axis[i]];
          output_vec[i] = output->dims()[axis[i]];
        }
        DataTranspose<T, 5>(ctx, input, &input_transpose, axis);
      }
    } else {
      input_transpose = *input;
    }

    // update padding and dilation
    auto in_dims = input_transpose.dims();
    auto filter_dims = filter->dims();
    framework::DDim in_data_dims;
    in_data_dims = framework::slice_ddim(in_dims, 2, in_dims.size());
    framework::DDim filter_data_dims =
        framework::slice_ddim(filter_dims, 2, filter_dims.size());
    std::vector<int> ksize = framework::vectorize<int>(filter_data_dims);
    UpdatePaddingAndDilation(&paddings, &dilations, padding_algorithm,
                             in_data_dims, strides, ksize);

    int data_dim = strides.size();  // 2d or 3d
    bool is_sys_pad = math::IsSymmetricPadding(paddings, data_dim);

    std::vector<int> input_pad(input_transpose.dims().size() * 2, 0);
    Tensor transformed_input;
    std::vector<int> padding_common(data_dim, 0);
    if (!is_sys_pad) {
      std::vector<int> padding_diff(data_dim);
      std::vector<int> new_input_shape_vec(data_dim + 2);
      new_input_shape_vec[0] = input_transpose.dims()[0];
      new_input_shape_vec[1] = input_transpose.dims()[1];

      for (size_t i = 0; i < data_dim; ++i) {
        padding_diff[i] = std::abs(paddings[2 * i] - paddings[2 * i + 1]);
        padding_common[i] = std::min(paddings[2 * i], paddings[2 * i + 1]);
        new_input_shape_vec[i + 2] =
            input_transpose.dims()[i + 2] + padding_diff[i];
        input_pad[2 * i + 4] = paddings[2 * i] - padding_common[i];
        input_pad[2 * i + 4 + 1] = paddings[2 * i + 1] - padding_common[i];
      }
      framework::DDim new_input_shape(
          framework::make_ddim(new_input_shape_vec));
      transformed_input.Resize(new_input_shape);
      auto& dev_ctx =
          ctx.template device_context<paddle::platform::CUDADeviceContext>();

      transformed_input =
          ctx.AllocateTmpTensor<T, paddle::platform::CUDADeviceContext>(
              new_input_shape, dev_ctx);
      const int rank = input_transpose.dims().size();
      T pad_value(0.0);
      switch (rank) {
        case 4: {
          math::PadFunction<paddle::platform::CUDADeviceContext, T, 4>(
              ctx, input_pad, input_transpose, pad_value, &transformed_input);
        } break;
        case 5: {
          math::PadFunction<paddle::platform::CUDADeviceContext, T, 5>(
              ctx, input_pad, input_transpose, pad_value, &transformed_input);
        } break;
        default:
          PADDLE_THROW(platform::errors::InvalidArgument(
              "Op(ConvTranspose) only supports 4-D or 5-D input Tensor."));
      }
    } else {
      transformed_input = input_transpose;
      if (paddings.size() == data_dim) {
        for (size_t i = 0; i < data_dim; ++i) {
          padding_common[i] = paddings[i];
        }
      } else {
        for (size_t i = 0; i < data_dim; ++i) {
          padding_common[i] = paddings[2 * i];
        }
      }
    }

    std::vector<int64_t> starts(data_dim, 0);
    std::vector<int64_t> ends(data_dim, 0);
    std::vector<int64_t> axes(data_dim, 0);
    for (size_t i = 0; i < data_dim; ++i) {
      starts[i] = input_pad[2 * i + 4] * (strides[i] + 1);
      ends[i] = starts[i] + output_vec[i + 2];
      axes[i] = i + 2;
    }

    const T* input_data = transformed_input.data<T>();
    input_vec = framework::vectorize<int>(transformed_input.dims());

    std::vector<int> transformed_output_vec = output_vec;
    for (size_t i = 0; i < data_dim; ++i) {
      transformed_output_vec[i + 2] =
          output_vec[i + 2] +
          (input_pad[2 * i + 4] + input_pad[2 * i + 5]) * strides[i] -
          2 * padding_common[i] + paddings[2 * i] + paddings[2 * i + 1];
    }

    Tensor transformed_output;
    if (!is_sys_pad) {
      DDim transformed_output_shape(
          framework::make_ddim(transformed_output_vec));
      transformed_output.mutable_data<T>(transformed_output_shape,
                                         ctx.GetPlace());
    } else {
      output->mutable_data<T>(ctx.GetPlace());
      transformed_output.ShareDataWith(*output);
      transformed_output.Resize(framework::make_ddim(transformed_output_vec));
    }
    T* transformed_output_data = transformed_output.data<T>();

    platform::CudnnDataLayout layout;

    int iwo_groups = groups;
    int c_groups = 1;
#if defined(PADDLE_WITH_HIP) || CUDNN_VERSION_MIN(7, 0, 1)
    iwo_groups = 1;
    c_groups = groups;
    groups = 1;
#endif

    if (strides.size() == 2U) {
      layout = platform::CudnnDataLayout::kNCHW;
    } else {
      layout = platform::CudnnDataLayout::kNCDHW;
    }

    size_t workspace_size = 0;
#ifdef PADDLE_WITH_HIP
    miopenConvBwdDataAlgorithm_t algo{};
#else
    cudnnConvolutionBwdDataAlgo_t algo{};
#endif
    // ------------------- cudnn conv algorithm ---------------------
    auto& dev_ctx = ctx.template device_context<platform::CUDADeviceContext>();
    auto handle = dev_ctx.cudnn_handle();
    auto layout_tensor = GetCudnnTensorFormat(layout);
    bool deterministic = FLAGS_cudnn_deterministic;

    auto dtype = platform::CudnnDataType<T>::type;
    // ------------------- cudnn descriptors ---------------------
    ConvArgs args{&transformed_output,
                  filter,
                  &transformed_input,
                  strides,
                  padding_common,
                  dilations,
                  dtype};
    args.handle = handle;
    args.idesc.set(transformed_output, iwo_groups);
    args.wdesc.set(*filter, layout_tensor, iwo_groups);
    args.odesc.set(transformed_input, iwo_groups);
    args.cdesc.set(dtype, padding_common, strides, dilations,
                   platform::AllowTF32Cudnn(), c_groups);

#ifdef PADDLE_WITH_HIP
    using search = SearchAlgorithm<miopenConvBwdDataAlgorithm_t>;
    workspace_size = std::max(workspace_size, search::GetWorkspaceSize(args));
    algo = search::Find<T>(args, false, deterministic, workspace_size, ctx);
#else
    using search = SearchAlgorithm<cudnnConvolutionBwdDataAlgoPerf_t>;
    algo = search::Find<T>(args, false, deterministic, ctx);
    workspace_size =
        std::max(workspace_size, search::GetWorkspaceSize(args, algo));
#endif

    // ------------------- cudnn conv transpose forward ---------------------
    int input_offset =
        transformed_input.numel() / transformed_input.dims()[0] / groups;
    int output_offset =
        transformed_output.numel() / transformed_output.dims()[0] / groups;
    int filter_offset = filter->numel() / groups;
    ScalingParamType<T> alpha = 1.0f;
    ScalingParamType<T> beta = 0.0f;
    auto workspace_handle = dev_ctx.cudnn_workspace_handle();
    for (int g = 0; g < groups; g++) {
#ifdef PADDLE_WITH_HIP
      auto cudnn_func = [&](void* cudnn_workspace) {
        PADDLE_ENFORCE_CUDA_SUCCESS(
            platform::dynload::miopenConvolutionBackwardData(
                handle, &alpha, args.odesc.desc(),
                input_data + input_offset * g, args.wdesc.desc(),
                filter_data + filter_offset * g, args.cdesc.desc(), algo, &beta,
                args.idesc.desc(), transformed_output_data + output_offset * g,
                cudnn_workspace, workspace_size));
      };
#else   // PADDLE_WITH_HIP
      auto cudnn_func = [&](void* cudnn_workspace) {
        PADDLE_ENFORCE_CUDA_SUCCESS(
            platform::dynload::cudnnConvolutionBackwardData(
                handle, &alpha, args.wdesc.desc(),
                filter_data + filter_offset * g, args.odesc.desc(),
                input_data + input_offset * g, args.cdesc.desc(), algo,
                cudnn_workspace, workspace_size, &beta, args.idesc.desc(),
                transformed_output_data + output_offset * g));
      };
#endif  // PADDLE_WITH_HIP
      workspace_handle.RunFunc(cudnn_func, workspace_size);
    }
    if (!is_sys_pad && strides.size() == 2U) {
      Slice<paddle::platform::CUDADeviceContext, T, 4>(
          ctx, &transformed_output, output, starts, ends, axes);
    } else if (!is_sys_pad && strides.size() == 3U) {
      Slice<paddle::platform::CUDADeviceContext, T, 5>(
          ctx, &transformed_output, output, starts, ends, axes);
    }

    if (data_layout == platform::CudnnDataLayout::kNHWC) {
      Tensor output_transpose;
      Tensor output_nchw;
      output_nchw.ShareDataWith(*output);
      output_nchw.Resize(framework::make_ddim(output_vec));
      if (strides.size() == 2U) {
        std::vector<int> axis = {0, 2, 3, 1};
        DataTranspose<T, 4>(ctx, &output_nchw, &output_transpose, axis);
        *output = output_transpose;
      } else if (strides.size() == 3U) {
        std::vector<int> axis = {0, 2, 3, 4, 1};
        DataTranspose<T, 5>(ctx, &output_nchw, &output_transpose, axis);
        *output = output_transpose;
      }
    }
  }
};

template <typename T>
class CUDNNConvTransposeGradOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    PADDLE_ENFORCE_EQ(
        platform::is_gpu_place(ctx.GetPlace()), true,
        paddle::platform::errors::PreconditionNotMet("It must use CUDAPlace."));
    auto input = ctx.Input<Tensor>("Input");
    auto filter = ctx.Input<Tensor>("Filter");
    auto output_grad = ctx.Input<Tensor>(framework::GradVarName("Output"));
    auto input_grad = ctx.Output<Tensor>(framework::GradVarName("Input"));
    auto filter_grad = ctx.Output<Tensor>(framework::GradVarName("Filter"));
    const T* filter_data = filter->data<T>();

    std::vector<int> strides = ctx.Attr<std::vector<int>>("strides");
    std::vector<int> paddings = ctx.Attr<std::vector<int>>("paddings");
    // cudnn v5 does not support dilations
    std::vector<int> dilations = ctx.Attr<std::vector<int>>("dilations");
    int groups = ctx.Attr<int>("groups");
    std::string padding_algorithm = ctx.Attr<std::string>("padding_algorithm");
    int user_workspace_size = ctx.Attr<int>("workspace_size_MB");
    const std::string data_layout_str = ctx.Attr<std::string>("data_format");
    const paddle::platform::CudnnDataLayout data_layout =
        (data_layout_str != "NHWC" ? platform::CudnnDataLayout::kNCHW
                                   : platform::CudnnDataLayout::kNHWC);

    // if channel_last, transpose to channel_first
    Tensor input_transpose;
    Tensor output_grad_transpose;
    std::vector<int> input_vec = framework::vectorize<int>(input->dims());
    std::vector<int> output_vec =
        framework::vectorize<int>(output_grad->dims());
    if (data_layout == platform::CudnnDataLayout::kNHWC) {
      if (strides.size() == 2U) {
        std::vector<int> axis = {0, 3, 1, 2};
        for (size_t i = 0; i < axis.size(); ++i) {
          input_vec[i] = input->dims()[axis[i]];
          output_vec[i] = output_grad->dims()[axis[i]];
        }
        DataTranspose<T, 4>(ctx, input, &input_transpose, axis);
        DataTranspose<T, 4>(ctx, output_grad, &output_grad_transpose, axis);
      } else if (strides.size() == 3U) {
        std::vector<int> axis = {0, 4, 1, 2, 3};
        for (size_t i = 0; i < axis.size(); ++i) {
          input_vec[i] = input->dims()[axis[i]];
          output_vec[i] = output_grad->dims()[axis[i]];
        }
        DataTranspose<T, 5>(ctx, input, &input_transpose, axis);
        DataTranspose<T, 5>(ctx, output_grad, &output_grad_transpose, axis);
      }
    } else {
      input_transpose = *input;
      output_grad_transpose = *output_grad;
    }

    // update padding and dilation
    auto in_dims = input_transpose.dims();
    auto filter_dims = filter->dims();
    framework::DDim in_data_dims;
    in_data_dims = framework::slice_ddim(in_dims, 2, in_dims.size());
    framework::DDim filter_data_dims =
        framework::slice_ddim(filter_dims, 2, filter_dims.size());
    std::vector<int> ksize = framework::vectorize<int>(filter_data_dims);
    UpdatePaddingAndDilation(&paddings, &dilations, padding_algorithm,
                             in_data_dims, strides, ksize);

    int data_dim = strides.size();  // 2d or 3d
    bool is_sys_pad = math::IsSymmetricPadding(paddings, data_dim);

    std::vector<int> input_pad(input_transpose.dims().size() * 2, 0);
    Tensor transformed_output_grad;
    std::vector<int> padding_common(data_dim, 0);
    if (!is_sys_pad) {
      std::vector<int> padding_diff(data_dim);
      std::vector<int> new_output_grad_shape_vec(data_dim + 2);
      new_output_grad_shape_vec[0] = output_grad_transpose.dims()[0];
      new_output_grad_shape_vec[1] = output_grad_transpose.dims()[1];

      for (size_t i = 0; i < data_dim; ++i) {
        padding_diff[i] = std::abs(paddings[2 * i] - paddings[2 * i + 1]);
        padding_common[i] = std::min(paddings[2 * i], paddings[2 * i + 1]);
        new_output_grad_shape_vec[i + 2] =
            output_grad_transpose.dims()[i + 2] + padding_diff[i];
        input_pad[2 * i + 4] = paddings[2 * i] - padding_common[i];
        input_pad[2 * i + 4 + 1] = paddings[2 * i + 1] - padding_common[i];
      }
      framework::DDim new_output_grad_shape(
          framework::make_ddim(new_output_grad_shape_vec));
      transformed_output_grad.Resize(new_output_grad_shape);
      auto& dev_ctx =
          ctx.template device_context<paddle::platform::CUDADeviceContext>();

      transformed_output_grad =
          ctx.AllocateTmpTensor<T, paddle::platform::CUDADeviceContext>(
              new_output_grad_shape, dev_ctx);
      const int rank = input_transpose.dims().size();
      T pad_value(0.0);
      switch (rank) {
        case 4: {
          math::PadFunction<paddle::platform::CUDADeviceContext, T, 4>(
              ctx, input_pad, output_grad_transpose, pad_value,
              &transformed_output_grad);
        } break;
        case 5: {
          math::PadFunction<paddle::platform::CUDADeviceContext, T, 5>(
              ctx, input_pad, output_grad_transpose, pad_value,
              &transformed_output_grad);
        } break;
        default:
          PADDLE_THROW(platform::errors::InvalidArgument(
              "Op(ConvTranspose) only supports 4-D or 5-D input Tensor."));
      }
    } else {
      transformed_output_grad = output_grad_transpose;
      if (paddings.size() == data_dim) {
        for (size_t i = 0; i < data_dim; ++i) {
          padding_common[i] = paddings[i];
        }
      } else {
        for (size_t i = 0; i < data_dim; ++i) {
          padding_common[i] = paddings[2 * i];
        }
      }
    }

    const T* input_data = input_transpose.data<T>();
    const T* output_grad_data = transformed_output_grad.data<T>();
    output_vec = framework::vectorize<int>(transformed_output_grad.dims());

    // ------------------- cudnn descriptors ---------------------
    platform::CudnnDataLayout layout;

    if (strides.size() == 2U) {
      layout = platform::CudnnDataLayout::kNCHW;
    } else {
      layout = platform::CudnnDataLayout::kNCDHW;
    }

    int iwo_groups = groups;
    int c_groups = 1;
#if defined(PADDLE_WITH_HIP) || CUDNN_VERSION_MIN(7, 0, 1)
    iwo_groups = 1;
    c_groups = groups;
    groups = 1;
#endif

    auto dtype = platform::CudnnDataType<T>::type;

    ConvArgs args1{&transformed_output_grad,
                   filter,
                   &input_transpose,
                   strides,
                   padding_common,
                   dilations,
                   dtype};
    ConvArgs args2{&transformed_output_grad,
                   filter,
                   &input_transpose,
                   strides,
                   padding_common,
                   dilations,
                   dtype};

#ifdef PADDLE_WITH_HIP
    miopenConvFwdAlgorithm_t data_algo{};
    miopenConvBwdWeightsAlgorithm_t filter_algo{};
#else
    cudnnConvolutionFwdAlgo_t data_algo{};
    cudnnConvolutionBwdFilterAlgo_t filter_algo{};
#endif

    auto layout_tensor = GetCudnnTensorFormat(layout);
    size_t workspace_size = 0;
    auto& dev_ctx = ctx.template device_context<platform::CUDADeviceContext>();
    auto handle = dev_ctx.cudnn_handle();
    bool deterministic = FLAGS_cudnn_deterministic;
    T* input_grad_data = nullptr;
    T* filter_grad_data = nullptr;

    if (input_grad) {
      input_grad_data = input_grad->mutable_data<T>(ctx.GetPlace());
      args1.handle = handle;
      args1.idesc.set(transformed_output_grad, iwo_groups);
      args1.wdesc.set(*filter, layout_tensor, iwo_groups);
      args1.odesc.set(input_transpose, iwo_groups);
      args1.cdesc.set(dtype, padding_common, strides, dilations,
                      platform::AllowTF32Cudnn(), c_groups);
#ifdef PADDLE_WITH_HIP
      using search1 = SearchAlgorithm<miopenConvFwdAlgorithm_t>;
      workspace_size =
          std::max(workspace_size, search1::GetWorkspaceSize(args1));
      data_algo =
          search1::Find<T>(args1, false, deterministic, workspace_size, ctx);
#else
      using search1 = SearchAlgorithm<cudnnConvolutionFwdAlgoPerf_t>;
      data_algo = search1::Find<T>(args1, false, deterministic, ctx);
      workspace_size =
          std::max(workspace_size, search1::GetWorkspaceSize(args1, data_algo));
#endif
    }

    if (filter_grad) {
      filter_grad_data = filter_grad->mutable_data<T>(ctx.GetPlace());
      args2.handle = handle;
      args2.idesc.set(transformed_output_grad, iwo_groups);
      args2.wdesc.set(*filter_grad, layout_tensor, iwo_groups);
      args2.odesc.set(input_transpose, iwo_groups);
      args2.cdesc.set(dtype, padding_common, strides, dilations,
                      platform::AllowTF32Cudnn(), c_groups);
#ifdef PADDLE_WITH_HIP
      using search2 = SearchAlgorithm<miopenConvBwdWeightsAlgorithm_t>;
      workspace_size =
          std::max(workspace_size, search2::GetWorkspaceSize(args2));
      filter_algo =
          search2::Find<T>(args2, false, deterministic, workspace_size, ctx);
#else
      using search2 = SearchAlgorithm<cudnnConvolutionBwdFilterAlgoPerf_t>;
      filter_algo = search2::Find<T>(args2, false, deterministic, ctx);
      workspace_size = std::max(workspace_size,
                                search2::GetWorkspaceSize(args2, filter_algo));
#endif
    }

    // ------------------- cudnn conv backward data ---------------------
    // FIXME(typhoonzero): template type T may not be the same as cudnn call.
    int input_offset = input->numel() / input->dims()[0] / groups;
    int output_grad_offset = transformed_output_grad.numel() /
                             transformed_output_grad.dims()[0] / groups;
    int filter_offset = filter->numel() / groups;
    ScalingParamType<T> alpha = 1.0f;
    ScalingParamType<T> beta = 0.0f;
    auto workspace_handle = dev_ctx.cudnn_workspace_handle();
    if (input_grad) {
      // Because beta is zero, it is unnecessary to reset input_grad.
      for (int g = 0; g < groups; g++) {
#ifdef PADDLE_WITH_HIP
        auto cudnn_func = [&](void* cudnn_workspace) {
          PADDLE_ENFORCE_CUDA_SUCCESS(
              platform::dynload::miopenConvolutionForward(
                  handle, &alpha, args1.idesc.desc(),
                  output_grad_data + output_grad_offset * g, args1.wdesc.desc(),
                  filter_data + filter_offset * g, args1.cdesc.desc(),
                  data_algo, &beta, args1.odesc.desc(),
                  input_grad_data + input_offset * g, cudnn_workspace,
                  workspace_size));
        };
#else   // PADDLE_WITH_HIP
        auto cudnn_func = [&](void* cudnn_workspace) {
          PADDLE_ENFORCE_CUDA_SUCCESS(
              platform::dynload::cudnnConvolutionForward(
                  handle, &alpha, args1.idesc.desc(),
                  output_grad_data + output_grad_offset * g, args1.wdesc.desc(),
                  filter_data + filter_offset * g, args1.cdesc.desc(),
                  data_algo, cudnn_workspace, workspace_size, &beta,
                  args1.odesc.desc(), input_grad_data + input_offset * g));
        };
#endif  // PADDLE_WITH_HIP
        workspace_handle.RunFunc(cudnn_func, workspace_size);
      }

      if (data_layout == platform::CudnnDataLayout::kNHWC) {
        Tensor input_grad_transpose;
        Tensor input_grad_nchw;
        input_grad_nchw.ShareDataWith(*input_grad);
        input_grad_nchw.Resize(framework::make_ddim(input_vec));
        if (strides.size() == 2U) {
          std::vector<int> axis = {0, 2, 3, 1};
          DataTranspose<T, 4>(ctx, &input_grad_nchw, &input_grad_transpose,
                              axis);
          *input_grad = input_grad_transpose;
        } else if (strides.size() == 3U) {
          std::vector<int> axis = {0, 2, 3, 4, 1};
          DataTranspose<T, 5>(ctx, &input_grad_nchw, &input_grad_transpose,
                              axis);
          *input_grad = input_grad_transpose;
        }
      }
    }

    // ------------------- cudnn conv backward filter ---------------------
    if (filter_grad) {
      // Because beta is zero, it is unnecessary to reset filter_grad.
      // Gradient with respect to the filter
      for (int g = 0; g < groups; g++) {
#ifdef PADDLE_WITH_HIP
        auto cudnn_func = [&](void* cudnn_workspace) {
          PADDLE_ENFORCE_CUDA_SUCCESS(
              platform::dynload::miopenConvolutionBackwardWeights(
                  handle, &alpha, args2.odesc.desc(),
                  input_data + input_offset * g, args2.idesc.desc(),
                  output_grad_data + output_grad_offset * g, args2.cdesc.desc(),
                  filter_algo, &beta, args2.wdesc.desc(),
                  filter_grad_data + filter_offset * g, cudnn_workspace,
                  workspace_size));
        };
#else   // PADDLE_WITH_HIP
        auto cudnn_func = [&](void* cudnn_workspace) {
          PADDLE_ENFORCE_CUDA_SUCCESS(
              platform::dynload::cudnnConvolutionBackwardFilter(
                  handle, &alpha, args2.idesc.desc(),
                  output_grad_data + output_grad_offset * g, args2.odesc.desc(),
                  input_data + input_offset * g, args2.cdesc.desc(),
                  filter_algo, cudnn_workspace, workspace_size, &beta,
                  args2.wdesc.desc(), filter_grad_data + filter_offset * g));
        };
#endif  // PADDLE_WITH_HIP
        workspace_handle.RunFunc(cudnn_func, workspace_size);
      }
    }
  }
};

/*
 * Inputs:  I, W, dO, ddI, ddW
 * Outputs: ddO, dW, dI
 * ddo = conv_bp_data(W, ddI) + conv_bp_data(ddW, I)
 * dW = conv_bp_filter(dO, ddI)
 * dI = conv(dO, ddW)
 */
template <typename T>
class CUDNNConvTransposeDoubleGradOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto& dev_ctx = ctx.template device_context<platform::CUDADeviceContext>();
    PADDLE_ENFORCE_EQ(
        platform::is_gpu_place(ctx.GetPlace()), true,
        paddle::platform::errors::PreconditionNotMet("It must use CUDAPlace."));
    auto X = ctx.Input<Tensor>("Input");
    auto W = ctx.Input<Tensor>("Filter");
    auto dO = ctx.Input<Tensor>("DOutput");
    auto ddX = ctx.Input<Tensor>("DDInput");
    auto ddW = ctx.Input<Tensor>("DDFilter");

    auto ddO = ctx.Output<Tensor>("DDOutput");
    auto dW = ctx.Output<Tensor>("DFilter");
    auto dX = ctx.Output<Tensor>("DInput");

    if (ddO) {
      ddO->mutable_data<T>(ctx.GetPlace());
      math::SetConstant<platform::CUDADeviceContext, T> set_zero;
      set_zero(dev_ctx, ddO, static_cast<T>(0));
    }
    if (dW) {
      dW->mutable_data<T>(ctx.GetPlace());
    }
    if (dX) {
      dX->mutable_data<T>(ctx.GetPlace());
    }

    const T* dy = dO->data<T>();
    const T* w = W->data<T>();

    const T* ddx = nullptr;
    const T* ddw = nullptr;
    T *dw, *dx, *ddy;
    dw = dx = ddy = nullptr;
    T* transformed_dx = nullptr;
    const std::vector<int>& strides = ctx.Attr<std::vector<int>>("strides");
    std::vector<int> dilations = ctx.Attr<std::vector<int>>("dilations");
    int groups = ctx.Attr<int>("groups");

    bool deterministic = FLAGS_cudnn_deterministic;

    std::vector<int> paddings = ctx.Attr<std::vector<int>>("paddings");

    std::string padding_algorithm = ctx.Attr<std::string>("padding_algorithm");
    const std::string data_format = ctx.Attr<std::string>("data_format");
    const bool channel_last = (data_format == "NHWC" || data_format == "NDHWC");

    // transform Tensors to channel first-----------
    Tensor transformed_X_channel(X->type());
    Tensor transformed_dO_channel(dO->type());
    Tensor transformed_ddX_channel(X->type());

    Tensor transformed_ddO_channel(dO->type());
    Tensor transformed_dX_channel(X->type());

    if (channel_last) {
      ResizeToChannelFirst<platform::CUDADeviceContext, T>(
          ctx, X, &transformed_X_channel);
      TransToChannelFirst<platform::CUDADeviceContext, T>(
          ctx, X, &transformed_X_channel);

      ResizeToChannelFirst<platform::CUDADeviceContext, T>(
          ctx, dO, &transformed_dO_channel);
      TransToChannelFirst<platform::CUDADeviceContext, T>(
          ctx, dO, &transformed_dO_channel);

      if (ddX) {
        ResizeToChannelFirst<platform::CUDADeviceContext, T>(
            ctx, ddX, &transformed_ddX_channel);
        TransToChannelFirst<platform::CUDADeviceContext, T>(
            ctx, ddX, &transformed_ddX_channel);
      }

      if (ddO) {
        ResizeToChannelFirst<platform::CUDADeviceContext, T>(
            ctx, ddO, &transformed_ddO_channel);
      }
      if (dX) {
        ResizeToChannelFirst<platform::CUDADeviceContext, T>(
            ctx, dX, &transformed_dX_channel);
        transformed_dX_channel.mutable_data<T>(ctx.GetPlace());
      }

    } else {
      transformed_X_channel = *X;
      transformed_dO_channel = *dO;
      if (ddX) {
        transformed_ddX_channel = *ddX;
      }
      if (dX) {
        transformed_dX_channel = *dX;
      }
    }
    std::vector<int> output_vec =
        framework::vectorize<int>(transformed_dO_channel.dims());

    auto in_dims = transformed_X_channel.dims();
    auto filter_dims = W->dims();
    framework::DDim in_data_dims =
        framework::slice_ddim(in_dims, 2, in_dims.size());
    framework::DDim filter_data_dims =
        framework::slice_ddim(filter_dims, 2, filter_dims.size());
    std::vector<int> ksize = framework::vectorize<int>(filter_data_dims);
    UpdatePaddingAndDilation(&paddings, &dilations, padding_algorithm,
                             in_data_dims, strides, ksize);

    int data_dim = strides.size();  // 2d or 3d
    bool is_sys_pad = math::IsSymmetricPadding(paddings, data_dim);
    Tensor transformed_X(X->type());
    Tensor transformed_ddX(X->type());

    Tensor transformed_dO(dO->type());

    std::vector<int> padding_common(data_dim, 0);
    std::vector<int> input_pad(X->dims().size() * 2, 0);

    if (!is_sys_pad) {
      // get pad
      std::vector<int> padding_diff(data_dim);
      std::vector<int> new_input_shape_vec(data_dim + 2);
      std::vector<int> new_output_grad_shape_vec(data_dim + 2);

      new_input_shape_vec[0] = transformed_X_channel.dims()[0];
      new_input_shape_vec[1] = transformed_X_channel.dims()[1];

      new_output_grad_shape_vec[0] = transformed_dO_channel.dims()[0];
      new_output_grad_shape_vec[1] = transformed_dO_channel.dims()[1];

      for (size_t i = 0; i < data_dim; ++i) {
        padding_diff[i] = std::abs(paddings[2 * i] - paddings[2 * i + 1]);
        padding_common[i] = std::min(paddings[2 * i], paddings[2 * i + 1]);
        new_input_shape_vec[i + 2] =
            transformed_X_channel.dims()[i + 2] + padding_diff[i];

        new_output_grad_shape_vec[i + 2] =
            transformed_dO_channel.dims()[i + 2] + padding_diff[i];

        input_pad[2 * i + 4] = paddings[2 * i] - padding_common[i];
        input_pad[2 * i + 4 + 1] = paddings[2 * i + 1] - padding_common[i];
      }
      framework::DDim new_input_shape(
          framework::make_ddim(new_input_shape_vec));
      transformed_X.Resize(new_input_shape);
      transformed_ddX.Resize(new_input_shape);

      framework::DDim new_output_grad_shape(
          framework::make_ddim(new_output_grad_shape_vec));
      transformed_dO.Resize(new_output_grad_shape);

      transformed_dO =
          ctx.AllocateTmpTensor<T, paddle::platform::CUDADeviceContext>(
              new_output_grad_shape, dev_ctx);

      transformed_X =
          ctx.AllocateTmpTensor<T, paddle::platform::CUDADeviceContext>(
              new_input_shape, dev_ctx);
      if (ddX) {
        transformed_ddX =
            ctx.AllocateTmpTensor<T, paddle::platform::CUDADeviceContext>(
                new_input_shape, dev_ctx);
      }

      // pad for input
      const int rank = X->dims().size();
      T pad_value(0.0);
      switch (rank) {
        case 4: {
          math::PadFunction<paddle::platform::CUDADeviceContext, T, 4>(
              ctx, input_pad, transformed_X_channel, pad_value, &transformed_X);
          if (dO) {
            math::PadFunction<paddle::platform::CUDADeviceContext, T, 4>(
                ctx, input_pad, transformed_dO_channel, pad_value,
                &transformed_dO);
          }

          if (ddX) {
            math::PadFunction<paddle::platform::CUDADeviceContext, T, 4>(
                ctx, input_pad, transformed_ddX_channel, pad_value,
                &transformed_ddX);
          }
        } break;
        case 5: {
          math::PadFunction<paddle::platform::CUDADeviceContext, T, 5>(
              ctx, input_pad, transformed_X_channel, pad_value, &transformed_X);
          if (ddX) {
            math::PadFunction<paddle::platform::CUDADeviceContext, T, 5>(
                ctx, input_pad, transformed_ddX_channel, pad_value,
                &transformed_ddX);
          }
        } break;
        default:
          PADDLE_THROW(platform::errors::InvalidArgument(
              "ConvOp only support tensors with 4 or 5 dimensions."));
      }

    } else {
      transformed_X = transformed_X_channel;
      transformed_dO = transformed_dO_channel;
      if (ddX) {
        transformed_ddX = transformed_ddX_channel;
      }

      if (paddings.size() == data_dim) {
        for (size_t i = 0; i < data_dim; ++i) {
          padding_common[i] = paddings[i];
        }
      } else {
        for (size_t i = 0; i < data_dim; ++i) {
          padding_common[i] = paddings[2 * i];
        }
      }
    }

    std::vector<int64_t> starts(data_dim, 0);
    std::vector<int64_t> ends(data_dim, 0);
    std::vector<int64_t> axes(data_dim, 0);
    for (size_t i = 0; i < data_dim; ++i) {
      starts[i] = input_pad[2 * i + 4] * (strides[i] + 1);
      ends[i] = starts[i] + output_vec[i + 2];
      axes[i] = i + 2;
    }

    std::vector<int> transformed_output_vec = output_vec;
    for (size_t i = 0; i < data_dim; ++i) {
      transformed_output_vec[i + 2] =
          output_vec[i + 2] +
          (input_pad[2 * i + 4] + input_pad[2 * i + 5]) * strides[i] -
          2 * padding_common[i] + paddings[2 * i] + paddings[2 * i + 1];
    }

    if (!is_sys_pad) {
      DDim transformed_output_shape(
          framework::make_ddim(transformed_output_vec));
      transformed_ddO_channel.mutable_data<T>(transformed_output_shape,
                                              ctx.GetPlace());
    } else {
      ddO->mutable_data<T>(ctx.GetPlace());
      transformed_ddO_channel = *ddO;
      transformed_ddO_channel.Resize(
          framework::make_ddim(transformed_output_vec));
    }

    const T* x = transformed_X.data<T>();

    int iwo_group = groups;
    int c_group = 1;
#if defined(PADDLE_WITH_HIP) || CUDNN_VERSION_MIN(7, 0, 1)
    iwo_group = 1;
    c_group = groups;
    groups = 1;
#endif
    auto dtype = platform::CudnnDataType<T>::type;

    auto handle = dev_ctx.cudnn_handle();

    ConvArgs args1{&transformed_ddO_channel,
                   W,
                   &transformed_ddX,
                   strides,
                   padding_common,
                   dilations,
                   dtype};
    ConvArgs args2{&transformed_ddO_channel, ddW,       &transformed_X, strides,
                   padding_common,           dilations, dtype};

    ConvArgs args3{&transformed_dO,
                   dW,
                   &transformed_ddX_channel,
                   strides,
                   padding_common,
                   dilations,
                   dtype};
    ConvArgs args4{
        &transformed_dO, ddW,  &transformed_dX_channel, strides, padding_common,
        dilations,       dtype};
#ifdef PADDLE_WITH_HIP
    miopenConvBwdDataAlgorithm_t bwd_algo1 =
        static_cast<miopenConvBwdDataAlgorithm_t>(0);
    miopenConvBwdDataAlgorithm_t bwd_algo2 =
        static_cast<miopenConvBwdDataAlgorithm_t>(0);
    miopenConvFwdAlgorithm_t data_algo =
        static_cast<miopenConvFwdAlgorithm_t>(0);
    miopenConvBwdWeightsAlgorithm_t filter_algo =
        static_cast<miopenConvBwdWeightsAlgorithm_t>(0);
#else
    cudnnConvolutionBwdDataAlgo_t bwd_algo1 =
        static_cast<cudnnConvolutionBwdDataAlgo_t>(0);
    cudnnConvolutionBwdDataAlgo_t bwd_algo2 =
        static_cast<cudnnConvolutionBwdDataAlgo_t>(0);
    cudnnConvolutionFwdAlgo_t data_algo =
        static_cast<cudnnConvolutionFwdAlgo_t>(0);
    cudnnConvolutionBwdFilterAlgo_t filter_algo =
        static_cast<cudnnConvolutionBwdFilterAlgo_t>(0);
#endif

    auto layout = GetCudnnTensorFormat(platform::CudnnDataLayout::kNCHW);

    // ddo = conv(ddI, W) + conv(I, ddW)
    size_t workspace_size = 0;

    T* transformed_ddy_channel = nullptr;

    if (ddO) {
      ddy = ddO->data<T>();
      transformed_ddy_channel = transformed_ddO_channel.data<T>();
      if (ddX) {
        args1.handle = handle;
        args1.idesc.set(transformed_ddO_channel, iwo_group);
        args1.wdesc.set(*W, layout, iwo_group);
        args1.odesc.set(transformed_ddX, iwo_group);
        args1.cdesc.set(dtype, padding_common, strides, dilations,
                        platform::AllowTF32Cudnn(), c_group);
#ifdef PADDLE_WITH_HIP
        using search1 = SearchAlgorithm<miopenConvBwdDataAlgorithm_t>;
        workspace_size = search1::GetWorkspaceSize(args1);
        bwd_algo1 =
            search1::Find<T>(args1, false, deterministic, workspace_size, ctx);
#else
        using search1 = SearchAlgorithm<cudnnConvolutionBwdDataAlgoPerf_t>;
        bwd_algo1 = search1::Find<T>(args1, false, deterministic, ctx);
        workspace_size = search1::GetWorkspaceSize(args1, bwd_algo1);
#endif
      }

      if (ddW) {
        ddw = ddW->data<T>();
        args2.handle = handle;
        args2.idesc.set(transformed_ddO_channel, iwo_group);
        args2.wdesc.set(*ddW, layout, iwo_group);
        args2.odesc.set(transformed_X, iwo_group);
        args2.cdesc.set(dtype, padding_common, strides, dilations,
                        platform::AllowTF32Cudnn(), c_group);
#ifdef PADDLE_WITH_HIP
        using search2 = SearchAlgorithm<miopenConvBwdDataAlgorithm_t>;
        workspace_size =
            std::max(workspace_size, search2::GetWorkspaceSize(args2));
        bwd_algo2 =
            search2::Find<T>(args2, false, deterministic, workspace_size, ctx);
#else
        using search2 = SearchAlgorithm<cudnnConvolutionBwdDataAlgoPerf_t>;
        bwd_algo2 = search2::Find<T>(args2, false, deterministic, ctx);
        workspace_size = std::max(workspace_size,
                                  search2::GetWorkspaceSize(args2, bwd_algo2));
#endif
      }
    }

    if (dW && ddX) {
      dw = dW->data<T>();
      args3.handle = handle;
      args3.idesc.set(transformed_dO, iwo_group);
      args3.wdesc.set(*dW, layout, iwo_group);

      args3.odesc.set(transformed_ddX_channel, iwo_group);

      args3.cdesc.set(dtype, padding_common, strides, dilations,
                      platform::AllowTF32Cudnn(), c_group);
#ifdef PADDLE_WITH_HIP
      using search3 = SearchAlgorithm<miopenConvBwdWeightsAlgorithm_t>;
      workspace_size =
          std::max(workspace_size, search3::GetWorkspaceSize(args3));
      filter_algo =
          search3::Find<T>(args3, false, deterministic, workspace_size, ctx);
#else
      using search3 = SearchAlgorithm<cudnnConvolutionBwdFilterAlgoPerf_t>;
      filter_algo = search3::Find<T>(args3, false, deterministic, ctx);
      workspace_size = std::max(workspace_size,
                                search3::GetWorkspaceSize(args3, filter_algo));
#endif
    }

    if (ddW && dX) {
      transformed_dx = transformed_dX_channel.data<T>();

      args4.handle = handle;
      args4.idesc.set(transformed_dO, iwo_group);
      args4.wdesc.set(*ddW, layout, iwo_group);
      args4.odesc.set(transformed_dX_channel, iwo_group);
      args4.cdesc.set(dtype, padding_common, strides, dilations,
                      platform::AllowTF32Cudnn(), c_group);
#ifdef PADDLE_WITH_HIP
      using search4 = SearchAlgorithm<miopenConvFwdAlgorithm_t>;
      workspace_size =
          std::max(workspace_size, search4::GetWorkspaceSize(args4));
      data_algo =
          search4::Find<T>(args4, false, deterministic, workspace_size, ctx);
#else
      using search4 = SearchAlgorithm<cudnnConvolutionFwdAlgoPerf_t>;
      data_algo = search4::Find<T>(args4, false, deterministic, ctx);
      workspace_size =
          std::max(workspace_size, search4::GetWorkspaceSize(args4, data_algo));
#endif
    }

    int i_n, i_c, i_d, i_h, i_w;
    GetNCDHW(transformed_X.dims(), platform::CudnnDataLayout::kNCHW, &i_n, &i_c,
             &i_d, &i_h, &i_w);

    int o_n, o_c, o_d, o_h, o_w;
    GetNCDHW(transformed_dO.dims(), platform::CudnnDataLayout::kNCHW, &o_n,
             &o_c, &o_d, &o_h, &o_w);

    int group_offset_in =
        transformed_X.numel() / transformed_X.dims()[0] / groups;
    int group_offset_out =
        transformed_dO.numel() / transformed_dO.dims()[0] / groups;
    int group_offset_filter = W->numel() / groups;

    ScalingParamType<T> alpha = 1.0f;
    ScalingParamType<T> beta = 0.0f;

    auto wkspace_handle = dev_ctx.cudnn_workspace_handle();

    if (ddO) {
      if (ddX) {
        ddx = transformed_ddX.data<T>();
        for (int i = 0; i < groups; i++) {
#ifdef PADDLE_WITH_HIP
          wkspace_handle.RunFunc(
              [&](void* workspace_ptr) {
                PADDLE_ENFORCE_CUDA_SUCCESS(
                    platform::dynload::miopenConvolutionBackwardData(
                        handle, &alpha, args1.odesc.desc(),
                        ddx + i * group_offset_in, args1.wdesc.desc(),
                        w + i * group_offset_filter, args1.cdesc.desc(),
                        bwd_algo1, &beta, args1.idesc.desc(),
                        transformed_ddy_channel + i * group_offset_out,
                        workspace_ptr, workspace_size));
              },
              workspace_size);
#else   // PADDLE_WITH_HIP
          wkspace_handle.RunFunc(
              [&](void* workspace_ptr) {
                PADDLE_ENFORCE_CUDA_SUCCESS(
                    platform::dynload::cudnnConvolutionBackwardData(
                        handle, &alpha, args1.wdesc.desc(),
                        w + i * group_offset_filter, args1.odesc.desc(),
                        ddx + i * group_offset_in, args1.cdesc.desc(),
                        bwd_algo1, workspace_ptr, workspace_size, &beta,
                        args1.idesc.desc(),
                        transformed_ddy_channel + i * group_offset_out));
              },
              workspace_size);
#endif  // PADDLE_WITH_HIP
        }
      }
      if (ddW) {
        for (int i = 0; i < groups; i++) {
#ifdef PADDLE_WITH_HIP
          // MIOPEN ONLY support beta to be 0.0f
          Tensor conv_x_ddw(dO->type());
          conv_x_ddw.Resize(transformed_ddO_channel.dims());
          T* conv_x_ddw_data = conv_x_ddw.mutable_data<T>(ctx.GetPlace());
          wkspace_handle.RunFunc(
              [&](void* workspace_ptr) {
                PADDLE_ENFORCE_CUDA_SUCCESS(
                    platform::dynload::miopenConvolutionBackwardData(
                        handle, &alpha, args2.odesc.desc(),
                        x + i * group_offset_in, args2.wdesc.desc(),
                        ddw + i * group_offset_filter, args2.cdesc.desc(),
                        bwd_algo2, &beta, args2.idesc.desc(),
                        conv_x_ddw_data + i * group_offset_out, workspace_ptr,
                        workspace_size));
              },
              workspace_size);
          PADDLE_ENFORCE_CUDA_SUCCESS(platform::dynload::miopenOpTensor(
              handle, miopenTensorOpAdd, &alpha, args2.idesc.desc(),
              transformed_ddy_channel + i * group_offset_out, &alpha,
              args2.idesc.desc(), conv_x_ddw_data + i * group_offset_out, &beta,
              args2.idesc.desc(),
              transformed_ddy_channel + i * group_offset_out));
#else   // PADDLE_WITH_HIP
          wkspace_handle.RunFunc(
              [&](void* workspace_ptr) {
                PADDLE_ENFORCE_CUDA_SUCCESS(
                    platform::dynload::cudnnConvolutionBackwardData(
                        handle, &alpha, args2.wdesc.desc(),
                        ddw + i * group_offset_filter, args2.odesc.desc(),
                        x + i * group_offset_in, args2.cdesc.desc(), bwd_algo2,
                        workspace_ptr, workspace_size, &alpha,
                        args2.idesc.desc(),
                        transformed_ddy_channel + i * group_offset_out));
              },
              workspace_size);
#endif  // PADDLE_WITH_HIP
        }
      }
      if ((!is_sys_pad) && (!channel_last)) {
        if (strides.size() == 2U) {
          Slice<paddle::platform::CUDADeviceContext, T, 4>(
              ctx, &transformed_ddO_channel, ddO, starts, ends, axes);
        } else if (!is_sys_pad && strides.size() == 3U) {
          Slice<paddle::platform::CUDADeviceContext, T, 5>(
              ctx, &transformed_ddO_channel, ddO, starts, ends, axes);
        }
      } else if ((!is_sys_pad) && (channel_last)) {
        if (strides.size() == 2U) {
          Slice<paddle::platform::CUDADeviceContext, T, 4>(
              ctx, &transformed_ddO_channel, &transformed_ddO_channel, starts,
              ends, axes);
        } else if (!is_sys_pad && strides.size() == 3U) {
          Slice<paddle::platform::CUDADeviceContext, T, 5>(
              ctx, &transformed_ddO_channel, &transformed_ddO_channel, starts,
              ends, axes);
        }

        TransToChannelLast<paddle::platform::CUDADeviceContext, T>(
            ctx, &transformed_ddO_channel, ddO);
      }
    }

    T* transformed_dy_channel = transformed_dO.data<T>();
    if (dW && ddX) {
      ddx = transformed_ddX_channel.data<T>();
      for (int i = 0; i < groups; i++) {
#ifdef PADDLE_WITH_HIP
        wkspace_handle.RunFunc(
            [&](void* workspace_ptr) {
              PADDLE_ENFORCE_CUDA_SUCCESS(
                  platform::dynload::miopenConvolutionBackwardWeights(
                      handle, &alpha, args3.odesc.desc(),
                      ddx + i * group_offset_in, args3.idesc.desc(),
                      transformed_dy_channel + i * group_offset_out,
                      args3.cdesc.desc(), filter_algo, &beta,
                      args3.wdesc.desc(), dw + i * group_offset_filter,
                      workspace_ptr, workspace_size));
            },
            workspace_size);
#else   // PADDLE_WITH_HIP
        wkspace_handle.RunFunc(
            [&](void* workspace_ptr) {
              PADDLE_ENFORCE_CUDA_SUCCESS(
                  platform::dynload::cudnnConvolutionBackwardFilter(
                      handle, &alpha, args3.idesc.desc(),
                      transformed_dy_channel + i * group_offset_out,
                      args3.odesc.desc(), ddx + i * group_offset_in,
                      args3.cdesc.desc(), filter_algo, workspace_ptr,
                      workspace_size, &beta, args3.wdesc.desc(),
                      dw + i * group_offset_filter));
            },
            workspace_size);
#endif  // PADDLE_WITH_HIP
      }
    }

    if (dX && ddW) {
      ddw = ddW->data<T>();
      for (int i = 0; i < groups; i++) {
#ifdef PADDLE_WITH_HIP
        wkspace_handle.RunFunc(
            [&](void* workspace_ptr) {
              PADDLE_ENFORCE_CUDA_SUCCESS(
                  platform::dynload::miopenConvolutionForward(
                      handle, &alpha, args4.idesc.desc(),
                      transformed_dy_channel + i * group_offset_out,
                      args4.wdesc.desc(), ddw + i * group_offset_filter,
                      args4.cdesc.desc(), data_algo, &beta, args4.odesc.desc(),
                      transformed_dx + i * group_offset_in, workspace_ptr,
                      workspace_size));
            },
            workspace_size);
#else   // PADDLE_WITH_HIP
        wkspace_handle.RunFunc(
            [&](void* workspace_ptr) {
              PADDLE_ENFORCE_CUDA_SUCCESS(
                  platform::dynload::cudnnConvolutionForward(
                      handle, &alpha, args4.idesc.desc(),
                      transformed_dy_channel + i * group_offset_out,
                      args4.wdesc.desc(), ddw + i * group_offset_filter,
                      args4.cdesc.desc(), data_algo, workspace_ptr,
                      workspace_size, &beta, args4.odesc.desc(),
                      transformed_dx + i * group_offset_in));
            },
            workspace_size);
#endif  // PADDLE_WITH_HIP
      }
      if (channel_last) {
        TransToChannelLast<paddle::platform::CUDADeviceContext, T>(
            ctx, &transformed_dX_channel, dX);
      }
    }
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;

#ifdef PADDLE_WITH_HIP
// MIOPEN do not support double
REGISTER_OP_KERNEL(conv2d_transpose, CUDNN, ::paddle::platform::CUDAPlace,
                   ops::CUDNNConvTransposeOpKernel<plat::float16>,
                   ops::CUDNNConvTransposeOpKernel<float>);
REGISTER_OP_KERNEL(conv2d_transpose_grad, CUDNN, ::paddle::platform::CUDAPlace,
                   ops::CUDNNConvTransposeGradOpKernel<plat::float16>,
                   ops::CUDNNConvTransposeGradOpKernel<float>);
REGISTER_OP_KERNEL(
    conv2d_transpose_grad_grad, CUDNN, plat::CUDAPlace,
    paddle::operators::CUDNNConvTransposeDoubleGradOpKernel<float>,
    paddle::operators::CUDNNConvTransposeDoubleGradOpKernel<plat::float16>);

REGISTER_OP_KERNEL(conv3d_transpose, CUDNN, ::paddle::platform::CUDAPlace,
                   ops::CUDNNConvTransposeOpKernel<plat::float16>,
                   ops::CUDNNConvTransposeOpKernel<float>);
REGISTER_OP_KERNEL(conv3d_transpose_grad, CUDNN, ::paddle::platform::CUDAPlace,
                   ops::CUDNNConvTransposeGradOpKernel<plat::float16>,
                   ops::CUDNNConvTransposeGradOpKernel<float>);
#else
REGISTER_OP_KERNEL(conv2d_transpose, CUDNN, ::paddle::platform::CUDAPlace,
                   ops::CUDNNConvTransposeOpKernel<plat::float16>,
                   ops::CUDNNConvTransposeOpKernel<float>,
                   ops::CUDNNConvTransposeOpKernel<double>);
REGISTER_OP_KERNEL(conv2d_transpose_grad, CUDNN, ::paddle::platform::CUDAPlace,
                   ops::CUDNNConvTransposeGradOpKernel<plat::float16>,
                   ops::CUDNNConvTransposeGradOpKernel<float>,
                   ops::CUDNNConvTransposeGradOpKernel<double>);
REGISTER_OP_KERNEL(
    conv2d_transpose_grad_grad, CUDNN, plat::CUDAPlace,
    paddle::operators::CUDNNConvTransposeDoubleGradOpKernel<float>,
    paddle::operators::CUDNNConvTransposeDoubleGradOpKernel<double>,
    paddle::operators::CUDNNConvTransposeDoubleGradOpKernel<plat::float16>);

REGISTER_OP_KERNEL(conv3d_transpose, CUDNN, ::paddle::platform::CUDAPlace,
                   ops::CUDNNConvTransposeOpKernel<plat::float16>,
                   ops::CUDNNConvTransposeOpKernel<float>,
                   ops::CUDNNConvTransposeOpKernel<double>);
REGISTER_OP_KERNEL(conv3d_transpose_grad, CUDNN, ::paddle::platform::CUDAPlace,
                   ops::CUDNNConvTransposeGradOpKernel<plat::float16>,
                   ops::CUDNNConvTransposeGradOpKernel<float>,
                   ops::CUDNNConvTransposeGradOpKernel<double>);
#endif
