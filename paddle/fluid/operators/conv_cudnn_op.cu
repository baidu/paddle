/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the spopecific language governing permissions and
limitations under the License. */

#include <utility>
#include <vector>

#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/memory/memory.h"
#ifdef PADDLE_WITH_HIP
#include "paddle/fluid/operators/conv_miopen_helper.h"
#else
#include "paddle/fluid/operators/conv_cudnn_helper.h"
#endif
#include "paddle/fluid/operators/conv_op.h"
#include "paddle/fluid/operators/math/padding.h"
#include "paddle/fluid/platform/cudnn_workspace_helper.h"
#include "paddle/fluid/platform/float16.h"
#include "paddle/fluid/platform/profiler.h"

DECLARE_bool(cudnn_deterministic);
DECLARE_uint64(conv_workspace_size_limit);
DECLARE_bool(cudnn_exhaustive_search);

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
using ScopedTensorDescriptor = platform::ScopedTensorDescriptor;
using ScopedFilterDescriptor = platform::ScopedFilterDescriptor;
using ScopedConvolutionDescriptor = platform::ScopedConvolutionDescriptor;
using DataLayout = platform::DataLayout;

static inline bool IsVoltaOrLater(const platform::CUDADeviceContext& dev_ctx) {
  return dev_ctx.GetComputeCapability() >= 70;
}

template <typename T>
class CUDNNConvOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto& dev_ctx = ctx.template device_context<platform::CUDADeviceContext>();
    PADDLE_ENFORCE_EQ(
        platform::is_gpu_place(ctx.GetPlace()), true,
        paddle::platform::errors::PreconditionNotMet("It must use CUDAPlace."));
    const Tensor* input = ctx.Input<Tensor>("Input");
    auto* filter = ctx.Input<Tensor>("Filter");
    auto* output = ctx.Output<Tensor>("Output");
    output->mutable_data<T>(ctx.GetPlace());
    const std::vector<int> strides = ctx.Attr<std::vector<int>>("strides");
    std::vector<int> paddings = ctx.Attr<std::vector<int>>("paddings");
    std::vector<int> dilations = ctx.Attr<std::vector<int>>("dilations");
    int groups = ctx.Attr<int>("groups");

    bool exhaustive_search =
        FLAGS_cudnn_exhaustive_search || ctx.Attr<bool>("exhaustive_search");
    bool deterministic = FLAGS_cudnn_deterministic;
    auto exhaustive_deterministic = exhaustive_search && deterministic;
    PADDLE_ENFORCE_EQ(exhaustive_deterministic, false,
                      platform::errors::InvalidArgument(
                          "Cann't set exhaustive_search True and "
                          "FLAGS_cudnn_deterministic True at same time."));

    const std::string padding_algorithm =
        ctx.Attr<std::string>("padding_algorithm");
    const std::string data_format = ctx.Attr<std::string>("data_format");
    const bool channel_last = (data_format == "NHWC" || data_format == "NDHWC");

    auto dtype = platform::CudnnDataType<T>::type;

#ifdef PADDLE_WITH_HIP
    // HIP MIOPEN ONLY SUPPORT NCHW format
    auto compute_format = DataLayout::kNCHW;
#else
    // Tensor Core introduced from Volta GPUs supports more faster conv op
    // with FP16 in NHWC data format.
    const bool compute_in_nhwc =
        dtype == CUDNN_DATA_HALF && IsVoltaOrLater(dev_ctx);
    // We will only do data format conversion from NHWC to NCHW.
    // cudnn will convert NCHW to NHWC automatically on Tensor Core.
    auto compute_format =
        compute_in_nhwc && channel_last ? DataLayout::kNHWC : DataLayout::kNCHW;
#endif
    VLOG(3) << "Compute ConvOp with cuDNN:"
            << " data_format=" << data_format << " compute_format="
            << (compute_format == DataLayout::kNHWC ? "NHWC" : "NCHW");

    // ------------ transformed tensor -----------
    Tensor transformed_input_channel(input->type());
    Tensor transformed_output(output->type());
    Tensor transformed_filter_channel(filter->type());
    T* output_data = nullptr;
    if (channel_last && compute_format == DataLayout::kNCHW) {
      VLOG(3) << "Transform input tensor from NHWC to NCHW.";
      ResizeToChannelFirst<platform::CUDADeviceContext, T>(
          ctx, input, &transformed_input_channel);
      TransToChannelFirst<platform::CUDADeviceContext, T>(
          ctx, input, &transformed_input_channel);

      ResizeToChannelFirst<platform::CUDADeviceContext, T>(ctx, output,
                                                           &transformed_output);

    } else {
      transformed_input_channel.ShareDataWith(*input);
      transformed_output.ShareDataWith(*output);
    }
    if (compute_format == DataLayout::kNHWC) {
      VLOG(3) << "Transform filter tensor from NCHW to NHWC.";
      ResizeToChannelLast<platform::CUDADeviceContext, T>(
          ctx, filter, &transformed_filter_channel);
      TransToChannelLast<platform::CUDADeviceContext, T>(
          ctx, filter, &transformed_filter_channel);
    } else {
      transformed_filter_channel.ShareDataWith(*filter);
    }
    output_data = transformed_output.data<T>();

    // update padding and dilation
    auto in_dims = transformed_input_channel.dims();
    auto filter_dims = transformed_filter_channel.dims();
    framework::DDim in_data_dims;
    framework::DDim filter_data_dims;

    if (compute_format == DataLayout::kNCHW) {
      in_data_dims = framework::slice_ddim(in_dims, 2, in_dims.size());
      filter_data_dims =
          framework::slice_ddim(filter_dims, 2, filter_dims.size());
    } else {
      in_data_dims = framework::slice_ddim(in_dims, 1, in_dims.size() - 1);
      filter_data_dims =
          framework::slice_ddim(filter_dims, 1, filter_dims.size() - 1);
    }

    std::vector<int> ksize = framework::vectorize<int>(filter_data_dims);
    UpdatePaddingAndDilation(&paddings, &dilations, padding_algorithm,
                             in_data_dims, strides, ksize);

    int data_dim = strides.size();  // 2d or 3d
    bool is_sys_pad = math::IsSymmetricPadding(paddings, data_dim);

    Tensor transformed_input;
    std::vector<int> padding_common(data_dim, 0);
    if (!is_sys_pad) {
      std::vector<int> padding_diff(data_dim);
      std::vector<int> new_input_shape_vec(data_dim + 2);
      new_input_shape_vec[0] = transformed_input_channel.dims()[0];

      if (compute_format == DataLayout::kNCHW) {
        new_input_shape_vec[1] = transformed_input_channel.dims()[1];
      } else {
        new_input_shape_vec[data_dim + 1] =
            transformed_input_channel.dims()[data_dim + 1];
      }

      std::vector<int> input_pad(transformed_input_channel.dims().size() * 2,
                                 0);
      for (size_t i = 0; i < data_dim; ++i) {
        padding_diff[i] = std::abs(paddings[2 * i] - paddings[2 * i + 1]);
        padding_common[i] = std::min(paddings[2 * i], paddings[2 * i + 1]);
        if (compute_format == DataLayout::kNCHW) {
          new_input_shape_vec[i + 2] =
              transformed_input_channel.dims()[i + 2] + padding_diff[i];
        } else {
          new_input_shape_vec[i + 1] =
              transformed_input_channel.dims()[i + 1] + padding_diff[i];
        }
        if (compute_format == DataLayout::kNCHW) {
          input_pad[2 * i + 4] = paddings[2 * i] - padding_common[i];
          input_pad[2 * i + 4 + 1] = paddings[2 * i + 1] - padding_common[i];
        } else {
          input_pad[2 * i + 2] = paddings[2 * i] - padding_common[i];
          input_pad[2 * i + 2 + 1] = paddings[2 * i + 1] - padding_common[i];
        }
      }
      framework::DDim new_input_shape(
          framework::make_ddim(new_input_shape_vec));
      transformed_input.Resize(new_input_shape);
      auto& dev_ctx =
          ctx.template device_context<paddle::platform::CUDADeviceContext>();

      transformed_input =
          ctx.AllocateTmpTensor<T, paddle::platform::CUDADeviceContext>(
              new_input_shape, dev_ctx);
      const int rank = transformed_input_channel.dims().size();
      T pad_value(0.0);
      switch (rank) {
        case 4: {
          math::PadFunction<paddle::platform::CUDADeviceContext, T, 4>(
              ctx, input_pad, transformed_input_channel, pad_value,
              &transformed_input);
        } break;
        case 5: {
          math::PadFunction<paddle::platform::CUDADeviceContext, T, 5>(
              ctx, input_pad, transformed_input_channel, pad_value,
              &transformed_input);
        } break;
        default:
          PADDLE_THROW(platform::errors::InvalidArgument(
              "ConvOp only support tensors with 4 or 5 dimensions."));
      }

    } else {
      transformed_input.ShareDataWith(transformed_input_channel);
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

    const T* input_data = transformed_input.data<T>();
    const T* filter_data = transformed_filter_channel.data<T>();

    // ------------------- cudnn descriptors ---------------------
    ConvArgs args{&transformed_input,
                  &transformed_filter_channel,
                  &transformed_output,
                  strides,
                  padding_common,
                  dilations,
                  dtype};

    auto handle = dev_ctx.cudnn_handle();
    auto workspace_handle = dev_ctx.cudnn_workspace_handle();
    DataLayout layout = compute_format == DataLayout::kNHWC ? DataLayout::kNHWC
                                                            : DataLayout::kNCHW;
    if (transformed_input.dims().size() == 5) {
      layout = compute_format == DataLayout::kNHWC ? DataLayout::kNDHWC
                                                   : DataLayout::kNCDHW;
    }
    auto layout_format = GetCudnnTensorFormat(layout);

    args.handle = handle;

#ifdef PADDLE_WITH_HIP
    // MIOPEN need to set groups in cdesc in miopen_desc.h
    args.cdesc.set(dtype, padding_common, strides, dilations,
                   platform::AllowTF32Cudnn(), groups);
#else
    args.cdesc.set(dtype, padding_common, strides, dilations,
                   platform::AllowTF32Cudnn());
#endif

#if defined(PADDLE_WITH_CUDA) && CUDNN_VERSION_MIN(7, 0, 1)
    // cudnn 7 can support groups, no need to do it manually
    // FIXME(typhoonzero): find a better way to disable groups
    // rather than setting it to 1.
    PADDLE_ENFORCE_CUDA_SUCCESS(
        platform::dynload::cudnnSetConvolutionGroupCount(args.cdesc.desc(),
                                                         groups));
    groups = 1;
#endif
#ifdef PADDLE_WITH_HIP
    // MIOPEN do not set groups in wdesc after set groups in cdesc
    groups = 1;
#endif
    args.idesc.set(transformed_input, layout_format);
    args.wdesc.set(transformed_filter_channel, layout_format, groups);
    args.odesc.set(transformed_output, layout_format);
    int i_n, i_c, i_d, i_h, i_w;
    int o_n, o_c, o_d, o_h, o_w;

    if (compute_format == DataLayout::kNHWC) {
      GetNCDHW(transformed_input.dims(), DataLayout::kNHWC, &i_n, &i_c, &i_d,
               &i_h, &i_w);
      GetNCDHW(transformed_output.dims(), DataLayout::kNHWC, &o_n, &o_c, &o_d,
               &o_h, &o_w);
    } else {
      GetNCDHW(transformed_input.dims(), DataLayout::kNCHW, &i_n, &i_c, &i_d,
               &i_h, &i_w);
      GetNCDHW(transformed_output.dims(), DataLayout::kNCHW, &o_n, &o_c, &o_d,
               &o_h, &o_w);
    }

    int group_offset_in = i_c / groups * i_h * i_w * i_d;
    int group_offset_out = o_c / groups * o_h * o_w * o_d;
    int group_offset_filter = transformed_filter_channel.numel() / groups;
    // ------------------- cudnn conv workspace ---------------------
    size_t workspace_size = 0;  // final workspace to allocate.
// ------------------- cudnn conv algorithm ---------------------
#ifdef PADDLE_WITH_HIP
    miopenConvFwdAlgorithm_t algo{};
    using search = SearchAlgorithm<miopenConvFwdAlgorithm_t>;
    workspace_size = search::GetWorkspaceSize(args);
    algo = search::Find<T>(args, exhaustive_search, false, workspace_size, ctx);
#else
    cudnnConvolutionFwdAlgo_t algo{};
    using search = SearchAlgorithm<cudnnConvolutionFwdAlgoPerf_t>;
    algo = search::Find<T>(args, exhaustive_search, false, ctx);
    workspace_size = search::GetWorkspaceSize(args, algo);
#endif

#if defined(PADDLE_WITH_CUDA) && CUDNN_VERSION_MIN(7, 0, 1)
    // when groups > 1, SearchAlgorithm find algo is CUDNN_CONVOLUTION_\
    // FWD_ALGO_WINOGRAD_NONFUSED, but this kind of algorithm is unstable
    // in forward computation, so change the algorithm to CUDNN_CONVOLUTION_\
    // FWD_ALGO_IMPLICIT_GEMM manually.
    if (ctx.Attr<int>("groups") > 1) {
      algo = static_cast<cudnnConvolutionFwdAlgo_t>(0);
    }
#endif

    // ------------------- cudnn conv forward ---------------------
    ScalingParamType<T> alpha = 1.0f;
    ScalingParamType<T> beta = 0.0f;

// NOTE(zhiqiu): inplace addto is not supportted in double grad yet.
// ScalingParamType<T> beta = ctx.Attr<bool>("use_addto") ? 1.0f : 0.0f;
// VLOG(4) << "Conv: use_addto = " << ctx.Attr<bool>("use_addto");

#ifdef PADDLE_WITH_HIP
    workspace_handle.RunFunc(
        [&](void* workspace_ptr) {
          PADDLE_ENFORCE_CUDA_SUCCESS(
              platform::dynload::miopenConvolutionForward(
                  handle, &alpha, args.idesc.desc(), input_data,
                  args.wdesc.desc(), filter_data, args.cdesc.desc(), algo,
                  &beta, args.odesc.desc(), output_data, workspace_ptr,
                  workspace_size));
        },
        workspace_size);
#else
    for (int i = 0; i < groups; i++) {
      workspace_handle.RunFunc(
          [&](void* workspace_ptr) {
            PADDLE_ENFORCE_CUDA_SUCCESS(
                platform::dynload::cudnnConvolutionForward(
                    handle, &alpha, args.idesc.desc(),
                    input_data + i * group_offset_in, args.wdesc.desc(),
                    filter_data + i * group_offset_filter, args.cdesc.desc(),
                    algo, workspace_ptr, workspace_size, &beta,
                    args.odesc.desc(), output_data + i * group_offset_out));
          },
          workspace_size);
    }
#endif

    if (channel_last && compute_format == DataLayout::kNCHW) {
      TransToChannelLast<paddle::platform::CUDADeviceContext, T>(
          ctx, &transformed_output, output);
    }
  }
};

template <typename T>
class CUDNNConvGradOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto& dev_ctx = ctx.template device_context<platform::CUDADeviceContext>();
    PADDLE_ENFORCE_EQ(
        platform::is_gpu_place(ctx.GetPlace()), true,
        paddle::platform::errors::PreconditionNotMet("It must use CUDAPlace."));
    auto input = ctx.Input<Tensor>("Input");
    auto filter = ctx.Input<Tensor>("Filter");
    auto output_grad = ctx.Input<Tensor>(framework::GradVarName("Output"));
    auto input_grad = ctx.Output<Tensor>(framework::GradVarName("Input"));
    auto filter_grad = ctx.Output<Tensor>(framework::GradVarName("Filter"));

    if (input_grad) {
      input_grad->mutable_data<T>(ctx.GetPlace());
    }
    if (filter_grad) {
      filter_grad->mutable_data<T>(ctx.GetPlace());
    }

    std::vector<int> dilations = ctx.Attr<std::vector<int>>("dilations");
    std::vector<int> strides = ctx.Attr<std::vector<int>>("strides");
    std::vector<int> paddings = ctx.Attr<std::vector<int>>("paddings");
    std::string padding_algorithm = ctx.Attr<std::string>("padding_algorithm");
    int groups = ctx.Attr<int>("groups");

    bool exhaustive_search =
        FLAGS_cudnn_exhaustive_search || ctx.Attr<bool>("exhaustive_search");
    bool deterministic = FLAGS_cudnn_deterministic;
    auto exhaustive_deterministic = exhaustive_search && deterministic;
    PADDLE_ENFORCE_EQ(exhaustive_deterministic, false,
                      platform::errors::InvalidArgument(
                          "Cann't set exhaustive_search True and "
                          "FLAGS_cudnn_deterministic True at same time."));

    const std::string data_format = ctx.Attr<std::string>("data_format");
    const bool channel_last = (data_format == "NHWC" || data_format == "NDHWC");

    auto dtype = platform::CudnnDataType<T>::type;

#ifdef PADDLE_WITH_HIP
    // HIP MIOPEN ONLY SUPPORT NCHW format
    auto compute_format = DataLayout::kNCHW;
#else
    const bool compute_in_nhwc =
        dtype == CUDNN_DATA_HALF && IsVoltaOrLater(dev_ctx);
    auto compute_format =
        compute_in_nhwc && channel_last ? DataLayout::kNHWC : DataLayout::kNCHW;
#endif
    VLOG(3) << "Compute ConvGradOp with cuDNN:"
            << " data_format=" << data_format << " compute_format="
            << (compute_format == DataLayout::kNHWC ? "NHWC" : "NCHW");

    // transform Tensor
    Tensor transformed_input_channel(input->type());
    Tensor transformed_output_grad_channel(output_grad->type());
    Tensor transformed_input_grad_channel(input->type());
    Tensor transformed_filter_channel(filter->type());
    Tensor transformed_filter_grad_channel(filter->type());

    if (channel_last && compute_format == DataLayout::kNCHW) {
      VLOG(3) << "Transform input, output_grad, input_grad and tensor from "
                 "NHWC to NCHW.";
      ResizeToChannelFirst<platform::CUDADeviceContext, T>(
          ctx, input, &transformed_input_channel);
      TransToChannelFirst<platform::CUDADeviceContext, T>(
          ctx, input, &transformed_input_channel);

      ResizeToChannelFirst<platform::CUDADeviceContext, T>(
          ctx, output_grad, &transformed_output_grad_channel);
      TransToChannelFirst<platform::CUDADeviceContext, T>(
          ctx, output_grad, &transformed_output_grad_channel);

      if (input_grad) {
        ResizeToChannelFirst<platform::CUDADeviceContext, T>(
            ctx, input_grad, &transformed_input_grad_channel);
        // NOTE(zhiqiu): If inplace_addto strategy is enabled, we need to copy
        // the data of input_grad to transformed_input_grad_channel.
        if (ctx.Attr<bool>("use_addto")) {
          TransToChannelFirst<platform::CUDADeviceContext, T>(
              ctx, input_grad, &transformed_input_grad_channel);
        }
      }
    } else {
      transformed_input_channel.ShareDataWith(*input);
      transformed_output_grad_channel.ShareDataWith(*output_grad);
      if (input_grad) {
        transformed_input_grad_channel.ShareDataWith(*input_grad);
      }
    }

    if (compute_format == DataLayout::kNHWC) {
      VLOG(3) << "Transform filter and filter_grad tensor from NCHW to NHWC.";
      ResizeToChannelLast<platform::CUDADeviceContext, T>(
          ctx, filter, &transformed_filter_channel);
      TransToChannelLast<platform::CUDADeviceContext, T>(
          ctx, filter, &transformed_filter_channel);

      if (filter_grad) {
        ResizeToChannelLast<platform::CUDADeviceContext, T>(
            ctx, filter_grad, &transformed_filter_grad_channel);
      }
    } else {
      transformed_filter_channel.ShareDataWith(*filter);
      if (filter_grad) {
        transformed_filter_grad_channel.ShareDataWith(*filter_grad);
      }
    }

    //  update paddings
    auto in_dims = transformed_input_channel.dims();
    auto filter_dims = transformed_filter_channel.dims();
    framework::DDim in_data_dims;
    framework::DDim filter_data_dims;
    if (compute_format == DataLayout::kNCHW) {
      in_data_dims = framework::slice_ddim(in_dims, 2, in_dims.size());
      filter_data_dims =
          framework::slice_ddim(filter_dims, 2, filter_dims.size());
    } else {
      in_data_dims = framework::slice_ddim(in_dims, 1, in_dims.size() - 1);
      filter_data_dims =
          framework::slice_ddim(filter_dims, 1, filter_dims.size() - 1);
    }
    std::vector<int> ksize = framework::vectorize<int>(filter_data_dims);
    UpdatePaddingAndDilation(&paddings, &dilations, padding_algorithm,
                             in_data_dims, strides, ksize);

    // cuDNN only supports padding the same amount on every dimension.
    // So we create a new padded input tensor.
    int data_dim = strides.size();  // 2d or 3d
    bool is_sys_pad = math::IsSymmetricPadding(paddings, data_dim);
    Tensor transformed_input(input->type());
    Tensor transformed_input_grad(input->type());
    std::vector<int> padding_common(data_dim, 0);
    std::vector<int> input_pad(transformed_input_channel.dims().size() * 2, 0);

    if (!is_sys_pad) {
      // get pad
      std::vector<int> padding_diff(data_dim);
      std::vector<int> new_input_shape_vec(data_dim + 2);
      new_input_shape_vec[0] = transformed_input_channel.dims()[0];
      if (compute_format == DataLayout::kNCHW) {
        new_input_shape_vec[1] = transformed_input_channel.dims()[1];
      } else {
        new_input_shape_vec[data_dim + 1] =
            transformed_input_channel.dims()[data_dim + 1];
      }

      for (size_t i = 0; i < data_dim; ++i) {
        padding_diff[i] = std::abs(paddings[2 * i] - paddings[2 * i + 1]);
        padding_common[i] = std::min(paddings[2 * i], paddings[2 * i + 1]);
        if (compute_format == DataLayout::kNCHW) {
          new_input_shape_vec[i + 2] =
              transformed_input_channel.dims()[i + 2] + padding_diff[i];
        } else {
          new_input_shape_vec[i + 1] =
              transformed_input_channel.dims()[i + 1] + padding_diff[i];
        }
        if (compute_format == DataLayout::kNCHW) {
          input_pad[2 * i + 4] = paddings[2 * i] - padding_common[i];
          input_pad[2 * i + 4 + 1] = paddings[2 * i + 1] - padding_common[i];
        } else {
          input_pad[2 * i + 2] = paddings[2 * i] - padding_common[i];
          input_pad[2 * i + 2 + 1] = paddings[2 * i + 1] - padding_common[i];
        }
      }
      framework::DDim new_input_shape(
          framework::make_ddim(new_input_shape_vec));
      transformed_input.Resize(new_input_shape);

      transformed_input_grad.Resize(new_input_shape);
      auto& dev_ctx =
          ctx.template device_context<paddle::platform::CUDADeviceContext>();

      transformed_input =
          ctx.AllocateTmpTensor<T, paddle::platform::CUDADeviceContext>(
              new_input_shape, dev_ctx);
      if (input_grad) {
        transformed_input_grad =
            ctx.AllocateTmpTensor<T, paddle::platform::CUDADeviceContext>(
                new_input_shape, dev_ctx);
      }
      // pad for input
      const int rank = transformed_input_channel.dims().size();
      T pad_value(0.0);
      switch (rank) {
        case 4: {
          math::PadFunction<paddle::platform::CUDADeviceContext, T, 4>(
              ctx, input_pad, transformed_input_channel, pad_value,
              &transformed_input);
        } break;
        case 5: {
          math::PadFunction<paddle::platform::CUDADeviceContext, T, 5>(
              ctx, input_pad, transformed_input_channel, pad_value,
              &transformed_input);
        } break;
        default:
          PADDLE_THROW(platform::errors::InvalidArgument(
              "ConvOp only support tensors with 4 or 5 dimensions."));
      }
    } else {
      transformed_input.ShareDataWith(transformed_input_channel);
      if (input_grad) {
        transformed_input_grad.ShareDataWith(transformed_input_grad_channel);
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

    const T* input_data = transformed_input.data<T>();
    const T* output_grad_data = transformed_output_grad_channel.data<T>();
    const T* filter_data = transformed_filter_channel.data<T>();
    T* filter_grad_data = nullptr;
    T* input_grad_data = nullptr;
    T* transformed_input_grad_data = nullptr;

    ConvArgs args1{&transformed_input_grad,
                   &transformed_filter_channel,
                   &transformed_output_grad_channel,
                   strides,
                   padding_common,
                   dilations,
                   dtype};
    ConvArgs args2{&transformed_input,
                   &transformed_filter_grad_channel,
                   &transformed_output_grad_channel,
                   strides,
                   padding_common,
                   dilations,
                   dtype};

    auto handle = dev_ctx.cudnn_handle();
    DataLayout layout = compute_format == DataLayout::kNHWC ? DataLayout::kNHWC
                                                            : DataLayout::kNCHW;
    if (transformed_input.dims().size() == 5) {
      layout = compute_format == DataLayout::kNHWC ? DataLayout::kNDHWC
                                                   : DataLayout::kNCDHW;
    }
    auto layout_tensor = GetCudnnTensorFormat(layout);
    auto workspace_handle = dev_ctx.cudnn_workspace_handle();

    int i_n, i_c, i_d, i_h, i_w;
    int o_n, o_c, o_d, o_h, o_w;
    if (compute_format == DataLayout::kNHWC) {
      GetNCDHW(transformed_input.dims(), DataLayout::kNHWC, &i_n, &i_c, &i_d,
               &i_h, &i_w);
      GetNCDHW(transformed_output_grad_channel.dims(), DataLayout::kNHWC, &o_n,
               &o_c, &o_d, &o_h, &o_w);
    } else {
      GetNCDHW(transformed_input.dims(), DataLayout::kNCHW, &i_n, &i_c, &i_d,
               &i_h, &i_w);
      GetNCDHW(transformed_output_grad_channel.dims(), DataLayout::kNCHW, &o_n,
               &o_c, &o_d, &o_h, &o_w);
    }

    int group_offset_in = i_c / groups * i_h * i_w * i_d;
    int group_offset_out = o_c / groups * o_h * o_w * o_d;
    int group_offset_filter = transformed_filter_channel.numel() / groups;
// ------------------- cudnn backward algorithm ---------------------
#ifdef PADDLE_WITH_HIP
    miopenConvBwdDataAlgorithm_t data_algo =
        static_cast<miopenConvBwdDataAlgorithm_t>(0);
    miopenConvBwdWeightsAlgorithm_t filter_algo =
        static_cast<miopenConvBwdWeightsAlgorithm_t>(0);
#else
    cudnnConvolutionBwdDataAlgo_t data_algo =
        static_cast<cudnnConvolutionBwdDataAlgo_t>(0);
    cudnnConvolutionBwdFilterAlgo_t filter_algo =
        static_cast<cudnnConvolutionBwdFilterAlgo_t>(0);
#endif
    size_t workspace_size = 0;
    int iwo_groups = groups;
    int c_groups = 1;

#if defined(PADDLE_WITH_HIP) || CUDNN_VERSION_MIN(7, 0, 1)
    iwo_groups = 1;
    c_groups = groups;
    groups = 1;
#endif

    if (input_grad) {
      // ------------------- cudnn descriptors ---------------------
      input_grad_data = input_grad->data<T>();
      transformed_input_grad_data = transformed_input_grad.data<T>();
      args1.handle = handle;
      args1.idesc.set(transformed_input_grad, layout_tensor);
      args1.wdesc.set(transformed_filter_channel, layout_tensor, iwo_groups);
      args1.odesc.set(transformed_output_grad_channel, layout_tensor);
      args1.cdesc.set(dtype, padding_common, strides, dilations,
                      platform::AllowTF32Cudnn(), c_groups);

#ifdef PADDLE_WITH_HIP
      using search1 = SearchAlgorithm<miopenConvBwdDataAlgorithm_t>;
      workspace_size =
          std::max(workspace_size, search1::GetWorkspaceSize(args1));
      data_algo = search1::Find<T>(args1, exhaustive_search, deterministic,
                                   workspace_size, ctx);
#else
      using search1 = SearchAlgorithm<cudnnConvolutionBwdDataAlgoPerf_t>;
      data_algo =
          search1::Find<T>(args1, exhaustive_search, deterministic, ctx);
      workspace_size =
          std::max(workspace_size, search1::GetWorkspaceSize(args1, data_algo));
#endif
    }

    if (filter_grad) {
      // ------------------- cudnn descriptors ---------------------
      filter_grad_data = transformed_filter_grad_channel.data<T>();
      args2.handle = handle;
      args2.idesc.set(transformed_input, layout_tensor);
      args2.wdesc.set(transformed_filter_grad_channel, layout_tensor,
                      iwo_groups);
      args2.odesc.set(transformed_output_grad_channel, layout_tensor);
      args2.cdesc.set(dtype, padding_common, strides, dilations,
                      platform::AllowTF32Cudnn(), c_groups);
#ifdef PADDLE_WITH_HIP
      using search2 = SearchAlgorithm<miopenConvBwdWeightsAlgorithm_t>;
      workspace_size =
          std::max(workspace_size, search2::GetWorkspaceSize(args2));
      filter_algo = search2::Find<T>(args2, exhaustive_search, deterministic,
                                     workspace_size, ctx);
#else
      using search2 = SearchAlgorithm<cudnnConvolutionBwdFilterAlgoPerf_t>;
      filter_algo =
          search2::Find<T>(args2, exhaustive_search, deterministic, ctx);
      workspace_size = std::max(workspace_size,
                                search2::GetWorkspaceSize(args2, filter_algo));
#endif
    }

    // ------------------- cudnn conv backward data ---------------------
    ScalingParamType<T> alpha = 1.0f;
#ifdef PADDLE_WITH_HIP
    // MIOPEN ONLY support beta to be 0.0f
    ScalingParamType<T> beta = 0.0f;
#else
    ScalingParamType<T> beta = ctx.Attr<bool>("use_addto") ? 1.0f : 0.0f;
#endif
    VLOG(4) << "Conv_grad: use_addto = " << ctx.Attr<bool>("use_addto");

    if (input_grad) {
// When beta is 0, it is unnecessary to reset input_grad.
// When beta is 1, the output cannot be reset since addt strategy used.
#ifdef PADDLE_WITH_HIP
      if (ctx.Attr<bool>("use_addto")) {
        Tensor temp_tensor(transformed_input_grad.type());
        temp_tensor.Resize(transformed_input_grad.dims());
        T* temp_tensor_data = temp_tensor.mutable_data<T>(ctx.GetPlace());
        workspace_handle.RunFunc(
            [&](void* cudnn_workspace_ptr) {
              PADDLE_ENFORCE_CUDA_SUCCESS(
                  platform::dynload::miopenConvolutionBackwardData(
                      handle, &alpha, args1.odesc.desc(), output_grad_data,
                      args1.wdesc.desc(), filter_data, args1.cdesc.desc(),
                      data_algo, &beta, args1.idesc.desc(), temp_tensor_data,
                      cudnn_workspace_ptr, workspace_size));
            },
            workspace_size);
        PADDLE_ENFORCE_CUDA_SUCCESS(platform::dynload::miopenOpTensor(
            handle, miopenTensorOpAdd, &alpha, args1.idesc.desc(),
            transformed_input_grad_data, &alpha, args1.idesc.desc(),
            temp_tensor_data, &beta, args1.idesc.desc(),
            transformed_input_grad_data));
      } else {
        workspace_handle.RunFunc(
            [&](void* cudnn_workspace_ptr) {
              PADDLE_ENFORCE_CUDA_SUCCESS(
                  platform::dynload::miopenConvolutionBackwardData(
                      handle, &alpha, args1.odesc.desc(), output_grad_data,
                      args1.wdesc.desc(), filter_data, args1.cdesc.desc(),
                      data_algo, &beta, args1.idesc.desc(),
                      transformed_input_grad_data, cudnn_workspace_ptr,
                      workspace_size));
            },
            workspace_size);
      }

#else
      for (int i = 0; i < groups; i++) {
        workspace_handle.RunFunc(
            [&](void* cudnn_workspace_ptr) {
              PADDLE_ENFORCE_CUDA_SUCCESS(
                  platform::dynload::cudnnConvolutionBackwardData(
                      handle, &alpha, args1.wdesc.desc(),
                      filter_data + i * group_offset_filter, args1.odesc.desc(),
                      output_grad_data + i * group_offset_out,
                      args1.cdesc.desc(), data_algo, cudnn_workspace_ptr,
                      workspace_size, &beta, args1.idesc.desc(),
                      transformed_input_grad_data + i * group_offset_in));
            },
            workspace_size);
      }
#endif
      if (!is_sys_pad) {
        std::vector<int> starts(transformed_input_channel.dims().size(), 0);
        std::vector<int> axes(transformed_input_channel.dims().size(), 0);

        for (size_t i = 0; i < transformed_input_channel.dims().size(); ++i) {
          starts[i] = input_pad[2 * i];
          axes[i] = i;
        }

        transformed_input_grad_channel.mutable_data(ctx.GetPlace());
        if (transformed_input_channel.dims().size() == 4) {
          RemovePaddingSlice<paddle::platform::CUDADeviceContext, T, 4>(
              ctx, &transformed_input_grad, &transformed_input_grad_channel,
              starts, axes);
        } else {
          RemovePaddingSlice<paddle::platform::CUDADeviceContext, T, 5>(
              ctx, &transformed_input_grad, &transformed_input_grad_channel,
              starts, axes);
        }
      }

      if (channel_last && compute_format == DataLayout::kNCHW) {
        TransToChannelLast<paddle::platform::CUDADeviceContext, T>(
            ctx, &transformed_input_grad_channel, input_grad);
      }
    }

    // filter_grad do not use inplace addto.
    ScalingParamType<T> beta_filter = 0.0f;
    // ------------------- cudnn conv backward filter ---------------------
    if (filter_grad) {
// Because beta is zero, it is unnecessary to reset filter_grad.
#ifdef PADDLE_WITH_HIP
      workspace_handle.RunFunc(
          [&](void* cudnn_workspace_ptr) {
            PADDLE_ENFORCE_CUDA_SUCCESS(
                platform::dynload::miopenConvolutionBackwardWeights(
                    handle, &alpha, args2.odesc.desc(), output_grad_data,
                    args2.idesc.desc(), input_data, args2.cdesc.desc(),
                    filter_algo, &beta, args2.wdesc.desc(), filter_grad_data,
                    cudnn_workspace_ptr, workspace_size));
          },
          workspace_size);
#else
      for (int i = 0; i < groups; i++) {
        workspace_handle.RunFunc(
            [&](void* cudnn_workspace_ptr) {
              PADDLE_ENFORCE_CUDA_SUCCESS(
                  platform::dynload::cudnnConvolutionBackwardFilter(
                      handle, &alpha, args2.idesc.desc(),
                      input_data + i * group_offset_in, args2.odesc.desc(),
                      output_grad_data + i * group_offset_out,
                      args2.cdesc.desc(), filter_algo, cudnn_workspace_ptr,
                      workspace_size, &beta_filter, args2.wdesc.desc(),
                      filter_grad_data + i * group_offset_filter));
            },
            workspace_size);
      }
#endif

      if (compute_format == DataLayout::kNHWC) {
        TransToChannelFirst<paddle::platform::CUDADeviceContext, T>(
            ctx, &transformed_filter_grad_channel, filter_grad);
      }
    }
  }
};

/*
 * Inputs:  I, W, dO, ddI, ddW
 * Outputs: ddO, dW, dI
 * ddo = conv(ddI, W) + conv(I, ddW)
 * dW = conv_bp_filter(ddI, dO)
 * dI = conv_bp_data(ddW, dO)
 */
template <typename T>
class CUDNNConvDoubleGradOpKernel : public framework::OpKernel<T> {
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

    // const T* x = X->data<T>();
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

    bool exhaustive_search =
        FLAGS_cudnn_exhaustive_search || ctx.Attr<bool>("exhaustive_search");
    bool deterministic = FLAGS_cudnn_deterministic;
    auto exhaustive_deterministic = exhaustive_search && deterministic;
    PADDLE_ENFORCE_EQ(exhaustive_deterministic, false,
                      platform::errors::InvalidArgument(
                          "Cann't set exhaustive_search True and "
                          "FLAGS_cudnn_deterministic True at same time."));

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
      if (ddO) {
        transformed_ddO_channel.ShareDataWith(*ddO);
      }
      if (dX) {
        transformed_dX_channel.ShareDataWith(*dX);
      }
    }

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

    Tensor transformed_dX(X->type());

    std::vector<int> padding_common(data_dim, 0);
    std::vector<int> input_pad(X->dims().size() * 2, 0);

    if (!is_sys_pad) {
      // get pad
      std::vector<int> padding_diff(data_dim);
      std::vector<int> new_input_shape_vec(data_dim + 2);
      new_input_shape_vec[0] = transformed_X_channel.dims()[0];
      new_input_shape_vec[1] = transformed_X_channel.dims()[1];

      for (size_t i = 0; i < data_dim; ++i) {
        padding_diff[i] = std::abs(paddings[2 * i] - paddings[2 * i + 1]);
        padding_common[i] = std::min(paddings[2 * i], paddings[2 * i + 1]);
        new_input_shape_vec[i + 2] =
            transformed_X_channel.dims()[i + 2] + padding_diff[i];
        input_pad[2 * i + 4] = paddings[2 * i] - padding_common[i];
        input_pad[2 * i + 4 + 1] = paddings[2 * i + 1] - padding_common[i];
      }
      framework::DDim new_input_shape(
          framework::make_ddim(new_input_shape_vec));
      transformed_X.Resize(new_input_shape);
      transformed_ddX.Resize(new_input_shape);
      transformed_dX.Resize(new_input_shape);

      transformed_X =
          ctx.AllocateTmpTensor<T, paddle::platform::CUDADeviceContext>(
              new_input_shape, dev_ctx);
      if (ddX) {
        transformed_ddX =
            ctx.AllocateTmpTensor<T, paddle::platform::CUDADeviceContext>(
                new_input_shape, dev_ctx);
      }
      if (dX) {
        transformed_dX =
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
      transformed_X.ShareDataWith(transformed_X_channel);
      if (ddX) {
        transformed_ddX.ShareDataWith(transformed_ddX_channel);
      }
      if (dX) {
        transformed_dX.ShareDataWith(transformed_dX_channel);
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

    ConvArgs args1{&transformed_ddX,
                   W,
                   &transformed_ddO_channel,
                   strides,
                   padding_common,
                   dilations,
                   dtype};
    ConvArgs args2{
        &transformed_X, ddW,  &transformed_ddO_channel, strides, padding_common,
        dilations,      dtype};
    ConvArgs args3{&transformed_ddX,
                   dW,
                   &transformed_dO_channel,
                   strides,
                   padding_common,
                   dilations,
                   dtype};
    ConvArgs args4{
        &transformed_dX, ddW,  &transformed_dO_channel, strides, padding_common,
        dilations,       dtype};

#ifdef PADDLE_WITH_HIP
    miopenConvFwdAlgorithm_t fwd_algo1 =
        static_cast<miopenConvFwdAlgorithm_t>(0);
    miopenConvFwdAlgorithm_t fwd_algo2 =
        static_cast<miopenConvFwdAlgorithm_t>(0);
    miopenConvBwdDataAlgorithm_t data_algo =
        static_cast<miopenConvBwdDataAlgorithm_t>(0);
    miopenConvBwdWeightsAlgorithm_t filter_algo =
        static_cast<miopenConvBwdWeightsAlgorithm_t>(0);
#else
    cudnnConvolutionFwdAlgo_t fwd_algo1 =
        static_cast<cudnnConvolutionFwdAlgo_t>(0);
    cudnnConvolutionFwdAlgo_t fwd_algo2 =
        static_cast<cudnnConvolutionFwdAlgo_t>(0);
    cudnnConvolutionBwdDataAlgo_t data_algo =
        static_cast<cudnnConvolutionBwdDataAlgo_t>(0);
    cudnnConvolutionBwdFilterAlgo_t filter_algo =
        static_cast<cudnnConvolutionBwdFilterAlgo_t>(0);
#endif

    auto layout = GetCudnnTensorFormat(DataLayout::kNCHW);

    // ddo = conv(ddI, W) + conv(I, ddW)
    size_t workspace_size = 0;

    T* transformed_ddy_channel = nullptr;
    if (ddO) {
      ddy = ddO->data<T>();
      transformed_ddy_channel = transformed_ddO_channel.data<T>();
      if (ddX) {
        args1.handle = handle;
        args1.idesc.set(transformed_ddX, iwo_group);
        args1.wdesc.set(*W, layout, iwo_group);
        args1.odesc.set(transformed_ddO_channel, iwo_group);
        args1.cdesc.set(dtype, padding_common, strides, dilations,
                        platform::AllowTF32Cudnn(), c_group);

#ifdef PADDLE_WITH_HIP
        using search1 = SearchAlgorithm<miopenConvFwdAlgorithm_t>;
        workspace_size = search1::GetWorkspaceSize(args1);
        fwd_algo1 = search1::Find<T>(args1, exhaustive_search, false,
                                     workspace_size, ctx);
#else
        using search1 = SearchAlgorithm<cudnnConvolutionFwdAlgoPerf_t>;
        fwd_algo1 = search1::Find<T>(args1, exhaustive_search, false, ctx);
        workspace_size = search1::GetWorkspaceSize(args1, fwd_algo1);
#endif
      }

      if (ddW) {
        ddw = ddW->data<T>();
        args2.handle = handle;
        args2.idesc.set(transformed_X, iwo_group);
        args2.wdesc.set(*ddW, layout, iwo_group);
        args2.odesc.set(transformed_ddO_channel, iwo_group);
        args2.cdesc.set(dtype, padding_common, strides, dilations,
                        platform::AllowTF32Cudnn(), c_group);

#ifdef PADDLE_WITH_HIP
        using search2 = SearchAlgorithm<miopenConvFwdAlgorithm_t>;
        workspace_size =
            std::max(workspace_size, search2::GetWorkspaceSize(args2));
        fwd_algo2 = search2::Find<T>(args2, exhaustive_search, false,
                                     workspace_size, ctx);
#else
        using search2 = SearchAlgorithm<cudnnConvolutionFwdAlgoPerf_t>;
        fwd_algo2 = search2::Find<T>(args2, exhaustive_search, false, ctx);
        workspace_size = std::max(workspace_size,
                                  search2::GetWorkspaceSize(args2, fwd_algo2));
#endif
      }
    }

    if (dW && ddX) {
      dw = dW->data<T>();
      args3.handle = handle;
      args3.idesc.set(transformed_ddX, iwo_group);
      args3.wdesc.set(*dW, layout, iwo_group);
      args3.odesc.set(transformed_dO_channel, iwo_group);
      args3.cdesc.set(dtype, padding_common, strides, dilations,
                      platform::AllowTF32Cudnn(), c_group);

#ifdef PADDLE_WITH_HIP
      using search3 = SearchAlgorithm<miopenConvBwdWeightsAlgorithm_t>;
      workspace_size =
          std::max(workspace_size, search3::GetWorkspaceSize(args3));
      filter_algo = search3::Find<T>(args3, exhaustive_search, deterministic,
                                     workspace_size, ctx);
#else
      using search3 = SearchAlgorithm<cudnnConvolutionBwdFilterAlgoPerf_t>;
      filter_algo =
          search3::Find<T>(args3, exhaustive_search, deterministic, ctx);
      workspace_size = std::max(workspace_size,
                                search3::GetWorkspaceSize(args3, filter_algo));
#endif
    }

    if (ddW && dX) {
      transformed_dx = transformed_dX.data<T>();

      args4.handle = handle;
      args4.idesc.set(transformed_dX, iwo_group);
      args4.wdesc.set(*ddW, layout, iwo_group);
      args4.odesc.set(transformed_dO_channel, iwo_group);
      args4.cdesc.set(dtype, padding_common, strides, dilations,
                      platform::AllowTF32Cudnn(), c_group);

#ifdef PADDLE_WITH_HIP
      using search4 = SearchAlgorithm<miopenConvBwdDataAlgorithm_t>;
      workspace_size =
          std::max(workspace_size, search4::GetWorkspaceSize(args4));
      data_algo = search4::Find<T>(args4, exhaustive_search, deterministic,
                                   workspace_size, ctx);
#else
      using search4 = SearchAlgorithm<cudnnConvolutionBwdDataAlgoPerf_t>;
      data_algo =
          search4::Find<T>(args4, exhaustive_search, deterministic, ctx);
      workspace_size =
          std::max(workspace_size, search4::GetWorkspaceSize(args4, data_algo));
#endif
    }

    int i_n, i_c, i_d, i_h, i_w;
    GetNCDHW(transformed_X.dims(), DataLayout::kNCHW, &i_n, &i_c, &i_d, &i_h,
             &i_w);

    int o_n, o_c, o_d, o_h, o_w;
    GetNCDHW(transformed_dO_channel.dims(), DataLayout::kNCHW, &o_n, &o_c, &o_d,
             &o_h, &o_w);

    int group_offset_in = i_c / groups * i_h * i_w * i_d;
    int group_offset_out = o_c / groups * o_h * o_w * o_d;
    int group_offset_filter = W->numel() / groups;

    ScalingParamType<T> alpha = 1.0f;
    ScalingParamType<T> beta = 0.0f;

    // NOTE(zhiqiu): inplace addto is not supportted in double grad yet.
    // ScalingParamType<T> beta = ctx.Attr<bool>("use_addto") ? 1.0f :
    // 0.0f;
    // VLOG(4) << "Conv_grad_grad: use_addto = " << ctx.Attr<bool>("use_addto");
    auto wkspace_handle = dev_ctx.cudnn_workspace_handle();

    if (ddO) {
      if (ddX) {
        ddx = transformed_ddX.data<T>();
#ifdef PADDLE_WITH_HIP
        wkspace_handle.RunFunc(
            [&](void* workspace_ptr) {
              PADDLE_ENFORCE_CUDA_SUCCESS(
                  platform::dynload::miopenConvolutionForward(
                      handle, &alpha, args1.idesc.desc(), ddx,
                      args1.wdesc.desc(), w, args1.cdesc.desc(), fwd_algo1,
                      &beta, args1.odesc.desc(), transformed_ddy_channel,
                      workspace_ptr, workspace_size));
            },
            workspace_size);
#else
        for (int i = 0; i < groups; i++) {
          wkspace_handle.RunFunc(
              [&](void* workspace_ptr) {
                PADDLE_ENFORCE_CUDA_SUCCESS(
                    platform::dynload::cudnnConvolutionForward(
                        handle, &alpha, args1.idesc.desc(),
                        ddx + i * group_offset_in, args1.wdesc.desc(),
                        w + i * group_offset_filter, args1.cdesc.desc(),
                        fwd_algo1, workspace_ptr, workspace_size, &beta,
                        args1.odesc.desc(),
                        transformed_ddy_channel + i * group_offset_out));
              },
              workspace_size);
        }
#endif
      }
      if (ddW) {
#ifdef PADDLE_WITH_HIP
        // MIOPEN ONLY support beta to be 0.0f
        Tensor conv_x_ddw(dO->type());
        conv_x_ddw.Resize(transformed_ddO_channel.dims());
        T* conv_x_ddw_data = conv_x_ddw.mutable_data<T>(ctx.GetPlace());
        wkspace_handle.RunFunc(
            [&](void* workspace_ptr) {
              PADDLE_ENFORCE_CUDA_SUCCESS(
                  platform::dynload::miopenConvolutionForward(
                      handle, &alpha, args2.idesc.desc(), x, args2.wdesc.desc(),
                      ddw, args2.cdesc.desc(), fwd_algo2, &beta,
                      args2.odesc.desc(), conv_x_ddw_data, workspace_ptr,
                      workspace_size));
            },
            workspace_size);
        PADDLE_ENFORCE_CUDA_SUCCESS(platform::dynload::miopenOpTensor(
            handle, miopenTensorOpAdd, &alpha, args2.odesc.desc(),
            transformed_ddy_channel, &alpha, args2.odesc.desc(),
            conv_x_ddw_data, &beta, args2.odesc.desc(),
            transformed_ddy_channel));
#else
        for (int i = 0; i < groups; i++) {
          wkspace_handle.RunFunc(
              [&](void* workspace_ptr) {
                PADDLE_ENFORCE_CUDA_SUCCESS(
                    platform::dynload::cudnnConvolutionForward(
                        handle, &alpha, args2.idesc.desc(),
                        x + i * group_offset_in, args2.wdesc.desc(),
                        ddw + i * group_offset_filter, args2.cdesc.desc(),
                        fwd_algo2, workspace_ptr, workspace_size, &alpha,
                        args2.odesc.desc(),
                        transformed_ddy_channel + i * group_offset_out));
              },
              workspace_size);
        }
#endif
      }
      if (channel_last) {
        TransToChannelLast<paddle::platform::CUDADeviceContext, T>(
            ctx, &transformed_ddO_channel, ddO);
      }
    }
    T* transformed_dy_channel = transformed_dO_channel.data<T>();
    if (dW && ddX) {
      ddx = transformed_ddX.data<T>();
#ifdef PADDLE_WITH_HIP
      wkspace_handle.RunFunc(
          [&](void* workspace_ptr) {
            PADDLE_ENFORCE_CUDA_SUCCESS(
                platform::dynload::miopenConvolutionBackwardWeights(
                    handle, &alpha, args3.odesc.desc(), transformed_dy_channel,
                    args3.idesc.desc(), ddx, args3.cdesc.desc(), filter_algo,
                    &beta, args3.wdesc.desc(), dw, workspace_ptr,
                    workspace_size));
          },
          workspace_size);
#else
      for (int i = 0; i < groups; i++) {
        wkspace_handle.RunFunc(
            [&](void* workspace_ptr) {
              PADDLE_ENFORCE_CUDA_SUCCESS(
                  platform::dynload::cudnnConvolutionBackwardFilter(
                      handle, &alpha, args3.idesc.desc(),
                      ddx + i * group_offset_in, args3.odesc.desc(),
                      transformed_dy_channel + i * group_offset_out,
                      args3.cdesc.desc(), filter_algo, workspace_ptr,
                      workspace_size, &beta, args3.wdesc.desc(),
                      dw + i * group_offset_filter));
            },
            workspace_size);
      }
#endif
    }

    if (dX && ddW) {
      ddw = ddW->data<T>();
#ifdef PADDLE_WITH_HIP
      wkspace_handle.RunFunc(
          [&](void* workspace_ptr) {
            PADDLE_ENFORCE_CUDA_SUCCESS(
                platform::dynload::miopenConvolutionBackwardData(
                    handle, &alpha, args4.odesc.desc(), transformed_dy_channel,
                    args4.wdesc.desc(), ddw, args4.cdesc.desc(), data_algo,
                    &beta, args4.idesc.desc(), transformed_dx, workspace_ptr,
                    workspace_size));
          },
          workspace_size);
#else
      for (int i = 0; i < groups; i++) {
        wkspace_handle.RunFunc(
            [&](void* workspace_ptr) {
              PADDLE_ENFORCE_CUDA_SUCCESS(
                  platform::dynload::cudnnConvolutionBackwardData(
                      handle, &alpha, args4.wdesc.desc(),
                      ddw + i * group_offset_filter, args4.odesc.desc(),
                      transformed_dy_channel + i * group_offset_out,
                      args4.cdesc.desc(), data_algo, workspace_ptr,
                      workspace_size, &beta, args4.idesc.desc(),
                      transformed_dx + i * group_offset_in));
            },
            workspace_size);
      }
#endif

      if (!is_sys_pad) {
        // reverse padded input
        std::vector<int> starts(X->dims().size(), 0);
        std::vector<int> axes(X->dims().size(), 0);

        for (size_t i = 0; i < X->dims().size(); ++i) {
          starts[i] = input_pad[2 * i];
          axes[i] = i;
        }
        if (X->dims().size() == 4) {
          RemovePaddingSlice<paddle::platform::CUDADeviceContext, T, 4>(
              ctx, &transformed_dX, &transformed_dX_channel, starts, axes);
        } else {
          RemovePaddingSlice<paddle::platform::CUDADeviceContext, T, 5>(
              ctx, &transformed_dX, &transformed_dX_channel, starts, axes);
        }
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

namespace plat = paddle::platform;
#ifdef PADDLE_WITH_HIP
// MIOPEN do not support double
REGISTER_OP_KERNEL(conv2d, CUDNN, plat::CUDAPlace,
                   paddle::operators::CUDNNConvOpKernel<float>,
                   paddle::operators::CUDNNConvOpKernel<plat::float16>);
REGISTER_OP_KERNEL(conv2d_grad, CUDNN, plat::CUDAPlace,
                   paddle::operators::CUDNNConvGradOpKernel<float>,
                   paddle::operators::CUDNNConvGradOpKernel<plat::float16>);
REGISTER_OP_KERNEL(
    conv2d_grad_grad, CUDNN, plat::CUDAPlace,
    paddle::operators::CUDNNConvDoubleGradOpKernel<float>,
    paddle::operators::CUDNNConvDoubleGradOpKernel<plat::float16>);
// ROCM has limit thread in depthwise_conv.cu and willl result in accuracy issue
// Use depthwise_conv2d in MIOPEN to resolve this issue
REGISTER_OP_KERNEL(depthwise_conv2d, CUDNN, plat::CUDAPlace,
                   paddle::operators::CUDNNConvOpKernel<float>,
                   paddle::operators::CUDNNConvOpKernel<plat::float16>);
REGISTER_OP_KERNEL(depthwise_conv2d_grad, CUDNN, plat::CUDAPlace,
                   paddle::operators::CUDNNConvGradOpKernel<float>,
                   paddle::operators::CUDNNConvGradOpKernel<plat::float16>);
REGISTER_OP_CUDA_KERNEL(
    depthwise_conv2d_grad_grad,
    paddle::operators::CUDNNConvDoubleGradOpKernel<float>,
    paddle::operators::CUDNNConvDoubleGradOpKernel<plat::float16>);

REGISTER_OP_KERNEL(conv3d, CUDNN, plat::CUDAPlace,
                   paddle::operators::CUDNNConvOpKernel<float>,
                   paddle::operators::CUDNNConvOpKernel<plat::float16>);
REGISTER_OP_KERNEL(conv3d_grad, CUDNN, plat::CUDAPlace,
                   paddle::operators::CUDNNConvGradOpKernel<float>);
REGISTER_OP_KERNEL(
    conv3d_grad_grad, CUDNN, plat::CUDAPlace,
    paddle::operators::CUDNNConvDoubleGradOpKernel<float>,
    paddle::operators::CUDNNConvDoubleGradOpKernel<plat::float16>);
#else
#if CUDNN_VERSION_MIN(8, 1, 0)
REGISTER_OP_KERNEL(conv2d, CUDNN, plat::CUDAPlace,
                   paddle::operators::CUDNNConvOpKernel<float>,
                   paddle::operators::CUDNNConvOpKernel<double>,
                   paddle::operators::CUDNNConvOpKernel<plat::float16>,
                   paddle::operators::CUDNNConvOpKernel<plat::bfloat16>);
REGISTER_OP_KERNEL(conv2d_grad, CUDNN, plat::CUDAPlace,
                   paddle::operators::CUDNNConvGradOpKernel<float>,
                   paddle::operators::CUDNNConvGradOpKernel<double>,
                   paddle::operators::CUDNNConvGradOpKernel<plat::float16>,
                   paddle::operators::CUDNNConvGradOpKernel<plat::bfloat16>);
REGISTER_OP_KERNEL(
    conv2d_grad_grad, CUDNN, plat::CUDAPlace,
    paddle::operators::CUDNNConvDoubleGradOpKernel<float>,
    paddle::operators::CUDNNConvDoubleGradOpKernel<double>,
    paddle::operators::CUDNNConvDoubleGradOpKernel<plat::float16>,
    paddle::operators::CUDNNConvDoubleGradOpKernel<plat::bfloat16>);

REGISTER_OP_CUDA_KERNEL(
    depthwise_conv2d_grad_grad,
    paddle::operators::CUDNNConvDoubleGradOpKernel<float>,
    paddle::operators::CUDNNConvDoubleGradOpKernel<double>,
    paddle::operators::CUDNNConvDoubleGradOpKernel<plat::float16>,
    paddle::operators::CUDNNConvDoubleGradOpKernel<plat::bfloat16>);
#else
REGISTER_OP_KERNEL(conv2d, CUDNN, plat::CUDAPlace,
                   paddle::operators::CUDNNConvOpKernel<float>,
                   paddle::operators::CUDNNConvOpKernel<double>,
                   paddle::operators::CUDNNConvOpKernel<plat::float16>);
REGISTER_OP_KERNEL(conv2d_grad, CUDNN, plat::CUDAPlace,
                   paddle::operators::CUDNNConvGradOpKernel<float>,
                   paddle::operators::CUDNNConvGradOpKernel<double>,
                   paddle::operators::CUDNNConvGradOpKernel<plat::float16>);
REGISTER_OP_KERNEL(
    conv2d_grad_grad, CUDNN, plat::CUDAPlace,
    paddle::operators::CUDNNConvDoubleGradOpKernel<float>,
    paddle::operators::CUDNNConvDoubleGradOpKernel<double>,
    paddle::operators::CUDNNConvDoubleGradOpKernel<plat::float16>);

REGISTER_OP_CUDA_KERNEL(
    depthwise_conv2d_grad_grad,
    paddle::operators::CUDNNConvDoubleGradOpKernel<float>,
    paddle::operators::CUDNNConvDoubleGradOpKernel<double>,
    paddle::operators::CUDNNConvDoubleGradOpKernel<plat::float16>);
#endif

REGISTER_OP_KERNEL(conv3d, CUDNN, plat::CUDAPlace,
                   paddle::operators::CUDNNConvOpKernel<float>,
                   paddle::operators::CUDNNConvOpKernel<double>,
                   paddle::operators::CUDNNConvOpKernel<plat::float16>);
REGISTER_OP_KERNEL(conv3d_grad, CUDNN, plat::CUDAPlace,
                   paddle::operators::CUDNNConvGradOpKernel<float>,
                   paddle::operators::CUDNNConvGradOpKernel<double>);
REGISTER_OP_KERNEL(
    conv3d_grad_grad, CUDNN, plat::CUDAPlace,
    paddle::operators::CUDNNConvDoubleGradOpKernel<float>,
    paddle::operators::CUDNNConvDoubleGradOpKernel<double>,
    paddle::operators::CUDNNConvDoubleGradOpKernel<plat::float16>);
#endif
