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

#include <array>
#include "paddle/fluid/framework/conv_search_cache.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/conv_cudnn_op_cache.h"
#include "paddle/fluid/operators/conv_op.h"
#include "paddle/fluid/operators/math/padding.h"
#ifdef PADDLE_WITH_HIP
#include "paddle/fluid/platform/miopen_helper.h"
#else
#include "paddle/fluid/platform/cudnn_helper.h"
#endif

DECLARE_int64(cudnn_exhaustive_search_times);

namespace paddle {
namespace operators {

#if PADDLE_WITH_HIP || CUDNN_VERSION >= 7100
using Tensor = framework::Tensor;
using ScopedTensorDescriptor = platform::ScopedTensorDescriptor;
using ScopedFilterDescriptor = platform::ScopedFilterDescriptor;
using ScopedConvolutionDescriptor = platform::ScopedConvolutionDescriptor;
using ScopedActivationDescriptor = platform::ScopedActivationDescriptor;
using DataLayout = platform::CudnnDataLayout;
using framework::AlgorithmsCache;
using framework::ConvSearchCache;

template <typename T>
using ScalingParamType = typename platform::CudnnDataType<T>::ScalingParamType;

template <typename T>
class CUDNNConvFusionOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto& dev_ctx = ctx.template device_context<platform::CUDADeviceContext>();
    auto* input = ctx.Input<Tensor>("Input");
    auto* filter = ctx.Input<Tensor>("Filter");
    auto* bias = ctx.Input<Tensor>("Bias");
    auto* residual = ctx.Input<Tensor>("ResidualData");
    auto* output = ctx.Output<Tensor>("Output");
    output->mutable_data<T>(ctx.GetPlace());

    std::vector<int> strides = ctx.Attr<std::vector<int>>("strides");
    std::vector<int> paddings = ctx.Attr<std::vector<int>>("paddings");
    std::vector<int> dilations = ctx.Attr<std::vector<int>>("dilations");
    const std::string activation = ctx.Attr<std::string>("activation");
    int groups = ctx.Attr<int>("groups");
    int64_t user_workspace_size =
        static_cast<size_t>(ctx.Attr<int>("workspace_size_MB"));
    bool exhaustive_search =
        FLAGS_cudnn_exhaustive_search || ctx.Attr<bool>("exhaustive_search");

    const T* filter_data = filter->data<T>();
    const T* bias_data = bias->data<T>();

    const std::string padding_algorithm =
        ctx.Attr<std::string>("padding_algorithm");

    Tensor transformed_input_channel(input->type());
    Tensor transformed_output(output->type());
    transformed_input_channel = *input;
    transformed_output = *output;
    T* output_data = transformed_output.data<T>();

    const T* residual_data = residual ? residual->data<T>() : output_data;

    // update padding and dilation
    auto in_dims = transformed_input_channel.dims();
    auto filter_dims = filter->dims();
    framework::DDim in_data_dims =
        framework::slice_ddim(in_dims, 2, in_dims.size());

    framework::DDim filter_data_dims =
        framework::slice_ddim(filter_dims, 2, filter_dims.size());
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
      new_input_shape_vec[1] = transformed_input_channel.dims()[1];

      std::vector<int> input_pad(transformed_input_channel.dims().size() * 2,
                                 0);
      for (size_t i = 0; i < data_dim; ++i) {
        padding_diff[i] = std::abs(paddings[2 * i] - paddings[2 * i + 1]);
        padding_common[i] = std::min(paddings[2 * i], paddings[2 * i + 1]);
        new_input_shape_vec[i + 2] =
            transformed_input_channel.dims()[i + 2] + padding_diff[i];
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
          PADDLE_THROW(platform::errors::PermissionDenied(
              "Operator Conv2DFusion expects Input to be a 4-D or 5-D Tensor. "
              "But recieved the actual dimension = %d, shape = [%s].",
              rank, transformed_input_channel.dims()));
      }

    } else {
      transformed_input = transformed_input_channel;
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

    // ------------------- cudnn descriptors ---------------------
    ScopedTensorDescriptor input_desc;
    ScopedTensorDescriptor output_desc;
    ScopedFilterDescriptor filter_desc;
    ScopedTensorDescriptor bias_desc;
    ScopedConvolutionDescriptor conv_desc;
    ScopedActivationDescriptor act_desc;
    DataLayout layout = DataLayout::kNCHW;
    if (input->dims().size() == 5) {
      layout = DataLayout::kNCDHW;
    }
#ifdef PADDLE_WITH_HIP
    miopenConvolutionDescriptor_t cudnn_conv_desc =
        conv_desc.descriptor<T>(padding_common, strides, dilations);
    PADDLE_ENFORCE_CUDA_SUCCESS(
        platform::dynload::miopenSetConvolutionGroupCount(cudnn_conv_desc,
                                                          groups));
    // Now only support NCHW
    std::vector<int> bias_dim = {
        1, static_cast<int>(transformed_output.dims()[1]), 1, 1};
    miopenTensorDescriptor_t cudnn_input_desc = input_desc.descriptor<T>(
        layout, framework::vectorize<int>(transformed_input.dims()));
    miopenTensorDescriptor_t cudnn_output_desc = output_desc.descriptor<T>(
        layout, framework::vectorize<int>(transformed_output.dims()));
    miopenTensorDescriptor_t cudnn_filter_desc = filter_desc.descriptor<T>(
        layout, framework::vectorize<int>(filter->dims()));
    miopenTensorDescriptor_t cudnn_bias_desc =
        bias_desc.descriptor<T>(layout, bias_dim);
    miopenActivationDescriptor_t cudnn_act_desc =
        act_desc.descriptor<T>(activation);

    miopenConvFwdAlgorithm_t algo;
    auto handle = dev_ctx.cudnn_handle();
    auto workspace_handle = dev_ctx.cudnn_workspace_handle();

    auto x_dims = framework::vectorize(transformed_input.dims());
    auto f_dims = framework::vectorize(filter->dims());

    size_t workspace_size = 0;
    PADDLE_ENFORCE_CUDA_SUCCESS(
        platform::dynload::miopenConvolutionForwardGetWorkSpaceSize(
            handle, cudnn_filter_desc, cudnn_input_desc, cudnn_conv_desc,
            cudnn_output_desc, &workspace_size));
    int find_count;
    miopenConvAlgoPerf_t find_result;
    auto cudnn_find_func = [&](void* cudnn_workspace_ptr) {
      PADDLE_ENFORCE_CUDA_SUCCESS(
          platform::dynload::miopenFindConvolutionForwardAlgorithm(
              handle, cudnn_input_desc, input_data, cudnn_filter_desc,
              filter_data, cudnn_conv_desc, cudnn_output_desc, output_data,
              kNUM_CUDNN_FWD_ALGS, &find_count, &find_result,
              cudnn_workspace_ptr, workspace_size, false));
    };
    workspace_handle.RunFuncSync(cudnn_find_func, workspace_size);
    algo = find_result.fwd_algo;
    VLOG(3) << "cuDNN forward algo " << algo;

    {
      ScalingParamType<T> alpha = 1.0f, beta = 0.0f;
      auto cudnn_func = [&](void* cudnn_workspace) {
        PADDLE_ENFORCE_CUDA_SUCCESS(platform::dynload::miopenConvolutionForward(
            handle, &alpha, cudnn_input_desc, input_data, cudnn_filter_desc,
            filter_data, cudnn_conv_desc, algo, &beta, cudnn_output_desc,
            output_data, cudnn_workspace, workspace_size));
      };
      workspace_handle.RunFunc(cudnn_func, workspace_size);
      PADDLE_ENFORCE_CUDA_SUCCESS(
          platform::dynload::miopenConvolutionForwardBias(
              handle, &alpha, cudnn_bias_desc, bias_data, &beta,
              cudnn_output_desc, output_data));
      if (activation != "identity") {
        PADDLE_ENFORCE_CUDA_SUCCESS(platform::dynload::miopenActivationForward(
            handle, cudnn_act_desc, &alpha, cudnn_output_desc, output_data,
            &beta, cudnn_output_desc, output_data));
      }
      if (residual) {
        PADDLE_ENFORCE_CUDA_SUCCESS(platform::dynload::miopenOpTensor(
            handle, miopenTensorOpAdd, &alpha, cudnn_output_desc, output_data,
            &alpha, cudnn_output_desc, residual_data, &beta, cudnn_output_desc,
            output_data));
      }
    }
#else  // PADDLE_WITH_HIP
    cudnnConvolutionDescriptor_t cudnn_conv_desc =
        conv_desc.descriptor<T>(padding_common, strides, dilations);
    PADDLE_ENFORCE_CUDA_SUCCESS(
        platform::dynload::cudnnSetConvolutionGroupCount(cudnn_conv_desc,
                                                         groups));

    cudnnTensorDescriptor_t cudnn_input_desc = input_desc.descriptor<T>(
        layout, framework::vectorize<int>(transformed_input.dims()));
    cudnnTensorDescriptor_t cudnn_output_desc = output_desc.descriptor<T>(
        layout, framework::vectorize<int>(transformed_output.dims()));
    cudnnFilterDescriptor_t cudnn_filter_desc = filter_desc.descriptor<T>(
        layout, framework::vectorize<int>(filter->dims()));
    // Now only support NCHW
    std::vector<int> bias_dim = {
        1, static_cast<int>(transformed_output.dims()[1]), 1, 1};
    cudnnTensorDescriptor_t cudnn_bias_desc =
        bias_desc.descriptor<T>(layout, bias_dim);
    cudnnActivationDescriptor_t cudnn_act_desc =
        act_desc.descriptor<T>(activation);

    // ------------------- cudnn conv workspace ---------------------
    size_t workspace_size_in_bytes;  // final workspace to allocate.
    size_t workspace_size_limit = 0;
    if (FLAGS_conv_workspace_size_limit > 0 || user_workspace_size > 0) {
      int64_t max_user_size =
          std::min(static_cast<int64_t>(FLAGS_conv_workspace_size_limit),
                   user_workspace_size);
      workspace_size_limit = max_user_size * 1024 * 1024;
    }

    // ------------------- cudnn conv algorithm ---------------------
    cudnnConvolutionFwdAlgo_t algo;
    auto handle = dev_ctx.cudnn_handle();
    auto workspace_handle = dev_ctx.cudnn_workspace_handle();

    PADDLE_ENFORCE_CUDA_SUCCESS(platform::dynload::cudnnSetConvolutionMathType(
        cudnn_conv_desc, CUDNN_DEFAULT_MATH));
#if CUDA_VERSION >= 11000 && CUDNN_VERSION >= 8000
    if (!platform::allow_tf32_cudnn) {
      PADDLE_ENFORCE_CUDA_SUCCESS(
          platform::dynload::cudnnSetConvolutionMathType(cudnn_conv_desc,
                                                         CUDNN_FMA_MATH));
    }
#endif  // CUDA_VERSION >= 11000 && CUDNN_VERSION >= 8000

    auto x_dims = framework::vectorize(transformed_input.dims());
    auto f_dims = framework::vectorize(filter->dims());
    if (!exhaustive_search) {
#if CUDNN_VERSION >= 8000
      int perf_count;
      int best_algo_idx = 0;
      size_t tmp_size = 0;
      std::unique_ptr<cudnnConvolutionFwdAlgoPerf_t[]> perf_results(
          new cudnnConvolutionFwdAlgoPerf_t[kNUM_CUDNN_FWD_ALGS]);
      PADDLE_ENFORCE_CUDA_SUCCESS(
          platform::dynload::cudnnGetConvolutionForwardAlgorithm_v7(
              handle, cudnn_input_desc, cudnn_filter_desc, cudnn_conv_desc,
              cudnn_output_desc, kNUM_CUDNN_FWD_ALGS, &perf_count,
              perf_results.get()));
      algo = (perf_results.get())[best_algo_idx].algo;
      PADDLE_ENFORCE_CUDA_SUCCESS(
          platform::dynload::cudnnGetConvolutionForwardWorkspaceSize(
              handle, cudnn_input_desc, cudnn_filter_desc, cudnn_conv_desc,
              cudnn_output_desc, algo, &workspace_size_in_bytes));
      if (workspace_size_in_bytes > workspace_size_limit)
        workspace_size_limit = workspace_size_in_bytes;
#else
      PADDLE_ENFORCE_CUDA_SUCCESS(
          platform::dynload::cudnnGetConvolutionForwardAlgorithm(
              handle, cudnn_input_desc, cudnn_filter_desc, cudnn_conv_desc,
              cudnn_output_desc, CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT,
              workspace_size_limit, &algo));
      VLOG(3) << "cuDNN forward algo " << algo;
#endif
    } else {
      std::function<cudnnConvolutionFwdAlgo_t()> search_func =
          [&]() -> cudnnConvolutionFwdAlgo_t {
        int returned_algo_count;
        std::array<cudnnConvolutionFwdAlgoPerf_t, kNUM_CUDNN_FWD_ALGS>
            fwd_perf_stat;
        auto cudnn_find_func = [&](void* cudnn_workspace) {
          PADDLE_ENFORCE_CUDA_SUCCESS(
              platform::dynload::cudnnFindConvolutionForwardAlgorithmEx(
                  handle, cudnn_input_desc, input_data, cudnn_filter_desc,
                  filter_data, cudnn_conv_desc, cudnn_output_desc, output_data,
                  kNUM_CUDNN_FWD_ALGS, &returned_algo_count,
                  fwd_perf_stat.data(), cudnn_workspace, workspace_size_limit));
        };
        workspace_handle.RunFuncSync(cudnn_find_func, workspace_size_limit);
        VLOG(3) << "Perf result: (algo: stat, time, memory)";
        for (int i = 0; i < returned_algo_count; ++i) {
          const auto& stat = fwd_perf_stat[i];
          VLOG(3) << stat.algo << ": " << stat.status << " " << stat.time << " "
                  << stat.memory;
        }
        return fwd_perf_stat[0].algo;
      };
      AlgorithmsCache<cudnnConvolutionFwdAlgo_t>& algo_cache =
          *(framework::ConvSearchCache::Instance().GetConvFusion());
      int search_times = ctx.Attr<int>("search_times");
      search_times = std::max(
          static_cast<int>(FLAGS_cudnn_exhaustive_search_times), search_times);
      // TODO(dangqingqing): Unify this if-else.
      if (search_times > 0) {
        // The searched algo will be cached by `search_times` times for
        // different input dimension. For other dimensions, select the algo
        // of closest area.
        algo = algo_cache.GetAlgorithm(x_dims[2] * x_dims[3], search_times, 0,
                                       search_func);
      } else {
        auto dtype = platform::CudnnDataType<T>::type;
        algo = algo_cache.GetAlgorithm(x_dims, f_dims, strides, paddings,
                                       dilations, 0, dtype, search_func);
      }
      VLOG(3) << "choose algo " << algo;
    }

    PADDLE_ENFORCE_CUDA_SUCCESS(
        platform::dynload::cudnnGetConvolutionForwardWorkspaceSize(
            handle, cudnn_input_desc, cudnn_filter_desc, cudnn_conv_desc,
            cudnn_output_desc, algo, &workspace_size_in_bytes));
    PADDLE_ENFORCE_LE(
        workspace_size_in_bytes, workspace_size_limit,
        platform::errors::InvalidArgument(
            "The actual workspace size to be allocated for cuDNN is expected "
            "to be less than the limit. But recieved: the actual workspace "
            "size = %d, limit = %d.",
            workspace_size_in_bytes, workspace_size_limit));

    if ((activation == "identity") && (!residual)) {
      // Only the CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM algo is
      // enabled with CUDNN_ACTIVATION_IDENTITY in cuDNN lib.
      // But test in some case, the speed is slower, change to use
      // cudnnConvolutionForward and cudnnAddTensor
      // ------------- cudnn conv forward and bias add ---------------------
      ScalingParamType<T> alpha = 1.0f, beta = 0.0f;
      auto cudnn_func = [&](void* cudnn_workspace) {
        PADDLE_ENFORCE_CUDA_SUCCESS(platform::dynload::cudnnConvolutionForward(
            handle, &alpha, cudnn_input_desc, input_data, cudnn_filter_desc,
            filter_data, cudnn_conv_desc, algo, cudnn_workspace,
            workspace_size_in_bytes, &beta, cudnn_output_desc, output_data));
      };
      workspace_handle.RunFunc(cudnn_func, workspace_size_in_bytes);
      PADDLE_ENFORCE_CUDA_SUCCESS(platform::dynload::cudnnAddTensor(
          handle, &alpha, cudnn_bias_desc, bias_data, &alpha, cudnn_output_desc,
          output_data));
    } else {
      if (activation == "identity") {
        algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;
      }
      // ------------------- cudnn conv+bias+act forward --------------------
      ScalingParamType<T> alpha1 = 1.0f;
      ScalingParamType<T> alpha2 = residual ? 1.0f : 0.0f;
      auto cudnn_func = [&](void* cudnn_workspace) {
        PADDLE_ENFORCE_CUDA_SUCCESS(
            platform::dynload::cudnnConvolutionBiasActivationForward(
                handle, &alpha1, cudnn_input_desc, input_data,
                cudnn_filter_desc, filter_data, cudnn_conv_desc, algo,
                cudnn_workspace, workspace_size_in_bytes, &alpha2,
                cudnn_output_desc, residual_data, cudnn_bias_desc, bias_data,
                cudnn_act_desc, cudnn_output_desc, output_data));
      };
      workspace_handle.RunFunc(cudnn_func, workspace_size_in_bytes);
    }
#endif
    std::vector<int> channels = ctx.Attr<std::vector<int>>("split_channels");
    if (channels.size()) {
      auto outs = ctx.MultiOutput<framework::Tensor>("Outputs");
      if (x_dims[0] == 1) {
        // share data with Output
        framework::Tensor t;
        t.ShareDataWith(*output);
        auto y_dims = output->dims();
        t.Resize({y_dims[1], y_dims[2], y_dims[3]});
        int s = 0;
        for (size_t i = 0; i < channels.size(); ++i) {
          int e = s + channels[i];
          outs[i]->ShareDataWith(t.Slice(s, e));
          outs[i]->Resize({x_dims[0], channels[i], y_dims[2], y_dims[3]});
          s = e;
        }
      } else {
        // TODO(qingiqng): do copy when batch size large than 1
        PADDLE_THROW(platform::errors::Unimplemented(
            "Input with batch size greater than 1 is unsupported. The recieved "
            "batch size is %d, Input's shape is [%s].",
            x_dims[0], framework::make_ddim(x_dims)));
      }
    }
  }
};
#endif

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
#if CUDNN_VERSION >= 7100
REGISTER_OP_CUDA_KERNEL(conv2d_fusion, ops::CUDNNConvFusionOpKernel<float>,
                        ops::CUDNNConvFusionOpKernel<double>);
#endif
#ifdef PADDLE_WITH_HIP
REGISTER_OP_CUDA_KERNEL(conv2d_fusion, ops::CUDNNConvFusionOpKernel<float>);
#endif
