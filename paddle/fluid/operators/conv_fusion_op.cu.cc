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

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/conv_cudnn_op_cache.h"
#include "paddle/fluid/platform/cudnn_helper.h"

DECLARE_int64(cudnn_exhaustive_search_times);

namespace paddle {
namespace operators {

#if CUDNN_VERSION >= 7100
using Tensor = framework::Tensor;
using ScopedTensorDescriptor = platform::ScopedTensorDescriptor;
using ScopedFilterDescriptor = platform::ScopedFilterDescriptor;
using ScopedConvolutionDescriptor = platform::ScopedConvolutionDescriptor;
using ScopedActivationDescriptor = platform::ScopedActivationDescriptor;
using DataLayout = platform::DataLayout;
using framework::AlgorithmsCache;

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
    PADDLE_ENFORCE(bias, "The bias should not be null.");
    auto* residual = ctx.Input<Tensor>("ResidualData");
    auto* output = ctx.Output<Tensor>("Output");

    std::vector<int> strides = ctx.Attr<std::vector<int>>("strides");
    std::vector<int> paddings = ctx.Attr<std::vector<int>>("paddings");
    std::vector<int> dilations = ctx.Attr<std::vector<int>>("dilations");
    const std::string activation = ctx.Attr<std::string>("activation");
    int groups = ctx.Attr<int>("groups");
    int64_t user_workspace_size =
        static_cast<size_t>(ctx.Attr<int>("workspace_size_MB"));
    bool exhaustive_search =
        FLAGS_cudnn_exhaustive_search || ctx.Attr<bool>("exhaustive_search");

    const T* input_data = input->data<T>();
    const T* filter_data = filter->data<T>();
    const T* bias_data = bias->data<T>();
    T* output_data = output->mutable_data<T>(ctx.GetPlace());
    const T* residual_data = residual ? residual->data<T>() : output_data;

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

    cudnnConvolutionDescriptor_t cudnn_conv_desc =
        conv_desc.descriptor<T>(paddings, strides, dilations);
    CUDNN_ENFORCE(platform::dynload::cudnnSetConvolutionGroupCount(
        cudnn_conv_desc, groups));

    cudnnTensorDescriptor_t cudnn_input_desc = input_desc.descriptor<T>(
        layout, framework::vectorize2int(input->dims()));
    cudnnTensorDescriptor_t cudnn_output_desc = output_desc.descriptor<T>(
        layout, framework::vectorize2int(output->dims()));
    cudnnFilterDescriptor_t cudnn_filter_desc = filter_desc.descriptor<T>(
        layout, framework::vectorize2int(filter->dims()));
    // Now only support NCHW
    std::vector<int> bias_dim = {1, static_cast<int>(output->dims()[1]), 1, 1};
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

    CUDNN_ENFORCE(platform::dynload::cudnnSetConvolutionMathType(
        cudnn_conv_desc, CUDNN_DEFAULT_MATH));

    auto x_dims = framework::vectorize(input->dims());
    auto f_dims = framework::vectorize(filter->dims());
    if (!exhaustive_search) {
      CUDNN_ENFORCE(platform::dynload::cudnnGetConvolutionForwardAlgorithm(
          handle, cudnn_input_desc, cudnn_filter_desc, cudnn_conv_desc,
          cudnn_output_desc, CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT,
          workspace_size_limit, &algo));
      VLOG(3) << "cuDNN forward algo " << algo;
    } else {
      auto search_func = [&]() {
        int returned_algo_count;
        std::array<cudnnConvolutionFwdAlgoPerf_t, kNUM_CUDNN_FWD_ALGS>
            fwd_perf_stat;
        auto cudnn_find_func = [&](void* cudnn_workspace) {
          CUDNN_ENFORCE(
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
          ctx.GetKernelConfig<AlgorithmsCache<cudnnConvolutionFwdAlgo_t>>(0);
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
        algo = algo_cache.GetAlgorithm(x_dims, f_dims, strides, paddings,
                                       dilations, 0, search_func);
      }
      VLOG(3) << "choose algo " << algo;
    }

    CUDNN_ENFORCE(platform::dynload::cudnnGetConvolutionForwardWorkspaceSize(
        handle, cudnn_input_desc, cudnn_filter_desc, cudnn_conv_desc,
        cudnn_output_desc, algo, &workspace_size_in_bytes));
    PADDLE_ENFORCE_LE(workspace_size_in_bytes, workspace_size_limit,
                      "workspace_size to be allocated exceeds the limit");

    if ((activation == "identity") && (!residual)) {
      // Only the CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM algo is
      // enabled with CUDNN_ACTIVATION_IDENTITY in cuDNN lib.
      // But test in some case, the speed is slower, change to use
      // cudnnConvolutionForward and cudnnAddTensor
      // ------------- cudnn conv forward and bias add ---------------------
      ScalingParamType<T> alpha = 1.0f, beta = 0.0f;
      auto cudnn_func = [&](void* cudnn_workspace) {
        CUDNN_ENFORCE(platform::dynload::cudnnConvolutionForward(
            handle, &alpha, cudnn_input_desc, input_data, cudnn_filter_desc,
            filter_data, cudnn_conv_desc, algo, cudnn_workspace,
            workspace_size_in_bytes, &beta, cudnn_output_desc, output_data));
      };
      workspace_handle.RunFunc(cudnn_func, workspace_size_in_bytes);
      CUDNN_ENFORCE(platform::dynload::cudnnAddTensor(
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
        CUDNN_ENFORCE(platform::dynload::cudnnConvolutionBiasActivationForward(
            handle, &alpha1, cudnn_input_desc, input_data, cudnn_filter_desc,
            filter_data, cudnn_conv_desc, algo, cudnn_workspace,
            workspace_size_in_bytes, &alpha2, cudnn_output_desc, residual_data,
            cudnn_bias_desc, bias_data, cudnn_act_desc, cudnn_output_desc,
            output_data));
      };
      workspace_handle.RunFunc(cudnn_func, workspace_size_in_bytes);
    }
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
        PADDLE_THROW("Batch size greater than 1 is Unsupported");
      }
    }
  }
};
#endif

}  // namespace operators
}  // namespace paddle

#if CUDNN_VERSION >= 7100
namespace ops = paddle::operators;
REGISTER_OP_CUDA_KERNEL(conv2d_fusion, ops::CUDNNConvFusionOpKernel<float>,
                        ops::CUDNNConvFusionOpKernel<double>);
#endif
