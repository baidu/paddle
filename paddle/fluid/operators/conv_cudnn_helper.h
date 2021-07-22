/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.

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

#include <algorithm>
#include <array>
#include <memory>
#include <string>
#include <vector>

#include "paddle/fluid/framework/conv_search_cache.h"
#include "paddle/fluid/framework/operator_kernel_configs.h"
#include "paddle/fluid/operators/conv_cudnn_op_cache.h"
#include "paddle/fluid/operators/eigen/eigen_function.h"
#include "paddle/fluid/platform/cudnn_desc.h"
namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
using DataLayout = platform::CudnnDataLayout;
template <typename T>
using ScalingParamType = typename platform::CudnnDataType<T>::ScalingParamType;
using framework::AlgorithmsCache;
static inline void GetNCDHW(const framework::DDim& dims,
                            const DataLayout& layout, int* N, int* C, int* D,
                            int* H, int* W) {
  *N = dims[0];
  *C = layout == DataLayout::kNCHW ? dims[1] : dims[dims.size() - 1];
  int i = layout == DataLayout::kNCHW ? 0 : 1;
  if (dims.size() == 5) {
    *D = dims[2 - i];
    *H = dims[3 - i];
    *W = dims[4 - i];
  } else {
    *D = 1;
    *H = dims[2 - i];
    *W = dims[3 - i];
  }
}

template <typename DeviceContext, typename T, size_t D>
static void RemovePaddingSlice(const framework::ExecutionContext& context,
                               const Tensor* input, Tensor* out,
                               const std::vector<int>& starts,
                               const std::vector<int>& axes) {
  auto& place =
      *context.template device_context<DeviceContext>().eigen_device();
  auto in_dims = input->dims();
  auto new_out_dims = out->dims();
  auto offsets = Eigen::DSizes<Eigen::DenseIndex, D>();
  auto extents = Eigen::DSizes<Eigen::DenseIndex, D>();
  for (size_t i = 0; i < D; ++i) {
    offsets[i] = 0;
    extents[i] = new_out_dims[i];
  }

  int start;
  for (size_t i = 0; i < axes.size(); ++i) {
    start = starts[i];
    if (start < 0) {
      start = (start + in_dims[axes[i]]);
    }
    start = std::max(start, 0);
    offsets[axes[i]] = start;
  }
  auto in_t =
      framework::EigenTensor<T, D, Eigen::RowMajor, Eigen::DenseIndex>::From(
          *input);

  auto out_t =
      framework::EigenTensor<T, D, Eigen::RowMajor, Eigen::DenseIndex>::From(
          *out, new_out_dims);
  EigenSlice<std::decay_t<decltype(place)>, T, D>::Eval(place, out_t, in_t,
                                                        offsets, extents);
}

template <typename T>
std::ostream& operator<<(std::ostream& out, const std::vector<T>& v) {
  out << "[";
  for (auto const& tmp : v) out << tmp << ",";
  out << "]";
  return out;
}

inline int MaxBwdFilterAlgos(cudnnHandle_t cudnn_handle) {
  int max_algos = 0;
#if CUDNN_VERSION_MIN(7, 0, 1)
  PADDLE_ENFORCE_CUDA_SUCCESS(
      platform::dynload::cudnnGetConvolutionBackwardFilterAlgorithmMaxCount(
          cudnn_handle, &max_algos));
#endif
  return max_algos;
}

template <typename PerfType, typename AlgoType>
void ChooseAlgoByWorkspace(PerfType* perf_results, size_t perf_num,
                           size_t workspace_byte, AlgoType* algo) {
  for (size_t i = 0; i < perf_num; ++i) {
    auto result = perf_results[i];
    if (result.status == CUDNN_STATUS_SUCCESS &&
        result.memory < workspace_byte) {
      *algo = result.algo;
      VLOG(3) << "    algo: " << result.algo << ", time: " << result.time
              << " ms, wksp = " << result.memory
              << ", status = " << result.status;
      return;
    }
  }
  VLOG(3) << "Can not find alog that requires memory < "
          << static_cast<double>(workspace_byte) / (1 << 20) << " MB";
}

template <typename PerfType, typename AlgoType>
void ChooseAlgo(const std::vector<PerfType>& perf_results,
                size_t workspace_byte, AlgoType* algo) {
  VLOG(3) << "=========BwdFilterAlgo Perf result=========";
  for (const auto& result : perf_results) {
    auto math_type_str = "False";
    if (result.mathType == CUDNN_TENSOR_OP_MATH) {
      math_type_str = "True";
    }
    VLOG(3) << "    algo: " << result.algo << ", TensorCore: " << math_type_str
            << ", time: " << result.time << " ms"
            << ", wksp = " << result.memory << ", status = " << result.status;
  }

  for (size_t i = 0; i != perf_results.size(); ++i) {
    const auto& result = perf_results[i];
    if (result.status == CUDNN_STATUS_SUCCESS &&
        (result.memory <= workspace_byte)) {
      if ((result.mathType == CUDNN_TENSOR_OP_MATH) &&
          (i != perf_results.size() - 1)) {
        const auto& next_result = perf_results[i + 1];
        if (next_result.status == CUDNN_STATUS_SUCCESS &&
            next_result.algo == result.algo &&
            next_result.memory == result.memory &&
            next_result.mathType != CUDNN_TENSOR_OP_MATH &&
            next_result.time < 1.01 * result.time) {
          // Skip over this result- it's not really a Tensor Core algo.
          // Because it is only 1% performance difference.
          // Prefer to choose the next equivalent non-Tensor Core algo.
          continue;
        }
      }
      *algo = result.algo;
      auto math_type_str = "0";
      if (result.mathType == CUDNN_TENSOR_OP_MATH) {
        math_type_str = "1";
      }
      VLOG(3) << "    choose algo: " << result.algo << ", TC: " << math_type_str
              << ", time: " << result.time << " ms"
              << ", wksp = " << result.memory << ", status = " << result.status;
      return;
    }
  }
}

using framework::ConvSearchCache;

struct ConvArgs {
  cudnnHandle_t handle;
  platform::TensorDescriptor idesc, odesc;
  platform::FilterDescriptor wdesc;
  platform::ConvolutionDescriptor cdesc;
  const framework::Tensor *x, *w, *o;
  cudnnDataType_t cudnn_dtype;

  // strides
  std::vector<int> s;
  // paddings
  std::vector<int> p;
  // dilations
  std::vector<int> d;

  ConvArgs(const framework::Tensor* x, const framework::Tensor* w,
           const framework::Tensor* o, const std::vector<int> s,
           const std::vector<int> p, const std::vector<int> d,
           cudnnDataType_t dtype)
      : x(x), w(w), o(o), s(s), p(p), d(d), cudnn_dtype(dtype) {}
};

template <typename perf_t>
struct SearchAlgorithm {};

template <>
struct SearchAlgorithm<cudnnConvolutionFwdAlgoPerf_t> {
  using perf_t = cudnnConvolutionFwdAlgoPerf_t;
  using algo_t = cudnnConvolutionFwdAlgo_t;

  template <typename T>
  static algo_t Find(const ConvArgs& args, bool exhaustive_search,
                     bool deterministic,
                     const framework::ExecutionContext& ctx) {
    auto dtype = platform::CudnnDataType<T>::type;
    bool has_got_workspace_size = true;
    size_t workspace_size_limit = FLAGS_conv_workspace_size_limit * 1024 * 1024;
    size_t workspace_size = 0;
    algo_t algo;

#if CUDA_VERSION >= 9000 && CUDNN_VERSION_MIN(7, 0, 1)
    auto& dev_ctx = ctx.template device_context<platform::CUDADeviceContext>();
    if (dev_ctx.GetComputeCapability() >= 70 && dtype == CUDNN_DATA_HALF) {
      PADDLE_ENFORCE_CUDA_SUCCESS(
          platform::dynload::cudnnSetConvolutionMathType(args.cdesc.desc(),
                                                         CUDNN_TENSOR_OP_MATH));
      VLOG(5) << "use cudnn_tensor_op_math";
#if CUDA_VERSION >= 11000
#if CUDNN_VERSION_MIN(8, 1, 0)
    } else if (dev_ctx.GetComputeCapability() >= 80 &&
               dtype == CUDNN_DATA_BFLOAT16) {
      PADDLE_ENFORCE_CUDA_SUCCESS(
          platform::dynload::cudnnSetConvolutionMathType(args.cdesc.desc(),
                                                         CUDNN_TENSOR_OP_MATH));
      VLOG(5) << "use cudnn_tensor_op_math";
#endif  // CUDNN_VERSION >= 8100
    } else if (dtype == CUDNN_DATA_FLOAT && !args.cdesc.allow_tf32_) {
      PADDLE_ENFORCE_CUDA_SUCCESS(
          platform::dynload::cudnnSetConvolutionMathType(args.cdesc.desc(),
                                                         CUDNN_FMA_MATH));
      VLOG(5) << "use cudnn_fma_math";
#endif  // CUDA_VERSION >= 11000
    } else {
      PADDLE_ENFORCE_CUDA_SUCCESS(
          platform::dynload::cudnnSetConvolutionMathType(args.cdesc.desc(),
                                                         CUDNN_DEFAULT_MATH));
      VLOG(5) << "use cudnn_default_math";
    }
#endif

    if (!exhaustive_search && !deterministic) {
#if CUDNN_VERSION >= 7001
      int perf_count;
      int best_algo_idx = 0;
      std::unique_ptr<perf_t[]> perf_results(new perf_t[kNUM_CUDNN_FWD_ALGS]);
      PADDLE_ENFORCE_CUDA_SUCCESS(
          platform::dynload::cudnnGetConvolutionForwardAlgorithm_v7(
              args.handle, args.idesc.desc(), args.wdesc.desc(),
              args.cdesc.desc(), args.odesc.desc(), kNUM_CUDNN_FWD_ALGS,
              &perf_count, perf_results.get()));
      algo = (perf_results.get())[best_algo_idx].algo;
      workspace_size = GetWorkspaceSize(args, algo);

      if (workspace_size > workspace_size_limit) {
#if CUDNN_VERSION >= 8000
        // cudnnGetConvolutionForwardAlgorithm is removed in CUDNN-8
        ChooseAlgoByWorkspace<perf_t, algo_t>(perf_results.get(),
                                              kNUM_CUDNN_FWD_ALGS,
                                              workspace_size_limit, &algo);
#else
        VLOG(1) << "Fallback to non-v7 method to find conv algorithm becasue "
                   "the workspace size request("
                << workspace_size << ") exceeds the limit("
                << workspace_size_limit << ")";
        PADDLE_ENFORCE_CUDA_SUCCESS(
            platform::dynload::cudnnGetConvolutionForwardAlgorithm(
                args.handle, args.idesc.desc(), args.wdesc.desc(),
                args.cdesc.desc(), args.odesc.desc(),
                CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT,
                workspace_size_limit, &algo));
#endif
      }
#else
      PADDLE_ENFORCE_CUDA_SUCCESS(
          platform::dynload::cudnnGetConvolutionForwardAlgorithm(
              args.handle, args.idesc.desc(), args.wdesc.desc(),
              args.cdesc.desc(), args.odesc.desc(),
              CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT,
              workspace_size_limit, &algo));
#endif
      VLOG(3) << "choose algo " << algo;
    } else if (deterministic) {
      algo = static_cast<cudnnConvolutionFwdAlgo_t>(1);
    } else {
      auto& dev_ctx =
          ctx.template device_context<platform::CUDADeviceContext>();
      auto workspace_handle = dev_ctx.cudnn_workspace_handle();

      auto& temp = ctx.cuda_device_context();
      AlgorithmsCache<algo_t>& algo_cache =
          *(framework::ConvSearchCache::Instance().GetForward());

      auto x_dims = framework::vectorize(args.x->dims());
      auto w_dims = framework::vectorize(args.w->dims());

      VLOG(10) << "cudnnConvolutionFwdAlgoPerf_t:"
               << ", x_dims:" << x_dims << ", w_dims:" << w_dims << ", args.s"
               << args.s << ", args.p" << args.p << ", args.d" << args.d;

      algo = algo_cache.GetAlgorithm(
          x_dims, w_dims, args.s, args.p, args.d, 0,
          static_cast<int64_t>(args.cudnn_dtype), [&]() {
            int returned_algo_count;
            std::array<perf_t, kNUM_CUDNN_FWD_ALGS> perf_stat;

            auto cudnn_find_func = [&](void* cudnn_workspace_ptr) {
              PADDLE_ENFORCE_CUDA_SUCCESS(
                  platform::dynload::cudnnFindConvolutionForwardAlgorithmEx(
                      args.handle, args.idesc.desc(), args.x->data<T>(),
                      args.wdesc.desc(), args.w->data<T>(), args.cdesc.desc(),
                      args.odesc.desc(), const_cast<T*>(args.o->data<T>()),
                      kNUM_CUDNN_FWD_ALGS, &returned_algo_count,
                      perf_stat.data(), cudnn_workspace_ptr,
                      workspace_size_limit));
            };
            workspace_handle.RunFuncSync(cudnn_find_func, workspace_size_limit);

            VLOG(3) << "FwdAlgo Perf result: (algo: stat, time, memory)";
            for (int i = 0; i < returned_algo_count; ++i) {
              const auto& stat = perf_stat[i];
              VLOG(3) << stat.algo << ": " << stat.status << " " << stat.time
                      << " " << stat.memory;
            }
            return perf_stat[0].algo;
          });
    }
    VLOG(3) << "choose algo " << algo;
    return algo;
  }

  static size_t GetWorkspaceSize(const ConvArgs& args, algo_t algo) {
    size_t workspace_size = 0;
    PADDLE_ENFORCE_CUDA_SUCCESS(
        platform::dynload::cudnnGetConvolutionForwardWorkspaceSize(
            args.handle, args.idesc.desc(), args.wdesc.desc(),
            args.cdesc.desc(), args.odesc.desc(), algo, &workspace_size));
    return workspace_size;
  }
};

template <>
struct SearchAlgorithm<cudnnConvolutionBwdDataAlgoPerf_t> {
  using perf_t = cudnnConvolutionBwdDataAlgoPerf_t;
  using algo_t = cudnnConvolutionBwdDataAlgo_t;

  template <typename T>
  static algo_t Find(const ConvArgs& args, bool exhaustive_search,
                     bool deterministic,
                     const framework::ExecutionContext& ctx) {
    auto dtype = platform::CudnnDataType<T>::type;
    size_t workspace_size_limit = FLAGS_conv_workspace_size_limit * 1024 * 1024;
    size_t workspace_size = 0;
    bool has_got_workspace_size = true;
    algo_t algo;
#if CUDA_VERSION >= 9000 && CUDNN_VERSION_MIN(7, 0, 1)
    auto& dev_ctx = ctx.template device_context<platform::CUDADeviceContext>();
    PADDLE_ENFORCE_CUDA_SUCCESS(platform::dynload::cudnnSetConvolutionMathType(
        args.cdesc.desc(), CUDNN_DEFAULT_MATH));
    VLOG(5) << "NOT use cudnn_tensor_op_math";
    if (dev_ctx.GetComputeCapability() >= 70 && dtype == CUDNN_DATA_HALF) {
      PADDLE_ENFORCE_CUDA_SUCCESS(
          platform::dynload::cudnnSetConvolutionMathType(args.cdesc.desc(),
                                                         CUDNN_TENSOR_OP_MATH));
      VLOG(5) << "use cudnn_tensor_op_math";
    } else if (dtype == CUDNN_DATA_FLOAT && !args.cdesc.allow_tf32_) {
#if CUDA_VERSION >= 11000
      PADDLE_ENFORCE_CUDA_SUCCESS(
          platform::dynload::cudnnSetConvolutionMathType(args.cdesc.desc(),
                                                         CUDNN_FMA_MATH));
#endif  // CUDA_VERSION >= 11000
    }
#endif

    if (!exhaustive_search && !deterministic) {
#if CUDNN_VERSION >= 7001
      int perf_count;
      int best_algo_idx = 0;
      std::unique_ptr<perf_t[]> perf_results(
          new perf_t[kNUM_CUDNN_BWD_DATA_ALGS]);
      PADDLE_ENFORCE_CUDA_SUCCESS(
          platform::dynload::cudnnGetConvolutionBackwardDataAlgorithm_v7(
              args.handle, args.wdesc.desc(), args.odesc.desc(),
              args.cdesc.desc(), args.idesc.desc(), kNUM_CUDNN_BWD_DATA_ALGS,
              &perf_count, perf_results.get()));
      algo = (perf_results.get())[best_algo_idx].algo;

#if CUDNN_VERSION < 7500
      int stride_dim = args.x->dims().size() - 2;
      bool blacklist = std::any_of(args.s.begin(), args.s.begin() + stride_dim,
                                   [=](int n) { return n != 1; });
      if (blacklist && (static_cast<cudnnConvolutionBwdDataAlgo_t>(
                            perf_results[best_algo_idx].algo) ==
                            CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT_TILING ||
                        static_cast<cudnnConvolutionBwdDataAlgo_t>(
                            perf_results[best_algo_idx].algo) ==
                            CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT)) {
        algo = CUDNN_CONVOLUTION_BWD_DATA_ALGO_1;
      }
#endif
      workspace_size = GetWorkspaceSize(args, algo);
      if (workspace_size > workspace_size_limit) {
        has_got_workspace_size = false;
#if CUDNN_VERSION >= 8000
        // cudnnGetConvolutionBackwardDataAlgorithm is removed in CUDNN-8
        ChooseAlgoByWorkspace<perf_t, algo_t>(perf_results.get(),
                                              kNUM_CUDNN_BWD_DATA_ALGS,
                                              workspace_size_limit, &algo);
#else
        VLOG(1) << "Fallback to non-v7 method to find conv algorithm becasue "
                   "the workspace size request("
                << workspace_size << ") exceeds the limit("
                << workspace_size_limit << ")";
        PADDLE_ENFORCE_CUDA_SUCCESS(
            platform::dynload::cudnnGetConvolutionBackwardDataAlgorithm(
                args.handle, args.wdesc.desc(), args.odesc.desc(),
                args.cdesc.desc(), args.idesc.desc(),
                CUDNN_CONVOLUTION_BWD_DATA_SPECIFY_WORKSPACE_LIMIT,
                workspace_size_limit, &algo));
#endif
      }
#else
      PADDLE_ENFORCE_CUDA_SUCCESS(
          platform::dynload::cudnnGetConvolutionBackwardDataAlgorithm(
              args.handle, args.wdesc.desc(), args.odesc.desc(),
              args.cdesc.desc(), args.idesc.desc(),
              CUDNN_CONVOLUTION_BWD_DATA_SPECIFY_WORKSPACE_LIMIT,
              workspace_size_limit, &algo));
#endif
    } else if (deterministic) {
      return CUDNN_CONVOLUTION_BWD_DATA_ALGO_1;
    } else {
      auto& dev_ctx =
          ctx.template device_context<platform::CUDADeviceContext>();
      auto workspace_handle = dev_ctx.cudnn_workspace_handle();

      AlgorithmsCache<algo_t>& algo_cache =
          *(framework::ConvSearchCache::Instance().GetBackwardData());

      auto x_dims = framework::vectorize(args.x->dims());
      auto w_dims = framework::vectorize(args.w->dims());

      VLOG(10) << "cudnnConvolutionFwdAlgoPerf_t"
               << ", x_dims:" << x_dims << ", w_dims:" << w_dims << ", args.s"
               << args.s << ", args.p" << args.p << ", args.d" << args.d;

      algo = algo_cache.GetAlgorithm(
          x_dims, w_dims, args.s, args.p, args.d, 0,
          static_cast<int64_t>(args.cudnn_dtype), [&]() {
            int returned_algo_count;
            std::array<perf_t, kNUM_CUDNN_BWD_DATA_ALGS> perf_stat;

            auto cudnn_find_func = [&](void* cudnn_workspace_ptr) {
              PADDLE_ENFORCE_CUDA_SUCCESS(
                  platform::dynload::
                      cudnnFindConvolutionBackwardDataAlgorithmEx(
                          args.handle, args.wdesc.desc(), args.w->data<T>(),
                          args.odesc.desc(), args.o->data<T>(),
                          args.cdesc.desc(), args.idesc.desc(),
                          const_cast<T*>(args.x->data<T>()),
                          kNUM_CUDNN_BWD_DATA_ALGS, &returned_algo_count,
                          perf_stat.data(), cudnn_workspace_ptr,
                          workspace_size_limit));
            };
            workspace_handle.RunFuncSync(cudnn_find_func, workspace_size_limit);

            VLOG(3) << "BwdDataAlgo Perf result: (algo: stat, time, memory)";
            for (int i = 0; i < returned_algo_count; ++i) {
              const auto& stat = perf_stat[i];
              VLOG(3) << stat.algo << ": " << stat.status << " " << stat.time
                      << " " << stat.memory;
            }

            return perf_stat[0].algo;
          });
    }
    VLOG(3) << "choose algo " << algo;
    return algo;
  }

  static size_t GetWorkspaceSize(const ConvArgs& args, algo_t algo) {
    size_t workspace_size = 0;
    PADDLE_ENFORCE_CUDA_SUCCESS(
        platform::dynload::cudnnGetConvolutionBackwardDataWorkspaceSize(
            args.handle, args.wdesc.desc(), args.odesc.desc(),
            args.cdesc.desc(), args.idesc.desc(), algo, &workspace_size));
    return workspace_size;
  }
};

template <>
struct SearchAlgorithm<cudnnConvolutionBwdFilterAlgoPerf_t> {
  using perf_t = cudnnConvolutionBwdFilterAlgoPerf_t;
  using algo_t = cudnnConvolutionBwdFilterAlgo_t;

  template <typename T>
  static algo_t Find(const ConvArgs& args, bool exhaustive_search,
                     bool deterministic,
                     const framework::ExecutionContext& ctx) {
    auto dtype = platform::CudnnDataType<T>::type;
    size_t workspace_size_limit = FLAGS_conv_workspace_size_limit * 1024 * 1024;
    size_t workspace_size = 0;
    bool has_got_workspace_size = true;

#if CUDA_VERSION >= 9000 && CUDNN_VERSION_MIN(7, 0, 1)
    auto& dev_ctx = ctx.template device_context<platform::CUDADeviceContext>();
    PADDLE_ENFORCE_CUDA_SUCCESS(platform::dynload::cudnnSetConvolutionMathType(
        args.cdesc.desc(), CUDNN_DEFAULT_MATH));
    VLOG(5) << "NOT use cudnn_tensor_op_math";
    if (dev_ctx.GetComputeCapability() >= 70 && dtype == CUDNN_DATA_HALF) {
      PADDLE_ENFORCE_CUDA_SUCCESS(
          platform::dynload::cudnnSetConvolutionMathType(args.cdesc.desc(),
                                                         CUDNN_TENSOR_OP_MATH));
      VLOG(5) << "use cudnn_tensor_op_math";
    } else if (dtype == CUDNN_DATA_FLOAT && !args.cdesc.allow_tf32_) {
#if CUDA_VERSION >= 11000
      PADDLE_ENFORCE_CUDA_SUCCESS(
          platform::dynload::cudnnSetConvolutionMathType(args.cdesc.desc(),
                                                         CUDNN_FMA_MATH));
#endif  // CUDA_VERSION >= 11000
    }
#endif

    algo_t algo;
    if (!exhaustive_search && !deterministic) {
#if CUDNN_VERSION >= 7001
      using perf_t = cudnnConvolutionBwdFilterAlgoPerf_t;
      int perf_count;
      int best_algo_idx = 0;
      std::unique_ptr<perf_t[]> perf_results(
          new perf_t[kNUM_CUDNN_BWD_FILTER_ALGS]);
      PADDLE_ENFORCE_CUDA_SUCCESS(
          platform::dynload::cudnnGetConvolutionBackwardFilterAlgorithm_v7(
              args.handle, args.idesc.desc(), args.odesc.desc(),
              args.cdesc.desc(), args.wdesc.desc(), kNUM_CUDNN_BWD_FILTER_ALGS,
              &perf_count, perf_results.get()));
      algo = (perf_results.get())[best_algo_idx].algo;
      workspace_size = GetWorkspaceSize(args, algo);
      if (workspace_size > workspace_size_limit) {
        workspace_size = workspace_size_limit;
#if CUDNN_VERSION >= 8000
        // cudnnGetConvolutionBackwardFilterAlgorithm is removed in CUDNN-8
        ChooseAlgoByWorkspace<perf_t, algo_t>(perf_results.get(),
                                              kNUM_CUDNN_BWD_FILTER_ALGS,
                                              workspace_size_limit, &algo);
#else
        VLOG(1) << "Fallback to non-v7 method to find conv algorithm becasue "
                   "the workspace size request("
                << workspace_size << ") exceeds the limit("
                << workspace_size_limit << ")";
        PADDLE_ENFORCE_CUDA_SUCCESS(
            platform::dynload::cudnnGetConvolutionBackwardFilterAlgorithm(
                args.handle, args.idesc.desc(), args.odesc.desc(),
                args.cdesc.desc(), args.wdesc.desc(),
                CUDNN_CONVOLUTION_BWD_FILTER_SPECIFY_WORKSPACE_LIMIT,
                workspace_size_limit, &algo));
#endif
      }
#else
      PADDLE_ENFORCE_CUDA_SUCCESS(
          platform::dynload::cudnnGetConvolutionBackwardFilterAlgorithm(
              args.handle, args.idesc.desc(), args.odesc.desc(),
              args.cdesc.desc(), args.wdesc.desc(),
              CUDNN_CONVOLUTION_BWD_FILTER_SPECIFY_WORKSPACE_LIMIT,
              workspace_size_limit, &algo));
#endif
    } else if (deterministic) {
      return CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1;
    } else {
      auto& dev_ctx =
          ctx.template device_context<platform::CUDADeviceContext>();
      auto workspace_handle = dev_ctx.cudnn_workspace_handle();
      AlgorithmsCache<algo_t>& algo_cache =
          *(framework::ConvSearchCache::Instance().GetBackwardFilter());

      auto x_dims = framework::vectorize(args.x->dims());
      auto w_dims = framework::vectorize(args.w->dims());

      VLOG(10) << "cudnnConvolutionFwdAlgoPerf_t:"
               << ", x_dims:" << x_dims << ", w_dims:" << w_dims << ", args.s"
               << args.s << ", args.p" << args.p << ", args.d" << args.d;
      if (dtype != CUDNN_DATA_HALF) {
        algo = algo_cache.GetAlgorithm(
            x_dims, w_dims, args.s, args.p, args.d, 0,
            static_cast<int64_t>(args.cudnn_dtype), [&]() {
              int returned_algo_count;
              std::array<perf_t, kNUM_CUDNN_BWD_FILTER_ALGS> perf_stat;
              auto cudnn_find_func = [&](void* cudnn_workspace_ptr) {
                PADDLE_ENFORCE_CUDA_SUCCESS(
                    platform::dynload::
                        cudnnFindConvolutionBackwardFilterAlgorithmEx(
                            args.handle, args.idesc.desc(), args.x->data<T>(),
                            args.odesc.desc(), args.o->data<T>(),
                            args.cdesc.desc(), args.wdesc.desc(),
                            const_cast<T*>(args.w->data<T>()),
                            kNUM_CUDNN_BWD_FILTER_ALGS, &returned_algo_count,
                            perf_stat.data(), cudnn_workspace_ptr,
                            workspace_size_limit));
              };
              workspace_handle.RunFuncSync(cudnn_find_func,
                                           workspace_size_limit);

              VLOG(3)
                  << "BwdFilterAlgo Perf result: (algo: stat, time, memory)";
              for (int i = 0; i < returned_algo_count; ++i) {
                const auto& stat = perf_stat[i];
                VLOG(3) << stat.algo << ": " << stat.status << " " << stat.time
                        << " " << stat.memory;
              }
              return perf_stat[0].algo;
            });
      } else {
        auto max_algos = MaxBwdFilterAlgos(args.handle);
        algo = algo_cache.GetAlgorithm(
            x_dims, w_dims, args.s, args.p, args.d, 0,
            static_cast<int64_t>(args.cudnn_dtype), [&]() {
              algo_t chosen_algo;
              std::vector<perf_t> perf_results(max_algos);
              int actual_algos = 0;
              PADDLE_ENFORCE_CUDA_SUCCESS(
                  platform::dynload::
                      cudnnFindConvolutionBackwardFilterAlgorithm(
                          args.handle, args.idesc.desc(), args.odesc.desc(),
                          args.cdesc.desc(), args.wdesc.desc(),
                          perf_results.size(), &actual_algos,
                          perf_results.data()));
              perf_results.resize(actual_algos);
              ChooseAlgo<perf_t, algo_t>(perf_results, workspace_size_limit,
                                         &chosen_algo);
              return chosen_algo;
            });
      }
    }
    VLOG(3) << "choose algo " << algo;
    return algo;
  }

  static size_t GetWorkspaceSize(const ConvArgs& args, algo_t algo) {
    size_t workspace_size = 0;
    PADDLE_ENFORCE_CUDA_SUCCESS(
        platform::dynload::cudnnGetConvolutionBackwardFilterWorkspaceSize(
            args.handle, args.idesc.desc(), args.odesc.desc(),
            args.cdesc.desc(), args.wdesc.desc(), algo, &workspace_size));
    return workspace_size;
  }
};

}  // namespace operators
}  // namespace paddle
