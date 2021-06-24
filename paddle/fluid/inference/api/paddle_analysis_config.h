// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

///
/// \file paddle_analysis_config.h
///
/// \brief Paddle Analysis Config API信息
///
/// \author paddle-infer@baidu.com
/// \date 2020-03-20
/// \since 1.7
///

#pragma once

#include <cassert>
#include <map>
#include <memory>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "paddle_infer_declare.h"  // NOLINT

/*! \file */
// Here we include some header files with relative paths, for that in deploy,
// the abstract path of this header file will be changed.
#include "paddle_api.h"           // NOLINT
#include "paddle_pass_builder.h"  // NOLINT
#ifdef PADDLE_WITH_MKLDNN
#include "paddle_mkldnn_quantizer_config.h"  // NOLINT
#endif

namespace paddle {

class AnalysisPredictor;
struct MkldnnQuantizerConfig;

///
/// \brief configuration manager for AnalysisPredictor.
/// \since 1.7.0
///
/// AnalysisConfig manages configurations of AnalysisPredictor.
/// During inference procedure, there are many parameters(model/params path,
/// place of inference, etc.)
/// to be specified, and various optimizations(subgraph fusion, memory
/// optimazation, TensorRT engine, etc.)
/// to be done. Users can manage these settings by creating and modifying an
/// AnalysisConfig,
/// and loading it into AnalysisPredictor.
///
struct PD_INFER_DECL AnalysisConfig {
  AnalysisConfig() = default;
  ///
  /// \brief Construct a new AnalysisConfig from another
  /// AnalysisConfig.
  ///
  /// \param[in] other another AnalysisConfig
  ///
  explicit AnalysisConfig(const AnalysisConfig& other);
  ///
  /// \brief Construct a new AnalysisConfig from a no-combined model.
  ///
  /// \param[in] model_dir model directory of the no-combined model.
  ///
  explicit AnalysisConfig(const std::string& model_dir);
  ///
  /// \brief Construct a new AnalysisConfig from a combined model.
  ///
  /// \param[in] prog_file model file path of the combined model.
  /// \param[in] params_file params file path of the combined model.
  ///
  explicit AnalysisConfig(const std::string& prog_file,
                          const std::string& params_file);
  ///
  /// \brief Precision of inference in TensorRT.
  ///
  enum class Precision {
    kFloat32 = 0,  ///< fp32
    kInt8,         ///< int8
    kHalf,         ///< fp16
  };

  ///
  /// \brief Set the no-combined model dir path.
  ///
  /// \param model_dir model dir path.
  ///
  void SetModel(const std::string& model_dir) { model_dir_ = model_dir; }

  ///
  /// \brief Set the combined model with two specific pathes for program and
  /// parameters.
  ///
  /// \param prog_file_path model file path of the combined model.
  /// \param params_file_path params file path of the combined model.
  ///
  void SetModel(const std::string& prog_file_path,
                const std::string& params_file_path);
  ///
  /// \brief Set the model file path of a combined model.
  ///
  /// \param x model file path.
  ///
  void SetProgFile(const std::string& x) { prog_file_ = x; }
  ///
  /// \brief Set the params file path of a combined model.
  ///
  /// \param x params file path.
  ///
  void SetParamsFile(const std::string& x) { params_file_ = x; }

  ///
  /// \brief Set the path of optimization cache directory.
  ///
  /// \param opt_cache_dir the path of optimization cache directory.
  ///
  void SetOptimCacheDir(const std::string& opt_cache_dir) {
    opt_cache_dir_ = opt_cache_dir;
  }
  ///
  /// \brief Get the model directory path.
  ///
  /// \return const std::string& The model directory path.
  ///
  const std::string& model_dir() const { return model_dir_; }
  ///
  /// \brief Get the program file path.
  ///
  /// \return const std::string& The program file path.
  ///
  const std::string& prog_file() const { return prog_file_; }
  ///
  /// \brief Get the combined parameters file.
  ///
  /// \return const std::string& The combined parameters file.
  ///
  const std::string& params_file() const { return params_file_; }

  // Padding related.

  ///
  /// \brief Turn off FC Padding.
  ///
  ///
  void DisableFCPadding();
  ///
  /// \brief A boolean state telling whether fc padding is used.
  ///
  /// \return bool Whether fc padding is used.
  ///
  bool use_fc_padding() const { return use_fc_padding_; }

  // GPU related.

  ///
  /// \brief Turn on GPU.
  ///
  /// \param memory_pool_init_size_mb initial size of the GPU memory pool in MB.
  /// \param device_id device_id the GPU card to use (default is 0).
  ///
  void EnableUseGpu(uint64_t memory_pool_init_size_mb, int device_id = 0);
  ///
  /// \brief Turn off GPU.
  ///
  ///
  void DisableGpu();

  ///
  /// \brief Turn on XPU.
  ///
  /// \param l3_workspace_size The size of the video memory allocated by the l3
  ///         cache, the maximum is 16M.
  /// \param locked Whether the allocated L3 cache can be locked. If false,
  ///       it means that the L3 cache is not locked, and the allocated L3
  ///       cache can be shared by multiple models, and multiple models
  ///       sharing the L3 cache will be executed sequentially on the card.
  /// \param autotune Whether to autotune the conv operator in the model. If
  ///       true, when the conv operator of a certain dimension is executed
  ///       for the first time, it will automatically search for a better
  ///       algorithm to improve the performance of subsequent conv operators
  ///       of the same dimension.
  /// \param autotune_file Specify the path of the autotune file. If
  ///       autotune_file is specified, the algorithm specified in the
  ///       file will be used and autotune will not be performed again.
  /// \param precision Calculation accuracy of multi_encoder
  /// \param adaptive_seqlen Is the input of multi_encoder variable length
  ///
  void EnableXpu(int l3_workspace_size = 0xfffc00, bool locked = false,
                 bool autotune = true, const std::string& autotune_file = "",
                 const std::string& precision = "int16",
                 bool adaptive_seqlen = false);
  ///
  /// \brief A boolean state telling whether the GPU is turned on.
  ///
  /// \return bool Whether the GPU is turned on.
  ///
  bool use_gpu() const { return use_gpu_; }
  ///
  /// \brief A boolean state telling whether the XPU is turned on.
  ///
  /// \return bool Whether the XPU is turned on.
  ///
  bool use_xpu() const { return use_xpu_; }
  ///
  /// \brief Get the GPU device id.
  ///
  /// \return int The GPU device id.
  ///
  int gpu_device_id() const { return gpu_device_id_; }
  ///
  /// \brief Get the XPU device id.
  ///
  /// \return int The XPU device id.
  ///
  int xpu_device_id() const { return xpu_device_id_; }
  ///
  /// \brief Get the initial size in MB of the GPU memory pool.
  ///
  /// \return int The initial size in MB of the GPU memory pool.
  ///
  int memory_pool_init_size_mb() const { return memory_pool_init_size_mb_; }
  ///
  /// \brief Get the proportion of the initial memory pool size compared to the
  /// device.
  ///
  /// \return float The proportion of the initial memory pool size.
  ///
  float fraction_of_gpu_memory_for_pool() const;

  // CUDNN related.
  ///
  /// \brief Turn on CUDNN.
  ///
  ///
  void EnableCUDNN();
  ///
  /// \brief A boolean state telling whether to use CUDNN.
  ///
  /// \return bool Whether to use CUDNN.
  ///
  bool cudnn_enabled() const { return use_cudnn_; }

  ///
  /// \brief Control whether to perform IR graph optimization.
  /// If turned off, the AnalysisConfig will act just like a NativeConfig.
  ///
  /// \param x Whether the ir graph optimization is actived.
  ///
  void SwitchIrOptim(int x = true) { enable_ir_optim_ = x; }
  ///
  /// \brief A boolean state telling whether the ir graph optimization is
  /// actived.
  ///
  /// \return bool Whether to use ir graph optimization.
  ///
  bool ir_optim() const { return enable_ir_optim_; }

  ///
  /// \brief INTERNAL Determine whether to use the feed and fetch operators.
  /// Just for internal development, not stable yet.
  /// When ZeroCopyTensor is used, this should be turned off.
  ///
  /// \param x Whether to use the feed and fetch operators.
  ///
  void SwitchUseFeedFetchOps(int x = true) { use_feed_fetch_ops_ = x; }
  ///
  /// \brief A boolean state telling whether to use the feed and fetch
  /// operators.
  ///
  /// \return bool Whether to use the feed and fetch operators.
  ///
  bool use_feed_fetch_ops_enabled() const { return use_feed_fetch_ops_; }

  ///
  /// \brief Control whether to specify the inputs' names.
  /// The ZeroCopyTensor type has a name member, assign it with the
  /// corresponding
  /// variable name. This is used only when the input ZeroCopyTensors passed to
  /// the
  /// AnalysisPredictor.ZeroCopyRun() cannot follow the order in the training
  /// phase.
  ///
  /// \param x Whether to specify the inputs' names.
  ///
  void SwitchSpecifyInputNames(bool x = true) { specify_input_name_ = x; }
  ///
  /// \brief A boolean state tell whether the input ZeroCopyTensor names
  /// specified should
  /// be used to reorder the inputs in AnalysisPredictor.ZeroCopyRun().
  ///
  /// \return bool Whether to specify the inputs' names.
  ///
  bool specify_input_name() const { return specify_input_name_; }

  ///
  /// \brief Turn on the TensorRT engine.
  /// The TensorRT engine will accelerate some subgraphes in the original Fluid
  /// computation graph. In some models such as resnet50, GoogleNet and so on,
  /// it gains significant performance acceleration.
  ///
  /// \param workspace_size The memory size(in byte) used for TensorRT
  /// workspace.
  /// \param max_batch_size The maximum batch size of this prediction task,
  /// better set as small as possible for less performance loss.
  /// \param min_subgraph_size The minimum TensorRT subgraph size needed, if a
  /// subgraph is smaller than this, it will not be transferred to TensorRT
  /// engine.
  /// \param precision The precision used in TensorRT.
  /// \param use_static Serialize optimization information to disk for reusing.
  /// \param use_calib_mode Use TRT int8 calibration(post training
  /// quantization).
  ///
  ///
  void EnableTensorRtEngine(int workspace_size = 1 << 20,
                            int max_batch_size = 1, int min_subgraph_size = 3,
                            Precision precision = Precision::kFloat32,
                            bool use_static = false,
                            bool use_calib_mode = true);
  ///
  /// \brief A boolean state telling whether the TensorRT engine is used.
  ///
  /// \return bool Whether the TensorRT engine is used.
  ///
  bool tensorrt_engine_enabled() const { return use_tensorrt_; }
  ///
  /// \brief Set min, max, opt shape for TensorRT Dynamic shape mode.
  /// \param min_input_shape The min input shape of the subgraph input.
  /// \param max_input_shape The max input shape of the subgraph input.
  /// \param opt_input_shape The opt input shape of the subgraph input.
  /// \param disable_trt_plugin_fp16 Setting this parameter to true means that
  /// TRT plugin will not run fp16.
  ///
  void SetTRTDynamicShapeInfo(
      std::map<std::string, std::vector<int>> min_input_shape,
      std::map<std::string, std::vector<int>> max_input_shape,
      std::map<std::string, std::vector<int>> optim_input_shape,
      bool disable_trt_plugin_fp16 = false);

  ///
  /// \brief Prevent ops running in Paddle-TRT
  /// NOTE: just experimental, not an official stable API, easy to be broken.
  ///
  void Exp_DisableTensorRtOPs(const std::vector<std::string>& ops);

  ///
  /// \brief Replace some TensorRT plugins to TensorRT OSS(
  /// https://github.com/NVIDIA/TensorRT), with which some models's inference
  /// may be more high-performance. Libnvinfer_plugin.so greater than
  /// V7.2.1 is needed.
  ///
  void EnableTensorRtOSS();

  ///
  /// \brief A boolean state telling whether to use the TensorRT OSS.
  ///
  /// \return bool Whether to use the TensorRT OSS.
  ///
  bool tensorrt_oss_enabled() { return trt_use_oss_; }

  ///
  /// \brief Enable TensorRT DLA
  /// \param dla_core ID of DLACore, which should be 0, 1,
  ///        ..., IBuilder.getNbDLACores() - 1
  ///
  void EnableTensorRtDLA(int dla_core = 0);

  ///
  /// \brief A boolean state telling whether to use the TensorRT DLA.
  ///
  /// \return bool Whether to use the TensorRT DLA.
  ///
  bool tensorrt_dla_enabled() { return trt_use_dla_; }

  void EnableDlnne(int min_subgraph_size = 3);
  bool dlnne_enabled() const { return use_dlnne_; }

  ///
  /// \brief Turn on the usage of Lite sub-graph engine.
  ///
  /// \param precision_mode Precion used in Lite sub-graph engine.
  /// \param passes_filter Set the passes used in Lite sub-graph engine.
  /// \param ops_filter Operators not supported by Lite.
  ///
  void EnableLiteEngine(
      AnalysisConfig::Precision precision_mode = Precision::kFloat32,
      bool zero_copy = false,
      const std::vector<std::string>& passes_filter = {},
      const std::vector<std::string>& ops_filter = {});

  ///
  /// \brief A boolean state indicating whether the Lite sub-graph engine is
  /// used.
  ///
  /// \return bool whether the Lite sub-graph engine is used.
  ///
  bool lite_engine_enabled() const { return use_lite_; }

  ///
  /// \brief Control whether to debug IR graph analysis phase.
  /// This will generate DOT files for visualizing the computation graph after
  /// each analysis pass applied.
  ///
  /// \param x whether to debug IR graph analysis phase.
  ///
  void SwitchIrDebug(int x = true);

  ///
  /// \brief Turn on MKLDNN.
  ///
  ///
  void EnableMKLDNN();
  ///
  /// \brief Set the cache capacity of different input shapes for MKLDNN.
  /// Default value 0 means not caching any shape.
  /// Please see MKL-DNN Data Caching Design Document:
  /// https://github.com/PaddlePaddle/FluidDoc/blob/develop/doc/fluid/design/mkldnn/caching/caching.md
  ///
  /// \param capacity The cache capacity.
  ///
  void SetMkldnnCacheCapacity(int capacity);
  ///
  /// \brief A boolean state telling whether to use the MKLDNN.
  ///
  /// \return bool Whether to use the MKLDNN.
  ///
  bool mkldnn_enabled() const { return use_mkldnn_; }

  ///
  /// \brief Set the number of cpu math library threads.
  ///
  /// \param cpu_math_library_num_threads The number of cpu math library
  /// threads.
  ///
  void SetCpuMathLibraryNumThreads(int cpu_math_library_num_threads);
  ///
  /// \brief An int state telling how many threads are used in the CPU math
  /// library.
  ///
  /// \return int The number of threads used in the CPU math library.
  ///
  int cpu_math_library_num_threads() const {
    return cpu_math_library_num_threads_;
  }

  ///
  /// \brief Transform the AnalysisConfig to NativeConfig.
  ///
  /// \return NativeConfig The NativeConfig transformed.
  ///
  NativeConfig ToNativeConfig() const;
  ///
  /// \brief Specify the operator type list to use MKLDNN acceleration.
  ///
  /// \param op_list The operator type list.
  ///
  void SetMKLDNNOp(std::unordered_set<std::string> op_list) {
    mkldnn_enabled_op_types_ = op_list;
  }

  ///
  /// \brief Turn on MKLDNN quantization.
  ///
  ///
  void EnableMkldnnQuantizer();

  ///
  /// \brief Turn on MKLDNN bfloat16.
  ///
  ///
  void EnableMkldnnBfloat16();

  ///
  /// \brief A boolean state telling whether to use the MKLDNN Bfloat16.
  ///
  /// \return bool Whether to use the MKLDNN Bfloat16.
  ///
  bool mkldnn_bfloat16_enabled() const { return use_mkldnn_bfloat16_; }

  /// \brief Specify the operator type list to use Bfloat16 acceleration.
  ///
  /// \param op_list The operator type list.
  ///
  void SetBfloat16Op(std::unordered_set<std::string> op_list) {
    bfloat16_enabled_op_types_ = op_list;
  }

  ///
  /// \brief A boolean state telling whether the thread local CUDA stream is
  /// enabled.
  ///
  /// \return bool Whether the thread local CUDA stream is enabled.
  ///
  bool thread_local_stream_enabled() const { return thread_local_stream_; }

  ///
  /// \brief A boolean state telling whether the MKLDNN quantization is enabled.
  ///
  /// \return bool Whether the MKLDNN quantization is enabled.
  ///
  bool mkldnn_quantizer_enabled() const { return use_mkldnn_quantizer_; }

  ///
  /// \brief Get MKLDNN quantizer config.
  ///
  /// \return MkldnnQuantizerConfig* MKLDNN quantizer config.
  ///
  MkldnnQuantizerConfig* mkldnn_quantizer_config() const;

  ///
  /// \brief Specify the memory buffer of program and parameter.
  /// Used when model and params are loaded directly from memory.
  ///
  /// \param prog_buffer The memory buffer of program.
  /// \param prog_buffer_size The size of the model data.
  /// \param params_buffer The memory buffer of the combined parameters file.
  /// \param params_buffer_size The size of the combined parameters data.
  ///
  void SetModelBuffer(const char* prog_buffer, size_t prog_buffer_size,
                      const char* params_buffer, size_t params_buffer_size);
  ///
  /// \brief A boolean state telling whether the model is set from the CPU
  /// memory.
  ///
  /// \return bool Whether model and params are loaded directly from memory.
  ///
  bool model_from_memory() const { return model_from_memory_; }

  ///
  /// \brief Turn on memory optimize
  /// NOTE still in development.
  ///
  void EnableMemoryOptim();
  ///
  /// \brief A boolean state telling whether the memory optimization is
  /// activated.
  ///
  /// \return bool Whether the memory optimization is activated.
  ///
  bool enable_memory_optim() const;

  ///
  /// \brief Turn on profiling report.
  /// If not turned on, no profiling report will be generated.
  ///
  void EnableProfile();
  ///
  /// \brief A boolean state telling whether the profiler is activated.
  ///
  /// \return bool Whether the profiler is activated.
  ///
  bool profile_enabled() const { return with_profile_; }

  ///
  /// \brief Mute all logs in Paddle inference.
  ///
  void DisableGlogInfo();
  ///
  /// \brief A boolean state telling whether logs in Paddle inference are muted.
  ///
  /// \return bool Whether logs in Paddle inference are muted.
  ///
  bool glog_info_disabled() const { return !with_glog_info_; }

  ///
  /// \brief Set the AnalysisConfig to be invalid.
  /// This is to ensure that an AnalysisConfig can only be used in one
  /// AnalysisPredictor.
  ///
  void SetInValid() const { is_valid_ = false; }
  ///
  /// \brief A boolean state telling whether the AnalysisConfig is valid.
  ///
  /// \return bool Whether the AnalysisConfig is valid.
  ///
  bool is_valid() const { return is_valid_; }

  friend class ::paddle::AnalysisPredictor;

  ///
  /// \brief Get a pass builder for customize the passes in IR analysis phase.
  /// NOTE: Just for developer, not an official API, easy to be broken.
  ///
  ///
  PassStrategy* pass_builder() const;

  ///
  /// \brief Enable the GPU multi-computing stream feature.
  /// NOTE: The current behavior of this interface is to bind the computation
  /// stream to the thread, and this behavior may be changed in the future.
  ///
  void EnableGpuMultiStream();
  void PartiallyRelease();

 protected:
  // Update the config.
  void Update();

  std::string SerializeInfoCache();

 protected:
  // Model pathes.
  std::string model_dir_;
  mutable std::string prog_file_;
  mutable std::string params_file_;

  // GPU related.
  bool use_gpu_{false};
  int gpu_device_id_{0};
  int xpu_device_id_{0};
  uint64_t memory_pool_init_size_mb_{100};  // initial size is 100MB.

  bool use_cudnn_{false};

  // Padding related
  bool use_fc_padding_{true};

  // TensorRT related.
  bool use_tensorrt_{false};
  // For workspace_size, refer it from here:
  // https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#troubleshooting
  int tensorrt_workspace_size_{1 << 30};
  // While TensorRT allows an engine optimized for a given max batch size
  // to run at any smaller size, the performance for those smaller
  // sizes may not be as well-optimized. Therefore, Max batch is best
  // equivalent to the runtime batch size.
  int tensorrt_max_batchsize_{1};
  //  We transform the Ops that can be converted into TRT layer in the model,
  //  and aggregate these Ops into subgraphs for TRT execution.
  //  We set this variable to control the minimum number of nodes in the
  //  subgraph, 3 as default value.
  int tensorrt_min_subgraph_size_{3};
  Precision tensorrt_precision_mode_{Precision::kFloat32};
  bool trt_use_static_engine_{false};
  bool trt_use_calib_mode_{true};
  bool trt_use_oss_{false};
  bool trt_use_dla_{false};
  int trt_dla_core_{0};
  std::map<std::string, std::vector<int>> min_input_shape_{};
  std::map<std::string, std::vector<int>> max_input_shape_{};
  std::map<std::string, std::vector<int>> optim_input_shape_{};
  std::vector<std::string> trt_disabled_ops_{};
  bool disable_trt_plugin_fp16_{false};

  // dlnne related.
  bool use_dlnne_{false};
  int dlnne_min_subgraph_size_{3};

  // memory reuse related.
  bool enable_memory_optim_{false};

  bool use_mkldnn_{false};
  std::unordered_set<std::string> mkldnn_enabled_op_types_;

  bool model_from_memory_{false};

  bool enable_ir_optim_{true};
  bool use_feed_fetch_ops_{true};
  bool ir_debug_{false};

  bool specify_input_name_{false};

  int cpu_math_library_num_threads_{1};

  bool with_profile_{false};

  bool with_glog_info_{true};

  // A runtime cache, shouldn't be transferred to others.
  std::string serialized_info_cache_;

  mutable std::unique_ptr<PassStrategy> pass_builder_;

  bool use_lite_{false};
  std::vector<std::string> lite_passes_filter_;
  std::vector<std::string> lite_ops_filter_;
  Precision lite_precision_mode_;
  bool lite_zero_copy_;

  bool thread_local_stream_{false};
  bool use_xpu_{false};
  int xpu_l3_workspace_size_;
  bool xpu_locked_;
  bool xpu_autotune_;
  std::string xpu_autotune_file_;
  std::string xpu_precision_;
  bool xpu_adaptive_seqlen_;

  // mkldnn related.
  int mkldnn_cache_capacity_{10};
  bool use_mkldnn_quantizer_{false};
  std::shared_ptr<MkldnnQuantizerConfig> mkldnn_quantizer_config_;
  bool use_mkldnn_bfloat16_{false};
  std::unordered_set<std::string> bfloat16_enabled_op_types_;

  // If the config is already used on a predictor, it becomes invalid.
  // Any config can only be used with one predictor.
  // Variables held by config can take up a lot of memory in some cases.
  // So we release the memory when the predictor is set up.
  mutable bool is_valid_{true};
  std::string opt_cache_dir_;
};

}  // namespace paddle
