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

#include "paddle/fluid/inference/api/analysis_predictor.h"

#include <glog/logging.h>

#include <algorithm>
#include <fstream>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "paddle/fluid/extension/include/ext_op_meta_info.h"
#include "paddle/fluid/framework/feed_fetch_method.h"
#include "paddle/fluid/framework/feed_fetch_type.h"
#include "paddle/fluid/framework/ir/fuse_pass_base.h"
#include "paddle/fluid/framework/ir/pass.h"
#include "paddle/fluid/framework/naive_executor.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/framework/var_type_traits.h"
#include "paddle/fluid/framework/version.h"
#include "paddle/fluid/inference/analysis/helper.h"
#include "paddle/fluid/inference/analysis/passes/memory_optimize_pass.h"
#include "paddle/fluid/inference/api/helper.h"
#include "paddle/fluid/inference/api/paddle_inference_pass.h"
#include "paddle/fluid/inference/utils/singleton.h"
#include "paddle/fluid/memory/memcpy.h"
#include "paddle/fluid/platform/cpu_helper.h"
#include "paddle/fluid/platform/device_context.h"
#include "paddle/fluid/platform/gpu_info.h"
#include "paddle/fluid/platform/place.h"
#include "paddle/fluid/platform/profiler.h"

#ifdef PADDLE_WITH_MKLML
#include "paddle/fluid/platform/dynload/mklml.h"
#endif

#ifdef PADDLE_WITH_MKLDNN
#include "paddle/fluid/inference/api/mkldnn_quantizer.h"
#endif

#if PADDLE_WITH_TENSORRT
#include "paddle/fluid/inference/tensorrt/convert/op_converter.h"
#include "paddle/fluid/inference/tensorrt/trt_int8_calibrator.h"
#endif

namespace paddle {

using inference::Singleton;
#if PADDLE_WITH_TENSORRT
using inference::tensorrt::TRTCalibratorEngine;
using inference::tensorrt::TRTCalibratorEngineManager;
using inference::tensorrt::TRTInt8Calibrator;
#endif

namespace {
bool IsPersistable(const framework::VarDesc *var) {
  if (var->Persistable() &&
      var->GetType() != framework::proto::VarType::FEED_MINIBATCH &&
      var->GetType() != framework::proto::VarType::FETCH_LIST &&
      var->GetType() != framework::proto::VarType::RAW) {
    return true;
  }
  return false;
}
}  // namespace

bool PaddleTensorToLoDTensor(const PaddleTensor &pt, framework::LoDTensor *t,
                             const platform::Place &place) {
  framework::DDim ddim = framework::make_ddim(pt.shape);
  void *input_ptr;
  if (pt.dtype == PaddleDType::INT64) {
    input_ptr = t->mutable_data<int64_t>(ddim, place);
  } else if (pt.dtype == PaddleDType::FLOAT32) {
    input_ptr = t->mutable_data<float>(ddim, place);
  } else if (pt.dtype == PaddleDType::INT32) {
    input_ptr = t->mutable_data<int32_t>(ddim, place);
  } else {
    LOG(ERROR) << "unsupported feed type " << pt.dtype;
    return false;
  }

  PADDLE_ENFORCE_NOT_NULL(
      input_ptr,
      paddle::platform::errors::Fatal(
          "Cannot convert to LoDTensor because LoDTensor creation failed."));
  PADDLE_ENFORCE_NOT_NULL(
      pt.data.data(),
      paddle::platform::errors::InvalidArgument(
          "The data contained in the input PaddleTensor is illegal."));

  if (platform::is_cpu_place(place)) {
    // TODO(panyx0718): Init LoDTensor from existing memcpy to save a copy.
    std::memcpy(static_cast<void *>(input_ptr), pt.data.data(),
                pt.data.length());
  } else if (platform::is_gpu_place(place)) {
    PADDLE_ENFORCE_EQ(platform::is_xpu_place(place), false,
                      platform::errors::InvalidArgument(
                          "Only one choice can be made between CPU and XPU."));
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
    platform::DeviceContextPool &pool = platform::DeviceContextPool::Instance();
    auto *dev_ctx =
        static_cast<const platform::CUDADeviceContext *>(pool.Get(place));
    auto dst_gpu_place = BOOST_GET_CONST(platform::CUDAPlace, place);
    memory::Copy(dst_gpu_place, static_cast<void *>(input_ptr),
                 platform::CPUPlace(), pt.data.data(), pt.data.length(),
                 dev_ctx->stream());
#else
    PADDLE_THROW(paddle::platform::errors::Fatal(
        "Not compile with CUDA, should not reach here."));
#endif
  } else if (platform::is_xpu_place(place)) {
#ifdef PADDLE_WITH_XPU
    auto dst_xpu_place = BOOST_GET_CONST(platform::XPUPlace, place);
    memory::Copy(dst_xpu_place, static_cast<void *>(input_ptr),
                 platform::CPUPlace(), pt.data.data(), pt.data.length());
#else
    PADDLE_THROW(paddle::platform::errors::Fatal(
        "Not compile with XPU, should not reach here."));
#endif
  } else {
    PADDLE_THROW(paddle::platform::errors::InvalidArgument(
        "The analysis predictor supports CPU, GPU and XPU now."));
  }
  // TODO(Superjomn) Low performance, need optimization for heavy LoD copy.
  framework::LoD lod;
  for (auto &level : pt.lod) {
    lod.emplace_back(level);
  }
  t->set_lod(lod);
  return true;
}

bool AnalysisPredictor::Init(
    const std::shared_ptr<framework::Scope> &parent_scope,
    const std::shared_ptr<framework::ProgramDesc> &program) {
  VLOG(3) << "Predictor::init()";
  if (config_.with_profile_) {
    LOG(WARNING) << "Profiler is activated, which might affect the performance";
    auto tracking_device = config_.use_gpu() ? platform::ProfilerState::kAll
                                             : platform::ProfilerState::kCPU;
    platform::EnableProfiler(tracking_device);
  } else {
    LOG(INFO) << "Profiler is deactivated, and no profiling report will be "
                 "generated.";
  }

  // no matter with or without MKLDNN
  paddle::platform::SetNumThreads(config_.cpu_math_library_num_threads());

  if (!PrepareScope(parent_scope)) {
    return false;
  }
  if (!CreateExecutor()) {
    return false;
  }
  if (!PrepareProgram(program)) {
    return false;
  }

  // Prepare executor, create local variables.
  if (!PrepareExecutor()) {
    return true;
  }

  // Get the feed_target_names and fetch_target_names
  PrepareFeedFetch();

  return true;
}

bool AnalysisPredictor::PrepareScope(
    const std::shared_ptr<framework::Scope> &parent_scope) {
  if (parent_scope) {
    PADDLE_ENFORCE_NOT_NULL(
        parent_scope,
        platform::errors::PreconditionNotMet(
            "Both program and parent_scope should be set in Clone mode."));
    scope_ = parent_scope;
    status_is_cloned_ = true;
  } else {
    paddle::framework::InitDevices();
    // TODO(wilber): we need to release memory occupied by weights.
    scope_ = std::make_shared<paddle::framework::Scope>();
    status_is_cloned_ = false;
  }
  sub_scope_ = &scope_->NewScope();
  return true;
}
bool AnalysisPredictor::PrepareProgram(
    const std::shared_ptr<framework::ProgramDesc> &program) {
  if (!program) {
    if (!LoadProgramDesc()) return false;
    // If not cloned, the parameters should be loaded.
    // If config_.ir_optim() is True, parameters is loaded in
    // OptimizeInferenceProgram(), but other persistable variables
    // (like RAW type var) are not created in scope.
    // If config_.ir_optim() is False, parameters is loaded in LoadParameters(),
    // still need to create other persistable variables.
    // So in both case, create persistable variables at first.
    executor_->CreateVariables(*inference_program_, 0, true, sub_scope_);

    // if enable_ir_optim_ is false,
    // the analysis pass(op fuse, graph analysis, trt subgraph, mkldnn etc) will
    // not be executed.
    OptimizeInferenceProgram();
  } else {
    // If the program is passed from external, no need to optimize it, this
    // logic is used in the clone scenario.
    inference_program_ = program;
  }

  executor_->CreateVariables(*inference_program_, 0, false, sub_scope_);

  return true;
}
bool AnalysisPredictor::CreateExecutor() {
  if (config_.use_gpu()) {
    PADDLE_ENFORCE_EQ(config_.use_xpu(), false,
                      platform::errors::InvalidArgument(
                          "Only one choice can be made between CPU and XPU."));
    place_ = paddle::platform::CUDAPlace(config_.gpu_device_id());
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
    if (config_.thread_local_stream_enabled()) {
      auto *ctx = static_cast<platform::CUDADeviceContext *>(
          platform::DeviceContextPool::Instance().Get(place_));
      VLOG(3) << "The prediction process will be completed using a separate "
                 "normal-priority stream on each thread.";
      ctx->ResetThreadContext(platform::stream::Priority::kNormal);
    }
#endif
  } else if (config_.use_xpu()) {
    if (config_.lite_engine_enabled()) {
#ifdef LITE_SUBGRAPH_WITH_XPU
      // Currently, Paddle-Lite's XPU user interface only supports the transfer
      // of Host data pointers. If it is currently used as a subgraph, execution
      // efficiency will be sacrificed, so it is temporarily set to cpu place.
      // And, the current lite engine of xpu must execute all parts of the
      // model.
      place_ = paddle::platform::CPUPlace();
#else
      PADDLE_THROW(platform::errors::Unavailable(
          "You tried to use an XPU lite engine, but Paddle was not compiled "
          "with it."));
#endif  // LITE_SUBGRAPH_WITH_XPU
    } else {
#ifdef PADDLE_WITH_XPU
      place_ = paddle::platform::XPUPlace(config_.xpu_device_id());
#else
      PADDLE_THROW(platform::errors::Unavailable(
          "You tried to use XPU forward propagation (inference without lite "
          "engine), but Paddle was not compiled "
          "with WITH_XPU."));
#endif  // PADDLE_WITH_XPU
    }
  } else {
    place_ = paddle::platform::CPUPlace();
  }
  executor_ = std::make_unique<paddle::framework::NaiveExecutor>(place_);
  return true;
}
bool AnalysisPredictor::PrepareExecutor() {
  executor_->Prepare(sub_scope_, *inference_program_, 0,
                     config_.use_feed_fetch_ops_);

  PADDLE_ENFORCE_NOT_NULL(sub_scope_,
                          platform::errors::PreconditionNotMet(
                              "The sub_scope should not be nullptr."));

  return true;
}

void AnalysisPredictor::MkldnnPreSet(const std::vector<PaddleTensor> &inputs) {
#ifdef PADDLE_WITH_MKLDNN
  std::vector<std::vector<int>> inputs_shape;
  for (size_t i = 0; i < inputs.size(); ++i) {
    inputs_shape.emplace_back(inputs[i].shape);
  }
  MkldnnPreSet(inputs_shape);
#endif
}

void AnalysisPredictor::MkldnnPreSet(
    const std::vector<std::vector<int>> &inputs_shape) {
#ifdef PADDLE_WITH_MKLDNN
  VLOG(2) << "AnalysisPredictor::ZeroCopyRun get_cur_mkldnn_session_id="
          << platform::MKLDNNDeviceContext::tls().get_cur_mkldnn_session_id();
  // In cache clearing mode.
  if (config_.mkldnn_cache_capacity_ > 0) {
    VLOG(2) << "In mkldnn cache clear mode.";
    platform::MKLDNNDeviceContext::tls().set_cur_mkldnn_session_id(
        platform::MKLDNNDeviceContextThreadLocals::
            kMKLDNNSessionID_CacheClearing);
    platform::MKLDNNDeviceContext::tls().set_cur_input_shape_cache_capacity(
        config_.mkldnn_cache_capacity_);
    // Set current_input_shape for caching dynamic shape.
    std::stringstream ss;
    for (size_t i = 0; i < inputs_shape.size(); ++i) {
      for (size_t j = 0; j < inputs_shape[i].size(); ++j) {
        ss << inputs_shape[i][j] << "-";
      }
    }
    VLOG(2) << "Set input shape=" << ss.str();
    platform::MKLDNNDeviceContext::tls().set_cur_input_shape_str(ss.str());
  }
#endif
}

void AnalysisPredictor::MkldnnPostReset() {
#ifdef PADDLE_WITH_MKLDNN
  // In cache clearing mode.
  if (config_.mkldnn_cache_capacity_ > 0) {
    if (VLOG_IS_ON(2)) {
      auto shape_blob_size = static_cast<platform::MKLDNNDeviceContext *>(
                                 (&platform::DeviceContextPool::Instance())
                                     ->Get(platform::CPUPlace()))
                                 ->GetShapeBlobSize();
      CHECK_LE(shape_blob_size,
               static_cast<size_t>(config_.mkldnn_cache_capacity_));
    }
    paddle::platform::MKLDNNDeviceContext::tls().set_cur_mkldnn_session_id(
        platform::MKLDNNDeviceContextThreadLocals::kMKLDNNSessionID_Default);
    platform::MKLDNNDeviceContext::tls().set_cur_input_shape_cache_capacity(0);
    platform::MKLDNNDeviceContext::tls().set_cur_input_shape_str("");
  }
#endif
}

bool AnalysisPredictor::Run(const std::vector<PaddleTensor> &inputs,
                            std::vector<PaddleTensor> *output_data,
                            int batch_size) {
  paddle::platform::SetNumThreads(config_.cpu_math_library_num_threads());
#ifdef PADDLE_WITH_MKLDNN
  if (config_.use_mkldnn_) MkldnnPreSet(inputs);
#endif
  VLOG(3) << "Predictor::predict";
  inference::Timer timer;
  timer.tic();
  // set feed variable
  framework::Scope *scope = sub_scope_ ? sub_scope_ : scope_.get();
  PADDLE_ENFORCE_NOT_NULL(scope, platform::errors::PreconditionNotMet(
                                     "The scope should not be nullptr."));
  if (!SetFeed(inputs, scope)) {
    LOG(ERROR) << "fail to set feed";
    return false;
  }

  // Run the inference program
  // if share variables, we need not create variables
  executor_->Run();

  // get fetch variable
  if (!GetFetch(output_data, scope)) {
    LOG(ERROR) << "fail to get fetches";
    return false;
  }

  VLOG(3) << "predict cost: " << timer.toc() << "ms";

  // All the containers in the scope will be hold in inference, but the
  // operators assume that the container will be reset after each batch.
  // Here is a bugfix, collect all the container variables, and reset then to a
  // bool; the next time, the operator will call MutableData and construct a new
  // container again, so that the container will be empty for each batch.
  if (sub_scope_) {
    tensor_array_batch_cleaner_.CollectNoTensorVars(sub_scope_);
  }
  tensor_array_batch_cleaner_.ResetNoTensorVars();

  // recover the cpu_math_library_num_threads to 1, in order to avoid thread
  // conflict when integrating it into deployment service.
  paddle::platform::SetNumThreads(1);
#ifdef PADDLE_WITH_MKLDNN
  if (config_.use_mkldnn_) MkldnnPostReset();
#endif
#if defined(PADDLE_WITH_MKLML)
  // Frees unused memory allocated by the Intel® MKL Memory Allocator to
  // avoid memory leak. See:
  // https://software.intel.com/en-us/mkl-developer-reference-c-mkl-free-buffers
  platform::dynload::MKL_Free_Buffers();
#endif
  return true;
}

bool AnalysisPredictor::SetFeed(const std::vector<PaddleTensor> &inputs,
                                framework::Scope *scope) {
  VLOG(3) << "Predictor::set_feed";
  if (inputs.size() != feeds_.size()) {
    LOG(ERROR) << "wrong feed input size, need " << feeds_.size() << " but get "
               << inputs.size();
    return false;
  }

  // Cache the inputs memory for better concurrency performance.
  feed_tensors_.resize(inputs.size());

  for (size_t i = 0; i < inputs.size(); ++i) {
    framework::LoDTensor *input = &feed_tensors_[i];
    if (!PaddleTensorToLoDTensor(inputs[i], input, place_)) {
      return false;
    }
    int idx = -1;
    if (config_.specify_input_name_) {
      auto name = inputs[i].name;
      if (feed_names_.find(name) == feed_names_.end()) {
        LOG(ERROR) << "feed names from program do not have name: [" << name
                   << "] from specified input";
      }
      idx = feed_names_[name];
    } else {
      idx = BOOST_GET_CONST(int, feeds_[i]->GetAttr("col"));
    }
    framework::SetFeedVariable(scope, *input, "feed", idx);
  }
  return true;
}

template <typename T>
void AnalysisPredictor::GetFetchOne(const framework::LoDTensor &fetch,
                                    PaddleTensor *output) {
  // set shape.
  auto shape = framework::vectorize(fetch.dims());
  output->shape.assign(shape.begin(), shape.end());
  // set data.
  const T *data = fetch.data<T>();
  int num_elems = inference::VecReduceToInt(shape);
  output->data.Resize(num_elems * sizeof(T));
  // The fetched tensor output by fetch op, should always in CPU memory, so just
  // copy.
  memcpy(output->data.data(), data, num_elems * sizeof(T));
  // set lod
  output->lod.clear();
  for (auto &level : fetch.lod()) {
    output->lod.emplace_back(level.begin(), level.end());
  }
}

bool AnalysisPredictor::GetFetch(std::vector<PaddleTensor> *outputs,
                                 framework::Scope *scope) {
  VLOG(3) << "Predictor::get_fetch";
  outputs->resize(fetches_.size());
  for (size_t i = 0; i < fetches_.size(); ++i) {
    int idx = BOOST_GET_CONST(int, fetches_[i]->GetAttr("col"));
    PADDLE_ENFORCE_EQ(
        static_cast<size_t>(idx), i,
        platform::errors::InvalidArgument(
            "Fetch op's col attr(%d) should be equal to the index(%d)", idx,
            i));
    framework::FetchType &fetch_var =
        framework::GetFetchVariable(*scope, "fetch", idx);
    auto &fetch = BOOST_GET(framework::LoDTensor, fetch_var);
    auto type = fetch.type();
    auto output = &(outputs->at(i));
    output->name = fetches_[idx]->Input("X")[0];
    if (type == framework::proto::VarType::FP32) {
      GetFetchOne<float>(fetch, output);
      output->dtype = PaddleDType::FLOAT32;
    } else if (type == framework::proto::VarType::INT64) {
      GetFetchOne<int64_t>(fetch, output);
      output->dtype = PaddleDType::INT64;
    } else if (type == framework::proto::VarType::INT32) {
      GetFetchOne<int32_t>(fetch, output);
      output->dtype = PaddleDType::INT32;
    } else {
      LOG(ERROR) << "unknown type, only support float32, int64 and int32 now.";
    }
  }
  return true;
}

void AnalysisPredictor::PrepareArgument() {
  argument_.SetUseGPU(config_.use_gpu());
  argument_.SetUseFcPadding(config_.use_fc_padding());
  argument_.SetGPUDeviceId(config_.gpu_device_id());
  argument_.SetEnableAnalysisOptim(config_.enable_ir_optim_);
  argument_.SetEnableMemoryOptim(config_.enable_memory_optim());
  argument_.SetModelFromMemory(config_.model_from_memory_);
  // Analyze inference_program
  argument_.SetPredictorID(predictor_id_);
  argument_.SetOptimCacheDir(config_.opt_cache_dir_);
  if (!config_.model_dir().empty()) {
    argument_.SetModelDir(config_.model_dir());
  } else {
    PADDLE_ENFORCE_EQ(config_.params_file().empty(), false,
                      platform::errors::PreconditionNotMet(
                          "Either model_dir or param_file should be set."));
    PADDLE_ENFORCE_EQ(config_.prog_file().empty(), false,
                      platform::errors::PreconditionNotMet(
                          "Either model_dir or prog_file should be set."));
    std::string dir = inference::analysis::GetDirRoot(config_.prog_file());

    argument_.SetModelProgramPath(config_.prog_file());
    argument_.SetModelParamsPath(config_.params_file());
  }

  if (config_.use_gpu() && config_.tensorrt_engine_enabled()) {
    LOG(INFO) << "TensorRT subgraph engine is enabled";
    argument_.SetUseTensorRT(true);
    argument_.SetTensorRtWorkspaceSize(config_.tensorrt_workspace_size_);
    argument_.SetTensorRtMaxBatchSize(config_.tensorrt_max_batchsize_);
    argument_.SetTensorRtMinSubgraphSize(config_.tensorrt_min_subgraph_size_);
    argument_.SetTensorRtDisabledOPs(config_.trt_disabled_ops_);
    argument_.SetTensorRtUseDLA(config_.trt_use_dla_);
    argument_.SetTensorRtDLACore(config_.trt_dla_core_);
    argument_.SetTensorRtPrecisionMode(config_.tensorrt_precision_mode_);
    argument_.SetTensorRtUseStaticEngine(config_.trt_use_static_engine_);
    argument_.SetTensorRtUseCalibMode(config_.trt_use_calib_mode_);
    argument_.SetTensorRtUseOSS(config_.trt_use_oss_);
    argument_.SetMinInputShape(config_.min_input_shape_);
    argument_.SetMaxInputShape(config_.max_input_shape_);
    argument_.SetOptimInputShape(config_.optim_input_shape_);
    argument_.SetCloseTrtPluginFp16(config_.disable_trt_plugin_fp16_);
  }

  if (config_.dlnne_enabled()) {
    LOG(INFO) << "Dlnne subgraph is enabled";
    argument_.SetUseDlnne(true);
    argument_.SetDlnneMinSubgraphSize(config_.dlnne_min_subgraph_size_);
  }

  if (config_.lite_engine_enabled()) {
    argument_.SetCpuMathLibraryNumThreads(
        config_.cpu_math_library_num_threads());
    argument_.SetLitePrecisionMode(config_.lite_precision_mode_);
    argument_.SetLitePassesFilter(config_.lite_passes_filter_);
    argument_.SetLiteOpsFilter(config_.lite_ops_filter_);
    argument_.SetLiteZeroCopy(config_.lite_zero_copy_);
    argument_.SetUseXpu(config_.use_xpu_);
    argument_.SetXpuL3WorkspaceSize(config_.xpu_l3_workspace_size_);
    argument_.SetXpuLocked(config_.xpu_locked_);
    argument_.SetXpuAutotune(config_.xpu_autotune_);
    argument_.SetXpuAutotuneFile(config_.xpu_autotune_file_);
    argument_.SetXpuPrecision(config_.xpu_precision_);
    argument_.SetXpuAdaptiveSeqlen(config_.xpu_adaptive_seqlen_);
    LOG(INFO) << "Lite subgraph engine is enabled";
  }

  if (config_.use_mkldnn_) {
    LOG(INFO) << "MKLDNN is enabled";
    argument_.SetMKLDNNEnabledOpTypes(config_.mkldnn_enabled_op_types_);
  }

#ifdef PADDLE_WITH_MKLDNN
  if (config_.mkldnn_quantizer_enabled()) {
    LOG(INFO) << "Quantization is enabled";
    argument_.SetQuantizeEnabledOpTypes(
        config_.mkldnn_quantizer_config()->enabled_op_types());
    argument_.SetQuantizeExcludedOpIds(
        config_.mkldnn_quantizer_config()->excluded_op_ids());
  }
  if (config_.use_mkldnn_bfloat16_) {
    LOG(INFO) << "Bfloat16 is enabled";
    argument_.SetBfloat16EnabledOpTypes(config_.bfloat16_enabled_op_types_);
  }
#endif

  auto passes = config_.pass_builder()->AllPasses();
  if (!config_.ir_optim()) {
    passes.clear();
    LOG(INFO) << "ir_optim is turned off, no IR pass will be executed";
  }
  argument_.SetDisableLogs(config_.glog_info_disabled());
  argument_.SetIrAnalysisPasses(passes);
  argument_.SetAnalysisPasses(config_.pass_builder()->AnalysisPasses());
  argument_.SetScopeNotOwned(scope_.get());
}

// NOTE All the members in AnalysisConfig should be copied to Argument.
void AnalysisPredictor::OptimizeInferenceProgram() {
  PrepareArgument();
  Analyzer().Run(&argument_);

  PADDLE_ENFORCE_EQ(
      argument_.scope_valid(), true,
      platform::errors::InvalidArgument("The argument scope should be valid."));
  VLOG(5) << "to prepare executor";
  ARGUMENT_CHECK_FIELD((&argument_), ir_analyzed_program);
  inference_program_ =
      std::make_shared<framework::ProgramDesc>(argument_.ir_analyzed_program());
  // The config and argument take a lot of storage,
  // when the predictor settings are complete, we release these stores.
  argument_.PartiallyRelease();
  config_.PartiallyRelease();
  LOG(INFO) << "======= optimize end =======";
}

template <>
std::unique_ptr<PaddlePredictor>
CreatePaddlePredictor<AnalysisConfig, PaddleEngineKind::kAnalysis>(
    const AnalysisConfig &config) {
  // TODO(NHZlX): Should add the link to the doc of
  // paddle_infer::CreatePredictor<paddle_infer::Config>
  if (config.glog_info_disabled()) {
    FLAGS_logtostderr = true;
    FLAGS_minloglevel = 2;  // GLOG_ERROR
  }
  VLOG(3) << "create AnalysisConfig";
  PADDLE_ENFORCE_EQ(
      config.is_valid(), true,
      platform::errors::InvalidArgument(
          "Note: Each config can only be used for one predictor."));

  // Register custom operators compiled by the user.
  // This function can only be executed once per process.
  static std::once_flag custom_operators_registered;
  std::call_once(custom_operators_registered,
                 []() { inference::RegisterAllCustomOperator(); });

  if (config.use_gpu()) {
    static std::once_flag gflags_initialized;
    static bool process_level_allocator_enabled;

    std::call_once(gflags_initialized, [&]() {
      std::vector<std::string> gflags;
      PADDLE_ENFORCE_GE(
          config.memory_pool_init_size_mb(), 0.f,
          platform::errors::InvalidArgument(
              "The size of memory pool should be greater than 0."));
      PADDLE_ENFORCE_GE(
          config.gpu_device_id(), 0,
          platform::errors::InvalidArgument(
              "Invalid device id (%d). The device id should be greater than 0.",
              config.gpu_device_id()));
      gflags.emplace_back("dummy");

      float fraction_of_gpu_memory = config.fraction_of_gpu_memory_for_pool();
      if (fraction_of_gpu_memory > 0.95f) {
        LOG(ERROR)
            << "Allocate too much memory for the GPU memory pool, assigned "
            << config.memory_pool_init_size_mb() << " MB";
        LOG(ERROR) << "Try to shink the value by setting "
                      "AnalysisConfig::EnableGpu(...)";
      }

      if (fraction_of_gpu_memory >= 0.0f || fraction_of_gpu_memory <= 0.95f) {
        std::string flag = "--fraction_of_gpu_memory_to_use=" +
                           std::to_string(fraction_of_gpu_memory);
        VLOG(3) << "set flag: " << flag;
        gflags.push_back(flag);
        gflags.emplace_back("--cudnn_deterministic=True");
      }

// TODO(wilber): jetson tx2 may fail to run the model due to insufficient memory
// under the native_best_fit strategy. Modify the default allocation strategy to
// auto_growth. todo, find a more appropriate way to solve the problem.
#ifdef WITH_NV_JETSON
      gflags.push_back("--allocator_strategy=auto_growth");
#endif

      // TODO(Shixiaowei02): Add a mandatory scheme to use the thread local
      // allocator when multi-stream is enabled.
      if (config.thread_local_stream_enabled()) {
        gflags.emplace_back("--allocator_strategy=thread_local");
        process_level_allocator_enabled = false;
      } else {
        process_level_allocator_enabled = true;
      }

      if (framework::InitGflags(gflags)) {
        VLOG(3) << "The following gpu analysis configurations only take effect "
                   "for the first predictor: ";
        for (size_t i = 1; i < gflags.size(); ++i) {
          VLOG(3) << gflags[i];
        }
      } else {
        LOG(WARNING) << "The one-time configuration of analysis predictor "
                        "failed, which may be due to native predictor called "
                        "first and its configurations taken effect.";
      }
    });

    if (config.thread_local_stream_enabled() &&
        process_level_allocator_enabled) {
      PADDLE_THROW(platform::errors::Fatal(
          "When binding threads and streams, the use of "
          "process-level allocators will result in undefined result "
          "errors due to memory asynchronous operations."
          "The thread and stream binding configuration of all "
          "predictors should be the same in a single process."));
    }
  }

  std::unique_ptr<PaddlePredictor> predictor(new AnalysisPredictor(config));
  // Each config can only be used for one predictor.
  config.SetInValid();
  auto predictor_p = dynamic_cast<AnalysisPredictor *>(predictor.get());

  if (!predictor_p->Init(nullptr)) {
    return nullptr;
  }

  if (config.mkldnn_quantizer_enabled() && !predictor_p->MkldnnQuantize()) {
    return nullptr;
  }

  return predictor;
}

bool AnalysisPredictor::MkldnnQuantize() {
#if PADDLE_WITH_MKLDNN
  if (!mkldnn_quantizer_)
    mkldnn_quantizer_ = new AnalysisPredictor::MkldnnQuantizer(
        *this, config_.mkldnn_quantizer_config());
  return mkldnn_quantizer_->Quantize();
#else
  LOG(ERROR) << "Please compile with MKLDNN first to use MkldnnQuantizer";
  return false;
#endif
}

void AnalysisPredictor::PrepareFeedFetch() {
  PADDLE_ENFORCE_NOT_NULL(sub_scope_,
                          platform::errors::InvalidArgument(
                              "The sub_scope should not be nullptr."));
  CreateFeedFetchVar(sub_scope_);
  for (auto *op : inference_program_->Block(0).AllOps()) {
    if (op->Type() == "feed") {
      int idx = BOOST_GET_CONST(int, op->GetAttr("col"));
      if (feeds_.size() <= static_cast<size_t>(idx)) {
        feeds_.resize(idx + 1);
      }
      feeds_[idx] = op;
      feed_names_[op->Output("Out")[0]] = idx;
      idx2feeds_[idx] = op->Output("Out")[0];
    } else if (op->Type() == "fetch") {
      int idx = BOOST_GET_CONST(int, op->GetAttr("col"));
      if (fetches_.size() <= static_cast<size_t>(idx)) {
        fetches_.resize(idx + 1);
      }
      fetches_[idx] = op;
      idx2fetches_[idx] = op->Input("X")[0];
    }
  }
}

void AnalysisPredictor::CreateFeedFetchVar(framework::Scope *scope) {
  PADDLE_ENFORCE_NOT_NULL(scope, platform::errors::InvalidArgument(
                                     "The scope should not be nullptr."));
  auto *var = scope->Var("feed");
  var->GetMutable<framework::FeedList>();
  var = scope->Var("fetch");
  var->GetMutable<framework::FetchList>();
}

std::vector<std::string> AnalysisPredictor::GetInputNames() {
  std::vector<std::string> input_names;
  for (auto &item : idx2feeds_) {
    input_names.push_back(item.second);
  }
  return input_names;
}

std::map<std::string, std::vector<int64_t>>
AnalysisPredictor::GetInputTensorShape() {
  std::map<std::string, std::vector<int64_t>> input_shapes;
  std::vector<std::string> names = GetInputNames();
  for (std::string name : names) {
    auto *var = inference_program_->Block(0).FindVar(name);
    PADDLE_ENFORCE_NOT_NULL(var, platform::errors::PreconditionNotMet(
                                     "Input %s does not exist.", name));
    input_shapes[name] = var->GetShape();
  }
  return input_shapes;
}

std::vector<std::string> AnalysisPredictor::GetOutputNames() {
  std::vector<std::string> output_names;
  for (auto &item : idx2fetches_) {
    output_names.push_back(item.second);
  }
  return output_names;
}

std::unique_ptr<ZeroCopyTensor> AnalysisPredictor::GetInputTensor(
    const std::string &name) {
  PADDLE_ENFORCE_NOT_NULL(
      executor_->scope()->FindVar(name),
      platform::errors::PreconditionNotMet(
          "The variable named %s is not found in the scope of the exector.",
          name));
  std::unique_ptr<ZeroCopyTensor> res(
      new ZeroCopyTensor(static_cast<void *>(executor_->scope())));
  res->input_or_output_ = true;
  res->SetName(name);
  if (platform::is_cpu_place(place_)) {
    res->SetPlace(PaddlePlace::kCPU);
  } else if (platform::is_xpu_place(place_)) {
    if (config_.lite_engine_enabled()) {
      // Currently, Paddle-Lite's XPU user interface only supports the transfer
      // of host data pointers. If it is currently used as a subgraph, execution
      // efficiency will be sacrificed, so it is temporarily set to cpu place.
      // And, the current lite engine of xpu must execute all parts of the
      // model.
      res->SetPlace(PaddlePlace::kCPU);
    } else {
      auto xpu_place = BOOST_GET_CONST(platform::XPUPlace, place_);
      res->SetPlace(PaddlePlace::kXPU, xpu_place.GetDeviceId());
    }
  } else {
    auto gpu_place = BOOST_GET_CONST(platform::CUDAPlace, place_);
    res->SetPlace(PaddlePlace::kGPU, gpu_place.GetDeviceId());
  }
  return res;
}

std::unique_ptr<ZeroCopyTensor> AnalysisPredictor::GetOutputTensor(
    const std::string &name) {
  PADDLE_ENFORCE_NOT_NULL(
      executor_->scope()->FindVar(name),
      platform::errors::PreconditionNotMet(
          "he variable named %s is not found in the scope of the exector.",
          name));
  std::unique_ptr<ZeroCopyTensor> res(
      new ZeroCopyTensor(static_cast<void *>(executor_->scope())));
  res->input_or_output_ = false;
  res->SetName(name);
  if (platform::is_cpu_place(place_)) {
    res->SetPlace(PaddlePlace::kCPU);
  } else if (platform::is_xpu_place(place_)) {
    if (config_.lite_engine_enabled()) {
      // Currently, Paddle-Lite's XPU user interface only supports the transfer
      // of host data pointers. If it is currently used as a subgraph, execution
      // efficiency will be sacrificed, so it is temporarily set to cpu place.
      // And, the current lite engine of xpu must execute all parts of the
      // model.
      res->SetPlace(PaddlePlace::kCPU);
    } else {
      auto xpu_place = BOOST_GET_CONST(platform::XPUPlace, place_);
      res->SetPlace(PaddlePlace::kXPU, xpu_place.GetDeviceId());
    }
  } else {
    auto gpu_place = BOOST_GET_CONST(platform::CUDAPlace, place_);
    res->SetPlace(PaddlePlace::kGPU, gpu_place.GetDeviceId());
  }
  return res;
}

bool AnalysisPredictor::ZeroCopyRun() {
  paddle::platform::SetNumThreads(config_.cpu_math_library_num_threads());
#ifdef PADDLE_WITH_MKLDNN
  if (config_.use_mkldnn_) {
    std::vector<std::vector<int>> shape_vector;
    auto names = GetInputNames();
    for (size_t i = 0; i < names.size(); ++i) {
      auto in_tensor = GetInputTensor(names[i]);
      shape_vector.emplace_back(in_tensor->shape());
    }
    MkldnnPreSet(shape_vector);
  }
#endif

  executor_->Run();
  // Fix TensorArray reuse not cleaned bug.
  tensor_array_batch_cleaner_.CollectTensorArrays(sub_scope_);
  tensor_array_batch_cleaner_.ResetTensorArray();

  // recover the cpu_math_library_num_threads to 1, in order to avoid thread
  // conflict when integrating it into deployment service.
  paddle::platform::SetNumThreads(1);
#ifdef PADDLE_WITH_MKLDNN
  if (config_.use_mkldnn_) MkldnnPostReset();
#endif
#if defined(PADDLE_WITH_MKLML)
  // Frees unused memory allocated by the Intel® MKL Memory Allocator to
  // avoid memory leak. See:
  // https://software.intel.com/en-us/mkl-developer-reference-c-mkl-free-buffers
  platform::dynload::MKL_Free_Buffers();
#endif
  return true;
}

bool AnalysisPredictor::LoadProgramDesc() {
  // Initialize the inference program
  std::string filename;
  if (!config_.model_dir().empty()) {
    filename = config_.model_dir() + "/__model__";
  } else if (!config_.prog_file().empty() && !config_.params_file().empty()) {
    // All parameters are saved in a single file.
    // The file names should be consistent with that used
    // in Python API `fluid.io.save_inference_model`.
    filename = config_.prog_file();
  } else {
    if (config_.model_dir().empty() && config_.prog_file().empty()) {
      LOG(ERROR)
          << "Either model_dir or (prog_file, param_file) should be set.";
      return false;
    }
    LOG(ERROR) << string::Sprintf(
        "not valid model path '%s' or program path '%s'.", config_.model_dir(),
        config_.params_file());
    return false;
  }

  // Create ProgramDesc
  framework::proto::ProgramDesc proto;
  if (!config_.model_from_memory()) {
    std::string pb_content;
    // Read binary
    std::ifstream fin(filename, std::ios::in | std::ios::binary);
    PADDLE_ENFORCE_EQ(
        static_cast<bool>(fin.is_open()), true,
        platform::errors::NotFound(
            "Cannot open file %s, please confirm whether the file is normal.",
            filename));
    fin.seekg(0, std::ios::end);
    pb_content.resize(fin.tellg());
    fin.seekg(0, std::ios::beg);
    fin.read(&(pb_content.at(0)), pb_content.size());
    fin.close();

    proto.ParseFromString(pb_content);
  } else {
    proto.ParseFromString(config_.prog_file());
  }
  inference_program_ = std::make_shared<framework::ProgramDesc>(proto);
  return true;
}

bool AnalysisPredictor::LoadParameters() {
  PADDLE_ENFORCE_NOT_NULL(inference_program_.get(),
                          platform::errors::PreconditionNotMet(
                              "The inference program should be loaded first."));

  const auto &global_block = inference_program_->MutableBlock(0);

  // create a temporary program to load parameters.

  std::unique_ptr<framework::ProgramDesc> load_program(
      new framework::ProgramDesc());
  framework::BlockDesc *load_block = load_program->MutableBlock(0);
  std::vector<std::string> params;

  for (auto *var : global_block->AllVars()) {
    if (IsPersistable(var)) {
      VLOG(3) << "persistable variable's name: " << var->Name();

      framework::VarDesc *new_var = load_block->Var(var->Name());
      new_var->SetShape(var->GetShape());
      new_var->SetDataType(var->GetDataType());
      new_var->SetType(var->GetType());
      new_var->SetLoDLevel(var->GetLoDLevel());
      new_var->SetPersistable(true);

      if (!config_.params_file().empty()) {
        params.push_back(new_var->Name());
      } else {
        // append_op
        framework::OpDesc *op = load_block->AppendOp();
        op->SetType("load");
        op->SetOutput("Out", {new_var->Name()});
        op->SetAttr("file_path", {config_.model_dir() + "/" + new_var->Name()});
        op->CheckAttrs();
      }
    }
  }

  if (!config_.params_file().empty()) {
    // sort paramlist to have consistent ordering
    std::sort(params.begin(), params.end());
    // append just the load_combine op
    framework::OpDesc *op = load_block->AppendOp();
    op->SetType("load_combine");
    op->SetOutput("Out", params);
    op->SetAttr("file_path", {config_.params_file()});
    op->CheckAttrs();
  }

  // Use NaiveExecutor to Load parameters.
  framework::NaiveExecutor e(place_);
  e.Prepare(scope_.get(), *load_program, 0, false);
  e.Run();
  VLOG(3) << "get " << scope_->LocalVarNames().size() << " vars after load";

  return true;
}

uint64_t AnalysisPredictor::TryShrinkMemory() {
  ClearIntermediateTensor();
  return paddle::memory::Release(place_);
}

void AnalysisPredictor::ClearIntermediateTensor() {
  PADDLE_ENFORCE_NOT_NULL(inference_program_.get(),
                          platform::errors::PreconditionNotMet(
                              "The inference program should be loaded first."));
  const auto &global_block = inference_program_->MutableBlock(0);
  for (auto *var : global_block->AllVars()) {
    if (!IsPersistable(var)) {
      const std::string name = var->Name();
      auto *variable = executor_->scope()->FindVar(name);
      if (variable != nullptr && variable->IsType<framework::LoDTensor>() &&
          name != "feed" && name != "fetch") {
        VLOG(3) << "Clear Intermediate Tensor: " << name;
        auto *t = variable->GetMutable<framework::LoDTensor>();
        t->clear();
      }
    }
  }
}

#if PADDLE_WITH_TENSORRT
bool AnalysisPredictor::SaveTrtCalibToDisk() {
  PADDLE_ENFORCE_EQ(config_.tensorrt_engine_enabled(), true,
                    platform::errors::PreconditionNotMet(
                        "This func can be invoked only in trt mode"));
  auto &block = inference_program_->Block(0);
  for (auto &op_desc : block.AllOps()) {
    if (op_desc->Type() == "tensorrt_engine") {
      std::string engine_name = BOOST_GET_CONST(
          std::string, op_desc->GetAttr("calibration_engine_key"));
      if (!Singleton<TRTCalibratorEngineManager>::Global().Has(engine_name)) {
        LOG(ERROR) << "You should run the predictor(with trt) on the real data "
                      "to generate calibration info";
        return false;
      }
      TRTCalibratorEngine *calib_engine =
          Singleton<TRTCalibratorEngineManager>::Global().Get(engine_name);
      LOG(INFO) << "Wait for calib threads done.";
      calib_engine->calib_->waitAndSetDone();
      LOG(INFO) << "Generating TRT Calibration table data, this may cost a lot "
                   "of time...";
      calib_engine->thr_->join();
      std::string calibration_table_data =
          calib_engine->calib_->getCalibrationTableAsString();

      if (calibration_table_data.empty()) {
        LOG(ERROR) << "the calibration table is empty.";
        return false;
      }

      std::string model_opt_cache_dir =
          argument_.Has("model_dir")
              ? argument_.model_dir()
              : inference::analysis::GetDirRoot(argument_.model_program_path());

      std::string calibration_table_data_path =
          inference::analysis::GetTrtCalibPath(
              inference::analysis::GetOrCreateModelOptCacheDir(
                  model_opt_cache_dir),
              engine_name);

      std::ofstream ofile(calibration_table_data_path, std::ios::out);
      LOG(INFO) << "Write Paddle-TRT INT8 calibration table data to file "
                << calibration_table_data_path;
      ofile << calibration_table_data;
      ofile.close();
    }
  }
  // Free all calibrator resources.
  Singleton<TRTCalibratorEngineManager>::Global().DeleteALL();
  return true;
}
#endif

AnalysisPredictor::~AnalysisPredictor() {
#if PADDLE_WITH_TENSORRT
  if (config_.tensorrt_engine_enabled() &&
      config_.tensorrt_precision_mode_ == AnalysisConfig::Precision::kInt8 &&
      Singleton<TRTCalibratorEngineManager>::Global().Has()) {
    SaveTrtCalibToDisk();
  }
#endif
  if (config_.with_profile_) {
    platform::DisableProfiler(platform::EventSortingKey::kTotal,
                              "./profile.log");
  }
  if (sub_scope_) {
    scope_->DeleteScope(sub_scope_);
  }

#if PADDLE_WITH_MKLDNN
  if (mkldnn_quantizer_) {
    delete mkldnn_quantizer_;
    mkldnn_quantizer_ = nullptr;
  }
#endif

  memory::Release(place_);
}

std::unique_ptr<PaddlePredictor> AnalysisPredictor::Clone() {
  std::lock_guard<std::mutex> lk(clone_mutex_);
  auto *x = new AnalysisPredictor(config_);
  x->Init(scope_, inference_program_);
  return std::unique_ptr<PaddlePredictor>(x);
}

std::string AnalysisPredictor::GetSerializedProgram() const {
  return inference_program_->Proto()->SerializeAsString();
}

// Add SaveOptimModel
void AnalysisPredictor::SaveOptimModel(const std::string &dir) {
  // save model
  std::string model_name = dir + "/model";
  std::ofstream outfile;
  outfile.open(model_name, std::ios::out | std::ios::binary);
  std::string inference_prog_desc = GetSerializedProgram();
  outfile << inference_prog_desc;
  // save params
  framework::ProgramDesc save_program;
  auto *save_block = save_program.MutableBlock(0);

  const framework::ProgramDesc &main_program = program();
  const framework::BlockDesc &global_block = main_program.Block(0);
  std::vector<std::string> save_var_list;
  for (framework::VarDesc *var : global_block.AllVars()) {
    if (IsPersistable(var)) {
      framework::VarDesc *new_var = save_block->Var(var->Name());
      new_var->SetShape(var->GetShape());
      new_var->SetDataType(var->GetDataType());
      new_var->SetType(var->GetType());
      new_var->SetLoDLevel(var->GetLoDLevel());
      new_var->SetPersistable(true);

      save_var_list.push_back(new_var->Name());
    }
  }
  std::sort(save_var_list.begin(), save_var_list.end());
  auto *op = save_block->AppendOp();
  op->SetType("save_combine");
  op->SetInput("X", save_var_list);
  op->SetAttr("file_path", dir + "/params");
  op->CheckAttrs();

  platform::CPUPlace place;
  framework::Executor exe(place);
  exe.Run(save_program, scope(), 0, true, true);
}

template <>
std::unique_ptr<PaddlePredictor> CreatePaddlePredictor<AnalysisConfig>(
    const AnalysisConfig &config) {
  LOG(WARNING) << "Deprecated. Please use CreatePredictor instead.";
  return CreatePaddlePredictor<AnalysisConfig, PaddleEngineKind::kAnalysis>(
      config);
}

}  // namespace paddle

#if PADDLE_WITH_TENSORRT
USE_TRT_CONVERTER(elementwise_add_weight);
USE_TRT_CONVERTER(elementwise_add_tensor);
USE_TRT_CONVERTER(elementwise_sub_tensor);
USE_TRT_CONVERTER(elementwise_div_tensor);
USE_TRT_CONVERTER(elementwise_mul_tensor);
USE_TRT_CONVERTER(elementwise_max_tensor);
USE_TRT_CONVERTER(elementwise_min_tensor);
USE_TRT_CONVERTER(elementwise_pow_tensor);
USE_TRT_CONVERTER(transpose);
USE_TRT_CONVERTER(flatten);
USE_TRT_CONVERTER(matmul);
USE_TRT_CONVERTER(conv2d);
USE_TRT_CONVERTER(relu);
USE_TRT_CONVERTER(sigmoid);
USE_TRT_CONVERTER(tanh);
USE_TRT_CONVERTER(fc);
USE_TRT_CONVERTER(pool2d);
USE_TRT_CONVERTER(softmax);
USE_TRT_CONVERTER(batch_norm);
USE_TRT_CONVERTER(concat);
USE_TRT_CONVERTER(dropout);
USE_TRT_CONVERTER(pad);
USE_TRT_CONVERTER(hard_sigmoid);
USE_TRT_CONVERTER(hard_swish);
USE_TRT_CONVERTER(split);
USE_TRT_CONVERTER(prelu);
USE_TRT_CONVERTER(conv2d_transpose);
USE_TRT_CONVERTER(leaky_relu);
USE_TRT_CONVERTER(shuffle_channel);
USE_TRT_CONVERTER(swish);
USE_TRT_CONVERTER(group_norm);
USE_TRT_CONVERTER(instance_norm);
USE_TRT_CONVERTER(layer_norm);
USE_TRT_CONVERTER(gelu);
USE_TRT_CONVERTER(multihead_matmul);
USE_TRT_CONVERTER(fused_embedding_eltwise_layernorm);
USE_TRT_CONVERTER(skip_layernorm);
USE_TRT_CONVERTER(slice);
USE_TRT_CONVERTER(scale);
USE_TRT_CONVERTER(stack);
USE_TRT_CONVERTER(clip);
USE_TRT_CONVERTER(gather);
USE_TRT_CONVERTER(anchor_generator);
USE_TRT_CONVERTER(yolo_box);
USE_TRT_CONVERTER(roi_align);
USE_TRT_CONVERTER(affine_channel);
USE_TRT_CONVERTER(multiclass_nms);
USE_TRT_CONVERTER(nearest_interp);
USE_TRT_CONVERTER(reshape);
#endif

namespace paddle_infer {

Predictor::Predictor(const Config &config) {
  const_cast<Config *>(&config)->SwitchUseFeedFetchOps(false);
  // The second parameter indicates that the discard log is not printed
  predictor_ = paddle::CreatePaddlePredictor<
      Config, paddle::PaddleEngineKind::kAnalysis>(config);
}

std::vector<std::string> Predictor::GetInputNames() {
  return predictor_->GetInputNames();
}

std::unique_ptr<Tensor> Predictor::GetInputHandle(const std::string &name) {
  return predictor_->GetInputTensor(name);
}

std::vector<std::string> Predictor::GetOutputNames() {
  return predictor_->GetOutputNames();
}

std::unique_ptr<Tensor> Predictor::GetOutputHandle(const std::string &name) {
  return predictor_->GetOutputTensor(name);
}

bool Predictor::Run() { return predictor_->ZeroCopyRun(); }

std::unique_ptr<Predictor> Predictor::Clone() {
  auto analysis_pred = predictor_->Clone();
  std::unique_ptr<Predictor> pred(new Predictor(std::move(analysis_pred)));
  return pred;
}

void Predictor::ClearIntermediateTensor() {
  predictor_->ClearIntermediateTensor();
}

uint64_t Predictor::TryShrinkMemory() { return predictor_->TryShrinkMemory(); }

int GetNumBytesOfDataType(DataType dtype) {
  switch (dtype) {
    case DataType::FLOAT32:
      return sizeof(float);
    case DataType::INT64:
      return sizeof(int64_t);
    case DataType::INT32:
      return sizeof(int32_t);
    case DataType::UINT8:
      return sizeof(uint8_t);
    default:
      assert(false);
      return -1;
  }
}

std::string GetVersion() { return paddle::get_version(); }

std::string UpdateDllFlag(const char *name, const char *value) {
  return paddle::UpdateDllFlag(name, value);
}

}  // namespace paddle_infer

namespace paddle_infer {
std::shared_ptr<Predictor> CreatePredictor(const Config &config) {  // NOLINT
  std::shared_ptr<Predictor> predictor(new Predictor(config));
  return predictor;
}

namespace services {
PredictorPool::PredictorPool(const Config &config, size_t size) {
  PADDLE_ENFORCE_GE(
      size, 1UL,
      paddle::platform::errors::InvalidArgument(
          "The predictor pool size should be greater than 1, but it's (%d)",
          size));
  Config copy_config(config);
  main_pred_ = std::make_shared<Predictor>(config);
  for (size_t i = 0; i < size - 1; i++) {
    if (config.tensorrt_engine_enabled()) {
      Config config_tmp(copy_config);
      preds_.push_back(std::move(std::make_unique<Predictor>(config_tmp)));
    } else {
      preds_.push_back(std::move(main_pred_->Clone()));
    }
  }
}

Predictor *PredictorPool::Retrive(size_t idx) {
  PADDLE_ENFORCE_LT(
      idx, preds_.size() + 1,
      paddle::platform::errors::InvalidArgument(
          "There are (%d) predictors in the pool, but the idx is (%d)", idx,
          preds_.size() + 1));
  if (idx == 0) {
    return main_pred_.get();
  }
  return preds_[idx - 1].get();
}
}  // namespace services
}  // namespace paddle_infer
