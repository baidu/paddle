// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/inference/capi_exp/pd_config.h"
#include "paddle/fluid/inference/api/paddle_inference_api.h"
#include "paddle/fluid/inference/capi_exp/pd_types.h"
#include "paddle/fluid/inference/capi_exp/utils_internal.h"
#include "paddle/fluid/platform/enforce.h"

#define CHECK_NULL_POINTER_PARM(param)                  \
  PADDLE_ENFORCE_NOT_NULL(                              \
      param, paddle::platform::errors::InvalidArgument( \
                 "The pointer of " #param " shouldn't be nullptr"))

#define CHECK_AND_CONVERT_PD_CONFIG                                         \
  PADDLE_ENFORCE_NOT_NULL(                                                  \
      pd_config, paddle::platform::errors::InvalidArgument(                 \
                     "The pointer of paddle config shouldn't be nullptr")); \
  Config* config = reinterpret_cast<Config*>(pd_config)

using paddle_infer::Config;

static Config::Precision ConvertToCxxPrecisionType(PD_PrecisionType precision) {
  switch (precision) {
    case PD_PRECISION_FLOAT32:
      return Config::Precision::kFloat32;
    case PD_PRECISION_INT8:
      return Config::Precision::kInt8;
    case PD_PRECISION_HALF:
      return Config::Precision::kHalf;
    default:
      PADDLE_THROW(paddle::platform::errors::InvalidArgument(
          "Unsupport paddle precision type %d.", precision));
      return Config::Precision::kFloat32;
  }
}

extern "C" {
__pd_give PD_Config* PD_ConfigCreate() {
  return reinterpret_cast<PD_Config*>(new Config());
}

void PD_ConfigDestroy(__pd_take PD_Config* pd_config) {
  CHECK_AND_CONVERT_PD_CONFIG;
  delete reinterpret_cast<Config*>(config);
}

void PD_ConfigSetModel(__pd_keep PD_Config* pd_config,
                       const char* prog_file_path,
                       const char* params_file_path) {
  CHECK_AND_CONVERT_PD_CONFIG;
  CHECK_NULL_POINTER_PARM(prog_file_path);
  CHECK_NULL_POINTER_PARM(params_file_path);
  config->SetModel(prog_file_path, params_file_path);
}
void PD_ConfigSetProgFile(__pd_keep PD_Config* pd_config,
                          const char* prog_file_path) {
  CHECK_AND_CONVERT_PD_CONFIG;
  CHECK_NULL_POINTER_PARM(prog_file_path);
  config->SetProgFile(prog_file_path);
}
void PD_ConfigSetParamsFile(__pd_keep PD_Config* pd_config,
                            const char* params_file_path) {
  CHECK_AND_CONVERT_PD_CONFIG;
  CHECK_NULL_POINTER_PARM(params_file_path);
  config->SetParamsFile(params_file_path);
}
void PD_ConfigSetOptimCacheDir(__pd_keep PD_Config* pd_config,
                               const char* opt_cache_dir) {
  CHECK_AND_CONVERT_PD_CONFIG;
  CHECK_NULL_POINTER_PARM(opt_cache_dir);
  config->SetOptimCacheDir(opt_cache_dir);
}

void PD_ConfigSetModelDir(__pd_keep PD_Config* pd_config,
                          const char* model_dir) {
  CHECK_AND_CONVERT_PD_CONFIG;
  CHECK_NULL_POINTER_PARM(model_dir);
  config->SetModel(model_dir);
}
const char* PD_ConfigGetModelDir(__pd_keep PD_Config* pd_config) {
  CHECK_AND_CONVERT_PD_CONFIG;
  return config->model_dir().c_str();
}
const char* PD_ConfigGetProgFile(__pd_keep PD_Config* pd_config) {
  CHECK_AND_CONVERT_PD_CONFIG;
  return config->prog_file().c_str();
}
const char* PD_ConfigGetParamsFile(__pd_keep PD_Config* pd_config) {
  CHECK_AND_CONVERT_PD_CONFIG;
  return config->params_file().c_str();
}

void PD_ConfigDisableFCPadding(__pd_keep PD_Config* pd_config) {
  CHECK_AND_CONVERT_PD_CONFIG;
  config->DisableFCPadding();
}
PD_Bool PD_ConfigUseFcPadding(__pd_keep PD_Config* pd_config) {
  CHECK_AND_CONVERT_PD_CONFIG;
  return config->use_fc_padding();
}

void PD_ConfigEnableUseGpu(__pd_keep PD_Config* pd_config,
                           uint64_t memory_pool_init_size_mb,
                           int32_t device_id) {
  CHECK_AND_CONVERT_PD_CONFIG;
  config->EnableUseGpu(memory_pool_init_size_mb, device_id);
}
void PD_ConfigDisableGpu(__pd_keep PD_Config* pd_config) {
  CHECK_AND_CONVERT_PD_CONFIG;
  config->DisableGpu();
}
PD_Bool PD_ConfigUseGpu(__pd_keep PD_Config* pd_config) {
  CHECK_AND_CONVERT_PD_CONFIG;
  return config->use_gpu();
}

void PD_ConfigEnableXpu(__pd_keep PD_Config* pd_config,
                        int32_t l3_workspace_size, PD_Bool locked,
                        PD_Bool autotune, const char* autotune_file,
                        const char* precision, PD_Bool adaptive_seqlen) {
  CHECK_AND_CONVERT_PD_CONFIG;
  config->EnableXpu(l3_workspace_size, locked, autotune, autotune_file,
                    precision, adaptive_seqlen);
}

PD_Bool PD_ConfigUseXpu(__pd_keep PD_Config* pd_config) {
  CHECK_AND_CONVERT_PD_CONFIG;
  return config->use_xpu();
}

int32_t PD_ConfigGpuDeviceId(__pd_keep PD_Config* pd_config) {
  CHECK_AND_CONVERT_PD_CONFIG;
  return config->gpu_device_id();
}
int32_t PD_ConfigXpuDeviceId(__pd_keep PD_Config* pd_config) {
  CHECK_AND_CONVERT_PD_CONFIG;
  return config->xpu_device_id();
}
int32_t PD_ConfigMemoryPoolInitSizeMb(__pd_keep PD_Config* pd_config) {
  CHECK_AND_CONVERT_PD_CONFIG;
  return config->memory_pool_init_size_mb();
}
float PD_ConfigFractionOfGpuMemoryForPool(__pd_keep PD_Config* pd_config) {
  CHECK_AND_CONVERT_PD_CONFIG;
  return config->fraction_of_gpu_memory_for_pool();
}
void PD_ConfigEnableCudnn(__pd_keep PD_Config* pd_config) {
  CHECK_AND_CONVERT_PD_CONFIG;
  config->EnableCUDNN();
}
PD_Bool PD_ConfigCudnnEnabled(__pd_keep PD_Config* pd_config) {
  CHECK_AND_CONVERT_PD_CONFIG;
  return config->cudnn_enabled();
}

void PD_ConfigSwitchIrOptim(__pd_keep PD_Config* pd_config, PD_Bool x) {
  CHECK_AND_CONVERT_PD_CONFIG;
  config->SwitchIrOptim(x);
}
PD_Bool PD_ConfigIrOptim(__pd_keep PD_Config* pd_config) {
  CHECK_AND_CONVERT_PD_CONFIG;
  return config->ir_optim();
}

void PD_ConfigEnableTensorRtEngine(__pd_keep PD_Config* pd_config,
                                   int32_t workspace_size,
                                   int32_t max_batch_size,
                                   int32_t min_subgraph_size,
                                   PD_PrecisionType precision,
                                   PD_Bool use_static, PD_Bool use_calib_mode) {
  CHECK_AND_CONVERT_PD_CONFIG;
  config->EnableTensorRtEngine(
      workspace_size, max_batch_size, min_subgraph_size,
      ConvertToCxxPrecisionType(precision), use_static, use_calib_mode);
}
PD_Bool PD_ConfigTensorRtEngineEnabled(__pd_keep PD_Config* pd_config) {
  CHECK_AND_CONVERT_PD_CONFIG;
  return config->tensorrt_engine_enabled();
}

void PD_ConfigSetTrtDynamicShapeInfo(__pd_keep PD_Config* pd_config,
                                     size_t tensor_num,
                                     const char** tensor_name,
                                     size_t* shapes_num, int32_t** min_shape,
                                     int32_t** max_shape, int32_t** optim_shape,
                                     PD_Bool disable_trt_plugin_fp16) {
  CHECK_AND_CONVERT_PD_CONFIG;
  std::map<std::string, std::vector<int>> min_input_shapes;
  std::map<std::string, std::vector<int>> max_input_shapes;
  std::map<std::string, std::vector<int>> optim_input_shapes;
  for (size_t tensor_index = 0; tensor_index < tensor_num; ++tensor_index) {
    std::string name(tensor_name[tensor_index]);
    std::vector<int> min_input_shape, max_input_shape, optim_input_shape;
    for (size_t shape_index = 0; shape_index < shapes_num[tensor_index];
         ++shape_index) {
      min_input_shape.emplace_back(min_shape[tensor_index][shape_index]);
      max_input_shape.emplace_back(max_shape[tensor_index][shape_index]);
      optim_input_shape.emplace_back(optim_shape[tensor_index][shape_index]);
    }
    min_input_shapes[name] = std::move(min_input_shape);
    max_input_shapes[name] = std::move(max_input_shape);
    optim_input_shapes[name] = std::move(optim_input_shape);
  }
  config->SetTRTDynamicShapeInfo(min_input_shapes, max_input_shapes,
                                 optim_input_shapes, disable_trt_plugin_fp16);
}

void PD_ConfigDisableTensorRtOPs(__pd_keep PD_Config* pd_config, size_t ops_num,
                                 const char** ops_name) {
  CHECK_AND_CONVERT_PD_CONFIG;
  std::vector<std::string> ops_list;
  for (size_t index = 0; index < ops_num; ++index) {
    ops_list.emplace_back(ops_name[index]);
  }
  config->Exp_DisableTensorRtOPs(ops_list);
}

void PD_ConfigEnableTensorRtOSS(__pd_keep PD_Config* pd_config) {
  CHECK_AND_CONVERT_PD_CONFIG;
  config->EnableTensorRtOSS();
}
PD_Bool PD_ConfigTensorRtOssEnabled(__pd_keep PD_Config* pd_config) {
  CHECK_AND_CONVERT_PD_CONFIG;
  return config->tensorrt_oss_enabled();
}

void PD_ConfigEnableTensorRtDla(__pd_keep PD_Config* pd_config,
                                int32_t dla_core) {
  CHECK_AND_CONVERT_PD_CONFIG;
  config->EnableTensorRtDLA(dla_core);
}
PD_Bool PD_ConfigTensorRtDlaEnabled(__pd_keep PD_Config* pd_config) {
  CHECK_AND_CONVERT_PD_CONFIG;
  return config->tensorrt_dla_enabled();
}

void PD_ConfigEnableLiteEngine(__pd_keep PD_Config* pd_config,
                               PD_PrecisionType precision, PD_Bool zero_copy,
                               size_t passes_filter_num,
                               const char** passes_filter,
                               size_t ops_filter_num, const char** ops_filter) {
  CHECK_AND_CONVERT_PD_CONFIG;
  std::vector<std::string> passes_filters, ops_filters;
  for (size_t index = 0; index < passes_filter_num; ++index) {
    passes_filters.emplace_back(passes_filter[index]);
  }
  for (size_t index = 0; index < ops_filter_num; ++index) {
    ops_filters.emplace_back(ops_filter[index]);
  }
  config->EnableLiteEngine(ConvertToCxxPrecisionType(precision), zero_copy,
                           passes_filters, ops_filters);
}
PD_Bool PD_ConfigLiteEngineEnabled(__pd_keep PD_Config* pd_config) {
  CHECK_AND_CONVERT_PD_CONFIG;
  return config->lite_engine_enabled();
}

void PD_ConfigSwitchIrDebug(__pd_keep PD_Config* pd_config, PD_Bool x) {
  CHECK_AND_CONVERT_PD_CONFIG;
  config->SwitchIrDebug(x);
}
void PD_ConfigEnableMKLDNN(__pd_keep PD_Config* pd_config) {
  CHECK_AND_CONVERT_PD_CONFIG;
  config->EnableMKLDNN();
}
void PD_ConfigSetMkldnnCacheCapacity(__pd_keep PD_Config* pd_config,
                                     int32_t capacity) {
  CHECK_AND_CONVERT_PD_CONFIG;
  config->SetMkldnnCacheCapacity(capacity);
}
PD_Bool PD_ConfigMkldnnEnabled(__pd_keep PD_Config* pd_config) {
  CHECK_AND_CONVERT_PD_CONFIG;
  return config->mkldnn_enabled();
}
void PD_ConfigSetCpuMathLibraryNumThreads(
    __pd_keep PD_Config* pd_config, int32_t cpu_math_library_num_threads) {
  CHECK_AND_CONVERT_PD_CONFIG;
  config->SetCpuMathLibraryNumThreads(cpu_math_library_num_threads);
}
int32_t PD_ConfigGetCpuMathLibraryNumThreads(__pd_keep PD_Config* pd_config) {
  CHECK_AND_CONVERT_PD_CONFIG;
  return config->cpu_math_library_num_threads();
}

void PD_ConfigSetMkldnnOp(__pd_keep PD_Config* pd_config, size_t ops_num,
                          const char** op_list) {
  CHECK_AND_CONVERT_PD_CONFIG;
  std::unordered_set<std::string> op_names;
  for (size_t index = 0; index < ops_num; ++index) {
    op_names.emplace(op_list[index]);
  }
  config->SetMKLDNNOp(std::move(op_names));
}
void PD_ConfigEnableMkldnnQuantizer(__pd_keep PD_Config* pd_config) {
  CHECK_AND_CONVERT_PD_CONFIG;
  config->EnableMkldnnQuantizer();
}
void PD_ConfigEnableMkldnnBfloat16(__pd_keep PD_Config* pd_config) {
  CHECK_AND_CONVERT_PD_CONFIG;
  config->EnableMkldnnBfloat16();
}
PD_Bool PD_ConfigMkldnnBfloat16Enabled(__pd_keep PD_Config* pd_config) {
  CHECK_AND_CONVERT_PD_CONFIG;
  return config->mkldnn_bfloat16_enabled();
}
void PD_ConfigSetBfloat16Op(__pd_keep PD_Config* pd_config, size_t ops_num,
                            const char** op_list) {
  CHECK_AND_CONVERT_PD_CONFIG;
  std::unordered_set<std::string> op_names;
  for (size_t index = 0; index < ops_num; ++index) {
    op_names.emplace(op_list[index]);
  }
  config->SetBfloat16Op(std::move(op_names));
}
PD_Bool PD_ConfigThreadLocalStreamEnabled(__pd_keep PD_Config* pd_config) {
  CHECK_AND_CONVERT_PD_CONFIG;
  return config->thread_local_stream_enabled();
}
PD_Bool PD_ConfigMkldnnQuantizerEnabled(__pd_keep PD_Config* pd_config) {
  CHECK_AND_CONVERT_PD_CONFIG;
  return config->mkldnn_quantizer_enabled();
}
void PD_ConfigSetModelBuffer(__pd_keep PD_Config* pd_config,
                             const char* prog_buffer, size_t prog_buffer_size,
                             const char* params_buffer,
                             size_t params_buffer_size) {
  CHECK_AND_CONVERT_PD_CONFIG;
  config->SetModelBuffer(prog_buffer, prog_buffer_size, params_buffer,
                         params_buffer_size);
}
PD_Bool PD_ConfigModelFromMemory(__pd_keep PD_Config* pd_config) {
  CHECK_AND_CONVERT_PD_CONFIG;
  return config->model_from_memory();
}
void PD_ConfigEnableMemoryOptim(__pd_keep PD_Config* pd_config) {
  CHECK_AND_CONVERT_PD_CONFIG;
  config->EnableMemoryOptim();
}
PD_Bool PD_ConfigMemoryOptimEnabled(__pd_keep PD_Config* pd_config) {
  CHECK_AND_CONVERT_PD_CONFIG;
  return config->enable_memory_optim();
}
void PD_ConfigEnableProfile(__pd_keep PD_Config* pd_config) {
  CHECK_AND_CONVERT_PD_CONFIG;
  config->EnableProfile();
}
PD_Bool PD_ConfigProfileEnabled(__pd_keep PD_Config* pd_config) {
  CHECK_AND_CONVERT_PD_CONFIG;
  return config->profile_enabled();
}
void PD_ConfigDisableGlogInfo(__pd_keep PD_Config* pd_config) {
  CHECK_AND_CONVERT_PD_CONFIG;
  config->DisableGlogInfo();
}
PD_Bool PD_ConfigGlogInfoDisabled(__pd_keep PD_Config* pd_config) {
  CHECK_AND_CONVERT_PD_CONFIG;
  return config->glog_info_disabled();
}
void PD_ConfigSetInvalid(__pd_keep PD_Config* pd_config) {
  CHECK_AND_CONVERT_PD_CONFIG;
  config->SetInValid();
}
PD_Bool PD_ConfigIsValid(__pd_keep PD_Config* pd_config) {
  CHECK_AND_CONVERT_PD_CONFIG;
  return config->is_valid();
}
void PD_ConfigEnableGpuMultiStream(__pd_keep PD_Config* pd_config) {
  CHECK_AND_CONVERT_PD_CONFIG;
  config->EnableGpuMultiStream();
}
void PD_ConfigPartiallyRelease(__pd_keep PD_Config* pd_config) {
  CHECK_AND_CONVERT_PD_CONFIG;
  config->PartiallyRelease();
}
void PD_ConfigDeletePass(__pd_keep PD_Config* pd_config, const char* pass) {
  CHECK_AND_CONVERT_PD_CONFIG;
  config->pass_builder()->DeletePass(pass);
}
void PD_ConfigInsertPass(__pd_keep PD_Config* pd_config, size_t idx,
                         const char* pass) {
  CHECK_AND_CONVERT_PD_CONFIG;
  config->pass_builder()->InsertPass(idx, pass);
}
void PD_ConfigAppendPass(__pd_keep PD_Config* pd_config, const char* pass) {
  CHECK_AND_CONVERT_PD_CONFIG;
  config->pass_builder()->AppendPass(pass);
}
__pd_give PD_OneDimArrayCstr* PD_ConfigAllPasses(
    __pd_keep PD_Config* pd_config) {
  CHECK_AND_CONVERT_PD_CONFIG;
  std::vector<std::string> passes = config->pass_builder()->AllPasses();
  return paddle_infer::CvtVecToOneDimArrayCstr(passes);
}

}  // extern "C"
