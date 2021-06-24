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

///
/// \file pd_config.h
///
/// \brief interface for paddle config
///
/// \author paddle-infer@baidu.com
/// \date 2021-04-21
/// \since 2.1
///

#pragma once

#include "pd_common.h"  // NOLINT
#include "pd_types.h"   // NOLINT

typedef struct PD_Config PD_Config;

#ifdef __cplusplus
extern "C" {
#endif

///
/// \brief Create a paddle config
///
/// \return new config.
///
PADDLE_CAPI_EXPORT extern __pd_give PD_Config* PD_ConfigCreate();
///
/// \brief Destroy the paddle config
///
/// \param[in] pd_onfig config
///
PADDLE_CAPI_EXPORT extern void PD_ConfigDestroy(__pd_take PD_Config* pd_config);
///
/// \brief Set the combined model with two specific pathes for program and
/// parameters.
///
/// \param[in] pd_onfig config
/// \param[in] prog_file_path model file path of the combined model.
/// \param[in] params_file_path params file path of the combined model.
///
PADDLE_CAPI_EXPORT extern void PD_ConfigSetModel(__pd_keep PD_Config* pd_config,
                                                 const char* prog_file_path,
                                                 const char* params_file_path);
///
/// \brief Set the model file path of a combined model.
///
/// \param[in] pd_onfig config
/// \param[in] prog_file_path model file path.
///
PADDLE_CAPI_EXPORT extern void PD_ConfigSetProgFile(
    __pd_keep PD_Config* pd_config, const char* prog_file_path);
///
/// \brief Set the params file path of a combined model.
///
/// \param[in] pd_onfig config
/// \param[in] params_file_path params file path.
///
PADDLE_CAPI_EXPORT extern void PD_ConfigSetParamsFile(
    __pd_keep PD_Config* pd_config, const char* params_file_path);
///
/// \brief Set the path of optimization cache directory.
/// \param[in] pd_onfig config
/// \param[in] opt_cache_dir the path of optimization cache directory.
///
PADDLE_CAPI_EXPORT extern void PD_ConfigSetOptimCacheDir(
    __pd_keep PD_Config* pd_config, const char* opt_cache_dir);
///
/// \brief Set the no-combined model dir path.
/// \param[in] pd_onfig config
/// \param[in] model_dir model dir path.
///
PADDLE_CAPI_EXPORT extern void PD_ConfigSetModelDir(
    __pd_keep PD_Config* pd_config, const char* model_dir);
///
/// \brief Get the model directory path.
///
/// \param[in] pd_onfig config
/// \return The model directory path.
///
PADDLE_CAPI_EXPORT extern const char* PD_ConfigGetModelDir(
    __pd_keep PD_Config* pd_config);
///
/// \brief Get the program file path.
///
/// \param[in] pd_onfig config
/// \return The program file path.
///
PADDLE_CAPI_EXPORT extern const char* PD_ConfigGetProgFile(
    __pd_keep PD_Config* pd_config);
///
/// \brief Get the params file path.
///
/// \param[in] pd_onfig config
/// \return The params file path.
///
PADDLE_CAPI_EXPORT extern const char* PD_ConfigGetParamsFile(
    __pd_keep PD_Config* pd_config);
///
/// \brief Turn off FC Padding.
///
/// \param[in] pd_onfig config
///
PADDLE_CAPI_EXPORT extern void PD_ConfigDisableFCPadding(
    __pd_keep PD_Config* pd_config);
///
/// \brief A boolean state telling whether fc padding is used.
///
/// \param[in] pd_onfig config
/// \return Whether fc padding is used.
///
PADDLE_CAPI_EXPORT extern PD_Bool PD_ConfigUseFcPadding(
    __pd_keep PD_Config* pd_config);
///
/// \brief Turn on GPU.
///
/// \param[in] pd_onfig config
/// \param[in] memory_pool_init_size_mb initial size of the GPU memory pool in
/// MB.
/// \param[in] device_id device_id the GPU card to use.
///
PADDLE_CAPI_EXPORT extern void PD_ConfigEnableUseGpu(
    __pd_keep PD_Config* pd_config, uint64_t memory_pool_init_size_mb,
    int32_t device_id);
///
/// \brief Turn off GPU.
///
/// \param[in] pd_onfig config
///
PADDLE_CAPI_EXPORT extern void PD_ConfigDisableGpu(
    __pd_keep PD_Config* pd_config);
///
/// \brief A boolean state telling whether the GPU is turned on.
///
/// \brief Turn off GPU.
/// \return Whether the GPU is turned on.
///
PADDLE_CAPI_EXPORT extern PD_Bool PD_ConfigUseGpu(
    __pd_keep PD_Config* pd_config);
///
/// \brief Turn on XPU.
///
/// \param[in] pd_onfig config
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
PADDLE_CAPI_EXPORT extern void PD_ConfigEnableXpu(
    __pd_keep PD_Config* pd_config, int32_t l3_workspace_size, PD_Bool locked,
    PD_Bool autotune, const char* autotune_file, const char* precision,
    PD_Bool adaptive_seqlen);
///
/// \brief A boolean state telling whether the XPU is turned on.
///
/// \param[in] pd_onfig config
/// \return Whether the XPU is turned on.
///
PADDLE_CAPI_EXPORT extern PD_Bool PD_ConfigUseXpu(
    __pd_keep PD_Config* pd_config);
///
/// \brief Get the GPU device id.
///
/// \param[in] pd_onfig config
/// \return The GPU device id.
///
PADDLE_CAPI_EXPORT extern int32_t PD_ConfigGpuDeviceId(
    __pd_keep PD_Config* pd_config);
///
/// \brief Get the XPU device id.
///
/// \param[in] pd_onfig config
/// \return The XPU device id.
///
PADDLE_CAPI_EXPORT extern int32_t PD_ConfigXpuDeviceId(
    __pd_keep PD_Config* pd_config);
///
/// \brief Get the initial size in MB of the GPU memory pool.
///
/// \param[in] pd_onfig config
/// \return The initial size in MB of the GPU memory pool.
///
PADDLE_CAPI_EXPORT extern int32_t PD_ConfigMemoryPoolInitSizeMb(
    __pd_keep PD_Config* pd_config);
///
/// \brief Get the proportion of the initial memory pool size compared to the
/// device.
///
/// \param[in] pd_onfig config
/// \return The proportion of the initial memory pool size.
///
PADDLE_CAPI_EXPORT extern float PD_ConfigFractionOfGpuMemoryForPool(
    __pd_keep PD_Config* pd_config);
///
/// \brief Turn on CUDNN.
///
/// \param[in] pd_onfig config
///
PADDLE_CAPI_EXPORT extern void PD_ConfigEnableCudnn(
    __pd_keep PD_Config* pd_config);
///
/// \brief A boolean state telling whether to use CUDNN.
///
/// \param[in] pd_onfig config
/// \return Whether to use CUDNN.
///
PADDLE_CAPI_EXPORT extern PD_Bool PD_ConfigCudnnEnabled(
    __pd_keep PD_Config* pd_config);
///
/// \brief Control whether to perform IR graph optimization.
/// If turned off, the AnalysisConfig will act just like a NativeConfig.
///
/// \param[in] pd_onfig config
/// \param[in] x Whether the ir graph optimization is actived.
///
PADDLE_CAPI_EXPORT extern void PD_ConfigSwitchIrOptim(
    __pd_keep PD_Config* pd_config, PD_Bool x);
///
/// \brief A boolean state telling whether the ir graph optimization is
/// actived.
///
/// \param[in] pd_onfig config
/// \return Whether to use ir graph optimization.
///
PADDLE_CAPI_EXPORT extern PD_Bool PD_ConfigIrOptim(
    __pd_keep PD_Config* pd_config);
///
/// \brief Turn on the TensorRT engine.
/// The TensorRT engine will accelerate some subgraphes in the original Fluid
/// computation graph. In some models such as resnet50, GoogleNet and so on,
/// it gains significant performance acceleration.
///
/// \param[in] pd_onfig config
/// \param[in] workspace_size The memory size(in byte) used for TensorRT
/// workspace.
/// \param[in] max_batch_size The maximum batch size of this prediction task,
/// better set as small as possible for less performance loss.
/// \param[in] min_subgrpah_size The minimum TensorRT subgraph size needed, if a
/// subgraph is smaller than this, it will not be transferred to TensorRT
/// engine.
/// \param[in] precision The precision used in TensorRT.
/// \param[in] use_static Serialize optimization information to disk for
/// reusing.
/// \param[in] use_calib_mode Use TRT int8 calibration(post training
/// quantization).
///
PADDLE_CAPI_EXPORT extern void PD_ConfigEnableTensorRtEngine(
    __pd_keep PD_Config* pd_config, int32_t workspace_size,
    int32_t max_batch_size, int32_t min_subgraph_size,
    PD_PrecisionType precision, PD_Bool use_static, PD_Bool use_calib_mode);
///
/// \brief A boolean state telling whether the TensorRT engine is used.
///
/// \param[in] pd_onfig config
/// \return Whether the TensorRT engine is used.
///
PADDLE_CAPI_EXPORT extern PD_Bool PD_ConfigTensorRtEngineEnabled(
    __pd_keep PD_Config* pd_config);
///
/// \brief Set min, max, opt shape for TensorRT Dynamic shape mode.
///
/// \param[in] pd_onfig config
/// \param[in] tensor_num The number of the subgraph input.
/// \param[in] tensor_name The name of every subgraph input.
/// \param[in] shapes_num The shape size of every subgraph input.
/// \param[in] min_shape The min input shape of every subgraph input.
/// \param[in] max_shape The max input shape of every subgraph input.
/// \param[in] optim_shape The opt input shape of every subgraph input.
/// \param[in] disable_trt_plugin_fp16 Setting this parameter to true means that
/// TRT plugin will not run fp16.
///
PADDLE_CAPI_EXPORT extern void PD_ConfigSetTrtDynamicShapeInfo(
    __pd_keep PD_Config* pd_config, size_t tensor_num, const char** tensor_name,
    size_t* shapes_num, int32_t** min_shape, int32_t** max_shape,
    int32_t** optim_shape, PD_Bool disable_trt_plugin_fp16);
///
/// \brief Prevent ops running in Paddle-TRT
/// NOTE: just experimental, not an official stable API, easy to be broken.
///
/// \param[in] pd_onfig config
/// \param[in] ops_num ops number
/// \param[in] ops_name ops name
///
PADDLE_CAPI_EXPORT extern void PD_ConfigDisableTensorRtOPs(
    __pd_keep PD_Config* pd_config, size_t ops_num, const char** ops_name);
///
/// \brief Replace some TensorRT plugins to TensorRT OSS(
/// https://github.com/NVIDIA/TensorRT), with which some models's inference
/// may be more high-performance. Libnvinfer_plugin.so greater than
/// V7.2.1 is needed.
///
/// \param[in] pd_onfig config
///
PADDLE_CAPI_EXPORT extern void PD_ConfigEnableTensorRtOSS(
    __pd_keep PD_Config* pd_config);
///
/// \brief A boolean state telling whether to use the TensorRT OSS.
///
/// \param[in] pd_onfig config
/// \return Whether to use the TensorRT OSS.
///
PADDLE_CAPI_EXPORT extern PD_Bool PD_ConfigTensorRtOssEnabled(
    __pd_keep PD_Config* pd_config);
///
/// \brief Enable TensorRT DLA
///
/// \param[in] pd_onfig config
/// \param[in] dla_core ID of DLACore, which should be 0, 1,
///        ..., IBuilder.getNbDLACores() - 1
///
PADDLE_CAPI_EXPORT extern void PD_ConfigEnableTensorRtDla(
    __pd_keep PD_Config* pd_config, int32_t dla_core);
///
/// \brief A boolean state telling whether to use the TensorRT DLA.
///
/// \param[in] pd_onfig config
/// \return Whether to use the TensorRT DLA.
///
PADDLE_CAPI_EXPORT extern PD_Bool PD_ConfigTensorRtDlaEnabled(
    __pd_keep PD_Config* pd_config);
///
/// \brief Turn on the usage of Lite sub-graph engine.
///
/// \param[in] pd_onfig config
/// \param[in] precision Precion used in Lite sub-graph engine.
/// \param[in] zero_copy whether use zero copy.
/// \param[in] passes_filter_num The number of passes used in Lite sub-graph
/// engine.
/// \param[in] passes_filter The name of passes used in Lite sub-graph engine.
/// \param[in] ops_filter_num The number of operators not supported by Lite.
/// \param[in] ops_filter The name of operators not supported by Lite.
///
PADDLE_CAPI_EXPORT extern void PD_ConfigEnableLiteEngine(
    __pd_keep PD_Config* pd_config, PD_PrecisionType precision,
    PD_Bool zero_copy, size_t passes_filter_num, const char** passes_filter,
    size_t ops_filter_num, const char** ops_filter);
///
/// \brief A boolean state indicating whether the Lite sub-graph engine is
/// used.
///
/// \param[in] pd_onfig config
/// \return Whether the Lite sub-graph engine is used.
///
PADDLE_CAPI_EXPORT extern PD_Bool PD_ConfigLiteEngineEnabled(
    __pd_keep PD_Config* pd_config);
///
/// \brief Control whether to debug IR graph analysis phase.
/// This will generate DOT files for visualizing the computation graph after
/// each analysis pass applied.
///
/// \param[in] pd_onfig config
/// \param[in] x whether to debug IR graph analysis phase.
///
PADDLE_CAPI_EXPORT extern void PD_ConfigSwitchIrDebug(
    __pd_keep PD_Config* pd_config, PD_Bool x);
///
/// \brief Turn on MKLDNN.
///
/// \param[in] pd_onfig config
///
PADDLE_CAPI_EXPORT extern void PD_ConfigEnableMKLDNN(
    __pd_keep PD_Config* pd_config);
///
/// \brief Set the cache capacity of different input shapes for MKLDNN.
/// Default value 0 means not caching any shape.
/// Please see MKL-DNN Data Caching Design Document:
/// https://github.com/PaddlePaddle/FluidDoc/blob/develop/doc/fluid/design/mkldnn/caching/caching.md
///
/// \param[in] pd_onfig config
/// \param[in] capacity The cache capacity.
///
PADDLE_CAPI_EXPORT extern void PD_ConfigSetMkldnnCacheCapacity(
    __pd_keep PD_Config* pd_config, int32_t capacity);
///
/// \brief A boolean state telling whether to use the MKLDNN.
///
/// \param[in] pd_onfig config
/// \return Whether to use the MKLDNN.
///
PADDLE_CAPI_EXPORT extern PD_Bool PD_ConfigMkldnnEnabled(
    __pd_keep PD_Config* pd_config);
///
/// \brief Set the number of cpu math library threads.
///
/// \param[in] pd_onfig config
/// \param cpu_math_library_num_threads The number of cpu math library
/// threads.
///
PADDLE_CAPI_EXPORT extern void PD_ConfigSetCpuMathLibraryNumThreads(
    __pd_keep PD_Config* pd_config, int32_t cpu_math_library_num_threads);
///
/// \brief An int state telling how many threads are used in the CPU math
/// library.
///
/// \param[in] pd_onfig config
/// \return The number of threads used in the CPU math library.
///
PADDLE_CAPI_EXPORT extern int32_t PD_ConfigGetCpuMathLibraryNumThreads(
    __pd_keep PD_Config* pd_config);
///
/// \brief Specify the operator type list to use MKLDNN acceleration.
///
/// \param[in] pd_onfig config
/// \param[in] ops_num The number of operator type list.
/// \param[in] op_list The name of operator type list.
///
PADDLE_CAPI_EXPORT extern void PD_ConfigSetMkldnnOp(
    __pd_keep PD_Config* pd_config, size_t ops_num, const char** op_list);
///
/// \brief Turn on MKLDNN quantization.
///
/// \param[in] pd_onfig config
///
PADDLE_CAPI_EXPORT extern void PD_ConfigEnableMkldnnQuantizer(
    __pd_keep PD_Config* pd_config);
///
/// \brief A boolean state telling whether the MKLDNN quantization is enabled.
///
/// \param[in] pd_onfig config
/// \return Whether the MKLDNN quantization is enabled.
///
PADDLE_CAPI_EXPORT extern PD_Bool PD_ConfigMkldnnQuantizerEnabled(
    __pd_keep PD_Config* pd_config);
///
/// \brief Turn on MKLDNN bfloat16.
///
/// \param[in] pd_onfig config
///
PADDLE_CAPI_EXPORT extern void PD_ConfigEnableMkldnnBfloat16(
    __pd_keep PD_Config* pd_config);
///
/// \brief A boolean state telling whether to use the MKLDNN Bfloat16.
///
/// \param[in] pd_onfig config
/// \return Whether to use the MKLDNN Bfloat16.
///
PADDLE_CAPI_EXPORT extern PD_Bool PD_ConfigMkldnnBfloat16Enabled(
    __pd_keep PD_Config* pd_config);
/// \brief Specify the operator type list to use Bfloat16 acceleration.
///
/// \param[in] pd_onfig config
/// \param[in] ops_num The number of operator type list.
/// \param[in] op_list The name of operator type list.
///
PADDLE_CAPI_EXPORT extern void PD_ConfigSetBfloat16Op(
    __pd_keep PD_Config* pd_config, size_t ops_num, const char** op_list);
///
/// \brief Enable the GPU multi-computing stream feature.
/// NOTE: The current behavior of this interface is to bind the computation
/// stream to the thread, and this behavior may be changed in the future.
///
/// \param[in] pd_onfig config
///
PADDLE_CAPI_EXPORT extern void PD_ConfigEnableGpuMultiStream(
    __pd_keep PD_Config* pd_config);
///
/// \brief A boolean state telling whether the thread local CUDA stream is
/// enabled.
///
/// \param[in] pd_onfig config
/// \return Whether the thread local CUDA stream is enabled.
///
PADDLE_CAPI_EXPORT extern PD_Bool PD_ConfigThreadLocalStreamEnabled(
    __pd_keep PD_Config* pd_config);
///
/// \brief Specify the memory buffer of program and parameter.
/// Used when model and params are loaded directly from memory.
///
/// \param[in] pd_onfig config
/// \param[in] prog_buffer The memory buffer of program.
/// \param[in] prog_buffer_size The size of the model data.
/// \param[in] params_buffer The memory buffer of the combined parameters file.
/// \param[in] params_buffer_size The size of the combined parameters data.
///
PADDLE_CAPI_EXPORT extern void PD_ConfigSetModelBuffer(
    __pd_keep PD_Config* pd_config, const char* prog_buffer,
    size_t prog_buffer_size, const char* params_buffer,
    size_t params_buffer_size);
///
/// \brief A boolean state telling whether the model is set from the CPU
/// memory.
///
/// \param[in] pd_onfig config
/// \return Whether model and params are loaded directly from memory.
///
PADDLE_CAPI_EXPORT extern PD_Bool PD_ConfigModelFromMemory(
    __pd_keep PD_Config* pd_config);
///
/// \brief Turn on memory optimize
/// NOTE still in development.
///
/// \param[in] pd_onfig config
///
PADDLE_CAPI_EXPORT extern void PD_ConfigEnableMemoryOptim(
    __pd_keep PD_Config* pd_config);
///
/// \brief A boolean state telling whether the memory optimization is
/// activated.
///
/// \param[in] pd_onfig config
/// \return Whether the memory optimization is activated.
///
PADDLE_CAPI_EXPORT extern PD_Bool PD_ConfigMemoryOptimEnabled(
    __pd_keep PD_Config* pd_config);
///
/// \brief Turn on profiling report.
/// If not turned on, no profiling report will be generated.
///
/// \param[in] pd_onfig config
///
PADDLE_CAPI_EXPORT extern void PD_ConfigEnableProfile(
    __pd_keep PD_Config* pd_config);
///
/// \brief A boolean state telling whether the profiler is activated.
///
/// \param[in] pd_onfig config
/// \return bool Whether the profiler is activated.
///
PADDLE_CAPI_EXPORT extern PD_Bool PD_ConfigProfileEnabled(
    __pd_keep PD_Config* pd_config);
///
/// \brief Mute all logs in Paddle inference.
///
/// \param[in] pd_onfig config
///
PADDLE_CAPI_EXPORT extern void PD_ConfigDisableGlogInfo(
    __pd_keep PD_Config* pd_config);
///
/// \brief A boolean state telling whether logs in Paddle inference are muted.
///
/// \param[in] pd_onfig config
/// \return Whether logs in Paddle inference are muted.
///
PADDLE_CAPI_EXPORT extern PD_Bool PD_ConfigGlogInfoDisabled(
    __pd_keep PD_Config* pd_config);
///
/// \brief Set the Config to be invalid.
/// This is to ensure that an Config can only be used in one
/// Predictor.
///
/// \param[in] pd_onfig config
///
PADDLE_CAPI_EXPORT extern void PD_ConfigSetInvalid(
    __pd_keep PD_Config* pd_config);
///
/// \brief A boolean state telling whether the Config is valid.
///
/// \param[in] pd_onfig config
/// \return Whether the Config is valid.
///
PADDLE_CAPI_EXPORT extern PD_Bool PD_ConfigIsValid(
    __pd_keep PD_Config* pd_config);
///
/// \brief Partially release the memory
///
/// \param[in] pd_onfig config
///
PADDLE_CAPI_EXPORT extern void PD_ConfigPartiallyRelease(
    __pd_keep PD_Config* pd_config);
///
/// \brief Delete all passes that has a certain type 'pass'.
///
/// \param[in] pass the certain pass type to be deleted.
///
PADDLE_CAPI_EXPORT extern void PD_ConfigDeletePass(
    __pd_keep PD_Config* pd_config, const char* pass);
///
/// \brief  Insert a pass to a specific position
///
/// \param[in] idx the position to insert.
/// \param[in] pass the new pass.
///
PADDLE_CAPI_EXPORT extern void PD_ConfigInsertPass(
    __pd_keep PD_Config* pd_config, size_t idx, const char* pass);
///
/// \brief Append a pass to the end of the passes
///
/// \param[in] pass the new pass.
///
PADDLE_CAPI_EXPORT extern void PD_ConfigAppendPass(
    __pd_keep PD_Config* pd_config, const char* pass);
///
/// \brief Get information of passes.
///
/// \return Return list of the passes.
///
PADDLE_CAPI_EXPORT extern __pd_give PD_OneDimArrayCstr* PD_ConfigAllPasses(
    __pd_keep PD_Config* pd_config);

#ifdef __cplusplus
}  // extern "C"
#endif
