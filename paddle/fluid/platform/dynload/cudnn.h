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

#pragma once
#include <cudnn.h>
#include <glog/logging.h>
#include <mutex>  // NOLINT

#include "paddle/fluid/platform/dynload/dynamic_loader.h"
#include "paddle/fluid/platform/port.h"

namespace paddle {
namespace platform {
namespace dynload {

extern std::once_flag cudnn_dso_flag;
extern void* cudnn_dso_handle;
extern bool HasCUDNN();

extern void EnforceCUDNNLoaded(const char* fn_name);
#define DECLARE_DYNAMIC_LOAD_CUDNN_WRAP(__name)                            \
  struct DynLoad__##__name {                                               \
    template <typename... Args>                                            \
    auto operator()(Args... args) -> DECLARE_TYPE(__name, args...) {       \
      using cudnn_func = decltype(&::__name);                              \
      std::call_once(cudnn_dso_flag, []() {                                \
        cudnn_dso_handle = paddle::platform::dynload::GetCUDNNDsoHandle(); \
      });                                                                  \
      EnforceCUDNNLoaded(#__name);                                         \
      static void* p_##__name = dlsym(cudnn_dso_handle, #__name);          \
      return reinterpret_cast<cudnn_func>(p_##__name)(args...);            \
    }                                                                      \
  };                                                                       \
  extern struct DynLoad__##__name __name

/**
 * include all needed cudnn functions in HPPL
 * different cudnn version has different interfaces
 **/
#define CUDNN_DNN_ROUTINE_EACH(__macro)                   \
  __macro(cudnnSetTensor4dDescriptor);                    \
  __macro(cudnnSetTensor4dDescriptorEx);                  \
  __macro(cudnnSetTensorNdDescriptor);                    \
  __macro(cudnnGetTensorNdDescriptor);                    \
  __macro(cudnnGetConvolutionNdForwardOutputDim);         \
  __macro(cudnnCreateTensorDescriptor);                   \
  __macro(cudnnDestroyTensorDescriptor);                  \
  __macro(cudnnCreateFilterDescriptor);                   \
  __macro(cudnnSetFilter4dDescriptor);                    \
  __macro(cudnnSetFilterNdDescriptor);                    \
  __macro(cudnnGetFilterNdDescriptor);                    \
  __macro(cudnnSetPooling2dDescriptor);                   \
  __macro(cudnnSetPoolingNdDescriptor);                   \
  __macro(cudnnGetPoolingNdDescriptor);                   \
  __macro(cudnnDestroyFilterDescriptor);                  \
  __macro(cudnnCreateConvolutionDescriptor);              \
  __macro(cudnnCreatePoolingDescriptor);                  \
  __macro(cudnnDestroyPoolingDescriptor);                 \
  __macro(cudnnSetConvolution2dDescriptor);               \
  __macro(cudnnDestroyConvolutionDescriptor);             \
  __macro(cudnnSetConvolutionNdDescriptor);               \
  __macro(cudnnGetConvolutionNdDescriptor);               \
  __macro(cudnnDeriveBNTensorDescriptor);                 \
  __macro(cudnnCreateSpatialTransformerDescriptor);       \
  __macro(cudnnSetSpatialTransformerNdDescriptor);        \
  __macro(cudnnDestroySpatialTransformerDescriptor);      \
  __macro(cudnnSpatialTfGridGeneratorForward);            \
  __macro(cudnnSpatialTfGridGeneratorBackward);           \
  __macro(cudnnSpatialTfSamplerForward);                  \
  __macro(cudnnSpatialTfSamplerBackward);                 \
  __macro(cudnnCreate);                                   \
  __macro(cudnnDestroy);                                  \
  __macro(cudnnSetStream);                                \
  __macro(cudnnActivationForward);                        \
  __macro(cudnnActivationBackward);                       \
  __macro(cudnnConvolutionForward);                       \
  __macro(cudnnConvolutionBackwardBias);                  \
  __macro(cudnnGetConvolutionForwardWorkspaceSize);       \
  __macro(cudnnTransformTensor);                          \
  __macro(cudnnPoolingForward);                           \
  __macro(cudnnPoolingBackward);                          \
  __macro(cudnnSoftmaxBackward);                          \
  __macro(cudnnSoftmaxForward);                           \
  __macro(cudnnGetVersion);                               \
  __macro(cudnnFindConvolutionForwardAlgorithmEx);        \
  __macro(cudnnFindConvolutionBackwardFilterAlgorithmEx); \
  __macro(cudnnFindConvolutionBackwardFilterAlgorithm);   \
  __macro(cudnnFindConvolutionBackwardDataAlgorithmEx);   \
  __macro(cudnnGetErrorString);                           \
  __macro(cudnnCreateDropoutDescriptor);                  \
  __macro(cudnnDropoutGetStatesSize);                     \
  __macro(cudnnSetDropoutDescriptor);                     \
  __macro(cudnnRestoreDropoutDescriptor);                 \
  __macro(cudnnCreateRNNDescriptor);                      \
  __macro(cudnnGetRNNParamsSize);                         \
  __macro(cudnnGetRNNWorkspaceSize);                      \
  __macro(cudnnGetRNNTrainingReserveSize);                \
  __macro(cudnnRNNForwardTraining);                       \
  __macro(cudnnRNNBackwardData);                          \
  __macro(cudnnRNNBackwardWeights);                       \
  __macro(cudnnRNNForwardInference);                      \
  __macro(cudnnDestroyDropoutDescriptor);                 \
  __macro(cudnnDestroyRNNDescriptor);                     \
  __macro(cudnnSetTensorNdDescriptorEx);

CUDNN_DNN_ROUTINE_EACH(DECLARE_DYNAMIC_LOAD_CUDNN_WRAP)

#define CUDNN_DNN_ROUTINE_EACH_R2(__macro) \
  __macro(cudnnAddTensor);                 \
  __macro(cudnnConvolutionBackwardData);   \
  __macro(cudnnConvolutionBackwardFilter);
CUDNN_DNN_ROUTINE_EACH_R2(DECLARE_DYNAMIC_LOAD_CUDNN_WRAP)

// APIs available after R3:
#if CUDNN_VERSION >= 3000
#define CUDNN_DNN_ROUTINE_EACH_AFTER_R3(__macro)           \
  __macro(cudnnGetConvolutionBackwardFilterWorkspaceSize); \
  __macro(cudnnGetConvolutionBackwardDataWorkspaceSize);
CUDNN_DNN_ROUTINE_EACH_AFTER_R3(DECLARE_DYNAMIC_LOAD_CUDNN_WRAP)
#endif

// APIs available after R3:
#if CUDNN_VERSION >= 3000 && CUDNN_VERSION < 8000
#define CUDNN_DNN_ROUTINE_EACH_AFTER_R3_LESS_R8(__macro) \
  __macro(cudnnGetConvolutionBackwardFilterAlgorithm);   \
  __macro(cudnnGetConvolutionForwardAlgorithm);          \
  __macro(cudnnGetConvolutionBackwardDataAlgorithm);     \
  __macro(cudnnSetRNNDescriptor);
CUDNN_DNN_ROUTINE_EACH_AFTER_R3_LESS_R8(DECLARE_DYNAMIC_LOAD_CUDNN_WRAP)
#endif

// APIs available after R4:
#if CUDNN_VERSION >= 4007
#define CUDNN_DNN_ROUTINE_EACH_AFTER_R4(__macro)    \
  __macro(cudnnBatchNormalizationForwardTraining);  \
  __macro(cudnnBatchNormalizationForwardInference); \
  __macro(cudnnBatchNormalizationBackward);
CUDNN_DNN_ROUTINE_EACH_AFTER_R4(DECLARE_DYNAMIC_LOAD_CUDNN_WRAP)
#endif

// APIs in R5
#if CUDNN_VERSION >= 5000
#define CUDNN_DNN_ROUTINE_EACH_R5(__macro)  \
  __macro(cudnnCreateActivationDescriptor); \
  __macro(cudnnSetActivationDescriptor);    \
  __macro(cudnnGetActivationDescriptor);    \
  __macro(cudnnDestroyActivationDescriptor);
CUDNN_DNN_ROUTINE_EACH_R5(DECLARE_DYNAMIC_LOAD_CUDNN_WRAP)
#endif

// APIs in R6
#if CUDNN_VERSION >= 6000
#define CUDNN_DNN_ROUTINE_EACH_R6(__macro) __macro(cudnnSetRNNDescriptor_v6);
CUDNN_DNN_ROUTINE_EACH_R6(DECLARE_DYNAMIC_LOAD_CUDNN_WRAP)
#endif

#if CUDNN_VERSION >= 7001
#define CUDNN_DNN_ROUTINE_EACH_R7(__macro)                \
  __macro(cudnnSetConvolutionGroupCount);                 \
  __macro(cudnnSetConvolutionMathType);                   \
  __macro(cudnnConvolutionBiasActivationForward);         \
  __macro(cudnnCreateCTCLossDescriptor);                  \
  __macro(cudnnDestroyCTCLossDescriptor);                 \
  __macro(cudnnGetCTCLossDescriptor);                     \
  __macro(cudnnSetCTCLossDescriptor);                     \
  __macro(cudnnGetCTCLossWorkspaceSize);                  \
  __macro(cudnnCTCLoss);                                  \
  __macro(cudnnGetConvolutionBackwardDataAlgorithm_v7);   \
  __macro(cudnnGetConvolutionBackwardFilterAlgorithm_v7); \
  __macro(cudnnGetConvolutionForwardAlgorithm_v7);        \
  __macro(cudnnGetConvolutionBackwardFilterAlgorithmMaxCount);
CUDNN_DNN_ROUTINE_EACH_R7(DECLARE_DYNAMIC_LOAD_CUDNN_WRAP)
#endif

#if CUDNN_VERSION >= 7201
#define CUDNN_DNN_ROUTINE_EACH_AFTER_TWO_R7(__macro) \
  __macro(cudnnCreateRNNDataDescriptor);             \
  __macro(cudnnDestroyRNNDataDescriptor);            \
  __macro(cudnnSetRNNDataDescriptor);                \
  __macro(cudnnSetRNNPaddingMode);                   \
  __macro(cudnnRNNForwardTrainingEx);                \
  __macro(cudnnRNNBackwardDataEx);                   \
  __macro(cudnnRNNBackwardWeightsEx);                \
  __macro(cudnnRNNForwardInferenceEx);
CUDNN_DNN_ROUTINE_EACH_AFTER_TWO_R7(DECLARE_DYNAMIC_LOAD_CUDNN_WRAP)
#endif

#if CUDNN_VERSION >= 7401
#define CUDNN_DNN_ROUTINE_EACH_AFTER_R7(__macro)                     \
  __macro(cudnnGetBatchNormalizationForwardTrainingExWorkspaceSize); \
  __macro(cudnnBatchNormalizationForwardTrainingEx);                 \
  __macro(cudnnGetBatchNormalizationBackwardExWorkspaceSize);        \
  __macro(cudnnBatchNormalizationBackwardEx);                        \
  __macro(cudnnGetBatchNormalizationTrainingExReserveSpaceSize);
CUDNN_DNN_ROUTINE_EACH_AFTER_R7(DECLARE_DYNAMIC_LOAD_CUDNN_WRAP)
#endif

#if CUDNN_VERSION >= 8000
#define CUDNN_DNN_ROUTINE_EACH_R8(__macro) __macro(cudnnSetRNNDescriptor_v8);
CUDNN_DNN_ROUTINE_EACH_R8(DECLARE_DYNAMIC_LOAD_CUDNN_WRAP)
#endif

}  // namespace dynload
}  // namespace platform
}  // namespace paddle
