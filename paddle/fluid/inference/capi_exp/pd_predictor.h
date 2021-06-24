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
/// \file pd_predictor.h
///
/// \brief interface for paddle predictor
///
/// \author paddle-infer@baidu.com
/// \date 2021-04-21
/// \since 2.1
///

#pragma once

#include "pd_common.h"  // NOLINT

typedef struct PD_Predictor PD_Predictor;
typedef struct PD_Config PD_Config;
typedef struct PD_Tensor PD_Tensor;
typedef struct PD_OneDimArrayCstr PD_OneDimArrayCstr;

#ifdef __cplusplus
extern "C" {
#endif

///
/// \brief Create a new Predictor
///
/// \param[in] Config config
/// \return new predicor.
///
PADDLE_CAPI_EXPORT extern __pd_give PD_Predictor* PD_PredictorCreate(
    __pd_take PD_Config* pd_config);
///
/// \brief Clone a new Predictor
///
/// \param[in] pd_predictor predictor
/// \return new predictor.
///
PADDLE_CAPI_EXPORT extern __pd_give PD_Predictor* PD_PredictorClone(
    __pd_keep PD_Predictor* pd_predictor);
///
/// \brief Get the input names
///
/// \param[in] pd_predictor predictor
/// \return input names
///
PADDLE_CAPI_EXPORT extern __pd_give PD_OneDimArrayCstr*
PD_PredictorGetInputNames(__pd_keep PD_Predictor* pd_predictor);
///
/// \brief Get the output names
///
/// \param[in] pd_predictor predictor
/// \return output names
///
PADDLE_CAPI_EXPORT extern __pd_give PD_OneDimArrayCstr*
PD_PredictorGetOutputNames(__pd_keep PD_Predictor* pd_predictor);

///
/// \brief Get the input number
///
/// \param[in] pd_predictor predictor
/// \return input number
///
PADDLE_CAPI_EXPORT extern size_t PD_PredictorGetInputNum(
    __pd_keep PD_Predictor* pd_predictor);

///
/// \brief Get the output number
///
/// \param[in] pd_predictor predictor
/// \return output number
///
PADDLE_CAPI_EXPORT extern size_t PD_PredictorGetOutputNum(
    __pd_keep PD_Predictor* pd_predictor);

///
/// \brief Get the Input Tensor object
///
/// \param[in] pd_predictor predictor
/// \param[in] name input name
/// \return input tensor
///
PADDLE_CAPI_EXPORT extern __pd_give PD_Tensor* PD_PredictorGetInputHandle(
    __pd_keep PD_Predictor* pd_predictor, const char* name);

///
/// \brief Get the Output Tensor object
///
/// \param[in] pd_predictor predictor
/// \param[in] name output name
/// \return output tensor
///
PADDLE_CAPI_EXPORT extern __pd_give PD_Tensor* PD_PredictorGetOutputHandle(
    __pd_keep PD_Predictor* pd_predictor, const char* name);

///
/// \brief Run the prediction engine
///
/// \param[in] pd_predictor predictor
/// \return Whether the function executed successfully
///
PADDLE_CAPI_EXPORT extern PD_Bool PD_PredictorRun(
    __pd_keep PD_Predictor* pd_predictor);

/// \brief Clear the intermediate tensors of the predictor
///
/// \param[in] pd_predictor predictor
///
PADDLE_CAPI_EXPORT extern void PD_PredictorClearIntermediateTensor(
    __pd_keep PD_Predictor* pd_predictor);

///
/// \brief Release all tmp tensor to compress the size of the memory pool.
/// The memory pool is considered to be composed of a list of chunks, if
/// the chunk is not occupied, it can be released.
///
/// \param[in] pd_predictor predictor
/// \return Number of bytes released. It may be smaller than the actual
/// released memory, because part of the memory is not managed by the
/// MemoryPool.
///
PADDLE_CAPI_EXPORT extern uint64_t PD_PredictorTryShrinkMemory(
    __pd_keep PD_Predictor* pd_predictor);

///
/// \brief Destroy a predictor object
///
/// \param[in] pd_predictor predictor
///
PADDLE_CAPI_EXPORT extern void PD_PredictorDestroy(
    __pd_take PD_Predictor* pd_predictor);

///
/// \brief Get version info.
///
/// \return version
///
PADDLE_CAPI_EXPORT extern const char* PD_GetVersion();

#ifdef __cplusplus
}  // extern "C"
#endif
