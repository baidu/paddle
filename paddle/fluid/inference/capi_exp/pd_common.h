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

#pragma once

#include <stdint.h>
#include <stdio.h>

#if defined(_WIN32)
#ifdef PADDLE_DLL_INFERENCE
#define PADDLE_CAPI_EXPORT __declspec(dllexport)
#else
#define PADDLE_CAPI_EXPORT __declspec(dllimport)
#endif  // PADDLE_DLL_INFERENCE
#else
#define PADDLE_CAPI_EXPORT __attribute__((visibility("default")))
#endif  // _WIN32

///
/// __pd_give means that a new object is returned. The user should make sure
/// that the returned pointer is used exactly once as a value for an __pd_take
/// argument. In between, it can be used as a value for as many __pd_keep
/// arguments as the user likes.
///
#ifndef __pd_give
#define __pd_give
#endif
///
/// __pd_take means that the object the argument points to is taken over by the
/// function and may no longer be used by the user as an argument to any other
/// function. The pointer value must be one returned by a function returning an
/// __pd_give pointer.
///
#ifndef __pd_take
#define __pd_take
#endif
///
/// __pd_keep means that the function will only use the object temporarily. The
/// object which the argument points to is not taken over by the function. After
/// the function has finished, the user can still use it as an argument to other
/// functions.
///
#ifndef __pd_keep
#define __pd_keep
#endif

typedef int8_t PD_Bool;
#define TRUE 1
#define FALSE 0

#define PD_ENUM(type)   \
  typedef int32_t type; \
  enum

PD_ENUM(PD_PrecisionType){PD_PRECISION_FLOAT32 = 0, PD_PRECISION_INT8,
                          PD_PRECISION_HALF};

PD_ENUM(PD_PlaceType){PD_PLACE_UNK = -1, PD_PLACE_CPU, PD_PLACE_GPU,
                      PD_PLACE_XPU};

PD_ENUM(PD_DataType){
    PD_DATA_UNK = -1, PD_DATA_FLOAT32, PD_DATA_INT32,
    PD_DATA_INT64,    PD_DATA_UINT8,   PD_DATA_INT8,
};
