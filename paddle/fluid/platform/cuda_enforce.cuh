// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

#include"paddle/fluid/platform/enforce.h"
#ifdef PADDLE_WITH_CUDA
#ifdef PADDLE_CUDA_KERNEL_CHECK

#include<cuda.h>
#include<stdio.h>
  
#define __LEN_ERROR_MSG 128

#define PADDLE_ENFORCE_CHECK_CUDA_KERNEL()                          \
    do {                                                            \
      if (UNLIKELY(ifCudaFail())) {                                 \
        char msg[__LEN_ERROR_MSG] = {0};                            \
        int line = 0;                                               \
        char file[__LEN_ERROR_MSG] = {0};                           \
        get_msg_from_cuda(msg, file, &line);                        \
        char err_msg_temp[__LEN_ERROR_MSG+32]={};                   \
        snprintf(err_msg_temp, __LEN_ERROR_MSG,                     \
        "Error:%s", msg);                           \
        throw ::paddle::platform::EnforceNotMet(err_msg_temp,       \
        file, line);                                                \
      }                                                             \
    } while (0)

#define _PADDLE_ENFORCE_CUDA_KERNEL(_IS_NOT_ERROR, __FORMAT)          \
  do {                                                                \
    if (!(_IS_NOT_ERROR)) {                                           \
        send_error_msg(__FORMAT, __FILE__, __LINE__);                 \
        return;                                                       \
    }                                                                 \
  } while (0)

__device__ __host__ void send_error_msg(char*msg, char* file, int line);



  char  ifCudaFail();
  void  get_msg_from_cuda(char* msg, char* file, int*line);


#define PADDLE_ENFORCE_CUDA_KERNEL _PADDLE_ENFORCE_CUDA_KERNEL

#else   // PADDLE_CUDA_KERNEL_CHECK
#define PADDLE_ENFORCE_CUDA_KERNEL PADDLE_ENFORCE
#define PADDLE_ENFORCE_CHECK_CUDA_KERNEL()
#endif   // PADDLE_CUDA_KERNEL_CHECK
#endif   // PADDLE_WITH_CUDA
