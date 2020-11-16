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

#ifdef PADDLE_WITH_CUDA
#ifdef PADDLE_CUDA_KERNEL_CHECK
#include "paddle/fluid/platform/cuda_enforce.cuh"

typedef struct _CudaKerenlErrorPro {
  char _CudaErrorMsg[__LEN_ERROR_MSG];
  int _line;
  char _file[__LEN_ERROR_MSG];
} CudaKerenlErrorPro;

__device__ char _CudaKernelErrorOccurred = 0;
__device__ CudaKerenlErrorPro _CudaKernelErrorMsg;

__device__ __host__ void _strcpy(char* src, char* target) {
  int i = 0;
  while (0 != src[i] && i < __LEN_ERROR_MSG - 1) {
    target[i] = src[i];
    i++;
  }
  target[i] = 0;
}

__host__ __device__ void send_error_msg(char* msg, char* file, int line) {
  if (0 == _CudaKernelErrorOccurred) {
    _strcpy(msg, _CudaKernelErrorMsg._CudaErrorMsg);
    _strcpy(file, _CudaKernelErrorMsg._file);
    _CudaKernelErrorMsg._line = line;
    _CudaKernelErrorOccurred = 127;
  }
}

char ifCudaFail() {
  char occur[1];
  PADDLE_ENFORCE_CUDA_SUCCESS(
      cudaMemcpyFromSymbol(occur, _CudaKernelErrorOccurred, sizeof(char)));
  return occur[0];
}

void get_msg_from_cuda(char* msg, char* file, int* line) {
  CudaKerenlErrorPro temp;
  PADDLE_ENFORCE_CUDA_SUCCESS(
      cudaMemcpyFromSymbol(&temp, (_CudaKernelErrorMsg), sizeof(temp)));
  snprintf(msg, __LEN_ERROR_MSG, temp._CudaErrorMsg);
  snprintf(file, __LEN_ERROR_MSG, temp._file);
  *line = temp._line;
}

#endif  // PADDLE_CUDA_KERNEL_CHECK
#endif  // PADDLE_WITH_CUDA
