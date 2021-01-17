// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

#include <mutex>  // NOLINT

#if defined PADDLE_WITH_CUDA
#include "paddle/fluid/platform/dynload/cublas.h"
#elif defined PADDLE_WITH_HIP
#include "paddle/fluid/platform/dynload/rocblas.h"
#endif
#include "paddle/fluid/platform/enforce.h"
#include "paddle/fluid/platform/macros.h"

#ifdef PADDLE_WITH_CUDA
#if CUDA_VERSION < 9000
enum cublasMath_t { CUBLAS_DEFAULT_MATH = 0 };
#endif
#endif
namespace paddle {
namespace platform {

/*
 * Summary: Grid stride looping macro in CUDA kernel
 *
 *  [ Why need this macro? ]
 *
 *    The original looping in CUDA kernel is:
 *
 *    `for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); \
 *        i += blockDim.x * gridDim.x)`
 *
 *    This for condition is risky. The value of `blockIdx.x * blockDim.x`
 *    may be large, such as over 1GB, the first iteration is no problem here,
 *    but when `i += blockDim.x * gridDim.x` is executed, the value of i
 *    will greater than INT_MAX and overflow becomes negative value, at
 *    this time, the cycle condition `i < (n)` is still satisfied, so it
 *    will cause illegal access to cuda memory.
 *
 *    Here is a real example in ERINE, it will trigger above error.
 *    The related data are:
 *      - blockIdx.x = 2172938
 *      - blockDim.x = 512
 *      - blockIdx.x * blockDim.x = 1112543864
 *      - INT_MAX = 2147483647
 *
 *    So we polish the for condition as follow, the int64_t __index__ will
 *    prevent overflow in the loop increment.
 *
 * Parameters:
 *    - i: loop index
 *    - num: total element numbers
 *
 * Examples:
 *    template <typename T>
 *    __global__ void Scale(T* logit_grad, const T* loss_grad, const int num,
 *                      const int d, const int remain) {
 *    CUDA_KERNEL_LOOP(index, num) {
 *      int idx_n = index / d;
 *      int idx_remain = index % remain;
 *      logit_grad[index] *= loss_grad[idx_n * remain + idx_remain];
 *      }
 *    }
 *
*/
#define CUDA_KERNEL_LOOP(i, num)                             \
  int64_t __index__ = blockIdx.x * blockDim.x + threadIdx.x; \
  for (int i = __index__; __index__ < (num);                 \
       __index__ += blockDim.x * gridDim.x, i = __index__)

class CublasHandleHolder {
 public:
#if defined PADDLE_WITH_CUDA
  CublasHandleHolder(cudaStream_t stream, cublasMath_t math_type) {
    PADDLE_RETRY_CUDA_SUCCESS(dynload::cublasCreate(&handle_));
    PADDLE_RETRY_CUDA_SUCCESS(dynload::cublasSetStream(handle_, stream));
#if CUDA_VERSION >= 9000
    if (math_type == CUBLAS_TENSOR_OP_MATH) {
      PADDLE_RETRY_CUDA_SUCCESS(
          dynload::cublasSetMathMode(handle_, CUBLAS_TENSOR_OP_MATH));
#if CUDA_VERSION >= 11000
    } else if (math_type == CUBLAS_TF32_TENSOR_OP_MATH) {
      PADDLE_RETRY_CUDA_SUCCESS(
          dynload::cublasSetMathMode(handle_, CUBLAS_TF32_TENSOR_OP_MATH));
#endif  // CUDA_VERSION >= 11000
    }
  }
#endif  // CUDA_VERSION >= 9000
#elif defined PADDLE_WITH_HIP
  explicit CublasHandleHolder(hipStream_t stream) {
    PADDLE_ENFORCE_CUDA_SUCCESS(dynload::rocblas_create_handle(&handle_));
    PADDLE_ENFORCE_CUDA_SUCCESS(dynload::rocblas_get_stream(handle_, &stream));
    // cublasSetMathMode not have corresponding API on rocm platform
  }
#endif
  ~CublasHandleHolder() PADDLE_MAY_THROW {
#if defined PADDLE_WITH_CUDA
    PADDLE_ENFORCE_CUDA_SUCCESS(dynload::cublasDestroy(handle_));
#elif defined PADDLE_WITH_HIP
    PADDLE_ENFORCE_CUDA_SUCCESS(dynload::rocblas_destroy_handle(handle_));
#endif
  }

  template <typename Callback>
  inline void Call(Callback&& callback) const {
    std::lock_guard<std::mutex> guard(mtx_);
    callback(handle_);
  }

 private:
  DISABLE_COPY_AND_ASSIGN(CublasHandleHolder);

#if defined PADDLE_WITH_CUDA
  cublasHandle_t handle_;
#elif defined PADDLE_WITH_HIP
  rocblas_handle handle_;
#endif
  mutable std::mutex mtx_;
};

}  // namespace platform
}  // namespace paddle
