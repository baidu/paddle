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

#include <algorithm>
#include "paddle/fluid/operators/math/blas.h"
#include "paddle/fluid/operators/math/fc.h"
#include <cuda_fp16.h>

namespace paddle {
namespace operators {
namespace math {

template <typename T>
struct FcTypeTraits;

template <>
struct FcTypeTraits<float> {
  typedef float4 Type;
};

template <>
struct FcTypeTraits<double> {
  typedef double4 Type;
};

template <>
struct FcTypeTraits<platform::float16> {
  typedef half4 Type;
};

template <typename T, bool DoRelu>
__global__ void bias_relu_v4(const int num, const T* bias, T* data, int K) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < num) {
    int bias_idx = tid % K;
    const T bias_ptr = bias[bias_idx];
    const T in_ptr = data[tid];
    T packed_val;
    packed_val.x = in_ptr.x + bias_ptr.x;
    packed_val.y = in_ptr.y + bias_ptr.y;
    packed_val.z = in_ptr.z + bias_ptr.z;
    packed_val.w = in_ptr.w + bias_ptr.w;
    if (DoRelu) {
      packed_val.x = fmaxf(0.f, packed_val.x);
      packed_val.y = fmaxf(0.f, packed_val.y);
      packed_val.z = fmaxf(0.f, packed_val.z);
      packed_val.w = fmaxf(0.f, packed_val.w);
    }
    data[tid] = packed_val;
  }
}

template <typename T, bool DoRelu, int BlockDim>
__global__ void InplaceAddReluKernel(const int N, const T* bias, T* data) {
  int offset = blockIdx.x * N;

  for (int i = threadIdx.x; i < N; i += BlockDim) {
    T temp;
#if defined(__HIPCC__) || __CUDA_ARCH__ >= 350
    temp = __ldg(data + offset + i) + __ldg(bias + i);
#else
    temp = data[offset + i] + bias[i];
#endif
    if (DoRelu) {
      data[offset + i] = static_cast<int>(temp > 0) * temp;
    } else {
      data[offset + i] = temp;
    }
  }
}

template <typename T>
class FCFunctor<platform::CUDADeviceContext, T> {
 public:
  void operator()(const platform::CUDADeviceContext& context, const int M,
                  const int N, const int K, const T* X, const T* W, T* Y,
                  const T* B = nullptr, bool relu = false,
                  bool padding_weights = false) {
    PADDLE_ENFORCE_EQ(
        padding_weights, false,
        platform::errors::PermissionDenied(
            "Weight padding in fc can not be used in GPU scope."));
    auto blas = math::GetBlas<platform::CUDADeviceContext, T>(context);
    blas.GEMM(false, false, M, N, K, static_cast<T>(1.0), X, K, W, N,
              static_cast<T>(0.0), Y, N);
    if (B == NULL) {
      return;
    }

    // M * N
    if (N % 4 == 0) {
      const int threads = 256;
      const int num = M * N / 4;
      const int blocks = (num + threads - 1) / threads;
      typedef typename FcTypeTraits<T>::Type trans_type;
      auto* bias_ptr_v4 = reinterpret_cast<const trans_type*>(B);
      auto* data_ptr_v4 = reinterpret_cast<trans_type*>(Y);
      if (relu) {
        bias_relu_v4<trans_type,
                     true><<<blocks, threads, 0, context.stream()>>>(
            num, bias_ptr_v4, data_ptr_v4, N / 4);
      } else {
        bias_relu_v4<trans_type,
                     false><<<blocks, threads, 0, context.stream()>>>(
            num, bias_ptr_v4, data_ptr_v4, N / 4);
      }
    } else {
      const int threads = 256;
      const int blocks = M;
      if (relu) {
        InplaceAddReluKernel<T, true,
                             threads><<<blocks, threads, 0, context.stream()>>>(
            N, B, Y);
      } else {
        InplaceAddReluKernel<T, false,
                             threads><<<blocks, threads, 0, context.stream()>>>(
            N, B, Y);
      }
    }
  }
};

template class FCFunctor<platform::CUDADeviceContext, float>;
template class FCFunctor<platform::CUDADeviceContext, double>;
template class FCFunctor<platform::CUDADeviceContext, platform::float16>;

}  // namespace math
}  // namespace operators
}  // namespace paddle
