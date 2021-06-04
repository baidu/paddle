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

#include "paddle/fluid/operators/math/fc.h"

#include "paddle/fluid/operators/jit/kernels.h"
#include "paddle/fluid/operators/math/blas.h"

namespace paddle {
namespace operators {
namespace math {

template <typename T>
class FCFunctor<platform::CPUDeviceContext, T> {
 public:
  void operator()(const platform::CPUDeviceContext& context, const int M,
                  const int N, const int K, const T* X, const T* W, T* Y,
                  const T* B = nullptr, bool relu = false,
                  bool padding_weights = false) {
    auto blas = math::GetBlas<platform::CPUDeviceContext, T>(context);
    framework::Tensor Y1;
    T* Y1_data = nullptr;
    if (padding_weights) {
      const int NN = N + 4;
      const int KK = K + 4;
      framework::Tensor X1;
      T* X1_data = X1.mutable_data<T>({M * KK}, platform::CPUPlace());
      Y1_data = Y1.mutable_data<T>({M * (N + 4)}, platform::CPUPlace());
#ifdef PADDLE_WITH_MKLML
#pragma omp parallel for
#endif
      for (int i = 0; i < M; i++) {
        memcpy(X1_data + i * KK, X + i * K, K * sizeof(T));
      }
      blas.GEMM(false, false, M, N, K, static_cast<T>(1.0), X1_data, KK, W, NN,
                static_cast<T>(0.0), Y1_data, NN);
    } else {
      blas.MatMul(M, N, K, X, W, Y);
    }
    if (B == nullptr) {
      if (padding_weights) {
#ifdef PADDLE_WITH_MKLML
#pragma omp parallel for
#endif
        for (int i = 0; i < M; i++) {
          memcpy(Y + i * N, Y1_data + i * (N + 4), N * sizeof(T));
        }
      }
      PADDLE_ENFORCE_EQ(relu, false,
                        platform::errors::PermissionDenied(
                            "When bias is NULL, relu can not be true."));
      return;
    }
    auto compute =
        relu ? jit::KernelFuncs<jit::VAddReluTuple<T>,
                                platform::CPUPlace>::Cache()
                   .At(N)
             : jit::KernelFuncs<jit::VAddTuple<T>, platform::CPUPlace>::Cache()
                   .At(N);
#ifdef PADDLE_WITH_MKLML
#pragma omp parallel for
#endif
    for (int i = 0; i < M; i++) {
      T* dst = Y + i * N;
      T* src = (padding_weights) ? Y1_data + i * (N + 4) : dst;
      compute(B, src, dst, N);
    }
  }
};

template class FCFunctor<platform::CPUDeviceContext, float>;
template class FCFunctor<platform::CPUDeviceContext, double>;

}  // namespace math
}  // namespace operators
}  // namespace paddle
