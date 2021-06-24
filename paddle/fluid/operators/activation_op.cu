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

#include "paddle/fluid/operators/activation_op.h"
#include "paddle/fluid/operators/amp/fp16_type_traits.h"
#include "paddle/fluid/operators/elementwise/elementwise_op_impl.cu.h"
#include "paddle/fluid/operators/math/math_cuda_utils.h"
#include "paddle/fluid/platform/bfloat16.h"
#include "paddle/fluid/platform/cuda_device_function.h"

namespace paddle {
namespace operators {

template <typename T>
struct CudaReluFunctor : public BaseActivationFunctor<T> {
  T zero = static_cast<T>(0.0f);

  // relu(x) = max(x, 0)
  // Inputs: args[0], the input x
  __device__ __forceinline__ T operator()(const T* args) const {
    return args[0] > zero ? args[0] : zero;
  }
};

template <typename T>
struct CudaReluGradFunctor : public BaseActivationFunctor<T> {
  T zero = static_cast<T>(0.0f);

  // dx = dout * (out > 0)
  // Inputs: args[0], the input dout
  //         args[1], the input out
  __device__ __forceinline__ T operator()(const T* args) const {
    return args[1] > zero ? args[0] : zero;
  }

  static constexpr ActBwdOpFwdDeps FwdDeps() { return kDepOut; }
};

template <typename T>
struct CudaLeakyReluFunctor : public BaseActivationFunctor<T> {
  T zero = static_cast<T>(0.0f);
  float alpha;

  typename BaseActivationFunctor<T>::AttrPair GetAttrs() {
    return {{"alpha", &alpha}};
  }

  // leakyrelu(x) = x > 0 ? x : alpha * x
  // Inputs: args[0], the input x
  __device__ __forceinline__ T operator()(const T* args) const {
    return args[0] > zero ? args[0] : static_cast<T>(alpha) * args[0];
  }
};

template <typename T>
struct CudaLeakyReluGradFunctor : public BaseActivationFunctor<T> {
  T zero = static_cast<T>(0.0f);
  float alpha;

  typename BaseActivationFunctor<T>::AttrPair GetAttrs() {
    return {{"alpha", &alpha}};
  }

  // dx = dout * (x > 0 ? 1 : alpha)
  // Inputs: args[0], the input dout
  //         args[1], the input x
  __device__ __forceinline__ T operator()(const T* args) const {
    return args[1] > zero ? args[0] : static_cast<T>(alpha) * args[0];
  }

  static constexpr ActBwdOpFwdDeps FwdDeps() { return kDepX; }
};

template <typename T>
struct CudaSigmoidFunctor : public BaseActivationFunctor<T> {
  using MPType = typename details::MPTypeTrait<T>::Type;
  MPType one = static_cast<MPType>(1.0f);

  // sigmoid(x) = 1 / (1 + exp(-x))
  // Inputs: args[0], the input x
  __device__ __forceinline__ T operator()(const T* args) const {
    MPType x = static_cast<MPType>(args[0]);
    return static_cast<T>(one / (one + exp(-x)));
  }
};

template <typename T>
struct CudaSigmoidGradFunctor : public BaseActivationFunctor<T> {
  T one = static_cast<T>(1.0f);

  // dx = dout * out * (1 - out)
  // Inputs: args[0], the input dout
  //         args[1], the input out
  __device__ __forceinline__ T operator()(const T* args) const {
    return args[0] * args[1] * (one - args[1]);
  }

  static constexpr ActBwdOpFwdDeps FwdDeps() { return kDepOut; }
};

template <typename T>
struct CudaSiluFunctor : public BaseActivationFunctor<T> {
  // MPType means Compute Type
  using MPType = typename details::MPTypeTrait<T>::Type;
  MPType one = static_cast<MPType>(1.0f);

  // silu(x) = x / (1 + exp(-x))
  // Inputs: args[0], the input x
  __device__ __forceinline__ T operator()(const T* args) const {
    MPType x = static_cast<MPType>(args[0]);
    return static_cast<T>(x / (one + exp(-x)));
  }
};

template <typename T>
struct CudaSiluGradFunctor : public BaseActivationFunctor<T> {
  using MPType = typename details::MPTypeTrait<T>::Type;
  MPType one = static_cast<MPType>(1.0f);

  // dx = dout * (1 + exp(-x) + x * exp(-x) / (1 + exp(-x))^2)
  // Inputs: args[0], the input dout
  //         args[1], the input x
  __device__ __forceinline__ T operator()(const T* args) const {
    MPType dout = static_cast<MPType>(args[0]);
    MPType x = static_cast<MPType>(args[1]);
    MPType temp = one / (one + exp(-x));
    return static_cast<T>(dout * (temp * (one + x * (one - temp))));
  }

  static constexpr ActBwdOpFwdDeps FwdDeps() { return kDepX; }
};

template <typename T>
struct CudaLogSigmoidFunctor : public BaseActivationFunctor<T> {
  using MPType = typename details::MPTypeTrait<T>::Type;
  MPType zero = static_cast<MPType>(0.0f);

  // logsigmoid(x) = log(1 / (1 + exp(-x)))
  // For numerical stability,
  // logsigmoid(x) =
  //          - (max(-x, 0) + log(exp(-max(-x, 0)) + exp(-x - max(-x, 0))))
  // Inputs: args[0], the input x
  __device__ __forceinline__ T operator()(const T* args) const {
    MPType x = static_cast<MPType>(args[0]);
    MPType temp = x > zero ? zero : -x;
    return static_cast<T>(-temp - log(exp(-temp) + exp(-x - temp)));
  }
};

template <typename T>
struct CudaLogSigmoidGradFunctor : public BaseActivationFunctor<T> {
  using MPType = typename details::MPTypeTrait<T>::Type;
  MPType zero = static_cast<MPType>(0.0f);

  // dx = dout * exp(-x) / (1 + exp(-x))
  // For numerical stability:
  // dx = dout * exp(-x - max(-x, 0)) / (exp(-max(-x, 0)) + exp(-x - max(-x,
  // 0)))
  // Inputs: args[0], the input dout
  //         args[1], the input x
  __device__ __forceinline__ T operator()(const T* args) const {
    MPType dout = static_cast<MPType>(args[0]);
    MPType x = static_cast<MPType>(args[1]);
    MPType temp1 = x > zero ? zero : -x;
    MPType temp2 = exp(-x - temp1);
    return static_cast<T>(dout * (temp2 / (exp(-temp1) + temp2)));
  }

  static constexpr ActBwdOpFwdDeps FwdDeps() { return kDepX; }
};

template <typename T>
struct CudaAtanFunctor : public BaseActivationFunctor<T> {
  using MPType = typename details::MPTypeTrait<T>::Type;

  // atan(x) = atan(x)
  // Inputs: args[0], the input x
  __device__ __forceinline__ T operator()(const T* args) const {
    MPType x = static_cast<MPType>(args[0]);
    return static_cast<T>(atan(x));
  }
};

template <typename T>
struct CudaAtanGradFunctor : public BaseActivationFunctor<T> {
  T one = static_cast<T>(1.0f);

  // dx = dout / (1 + x^2)
  // Inputs: args[0], the input dout
  //         args[1], the input x
  __device__ __forceinline__ T operator()(const T* args) const {
    return args[0] / (one + args[1] * args[1]);
  }

  static constexpr ActBwdOpFwdDeps FwdDeps() { return kDepX; }
};

template <typename T>
struct CudaSoftShrinkFunctor : public BaseActivationFunctor<T> {
  float lambda;

  typename BaseActivationFunctor<T>::AttrPair GetAttrs() {
    return {{"lambda", &lambda}};
  }

  // softshrink(x) = x - lambda, if x > lambda;
  //                 x + lambda, if x < -lambda;
  //                 0, otherwise.
  // Inputs: args[0], the input x
  __device__ __forceinline__ T operator()(const T* args) const {
    T x = args[0];
    T l = static_cast<T>(lambda);
    T temp1 = static_cast<T>(x > l);
    T temp2 = static_cast<T>(x < -l);
    return temp1 * (x - l) + temp2 * (x + l);
  }
};

template <typename T>
struct CudaSoftShrinkGradFunctor : public BaseActivationFunctor<T> {
  T zero = static_cast<T>(0.0f);
  float lambda;

  typename BaseActivationFunctor<T>::AttrPair GetAttrs() {
    return {{"lambda", &lambda}};
  }

  // dx = dout, if x > lambda or x < -lambda else 0
  // Inputs: args[0], the input dout
  //         args[1], the input x
  __device__ __forceinline__ T operator()(const T* args) const {
    T x = args[1];
    T l = static_cast<T>(lambda);
    return (x >= -l && x <= l) ? zero : args[0];
  }

  static constexpr ActBwdOpFwdDeps FwdDeps() { return kDepX; }
};

template <typename T>
struct CudaCeilFunctor : public BaseActivationFunctor<T> {
  using MPType = typename details::MPTypeTrait<T>::Type;

  // ceil(x) = ceil(x)
  // Inputs: args[0], the input x
  __device__ __forceinline__ T operator()(const T* args) const {
    MPType x = static_cast<MPType>(args[0]);
    return static_cast<T>(ceil(x));
  }
};

template <typename T>
struct CudaFloorFunctor : public BaseActivationFunctor<T> {
  using MPType = typename details::MPTypeTrait<T>::Type;

  // floor(x) = floor(x)
  // Inputs: args[0], the input x
  __device__ __forceinline__ T operator()(const T* args) const {
    MPType x = static_cast<MPType>(args[0]);
    return static_cast<T>(floor(x));
  }
};

template <typename T>
struct CudaRoundFunctor : public BaseActivationFunctor<T> {
  using MPType = typename details::MPTypeTrait<T>::Type;

  // round(x) = round(x)
  // Inputs: args[0], the input x
  __device__ __forceinline__ T operator()(const T* args) const {
    MPType x = static_cast<MPType>(args[0]);
    return static_cast<T>(round(x));
  }
};

// grad functor for ceil, floor and round
template <typename T>
struct CudaZeroGradFunctor : public BaseActivationFunctor<T> {
  __device__ __forceinline__ T operator()(const T* args) const {
    return static_cast<T>(0.0f);
  }

  static constexpr ActBwdOpFwdDeps FwdDeps() { return kNoDeps; }
};

template <typename T>
struct CudaCosFunctor : public BaseActivationFunctor<T> {
  using MPType = typename details::MPTypeTrait<T>::Type;

  // cos(x) = cos(x)
  // Inputs: args[0], the input x
  __device__ __forceinline__ T operator()(const T* args) const {
    MPType x = static_cast<MPType>(args[0]);
    return static_cast<T>(cos(x));
  }
};

template <typename T>
struct CudaCosGradFunctor : public BaseActivationFunctor<T> {
  using MPType = typename details::MPTypeTrait<T>::Type;

  // dx = dout * (-sin(x))
  // Inputs: args[0], the input dout
  //         args[1], the input x
  __device__ __forceinline__ T operator()(const T* args) const {
    MPType dout = static_cast<MPType>(args[0]);
    MPType x = static_cast<MPType>(args[1]);
    return static_cast<T>(-dout * sin(x));
  }

  static constexpr ActBwdOpFwdDeps FwdDeps() { return kDepX; }
};

template <typename T>
struct CudaSinFunctor : public BaseActivationFunctor<T> {
  using MPType = typename details::MPTypeTrait<T>::Type;

  // sin(x) = sin(x)
  // Inputs: args[0], the input x
  __device__ __forceinline__ T operator()(const T* args) const {
    MPType x = static_cast<MPType>(args[0]);
    return static_cast<T>(sin(x));
  }
};

template <typename T>
struct CudaSinGradFunctor : public BaseActivationFunctor<T> {
  using MPType = typename details::MPTypeTrait<T>::Type;

  // dx = dout * cos(x)
  // Inputs: args[0], the input dout
  //         args[1], the input x
  __device__ __forceinline__ T operator()(const T* args) const {
    MPType dout = static_cast<MPType>(args[0]);
    MPType x = static_cast<MPType>(args[1]);
    return static_cast<T>(dout * cos(x));
  }

  static constexpr ActBwdOpFwdDeps FwdDeps() { return kDepX; }
};

template <typename T>
struct CudaTanFunctor : public BaseActivationFunctor<T> {
  using MPType = typename details::MPTypeTrait<T>::Type;

  // tan(x) = tan(x)
  // Inputs: args[0], the input x
  __device__ __forceinline__ T operator()(const T* args) const {
    MPType x = static_cast<MPType>(args[0]);
    return static_cast<T>(tan(x));
  }
};

template <typename T>
struct CudaTanGradFunctor : public BaseActivationFunctor<T> {
  using MPType = typename details::MPTypeTrait<T>::Type;

  // dx = dout / cos(x)^2
  // Inputs: args[0], the input dout
  //         args[1], the input x
  __device__ __forceinline__ T operator()(const T* args) const {
    MPType dout = static_cast<MPType>(args[0]);
    MPType x = static_cast<MPType>(args[1]);
    return static_cast<T>(dout / (cos(x) * cos(x)));
  }

  static constexpr ActBwdOpFwdDeps FwdDeps() { return kDepX; }
};

template <typename T>
struct CudaAsinFunctor : public BaseActivationFunctor<T> {
  using MPType = typename details::MPTypeTrait<T>::Type;

  // asin(x) = asin(x)
  // Inputs: args[0], the input x
  __device__ __forceinline__ T operator()(const T* args) const {
    MPType x = static_cast<MPType>(args[0]);
    return static_cast<T>(asin(x));
  }
};

template <typename T>
struct CudaAsinGradFunctor : public BaseActivationFunctor<T> {
  using MPType = typename details::MPTypeTrait<T>::Type;
  MPType one = static_cast<MPType>(1.0f);

  // dx = dout / sqrt(1 - x^2)
  // Inputs: args[0], the input dout
  //         args[1], the input x
  __device__ __forceinline__ T operator()(const T* args) const {
    MPType dout = static_cast<MPType>(args[0]);
    MPType x = static_cast<MPType>(args[1]);
    return static_cast<T>(dout / sqrt(one - x * x));
  }

  static constexpr ActBwdOpFwdDeps FwdDeps() { return kDepX; }
};

template <typename T>
struct CudaAcosFunctor : public BaseActivationFunctor<T> {
  using MPType = typename details::MPTypeTrait<T>::Type;

  // acos(x) = acos(x)
  // Inputs: args[0], the input x
  __device__ __forceinline__ T operator()(const T* args) const {
    MPType x = static_cast<MPType>(args[0]);
    return static_cast<T>(acos(x));
  }
};

template <typename T>
struct CudaAcosGradFunctor : public BaseActivationFunctor<T> {
  using MPType = typename details::MPTypeTrait<T>::Type;
  MPType one = static_cast<MPType>(1.0f);

  // dx = -dout / sqrt(1 - x^2)
  // Inputs: args[0], the input dout
  //         args[1], the input x
  __device__ __forceinline__ T operator()(const T* args) const {
    MPType dout = static_cast<MPType>(args[0]);
    MPType x = static_cast<MPType>(args[1]);
    return static_cast<T>(-dout / sqrt(one - x * x));
  }

  static constexpr ActBwdOpFwdDeps FwdDeps() { return kDepX; }
};

template <typename T>
struct CudaCoshFunctor : public BaseActivationFunctor<T> {
  using MPType = typename details::MPTypeTrait<T>::Type;

  // cosh(x) = cosh(x)
  // Inputs: args[0], the input x
  __device__ __forceinline__ T operator()(const T* args) const {
    MPType x = static_cast<MPType>(args[0]);
    return static_cast<T>(cosh(x));
  }
};

template <typename T>
struct CudaCoshGradFunctor : public BaseActivationFunctor<T> {
  using MPType = typename details::MPTypeTrait<T>::Type;

  // dx = dout * sinh(x)
  // Inputs: args[0], the input dout
  //         args[1], the input x
  __device__ __forceinline__ T operator()(const T* args) const {
    MPType dout = static_cast<MPType>(args[0]);
    MPType x = static_cast<MPType>(args[1]);
    return static_cast<T>(dout * sinh(x));
  }

  static constexpr ActBwdOpFwdDeps FwdDeps() { return kDepX; }
};

template <typename T>
struct CudaSinhFunctor : public BaseActivationFunctor<T> {
  using MPType = typename details::MPTypeTrait<T>::Type;

  // sinh(x) = sinh(x)
  // Inputs: args[0], the input x
  __device__ __forceinline__ T operator()(const T* args) const {
    MPType x = static_cast<MPType>(args[0]);
    return static_cast<T>(sinh(x));
  }
};

template <typename T>
struct CudaSinhGradFunctor : public BaseActivationFunctor<T> {
  using MPType = typename details::MPTypeTrait<T>::Type;

  // dx = dout * cosh(x)
  // Inputs: args[0], the input dout
  //         args[1], the input x
  __device__ __forceinline__ T operator()(const T* args) const {
    MPType dout = static_cast<MPType>(args[0]);
    MPType x = static_cast<MPType>(args[1]);
    return static_cast<T>(dout * cosh(x));
  }

  static constexpr ActBwdOpFwdDeps FwdDeps() { return kDepX; }
};

template <typename T>
struct CudaTanhFunctor : public BaseActivationFunctor<T> {
  using MPType = typename details::MPTypeTrait<T>::Type;

  // tanh(x) = tanh(x)
  // Inputs: args[0], the input x
  __device__ __forceinline__ T operator()(const T* args) const {
    MPType x = static_cast<MPType>(args[0]);
    return static_cast<T>(tanh(x));
  }
};

template <typename T>
struct CudaTanhGradFunctor : public BaseActivationFunctor<T> {
  T one = static_cast<T>(1.0f);

  // dx = dout * (1 - out^2)
  // Inputs: args[0], the input dout
  //         args[1], the input out
  __device__ __forceinline__ T operator()(const T* args) const {
    T dout = static_cast<T>(args[0]);
    T out = static_cast<T>(args[1]);
    return dout * (one - out * out);
  }

  static constexpr ActBwdOpFwdDeps FwdDeps() { return kDepOut; }
};

template <typename T>
struct CudaReciprocalFunctor : public BaseActivationFunctor<T> {
  T one = static_cast<T>(1.0f);

  // reciprocal(x) = 1 / x
  // Inputs: args[0], the input x
  __device__ __forceinline__ T operator()(const T* args) const {
    return one / args[0];
  }
};

template <typename T>
struct CudaReciprocalGradFunctor : public BaseActivationFunctor<T> {
  // dx = -dout * out^2
  // Inputs: args[0], the input dout
  //         args[1], the input out
  __device__ __forceinline__ T operator()(const T* args) const {
    return -args[0] * args[1] * args[1];
  }

  static constexpr ActBwdOpFwdDeps FwdDeps() { return kDepOut; }
};

template <typename T>
struct CudaExpFunctor : public BaseActivationFunctor<T> {
  using MPType = typename details::MPTypeTrait<T>::Type;

  // exp(x) = exp(x)
  // Inputs: args[0], the input x
  __device__ __forceinline__ T operator()(const T* args) const {
    MPType x = static_cast<MPType>(args[0]);
    return static_cast<T>(exp(x));
  }
};

template <typename T>
struct CudaExpGradFunctor : public BaseActivationFunctor<T> {
  // dx = dout * out
  // Inputs: args[0], the input dout
  //         args[1], the input out
  __device__ __forceinline__ T operator()(const T* args) const {
    return args[0] * args[1];
  }

  static constexpr ActBwdOpFwdDeps FwdDeps() { return kDepOut; }
};

template <typename T>
struct CudaExpm1Functor : public BaseActivationFunctor<T> {
  using MPType = typename details::MPTypeTrait<T>::Type;

  // expm1(x) = expm1(x)
  // Inputs: args[0], the input x
  __device__ __forceinline__ T operator()(const T* args) const {
    MPType x = static_cast<MPType>(args[0]);
    return static_cast<T>(expm1(x));
  }
};

template <typename T>
struct CudaExpm1GradFunctor : public BaseActivationFunctor<T> {
  // dx = dout * out
  // Inputs: args[0], the input dout
  //         args[1], the input out
  __device__ __forceinline__ T operator()(const T* args) const {
    return args[0] * args[1] + args[0];
  }

  static constexpr ActBwdOpFwdDeps FwdDeps() { return kDepOut; }
};

template <typename T>
struct CudaLogFunctor : public BaseActivationFunctor<T> {
  using MPType = typename details::MPTypeTrait<T>::Type;

  // log(x) = log(x)
  // Inputs: args[0], the input x
  __device__ __forceinline__ T operator()(const T* args) const {
    MPType x = static_cast<MPType>(args[0]);
    return static_cast<T>(log(x));
  }
};

template <typename T>
struct CudaLogGradFunctor : public BaseActivationFunctor<T> {
  // dx = dout / x
  // Inputs: args[0], the input dout
  //         args[1], the input x
  __device__ __forceinline__ T operator()(const T* args) const {
    return args[0] / args[1];
  }

  static constexpr ActBwdOpFwdDeps FwdDeps() { return kDepX; }
};

template <typename T>
struct CudaSquareFunctor : public BaseActivationFunctor<T> {
  // square(x) = x * x
  // Inputs: args[0], the input x
  __device__ __forceinline__ T operator()(const T* args) const {
    return args[0] * args[0];
  }
};

template <typename T>
struct CudaSquareGradFunctor : public BaseActivationFunctor<T> {
  T two = static_cast<T>(2.0f);

  // dx = dout * 2 * x
  // Inputs: args[0], the input dout
  //         args[1], the input x
  __device__ __forceinline__ T operator()(const T* args) const {
    return args[0] * two * args[1];
  }

  static constexpr ActBwdOpFwdDeps FwdDeps() { return kDepX; }
};

template <typename T>
struct CudaSqrtFunctor : public BaseActivationFunctor<T> {
  using MPType = typename details::MPTypeTrait<T>::Type;

  // sqrt(x) = sqrt(x)
  // Inputs: args[0], the input x
  __device__ __forceinline__ T operator()(const T* args) const {
    MPType x = static_cast<MPType>(args[0]);
    return static_cast<T>(sqrt(x));
  }
};

template <typename T>
struct CudaSqrtGradFunctor : public BaseActivationFunctor<T> {
  T one_half = static_cast<T>(0.5f);

  // dx = dout * 0.5 / out
  // Inputs: args[0], the input dout
  //         args[1], the input out
  __device__ __forceinline__ T operator()(const T* args) const {
    return one_half * args[0] / args[1];
  }

  static constexpr ActBwdOpFwdDeps FwdDeps() { return kDepOut; }
};

template <typename T>
struct CudaRsqrtFunctor : public BaseActivationFunctor<T> {
  using MPType = typename details::MPTypeTrait<T>::Type;

  // rsqrt(x) = rsqrt(x)
  // Inputs: args[0], the input x
  __device__ __forceinline__ T operator()(const T* args) const {
    MPType x = static_cast<MPType>(args[0]);
    return static_cast<T>(rsqrt(x));
  }
};

template <typename T>
struct CudaRsqrtGradFunctor : public BaseActivationFunctor<T> {
  T minus_one_half = static_cast<T>(-0.5f);

  // dx = dout * -0.5 / out^3
  // Inputs: args[0], the input dout
  //         args[1], the input out
  __device__ __forceinline__ T operator()(const T* args) const {
    T out = args[1];
    return minus_one_half * args[0] * out * out * out;
  }

  static constexpr ActBwdOpFwdDeps FwdDeps() { return kDepOut; }
};

template <typename T>
struct CudaLog1pFunctor : public BaseActivationFunctor<T> {
  using MPType = typename details::MPTypeTrait<T>::Type;
  MPType one = static_cast<MPType>(1.0f);

  // log1p(x) = log(1 + x)
  // Inputs: args[0], the input x
  __device__ __forceinline__ T operator()(const T* args) const {
    MPType x = static_cast<MPType>(args[0]);
    return static_cast<T>(log(one + x));
  }
};

template <typename T>
struct CudaLog1pGradFunctor : public BaseActivationFunctor<T> {
  T one = static_cast<T>(1.0f);

  // dx = dout / (1 + x)
  // Inputs: args[0], the input dout
  //         args[1], the input x
  __device__ __forceinline__ T operator()(const T* args) const {
    return args[0] / (one + args[1]);
  }

  static constexpr ActBwdOpFwdDeps FwdDeps() { return kDepX; }
};

template <typename T>
struct CudaLog2Functor : public BaseActivationFunctor<T> {
  using MPType = typename details::MPTypeTrait<T>::Type;

  // log2(x) = log2(x)
  // Inputs: args[0], the input x
  __device__ __forceinline__ T operator()(const T* args) const {
    MPType x = static_cast<MPType>(args[0]);
    return static_cast<T>(log2(x));
  }
};

template <typename T>
struct CudaLog2GradFunctor : public BaseActivationFunctor<T> {
  using MPType = typename details::MPTypeTrait<T>::Type;
  T log_two = static_cast<T>(log(static_cast<MPType>(2.0f)));

  // dx = dout / (x * log(2))
  // Inputs: args[0], the input dout
  //         args[1], the input x
  __device__ __forceinline__ T operator()(const T* args) const {
    return args[0] / (args[1] * log_two);
  }

  static constexpr ActBwdOpFwdDeps FwdDeps() { return kDepX; }
};

template <typename T>
struct CudaLog10Functor : public BaseActivationFunctor<T> {
  using MPType = typename details::MPTypeTrait<T>::Type;

  // log10(x) = log10(x)
  // Inputs: args[0], the input x
  __device__ __forceinline__ T operator()(const T* args) const {
    MPType x = static_cast<MPType>(args[0]);
    return static_cast<T>(log10(x));
  }
};

template <typename T>
struct CudaLog10GradFunctor : public BaseActivationFunctor<T> {
  using MPType = typename details::MPTypeTrait<T>::Type;
  T log_ten = static_cast<T>(log(static_cast<MPType>(10.0f)));

  // dx = dout / (x * log(10))
  // Inputs: args[0], the input dout
  //         args[1], the input x
  __device__ __forceinline__ T operator()(const T* args) const {
    return args[0] / (args[1] * log_ten);
  }

  static constexpr ActBwdOpFwdDeps FwdDeps() { return kDepX; }
};

template <typename T>
struct CudaBReluFunctor : public BaseActivationFunctor<T> {
  float t_min;
  float t_max;

  typename BaseActivationFunctor<T>::AttrPair GetAttrs() {
    return {{"t_min", &t_min}, {"t_max", &t_max}};
  }

  // brelu(x) = min(max(x, t_min), t_max)
  // Inputs: args[0], the input x
  __device__ __forceinline__ T operator()(const T* args) const {
    T x = args[0];
    T t_min_cast = static_cast<T>(t_min);
    T t_max_cast = static_cast<T>(t_max);
    T temp_max = x > t_min_cast ? x : t_min_cast;
    T temp_min = temp_max < t_max_cast ? temp_max : t_max_cast;
    return temp_min;
  }
};

template <typename T>
struct CudaBReluGradFunctor : public BaseActivationFunctor<T> {
  T zero = static_cast<T>(0.0f);
  float t_min;
  float t_max;

  typename BaseActivationFunctor<T>::AttrPair GetAttrs() {
    return {{"t_min", &t_min}, {"t_max", &t_max}};
  }

  // dx = (x > t_min && x < t_max) ? dout : 0
  // Inputs: args[0], the input dout
  //         args[1], the input x
  __device__ __forceinline__ T operator()(const T* args) const {
    T dout = args[0];
    T x = args[1];
    T t_min_cast = static_cast<T>(t_min);
    T t_max_cast = static_cast<T>(t_max);
    return (x > t_min_cast && x < t_max_cast) ? dout : zero;
  }

  static constexpr ActBwdOpFwdDeps FwdDeps() { return kDepX; }
};

template <typename T>
struct CudaSoftReluFunctor : public BaseActivationFunctor<T> {
  using MPType = typename details::MPTypeTrait<T>::Type;
  MPType one = static_cast<MPType>(1.0f);
  float threshold;

  typename BaseActivationFunctor<T>::AttrPair GetAttrs() {
    return {{"threshold", &threshold}};
  }

  // soft_relu(x) = log(1 + exp(max(min(x, threshold), -threshold)))
  // Inputs: args[0], the input x
  // threshold should not be negative
  __device__ __forceinline__ T operator()(const T* args) const {
    MPType x = static_cast<MPType>(args[0]);
    MPType t = static_cast<MPType>(threshold);
    MPType temp_min = x < t ? x : t;
    MPType temp_max = temp_min > -t ? temp_min : -t;
    return static_cast<T>(log(one + exp(temp_max)));
  }
};

template <typename T>
struct CudaSoftReluGradFunctor : public BaseActivationFunctor<T> {
  using MPType = typename details::MPTypeTrait<T>::Type;
  MPType one = static_cast<MPType>(1.0f);
  float threshold;

  typename BaseActivationFunctor<T>::AttrPair GetAttrs() {
    return {{"threshold", &threshold}};
  }

  // dx = (out > -threshold && out < threshold) ? dout * (1 - exp(-out)) : 0
  // Inputs: args[0], the input dout
  //         args[1], the input out
  // threshold should not be negative
  __device__ __forceinline__ T operator()(const T* args) const {
    MPType dout = static_cast<MPType>(args[0]);
    MPType out = static_cast<MPType>(args[1]);
    MPType t = static_cast<MPType>(threshold);
    return (out > -t && out < t) ? static_cast<T>(dout * (one - exp(-out)))
                                 : static_cast<T>(0.0f);
  }

  static constexpr ActBwdOpFwdDeps FwdDeps() { return kDepOut; }
};

template <typename T>
struct CudaSTanhFunctor : public BaseActivationFunctor<T> {
  using MPType = typename details::MPTypeTrait<T>::Type;
  float scale_a;
  float scale_b;

  typename BaseActivationFunctor<T>::AttrPair GetAttrs() {
    return {{"scale_a", &scale_a}, {"scale_b", &scale_b}};
  }

  // stanh(x) = b * tanh(a * x)
  // Inputs: args[0], the input x
  __device__ __forceinline__ T operator()(const T* args) const {
    MPType x = static_cast<MPType>(args[0]);
    MPType a = static_cast<MPType>(scale_a);
    MPType b = static_cast<MPType>(scale_b);
    return static_cast<T>(b * tanh(a * x));
  }
};

template <typename T>
struct CudaSTanhGradFunctor : public BaseActivationFunctor<T> {
  using MPType = typename details::MPTypeTrait<T>::Type;
  MPType one = static_cast<MPType>(1.0f);
  float scale_a;
  float scale_b;

  typename BaseActivationFunctor<T>::AttrPair GetAttrs() {
    return {{"scale_a", &scale_a}, {"scale_b", &scale_b}};
  }

  // dx = dout * a * b * (1 - tanh(a * x) * tanh(a * x))
  // Inputs: args[0], the input dout
  //         args[1], the input x
  __device__ __forceinline__ T operator()(const T* args) const {
    MPType dout = static_cast<MPType>(args[0]);
    MPType x = static_cast<MPType>(args[1]);
    MPType a = static_cast<MPType>(scale_a);
    MPType b = static_cast<MPType>(scale_b);
    MPType temp = tanh(a * x);
    return static_cast<T>(dout * a * b * (one - temp * temp));
  }

  static constexpr ActBwdOpFwdDeps FwdDeps() { return kDepX; }
};

template <typename T>
struct CudaSoftplusFunctor : public BaseActivationFunctor<T> {
  using MPType = typename details::MPTypeTrait<T>::Type;
  MPType one = static_cast<MPType>(1.0f);
  float beta;
  float threshold;

  typename BaseActivationFunctor<T>::AttrPair GetAttrs() {
    return {{"beta", &beta}, {"threshold", &threshold}};
  }

  // softplus(x) = beta * x > threshold ? x : log(1 + exp(beta * x)) / beta
  // Inputs: args[0], the input x
  __device__ __forceinline__ T operator()(const T* args) const {
    MPType x = static_cast<MPType>(args[0]);
    MPType b = static_cast<MPType>(beta);
    MPType t = static_cast<MPType>(threshold);
    MPType x_beta = x * beta;
    return static_cast<T>(x_beta > t ? x : log(one + exp(x_beta)) / b);
  }
};

template <typename T>
struct CudaSoftplusGradFunctor : public BaseActivationFunctor<T> {
  using MPType = typename details::MPTypeTrait<T>::Type;
  MPType one = static_cast<MPType>(1.0f);
  float beta;
  float threshold;

  typename BaseActivationFunctor<T>::AttrPair GetAttrs() {
    return {{"beta", &beta}, {"threshold", &threshold}};
  }

  // dx = x * beta > threshold ? dout : dout / (1 + exp(-beta * x))
  // Inputs: args[0], the input dout
  //         args[1], the input x
  __device__ __forceinline__ T operator()(const T* args) const {
    MPType dout = static_cast<MPType>(args[0]);
    MPType x = static_cast<MPType>(args[1]);
    MPType b = static_cast<MPType>(beta);
    MPType t = static_cast<MPType>(threshold);
    MPType x_beta = x * beta;
    return x_beta > t ? args[0] : static_cast<T>(dout / (one + exp(-x_beta)));
  }

  static constexpr ActBwdOpFwdDeps FwdDeps() { return kDepX; }
};

template <typename T>
struct CudaSoftsignFunctor : public BaseActivationFunctor<T> {
  T one = static_cast<T>(1.0f);

  // softsign(x) = x / (1 + abs(x))
  // Inputs: args[0], the input x
  __device__ __forceinline__ T operator()(const T* args) const {
    return args[0] / (one + abs(args[0]));
  }
};

template <typename T>
struct CudaSoftsignGradFunctor : public BaseActivationFunctor<T> {
  T one = static_cast<T>(1.0f);

  // dx = dout / (1 + abs(x))^2
  // Inputs: args[0], the input dout
  //         args[1], the input x
  __device__ __forceinline__ T operator()(const T* args) const {
    T temp = one + abs(args[1]);
    return args[0] / (temp * temp);
  }

  static constexpr ActBwdOpFwdDeps FwdDeps() { return kDepX; }
};

template <typename T>
struct CudaRelu6Functor : public BaseActivationFunctor<T> {
  T zero = static_cast<T>(0.0f);
  float threshold;

  typename BaseActivationFunctor<T>::AttrPair GetAttrs() {
    return {{"threshold", &threshold}};
  }

  // relu6(x) = min(max(0, x), 6)
  // Inputs: args[0], the input x
  __device__ __forceinline__ T operator()(const T* args) const {
    T t = static_cast<T>(threshold);
    return args[0] <= zero ? zero : (args[0] < t ? args[0] : t);
  }
};

template <typename T>
struct CudaRelu6GradFunctor : public BaseActivationFunctor<T> {
  T zero = static_cast<T>(0.0f);
  float threshold;

  typename BaseActivationFunctor<T>::AttrPair GetAttrs() {
    return {{"threshold", &threshold}};
  }

  // dx = (out > 0 && out < t) ? dout : 0
  // Inputs: args[0], the input dout
  //         args[1], the input out
  __device__ __forceinline__ T operator()(const T* args) const {
    T t = static_cast<T>(threshold);
    return (args[1] > zero && args[1] < t) ? args[0] : zero;
  }

  static constexpr ActBwdOpFwdDeps FwdDeps() { return kDepOut; }
};

template <typename T>
struct CudaTanhShrinkFunctor : public BaseActivationFunctor<T> {
  using MPType = typename details::MPTypeTrait<T>::Type;

  // tanhshrink(x) = x - tanh(x)
  // Inputs: args[0], the input x
  __device__ __forceinline__ T operator()(const T* args) const {
    MPType x = static_cast<MPType>(args[0]);
    return static_cast<T>(x - tanh(x));
  }
};

template <typename T>
struct CudaTanhShrinkGradFunctor : public BaseActivationFunctor<T> {
  using MPType = typename details::MPTypeTrait<T>::Type;

  // dx = dout * tanh(x)^2
  // Inputs: args[0], the input dout
  //         args[1], the input x
  __device__ __forceinline__ T operator()(const T* args) const {
    MPType dout = static_cast<MPType>(args[0]);
    MPType x = static_cast<MPType>(args[1]);
    return static_cast<T>(dout * tanh(x) * tanh(x));
  }

  static constexpr ActBwdOpFwdDeps FwdDeps() { return kDepX; }
};

template <typename T>
struct CudaHardShrinkFunctor : public BaseActivationFunctor<T> {
  T zero = static_cast<T>(0.0f);
  float threshold;

  typename BaseActivationFunctor<T>::AttrPair GetAttrs() {
    return {{"threshold", &threshold}};
  }

  // hadrshrink(x) = (x > -threshold && x < threshold) ? 0 : x
  // Inputs: args[0], the input x
  __device__ __forceinline__ T operator()(const T* args) const {
    T x = args[0];
    T t = static_cast<T>(threshold);
    return (x > -t && x < t) ? zero : x;
  }
};

template <typename T>
struct CudaHardShrinkGradFunctor : public BaseActivationFunctor<T> {
  T zero = static_cast<T>(0.0f);
  float threshold;

  typename BaseActivationFunctor<T>::AttrPair GetAttrs() {
    return {{"threshold", &threshold}};
  }

  // dx = (x > -threshold && x < threshold) ? 0 : dout
  // Inputs: args[0], the input dout
  //         args[1], the input x
  __device__ __forceinline__ T operator()(const T* args) const {
    T x = args[1];
    T t = static_cast<T>(threshold);
    return (x > -t && x < t) ? zero : args[0];
  }

  static constexpr ActBwdOpFwdDeps FwdDeps() { return kDepX; }
};

template <typename T>
struct CudaHardSigmoidFunctor : public BaseActivationFunctor<T> {
  T zero = static_cast<T>(0.0f);
  T one = static_cast<T>(1.0f);
  float slope;
  float offset;

  typename BaseActivationFunctor<T>::AttrPair GetAttrs() {
    return {{"slope", &slope}, {"offset", &offset}};
  }

  // hard_sigmoid(x) = 0, when x <= -3
  //                   1, when x >= 3
  //                   x * slope + offset, otherwise
  // Inputs: args[0], the input x
  __device__ __forceinline__ T operator()(const T* args) const {
    T temp = args[0] * static_cast<T>(slope) + static_cast<T>(offset);
    T temp_max = temp > zero ? temp : zero;
    T temp_min = temp_max < one ? temp_max : one;
    return temp_min;
  }
};

template <typename T>
struct CudaHardSigmoidGradFunctor : public BaseActivationFunctor<T> {
  T zero = static_cast<T>(0.0f);
  T one = static_cast<T>(1.0f);
  float slope;
  float offset;

  typename BaseActivationFunctor<T>::AttrPair GetAttrs() {
    return {{"slope", &slope}, {"offset", &offset}};
  }

  // dx = (out > 0 && out < 1) ? dout * slope : 0
  // Inputs: args[0], the input dout
  //         args[1], the input out
  __device__ __forceinline__ T operator()(const T* args) const {
    T out = args[1];
    return (out > zero && out < one) ? args[0] * static_cast<T>(slope) : zero;
  }

  static constexpr ActBwdOpFwdDeps FwdDeps() { return kDepOut; }
};

template <typename T>
struct CudaSwishFunctor : public BaseActivationFunctor<T> {
  using MPType = typename details::MPTypeTrait<T>::Type;
  MPType one = static_cast<MPType>(1.0f);
  float beta;

  typename BaseActivationFunctor<T>::AttrPair GetAttrs() {
    return {{"beta", &beta}};
  }

  // swish(x) = x / (1 + exp(-beta * x))
  // Inputs: args[0], the input x
  __device__ __forceinline__ T operator()(const T* args) const {
    MPType x = static_cast<MPType>(args[0]);
    MPType b = static_cast<MPType>(beta);
    return static_cast<T>(x / (one + exp(-b * x)));
  }
};

template <typename T>
struct CudaSwishGradFunctor : public BaseActivationFunctor<T> {
  using MPType = typename details::MPTypeTrait<T>::Type;
  MPType one = static_cast<MPType>(1.0f);
  float beta;

  typename BaseActivationFunctor<T>::AttrPair GetAttrs() {
    return {{"beta", &beta}};
  }

  // dx = dout * (1 + exp(-b * x) + b * x * exp(-b * x) / (1 + exp(-b * x))^2)
  // Inputs: args[0], the input dout
  //         args[1], the input x
  __device__ __forceinline__ T operator()(const T* args) const {
    MPType dout = static_cast<MPType>(args[0]);
    MPType x = static_cast<MPType>(args[1]);
    MPType b = static_cast<MPType>(beta);
    MPType temp1 = one / (one + exp(-b * x));
    MPType out = x * temp1;
    MPType temp2 = b * out;
    MPType temp3 = temp1 * (one - temp2);
    return static_cast<T>(dout * (temp2 + temp3));
  }

  static constexpr ActBwdOpFwdDeps FwdDeps() { return kDepX; }
};

template <typename T>
struct CudaThresholdedReluFunctor : public BaseActivationFunctor<T> {
  T zero = static_cast<T>(0.0f);
  float threshold;

  typename BaseActivationFunctor<T>::AttrPair GetAttrs() {
    return {{"threshold", &threshold}};
  }

  // thresholded_relu(x) = x > threshold ? x : 0
  // Inputs: args[0], the input x
  __device__ __forceinline__ T operator()(const T* args) const {
    return args[0] > static_cast<T>(threshold) ? args[0] : zero;
  }
};

template <typename T>
struct CudaThresholdedReluGradFunctor : public BaseActivationFunctor<T> {
  T zero = static_cast<T>(0.0f);
  float threshold;

  typename BaseActivationFunctor<T>::AttrPair GetAttrs() {
    return {{"threshold", &threshold}};
  }

  // dx = x > threshold ? dout : 0
  // Inputs: args[0], the input dout
  //         args[1], the input x
  __device__ __forceinline__ T operator()(const T* args) const {
    return args[1] > static_cast<T>(threshold) ? args[0] : zero;
  }

  static constexpr ActBwdOpFwdDeps FwdDeps() { return kDepX; }
};

template <typename T>
struct CudaHardSwishFunctor : public BaseActivationFunctor<T> {
  T zero = static_cast<T>(0.0f);
  float threshold;
  float scale;
  float offset;

  typename BaseActivationFunctor<T>::AttrPair GetAttrs() {
    return {{"threshold", &threshold}, {"scale", &scale}, {"offset", &offset}};
  }

  // hard_swish(x) = 0, when x <= -offset
  //                 x , when x >= threshold - offset
  //                 x * (x + offset) / scale, otherwise
  // threshold = scale = 6, offset = 3 by default
  // Inputs: args[0], the input x
  __device__ __forceinline__ T operator()(const T* args) const {
    T x = args[0];
    T t = static_cast<T>(threshold);
    T temp = x + static_cast<T>(offset);
    T temp_max = temp > zero ? temp : zero;
    T temp_min = temp_max < t ? temp_max : t;
    return temp_min * x / static_cast<T>(scale);
  }
};

template <typename T>
struct CudaHardSwishGradFunctor : public BaseActivationFunctor<T> {
  T zero = static_cast<T>(0.0f);
  T one = static_cast<T>(1.0f);
  T two = static_cast<T>(2.0f);
  float threshold;
  float scale;
  float offset;

  typename BaseActivationFunctor<T>::AttrPair GetAttrs() {
    return {{"threshold", &threshold}, {"scale", &scale}, {"offset", &offset}};
  }

  // dx = 0, when x <= -offset
  //      dout , when x >= threshold - offset
  //      dout * (2 * x / scale + offset / scale), otherwise
  // threshold = scale = 6, offset = 3 by default
  // Inputs: args[0], the input dout
  //         args[1], the input x
  __device__ __forceinline__ T operator()(const T* args) const {
    T x = args[1];
    T o = static_cast<T>(offset);
    T s = static_cast<T>(scale);
    T temp1 = static_cast<T>(x + o > zero);
    T temp2 = static_cast<T>(x + o < static_cast<T>(threshold));
    return args[0] * (temp1 * temp2 * (two * x + o) / s + one - temp2);
  }

  static constexpr ActBwdOpFwdDeps FwdDeps() { return kDepX; }
};

template <typename T>
struct CudaELUFunctor : public BaseActivationFunctor<T> {
  using CT = typename details::MPTypeTrait<T>::Type;
  CT zero = static_cast<CT>(0.0f);
  CT one = static_cast<CT>(1.0f);
  float alpha;

  typename BaseActivationFunctor<T>::AttrPair GetAttrs() {
    return {{"alpha", &alpha}};
  }

  // elu(x) = max(0, x) + min(0, alpha * (exp(x) - 1))
  // Inputs: args[0], the input x
  __device__ __forceinline__ T operator()(const T* args) const {
    CT x = static_cast<CT>(args[0]);
    CT temp = static_cast<CT>(alpha) * (exp(x) - one);
    CT res = (x > zero ? x : zero) + (temp > zero ? zero : temp);
    return static_cast<T>(res);
  }
};

template <typename T>
struct CudaELUGradFunctor : public BaseActivationFunctor<T> {
  using MPType = typename details::MPTypeTrait<T>::Type;
  MPType zero = static_cast<MPType>(0.0f);
  MPType one = static_cast<MPType>(1.0f);
  float alpha;

  typename BaseActivationFunctor<T>::AttrPair GetAttrs() {
    return {{"alpha", &alpha}};
  }

  // dx = dout, if alpha > 0 and x > 0
  // dx = dout * alpha * x.exp(), if alpha > 0 and x <= 0
  // dx = dout * (1 + alpha * x.exp()), if alpha <= 0 and x > 0
  // dx = 0, if alpha <= 0 and x <=0
  // Inputs: args[0], the input dout
  //         args[1], the input x
  __device__ __forceinline__ T operator()(const T* args) const {
    MPType dout = static_cast<MPType>(args[0]);
    MPType x = static_cast<MPType>(args[1]);
    MPType a = static_cast<MPType>(alpha);
    MPType temp_a_pos = static_cast<MPType>(alpha > 0.0f);
    MPType temp_a_neg = static_cast<MPType>(alpha <= 0.0f);
    MPType temp_x_pos = static_cast<MPType>(x > zero);
    MPType temp_x_neg = static_cast<MPType>(x <= zero);
    return static_cast<T>(
        dout * (temp_a_pos * temp_x_pos + temp_a_pos * temp_x_neg * a * exp(x) +
                temp_a_neg * temp_x_pos * (one + a * exp(x))));
  }

  static constexpr ActBwdOpFwdDeps FwdDeps() { return kDepX; }
};

template <typename DeviceContext, typename Functor>
class ActivationCudaKernel
    : public framework::OpKernel<typename Functor::ELEMENT_TYPE> {
 public:
  using T = typename Functor::ELEMENT_TYPE;
  void Compute(const framework::ExecutionContext& ctx) const override {
    const framework::Tensor* x = nullptr;
    framework::Tensor* out = nullptr;
    ExtractActivationTensor(ctx, &x, &out);
    out->mutable_data<T>(ctx.GetPlace());
    auto& dev_ctx = ctx.template device_context<DeviceContext>();
    std::vector<const framework::Tensor*> ins = {x};
    std::vector<framework::Tensor*> outs = {out};
    auto functor = Functor();
    auto attrs = functor.GetAttrs();
    for (auto& attr : attrs) {
      *attr.second = ctx.Attr<float>(attr.first);
    }
    LaunchSameDimsElementwiseCudaKernel<ElementwiseType::kUnary, T, T>(
        dev_ctx, ins, &outs, functor);
  }
};

template <typename DeviceContext, typename Functor>
class ActivationGradCudaKernel
    : public framework::OpKernel<typename Functor::ELEMENT_TYPE> {
 public:
  using T = typename Functor::ELEMENT_TYPE;
  void Compute(const framework::ExecutionContext& ctx) const override {
    const framework::Tensor *x, *out, *d_out;
    framework::Tensor* d_x = nullptr;
    x = out = d_out = nullptr;
    ExtractActivationGradTensor<Functor::FwdDeps()>(ctx, &x, &out, &d_out,
                                                    &d_x);
    d_x->mutable_data<T>(ctx.GetPlace());
    auto& dev_ctx = ctx.template device_context<DeviceContext>();
    auto functor = Functor();
    auto attrs = functor.GetAttrs();
    for (auto& attr : attrs) {
      *attr.second = ctx.Attr<float>(attr.first);
    }

    std::vector<const framework::Tensor*> ins = {d_out};
    std::vector<framework::Tensor*> outs = {d_x};

    if (static_cast<int>(Functor::FwdDeps()) == static_cast<int>(kDepOut)) {
      // Only need forward output Out
      ins.push_back(out);
      LaunchSameDimsElementwiseCudaKernel<ElementwiseType::kBinary, T, T>(
          dev_ctx, ins, &outs, functor);
    } else if (static_cast<int>(Functor::FwdDeps()) ==
               static_cast<int>(kDepX)) {
      // Only need forward input X
      ins.push_back(x);
      LaunchSameDimsElementwiseCudaKernel<ElementwiseType::kBinary, T, T>(
          dev_ctx, ins, &outs, functor);
    } else {
      LaunchSameDimsElementwiseCudaKernel<ElementwiseType::kUnary, T, T>(
          dev_ctx, ins, &outs, functor);
    }
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;

#define REGISTER_ACTIVATION_CUDA_KERNEL(act_type, op_name, functor,            \
                                        grad_functor)                          \
  REGISTER_OP_CUDA_KERNEL(                                                     \
      act_type, ops::ActivationCudaKernel<paddle::platform::CUDADeviceContext, \
                                          ops::functor<float>>,                \
      ops::ActivationCudaKernel<paddle::platform::CUDADeviceContext,           \
                                ops::functor<double>>,                         \
      ops::ActivationCudaKernel<plat::CUDADeviceContext,                       \
                                ops::functor<plat::float16>>);                 \
  REGISTER_OP_CUDA_KERNEL(                                                     \
      act_type##_grad,                                                         \
      ops::ActivationGradCudaKernel<plat::CUDADeviceContext,                   \
                                    ops::grad_functor<float>>,                 \
      ops::ActivationGradCudaKernel<plat::CUDADeviceContext,                   \
                                    ops::grad_functor<double>>,                \
      ops::ActivationGradCudaKernel<plat::CUDADeviceContext,                   \
                                    ops::grad_functor<plat::float16>>);

#define REGISTER_ACTIVATION_CUDA_KERNEL_INT(act_type, op_name, functor,        \
                                            grad_functor)                      \
  REGISTER_OP_CUDA_KERNEL(                                                     \
      act_type, ops::ActivationCudaKernel<paddle::platform::CUDADeviceContext, \
                                          ops::functor<float>>,                \
      ops::ActivationCudaKernel<paddle::platform::CUDADeviceContext,           \
                                ops::functor<double>>,                         \
      ops::ActivationCudaKernel<paddle::platform::CUDADeviceContext,           \
                                ops::functor<int>>,                            \
      ops::ActivationCudaKernel<paddle::platform::CUDADeviceContext,           \
                                ops::functor<int64_t>>,                        \
      ops::ActivationCudaKernel<plat::CUDADeviceContext,                       \
                                ops::functor<plat::float16>>);                 \
  REGISTER_OP_CUDA_KERNEL(                                                     \
      act_type##_grad,                                                         \
      ops::ActivationGradCudaKernel<plat::CUDADeviceContext,                   \
                                    ops::grad_functor<float>>,                 \
      ops::ActivationGradCudaKernel<plat::CUDADeviceContext,                   \
                                    ops::grad_functor<double>>,                \
      ops::ActivationGradCudaKernel<plat::CUDADeviceContext,                   \
                                    ops::grad_functor<int>>,                   \
      ops::ActivationGradCudaKernel<plat::CUDADeviceContext,                   \
                                    ops::grad_functor<int64_t>>,               \
      ops::ActivationGradCudaKernel<plat::CUDADeviceContext,                   \
                                    ops::grad_functor<plat::float16>>);

/* ======================== leaky relu register  ============================ */
REGISTER_ACTIVATION_CUDA_KERNEL(leaky_relu, LeakyRelu, CudaLeakyReluFunctor,
                                CudaLeakyReluGradFunctor);

REGISTER_OP_CUDA_KERNEL(
    leaky_relu_grad_grad,
    ops::ActivationDoubleGradKernel<plat::CUDADeviceContext,
                                    ops::LeakyReluGradGradFunctor<float>>,
    ops::ActivationDoubleGradKernel<plat::CUDADeviceContext,
                                    ops::LeakyReluGradGradFunctor<double>>,
    ops::ActivationDoubleGradKernel<
        plat::CUDADeviceContext, ops::LeakyReluGradGradFunctor<plat::float16>>);
/* ========================================================================== */

/* ======================== elu register  ============================ */
REGISTER_ACTIVATION_CUDA_KERNEL(elu, ELU, CudaELUFunctor, CudaELUGradFunctor);

REGISTER_OP_CUDA_KERNEL(
    elu_grad_grad, ops::ELUDoubleGradKernel<plat::CUDADeviceContext,
                                            ops::ELUGradGradFunctor<float>>,
    ops::ELUDoubleGradKernel<plat::CUDADeviceContext,
                             ops::ELUGradGradFunctor<double>>,
    ops::ELUDoubleGradKernel<plat::CUDADeviceContext,
                             ops::ELUGradGradFunctor<plat::float16>>);
/* ========================================================================== */

/* ===========================    relu register  ============================ */
#ifdef PADDLE_WITH_HIP
REGISTER_ACTIVATION_CUDA_KERNEL(relu, Relu, CudaReluFunctor,
                                CudaReluGradFunctor);
REGISTER_OP_CUDA_KERNEL(
    relu_grad_grad,
    ops::ActivationDoubleGradKernel<paddle::platform::CUDADeviceContext,
                                    ops::ReluGradGradFunctor<float>>,
    ops::ActivationDoubleGradKernel<paddle::platform::CUDADeviceContext,
                                    ops::ReluGradGradFunctor<double>>,
    ops::ActivationDoubleGradKernel<plat::CUDADeviceContext,
                                    ops::ReluGradGradFunctor<plat::float16>>);
#else
REGISTER_OP_CUDA_KERNEL(
    relu, ops::ActivationCudaKernel<paddle::platform::CUDADeviceContext,
                                    ops::CudaReluFunctor<float>>,
    ops::ActivationCudaKernel<paddle::platform::CUDADeviceContext,
                              ops::CudaReluFunctor<double>>,
    ops::ActivationCudaKernel<plat::CUDADeviceContext,
                              ops::CudaReluFunctor<plat::float16>>,
    ops::ActivationCudaKernel<plat::CUDADeviceContext,
                              ops::CudaReluFunctor<plat::bfloat16>>);
REGISTER_OP_CUDA_KERNEL(
    relu_grad, ops::ActivationGradCudaKernel<plat::CUDADeviceContext,
                                             ops::CudaReluGradFunctor<float>>,
    ops::ActivationGradCudaKernel<plat::CUDADeviceContext,
                                  ops::CudaReluGradFunctor<double>>,
    ops::ActivationGradCudaKernel<plat::CUDADeviceContext,
                                  ops::CudaReluGradFunctor<plat::float16>>,
    ops::ActivationGradCudaKernel<plat::CUDADeviceContext,
                                  ops::CudaReluGradFunctor<plat::bfloat16>>);
REGISTER_OP_CUDA_KERNEL(
    relu_grad_grad,
    ops::ActivationDoubleGradKernel<paddle::platform::CUDADeviceContext,
                                    ops::ReluGradGradFunctor<float>>,
    ops::ActivationDoubleGradKernel<paddle::platform::CUDADeviceContext,
                                    ops::ReluGradGradFunctor<double>>,
    ops::ActivationDoubleGradKernel<plat::CUDADeviceContext,
                                    ops::ReluGradGradFunctor<plat::float16>>,
    ops::ActivationDoubleGradKernel<plat::CUDADeviceContext,
                                    ops::ReluGradGradFunctor<plat::bfloat16>>);
#endif
/* ========================================================================== */

/* ===========================    sigmoid register  ============================
 */
REGISTER_ACTIVATION_CUDA_KERNEL(sigmoid, Sigmoid, CudaSigmoidFunctor,
                                CudaSigmoidGradFunctor);

REGISTER_OP_CUDA_KERNEL(
    sigmoid_grad_grad,
    ops::SigmoidDoubleGradKernel<paddle::platform::CUDADeviceContext,
                                 ops::SigmoidGradGradFunctor<float>>,
    ops::SigmoidDoubleGradKernel<paddle::platform::CUDADeviceContext,
                                 ops::SigmoidGradGradFunctor<double>>,
    ops::SigmoidDoubleGradKernel<plat::CUDADeviceContext,
                                 ops::SigmoidGradGradFunctor<plat::float16>>);
/* ========================================================================== */

/* ===========================    tanh register  ============================ */
REGISTER_ACTIVATION_CUDA_KERNEL(tanh, Tanh, CudaTanhFunctor,
                                CudaTanhGradFunctor);

REGISTER_OP_CUDA_KERNEL(
    tanh_grad_grad,
    ops::TanhDoubleGradKernel<paddle::platform::CUDADeviceContext,
                              ops::TanhGradGradFunctor<float>>,
    ops::TanhDoubleGradKernel<paddle::platform::CUDADeviceContext,
                              ops::TanhGradGradFunctor<double>>,
    ops::TanhDoubleGradKernel<plat::CUDADeviceContext,
                              ops::TanhGradGradFunctor<plat::float16>>);
/* ========================================================================== */

/* ===========================   sqrt register  ============================= */
REGISTER_ACTIVATION_CUDA_KERNEL(sqrt, Sqrt, CudaSqrtFunctor,
                                CudaSqrtGradFunctor);

REGISTER_OP_CUDA_KERNEL(
    sqrt_grad_grad,
    ops::SqrtDoubleGradKernel<paddle::platform::CUDADeviceContext,
                              ops::SqrtGradGradFunctor<float>>,
    ops::SqrtDoubleGradKernel<paddle::platform::CUDADeviceContext,
                              ops::SqrtGradGradFunctor<double>>,
    ops::SqrtDoubleGradKernel<paddle::platform::CUDADeviceContext,
                              ops::SqrtGradGradFunctor<plat::float16>>);
/* ========================================================================== */

/* ===========================   rsqrt register  =============================
 */
REGISTER_ACTIVATION_CUDA_KERNEL(rsqrt, Rsqrt, CudaRsqrtFunctor,
                                CudaRsqrtGradFunctor);

REGISTER_OP_CUDA_KERNEL(
    rsqrt_grad_grad,
    ops::RsqrtDoubleGradKernel<paddle::platform::CUDADeviceContext,
                               ops::RsqrtGradGradFunctor<float>>,
    ops::RsqrtDoubleGradKernel<paddle::platform::CUDADeviceContext,
                               ops::RsqrtGradGradFunctor<double>>,
    ops::RsqrtDoubleGradKernel<paddle::platform::CUDADeviceContext,
                               ops::RsqrtGradGradFunctor<plat::float16>>);
/* ========================================================================== */

/* ===========================  square register  ============================ */
REGISTER_ACTIVATION_CUDA_KERNEL_INT(square, Square, CudaSquareFunctor,
                                    CudaSquareGradFunctor);

REGISTER_OP_CUDA_KERNEL(
    square_grad_grad,
    ops::SquareDoubleGradKernel<paddle::platform::CUDADeviceContext,
                                ops::SquareGradGradFunctor<float>>,
    ops::SquareDoubleGradKernel<paddle::platform::CUDADeviceContext,
                                ops::SquareGradGradFunctor<double>>,
    ops::SquareDoubleGradKernel<plat::CUDADeviceContext,
                                ops::SquareGradGradFunctor<plat::float16>>,
    ops::SquareDoubleGradKernel<paddle::platform::CUDADeviceContext,
                                ops::SquareGradGradFunctor<int>>,
    ops::SquareDoubleGradKernel<paddle::platform::CUDADeviceContext,
                                ops::SquareGradGradFunctor<int64_t>>);
/* ========================================================================== */

/* ==========================   pow register  ============================ */
REGISTER_OP_CUDA_KERNEL(
    pow, ops::PowKernel<plat::CUDADeviceContext, ops::PowFunctor<float>>,
    ops::PowKernel<plat::CUDADeviceContext, ops::PowFunctor<double>>,
    ops::PowKernel<plat::CUDADeviceContext, ops::PowFunctor<int>>,
    ops::PowKernel<plat::CUDADeviceContext, ops::PowFunctor<int64_t>>,
    ops::PowKernel<plat::CUDADeviceContext, ops::PowFunctor<plat::float16>>);
REGISTER_OP_CUDA_KERNEL(
    pow_grad,
    ops::PowGradKernel<plat::CUDADeviceContext, ops::PowGradFunctor<float>>,
    ops::PowGradKernel<plat::CUDADeviceContext, ops::PowGradFunctor<double>>,
    ops::PowGradKernel<plat::CUDADeviceContext, ops::PowGradFunctor<int>>,
    ops::PowGradKernel<plat::CUDADeviceContext, ops::PowGradFunctor<int64_t>>,
    ops::PowGradKernel<plat::CUDADeviceContext,
                       ops::PowGradFunctor<plat::float16>>);
/* ========================================================================== */

/* ==========================   exp register  ============================ */
REGISTER_OP_CUDA_KERNEL(
    exp, ops::ActivationCudaKernel<plat::CUDADeviceContext,
                                   ops::CudaExpFunctor<float>>,
    ops::ActivationCudaKernel<plat::CUDADeviceContext,
                              ops::CudaExpFunctor<double>>,
    ops::ActivationKernel<plat::CUDADeviceContext, ops::ExpFunctor<int>>,
    ops::ActivationKernel<plat::CUDADeviceContext, ops::ExpFunctor<int64_t>>,
    ops::ActivationCudaKernel<plat::CUDADeviceContext,
                              ops::CudaExpFunctor<plat::float16>>);
REGISTER_OP_CUDA_KERNEL(
    exp_grad, ops::ActivationGradCudaKernel<plat::CUDADeviceContext,
                                            ops::CudaExpGradFunctor<float>>,
    ops::ActivationGradCudaKernel<plat::CUDADeviceContext,
                                  ops::CudaExpGradFunctor<double>>,
    ops::ActivationGradCudaKernel<plat::CUDADeviceContext,
                                  ops::CudaExpGradFunctor<int>>,
    ops::ActivationGradCudaKernel<plat::CUDADeviceContext,
                                  ops::CudaExpGradFunctor<int64_t>>,
    ops::ActivationGradCudaKernel<plat::CUDADeviceContext,
                                  ops::CudaExpGradFunctor<plat::float16>>);
/* ========================================================================== */

/* ==========================   expm1 register  ============================ */

REGISTER_OP_CUDA_KERNEL(
    expm1, ops::ActivationCudaKernel<plat::CUDADeviceContext,
                                     ops::CudaExpm1Functor<float>>,
    ops::ActivationCudaKernel<plat::CUDADeviceContext,
                              ops::CudaExpm1Functor<double>>,
    ops::ActivationCudaKernel<plat::CUDADeviceContext,
                              ops::CudaExpm1Functor<plat::float16>>);
REGISTER_OP_CUDA_KERNEL(
    expm1_grad, ops::ActivationGradCudaKernel<plat::CUDADeviceContext,
                                              ops::CudaExpm1GradFunctor<float>>,
    ops::ActivationGradCudaKernel<plat::CUDADeviceContext,
                                  ops::CudaExpm1GradFunctor<double>>,
    ops::ActivationGradCudaKernel<plat::CUDADeviceContext,
                                  ops::CudaExpm1GradFunctor<plat::float16>>);
/* ========================================================================== */

/* ==========================  Log register ==================================*/
REGISTER_ACTIVATION_CUDA_KERNEL(log, Log, CudaLogFunctor, CudaLogGradFunctor);

REGISTER_OP_CUDA_KERNEL(
    log_grad_grad, ops::LogDoubleGradKernel<plat::CUDADeviceContext,
                                            ops::LogGradGradFunctor<float>>,
    ops::LogDoubleGradKernel<plat::CUDADeviceContext,
                             ops::LogGradGradFunctor<double>>,
    ops::LogDoubleGradKernel<plat::CUDADeviceContext,
                             ops::LogGradGradFunctor<plat::float16>>);
/* ========================================================================== */

#define FOR_EACH_ACTIVATION_CUDA_OP(__macro)                                  \
  __macro(silu, Silu, CudaSiluFunctor, CudaSiluGradFunctor);                  \
  __macro(logsigmoid, LogSigmoid, CudaLogSigmoidFunctor,                      \
          CudaLogSigmoidGradFunctor);                                         \
  __macro(atan, Atan, CudaAtanFunctor, CudaAtanGradFunctor);                  \
  __macro(softshrink, SoftShrink, CudaSoftShrinkFunctor,                      \
          CudaSoftShrinkGradFunctor);                                         \
  __macro(ceil, Ceil, CudaCeilFunctor, CudaZeroGradFunctor);                  \
  __macro(floor, Floor, CudaFloorFunctor, CudaZeroGradFunctor);               \
  __macro(cos, Cos, CudaCosFunctor, CudaCosGradFunctor);                      \
  __macro(tan, Tan, CudaTanFunctor, CudaTanGradFunctor);                      \
  __macro(acos, Acos, CudaAcosFunctor, CudaAcosGradFunctor);                  \
  __macro(sin, Sin, CudaSinFunctor, CudaSinGradFunctor);                      \
  __macro(asin, Asin, CudaAsinFunctor, CudaAsinGradFunctor);                  \
  __macro(sinh, Sinh, CudaSinhFunctor, CudaSinhGradFunctor);                  \
  __macro(cosh, Cosh, CudaCoshFunctor, CudaCoshGradFunctor);                  \
  __macro(round, Round, CudaRoundFunctor, CudaZeroGradFunctor);               \
  __macro(reciprocal, Reciprocal, CudaReciprocalFunctor,                      \
          CudaReciprocalGradFunctor);                                         \
  __macro(log1p, Log1p, CudaLog1pFunctor, CudaLog1pGradFunctor);              \
  __macro(log2, Log2, CudaLog2Functor, CudaLog2GradFunctor);                  \
  __macro(log10, Log10, CudaLog10Functor, CudaLog10GradFunctor);              \
  __macro(brelu, BRelu, CudaBReluFunctor, CudaBReluGradFunctor);              \
  __macro(soft_relu, SoftRelu, CudaSoftReluFunctor, CudaSoftReluGradFunctor); \
  __macro(stanh, STanh, CudaSTanhFunctor, CudaSTanhGradFunctor);              \
  __macro(softplus, Softplus, CudaSoftplusFunctor, CudaSoftplusGradFunctor);  \
  __macro(softsign, Softsign, CudaSoftsignFunctor, CudaSoftsignGradFunctor);  \
  __macro(relu6, Relu6, CudaRelu6Functor, CudaRelu6GradFunctor);              \
  __macro(tanh_shrink, TanhShrink, CudaTanhShrinkFunctor,                     \
          CudaTanhShrinkGradFunctor);                                         \
  __macro(hard_shrink, HardShrink, CudaHardShrinkFunctor,                     \
          CudaHardShrinkGradFunctor);                                         \
  __macro(hard_sigmoid, HardSigmoid, CudaHardSigmoidFunctor,                  \
          CudaHardSigmoidGradFunctor);                                        \
  __macro(swish, Swish, CudaSwishFunctor, CudaSwishGradFunctor);              \
  __macro(thresholded_relu, ThresholdedRelu, CudaThresholdedReluFunctor,      \
          CudaThresholdedReluGradFunctor);                                    \
  __macro(hard_swish, HardSwish, CudaHardSwishFunctor,                        \
          CudaHardSwishGradFunctor);
FOR_EACH_ACTIVATION_CUDA_OP(REGISTER_ACTIVATION_CUDA_KERNEL)
