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

#include "paddle/fluid/operators/controlflow/bitwise_op.h"
#include "paddle/fluid/operators/elementwise/elementwise_op_broadcast.cu.h"

namespace paddle {
namespace operators {

#define BITWISE_BINARY_FUNCTOR(func, expr, bool_expr)    \
  template <typename T>                                  \
  struct Bitwise##func##CUDAFunctor {                    \
    using ELEM_TYPE = T;                                 \
    HOSTDEVICE T operator()(const T* args) const {       \
      return args[0] expr args[1];                       \
    }                                                    \
  };                                                     \
                                                         \
  template <>                                            \
  struct Bitwise##func##CUDAFunctor<bool> {              \
    using ELEM_TYPE = bool;                              \
    HOSTDEVICE bool operator()(const bool* args) const { \
      return args[0] bool_expr args[1];                  \
    }                                                    \
  };

BITWISE_BINARY_FUNCTOR(And, &, &&)
BITWISE_BINARY_FUNCTOR(Or, |, ||)
BITWISE_BINARY_FUNCTOR(Xor, ^, !=)
#undef BITWISE_BINARY_FUNCTOR

template <typename T>
struct BitwiseNotCUDAFunctor {
  using ELEM_TYPE = T;
  HOSTDEVICE T operator()(const T* args) const { return ~args[0]; }
};

template <>
struct BitwiseNotCUDAFunctor<bool> {
  using ELEM_TYPE = bool;
  HOSTDEVICE bool operator()(const bool* args) const { return !args[0]; }
};

template <typename Functor>
class BinaryBitwiseOpKernel<platform::CUDADeviceContext, Functor>
    : public framework::OpKernel<typename Functor::ELEM_TYPE> {
 public:
  using T = typename Functor::ELEM_TYPE;
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto functor = Functor();
    std::vector<const framework::Tensor*> ins;
    std::vector<framework::Tensor*> outs;
    const auto& cuda_ctx =
        ctx.template device_context<platform::CUDADeviceContext>();
    int axis = PackTensorsIntoVector<T>(ctx, &ins, &outs);

    if (ins.size() == 1) {
      LaunchElementwiseCudaKernel<ElementwiseType::kUnary, T, T>(
          cuda_ctx, ins, &outs, axis, functor);
    } else {
      LaunchElementwiseCudaKernel<ElementwiseType::kBinary, T, T>(
          cuda_ctx, ins, &outs, axis, functor);
    }
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = ::paddle::operators;
namespace plat = ::paddle::platform;

REGISTER_BINARY_BITWISE_KERNEL(bitwise_and, CUDA, ops::BitwiseAndCUDAFunctor);
REGISTER_BINARY_BITWISE_KERNEL(bitwise_or, CUDA, ops::BitwiseOrCUDAFunctor);
REGISTER_BINARY_BITWISE_KERNEL(bitwise_xor, CUDA, ops::BitwiseXorCUDAFunctor);
REGISTER_BINARY_BITWISE_KERNEL(bitwise_not, CUDA, ops::BitwiseNotCUDAFunctor);
