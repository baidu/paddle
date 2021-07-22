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

#include "paddle/fluid/operators/controlflow/logical_op.h"
#include "paddle/fluid/operators/elementwise/elementwise_op_broadcast.cu.h"

namespace ops = paddle::operators;
namespace plat = paddle::platform;

namespace paddle {
namespace operators {

#define LOGICAL_BINARY_FUNCTOR(func_name, op)                          \
  template <typename T>                                                \
  struct func_name {                                                   \
    using ELEMENT_TYPE = T;                                            \
    HOSTDEVICE bool operator()(const T* args) const {                  \
      return static_cast<bool>(args[0]) op static_cast<bool>(args[1]); \
    }                                                                  \
  };

LOGICAL_BINARY_FUNCTOR(CudaOrFunctor, ||)
LOGICAL_BINARY_FUNCTOR(CudaAndFunctor, &&)
LOGICAL_BINARY_FUNCTOR(CudaXorFunctor, ^)
#undef LOGICAL_BINARY_FUNCTOR

template <typename T>
struct CudaNotFunctor {
  using ELEMENT_TYPE = T;
  HOSTDEVICE bool operator()(const T* args) const { return !args[0]; }
};

template <typename Functor>
class BinaryLogicalOpKernel<platform::CUDADeviceContext, Functor>
    : public framework::OpKernel<typename Functor::ELEMENT_TYPE> {
 public:
  using InT = typename Functor::ELEMENT_TYPE;
  using OutT = bool;
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto functor = Functor();
    std::vector<const framework::Tensor*> ins;
    std::vector<framework::Tensor*> outs;
    const auto& cuda_ctx =
        ctx.template device_context<platform::CUDADeviceContext>();
    int axis = PackTensorsIntoVector<OutT>(ctx, &ins, &outs);

    if (ins.size() == 1) {
      LaunchElementwiseCudaKernel<ElementwiseType::kUnary, InT, OutT>(
          cuda_ctx, ins, &outs, axis, functor);
    } else {
      LaunchElementwiseCudaKernel<ElementwiseType::kBinary, InT, OutT>(
          cuda_ctx, ins, &outs, axis, functor);
    }
  }
};

}  // namespace operators
}  // namespace paddle

#define REGISTER_LOGICAL_CUDA_KERNEL(op_name, func)                            \
  REGISTER_OP_CUDA_KERNEL(                                                     \
      op_name,                                                                 \
      ops::BinaryLogicalOpKernel<plat::CUDADeviceContext, ops::func<bool>>,    \
      ops::BinaryLogicalOpKernel<plat::CUDADeviceContext, ops::func<int8_t>>,  \
      ops::BinaryLogicalOpKernel<plat::CUDADeviceContext, ops::func<int16_t>>, \
      ops::BinaryLogicalOpKernel<plat::CUDADeviceContext, ops::func<int>>,     \
      ops::BinaryLogicalOpKernel<plat::CUDADeviceContext, ops::func<int64_t>>, \
      ops::BinaryLogicalOpKernel<plat::CUDADeviceContext, ops::func<float>>,   \
      ops::BinaryLogicalOpKernel<plat::CUDADeviceContext, ops::func<double>>);

REGISTER_LOGICAL_CUDA_KERNEL(logical_or, CudaOrFunctor)
REGISTER_LOGICAL_CUDA_KERNEL(logical_and, CudaAndFunctor)
REGISTER_LOGICAL_CUDA_KERNEL(logical_xor, CudaXorFunctor)
REGISTER_LOGICAL_CUDA_KERNEL(logical_not, CudaNotFunctor)
#undef REGISTER_LOGICAL_CUDA_KERNEL
