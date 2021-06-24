/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */
#include "paddle/fluid/operators/elementwise/elementwise_floordiv_op.h"
#include "paddle/fluid/operators/elementwise/elementwise_op_broadcast.cu.h"

namespace ops = paddle::operators;
namespace plat = paddle::platform;

namespace paddle {
namespace operators {

template <typename T>
struct CudaFloorDivFunctor {
  inline HOSTDEVICE T operator()(const T argv[]) const {
    PADDLE_ENFORCE(argv[1] != 0,
                   "InvalidArgument: divide by zero "
                   "encountered in floor-divide ops, please check.\n");
    return static_cast<T>(std::trunc(argv[0] / argv[1]));
  }
};

template <typename T>
class ElementwiseFloorDivKernel<platform::CUDADeviceContext, T>
    : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    std::vector<const framework::Tensor*> ins;
    std::vector<framework::Tensor*> outs;
    const auto& cuda_ctx =
        ctx.template device_context<platform::CUDADeviceContext>();

    int axis = PackTensorsIntoVector<T>(ctx, &ins, &outs);
    LaunchElementwiseCudaKernel<ElementwiseType::kBinary, T, T>(
        cuda_ctx, ins, &outs, axis, CudaFloorDivFunctor<T>());
  }
};

}  // namespace operators
}  // namespace paddle

REGISTER_OP_CUDA_KERNEL(
    elementwise_floordiv,
    ops::ElementwiseFloorDivKernel<plat::CUDADeviceContext, int>,
    ops::ElementwiseFloorDivKernel<plat::CUDADeviceContext, int64_t>);
