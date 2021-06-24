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
#include "paddle/fluid/operators/elementwise/elementwise_max_op.h"
#include "paddle/fluid/operators/elementwise/elementwise_op_broadcast.cu.h"

namespace ops = paddle::operators;

namespace paddle {
namespace operators {

template <typename T>
struct CudaMaxFunctor {
  inline HOSTDEVICE T operator()(const T* args) const {
    return (args[0] > args[1] ? args[0] : args[1]);
  }
};

template <typename T>
class ElementwiseMaxKernel<platform::CUDADeviceContext, T>
    : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    std::vector<const framework::Tensor*> ins;
    std::vector<framework::Tensor*> outs;
    const auto& cuda_ctx =
        ctx.template device_context<platform::CUDADeviceContext>();

    int axis = PackTensorsIntoVector<T>(ctx, &ins, &outs);
    LaunchElementwiseCudaKernel<ElementwiseType::kBinary, T, T>(
        cuda_ctx, ins, &outs, axis, CudaMaxFunctor<T>());
  }
};

}  // namespace operators
}  // namespace paddle

REGISTER_OP_CUDA_KERNEL(
    elementwise_max,
    ops::ElementwiseMaxKernel<paddle::platform::CUDADeviceContext, float>,
    ops::ElementwiseMaxKernel<paddle::platform::CUDADeviceContext, double>,
    ops::ElementwiseMaxKernel<paddle::platform::CUDADeviceContext, int>,
    ops::ElementwiseMaxKernel<paddle::platform::CUDADeviceContext, int64_t>);
REGISTER_OP_CUDA_KERNEL(
    elementwise_max_grad,
    ops::ElementwiseMaxGradKernel<paddle::platform::CUDADeviceContext, float>,
    ops::ElementwiseMaxGradKernel<paddle::platform::CUDADeviceContext, double>,
    ops::ElementwiseMaxGradKernel<paddle::platform::CUDADeviceContext, int>,
    ops::ElementwiseMaxGradKernel<paddle::platform::CUDADeviceContext,
                                  int64_t>);
