/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */
#include "paddle/fluid/operators/elementwise/elementwise_op_broadcast.cu.h"
#include "paddle/fluid/operators/elementwise/elementwise_pow_op.h"

namespace ops = paddle::operators;

namespace paddle {
namespace operators {

template <typename T, typename Enable = void>
struct CudaPowFunctor {
  inline HOSTDEVICE T operator()(const T args[]) const {
    return std::pow(args[0], args[1]);
  }
};

template <typename T>
struct CudaPowFunctor<
    T, typename std::enable_if<std::is_integral<T>::value>::type> {
  // On CUDAPlace, std::pow(3, 1) calls pow(float, float), and
  // it will return a float number like 2.99... , which floor to 2
  // when cast to int by default and it is wrong.
  // Use llrint to cast it to the nearest integer, which is 3.
  inline HOSTDEVICE T operator()(const T args[]) const {
    return std::llrint(std::pow(args[0], args[1]));
  }
};

template <typename T>
class ElementwisePowKernel<platform::CUDADeviceContext, T>
    : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    std::vector<const framework::Tensor*> ins;
    std::vector<framework::Tensor*> outs;
    const auto& cuda_ctx =
        ctx.template device_context<platform::CUDADeviceContext>();

    int axis = PackTensorsIntoVector<T>(ctx, &ins, &outs);
    LaunchElementwiseCudaKernel<ElementwiseType::kBinary, T, T>(
        cuda_ctx, ins, &outs, axis, CudaPowFunctor<T>());
  }
};

}  // namespace operators
}  // namespace paddle

REGISTER_OP_CUDA_KERNEL(
    elementwise_pow,
    ops::ElementwisePowKernel<paddle::platform::CUDADeviceContext, float>,
    ops::ElementwisePowKernel<paddle::platform::CUDADeviceContext, double>,
    ops::ElementwisePowKernel<paddle::platform::CUDADeviceContext, int>,
    ops::ElementwisePowKernel<paddle::platform::CUDADeviceContext, int64_t>);
REGISTER_OP_CUDA_KERNEL(
    elementwise_pow_grad,
    ops::ElementwisePowGradKernel<paddle::platform::CUDADeviceContext, float>,
    ops::ElementwisePowGradKernel<paddle::platform::CUDADeviceContext, double>,
    ops::ElementwisePowGradKernel<paddle::platform::CUDADeviceContext, int>,
    ops::ElementwisePowGradKernel<paddle::platform::CUDADeviceContext,
                                  int64_t>);
