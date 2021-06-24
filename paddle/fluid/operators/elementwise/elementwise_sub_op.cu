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
#include "paddle/fluid/operators/elementwise/elementwise_op_broadcast.cu.h"
#include "paddle/fluid/operators/elementwise/elementwise_sub_op.h"
#include "paddle/fluid/platform/complex.h"
#include "paddle/fluid/platform/float16.h"

namespace ops = paddle::operators;
namespace plat = paddle::platform;

namespace paddle {
namespace operators {

template <typename T>
struct CudaSubFunctor {
  inline HOSTDEVICE T operator()(const T* args) const {
    return args[0] - args[1];
  }
};

template <typename T>
class ElementwiseSubKernel<platform::CUDADeviceContext, T>
    : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    std::vector<const framework::Tensor*> ins;
    std::vector<framework::Tensor*> outs;
    const auto& cuda_ctx =
        ctx.template device_context<platform::CUDADeviceContext>();

    int axis = PackTensorsIntoVector<T>(ctx, &ins, &outs);
    LaunchElementwiseCudaKernel<ElementwiseType::kBinary, T, T>(
        cuda_ctx, ins, &outs, axis, CudaSubFunctor<T>());
  }
};

template <typename T>
static __global__ void SimpleElemwiseSubGradCUDAKernel(const T* dout,
                                                       int64_t size, T* dx,
                                                       T* dy) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  while (col < size) {
    dx[col] = dout[col];
    dy[col] = -dout[col];
    col += blockDim.x * gridDim.x;
  }
}

template <typename DeviceContext, typename T>
typename std::enable_if<
    std::is_same<DeviceContext, plat::CUDADeviceContext>::value>::type
elementwise_sub_grad(const framework::ExecutionContext& ctx,
                     const framework::Tensor* x, const framework::Tensor* y,
                     const framework::Tensor* out,
                     const framework::Tensor* dout, framework::Tensor* dx,
                     framework::Tensor* dy) {
  dim3 block_size = dim3(PADDLE_CUDA_THREAD_SIZE, 1);
  auto size = x->numel();
  dim3 grid_size =
      dim3((size + PADDLE_CUDA_THREAD_SIZE - 1) / PADDLE_CUDA_THREAD_SIZE, 1);
  SimpleElemwiseSubGradCUDAKernel<
      T><<<grid_size, block_size, 0,
           ctx.template device_context<plat::CUDADeviceContext>().stream()>>>(
      dout->data<T>(), size, dx->mutable_data<T>(ctx.GetPlace()),
      dy->mutable_data<T>(ctx.GetPlace()));
}

}  // namespace operators
}  // namespace paddle

REGISTER_OP_CUDA_KERNEL(
    elementwise_sub,
    ops::ElementwiseSubKernel<paddle::platform::CUDADeviceContext, float>,
    ops::ElementwiseSubKernel<paddle::platform::CUDADeviceContext,
                              paddle::platform::float16>,
    ops::ElementwiseSubKernel<paddle::platform::CUDADeviceContext, double>,
    ops::ElementwiseSubKernel<paddle::platform::CUDADeviceContext, int>,
    ops::ElementwiseSubKernel<paddle::platform::CUDADeviceContext, int64_t>,
    ops::ElementwiseSubKernel<paddle::platform::CUDADeviceContext,
                              paddle::platform::complex<float>>,
    ops::ElementwiseSubKernel<paddle::platform::CUDADeviceContext,
                              paddle::platform::complex<double>>);
REGISTER_OP_CUDA_KERNEL(
    elementwise_sub_grad,
    ops::ElementwiseSubGradKernel<paddle::platform::CUDADeviceContext, float>,
    ops::ElementwiseSubGradKernel<paddle::platform::CUDADeviceContext,
                                  paddle::platform::float16>,
    ops::ElementwiseSubGradKernel<paddle::platform::CUDADeviceContext, double>,
    ops::ElementwiseSubGradKernel<paddle::platform::CUDADeviceContext, int>,
    ops::ElementwiseSubGradKernel<paddle::platform::CUDADeviceContext, int64_t>,
    ops::ElementwiseSubGradKernel<paddle::platform::CUDADeviceContext,
                                  paddle::platform::complex<float>>,
    ops::ElementwiseSubGradKernel<paddle::platform::CUDADeviceContext,
                                  paddle::platform::complex<double>>);
REGISTER_OP_CUDA_KERNEL(
    elementwise_sub_grad_grad,
    ops::ElementwiseSubDoubleGradKernel<paddle::platform::CUDADeviceContext,
                                        float>,
    ops::ElementwiseSubDoubleGradKernel<paddle::platform::CUDADeviceContext,
                                        double>,
    ops::ElementwiseSubDoubleGradKernel<paddle::platform::CUDADeviceContext,
                                        int>,
    ops::ElementwiseSubDoubleGradKernel<paddle::platform::CUDADeviceContext,
                                        int64_t>,
    ops::ElementwiseSubDoubleGradKernel<paddle::platform::CUDADeviceContext,
                                        paddle::platform::complex<float>>,
    ops::ElementwiseSubDoubleGradKernel<paddle::platform::CUDADeviceContext,
                                        paddle::platform::complex<double>>);
