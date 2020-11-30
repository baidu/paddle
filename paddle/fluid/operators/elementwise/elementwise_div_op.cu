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
#include "paddle/fluid/operators/elementwise/elementwise_div_op.h"
#include "paddle/fluid/operators/elementwise/elementwise_op_function.cu.h"
#include "paddle/fluid/operators/elementwise/elementwise_op_function.h"
#include "paddle/fluid/platform/complex128.h"
#include "paddle/fluid/platform/complex64.h"
#include "paddle/fluid/platform/float16.h"

namespace ops = paddle::operators;
namespace plat = paddle::platform;

namespace paddle {
namespace operators {

template <typename T>
struct SameDimsElemwiseDiv<platform::CUDADeviceContext, T> {
  void operator()(const framework::ExecutionContext& ctx,
                  const framework::Tensor* x, const framework::Tensor* y,
                  framework::Tensor* z) {
    DivRangeFunctor<T> functor(x->data<T>(), y->data<T>(), z->data<T>());
    auto& dev_ctx = ctx.template device_context<platform::CUDADeviceContext>();
    platform::ForRange<platform::CUDADeviceContext> for_range(dev_ctx,
                                                              x->numel());
    for_range(functor);
  }
};

template <>
struct SameDimsElemwiseDiv<platform::CUDADeviceContext, platform::float16> {
  void operator()(const framework::ExecutionContext& ctx,
                  const framework::Tensor* x, const framework::Tensor* y,
                  framework::Tensor* z) {
    auto size = x->numel();
    dim3 grid_size = dim3(((size + 1) / 2 + PADDLE_CUDA_THREAD_SIZE - 1) /
                              PADDLE_CUDA_THREAD_SIZE,
                          1);
    dim3 block_size = dim3(PADDLE_CUDA_THREAD_SIZE, 1);
    const half* x2 =
        reinterpret_cast<const half*>(x->data<platform::float16>());
    const half* y2 =
        reinterpret_cast<const half*>(y->data<platform::float16>());
    half* z2 = reinterpret_cast<half*>(z->data<platform::float16>());
    SameDimsElemwiseDivCUDAKernel<<<
        grid_size, block_size, 0,
        ctx.template device_context<platform::CUDADeviceContext>().stream()>>>(
        x2, y2, z2, size);
  }
};

template <typename T>
static __global__ void SimpleElemwiseDivGradCUDAKernel(const T* x, const T* y,
                                                       const T* out,
                                                       const T* dout,
                                                       int64_t size, T* dx,
                                                       T* dy) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  while (col < size) {
    T o = dout[col];
    dx[col] = o / y[col];
    dy[col] = -o * out[col] / y[col];
    col += blockDim.x * gridDim.x;
  }
}

template <typename DeviceContext, typename T>
typename std::enable_if<
    std::is_same<DeviceContext, plat::CUDADeviceContext>::value>::type
elementwise_div_grad(const framework::ExecutionContext& ctx,
                     const framework::Tensor* x, const framework::Tensor* y,
                     const framework::Tensor* out,
                     const framework::Tensor* dout, framework::Tensor* dx,
                     framework::Tensor* dy) {
  dim3 block_size = dim3(PADDLE_CUDA_THREAD_SIZE, 1);
  auto size = x->numel();
  dim3 grid_size =
      dim3((size + PADDLE_CUDA_THREAD_SIZE - 1) / PADDLE_CUDA_THREAD_SIZE, 1);
  SimpleElemwiseDivGradCUDAKernel<
      T><<<grid_size, block_size, 0,
           ctx.template device_context<plat::CUDADeviceContext>().stream()>>>(
      x->data<T>(), y->data<T>(), out->data<T>(), dout->data<T>(), size,
      dx->mutable_data<T>(ctx.GetPlace()), dy->mutable_data<T>(ctx.GetPlace()));
}

}  // namespace operators
}  // namespace paddle

REGISTER_OP_CUDA_KERNEL(
    elementwise_div,
    ops::ElementwiseDivKernel<paddle::platform::CUDADeviceContext, float>,
    ops::ElementwiseDivKernel<paddle::platform::CUDADeviceContext,
                              paddle::platform::complex64>,
    ops::ElementwiseDivKernel<paddle::platform::CUDADeviceContext,
                              paddle::platform::complex128>,
    ops::ElementwiseDivKernel<paddle::platform::CUDADeviceContext,
                              paddle::platform::float16>,
    ops::ElementwiseDivKernel<paddle::platform::CUDADeviceContext, double>,
    ops::ElementwiseDivKernel<paddle::platform::CUDADeviceContext, int>,
    ops::ElementwiseDivKernel<paddle::platform::CUDADeviceContext, int64_t>);
REGISTER_OP_CUDA_KERNEL(
    elementwise_div_grad,
    ops::ElementwiseDivGradKernel<paddle::platform::CUDADeviceContext, float>,
    ops::ElementwiseDivGradKernel<paddle::platform::CUDADeviceContext,
                                  paddle::platform::complex64>,
    ops::ElementwiseDivGradKernel<paddle::platform::CUDADeviceContext,
                                  paddle::platform::complex128>,
    ops::ElementwiseDivGradKernel<paddle::platform::CUDADeviceContext,
                                  paddle::platform::float16>,
    ops::ElementwiseDivGradKernel<paddle::platform::CUDADeviceContext, double>,
    ops::ElementwiseDivGradKernel<paddle::platform::CUDADeviceContext, int>,
    ops::ElementwiseDivGradKernel<paddle::platform::CUDADeviceContext,
                                  int64_t>);
REGISTER_OP_CUDA_KERNEL(
    elementwise_div_grad_grad,
    ops::ElementwiseDivDoubleGradKernel<paddle::platform::CUDADeviceContext,
                                        paddle::platform::complex64>,
    ops::ElementwiseDivDoubleGradKernel<paddle::platform::CUDADeviceContext,
                                        paddle::platform::complex128>,
    ops::ElementwiseDivDoubleGradKernel<paddle::platform::CUDADeviceContext,
                                        float>,
    ops::ElementwiseDivDoubleGradKernel<paddle::platform::CUDADeviceContext,
                                        paddle::platform::float16>,
    ops::ElementwiseDivDoubleGradKernel<paddle::platform::CUDADeviceContext,
                                        double>,
    ops::ElementwiseDivDoubleGradKernel<paddle::platform::CUDADeviceContext,
                                        int>,
    ops::ElementwiseDivDoubleGradKernel<paddle::platform::CUDADeviceContext,
                                        int64_t>);
