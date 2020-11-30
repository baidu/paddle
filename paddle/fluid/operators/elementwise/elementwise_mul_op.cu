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

#include "paddle/fluid/operators/elementwise/elementwise_mul_op.h"
#include "paddle/fluid/operators/elementwise/elementwise_op_function.cu.h"
#include "paddle/fluid/platform/complex128.h"
#include "paddle/fluid/platform/complex64.h"
#include "paddle/fluid/platform/float16.h"

namespace ops = paddle::operators;
namespace plat = paddle::platform;

namespace paddle {
namespace operators {

template <typename T>
struct SameDimsElemwiseMul<platform::CUDADeviceContext, T> {
  void operator()(const framework::ExecutionContext& ctx,
                  const framework::Tensor* x, const framework::Tensor* y,
                  framework::Tensor* z) {
    MulRangeFunctor<T> functor(x->data<T>(), y->data<T>(), z->data<T>());
    auto& dev_ctx = ctx.template device_context<platform::CUDADeviceContext>();
    platform::ForRange<platform::CUDADeviceContext> for_range(dev_ctx,
                                                              x->numel());
    for_range(functor);
  }
};

template <>
struct SameDimsElemwiseMul<platform::CUDADeviceContext, platform::float16> {
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
    SameDimsElemwiseMulCUDAKernel<<<
        grid_size, block_size, 0,
        ctx.template device_context<platform::CUDADeviceContext>().stream()>>>(
        x2, y2, z2, size);
  }
};

template <typename T>
static __global__ void SimpleElemwiseMulGradCUDAKernel(const T* x, const T* y,
                                                       const T* out,
                                                       const T* dout,
                                                       int64_t size, T* dx,
                                                       T* dy) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  while (col < size) {
    T o = dout[col];
    dx[col] = y[col] * o;
    dy[col] = x[col] * o;
    col += blockDim.x * gridDim.x;
  }
}

template <typename DeviceContext, typename T>
typename std::enable_if<
    std::is_same<DeviceContext, plat::CUDADeviceContext>::value>::type
elementwise_mul_grad(const framework::ExecutionContext& ctx,
                     const framework::Tensor* x, const framework::Tensor* y,
                     const framework::Tensor* out,
                     const framework::Tensor* dout, framework::Tensor* dx,
                     framework::Tensor* dy) {
  dim3 block_size = dim3(PADDLE_CUDA_THREAD_SIZE, 1);
  auto size = x->numel();
  dim3 grid_size =
      dim3((size + PADDLE_CUDA_THREAD_SIZE - 1) / PADDLE_CUDA_THREAD_SIZE, 1);
  SimpleElemwiseMulGradCUDAKernel<
      T><<<grid_size, block_size, 0,
           ctx.template device_context<plat::CUDADeviceContext>().stream()>>>(
      x->data<T>(), y->data<T>(), out->data<T>(), dout->data<T>(), size,
      dx->mutable_data<T>(ctx.GetPlace()), dy->mutable_data<T>(ctx.GetPlace()));
}

}  // namespace operators
}  // namespace paddle

REGISTER_OP_CUDA_KERNEL(
    elementwise_mul, ops::ElementwiseMulKernel<plat::CUDADeviceContext, float>,
    ops::ElementwiseMulKernel<plat::CUDADeviceContext,
                              paddle::platform::complex64>,
    ops::ElementwiseMulKernel<plat::CUDADeviceContext,
                              paddle::platform::complex128>,
    ops::ElementwiseMulKernel<plat::CUDADeviceContext, double>,
    ops::ElementwiseMulKernel<plat::CUDADeviceContext, int>,
    ops::ElementwiseMulKernel<plat::CUDADeviceContext, int64_t>,
    ops::ElementwiseMulKernel<plat::CUDADeviceContext, plat::float16>);
REGISTER_OP_CUDA_KERNEL(
    elementwise_mul_grad,
    ops::ElementwiseMulGradKernel<plat::CUDADeviceContext,
                                  paddle::platform::complex64>,
    ops::ElementwiseMulGradKernel<plat::CUDADeviceContext,
                                  paddle::platform::complex128>,
    ops::ElementwiseMulGradKernel<plat::CUDADeviceContext, float>,
    ops::ElementwiseMulGradKernel<plat::CUDADeviceContext, double>,
    ops::ElementwiseMulGradKernel<plat::CUDADeviceContext, int>,
    ops::ElementwiseMulGradKernel<plat::CUDADeviceContext, int64_t>,
    ops::ElementwiseMulGradKernel<plat::CUDADeviceContext, plat::float16>);
REGISTER_OP_CUDA_KERNEL(
    elementwise_mul_grad_grad,
    ops::ElementwiseMulDoubleGradKernel<plat::CUDADeviceContext,
                                        paddle::platform::complex64>,
    ops::ElementwiseMulDoubleGradKernel<plat::CUDADeviceContext,
                                        paddle::platform::complex128>,
    ops::ElementwiseMulDoubleGradKernel<plat::CUDADeviceContext, float>,
    ops::ElementwiseMulDoubleGradKernel<plat::CUDADeviceContext, double>,
    ops::ElementwiseMulDoubleGradKernel<plat::CUDADeviceContext, int>,
    ops::ElementwiseMulDoubleGradKernel<plat::CUDADeviceContext, int64_t>,
    ops::ElementwiseMulDoubleGradKernel<plat::CUDADeviceContext,
                                        plat::float16>);
