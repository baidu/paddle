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
#include "paddle/fluid/operators/elementwise/elementwise_add_op.h"
#include "paddle/fluid/operators/elementwise/elementwise_op_function.cu.h"
#include "paddle/fluid/platform/complex128.h"
#include "paddle/fluid/platform/complex64.h"
#include "paddle/fluid/platform/float16.h"

namespace ops = paddle::operators;
namespace plat = paddle::platform;

namespace paddle {
namespace operators {

template <typename T>
struct SameDimsElemwiseAdd<platform::CUDADeviceContext, T> {
  void operator()(const framework::ExecutionContext& ctx,
                  const framework::Tensor* x, const framework::Tensor* y,
                  framework::Tensor* z) {
    AddRangeFunctor<T> functor(x->data<T>(), y->data<T>(), z->data<T>());
    auto& dev_ctx = ctx.template device_context<platform::CUDADeviceContext>();
    platform::ForRange<platform::CUDADeviceContext> for_range(dev_ctx,
                                                              x->numel());
    for_range(functor);
  }
};

template <>
struct SameDimsElemwiseAdd<platform::CUDADeviceContext, platform::float16> {
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
    SameDimsElemwiseAddCUDAKernel<<<
        grid_size, block_size, 0,
        ctx.template device_context<platform::CUDADeviceContext>().stream()>>>(
        x2, y2, z2, size);
  }
};

template <typename T>
static __global__ void SimpleElemwiseAddGradCUDAKernel(const T* dout,
                                                       int64_t size, T* dx,
                                                       T* dy) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  while (col < size) {
    dx[col] = dout[col];
    dy[col] = dout[col];
    col += blockDim.x * gridDim.x;
  }
}

template <typename DeviceContext, typename T>
typename std::enable_if<
    std::is_same<DeviceContext, plat::CUDADeviceContext>::value>::type
elementwise_add_grad(const framework::ExecutionContext& ctx,
                     const framework::Tensor* x, const framework::Tensor* y,
                     const framework::Tensor* out,
                     const framework::Tensor* dout, framework::Tensor* dx,
                     framework::Tensor* dy) {
  dim3 block_size = dim3(PADDLE_CUDA_THREAD_SIZE, 1);
  auto size = x->numel();
  dim3 grid_size =
      dim3((size + PADDLE_CUDA_THREAD_SIZE - 1) / PADDLE_CUDA_THREAD_SIZE, 1);
  SimpleElemwiseAddGradCUDAKernel<
      T><<<grid_size, block_size, 0,
           ctx.template device_context<plat::CUDADeviceContext>().stream()>>>(
      dout->data<T>(), size, dx->mutable_data<T>(ctx.GetPlace()),
      dy->mutable_data<T>(ctx.GetPlace()));
}

}  // namespace operators
}  // namespace paddle
REGISTER_OP_CUDA_KERNEL(
    elementwise_add, ops::ElementwiseAddKernel<plat::CUDADeviceContext, float>,
    ops::ElementwiseAddKernel<plat::CUDADeviceContext,
                              paddle::platform::complex64>,
    ops::ElementwiseAddKernel<plat::CUDADeviceContext,
                              paddle::platform::complex128>,
    ops::ElementwiseAddKernel<plat::CUDADeviceContext, double>,
    ops::ElementwiseAddKernel<plat::CUDADeviceContext, int>,
    ops::ElementwiseAddKernel<plat::CUDADeviceContext, int64_t>,
    ops::ElementwiseAddKernel<plat::CUDADeviceContext, plat::float16>);
REGISTER_OP_CUDA_KERNEL(
    elementwise_add_grad,
    ops::ElementwiseAddGradKernel<plat::CUDADeviceContext,
                                  paddle::platform::complex64>,
    ops::ElementwiseAddGradKernel<plat::CUDADeviceContext,
                                  paddle::platform::complex128>,
    ops::ElementwiseAddGradKernel<plat::CUDADeviceContext, float>,
    ops::ElementwiseAddGradKernel<plat::CUDADeviceContext, double>,
    ops::ElementwiseAddGradKernel<plat::CUDADeviceContext, int>,
    ops::ElementwiseAddGradKernel<plat::CUDADeviceContext, int64_t>,
    ops::ElementwiseAddGradKernel<plat::CUDADeviceContext, plat::float16>);
REGISTER_OP_CUDA_KERNEL(
    elementwise_add_grad_grad,
    ops::ElementwiseAddDoubleGradKernel<plat::CUDADeviceContext, float>,
    ops::ElementwiseAddDoubleGradKernel<plat::CUDADeviceContext,
                                        paddle::platform::complex64>,
    ops::ElementwiseAddDoubleGradKernel<plat::CUDADeviceContext,
                                        paddle::platform::complex128>,
    ops::ElementwiseAddDoubleGradKernel<plat::CUDADeviceContext, double>,
    ops::ElementwiseAddDoubleGradKernel<plat::CUDADeviceContext, int>,
    ops::ElementwiseAddDoubleGradKernel<plat::CUDADeviceContext, int64_t>,
    ops::ElementwiseAddDoubleGradKernel<plat::CUDADeviceContext,
                                        plat::float16>);

REGISTER_OP_CUDA_KERNEL(
    grad_add, ops::ElementwiseAddKernel<plat::CUDADeviceContext, float>,
    ops::ElementwiseAddKernel<plat::CUDADeviceContext,
                              paddle::platform::complex64>,
    ops::ElementwiseAddKernel<plat::CUDADeviceContext,
                              paddle::platform::complex128>,
    ops::ElementwiseAddKernel<plat::CUDADeviceContext, double>,
    ops::ElementwiseAddKernel<plat::CUDADeviceContext, int>,
    ops::ElementwiseAddKernel<plat::CUDADeviceContext, int64_t>,
    ops::ElementwiseAddKernel<plat::CUDADeviceContext, plat::float16>);
