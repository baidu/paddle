/* Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/operators/mv_op.h"
#include "paddle/fluid/platform/gpu_launch_param_config.h"

namespace paddle {
namespace operators {

template <typename T>
__global__ void MVGradDxCUDAKernel(const int m, const int n, const T *dout,
                                   const T *vec, T *dx) {
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  for (; idx < m * n; idx += blockDim.x * gridDim.x) {
    int i = idx / n;
    int j = idx % n;
    dx[idx] = dout[i] * vec[j];
  }
}

// Using dimensional constraints on matrix multiplication, it is
// straight-forward to check the following table for when X and Y
// are both matrices.
//
// dX = | dOut Vec^T
// dVec = | X^T dOut
template <typename T>
class MVGradKernel<platform::CUDADeviceContext, T>
    : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &context) const override {
    auto *x = context.Input<framework::Tensor>("X");
    auto *vec = context.Input<framework::Tensor>("Vec");
    auto *dout =
        context.Input<framework::Tensor>(framework::GradVarName("Out"));
    auto *dx = context.Output<framework::Tensor>(framework::GradVarName("X"));
    auto *dvec =
        context.Output<framework::Tensor>(framework::GradVarName("Vec"));

    auto dim_x = x->dims();
    int m = dim_x[0];
    int n = dim_x[1];

    // get data ptr
    const T *x_data = x->data<T>();
    const T *vec_data = vec->data<T>();
    const T *dout_data = dout->data<T>();

    auto &dev_ctx =
        context.template device_context<platform::CUDADeviceContext>();
    auto blas = math::GetBlas<platform::CUDADeviceContext, T>(dev_ctx);
    auto stream = context.cuda_device_context().stream();
    auto config = GetGpuLaunchConfig1D(dev_ctx, m * n);

    if (dx) {
      T *dx_data = dx->mutable_data<T>(context.GetPlace());

      MVGradDxCUDAKernel<
          T><<<config.block_per_grid.x, config.thread_per_block.x, 0, stream>>>(
          m, n, dout_data, vec_data, dx_data);
    }

    if (dvec) {
      T *dvec_data = dvec->mutable_data<T>(context.GetPlace());

      blas.GEMV(true, dim_x[0], dim_x[1], static_cast<T>(1), x_data, dout_data,
                static_cast<T>(0), dvec_data);
    }
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;

REGISTER_OP_CUDA_KERNEL(
    mv, ops::MVKernel<paddle::platform::CUDADeviceContext, float>,
    ops::MVKernel<paddle::platform::CUDADeviceContext, double>);
REGISTER_OP_CUDA_KERNEL(
    mv_grad, ops::MVGradKernel<paddle::platform::CUDADeviceContext, float>,
    ops::MVGradKernel<paddle::platform::CUDADeviceContext, double>);
