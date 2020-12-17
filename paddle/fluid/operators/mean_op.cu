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
#include "cub/cub.cuh"
#include "paddle/fluid/operators/mean_op.h"
#include "paddle/fluid/platform/cuda_primitives.h"
#include "paddle/fluid/platform/float16.h"

namespace paddle {
namespace operators {

template <typename T>
struct DivideFunctor {
  HOSTDEVICE explicit inline DivideFunctor(int n)
      : n_inv(static_cast<T>(1.0 / n)) {}

  HOSTDEVICE inline T operator()(const T& x) const { return x * n_inv; }

 private:
  T n_inv;
};

template <typename T>
__global__ void MeanRunKernel(const T* in_data, T* out_data, int N) {
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  T data = in_data[0];
  for (; idx < N; idx += blockDim.x * gridDim.x) {
    out_data[idx] = data / (static_cast<T>(N));
  }
}

template <typename DeviceContext, typename T>
class MeanCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* input = context.Input<Tensor>("X");
    auto* output = context.Output<Tensor>("Out");

    output->mutable_data<T>(context.GetPlace());
    auto size_prob = input->numel();
    const T* in_data = input->data<T>();
    T* out_data = output->mutable_data<T>(context.GetPlace());
    auto stream = context.cuda_device_context().stream();

    DivideFunctor<T> transformer(size_prob);
    cub::TransformInputIterator<T, DivideFunctor<T>, const T*> trans_x(
        in_data, transformer);
    size_t temp_storage_bytes = 0;

    auto err = cub::DeviceReduce::Sum(nullptr, temp_storage_bytes, trans_x,
                                      out_data, size_prob, stream);
    PADDLE_ENFORCE_CUDA_SUCCESS(err);
    framework::Tensor tmp;
    auto* temp_storage = tmp.mutable_data<uint8_t>(
        framework::make_ddim({static_cast<int64_t>(temp_storage_bytes)}),
        context.GetPlace());
    err = cub::DeviceReduce::Sum(temp_storage, temp_storage_bytes, trans_x,
                                 out_data, size_prob, stream);
    PADDLE_ENFORCE_CUDA_SUCCESS(err);
  }
};

template <typename DeviceContext, typename T>
class MeanCUDAGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto OG = context.Input<Tensor>(framework::GradVarName("Out"));
    PADDLE_ENFORCE_EQ(OG->numel(), 1,
                      platform::errors::InvalidArgument(
                          "Mean Gradient Input Tensor len should be 1. But "
                          "received Out@Grad's elements num is %d.",
                          OG->numel()));
    auto IG = context.Output<Tensor>(framework::GradVarName("X"));
    IG->mutable_data<T>(context.GetPlace());

    auto in_data = OG->data<T>();
    auto size_prob = IG->numel();
    auto out_data = IG->data<T>();
    int threads = 512;
    int grid = (size_prob + threads - 1) / threads;
    auto stream = context.cuda_device_context().stream();
    MeanRunKernel<T><<<grid, threads, 0, stream>>>(in_data, out_data,
                                                   size_prob);
  }
};
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;
REGISTER_OP_CUDA_KERNEL(
    mean, ops::MeanCUDAKernel<paddle::platform::CUDADeviceContext, float>,
    ops::MeanCUDAKernel<paddle::platform::CUDADeviceContext, double>,
    ops::MeanCUDAKernel<paddle::platform::CUDADeviceContext, plat::float16>,
    ops::MeanCUDAKernel<paddle::platform::CUDADeviceContext, plat::complex64>,
    ops::MeanCUDAKernel<paddle::platform::CUDADeviceContext, plat::complex128>);
REGISTER_OP_CUDA_KERNEL(
    mean_grad,
    ops::MeanCUDAGradKernel<paddle::platform::CUDADeviceContext, float>,
    ops::MeanCUDAGradKernel<paddle::platform::CUDADeviceContext, double>,
    ops::MeanCUDAGradKernel<paddle::platform::CUDADeviceContext, plat::float16>,
    ops::MeanCUDAGradKernel<paddle::platform::CUDADeviceContext,
                            plat::complex64>,
    ops::MeanCUDAGradKernel<paddle::platform::CUDADeviceContext,
                            plat::complex128>);
