/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <vector>
#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/memory/malloc.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
using CUDADeviceContext = paddle::platform::CUDADeviceContext;

__constant__ int64_t convar[1];

template <typename T>
__global__ void fill_constant_kernel(const int64_t featuresize, T* in_data) {
  T* temp = reinterpret_cast<T*>(convar);
  for (int idx = blockIdx.x * featuresize + threadIdx.x;
       idx < (blockIdx.x + 1) * featuresize; idx += blockDim.x) {
    in_data[idx] = *temp;
  }
}

template <typename T>
class FillInplaceCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
#ifdef __HIPCC__
    const int64_t kMaxBlockDim = 256;
#else
    const int64_t kMaxBlockDim = 512;
#endif
    auto* x = ctx.Input<Tensor>("X");
    auto* value = ctx.Input<Tensor>("value");
    auto* out = ctx.Output<Tensor>("Out");
    T* out_data = out->mutable_data<T>(ctx.GetPlace());
    T* fill_value = const_cast<T*>(value->data<T>());
    cudaMemcpyToSymbol(convar, fill_value, sizeof(T));
    auto x_dims = x->dims();
    auto matrix_dim = framework::flatten_to_2d(x_dims, 1);
    int64_t batchsize = static_cast<int64_t>(matrix_dim[0]);
    int64_t featuresize = static_cast<int64_t>(matrix_dim[1]);
    int64_t kBlockDim = std::min(featuresize, kMaxBlockDim);
    fill_constant_kernel<T><<<batchsize, kBlockDim, 0>>>(featuresize, out_data);
  }
};

template <typename T>
class FillInplaceGradCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
#ifdef __HIPCC__
    const int64_t kMaxBlockDim = 256;
#else
    const int64_t kMaxBlockDim = 512;
#endif
    auto* dx = ctx.Output<framework::Tensor>(framework::GradVarName("X"));
    auto* in_data = dx->mutable_data<T>(ctx.GetPlace());
    T temp = T(0);
    cudaMemcpyToSymbol(convar, &temp, sizeof(T));
    const auto x_dims = dx->dims();
    auto matrix_dim = framework::flatten_to_2d(x_dims, 1);
    int64_t batchsize = static_cast<int64_t>(matrix_dim[0]);
    int64_t featuresize = static_cast<int64_t>(matrix_dim[1]);
    int64_t kBlockDim = std::min(featuresize, kMaxBlockDim);
    fill_constant_kernel<T><<<batchsize, kBlockDim, 0>>>(featuresize, in_data);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;

REGISTER_OP_CUDA_KERNEL(fill_inplace, ops::FillInplaceCUDAKernel<float>,
                        ops::FillInplaceCUDAKernel<double>,
                        ops::FillInplaceCUDAKernel<plat::float16>,
                        ops::FillInplaceCUDAKernel<int>,
                        ops::FillInplaceCUDAKernel<int64_t>,
                        ops::FillInplaceCUDAKernel<bool>);

REGISTER_OP_CUDA_KERNEL(fill_inplace_grad,
                        ops::FillInplaceGradCUDAKernel<float>,
                        ops::FillInplaceGradCUDAKernel<double>,
                        ops::FillInplaceGradCUDAKernel<int>,
                        ops::FillInplaceGradCUDAKernel<int64_t>,
                        ops::FillInplaceGradCUDAKernel<plat::float16>,
                        ops::FillInplaceGradCUDAKernel<bool>);
