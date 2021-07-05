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

template <typename T>
__global__ void fill_constant_kernel(const int64_t featuresize, T* in_data,
                                     const T* value) {
  for (int idx = blockIdx.x * featuresize + threadIdx.x;
       idx < (blockIdx.x + 1) * featuresize; idx += blockDim.x) {
    in_data[idx] = *value;
  }
}

template <typename T>
class Fill_CUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
#ifdef __HIPCC__
    const int64_t kMaxBlockDim = 256;
#else
    const int64_t kMaxBlockDim = 512;
#endif

    auto* in = ctx.Input<Tensor>("X");
    auto* value = ctx.Input<Tensor>("value");

    auto* in_data = in->data<T>();
    const auto x_dims = in->dims();
    auto matrix_dim = framework::flatten_to_2d(x_dims, 1);
    int64_t batchsize = static_cast<int64_t>(matrix_dim[0]);
    int64_t featuresize = static_cast<int64_t>(matrix_dim[1]);
    int64_t kBlockDim = std::min(featuresize, kMaxBlockDim);
    fill_constant_kernel<T><<<batchsize, kBlockDim, 0>>>(
        featuresize, const_cast<T*>(in_data), value->data<T>());
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;

REGISTER_OP_CUDA_KERNEL(fill_inplace, ops::Fill_CUDAKernel<float>,
                        ops::Fill_CUDAKernel<double>,
                        ops::Fill_CUDAKernel<plat::float16>,
                        ops::Fill_CUDAKernel<int>,
                        ops::Fill_CUDAKernel<int64_t>,
                        ops::Fill_CUDAKernel<bool>);
