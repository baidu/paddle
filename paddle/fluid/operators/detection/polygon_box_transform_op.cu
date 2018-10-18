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

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/platform/cuda_primitives.h"
#include "paddle/fluid/platform/gpu_info.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
using platform::PADDLE_CUDA_NUM_THREADS;
#define CUDA_BLOCK_SIZE 16

template <typename T>
__global__ void PolygonBoxTransformKernel(const int n, const int h, const int w,
                                          const T* input, T* output) {
  int id_n = threadIdx.x + blockDim.x * blockIdx.x;
  int id_h = threadIdx.y + blockDim.y * blockIdx.y;
  int id_w = threadIdx.z + blockDim.z * blockIdx.z;
  if (id_n < n && id_h < h && id_w < w) {
    int id = id_n * h * w + w * id_h + id_w;
    if (id_n % 2 == 0) {
      output[id] = id_w * 4 - input[id];
    } else {
      output[id] = id_h * 4 - input[id];
    }
  }
}

template <typename T>
class PolygonBoxTransformOpCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    PADDLE_ENFORCE(platform::is_gpu_place(ctx.GetPlace()),
                   "It must use CUDAPlace.");
    auto* in = ctx.Input<Tensor>("Input");
    auto in_dims = in->dims();
    const T* in_data = in->data<T>();
    auto* out = ctx.Output<Tensor>("Output");
    T* out_data = out->mutable_data<T>(ctx.GetPlace());

    int batch_size = in_dims[0];
    int geo_channels = in_dims[1];
    int height = in_dims[2];
    int width = in_dims[3];
    dim3 threadsPerBlock(
        PADDLE_CUDA_NUM_THREADS / (CUDA_BLOCK_SIZE * CUDA_BLOCK_SIZE),
        CUDA_BLOCK_SIZE, CUDA_BLOCK_SIZE);
    dim3 numBlocks((batch_size * geo_channels) / threadsPerBlock.x,
                   (height + threadsPerBlock.y - 1) / threadsPerBlock.y,
                   (width + threadsPerBlock.z - 1) / threadsPerBlock.z);
    auto stream = ctx.cuda_device_context().stream();
    PolygonBoxTransformKernel<T><<<numBlocks, threadsPerBlock, 0, stream>>>(
        batch_size * geo_channels, height, width, in_data, out_data);
  }
};

}  // namespace operators
}  // namespace paddle

REGISTER_OP_CUDA_KERNEL(
    polygon_box_transform,
    paddle::operators::PolygonBoxTransformOpCUDAKernel<float>,
    paddle::operators::PolygonBoxTransformOpCUDAKernel<double>);
