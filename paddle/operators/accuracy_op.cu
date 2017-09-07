/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <iostream>
#include "paddle/operators/accuracy_op.h"

namespace paddle {
namespace operators {

// FIXME(typhoonzero): use cub:blockreduce to do accuracy add.
// import cub libraries and make this more efficient.
__global__ void AccuracyDivideKernel(const int N, const int D, const int top_k,
                                     const int* Xdata, const int* labelData,
                                     float* accuracy) {
  int correct = 0;
  for (int row = 0; row < N; row++) {
    const int label = labelData[row];
    for (int col = 0; col < D; col++) {
      const int pred = Xdata[row * D + col];
      if (pred == label) {
        ++correct;
        break;
      }
      __syncthreads();
    }
  }
  *accuracy = static_cast<float>(correct) / static_cast<float>(N);
}

template <typename T>
class AccuracyOpCUDAKernel : public framework::OpKernel {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    PADDLE_ENFORCE(platform::is_gpu_place(ctx.GetPlace()),
                   "It must use GPUPlace.");
    auto* inference = ctx.Input<Tensor>("Inference");
    auto* label = ctx.Input<Tensor>("Label");
    auto* accuracy = ctx.Output<Tensor>("Accuracy");
    const int* inference_data = inference->data<int>();
    const int* label_data = label->data<int>();
    T* accuracy_data = accuracy->mutable_data<T>(ctx.GetPlace());

    size_t num_samples = inference->dims()[0];
    size_t infer_width = inference->dims()[1];

    AccuracyDivideKernel<<<1, 1>>>(num_samples, infer_width, 1, inference_data,
                                   label_data, accuracy_data);
  }
};

}  // namespace operators
}  // namespace paddle

REGISTER_OP_GPU_KERNEL(accuracy,
                       paddle::operators::AccuracyOpCUDAKernel<float>);
