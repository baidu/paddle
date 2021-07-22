/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <algorithm>
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/gather_tree_op.h"

namespace paddle {
namespace operators {

template <typename T>
__global__ void GatherTree(const T *ids_data, const T *parents_data,
                           T *out_data, const int64_t max_length,
                           const int64_t batch_size, const int64_t beam_size) {
  CUDA_KERNEL_LOOP(i, batch_size * beam_size) {
    int batch = i / beam_size;
    int beam = i % beam_size;
    auto idx =
        (max_length - 1) * batch_size * beam_size + batch * beam_size + beam;
    out_data[idx] = ids_data[idx];
    auto parent = parents_data[idx];
    for (int step = max_length - 2; step >= 0; step--) {
      idx = step * batch_size * beam_size + batch * beam_size;
      out_data[idx + beam] = ids_data[idx + parent];
      parent = parents_data[idx + parent];
    }
  }
}

template <typename T>
class GatherTreeOpCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    auto *ids = ctx.Input<Tensor>("Ids");
    auto *parents = ctx.Input<Tensor>("Parents");
    auto *out = ctx.Output<Tensor>("Out");

    const auto *ids_data = ids->data<T>();
    const auto *parents_data = parents->data<T>();
    auto *out_data = out->mutable_data<T>(ctx.GetPlace());

    PADDLE_ENFORCE_NOT_NULL(
        ids_data, platform::errors::InvalidArgument(
                      "Input(Ids) of gather_tree should not be null."));

    PADDLE_ENFORCE_NOT_NULL(
        parents_data, platform::errors::InvalidArgument(
                          "Input(Parents) of gather_tree should not be null."));

    auto &ids_dims = ids->dims();
    int64_t max_length = ids_dims[0];
    int64_t batch_size = ids_dims[1];
    int64_t beam_size = ids_dims[2];

    auto &dev_ctx = ctx.cuda_device_context();

    const int block = 512;
    int max_threads =
        std::min(static_cast<int64_t>(dev_ctx.GetMaxPhysicalThreadCount()),
                 batch_size * beam_size);
    const int grid = std::max(max_threads / block, 1);
    GatherTree<<<grid, block>>>(ids_data, parents_data, out_data, max_length,
                                batch_size, beam_size);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OP_CUDA_KERNEL(gather_tree, ops::GatherTreeOpCUDAKernel<int32_t>,
                        ops::GatherTreeOpCUDAKernel<int64_t>);
