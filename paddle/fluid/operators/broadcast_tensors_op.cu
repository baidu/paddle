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

#include "paddle/fluid/operators/broadcast_tensors_op.h"

#include <algorithm>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "paddle/fluid/operators/reduce_ops/cub_reduce.h"

namespace paddle {
namespace operators {

using framework::Tensor;
using framework::DDim;

template <typename Tout>
struct IdentityFunctor {
  HOSTDEVICE explicit inline IdentityFunctor() {}

  template <typename U>
  HOSTDEVICE inline Tout operator()(const U& x) const {
    return static_cast<Tout>(x);
  }
};

template <typename T>
class CUDABroadcastTensorsGradOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    // Find reduce dimensions
    const auto& in_tensors =
        context.MultiInput<Tensor>(framework::GradVarName("Out"));
    auto out_tensors = context.MultiOutput<Tensor>(framework::GradVarName("X"));

    size_t num_ins = in_tensors.size();

    PADDLE_ENFORCE_GT(
        num_ins, 1,
        platform::errors::InvalidArgument(
            "Expected at least 2 input tensors, but only received d%.",
            in_tensors.size()));

    PADDLE_ENFORCE_EQ(
        num_ins, out_tensors.size(),
        platform::errors::InvalidArgument(
            "BroadcastTensorsOp expects equal number of inputs and outputs,"
            "but received: %d inputs v.s %d outputs",
            num_ins, out_tensors.size()));

    // For each In-Out tensor pair,
    // Prepare and apply broadcast dims array
    for (size_t i = 0; i < num_ins; i++) {
      auto* input_tensor = in_tensors[i];
      auto* output_tensor = out_tensors[i];

      const DDim& input_dims = input_tensor->dims();
      const DDim& output_dims = output_tensor->dims();

      int in_rank = input_dims.size();
      int out_rank = output_dims.size();

      // Collect reduce_dims
      // Example:
      // dX  = [1,1,1,1]
      // dOut = [1,1,1,4]
      //
      // reduce_dims  = [3] // reduce along the broadcasted axis
      std::vector<int> reduce_dims_vec;
      for (int j = 0; j < in_rank; j++) {
        int out_axis = out_rank - j - 1;
        int in_axis = in_rank - j - 1;

        if (out_axis < 0 || output_dims[out_axis] != input_dims[in_axis]) {
          reduce_dims_vec.push_back(in_axis);
        }
      }

      bool just_copy = (reduce_dims_vec.size() == 0);
      output_tensor->mutable_data<T>(context.GetPlace());
      if (just_copy) {
        // Turns out to be a No-Op, simply copy tensors
        framework::TensorCopy(*input_tensor, context.GetPlace(),
                              context.device_context(), output_tensor);
      } else {
        // reduce_sum implementation on CUDA
        auto stream = context.cuda_device_context().stream();
        TensorReduce<T, T, cub::Sum, IdentityFunctor<T>>(
            *input_tensor, output_tensor, reduce_dims_vec, static_cast<T>(0),
            cub::Sum(), IdentityFunctor<T>(), stream);
      }
    }
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;

REGISTER_OP_CUDA_KERNEL(
    broadcast_tensors,
    ops::BroadcastTensorsOpKernel<paddle::platform::CUDADeviceContext,
                                  plat::float16>,
    ops::BroadcastTensorsOpKernel<paddle::platform::CUDADeviceContext, float>,
    ops::BroadcastTensorsOpKernel<paddle::platform::CUDADeviceContext, double>,
    ops::BroadcastTensorsOpKernel<paddle::platform::CUDADeviceContext, bool>,
    ops::BroadcastTensorsOpKernel<paddle::platform::CUDADeviceContext, int>,
    ops::BroadcastTensorsOpKernel<paddle::platform::CUDADeviceContext,
                                  int64_t>);

REGISTER_OP_CUDA_KERNEL(broadcast_tensors_grad,
                        ops::CUDABroadcastTensorsGradOpKernel<plat::float16>,
                        ops::CUDABroadcastTensorsGradOpKernel<float>,
                        ops::CUDABroadcastTensorsGradOpKernel<double>,
                        ops::CUDABroadcastTensorsGradOpKernel<int>,
                        ops::CUDABroadcastTensorsGradOpKernel<int64_t>);
