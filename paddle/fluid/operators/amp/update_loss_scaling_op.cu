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

#include <vector>
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/amp/update_loss_scaling_op.h"
#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace operators {

template <typename T>
__global__ void GpuUpdateLossScaling(
    const bool* found_inf_data, const T* pre_loss_scaling_data,
    const int* good_in_data, const int* bad_in_data,
    const int incr_every_n_steps, const int decr_every_n_nan_or_inf,
    const float incr_ratio, const float decr_ratio,
    T* updated_loss_scaling_data, int* good_out_data, int* bad_out_data) {
  Update<T>(found_inf_data, pre_loss_scaling_data, good_in_data, bad_in_data,
            incr_every_n_steps, decr_every_n_nan_or_inf, incr_ratio, decr_ratio,
            updated_loss_scaling_data, good_out_data, bad_out_data);
}

template <typename T>
class UpdateLossScalingFunctor<platform::CUDADeviceContext, T> {
 public:
  void operator()(const platform::CUDADeviceContext& dev_ctx,
                  const bool* found_inf_data, const T* pre_loss_scaling_data,
                  const int* good_in_data, const int* bad_in_data,
                  const int incr_every_n_steps,
                  const int decr_every_n_nan_or_inf, const float incr_ratio,
                  const float decr_ratio, T* updated_loss_scaling_data,
                  int* good_out_data, int* bad_out_data) const {
    GpuUpdateLossScaling<T><<<1, 1, 0, dev_ctx.stream()>>>(
        found_inf_data, pre_loss_scaling_data, good_in_data, bad_in_data,
        incr_every_n_steps, decr_every_n_nan_or_inf, incr_ratio, decr_ratio,
        updated_loss_scaling_data, good_out_data, bad_out_data);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
using GPU = paddle::platform::CUDADeviceContext;

REGISTER_OP_CUDA_KERNEL(update_loss_scaling,
                        ops::UpdateLossScalingKernel<GPU, float>,
                        ops::UpdateLossScalingKernel<GPU, double>);
