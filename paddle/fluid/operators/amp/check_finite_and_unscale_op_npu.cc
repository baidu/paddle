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

#include <memory>
#include <string>

#include "paddle/fluid/framework/tensor_util.h"
#include "paddle/fluid/operators/amp/check_finite_and_unscale_op.h"
#include "paddle/fluid/operators/npu_op_runner.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;

// NOTE(zhiqiu): The CheckFiniteAndUnscaleNPUKernel is different from CUDA.
// On NPU, we do not really check the data of input tensors,
// but use NPUGetFloatStatus to check whether the nan/inf occurs on device,
// and clear it after this op.
// Which may leads to wrong result if the input tensors is not calculated
// on NPU device, but got from other way, for example, feeding.
template <typename T>
class CheckFiniteAndUnscaleNPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const {
    const auto xs = ctx.MultiInput<framework::Tensor>("X");
    const auto* scale = ctx.Input<framework::Tensor>("Scale");
    const auto* float_status = ctx.Input<framework::Tensor>("FloatStatus");
    auto outs = ctx.MultiOutput<framework::Tensor>("Out");
    auto* found_inf = ctx.Output<framework::Tensor>("FoundInfinite");

    found_inf->mutable_data<bool>(ctx.GetPlace());

    auto stream =
        ctx.template device_context<paddle::platform::NPUDeviceContext>()
            .stream();

    // step1: inverse scale
    Tensor const_tensor;
    const_tensor.mutable_data<T>({1}, ctx.GetPlace());
    FillNpuTensorWithConstant<T>(&const_tensor, static_cast<T>(1.0));

    // Inverse(1.0/scale)
    Tensor* tmp_inverse_out = const_cast<Tensor*>(scale);
    Tensor inverse_out(scale->type());
    inverse_out.Resize(scale->dims());
    inverse_out.mutable_data<T>(ctx.GetPlace());
    const auto& runner_inverse =
        NpuOpRunner("Div", {const_tensor, *scale}, {inverse_out}, {});
    runner_inverse.Run(stream);
    tmp_inverse_out = &inverse_out;

    bool found_inf_data = false;

    size_t x_size = xs.size();
    for (size_t i = 0; i < x_size; ++i) {
      const auto* x = xs[i];

      // step2: CheckNumerics
      // CheckNumerics runs on the Ascend AI CPU, which delivers poor
      // performance.
      Tensor check_xout(x->type());
      check_xout.Resize(x->dims());
      check_xout.mutable_data<T>(ctx.GetPlace());
      try {
        const auto& runner_checknumerics =
            NpuOpRunner("CheckNumerics", {*x}, {check_xout},
                        {{"message", std::string("check_nan_and_inf")}});
        runner_checknumerics.Run(stream);
        ctx.template device_context<paddle::platform::NPUDeviceContext>()
            .Wait();
      } catch (platform::EnforceNotMet& exception) {
        LOG(WARNING) << "[check_nan_and_inf] detected contains NaN or INF!!!";
        found_inf_data = true;
      } catch (...) {
        LOG(WARNING) << "[check_nan_and_inf] detected contains NaN or INF!!!";
        found_inf_data = true;
      }
    }  // end for

    VLOG(0) << "yoki: found_inf in CheckNumerics method: "
            << static_cast<float>(found_inf_data);

    // NOTE(zhiqiu):
    Tensor tmp;
    tmp.mutable_data<float>({8}, ctx.GetPlace());
    // NOTE(zhiqiu): NPUGetFloatStatus updates data on input in-place.
    // tmp is only placeholder.
    const auto& runner_float_status =
        NpuOpRunner("NPUGetFloatStatus", {*float_status}, {tmp},
                    {{"message", std::string("check_nan_and_inf")}});
    runner_float_status.Run(stream);

    Tensor sum;
    sum.mutable_data<float>({1}, ctx.GetPlace());
    const auto& runner_reduce_sum =
        NpuOpRunner("ReduceSumD", {*float_status}, {sum},
                    {{"axes", std::vector<int>{0}}, {"keep_dims", true}});
    runner_reduce_sum.Run(stream);

    const auto& runner_greater =
        NpuOpRunner("GreaterEqual", {sum, const_tensor}, {*found_inf}, {});
    runner_greater.Run(stream);

    std::vector<bool> found_inf_cpu;
    TensorToVector(*found_inf, ctx.device_context(), &found_inf_cpu);
    VLOG(0) << "yoki: found_inf in check_finite npu op: "
            << static_cast<float>(found_inf_cpu[0]);

    if (found_inf_cpu[0]) {
      size_t x_size = xs.size();
      for (size_t i = 0; i < x_size; ++i) {
        const auto* x = xs[i];

        std::vector<T> check_x;
        TensorToVector(*x, ctx.device_context(), &check_x);
        for (int i = 0; i < static_cast<int>(check_x.size()); ++i) {
          // VLOG(0) << "check_x[" << i<< "] : yokinum: "<< check_x[i];
          if (std::isnan(check_x[i]) || std::isinf(check_x[i])) {
            VLOG(0) << "yokinan: " << check_x[i];
          }
        }
      }
    }

    // NOTE(zhiqiu): The normal logic is :
    // out = in, if found_inf = true
    // out = in/scale, if found_inf = false
    // However, on NPU, in order to avoid stream sync, we do not copy the
    // found_inf data to cpu to check whether to unscale or not.
    // Instead, we do the Mul no matter found_inf or not.
    // And, a fact is, only few steps contains nan/inf during training.
    for (size_t i = 0; i < xs.size(); ++i) {
      const auto* x = xs[i];
      auto* out = outs[i];
      out->mutable_data<T>(ctx.GetPlace());
      const auto& runner_mul =
          NpuOpRunner("Mul", {*x, *tmp_inverse_out}, {*out}, {});
      runner_mul.Run(stream);
    }
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;
REGISTER_OP_NPU_KERNEL(check_finite_and_unscale,
                       ops::CheckFiniteAndUnscaleNPUKernel<float>,
                       ops::CheckFiniteAndUnscaleNPUKernel<plat::float16>);
