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

#ifdef PADDLE_WITH_XPU
#include "paddle/fluid/operators/amp/update_loss_scaling_op.h"
#include <cstring>
#include <string>
#include <vector>
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/amp/fp16_type_traits.h"
#include "paddle/fluid/platform/float16.h"

namespace paddle {
namespace operators {

template <typename T>
class UpdateLossScalingXPUKernel : public framework::OpKernel<T> {
  using MPDType = typename details::MPTypeTrait<T>::Type;
  using XPUTyp = typename XPUTypeTrait<T>::Type;

 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto& dev_ctx = ctx.template device_context<platform::XPUDeviceContext>();

    const auto xs = ctx.MultiInput<framework::Tensor>("X");
    auto outs = ctx.MultiOutput<framework::Tensor>("Out");
    const auto* found_inf = ctx.Input<Tensor>("FoundInfinite");
    PADDLE_ENFORCE_EQ(found_inf->numel(), 1,
                      platform::errors::InvalidArgument(
                          "FoundInfinite must has only one element."));
    const bool* found_inf_data = found_inf->data<bool>();
    bool cpu_found_inf_data = false;
    if (platform::is_xpu_place(found_inf->place())) {
      xpu_memcpy(&cpu_found_inf_data, found_inf_data, sizeof(bool),
                 XPUMemcpyKind::XPU_DEVICE_TO_HOST);
    } else {
      cpu_found_inf_data = (*found_inf_data);
    }

    for (size_t i = 0; i < xs.size(); ++i) {
      auto* out = outs[i];
      T* out_data = out->mutable_data<T>(dev_ctx.GetPlace());
      int num = out->numel();
      if (cpu_found_inf_data) {
        VLOG(1) << "-- UpdateLossScaling: Find infinite grads. --";
        int r = 0;
        r = xpu::constant(dev_ctx.x_context(),
                          reinterpret_cast<XPUTyp*>(out_data), num,
                          XPUTyp(0.0));
        PADDLE_ENFORCE_EQ(r == xpu::Error_t::SUCCESS, true,
                          platform::errors::External(
                              "XPU API return wrong value[%d], please check "
                              "where Baidu Kunlun Card is properly installed.",
                              r));
      }
    }
    const bool stop_update = ctx.Attr<bool>("stop_update");
    if (stop_update) {
      return;
    }

    const auto* pre_loss_scaling = ctx.Input<Tensor>("PrevLossScaling");
    const auto* good_in = ctx.Input<Tensor>("InGoodSteps");
    const auto* bad_in = ctx.Input<Tensor>("InBadSteps");
    auto* updated_loss_scaling = ctx.Output<Tensor>("LossScaling");
    auto* good_out = ctx.Output<Tensor>("OutGoodSteps");
    auto* bad_out = ctx.Output<Tensor>("OutBadSteps");
    const MPDType* pre_loss_scaling_data = pre_loss_scaling->data<MPDType>();
    const int* good_in_data = good_in->data<int>();
    const int* bad_in_data = bad_in->data<int>();

    MPDType* updated_loss_scaling_data =
        updated_loss_scaling->mutable_data<MPDType>(dev_ctx.GetPlace());
    int* good_out_data = good_out->mutable_data<int>(dev_ctx.GetPlace());
    int* bad_out_data = bad_out->mutable_data<int>(dev_ctx.GetPlace());

    const int incr_every_n_steps = ctx.Attr<int>("incr_every_n_steps");
    const int decr_every_n_nan_or_inf =
        ctx.Attr<int>("decr_every_n_nan_or_inf");
    const float incr_ratio = ctx.Attr<float>("incr_ratio");
    const float decr_ratio = ctx.Attr<float>("decr_ratio");

    int cpu_bad_in_data;
    int cpu_good_in_data;
    MPDType cpu_pre_loss_scaling_data;
    if (platform::is_xpu_place(bad_in->place())) {
      xpu_memcpy(&cpu_bad_in_data, bad_in_data, sizeof(int),
                 XPUMemcpyKind::XPU_DEVICE_TO_HOST);
    } else {
      cpu_bad_in_data = (*bad_in_data);
    }

    if (platform::is_xpu_place(good_in->place())) {
      xpu_memcpy(&cpu_good_in_data, good_in_data, sizeof(int),
                 XPUMemcpyKind::XPU_DEVICE_TO_HOST);
    } else {
      cpu_good_in_data = (*good_in_data);
    }

    if (platform::is_xpu_place(pre_loss_scaling->place())) {
      xpu_memcpy(&cpu_pre_loss_scaling_data, pre_loss_scaling_data,
                 sizeof(MPDType), XPUMemcpyKind::XPU_DEVICE_TO_HOST);
    } else {
      cpu_pre_loss_scaling_data = (*pre_loss_scaling_data);
    }

    int cpu_good_out_data = 0;
    int cpu_bad_out_data = 0;
    MPDType cpu_updated_loss_scaling_data;

    if (cpu_found_inf_data) {
      cpu_good_out_data = 0;
      cpu_bad_out_data = cpu_bad_in_data + 1;
      if (cpu_bad_out_data == decr_every_n_nan_or_inf) {
        MPDType new_loss_scaling = cpu_pre_loss_scaling_data * decr_ratio;
        cpu_updated_loss_scaling_data =
            (new_loss_scaling < static_cast<MPDType>(1))
                ? (static_cast<MPDType>(1))
                : (new_loss_scaling);
        cpu_bad_out_data = 0;
      }
    } else {
      cpu_bad_out_data = 0;
      cpu_good_out_data = cpu_good_in_data + 1;
      if (cpu_good_out_data == incr_every_n_steps) {
        MPDType new_loss_scaling = cpu_pre_loss_scaling_data * incr_ratio;
        cpu_updated_loss_scaling_data = (std::isfinite(new_loss_scaling))
                                            ? new_loss_scaling
                                            : cpu_pre_loss_scaling_data;
        cpu_good_out_data = 0;
      }
    }

    // copy to host
    memory::Copy(BOOST_GET_CONST(platform::XPUPlace, dev_ctx.GetPlace()),
                 bad_out_data, platform::CPUPlace(), &cpu_bad_out_data,
                 sizeof(int));
    memory::Copy(BOOST_GET_CONST(platform::XPUPlace, dev_ctx.GetPlace()),
                 good_out_data, platform::CPUPlace(), &cpu_good_out_data,
                 sizeof(int));
    memory::Copy(BOOST_GET_CONST(platform::XPUPlace, dev_ctx.GetPlace()),
                 updated_loss_scaling_data, platform::CPUPlace(),
                 &cpu_updated_loss_scaling_data, sizeof(MPDType));
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;

REGISTER_OP_XPU_KERNEL(update_loss_scaling,
                       ops::UpdateLossScalingXPUKernel<float>,
                       ops::UpdateLossScalingXPUKernel<plat::float16>);
#endif
