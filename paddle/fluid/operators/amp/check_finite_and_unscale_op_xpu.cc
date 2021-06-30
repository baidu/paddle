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
#include "paddle/fluid/operators/amp/check_finite_and_unscale_op.h"
#include "paddle/fluid/operators/amp/fp16_type_traits.h"
#include "paddle/fluid/platform/float16.h"
namespace paddle {
namespace operators {
template <typename T>
class CheckFiniteAndUnscaleXPUKernel : public framework::OpKernel<T> {
  using MPDType = typename details::MPTypeTrait<T>::Type;
  using XPUTyp = typename XPUTypeTrait<T>::Type;

 public:
  void Compute(const framework::ExecutionContext& ctx) const {
    auto& dev_ctx = ctx.template device_context<platform::XPUDeviceContext>();
    const auto xs = ctx.MultiInput<framework::Tensor>("X");
    const auto* scale = ctx.Input<framework::Tensor>("Scale");
    auto outs = ctx.MultiOutput<framework::Tensor>("Out");
    auto* found_inf = ctx.Output<framework::Tensor>("FoundInfinite");

    const MPDType* scale_data = scale->data<MPDType>();
    bool* found_inf_data = found_inf->mutable_data<bool>(dev_ctx.GetPlace());

    // cpy to cpu
    bool cpu_found_inf_data = false;

    MPDType cpu_scale_data;
    if (platform::is_xpu_place(scale->place())) {
      xpu_memcpy(&cpu_scale_data, scale_data, sizeof(MPDType),
                 XPUMemcpyKind::XPU_DEVICE_TO_HOST);
    } else {
      cpu_scale_data = (*scale_data);
    }
    MPDType inverse_scale = 1.0 / cpu_scale_data;
    for (size_t i = 0; i < xs.size(); ++i) {
      const auto* x = xs[i];
      auto* out = outs[i];
      out->mutable_data<T>(dev_ctx.GetPlace());
      framework::Tensor is_finite =
          ctx.AllocateTmpTensor<bool, platform::XPUDeviceContext>(x->dims(),
                                                                  dev_ctx);
      framework::Tensor is_nan =
          ctx.AllocateTmpTensor<bool, platform::XPUDeviceContext>(x->dims(),
                                                                  dev_ctx);
      framework::Tensor is_finite_and_nan =
          ctx.AllocateTmpTensor<bool, platform::XPUDeviceContext>(x->dims(),
                                                                  dev_ctx);
      if (cpu_found_inf_data == false) {
        int r = xpu::isfinite(dev_ctx.x_context(),
                              reinterpret_cast<const XPUTyp*>(x->data<T>()),
                              is_finite.data<bool>(), x->numel());
        PADDLE_ENFORCE_EQ(r, XPU_SUCCESS, platform::errors::External(
                                              "XPU API(isfinite) return wrong "
                                              "value[%d %s]",
                                              r, XPUAPIErrorMsg[r]));
        r = xpu::logical_not(dev_ctx.x_context(), reinterpret_cast<const bool*>(
                                                      is_finite.data<bool>()),
                             is_finite.data<bool>(), x->numel());
        PADDLE_ENFORCE_EQ(r, XPU_SUCCESS, platform::errors::External(
                                              "XPU API(isfinite) return wrong "
                                              "value[%d %s]",
                                              r, XPUAPIErrorMsg[r]));
        r = xpu::isnan(dev_ctx.x_context(),
                       reinterpret_cast<const XPUTyp*>(x->data<T>()),
                       is_nan.data<bool>(), x->numel());
        PADDLE_ENFORCE_EQ(r, XPU_SUCCESS, platform::errors::External(
                                              "XPU API(isnan) return wrong "
                                              "value[%d %s]",
                                              r, XPUAPIErrorMsg[r]));
        r = xpu::logical_or(dev_ctx.x_context(), is_finite.data<bool>(),
                            is_nan.data<bool>(), is_finite.data<bool>(),
                            x->numel());
        PADDLE_ENFORCE_EQ(r, XPU_SUCCESS, platform::errors::External(
                                              "XPU API(any) return wrong "
                                              "value[%d %s]",
                                              r, XPUAPIErrorMsg[r]));
        r = xpu::any(dev_ctx.x_context(), is_finite.data<bool>(),
                     found_inf_data, x->numel());
        PADDLE_ENFORCE_EQ(r, XPU_SUCCESS, platform::errors::External(
                                              "XPU API(any) return wrong "
                                              "value[%d %s]",
                                              r, XPUAPIErrorMsg[r]));
        memory::Copy(platform::CPUPlace(), &cpu_found_inf_data,
                     BOOST_GET_CONST(platform::XPUPlace, dev_ctx.GetPlace()),
                     found_inf_data, sizeof(bool));
      }

      if (cpu_found_inf_data) {
        inverse_scale = 0.0;
      }
      auto dev_env = XPUEnv::getenv("XPUSIM_DEVICE_MODEL");

      if (std::is_same<T, paddle::platform::float16>::value &&
          (dev_env == nullptr || std::strcmp(dev_env, "KUNLUN1"))) {
        framework::Tensor float_x;
        framework::Tensor float_out;
        float_x.mutable_data<MPDType>(dev_ctx.GetPlace(),
                                      x->numel() * sizeof(MPDType));
        float_out.mutable_data<MPDType>(dev_ctx.GetPlace(),
                                        out->numel() * sizeof(MPDType));
        int r = xpu::cast_v2(dev_ctx.x_context(),
                             reinterpret_cast<const float16*>(x->data<T>()),
                             float_x.data<MPDType>(), x->numel());
        PADDLE_ENFORCE_EQ(r, XPU_SUCCESS, platform::errors::External(
                                              "XPU API(cast_v2) return wrong "
                                              "value[%d %s]",
                                              r, XPUAPIErrorMsg[r]));

        r = xpu::scale(dev_ctx.x_context(), float_x.data<MPDType>(),
                       float_out.data<MPDType>(), x->numel(), false,
                       inverse_scale, 0.0);
        PADDLE_ENFORCE_EQ(r, XPU_SUCCESS, platform::errors::External(
                                              "XPU API(scale) return wrong "
                                              "value[%d %s]",
                                              r, XPUAPIErrorMsg[r]));

        r = xpu::cast_v2(dev_ctx.x_context(), float_out.data<MPDType>(),
                         reinterpret_cast<float16*>(out->data<T>()),
                         out->numel());

        PADDLE_ENFORCE_EQ(r, XPU_SUCCESS, platform::errors::External(
                                              "XPU API(cast_v2) return wrong "
                                              "value[%d %s]",
                                              r, XPUAPIErrorMsg[r]));

      } else {
        int r = xpu::scale(dev_ctx.x_context(),
                           reinterpret_cast<const XPUTyp*>(x->data<T>()),
                           reinterpret_cast<XPUTyp*>(out->data<T>()),
                           x->numel(), false, inverse_scale, 0.0);
        PADDLE_ENFORCE_EQ(r, XPU_SUCCESS, platform::errors::External(
                                              "XPU API(scale) return wrong "
                                              "value[%d %s]",
                                              r, XPUAPIErrorMsg[r]));
      }
    }
    memory::Copy(BOOST_GET_CONST(platform::XPUPlace, dev_ctx.GetPlace()),
                 found_inf_data, platform::CPUPlace(), &cpu_found_inf_data,
                 sizeof(bool));
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;
REGISTER_OP_XPU_KERNEL(check_finite_and_unscale,
                       ops::CheckFiniteAndUnscaleXPUKernel<float>,
                       ops::CheckFiniteAndUnscaleXPUKernel<plat::float16>);

#endif
