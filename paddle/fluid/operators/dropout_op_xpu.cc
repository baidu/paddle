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
#include "paddle/fluid/operators/dropout_op.h"
#include <memory>
#include <string>
#include "paddle/fluid/platform/xpu_header.h"
namespace paddle {
namespace operators {

#ifdef PADDLE_WITH_XPU

template <typename DeviceContext, typename T>
class DropoutXPUKernel : public framework::OpKernel<T> {
  using XPUTyp = typename XPUTypeTrait<T>::Type;

 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* x = context.Input<Tensor>("X");
    auto* y = context.Output<Tensor>("Out");
    const auto* x_data = x->data<T>();
    auto* y_data = y->mutable_data<T>(context.GetPlace());
    float dropout_prob = context.Attr<float>("dropout_prob");
    auto dropout_implementation =
        context.Attr<std::string>("dropout_implementation");
    auto& dev_ctx = context.template device_context<DeviceContext>();

    PADDLE_ENFORCE_EQ(!context.HasInput("Seed"), true,
                      platform::errors::InvalidArgument(
                          ("Input(Seed) not supported on XPU")));
    int is_upscale = (dropout_implementation == "upscale_in_train");

    if (!context.Attr<bool>("is_test")) {
      std::random_device rnd;
      // int seed = (context.Attr<bool>("fix_seed")) ?
      // int(context.Attr<int>("seed")) : (rnd());
      int seed = 0;
      if (context.Attr<bool>("fix_seed") == true) {
        seed = static_cast<int>(context.Attr<int>("seed"));
      } else {
        seed = rnd();
      }

      auto* mask = context.Output<Tensor>("Mask");
      auto* mask_data = mask->mutable_data<T>(context.GetPlace());
      // Special case when dropout_prob is 1.0
      if (dropout_prob == 1.0f) {
        int r = xpu::constant(dev_ctx.x_context(),
                              reinterpret_cast<XPUTyp*>(y_data), y->numel(),
                              XPUTyp(0));
        PADDLE_ENFORCE_EQ(r, XPU_SUCCESS, platform::errors::External(
                                              "XPU API(constant) return wrong "
                                              "value[%d %s]",
                                              r, XPUAPIErrorMsg[r]));
        r = xpu::constant(dev_ctx.x_context(),
                          reinterpret_cast<XPUTyp*>(mask_data), mask->numel(),
                          XPUTyp(0));
        PADDLE_ENFORCE_EQ(r, XPU_SUCCESS, platform::errors::External(
                                              "XPU API(constant) return wrong "
                                              "value[%d %s]",
                                              r, XPUAPIErrorMsg[r]));
        return;
      }
      int r = xpu::dropout(dev_ctx.x_context(),
                           reinterpret_cast<const XPUTyp*>(x->data<T>()),
                           reinterpret_cast<XPUTyp*>(y->data<T>()),
                           reinterpret_cast<XPUTyp*>(mask_data), seed,
                           mask->numel(), is_upscale, dropout_prob);
      PADDLE_ENFORCE_EQ(r, XPU_SUCCESS, platform::errors::External(
                                            "XPU API(dropout) return wrong "
                                            "value[%d %s]",
                                            r, XPUAPIErrorMsg[r]));
    } else {
      float scale =
          (is_upscale) ? (1.0) : (static_cast<float>(1.0f - dropout_prob));
      int r = xpu::scale(
          dev_ctx.x_context(), reinterpret_cast<const XPUTyp*>(x_data),
          reinterpret_cast<XPUTyp*>(y_data), x->numel(), false, scale, 0.0f);
      PADDLE_ENFORCE_EQ(r, XPU_SUCCESS, platform::errors::External(
                                            "XPU API(scale) return wrong "
                                            "value[%d %s]",
                                            r, XPUAPIErrorMsg[r]));
    }
  }
};
template <typename DeviceContext, typename T>
class DropoutGradXPUKernel : public framework::OpKernel<T> {
  using XPUTyp = typename XPUTypeTrait<T>::Type;

 public:
  void Compute(const framework::ExecutionContext& context) const override {
    PADDLE_ENFORCE_EQ(!context.Attr<bool>("is_test"), true,
                      platform::errors::InvalidArgument(
                          "GradOp is only callable when is_test is false"));
    auto* grad_x = context.Output<Tensor>(framework::GradVarName("X"));
    auto* grad_y = context.Input<Tensor>(framework::GradVarName("Out"));
    auto* mask = context.Input<Tensor>("Mask");
    grad_x->mutable_data<T>(context.GetPlace());
    auto& dev_ctx = context.template device_context<DeviceContext>();
    auto& dropout_implementation =
        context.Attr<std::string>("dropout_implementation");
    float dropout_prob = context.Attr<float>("dropout_prob");
    const T* mask_data = mask->data<T>();
    framework::Tensor mask_new;
    if (dropout_implementation == "upscale_in_train") {
      mask_new = context.AllocateTmpTensor<T, platform::XPUDeviceContext>(
          mask->dims(), dev_ctx);
      float scale =
          (dropout_prob == 1.0f) ? (1.0f) : (1.0f / (1.0f - dropout_prob));
      int r = xpu::scale(dev_ctx.x_context(),
                         reinterpret_cast<const XPUTyp*>(mask->data<T>()),
                         reinterpret_cast<XPUTyp*>(mask_new.data<T>()),
                         mask->numel(), false, scale, 0.0f);
      PADDLE_ENFORCE_EQ(
          r, xpu::Error_t::SUCCESS,
          platform::errors::External(
              "XPU dropout return wrong value[%d], please check whether "
              "Baidu Kunlun Card is properly installed.",
              r));
      mask_data = mask_new.data<T>();
    }

    int r = xpu::mul(
        dev_ctx.x_context(), reinterpret_cast<const XPUTyp*>(grad_y->data<T>()),
        reinterpret_cast<const XPUTyp*>(mask_data),
        reinterpret_cast<XPUTyp*>(grad_x->data<T>()), grad_y->numel());
    PADDLE_ENFORCE_EQ(
        r, xpu::Error_t::SUCCESS,
        platform::errors::External(
            "XPU dropout return wrong value[%d], please check whether "
            "Baidu Kunlun Card is properly installed.",
            r));
  }
};
}  // namespace operators
}  // namespace paddle
namespace ops = paddle::operators;
namespace plat = paddle::platform;
REGISTER_OP_XPU_KERNEL(
    dropout, ops::DropoutXPUKernel<paddle::platform::XPUDeviceContext, float>,
    ops::DropoutXPUKernel<paddle::platform::XPUDeviceContext, plat::float16>);
REGISTER_OP_XPU_KERNEL(
    dropout_grad,
    ops::DropoutGradXPUKernel<paddle::platform::XPUDeviceContext, float>,
    ops::DropoutGradXPUKernel<paddle::platform::XPUDeviceContext,
                              plat::float16>);
#endif
