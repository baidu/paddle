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
#include "paddle/fluid/operators/conv_op.h"
#include <memory>
#include <string>
#include <vector>
#include "paddle/fluid/platform/cudnn_workspace_helper.h"
#ifdef PADDLE_WITH_XPU
namespace paddle {
namespace operators {

template <typename DeviceContext, typename T>
class GemmConvXPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    const Tensor* input = context.Input<Tensor>("Input");
    // The filter will be reshaped in the calculations,
    // so here use an assignment operation,
    // that avoids modifying the variable in the Scope.
    Tensor filter = *context.Input<Tensor>("Filter");
    Tensor* output = context.Output<Tensor>("Output");
    Tensor* max_input = context.Output<Tensor>("MaxInput");
    Tensor* max_filter = context.Output<Tensor>("MaxFilter");
    max_input->mutable_data<T>(context.GetPlace());
    max_filter->mutable_data<T>(context.GetPlace());
    output->mutable_data<T>(context.GetPlace());
    int groups = context.Attr<int>("groups");
    std::vector<int> strides = context.Attr<std::vector<int>>("strides");
    std::vector<int> paddings = context.Attr<std::vector<int>>("paddings");
    std::vector<int> dilations = context.Attr<std::vector<int>>("dilations");
    const int batch_size = static_cast<int>(input->dims()[0]);
    const int img_c = static_cast<int>(input->dims()[1]);
    const int img_h = static_cast<int>(input->dims()[2]);
    const int img_w = static_cast<int>(input->dims()[3]);
    const int f = static_cast<int>(filter.dims()[0]);
    const int win_h = static_cast<int>(filter.dims()[2]);
    const int win_w = static_cast<int>(filter.dims()[3]);
    PADDLE_ENFORCE_EQ(
        dilations[0] == 1 && dilations[1] == 1, true,
        platform::errors::InvalidArgument("XPU only support dilation == 1."));
    auto& dev_ctx = context.template device_context<DeviceContext>();
    PADDLE_ENFORCE_EQ(
        xpu::findmax(dev_ctx.x_context(), input->data<T>(), input->numel(),
                     max_input->data<T>()) == xpu::Error_t::SUCCESS,
        true, platform::errors::InvalidArgument("XPU kernel error!"));
    PADDLE_ENFORCE_EQ(
        xpu::findmax(dev_ctx.x_context(), filter.data<T>(), filter.numel(),
                     max_filter->data<T>()) == xpu::Error_t::SUCCESS,
        true, platform::errors::InvalidArgument("XPU kernel error!"));
    if (groups == 1) {
      int r = xpu::conv2d_forward_int16<float, float, float, float>(
          dev_ctx.x_context(), batch_size, img_c, img_h, img_w, f, win_h, win_w,
          strides[0], strides[1], paddings[0], paddings[1], dilations[0],
          dilations[1], groups, input->data<float>(), filter.data<float>(),
          output->data<float>(), nullptr, nullptr, xpu::Activation_t::LINEAR,
          // nullptr, nullptr);
          max_input->data<float>(), max_filter->data<float>());
      PADDLE_ENFORCE_EQ(r == xpu::Error_t::SUCCESS, true,
                        platform::errors::InvalidArgument("XPU kernel error!"));
    } else {
      int r = xpu::conv2d_int16_with_group<float, float, float>(
          dev_ctx.x_context(), input->data<float>(), filter.data<float>(),
          output->data<float>(), batch_size, img_c, img_h, img_w, f, win_h,
          win_w, groups, strides[0], strides[1], paddings[0], paddings[1],
          // nullptr, nullptr);
          max_input->data<float>(), max_filter->data<float>());
      PADDLE_ENFORCE_EQ(r == xpu::Error_t::SUCCESS, true,
                        platform::errors::InvalidArgument("XPU kernel error!"));
    }
  }
};
template <typename DeviceContext, typename T>
class GemmConvGradXPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    const Tensor* input = context.Input<Tensor>("Input");
    const Tensor* max_input = context.Input<Tensor>("MaxInput");
    const Tensor* max_filter = context.Input<Tensor>("MaxFilter");
    Tensor* max_output_grad = context.Output<Tensor>("MaxOutputGrad");
    const Tensor* output_grad =
        context.Input<Tensor>(framework::GradVarName("Output"));
    Tensor* input_grad =
        context.Output<Tensor>(framework::GradVarName("Input"));
    Tensor* filter_grad =
        context.Output<Tensor>(framework::GradVarName("Filter"));
    // The filter and filter_grad will be reshaped in the calculations,
    // so here use an assignment operation,
    // that avoids modifying the variable in the Scope.
    Tensor filter = *context.Input<Tensor>("Filter");
    if (!input_grad && !filter_grad) return;
    int groups = context.Attr<int>("groups");
    std::vector<int> strides = context.Attr<std::vector<int>>("strides");
    std::vector<int> paddings = context.Attr<std::vector<int>>("paddings");
    std::vector<int> dilations = context.Attr<std::vector<int>>("dilations");
    const int batch_size = static_cast<int>(input->dims()[0]);
    PADDLE_ENFORCE_EQ(groups == 1, true, platform::errors::InvalidArgument(
                                             "XPU only support groups == 1."));
    PADDLE_ENFORCE_EQ(
        dilations[0] == 1 && dilations[1] == 1, true,
        platform::errors::InvalidArgument("XPU only support dilation == 1."));
    const int img_c = static_cast<int>(input->dims()[1]);
    const int img_h = static_cast<int>(input->dims()[2]);
    const int img_w = static_cast<int>(input->dims()[3]);
    const int f = static_cast<int>(filter.dims()[0]);
    const int win_h = static_cast<int>(filter.dims()[2]);
    const int win_w = static_cast<int>(filter.dims()[3]);
    if (input_grad) {
      input_grad->mutable_data<T>(context.GetPlace());
    }
    if (filter_grad) {
      filter_grad->mutable_data<T>(context.GetPlace());
    }
    auto& dev_ctx = context.template device_context<DeviceContext>();
    max_output_grad->Resize({4});
    max_output_grad->mutable_data<T>(context.GetPlace());
    PADDLE_ENFORCE_EQ(
        xpu::findmax(dev_ctx.x_context(), output_grad->data<T>(),
                     output_grad->numel(),
                     max_output_grad->data<T>()) == xpu::Error_t::SUCCESS,
        true, platform::errors::InvalidArgument("XPU kernel error!"));
    if (input_grad) {
      int r = xpu::conv2d_backward_int16(
          dev_ctx.x_context(), batch_size, img_c, img_h, img_w, f, win_h, win_w,
          strides[0], strides[1], paddings[0], paddings[1], dilations[0],
          dilations[1], groups, output_grad->data<float>(),
          filter.data<float>(), input_grad->data<float>(),
          // nullptr, nullptr,
          max_output_grad->data<float>(), max_filter->data<float>());
      PADDLE_ENFORCE_EQ(r == xpu::Error_t::SUCCESS, true,
                        platform::errors::InvalidArgument("XPU kernel error!"));
    }
    if (filter_grad) {
      int r = xpu::conv2d_backward_weight_int16(
          dev_ctx.x_context(), batch_size, img_c, img_h, img_w, f, win_h, win_w,
          strides[0], strides[1], paddings[0], paddings[1], dilations[0],
          dilations[1], groups, output_grad->data<float>(),
          input->data<float>(), filter_grad->data<float>(),
          // nullptr, nullptr,
          max_output_grad->data<float>(), max_input->data<float>());
      PADDLE_ENFORCE_EQ(r == xpu::Error_t::SUCCESS, true,
                        platform::errors::InvalidArgument("XPU kernel error!"));
    }
  }
};
}  // namespace operators
}  // namespace paddle
namespace ops = paddle::operators;
// TODO(xingzhaolong): neon kernel for mobile
REGISTER_OP_XPU_KERNEL(
    depthwise_conv2d,
    ops::GemmConvXPUKernel<paddle::platform::XPUDeviceContext, float>);
REGISTER_OP_XPU_KERNEL(
    conv2d, ops::GemmConvXPUKernel<paddle::platform::XPUDeviceContext, float>);
REGISTER_OP_XPU_KERNEL(
    conv2d_grad,
    ops::GemmConvGradXPUKernel<paddle::platform::XPUDeviceContext, float>);
#endif
