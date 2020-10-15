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
#include "paddle/fluid/operators/pool_op.h"
#include <unordered_map>

#ifdef PADDLE_WITH_XPU
namespace paddle {
namespace operators {

xpu::Pooling_t XPUPoolingType(const std::string& pooltype, bool exclusive,
                              bool is_test) {
  if (pooltype == "max") {
    return xpu::Pooling_t::MAX_WITHOUT_INDEX;
  } else if (pooltype == "avg") {
    if (exclusive) {
      return xpu::Pooling_t::AVG_WITHOUT_PAD;
    } else {
      return xpu::Pooling_t::AVG_WITH_PAD;
    }
  } else {
    PADDLE_THROW(platform::errors::InvalidArgument(
        "Pool op only supports 2D and 3D input."));
  }
}
template <typename DeviceContext, typename T>
class PoolXPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    const Tensor* in_x = context.Input<Tensor>("X");
    Tensor* out = context.Output<Tensor>("Out");
    std::string pooling_type = context.Attr<std::string>("pooling_type");
    std::vector<int> ksize = context.Attr<std::vector<int>>("ksize");
    std::vector<int> strides = context.Attr<std::vector<int>>("strides");
    std::vector<int> paddings = context.Attr<std::vector<int>>("paddings");
    bool exclusive = context.Attr<bool>("exclusive");
    bool is_test = context.Attr<bool>("is_test");
    bool adaptive = context.Attr<bool>("adaptive");
    PADDLE_ENFORCE_EQ(!adaptive, true,
                      platform::errors::InvalidArgument(
                          "XPU does not support adaptive == true!"));
    PADDLE_ENFORCE_EQ(ksize.size(), 2,
                      platform::errors::InvalidArgument(
                          "XPU only support 2 dimension pooling!"));
    int* index_data = nullptr;
    if (context.Attr<bool>("global_pooling")) {
      for (size_t i = 0; i < ksize.size(); ++i) {
        paddings[i] = 0;
        ksize[i] = static_cast<int>(in_x->dims()[i + 2]);
      }
    }
    const int c = in_x->dims()[0] * in_x->dims()[1];
    const int in_h = in_x->dims()[2];
    const int in_w = in_x->dims()[3];
    const int out_h = out->dims()[2];
    const int out_w = out->dims()[3];
    const int win_h = ksize[0];
    const int win_w = ksize[1];
    const int stride_h = strides[0];
    const int stride_w = strides[1];
    const int pad_up = paddings[0];
    const int pad_down = paddings[0];
    const int pad_left = paddings[1];
    const int pad_right = paddings[1];
    const float* input = in_x->data<float>();
    out->mutable_data<T>(context.GetPlace());
    float* output = out->data<float>();
    xpu::Pooling_t pool_type = XPUPoolingType(pooling_type, exclusive, is_test);
    auto& dev_ctx = context.template device_context<DeviceContext>();
    int r = xpu::pooling_forward<float, float>(
        dev_ctx.x_context(), input, output, index_data, pool_type, c, in_h,
        in_w, pad_left, pad_right, pad_up, pad_down, win_h, win_w, stride_h,
        stride_w, out_h, out_w);
    PADDLE_ENFORCE_EQ(
        r, xpu::Error_t::SUCCESS,
        platform::errors::InvalidArgument("pool2d XPU kernel error!"));
  }
};
template <typename DeviceContext, typename T>
class PoolGradXPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    const Tensor* in_x = context.Input<Tensor>("X");
    const Tensor* out = context.Input<Tensor>("Out");
    const Tensor* out_grad =
        context.Input<Tensor>(framework::GradVarName("Out"));
    Tensor* in_x_grad = context.Output<Tensor>(framework::GradVarName("X"));
    std::string pooling_type = context.Attr<std::string>("pooling_type");
    std::vector<int> ksize = context.Attr<std::vector<int>>("ksize");
    std::vector<int> strides = context.Attr<std::vector<int>>("strides");
    std::vector<int> paddings = context.Attr<std::vector<int>>("paddings");
    bool exclusive = context.Attr<bool>("exclusive");
    bool adaptive = context.Attr<bool>("adaptive");
    const int* index_data = nullptr;
    PADDLE_ENFORCE_EQ(!adaptive, true,
                      platform::errors::InvalidArgument(
                          "XPU does not support adaptive == true!"));
    PADDLE_ENFORCE_EQ(ksize.size(), 2,
                      platform::errors::InvalidArgument(
                          "XPU only support 2 dimension pooling!"));
    if (context.Attr<bool>("global_pooling")) {
      for (size_t i = 0; i < ksize.size(); ++i) {
        paddings[i] = 0;
        ksize[i] = static_cast<int>(in_x->dims()[i + 2]);
      }
    }
    if (!in_x_grad) {
      return;
    }
    const int c = in_x->dims()[0] * in_x->dims()[1];
    const int in_h = in_x->dims()[2];
    const int in_w = in_x->dims()[3];
    const int out_h = out->dims()[2];
    const int out_w = out->dims()[3];
    const int win_h = ksize[0];
    const int win_w = ksize[1];
    const int stride_h = strides[0];
    const int stride_w = strides[1];
    const int pad_up = paddings[0];
    const int pad_down = paddings[0];
    const int pad_left = paddings[1];
    const int pad_right = paddings[1];
    const float* input = in_x->data<float>();
    const float* output = out->data<float>();
    const float* output_grad = out_grad->data<float>();
    in_x_grad->mutable_data<T>(context.GetPlace());
    float* input_grad = in_x_grad->data<float>();
    xpu::Pooling_t pool_type = XPUPoolingType(pooling_type, exclusive, false);
    auto& dev_ctx = context.template device_context<DeviceContext>();
    // Need to init memory in the first place
    const int zero = 0;
    int r =
        xpu::memset(dev_ctx.x_context(), reinterpret_cast<void**>(input_grad),
                    zero, in_x_grad->numel() * sizeof(float));
    PADDLE_ENFORCE_EQ(r, xpu::Error_t::SUCCESS,
                      platform::errors::InvalidArgument(
                          "There are pool2d grad XPU kernel error raised!"));
    r = xpu::pooling_backward(dev_ctx.x_context(), input, output, index_data,
                              output_grad, input_grad, pool_type, c, in_h, in_w,
                              pad_left, pad_right, pad_up, pad_down, win_h,
                              win_w, stride_h, stride_w, out_h, out_w);
    PADDLE_ENFORCE_EQ(r, xpu::Error_t::SUCCESS,
                      platform::errors::InvalidArgument(
                          "There are pool2d grad XPU kernel error raised!"));
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_XPU_KERNEL(
    pool2d, ops::PoolXPUKernel<paddle::platform::XPUDeviceContext, float>);
REGISTER_OP_XPU_KERNEL(
    pool2d_grad,
    ops::PoolGradXPUKernel<paddle::platform::XPUDeviceContext, float>);

#endif
