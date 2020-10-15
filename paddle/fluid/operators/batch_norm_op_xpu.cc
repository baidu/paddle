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

#ifdef PADDLE_WITH_XPU

#include "paddle/fluid/operators/batch_norm_op.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
using DDim = framework::DDim;

template <typename DeviceContext, typename T>
class BatchNormXPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    const auto epsilon = ctx.Attr<float>("epsilon");
    const auto momentum = ctx.Attr<float>("momentum");
    const auto is_test = ctx.Attr<bool>("is_test");
    const auto use_global_stats = ctx.Attr<bool>("use_global_stats");
    const auto trainable_stats = ctx.Attr<bool>("trainable_statistics");
    bool test_mode = is_test && (!trainable_stats);
    bool global_stats = test_mode || use_global_stats;
    const auto& data_layout_str = ctx.Attr<std::string>("data_layout");
    const auto data_layout = framework::StringToDataLayout(data_layout_str);
    PADDLE_ENFORCE_EQ(data_layout, DataLayout::kNCHW,
                      platform::errors::InvalidArgument(
                          "The 'data_layout' attribute must be NCHW. But "
                          "recevived 'data_layout' is [%s].",
                          data_layout_str));
    const auto* x = ctx.Input<Tensor>("X");
    const auto& x_dims = x->dims();
    PADDLE_ENFORCE_EQ(x_dims.size(), 4,
                      platform::errors::InvalidArgument(
                          "The input tensor X's dimension must equal to 4. But "
                          "received X's shape = [%s], X's dimension = [%d].",
                          x_dims, x_dims.size()));
    const int N = x_dims[0];
    const int C = x_dims[1];
    const int H = x_dims[2];
    const int W = x_dims[3];
    const auto* scale = ctx.Input<Tensor>("Scale");
    const auto* bias = ctx.Input<Tensor>("Bias");
    const auto* x_data = x->data<T>();
    const auto* scale_data = scale->data<T>();
    const auto* bias_data = bias->data<T>();
    auto* y = ctx.Output<Tensor>("Y");
    auto* y_data = y->mutable_data<T>(ctx.GetPlace());
    auto& dev_ctx = ctx.template device_context<DeviceContext>();
    if (!global_stats) {
      auto* mean_out = ctx.Output<Tensor>("MeanOut");
      auto* variance_out = ctx.Output<Tensor>("VarianceOut");
      auto* saved_mean = ctx.Output<Tensor>("SavedMean");
      auto* saved_variance = ctx.Output<Tensor>("SavedVariance");
      mean_out->mutable_data<T>(ctx.GetPlace());
      variance_out->mutable_data<T>(ctx.GetPlace());
      saved_mean->mutable_data<T>(ctx.GetPlace());
      saved_variance->mutable_data<T>(ctx.GetPlace());
      auto* mean_out_data = mean_out->data<T>();
      auto* variance_out_data = variance_out->data<T>();
      auto* saved_mean_data = saved_mean->data<T>();
      auto* saved_variance_data = saved_variance->data<T>();
      int r = xpu::batch_norm_train_forward(
          dev_ctx.x_context(), epsilon, momentum, N, C, H, W, x_data, y_data,
          scale_data, bias_data, mean_out_data, variance_out_data,
          saved_mean_data, saved_variance_data);
      PADDLE_ENFORCE_EQ(
          r, XPU_SUCCESS,
          platform::errors::External("XPU API(batch_norm_train_forward) return "
                                     "wrong value[%d], please check whether "
                                     "Baidu Kunlun Card is properly installed.",
                                     r));
    } else {
      const auto* mean = ctx.Input<Tensor>("Mean");
      const auto* variance = ctx.Input<Tensor>("Variance");
      const auto* mean_data = mean->data<T>();
      const auto* variance_data = variance->data<T>();
      int r = xpu::batch_norm_infer_forward(
          dev_ctx.x_context(), epsilon, N, C, H, W, x_data, y_data, scale_data,
          bias_data, mean_data, variance_data);
      PADDLE_ENFORCE_EQ(
          r, XPU_SUCCESS,
          platform::errors::External("XPU API(batch_norm_infer_forward) return "
                                     "wrong value[%d], please check whether "
                                     "Baidu Kunlun Card is properly installed.",
                                     r));
    }
  }
};

template <typename DeviceContext, typename T>
class BatchNormGradXPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    const auto* x = ctx.Input<Tensor>("X");
    const auto* dy = ctx.Input<Tensor>(framework::GradVarName("Y"));
    const auto* scale = ctx.Input<Tensor>("Scale");
    const auto* saved_mean = ctx.Input<Tensor>("SavedMean");
    // SavedVariance have been reverted in forward operator
    const auto* saved_inv_variance = ctx.Input<Tensor>("SavedVariance");
    const auto& data_layout_str = ctx.Attr<std::string>("data_layout");
    const auto data_layout = framework::StringToDataLayout(data_layout_str);
    PADDLE_ENFORCE_EQ(data_layout, DataLayout::kNCHW,
                      platform::errors::InvalidArgument(
                          "The 'data_layout' attribute must be NCHW. But "
                          "recevived 'data_layout' is [%s].",
                          data_layout_str));
    const auto& x_dims = x->dims();
    PADDLE_ENFORCE_EQ(x_dims.size(), 4,
                      platform::errors::InvalidArgument(
                          "The input tensor X's dimension must equal to 4. But "
                          "received X's shape = [%s], X's dimension = [%d].",
                          x_dims, x_dims.size()));
    const int N = x_dims[0];
    const int C = x_dims[1];
    const int H = x_dims[2];
    const int W = x_dims[3];
    const auto* x_data = x->data<T>();
    const auto* dy_data = dy->data<T>();
    const auto* scale_data = scale->data<T>();
    const auto* saved_mean_data = saved_mean->data<T>();
    const auto* saved_inv_variance_data = saved_inv_variance->data<T>();
    auto* dx = ctx.Output<Tensor>(framework::GradVarName("X"));
    auto* dscale = ctx.Output<Tensor>(framework::GradVarName("Scale"));
    auto* dbias = ctx.Output<Tensor>(framework::GradVarName("Bias"));
    auto* dx_data = dx->mutable_data<T>(ctx.GetPlace());
    auto* dscale_data = dscale->mutable_data<T>(ctx.GetPlace());
    auto* dbias_data = dbias->mutable_data<T>(ctx.GetPlace());
    auto& dev_ctx = ctx.template device_context<DeviceContext>();
    int r = xpu::batch_norm_backward(dev_ctx.x_context(), N, C, H, W, x_data,
                                     dy_data, scale_data, saved_mean_data,
                                     saved_inv_variance_data, dx_data,
                                     dscale_data, dbias_data);
    PADDLE_ENFORCE_EQ(
        r, XPU_SUCCESS,
        platform::errors::External("XPU API(batch_norm_infer_forward) return "
                                   "wrong value[%d], please check whether "
                                   "Baidu Kunlun Card is properly installed.",
                                   r));
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OP_XPU_KERNEL(
    batch_norm,
    ops::BatchNormXPUKernel<paddle::platform::XPUDeviceContext, float>);
REGISTER_OP_XPU_KERNEL(
    batch_norm_grad,
    ops::BatchNormGradXPUKernel<paddle::platform::XPUDeviceContext, float>);

#endif  // PADDLE_WITH_XPU
