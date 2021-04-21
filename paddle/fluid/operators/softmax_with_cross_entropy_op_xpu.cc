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

#include "paddle/fluid/operators/softmax_with_cross_entropy_op.h"
#ifdef PADDLE_WITH_XPU
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "xpu/refactor/math.h"
#include "xpu/refactor/nn.h"

namespace paddle {
namespace operators {

template <typename T>
class SoftmaxWithCrossEntropyXPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    PADDLE_ENFORCE_EQ(
        platform::is_xpu_place(context.GetPlace()), true,
        platform::errors::PreconditionNotMet("This kernel only runs on XPU."));
    const Tensor* logits = context.Input<Tensor>("Logits");
    const Tensor* labels = context.Input<Tensor>("Label");
    Tensor* softmax = context.Output<Tensor>("Softmax");
    Tensor* loss = context.Output<Tensor>("Loss");
    const int rank = logits->dims().size();
    const int axis = CanonicalAxis(context.Attr<int>("axis"), rank);
    PADDLE_ENFORCE_EQ(axis, rank - 1, platform::errors::InvalidArgument(
                                          "axis should == rank - 1"));
    softmax->mutable_data<T>(context.GetPlace());
    loss->mutable_data<T>(context.GetPlace());
    const int n = SizeToAxis(axis, logits->dims());
    const int d = SizeFromAxis(axis, logits->dims());
    std::vector<int> logits_dims = framework::vectorize<int>(logits->dims());

    // softmax
    auto& dev_ctx =
        context.template device_context<platform::XPUDeviceContext>();
    int r = XPU_SUCCESS;
    Tensor clip_logits;
    int len = logits->numel();
    T* clip_logits_data =
        clip_logits.mutable_data<T>(context.GetPlace(), len * sizeof(T));
    r = xpu::clip(dev_ctx.x_context(), logits->data<float>(), clip_logits_data,
                  len, -1e30, 1e30);
    PADDLE_ENFORCE_EQ(
        r, xpu::Error_t::SUCCESS,
        platform::errors::External("XPU kernel error. clip "
                                   "execution not succeed, error code=%d",
                                   r));

    r = xpu::softmax(dev_ctx.x_context(), clip_logits_data,
                     softmax->data<float>(), logits_dims, axis);

    PADDLE_ENFORCE_EQ(
        r, xpu::Error_t::SUCCESS,
        platform::errors::External("XPU kernel error. Softmax2d_forward "
                                   "execution not succeed, error code=%d",
                                   r));
    // cross_entropy
    auto ignore_index = context.Attr<int>("ignore_index");
    const bool soft_label = context.Attr<bool>("soft_label");
    if (soft_label) {
      r = xpu::soft_cross_entropy<float>(
          dev_ctx.x_context(), softmax->data<float>(), labels->data<float>(),
          loss->data<float>(), n, d);
      PADDLE_ENFORCE_EQ(
          r, xpu::Error_t::SUCCESS,
          platform::errors::External("XPU kernel error. soft_cross_entropy "
                                     "execution not succeed, error code=%d",
                                     r));
    } else {
      Tensor labels_int32;
      labels_int32.mutable_data<int32_t>(context.GetPlace(),
                                         labels->numel() * sizeof(int32_t));
      r = xpu::cast_v2<int64_t, int32_t>(
          dev_ctx.x_context(), labels->data<int64_t>(),
          labels_int32.data<int32_t>(), labels->numel());
      PADDLE_ENFORCE_EQ(
          r, xpu::Error_t::SUCCESS,
          platform::errors::External("XPU kernel error. cast_v2 "
                                     "execution not succeed, error code=%d",
                                     r));

      r = xpu::hard_cross_entropy<float, int32_t>(
          dev_ctx.x_context(), softmax->data<float>(),
          labels_int32.data<int32_t>(), loss->data<float>(), nullptr, n, d,
          ignore_index);
      PADDLE_ENFORCE_EQ(
          r, xpu::Error_t::SUCCESS,
          platform::errors::External("XPU kernel error. hard_cross_entropy "
                                     "execution not succeed, error code=%d",
                                     r));
    }
  }
};

template <typename T>
class SoftmaxWithCrossEntropyGradXPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    PADDLE_ENFORCE_EQ(
        platform::is_xpu_place(context.GetPlace()), true,
        platform::errors::PreconditionNotMet("This kernel only runs on XPU."));
    const Tensor* softmax = context.Input<Tensor>("Softmax");
    const Tensor* labels = context.Input<Tensor>("Label");
    const Tensor* loss_grad =
        context.Input<Tensor>(framework::GradVarName("Loss"));
    Tensor* logits_grad =
        context.Output<Tensor>(framework::GradVarName("Logits"));
    const int rank = logits_grad->dims().size();
    const int axis = CanonicalAxis(context.Attr<int>("axis"), rank);
    PADDLE_ENFORCE_EQ(axis, rank - 1, platform::errors::InvalidArgument(
                                          "axis should == rank - 1"));

    const int n = SizeToAxis(axis, logits_grad->dims());
    const int d = SizeFromAxis(axis, logits_grad->dims());
    std::vector<int> softmax_dims = framework::vectorize<int>(softmax->dims());

    if (logits_grad) {
      logits_grad->mutable_data<float>(context.GetPlace());
    }
    Tensor softmax_grad;
    softmax_grad.mutable_data<float>(softmax->dims(), context.GetPlace());

    auto& dev_ctx =
        context.template device_context<paddle::platform::XPUDeviceContext>();

    // cross_entropy_grad
    const bool soft_label = context.Attr<bool>("soft_label");
    auto ignore_index = context.Attr<int>("ignore_index");
    if (soft_label) {
      int r = xpu::soft_cross_entropy_grad<float>(
          dev_ctx.x_context(), softmax->data<float>(), labels->data<float>(),
          loss_grad->data<float>(), softmax_grad.data<float>(), n, d);
      PADDLE_ENFORCE_EQ(r, xpu::Error_t::SUCCESS,
                        platform::errors::External(
                            "XPU kernel error. soft_cross_entropy_grad "
                            "execution not succeed, error code=%d, %s",
                            r, XPUAPIErrorMsg[r]));
    } else {
      Tensor labels_int32;
      labels_int32.mutable_data<int32_t>(context.GetPlace(),
                                         labels->numel() * sizeof(int32_t));
      int r = xpu::cast_v2<int64_t, int32_t>(
          dev_ctx.x_context(), labels->data<int64_t>(),
          labels_int32.data<int32_t>(), labels->numel());
      PADDLE_ENFORCE_EQ(
          r, xpu::Error_t::SUCCESS,
          platform::errors::External("XPU kernel error. cast_v2 "
                                     "execution not succeed, error code=%d, %s",
                                     r, XPUAPIErrorMsg[r]));

      Tensor match_x_cpu;
      auto* match_x_data = match_x_cpu.mutable_data<T>(
          framework::make_ddim({n}), platform::CPUPlace());
      Tensor softmax_copy;
      TensorCopySync(*softmax, platform::CPUPlace(), &softmax_copy);
      Tensor label_copy;
      TensorCopySync(*labels, platform::CPUPlace(), &label_copy);
      auto* softmax_data = softmax_copy.data<T>();
      auto* label_data = label_copy.data<int64_t>();
      for (int i = 0; i < n; i++) {
        T val = static_cast<T>(softmax_data[i * d + label_data[i]]);
        match_x_data[i] = val;
      }
      Tensor match_x;
      TensorCopySync(match_x_cpu, context.GetPlace(), &match_x);
      r = xpu::hard_cross_entropy_grad<float, int32_t>(
          dev_ctx.x_context(), labels_int32.data<int32_t>(), match_x.data<T>(),
          loss_grad->data<float>(), softmax_grad.data<float>(), n, d,
          ignore_index);
      PADDLE_ENFORCE_EQ(r, xpu::Error_t::SUCCESS,
                        platform::errors::External(
                            "XPU kernel error. hard_cross_entropy_grad "
                            "execution not succeed, error code=%d, %s",
                            r, XPUAPIErrorMsg[r]));
      dev_ctx.Wait();
    }

    // softmax_grad
    int r = xpu::softmax_grad(dev_ctx.x_context(), softmax->data<float>(),
                              softmax_grad.data<float>(),
                              logits_grad->data<T>(), softmax_dims, axis);
    PADDLE_ENFORCE_EQ(
        r, xpu::Error_t::SUCCESS,
        platform::errors::External("XPU kernel error. softmax_grad "
                                   "execution not succeed, error code=%d, %s",
                                   r, XPUAPIErrorMsg[r]));
  }
};
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_XPU_KERNEL(softmax_with_cross_entropy,
                       ops::SoftmaxWithCrossEntropyXPUKernel<float>);
REGISTER_OP_XPU_KERNEL(softmax_with_cross_entropy_grad,
                       ops::SoftmaxWithCrossEntropyGradXPUKernel<float>);
#endif
