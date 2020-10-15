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

namespace paddle {
namespace operators {

template <typename T>
class SoftmaxWithCrossEntropyXPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    PADDLE_ENFORCE_EQ(
        platform::is_xpu_place(context.GetPlace()), true,
        platform::errors::InvalidArgument("This kernel only runs on XPU."));
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
    // softmax
    auto& dev_ctx =
        context.template device_context<platform::XPUDeviceContext>();
    int r = xpu::softmax2d_forward(dev_ctx.x_context(), logits->data<float>(),
                                   softmax->data<float>(), n, d);
    PADDLE_ENFORCE_EQ(r, xpu::Error_t::SUCCESS,
                      platform::errors::InvalidArgument("XPU kernel error!"));
    // cross_entropy
    auto ignore_index = context.Attr<int>("ignore_index");
    const bool soft_label = context.Attr<bool>("soft_label");
    if (soft_label) {
      PADDLE_THROW(platform::errors::InvalidArgument(
          "XPU only support soft_label == false for now!"));
    } else {
      auto* p_labels = labels->data<int64_t>();
      int64_t* labels_int64_host =
          reinterpret_cast<int64_t*>(std::malloc(n * sizeof(int64_t)));
      int* labels_int32_host =
          reinterpret_cast<int*>(std::malloc(n * sizeof(int)));
      int* labels_int32_device = NULL;
      PADDLE_ENFORCE_EQ(
          xpu_malloc(reinterpret_cast<void**>(&labels_int32_device),
                     n * sizeof(int)),
          XPU_SUCCESS, platform::errors::InvalidArgument("XPU kernel error!"));
      dev_ctx.Wait();
      memory::Copy(platform::CPUPlace(), labels_int64_host,
                   BOOST_GET_CONST(platform::XPUPlace, context.GetPlace()),
                   p_labels, n * sizeof(int64_t));
      for (int i = 0; i < n; ++i) {
        labels_int32_host[i] = labels_int64_host[i];
      }
      memory::Copy(BOOST_GET_CONST(platform::XPUPlace, context.GetPlace()),
                   labels_int32_device, platform::CPUPlace(), labels_int32_host,
                   n * sizeof(int));
      int r = xpu::cross_entropy_forward(
          dev_ctx.x_context(), n, d, softmax->data<float>(),
          labels_int32_device, loss->data<float>(), nullptr, ignore_index);
      PADDLE_ENFORCE_EQ(r, xpu::Error_t::SUCCESS,
                        platform::errors::InvalidArgument("XPU kernel error!"));
      dev_ctx.Wait();
      std::free(labels_int32_host);
      std::free(labels_int64_host);
      xpu_free(labels_int32_device);
    }
  }
};
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_XPU_KERNEL(softmax_with_cross_entropy,
                       ops::SoftmaxWithCrossEntropyXPUKernel<float>);
#endif
