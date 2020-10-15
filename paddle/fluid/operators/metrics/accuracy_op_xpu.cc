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

#include "paddle/fluid/operators/metrics/accuracy_op.h"
#include "paddle/fluid/platform/xpu_header.h"

namespace paddle {
namespace operators {

template <typename DeviceContext, typename T>
class AccuracyXPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* inference = ctx.Input<Tensor>("Out");
    auto* indices = ctx.Input<Tensor>("Indices");
    auto* label = ctx.Input<Tensor>("Label");
    auto* accuracy = ctx.Output<Tensor>("Accuracy");
    auto* correct = ctx.Output<Tensor>("Correct");
    auto* total = ctx.Output<Tensor>("Total");
    int* correct_data = correct->mutable_data<int>(ctx.GetPlace());
    int* total_data = total->mutable_data<int>(ctx.GetPlace());
    float* accuracy_data = accuracy->mutable_data<float>(ctx.GetPlace());
    const int64_t* indices_data = indices->data<int64_t>();
    const int64_t* label_data = label->data<int64_t>();
    size_t num_samples = inference->dims()[0];
    size_t class_dim = inference->dims()[1];
    if (num_samples == 0) {
      return;
    }
    size_t indices_int32_size = num_samples * class_dim * sizeof(int);
    size_t indices_int64_size = num_samples * class_dim * sizeof(int64_t);
    size_t label_int32_size = num_samples * sizeof(int);
    size_t label_int64_size = num_samples * sizeof(int64_t);
    auto& dev_ctx = ctx.template device_context<DeviceContext>();
    int* indices_int32_device = NULL;
    PADDLE_ENFORCE_EQ(
        xpu_malloc(reinterpret_cast<void**>(&indices_int32_device),
                   indices_int32_size),
        XPU_SUCCESS,
        platform::errors::ResourceExhausted(
            "\n\nOut of memory error on XPU, Cannot allocate %s memory"
            " on XPU. \n\nPlease check whether there is any other process "
            "using XPU.\n",
            string::HumanReadableSize(indices_int32_size)));
    int* label_int32_device = NULL;
    PADDLE_ENFORCE_EQ(
        xpu_malloc(reinterpret_cast<void**>(&label_int32_device),
                   label_int32_size),
        XPU_SUCCESS,
        platform::errors::ResourceExhausted(
            "\n\nOut of memory error on XPU, Cannot allocate %s memory"
            " on XPU. \n\nPlease check whether there is any other process "
            "using XPU.\n",
            string::HumanReadableSize(label_int32_size)));

    int* indices_int32_host =
        reinterpret_cast<int*>(std::malloc(indices_int32_size));
    int64_t* indices_int64_host =
        reinterpret_cast<int64_t*>(std::malloc(indices_int64_size));
    int* label_int32_host =
        reinterpret_cast<int*>(std::malloc(label_int32_size));
    int64_t* label_int64_host =
        reinterpret_cast<int64_t*>(std::malloc(label_int64_size));
    dev_ctx.Wait();
    memory::Copy(platform::CPUPlace(), indices_int64_host,
                 BOOST_GET_CONST(platform::XPUPlace, ctx.GetPlace()),
                 indices_data, indices_int64_size);
    memory::Copy(platform::CPUPlace(), label_int64_host,
                 BOOST_GET_CONST(platform::XPUPlace, ctx.GetPlace()),
                 label_data, label_int64_size);
    for (size_t i = 0; i < num_samples; ++i) {
      label_int32_host[i] = label_int64_host[i];
      for (size_t j = 0; j < class_dim; ++j) {
        indices_int32_host[i * class_dim + j] =
            indices_int64_host[i * class_dim + j];
      }
    }
    memory::Copy(BOOST_GET_CONST(platform::XPUPlace, ctx.GetPlace()),
                 indices_int32_device, platform::CPUPlace(), indices_int32_host,
                 indices_int32_size);
    memory::Copy(BOOST_GET_CONST(platform::XPUPlace, ctx.GetPlace()),
                 label_int32_device, platform::CPUPlace(), label_int32_host,
                 label_int32_size);
    int r = xpu::accuracy(dev_ctx.x_context(), indices_int32_device,
                          label_int32_device, num_samples, class_dim,
                          correct_data, total_data, accuracy_data);
    PADDLE_ENFORCE_EQ(r, xpu::Error_t::SUCCESS,
                      platform::errors::Fatal("XPU accuracy kernel error!"));
    dev_ctx.Wait();
    xpu_free(indices_int32_device);
    xpu_free(label_int32_device);
    std::free(indices_int32_host);
    std::free(indices_int64_host);
    std::free(label_int32_host);
    std::free(label_int64_host);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_XPU_KERNEL(
    accuracy,
    ops::AccuracyXPUKernel<paddle::platform::XPUDeviceContext, float>);

#endif
