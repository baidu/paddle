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

#include "paddle/fluid/operators/sum_op.h"
#include <vector>
#include "paddle/fluid/platform/xpu_header.h"

namespace paddle {
namespace operators {
using framework::Tensor;

template <typename DeviceContext, typename T>
class SumXPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &context) const override {
    auto in_vars = context.MultiInputVar("X");
    auto out_var = context.OutputVar("Out");
    auto *out = context.Output<LoDTensor>("Out");
    bool in_place = out_var == in_vars[0];
    int N = in_vars.size();
    PADDLE_ENFORCE_EQ(
        out_var->IsType<framework::LoDTensor>(), true,
        platform::errors::InvalidArgument("XPU only surpport LodTensor"));
    if (!in_place) {
      out->mutable_data<T>(context.GetPlace());
    }
    auto &dev_ctx = context.template device_context<DeviceContext>();
    std::vector<const float *> ptrs(N, nullptr);
    int valid_count = 0;
    for (int i = 0; i < N; ++i) {
      PADDLE_ENFORCE_EQ(
          in_vars[i]->IsType<framework::LoDTensor>(), true,
          platform::errors::InvalidArgument("XPU only surpport LodTensor"));
      auto &in_t = in_vars[i]->Get<framework::LoDTensor>();
      if (in_t.numel() == 0) {
        continue;
      }
      ptrs[valid_count] = reinterpret_cast<const float *>(in_t.data<T>());
      valid_count++;
    }
    int r = xpu::sum_batch(dev_ctx.x_context(), ptrs.data(), out->data<T>(),
                           valid_count, out->numel());
    PADDLE_ENFORCE_EQ(r, xpu::Error_t::SUCCESS,
                      platform::errors::Fatal("XPU sum kernel error!"));
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OP_XPU_KERNEL(
    sum, ops::SumXPUKernel<paddle::platform::XPUDeviceContext, float>);
#endif
