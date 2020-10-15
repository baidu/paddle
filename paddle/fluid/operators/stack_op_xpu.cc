// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "paddle/fluid/operators/stack_op.h"
#ifdef PADDLE_WITH_XPU

namespace paddle {
namespace operators {

using framework::Tensor;
template <typename DeviceContext, typename T>
class StackXPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto x = ctx.MultiInput<Tensor>("X");
    auto* y = ctx.Output<Tensor>("Y");
    int axis = ctx.Attr<int>("axis");
    if (axis < 0) {
      axis += (x[0]->dims().size() + 1);
    }
    int n = static_cast<int>(x.size());
    PADDLE_ENFORCE_LE(n, 24,
                      platform::errors::InvalidArgument(
                          "XPU only surpport at most 24 tensors for now"));
    auto* y_data = y->mutable_data<T>(ctx.GetPlace());
    int pre = 1, post = 1;
    auto& dim = x[0]->dims();
    for (auto i = 0; i < axis; ++i) {
      pre *= dim[i];
    }
    for (auto i = axis; i < dim.size(); ++i) {
      post *= dim[i];
    }
    auto& dev_ctx = ctx.template device_context<DeviceContext>();
    void* x_datas_host = std::malloc(n * sizeof(void*));
    void* x_datas_device = nullptr;
    PADDLE_ENFORCE(xpu_malloc(reinterpret_cast<void**>(&x_datas_device),
                              n * sizeof(void*)) == XPU_SUCCESS);
    for (auto i = 0; i < n; ++i) {
      ((const void**)x_datas_host)[i] = x[i]->data<T>();
    }
    memory::Copy(BOOST_GET_CONST(platform::XPUPlace, ctx.GetPlace()),
                 x_datas_device, platform::CPUPlace(), x_datas_host,
                 n * sizeof(void*));
    int r = xpu::stack_forward<float>(dev_ctx.x_context(), pre, post, n,
                                      x_datas_device, y_data);
    PADDLE_ENFORCE_EQ(r, xpu::Error_t::SUCCESS,
                      platform::errors::InvalidArgument(
                          "There are stack XPU kernel error raised!"));
    dev_ctx.Wait();
    std::free(x_datas_host);
    xpu_free(x_datas_device);
  }
};

}  // namespace operators
}  // namespace paddle

namespace plat = paddle::platform;
namespace ops = paddle::operators;

REGISTER_OP_XPU_KERNEL(stack,
                       ops::StackXPUKernel<plat::XPUDeviceContext, float>);
#endif
