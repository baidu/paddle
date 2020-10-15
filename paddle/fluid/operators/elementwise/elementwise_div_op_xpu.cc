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
#include "paddle/fluid/operators/elementwise/elementwise_div_op.h"
#include "paddle/fluid/operators/elementwise/elementwise_op.h"
#include "paddle/fluid/operators/elementwise/elementwise_xpu.h"
namespace paddle {
namespace operators {

template <typename T>
struct XPUDivFunctor {
  int operator()(xpu::Context* ctx, const T* x, const T* y, T* z, int len) {
    return xpu::elementwise_div(ctx, x, y, z, len);
  }
};

template <typename DeviceContext, typename T>
class ElementwiseDivXPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    XPUElementwise<T, XPUDivFunctor<T>>(ctx);
  }
};

}  // namespace operators
}  // namespace paddle
namespace ops = paddle::operators;
REGISTER_OP_XPU_KERNEL(
    elementwise_div,
    ops::ElementwiseDivXPUKernel<paddle::platform::XPUDeviceContext, float>);
#endif
