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

#include "paddle/fluid/operators/scale_op.h"
#include <string>
#include "paddle/fluid/platform/xpu_header.h"

namespace paddle {
namespace operators {
template <typename DeviceContext, typename T>
class ScaleXPUKernel : public framework::OpKernel<T> {
 public:
  virtual void Compute(const framework::ExecutionContext& ctx) const {
    auto* in_var = ctx.InputVar("X");
    auto* in = framework::GetLoDTensorOrSelectedRowsValueFromVar(*in_var);
    auto scale = static_cast<T>(ctx.Attr<float>("scale"));
    auto bias = static_cast<T>(ctx.Attr<float>("bias"));
    auto bias_after_scale = ctx.Attr<bool>("bias_after_scale");
    auto* out_var = ctx.OutputVar("Out");
    if (in_var->IsType<framework::SelectedRows>() && in_var != out_var) {
      auto& in_slr = in_var->Get<framework::SelectedRows>();
      auto* out_slr = out_var->GetMutable<framework::SelectedRows>();
      out_slr->set_rows(in_slr.rows());
      out_slr->set_height(in_slr.height());
    }
    auto* out =
        framework::GetMutableLoDTensorOrSelectedRowsValueFromVar(out_var);
    out->mutable_data<T>(in->place());
    PADDLE_ENFORCE_EQ(
        in->dims(), out->dims(),
        platform::errors::InvalidArgument("In and out should have the same dim,"
                                          " expected %s, but got %s.",
                                          in->dims().to_str().c_str(),
                                          out->dims().to_str().c_str()));
    auto& dev_ctx = ctx.template device_context<DeviceContext>();
    int r = xpu::scale(dev_ctx.x_context(), in->numel(), scale, bias,
                       bias_after_scale, in->data<float>(), out->data<float>());
    PADDLE_ENFORCE_EQ(r, xpu::Error_t::SUCCESS,
                      platform::errors::Fatal("XPU scale kernel error!"));
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_XPU_KERNEL(
    scale, ops::ScaleXPUKernel<paddle::platform::XPUDeviceContext, float>);

#endif
