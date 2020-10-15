// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

#ifdef PADDLE_WITH_XPU

#include "paddle/fluid/operators/reduce_ops/reduce_mean_op.h"
#include <memory>
#include <string>
#include <vector>

namespace paddle {
namespace operators {
template <typename DeviceContext, typename T>
class ReduceMeanXPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    PADDLE_ENFORCE_EQ(
        platform::is_xpu_place(context.GetPlace()), true,
        platform::errors::Unavailable("This kernel only runs on XPU."));
    // bool reduce_all = context.Attr<bool>("reduce_all");
    auto* input = context.Input<Tensor>("X");
    auto* output = context.Output<Tensor>("Out");
    output->mutable_data<T>(context.GetPlace());
    auto& dev_ctx = context.template device_context<DeviceContext>();
    int ndim = input->dims().size();
    std::vector<int> idims;
    for (int i = 0; i < input->dims().size(); i++) {
      idims.push_back(input->dims()[i]);
    }
    auto dims = context.Attr<std::vector<int>>("dim");
    int rdim = dims.size();
    int r =
        xpu::reduce(dev_ctx.x_context(), input->data<T>(), output->data<T>(),
                    idims.data(), ndim, dims.data(), rdim, xpu::REDUCE_MEAN);
    PADDLE_ENFORCE_EQ(r == xpu::Error_t::SUCCESS, true,
                      platform::errors::External("XPU kernel error!"));
  }
};
}  // namespace operators
}  // namespace paddle

REGISTER_OP_XPU_KERNEL(
    reduce_mean,
    ops::ReduceMeanXPUKernel<paddle::platform::XPUDeviceContext, float>);

#endif
