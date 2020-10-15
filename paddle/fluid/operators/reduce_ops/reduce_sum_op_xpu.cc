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

#ifdef PADDLE_WITH_XPU
#include "paddle/fluid/operators/reduce_ops/reduce_sum_op.h"
#include <memory>
#include <string>
namespace paddle {
namespace operators {

template <typename DeviceContext, typename T>
class ReduceSumXPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    PADDLE_ENFORCE_EQ(
        platform::is_xpu_place(context.GetPlace()), true,
        platform::errors::Unavailable("This kernel only runs on XPU."));
    bool reduce_all = context.Attr<bool>("reduce_all");
    auto* input = context.Input<Tensor>("X");
    auto* output = context.Output<Tensor>("Out");
    output->mutable_data<T>(context.GetPlace());
    auto& dev_ctx = context.template device_context<DeviceContext>();
    if (reduce_all) {
      int input_len = input->numel();
      int r = xpu::sum(dev_ctx.x_context(), input->data<T>(), output->data<T>(),
                       input_len);
      PADDLE_ENFORCE_EQ(r == xpu::Error_t::SUCCESS, true,
                        platform::errors::External("XPU kernel error!"));
    } else {
      int ndim = input->dims().size();
      std::vector<int> idims;
      for (int i = 0; i < input->dims().size(); i++) {
        idims.push_back(input->dims()[i]);
      }
      auto dims = context.Attr<std::vector<int>>("dim");
      int rdim = dims.size();
      int r =
          xpu::reduce(dev_ctx.x_context(), input->data<T>(), output->data<T>(),
                      idims.data(), ndim, dims.data(), rdim, xpu::REDUCE_SUM);
      PADDLE_ENFORCE_EQ(r == xpu::Error_t::SUCCESS, true,
                        platform::errors::External("XPU kernel error!"));
    }
  }
};
template <typename DeviceContext, typename T>
class ReduceSumGradXPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto dims = context.Attr<std::vector<int>>("dim");
    bool reduce_all = context.Attr<bool>("reduce_all");
    auto* input0 = context.Input<Tensor>("X");
    auto* input2 = context.Input<Tensor>(framework::GradVarName("Out"));
    auto* output = context.Output<Tensor>(framework::GradVarName("X"));
    output->mutable_data<T>(context.GetPlace());
    const auto* input2_d = input2->data<T>();
    auto* output_d = output->data<T>();
    auto& dev_ctx = context.template device_context<DeviceContext>();
    int r = 0;
    std::vector<int> idims;
    int reduce_dim = 0;
    if (reduce_all) {
      idims.push_back(input0->numel());
      idims.push_back(1);
      idims.push_back(1);
      r = xpu::reduce_grad(dev_ctx.x_context(), input2_d, output_d,
                           idims.data(), idims.size(), &reduce_dim, 1,
                           xpu::REDUCE_SUM);
      PADDLE_ENFORCE_EQ(r == xpu::Error_t::SUCCESS, true,
                        platform::errors::External("XPU kernel error!"));
    } else if (dims.size() == 1) {
      // handle reduce by one dimension
      int reduce_dim_index = dims[0];
      if (reduce_dim_index < 0) {
        reduce_dim_index += input0->dims().size();
      }
      auto& input_dim = input0->dims();
      int before_dim = 1;
      for (int i = 0; i < reduce_dim_index; ++i) {
        before_dim *= input_dim[i];
      }
      int reduce_dim = input_dim[reduce_dim_index];
      int after_dim = 1;
      for (int i = reduce_dim_index + 1; i < input_dim.size(); ++i) {
        after_dim *= input_dim[i];
      }
      idims.push_back(before_dim);
      idims.push_back(input_dim[reduce_dim_index]);
      idims.push_back(after_dim);
      reduce_dim = 1;
      r = xpu::reduce_grad(dev_ctx.x_context(), input2_d, output_d,
                           idims.data(), idims.size(), &reduce_dim, 1,
                           xpu::REDUCE_SUM);
      PADDLE_ENFORCE_EQ(r == xpu::Error_t::SUCCESS, true,
                        platform::errors::External("XPU kernel error!"));
    } else {
      PADDLE_THROW(
          platform::errors::Unimplemented("unsupport reduce sum grad"));
    }
  }
};

}  // namespace operators
}  // namespace paddle

REGISTER_OP_XPU_KERNEL(
    reduce_sum,
    ops::ReduceSumXPUKernel<paddle::platform::XPUDeviceContext, float>);
REGISTER_OP_XPU_KERNEL(
    reduce_sum_grad,
    ops::ReduceSumGradXPUKernel<paddle::platform::XPUDeviceContext, float>);

#endif
