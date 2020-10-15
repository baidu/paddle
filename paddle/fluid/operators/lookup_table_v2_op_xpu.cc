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

#include "paddle/fluid/operators/lookup_table_v2_op.h"
#include <memory>
#include "paddle/fluid/framework/no_need_buffer_vars_inference.h"
#include "paddle/fluid/framework/op_version_registry.h"
#include "paddle/fluid/framework/var_type_inference.h"

namespace paddle {
namespace operators {

#ifdef PADDLE_WITH_XPU
template <typename DeviceContext, typename T>
class LookupTableV2XPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &context) const override {
    auto *ids_t = context.Input<LoDTensor>("Ids");      // int
    auto *output_t = context.Output<LoDTensor>("Out");  // float
    auto *table_var = context.InputVar("W");
    PADDLE_ENFORCE_EQ(
        (std::is_same<DeviceContext, platform::XPUDeviceContext>::value), true,
        platform::errors::InvalidArgument("Unsupported place!"));

    PADDLE_ENFORCE_EQ(table_var->IsType<LoDTensor>(), true,
                      platform::errors::InvalidArgument(
                          "idx in LookupTableV2XPUKernel should be LoDTensor"));

    int64_t padding_idx = context.Attr<int64_t>("padding_idx");
    int64_t ids_numel = ids_t->numel();

    auto *table_t = context.Input<LoDTensor>("W");
    auto &dev_ctx = context.template device_context<DeviceContext>();
    // size_t N = table_t->dims()[0];
    size_t D = table_t->dims()[1];

    auto *table = table_t->data<T>();
    auto *output = output_t->mutable_data<T>(context.GetPlace());
    const int64_t *ids = ids_t->data<int64_t>();

    PADDLE_ENFORCE_EQ(ids_numel <= std::numeric_limits<int32_t>::max(), true,
                      platform::errors::InvalidArgument(
                          "idx_numel in LookupTableV2XPUKernel should not "
                          "greater than int32_t::max."));
    int ids_numel_int32 = static_cast<int>(ids_numel);
    int r = xpu::embedding<T>(dev_ctx.x_context(), ids_numel_int32, ids, D,
                              table, output, padding_idx);
    PADDLE_ENFORCE_EQ(r == xpu::Error_t::SUCCESS, true,
                      platform::errors::InvalidArgument("XPU kernel error!"));
  }
};

template <typename DeviceContext, typename T>
class LookupTableV2GradXPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &context) const override {
    auto *table_var = context.InputVar("W");
    DDim table_dim;
    PADDLE_ENFORCE_EQ(
        table_var->IsType<LoDTensor>(), true,
        platform::errors::InvalidArgument(
            "idx in LookupTableV2GradXPUKernel should be LoDTensor"));
    table_dim = context.Input<LoDTensor>("W")->dims();

    bool is_sparse = context.Attr<bool>("is_sparse");
    PADDLE_ENFORCE_EQ(
        is_sparse, false,
        platform::errors::InvalidArgument(
            "LookupTableV2GradXPUKernel dose NOT support is_sparse = True"));

    auto ids_t = context.Input<LoDTensor>("Ids");
    auto d_output_t = context.Input<LoDTensor>(framework::GradVarName("Out"));
    auto d_table_t = context.Output<LoDTensor>(framework::GradVarName("W"));

    int64_t ids_numel = ids_t->numel();
    PADDLE_ENFORCE_EQ(ids_numel <= std::numeric_limits<int32_t>::max(), true,
                      platform::errors::InvalidArgument(
                          "idx_numel in LookupTableV2GradXPUKernel should not "
                          "greater than int32_t::max."));
    int ids_numel_int32 = static_cast<int>(ids_numel);
    const int64_t *ids_data = ids_t->data<int64_t>();

    int D = d_table_t->dims()[1];
    const T *d_output_data = d_output_t->data<T>();
    T *d_table_data = d_table_t->mutable_data<T>(context.GetPlace());
    auto &dev_ctx = context.template device_context<DeviceContext>();
    // set zeros for d_table_data
    const int zero = 0;
    int r = xpu::memset(dev_ctx.x_context(), d_table_data, zero,
                        d_table_t->numel() * sizeof(T));
    PADDLE_ENFORCE_EQ(r == xpu::Error_t::SUCCESS, true,
                      platform::errors::InvalidArgument("XPU kernel error!"));

    r = xpu::embedding_backward<T, int64_t>(dev_ctx.x_context(),
                                            ids_numel_int32, ids_data, D,
                                            d_output_data, d_table_data);
    PADDLE_ENFORCE_EQ(r == xpu::Error_t::SUCCESS, true,
                      platform::errors::InvalidArgument("XPU kernel error!"));
  }
};
#endif

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
#ifdef PADDLE_WITH_XPU
REGISTER_OP_XPU_KERNEL(
    lookup_table_v2,
    ops::LookupTableV2XPUKernel<paddle::platform::XPUDeviceContext, float>);
REGISTER_OP_XPU_KERNEL(
    lookup_table_v2_grad,
    ops::LookupTableV2GradXPUKernel<paddle::platform::XPUDeviceContext, float>);
#endif
