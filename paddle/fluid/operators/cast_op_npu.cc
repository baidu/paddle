/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <memory>
#include <string>

#include "paddle/fluid/operators/cast_op.h"
#include "paddle/fluid/operators/npu_op_runner.h"

namespace paddle {
namespace operators {

static std::map<framework::proto::VarType::Type, aclDataType>
    DTYPE_2_ACL_DTYPE = {
        {framework::proto::VarType::BOOL, ACL_BOOL},
        {framework::proto::VarType::INT16, ACL_INT16},
        {framework::proto::VarType::INT32, ACL_INT32},
        {framework::proto::VarType::INT64, ACL_INT64},
        {framework::proto::VarType::FP16, ACL_FLOAT16},
        {framework::proto::VarType::FP32, ACL_FLOAT},
        {framework::proto::VarType::FP64, ACL_DOUBLE},
};

using Tensor = framework::Tensor;

template <typename DeviceContext, typename T>
class CastNPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* x = ctx.Input<Tensor>("X");
    int dtype = ctx.Attr<int>("out_dtype");
    auto* out = ctx.Output<Tensor>("Out");
    auto place = ctx.GetPlace();

    InferNPUStorageFormatAndDims(out, x->npu_storage_layout());

    if (x->type() == dtype) {
      // NOTE(zhiqiu): NPU cast op may result in wrong value, so
      // add special case here.
      VLOG(4) << "cast to same dtype:" << dtype;
      out->mutable_data(place, x->type());
      framework::TensorCopy(
          *x, ctx.GetPlace(),
          ctx.template device_context<platform::DeviceContext>(), out);
      return;
    }

    auto iter = DTYPE_2_ACL_DTYPE.find(
        static_cast<framework::proto::VarType::Type>(dtype));
    int aclDtype = iter->second;

    if (dtype == framework::proto::VarType::FP32) {
      out->mutable_data<float>(place);
    } else if (dtype == framework::proto::VarType::FP16) {
      out->mutable_data<paddle::platform::float16>(place);
    } else if (dtype == framework::proto::VarType::INT16) {
      out->mutable_data<int16_t>(place);
    } else if (dtype == framework::proto::VarType::INT32) {
      out->mutable_data<int32_t>(place);
    } else if (dtype == framework::proto::VarType::INT64) {
      out->mutable_data<int64_t>(place);
    } else if (dtype == framework::proto::VarType::FP64) {
      out->mutable_data<double>(place);
    } else if (dtype == framework::proto::VarType::BOOL) {
      out->mutable_data<bool>(place);
    }

    auto stream =
        ctx.template device_context<paddle::platform::NPUDeviceContext>()
            .stream();

    const auto& runner = NpuOpRunner(
        "Cast", {*x}, {*out}, {{"dst_type", static_cast<int32_t>(aclDtype)}});
    runner.Run(stream);
  }
};
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OP_NPU_KERNEL(
    cast, ops::CastNPUKernel<paddle::platform::NPUDeviceContext, int16_t>,
    ops::CastNPUKernel<paddle::platform::NPUDeviceContext, int32_t>,
    ops::CastNPUKernel<paddle::platform::NPUDeviceContext, int64_t>,
    ops::CastNPUKernel<paddle::platform::NPUDeviceContext, int>,
    ops::CastNPUKernel<paddle::platform::NPUDeviceContext, bool>,
    ops::CastNPUKernel<paddle::platform::NPUDeviceContext, double>,
    ops::CastNPUKernel<paddle::platform::NPUDeviceContext, float>,
    ops::CastNPUKernel<paddle::platform::NPUDeviceContext,
                       paddle::platform::float16>);
