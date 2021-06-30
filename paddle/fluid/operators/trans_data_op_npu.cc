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

#include "paddle/fluid/operators/npu_op_runner.h"
#include "paddle/fluid/operators/trans_data_op.h"

namespace paddle {
namespace operators {

template <typename DeviceContext, typename T>
class TransDataNPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* x = ctx.Input<framework::Tensor>("X");
    auto* out = ctx.Output<framework::Tensor>("Out");
    int acl_format = ctx.Attr<int>("acl_format");

    PADDLE_ENFORCE_EQ(acl_format, 29,
                      platform::errors::InvalidArgument(
                          "The data_format to be transformed must be "
                          "FRACTAL_NZ (29), but got %d",
                          acl_format));

    auto stream =
        ctx.template device_context<paddle::platform::NPUDeviceContext>()
            .stream();

    if (framework::DataLayoutToString(out->npu_storage_layout()) ==
        "FRACTAL_NZ") {
      VLOG(4) << "Input tensor's data_format is FRACTAL_NZ, don't need to do "
                 "TransData.";
      return;
    }

    Tensor tmp_x(x->type());
    tmp_x.Resize(x->dims());
    tmp_x.mutable_data<T>(ctx.GetPlace());
    framework::TensorCopy(
        *x, ctx.GetPlace(),
        ctx.template device_context<platform::DeviceContext>(), &tmp_x);
    tmp_x.set_npu_storage_layout(x->layout());
    tmp_x.ResizeNPUDims(x->dims());

    out->ResizeNPUDims(framework::make_ddim(
        InferShapeNDToNZ(framework::vectorize(out->dims()))));
    out->set_npu_storage_layout(DataLayout::kFractalNZ);

    auto place = GetCurrentNPUPlace();
    size_t npu_storage_size =
        out->npu_storage_numel() * framework::SizeOfType(x->type());
    out->mutable_data(ctx.GetPlace(), x->type(), npu_storage_size);

    std::string src_format_name =
        framework::DataLayoutToString(tmp_x.npu_storage_layout());
    std::string dst_format_name =
        framework::DataLayoutToString(out->npu_storage_layout());
    const auto& runner_trans_data = NpuOpRunner(
        "TransData", {tmp_x}, {*out}, {{"src_format", src_format_name},
                                       {"dst_format", dst_format_name},
                                       {"groups", 1}});
    runner_trans_data.Run(stream);
    VLOG(4) << "Run TransData OP to cast NPU format from " << src_format_name
            << " to " << dst_format_name;
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OP_NPU_KERNEL(
    trans_data,
    ops::TransDataNPUKernel<paddle::platform::NPUDeviceContext, float>,
    ops::TransDataNPUKernel<paddle::platform::NPUDeviceContext,
                            paddle::platform::float16>);
