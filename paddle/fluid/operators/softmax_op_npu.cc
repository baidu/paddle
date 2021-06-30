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
#include "paddle/fluid/operators/softmax_op.h"

namespace paddle {
namespace operators {

template <typename DeviceContext, typename T>
class SoftmaxNPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* in = ctx.Input<framework::LoDTensor>("X");
    auto axis = ctx.Attr<int>("axis");
    std::vector<int> axes;
    axes.push_back(axis);
    framework::NPUAttributeMap attr_input = {{"axes", axes}};

    auto* out = ctx.Output<framework::LoDTensor>("Out");
    InferNPUStorageFormatAndDims(out, in->npu_storage_layout());
    out->mutable_data<T>(ctx.GetPlace());

    const auto& runner = NpuOpRunner("SoftmaxV2", {*in}, {*out}, attr_input);

    auto stream =
        ctx.template device_context<paddle::platform::NPUDeviceContext>()
            .stream();
    runner.Run(stream);
  }
};

template <typename DeviceContext, typename T>
class SoftmaxGradNPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* out = ctx.Input<framework::LoDTensor>("Out");
    auto* dOut = ctx.Input<framework::LoDTensor>(framework::GradVarName("Out"));

    auto* dX = ctx.Output<Tensor>(framework::GradVarName("X"));

    auto dims = dX->dims();
    const int rank = dims.size();
    const int axis = CanonicalAxis(ctx.Attr<int>("axis"), rank);
    int64_t first_dim = 1;
    int64_t sec_dim = 1;
    for (int i = 0; i < axis; i++) {
      first_dim *= dims[i];
    }
    for (int i = axis; i < rank; i++) {
      sec_dim *= dims[i];
    }

    auto stream =
        ctx.template device_context<paddle::platform::NPUDeviceContext>()
            .stream();

    Tensor tmp_out;
    if (dOut->npu_storage_layout() != DataLayout::kFractalNZ) {
      tmp_out.ShareDataWith(*out).Resize({first_dim, sec_dim});
    } else {
      tmp_out.Resize(out->dims());
      InferNPUStorageFormatAndDims(&tmp_out, DataLayout::kNCHW);
      tmp_out.mutable_data<T>(ctx.GetPlace());
      RunTransDataNPUOP(*out, &tmp_out, stream);
      VLOG(3) << "Transform data_format of softmax grad op.";

      tmp_out.Resize({first_dim, sec_dim});
      tmp_out.ResizeNPUDims({first_dim, sec_dim});
    }

    Tensor tmp_dOut;
    if (dOut->npu_storage_layout() != DataLayout::kFractalNZ) {
      tmp_dOut.ShareDataWith(*dOut).Resize({first_dim, sec_dim});
    } else {
      tmp_dOut.Resize(dOut->dims());
      InferNPUStorageFormatAndDims(&tmp_dOut, DataLayout::kNCHW);
      tmp_dOut.mutable_data<T>(ctx.GetPlace());
      RunTransDataNPUOP(*dOut, &tmp_dOut, stream);
      VLOG(3) << "Transform data_format of softmax grad op.";

      tmp_dOut.Resize({first_dim, sec_dim});
      tmp_dOut.ResizeNPUDims({first_dim, sec_dim});
    }

    dX->Resize(framework::make_ddim({first_dim, sec_dim}));
    dX->mutable_data<T>(ctx.GetPlace());

    framework::NPUAttributeMap attr_input = {};
    const auto& runner = NpuOpRunner(std::string("SoftmaxGrad"),
                                     {tmp_out, tmp_dOut}, {*dX}, attr_input);

    runner.Run(stream);

    dX->Resize(dims);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;

REGISTER_OP_NPU_KERNEL(
    softmax, ops::SoftmaxNPUKernel<plat::NPUDeviceContext, float>,
    ops::SoftmaxNPUKernel<plat::NPUDeviceContext, double>,
    ops::SoftmaxNPUKernel<plat::NPUDeviceContext, plat::float16>);

REGISTER_OP_NPU_KERNEL(
    softmax_grad, ops::SoftmaxGradNPUKernel<plat::NPUDeviceContext, float>,
    ops::SoftmaxGradNPUKernel<plat::NPUDeviceContext, double>,
    ops::SoftmaxGradNPUKernel<plat::NPUDeviceContext,
                              paddle::platform::float16>);
