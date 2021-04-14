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

#ifdef PADDLE_WITH_ASCEND_CL
#include <memory>
#include <string>

#include "paddle/fluid/operators/kron_op.h"
#include "paddle/fluid/operators/npu_op_runner.h"
#include "paddle/fluid/operators/scatter_op.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;

template <typename DeviceContext, typename T>
class ScatterNPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* x = ctx.Input<Tensor>("X");
    auto* index = ctx.Input<Tensor>("Ids");
    auto* updates = ctx.Input<Tensor>("Updates");
    bool overwrite = ctx.Attr<bool>("overwrite");

    auto* out = ctx.Output<Tensor>("Out");

    auto place = ctx.GetPlace();
    out->mutable_data<T>(place);

    framework::Tensor tmp_tensor(index->type());
    const auto index_dims = index->dims();
    if (index_dims.size() == 1) {
      tmp_tensor.ShareDataWith(*index);
      std::vector<int64_t> new_dim = {index_dims[0], 1};
      tmp_tensor.Resize(framework::make_ddim(new_dim));
      index = &tmp_tensor;
    }

    auto stream =
        ctx.template device_context<paddle::platform::NPUDeviceContext>()
            .stream();

    if (overwrite) {
      auto runner_update = NpuOpRunner("TensorScatterUpdate",
                                       {*x, *index, *updates}, {*out}, {});
      runner_update.Run(stream);
    } else {
      auto runner_add =
          NpuOpRunner("TensorScatterAdd", {*x, *index, *updates}, {*out}, {});
      runner_add.Run(stream);
    }
  }
};
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OP_NPU_KERNEL(
    scatter, ops::ScatterNPUKernel<paddle::platform::NPUDeviceContext, float>,
    ops::ScatterNPUKernel<paddle::platform::NPUDeviceContext,
                          paddle::platform::float16>);
#endif
