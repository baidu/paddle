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

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/framework/program_desc.h"
#include "paddle/fluid/framework/tensor_util.h"
#include "paddle/fluid/operators/dropout_op.h"
#include "paddle/fluid/operators/math/math_function.h"
#include "paddle/fluid/operators/npu_op_runner.h"
#include "paddle/fluid/operators/range_op.h"
#include "paddle/fluid/operators/utils.h"

namespace paddle {
namespace operators {

template <typename DeviceContext, typename T>
class RangeNPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* start_t = context.Input<framework::Tensor>("Start");
    auto* end_t = context.Input<framework::Tensor>("End");
    auto* step_t = context.Input<framework::Tensor>("Step");
    auto* out = context.Output<framework::Tensor>("Out");

    framework::Tensor n;
    framework::TensorCopySync(*start_t, platform::CPUPlace(), &n);
    T start = n.data<T>()[0];
    framework::TensorCopySync(*end_t, platform::CPUPlace(), &n);
    T end = n.data<T>()[0];
    framework::TensorCopySync(*step_t, platform::CPUPlace(), &n);
    T step = n.data<T>()[0];

    int64_t size = 0;
    GetSize(start, end, step, &size);

    out->Resize(framework::make_ddim({size}));
    out->mutable_data<T>(context.GetPlace());

    std::vector<T> odata;
    T value = start;
    for (int64_t i = 0; i < size; ++i) {
      odata.push_back(value);
      value += step;
    }

    framework::TensorFromVector(odata, context.device_context(), out);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OP_NPU_KERNEL(
    range, ops::RangeNPUKernel<paddle::platform::NPUDeviceContext, int>,
    ops::RangeNPUKernel<paddle::platform::NPUDeviceContext, float>,
    ops::RangeNPUKernel<paddle::platform::NPUDeviceContext, double>)

#endif
