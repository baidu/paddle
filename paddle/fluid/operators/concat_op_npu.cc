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

#include "paddle/fluid/operators/concat_op.h"
#include "paddle/fluid/operators/npu_op_runner.h"

namespace paddle {
namespace operators {

template <typename T>
class ConcatNPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto ins = ctx.MultiInput<framework::LoDTensor>("X");
    framework::LoDTensor* out = ctx.Output<framework::LoDTensor>("Out");
    PADDLE_ENFORCE_NOT_NULL(ins[0],
                            platform::errors::NotFound(
                                "The first input tensor is not initalized."));
    auto axis = ctx.Attr<int>("axis");

    if (ctx.HasInput("AxisTensor")) {
      PADDLE_THROW(platform::errors::NotFound(
          "The AxisTensor is not supported on NPU now."));
    }
    axis = ComputeAxis(static_cast<int64_t>(axis),
                       static_cast<int64_t>(ins[0]->dims().size()));

    auto place = ctx.GetPlace();
    out->mutable_data<T>(place);

    std::vector<framework::Tensor> inputs;
    std::vector<std::string> names;
    for (size_t i = 0; i < ins.size(); ++i) {
      if (ins[i] && ins[i]->numel() > 0) {
        inputs.push_back(*ins[i]);
        names.push_back("x" + std::to_string(i));
      } else {
        continue;
      }
    }
    auto stream =
        ctx.template device_context<paddle::platform::NPUDeviceContext>()
            .stream();
    auto runner = NpuOpRunner(
        "ConcatD", {inputs}, {*out},
        {{"concat_dim", axis}, {"N", static_cast<int>(inputs.size())}});
    runner.AddInputNames(names);
    runner.Run(stream);
  }
};

template <typename T>
class ConcatGradNPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* out_grad =
        ctx.Input<framework::Tensor>(framework::GradVarName("Out"));
    auto ins = ctx.MultiInput<framework::LoDTensor>("X");
    auto out_var_names = ctx.OutputNames(framework::GradVarName("X"));
    auto outs =
        ctx.MultiOutput<framework::LoDTensor>(framework::GradVarName("X"));

    PADDLE_ENFORCE_NOT_NULL(ins[0],
                            platform::errors::NotFound(
                                "The first input tensor is not initalized."));

    auto axis = ctx.Attr<int>("axis");

    axis = ComputeAxis(static_cast<int64_t>(axis),
                       static_cast<int64_t>(ins[0]->dims().size()));

    int offset = 0;
    auto stream =
        ctx.template device_context<paddle::platform::NPUDeviceContext>()
            .stream();
    for (size_t j = 0; j < outs.size(); ++j) {
      // For stop gradient
      // get output tensor that the name is not kEmptyVarName
      if (out_var_names[j] != framework::kEmptyVarName &&
          outs[j]->numel() != 0UL) {
        outs[j]->mutable_data<T>(ctx.GetPlace());
        std::vector<int> offsets;
        std::vector<int> sizes;
        for (int dim = 0; dim < ins[j]->dims().size(); ++dim) {
          if (dim == axis) {
            offsets.push_back(offset);
            sizes.push_back(ins[j]->dims()[dim]);
          } else {
            offsets.push_back(0);
            sizes.push_back(ins[j]->dims()[dim]);
          }
        }
        auto runner = NpuOpRunner("SliceD", {*out_grad}, {*outs[j]},
                                  {{"offsets", offsets}, {"size", sizes}});
        runner.Run(stream);
      }
      if (ins[j]->numel() != 0UL) {
        offset += ins[j]->dims()[axis];
      }
    }
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OP_NPU_KERNEL(concat, ops::ConcatNPUKernel<float>,
                       ops::ConcatNPUKernel<paddle::platform::float16>,
                       ops::ConcatNPUKernel<int>);

REGISTER_OP_NPU_KERNEL(concat_grad, ops::ConcatGradNPUKernel<float>,
                       ops::ConcatGradNPUKernel<paddle::platform::float16>,
                       ops::ConcatGradNPUKernel<int>);
