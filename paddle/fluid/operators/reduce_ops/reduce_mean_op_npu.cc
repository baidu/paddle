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
#include "paddle/fluid/operators/reduce_ops/reduce_mean_op.h"
#include "paddle/fluid/framework/fleet/ascend_wrapper.h"
#include "paddle/fluid/framework/generator.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/operators/elementwise/elementwise_npu.h"

#include "paddle/fluid/operators/npu_op_runner.h"

namespace paddle {
namespace operators {

template <typename T>
class NPUReduceMeanOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* input = ctx.Input<Tensor>("X");
    auto* output = ctx.Output<Tensor>("Out");
    output->mutable_data<T>(ctx.GetPlace());

    bool reduce_all = ctx.Attr<bool>("reduce_all");
    auto dims = ctx.Attr<std::vector<int>>("dim");
    bool keep_dim = ctx.Attr<bool>("keep_dim");

    auto input_dims_vec = framework::vectorize(input->dims());
    if (reduce_all) {
      dims.clear();
      for (size_t i = 0; i < input_dims_vec.size(); i++) {
        dims.push_back(static_cast<int>(i));
      }
    }

    const auto& runner = NpuOpRunner("ReduceMeanD", {*input}, {*output},
                                     {{"axes", dims}, {"keep_dims", keep_dim}});
    auto stream =
        ctx.template device_context<paddle::platform::NPUDeviceContext>()
            .stream();
    runner.Run(stream);
  }
};

template <typename T>
class NPUReduceMeanGradOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* input = ctx.Input<Tensor>("X");
    auto* output_grad = ctx.Input<Tensor>(framework::GradVarName("Out"));
    auto* input_grad = ctx.Output<Tensor>(framework::GradVarName("X"));
    input_grad->mutable_data<T>(ctx.GetPlace());

    bool reduce_all = ctx.Attr<bool>("reduce_all");
    auto dims = ctx.Attr<std::vector<int>>("dim");

    auto input_dims_vec = framework::vectorize(input->dims());

    int reduce_numel = 1;
    auto reduce_dims = dims;
    if (reduce_all) {
      reduce_dims.clear();
      for (size_t d = 0; d < input_dims_vec.size(); ++d) {
        reduce_dims.push_back(static_cast<int>(d));
      }
    }
    for (auto d : dims) {
      reduce_numel *= input_dims_vec[d];
    }

    const auto& runner =
        NpuOpRunner("FillV2D", {}, {*input_grad},
                    {{"value", 1.0f / static_cast<float>(reduce_numel)},
                     {"dims", input_dims_vec}});
    auto stream =
        ctx.template device_context<paddle::platform::NPUDeviceContext>()
            .stream();
    runner.Run(stream);

    Tensor transformed_input_grad, transformed_out_grad;
    Tensor tmp_output_grad;
    auto tmp_output_dims_vec = input_dims_vec;
    for (auto d : reduce_dims) {
      tmp_output_dims_vec[d] = 1;
    }
    tmp_output_grad.ShareDataWith(*output_grad);
    tmp_output_grad.Resize(framework::make_ddim(tmp_output_dims_vec));
    auto& dev_ctx =
        ctx.template device_context<paddle::platform::NPUDeviceContext>();
    NpuElementWiseOpBroadcast<T>(dev_ctx, input_grad, &tmp_output_grad, 0,
                                 &transformed_input_grad,
                                 &transformed_out_grad);
    const auto& runner2 =
        NpuOpRunner("Mul", {transformed_input_grad, transformed_out_grad},
                    {*input_grad}, {});
    runner2.Run(stream);
  }
};
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_NPU_KERNEL(reduce_mean, ops::NPUReduceMeanOpKernel<float>);
REGISTER_OP_NPU_KERNEL(reduce_mean_grad, ops::NPUReduceMeanGradOpKernel<float>);
