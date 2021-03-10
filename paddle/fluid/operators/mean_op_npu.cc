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
//#include "cub/cub.cuh"
#include "paddle/fluid/operators/mean_op.h"
#include "paddle/fluid/platform/float16.h"
#include "paddle/fluid/operators/npu_op_runner.h"


namespace paddle {
namespace operators {

template <typename DeviceContext, typename T>
class MeanNPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* x = ctx.Input<framework::LoDTensor>("X");
    auto* out = ctx.Output<framework::LoDTensor>("Out");

    auto reduce_ndim = x->dims().size();
    std::vector<int> axes;
    for (auto i = 0; i < reduce_ndim; ++i) {
      axes.push_back(i);
      // VLgrad(3) << " axes " << i ;
    }

    // framework::AttributeMap attr_input = {{"keep_dims", false}, {"axes", axes}};
    framework::NPUAttributeMap attr_input = {{"keep_dims", false}, {"axes", axes}};

    std::vector<int64_t> out_dims;
    out_dims.push_back(1);
    out->Resize(framework::make_ddim(out_dims));
    out->mutable_data<T>(ctx.GetPlace());

    Tensor reduced_out(x->type());
    std::vector<int64_t> reduced_dout_dims;
    reduced_dout_dims.push_back(1);
    reduced_out.Resize(framework::make_ddim(reduced_dout_dims));
    reduced_out.mutable_data<T>(ctx.GetPlace());

    auto runner = NpuOpRunner("ReduceMeanD", {*x}, {*out}, attr_input);
    auto stream =
      ctx.template device_context<paddle::platform::NPUDeviceContext>()
                .stream();
    runner.Run(stream);
    /*
    framework::Tensor cpu_tensor2;
    TensorCopySync(*out, platform::CPUPlace(), &cpu_tensor2);
    framework::Tensor cpu_tensor2;
    TensorCopySync(reduced_out, platform::CPUPlace(), &cpu_tensor2);
    data = cpu_tensor2.data<T>();
    framework::TensorCopySync(*tmp_dout, ctx.GetPlace(), dx);
    */

  }
};


template <typename DeviceContext, typename T>
class MeanGradNPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto stream =
      context.template device_context<paddle::platform::NPUDeviceContext>()
                .stream();

    auto grad = context.Input<Tensor>(framework::GradVarName("Out"));

    PADDLE_ENFORCE_EQ(grad->numel(), 1,
                      platform::errors::InvalidArgument(
                          "Mean Gradient Input Tensor len should be 1. But "
                          "received Out@Grad's elements num is %d.",
                          grad->numel()));

    auto IG = context.Output<Tensor>(framework::GradVarName("X"));
    IG->mutable_data<T>(context.GetPlace());

    // ones 
    Tensor ones(grad->type());
    std::vector<int64_t> dout_dims;
    for (auto i = 0; i < IG->dims().size(); ++i) {
      dout_dims.push_back(IG->dims()[i]);
    }
    ones.Resize(framework::make_ddim(dout_dims));
    ones.mutable_data<T>(context.GetPlace());
    auto runner_ones = NpuOpRunner("OnesLike", {*IG}, {ones}, {});
    runner_ones.Run(stream);

    // means
    Tensor mean_tensor(grad->type());
    mean_tensor.Resize({1});
    mean_tensor.mutable_data<T>(context.GetPlace());
    std::vector<float> mean_vec;
    mean_vec.push_back(1.0/static_cast<float>(IG->numel()));
    framework::TensorFromVector(mean_vec, context.device_context(), &mean_tensor);

    // means mul ones
    Tensor mean_ma(grad->type());
    mean_ma.Resize(framework::make_ddim(dout_dims));
    mean_ma.mutable_data<T>(context.GetPlace());
    auto runner_mul_1 = NpuOpRunner("Mul", {mean_tensor, ones}, {mean_ma}, {});
    runner_mul_1.Run(stream);

    // and mul grad
    auto runner_mul_2 = NpuOpRunner("Mul", {mean_ma, *grad}, {*IG}, {});
    runner_mul_2.Run(stream);

  }
};


}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;
REGISTER_OP_NPU_KERNEL(
    mean, 
    ops::MeanNPUKernel<paddle::platform::NPUDeviceContext, int>,
    ops::MeanNPUKernel<paddle::platform::NPUDeviceContext, float>,
    ops::MeanNPUKernel<paddle::platform::NPUDeviceContext, double>)


REGISTER_OP_NPU_KERNEL(
    mean_grad, 
    ops::MeanGradNPUKernel<paddle::platform::NPUDeviceContext, int>,
    ops::MeanGradNPUKernel<paddle::platform::NPUDeviceContext, float>,
    ops::MeanGradNPUKernel<paddle::platform::NPUDeviceContext, double>)

