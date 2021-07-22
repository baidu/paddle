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
#include "paddle/fluid/operators/batch_norm_op.h"

#include "paddle/fluid/operators/npu_op_runner.h"

namespace paddle {
namespace operators {

template <typename T>
class NPUBatchNormOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    auto &dev_ctx = ctx.template device_context<platform::NPUDeviceContext>();
    const float epsilon = ctx.Attr<float>("epsilon");
    float momentum = ctx.Attr<float>("momentum");
    const bool is_test = ctx.Attr<bool>("is_test");
    const bool use_global_stats = ctx.Attr<bool>("use_global_stats");
    const bool trainable_stats = ctx.Attr<bool>("trainable_statistics");
    const bool test_mode = is_test && (!trainable_stats);
    const std::string data_layout = ctx.Attr<std::string>("data_layout");

    const auto *x = ctx.Input<Tensor>("X");
    const auto &x_dims = x->dims();
    PADDLE_ENFORCE_EQ(x_dims.size(), 4,
                      platform::errors::InvalidArgument(
                          "The input tensor X's dimension must equal to 4. But "
                          "received X's shape = [%s], X's dimension = [%d].",
                          x_dims, x_dims.size()));

    auto *y = ctx.Output<Tensor>("Y");
    y->mutable_data<T>(ctx.GetPlace());

    const auto *scale = ctx.Input<Tensor>("Scale");
    const auto *bias = ctx.Input<Tensor>("Bias");

    Tensor x_tensor, y_tesnor;
    x_tensor.ShareDataWith(*x);
    y_tesnor.ShareDataWith(*y);
    if (data_layout == "NHWC") {
      x_tensor.set_layout(DataLayout::kNHWC);
      y_tesnor.set_layout(DataLayout::kNHWC);
    }

    bool training = !test_mode && !use_global_stats;
    if (!training) {
      const auto *est_mean = ctx.Input<Tensor>("Mean");
      const auto *est_var = ctx.Input<Tensor>("Variance");

      const auto &runner = NpuOpRunner(
          "BatchNorm", {x_tensor, *scale, *bias, *est_mean, *est_var},
          {y_tesnor, *est_mean, *est_var, *est_mean, *est_var},
          {{"epsilon", epsilon},
           {"is_training", training},
           {"data_format", data_layout}});
      auto stream = dev_ctx.stream();
      runner.Run(stream);
    } else {
      // if MomentumTensor is set, use MomentumTensor value, momentum
      // is only used in this training branch
      if (ctx.HasInput("MomentumTensor")) {
        const auto *mom_tensor = ctx.Input<Tensor>("MomentumTensor");
        Tensor mom_cpu;
        TensorCopySync(*mom_tensor, platform::CPUPlace(), &mom_cpu);
        momentum = mom_cpu.data<float>()[0];
      }

      auto *mean_out = ctx.Output<Tensor>("MeanOut");
      auto *variance_out = ctx.Output<Tensor>("VarianceOut");
      auto *saved_mean = ctx.Output<Tensor>("SavedMean");
      auto *saved_variance = ctx.Output<Tensor>("SavedVariance");
      mean_out->mutable_data<T>(ctx.GetPlace());
      variance_out->mutable_data<T>(ctx.GetPlace());
      saved_mean->mutable_data<T>(ctx.GetPlace());
      saved_variance->mutable_data<T>(ctx.GetPlace());

      framework::Tensor mean_tmp, variance_tmp;
      mean_tmp.mutable_data<float>(mean_out->dims(), ctx.GetPlace());
      variance_tmp.mutable_data<float>(variance_out->dims(), ctx.GetPlace());

      const auto &runner = NpuOpRunner(
          "BatchNorm", {x_tensor, *scale, *bias},
          {y_tesnor, mean_tmp, variance_tmp, *saved_mean, *saved_variance},
          {{"epsilon", epsilon},
           {"is_training", training},
           {"data_format", data_layout}});
      auto stream = dev_ctx.stream();
      runner.Run(stream);
      // Ascend can't output the estimated mean and variance
      framework::Tensor this_factor_tensor;
      this_factor_tensor.mutable_data<float>(framework::make_ddim({1}),
                                             ctx.GetPlace());
      framework::TensorFromVector<float>({static_cast<float>(1. - momentum)},
                                         dev_ctx, &this_factor_tensor);
      framework::Tensor momentum_tensor;
      momentum_tensor.mutable_data<float>(framework::make_ddim({1}),
                                          ctx.GetPlace());
      framework::TensorFromVector<float>({static_cast<float>(momentum)},
                                         dev_ctx, &momentum_tensor);
      framework::Tensor ones_tensor;
      ones_tensor.mutable_data<float>(mean_out->dims(), ctx.GetPlace());
      framework::TensorFromVector<float>(
          std::vector<float>(framework::product(mean_out->dims()), 1.0f),
          dev_ctx, &ones_tensor);

      const auto &runner1 = NpuOpRunner("AddMatMatElements",
                                        {*mean_out, *saved_mean, ones_tensor,
                                         momentum_tensor, this_factor_tensor},
                                        {*mean_out}, {});
      runner1.Run(stream);
      const auto &runner2 = NpuOpRunner(
          "AddMatMatElements", {*variance_out, *saved_variance, ones_tensor,
                                momentum_tensor, this_factor_tensor},
          {*variance_out}, {});
      runner2.Run(stream);
    }
  }
};

template <typename T>
class NPUBatchNormGradOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    auto &dev_ctx = ctx.template device_context<platform::NPUDeviceContext>();
    const float epsilon = ctx.Attr<float>("epsilon");
    const std::string data_layout = ctx.Attr<std::string>("data_layout");
    bool use_global_stats = ctx.Attr<bool>("use_global_stats");

    const auto *y_grad = ctx.Input<Tensor>(framework::GradVarName("Y"));
    const auto *scale = ctx.Input<Tensor>("Scale");
    const auto *bias = ctx.Input<Tensor>("Bias");
    auto *saved_mean = ctx.Input<Tensor>("SavedMean");
    auto *saved_variance = ctx.Input<Tensor>("SavedVariance");

    auto *x_grad = ctx.Output<Tensor>(framework::GradVarName("X"));
    auto *scale_grad = ctx.Output<Tensor>(framework::GradVarName("Scale"));
    auto *bias_grad = ctx.Output<Tensor>(framework::GradVarName("Bias"));

    const bool is_test = ctx.Attr<bool>("is_test");
    use_global_stats = is_test || use_global_stats;

    const Tensor *x = ctx.Input<Tensor>("X");
    const auto &x_dims = x->dims();
    PADDLE_ENFORCE_EQ(x_dims.size(), 4,
                      platform::errors::InvalidArgument(
                          "The input tensor X's dimension must equal to 4. But "
                          "received X's shape = [%s], X's dimension = [%d].",
                          x_dims, x_dims.size()));

    // init output
    Tensor scale_grad_tmp, bias_grad_tmp, x_grad_tmp;
    if (scale_grad && bias_grad) {
      scale_grad->mutable_data<T>(ctx.GetPlace());
      bias_grad->mutable_data<T>(ctx.GetPlace());
      scale_grad_tmp.ShareDataWith(*scale_grad);
      bias_grad_tmp.ShareDataWith(*bias_grad);
    } else {
      scale_grad_tmp.mutable_data<T>(scale->dims(), ctx.GetPlace());
      bias_grad_tmp.mutable_data<T>(bias->dims(), ctx.GetPlace());
    }

    Tensor x_tensor, y_grad_tensor, x_grad_tensor;
    x_tensor.ShareDataWith(*x);
    y_grad_tensor.ShareDataWith(*y_grad);
    if (x_grad) {
      x_grad->mutable_data<T>(ctx.GetPlace());
      x_grad_tensor.ShareDataWith(*x_grad);
    } else {
      x_grad_tensor.mutable_data<T>(x->dims(), ctx.GetPlace());
    }
    if (data_layout == "NHWC") {
      x_tensor.set_layout(DataLayout::kNHWC);
      y_grad_tensor.set_layout(DataLayout::kNHWC);
      x_grad_tensor.set_layout(DataLayout::kNHWC);
    }
    if (!use_global_stats) {
      const auto &runner = NpuOpRunner(
          "BatchNormGrad",
          {y_grad_tensor, x_tensor, *scale, *saved_mean, *saved_variance},
          {x_grad_tensor, scale_grad_tmp, bias_grad_tmp, *saved_mean,
           *saved_variance},  // segment fault if no reserve_space_3 and
                              // reserve_space_4
          {{"epsilon", epsilon},
           {"is_training", true},
           {"data_format", data_layout}});
      auto stream = dev_ctx.stream();
      runner.Run(stream);
    } else {
      const auto *running_mean = ctx.Input<Tensor>("Mean");
      const auto *running_var = ctx.Input<Tensor>("Variance");

      const auto &runner = NpuOpRunner(
          "BatchNormGrad",
          {y_grad_tensor, x_tensor, *scale, *running_mean, *running_var},
          {x_grad_tensor, scale_grad_tmp, bias_grad_tmp, *running_mean,
           *running_var},  // segment fault if no reserve_space_3 and
                           // reserve_space_4
          {{"epsilon", epsilon},
           {"is_training", true},
           {"data_format", data_layout}});
      auto stream = dev_ctx.stream();
      runner.Run(stream);
    }
  }
};
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;

REGISTER_OP_NPU_KERNEL(batch_norm, ops::NPUBatchNormOpKernel<float>,
                       ops::NPUBatchNormOpKernel<plat::float16>);
REGISTER_OP_NPU_KERNEL(batch_norm_grad, ops::NPUBatchNormGradOpKernel<float>,
                       ops::NPUBatchNormGradOpKernel<plat::float16>);
