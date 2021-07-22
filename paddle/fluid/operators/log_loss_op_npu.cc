/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the Licnse. */

#include "paddle/fluid/operators/log_loss_op.h"
#include "paddle/fluid/operators/npu_op_runner.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;

template <typename T>
void AddsFun(const platform::Place& place, const aclrtStream& stream,
             const Tensor* x, float scale, Tensor* y) {
  //  Calculate y = x + scale
  y->mutable_data<T>(x->dims(), place);
  const auto& runner = NpuOpRunner("Adds", {*x}, {*y}, {{"value", scale}});
  runner.Run(stream);
}

template <typename T>
void MulsFun(const platform::Place& place, const aclrtStream& stream,
             const Tensor* x, float scale, Tensor* y) {
  //  Calculate y = x + scale
  y->mutable_data<T>(x->dims(), place);
  const auto& runner = NpuOpRunner("Muls", {*x}, {*y}, {{"value", scale}});
  runner.Run(stream);
}

template <typename T>
void MulFun(const platform::Place& place, const aclrtStream& stream,
            const Tensor* x, const Tensor* y, Tensor* z) {
  //  Calculate z = x * y
  z->mutable_data<T>(x->dims(), place);
  const auto& runner = NpuOpRunner("Mul", {*x, *y}, {*z}, {});
  runner.Run(stream);
}

template <typename T>
void DivFun(const platform::Place& place, const aclrtStream& stream,
            const Tensor* x, const Tensor* y, Tensor* z) {
  //  Calculate z = x / y
  z->mutable_data<T>(x->dims(), place);
  const auto& runner = NpuOpRunner("Div", {*x, *y}, {*z}, {});
  runner.Run(stream);
}

template <typename T>
void AddFun(const platform::Place& place, const aclrtStream& stream,
            const Tensor* x, const Tensor* y, Tensor* z) {
  //  Calculate z = x + y
  z->mutable_data<T>(x->dims(), place);
  const auto& runner = NpuOpRunner("Add", {*x, *y}, {*z}, {});
  runner.Run(stream);
}

template <typename T>
void SubFun(const platform::Place& place, const aclrtStream& stream,
            const Tensor* x, const Tensor* y, Tensor* z) {
  //  Calculate z = x - y
  z->mutable_data<T>(x->dims(), place);
  const auto& runner = NpuOpRunner("Sub", {*x, *y}, {*z}, {});
  runner.Run(stream);
}

template <typename T>
void LogFun(const platform::Place& place, const aclrtStream& stream,
            const Tensor* x, float scale, float shift, Tensor* y) {
  //  Calculate y = log ( scale * x + shift )
  //  Try to use Log API directly but failed, it seems that something is wrong
  //  with this API
  y->mutable_data<T>(x->dims(), place);
  Tensor t_x_scale;
  MulsFun<T>(place, stream, x, scale, &t_x_scale);
  Tensor t_add;
  AddsFun<T>(place, stream, &t_x_scale, shift - 1, &t_add);
  const auto& runner = NpuOpRunner("Log1p", {t_add}, {*y}, {});
  runner.Run(stream);
}

template <typename T, typename AttrType = T>
class LogLossNPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    //  y = - label * log (pred+epsilon) - (1-label) * log (1-pred+epsilon)
    //    = - label * log (pred+epsilon) + (label-1) * log (1-pred+epsilon)
    auto* y = ctx.Output<Tensor>("Loss");
    auto* pred = ctx.Input<Tensor>("Predicted");
    auto* label = ctx.Input<Tensor>("Labels");
    auto epsilon = static_cast<T>(ctx.Attr<AttrType>("epsilon"));

    auto place = ctx.GetPlace();
    auto stream =
        ctx.template device_context<paddle::platform::NPUDeviceContext>()
            .stream();

    //  t_log_0 = log ( input + epsilon )
    Tensor t_log_0;
    LogFun<T>(place, stream, pred, 1, epsilon, &t_log_0);
    //  t_log_1 = log ( 1 - input + epsilon )
    Tensor t_log_1;
    LogFun<T>(place, stream, pred, -1, epsilon + 1, &t_log_1);
    //  t_mul_0 = label * t_log_0
    Tensor t_mul_0;
    MulFun<T>(place, stream, label, &t_log_0, &t_mul_0);
    //  t_label_m1 = label - 1
    Tensor t_label_m1;
    AddsFun<T>(place, stream, label, -1, &t_label_m1);
    //  t_mul_1 = t_label_m1 * t_log_1
    Tensor t_mul_1;
    MulFun<T>(place, stream, &t_label_m1, &t_log_1, &t_mul_1);
    //  loss_out = t_mul_1 - t_mul_0
    SubFun<T>(place, stream, &t_mul_1, &t_mul_0, y);
  }
};

template <typename T, typename AttrType = T>
class LogLossGradNPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    //  dpred = dloss * ( -label/(pred+epsilon) + (1-label)/(1-pred+epsilon) )
    //        = dloss * ( (label-1)/(pred-1-epsilon) - label/(pred+epsilon) )
    auto* pred = ctx.Input<Tensor>("Predicted");
    auto* label = ctx.Input<Tensor>("Labels");
    auto* dloss = ctx.Input<Tensor>(framework::GradVarName("Loss"));
    auto* dpred = ctx.Output<Tensor>(framework::GradVarName("Predicted"));
    auto epsilon = static_cast<T>(ctx.Attr<AttrType>("epsilon"));

    auto place = ctx.GetPlace();
    auto stream =
        ctx.template device_context<paddle::platform::NPUDeviceContext>()
            .stream();

    if (dpred) {
      //  t_label_m1 = label - 1
      Tensor t_label_m1;
      AddsFun<T>(place, stream, label, -1, &t_label_m1);
      //  t_pred_m = pred - 1 - epsilon
      Tensor t_pred_m;
      AddsFun<T>(place, stream, pred, -1 - epsilon, &t_pred_m);
      //  t_div_0 = t_label_m1 / t_pred_m
      Tensor t_div_0;
      DivFun<T>(place, stream, &t_label_m1, &t_pred_m, &t_div_0);
      //  t_pred_p = pred + epsilon
      Tensor t_pred_p;
      AddsFun<T>(place, stream, pred, epsilon, &t_pred_p);
      //  t_div_1 = label / t_pred_p
      Tensor t_div_1;
      DivFun<T>(place, stream, label, &t_pred_p, &t_div_1);
      //  t_sub = t_div_0 - t_div_1
      Tensor t_sub;
      SubFun<T>(place, stream, &t_div_0, &t_div_1, &t_sub);
      //  dpred = dloss * t_sub
      MulFun<T>(place, stream, dloss, &t_sub, dpred);
    }
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OP_NPU_KERNEL(log_loss, ops::LogLossNPUKernel<float>);

REGISTER_OP_NPU_KERNEL(log_loss_grad, ops::LogLossGradNPUKernel<float>);
