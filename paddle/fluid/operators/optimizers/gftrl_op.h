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

#pragma once
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/math/selected_rows_functor.h"
#include "paddle/fluid/platform/for_range.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
using SelectedRows = framework::SelectedRows;

template <typename T>
class UpdateFunctor {
 private:
  const T* g;
  const T* p;
  const T* s_acc;
  const T* l_acc;
  const T* lr_;
  const T l1;
  const T l2;
  const T lr_power;
  const int64_t row_numel;
  T* p_out;
  T* s_acc_out;
  T* l_acc_out;
  const int64_t* rows;

 public:
  UpdateFunctor(const T* g, const T* p, const T* s_acc, const T* l_acc,
                const T* lr, const T l1, const T l2, const T lr_power,
                const int64_t row_numel, T* p_out, T* s_acc_out, T* l_acc_out,
                const int64_t* rows = nullptr)
      : g(g),
        p(p),
        s_acc(s_acc),
        l_acc(l_acc),
        lr_(lr),
        l1(l1),
        l2(l2),
        lr_power(lr_power),
        row_numel(row_numel),
        p_out(p_out),
        s_acc_out(s_acc_out),
        l_acc_out(l_acc_out),
        rows(rows) {}

  inline HOSTDEVICE void operator()(size_t i) {
    auto lr = *lr_;

    auto m_base = i * row_numel;
    auto n_base = rows == nullptr ? m_base : rows[i] * row_numel;

    bool sparsity_flag = true;

    for (int64_t k = 0; k < row_numel; ++k) {
      auto m = m_base + k;
      auto n = n_base + k;

      auto new_acc = s_acc[n] + g[m] * g[m];

      if (lr_power == static_cast<T>(-0.5)) {
        l_acc_out[n] = l_acc[n] + g[m] -
                       (std::sqrt(new_acc) - std::sqrt(s_acc[n])) / lr * p[n];
      } else {
        l_acc_out[n] =
            l_acc[n] + g[m] -
            (std::pow(new_acc, -lr_power) - std::pow(s_acc[n], -lr_power)) /
                lr * p[n];
      }

      s_acc_out[n] = new_acc;

      if (sparsity_flag && std::fabs(l_acc_out[n]) > l1) {
        sparsity_flag = false;
      }
    }

    if (sparsity_flag) {
      for (int64_t k = 0; k < row_numel; ++k) {
        auto n = n_base + k;
        p_out[n] = static_cast<T>(0);
      }
    } else {
      auto x_square_acc = static_cast<T>(0);

      for (int64_t k = 0; k < row_numel; ++k) {
        auto n = n_base + k;

        auto x = (l_acc_out[n] >= static_cast<T>(0)) ? (-l_acc_out[n] + l1)
                                                     : (-l_acc_out[n] - l1);
        x_square_acc += x * x;

        auto y = (lr_power == static_cast<T>(-0.5))
                     ? (l2 + std::sqrt(s_acc_out[n]) / lr)
                     : (l2 + std::pow(s_acc_out[n], -lr_power) / lr);

        p_out[n] = x / (y + static_cast<T>(1e-6));
      }

      auto x_norm = std::sqrt(x_square_acc);
      auto z = static_cast<T>(1) - l1 / (x_norm + static_cast<T>(1e-6));

      for (int64_t k = 0; k < row_numel; ++k) {
        auto n = n_base + k;
        p_out[n] *= z;
      }
    }
  }
};

template <typename DeviceContext, typename T>
class GFTRLOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    const auto* param_var = ctx.InputVar("Param");
    PADDLE_ENFORCE_EQ(param_var->IsType<framework::LoDTensor>(), true,
                      platform::errors::InvalidArgument(
                          "The Var(%s)'s type should be LoDTensor, "
                          "but the received is %s",
                          ctx.InputNames("Param").front(),
                          framework::ToTypeName(param_var->Type())));
    const auto* grad_var = ctx.InputVar("Grad");

    auto* lr_in = ctx.Input<Tensor>("LearningRate");

    auto* param_in = ctx.Input<Tensor>("Param");
    auto* sq_accum_in = ctx.Input<Tensor>("SquaredAccumulator");
    auto* lin_accum_in = ctx.Input<Tensor>("LinearAccumulator");

    auto* param_out = ctx.Output<Tensor>("ParamOut");
    auto* sq_accum_out = ctx.Output<Tensor>("SquaredAccumOut");
    auto* lin_accum_out = ctx.Output<Tensor>("LinearAccumOut");

    auto l1 = static_cast<T>(ctx.Attr<float>("l1"));
    auto l2 = static_cast<T>(ctx.Attr<float>("l2"));
    auto lr_power = static_cast<T>(ctx.Attr<float>("lr_power"));

    if (grad_var->IsType<framework::LoDTensor>()) {
      auto grad = ctx.Input<Tensor>("Grad");

      auto row_height = static_cast<int64_t>(grad->dims()[0]);
      auto row_numel = static_cast<int64_t>(grad->numel() / row_height);

      platform::ForRange<DeviceContext> for_range(
          static_cast<const DeviceContext&>(ctx.device_context()), row_height);

      UpdateFunctor<T> functor(grad->data<T>(), param_in->data<T>(),
                               sq_accum_in->data<T>(), lin_accum_in->data<T>(),
                               lr_in->data<T>(), l1, l2, lr_power, row_numel,
                               param_out->mutable_data<T>(ctx.GetPlace()),
                               sq_accum_out->mutable_data<T>(ctx.GetPlace()),
                               lin_accum_out->mutable_data<T>(ctx.GetPlace()));
      for_range(functor);
    } else if (grad_var->IsType<SelectedRows>()) {
      auto grad = ctx.Input<SelectedRows>("Grad");

      SelectedRows tmp_merged_grad;
      SelectedRows* merged_grad = &tmp_merged_grad;
      math::scatter::MergeAdd<DeviceContext, T> merge_func;
      merge_func(ctx.template device_context<DeviceContext>(), *grad,
                 merged_grad);

      const auto* rows = merged_grad->rows().Data(ctx.GetPlace());
      auto row_height = static_cast<int64_t>(merged_grad->value().dims()[0]);
      auto row_numel =
          static_cast<int64_t>(merged_grad->value().numel() / row_height);

      platform::ForRange<DeviceContext> for_range(
          static_cast<const DeviceContext&>(ctx.device_context()), row_height);

      UpdateFunctor<T> functor(
          merged_grad->value().data<T>(), param_in->data<T>(),
          sq_accum_in->data<T>(), lin_accum_in->data<T>(), lr_in->data<T>(), l1,
          l2, lr_power, row_numel, param_out->mutable_data<T>(ctx.GetPlace()),
          sq_accum_out->mutable_data<T>(ctx.GetPlace()),
          lin_accum_out->mutable_data<T>(ctx.GetPlace()), rows);
      for_range(functor);
    } else {
      PADDLE_THROW(platform::errors::InvalidArgument(
          "Unsupported Variable Type of Grad"));
    }
  }
};

}  // namespace operators
}  // namespace paddle
