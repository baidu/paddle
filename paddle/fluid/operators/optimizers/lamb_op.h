/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.

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
#include <math.h>  // for sqrt in CPU and CUDA
#include <Eigen/Dense>
#include <vector>
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/amp/fp16_type_traits.h"
#include "paddle/fluid/operators/math/algorithm.h"
#include "paddle/fluid/operators/math/selected_rows_functor.h"
#include "paddle/fluid/platform/for_range.h"

namespace paddle {
namespace operators {

namespace scatter = paddle::operators::math::scatter;

template <typename T, typename MT>
struct LambMomentREGUpdateFunctor {
  MT weight_decay_;
  MT beta1_;
  MT beta2_;
  MT epsilon_;

  MT beta1_pow_;
  MT beta2_pow_;
  const MT* moment1_;
  MT* moment1_out_;
  const MT* moment2_;
  MT* moment2_out_;
  const T* grad_;
  const T* param_;
  MT* trust_ratio_div_;
  const MT* master_param_;

  LambMomentREGUpdateFunctor(MT weight_decay, MT beta1, MT beta2, MT epsilon,
                             MT beta1_pow, MT beta2_pow, const MT* mom1,
                             MT* mom1_out, const MT* mom2, MT* mom2_out,
                             const T* grad, const T* param, MT* trust_ratio_div,
                             const MT* master_param)
      : weight_decay_(weight_decay),
        beta1_(beta1),
        beta2_(beta2),
        epsilon_(epsilon),
        beta1_pow_(beta1_pow),
        beta2_pow_(beta2_pow),
        moment1_(mom1),
        moment1_out_(mom1_out),
        moment2_(mom2),
        moment2_out_(mom2_out),
        grad_(grad),
        param_(param),
        trust_ratio_div_(trust_ratio_div),
        master_param_(master_param) {}

  inline HOSTDEVICE void operator()(size_t i) const {
    MT g = static_cast<MT>(grad_[i]);
    MT mom1 = moment1_[i];
    MT mom2 = moment2_[i];
    MT beta1_pow = beta1_pow_;
    MT beta2_pow = beta2_pow_;
    MT p = master_param_ ? master_param_[i] : static_cast<MT>(param_[i]);

    mom1 = beta1_ * mom1 + (static_cast<MT>(1) - beta1_) * g;
    mom2 = beta2_ * mom2 + (static_cast<MT>(1) - beta2_) * g * g;

    moment1_out_[i] = mom1;
    moment2_out_[i] = mom2;

    MT mom1_unbiased = mom1 / (static_cast<MT>(1) - beta1_pow);
    MT mom2_unbiased = mom2 / (static_cast<MT>(1) - beta2_pow);
    trust_ratio_div_[i] =
        mom1_unbiased / (sqrt(mom2_unbiased) + epsilon_) + weight_decay_ * p;
  }
};

template <typename T, typename MT>
struct LambMomentMENUpdateFunctor {
  MT weight_decay_;
  MT beta1_;
  MT beta2_;
  MT epsilon_;

  const MT* beta1_pow_;
  MT* beta1_pow_out_;
  const MT* beta2_pow_;
  MT* beta2_pow_out_;
  const MT* moment1_;
  MT* moment1_out_;
  const MT* moment2_;
  MT* moment2_out_;
  const T* grad_;
  const T* param_;
  MT* trust_ratio_div_;
  const MT* master_param_;

  LambMomentMENUpdateFunctor(MT weight_decay, MT beta1, MT beta2, MT epsilon,
                             const MT* beta1_pow, MT* beta1_pow_out,
                             const MT* beta2_pow, MT* beta2_pow_out,
                             const MT* mom1, MT* mom1_out, const MT* mom2,
                             MT* mom2_out, const T* grad, const T* param,
                             MT* trust_ratio_div, const MT* master_param)
      : weight_decay_(weight_decay),
        beta1_(beta1),
        beta2_(beta2),
        epsilon_(epsilon),
        beta1_pow_(beta1_pow),
        beta1_pow_out_(beta1_pow_out),
        beta2_pow_(beta2_pow),
        beta2_pow_out_(beta2_pow_out),
        moment1_(mom1),
        moment1_out_(mom1_out),
        moment2_(mom2),
        moment2_out_(mom2_out),
        grad_(grad),
        param_(param),
        trust_ratio_div_(trust_ratio_div),
        master_param_(master_param) {}

  inline HOSTDEVICE void operator()(size_t i) const {
    MT g = static_cast<MT>(grad_[i]);
    MT mom1 = moment1_[i];
    MT mom2 = moment2_[i];
    MT beta1_pow = *beta1_pow_;
    MT beta2_pow = *beta2_pow_;
    MT p = master_param_ ? master_param_[i] : static_cast<MT>(param_[i]);

    mom1 = beta1_ * mom1 + (static_cast<MT>(1) - beta1_) * g;
    mom2 = beta2_ * mom2 + (static_cast<MT>(1) - beta2_) * g * g;

    moment1_out_[i] = mom1;
    moment2_out_[i] = mom2;

    MT mom1_unbiased = mom1 / (static_cast<MT>(1) - beta1_pow);
    MT mom2_unbiased = mom2 / (static_cast<MT>(1) - beta2_pow);
    trust_ratio_div_[i] =
        mom1_unbiased / (sqrt(mom2_unbiased) + epsilon_) + weight_decay_ * p;
    if (beta1_pow_out_ && beta2_pow_out_) {
      beta1_pow_out_[0] = beta1_pow * beta1_;
      beta2_pow_out_[0] = beta2_pow * beta2_;
    }
  }
};

template <typename T, typename MT>
struct SparseLambMomentREGUpdateFunctor {
  MT weight_decay_;
  MT beta1_;
  MT beta2_;
  MT epsilon_;

  MT beta1_pow_;
  MT beta2_pow_;
  const MT* moment1_;
  MT* moment1_out_;
  const MT* moment2_;
  MT* moment2_out_;
  const T* grad_;
  const T* param_;
  MT* trust_ratio_div_;
  const MT* master_param_;

  const int64_t* rows_;
  int64_t row_numel_;
  int64_t row_count_;

  SparseLambMomentREGUpdateFunctor(MT weight_decay, MT beta1, MT beta2,
                                   MT epsilon, MT beta1_pow, MT beta2_pow,
                                   const MT* mom1, MT* mom1_out, const MT* mom2,
                                   MT* mom2_out, const T* grad, const T* param,
                                   MT* trust_ratio_div, const MT* master_param,
                                   const int64_t* rows, int64_t row_numel,
                                   int64_t row_count)
      : weight_decay_(weight_decay),
        beta1_(beta1),
        beta2_(beta2),
        epsilon_(epsilon),
        beta1_pow_(beta1_pow),
        beta2_pow_(beta2_pow),
        moment1_(mom1),
        moment1_out_(mom1_out),
        moment2_(mom2),
        moment2_out_(mom2_out),
        grad_(grad),
        param_(param),
        trust_ratio_div_(trust_ratio_div),
        master_param_(master_param),
        rows_(rows),
        row_numel_(row_numel),
        row_count_(row_count) {}

  inline HOSTDEVICE void update(size_t i, MT g) const {
    // The following code is same as dense
    MT mom1 = moment1_[i];
    MT mom2 = moment2_[i];
    MT beta1_pow = beta1_pow_;
    MT beta2_pow = beta2_pow_;
    MT p = master_param_ ? master_param_[i] : static_cast<MT>(param_[i]);

    mom1 = beta1_ * mom1 + (static_cast<MT>(1) - beta1_) * g;
    mom2 = beta2_ * mom2 + (static_cast<MT>(1) - beta2_) * g * g;

    moment1_out_[i] = mom1;
    moment2_out_[i] = mom2;

    MT mom1_unbiased = mom1 / (static_cast<MT>(1) - beta1_pow);
    MT mom2_unbiased = mom2 / (static_cast<MT>(1) - beta2_pow);
    trust_ratio_div_[i] =
        mom1_unbiased / (sqrt(mom2_unbiased) + epsilon_) + weight_decay_ * p;
  }

  inline HOSTDEVICE void operator()(size_t i) const {
    auto row_idx =
        math::BinarySearch<int64_t>(rows_, row_count_, i / row_numel_);
    MT g = row_idx >= 0
               ? static_cast<MT>(grad_[row_idx * row_numel_ + i % row_numel_])
               : static_cast<MT>(0);
    update(i, g);
  }
};

// template <typename T, typename MT=T>
// class SparseLambMomentMENUpdateFunctor;

template <typename T, typename MT>
struct SparseLambMomentMENUpdateFunctor {
  // private:
  MT weight_decay_;
  MT beta1_;
  MT beta2_;
  MT epsilon_;

  const MT* beta1_pow_;
  MT* beta1_pow_out_;
  const MT* beta2_pow_;
  MT* beta2_pow_out_;
  const MT* moment1_;
  MT* moment1_out_;
  const MT* moment2_;
  MT* moment2_out_;
  const T* grad_;
  const T* param_;
  MT* trust_ratio_div_;
  const MT* master_param_;

  const int64_t* rows_;
  int64_t row_numel_;
  int64_t row_count_;

  // public:
  SparseLambMomentMENUpdateFunctor(
      MT weight_decay, MT beta1, MT beta2, MT epsilon, const MT* beta1_pow,
      MT* beta1_pow_out, const MT* beta2_pow, MT* beta2_pow_out, const MT* mom1,
      MT* mom1_out, const MT* mom2, MT* mom2_out, const T* grad, const T* param,
      MT* trust_ratio_div, const MT* master_param, const int64_t* rows,
      int64_t row_numel, int64_t row_count)
      : weight_decay_(weight_decay),
        beta1_(beta1),
        beta2_(beta2),
        epsilon_(epsilon),
        beta1_pow_(beta1_pow),
        beta1_pow_out_(beta1_pow_out),
        beta2_pow_(beta2_pow),
        beta2_pow_out_(beta2_pow_out),
        moment1_(mom1),
        moment1_out_(mom1_out),
        moment2_(mom2),
        moment2_out_(mom2_out),
        grad_(grad),
        param_(param),
        trust_ratio_div_(trust_ratio_div),
        master_param_(master_param),
        rows_(rows),
        row_numel_(row_numel),
        row_count_(row_count) {}

  inline HOSTDEVICE void update(size_t i, MT g) const {
    // The following code is same as dense
    MT mom1 = moment1_[i];
    MT mom2 = moment2_[i];
    MT beta1_pow = *beta1_pow_;
    MT beta2_pow = *beta2_pow_;
    MT p = master_param_ ? master_param_[i] : static_cast<MT>(param_[i]);

    mom1 = beta1_ * mom1 + (static_cast<MT>(1) - beta1_) * g;
    mom2 = beta2_ * mom2 + (static_cast<MT>(1) - beta2_) * g * g;

    moment1_out_[i] = mom1;
    moment2_out_[i] = mom2;

    MT mom1_unbiased = mom1 / (static_cast<MT>(1) - beta1_pow);
    MT mom2_unbiased = mom2 / (static_cast<MT>(1) - beta2_pow);
    trust_ratio_div_[i] =
        mom1_unbiased / (sqrt(mom2_unbiased) + epsilon_) + weight_decay_ * p;
    if (beta1_pow_out_ && beta1_pow_out_) {
      beta1_pow_out_[0] = beta1_pow * beta1_;
      beta2_pow_out_[0] = beta2_pow * beta2_;
    }
  }

  inline HOSTDEVICE void operator()(size_t i) const {
    auto row_idx =
        math::BinarySearch<int64_t>(rows_, row_count_, i / row_numel_);
    MT g = row_idx >= 0
               ? static_cast<MT>(grad_[row_idx * row_numel_ + i % row_numel_])
               : static_cast<MT>(0);
    update(i, g);
  }
};

template <typename T, typename MT>
struct LambParamUpateFunctor {
  const MT* lr_;
  const T* param_;
  const T* param_norm_;
  const MT* trust_ratio_div_;
  const MT* trust_ratio_div_norm_;
  T* param_out_;
  const MT* master_param_;
  MT* master_param_out_;

  LambParamUpateFunctor(const MT* lr, const T* param, const T* param_norm,
                        const MT* trust_ratio_div,
                        const MT* trust_ratio_div_norm, T* param_out,
                        const MT* master_param, MT* master_param_out)
      : lr_(lr),
        param_(param),
        param_norm_(param_norm),
        trust_ratio_div_(trust_ratio_div),
        trust_ratio_div_norm_(trust_ratio_div_norm),
        param_out_(param_out),
        master_param_(master_param),
        master_param_out_(master_param_out) {}

  inline HOSTDEVICE void operator()(size_t i) const {
    MT lr = *lr_;
    MT p_norm = static_cast<MT>(*param_norm_);
    MT t_norm = *trust_ratio_div_norm_;

    MT r = (p_norm > static_cast<MT>(0) && t_norm > static_cast<MT>(0))
               ? p_norm / t_norm
               : static_cast<MT>(1.0);
    lr *= r;
    MT p = master_param_ ? master_param_[i] : static_cast<MT>(param_[i]);
    p -= lr * trust_ratio_div_[i];
    param_out_[i] = static_cast<T>(p);
    if (master_param_) {
      master_param_out_[i] = p;
    }
  }
};

template <typename DeviceContext, typename T>
class LambOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    const auto* param_var = ctx.InputVar("Param");
    PADDLE_ENFORCE_EQ(param_var->IsType<framework::LoDTensor>(), true,
                      platform::errors::InvalidArgument(
                          "The Var(%s)'s type should be LoDTensor, "
                          "but the received is %s",
                          ctx.InputNames("Param").front(),
                          framework::ToTypeName(param_var->Type())));

    using paddle::framework::LoDTensor;
    using MPDType = typename details::MPTypeTrait<T>::Type;

    MPDType weight_decay =
        static_cast<MPDType>(ctx.Attr<float>("weight_decay"));
    MPDType beta1 = static_cast<MPDType>(ctx.Attr<float>("beta1"));
    MPDType beta2 = static_cast<MPDType>(ctx.Attr<float>("beta2"));
    MPDType epsilon = static_cast<MPDType>(ctx.Attr<float>("epsilon"));
    auto& param = GET_DATA_SAFELY(ctx.Input<LoDTensor>("Param"), "Input",
                                  "Param", "Lamb");
    auto* grad_var = ctx.InputVar("Grad");
    auto& mom1 = GET_DATA_SAFELY(ctx.Input<LoDTensor>("Moment1"), "Input",
                                 "Moment1", "Lamb");
    auto& mom2 = GET_DATA_SAFELY(ctx.Input<LoDTensor>("Moment2"), "Input",
                                 "Moment2", "Lamb");
    auto& lr = GET_DATA_SAFELY(ctx.Input<LoDTensor>("LearningRate"), "Input",
                               "LearningRate", "Lamb");

    auto& beta1_pow = GET_DATA_SAFELY(ctx.Input<LoDTensor>("Beta1Pow"), "Input",
                                      "Beta1Pow", "Lamb");
    auto& beta2_pow = GET_DATA_SAFELY(ctx.Input<LoDTensor>("Beta2Pow"), "Input",
                                      "Beta2Pow", "Lamb");

    auto& param_out = GET_DATA_SAFELY(ctx.Output<LoDTensor>("ParamOut"),
                                      "Output", "ParamOut", "Lamb");
    auto& mom1_out = GET_DATA_SAFELY(ctx.Output<LoDTensor>("Moment1Out"),
                                     "Output", "Moment1Out", "Lamb");
    auto& mom2_out = GET_DATA_SAFELY(ctx.Output<LoDTensor>("Moment2Out"),
                                     "Output", "Moment2Out", "Lamb");
    auto& beta1_pow_out = GET_DATA_SAFELY(ctx.Output<LoDTensor>("Beta1PowOut"),
                                          "Output", "Beta1PowOut", "Lamb");
    auto& beta2_pow_out = GET_DATA_SAFELY(ctx.Output<LoDTensor>("Beta2PowOut"),
                                          "Output", "Beta2PowOut", "Lamb");

    const bool multi_precision = ctx.Attr<bool>("multi_precision");
    const LoDTensor* master_param = nullptr;
    LoDTensor* master_param_out = nullptr;

    if (multi_precision) {
      bool has_master =
          ctx.HasInput("MasterParam") && ctx.HasOutput("MasterParamOut");
      PADDLE_ENFORCE_EQ(has_master, true,
                        platform::errors::InvalidArgument(
                            "The Input(MasterParam) and Output(MasterParamOut) "
                            "should not be null when "
                            "the attr `multi_precision` is true"));
      master_param = ctx.Input<LoDTensor>("MasterParam");
      master_param_out = ctx.Output<LoDTensor>("MasterParamOut");
    }
    const MPDType* master_in_data =
        multi_precision ? master_param->data<MPDType>() : nullptr;
    MPDType* master_out_data =
        multi_precision
            ? master_param_out->mutable_data<MPDType>(ctx.GetPlace())
            : nullptr;

    auto& dev_ctx = ctx.template device_context<DeviceContext>();
    platform::ForRange<DeviceContext> for_range(dev_ctx, param.numel());
    framework::Tensor trust_ratio_div =
        ctx.AllocateTmpTensor<MPDType, DeviceContext>(param.dims(), dev_ctx);

    // Update moments
    if (grad_var->IsType<framework::LoDTensor>()) {
      auto& grad = *ctx.Input<LoDTensor>("Grad");
      if (platform::is_gpu_place(ctx.GetPlace()) &&
          beta1_pow.place() == platform::CPUPlace() &&
          beta2_pow.place() == platform::CPUPlace()) {
        LambMomentREGUpdateFunctor<T, MPDType> moment_update_functor(
            weight_decay, beta1, beta2, epsilon,
            *beta1_pow.template data<MPDType>(),
            *beta2_pow.template data<MPDType>(), mom1.template data<MPDType>(),
            mom1_out.template mutable_data<MPDType>(ctx.GetPlace()),
            mom2.template data<MPDType>(),
            mom2_out.template mutable_data<MPDType>(ctx.GetPlace()),
            grad.template data<T>(), param.template data<T>(),
            trust_ratio_div.template data<MPDType>(), master_in_data);
        for_range(moment_update_functor);
        beta1_pow_out.template mutable_data<MPDType>(platform::CPUPlace())[0] =
            beta1 * beta1_pow.template data<MPDType>()[0];
        beta2_pow_out.template mutable_data<MPDType>(platform::CPUPlace())[0] =
            beta2 * beta2_pow.template data<MPDType>()[0];
      } else {
        LambMomentMENUpdateFunctor<T, MPDType> moment_update_functor(
            weight_decay, beta1, beta2, epsilon,
            beta1_pow.template data<MPDType>(),
            beta1_pow_out.template mutable_data<MPDType>(ctx.GetPlace()),
            beta2_pow.template data<MPDType>(),
            beta2_pow_out.template mutable_data<MPDType>(ctx.GetPlace()),
            mom1.template data<MPDType>(),
            mom1_out.template mutable_data<MPDType>(ctx.GetPlace()),
            mom2.template data<MPDType>(),
            mom2_out.template mutable_data<MPDType>(ctx.GetPlace()),
            grad.template data<T>(), param.template data<T>(),
            trust_ratio_div.template data<MPDType>(), master_in_data);
        for_range(moment_update_functor);
      }
    } else if (grad_var->IsType<framework::SelectedRows>()) {
      auto& grad = GET_DATA_SAFELY(ctx.Input<framework::SelectedRows>("Grad"),
                                   "Input", "Grad", "Lamb");
      if (grad.rows().size() == 0) {
        VLOG(3) << "grad row size is 0!!";
        return;
      }

      std::vector<int64_t> cpu_rows(grad.rows().begin(), grad.rows().end());
      bool is_strict_sorted = true;
      for (size_t i = 1; i < cpu_rows.size(); ++i) {
        if (cpu_rows[i - 1] >= cpu_rows[i]) {
          is_strict_sorted = false;
          break;
        }
      }

      framework::SelectedRows tmp_grad_merge;
      const framework::SelectedRows* grad_merge_ptr;
      if (is_strict_sorted) {
        grad_merge_ptr = &grad;
      } else {
        // merge duplicated rows if any.
        // The rows of grad_merge have been sorted inside MergeAdd functor
        scatter::MergeAdd<DeviceContext, T> merge_func;
        merge_func(dev_ctx, grad, &tmp_grad_merge, true);
        grad_merge_ptr = &tmp_grad_merge;
      }

      auto& grad_merge = *grad_merge_ptr;
      auto& grad_tensor = grad_merge.value();
      const T* grad_data = grad_tensor.template data<T>();
      const int64_t* rows = grad_merge.rows().Data(ctx.GetPlace());
      auto row_numel = grad_tensor.numel() / grad_merge.rows().size();
      if (platform::is_gpu_place(ctx.GetPlace()) &&
          beta1_pow.place() == platform::CPUPlace() &&
          beta2_pow.place() == platform::CPUPlace()) {
        SparseLambMomentREGUpdateFunctor<T, MPDType> moment_update_functor(
            weight_decay, beta1, beta2, epsilon,
            *beta1_pow.template data<MPDType>(),
            *beta2_pow.template data<MPDType>(), mom1.template data<MPDType>(),
            mom1_out.template mutable_data<MPDType>(ctx.GetPlace()),
            mom2.template data<MPDType>(),
            mom2_out.template mutable_data<MPDType>(ctx.GetPlace()), grad_data,
            param.template data<T>(), trust_ratio_div.template data<MPDType>(),
            master_in_data, rows, row_numel, grad_merge.rows().size());
        for_range(moment_update_functor);
        beta1_pow_out.template mutable_data<MPDType>(platform::CPUPlace())[0] =
            beta1 * beta1_pow.template data<MPDType>()[0];
        beta2_pow_out.template mutable_data<MPDType>(platform::CPUPlace())[0] =
            beta2 * beta2_pow.template data<MPDType>()[0];
      } else {
        SparseLambMomentMENUpdateFunctor<T, MPDType> moment_update_functor(
            weight_decay, beta1, beta2, epsilon,
            beta1_pow.template data<MPDType>(),
            beta1_pow_out.template mutable_data<MPDType>(ctx.GetPlace()),
            beta2_pow.template data<MPDType>(),
            beta2_pow_out.template mutable_data<MPDType>(ctx.GetPlace()),
            mom1.template data<MPDType>(),
            mom1_out.template mutable_data<MPDType>(ctx.GetPlace()),
            mom2.template data<MPDType>(),
            mom2_out.template mutable_data<MPDType>(ctx.GetPlace()), grad_data,
            param.template data<T>(), trust_ratio_div.template data<MPDType>(),
            master_in_data, rows, row_numel, grad_merge.rows().size());
        for_range(moment_update_functor);
      }
    } else {
      PADDLE_THROW(platform::errors::InvalidArgument(
          "Variable type not supported by lamb_op. Expect LoDTensor or "
          "SelectedRows, but got %s",
          framework::ToTypeName(param_var->Type())));
    }

    // Update parameter
    framework::Tensor p_norm_t =
        ctx.AllocateTmpTensor<T, DeviceContext>({1}, dev_ctx);
    framework::Tensor trust_ratio_div_norm_t =
        ctx.AllocateTmpTensor<MPDType, DeviceContext>({1}, dev_ctx);
    auto p_norm = framework::EigenScalar<T>::From(p_norm_t);
    auto trust_ratio_div_norm =
        framework::EigenScalar<MPDType>::From(trust_ratio_div_norm_t);

    auto p = framework::EigenVector<T>::Flatten(param);
    auto t = framework::EigenVector<MPDType>::Flatten(trust_ratio_div);

    auto* place = dev_ctx.eigen_device();
    p_norm.device(*place) = p.square().sum().sqrt();
    trust_ratio_div_norm.device(*place) = t.square().sum().sqrt();
    LambParamUpateFunctor<T, MPDType> param_update_functor(
        lr.template data<MPDType>(), param.template data<T>(),
        p_norm_t.template data<T>(), trust_ratio_div.template data<MPDType>(),
        trust_ratio_div_norm_t.template data<MPDType>(),
        param_out.template mutable_data<T>(ctx.GetPlace()), master_in_data,
        master_out_data);
    for_range(param_update_functor);
  }
};

}  // namespace operators
}  // namespace paddle
