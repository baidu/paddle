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
#include <math.h>
#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/math/algorithm.h"
#include "paddle/fluid/operators/math/selected_rows_functor.h"
#include "paddle/fluid/platform/for_range.h"

namespace paddle {
namespace operators {


static inline float GetAttrFromTensor(const framework::Tensor* tensor) {
  const float* tensor_data = tensor->data<float>();
  framework::Tensor cpu_tensor;
  if (platform::is_gpu_place(tensor->place())) {
    TensorCopySync(*tensor, platform::CPUPlace(), &cpu_tensor);
    tensor_data = cpu_tensor.data<float>();
  }
  if (platform::is_xpu_place(tensor->place())) {
    TensorCopySync(*tensor, platform::CPUPlace(), &cpu_tensor);
    tensor_data = cpu_tensor.data<float>();
  }
  return tensor_data[0];
}

template <typename T, int MajorType = Eigen::RowMajor,
          typename IndexType = Eigen::DenseIndex>
using EigenVector = framework::EigenVector<T, MajorType, IndexType>;

template <typename T>
struct DenseRmspropGradFunctor {
  inline explicit DenseRmspropGradFunctor(const T *grad) : grad_(grad) {}

  HOSTDEVICE inline T operator()(int64_t idx) const { return grad_[idx]; }

  const T *grad_;
};

template <typename T>
struct SparseRmspropGradFunctor {
  inline SparseRmspropGradFunctor(const T *grad, const int64_t *rows,
                                  int64_t row_numel, int64_t row_count)
      : grad_(grad),
        rows_(rows),
        row_numel_(row_numel),
        row_count_(row_count) {}

  HOSTDEVICE inline T operator()(int64_t idx) const {
    auto row_idx = math::BinarySearch(rows_, row_count_, idx / row_numel_);
    return row_idx >= 0 ? grad_[row_idx * row_numel_ + idx % row_numel_] : 0;
  }

  const T *grad_;
  const int64_t *rows_;
  int64_t row_numel_;
  int64_t row_count_;
};

template <typename T, typename GradFunctor>
struct UncenteredRmspropFunctor {
  UncenteredRmspropFunctor(T *param, T *ms, T *mom, const T *lr, T rho,
                           T epsilon, T momentum,
                           const GradFunctor &grad_functor)
      : param_(param),
        ms_(ms),
        mom_(mom),
        lr_(lr),
        rho_(rho),
        epsilon_(epsilon),
        momentum_(momentum),
        grad_functor_(grad_functor) {}

  HOSTDEVICE inline void operator()(int64_t idx) const {
    T g = grad_functor_(idx);
    T ms_out = rho_ * ms_[idx] + (1 - rho_) * g * g;
    T mom_out = momentum_ * mom_[idx] + lr_[0] * g / sqrt(ms_out + epsilon_);
    param_[idx] -= mom_out;
    ms_[idx] = ms_out;
    mom_[idx] = mom_out;
  }

  T *param_;
  T *ms_;
  T *mom_;
  const T *lr_;
  T rho_;
  T epsilon_;
  T momentum_;
  GradFunctor grad_functor_;
};

template <typename T, typename GradFunctor>
struct CenteredRmspropFunctor {
  CenteredRmspropFunctor(T *param, T *ms, T *mom, T *mean_grad, const T *lr,
                         T rho, T epsilon, T momentum,
                         const GradFunctor &grad_functor)
      : param_(param),
        ms_(ms),
        mom_(mom),
        mean_grad_(mean_grad),
        lr_(lr),
        rho_(rho),
        epsilon_(epsilon),
        momentum_(momentum),
        grad_functor_(grad_functor) {}

  HOSTDEVICE inline void operator()(int64_t idx) const {
    T g = grad_functor_(idx);
    T ms_out = rho_ * ms_[idx] + (1 - rho_) * g * g;
    T mg_out = rho_ * mean_grad_[idx] + (1 - rho_) * g;
    T mom_out = momentum_ * mom_[idx] +
                lr_[0] * g / sqrt(ms_out - mg_out * mg_out + epsilon_);
    param_[idx] -= mom_out;
    ms_[idx] = ms_out;
    mom_[idx] = mom_out;
    mean_grad_[idx] = mg_out;
  }

  T *param_;
  T *ms_;
  T *mom_;
  T *mean_grad_;
  const T *lr_;
  T rho_;
  T epsilon_;
  T momentum_;
  GradFunctor grad_functor_;
};

template <typename DeviceContext, typename T>
class RmspropOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    using LoDTensor = framework::LoDTensor;
    auto *grad_var = ctx.InputVar("Grad");
    auto *param_out = ctx.Output<LoDTensor>("ParamOut");
    auto *moment_out = ctx.Output<LoDTensor>("MomentOut");
    auto *mean_square_out = ctx.Output<LoDTensor>("MeanSquareOut");

    auto epsilon = static_cast<T>(ctx.Attr<float>("epsilon"));
    auto rho = static_cast<T>(ctx.Attr<float>("decay"));
    auto momentum = static_cast<T>(ctx.Attr<float>("momentum"));
    bool centered = ctx.Attr<bool>("centered");

    auto &p_tensor = *ctx.Input<LoDTensor>("Param");
    auto &ms_tensor = *ctx.Input<LoDTensor>("MeanSquare");
    auto &lr_tensor = *ctx.Input<LoDTensor>("LearningRate");
    auto &mom_tensor = *ctx.Input<LoDTensor>("Moment");

    PADDLE_ENFORCE_EQ(&p_tensor, param_out,
                      platform::errors::InvalidArgument(
                          "Param and ParamOut must be the same Tensor"));
    PADDLE_ENFORCE_EQ(&mom_tensor, moment_out,
                      platform::errors::InvalidArgument(
                          "Moment and MomentOut must be the same Tensor"));
    PADDLE_ENFORCE_EQ(
        &ms_tensor, mean_square_out,
        platform::errors::InvalidArgument(
            "MeanSquare and MeanSquareOut must be the same Tensor"));

    auto &dev_ctx = ctx.template device_context<DeviceContext>();
    size_t limit = static_cast<size_t>(ms_tensor.numel());

    if (grad_var->IsType<LoDTensor>()) {
      auto &grad_tensor = grad_var->Get<LoDTensor>();

      if (std::is_same<DeviceContext, platform::CPUDeviceContext>::value) {
        auto &place =
            *ctx.template device_context<DeviceContext>().eigen_device();
        auto lr_value = lr_tensor.data<T>()[0];

        auto p = EigenVector<T>::Flatten(p_tensor);
        auto ms = EigenVector<T>::Flatten(ms_tensor);
        auto g = EigenVector<T>::Flatten(grad_tensor);
        auto mom = EigenVector<T>::Flatten(mom_tensor);

        auto p_out = EigenVector<T>::Flatten(*param_out);
        auto mom_out = EigenVector<T>::Flatten(*moment_out);
        auto ms_out = EigenVector<T>::Flatten(*mean_square_out);

        ms_out.device(place) = rho * ms + (1 - rho) * g * g;
        if (centered) {
          auto &mg_tensor = *ctx.Input<LoDTensor>("MeanGrad");
          auto mg = EigenVector<T>::Flatten(mg_tensor);
          auto *mean_grad_out = ctx.Output<LoDTensor>("MeanGradOut");
          PADDLE_ENFORCE_EQ(
              &mg_tensor, mean_grad_out,
              platform::errors::InvalidArgument(
                  "MeanGrad and MeanGradOut must be the same Tensor"));
          auto mg_out = EigenVector<T>::Flatten(*mean_grad_out);

          mg_out.device(place) = rho * mg + (1 - rho) * g;
          mom_out.device(place) =
              momentum * mom +
              lr_value * g / (ms_out - mg_out.square() + epsilon).sqrt();
        } else {
          mom_out.device(place) =
              momentum * mom + lr_value * g / (ms_out + epsilon).sqrt();
        }
        p_out.device(place) = p - mom_out;
      } else {
        DenseRmspropGradFunctor<T> grad_func(grad_tensor.data<T>());
        platform::ForRange<DeviceContext> for_range(dev_ctx, limit);
        if (centered) {
          auto &mg_tensor = *ctx.Input<LoDTensor>("MeanGrad");
          auto *mean_grad_out = ctx.Output<LoDTensor>("MeanGradOut");
          PADDLE_ENFORCE_EQ(
              &mg_tensor, mean_grad_out,
              platform::errors::InvalidArgument(
                  "MeanGrad and MeanGradOut must be the same Tensor"));
          for_range(CenteredRmspropFunctor<T, DenseRmspropGradFunctor<T>>(
              param_out->mutable_data<T>(ctx.GetPlace()),
              mean_square_out->mutable_data<T>(ctx.GetPlace()),
              moment_out->mutable_data<T>(ctx.GetPlace()),
              mean_grad_out->mutable_data<T>(ctx.GetPlace()),
              lr_tensor.data<T>(), rho, epsilon, momentum, grad_func));
        } else {
          for_range(UncenteredRmspropFunctor<T, DenseRmspropGradFunctor<T>>(
              param_out->mutable_data<T>(ctx.GetPlace()),
              mean_square_out->mutable_data<T>(ctx.GetPlace()),
              moment_out->mutable_data<T>(ctx.GetPlace()), lr_tensor.data<T>(),
              rho, epsilon, momentum, grad_func));
        }
      }
    } else if (grad_var->IsType<framework::SelectedRows>()) {
      auto &grad = grad_var->Get<framework::SelectedRows>();
      framework::SelectedRows tmp_merged_grad;
      framework::SelectedRows *merged_grad = &tmp_merged_grad;
      math::scatter::MergeAdd<DeviceContext, T> merge_func;
      merge_func(dev_ctx, grad, merged_grad);

      platform::ForRange<DeviceContext> for_range(dev_ctx, limit);
      const int64_t *rows = merged_grad->rows().Data(ctx.GetPlace());

      auto &merged_tensor = merged_grad->value();
      int64_t row_count = merged_grad->rows().size();
      int64_t row_numel = merged_tensor.numel() / row_count;
      SparseRmspropGradFunctor<T> grad_func(merged_tensor.data<T>(), rows,
                                            row_numel, row_count);

      if (centered) {
        auto &mg_tensor = *ctx.Input<LoDTensor>("MeanGrad");
        auto *mean_grad_out = ctx.Output<LoDTensor>("MeanGradOut");
        PADDLE_ENFORCE_EQ(
            &mg_tensor, mean_grad_out,
            platform::errors::InvalidArgument(
                "MeanGrad and MeanGradOut must be the same Tensor"));
        for_range(CenteredRmspropFunctor<T, SparseRmspropGradFunctor<T>>(
            param_out->mutable_data<T>(ctx.GetPlace()),
            mean_square_out->mutable_data<T>(ctx.GetPlace()),
            moment_out->mutable_data<T>(ctx.GetPlace()),
            mean_grad_out->mutable_data<T>(ctx.GetPlace()), lr_tensor.data<T>(),
            rho, epsilon, momentum, grad_func));
      } else {
        for_range(UncenteredRmspropFunctor<T, SparseRmspropGradFunctor<T>>(
            param_out->mutable_data<T>(ctx.GetPlace()),
            mean_square_out->mutable_data<T>(ctx.GetPlace()),
            moment_out->mutable_data<T>(ctx.GetPlace()), lr_tensor.data<T>(),
            rho, epsilon, momentum, grad_func));
      }
    } else {
      PADDLE_ENFORCE_EQ(false, true,
                        platform::errors::PermissionDenied(
                            "Unsupported Variable Type of Grad "
                            "in RmspropOp. Excepted LodTensor "
                            "or SelectedRows, But received [%s]",
                            paddle::framework::ToTypeName(grad_var->Type())));
    }
  }
};

}  // namespace operators
}  // namespace paddle
