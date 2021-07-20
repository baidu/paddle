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
#include <math.h>  // for sqrt in CPU and CUDA
#include <Eigen/Dense>
#include <string>
#include <unordered_map>
#include <vector>
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/threadpool.h"
#include "paddle/fluid/operators/math/algorithm.h"
#include "paddle/fluid/operators/math/selected_rows_functor.h"
#include "paddle/fluid/platform/for_range.h"

namespace paddle {
namespace operators {

namespace scatter = paddle::operators::math::scatter;

static inline float GetAttriFromTensor(const framework::Tensor* tensor) {
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

class AdamWOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override;
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override;
  framework::OpKernelType GetKernelTypeForVar(
      const std::string& var_name, const framework::Tensor& tensor,
      const framework::OpKernelType& expected_kernel_type) const override;
};

struct GPUAdamW;
struct CPUAdamW;

template <typename T, typename Flavour>
class AdamWFunctor;

template <typename T>
class AdamWFunctor<T, GPUAdamW> {
 private:
  T beta1_;
  T beta2_;
  T epsilon_;
  T weight_decay_;

  const T* beta1_pow_;
  const T* beta2_pow_;
  const T* moment1_;
  T* moment1_out_;
  const T* moment2_;
  T* moment2_out_;
  const T* lr_;
  const T* grad_;
  const T* param_;
  T* param_out_;

 public:
  AdamWFunctor(T beta1, T beta2, T epsilon, T weight_decay, const T* beta1_pow,
               const T* beta2_pow, const T* mom1, T* mom1_out, const T* mom2,
               T* mom2_out, const T* lr, const T* grad, const T* param,
               T* param_out)
      : beta1_(beta1),
        beta2_(beta2),
        epsilon_(epsilon),
        weight_decay_(weight_decay),
        beta1_pow_(beta1_pow),
        beta2_pow_(beta2_pow),
        moment1_(mom1),
        moment1_out_(mom1_out),
        moment2_(mom2),
        moment2_out_(mom2_out),
        lr_(lr),
        grad_(grad),
        param_(param),
        param_out_(param_out) {}

  inline HOSTDEVICE void operator()(size_t i) const {
    // Merge all memory access together.
    T g = grad_[i];
    T mom1 = moment1_[i];
    T mom2 = moment2_[i];
    T lr = *lr_;
    T beta1_pow = *beta1_pow_;
    T beta2_pow = *beta2_pow_;
    T p = param_[i];

    // Calculation
    T wd = 1 - weight_decay_ * lr;
    lr *= sqrt(1 - beta2_pow) / (1 - beta1_pow);

    mom1 = beta1_ * mom1 + (1 - beta1_) * g;
    mom2 = beta2_ * mom2 + (1 - beta2_) * g * g;
    p = wd * p - lr * (mom1 / (sqrt(mom2) + epsilon_ * sqrt(1 - beta2_pow)));

    // Write back to global memory
    moment1_out_[i] = mom1;
    moment2_out_[i] = mom2;
    param_out_[i] = p;
  }
};

template <typename T>
class AdamWFunctor<T, CPUAdamW> {
 private:
  T beta1_;
  T beta2_;
  T epsilon_;
  T weight_decay_;

  const T* beta1_pow_;
  const T* beta2_pow_;
  const T* moment1_;
  T* moment1_out_;
  const T* moment2_;
  T* moment2_out_;
  const T* lr_;
  const T* grad_;
  const T* param_;
  T* param_out_;

 public:
  AdamWFunctor(T beta1, T beta2, T epsilon, T weight_decay, const T* beta1_pow,
               const T* beta2_pow, const T* mom1, T* mom1_out, const T* mom2,
               T* mom2_out, const T* lr, const T* grad, const T* param,
               T* param_out)
      : beta1_(beta1),
        beta2_(beta2),
        epsilon_(epsilon),
        weight_decay_(weight_decay),
        beta1_pow_(beta1_pow),
        beta2_pow_(beta2_pow),
        moment1_(mom1),
        moment1_out_(mom1_out),
        moment2_(mom2),
        moment2_out_(mom2_out),
        lr_(lr),
        grad_(grad),
        param_(param),
        param_out_(param_out) {}

  void operator()(size_t numel) const {
    Eigen::Map<const Eigen::Array<T, 1, Eigen::Dynamic>> g{
        grad_, static_cast<Eigen::Index>(numel)};
    Eigen::Map<const Eigen::Array<T, 1, Eigen::Dynamic>> mom1{
        moment1_, static_cast<Eigen::Index>(numel)};
    Eigen::Map<const Eigen::Array<T, 1, Eigen::Dynamic>> mom2{
        moment2_, static_cast<Eigen::Index>(numel)};
    Eigen::Map<const Eigen::Array<T, 1, Eigen::Dynamic>> param{
        param_, static_cast<Eigen::Index>(numel)};

    Eigen::Map<Eigen::Array<T, 1, Eigen::Dynamic>> param_out{
        param_out_, static_cast<Eigen::Index>(numel)};
    Eigen::Map<Eigen::Array<T, 1, Eigen::Dynamic>> moment1_out{
        moment1_out_, static_cast<Eigen::Index>(numel)};
    Eigen::Map<Eigen::Array<T, 1, Eigen::Dynamic>> moment2_out{
        moment2_out_, static_cast<Eigen::Index>(numel)};

    T lr = *lr_;
    T beta1_pow = *beta1_pow_;
    T beta2_pow = *beta2_pow_;

    // Calculation
    T wd = 1 - weight_decay_ * lr;
    lr *= sqrt(1 - beta2_pow) / (1 - beta1_pow);

    moment1_out = beta1_ * mom1 + (1 - beta1_) * g;
    moment2_out = beta2_ * mom2 + (1 - beta2_) * g * g;
    param_out = wd * param -
                lr * (moment1_out /
                      (moment2_out.sqrt() + epsilon_ * sqrt(1 - beta2_pow)));
  }
};

template <typename T, typename Flavour, typename MT = T>
class SparseAdamWFunctor;

template <typename T, typename MT>
class SparseAdamWFunctor<T, GPUAdamW, MT> {
 private:
  MT beta1_;
  MT beta2_;
  MT epsilon_;
  MT weight_decay_;

  const MT* beta1_pow_;
  const MT* beta2_pow_;
  const MT* moment1_;
  MT* moment1_out_;
  const MT* moment2_;
  MT* moment2_out_;
  const MT* lr_;
  const T* grad_;
  const T* param_;
  T* param_out_;
  const MT* master_param_;
  MT* master_param_out_;

  const int64_t* rows_;
  int64_t row_numel_;
  int64_t row_count_;
  bool lazy_mode_;

 public:
  SparseAdamWFunctor(MT beta1, MT beta2, MT epsilon, MT weight_decay,
                     const MT* beta1_pow, const MT* beta2_pow, const MT* mom1,
                     MT* mom1_out, const MT* mom2, MT* mom2_out, const MT* lr,
                     const T* grad, const T* param, T* param_out,
                     const MT* master_param, MT* master_param_out,
                     const int64_t* rows, int64_t row_numel, int64_t row_count,
                     bool lazy_mode)
      : beta1_(beta1),
        beta2_(beta2),
        epsilon_(epsilon),
        weight_decay_(weight_decay),
        beta1_pow_(beta1_pow),
        beta2_pow_(beta2_pow),
        moment1_(mom1),
        moment1_out_(mom1_out),
        moment2_(mom2),
        moment2_out_(mom2_out),
        lr_(lr),
        grad_(grad),
        param_(param),
        param_out_(param_out),
        master_param_(master_param),
        master_param_out_(master_param_out),
        rows_(rows),
        row_numel_(row_numel),
        row_count_(row_count),
        lazy_mode_(lazy_mode) {}

  inline HOSTDEVICE void adamw_update(size_t i, MT g) const {
    // The following code is the same as dense
    MT mom1 = moment1_[i];
    MT mom2 = moment2_[i];
    MT lr = *lr_;
    MT beta1_pow = *beta1_pow_;
    MT beta2_pow = *beta2_pow_;
    MT p = master_param_ ? master_param_[i] : static_cast<MT>(param_[i]);

    // Calculation
    MT wd = static_cast<MT>(1.0) - weight_decay_ * lr;
    lr *= sqrt(static_cast<MT>(1.0) - beta2_pow) /
          (static_cast<MT>(1.0) - beta1_pow);

    mom1 = beta1_ * mom1 + (static_cast<MT>(1.0) - beta1_) * g;
    mom2 = beta2_ * mom2 + (static_cast<MT>(1.0) - beta2_) * g * g;
    p = wd * p -
        lr * (mom1 /
              (sqrt(mom2) + epsilon_ * sqrt(static_cast<MT>(1.0) - beta2_pow)));

    // Write back to global memory
    moment1_out_[i] = mom1;
    moment2_out_[i] = mom2;
    param_out_[i] = static_cast<T>(p);
    if (master_param_out_) {
      master_param_out_[i] = p;
    }
  }

  inline HOSTDEVICE void operator()(size_t i) const {
    auto row_idx =
        math::BinarySearch<int64_t>(rows_, row_count_, i / row_numel_);
    if (lazy_mode_ && row_idx < 0) {
      return;
    } else {
      MT g = row_idx >= 0
                 ? static_cast<MT>(grad_[row_idx * row_numel_ + i % row_numel_])
                 : static_cast<MT>(0);
      adamw_update(i, g);
    }
  }
};

template <typename T>
class SparseAdamWFunctor<T, CPUAdamW, T> {
 private:
  T beta1_;
  T beta2_;
  T epsilon_;
  T weight_decay_;

  const T* beta1_pow_;
  const T* beta2_pow_;
  const T* moment1_;
  T* moment1_out_;
  const T* moment2_;
  T* moment2_out_;
  const T* lr_;
  const T* grad_;
  const T* param_;
  T* param_out_;

  const int64_t* rows_;
  int64_t row_numel_;
  int64_t row_count_;

 public:
  SparseAdamWFunctor(T beta1, T beta2, T epsilon, T weight_decay,
                     const T* beta1_pow, const T* beta2_pow, const T* mom1,
                     T* mom1_out, const T* mom2, T* mom2_out, const T* lr,
                     const T* grad, const T* param, T* param_out,
                     const int64_t* rows, int64_t row_numel, int64_t row_count,
                     bool lazy_mode)
      : beta1_(beta1),
        beta2_(beta2),
        epsilon_(epsilon),
        weight_decay_(weight_decay),
        beta1_pow_(beta1_pow),
        beta2_pow_(beta2_pow),
        moment1_(mom1),
        moment1_out_(mom1_out),
        moment2_(mom2),
        moment2_out_(mom2_out),
        lr_(lr),
        grad_(grad),
        param_(param),
        param_out_(param_out),
        rows_(rows),
        row_numel_(row_numel),
        row_count_(row_count) {}

  inline HOSTDEVICE void adamw_update(size_t i, T g) const {
    // The following code is the same as dense
    T mom1 = moment1_[i];
    T mom2 = moment2_[i];
    T lr = *lr_;
    T beta1_pow = *beta1_pow_;
    T beta2_pow = *beta2_pow_;
    T p = param_[i];

    // Calculation
    T wd = 1 - weight_decay_ * lr;
    lr *= sqrt(1 - beta2_pow) / (1 - beta1_pow);

    mom1 = beta1_ * mom1 + (1 - beta1_) * g;
    mom2 = beta2_ * mom2 + (1 - beta2_) * g * g;
    p = wd * p - lr * (mom1 / (sqrt(mom2) + epsilon_ * sqrt(1 - beta2_pow)));

    // Write back to global memory
    moment1_out_[i] = mom1;
    moment2_out_[i] = mom2;
    param_out_[i] = p;
  }

  inline void operator()(size_t numel) const {
    // lr could be reuse
    T lr = *lr_;
    T beta1_pow = *beta1_pow_;
    T beta2_pow = *beta2_pow_;

    T wd = 1 - weight_decay_ * lr;
    lr *= sqrt(1 - beta2_pow) / (1 - beta1_pow);
    int64_t row_count = static_cast<int64_t>(numel / row_numel_);

    for (int64_t i = 0, j = 0; i != row_count; ++i) {
      if (i == *(rows_ + j)) {
        for (int64_t k = 0; k != row_numel_; ++k) {
          T g = grad_[j * row_numel_ + k];
          adamw_update(i * row_numel_ + k, g);
        }
        ++j;
      } else {
        for (int64_t k = 0; k != row_numel_; ++k) {
          T mom1 = moment1_[i * row_numel_ + k];
          T mom2 = moment2_[i * row_numel_ + k];
          T p = param_[i * row_numel_ + k];

          mom1 = beta1_ * mom1;
          mom2 = beta2_ * mom2;

          p = wd * p -
              lr * (mom1 / (sqrt(mom2) + epsilon_ * sqrt(1 - beta2_pow)));
          // Write back to global memory
          moment1_out_[i * row_numel_ + k] = mom1;
          moment2_out_[i * row_numel_ + k] = mom2;
          param_out_[i * row_numel_ + k] = p;
        }
      }
    }
  }
};

template <typename DeviceContext, typename T>
class AdamWOpKernel : public framework::OpKernel<T> {
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

    int64_t min_row_size_to_use_multithread =
        ctx.Attr<int64_t>("min_row_size_to_use_multithread");
    bool lazy_mode = ctx.Attr<bool>("lazy_mode");
    bool use_global_beta_pow = ctx.Attr<bool>("use_global_beta_pow");
    VLOG(4) << "use_global_beta_pow:" << use_global_beta_pow;

    auto* param = ctx.Input<LoDTensor>("Param");
    auto* grad_var = ctx.InputVar("Grad");
    auto* mom1 = ctx.Input<LoDTensor>("Moment1");
    auto* mom2 = ctx.Input<LoDTensor>("Moment2");
    auto* lr = ctx.Input<LoDTensor>("LearningRate");

    auto* beta1_pow = ctx.Input<LoDTensor>("Beta1Pow");
    auto* beta2_pow = ctx.Input<LoDTensor>("Beta2Pow");

    auto* param_out = ctx.Output<LoDTensor>("ParamOut");
    auto* mom1_out = ctx.Output<LoDTensor>("Moment1Out");
    auto* mom2_out = ctx.Output<LoDTensor>("Moment2Out");
    auto* beta1_pow_out = ctx.Output<LoDTensor>("Beta1PowOut");
    auto* beta2_pow_out = ctx.Output<LoDTensor>("Beta2PowOut");

    T beta1 = static_cast<T>(ctx.Attr<float>("beta1"));
    if (ctx.HasInput("Beta1Tensor")) {
      auto* beta1_tensor = ctx.Input<framework::Tensor>("Beta1Tensor");
      PADDLE_ENFORCE_EQ(beta1_tensor->numel(), 1,
                        platform::errors::InvalidArgument(
                            "Input(Beta1Tensor) size must be 1, but get %d",
                            beta1_tensor->numel()));
      beta1 = static_cast<T>(GetAttriFromTensor(beta1_tensor));
    }
    T beta2 = static_cast<T>(ctx.Attr<float>("beta2"));
    if (ctx.HasInput("Beta2Tensor")) {
      auto* beta2_tensor = ctx.Input<framework::Tensor>("Beta2Tensor");
      PADDLE_ENFORCE_EQ(beta2_tensor->numel(), 1,
                        platform::errors::InvalidArgument(
                            "Input(Beta2Tensor) size must be 1, but get %d",
                            beta2_tensor->numel()));
      beta2 = static_cast<T>(GetAttriFromTensor(beta2_tensor));
    }
    T epsilon = static_cast<T>(ctx.Attr<float>("epsilon"));
    if (ctx.HasInput("EpsilonTensor")) {
      auto* epsilon_tensor = ctx.Input<framework::Tensor>("EpsilonTensor");
      PADDLE_ENFORCE_EQ(epsilon_tensor->numel(), 1,
                        platform::errors::InvalidArgument(
                            "Input(EpsilonTensor) size must be 1, but get %d",
                            epsilon_tensor->numel()));
      epsilon = static_cast<T>(GetAttriFromTensor(epsilon_tensor));
    }
    T weight_decay = static_cast<T>(ctx.Attr<float>("weightdecay"));
    if (ctx.HasInput("WeightDecayTensor")) {
      auto* weight_decay_tensor =
          ctx.Input<framework::Tensor>("WeightDecayTensor");
      PADDLE_ENFORCE_EQ(
          weight_decay_tensor->numel(), 1,
          platform::errors::InvalidArgument(
              "Input(WeightDecayTensor) size must be 1, but get %d",
              weight_decay_tensor->numel()));
      weight_decay = static_cast<T>(GetAttriFromTensor(weight_decay_tensor));
    }
    VLOG(3) << "beta1_pow.numel() : " << beta1_pow->numel()
            << "beta2_pow.numel() : " << beta2_pow->numel();
    VLOG(3) << "param.numel(): " << param->numel();

    PADDLE_ENFORCE_EQ(beta1_pow_out->numel(), 1,
                      platform::errors::InvalidArgument(
                          "beta1 pow output size should be 1, but received "
                          "value is:%d.",
                          beta1_pow_out->numel()));

    PADDLE_ENFORCE_EQ(beta2_pow_out->numel(), 1,
                      platform::errors::InvalidArgument(
                          "beta2 pow output size should be 1, but received "
                          "value is:%d.",
                          beta2_pow_out->numel()));

    if (grad_var->IsType<framework::LoDTensor>()) {
      auto* grad = ctx.Input<LoDTensor>("Grad");

      AdamWFunctor<T, CPUAdamW> functor(
          beta1, beta2, epsilon, weight_decay, beta1_pow->data<T>(),
          beta2_pow->data<T>(), mom1->data<T>(),
          mom1_out->mutable_data<T>(ctx.GetPlace()), mom2->data<T>(),
          mom2_out->mutable_data<T>(ctx.GetPlace()), lr->data<T>(),
          grad->data<T>(), param->data<T>(),
          param_out->mutable_data<T>(ctx.GetPlace()));
      functor(param->numel());
      if (!use_global_beta_pow) {
        beta1_pow_out->mutable_data<T>(ctx.GetPlace())[0] =
            beta1 * beta1_pow->data<T>()[0];
        beta2_pow_out->mutable_data<T>(ctx.GetPlace())[0] =
            beta2 * beta2_pow->data<T>()[0];
      }
    } else if (grad_var->IsType<framework::SelectedRows>()) {
      auto* grad = ctx.Input<framework::SelectedRows>("Grad");
      if (grad->rows().size() == 0) {
        VLOG(3) << "grad row size is 0!!";
        return;
      }

      std::vector<int64_t> cpu_rows(grad->rows().begin(), grad->rows().end());
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
        grad_merge_ptr = grad;
      } else {
        // merge duplicated rows if any.
        // The rows of grad_merge have been sorted inside MergeAdd functor
        scatter::MergeAdd<DeviceContext, T> merge_func;
        merge_func(ctx.template device_context<DeviceContext>(), *grad,
                   &tmp_grad_merge, true);
        grad_merge_ptr = &tmp_grad_merge;
      }

      auto& grad_merge = *grad_merge_ptr;
      auto& grad_tensor = grad_merge.value();
      const T* grad_data = grad_tensor.template data<T>();
      const int64_t* rows = grad_merge.rows().Data(ctx.GetPlace());
      auto row_numel = grad_tensor.numel() / grad_merge.rows().size();

      SparseAdamWFunctor<T, CPUAdamW> functor(
          beta1, beta2, epsilon, weight_decay, beta1_pow->data<T>(),
          beta2_pow->data<T>(), mom1->data<T>(),
          mom1_out->mutable_data<T>(ctx.GetPlace()), mom2->data<T>(),
          mom2_out->mutable_data<T>(ctx.GetPlace()), lr->data<T>(), grad_data,
          param->data<T>(), param_out->mutable_data<T>(ctx.GetPlace()), rows,
          row_numel, grad_merge.rows().size(), lazy_mode);
      // update beta1 and beta2
      if (!use_global_beta_pow) {
        beta1_pow_out->mutable_data<T>(ctx.GetPlace())[0] =
            beta1 * beta1_pow->data<T>()[0];
        beta2_pow_out->mutable_data<T>(ctx.GetPlace())[0] =
            beta2 * beta2_pow->data<T>()[0];
      }
      if (lazy_mode) {
        VLOG(3) << "run cpu lazy mode";
        size_t row_count = grad_merge.rows().size();
        std::vector<int64_t> cpu_rows(grad_merge.rows());
        for (size_t row_index = 0; row_index < row_count; ++row_index) {
          for (size_t offset = 0; offset < row_numel; ++offset) {
            size_t i = cpu_rows[row_index] * row_numel + offset;
            functor.adamw_update(i, grad_data[row_index * row_numel + offset]);
          }
        }
      }
#ifndef _WIN32
      else if (FLAGS_inner_op_parallelism > 1 &&  // NOLINT
               min_row_size_to_use_multithread > 0 &&
               param->dims()[0] > min_row_size_to_use_multithread) {
        VLOG(3) << "use multi thread, inner_op_parallelism="
                << FLAGS_inner_op_parallelism
                << " min_row_size_to_use_multithread="
                << min_row_size_to_use_multithread;
        if (FLAGS_inner_op_parallelism > 10) {
          VLOG(1) << "FLAGS_inner_op_parallelism " << FLAGS_inner_op_parallelism
                  << " is two large!";
        }
        auto& grad_rows = grad_merge.rows();
        std::unordered_map<size_t, int> row_id_to_grad_row_offset;
        size_t param_row_count = param->numel() / row_numel;
        if (param_row_count < 1000) {
          VLOG(1) << "param_row_count should be larger then 1000 to use "
                     "multi thread, currently "
                  << param_row_count;
        }
        for (size_t i = 0; i < grad_rows.size(); ++i) {
          row_id_to_grad_row_offset[grad_rows[i]] = i;
        }
        std::vector<std::future<void>> fs;
        int64_t line_in_each_thread =
            param_row_count / FLAGS_inner_op_parallelism + 1;
        for (int i = 0; i < FLAGS_inner_op_parallelism; ++i) {
          int64_t start = i * line_in_each_thread;
          int64_t end = (i + 1) * line_in_each_thread;
          if (start >= static_cast<int64_t>(param_row_count)) {
            break;
          }
          if (end > static_cast<int64_t>(param_row_count)) {
            end = static_cast<int64_t>(param_row_count);
          }
          fs.push_back(framework::Async([&functor, &row_id_to_grad_row_offset,
                                         &grad_data, row_numel, start, end]() {
            for (int64_t row_id = start; row_id < end; ++row_id) {
              auto iter = row_id_to_grad_row_offset.find(row_id);
              if (iter != row_id_to_grad_row_offset.end()) {
                for (size_t row_offset = 0U; row_offset < row_numel;
                     ++row_offset) {
                  functor.adamw_update(
                      row_id * row_numel + row_offset,
                      grad_data[iter->second * row_numel + row_offset]);
                }
              } else {
                for (size_t row_offset = 0U; row_offset < row_numel;
                     ++row_offset) {
                  functor.adamw_update(row_id * row_numel + row_offset, 0);
                }
              }
            }
          }));
        }
        for (size_t i = 0; i < fs.size(); ++i) fs[i].wait();
      }
#endif        // !_WIN32
      else {  // NOLINT
        functor(param->numel());
      }
    } else {
      PADDLE_THROW(platform::errors::InvalidArgument(
          "Variable type not supported by adamw_op"));
    }
  }
};

}  // namespace operators
}  // namespace paddle
