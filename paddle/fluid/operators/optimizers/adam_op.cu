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
#include "paddle/fluid/operators/optimizers/adam_op.h"

namespace paddle {
namespace operators {

template <typename T>
__global__ void AdamKernelREG(T beta1, T beta2, T epsilon, T beta1_pow_,
                              T beta2_pow_, const T* moment1, T* moment1_out,
                              const T* moment2, T* moment2_out, const T* lr_,
                              const T* grad, const T* param, T* param_out,
                              int ndim) {
  T lr = *lr_;
  T beta1_pow = beta1_pow_;
  T beta2_pow = beta2_pow_;

  lr *=
      sqrt(static_cast<T>(1.0) - beta2_pow) / (static_cast<T>(1.0) - beta1_pow);

  int id = blockIdx.x * blockDim.x + threadIdx.x;

  for (; id < ndim; id += gridDim.x * blockDim.x) {
    T p = param[id];
    T g = grad[id];
    T mom1 = moment1[id];
    T mom2 = moment2[id];
    mom1 = beta1 * mom1 + (static_cast<T>(1.0) - beta1) * g;
    mom2 = beta2 * mom2 + (static_cast<T>(1.0) - beta2) * g * g;
    p -= lr * (mom1 /
               (sqrt(mom2) + epsilon * sqrt(static_cast<T>(1.0) - beta2_pow)));

    moment1_out[id] = mom1;
    moment2_out[id] = mom2;
    param_out[id] = p;
  }
}

template <typename T>
__global__ void AdamKernelMEM(T beta1, T beta2, T epsilon, const T* beta1_pow_,
                              const T* beta2_pow_, const T* moment1,
                              T* moment1_out, const T* moment2, T* moment2_out,
                              const T* lr_, const T* grad, const T* param,
                              T* param_out, int ndim) {
  T lr = *lr_;
  T beta1_pow = *beta1_pow_;
  T beta2_pow = *beta2_pow_;

  lr *=
      sqrt(static_cast<T>(1.0) - beta2_pow) / (static_cast<T>(1.0) - beta1_pow);

  int id = blockIdx.x * blockDim.x + threadIdx.x;

  for (; id < ndim; id += gridDim.x * blockDim.x) {
    T p = param[id];
    T g = grad[id];
    T mom1 = moment1[id];
    T mom2 = moment2[id];
    mom1 = beta1 * mom1 + (static_cast<T>(1.0) - beta1) * g;
    mom2 = beta2 * mom2 + (static_cast<T>(1.0) - beta2) * g * g;
    p -= lr * (mom1 /
               (sqrt(mom2) + epsilon * sqrt(static_cast<T>(1.0) - beta2_pow)));

    moment1_out[id] = mom1;
    moment2_out[id] = mom2;
    param_out[id] = p;
  }
}
template <typename T>
__global__ void UpdateBetaPow(T beta1, T beta2, const T* beta1_pow_,
                              const T* beta2_pow_, T* beta1_pow_out,
                              T* beta2_pow_out) {
  *beta1_pow_out = beta1 * beta1_pow_[0];
  *beta2_pow_out = beta2 * beta2_pow_[0];
}

template <typename T>
__global__ void SparseAdamCUDAKernelREG(
    T beta1, T beta2, T epsilon, const T beta1_pow, const T beta2_pow,
    const T* mom1_, T* mom1_out_, const T* mom2_, T* mom2_out_, const T* lr_,
    const T* grad_, const T* param_, T* param_out_, const int64_t* rows_,
    int64_t row_numel, int64_t row_count, bool lazy_mode, int ndim) {
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  T lr = *lr_;
  lr *= sqrt(1 - beta2_pow) / (1 - beta1_pow);

  for (; id < ndim; id += blockDim.x * gridDim.x) {
    auto row_idx =
        math::BinarySearch<int64_t>(rows_, row_count, id / row_numel);
    if (lazy_mode && row_idx < 0) {
      return;
    } else {
      T mom1 = mom1_[id];
      T mom2 = mom2_[id];
      T p = param_[id];
      T g = row_idx >= 0 ? grad_[row_idx * row_numel + id % row_numel] : 0;
      mom1 = beta1 * mom1 + (1 - beta1) * g;
      mom2 = beta2 * mom2 + (1 - beta2) * g * g;
      p -= lr * (mom1 / (sqrt(mom2) +
                         epsilon * sqrt(static_cast<T>(1.0) - beta2_pow)));

      // Write back to global memory
      mom1_out_[id] = mom1;
      mom2_out_[id] = mom2;
      param_out_[id] = p;
    }
  }
}

template <typename T>
class AdamOpCUDAKernel : public framework::OpKernel<T> {
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
    T epsilon = static_cast<T>(ctx.Attr<float>("epsilon"));
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
      beta1 = static_cast<T>(GetAttrFromTensor(beta1_tensor));
    }
    T beta2 = static_cast<T>(ctx.Attr<float>("beta2"));
    if (ctx.HasInput("Beta2Tensor")) {
      auto* beta2_tensor = ctx.Input<framework::Tensor>("Beta2Tensor");
      PADDLE_ENFORCE_EQ(beta2_tensor->numel(), 1,
                        platform::errors::InvalidArgument(
                            "Input(Beta2Tensor) size must be 1, but get %d",
                            beta2_tensor->numel()));
      beta2 = static_cast<T>(GetAttrFromTensor(beta2_tensor));
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
    auto& dev_ctx = ctx.template device_context<platform::CUDADeviceContext>();

    if (grad_var->IsType<framework::LoDTensor>()) {
      auto* grad = ctx.Input<LoDTensor>("Grad");

      // update param and moment
      int threads = 512;
      int blocks = (param->numel() + threads - 1) / threads;

      if (beta1_pow->place() == platform::CPUPlace() &&
          beta2_pow->place() == platform::CPUPlace()) {
        // Compute with betapow in REG
        AdamKernelREG<T><<<blocks, threads, 0, dev_ctx.stream()>>>(
            beta1, beta2, epsilon, *beta1_pow->data<T>(), *beta2_pow->data<T>(),
            mom1->data<T>(), mom1_out->mutable_data<T>(ctx.GetPlace()),
            mom2->data<T>(), mom2_out->mutable_data<T>(ctx.GetPlace()),
            lr->data<T>(), grad->data<T>(), param->data<T>(),
            param_out->mutable_data<T>(ctx.GetPlace()), param->numel());
        // Cpu update
        beta1_pow_out->mutable_data<T>(platform::CPUPlace())[0] =
            beta1 * beta1_pow->data<T>()[0];
        beta2_pow_out->mutable_data<T>(platform::CPUPlace())[0] =
            beta2 * beta2_pow->data<T>()[0];
      } else {
        AdamKernelMEM<T><<<blocks, threads, 0, dev_ctx.stream()>>>(
            beta1, beta2, epsilon, beta1_pow->data<T>(), beta2_pow->data<T>(),
            mom1->data<T>(), mom1_out->mutable_data<T>(ctx.GetPlace()),
            mom2->data<T>(), mom2_out->mutable_data<T>(ctx.GetPlace()),
            lr->data<T>(), grad->data<T>(), param->data<T>(),
            param_out->mutable_data<T>(ctx.GetPlace()), param->numel());
        // Update with gpu
        UpdateBetaPow<T><<<1, 32, 0, dev_ctx.stream()>>>(
            beta1, beta2, beta1_pow->data<T>(), beta2_pow->data<T>(),
            beta1_pow_out->mutable_data<T>(ctx.GetPlace()),
            beta2_pow_out->mutable_data<T>(ctx.GetPlace()));
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
        scatter::MergeAdd<platform::CUDADeviceContext, T> merge_func;
        merge_func(ctx.template device_context<platform::CUDADeviceContext>(),
                   *grad, &tmp_grad_merge, true);
        grad_merge_ptr = &tmp_grad_merge;
      }
      auto& grad_merge = *grad_merge_ptr;
      auto& grad_tensor = grad_merge.value();
      const T* grad_data = grad_tensor.template data<T>();
      const int64_t* rows = grad_merge.rows().Data(ctx.GetPlace());
      auto row_numel = grad_tensor.numel() / grad_merge.rows().size();

      if (beta1_pow->place() == platform::CPUPlace() &&
          beta2_pow->place() == platform::CPUPlace()) {
        int threads = 512;
        int ndim = param->numel();
        int blocks = (ndim + threads - 1) / threads;

        SparseAdamCUDAKernelREG<T><<<blocks, threads, 0, dev_ctx.stream()>>>(
            beta1, beta2, epsilon, *beta1_pow->data<T>(), *beta2_pow->data<T>(),
            mom1->data<T>(), mom1_out->mutable_data<T>(ctx.GetPlace()),
            mom2->data<T>(), mom2_out->mutable_data<T>(ctx.GetPlace()),
            lr->data<T>(), grad_data, param->data<T>(),
            param_out->mutable_data<T>(ctx.GetPlace()), rows, row_numel,
            grad_merge.rows().size(), lazy_mode, ndim);
        // Update with cpu
        beta1_pow_out->mutable_data<T>(platform::CPUPlace())[0] =
            beta1 * beta1_pow->data<T>()[0];
        beta2_pow_out->mutable_data<T>(platform::CPUPlace())[0] =
            beta2 * beta2_pow->data<T>()[0];
      } else {
        SparseAdamFunctor<T, GPUAdam> functor(
            beta1, beta2, epsilon, beta1_pow->data<T>(), beta2_pow->data<T>(),
            mom1->data<T>(), mom1_out->mutable_data<T>(ctx.GetPlace()),
            mom2->data<T>(), mom2_out->mutable_data<T>(ctx.GetPlace()),
            lr->data<T>(), grad_data, param->data<T>(),
            param_out->mutable_data<T>(ctx.GetPlace()), rows, row_numel,
            grad_merge.rows().size(), lazy_mode);

        // FIXME(minqiyang): remove BinarySearch in GPU later
        platform::ForRange<platform::CUDADeviceContext> for_range(
            static_cast<const platform::CUDADeviceContext&>(
                ctx.device_context()),
            param->numel());
        for_range(functor);
        // update beta1 and beta2
        UpdateBetaPow<T><<<1, 32, 0, dev_ctx.stream()>>>(
            beta1, beta2, beta1_pow->data<T>(), beta2_pow->data<T>(),
            beta1_pow_out->mutable_data<T>(ctx.GetPlace()),
            beta2_pow_out->mutable_data<T>(ctx.GetPlace()));
      }
    } else {
      PADDLE_THROW(platform::errors::InvalidArgument(
          "Variable type not supported by adam_op"));
    }
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_CUDA_KERNEL(adam, ops::AdamOpCUDAKernel<float>,
                        ops::AdamOpCUDAKernel<double>);
