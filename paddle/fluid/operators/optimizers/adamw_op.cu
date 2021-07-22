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
#include "paddle/fluid/operators/amp/fp16_type_traits.h"
#include "paddle/fluid/operators/optimizers/adamw_op.h"
#include "paddle/fluid/platform/float16.h"

namespace paddle {
namespace operators {

template <typename T, typename MT>
__global__ void AdamWKernelREG(MT beta1, MT beta2, MT epsilon, MT weight_decay,
                               MT beta1_pow_, MT beta2_pow_, const MT* moment1,
                               MT* moment1_out, const MT* moment2,
                               MT* moment2_out, const MT* lr_, const T* grad,
                               const T* param, T* param_out,
                               const MT* master_param, MT* master_param_out,
                               int ndim) {
  MT lr = *lr_;
  MT beta1_pow = beta1_pow_;
  MT beta2_pow = beta2_pow_;

  MT wd = static_cast<MT>(1.0) - weight_decay * lr;
  lr *= sqrt(static_cast<MT>(1.0) - beta2_pow) /
        (static_cast<MT>(1.0) - beta1_pow);

  int id = blockIdx.x * blockDim.x + threadIdx.x;

  for (; id < ndim; id += gridDim.x * blockDim.x) {
    MT p = master_param ? master_param[id] : static_cast<MT>(param[id]);
    MT g = static_cast<MT>(grad[id]);
    MT mom1 = moment1[id];
    MT mom2 = moment2[id];
    mom1 = beta1 * mom1 + (static_cast<MT>(1.0) - beta1) * g;
    mom2 = beta2 * mom2 + (static_cast<MT>(1.0) - beta2) * g * g;
    p = wd * p -
        lr * (mom1 /
              (sqrt(mom2) + epsilon * sqrt(static_cast<MT>(1.0) - beta2_pow)));

    moment1_out[id] = mom1;
    moment2_out[id] = mom2;
    param_out[id] = static_cast<T>(p);
    if (master_param_out) {
      master_param_out[id] = p;
    }
  }
}

template <typename T, typename MT>
__global__ void AdamWKernelMEM(MT beta1, MT beta2, MT epsilon, MT weight_decay,
                               const MT* beta1_pow_, const MT* beta2_pow_,
                               const MT* moment1, MT* moment1_out,
                               const MT* moment2, MT* moment2_out,
                               const MT* lr_, const T* grad, const T* param,
                               T* param_out, const MT* master_param,
                               MT* master_param_out, int ndim) {
  MT lr = *lr_;
  MT beta1_pow = *beta1_pow_;
  MT beta2_pow = *beta2_pow_;

  MT wd = static_cast<MT>(1.0) - weight_decay * lr;
  lr *= sqrt(static_cast<MT>(1.0) - beta2_pow) /
        (static_cast<MT>(1.0) - beta1_pow);

  int id = blockIdx.x * blockDim.x + threadIdx.x;

  for (; id < ndim; id += gridDim.x * blockDim.x) {
    MT p = master_param ? master_param[id] : static_cast<MT>(param[id]);
    MT g = static_cast<MT>(grad[id]);
    MT mom1 = static_cast<MT>(moment1[id]);
    MT mom2 = static_cast<MT>(moment2[id]);
    mom1 = beta1 * mom1 + (static_cast<MT>(1.0) - beta1) * g;
    mom2 = beta2 * mom2 + (static_cast<MT>(1.0) - beta2) * g * g;
    p = wd * p -
        lr * (mom1 /
              (sqrt(mom2) + epsilon * sqrt(static_cast<MT>(1.0) - beta2_pow)));

    moment1_out[id] = mom1;
    moment2_out[id] = mom2;
    param_out[id] = static_cast<T>(p);
    if (master_param_out) {
      master_param_out[id] = p;
    }
  }
}
template <typename T>
__global__ void AdamWUpdateBetaPow(T beta1, T beta2, const T* beta1_pow_,
                                   const T* beta2_pow_, T* beta1_pow_out,
                                   T* beta2_pow_out) {
  *beta1_pow_out = beta1 * beta1_pow_[0];
  *beta2_pow_out = beta2 * beta2_pow_[0];
}

template <typename T, typename MT>
__global__ void SparseAdamWCUDAKernelREG(
    MT beta1, MT beta2, MT epsilon, MT weight_decay, const MT beta1_pow,
    const MT beta2_pow, const MT* mom1_, MT* mom1_out_, const MT* mom2_,
    MT* mom2_out_, const MT* lr_, const T* grad_, const T* param_,
    T* param_out_, const MT* master_param, MT* master_param_out,
    const int64_t* rows_, int64_t row_numel, int64_t row_count, bool lazy_mode,
    int ndim) {
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  MT lr = *lr_;

  MT wd = static_cast<MT>(1.0) - weight_decay * lr;
  lr *= sqrt(static_cast<MT>(1.0) - beta2_pow) /
        (static_cast<MT>(1.0) - beta1_pow);

  for (; id < ndim; id += blockDim.x * gridDim.x) {
    auto row_idx =
        math::BinarySearch<int64_t>(rows_, row_count, id / row_numel);
    if (lazy_mode && row_idx < 0) {
      return;
    } else {
      MT mom1 = mom1_[id];
      MT mom2 = mom2_[id];
      MT p = master_param ? master_param[id] : static_cast<MT>(param_[id]);
      MT g = row_idx >= 0
                 ? static_cast<MT>(grad_[row_idx * row_numel + id % row_numel])
                 : static_cast<MT>(0);
      mom1 = beta1 * mom1 + (static_cast<MT>(1.0) - beta1) * g;
      mom2 = beta2 * mom2 + (static_cast<MT>(1.0) - beta2) * g * g;
      p = wd * p -
          lr * (mom1 / (sqrt(mom2) +
                        epsilon * sqrt(static_cast<MT>(1.0) - beta2_pow)));

      // Write back to global memory
      mom1_out_[id] = mom1;
      mom2_out_[id] = mom2;
      param_out_[id] = static_cast<T>(p);
      if (master_param_out) {
        master_param_out[id] = p;
      }
    }
  }
}

template <typename T>
class AdamWOpCUDAKernel : public framework::OpKernel<T> {
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

    MPDType beta1 = static_cast<MPDType>(ctx.Attr<float>("beta1"));
    if (ctx.HasInput("Beta1Tensor")) {
      auto* beta1_tensor = ctx.Input<framework::Tensor>("Beta1Tensor");
      PADDLE_ENFORCE_EQ(beta1_tensor->numel(), 1,
                        platform::errors::InvalidArgument(
                            "Input(Beta1Tensor) size must be 1, but get %d",
                            beta1_tensor->numel()));
      beta1 = static_cast<MPDType>(GetAttriFromTensor(beta1_tensor));
    }
    MPDType beta2 = static_cast<MPDType>(ctx.Attr<float>("beta2"));
    if (ctx.HasInput("Beta2Tensor")) {
      auto* beta2_tensor = ctx.Input<framework::Tensor>("Beta2Tensor");
      PADDLE_ENFORCE_EQ(beta2_tensor->numel(), 1,
                        platform::errors::InvalidArgument(
                            "Input(Beta2Tensor) size must be 1, but get %d",
                            beta2_tensor->numel()));
      beta2 = static_cast<MPDType>(GetAttriFromTensor(beta2_tensor));
    }
    MPDType epsilon = static_cast<MPDType>(ctx.Attr<float>("epsilon"));
    if (ctx.HasInput("EpsilonTensor")) {
      auto* epsilon_tensor = ctx.Input<framework::Tensor>("EpsilonTensor");
      PADDLE_ENFORCE_EQ(epsilon_tensor->numel(), 1,
                        platform::errors::InvalidArgument(
                            "Input(EpsilonTensor) size must be 1, but get %d",
                            epsilon_tensor->numel()));
      epsilon = static_cast<MPDType>(GetAttriFromTensor(epsilon_tensor));
    }
    MPDType weight_decay = static_cast<MPDType>(ctx.Attr<float>("weightdecay"));
    if (ctx.HasInput("WeightDecayTensor")) {
      auto* weight_decay_tensor =
          ctx.Input<framework::Tensor>("WeightDecayTensor");
      PADDLE_ENFORCE_EQ(
          weight_decay_tensor->numel(), 1,
          platform::errors::InvalidArgument(
              "Input(WeightDecayTensor) size must be 1, but get %d",
              weight_decay_tensor->numel()));
      weight_decay =
          static_cast<MPDType>(GetAttriFromTensor(weight_decay_tensor));
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

    auto& dev_ctx = ctx.template device_context<platform::CUDADeviceContext>();

    if (grad_var->IsType<framework::LoDTensor>()) {
      auto* grad = ctx.Input<LoDTensor>("Grad");

      // update param and moment
      int threads = 512;
      int blocks = (param->numel() + threads - 1) / threads;

      if (beta1_pow->place() == platform::CPUPlace() &&
          beta2_pow->place() == platform::CPUPlace()) {
        // Compute with betapow in REG
        AdamWKernelREG<T, MPDType><<<blocks, threads, 0, dev_ctx.stream()>>>(
            beta1, beta2, epsilon, weight_decay, *beta1_pow->data<MPDType>(),
            *beta2_pow->data<MPDType>(), mom1->data<MPDType>(),
            mom1_out->mutable_data<MPDType>(ctx.GetPlace()),
            mom2->data<MPDType>(),
            mom2_out->mutable_data<MPDType>(ctx.GetPlace()),
            lr->data<MPDType>(), grad->data<T>(), param->data<T>(),
            param_out->mutable_data<T>(ctx.GetPlace()), master_in_data,
            master_out_data, param->numel());
        if (!use_global_beta_pow) {
          // Cpu update
          beta1_pow_out->mutable_data<MPDType>(platform::CPUPlace())[0] =
              beta1 * beta1_pow->data<MPDType>()[0];
          beta2_pow_out->mutable_data<MPDType>(platform::CPUPlace())[0] =
              beta2 * beta2_pow->data<MPDType>()[0];
        }
      } else {
        AdamWKernelMEM<T, MPDType><<<blocks, threads, 0, dev_ctx.stream()>>>(
            beta1, beta2, epsilon, weight_decay, beta1_pow->data<MPDType>(),
            beta2_pow->data<MPDType>(), mom1->data<MPDType>(),
            mom1_out->mutable_data<MPDType>(ctx.GetPlace()),
            mom2->data<MPDType>(),
            mom2_out->mutable_data<MPDType>(ctx.GetPlace()),
            lr->data<MPDType>(), grad->data<T>(), param->data<T>(),
            param_out->mutable_data<T>(ctx.GetPlace()), master_in_data,
            master_out_data, param->numel());
        if (!use_global_beta_pow) {
          // Update with gpu
          AdamWUpdateBetaPow<MPDType><<<1, 32, 0, dev_ctx.stream()>>>(
              beta1, beta2, beta1_pow->data<MPDType>(),
              beta2_pow->data<MPDType>(),
              beta1_pow_out->mutable_data<MPDType>(ctx.GetPlace()),
              beta2_pow_out->mutable_data<MPDType>(ctx.GetPlace()));
        }
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

        SparseAdamWCUDAKernelREG<
            T, MPDType><<<blocks, threads, 0, dev_ctx.stream()>>>(
            beta1, beta2, epsilon, weight_decay, *beta1_pow->data<MPDType>(),
            *beta2_pow->data<MPDType>(), mom1->data<MPDType>(),
            mom1_out->mutable_data<MPDType>(ctx.GetPlace()),
            mom2->data<MPDType>(),
            mom2_out->mutable_data<MPDType>(ctx.GetPlace()),
            lr->data<MPDType>(), grad_data, param->data<T>(),
            param_out->mutable_data<T>(ctx.GetPlace()), master_in_data,
            master_out_data, rows, row_numel, grad_merge.rows().size(),
            lazy_mode, ndim);
        if (!use_global_beta_pow) {
          // Update with cpu
          beta1_pow_out->mutable_data<MPDType>(platform::CPUPlace())[0] =
              beta1 * beta1_pow->data<MPDType>()[0];
          beta2_pow_out->mutable_data<MPDType>(platform::CPUPlace())[0] =
              beta2 * beta2_pow->data<MPDType>()[0];
        }
      } else {
        SparseAdamWFunctor<T, GPUAdamW, MPDType> functor(
            beta1, beta2, epsilon, weight_decay, beta1_pow->data<MPDType>(),
            beta2_pow->data<MPDType>(), mom1->data<MPDType>(),
            mom1_out->mutable_data<MPDType>(ctx.GetPlace()),
            mom2->data<MPDType>(),
            mom2_out->mutable_data<MPDType>(ctx.GetPlace()),
            lr->data<MPDType>(), grad_data, param->data<T>(),
            param_out->mutable_data<T>(ctx.GetPlace()), master_in_data,
            master_out_data, rows, row_numel, grad_merge.rows().size(),
            lazy_mode);

        // FIXME(minqiyang): remove BinarySearch in GPU later
        platform::ForRange<platform::CUDADeviceContext> for_range(
            static_cast<const platform::CUDADeviceContext&>(
                ctx.device_context()),
            param->numel());
        for_range(functor);
        if (!use_global_beta_pow) {
          // update beta1 and beta2
          AdamWUpdateBetaPow<MPDType><<<1, 32, 0, dev_ctx.stream()>>>(
              beta1, beta2, beta1_pow->data<MPDType>(),
              beta2_pow->data<MPDType>(),
              beta1_pow_out->mutable_data<MPDType>(ctx.GetPlace()),
              beta2_pow_out->mutable_data<MPDType>(ctx.GetPlace()));
        }
      }
    } else {
      PADDLE_THROW(platform::errors::InvalidArgument(
          "Variable type not supported by adamw_op"));
    }
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;

REGISTER_OP_CUDA_KERNEL(adamw, ops::AdamWOpCUDAKernel<float>,
                        ops::AdamWOpCUDAKernel<double>,
                        ops::AdamWOpCUDAKernel<plat::float16>);
