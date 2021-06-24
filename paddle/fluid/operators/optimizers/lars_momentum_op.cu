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

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/amp/fp16_type_traits.h"
#include "paddle/fluid/operators/optimizers/lars_momentum_op.h"

namespace paddle {
namespace operators {

template <typename T>
using MultiPrecisionType = typename details::MPTypeTrait<T>::Type;

template <typename T, typename MT>
__global__ void MomentumLarsKernel(
    const T* p, const T* g, const MT* v,
    const MultiPrecisionType<T>* learning_rate, const MT mu, const int64_t num,
    const MT lars_coeff, const MT lars_weight_decay,
    const MultiPrecisionType<T>* p_norm, const MultiPrecisionType<T>* g_norm,
    T* p_out, MT* v_out, const MT epsilon, const MT* master_p, MT* master_p_out,
    const MultiPrecisionType<T> rescale_grad) {
  const MT lr = static_cast<MT>(learning_rate[0]);
  MT local_lr = lr;
  const MT p_n = static_cast<MT>(p_norm[0]);
  const MT g_n = static_cast<MT>(g_norm[0]);

  if (lars_weight_decay > static_cast<MT>(0) && p_n > static_cast<MT>(0) &&
      g_n > static_cast<MT>(0)) {
    local_lr =
        lr * lars_coeff * p_n / (g_n + lars_weight_decay * p_n + epsilon);
  }
  CUDA_KERNEL_LOOP(i, num) {
    MT grad = static_cast<MT>(g[i]) * static_cast<MT>(rescale_grad);
    MT param = master_p ? master_p[i] : static_cast<MT>(p[i]);

    MT v_new = v[i] * mu + local_lr * (grad + lars_weight_decay * param);
    MT p_new = param - v_new;

    v_out[i] = v_new;
    p_out[i] = static_cast<T>(p_new);
    if (master_p_out) master_p_out[i] = p_new;
  }
}

template <typename DeviceContext, typename T>
class LarsMomentumOpCUDAKernel : public framework::OpKernel<T> {
  using MPDType = MultiPrecisionType<T>;

 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    const bool multi_precision = ctx.Attr<bool>("multi_precision");
    if (multi_precision) {
      InnerCompute<MPDType>(ctx, multi_precision);
    } else {
      InnerCompute<T>(ctx, multi_precision);
    }
  }

 private:
  template <typename MT>
  void InnerCompute(const framework::ExecutionContext& ctx,
                    const bool multi_precision) const {
    auto param_out = ctx.Output<framework::LoDTensor>("ParamOut");
    auto velocity_out = ctx.Output<framework::LoDTensor>("VelocityOut");
    auto param = ctx.Input<framework::LoDTensor>("Param");
    auto velocity = ctx.Input<framework::LoDTensor>("Velocity");
    auto grad = ctx.Input<framework::LoDTensor>("Grad");
    auto learning_rate = ctx.Input<framework::LoDTensor>("LearningRate");

    const framework::Tensor* master_param = nullptr;
    framework::Tensor* master_param_out = nullptr;
    if (multi_precision) {
      bool has_master =
          ctx.HasInput("MasterParam") && ctx.HasOutput("MasterParamOut");
      PADDLE_ENFORCE_EQ(has_master, true,
                        platform::errors::InvalidArgument(
                            "The Input(MasterParam) and Output(MasterParamOut) "
                            "should not be null when "
                            "the attr `multi_precision` is true"));
      master_param = ctx.Input<framework::Tensor>("MasterParam");
      master_param_out = ctx.Output<framework::Tensor>("MasterParamOut");
    }

    const MT* master_p = multi_precision ? master_param->data<MT>() : nullptr;
    MT* master_p_out = multi_precision
                           ? master_param_out->mutable_data<MT>(ctx.GetPlace())
                           : nullptr;

    T* p_out = param_out->mutable_data<T>(ctx.GetPlace());
    MT* v_out = velocity_out->mutable_data<MT>(ctx.GetPlace());

    MT mu = static_cast<MT>(ctx.Attr<float>("mu"));
    MT lars_coeff = static_cast<MT>(ctx.Attr<float>("lars_coeff"));
    MT lars_weight_decay =
        static_cast<MT>(ctx.Attr<float>("lars_weight_decay"));
    MT epsilon = static_cast<MT>(ctx.Attr<float>("epsilon"));
    MPDType rescale_grad =
        static_cast<MPDType>(ctx.Attr<float>("rescale_grad"));

    auto* p = param->data<T>();
    auto* g = grad->data<T>();
    auto* v = velocity->data<MT>();
    auto* lr = learning_rate->data<MPDType>();

    int block = 512;
    int grid = (param->numel() + block - 1) / block;

    auto eigen_p = framework::EigenVector<T>::Flatten(*param);
    auto eigen_g = framework::EigenVector<T>::Flatten(*grad);
    // calculate norms using eigein and launch the kernel.
    framework::Tensor p_norm_t, g_norm_t;
    p_norm_t.Resize({1});
    g_norm_t.Resize({1});
    auto* p_norm_data = p_norm_t.mutable_data<MPDType>(ctx.GetPlace());
    auto* g_norm_data = g_norm_t.mutable_data<MPDType>(ctx.GetPlace());
    auto ep_norm = framework::EigenScalar<MPDType>::From(p_norm_t);
    auto eg_norm = framework::EigenScalar<MPDType>::From(g_norm_t);

    auto* place = ctx.template device_context<DeviceContext>().eigen_device();

    // eigen unsupport fp16 l2-norm
    ep_norm.device(*place) =
        eigen_p.template cast<MPDType>().square().sum().sqrt();
    eg_norm.device(*place) =
        (eigen_g.template cast<MPDType>() * rescale_grad).square().sum().sqrt();

    MomentumLarsKernel<
        T, MT><<<grid, block, 0, ctx.cuda_device_context().stream()>>>(
        p, g, v, lr, mu, param->numel(), lars_coeff, lars_weight_decay,
        p_norm_data, g_norm_data, p_out, v_out, epsilon, master_p, master_p_out,
        rescale_grad);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_CUDA_KERNEL(
    lars_momentum,
    ops::LarsMomentumOpCUDAKernel<paddle::platform::CUDADeviceContext, float>,
    ops::LarsMomentumOpCUDAKernel<paddle::platform::CUDADeviceContext, double>,
    ops::LarsMomentumOpCUDAKernel<paddle::platform::CUDADeviceContext,
                                  paddle::platform::float16>);
