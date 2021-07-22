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

#include <vector>
#include "dgc/dgc.h"
#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/memory/malloc.h"
#include "paddle/fluid/operators/elementwise/elementwise_add_op.h"

#if defined(__NVCC__) || defined(__HIPCC__)
#include "paddle/fluid/operators/elementwise/elementwise_op_broadcast.cu.h"
#endif

namespace paddle {
namespace operators {

template <typename T, typename Enable = void>
struct CudaTwiceAddFunctor {
  inline HOSTDEVICE T operator()(const T* args) const {
    return args[0] + args[1] + args[2];
  }
};

template <typename T, typename Enable = void>
struct CudaAddFunctor {
  inline HOSTDEVICE T operator()(const T* args) const {
    return args[0] + args[1];
  }
};

inline float get_period_sparcity(const std::vector<float>& sparsity,
                                 float cur_step, float rampup_steps) {
  PADDLE_ENFORCE_GE(static_cast<int>(cur_step), 0,
                    platform::errors::InvalidArgument(
                        "DGC current step=%d, but it must >= 0, "
                        "please submit issue in github",
                        static_cast<int>(cur_step)));

  size_t idx = static_cast<int>(cur_step * sparsity.size() / rampup_steps);
  if (idx >= sparsity.size()) {
    idx = sparsity.size() - 1;
  }

  PADDLE_ENFORCE_LT(
      idx, sparsity.size(),
      platform::errors::OutOfRange(
          "sparsity index out of bounds. idx=%d >= sparsity.size=%d", idx,
          sparsity.size()));
  return sparsity[idx];
}

template <typename DeviceContext, typename T>
class DGCOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto u = ctx.Input<framework::Tensor>("U");
    auto v = ctx.Input<framework::Tensor>("V");
    auto g = ctx.Input<framework::Tensor>("Grad");

    auto grad_out = ctx.Output<framework::Tensor>("Grad_out");

    // attrs
    float m = ctx.Attr<float>("m");
    bool use_nesterov = ctx.Attr<bool>("use_nesterov");
    auto sparsity = ctx.Attr<std::vector<float>>("sparsity");
    auto rampup_begin_step = ctx.Attr<float>("rampup_begin_step");
    auto rampup_step = ctx.Attr<float>("rampup_step");

    // nranks
    auto nranks_tensor = ctx.Input<framework::Tensor>("nranks");
    const int nranks = static_cast<const int>(*nranks_tensor->data<float>());
    PADDLE_ENFORCE_GT(nranks, 1,
                      platform::errors::PreconditionNotMet(
                          "DGC is not useful when num_trainers <= 1. Please "
                          "use multi card or multi machine GPU"));

    // regularization
    auto p = ctx.Input<framework::Tensor>("Param");
    float regular_coeff = ctx.Attr<float>("regular_coeff");
    int regular_type = ctx.Attr<int>("regular_type");

    auto p_e = framework::EigenVector<T>::Flatten(*p);
    auto g_e = framework::EigenVector<T>::Flatten(*g);
    auto grad_out_e = framework::EigenVector<T>::Flatten(*grad_out);

    auto& dev_ctx = ctx.template device_context<DeviceContext>();
    auto& eigen_ctx = *dev_ctx.eigen_device();

    // NOTE. In paddle, loss has divided by nranks. Because dgc_op is before
    // allreduce, so local regular_coeff need div nranks too. But now we
    // multi grad with nranks in dgc_op, in that case regular_coeff don't
    // need to /nranks, can prevent precision loss. For coeff often equal
    // with 1e-4, if nranks=32, coeff/nranks will be 3.125e-6, the numerical
    // accuracy of coeff/nranks will be too low.
    PADDLE_ENFORCE_EQ(regular_type >= 0 && regular_type <= 2, true,
                      platform::errors::InvalidArgument(
                          "DGC only support one of None|L1Decay|L2Decay "
                          "Regularization for now."));
    if (regular_type == 0) {
      grad_out_e.device(eigen_ctx) = (1.0 * nranks) * g_e;
    } else if (regular_type == 1) {
      // L1Decay. grad = grad + coeff * sign(param)
      grad_out_e.device(eigen_ctx) =
          (1.0 * nranks) * g_e + regular_coeff * p_e.sign();
    } else if (regular_type == 2) {
      // L2Decay. grad = grad + coeff * param
      grad_out_e.device(eigen_ctx) = (1.0 * nranks) * g_e + regular_coeff * p_e;
    }

    // current step
    auto current_step_tensor = ctx.Input<framework::Tensor>("current_step");
    const float* current_step = current_step_tensor->data<float>();

    if (static_cast<int>(*current_step) < static_cast<int>(rampup_begin_step)) {
      VLOG(10) << "current_step:" << *current_step
               << " < rampup_begin_step:" << rampup_begin_step
               << " so does't use dgc";
      return;
    }

    float ratio =
        1 - get_period_sparcity(
                sparsity, static_cast<float>(*current_step - rampup_begin_step),
                rampup_step);
    PADDLE_ENFORCE_GE(ratio, 0.0, platform::errors::InvalidArgument(
                                      "DGC sparsity ratio must >= 0"));
    PADDLE_ENFORCE_LT(ratio, 1.0, platform::errors::InvalidArgument(
                                      "DGC sparsity ratio must < 1"));
    int k = static_cast<int>(g->numel() * ratio);

    VLOG(10) << "m:" << m << ", use_nesterov:" << use_nesterov
             << ", rampup_begin_step:" << rampup_begin_step
             << ", rampup_step:" << rampup_step
             << ",  current_step:" << *current_step << ", ratio:" << ratio
             << ", k:" << k << ", nranks:" << nranks;

    auto k_out = ctx.Output<framework::Tensor>("k");
    T* k_out_data = k_out->data<T>();
    *k_out_data = k;

    auto u_out = ctx.Output<framework::Tensor>("U_out");
    auto v_out = ctx.Output<framework::Tensor>("V_out");
    auto encode_grad_out = ctx.Output<framework::Tensor>("EncodeGrad");
    auto gather_buff = ctx.Output<framework::Tensor>("GatherBuff");

    // FIXME(gongwb): use cublas.
    auto u_out_e = framework::EigenVector<T>::Flatten(*u_out);
    auto u_e = framework::EigenVector<T>::Flatten(*u);

    // calc local momentum from global momentum
    // NOTE. If grad not multi nranks, need add below code.
    // if (static_cast<int>(*current_step) ==
    //     static_cast<int>(rampup_begin_step)) {
    //   u_out_e.device(eigen_ctx) = (1.0 / nranks) * u_e;
    // }
    if (platform::is_gpu_place(ctx.GetPlace())) {
#if defined(__NVCC__) || defined(__HIPCC__)
      std::vector<const framework::Tensor*> ins = {u, v};
      std::vector<framework::Tensor*> outs = {v_out};
      const auto& cuda_ctx =
          ctx.template device_context<platform::CUDADeviceContext>();
      if (use_nesterov) {
        // u = m * (u + g)
        u_out_e.device(eigen_ctx) = m * (u_e + grad_out_e);

        // v = u + v + g
        ins.emplace_back(g);
        LaunchElementwiseCudaKernel<ElementwiseType::kTernary, T, T>(
            cuda_ctx, ins, &outs, 0, CudaTwiceAddFunctor<T>());
      } else {
        // u = m * u + g
        u_out_e.device(eigen_ctx) = m * u_e + grad_out_e;

        // v = u + v
        LaunchElementwiseCudaKernel<ElementwiseType::kBinary, T, T>(
            cuda_ctx, ins, &outs, 0, CudaAddFunctor<T>());
      }
#endif
    } else {
      if (use_nesterov) {
        // u = m * (u + g)
        u_out_e.device(eigen_ctx) = m * (u_e + grad_out_e);

        // v = u + v + g
        ElementwiseComputeEx<AddFunctor<T>, DeviceContext, T>(
            ctx, u, v, 0, AddFunctor<T>(), v_out);

        ElementwiseComputeEx<AddFunctor<T>, DeviceContext, T>(
            ctx, g, v_out, 0, AddFunctor<T>(), v_out);
      } else {
        // u = m * u + g
        u_out_e.device(eigen_ctx) = m * u_e + grad_out_e;

        // v = u + v
        ElementwiseComputeEx<AddFunctor<T>, DeviceContext, T>(
            ctx, u, v, 0, AddFunctor<T>(), v_out);
      }
    }

    T* v_out_data = v_out->mutable_data<T>(ctx.GetPlace());
    T* u_out_data = u_out->mutable_data<T>(ctx.GetPlace());
    T* encode_grad_out_data = encode_grad_out->mutable_data<T>(
        framework::DDim{2 * k}, ctx.GetPlace());
    gather_buff->mutable_data<T>(framework::DDim{2 * k * nranks},
                                 ctx.GetPlace());

    int buf_size = paddle::communication::dgc::get_buffer_size(k);
    auto tmp_ious_data = memory::Alloc(dev_ctx, buf_size);
    void* buf = reinterpret_cast<void*>(tmp_ious_data->ptr());

    if (!paddle::communication::dgc::k_select(
            static_cast<void*>(encode_grad_out_data), k, v_out_data,
            static_cast<int>(v_out->numel()), buf, dev_ctx.stream(),
            u_out_data)) {
      // TODO(weihang): owner should polish this error message
      PADDLE_THROW(platform::errors::InvalidArgument(
          "V_out numel error, V_out numel is %d.", v_out->numel()));
    }

    math::SetConstant<DeviceContext, T> tset;
    tset(dev_ctx, grad_out, static_cast<T>(0));
  }
};
}  // namespace operators
}  // namespace paddle
