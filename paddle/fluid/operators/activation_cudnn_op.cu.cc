// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/activation_op.h"
#include "paddle/fluid/platform/cudnn_desc.h"

namespace paddle {
namespace platform {
struct CUDAPlace;
}  // namespace platform
}  // namespace paddle

namespace paddle {
namespace operators {
using framework::Tensor;
using platform::ActivationDescriptor;
using platform::TensorDescriptor;
using platform::CUDADeviceContext;

template <typename T>
struct CudnnActivationFunctor {
  using ELEMENT_TYPE = T;
  CudnnActivationFunctor(const CUDADeviceContext& ctx, const T& c,
                         const cudnnActivationMode_t& m)
      : ctx_(ctx), coef_(c), mode_(m) {}
  void operator()(const Tensor& x, Tensor* out) {
    ActivationDescriptor act_desc;
    act_desc.set(mode_, coef_);
    TensorDescriptor x_desc, out_desc;
    x_desc.set(x);
    out_desc.set(GET_DATA_SAFELY(out, "Output", "Out", "CudnnActivation"));
    PADDLE_ENFORCE_CUDA_SUCCESS(platform::dynload::cudnnActivationForward(
        ctx_.cudnn_handle(), act_desc.desc(),
        platform::CudnnDataType<T>::kOne(), x_desc.desc(), x.data<T>(),
        platform::CudnnDataType<T>::kZero(), out_desc.desc(),
        out->mutable_data<T>(ctx_.GetPlace())));
  }
  const CUDADeviceContext& ctx_;
  const T coef_;
  const cudnnActivationMode_t mode_;
};

template <typename T>
struct CudnnActivationGradFunctor {
  using ELEMENT_TYPE = T;
  CudnnActivationGradFunctor(const CUDADeviceContext& ctx, const T& c,
                             const cudnnActivationMode_t& m)
      : ctx_(ctx), coef_(c), mode_(m) {}
  void operator()(const Tensor& x, const Tensor& out, const Tensor dout,
                  Tensor* dx) {
    ActivationDescriptor act_desc;
    act_desc.set(mode_, coef_);
    TensorDescriptor x_desc, out_desc, dout_desc, dx_desc;
    x_desc.set(x);
    out_desc.set(out);
    dout_desc.set(dout);
    dx_desc.set(GET_DATA_SAFELY(dx, "Output", "X@GRAD", "CudnnActivationGrad"));
    PADDLE_ENFORCE_CUDA_SUCCESS(platform::dynload::cudnnActivationBackward(
        ctx_.cudnn_handle(), act_desc.desc(),
        platform::CudnnDataType<T>::kOne(), out_desc.desc(), out.data<T>(),
        dout_desc.desc(), dout.data<T>(), x_desc.desc(), x.data<T>(),
        platform::CudnnDataType<T>::kZero(), dx_desc.desc(),
        dx->mutable_data<T>(ctx_.GetPlace())));
  }
  const CUDADeviceContext& ctx_;
  const T coef_;
  const cudnnActivationMode_t mode_;
};

template <typename T>
struct CudnnReluFunctor : public CudnnActivationFunctor<T> {
  explicit CudnnReluFunctor(const CUDADeviceContext& ctx)
      : CudnnActivationFunctor<T>(ctx, 0.0, CUDNN_ACTIVATION_RELU) {}
};
template <typename T>
struct CudnnReluGradFunctor : public CudnnActivationGradFunctor<T> {
  explicit CudnnReluGradFunctor(const CUDADeviceContext& ctx)
      : CudnnActivationGradFunctor<T>(ctx, 0.0, CUDNN_ACTIVATION_RELU) {}

  static constexpr ActBwdOpFwdDeps FwdDeps() { return kDepOut; }
};

template <typename T>
struct CudnnRelu6Functor : public CudnnActivationFunctor<T> {
  explicit CudnnRelu6Functor(const CUDADeviceContext& ctx)
      : CudnnActivationFunctor<T>(ctx, 6.0, CUDNN_ACTIVATION_CLIPPED_RELU) {}
};
template <typename T>
struct CudnnRelu6GradFunctor : public CudnnActivationGradFunctor<T> {
  explicit CudnnRelu6GradFunctor(const CUDADeviceContext& ctx)
      : CudnnActivationGradFunctor<T>(ctx, 6.0, CUDNN_ACTIVATION_CLIPPED_RELU) {
  }

  static constexpr ActBwdOpFwdDeps FwdDeps() { return kDepOut; }
};

template <typename T>
struct CudnnSigmoidFunctor : public CudnnActivationFunctor<T> {
  explicit CudnnSigmoidFunctor(const CUDADeviceContext& ctx)
      : CudnnActivationFunctor<T>(ctx, 0.0, CUDNN_ACTIVATION_SIGMOID) {}
};
template <typename T>
struct CudnnSigmoidGradFunctor : public CudnnActivationGradFunctor<T> {
  explicit CudnnSigmoidGradFunctor(const CUDADeviceContext& ctx)
      : CudnnActivationGradFunctor<T>(ctx, 0.0, CUDNN_ACTIVATION_SIGMOID) {}

  static constexpr ActBwdOpFwdDeps FwdDeps() { return kDepOut; }
};

template <typename T>
struct CudnnTanhFunctor : public CudnnActivationFunctor<T> {
  explicit CudnnTanhFunctor(const CUDADeviceContext& ctx)
      : CudnnActivationFunctor<T>(ctx, 0.0, CUDNN_ACTIVATION_TANH) {}
};
template <typename T>
struct CudnnTanhGradFunctor : public CudnnActivationGradFunctor<T> {
  explicit CudnnTanhGradFunctor(const CUDADeviceContext& ctx)
      : CudnnActivationGradFunctor<T>(ctx, 0.0, CUDNN_ACTIVATION_TANH) {}

  static constexpr ActBwdOpFwdDeps FwdDeps() { return kDepOut; }
};

template <typename Functor>
class CudnnActivationKernel
    : public framework::OpKernel<typename Functor::ELEMENT_TYPE> {
 public:
  using T = typename Functor::ELEMENT_TYPE;
  void Compute(const framework::ExecutionContext& context) const override {
    const framework::Tensor* X = nullptr;
    framework::Tensor* Out = nullptr;
    ExtractActivationTensor(context, &X, &Out);
    Out->mutable_data<T>(context.GetPlace());
    auto& dev_ctx = context.template device_context<CUDADeviceContext>();
    Functor functor(dev_ctx);
    functor(GET_DATA_SAFELY(X, "Input", "X", "CudnnActivation"), Out);
  }
};

template <typename Functor>
class CudnnActivationGradKernel
    : public framework::OpKernel<typename Functor::ELEMENT_TYPE> {
 public:
  using T = typename Functor::ELEMENT_TYPE;
  void Compute(const framework::ExecutionContext& context) const override {
    static_assert(Functor::FwdDeps() == kDepOut, "Forward deps must be Out.");

    const framework::Tensor *X, *Out, *dOut;
    X = Out = dOut = nullptr;
    framework::Tensor* dX = nullptr;
    ExtractActivationGradTensor<Functor::FwdDeps()>(context, &X, &Out, &dOut,
                                                    &dX);
    dX->mutable_data<T>(context.GetPlace());
    auto& dev_ctx = context.template device_context<CUDADeviceContext>();
    Functor functor(dev_ctx);
    functor(GET_DATA_SAFELY(X, "Input", "X", "CudnnActivationGrad"),
            GET_DATA_SAFELY(Out, "Input", "Out", "CudnnActivationGrad"),
            GET_DATA_SAFELY(dOut, "Input", "Out@GRAD", "CudnnActivationGrad"),
            dX);
  }
};

}  // namespace operators
}  // namespace paddle

namespace plat = paddle::platform;
namespace ops = paddle::operators;

#define FOR_EACH_CUDNN_OP_FUNCTOR(__macro)                        \
  __macro(relu, CudnnReluFunctor, CudnnReluGradFunctor);          \
  __macro(relu6, CudnnRelu6Functor, CudnnRelu6GradFunctor);       \
  __macro(sigmoid, CudnnSigmoidFunctor, CudnnSigmoidGradFunctor); \
  __macro(tanh, CudnnTanhFunctor, CudnnTanhGradFunctor)

#define REGISTER_ACTIVATION_CUDNN_KERNEL(act_type, functor, grad_functor) \
  REGISTER_OP_KERNEL(act_type, CUDNN, plat::CUDAPlace,                    \
                     ops::CudnnActivationKernel<ops::functor<float>>,     \
                     ops::CudnnActivationKernel<ops::functor<double>>);   \
  REGISTER_OP_KERNEL(                                                     \
      act_type##_grad, CUDNN, plat::CUDAPlace,                            \
      ops::CudnnActivationGradKernel<ops::grad_functor<float>>,           \
      ops::CudnnActivationGradKernel<ops::grad_functor<double>>);

FOR_EACH_CUDNN_OP_FUNCTOR(REGISTER_ACTIVATION_CUDNN_KERNEL);
