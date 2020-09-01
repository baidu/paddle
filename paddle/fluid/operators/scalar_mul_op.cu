// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/scalar_mul_op.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;

template <typename DeviceContext, typename T>
class CUDAScalarMulKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* in_t = ctx.Input<Tensor>("X");
    auto* out_t = ctx.Output<Tensor>("Out");
    auto a = static_cast<T>(ctx.Attr<float>("a"));
    auto b = static_cast<T>(ctx.Attr<float>("b"));
    auto x = in_t->data<T>();

    /*
    auto dim = in_t->dims();
    Tensor out_tmp;
    T* out;
    auto& dev_ctx = ctx.template device_context<DeviceContext>();
    out_tmp = ctx.AllocateTmpTensor<T, DeviceContext>(dim, dev_ctx);
    out = out_tmp.mutable_data<T>(ctx.GetPlace());

    for (int i = 0; i < in_t->numel(); ++i) {
      out[i] = a * x[i] + b;
    }
    */

    out_t->mutable_data<T>(ctx.GetPlace());

    auto eigen_out = framework::EigenVector<T>::Flatten(*out_t);
    auto eigen_in = framework::EigenVector<T>::Flatten(*in_t);
    auto& dev = *ctx.template device_context<DeviceContext>().eigen_device();
    eigen_out.device(dev) = a * eigen_in + b;
  }
};

template <typename DeviceContext, typename T>
class CUDAScalarMulGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* dout_t = ctx.Input<Tensor>(framework::GradVarName("Out"));
    auto* dx_t = ctx.Output<Tensor>(framework::GradVarName("X"));
    auto a = static_cast<T>(ctx.Attr<float>("a"));

    /*
    auto dout = dout_t->data<T>();

    auto dim = dout_t->dims();
    Tensor dx_tmp;
    T* dx;
    auto& dev_ctx = ctx.template device_context<DeviceContext>();
    dx_tmp = ctx.AllocateTmpTensor<T, DeviceContext>(dim, dev_ctx);
    dx = dx_tmp.mutable_data<T>(ctx.GetPlace());

    for (int i = 0; i < dout_t->numel(); ++i) {
      dx[i] = dout[i] * a;
    }
    */

    dx_t->mutable_data<T>(ctx.GetPlace());

    auto eigen_out = framework::EigenVector<T>::Flatten(*dx_t);
    auto eigen_in = framework::EigenVector<T>::Flatten(*dout_t);
    auto& dev = *ctx.template device_context<DeviceContext>().eigen_device();
    eigen_out.device(dev) = a * eigen_in;
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;

REGISTER_OP_CUDA_KERNEL(
    scalar_mul, ops::CUDAScalarMulKernel<plat::CUDADeviceContext, float>,
    ops::CUDAScalarMulKernel<plat::CUDADeviceContext, double>);
REGISTER_OP_CUDA_KERNEL(
    scalar_mul_grad,
    ops::CUDAScalarMulGradKernel<plat::CUDADeviceContext, float>,
    ops::CUDAScalarMulGradKernel<plat::CUDADeviceContext, double>);
