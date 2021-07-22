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

#include <algorithm>
#include <limits>
#include <utility>

#include "paddle/fluid/operators/transpose_op.h"

namespace paddle {
namespace operators {

template <typename DeviceContext, typename T>
class TransposeGPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* x = context.InputVar("X");
    auto* out = context.OutputVar("Out");

    const framework::Tensor* x_tensor =
        GetLoDTensorOrSelectedRowsValueFromVar(*x);
    framework::Tensor* out_tensor =
        GetMutableLoDTensorOrSelectedRowsValueFromVar(out);

    out_tensor->mutable_data<T>(context.GetPlace());
    if (out_tensor->numel() == 0) {
      return;
    }

    std::vector<int> axis = context.Attr<std::vector<int>>("axis");
    paddle::operators::math::TransposeFunctor<
        paddle::platform::CUDADeviceContext, T>
        transpose;
    auto& dev_ctx =
        context.template device_context<paddle::platform::CUDADeviceContext>();
    transpose(dev_ctx, *x_tensor, out_tensor, axis);
  }
};

template <typename DeviceContext, typename T>
class TransposeGradGPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* out_grad = context.InputVar(framework::GradVarName("Out"));
    auto* x_grad = context.OutputVar(framework::GradVarName("X"));

    if (!x_grad) {
      return;
    }
    const framework::Tensor* out_grad_tensor =
        GetLoDTensorOrSelectedRowsValueFromVar(*out_grad);
    framework::Tensor* x_grad_tensor =
        GetMutableLoDTensorOrSelectedRowsValueFromVar(x_grad);

    x_grad_tensor->mutable_data<T>(context.GetPlace());
    if (x_grad_tensor->numel() == 0) {
      return;
    }

    std::vector<int> axis = context.Attr<std::vector<int>>("axis");
    std::vector<int> reversed_axis(axis);

    for (size_t i = 0; i < axis.size(); i++) {
      reversed_axis[axis[i]] = i;
    }

    paddle::operators::math::TransposeFunctor<
        paddle::platform::CUDADeviceContext, T>
        transpose;
    auto& dev_ctx =
        context.template device_context<paddle::platform::CUDADeviceContext>();
    transpose(dev_ctx, *out_grad_tensor, x_grad_tensor, reversed_axis);
  }
};
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;

REGISTER_OP_CUDA_KERNEL(
    transpose,
    ops::TransposeGPUKernel<paddle::platform::CUDADeviceContext, float>,
    ops::TransposeGPUKernel<paddle::platform::CUDADeviceContext, double>,
    ops::TransposeGPUKernel<paddle::platform::CUDADeviceContext, plat::float16>,
    ops::TransposeGPUKernel<paddle::platform::CUDADeviceContext,
                            paddle::platform::complex<float>>,
    ops::TransposeGPUKernel<paddle::platform::CUDADeviceContext,
                            paddle::platform::complex<double>>);
REGISTER_OP_CUDA_KERNEL(
    transpose_grad,
    ops::TransposeGradGPUKernel<paddle::platform::CUDADeviceContext, float>,
    ops::TransposeGradGPUKernel<paddle::platform::CUDADeviceContext, double>,
    ops::TransposeGradGPUKernel<paddle::platform::CUDADeviceContext,
                                plat::float16>,
    ops::TransposeGradGPUKernel<paddle::platform::CUDADeviceContext,
                                paddle::platform::complex<float>>,
    ops::TransposeGradGPUKernel<paddle::platform::CUDADeviceContext,
                                paddle::platform::complex<double>>);

REGISTER_OP_CUDA_KERNEL(
    transpose2,
    ops::TransposeGPUKernel<paddle::platform::CUDADeviceContext, int32_t>,
    ops::TransposeGPUKernel<paddle::platform::CUDADeviceContext, int64_t>,
    ops::TransposeGPUKernel<paddle::platform::CUDADeviceContext, float>,
    ops::TransposeGPUKernel<paddle::platform::CUDADeviceContext, double>,
    ops::TransposeGPUKernel<paddle::platform::CUDADeviceContext, plat::float16>,
    ops::TransposeGPUKernel<paddle::platform::CUDADeviceContext,
                            paddle::platform::complex<float>>,
    ops::TransposeGPUKernel<paddle::platform::CUDADeviceContext,
                            paddle::platform::complex<double>>);
REGISTER_OP_CUDA_KERNEL(
    transpose2_grad,
    ops::TransposeGradGPUKernel<paddle::platform::CUDADeviceContext, int32_t>,
    ops::TransposeGradGPUKernel<paddle::platform::CUDADeviceContext, int64_t>,
    ops::TransposeGradGPUKernel<paddle::platform::CUDADeviceContext, float>,
    ops::TransposeGradGPUKernel<paddle::platform::CUDADeviceContext, double>,
    ops::TransposeGradGPUKernel<paddle::platform::CUDADeviceContext,
                                plat::float16>,
    ops::TransposeGradGPUKernel<paddle::platform::CUDADeviceContext,
                                paddle::platform::complex<float>>,
    ops::TransposeGradGPUKernel<paddle::platform::CUDADeviceContext,
                                paddle::platform::complex<double>>);
