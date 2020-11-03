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
#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/math/math_function.h"
#include "paddle/fluid/operators/math/pooling.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;

template <typename DeviceContext, typename T1, typename T2>
class MaxPoolWithIndexKernel : public framework::OpKernel<T1> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    const Tensor* in_x = context.Input<Tensor>("X");
    Tensor* out = context.Output<Tensor>("Out");
    Tensor* mask = context.Output<Tensor>("Mask");

    std::vector<int> ksize = context.Attr<std::vector<int>>("ksize");
    std::vector<int> strides = context.Attr<std::vector<int>>("strides");
    std::vector<int> paddings = context.Attr<std::vector<int>>("paddings");
    bool adaptive = context.Attr<bool>("adaptive");

    auto& dev_ctx = context.template device_context<DeviceContext>();
    if (context.Attr<bool>("global_pooling")) {
      for (size_t i = 0; i < ksize.size(); ++i) {
        paddings[i] = 0;
        ksize[i] = static_cast<int>(in_x->dims()[i + 2]);
      }
    }

    switch (ksize.size()) {
      case 2: {
        paddle::operators::math::MaxPool2dWithIndexFunctor<DeviceContext, T1,
                                                           T2>
            pool2d_forward;
        pool2d_forward(dev_ctx, *in_x, ksize, strides, paddings, adaptive, out,
                       mask);
      } break;
      case 3: {
        paddle::operators::math::MaxPool3dWithIndexFunctor<DeviceContext, T1,
                                                           T2>
            pool3d_forward;
        pool3d_forward(dev_ctx, *in_x, ksize, strides, paddings, adaptive, out,
                       mask);
      } break;
      default: {
        PADDLE_THROW(platform::errors::InvalidArgument(
            "Pool op only supports 2D and 3D input."));
      }
    }
  }
};

template <typename DeviceContext, typename T1, typename T2>
class MaxPoolWithIndexGradKernel : public framework::OpKernel<T1> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    const Tensor* mask = context.Input<Tensor>("Mask");
    const Tensor* out_grad =
        context.Input<Tensor>(framework::GradVarName("Out"));
    Tensor* in_x_grad = context.Output<Tensor>(framework::GradVarName("X"));

    std::vector<int> ksize = context.Attr<std::vector<int>>("ksize");
    std::vector<int> strides = context.Attr<std::vector<int>>("strides");
    std::vector<int> paddings = context.Attr<std::vector<int>>("paddings");
    bool adaptive = context.Attr<bool>("adaptive");
    if (context.Attr<bool>("global_pooling")) {
      for (size_t i = 0; i < ksize.size(); ++i) {
        paddings[i] = 0;
        ksize[i] = static_cast<int>(in_x_grad->dims()[i + 2]);
      }
    }

    if (in_x_grad) {
      in_x_grad->mutable_data<T1>(context.GetPlace());
      auto& device_ctx = context.template device_context<DeviceContext>();
      math::set_constant(device_ctx, in_x_grad, 0);

      switch (ksize.size()) {
        case 2: {
          paddle::operators::math::MaxPool2dWithIndexGradFunctor<DeviceContext,
                                                                 T1, T2>
              pool2d_backward;
          pool2d_backward(device_ctx, *out_grad, *mask, ksize, strides,
                          paddings, adaptive, in_x_grad);
        } break;
        case 3: {
          paddle::operators::math::MaxPool3dWithIndexGradFunctor<DeviceContext,
                                                                 T1, T2>
              pool3d_backward;
          pool3d_backward(device_ctx, *out_grad, *mask, ksize, strides,
                          paddings, adaptive, in_x_grad);
        } break;
        default: {
          PADDLE_THROW(platform::errors::InvalidArgument(
              "Pool op only supports 2D and 3D input."));
        }
      }
    }
  }
};
}  // namespace operators
}  // namespace paddle
