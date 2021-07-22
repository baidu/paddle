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
#ifdef PADDLE_WITH_XPU
#include <algorithm>
#include <string>
#include <vector>

#include "paddle/fluid/framework/op_registry.h"
#include "xpu/refactor/math.h"

namespace paddle {

namespace operators {
typedef enum { XPU_OR, XPU_AND } XpuLogicalType;

std::string XpuLogicalType2Str(XpuLogicalType ty) {
  switch (ty) {
    case XpuLogicalType::XPU_OR:
      return std::string("logical or");
    case XpuLogicalType::XPU_AND:
      return std::string("logical and");
    default:
      return std::string("unknown type");
  }
  return std::string("unknown");
}

template <XpuLogicalType xpu_type, typename T>
class BinaryLogicalOpXPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* x = context.Input<framework::Tensor>("X");
    auto* y = context.Input<framework::Tensor>("Y");
    auto* out = context.Output<framework::Tensor>("Out");
    bool* out_ptr = out->mutable_data<bool>(context.GetPlace());
    const T* x_ptr = x->data<T>();
    const T* y_ptr = y->data<T>();
    auto& dev_ctx =
        context.template device_context<paddle::platform::XPUDeviceContext>();
    framework::Tensor broadcast_x;
    framework::Tensor broadcast_y;
    bool need_broad_cast = false;
    if (x->numel() != out->numel()) {
      // x need broadcast
      T* broadcast_x_ptr =
          broadcast_x.mutable_data<T>(context.GetPlace(), out->numel());
      auto& out_dim = out->dims();
      auto& x_dim = x->dims();
      int dims = out_dim.size();
      std::vector<int> bcast_xdims;
      std::vector<int> bcast_ydims;
      for (int i = 0; i < dims; ++i) {
        if (out_dim[i] == x_dim[i]) {
          bcast_xdims.push_back(x_dim[i]);
          bcast_ydims.push_back(x_dim[i]);
          continue;
        }
        bcast_xdims.push_back(1);
        bcast_xdims.push_back(x_dim[i]);
        bcast_ydims.push_back(out_dim[i] / x_dim[i]);
        bcast_ydims.push_back(x_dim[i]);
      }

      int ret = xpu::broadcast<int8_t>(
          dev_ctx.x_context(), reinterpret_cast<const int8_t*> x_ptr,
          reinterpret_cast<int8_t*> broadcast_x_ptr, bcast_xdims, bcast_ydims);
      PADDLE_ENFORCE_EQ(ret, XPU_SUCCESS,
                        platform::errors::External(
                            "XPU broadcast kernel return wrong value[%d %s]",
                            ret, XPUAPIErrorMsg[ret]));
      x_ptr = (const T*)broadcast_x_ptr;
      need_broad_cast = true;
    }
    if (y->numel() != out->numel()) {
      // y need broadcast
      T* broadcast_y_ptr =
          broadcast_y.mutable_data<T>(context.GetPlace(), out->numel());
      auto& out_dim = out->dims();
      auto& y_dim = y->dims();
      int dims = out_dim.size();
      std::vector<int> bcast_xdims;
      std::vector<int> bcast_ydims;
      for (int i = 0; i < dims; ++i) {
        if (out_dim[i] == y_dim[i]) {
          bcast_xdims.push_back(y_dim[i]);
          bcast_ydims.push_back(y_dim[i]);
          continue;
        }
        bcast_xdims.push_back(1);
        bcast_xdims.push_back(y_dim[i]);
        bcast_ydims.push_back(out_dim[i] / y_dim[i]);
        bcast_ydims.push_back(y_dim[i]);
      }

      int ret = xpu::broadcast<int8_t>(
          dev_ctx.x_context(), reinterpret_cast<const int8_t*> y_ptr,
          reinterpret_cast<int8_t*> broadcast_y_ptr, bcast_xdims, bcast_ydims);
      PADDLE_ENFORCE_EQ(ret, XPU_SUCCESS,
                        platform::errors::External(
                            "XPU broadcast kernel return wrong value[%d %s]",
                            ret, XPUAPIErrorMsg[ret]));
      y_ptr = (const T*)broadcast_y_ptr;
      need_broad_cast = true;
    }

    // logical kernel
    int ret = XPU_SUCCESS;
    switch (xpu_type) {
      case XpuLogicalType::XPU_OR:
        ret = xpu::logical_or<bool>(dev_ctx.x_context(), x_ptr, y_ptr, out_ptr,
                                    out->numel());
        break;
      case XpuLogicalType::XPU_AND:
        ret = xpu::logical_and<bool>(dev_ctx.x_context(), x_ptr, y_ptr, out_ptr,
                                     out->numel());
      default:
        LOG(ERROR) << "xpu not support logical xpu type = "
                   << XpuLogicalType2Str(xpu_type);
        break;
    }
    PADDLE_ENFORCE_EQ(
        ret, XPU_SUCCESS,
        platform::errors::External("XPU API return wrong value[%d %s] in "
                                   "op_name[%s].",
                                   ret, XPUAPIErrorMsg[ret],
                                   XpuLogicalType2Str(xpu_type)));

    if (need_broad_cast && dev_ctx.x_context()->xpu_stream != nullptr) {
      xpu_wait();
    }
  }
};

template <typename T>
class UnaryLogicalOpXPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* x = context.Input<framework::Tensor>("X");
    auto* out = context.Output<framework::Tensor>("Out");
    if (x->numel() == 0) {
      return;
    }
    out->mutable_data<bool>(context.GetPlace());
    auto& dev_ctx =
        context.template device_context<paddle::platform::XPUDeviceContext>();
    int ret = xpu::logical_not<bool>(dev_ctx.x_context(), x->data<T>(),
                                     out->data<T>(), x->numel());
    PADDLE_ENFORCE_EQ(
        ret, XPU_SUCCESS,
        platform::errors::External("XPU API return wrong value[%d %s].", ret,
                                   XPUAPIErrorMsg[ret]));
  }
};

}  // namespace operators
}  // namespace paddle
#endif
