/* Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.

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
#include <string>
#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/platform/place.h"

#define XPU_MALLOC(addr, num_bytes)                                        \
  PADDLE_ENFORCE_EQ(xpu_malloc(reinterpret_cast<void**>(addr), num_bytes), \
                    XPU_SUCCESS,                                           \
                    platform::errors::ResourceExhausted(                   \
                        "\n\nOut of memory error on XPU, Cannot"           \
                        "allocate %s memory on XPU. \n\nPlease "           \
                        "check whether there is any other process "        \
                        "using XPU.\n",                                    \
                        string::HumanReadableSize(num_bytes)))

#define DEFINE_XPU_GRAD_KERNEL(kernel_type, kernel_name, use_x_y_data)         \
  template <typename DeviceContext, typename T>                                \
  class Elementwise##kernel_type##GradXPUKernel                                \
      : public ElemwiseGradKernel<T> {                                         \
   public:                                                                     \
    void Compute(const framework::ExecutionContext& ctx) const override {      \
      ElemwiseGradKernel<T>::Compute(ctx);                                     \
      using Tensor = framework::Tensor;                                        \
      auto* dout = ctx.Input<Tensor>(framework::GradVarName("Out"));           \
      auto* dx = ctx.Output<Tensor>(framework::GradVarName("X"));              \
      auto* dy = ctx.Output<Tensor>(framework::GradVarName("Y"));              \
      auto dx_dims = dout->dims();                                             \
      auto dy_dims_untrimed = dout->dims();                                    \
      T* dx_data = NULL;                                                       \
      T* dy_data = NULL;                                                       \
      const T* y_data = nullptr;                                               \
      const T* x_data = nullptr;                                               \
      T* y_broadcast = nullptr;                                                \
      if (use_x_y_data) {                                                      \
        auto* x = ctx.Input<Tensor>("X");                                      \
        auto* y = ctx.Input<Tensor>("Y");                                      \
        y_data = y->data<T>();                                                 \
        x_data = x->data<T>();                                                 \
      } else {                                                                 \
        x_data = dout->data<T>();                                              \
        y_data = dout->data<T>();                                              \
      }                                                                        \
      int axis = ctx.Attr<int>("axis");                                        \
      PADDLE_ENFORCE_GE(                                                       \
          dx_dims.size(), dy_dims_untrimed.size(),                             \
          platform::errors::InvalidArgument(                                   \
              "Rank of first input must >= rank of second input."));           \
      if (dx != nullptr) {                                                     \
        dx->mutable_data<T>(ctx.GetPlace());                                   \
        dx_dims = dx->dims();                                                  \
        dx_data = dx->data<T>();                                               \
      }                                                                        \
      if (dy != nullptr) {                                                     \
        dy->mutable_data<T>(ctx.GetPlace());                                   \
        dy_dims_untrimed = dy->dims();                                         \
        dy_data = dy->data<T>();                                               \
      }                                                                        \
      int pre, n, post, is_run_common_broadcast;                               \
      if (dx_dims == dy_dims_untrimed) {                                       \
        pre = post = 1;                                                        \
        n = dout->numel();                                                     \
      } else {                                                                 \
        axis = (axis == -1 ? dx_dims.size() - dy_dims_untrimed.size() : axis); \
        PADDLE_ENFORCE_EQ(axis >= 0 && axis < dx_dims.size(), true,            \
                          platform::errors::InvalidArgument(                   \
                              "Axis should be in range [0, dx_dims)"));        \
        auto dy_dims = trim_trailing_singular_dims(dy_dims_untrimed);          \
        axis = (dy_dims.size() == 0) ? dx_dims.size() : axis;                  \
        get_mid_dims(dx_dims, dy_dims, axis, &pre, &n, &post,                  \
                     &is_run_common_broadcast);                                \
      }                                                                        \
      int len = pre * n * post;                                                \
      auto& dev_ctx =                                                          \
          ctx.template device_context<paddle::platform::XPUDeviceContext>();   \
      if (dx == nullptr) {                                                     \
        XPU_MALLOC(&dx_data, len * sizeof(float));                             \
      }                                                                        \
      if (dy == nullptr) {                                                     \
        XPU_MALLOC(&dy_data, len * sizeof(float));                             \
      } else {                                                                 \
        if (len != n) {                                                        \
          XPU_MALLOC(&dy_data, len * sizeof(float));                           \
        }                                                                      \
      }                                                                        \
      if (use_x_y_data) {                                                      \
        if (len != n) {                                                        \
          XPU_MALLOC(&y_broadcast, len * sizeof(float));                       \
          int res =                                                            \
              xpu::broadcast_ew(dev_ctx.x_context(), y_data, y_broadcast, pre, \
                                n, post, xpu::ElementwiseOp::ASSIGN);          \
          PADDLE_ENFORCE_EQ(res, xpu::Error_t::SUCCESS,                        \
                            platform::errors::Fatal("XPU kernel error!"));     \
          y_data = y_broadcast;                                                \
        }                                                                      \
      }                                                                        \
      int res = xpu::elementwise_##kernel_name##_grad(                         \
          dev_ctx.x_context(), x_data, y_data, dout->data<T>() /*out*/,        \
          dout->data<T>(), dx_data, dy_data, len);                             \
      PADDLE_ENFORCE_EQ(res, xpu::Error_t::SUCCESS,                            \
                        platform::errors::Fatal("XPU kernel error!"));         \
      if ((dy != nullptr) && (len != n)) {                                     \
        int res = xpu::reduce_ew(dev_ctx.x_context(), dy_data, dy->data<T>(),  \
                                 pre, n, post, xpu::ElementwiseOp::ASSIGN);    \
        PADDLE_ENFORCE_EQ(res, xpu::Error_t::SUCCESS,                          \
                          platform::errors::Fatal("XPU kernel error!"));       \
        dev_ctx.Wait();                                                        \
        xpu_free(dy_data);                                                     \
      }                                                                        \
      if ((len != n || dx == nullptr || dy == nullptr) &&                      \
          !(dy != nullptr && len != n)) {                                      \
        dev_ctx.Wait();                                                        \
      }                                                                        \
      if (dx == nullptr) {                                                     \
        xpu_free(dx_data);                                                     \
      }                                                                        \
      if (dy == nullptr) {                                                     \
        xpu_free(dy_data);                                                     \
      }                                                                        \
      if (use_x_y_data) {                                                      \
        if (len != n) {                                                        \
          xpu_free(y_broadcast);                                               \
        }                                                                      \
      }                                                                        \
    }                                                                          \
  }

namespace paddle {
namespace operators {

template <typename T>
struct XPUAddFunctor {
  int operator()(xpu::Context* ctx, const T* x, const T* y, T* z, int len) {
    return xpu::elementwise_add(ctx, x, y, z, len);
  }
};

template <typename T>
struct XPUMulFunctor {
  int operator()(xpu::Context* ctx, const T* x, const T* y, T* z, int len) {
    return xpu::elementwise_mul(ctx, x, y, z, len);
  }
};

template <typename T, typename Functor>
void XPUElementwise(const framework::ExecutionContext& ctx) {
  PADDLE_ENFORCE_EQ(platform::is_xpu_place(ctx.GetPlace()), true,
                    platform::errors::PreconditionNotMet(
                        "This kernel only runs on XPU device."));
  auto x_var = ctx.InputVar("X");
  PADDLE_ENFORCE_NE(x_var, nullptr,
                    platform::errors::Fatal("Cannot get input Variable X"));
  PADDLE_ENFORCE_EQ(
      x_var->IsType<framework::LoDTensor>(), true,
      platform::errors::InvalidArgument(
          "XPU only support LoDTensor, Input(X) is not LoDTensor"));

  auto x = x_var->Get<framework::LoDTensor>();
  auto* y = ctx.Input<framework::LoDTensor>("Y");
  auto* z = ctx.Output<framework::LoDTensor>("Out");
  z->mutable_data<T>(ctx.GetPlace());

  int axis = ctx.Attr<int>("axis");
  auto x_dims = x.dims();
  auto y_dims_untrimed = y->dims();
  PADDLE_ENFORCE_GE(x_dims.size(), y_dims_untrimed.size(),
                    platform::errors::InvalidArgument(
                        "Rank of first input must >= rank of second input."));
  axis = (axis == -1 ? x_dims.size() - y_dims_untrimed.size() : axis);
  PADDLE_ENFORCE_EQ(
      axis >= 0 && axis < x_dims.size(), true,
      platform::errors::InvalidArgument("Axis should be in range [0, x_dims)"));
  auto y_dims = trim_trailing_singular_dims(y_dims_untrimed);
  axis = (y_dims.size() == 0) ? x_dims.size() : axis;
  int pre, n, post, is_common_broadcast;
  get_mid_dims(x_dims, y_dims, axis, &pre, &n, &post, &is_common_broadcast);

  PADDLE_ENFORCE_NE(is_common_broadcast, 1,
                    platform::errors::Unimplemented(
                        "X's shape should be equal to Y's shape."));

  int len = pre * n * post;

  const T* x_data = x.data<T>();
  const T* y_data = y->data<T>();
  T* z_data = z->data<T>();
  T* y_broadcast = nullptr;

  auto& dev_ctx =
      ctx.template device_context<paddle::platform::XPUDeviceContext>();

  if (post == 1) {
    if (std::is_same<Functor, XPUAddFunctor<T>>::value) {
      int res = xpu::matrix_vector_add(dev_ctx.x_context(), x_data, y_data,
                                       z_data, pre, n);
      PADDLE_ENFORCE_EQ(
          res, xpu::Error_t::SUCCESS,
          platform::errors::Fatal("XPU kernel error! res = %d", res));
      return;
    }
    if (std::is_same<Functor, XPUMulFunctor<T>>::value) {
      int res = xpu::matrix_vector_mul(dev_ctx.x_context(), x_data, y_data,
                                       z_data, pre, n);
      PADDLE_ENFORCE_EQ(
          res, xpu::Error_t::SUCCESS,
          platform::errors::Fatal("XPU kernel error! res = %d", res));
      return;
    }
  }

  if (pre != 1 || post != 1) {
    PADDLE_ENFORCE(xpu_malloc(reinterpret_cast<void**>(&y_broadcast),
                              len * sizeof(T)) == XPU_SUCCESS);
    int res = xpu::broadcast_ew(dev_ctx.x_context(), y_data, y_broadcast, pre,
                                n, post, xpu::ElementwiseOp::ASSIGN);
    PADDLE_ENFORCE_EQ(
        res, xpu::Error_t::SUCCESS,
        platform::errors::Fatal("XPU kernel error! res = %d", res));
    y_data = y_broadcast;
  }

  Functor functor;
  int res = functor(dev_ctx.x_context(), x_data, y_data, z_data, len);
  PADDLE_ENFORCE_EQ(res, xpu::Error_t::SUCCESS,
                    platform::errors::Fatal("XPU kernel error! res = %d", res));

  if (pre != 1 || post != 1) {
    dev_ctx.Wait();
    xpu_free(y_broadcast);
  }
}

}  // namespace operators
}  // namespace paddle
#endif
