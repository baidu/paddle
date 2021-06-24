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

#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/selected_rows.h"
#include "paddle/fluid/framework/var_type_traits.h"
#include "paddle/fluid/operators/jit/kernels.h"
#ifdef PADDLE_WITH_MKLDNN
#include "paddle/fluid/operators/mkldnn/axpy_handler.h"
#endif
#include "paddle/fluid/platform/bfloat16.h"

namespace paddle {
namespace operators {

namespace detail {

template <typename T, int VariableTypeId>
struct sgd_dense_param_kernel {
  void operator()() const {}
};

// LodTensor
template <typename T>
struct sgd_dense_param_kernel<
    T, framework::VarTypeTrait<framework::LoDTensor>::kId> {
  void operator()(const framework::ExecutionContext &ctx) const {
    VLOG(4) << "[CPU]: sgd_dense_param_kernel<T, LoDTensor>";
    const auto *learning_rate = ctx.Input<framework::Tensor>("LearningRate");
    const auto *param = ctx.Input<framework::Tensor>("Param");
    auto *param_out = ctx.Output<framework::Tensor>("ParamOut");
    const auto *grad = ctx.Input<framework::Tensor>("Grad");

    const auto sz = param_out->numel();
    jit::sgd_attr_t attr(1, sz, 1, sz, 1);
    const T *lr = learning_rate->data<T>();
    const T *param_data = param->data<T>();
    const T *grad_data = grad->data<T>();
    int64_t rows_idx = 0;
    T *out_data = param_out->mutable_data<T>(ctx.GetPlace());

    auto sgd =
        jit::KernelFuncs<jit::SgdTuple<T>, platform::CPUPlace>::Cache().At(
            attr);
    sgd(lr, param_data, grad_data, &rows_idx, out_data, &attr);
  }
};

// SelectedRows
template <typename T>
struct sgd_dense_param_kernel<
    T, framework::VarTypeTrait<framework::SelectedRows>::kId> {
  void operator()(const framework::ExecutionContext &ctx) const {
    VLOG(4) << "[CPU]: sgd_dense_param_kernel<T, SelectedRows>";
    const auto *learning_rate = ctx.Input<framework::Tensor>("LearningRate");
    const auto *param = ctx.Input<framework::Tensor>("Param");
    auto *param_out = ctx.Output<framework::Tensor>("ParamOut");
    const auto *grad = ctx.Input<framework::SelectedRows>("Grad");

    const auto &grad_value = grad->value();
    const auto &grad_rows = grad->rows();
    const T *param_data = param->data<T>();
    const T *grad_data = grad_value.data<T>();
    const T *lr = learning_rate->data<T>();
    const int64_t *rows_data = grad_rows.data();
    T *out_data = param_out->mutable_data<T>(ctx.GetPlace());

    jit::sgd_attr_t attr;
    attr.param_height = param_out->dims()[0];
    attr.param_width = param_out->numel() / attr.param_height;
    attr.grad_height = grad_rows.size();  // note: it is not grad->height()
    attr.grad_width = grad_value.numel() / attr.grad_height;
    attr.selected_rows_size = grad_rows.size();

    auto sgd =
        jit::KernelFuncs<jit::SgdTuple<T>, platform::CPUPlace>::Cache().At(
            attr);
    sgd(lr, param_data, grad_data, rows_data, out_data, &attr);
  }
};

// LodTensor
template <>
struct sgd_dense_param_kernel<
    platform::bfloat16, framework::VarTypeTrait<framework::LoDTensor>::kId> {
  void operator()(const framework::ExecutionContext &ctx) const {
    VLOG(4) << "[CPU]: sgd_dense_param_kernel<bfloat16, LoDTensor>";
    const auto *learning_rate = ctx.Input<framework::Tensor>("LearningRate");
    const auto *param = ctx.Input<framework::Tensor>("Param");
    auto *param_out = ctx.Output<framework::Tensor>("ParamOut");
    const auto *grad = ctx.Input<framework::Tensor>("Grad");
    param_out->mutable_data<platform::bfloat16>(ctx.GetPlace());

    auto p = framework::EigenVector<platform::bfloat16>::Flatten(*param);
    auto g = framework::EigenVector<platform::bfloat16>::Flatten(*grad);
    auto o = framework::EigenVector<platform::bfloat16>::Flatten(*param_out);
    const auto *lr = learning_rate->data<platform::bfloat16>();

    o = p - lr[0] * g;
  }
};

// SelectedRows
template <>
struct sgd_dense_param_kernel<
    platform::bfloat16, framework::VarTypeTrait<framework::SelectedRows>::kId> {
  void operator()(const framework::ExecutionContext &ctx) const {
    VLOG(4) << "[CPU]: sgd_dense_param_kernel<bfloat16, SelectedRows>";
    const auto *learning_rate = ctx.Input<framework::Tensor>("LearningRate");
    auto *param_out = ctx.Output<framework::Tensor>("ParamOut");
    const auto *grad = ctx.Input<framework::SelectedRows>("Grad");

    const auto &grad_value = grad->value();
    const auto &grad_rows = grad->rows();
    const auto grad_height = grad->height();
    const int64_t grad_val_height = static_cast<int64_t>(grad_rows.size());
    const auto grad_width = grad_value.numel() / grad_val_height;

    const auto *grad_data = grad_value.data<platform::bfloat16>();
    auto *out_data = param_out->data<platform::bfloat16>();
    const auto *lr = learning_rate->data<platform::bfloat16>();

    for (size_t i = 0; i < grad_rows.size(); ++i) {
      PADDLE_ENFORCE_LT(
          grad_rows[i], grad_height,
          platform::errors::OutOfRange(
              "Grad rows index value should be less than grad height."
              "Got [%s], but expected less than [%s]",
              grad_rows[i], grad_height));
      const int64_t row = grad_rows[i];
#ifdef PADDLE_WITH_MKLDNN
      operators::onednn_handler_axpy(grad_width, -lr[0],
                                     grad_data + i * grad_width,
                                     out_data + row * grad_width);
#else
      for (int64_t j = 0; j < grad_width; ++j) {
        out_data[row * grad_width + j] -= lr[0] * grad_data[i * grad_width + j];
      }
#endif
    }
  }
};

template <typename T>
void sgd_op_invoke_dense_param_kernel(const framework::ExecutionContext &ctx) {
  const auto *param = ctx.Input<framework::Tensor>("Param");
  auto *param_out = ctx.Output<framework::Tensor>("ParamOut");
  const auto *grad_var = ctx.InputVar("Grad");

  if (grad_var->IsType<framework::LoDTensor>()) {
    const auto *grad = ctx.Input<framework::Tensor>("Grad");
    const auto sz = param_out->numel();
    PADDLE_ENFORCE_EQ(param->numel(), sz,
                      platform::errors::InvalidArgument(
                          "The input tensor Param's numel of SgdOp "
                          "should be equal with ParamOut's numel. "
                          "But received Param's "
                          "numel = [%s], ParamOut's numel = [%s]",
                          param->numel(), sz));
    PADDLE_ENFORCE_EQ(grad->numel(), sz,
                      platform::errors::InvalidArgument(
                          "The input tensor Grad's numel of SgdOp "
                          "should be equal with ParamOut's numel. "
                          "But received Grad's "
                          "numel = [%s], ParamOut's numel = [%s]",
                          grad->numel(), sz));

    sgd_dense_param_kernel<
        T, framework::VarTypeTrait<framework::LoDTensor>::kId>()(ctx);
  } else if (grad_var->IsType<framework::SelectedRows>()) {
    // TODO(qijun): In Sparse SGD operator, in-place update is enforced.
    // This manual optimization brings difficulty to track data dependency.
    // It's better to find a more elegant solution.
    PADDLE_ENFORCE_EQ(param, param_out,
                      platform::errors::InvalidArgument(
                          "The input tensor Param of SgdOp "
                          "should be equal with ParamOut if variable's "
                          "type is SelectedRows. "));
    const auto *grad = ctx.Input<framework::SelectedRows>("Grad");

    // for distributed training, a sparse var may be empty,
    // just skip updating.
    if (grad->rows().size() == 0) {
      return;
    }

    auto out_dims = param_out->dims();
    PADDLE_ENFORCE_EQ(
        grad->height(), out_dims[0],
        platform::errors::InvalidArgument(
            "The input tensor Grad's height of SgdOp "
            "should be equal with ParamOut's dims. But received  Grad's "
            "height [%s] and ParamOut's dims [%s]",
            grad->height(), out_dims[0]));

    auto &grad_value = grad->value();
    auto &grad_rows = grad->rows();
    const auto param_height = param_out->dims()[0];
    const auto param_width = param_out->numel() / param_height;
    // note: it is not grad->height()
    const auto grad_height = static_cast<int64_t>(grad_rows.size());
    const auto grad_width = grad_value.numel() / grad_height;

    PADDLE_ENFORCE_EQ(
        grad_width, param_width,
        platform::errors::InvalidArgument(
            "The grad_value's numel of SgdOp "
            "should be equal with param_out's numel. But received "
            "grad_value's numel [%s] and param_out's numel [%s]",
            grad_width, param_width));

    sgd_dense_param_kernel<
        T, framework::VarTypeTrait<framework::SelectedRows>::kId>()(ctx);
  } else {
    PADDLE_ENFORCE_EQ(
        false, true, platform::errors::PermissionDenied(
                         "Unsupported Variable Type of Grad in SgdOp. Excepted "
                         "LodTensor or SelectedRows, But received [%s]",
                         paddle::framework::ToTypeName(grad_var->Type())));
  }
}

}  // namespace detail

template <typename DeviceContext, typename T>
class SGDOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override;
};

template <typename T>
class SGDOpKernel<platform::CPUDeviceContext, T>
    : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    const auto *learning_rate = ctx.Input<framework::Tensor>("LearningRate");

    const auto *param_var = ctx.InputVar("Param");
    const auto *grad_var = ctx.InputVar("Grad");

    if (param_var->IsType<framework::LoDTensor>()) {
      detail::sgd_op_invoke_dense_param_kernel<T>(ctx);
    } else if (param_var->IsType<framework::SelectedRows>()) {
      PADDLE_ENFORCE_EQ(grad_var->IsType<framework::SelectedRows>(), true,
                        platform::errors::InvalidArgument(
                            "When param is SelectedRows, gradient should also "
                            "be SelectedRows"));
      const auto &param = param_var->Get<framework::SelectedRows>();
      auto *param_out = ctx.Output<framework::SelectedRows>("ParamOut");
      const auto &grad = grad_var->Get<framework::SelectedRows>();

      // for distributed training, a sparse var may be empty,
      // just skip updating.
      if (grad.rows().size() == 0) {
        return;
      }

      auto param_row_width = param.value().dims()[1];
      auto grad_row_width = grad.value().dims()[1];
      PADDLE_ENFORCE_EQ(
          param_row_width, grad_row_width,
          platform::errors::InvalidArgument(
              "The param_row in SgdOP should have the same size with grad_row. "
              "But received param_row's width is [%s], and grad_row's width is "
              "[%s]",
              param_row_width, grad_row_width));

      const auto *lr = learning_rate->data<T>();
      const auto *grad_data = grad.value().data<T>();
      auto *out_data = param_out->mutable_value()->data<T>();
      for (size_t i = 0; i < grad.rows().size(); i++) {
        int64_t id_index = param_out->AutoGrownIndex(grad.rows()[i], false);
        PADDLE_ENFORCE_GE(
            id_index, static_cast<int64_t>(0),
            platform::errors::InvalidArgument(
                "The id in SgdOp should be >= 0. But recevied id_index is [%s]",
                id_index));
        for (int64_t j = 0; j < grad_row_width; j++) {
          out_data[id_index * grad_row_width + j] -=
              lr[0] * grad_data[i * grad_row_width + j];
        }
      }
    } else {
      PADDLE_ENFORCE_EQ(
          false, true,
          platform::errors::PermissionDenied(
              "Unsupported Variable Type of Parameter in SgdOp. Excepted "
              "LodTensor or SelectedRows, But received [%s]",
              paddle::framework::ToTypeName(param_var->Type())));
    }
  }
};

}  // namespace operators
}  // namespace paddle
