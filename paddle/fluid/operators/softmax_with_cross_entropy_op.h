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
#include "paddle/fluid/operators/math/cross_entropy.h"
#include "paddle/fluid/operators/math/softmax.h"
#include "paddle/fluid/operators/softmax_op.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;

template <typename T>
class SoftmaxWithCrossEntropyKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
   
    PADDLE_ENFORCE_EQ(
        platform::is_cpu_place(context.GetPlace()), true,
        platform::errors::Unimplemented("This kernel only runs on CPU."));
    const bool use_softmax = context.Attr<bool>("use_softmax");

    // do not with softmax op, and input is softmax
    if (!use_softmax) {
      const Tensor* softmax = context.Input<Tensor>("Logits");
      const Tensor* labels = context.Input<Tensor>("Label");
      Tensor* softmax_out = context.Output<Tensor>("Softmax");
      Tensor* loss = context.Output<Tensor>("Loss");
      const bool soft_label = context.Attr<bool>("soft_label");
      const int rank = softmax->dims().size();
      const int axis = CanonicalAxis(context.Attr<int>("axis"), rank);
      int axis_dim = softmax->dims()[axis];

      softmax_out->mutable_data<T>(context.GetPlace());
      loss->mutable_data<T>(context.GetPlace());

      const int n = SizeToAxis(axis, softmax->dims());
      const int d = SizeFromAxis(axis, softmax->dims());

      Tensor softmax_2d, labels_2d, loss_2d, softmax_out_2d;
      softmax_2d.ShareDataWith(*softmax).Resize({n, d});
      labels_2d.ShareDataWith(*labels).Resize({n, labels->numel() / n});
      loss_2d.ShareDataWith(*loss).Resize({n, d / axis_dim});
      softmax_out_2d.ShareDataWith(*softmax_out).Resize({n, d});

      auto& dev_ctx =
          context.template device_context<platform::CPUDeviceContext>();

      math::CrossEntropyFunctor<platform::CPUDeviceContext, T>()(
          dev_ctx, &loss_2d, &softmax_2d, &labels_2d, soft_label,
          context.Attr<int>("ignore_index"), axis_dim);

      // cause of input is softmax
      // copy to output softmax, directly
      framework::TensorCopy(*softmax, context.GetPlace(),
                            context.device_context(), softmax_out);

      return;
    }

    const Tensor* logits = context.Input<Tensor>("Logits");
    const Tensor* labels = context.Input<Tensor>("Label");
    Tensor* softmax = context.Output<Tensor>("Softmax");
    Tensor* loss = context.Output<Tensor>("Loss");
    const bool soft_label = context.Attr<bool>("soft_label");

    const int rank = logits->dims().size();
    const int axis = CanonicalAxis(context.Attr<int>("axis"), rank);
    int axis_dim = logits->dims()[axis];

    softmax->mutable_data<T>(context.GetPlace());
    loss->mutable_data<T>(context.GetPlace());

    const int n = SizeToAxis(axis, logits->dims());
    const int d = SizeFromAxis(axis, logits->dims());
    Tensor logits_2d, softmax_2d, labels_2d, loss_2d;
    logits_2d.ShareDataWith(*logits).Resize({n, d});
    softmax_2d.ShareDataWith(*softmax).Resize({n, d});
    labels_2d.ShareDataWith(*labels).Resize({n, labels->numel() / n});
    loss_2d.ShareDataWith(*loss).Resize({n, d / axis_dim});

    auto& dev_ctx =
        context.template device_context<platform::CPUDeviceContext>();
    math::SoftmaxFunctor<platform::CPUDeviceContext, T, false>()(
        dev_ctx, axis_dim, &logits_2d, &softmax_2d);
    math::CrossEntropyFunctor<platform::CPUDeviceContext, T>()(
        dev_ctx, &loss_2d, &softmax_2d, &labels_2d, soft_label,
        context.Attr<int>("ignore_index"), axis_dim);
  }
};

template <typename T>
class SoftmaxWithCrossEntropyGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    
    const Tensor* out_grad =
        context.Input<Tensor>(framework::GradVarName("Loss"));
    
    const Tensor* labels = context.Input<Tensor>("Label");
    Tensor* logit_grad =
        context.Output<Tensor>(framework::GradVarName("Logits"));
    
    const Tensor* softmax = context.Input<Tensor>("Softmax");
    
    const bool use_softmax = context.Attr<bool>("use_softmax");
    
    if (logit_grad != softmax || !use_softmax) {
      framework::TensorCopy(*softmax, context.GetPlace(),
                            context.device_context(), logit_grad);
    }
    
    const bool soft_label = context.Attr<bool>("soft_label");
    auto ignore_index = context.Attr<int>("ignore_index");

    const int rank = logit_grad->dims().size();
    const int axis = CanonicalAxis(context.Attr<int>("axis"), rank);
    int axis_dim = logit_grad->dims()[axis];

    const int n = SizeToAxis(axis, logit_grad->dims());
    const int d = SizeFromAxis(axis, logit_grad->dims());
    Tensor logit_grad_2d, labels_2d, out_grad_2d;
    logit_grad_2d.ShareDataWith(*logit_grad).Resize({n, d});
    labels_2d.ShareDataWith(*labels).Resize({n, labels->numel() / n});
    out_grad_2d.ShareDataWith(*out_grad).Resize({n, d / axis_dim});
    
    auto out_grad_mat = framework::EigenMatrix<T>::From(out_grad_2d);
    auto logit_grad_mat = framework::EigenMatrix<T>::From(logit_grad_2d);
    auto& place = *context.template device_context<platform::CPUDeviceContext>()
                       .eigen_device();
    if (!use_softmax) {
      // use_softmax step1
      if (soft_label) {
        auto lbl_mat = framework::EigenMatrix<T>::From(labels_2d);
        logit_grad_mat.device(place) =
            (-lbl_mat / logit_grad_mat);  // for each sample ,i  is sample id
        logit_grad_mat.device(place) =
            out_grad_mat.broadcast(Eigen::DSizes<int, 2>(1, axis_dim)) *
            logit_grad_mat;
      }
      // use_softmax step2
      else {
        const int64_t* label_data = labels->data<int64_t>();
        T* logit_grad_data = logit_grad->data<T>();
        const T* out_grad_data = out_grad->data<T>();
        const int remain = d / axis_dim;
        for (int i = 0; i < n; ++i) {         // for each sample_1_dim
          for (int j = 0; j < remain; j++) {  // for each sample_other_dims
            int idx = i * remain + j;  // this sample's label_idx. for 1d case,
                                       // remain=1 and j=0, so, idx = i
            if (label_data[idx] == ignore_index) {
              for (int k = 0; k < axis_dim; ++k) {  // for each class id's label
                logit_grad_data[i * d + k * remain + j] = 0;
              }
            } else {
              // only for this sample's label_idx, the label is 1, others is 0,
              // so, only compute this label_idx's class
              logit_grad_data[i * d + label_data[idx] * remain + j] =
                  (-1 / logit_grad_data[i * d + label_data[idx] * remain + j]) *
                  out_grad_data[idx];
              for (int k = 0; k < axis_dim; ++k) {  // for each class id's label
                if (k !=
                    label_data[idx]) {  // label_data[idx]: this sample's label
                  logit_grad_data[i * d + k * remain + j] = 0;
                }
              }
            }
          }
        }
      }
      return;
    }
    
    // for use_softmax=False, continue

    if (soft_label) {
      // when soft_label = True, ignore_index is not supported
      auto lbl_mat = framework::EigenMatrix<T>::From(labels_2d);
      logit_grad_mat.device(place) =
          out_grad_mat.broadcast(Eigen::DSizes<int, 2>(1, axis_dim)) *
          (logit_grad_mat - lbl_mat);  // for each sample ,i  is sample id
      //         1) compute dy/dx by p_j - y_j or P-Y, where j is class id,
      //            P=logit_grad_mat[i] is all class's probs, Y=lbl_mat[i] is
      //            all class's labels
      //         2) compute dy * dy/dx by   Chain rule, dy=out_grad_mat[i]
      // for high dims, e.g. (n,c) or (n,d1,...,dm, c), compute grad by matrix
      // operation

    } else {
      logit_grad_mat.device(place) =
          logit_grad_mat *  // element_wise multiply
          out_grad_mat.broadcast(Eigen::DSizes<int, 2>(1, axis_dim));

      const int64_t* label_data = labels->data<int64_t>();
      T* logit_grad_data = logit_grad->data<T>();
      const T* out_grad_data = out_grad->data<T>();
      const int remain = d / axis_dim;
      for (int i = 0; i < n; ++i) {         // for each sample_1_dim
        for (int j = 0; j < remain; j++) {  // for each sample_other_dims
          int idx = i * remain + j;  // this sample's label_idx. for 1d case,
                                     // remain=1 and j=0, so, idx = i
          if (label_data[idx] == ignore_index) {
            for (int k = 0; k < axis_dim; ++k) {  // for each class id's label
              logit_grad_data[i * d + k * remain + j] = 0;
            }
          } else {
            // only for this sample's label_idx, the label is 1, others is 0,
            // so, only compute this label_idx's class
            // for 1d case, remain=1 and j=0, so, [i * d + label_data[idx] *
            // remain + j] = [i * d + label_data[idx]]
            // let idx_x = i * d + label_data[idx] * remain + j,
            //   logit_grad_data[idx_x] = logit_grad_data[idx_x] -
            //   out_grad_data[idx]
            // note: logit_grad_mat = logit_grad_mat * out_grad_mat
            // so: logit_grad_data[idx_x] =  (logit_grad_data[idx_x] - 1) *
            // out_grad_data[idx]
            // means:           dy/dp * dy=   ( p - y ) * dy

            logit_grad_data[i * d + label_data[idx] * remain + j] -=
                out_grad_data[idx];
          }
        }
      }
    }
  }
};

}  // namespace operators
}  // namespace paddle
