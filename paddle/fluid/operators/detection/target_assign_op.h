/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

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
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/platform/for_range.h"

namespace paddle {
namespace operators {
template <typename T, typename WT>
struct TargetAssignFunctor {
  // old version for LoDTensor
  const T* in_;
  const int* match_indices_;
  const size_t* lod_;
  const int mismatch_value_;
  const int64_t N_;
  const int64_t M_;
  const int64_t P_;
  const int64_t K_;

  T* out_;
  WT* out_wt_;

  TargetAssignFunctor(const T* input, const int* match_indices,
                      const size_t* lod, const int mismatch_value,
                      const int64_t N, const int64_t M, const int64_t P,
                      const int64_t K, T* out, WT* out_wt)
      : in_(input),
        match_indices_(match_indices),
        lod_(lod),
        mismatch_value_(mismatch_value),
        N_(N),
        M_(M),
        P_(P),
        K_(K),
        out_(out),
        out_wt_(out_wt) {}

  HOSTDEVICE void operator()(size_t i) const {
    int h = i / M_;
    int w = i - h * M_;

    size_t off = lod_[h];
    int id = match_indices_[i];

    T* out = out_ + i * K_;
    WT* out_wt = out_wt_ + i;

    if (id > -1) {
      int w_off = w % P_;
      const T* in = in_ + ((off + id) * P_ + w_off) * K_;
      for (int64_t k = 0; k < K_; ++k) {
        out[k] = in[k];
      }
      out_wt[0] = static_cast<WT>(1.);
    } else {
      for (int64_t k = 0; k < K_; ++k) {
        out[k] = static_cast<T>(mismatch_value_);
      }
      out_wt[0] = static_cast<WT>(0.);
    }
  }
};

template <typename T, typename WT>
struct TargetAssignDynamicFunctor {
  // new version for dygraph Tensor
  const T* in_;
  const int* match_indices_;
  const int mismatch_value_;
  const int64_t x_dim_size_;
  const int64_t batch_size_;
  const int64_t num_anchors_;
  const int64_t num_max_boxes_;
  const int64_t K_;
  const int* neg_mask_data_;

  T* out_;
  WT* out_wt_;

  TargetAssignDynamicFunctor(const T* input, const int* match_indices,
                             const int mismatch_value, const int64_t x_dim_size,
                             const int64_t batch_size,
                             const int64_t num_anchors,
                             const int64_t num_max_boxes, const int64_t k,
                             const int* neg_idx_data, T* out, WT* out_wt)
      : in_(input),
        match_indices_(match_indices),
        mismatch_value_(mismatch_value),
        x_dim_size_(x_dim_size),
        batch_size_(batch_size),
        num_anchors_(num_anchors),
        num_max_boxes_(num_max_boxes),
        K_(k),
        neg_mask_data_(neg_idx_data),
        out_(out),
        out_wt_(out_wt) {}

  HOSTDEVICE void operator()(size_t i) const {
    int h = i / num_anchors_;
    int w = i % num_anchors_;

    int id = match_indices_[i];

    T* out = out_ + i * K_;
    WT* out_wt = out_wt_ + i;

    if (id > -1) {
      if (x_dim_size_ == 3) {
        const T* in = in_ + h * num_max_boxes_ + id;
        out[0] = in[0];
      } else {
        const T* in = in_ + ((h * num_max_boxes_ + id) * num_anchors_ + w) * K_;
        for (int64_t k = 0; k < K_; ++k) {
          out[k] = in[k];
        }
      }
      out_wt[0] = static_cast<WT>(1.);
    } else {
      for (int64_t k = 0; k < K_; ++k) {
        out[k] = static_cast<T>(mismatch_value_);
      }
      out_wt[0] = static_cast<WT>(0.);
    }

    if (neg_mask_data_) {
      out_wt[0] += neg_mask_data_[i];
    }
  }
};

template <typename DeviceContext, typename T, typename WT>
struct NegTargetAssignFunctor {
  void operator()(const platform::DeviceContext& ctx, const int* neg_indices,
                  const size_t* lod, const int N, const int M, const int K,
                  const int mismatch_value, T* out, WT* out_wt) const;
};

template <typename DeviceContext, typename T, typename WT>
class TargetAssignKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* x = ctx.Input<framework::LoDTensor>("X");
    auto* match_indices = ctx.Input<framework::Tensor>("MatchIndices");

    auto* out = ctx.Output<framework::Tensor>("Out");
    auto* out_wt = ctx.Output<framework::Tensor>("OutWeight");

    if (x->lod().size()) {
      PADDLE_ENFORCE_EQ(x->lod().size(), 1UL,
                        platform::errors::InvalidArgument(
                            "TargetAssignOp input(X) needs 1 level of LoD"));
    }

    int mismatch_value = ctx.Attr<int>("mismatch_value");

    const T* x_data = x->data<T>();
    const int* match_idx_data = match_indices->data<int>();

    T* out_data = out->mutable_data<T>(ctx.GetPlace());
    WT* out_wt_data = out_wt->mutable_data<WT>(ctx.GetPlace());

    auto& device_ctx = ctx.template device_context<DeviceContext>();

    if (x->lod().size()) {
      int64_t n = match_indices->dims()[0];
      int64_t m = match_indices->dims()[1];
      int64_t p = x->dims()[1];
      int64_t k = x->dims()[2];

      auto x_lod = x->lod().back();
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      size_t* x_lod_data = x_lod.MutableData(ctx.GetPlace());
#else
      size_t* x_lod_data = x_lod.data();
#endif

      TargetAssignFunctor<T, WT> functor(x_data, match_idx_data, x_lod_data,
                                         mismatch_value, n, m, p, k, out_data,
                                         out_wt_data);

      platform::ForRange<DeviceContext> for_range(device_ctx, n * m);
      for_range(functor);

      auto* neg_indices = ctx.Input<framework::LoDTensor>("NegIndices");
      if (neg_indices) {
        PADDLE_ENFORCE_EQ(
            neg_indices->lod().size(), 1UL,
            platform::errors::InvalidArgument(
                "TargetAssignOp input(NegIndices) needs 1 level of LoD"));
        const int* neg_idx_data = neg_indices->data<int>();
        auto neg_lod = neg_indices->lod().back();
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
        size_t* neg_lod_data = neg_lod.MutableData(ctx.GetPlace());
#else
        size_t* neg_lod_data = neg_lod.data();
#endif
        NegTargetAssignFunctor<DeviceContext, T, WT> neg_trg_functor;
        neg_trg_functor(device_ctx, neg_idx_data, neg_lod_data, n, m, k,
                        mismatch_value, out_data, out_wt_data);
      }
    } else {
      int64_t x_dim_size = x->dims().size();
      int64_t batch_size = match_indices->dims()[0];
      int64_t num_anchors = match_indices->dims()[1];
      int64_t num_max_boxes = x->dims()[1];
      int64_t k = x->dims()[x_dim_size - 1];

      auto* neg_indices = ctx.Input<framework::LoDTensor>("NegIndices");
      if (neg_indices) {
        PADDLE_ENFORCE_EQ(
            neg_indices->dims().size(), 3UL,
            platform::errors::InvalidArgument(
                "TargetAssignOp input(NegIndices) must be 3D Tensor"));
      }
      const int* neg_idx_data =
          neg_indices ? neg_indices->data<int>() : nullptr;

      TargetAssignDynamicFunctor<T, WT> functor(
          x_data, match_idx_data, mismatch_value, x_dim_size, batch_size,
          num_anchors, num_max_boxes, k, neg_idx_data, out_data, out_wt_data);

      platform::ForRange<DeviceContext> for_range(device_ctx,
                                                  batch_size * num_anchors);
      for_range(functor);
    }
  }
};

}  // namespace operators
}  // namespace paddle
