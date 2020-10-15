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
#include <iostream>
#include <string>
#include <utility>
#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/gather.h"
#include "paddle/fluid/operators/math/math_function.h"
#include "paddle/fluid/platform/hostdevice.h"

namespace paddle {
namespace operators {

enum class Mode {
  bilinear,
  nearest,
};

enum class PaddingMode { zeros, border, reflect };

using Tensor = framework::Tensor;
template <typename T, size_t D, int MajorType = Eigen::RowMajor,
          typename IndexType = Eigen::DenseIndex>
using EigenTensor = framework::EigenTensor<T, D, MajorType, IndexType>;

using Array3 = Eigen::DSizes<int64_t, 3>;
using Array4 = Eigen::DSizes<int64_t, 4>;

template <typename T>
static inline bool isInBound(T x, T y, T x_max, T y_max) {
  if (x < 0 || x > x_max || y < 0 || y > y_max) {
    return false;
  }
  return true;
}

template <typename T>
static inline void unnormalize(const platform::CPUDeviceContext& ctx,
                               Tensor* grid_slice,
                               const int max_val,  // height-1 or width-1
                               bool align_corners) {
  auto& place = *ctx.eigen_device();
  auto grid_slice_t = EigenTensor<T, 3>::From(*grid_slice);

  if (!align_corners) {
    auto factor = static_cast<T>((max_val + 1) * 0.5);
    grid_slice_t.device(place) =
        (grid_slice_t + static_cast<T>(1)) * factor - static_cast<T>(0.5);
  } else {
    auto factor = static_cast<T>(max_val * 0.5);
    grid_slice_t.device(place) = (grid_slice_t + static_cast<T>(1)) * factor;
  }
}

template <typename T>
static inline void clip(const platform::CPUDeviceContext& ctx,
                        Tensor* grid_slice,
                        const int max_val,  // height-1 or width-1
                        bool align_corners, std::string padding_mode) {
  auto& place = *ctx.eigen_device();
  auto grid_slice_t = EigenTensor<T, 3>::From(*grid_slice);
  if (padding_mode == "border") {
    grid_slice_t.device(place) = grid_slice_t.cwiseMax(static_cast<T>(0))
                                     .cwiseMin(static_cast<T>(max_val));
  } else if (padding_mode == "reflection") {
    if (align_corners) {
      auto double_range = static_cast<T>(max_val * 2);
      auto grid_abs = grid_slice_t.abs();
      auto extra = grid_abs - (grid_abs / double_range).floor() * double_range;
      grid_slice_t.device(place) = extra.cwiseMin(double_range - extra);
    } else {
      auto double_range = static_cast<T>((max_val + 1) * 2);
      auto grid_abs = (grid_slice_t + static_cast<T>(0.5)).abs();
      auto extra = grid_abs - (grid_abs / double_range).floor() * double_range;
      grid_slice_t.device(place) =
          extra.cwiseMin(double_range - extra) - static_cast<T>(0.5);
      grid_slice_t.device(place) = grid_slice_t.cwiseMax(static_cast<T>(0))
                                       .cwiseMin(static_cast<T>(max_val));
    }
  }
}

template <typename T>
static inline void clipWithMask(const platform::CPUDeviceContext& ctx,
                                const int max_val,  // height-1 or width-1
                                bool align_corners, std::string padding_mode,
                                Tensor* grid_slice, Tensor* grid_scale) {
  auto& place = *ctx.eigen_device();
  grid_scale->mutable_data<T>(grid_slice->dims(), ctx.GetPlace());

  auto grid_slice_t = EigenTensor<T, 3>::From(*grid_slice);
  auto factor = static_cast<T>(max_val * 0.5);
  if (!align_corners) {
    factor = static_cast<T>((max_val + 1) * 0.5);
  }
  auto grid_scale_t = EigenTensor<T, 3>::From(*grid_scale).setConstant(factor);

  if (padding_mode == "border") {
    //    auto bounded_lo = grid_slice_t.cwiseMax(static_cast<T>(0));
    auto res = grid_slice_t.cwiseMax(static_cast<T>(0))
                   .cwiseMin(static_cast<T>(max_val));

    auto in_bound = (res == grid_slice_t);
    grid_scale_t.device(place) = grid_scale_t * in_bound.template cast<T>();
    grid_slice_t.device(place) = res;
  } else if (padding_mode == "reflection") {
    if (align_corners) {
      auto double_range = static_cast<T>(max_val * 2);
      auto is_neg = (grid_slice_t < static_cast<T>(0));
      auto grid_abs = grid_slice_t.abs();
      auto extra = grid_abs - (grid_abs / double_range).floor() * double_range;
      auto one_more_flip = (extra > (double_range - extra));
      grid_scale_t.device(place) =
          grid_scale_t * ((is_neg == one_more_flip).template cast<T>() -
                          (is_neg != one_more_flip).template cast<T>());
      grid_slice_t.device(place) = extra.cwiseMin(double_range - extra);
    } else {
      auto double_range = static_cast<T>((max_val + 1) * 2);
      auto grid_abs = (grid_slice_t + static_cast<T>(0.5)).abs();
      auto is_neg = ((grid_slice_t + static_cast<T>(0.5)) < static_cast<T>(0));
      auto extra = grid_abs - (grid_abs / double_range).floor() * double_range;
      auto one_more_flip = (extra > (double_range - extra));
      auto reflected =
          extra.cwiseMin(double_range - extra) - static_cast<T>(0.5);
      auto clipped = reflected.cwiseMax(static_cast<T>(0))
                         .cwiseMin(static_cast<T>(max_val));
      auto in_bound = (clipped == reflected).template cast<T>();
      grid_scale_t.device(place) =
          grid_scale_t * ((is_neg == one_more_flip).template cast<T>() -
                          (is_neg != one_more_flip).template cast<T>()) *
          in_bound;
      grid_slice_t.device(place) = clipped;
    }
  }
}

template <typename T>
static void calcGridLocations(const platform::CPUDeviceContext& ctx,
                              const Tensor& grid, const int in_h,
                              const int in_w, bool align_corners,
                              std::string padding_mode, Tensor* grid_x,
                              Tensor* grid_y) {
  const int n = grid.dims()[0];
  const int out_h = grid.dims()[1];
  const int out_w = grid.dims()[2];

  // split grid with shape (n, h, w, 2) into (x, y) by the 3rd Dim
  T* grid_x_data = grid_x->mutable_data<T>({n, out_h, out_w}, ctx.GetPlace());
  T* grid_y_data = grid_y->mutable_data<T>({n, out_h, out_w}, ctx.GetPlace());
  const T* grid_data = grid.data<T>();
  for (int i = 0; i < n * out_h * out_w; i++) {
    grid_x_data[i] = grid_data[2 * i];
    grid_y_data[i] = grid_data[(2 * i) + 1];
  }

  unnormalize<T>(ctx, grid_x, in_w - 1, align_corners);
  unnormalize<T>(ctx, grid_y, in_h - 1, align_corners);

  clip<T>(ctx, grid_x, in_w - 1, align_corners, padding_mode);
  clip<T>(ctx, grid_y, in_h - 1, align_corners, padding_mode);
}

template <typename T>
static void calcGridLocationsWithGrad(const platform::CPUDeviceContext& ctx,
                                      const Tensor& grid, const int in_h,
                                      const int in_w, bool align_corners,
                                      std::string padding_mode, Tensor* grid_x,
                                      Tensor* grid_y, Tensor* grid_x_scale,
                                      Tensor* grid_y_scale) {
  const int n = grid.dims()[0];
  const int out_h = grid.dims()[1];
  const int out_w = grid.dims()[2];

  // split grid with shape (n, h, w, 2) into (x, y) by the 3rd Dim
  T* grid_x_data = grid_x->mutable_data<T>({n, out_h, out_w}, ctx.GetPlace());
  T* grid_y_data = grid_y->mutable_data<T>({n, out_h, out_w}, ctx.GetPlace());

  const T* grid_data = grid.data<T>();
  for (int i = 0; i < n * out_h * out_w; i++) {
    grid_x_data[i] = grid_data[2 * i];
    grid_y_data[i] = grid_data[(2 * i) + 1];
  }

  unnormalize<T>(ctx, grid_x, in_w - 1, align_corners);
  unnormalize<T>(ctx, grid_y, in_h - 1, align_corners);

  clipWithMask<T>(ctx, in_w - 1, align_corners, padding_mode, grid_x,
                  grid_x_scale);
  clipWithMask<T>(ctx, in_h - 1, align_corners, padding_mode, grid_y,
                  grid_y_scale);
}

template <typename T>
static void getGridPointValue(const Tensor& input, Tensor* output,
                              const Tensor& x, const Tensor& y) {
  const int n = input.dims()[0];
  const int c = input.dims()[1];
  const int in_h = input.dims()[2];
  const int in_w = input.dims()[3];
  const int out_h = x.dims()[1];
  const int out_w = x.dims()[2];
  auto x_t = EigenTensor<T, 3>::From(x);
  auto y_t = EigenTensor<T, 3>::From(y);
  auto output_t = EigenTensor<T, 4>::From(*output).setConstant((T)0);
  auto input_t = EigenTensor<T, 4>::From(input);

  for (int i = 0; i < n; i++) {
    for (int k = 0; k < out_h; k++) {
      for (int l = 0; l < out_w; l++) {
        if (isInBound(x_t(i, k, l), y_t(i, k, l), (T)(in_w - 1),
                      (T)(in_h - 1))) {
          for (int j = 0; j < c; j++) {
            output_t(i, j, k, l) =
                input_t(i, j, static_cast<int>(round(y_t(i, k, l))),
                        static_cast<int>(round(x_t(i, k, l))));
          }
        }
      }
    }
  }
}

template <typename T>
static void allNeigbors(const platform::CPUDeviceContext& ctx,
                        const Tensor& input, Tensor* grid_x, Tensor* grid_y,
                        Tensor* x_w, Tensor* x_e, Tensor* y_n,
                        Tensor* y_s,  // positions
                        Tensor* d_w, Tensor* d_e, Tensor* d_n,
                        Tensor* d_s,  // distance
                        Tensor* v_wn, Tensor* v_en, Tensor* v_ws,
                        Tensor* v_es) {  // values
  auto& place = *ctx.eigen_device();

  const int c = input.dims()[1];
  const int n = grid_x->dims()[0];
  const int out_h = grid_x->dims()[1];
  const int out_w = grid_x->dims()[2];
  // calculate coords of 4 corner points
  x_w->mutable_data<T>({n, out_h, out_w}, ctx.GetPlace());
  x_e->mutable_data<T>({n, out_h, out_w}, ctx.GetPlace());
  y_n->mutable_data<T>({n, out_h, out_w}, ctx.GetPlace());
  y_s->mutable_data<T>({n, out_h, out_w}, ctx.GetPlace());
  auto x_w_t = EigenTensor<T, 3>::From(*x_w);
  auto x_e_t = EigenTensor<T, 3>::From(*x_e);
  auto y_n_t = EigenTensor<T, 3>::From(*y_n);
  auto y_s_t = EigenTensor<T, 3>::From(*y_s);

  auto grid_x_t = EigenTensor<T, 3>::From(*grid_x);
  auto grid_y_t = EigenTensor<T, 3>::From(*grid_y);

  x_w_t.device(place) = grid_x_t.floor();
  x_e_t.device(place) = x_w_t + static_cast<T>(1);
  y_n_t.device(place) = grid_y_t.floor();
  y_s_t.device(place) = y_n_t + static_cast<T>(1);

  // calculate distances to 4 sides
  d_w->mutable_data<T>({n, out_h, out_w}, ctx.GetPlace());
  d_e->mutable_data<T>({n, out_h, out_w}, ctx.GetPlace());
  d_n->mutable_data<T>({n, out_h, out_w}, ctx.GetPlace());
  d_s->mutable_data<T>({n, out_h, out_w}, ctx.GetPlace());
  auto d_w_t = EigenTensor<T, 3>::From(*d_w);
  auto d_e_t = EigenTensor<T, 3>::From(*d_e);
  auto d_n_t = EigenTensor<T, 3>::From(*d_n);
  auto d_s_t = EigenTensor<T, 3>::From(*d_s);
  d_w_t.device(place) = grid_x_t - x_w_t;
  d_e_t.device(place) = x_e_t - grid_x_t;
  d_n_t.device(place) = grid_y_t - y_n_t;
  d_s_t.device(place) = y_s_t - grid_y_t;

  // calc 4 corner points value
  v_wn->mutable_data<T>({n, c, out_h, out_w}, ctx.GetPlace());
  v_en->mutable_data<T>({n, c, out_h, out_w}, ctx.GetPlace());
  v_ws->mutable_data<T>({n, c, out_h, out_w}, ctx.GetPlace());
  v_es->mutable_data<T>({n, c, out_h, out_w}, ctx.GetPlace());
  getGridPointValue<T>(input, v_wn, *x_w, *y_n);
  getGridPointValue<T>(input, v_en, *x_e, *y_n);
  getGridPointValue<T>(input, v_ws, *x_w, *y_s);
  getGridPointValue<T>(input, v_es, *x_e, *y_s);
}

template <typename T>
static void bilinearInter(const platform::CPUDeviceContext& ctx,
                          const Tensor& input, Tensor* grid_x, Tensor* grid_y,
                          Tensor* out) {
  auto& place = *ctx.eigen_device();
  const int n = grid_x->dims()[0];
  const int out_h = grid_x->dims()[1];
  const int out_w = grid_x->dims()[2];
  const int c = input.dims()[1];

  Tensor x_w, x_e, y_n, y_s;
  Tensor d_w, d_e, d_n, d_s;
  Tensor v_wn, v_en, v_ws, v_es;

  allNeigbors<T>(ctx, input, grid_x, grid_y, &x_w, &x_e, &y_n, &y_s, &d_w, &d_e,
                 &d_n, &d_s, &v_wn, &v_en, &v_ws, &v_es);

  auto d_w_t = EigenTensor<T, 3>::From(d_w);
  auto d_e_t = EigenTensor<T, 3>::From(d_e);
  auto d_n_t = EigenTensor<T, 3>::From(d_n);
  auto d_s_t = EigenTensor<T, 3>::From(d_s);

  auto d_w_scaled_t =
      d_w_t.reshape(Array4(n, 1, out_h, out_w)).broadcast(Array4(1, c, 1, 1));
  auto d_e_scaled_t =
      d_e_t.reshape(Array4(n, 1, out_h, out_w)).broadcast(Array4(1, c, 1, 1));
  auto d_n_scaled_t =
      d_n_t.reshape(Array4(n, 1, out_h, out_w)).broadcast(Array4(1, c, 1, 1));
  auto d_s_scaled_t =
      d_s_t.reshape(Array4(n, 1, out_h, out_w)).broadcast(Array4(1, c, 1, 1));
  auto v_wn_t = EigenTensor<T, 4>::From(v_wn);
  auto v_en_t = EigenTensor<T, 4>::From(v_en);
  auto v_ws_t = EigenTensor<T, 4>::From(v_ws);
  auto v_es_t = EigenTensor<T, 4>::From(v_es);
  auto output_t = EigenTensor<T, 4>::From(*out);
  // bilinear interpolaetion by 4 corner points
  output_t.device(place) = v_wn_t * d_e_scaled_t * d_s_scaled_t +
                           v_en_t * d_w_scaled_t * d_s_scaled_t +
                           v_ws_t * d_e_scaled_t * d_n_scaled_t +
                           v_es_t * d_w_scaled_t * d_n_scaled_t;
}

template <typename T>
static void nearestInter(const platform::CPUDeviceContext& ctx,
                         const Tensor& input, Tensor* grid_x, Tensor* grid_y,
                         Tensor* out) {
  auto& place = *ctx.eigen_device();

  auto grid_x_t = EigenTensor<T, 3>::From(*grid_x);
  auto grid_y_t = EigenTensor<T, 3>::From(*grid_y);
  grid_x_t = grid_x_t.round();
  grid_y_t = grid_y_t.round();
  getGridPointValue<T>(input, out, *grid_x, *grid_y);
}

template <typename T>
static void gatherOutputGradToInputGrad(const Tensor& output_grad,
                                        Tensor* input_grad, const Tensor& x,
                                        const Tensor& y, const Tensor& d1,
                                        const Tensor& d2) {
  const int n = output_grad.dims()[0];
  const int c = output_grad.dims()[1];
  const int out_h = output_grad.dims()[2];
  const int out_w = output_grad.dims()[3];
  const int in_h = input_grad->dims()[2];
  const int in_w = input_grad->dims()[3];
  auto x_t = EigenTensor<T, 3>::From(x);
  auto y_t = EigenTensor<T, 3>::From(y);
  auto d1_t = EigenTensor<T, 3>::From(d1);
  auto d2_t = EigenTensor<T, 3>::From(d2);
  auto input_grad_t = EigenTensor<T, 4>::From(*input_grad);
  auto output_grad_t = EigenTensor<T, 4>::From(output_grad);

  for (int i = 0; i < n; i++) {
    for (int k = 0; k < out_h; k++) {
      for (int l = 0; l < out_w; l++) {
        if (isInBound(x_t(i, k, l), y_t(i, k, l), (T)(in_w - 1),
                      (T)(in_h - 1))) {
          for (int j = 0; j < c; j++) {
            input_grad_t(i, j, static_cast<int>(round(y_t(i, k, l))),
                         static_cast<int>(round(x_t(i, k, l)))) +=
                output_grad_t(i, j, k, l) * d1_t(i, k, l) * d2_t(i, k, l);
          }
        }
      }
    }
  }
}

template <typename T>
static void gatherOutputGradToInputGrad(const Tensor& output_grad,
                                        Tensor* input_grad, const Tensor& x,
                                        const Tensor& y) {
  const int n = output_grad.dims()[0];
  const int c = output_grad.dims()[1];
  const int out_h = output_grad.dims()[2];
  const int out_w = output_grad.dims()[3];
  const int in_h = input_grad->dims()[2];
  const int in_w = input_grad->dims()[3];
  auto x_t = EigenTensor<T, 3>::From(x);
  auto y_t = EigenTensor<T, 3>::From(y);
  auto input_grad_t = EigenTensor<T, 4>::From(*input_grad);
  auto output_grad_t = EigenTensor<T, 4>::From(output_grad);
  for (int i = 0; i < n; i++) {
    for (int k = 0; k < out_h; k++) {
      for (int l = 0; l < out_w; l++) {
        if (isInBound(x_t(i, k, l), y_t(i, k, l), (T)(in_w - 1),
                      (T)(in_h - 1))) {
          for (int j = 0; j < c; j++) {
            input_grad_t(i, j, static_cast<int>(round(y_t(i, k, l))),
                         static_cast<int>(round(x_t(i, k, l)))) +=
                output_grad_t(i, j, k, l);
          }
        }
      }
    }
  }
}

template <typename T>
static void gatherBilinearGrad(const platform::CPUDeviceContext& ctx,
                               const Tensor& input, const Tensor& output_grad,
                               Tensor* grid_x, Tensor* grid_y,
                               Tensor* grid_x_scale, Tensor* grid_y_scale,
                               Tensor* input_grad, Tensor* grid_grad) {
  const int n = grid_x->dims()[0];
  const int out_h = grid_x->dims()[1];
  const int out_w = grid_x->dims()[2];
  const int c = input.dims()[1];

  Tensor x_w, x_e, y_n, y_s;
  Tensor d_w, d_e, d_n, d_s;
  Tensor v_wn, v_en, v_ws, v_es;

  allNeigbors<T>(ctx, input,
                 grid_x,  // grid_x
                 grid_y,  // grid_y
                 &x_w, &x_e, &y_n, &y_s, &d_w, &d_e, &d_n, &d_s, &v_wn, &v_en,
                 &v_ws, &v_es);

  // gather output grad value to input grad by corner point coords and weight
  gatherOutputGradToInputGrad<T>(output_grad, input_grad, x_w, y_n, d_e, d_s);
  gatherOutputGradToInputGrad<T>(output_grad, input_grad, x_w, y_s, d_e, d_n);
  gatherOutputGradToInputGrad<T>(output_grad, input_grad, x_e, y_n, d_w, d_s);
  gatherOutputGradToInputGrad<T>(output_grad, input_grad, x_e, y_s, d_w, d_n);

  auto v_wn_t = EigenTensor<T, 4>::From(v_wn);
  auto v_en_t = EigenTensor<T, 4>::From(v_en);
  auto v_ws_t = EigenTensor<T, 4>::From(v_ws);
  auto v_es_t = EigenTensor<T, 4>::From(v_es);

  auto d_w_t = EigenTensor<T, 3>::From(d_w);
  auto d_e_t = EigenTensor<T, 3>::From(d_e);
  auto d_n_t = EigenTensor<T, 3>::From(d_n);
  auto d_s_t = EigenTensor<T, 3>::From(d_s);

  auto output_grad_t = EigenTensor<T, 4>::From(output_grad);

  if (grid_grad != nullptr) {
    Tensor grid_grad_x, grid_grad_y;
    grid_grad_x.mutable_data<T>({n, out_h, out_w}, ctx.GetPlace());
    grid_grad_y.mutable_data<T>({n, out_h, out_w}, ctx.GetPlace());
    auto grid_grad_x_t =
        EigenTensor<T, 3>::From(grid_grad_x).setConstant(static_cast<T>(0.0));
    auto grid_grad_y_t =
        EigenTensor<T, 3>::From(grid_grad_y).setConstant(static_cast<T>(0.0));
    for (int i = 0; i < n; i++) {
      for (int j = 0; j < c; j++) {
        for (int k = 0; k < out_h; k++) {
          for (int l = 0; l < out_w; l++) {
            grid_grad_x_t(i, k, l) +=
                ((v_en_t(i, j, k, l) - v_wn_t(i, j, k, l)) * d_s_t(i, k, l) +
                 (v_es_t(i, j, k, l) - v_ws_t(i, j, k, l)) * d_n_t(i, k, l)) *
                output_grad_t(i, j, k, l);
            grid_grad_y_t(i, k, l) +=
                ((v_ws_t(i, j, k, l) - v_wn_t(i, j, k, l)) * d_e_t(i, k, l) +
                 (v_es_t(i, j, k, l) - v_en_t(i, j, k, l)) * d_w_t(i, k, l)) *
                output_grad_t(i, j, k, l);
          }
        }
      }
    }

    //  const T x_max = static_cast<T>(in_w - 1);
    //  const T y_max = static_cast<T>(in_h - 1);

    auto grid_x_scale_t = EigenTensor<T, 3>::From(*grid_x_scale);
    auto grid_y_scale_t = EigenTensor<T, 3>::From(*grid_y_scale);
    grid_grad_x_t = grid_grad_x_t * grid_x_scale_t;
    grid_grad_y_t = grid_grad_y_t * grid_y_scale_t;

    // gather grid_grad [x, y] in 3rd Dim
    T* grid_grad_data = grid_grad->data<T>();
    T* grid_grad_x_data = grid_grad_x.data<T>();
    T* grid_grad_y_data = grid_grad_y.data<T>();
    for (int i = 0; i < n * out_h * out_w; i++) {
      grid_grad_data[2 * i] = grid_grad_x_data[i];
      grid_grad_data[2 * i + 1] = grid_grad_y_data[i];
    }
  }
}

template <typename DeviceContext, typename T>
class GridSampleOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto align_corners = ctx.Attr<bool>("align_corners");
    auto padding_mode = ctx.Attr<std::string>("padding_mode");
    auto mode = ctx.Attr<std::string>("mode");

    auto* input = ctx.Input<Tensor>("X");
    auto* grid = ctx.Input<Tensor>("Grid");

    const int n = grid->dims()[0];
    const int out_h = grid->dims()[1];
    const int out_w = grid->dims()[2];
    const int c = input->dims()[1];
    const int in_h = input->dims()[2];
    const int in_w = input->dims()[3];

    auto* output = ctx.Output<Tensor>("Output");
    output->mutable_data<T>({n, c, out_h, out_w}, ctx.GetPlace());
    math::SetConstant<DeviceContext, T>()(
        ctx.template device_context<DeviceContext>(), output,
        static_cast<T>(0));

    Tensor grid_x, grid_y;
    calcGridLocations<T>(
        ctx.template device_context<platform::CPUDeviceContext>(), *grid, in_h,
        in_w, align_corners, padding_mode, &grid_x, &grid_y);
    if (mode == "bilinear") {
      bilinearInter<T>(
          ctx.template device_context<platform::CPUDeviceContext>(), *input,
          &grid_x, &grid_y, output);
    } else if (mode == "nearest") {
      auto grid_x_t = EigenTensor<T, 3>::From(grid_x);
      auto grid_y_t = EigenTensor<T, 3>::From(grid_y);
      grid_x_t = grid_x_t.round();
      grid_y_t = grid_y_t.round();
      getGridPointValue<T>(*input, output, grid_x, grid_y);
    }
  }
};

template <typename DeviceContext, typename T>
class GridSampleGradOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto align_corners = ctx.Attr<bool>("align_corners");
    auto padding_mode = ctx.Attr<std::string>("padding_mode");
    auto mode = ctx.Attr<std::string>("mode");

    auto* input = ctx.Input<Tensor>("X");
    auto* grid = ctx.Input<Tensor>("Grid");
    auto* output_grad = ctx.Input<Tensor>(framework::GradVarName("Output"));

    const int n = grid->dims()[0];
    const int out_h = grid->dims()[1];
    const int out_w = grid->dims()[2];
    const int c = input->dims()[1];
    const int in_h = input->dims()[2];
    const int in_w = input->dims()[3];

    auto* input_grad = ctx.Output<Tensor>(framework::GradVarName("X"));
    input_grad->mutable_data<T>({n, c, in_h, in_w}, ctx.GetPlace());
    math::SetConstant<DeviceContext, T>()(
        ctx.template device_context<DeviceContext>(), input_grad,
        static_cast<T>(0));

    Tensor* grid_grad = nullptr;
    if (ctx.HasOutput(framework::GradVarName("Grid"))) {
      grid_grad = ctx.Output<Tensor>(framework::GradVarName("Grid"));
      grid_grad->mutable_data<T>({n, out_h, out_w, 2}, ctx.GetPlace());
      math::SetConstant<DeviceContext, T>()(
          ctx.template device_context<DeviceContext>(), grid_grad,
          static_cast<T>(0));
    }

    Tensor grid_x, grid_y;
    Tensor grid_x_scale, grid_y_scale;
    calcGridLocationsWithGrad<T>(
        ctx.template device_context<platform::CPUDeviceContext>(), *grid, in_h,
        in_w, align_corners, padding_mode, &grid_x, &grid_y, &grid_x_scale,
        &grid_y_scale);
    if (mode == "bilinear") {
      gatherBilinearGrad<T>(ctx.template device_context<DeviceContext>(),
                            *input, *output_grad, &grid_x, &grid_y,
                            &grid_x_scale, &grid_y_scale, input_grad,
                            grid_grad);
    } else {
      auto grid_x_t = EigenTensor<T, 3>::From(grid_x);
      auto grid_y_t = EigenTensor<T, 3>::From(grid_y);
      grid_x_t = grid_x_t.round();
      grid_y_t = grid_y_t.round();
      gatherOutputGradToInputGrad<T>(*output_grad, input_grad, grid_x, grid_y);
    }
  }
};

}  // namespace operators
}  // namespace paddle
